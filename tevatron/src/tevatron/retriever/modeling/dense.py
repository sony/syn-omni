import json
import logging
import os

import torch

from safetensors.torch import save_file
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

from tevatron.utils.dist_utils import is_main

from .encoder import EncoderModel, prepare_base_model
from .lora import (
    apply_lora,
    build_routing_scheduler,
    get_lora_config,
    provide_modality_mask,
)

logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):

    def encode_query(self, qry):
        qry.pop("modality_mask", None)  # not used here
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            assert left_padding
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device),
                                         sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


class MultiModalDenseModel(DenseModel):
    TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration

    def __init__(self, encoder, pooling='eos', normalize=True, temperature=0.02):
        super().__init__(encoder, pooling, normalize, temperature)
        # freeze visual encoder
        self.encoder = encoder
        for param in self.encoder.visual.parameters():
            param.requires_grad = False
        # freeze audio_tower
        for param in self.encoder.audio_tower.parameters():
            param.requires_grad = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def encode_query(self, qry, is_query=True):
        modality_mask = qry.pop("modality_mask", None)  # noqa  # not used here

        cache_position = torch.arange(0, qry['input_ids'].shape[1], device=qry['input_ids'].device)
        qry = self.encoder.prepare_inputs_for_generation(
            **qry, use_cache=False, cache_position=cache_position
        )
        query_hidden_states = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        query_hidden_states = query_hidden_states.hidden_states[-1]
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg, is_query=False)


class CustomLoRA_MMDenseModel(MultiModalDenseModel):
    TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration

    def __init__(self, encoder, pooling='eos', normalize=True, temperature=0.02, lora_config={}):
        super().__init__(encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        self.lora_config = lora_config  # for save
        if is_main():
            # print lora config
            print("custom lora configuration:")
            for k, v in self.lora_config.items():
                print(f"`{k}`: {v}")
            print()
        self.alpha_scheduler = build_routing_scheduler(lora_config)

    def encode_query(self, qry, is_query=True):
        qry_local = qry.copy()  # preserve the original input dict (`qry`) after `pop`
        modality_mask = qry_local.pop("modality_mask", None)
        assert modality_mask is not None, modality_mask

        with provide_modality_mask(modality_mask):
            # lora/router.py
            cache_position = torch.arange(
                0, qry_local['input_ids'].shape[1], device=qry_local['input_ids'].device
            )
            qry_inputs = self.encoder.prepare_inputs_for_generation(
                **qry_local, use_cache=False, cache_position=cache_position
            )
            query_hidden_states = self.encoder(
                **qry_inputs, return_dict=True, output_hidden_states=True
            )
            query_hidden_states = query_hidden_states.hidden_states[-1]
            return self._pooling(query_hidden_states, qry_local['attention_mask'])

    @classmethod
    def build(cls, model_args, train_args):
        torch_dtype = torch.bfloat16 if train_args.bf16 else (
            torch.float16 if train_args.fp16 else torch.float32
        )
        hf_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": model_args.attn_implementation
        }

        # init pretrained model
        model_path = prepare_base_model(model_args.model_name_or_path, model_args.model_save_root)
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_path, **hf_kwargs)

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        if train_args.gradient_checkpointing:
            raise NotImplementedError  # we dont use

        # ensure all parameters are frozen before applying lora
        for _, p in base_model.named_parameters():
            p.requires_grad_(False)

        # Prepare configuration and model for Custom LoRA
        custom_lora_config = get_lora_config(model_args)
        model_with_lora = apply_lora(base_model, custom_lora_config)

        # Wrap with current class
        return cls(
            encoder=model_with_lora,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
            lora_config=custom_lora_config
        )

    def save(self, output_dir: str):
        """
        Save both the custom adapter weights (using safetensors)
        and the architecture configuration.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save weights: only trainable parameters (safetensors format)
        trainable_state_dict = {
            n: p.detach().cpu()
            for n, p in self.encoder.named_parameters() if p.requires_grad
        }

        weights_path = os.path.join(output_dir, "custom_adapter_model.safetensors")
        save_file(trainable_state_dict, weights_path)

        # Save config: serialize lora_config to JSON
        if self.lora_config:
            config_path = os.path.join(output_dir, "custom_adapter_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.lora_config, f, indent=4)

        logger.info(f"Model adapter and config saved to {output_dir}")
