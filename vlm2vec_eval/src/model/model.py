import json
import os
import shutil

from typing import Dict

import torch
import torch.distributed as dist

from peft import LoraConfig, PeftModel
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import (
    AutoModel,
    PreTrainedModel,
    Qwen2_5OmniThinkerForConditionalGeneration,
    modeling_utils,
)

from src.arguments import ModelArguments
from src.utils.basic_utils import print_master
from src.utils.dist_utils import get_rank, is_main
from src.utils.hf_utils import resolve_hf_checkpoint

from .lora import LoRAAdaptation, MixtureOfLoRA, MoELoRAwithShared, apply_lora

LORA_CLS_LIST = (LoRAAdaptation, MixtureOfLoRA, MoELoRAwithShared)
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


def _handle_qwen_omni_3b(model_name):
    if "Qwen2.5-Omni-3B-Thinker" not in model_name:
        return model_name

    model_dir = os.path.expanduser(model_name)
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return model_dir

    from huggingface_hub import snapshot_download
    from transformers import Qwen2_5OmniForConditionalGeneration

    tmp_dir = os.path.join(model_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # download original qwen 2.5 omni pretrained checkpoints
        snapshot_download(repo_id="Qwen/Qwen2.5-Omni-3B", local_dir=tmp_dir)
        # also populate the default HF cache with the processor-only files (no weights) so
        # that processor_name="Qwen/Qwen2.5-Omni-3B" resolves later with local_files_only=True
        snapshot_download(
            repo_id="Qwen/Qwen2.5-Omni-3B",
            allow_patterns=["*.json", "*.jinja", "*.txt", "tokenizer*", "vocab*", "merges*"],
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            tmp_dir, torch_dtype="auto", device_map="cpu", low_cpu_mem_usage=True
        )

        # extract and save thinker part
        thinker = model.thinker
        thinker.save_pretrained(model_dir)

        del thinker, model
    except Exception:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return model_dir


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration

    def __init__(
        self,
        encoder: PreTrainedModel,
        pooling: str = 'last',
        normalize: bool = False,
        temperature: float = 0.02,
    ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_input(self, input, **kwargs):
        # last token pooling?
        cache_position = torch.arange(
            0, input['input_ids'].shape[1], device=input['input_ids'].device
        )
        input = self.encoder.prepare_inputs_for_generation(
            **input, use_cache=False, cache_position=cache_position
        )
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        return self._pooling(hidden_states, input['attention_mask'])

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ["last", "eos"]:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            assert left_padding
            reps = last_hidden_state[torch.arange(last_hidden_state.shape[0]), -1, :]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        else:
            raise NotImplementedError(f"Unknown pooling type: {self.pooling}")
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        assert is_trainable is False

        # Loading the base model
        print_master(f'Loading backbone from {model_args.model_name}')

        # check nvomninemotron
        model_cls = cls.TRANSFORMER_CLS
        if model_args.model_backbone == "nv_omniembed_nemotron":
            model_cls = AutoModel

        model_name = _handle_qwen_omni_3b(model_args.model_name)
        base_model = model_cls.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        # Building the model on top of the base
        if model_args.lora:
            assert model_args.checkpoint_path
            model_name_or_path = model_args.checkpoint_path
            print_master(f'Loading LoRA from {model_name_or_path}')

            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(
                base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable
            )
            lora_model.load_adapter(
                model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable
            )
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(
        self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs
    ):
        qry_reps = self.encode_input(qry, **kwargs) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt, **kwargs) if tgt else None  # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        if hasattr(self.encoder, "enable_input_require_grads"):
            self.encoder.enable_input_require_grads()


class CustomLoRAMMEBModel(MMEBModel):
    TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration

    def __init__(self, *args, lora_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        self.local_rank = get_rank()
        if self.local_rank == 0:
            print("\nCustom LoRA Config:")
            for k, v in self.lora_config.items():
                print(f"`{k}`: {v}")
            print()

    def encode_input(self, input, modality_mask=None, **kwargs):
        # assert modality_mask is not None
        cache_position = torch.arange(
            0, input['input_ids'].shape[1], device=input['input_ids'].device
        )
        input = self.encoder.prepare_inputs_for_generation(
            **input, use_cache=False, cache_position=cache_position
        )
        hidden_states = self.encoder(
            **input,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = hidden_states.hidden_states[-1]
        return self._pooling(hidden_states, input['attention_mask'])

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        assert is_trainable is False

        # Loading the base model
        print_master(f'Loading backbone from {model_args.model_name}')
        model_name = _handle_qwen_omni_3b(model_args.model_name)
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # trust_remote_code=True,
            **kwargs
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        # Building the model on top of the base
        assert model_args.lora  # custom lora
        # assert model_args.checkpoint_path
        # 1. apply lora layer
        # 2. load corresponding checkpoint
        # 3. weight merge and unload

        model_name_or_path = resolve_hf_checkpoint(model_args.checkpoint_path)
        print_master(f'Loading LoRA from {model_name_or_path}')

        # load config and build lora model
        config_path = os.path.join(model_name_or_path, "custom_adapter_config.json")
        lora_config = json.load(open(config_path, "r", encoding='utf-8'))
        model_with_lora = apply_lora(base_model, lora_config)

        # load adapter checkpoint
        weight_path = os.path.join(model_name_or_path, "custom_adapter_model.safetensors")
        state_dict = load_file(weight_path, device="cpu")

        model_with_lora.load_state_dict(state_dict, strict=False)
        print_master(f"Loaded adapter weights from {weight_path}")

        model = cls(
            encoder=model_with_lora,
            pooling=lora_config["pooling"],
            normalize=lora_config["normalize"],
            temperature=lora_config["temperature"],
            lora_config=lora_config
        )

        if not is_trainable:
            model = model.eval()
            if model_args.merge_and_unload:
                model.merge_and_unload()
            else:
                print("Do not merge and unload lora parameters.")

        model.model_backbone = model_args.model_backbone
        return model

    @torch.no_grad()
    def merge_and_unload(self):
        """
        Finds all standalone LoRA layers and merges them into the base weight.
        Standard LoRA layer (LoRAAdaptation) is converted to merged fc layer
        """
        print_master("Merging LoRA layers into base weights...")
        names = []
        for name, module in self.encoder.named_modules():
            if isinstance(module, LORA_CLS_LIST) and module.base_layer is not None:
                if "lora_shared" in name:
                    # this will not happen
                    raise NotImplementedError

                # Get the parent module to perform the replacement
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = self.encoder.get_submodule(parent_name)

                # Convert back to plain nn.Linear
                merged_linear = module.to_linear(name)
                if merged_linear is not None:
                    setattr(parent, child_name, merged_linear)
                names.append(name)

        print_master("Merge and unload for standard layers complete.")
        if is_main():
            print(f"applied layers:\n{names}")
