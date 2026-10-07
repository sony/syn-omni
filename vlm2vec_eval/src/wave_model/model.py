from typing import Dict

import torch
import torch.distributed as dist

from torch import Tensor, nn
from transformers import modeling_utils

from src.arguments import ModelArguments
from src.utils.basic_utils import print_master

from .wave_official.qwenvl.model.qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


class WAVEModel(nn.Module):

    def __init__(
        self,
        encoder,
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
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # inside MMEBModel.__init__(...)
        self.rep_dim = self._infer_rep_dim(encoder=self.encoder)

        if self.rep_dim is None or int(self.rep_dim) <= 0:
            raise ValueError(f"Cannot infer rep_dim from config. Got rep_dim={self.rep_dim}")
        self.rep_dim = int(self.rep_dim)
        print(f"rep_dim: {self.rep_dim}")

    def _infer_rep_dim(self, encoder):
        cfg = getattr(encoder, "config", None)
        model_cfg = getattr(getattr(encoder, "model", None), "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        vision_cfg = getattr(cfg, "vision_config", None)

        for val in (
            # legacy paths (preserve behavior for existing models)
            getattr(cfg, "hidden_size", None),
            getattr(cfg, "d_model", None),
            getattr(cfg, "embed_dim", None),
            getattr(model_cfg, "hidden_size", None),
            # nested paths (e.g., qwen3_vl)
            getattr(text_cfg, "hidden_size", None),
            getattr(text_cfg, "d_model", None),
            getattr(text_cfg, "embed_dim", None),
            getattr(vision_cfg, "out_hidden_size", None),
            getattr(vision_cfg, "hidden_size", None),
        ):
            if val is not None:
                return val
        return None

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_input(self, input, **kwargs):
        cache_position = torch.arange(
            0, input['input_ids'].shape[1], device=input['input_ids'].device
        )
        input = self.encoder.prepare_inputs_for_generation(
            **input, use_cache=False, cache_position=cache_position
        )
        input["pred_embeds"] = bool(getattr(self, "wave_pred_embeds", True))

        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        assert "mllm_embeds" in hidden_states and hidden_states["mllm_embeds"] is not None
        embeds = hidden_states["mllm_embeds"]
        if self.normalize:
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)
        return embeds

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        assert is_trainable is False
        model_name_or_path = model_args.model_name
        config_source = model_name_or_path
        print_master(f"Loading backbone [{model_args.model_backbone}] from {model_name_or_path}")

        config = Qwen2_5OmniThinkerConfig.from_pretrained(config_source)
        config.use_cache = False
        config.padding_side = "left"
        config._attn_implementation = "flash_attention_2"
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = "flash_attention_2"

        # WAVE official classify settings.
        config.train_classify = bool(getattr(model_args, "wave_train_classify", True))
        config.classify_type = getattr(model_args, "wave_classify_type", "all_layer")
        config.sim_temperature = float(getattr(model_args, "temperature", 0.02))

        wave_use_beats = bool(getattr(model_args, "wave_use_beats", False))
        wave_beats_path = getattr(model_args, "wave_beats_path", None)
        wave_beats_only = bool(getattr(model_args, "wave_beats_only", False))
        if hasattr(config, "audio_config"):
            if wave_use_beats:
                if wave_beats_path:
                    config.audio_config.beats_path = wave_beats_path
            else:
                config.audio_config.beats_path = "No"
            config.audio_config.beats_only = wave_beats_only

        base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            config=config,
        )

        model = cls(
            encoder=base_model,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature
        )

        model.model_backbone = model_args.model_backbone
        model.wave_pred_embeds = bool(getattr(model_args, "wave_pred_embeds", True))
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
