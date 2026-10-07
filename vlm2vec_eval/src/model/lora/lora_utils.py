from functools import partial

import torch.nn as nn

from .lora_layer import LoRAAdaptation, MixtureOfLoRA, MoELoRAwithShared


def get_lora_config(model_args):
    config = {
        # Target Modules
        "target_modules": model_args.lora_target_modules.split(','),

        # Basic LoRA/DoRA Parameters
        "rank": model_args.lora_r,
        "alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "use_dora": model_args.use_dora,

        # Hybrid Routing Logic Settings
        # Determines where to switch from Standard to Custom LoRA in LLM
        "custom_start_layer": model_args.custom_start_layer,

        # Custom LoRA Specifics
        "custom_lora_name": model_args.custom_lora_name,
        "num_specific_lora": model_args.num_specific_lora,
        "specific_rank_ratio": model_args.specific_rank_ratio,

        # orthogonal loss weight
        "ortho_loss_weight": model_args.ortho_loss_weight,

        # Router Settings
        "router_type": model_args.router_type,  # ["linear", "modality", "progressive"]
        "routing_mode": model_args.routing_mode,  # ["soft", "hard"]
        "routing_activation_soft": model_args.routing_activation_soft,  # ["softmax", "sigmoid"]
        "normalize_routing_weight": model_args.normalize_routing_weight,

        # router loss config (progressive)
        "router_loss_type": model_args.router_loss_type,
        "router_loss_weight": model_args.router_loss_weight,
        "router_loss_margin": model_args.router_loss_margin,

        # progressive router config
        "router_warmup_ratio": model_args.router_warmup_ratio,
        "router_transition_ratio": model_args.router_transition_ratio,
        "router_schedule_type": model_args.router_schedule_type,

        # Model Instance Metadata
        "pooling": model_args.pooling,
        "normalize": model_args.normalize,
        "temperature": model_args.temperature,

        # ## options not used ##
        # gating config
        "gating_bottleneck_dim": model_args.gating_bottleneck_dim,
        "activation": model_args.lora_activation,  # [None, "leaky_relu", "silu"]
        "negative_slope": model_args.negative_slope,  # Only for LeakyReLU
        "filter_absent_modality": model_args.filter_absent_modality,
    }
    return config


def build_custom_lora(config):
    custom_name = config["custom_lora_name"]
    if custom_name == "stacked":
        # new_layer = partial(CustomLoRALayer, **config)
        raise NotImplementedError  # NOTE: currently disable
    elif custom_name == "MoE":
        new_layer = partial(MixtureOfLoRA, **config)
    elif custom_name == "MoEwithShared":
        new_layer = partial(MoELoRAwithShared, **config)
    else:
        raise ValueError(f"Unknown custom lora name: {custom_name}")
    return new_layer


def apply_lora(model, config):
    custom_start_layer = config.get("custom_start_layer", -1)
    target_modules = config["target_modules"]
    assert len(target_modules)

    for name, module in model.named_modules():
        # exclude vision and audio encoders for injection
        if "visual" in name or "audio_tower" in name:
            continue

        # target only specified linear layers (q_proj, v_proj, etc.)
        if any(target in name for target in target_modules):
            if not isinstance(module, nn.Linear):
                continue

            # Check layer index for LLM backbone (model.layers.N...)
            is_custom_target = False
            if "model.layers." in name:
                try:
                    # Extract index N from 'model.layers.N.self_attn.q_proj'
                    layer_idx = int(name.split("model.layers.")[1].split(".")[0])
                    if layer_idx >= custom_start_layer:
                        is_custom_target = True
                except (IndexError, ValueError):
                    pass

            # Get parent and layer name for replacement
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            # Choose and Inject
            if is_custom_target:
                # Modality-specific Custom LoRA (for upper layers)
                new_layer_fn = build_custom_lora(config)
                new_layer = new_layer_fn(base_layer=module)
            else:
                # Standard LoRA (for lower layers)
                # Using the integrated LoRAAdaptation block
                # NOTE: currently, disable this feature temporarilly
                raise NotImplementedError
                new_layer = LoRAAdaptation(
                    base_layer=module,
                    rank=config['rank'],
                    alpha=config['alpha'],
                    dropout_p=config["lora_dropout"],
                    use_dora=config["use_dora"]
                )

            setattr(parent, layer_name, new_layer)

    return model
