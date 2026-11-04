from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_save_root: str = field(
        default="~/.cache/huggingface/hub",
        metadata={"help": "Download pre-trained model to local and load from it"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    pooling: str = field(
        default='last', metadata={"help": "pooling method for query and passage encoder"}
    )
    normalize: bool = field(
        default=True, metadata={"help": "normalize query and passage representations"}
    )

    temperature: float = field(default=0.02, metadata={"help": "temperature for softmax"})

    # for lora
    lora: bool = field(
        default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    lora_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained lora model or model identifier from huggingface.co/models"
        }
    )
    lora_r: int = field(default=16, metadata={"help": "lora r"})
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "lora dropout"})
    use_dora: bool = field(default=False)
    custom_start_layer: int = field(default=-1)
    custom_lora_name: str = field(default="MoEwithShared")
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "lora target modules"}
    )
    num_specific_lora: int = field(default=4)
    specific_rank_ratio: int = field(default=4)
    lora_activation: str = field(default=None)  # check activation status
    negative_slope: float = field(default=0.1)  # LeakyReLU
    ortho_loss_weight: float = field(default=0.0)  # orthogonal loss

    # progressive router config
    router_loss_type: str = field(default="margin")  # margin or bce
    router_loss_weight: float = field(default=0.0)  # progressive router loss
    router_loss_margin: float = field(default=0.2)  # progressive router loss
    router_warmup_ratio: float = field(default=0.1)
    router_transition_ratio: float = field(default=0.5)
    router_schedule_type: str = field(default="cosine")
    disable_routing_loss: bool = field(default=False)

    # router
    router_type: str = field(default="linear")
    routing_mode: str = field(default="soft")
    routing_activation_soft: str = field(default="softmax")
    filter_absent_modality: bool = field(default=True)
    normalize_routing_weight: bool = field(default=False)

    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help":
            'The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)).'
        },
    )


@dataclass
class DataArguments:
    data_root: str = field(default=None)
    train_yaml: str = field(default=None, metadata={"help": "yaml file for training datasets"})
    dataset_download: bool = field(default=False, metadata={"help": "whether to download datasets"})
    max_text_len: Optional[int] = field(
        default=256,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer"
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    append_eos_token: bool = field(
        default=True,
        metadata={
            "help": "append eos token to query and passage, this is currently used for repllama"
        }
    )
    append_gen_prompt: bool = field(
        default=False,
        metadata={
            "help": "append gen token to query and passage, this is currently used for repllama"
        }
    )

    # if true, will use adaptive number of visual tokens depending on its image/video inputs,
    #  which is limited by minimum and maximum number of pixels.
    apply_native_resolution: bool = field(default=True)
    image_min_pixels: int = field(default=64 * 28 * 28)
    image_max_pixels: int = field(default=1000 * 28 * 28)
    video_min_pixels: int = field(default=100 * 28 * 28)
    video_max_pixels: int = field(default=200 * 28 * 28)

    # otherwise, image and videos are resized to a fixed resolution.
    image_resized_height: int = field(default=616, metadata={"help": "resized image height"})
    image_resized_width: int = field(default=616, metadata={"help": "resized image height"})
    video_resized_height: int = field(default=280, metadata={"help": "resized video height"})
    video_resized_width: int = field(default=280, metadata={"help": "resized video height"})
    video_nframes: int = field(default=8, metadata={"help": "number of video frames"})


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.02)

    # gradcache options
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=4)

    # options: base, custom_lora
    training_component: str = field(default="base")
