import logging

from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
from src.model.vlm_backbone.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from src.utils.basic_utils import print_master

# from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'
INTERNVIDEO2 = 'internvideo2'
GME = 'gme'  # QWEN2-VL
LamRA = 'lamra'  # QWEN2-VL
LamRA_QWEN2_5 = 'lamra_qwen25'  # QWEN2.5-VL
COLPALI = 'colpali'  # PaliGemma-3B
E5_V = 'e5_v'  # Llava_next
QWEN2_5_OMNI = "qwen2_5_omni"
WAVE = "wave"
NV_OMNIEMBED_NEMOTRON = "nv_omniembed_nemotron"
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'colpali': COLPALI,
    "qwen2_5_omni": QWEN2_5_OMNI,
    "nv_omniembed_nemotron": NV_OMNIEMBED_NEMOTRON,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    COLPALI: "",
    QWEN2_5_OMNI: "",
    WAVE: "",
    NV_OMNIEMBED_NEMOTRON: "",
}

VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    COLPALI: "",
    QWEN2_5_OMNI: "",
    WAVE: "",
    NV_OMNIEMBED_NEMOTRON: "",
}

VLM_AUDIO_TOKENS = {
    QWEN2_5_OMNI: "",
    WAVE: "",
    NV_OMNIEMBED_NEMOTRON: "",
}

backbone2model = {
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    """
    model_name_or_path = model_args.processor_name
    print_master(f'Loading processor from: {model_name_or_path}')

    if model_args.model_backbone == NV_OMNIEMBED_NEMOTRON:
        from transformers import AutoProcessor
        processor_cls = AutoProcessor
    else:
        processor_cls = Qwen2_5OmniProcessor

    processor = processor_cls.from_pretrained(
        model_name_or_path, trust_remote_code=True, local_files_only=True
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        hf_config.model_type = model_type
    assert hf_config.model_type in SUPPORTED_MODELS, f"{hf_config.model_type}.{SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def process_input_text(
    instruction,
    model_backbone,
    text=None,
    add_video_token=False,
    add_image_token=False,
    add_audio_token=False
):
    prompt = instruction
    if text:
        prompt = prompt + " " + text

    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + prompt

    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        # if image_token:
        prompt = image_token + prompt

    if add_audio_token:
        audio_token = VLM_AUDIO_TOKENS[model_backbone]
        # if audio_token:
        prompt = audio_token + prompt

    return prompt.strip()


process_vlm_inputs_fns = {}
