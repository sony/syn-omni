import logging

from src.utils.basic_utils import print_master

from .wave_official.qwenvl.data import Qwen2_5OmniProcessor

logger = logging.getLogger(__name__)

PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

QWEN2_5_OMNI = "qwen2_5_omni"
WAVE = "wave"
MODEL2BACKBONE = {"qwen2_5_omni": QWEN2_5_OMNI, "wave": WAVE}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {QWEN2_5_OMNI: "", WAVE: ""}
VLM_VIDEO_TOKENS = {QWEN2_5_OMNI: "", WAVE: ""}
VLM_AUDIO_TOKENS = {QWEN2_5_OMNI: "", WAVE: ""}

backbone2model = {}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    """
    model_name_or_path = model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    processor_path = model_args.processor_name if model_args.processor_name else model_name_or_path
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)
    processor.tokenizer.padding_side = "left"
    return processor


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
