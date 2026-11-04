import logging

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from PIL import Image
from qwen_omni_utils import process_mm_info
from transformers import ProcessorMixin

from tevatron.retriever.arguments import DataArguments

logging.getLogger("qwen_omni_utils.v2_5.vision_process").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _build_modality_mapping_from_messages(
    messages: List[List[Dict[str, Any]]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build explicit modality->batch mapping from Qwen chat messages.
    Specify the indices for each modality with their counts in the batch, used for
    gradcache to split this batch into chunk and assign each modality to each chunk
    properly.

    Returns a dict with:
      - image_batch_idx: LongTensor[num_image_slots]
      - video_batch_idx: LongTensor[num_video_slots]
      - audio_batch_idx: LongTensor[num_audio_slots]
      - image_counts: LongTensor[batch_size]  # per sample slots
      - video_counts: LongTensor[batch_size]
      - audio_counts: LongTensor[batch_size]
    """
    bsz = len(messages)

    image_batch_idx: List[int] = []
    video_batch_idx: List[int] = []
    audio_batch_idx: List[int] = []

    image_counts = torch.zeros(bsz, dtype=torch.long, device=device)
    video_counts = torch.zeros(bsz, dtype=torch.long, device=device)
    audio_counts = torch.zeros(bsz, dtype=torch.long, device=device)

    for b, msg in enumerate(messages):
        if not msg or "content" not in msg[0]:
            print("`collator.py` something went wrong")
            continue
        content = msg[0]["content"] or []
        for item in content:
            t = item.get("type", None)
            if t == "image":
                image_batch_idx.append(b)
                image_counts[b] += 1
            elif t == "video":
                video_batch_idx.append(b)
                video_counts[b] += 1
            elif t == "audio":
                audio_batch_idx.append(b)
                audio_counts[b] += 1

    return {
        "image_batch_idx": torch.tensor(image_batch_idx, dtype=torch.long, device=device),
        "video_batch_idx": torch.tensor(video_batch_idx, dtype=torch.long, device=device),
        "audio_batch_idx": torch.tensor(audio_batch_idx, dtype=torch.long, device=device),
        "image_counts": image_counts,
        "video_counts": video_counts,
        "audio_counts": audio_counts,
    }


MMDict = Dict[str, Any]  # keys: (prompt, text, image, video, audio), or content


def _build_modality_masks(modality_dicts: List[MMDict], device=None):
    # Returns: a list of indicators for 'T', 'I', 'V', 'A'
    indicators = [[0, 0, 0, 0] for _ in range(len(modality_dicts))]
    modality_order = ["text", "image", "video", "audio"]
    for i, modality_dict in enumerate(modality_dicts):
        for j, modality in enumerate(modality_order):
            if modality_dict.get(modality, None) is not None:
                indicators[i][j] = 1
    indicators = torch.Tensor(indicators).long().to(device=device)
    return indicators


@dataclass
class MultiModalTrainCollator:
    """
    Collator for multimodal (text/image/video/audio) retrieval training.

    Assumption:
      - each feature is a (query_dict, passage_dict) tuple
      - query_dict / passage_dict have keys: (prompt, text, image, video, audio)
      - content: dict[list]
    """
    data_args: DataArguments
    processor: ProcessorMixin

    def _truncate_text(self, text: str, max_len: int) -> str:
        # encode w/ truncation then decode back to string
        return self.processor.tokenizer.decode(
            self.processor.tokenizer.encode(text, max_length=max_len, truncation=True)
        )

    def _build_content(self, mm_dict, max_text_len):
        prompt = (mm_dict.get("prompt", None) or "").strip()
        text = mm_dict.get("text", None)
        image = mm_dict.get("image", None)
        video = mm_dict.get("video", None)
        audio = mm_dict.get("audio", None)

        content = []

        if image is not None:
            resize_params = {}
            if self.data_args.apply_native_resolution:
                resize_params["min_pixels"] = self.data_args.image_min_pixels
                resize_params["max_pixels"] = self.data_args.image_max_pixels
            else:
                resize_params["resized_height"] = self.data_args.image_resized_height
                resize_params["resized_width"] = self.data_args.image_resized_width
            content.append({"type": "image", "image": image, **resize_params})

        if video is not None:
            resize_params = {}
            if self.data_args.apply_native_resolution:
                resize_params["min_pixels"] = self.data_args.video_min_pixels
                resize_params["max_pixels"] = self.data_args.video_max_pixels
            else:
                resize_params["resized_height"] = self.data_args.video_resized_height
                resize_params["resized_width"] = self.data_args.video_resized_width

            content.append(
                {
                    "type": "video",
                    "video": video,
                    "nframes": self.data_args.video_nframes,
                    **resize_params
                }
            )

        if audio is not None:
            content.append({"type": "audio", "audio": audio})

        # append the texts on the last
        text = prompt if text is None else (prompt + " " + text).strip()
        if ((image is not None) or (video is not None) or (audio is not None)):
            # prepend a spacing when any other preceding modality exists
            text = " " + text

        text = self._truncate_text(str(text), max_text_len)
        content.append({"type": "text", "text": text})

        if len(content) == 0:
            content.append({"type": "text", "text": " "})  # prevent none input
        return content

    def _build_inputs_from_mm_dicts(
        self,
        mm_dicts: Sequence[MMDict],
        *,
        max_text_len: int,
    ):
        # Convert list of multimodal dicts into processor inputs.
        messages = []
        for d in mm_dicts:
            # modality dictionary
            content = self._build_content(d, max_text_len)
            messages.append([{"role": "user", "content": content}])

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=self.data_args.append_gen_prompt
            ) for m in messages
        ]

        if self.data_args.append_eos_token:
            texts = [t + "<|endoftext|>" for t in texts]

        audio_inputs, image_inputs, video_inputs = process_mm_info(
            messages, use_audio_in_video=False
        )

        if video_inputs is not None and len(video_inputs):
            for i, v in enumerate(video_inputs):
                if isinstance(v, list) and isinstance(v[0], Image.Image):
                    # PIL -> numpy (H, W, 3), stack to (T, H, W, 3) in case of 'frames'
                    video = np.stack([np.array(img) for img in v], axis=0)
                    video_inputs[i] = torch.from_numpy(video).permute(0, 3, 1, 2)

        processed_inputs = self.processor(
            text=texts,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="longest",
        )
        return processed_inputs, messages, texts

    def __call__(self, features):
        all_queries = [f[0] for f in features]
        all_passages = [f[1] for f in features]

        query_inputs, query_messages, query_texts = self._build_inputs_from_mm_dicts(
            all_queries, max_text_len=self.data_args.max_text_len
        )
        passage_inputs, passage_messages, corpus_texts = self._build_inputs_from_mm_dicts(
            all_passages, max_text_len=self.data_args.max_text_len
        )
        query_masks = _build_modality_masks(all_queries)
        passage_masks = _build_modality_masks(all_passages)

        assert "input_ids" in query_inputs
        device = query_inputs["input_ids"].device

        # modality mappings: specify the indices for each modality with counts in the batch
        query_mm = _build_modality_mapping_from_messages(query_messages, device=device)
        passage_mm = _build_modality_mapping_from_messages(passage_messages, device=device)

        # now, it should be returned as dict for clean code?
        return (query_inputs, query_masks, query_mm), (passage_inputs, passage_masks, passage_mm)


COLLATOR_DICT = {
    "base": MultiModalTrainCollator,
    "custom_lora": MultiModalTrainCollator,
}


def build_collator(collator_name: str, *args, **kwargs):
    collator_name = collator_name or "base"
    assert collator_name in COLLATOR_DICT
    return COLLATOR_DICT[collator_name](*args, **kwargs)
