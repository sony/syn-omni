import logging
import os

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
import torchaudio.functional as F

from PIL import Image
from qwen_omni_utils import process_mm_info
from torchcodec.decoders import AudioDecoder, VideoDecoder
from transformers import ProcessorMixin

from src.arguments import DataArguments, ModelArguments
from src.wave_model.processor import WAVE


def load_audio_from_torchcodec(audio, sampling_rate=16000):
    # torchcodec AudioDecoder with resample
    audio = audio.get_all_samples()
    wav = audio.data  # shape: (C, T) or (T,)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # (1, T)
    if audio.sample_rate != sampling_rate:
        wav = F.resample(wav, orig_freq=audio.sample_rate, new_freq=sampling_rate)
    audio = wav.cpu().numpy()
    if audio.ndim == 2:
        audio = audio.mean(axis=0)  # to mono audio
    return audio


def load_video_from_torchcodec(video_bytes, target_nframes=8):
    decoder = VideoDecoder(source=video_bytes)
    num_frames = decoder.metadata.num_frames
    assert num_frames >= target_nframes

    sample_inds = np.linspace(0, num_frames - 1, target_nframes, dtype=int)
    video = decoder.get_frames_at(indices=sample_inds).data

    # TCHW -> THWC
    video = video.permute(0, 2, 3, 1).cpu().numpy()
    pil_images = [Image.fromarray(frame) for frame in video]
    return pil_images


T = TypeVar("T")


def _unwrap(x: T | list[T] | None, ) -> T | None:
    """
    Unwrap a singleton list by one level.

    Examples:
        None                  -> None
        [value]               -> value
        [[a, b, c]]           -> [a, b, c]
        [[[a, b, c]]]         -> [[a, b, c]]
        [a, b, c]             -> [a, b, c]
    """
    if x is None:
        return None

    if isinstance(x, list) and len(x) == 1:
        return x[0]

    return x


logger = logging.getLogger(__name__)


@dataclass
class MultimodalEvalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    encode_side: str

    def _get_batch_inputs(self, batch, keyname):
        # batched input; {query_text, query_image, query_audio, pos_text, pos_image, pos_audio}
        processed = []  # a flattened modality list of either query or corpus;
        for ex in batch:
            if ex is None or not ex:
                raise RuntimeError  # something went wrong

            # a list of text and image, in length of candidate. len(query)=1
            ex_text, ex_images = ex[f"{keyname}_text"], ex[f"{keyname}_image"]
            ex_audio = ex.get(f"{keyname}_audio", [None] * len(ex_text))
            assert isinstance(ex_text, list), ex
            assert isinstance(ex_images, list), ex
            assert isinstance(ex_audio, list), ex

            for text, image, audio in zip(ex_text, ex_images, ex_audio):
                # single image/text/audio instance
                if text is None:
                    text = " "

                item_dict = {"text": text, "image": None, "video": None, "audio": None}
                if image is not None:

                    assert isinstance(image, dict)
                    paths = image.get("paths", None)
                    if paths is None and "bytes" in image:
                        # a list of converted pil images
                        item_dict["video"] = load_video_from_torchcodec(image["bytes"])

                    else:
                        assert len(image["paths"]) > 0
                        num_images = len(image["paths"])
                        if num_images > 1:  # video
                            assert all(os.path.exists(p) for p in image["paths"])  # path debug
                            item_dict["video"] = image["paths"]  # a list of frame dirs
                        else:  # image
                            assert len(image["paths"]) == 1
                            assert os.path.exists(image["paths"][0])  # path debug
                            item_dict["image"] = image["paths"][0]  # a single str path

                if audio is not None:
                    if isinstance(audio, dict) and "bytes" in audio:
                        audio = AudioDecoder(source=audio["bytes"])  # load from bytes
                        audio = load_audio_from_torchcodec(audio)  # convert to numpy with resample
                    else:
                        assert isinstance(audio, str) and os.path.exists(audio)
                    item_dict["audio"] = audio

                processed.append(item_dict)
        return processed

    def _truncate_text(self, text: str, max_len: int) -> str:
        # encode w/ truncation then decode back to string
        return self.processor.tokenizer.decode(
            self.processor.tokenizer.encode(text, max_length=max_len, truncation=True)
        )

    # content, message, apply chat template / process_mm_info
    def _build_content(self, mm_dict, max_text_len):
        text = mm_dict.get("text", None)
        image = mm_dict.get("image", None)
        video = mm_dict.get("video", None)
        audio = mm_dict.get("audio", None)

        content = []  # insert modality content into each token position

        # prepend modality tokens first, as in mmeb-eval benchmark
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

            # num_frames fixed to 8 for evaluation
            content.append({"type": "video", "video": video, "nframes": 8, **resize_params})

        if audio is not None:
            # path or numpy array
            content.append({"type": "audio", "audio": audio})

        if text is not None:
            assert isinstance(text, str), text
            if ((image is not None) or (video is not None) or (audio is not None)):
                text = " " + text  # prepend a spacing after modality tokens
            text = self._truncate_text(str(text), max_text_len)
            content.append({"type": "text", "text": text})

        if len(content) == 0:
            content.append({"type": "text", "text": " "})  # prevent none input
        return content

    def _process_batch(self, batch, keyname):
        # a flattened list of dict(text, image, video, audio)
        keep_input_raw_wav = (self.model_args.model_backbone == WAVE)
        batch_inputs = self._get_batch_inputs(batch, keyname)

        # build a batched chat-style messages
        messages = []
        for d in batch_inputs:
            content = self._build_content(d, max_text_len=self.data_args.max_len)
            messages.append([{"role": "user", "content": content}])

        sample_records = []
        texts = []  # for debug
        for m in messages:
            # single item in batch
            text = self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            text = _unwrap(text)
            texts.append(text)

            # handle one element in the batch
            audio, image, video = process_mm_info(m, use_audio_in_video=False)
            audio, image, video = _unwrap(audio), _unwrap(image), _unwrap(video)

            # handle video; PIL List to Tensor
            if video is not None and isinstance(video, list) and isinstance(video[0], Image.Image):
                video = np.stack([np.array(img) for img in video], axis=0)
                video = torch.from_numpy(video).permute(0, 3, 1, 2)

            sample_records.append({
                "text": text,
                "audio": audio,
                "image": image,
                "video": video,
            })

        # assert len(messages) == 1  # eval purpose
        texts = [r["text"] for r in sample_records]
        images = [r["image"] for r in sample_records]
        videos = [r["video"] for r in sample_records]
        audios = [r["audio"] for r in sample_records]

        # ## chat template debug
        # if is_main():
        #     print(f"keyname: {keyname}")
        #     for t in texts[:1]:
        #         print(t)
        #         print()
        #     print()
        # ##

        processed_inputs = self.processor(
            text=texts,
            audio=audios if audios[0] is not None else None,
            images=images if images[0] is not None else None,
            videos=videos if videos[0] is not None else None,
            return_tensors="pt",
            padding="longest",
        )

        # process
        # assert len(audios) == 1
        if audios[0] is not None and keep_input_raw_wav:
            pass
            '''
            raw_wavs = []
            for aud in audios:
                assert isinstance(aud, torch.Tensor)
                wav = aud.detach().float().cpu().reshape(-1)
                raw_wavs.append(wav)
            processed_inputs["input_raw_wav"] = torch.stack(raw_wavs, dim=0)
            '''

        feats = processed_inputs.get("input_features", None)
        if isinstance(feats, torch.Tensor) and feats.dim() == 3:
            assert feats.shape[1] == 128
        processed_inputs.pop("audio_attention_mask", None)
        processed_inputs.pop("audio_feature_lengths", None)

        return processed_inputs

    def __call__(self, examples, **kwargs):
        input_key = "query" if self.encode_side == "qry" else "cand"
        processed_inputs = self._process_batch(examples, input_key)

        dataset_infos = [e["dataset_infos"] for e in examples]
        return processed_inputs, dataset_infos
