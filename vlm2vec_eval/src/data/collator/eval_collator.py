import logging
import os

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio.functional as F

from PIL import Image
from qwen_omni_utils import process_mm_info
from torchcodec.decoders import AudioDecoder, VideoDecoder
from transformers import ProcessorMixin

from src.arguments import DataArguments, ModelArguments


def load_audio_from_torchcodec(audio, sampling_rate=16000):
    # torchcodec AudioDecoder with resample
    audio = AudioDecoder(source=audio).get_all_samples()
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
                    if paths is None and "bytes" in image:  # for daily-omni dataset
                        # a list of converted pil images
                        item_dict["video"] = load_video_from_torchcodec(image["bytes"])

                    elif (
                        paths is not None and len(paths) == 1 and paths[0].lower().endswith(".mp4")
                    ):
                        # a raw video file path (e.g. AVHBench); decode frames directly
                        assert os.path.exists(paths[0]), paths[0]
                        item_dict["video"] = load_video_from_torchcodec(paths[0])

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
                        # audio = AudioDecoder(source=audio["bytes"])  # load from bytes
                        audio = load_audio_from_torchcodec(audio["bytes"])
                    elif isinstance(audio, str) and audio.lower().endswith(".mp4"):
                        # a raw video file path (e.g. AVHBench); decode+resample its audio track
                        assert os.path.exists(audio), audio
                        audio = load_audio_from_torchcodec(audio)
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

        if "LCO-Embedding" in self.model_args.model_name:
            # append special prompt
            interleaved_prompt = "\nSummarize the above in one word:"
            content.append({"type": "text", "text": interleaved_prompt})

        if len(content) == 0:
            content.append({"type": "text", "text": " "})  # prevent none input
        return content

    def _process_batch(self, batch, keyname):
        # a flattened list of dict(text, image, video, audio)
        batch_inputs = self._get_batch_inputs(batch, keyname)

        def _unwrap(text):
            if isinstance(text, list):
                assert len(text) == 1 and isinstance(text[0], str)
                return text[0]
            assert isinstance(text, str)
            return text

        # build a batched chat-style messages
        messages = []
        for d in batch_inputs:
            content = self._build_content(d, max_text_len=self.data_args.max_len)
            messages.append([{"role": "user", "content": content}])

        texts = [
            _unwrap(
                self.processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=self.data_args.add_generation_prompt
                )
            ) for m in messages
        ]

        if self.data_args.add_eos_token:
            texts = [t + "<|endoftext|>" for t in texts]

        audio_inputs, image_inputs, video_inputs = process_mm_info(
            messages, use_audio_in_video=False
        )

        # ## chat template debug
        # if is_main():
        #     print(f"keyname: {keyname}")
        #     for t in texts[:1]:
        #         print(t)
        #         print()
        #     print()
        # ##

        # convert a list of PIL images as frames to tensor
        if video_inputs is not None and len(video_inputs):
            for i, v in enumerate(video_inputs):
                if isinstance(v, list) and isinstance(v[0], Image.Image):
                    # PIL -> numpy (H, W, 3), stack to (T, H, W, 3) -> (T,3,H,W)
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

        if self.model_args.model_backbone == "nv_omniembed_nemotron":
            # Expect (B, T, C=128) -> (B, C=128, T) for qwen2_5_omni audio.
            # specifically nvomniembed_nemotron
            feats = processed_inputs.get("input_features", None)
            fam = processed_inputs.get("feature_attention_mask", None)
            if feats is not None:
                assert isinstance(feats, torch.Tensor)
                assert fam is not None and isinstance(fam, torch.Tensor)
                if feats.shape[1] != 128 and feats.shape[2] == 128:
                    feats = feats.transpose(1, 2)
                assert fam.dim() == 2
                if fam.shape[1] != feats.shape[2]:
                    min_len = min(fam.shape[1], feats.shape[2])
                    feats = feats[:, :, :min_len]
                    fam = fam[:, :min_len]
                processed_inputs["input_features"] = feats
                processed_inputs["feature_attention_mask"] = fam

        return processed_inputs

    def __call__(self, examples, **kwargs):
        input_key = "query" if self.encode_side == "qry" else "cand"
        processed_inputs = self._process_batch(examples, input_key)

        dataset_infos = [e["dataset_infos"] for e in examples]
        return processed_inputs, dataset_infos
