import logging
import os
import random

from typing import Any, Dict, Tuple

import torchaudio.functional as F

from datasets import load_dataset
from torchcodec.decoders import AudioDecoder

from .base_dataset_class import BaseDataset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class BaseRetrievalDataset(BaseDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        self.dataset = load_dataset(
            path=self.name,
            name=self.config_name,
            split=self.split,
            data_files=self.data_files,
            num_proc=self.num_proc,
        )

    def _dataset_filter_fn(self):
        if self.data == "activitynet":
            # print_master(f"before filtering `{self.data}`: {len(self.dataset)}")
            to_remove = ['v_C4V6fqELvPY.mp4']
            assert self.dataset is not None

            def _keep_mask(batch):
                videos = batch["video"]
                keep = [video not in to_remove for video in videos]
                return keep

            self.dataset = self.dataset.filter(_keep_mask, batched=True, num_proc=self.num_proc)
            # print_master(f"after filtering `{self.data}`: {len(self.dataset)}")
        return

    def _join_asset_path(self, v: Any) -> Any:
        if v is None or not isinstance(v, str):
            return v

        assert self.asset_prefix is not None and isinstance(v, str)
        return os.path.join(self.asset_prefix, v)

    def _pack_side(
        self,
        row: Dict[str, Any],
        side_cfg: Dict[str, Any],
        prefix: str,
    ) -> Tuple[str, Any, Any, Any]:
        """
        Build (text, image, video, audio) using mapping:
          side_cfg["modalities"] = {"text": "<col>", "video": "<col>", ...}
        """
        modalities = (side_cfg or {}).get("modalities", {}) or {}

        # text: if not configured, keep empty string (legacy expects text always present)
        prompt = prefix
        if "text" in modalities:
            col = modalities["text"]
            if isinstance(col, list):  # assuming qa format, where the two columns are q and a.
                raise NotImplementedError  # currently not in use
                assert len(col) == 2
                qa_text = " ".join([row[c] for c in col])
                text = prefix + qa_text
            else:
                # simple retrieval setting
                assert isinstance(col, str)
                text_val = row.get(col, "") or ""
                assert text_val  # if specified, should exist
                if isinstance(text_val, list):
                    # multiple training captions: randomly sample one
                    text_val = text_val[random.randrange(len(text_val))]
                text = str(text_val)
        else:
            text = None

        # image/video/audio: if not configured -> None
        if "image" in modalities:
            image = row.get(modalities["image"], None)
            assert image
            image = self._join_asset_path(image)  # if stored as path string
        else:
            image = None

        if "video" in modalities:
            video = row.get(modalities["video"], None)
            assert video
            video = self._join_asset_path(video)  # if stored as path string

        else:
            video = None

        if "audio" in modalities:
            audio = row.get(modalities["audio"], None)
            assert audio

            # torchcodec AudioDecoder
            if isinstance(audio, AudioDecoder):
                audio = audio.get_all_samples()
                wav = audio.data  # shape: (C, T) or (T,)
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)  # (1, T)
                if audio.sample_rate != SAMPLE_RATE:
                    wav = F.resample(wav, orig_freq=audio.sample_rate, new_freq=SAMPLE_RATE)
                audio = wav.cpu().numpy()
                if audio.ndim == 2:
                    audio = audio.mean(axis=0)  # to mono audio

            # path string (.mp3 / .wav)
            else:
                audio_ext = (".mp3", ".wav", ".flac")
                assert (
                    isinstance(audio, str) and (audio.endswith(audio_ext))
                ), f"Unsupported audio format: {type(audio)}"
                audio = self._join_asset_path(audio)
        else:
            audio = None

        return {"prompt": prompt, "text": text, "audio": audio, "image": image, "video": video}

    def _build_modality_dict(self, row, **kwargs):
        q_prefix = str((self.query_cfg or {}).get("prefix", "") or "")
        p_prefix = str((self.corpus_cfg or {}).get("prefix", "") or "")

        formatted_query = self._pack_side(row, self.query_cfg, q_prefix)
        formatted_pos_doc = self._pack_side(row, self.corpus_cfg, p_prefix)

        return formatted_query, formatted_pos_doc
