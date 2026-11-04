import json
import logging
import os

import torchaudio.functional as F

from datasets import Dataset as HF_Dataset
from torchcodec.decoders import AudioDecoder

from tevatron.retriever.datasets.hf_utils import load_hf_dataset_via_git_lfs

from .base_dataset_class import BaseDataset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
QA_QUERY_PROMPT = "Answer a question based on the content of the provided music. "


class MusicInstructDataset(BaseDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        data_files = os.path.join(self.asset_prefix, self.data_files)

        anns = json.load(open(data_files, "r"))["QA"]
        anns = [ann for ann in anns if ann["is_audioset_eval"] is False]

        # load musiccaps data for loading wav files
        musiccaps_data = load_hf_dataset_via_git_lfs(
            repo_id="CLAPv2/MusicCaps",
            data_root=self.data_root,
            split_globs="*.parquet",
            split="train",
        )

        # musiccaps dataset's youtube id -> index
        musiccaps_ytid2ind = {}
        for ind, ytid in enumerate(musiccaps_data["index"]):
            musiccaps_ytid2ind[ytid.replace(".wav", "")] = ind

        anns = [ann for ann in anns if ann["ytid"] in musiccaps_ytid2ind.keys()]
        self.dataset = HF_Dataset.from_list(anns)

        self._musiccaps_data = musiccaps_data
        self._musiccaps_ytid2ind = musiccaps_ytid2ind

    def _build_modality_dict(self, row, idx=None):
        question, answer = row["question"], row["answer"]

        audio = self._musiccaps_data[self._musiccaps_ytid2ind[row["ytid"]]]["audio"]
        assert audio

        # torchcodec AudioDecoder with resample
        assert isinstance(audio, AudioDecoder)
        audio = audio.get_all_samples()
        wav = audio.data  # shape: (C, T) or (T,)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # (1, T)
        if audio.sample_rate != SAMPLE_RATE:
            wav = F.resample(wav, orig_freq=audio.sample_rate, new_freq=SAMPLE_RATE)
        audio = wav.cpu().numpy()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)  # to mono audio

        formatted_query = {"prompt": QA_QUERY_PROMPT, "text": question, "audio": audio}
        formatted_pos_doc = {"text": answer}

        return formatted_query, formatted_pos_doc
