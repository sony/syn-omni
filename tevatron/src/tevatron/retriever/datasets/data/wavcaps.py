import json
import os

import torchaudio
import torchaudio.functional as F

from datasets import Dataset as HF_Dataset

from .base_dataset import BaseRetrievalDataset

SAMPLE_RATE = 16000

AUDIO_PREFIX = "Zip_files/AudioSet_SL/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac"  # noqa
A2T_QUERY_INSTRUCT = "Find a text that best describes what can be heard in the provided audio."  # noqa
T2A_QUERY_INSTRUCT = "Find an audio clip that matches the following description: "
T2A_POS_INSTRUCT = "Understand what can be heard in the provided audio."

TO_REMOVE_IDS = [
    "YVcu0pVF1npM.flac", "Y06-g5jz-OGc.flac", "Y3sSblRfEG2o.flac", "YWudGD6ZHRoY.flac",
    "YFli8wjBFV2M.flac", "YmW3S0u8bj58.flac"
]  # no audio files


def _read_flac_audio(audio_file):
    wav, sample_rate = torchaudio.load(audio_file)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # (1, T)
    if sample_rate != SAMPLE_RATE:
        wav = F.resample(wav, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    audio = wav.cpu().numpy()
    if audio.ndim == 2:
        audio = audio.mean(axis=0)  # to mono audio
    return audio


class WavCapsDataset(BaseRetrievalDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)
        self.audio_prefix = os.path.join(self.asset_prefix, AUDIO_PREFIX)
        self.failed_ids = []

    def _build_dataset(self):
        ann_file = os.path.join(self.asset_prefix, "json_files/AudioSet_SL/as_final.json")
        anns = json.load(open(ann_file, "r"))["data"]

        filtered_anns = []
        for ann in anns:
            ann["id"] = ann["id"].replace(".wav", ".flac")
            if ann["id"] in TO_REMOVE_IDS:
                continue
            filtered_anns.append(ann)
        self.dataset = HF_Dataset.from_list(filtered_anns)

    def _build_modality_dict(self, row, **kwargs):
        audio_file = os.path.join(self.audio_prefix, str(row["id"]))
        audio = _read_flac_audio(audio_file)
        caption = row["caption"]

        if "a2t" in self.view:
            formatted_query = {"audio": audio, "prompt": A2T_QUERY_INSTRUCT}
            formatted_pos_doc = {"text": caption}
        else:
            "t2a" in self.view
            formatted_query = {"prompt": T2A_QUERY_INSTRUCT, "text": caption}
            formatted_pos_doc = {"audio": audio, "prompt": T2A_POS_INSTRUCT}

        return formatted_query, formatted_pos_doc
