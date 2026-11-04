# Not used.
import json
import logging
import os

import numpy as np

from datasets import Dataset as HF_Dataset

from .base_dataset import BaseRetrievalDataset

logger = logging.getLogger(__name__)

QUERY_INSTRUCT = "Find a video that matches the scene described by the given changes:"
TARGET_INSTRUCT = "Understand the content of the provided video."


def _load_video_frames(frame_prefix: str, frame_dir: str, num_frames: int):
    frame_path = os.path.join(frame_prefix, "finecvr", frame_dir)
    frames = sorted(os.listdir(frame_path))
    assert len(frames)

    if len(frames) < num_frames:
        idxs = list(range(len(frames))) + [len(frames) - 1] * (num_frames - len(frames))
    else:
        idxs = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
    frames = [os.path.join(frame_path, frames[i]) for i in idxs]
    return frames


class FineCVRDataset(BaseRetrievalDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        data_files = os.path.join(self.asset_prefix, self.data_files)
        anns = json.load(open(data_files, "r"))
        self.dataset = HF_Dataset.from_list(anns)

    def _build_modality_dict(self, row, **kwargs):
        source = _load_video_frames(self.asset_prefix, row["source"], self.data_args.video_nframes)
        target = _load_video_frames(self.asset_prefix, row["target"], self.data_args.video_nframes)

        formatted_query = {"video": source, "prompt": QUERY_INSTRUCT, "text": row["instruct"]}
        formatted_pos_doc = {"video": target, "prompt": TARGET_INSTRUCT}

        return formatted_query, formatted_pos_doc
