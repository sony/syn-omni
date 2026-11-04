import glob
import logging
import os

import numpy as np

from datasets import load_dataset

from .base_dataset_class import BaseDataset

logger = logging.getLogger(__name__)

VRET_QRY_PROMPT = "Find a video that contains the following visual content: "
VRET_TGT_PROMPT = "Understand the content of the provided video."
QA_QUERY_PROMPT = "Answer a question based on the content of a video. "


class LLaVAHoundCaptionDataset(BaseDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        data_files = os.path.join(self.asset_prefix, self.data_files)
        self.dataset = load_dataset(
            "json", split="train", data_files=data_files, num_proc=self.num_proc
        )

    def _build_modality_dict(self, row, **kwargs):
        video_inst, caption = row["conversations"][0]["value"], row["conversations"][1]["value"]
        assert "<video>" in video_inst  # video token

        # get frames
        frames_dir = os.path.join(self.asset_prefix, "train_300k", row["video"])
        assert os.path.exists(frames_dir)
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpeg")))
        if len(frames) > self.data_args.video_nframes:
            # uniformly sample indices over the full range
            idxs = np.linspace(0, len(frames) - 1, num=self.data_args.video_nframes, dtype=int)
            frames = [frames[i] for i in idxs]
        # lengths?
        assert len(frames) == self.data_args.video_nframes, row

        if self.data_mode == "v2t":
            # split texts into video token
            video_split = video_inst.split("<video>")
            left, right = video_split[0].strip(), video_split[1].strip()
            assert (not left) or (not right)

            instruction = left if left else right
            formatted_query = {"prompt": instruction, "video": frames}
            formatted_pos_doc = {"text": caption}

        else:
            assert self.data_mode == "t2v"
            formatted_query = {"prompt": VRET_QRY_PROMPT, "text": caption}
            formatted_pos_doc = {"prompt": VRET_TGT_PROMPT, "video": frames}

        return formatted_query, formatted_pos_doc


class LLaVAHoundQADataset(BaseDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        data_files = os.path.join(self.asset_prefix, self.data_files)
        self.dataset = load_dataset(
            "json", split="train", data_files=data_files, num_proc=self.num_proc
        )

    def _build_modality_dict(self, row, **kwargs):
        question, answer = row["conversations"][0]["value"], row["conversations"][1]["value"]
        assert "<video>" in question  # video token

        # get frames
        frames_dir = os.path.join(self.asset_prefix, "train_300k", row["video"])
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpeg")))
        if len(frames) > self.data_args.video_nframes:
            # uniformly sample indices over the full range
            idxs = np.linspace(0, len(frames) - 1, num=self.data_args.video_nframes, dtype=int)
            frames = [frames[i] for i in idxs]
        assert len(frames) and len(frames) == self.data_args.video_nframes

        # split texts into video token
        video_split = question.split("<video>")
        left, right = video_split[0].strip(), video_split[1].strip()
        assert (not left)

        formatted_query = {"prompt": QA_QUERY_PROMPT, "text": right, "video": frames}
        formatted_pos_doc = {"text": answer}
        return formatted_query, formatted_pos_doc
