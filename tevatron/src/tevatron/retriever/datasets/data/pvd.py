import json
import logging
import os

from datasets import Dataset as HF_Dataset
from datasets import Features, Value

from tevatron.retriever.datasets.utils import load_video_frames

from .base_dataset import BaseRetrievalDataset

logger = logging.getLogger(__name__)

V2T_QUERY_INSTRUCT = "Find a text that best describes the visual content of the provided video."  # noqa
T2V_QUERY_INSTRUCT = "Find a video that contains the following visual content: "
T2V_POS_INSTRUCT = "Understand the content of the provided video."

features = Features(
    {
        "caption": Value("string"),
        "video_id": Value("int64"),
        "duration": Value("float32"),
        "fps": Value("float32"),
    }
)


class PVDataset(BaseRetrievalDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        ann_path = os.path.join(self.asset_prefix, self.data_files)
        assert os.path.exists(ann_path), \
            f"no pvd videos prepared: {ann_path}; run with --dataset_download True"
        anns = json.load(open(ann_path, "r"))
        self.dataset = HF_Dataset.from_list(anns, features=features)

    def _build_modality_dict(self, row, **kwargs):
        video_id = str(row["video_id"])
        frame_path = os.path.join(self.asset_prefix, "frames", video_id)
        frames = load_video_frames(frame_path, self.data_args.video_nframes)
        assert len(frames)

        if "v2t" in self.view:
            formatted_query = {"video": frames, "prompt": V2T_QUERY_INSTRUCT}
            formatted_pos_doc = {"text": row["caption"]}
        else:
            "t2v" in self.view
            formatted_query = {"prompt": T2V_QUERY_INSTRUCT, "text": row["caption"]}
            formatted_pos_doc = {"video": frames, "prompt": T2V_POS_INSTRUCT}

        return formatted_query, formatted_pos_doc
