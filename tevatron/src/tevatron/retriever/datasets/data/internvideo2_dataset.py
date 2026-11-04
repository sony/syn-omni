# Not used.
import json
import logging
import os

from datasets import Dataset as HF_Dataset

from tevatron.retriever.datasets.utils import load_video_frames

from .base_dataset import BaseRetrievalDataset

logger = logging.getLogger(__name__)

AV2T_QUERY_INSTRUCT = "Find a text that best describes the visual and audio content of the provided video."  # noqa

T2AV_QUERY_INSTRUCT = "Find a video that contains the following content: "
T2AV_POS_INSTRUCT = "Understand the visual and audio content of the provided video."


class InternVideo2Dataset(BaseRetrievalDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)
        # self._check_paths()

    def _build_dataset(self):
        anns = []
        for data_file in self.data_files:
            anns.extend(json.load(open(os.path.join(self.asset_prefix, data_file), "r")))

        exclude_ids_file = os.path.join(self.asset_prefix, "exclude_ids.json")
        if os.path.exists(exclude_ids_file):
            remove_ids = json.load(open(exclude_ids_file, "r"))
            anns = [ann for ann in anns if ann["video_id"] not in remove_ids]
        self.dataset = HF_Dataset.from_list(anns)

    def _check_paths(self):
        for row in self.dataset:
            video_id = str(row["video_id"])
            frame_path = os.path.join(self.asset_prefix, "frames", video_id)
            audio_path = os.path.join(self.asset_prefix, "audio", f"{video_id}.wav")
            assert os.path.exists(frame_path), frame_path
            assert os.path.exists(audio_path), audio_path

    def _build_modality_dict(self, row, **kwargs):
        video_id = str(row["video_id"])
        frame_path = os.path.join(self.asset_prefix, "frames", video_id)
        frames = load_video_frames(frame_path, self.data_args.video_nframes)
        audio_path = os.path.join(self.asset_prefix, "audio", f"{video_id}.wav")
        audio = audio_path
        # assert len(frames) and os.path.exists(audio_path)

        if "av2t" in self.view:
            formatted_query = {"video": frames, "audio": audio, "prompt": AV2T_QUERY_INSTRUCT}
            formatted_pos_doc = {"text": row["caption"]}
        else:
            assert "t2av" in self.view
            formatted_query = {"prompt": T2AV_QUERY_INSTRUCT, "text": row["caption"]}
            formatted_pos_doc = {"video": frames, "audio": audio, "prompt": T2AV_POS_INSTRUCT}

        return formatted_query, formatted_pos_doc
