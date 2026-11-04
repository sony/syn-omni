# Not used.
import json
import logging
import os

from datasets import Dataset as HF_Dataset

from tevatron.retriever.datasets.utils import load_video_frames

from .base_dataset import BaseRetrievalDataset

logger = logging.getLogger(__name__)


class AudioSetDataset(BaseRetrievalDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        data_files = os.path.join(self.asset_prefix, self.data_files)
        anns = json.load(open(data_files, "r"))["train"]
        self.dataset = HF_Dataset.from_list(anns)
        # self._check_paths()

    def _check_paths(self):
        for row in self.dataset:
            frame_path = os.path.join(self.asset_prefix, "frames", row["video_id"])
            audio_path = os.path.join(self.asset_prefix, "audio", f"{row['video_id']}.wav")
            assert os.path.exists(frame_path), frame_path
            assert os.path.exists(audio_path), audio_path

    def _build_modality_dict(self, row, **kwargs):
        frame_path = os.path.join(self.asset_prefix, "frames", row["video_id"])
        frames = load_video_frames(frame_path, self.data_args.video_nframes)
        audio_path = os.path.join(self.asset_prefix, "audio", f"{row['video_id']}.wav")
        audio = audio_path
        assert len(frames) and os.path.exists(audio_path)

        if "a2v" in self.view:
            QUERY_INSTRUCT = "Find a video that best matches the content of the given audio."
            TARGET_INSTRUCT = "Understand the content of the provided video."

            # inst
            formatted_query = {"audio": audio, "prompt": QUERY_INSTRUCT}
            formatted_pos_doc = {"video": frames, "prompt": TARGET_INSTRUCT}
        else:
            assert "v2a" in self.view
            QUERY_INSTRUCT = "Find a audio that best corresponds to the content of given video."  # noqa
            TARGET_INSTRUCT = "Understand what can be heard in the provided audio."

            formatted_query = {"video": frames, "prompt": QUERY_INSTRUCT}
            formatted_pos_doc = {"audio": audio, "prompt": TARGET_INSTRUCT}

        return formatted_query, formatted_pos_doc
