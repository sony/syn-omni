from typing import Any, Dict, List

from .audio_text_dataset import AudioTextDataset


def _filter_eval_samples(ds, num_proc: int | None = None, batched: bool = True):

    def _keep_mask(batch: Dict[str, List[Any]]):
        is_audioset_evals = batch["is_audioset_eval"]
        keep = [is_audioset_eval == "False" for is_audioset_eval in is_audioset_evals]
        return keep

    return ds.filter(_keep_mask, batched=True, num_proc=num_proc)


class MusicCapsDataset(AudioTextDataset):
    """same as base retrieval dataset, but additional filtering"""

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _dataset_filter_fn(self):
        assert self.dataset is not None
        self.dataset = _filter_eval_samples(self.dataset, num_proc=self.num_proc)
