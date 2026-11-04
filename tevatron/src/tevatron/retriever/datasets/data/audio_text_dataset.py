from tevatron.retriever.datasets.hf_utils import load_hf_dataset_via_git_lfs

from .base_dataset import BaseRetrievalDataset


class AudioTextDataset(BaseRetrievalDataset):
    """
    git lfs clone-based downloading and load datasets from local directory
    """

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        self.dataset = load_hf_dataset_via_git_lfs(
            repo_id=self.name,
            data_root=self.data_root,
            split_globs=self.glob_pattern,
            split=self.split
        )
