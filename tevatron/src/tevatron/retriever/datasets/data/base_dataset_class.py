from __future__ import annotations
import random

from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from tevatron.retriever.datasets.asset_registry import ASSET_REGISTRY


# -----------------------------
# Base prototype
# -----------------------------
class BaseDataset(Dataset, ABC):

    def __init__(self, data_args, cfg):
        assert data_args is not None and cfg is not None
        self.data_args = data_args
        self.data_root = data_args.data_root

        self.cfg = cfg
        self.view = cfg.get("view", None)
        self.name = cfg.get("name", None)
        self.corpus_name = cfg.get("corpus_name", None)
        self.split = cfg.get("split", None)
        self.max_samples = cfg.get("max_samples", -1)
        self.data = cfg.get("data", None)
        self.data_mode = cfg.get("data_mode", None)
        self.config_name = cfg.get("config_name", None)
        self.data_files = cfg.get("data_files", None)
        self.glob_pattern = cfg.get("glob_pattern", None)
        self.num_proc = cfg.get("num_proc", 1)
        self.seed = cfg.get("seed", 42)
        self.modality_composition = cfg.get("modality_composition", None)
        assert self.modality_composition is not None

        self.query_cfg = cfg.get("query", {}) or {}
        self.corpus_cfg = cfg.get("corpus", {}) or {}

        # ---- check if any assets download necessary ----
        self.asset_prefix = None
        self._check_and_download_assets()

        # ---- Load dataset ----
        self.dataset = None
        self._build_dataset()

        # ---- filter invalid samples ----
        self._dataset_filter_fn()
        self.orig_length = self.dataset.num_rows

        # ---- Deterministic subsampling ----
        self._apply_max_samples()

        # Final length
        self._length = self.dataset.num_rows

    # -----------------------------
    # Required overrides
    # -----------------------------
    @abstractmethod
    def _build_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def _build_modality_dict(self, row, **kwargs):
        raise NotImplementedError

    # -----------------------------
    # Standard methods
    # -----------------------------
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        row = self.dataset[idx]
        if not isinstance(row, dict):
            row = dict(row)

        # idx: optional, and used in musicinstruct only.
        formatted_query, formatted_corpus = self._build_modality_dict(row, idx=idx)
        return formatted_query, formatted_corpus, self.modality_composition

    # -----------------------------
    # Common utilities
    # -----------------------------
    def _dataset_filter_fn(self):
        return

    def _check_and_download_assets(self):
        downloader = ASSET_REGISTRY.get(self.data, None)
        if downloader is not None:
            download = self.data_args.dataset_download
            self.asset_prefix = downloader(self.data_root, self.name, download=download)

    def _apply_max_samples(self) -> None:
        if self.max_samples is None:
            return

        n = int(self.max_samples)
        if n <= 0 or n >= self.dataset.num_rows:
            return

        rng = random.Random(self.seed)
        indices = list(range(self.dataset.num_rows))
        rng.shuffle(indices)
        self.dataset = self.dataset.select(indices[:n])
