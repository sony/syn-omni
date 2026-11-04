import logging

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import yaml

from torch.utils.data import Dataset

from .data import DATASET_REGISTRY

logger = logging.getLogger(__name__)


def build_train_datasets(data_args):
    assert data_args.train_yaml is not None
    with open(data_args.train_yaml, "r") as f:
        train_yaml = yaml.safe_load(f)

    defaults_cfg = train_yaml.get("defaults", {}) or {}
    datasets_cfg = train_yaml.get("datasets", {}) or {}

    dataset_list = []
    for view_key, cfg in datasets_cfg.items():
        assert cfg, cfg
        cfg["view"] = view_key
        dataset_list.append(cfg)

    train_dataset = MultiTrainDatasetFromConfig(
        dataset_list=dataset_list,  # a list of configs
        data_args=data_args,
        defaults=defaults_cfg,
    )
    return train_dataset


def _update_dataset_config(
    cfg: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge per-dataset cfg with global defaults.
    dataset-level value wins if not None. otherwise, fall back to defaults
    """
    defaults = defaults or {}
    cfg = deepcopy(cfg)

    # Top-level fallbacks
    if cfg.get("num_proc") is None:
        cfg["num_proc"] = defaults.get("num_proc", None)

    if cfg.get("seed") is None:
        cfg["seed"] = defaults.get("seed", None)

    # prefixes fallback
    default_prefixes = defaults.get("prefixes", {}) or {}
    default_query_prefix = default_prefixes.get("query", "")
    default_corpus_prefix = default_prefixes.get("corpus", "")

    cfg.setdefault("query", {})
    cfg.setdefault("corpus", {})

    # Query prefix fallback: cfg.query.prefix -> defaults.prefixes.query
    if cfg["query"].get("prefix") is None:
        cfg["query"]["prefix"] = default_query_prefix

    # Corpus prefix fallback: cfg.corpus.prefix -> defaults.prefixes.corpus
    if cfg["corpus"].get("prefix") is None:
        cfg["corpus"]["prefix"] = default_corpus_prefix

    # Ensure modalities dicts exist
    cfg["query"].setdefault("modalities", {})
    cfg["corpus"].setdefault("modalities", {})

    return cfg


class MultiTrainDatasetFromConfig(Dataset):
    """
    Multi-dataset wrapper built from YAML config.
    """

    def __init__(
        self,
        dataset_list: List[Dict[str, Any]],
        data_args: Any,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        if not dataset_list:
            raise ValueError("dataset_list is empty")

        self.data_args = data_args
        self.defaults = defaults or {}

        self.sub_datasets: List[Dataset] = []
        self.sub_meta: List[Dict[str, Any]] = []

        self._build_subdatasets(dataset_list)
        self._cumulative_sizes = self._compute_cumulative_sizes()

        logger.info(
            "Built `MultiTrainDatasetFromConfig`: %d sub-datasets, total length=%d",
            len(self.sub_datasets),
            len(self),
        )

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        ds_idx, local_idx = self._locate(index)
        return self.sub_datasets[ds_idx][local_idx]

    def _build_subdatasets(self, dataset_list: List[Dict[str, Any]]) -> None:
        for cfg in dataset_list:
            logger.info("Initializing `%s` dataset...", cfg.get("view"))
            dataset, meta = self._build_single_dataset(cfg)
            self.sub_datasets.append(dataset)
            self.sub_meta.append(meta)

            logger.info(
                "Registered dataset `%s`, original size=%d,  subset length=%d", cfg.get("view"),
                dataset.orig_length, len(dataset)
            )

    def _build_single_dataset(self, cfg: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        cfg = _update_dataset_config(cfg, self.defaults)

        # try to build with default retrieval dataset class by default if not specified
        build_cls = DATASET_REGISTRY[cfg.pop("build_cls_name", "base_retrieval")]
        assert self.data_args.data_root is not None
        dataset = build_cls(data_args=self.data_args, cfg=cfg)

        meta = {
            "view": cfg["view"],
            "data": cfg.get("data", None),
            "name": cfg.get("name", None),
            "modality_composition": cfg.get("modality_composition", None),
            "original size": dataset.orig_length,
            "subset size": len(dataset),
        }

        return dataset, meta

    def _compute_cumulative_sizes(self) -> List[int]:
        sizes = []
        total = 0
        for ds in self.sub_datasets:
            total += len(ds)
            sizes.append(total)
        return sizes

    def _locate(self, global_index: int) -> Tuple[int, int]:
        prev = 0
        for i, end in enumerate(self._cumulative_sizes):
            if global_index < end:
                return i, global_index - prev
            prev = end
        raise RuntimeError("Index mapping failed")
