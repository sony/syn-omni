import os
import subprocess

from typing import Dict, Optional, Union

from datasets import DatasetDict, load_dataset

from .download_utils import sanitize_repo_dirname


def _run(cmd, cwd: Optional[str] = None):
    cmd = [str(c) for c in cmd]
    subprocess.check_call(cmd, cwd=cwd)


def hf_git_lfs_clone(
    repo_id: str,
    data_root: str,
    *,
    repo_type: str = "datasets",  # "datasets" or "models"
) -> str:
    """
    Clone a Hugging Face repo locally using git + LFS.
    - The repo is cloned into data_root/{last segment of repo_id}.
    """
    os.makedirs(data_root, exist_ok=True)

    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name)

    if os.path.isdir(os.path.join(local_dir, ".git")):
        return local_dir

    # Fresh clone
    url = f"https://huggingface.co/{repo_type}/{repo_id}"
    clone_cmd = ["git", "clone"]
    clone_cmd += [url, local_dir]
    print(f"cloning the repo {url} --> {data_root}")
    _run(clone_cmd)

    _run(["git", "lfs", "pull"], cwd=local_dir)
    return local_dir


def load_dataset_from_local_parquet_repo(
    repo_local_dir: str,
    *,
    data_type: str = "parquet",
    split_globs: Optional[Dict[str, Union[str, list[str]]]] = None,
    columns: Optional[list[str]] = None,
    split: str = None
) -> DatasetDict:
    """
    Load parquet shards from a locally cloned dataset repo and return a DatasetDict.
    - If split_globs is not provided, common directory structures are auto-detected.
    - Using glob patterns avoids explicitly listing thousands of shard filenames.
    """
    assert split_globs is not None
    if isinstance(split_globs, dict):
        abs_globs = {}
        for split, pat in split_globs.items():
            if isinstance(pat, list):
                abs_globs[split] = [os.path.join(repo_local_dir, p) for p in pat]
            else:
                abs_globs[split] = os.path.join(repo_local_dir, pat)
    else:
        abs_globs = os.path.join(repo_local_dir, split_globs)
    split_globs = abs_globs
    if data_type == "parquet":
        dataset = load_dataset(
            data_type,
            data_files=split_globs,
            columns=columns,
        )
    else:
        raise NotImplementedError
    if split is not None:
        assert split in dataset
    return dataset[split]


def load_hf_dataset_via_git_lfs(
    repo_id: str,
    data_root: str,
    *,
    data_type="parquet",
    repo_type: str = "datasets",
    split_globs: Optional[Dict[str, Union[str, list[str]]]] = None,
    columns: Optional[list[str]] = None,
    split: str = None
) -> DatasetDict:
    """
    End-to-end helper:
    repo_id -> clone with git+lfs into data_root -> load local parquet shards.
    """
    local_dir = hf_git_lfs_clone(repo_id, data_root, repo_type=repo_type)
    return load_dataset_from_local_parquet_repo(
        local_dir, data_type=data_type, split_globs=split_globs, columns=columns, split=split
    )
