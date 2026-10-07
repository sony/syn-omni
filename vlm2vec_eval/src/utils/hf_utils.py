import json
import os
import re
import subprocess

from typing import Dict, Optional, Union

from datasets import Dataset as HF_Dataset
from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download

from src.utils.dist_utils import barrier, is_main


def _run(cmd, cwd: Optional[str] = None):
    cmd = [str(c) for c in cmd]
    subprocess.check_call(cmd, cwd=cwd)


def _unzip(archive_path: str):
    # only run at main process
    if is_main():
        base = os.path.basename(archive_path)
        target_dir = os.path.dirname(archive_path)
        if archive_path.endswith(".zip"):
            _run(["unzip", "-q", base], cwd=target_dir)
        elif archive_path.endswith(".tar.gz"):
            _run(["tar", "-zxf", base], cwd=target_dir)
        elif archive_path.endswith(".tar"):
            _run(["tar", "-xf", base], cwd=target_dir)
        else:
            raise ValueError(f"Unknown archive type: {archive_path}")
    barrier()


def sanitize_repo_dirname(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id)


def resolve_hf_checkpoint(model_name_or_path: str) -> str:
    """
    Resolve a checkpoint source to a local directory.
    - If it's already a local directory, return it unchanged.
    - Otherwise, treat it as a HF Hub model repo id and snapshot_download it into
      the default HF cache (no explicit local_dir), returning the resolved cache path.
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    return snapshot_download(repo_id=model_name_or_path, repo_type="model")


def hf_snapshot_download_with_pattern(
    repo_id: str,
    data_root: str,
    pattern: str,
    repo_type: str = "dataset",  # "dataset" or "model"
):
    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name)
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id, repo_type=repo_type, allow_patterns=[pattern], local_dir=local_dir
    )
    return local_dir


def hf_git_lfs_clone(repo_id: str, local_dir: str = None, repo_type: str = "datasets") -> str:
    """
    Clone a Hugging Face repo locally using git + LFS.
    - The repo is cloned into data_root/{last segment of repo_id}.
    """
    assert local_dir is not None
    os.makedirs(local_dir, exist_ok=True)

    if os.path.isdir(os.path.join(local_dir, ".git")):
        return local_dir

    # Fresh clone
    url = f"https://huggingface.co/{repo_type}/{repo_id}"
    clone_cmd = ["git", "clone"]
    clone_cmd += [url, local_dir]
    print(f"cloning the repo {url} --> {local_dir}")
    _run(clone_cmd)

    _run(["git", "lfs", "pull"], cwd=local_dir)
    return local_dir


def load_dataset_from_local_parquet_repo(
    repo_local_dir: str,
    *,
    data_type: str = "parquet",
    split_globs: Optional[Dict[str, Union[str, list[str]]]] = None,
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
        dataset = load_dataset(data_type, data_files=split_globs)
        if split is not None:
            assert split in dataset
            dataset = dataset[split]

    elif data_type == "json":
        assert os.path.exists(split_globs)  # absolute json path
        dataset = HF_Dataset.from_list(json.load(open(split_globs, "r")))

    else:
        raise NotImplementedError

    return dataset


def load_hf_csv_dataset_local(
    repo_id,
    data_root,
    data_files=None,
    split_globs=None,
    download_method=None,
    repo_type="datasets",
    split="train",
):
    # assert download_method in ["snapshot", "git"], download_method
    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name)
    if not os.path.exists(local_dir):
        hf_git_lfs_clone(repo_id, local_dir=local_dir, repo_type="datasets")

    if not isinstance(data_files, list):
        data_files = [data_files]
    data_files = [os.path.join(local_dir, d) for d in data_files]
    if len(data_files) == 1:
        data_files = data_files[0]
    dataset = load_dataset("csv", data_files=data_files)
    if split is not None:
        dataset = dataset[split]
    return dataset, local_dir


def load_gtzan_dataset_local(repo_id, data_root):
    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name, "Data")
    if not os.path.exists(local_dir):
        zip_file = "gtzan-dataset-music-genre-classification.zip"
        hf_snapshot_download_with_pattern(repo_id, data_root, pattern=zip_file)
        zip_path = os.path.join(local_dir, zip_file)
        _unzip(zip_path)
    ann_file = "GTZANGenre.test.jsonl"
    hf_snapshot_download_with_pattern(repo_id, data_root, pattern=ann_file)
    ann_file = os.path.join(data_root, repo_name, ann_file)
    return load_dataset("json", data_files=ann_file)["train"], local_dir


def load_clotho_aqa(repo_id, data_root):
    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name)
    audio_root = os.path.join(local_dir, "audio_files")

    unzip_file = os.path.join(local_dir, "audio_files.zip")
    data_files = os.path.join(local_dir, "clotho_aqa_test.csv")
    if not os.path.exists(audio_root):
        local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        assert os.path.exists(unzip_file)
        _unzip(unzip_file)
        os.remove(unzip_file)
    dataset = load_dataset("csv", data_files=data_files)["train"]
    return dataset, local_dir


def download_sounddescs(repo_id, data_root):
    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name)
    zip_file = "audio_tasks/sounddescs-1k.tar"

    if not os.path.exists(os.path.join(local_dir, "audio_tasks", "sounddescs-1k")):
        hf_snapshot_download_with_pattern(repo_id, data_root, pattern=zip_file)
        zip_path = os.path.join(local_dir, zip_file)
        assert os.path.exists(zip_path)
        _unzip(zip_path)
    eval_path = os.path.join(local_dir, "audio_tasks", "sounddescs-1k/eval")
    return eval_path


def download_ave(repo_id, data_root):
    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name)
    zip_file = "audio_tasks/AVE.tar"

    if not os.path.exists(os.path.join(local_dir, "audio_tasks", "AVE")):
        hf_snapshot_download_with_pattern(repo_id, data_root, pattern=zip_file)
        zip_path = os.path.join(local_dir, zip_file)
        assert os.path.exists(zip_path)
        _unzip(zip_path)
    eval_path = os.path.join(local_dir, "audio_tasks", "AVE/AVE_Dataset")
    return eval_path


def download_mmar(repo_id, data_root):
    repo_name = sanitize_repo_dirname(repo_id)
    local_dir = os.path.join(data_root, repo_name)
    if not os.path.exists(os.path.join(local_dir, "audio")):
        local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        zip_path = os.path.join(local_dir, "mmar-audio.tar.gz")
        assert os.path.exists(zip_path)
        _unzip(zip_path)

    ann_file = os.path.join(local_dir, "MMAR-meta.json")
    assert os.path.exists(ann_file)
    dataset = load_dataset("json", data_files=ann_file)["train"]
    return dataset, local_dir


# AVHBench is only distributed via Google Drive (no HF mirror yet).
AVHBENCH_VIDEOS_ZIP_GDRIVE_ID = "10-Qp8zxA3ITT-ileEnCgJkf5Nzx1wry7"
AVHBENCH_QA_JSON_GDRIVE_ID = "1KcYDAv9lLy3hsx5rWdfRqMFV2NYcZ94W"


def download_avhbench(data_root: str) -> str:
    """
    Download AVHBench (videos + QA.json) from Google Drive, then extract frames
    (ffmpeg) + 16kHz mono audio (PyAV) for every video_id referenced in QA.json
    into `{local_dir}/frames/` and `{local_dir}/audio/`. Returns local_dir; the
    set of successfully-extracted video_ids is cached at
    `{local_dir}/converted_video_ids.json` (see get_avhbench_valid_video_ids()).
    """
    from src.utils.av_extraction_utils import extract_frames_and_audio

    local_dir = os.path.join(data_root, "AVHBench")
    videos_dir = os.path.join(local_dir, "videos")
    ann_file = os.path.join(local_dir, "QA.json")
    manifest_path = os.path.join(local_dir, "converted_video_ids.json")

    if is_main() and not (os.path.isdir(videos_dir) and os.path.exists(ann_file)):
        import gdown

        os.makedirs(local_dir, exist_ok=True)

        if not os.path.exists(ann_file):
            gdown.download(id=AVHBENCH_QA_JSON_GDRIVE_ID, output=ann_file, quiet=False)

        if not os.path.isdir(videos_dir):
            zip_path = os.path.join(local_dir, "videos.zip")
            gdown.download(id=AVHBENCH_VIDEOS_ZIP_GDRIVE_ID, output=zip_path, quiet=False)
            _run(["unzip", "-q", "videos.zip"], cwd=local_dir)
            os.remove(zip_path)
    barrier()

    assert os.path.isdir(videos_dir), videos_dir
    assert os.path.exists(ann_file), ann_file

    if is_main() and not os.path.exists(manifest_path):
        anns = json.load(open(ann_file))
        video_ids = {ann["video_id"] for ann in anns}
        extract_frames_and_audio(
            video_dir=videos_dir,
            frame_root=os.path.join(local_dir, "frames"),
            audio_root=os.path.join(local_dir, "audio"),
            video_ids=video_ids,
            max_frames_saved=32,
            manifest_path=manifest_path,
        )
    barrier()

    assert os.path.exists(manifest_path), manifest_path
    return local_dir


def get_avhbench_valid_video_ids(local_dir: str) -> set:
    manifest_path = os.path.join(local_dir, "converted_video_ids.json")
    return set(json.load(open(manifest_path)))


def load_hf_dataset_local(
    repo_id: str,
    data_root: str,
    *,
    download_method="snapshot",
    data_type="parquet",
    repo_type: str = "datasets",
    split_globs: Optional[Dict[str, Union[str, list[str]]]] = None,
    split: str = "train",
    download_all=False,
    unzip_assets=None
) -> DatasetDict:
    assert download_method in ["snapshot", "git"], download_method
    if download_all:
        repo_name = sanitize_repo_dirname(repo_id)
        local_dir = os.path.join(data_root, repo_name)
        if not os.path.exists(local_dir):
            local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        if unzip_assets is not None and (
            (not os.path.exists(os.path.join(local_dir, "frames"))) or  # noqa
            (not os.path.exists(os.path.join(local_dir, "audio")))
        ):
            for to_unzip in unzip_assets:
                to_unzip = os.path.join(local_dir, to_unzip)
                assert os.path.exists(to_unzip)
                _unzip(to_unzip)

    else:
        local_dir = hf_snapshot_download_with_pattern(repo_id, data_root, pattern=split_globs)

    return load_dataset_from_local_parquet_repo(
        local_dir, data_type=data_type, split_globs=split_globs, split=split
    )
