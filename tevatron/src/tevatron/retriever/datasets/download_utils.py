import glob
import logging
import os
import re
import shutil
import subprocess

from dataclasses import dataclass
from os.path import join as join_path
from typing import List, Optional

import huggingface_hub as hf_hub

logger = logging.getLogger(__name__)

_LAST_INT_RE = re.compile(r"(\d+)(?!.*\d)")


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    ret = subprocess.call(cmd, cwd=cwd)
    if ret != 0:
        raise RuntimeError(f"Command failed (exit={ret}): {' '.join(cmd)}\n"
                           f"cwd={cwd}")


def extract_archive(path: str, *, cwd: str | None = None):
    print(f"unziping {path}...")
    name_l = path.lower()
    if cwd is None:
        cwd = os.path.dirname(path)
    fname = os.path.basename(path)

    if name_l.endswith(".zip"):
        _run(["unzip", "-q", fname], cwd=cwd)
    elif name_l.endswith((".tar", ".tgz")):
        _run(["tar", "-xf", fname], cwd=cwd)
    elif name_l.endswith(".tar.gz"):
        _run(["tar", "-zxf", fname], cwd=cwd)
    else:
        raise ValueError(f"Unknown archive type: {path}")


def extract_all(path, cwd=None, remove_zip=False, merged_name=None):
    # if merged_name is given, merge all zip splits and extract the integrated one
    if isinstance(path, str) and "*" in path:
        path = sorted(glob.glob(path))
    elif isinstance(path, str):
        path = [path]

    if merged_name is not None:
        assert len(path) > 1

        def _key(p: str):
            b = os.path.basename(p)
            m = _LAST_INT_RE.search(b)
            return (0, int(m.group(1))) if m else (1, b)

        path.sort(key=_key)
        splits = ", ".join([os.path.basename(p) for p in path])
        print(f"integrating {splits} -> {merged_name}")
        # 4) Merge all parts into a single archive
        merged = os.path.join(os.path.dirname(path[0]), merged_name)
        if not os.path.exists(merged):
            with open(merged, "wb") as w:
                for p in path:
                    with open(p, "rb") as r:
                        w.write(r.read())
        path = [merged]

    path = [p for p in path if p.endswith((".zip", ".tar", ".tar.gz", ".tgz"))]
    to_extract = ", ".join([os.path.basename(p) for p in path])
    print(f"total {len(path)} archieve files will be extracted: {to_extract}")
    for p in path:
        extract_archive(p, cwd=cwd)
        if os.path.exists(p) and remove_zip is True:
            os.remove(p)
            print(f"Removed archive: {p}")
    return


# ---------------------------
# Download helpers
# ---------------------------


def sanitize_repo_dirname(repo_id: str) -> str:
    # "Tevatron/msrvtt-corpus" -> "Tevatron__msrvtt-corpus"
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id)


def _is_http_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _to_gdrive_uc(url_or_id: str) -> str:
    # Accept google drive file id, return a download URL
    if _is_http_url(url_or_id):
        return url_or_id
    return f"https://drive.google.com/uc?id={url_or_id}"


@dataclass(frozen=True)
class HfRef:
    repo_id: str
    repo_type: Optional[str]  # None or "dataset"
    path_in_repo: str


def _parse_hf_single_file_url(url: str) -> HfRef:
    """
    Supports:
      - https://huggingface.co/<org>/<repo>/blob/main/<path>
      - https://huggingface.co/<org>/<repo>/resolve/main/<path>
      - https://huggingface.co/datasets/<org>/<repo>/blob/main/<path>
      - https://huggingface.co/datasets/<org>/<repo>/resolve/main/<path>
    """
    is_dataset = "huggingface.co/datasets/" in url
    marker = "huggingface.co/datasets/" if is_dataset else "huggingface.co/"

    tail = url.split(marker, 1)[1]
    parts = tail.split("/")

    if is_dataset:
        if len(parts) < 5:
            raise ValueError(f"Invalid HF dataset URL: {url}")
        # parts: <org>, <repo>, (blob|resolve), main, <path...>
        repo_id = f"{parts[0]}/{parts[1]}"
        repo_type = "dataset"
        kind = parts[2]
        if kind not in ("blob", "resolve"):
            raise ValueError(f"Invalid HF URL (expected blob/resolve): {url}")
        if parts[3] != "main":
            raise ValueError(f"Only main branch supported: {url}")
        path_in_repo = "/".join(parts[4:])
    else:
        if len(parts) < 5:
            raise ValueError(f"Invalid HF repo URL: {url}")
        repo_id = f"{parts[0]}/{parts[1]}"
        repo_type = None
        kind = parts[2]
        if kind not in ("blob", "resolve"):
            raise ValueError(f"Invalid HF URL (expected blob/resolve): {url}")
        if parts[3] != "main":
            raise ValueError(f"Only main branch supported: {url}")
        path_in_repo = "/".join(parts[4:])

    # Strip query params like ?download=1
    if "?" in path_in_repo:
        path_in_repo = path_in_repo.split("?", 1)[0]

    if not path_in_repo:
        raise ValueError(f"HF URL does not contain a file path: {url}")

    return HfRef(repo_id=repo_id, repo_type=repo_type, path_in_repo=path_in_repo)


def _download_hf_single_file(url: str, root: str, filename: str) -> str:
    ref = _parse_hf_single_file_url(url)
    logger.info(
        "Downloading from Hugging Face Hub: repo=%s type=%s path=%s", ref.repo_id, ref.repo_type,
        ref.path_in_repo
    )

    # Returns the resolved local cache file path
    downloaded_path = hf_hub.hf_hub_download(
        repo_id=ref.repo_id,
        filename=ref.path_in_repo,
        local_dir=root,
        repo_type=ref.repo_type,
    )

    # Place/copy to requested output file name under root
    out_path = join_path(root, filename)
    if os.path.abspath(downloaded_path) != os.path.abspath(out_path):
        os.makedirs(root, exist_ok=True)
        shutil.move(downloaded_path, out_path)

    return out_path


def _download_wget(url: str, root: str, filename: str) -> str:
    os.makedirs(root, exist_ok=True)
    out_path = join_path(root, filename)
    tmp_path = out_path + ".partial"

    # -c (continue), -O output
    _run(["wget", "-c", "-O", tmp_path, url])

    # atomic-ish finalize
    os.replace(tmp_path, out_path)
    return out_path


def _download_gdown(url_or_id: str, root: str, filename: str) -> str:
    os.makedirs(root, exist_ok=True)
    out_path = join_path(root, filename)

    # gdown accepts either full URL or file id URL; we normalize to uc?id=...
    gurl = _to_gdrive_uc(url_or_id)
    _run(["gdown", gurl, "--output", out_path])
    return out_path


def _choose_downloader(url: str):
    # Hugging Face single-file URL
    if "huggingface.co" in url:
        return _download_hf_single_file

    # Other HTTP(S) sources
    if _is_http_url(url):
        return _download_wget

    # Otherwise treat as Google Drive file id
    return _download_gdown


def download_url(url: str, root_dir: str, fname: Optional[str] = None) -> str:
    """
    Download a single file identified by:
      - Hugging Face single-file URL (blob/main or resolve/main)
      - Any http(s) URL (via wget)
      - Google Drive file id (via gdown)
    """
    if fname is None:
        root_dir, fname = os.path.split(root_dir)

    if not fname:
        raise ValueError(
            "fname resolved to empty string. Provide fname or a full target file path."
        )

    os.makedirs(root_dir, exist_ok=True)
    out_path = join_path(root_dir, fname)

    if os.path.exists(out_path):
        return out_path

    logger.info("Downloading %s -> %s", url, out_path)
    downloader = _choose_downloader(url)
    downloaded = downloader(url=url, root=root_dir, filename=fname)

    if not os.path.exists(downloaded):
        raise RuntimeError(f"Download failed. Please manually download from {url} to {out_path}")
    return downloaded


def download_and_unzip(
    url: str,
    target_path: str,
    fname: Optional[str] = None,
    remove_zip: bool = True,
) -> str:
    """
    Downloads an archive (zip/tar/tar.gz) and extracts into target directory.
    - If fname is None, target_path is treated as the full file path.
    - Returns the downloaded archive path.
    """
    if fname is None:
        target_dir, fname = os.path.split(target_path)
    else:
        target_dir = target_path

    archive_path = download_url(url, target_dir, fname=fname)
    logger.info("Extracting %s -> %s", os.path.basename(archive_path), target_dir)

    base = os.path.basename(archive_path)

    if archive_path.endswith(".zip"):
        _run(["unzip", "-q", base], cwd=target_dir)
    elif archive_path.endswith(".tar.gz"):
        _run(["tar", "-zxf", base], cwd=target_dir)
    elif archive_path.endswith(".tar"):
        _run(["tar", "-xf", base], cwd=target_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive_path}")

    if remove_zip:
        os.remove(archive_path)
        logger.info("Removed archive: %s", archive_path)

    return archive_path


def hf_snapshot_download_extract(
    *,
    repo_id: str,
    assets_path: str,
    filename_pattern: str,  # e.g. "DiDeMo_Videos_mp4_train.tar" or any substring
    repo_type: str = "dataset",
    merge_before_extract: bool = True,

    # Only used when merge_before_extract=True
    merged_name: str = "merged_archive",  # include proper extension (e.g. merged.tar.gz)
    remove_parts: bool = True,
    remove_merged: bool = True,

    # Only used when merge_before_extract=False
    remove_archives: bool = True,  # remove each archive after extraction
):
    """
    Download files from a HuggingFace dataset snapshot and extract them.

    - merge_before_extract=True:
        Sort matched files, merge them into a single archive (merged_name),
        then extract based on merged_name extension.
    - merge_before_extract=False:
        Extract each archive file independently in its own directory
        (no merge, no sorting).
    """
    os.makedirs(assets_path, exist_ok=True)

    # 1) Download all matching files
    hf_hub.snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=assets_path,
        allow_patterns=filename_pattern,
    )

    # 2) Collect matched files
    files = glob.glob(
        os.path.join(assets_path, "**", f"*{filename_pattern}*"),
        recursive=True,
    )
    if not files:
        raise RuntimeError(f"No files found for pattern: {filename_pattern}")

    if merge_before_extract:
        # 3) Sort by the last integer in the basename (fallback to lexicographic)
        def _key(p: str):
            b = os.path.basename(p)
            m = _LAST_INT_RE.search(b)
            return (0, int(m.group(1))) if m else (1, b)

        files.sort(key=_key)

        # 4) Merge all parts into a single archive
        merged = os.path.join(assets_path, merged_name)
        if not os.path.exists(merged):
            with open(merged, "wb") as w:
                for p in files:
                    with open(p, "rb") as r:
                        w.write(r.read())

        # 5) Extract the merged archive
        extract_archive(merged, cwd=os.path.dirname(merged))

        # 6) Cleanup
        if remove_parts:
            for p in files:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        if remove_merged:
            try:
                os.remove(merged)
            except FileNotFoundError:
                pass

        return merged

    else:
        # 3) Extract each archive independently (no sort, no merge)
        extracted = []
        for arch in files:
            name_l = arch.lower()
            if not (
                name_l.endswith(".zip") or name_l.endswith(".tar") or name_l.endswith(".tar.gz")
                or name_l.endswith(".tgz")
            ):
                continue  # skip non-archive files

            extract_archive(arch)
            extracted.append(arch)

            if remove_archives:
                try:
                    os.remove(arch)
                except FileNotFoundError:
                    pass

        if not extracted:
            raise RuntimeError(
                f"Matched files exist but no extractable archives found: {filename_pattern}"
            )

        return extracted
