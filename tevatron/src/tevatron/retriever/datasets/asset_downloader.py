import glob
import json
import os
import shutil
import subprocess
import tarfile
import zipfile

from concurrent.futures import ThreadPoolExecutor, as_completed

import huggingface_hub as hf_hub

from tqdm import tqdm

from .av_extraction_utils import (
    download_and_extract,
    extract_one,
    make_extract_pool,
    outputs_valid,
)
from .download_utils import (
    download_and_unzip,
    download_url,
    extract_all,
    hf_snapshot_download_extract,
    sanitize_repo_dirname,
)


def msvd_video_downloader(assets_path: str, repo_id: str, download: bool = False):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    download_url = "https://huggingface.co/datasets/friedrichor/MSVD/resolve/main/MSVD_Videos.zip"  # noqa
    assets_prefix = os.path.join(assets_path, "YouTubeClips")
    if download and (not os.path.exists(assets_prefix)):
        download_and_unzip(download_url, assets_path, fname="MSVD_Videos.zip", remove_zip=True)
    return assets_prefix


def msrvtt_video_downloader(assets_path: str, repo_id: str, download: bool = False):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    download_url = "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/MSRVTT_Videos.zip"  # noqa
    assets_prefix = os.path.join(assets_path, "video")
    if download and (not os.path.exists(assets_prefix)):
        download_and_unzip(download_url, assets_path, fname="MSRVTT_Videos.zip", remove_zip=True)
    return assets_prefix


def activitynet_video_downloader(
    assets_path: str, repo_id: str, repo_type: str = "dataset", download: bool = False
):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    assets_prefix = os.path.join(assets_path, "Activity_Videos")
    if download and (not os.path.exists(assets_prefix)):
        hf_snapshot_download_extract(
            repo_id=repo_id,
            assets_path=assets_path,
            filename_pattern="ActivityNet_Videos.tar*",
            repo_type=repo_type,
            merged_name="merged_ActivityNet_Videos.tar",
            remove_merged=True,
            remove_archives=True
        )
    return assets_prefix


def didemo_video_downloader(
    assets_path: str, repo_id: str, repo_type: str = "dataset", download: bool = False
):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    assets_prefix = os.path.join(assets_path, "video")
    if download and (not os.path.exists(assets_prefix)):
        hf_snapshot_download_extract(
            repo_id=repo_id,
            assets_path=assets_path,
            filename_pattern="DiDeMo_Videos_mp4_train.tar*",
            repo_type=repo_type,
            merged_name="merged_DiDeMo_Videos_mp4_train.tar",
            remove_merged=True,
            remove_archives=True
        )
    return assets_prefix


def musicbench_audio_downloader(assets_path: str, repo_id: str, download: bool = False):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    download_url = "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench.tar.gz"  # noqa
    assets_prefix = os.path.join(assets_path, "datashare")
    if download and (not os.path.exists(assets_prefix)):
        download_and_unzip(download_url, assets_path, fname="MusicBench.tar.gz", remove_zip=True)
    return assets_prefix


def llavahound_downloader(
    assets_path: str, repo_id: str, repo_type: str = "dataset", download: bool = False
):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    # download annotations
    caption_url = "https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/sft/video_caption_300k.jsonl"  # noqa
    download_url(caption_url, assets_path, "video_caption_300k.jsonl")

    qa_url = "https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/qa/chatgpt_qa_240k.jsonl"  # noqa
    download_url(qa_url, assets_path, "chatgpt_qa_240k.jsonl")

    train_300k = os.path.join(assets_path, "train_300k")
    dirs = [p for p in glob.glob(os.path.join(train_300k, "*")) if not p.endswith(".gz")]
    os.makedirs(train_300k, exist_ok=True)
    if download and (len(dirs) < 1000):
        hf_snapshot_download_extract(
            repo_id=repo_id,
            assets_path=assets_path,
            filename_pattern="train_300k/*.tar.gz",
            repo_type=repo_type,
            merge_before_extract=False,
            remove_archives=True
        )
    return assets_path


def mmeb_train_downloader(
    assets_path: str,
    repo_id: str,
    repo_type: str = "dataset",
    dataset_name: str = None,
    download: bool = False,
    remove_zip=True
):
    os.makedirs(assets_path, exist_ok=True)
    assets_path = os.path.join(assets_path, "MMEB-train")
    img_path = os.path.join(assets_path, "images", dataset_name, "Train")

    def count_images(path):
        if not os.path.exists(path):
            return 0
        return len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if download and count_images(img_path) < 1:
        print(f"download MMEB-train -> {assets_path}")

        if not os.path.exists(os.path.join(assets_path, ".git")):
            subprocess.check_call(
                ["git", "clone", f"https://huggingface.co/datasets/{repo_id}", assets_path]
            )
        subprocess.check_call(["git", "lfs", "pull"], cwd=assets_path)

        zip_dir = os.path.join(assets_path, "images_zip")
        out_dir = os.path.join(assets_path, "images")
        os.makedirs(out_dir, exist_ok=True)

        for z in os.listdir(zip_dir):
            if z.endswith(".zip"):
                print(f"Unzip {z}...")
                with zipfile.ZipFile(os.path.join(zip_dir, z)) as f:
                    f.extractall(out_dir)
                target_zip = os.path.join(zip_dir, z)
                if remove_zip and os.path.exists(target_zip):
                    print(f"removing {target_zip}...")
                    os.remove(target_zip)

    return assets_path


def musicinstruct_downloader(
    assets_path: str, repo_id: str, repo_type: str = "dataset", download: bool = False
):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    # download annotations
    ann_url = "https://huggingface.co/datasets/m-a-p/Music-Instruct/resolve/main/MusicInstruct/music_instruct.json"  # noqa
    download_url(ann_url, assets_path, "music_instruct.json")

    return assets_path


def avinstruct_downloader(
    assets_path: str, repo_id: str, repo_type: str = "dataset", download: bool = False
):
    assets_path = os.path.join(assets_path, repo_id)
    os.makedirs(assets_path, exist_ok=True)

    # download annotations
    ann_url1 = "https://raw.githubusercontent.com/rikeilong/Bay-CAT/refs/heads/main/AVinstruct/avqa_data1.json"  # noqa
    download_url(ann_url1, assets_path, "avqa_data1.json")

    ann_url2 = "https://raw.githubusercontent.com/rikeilong/Bay-CAT/refs/heads/main/AVinstruct/avqa_data2.json"  # noqa
    download_url(ann_url2, assets_path, "avqa_data2.json")

    # download the avqa videos used for training (only those available in the HF repo),
    # then extract frames + audio from them;
    # videos that fail to download or extract are left out of `valid_ids.json`
    valid_ids_path = os.path.join(assets_path, "valid_ids.json")
    if download and not os.path.exists(valid_ids_path):
        ids_path = os.path.join(os.path.dirname(__file__), "resources", "avinstruct_video_ids.json")
        with open(ids_path, "r") as f:
            video_ids = json.load(f)

        def download_video(video_id):
            hf_hub.hf_hub_download(
                repo_id="juyil/AVQA-videos",
                repo_type="dataset",
                filename=f"videos/{video_id}.mp4",
                local_dir=assets_path
            )

        def is_retryable(err):  # failed download, except for videos missing in the repo
            return err.startswith("download") and "NotFound" not in err

        for _ in range(3):
            succeeded, failed = download_and_extract(
                video_ids,
                download_video,
                video_dir=os.path.join(assets_path, "videos"),
                frame_root=os.path.join(assets_path, "frames"),
                audio_root=os.path.join(assets_path, "audio"),
            )
            if not any(is_retryable(err) for _, err in failed):
                break
        else:
            raise RuntimeError(
                f"avinstruct: failed to download videos; rerun to resume: {failed[:5]}"
            )

        failed_path = os.path.join(assets_path, "failed_list.txt")
        with open(failed_path, "w") as f:
            for video_id, err in failed:
                f.write(f"{video_id}\t{' '.join(err.split())}\n")
        with open(valid_ids_path, "w") as f:
            json.dump(succeeded, f, indent=0)
        print(
            f"avinstruct: {len(succeeded)}/{len(video_ids)} videos ready, "
            f"{len(failed)} failed (see {failed_path})"
        )

    return assets_path


def pvd_downloader(
    assets_path: str, repo_id: str, repo_type: str = "dataset", download: bool = False
):
    assets_path = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(assets_path, exist_ok=True)

    # scan the train shards, keep only the videos used for training and extract frames from them;
    # annotations are built from the shard captions of the successfully extracted videos
    ann_path = os.path.join(assets_path, "pvd_annotations.json")
    if download and not os.path.exists(ann_path):
        ids_path = os.path.join(os.path.dirname(__file__), "resources", "pvd_video_ids.json")
        with open(ids_path, "r") as f:
            video_ids = json.load(f)
        wanted = set(video_ids)

        video_dir = os.path.join(assets_path, "videos")
        caption_dir = os.path.join(assets_path, "shard_captions")  # also marks scanned shards
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(caption_dir, exist_ok=True)

        def video_path(vid):
            return os.path.join(video_dir, f"{vid}.mp4")

        def frame_dir(vid):
            return os.path.join(assets_path, "frames", vid)

        def caption_path(shard):
            return os.path.join(caption_dir, os.path.basename(shard).replace(".tar", ".json"))

        def download_shard(shard):
            return hf_hub.hf_hub_download(
                repo_id=repo_id, repo_type=repo_type, filename=shard, local_dir=assets_path
            )

        shards = sorted(
            f.path for f in
            hf_hub.HfApi().list_repo_tree(repo_id, path_in_repo="train", repo_type=repo_type)
            if f.path.endswith(".tar")
        )
        todo = [s for s in shards if not os.path.exists(caption_path(s))]

        ex_futures = []
        with ThreadPoolExecutor(2) as dl_pool, make_extract_pool() as ex_pool:

            def submit_extract(vid):
                ex_futures.append(
                    ex_pool.submit(extract_one, vid, video_path(vid), frame_dir(vid), None, 32)
                )

            # videos saved by an earlier, interrupted run
            for vid in video_ids:
                if os.path.exists(video_path(vid)) and not outputs_valid(frame_dir(vid)):
                    submit_extract(vid)

            # download the next shard while the current one is scanned
            dl_futures = {s: dl_pool.submit(download_shard, s) for s in todo[:2]}
            for i, shard in enumerate(tqdm(todo, desc="pvd shards")):
                if i + 2 < len(todo):
                    dl_futures[todo[i + 2]] = dl_pool.submit(download_shard, todo[i + 2])
                shard_path = dl_futures.pop(shard).result()

                captions = {}
                with tarfile.open(shard_path) as tf:
                    for member in tf:
                        vid, _, ext = os.path.basename(member.name).partition(".")
                        if vid not in wanted:
                            continue
                        if ext == "json":
                            meta = json.load(tf.extractfile(member))
                            captions[vid] = {
                                "caption": meta["human_caption"],
                                "duration": float(meta["video_duration_in_s"]),
                                "fps": float(meta["fps"]),
                            }
                        elif ext == "mp4":
                            if not os.path.exists(video_path(vid)):
                                tmp_path = f"{video_path(vid)}.tmp"
                                with open(tmp_path, "wb") as f:
                                    shutil.copyfileobj(tf.extractfile(member), f)
                                os.replace(tmp_path, video_path(vid))
                            if not outputs_valid(frame_dir(vid)):
                                submit_extract(vid)

                with open(caption_path(shard), "w") as f:
                    json.dump(captions, f)
                os.remove(shard_path)

            failed = {}
            for fut in tqdm(as_completed(ex_futures), total=len(ex_futures), desc="extract"):
                vid, err = fut.result()
                if err is not None:
                    failed[vid] = err

        captions = {}
        for shard in shards:
            with open(caption_path(shard), "r") as f:
                captions.update(json.load(f))

        # keep the order of `video_ids` (the sampled training order)
        anns = []
        for vid in video_ids:
            if vid in failed:
                continue
            if vid not in captions:
                failed[vid] = "not found in train shards"
            elif not outputs_valid(frame_dir(vid)):
                failed[vid] = "no frames"
            else:
                anns.append({"video_id": int(vid), **captions[vid]})

        failed_path = os.path.join(assets_path, "failed_list.txt")
        with open(failed_path, "w") as f:
            for vid, err in sorted(failed.items()):
                f.write(f"{vid}\t{' '.join(err.split())}\n")
        with open(ann_path, "w") as f:
            json.dump(anns, f)
        print(
            f"pvd: {len(anns)}/{len(video_ids)} videos ready, "
            f"{len(failed)} failed (see {failed_path})"
        )

    return assets_path


def wavcaps_downloader(
    assets_path: str, repo_id: str, repo_type: str = "dataset", download: bool = False
):
    local_dir = os.path.join(assets_path, sanitize_repo_dirname(repo_id))
    os.makedirs(local_dir, exist_ok=True)

    zip_path = os.path.join(local_dir, "Zip_files")
    json_path = os.path.join(local_dir, "json_files")
    if download:
        # download anns
        as_url = "https://huggingface.co/datasets/cvssp/WavCaps/blob/main/json_files/AudioSet_SL/as_final.json"  # noqa
        download_url(as_url, os.path.join(json_path, "AudioSet_SL"), fname="as_final.json")

        # download zips and extract
        hf_hub.snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            allow_patterns="Zip_files/AudioSet_SL/AudioSet_SL.z*"
        )  # download only AudioSet-SL zipfiles

        # remove zip files when download is already completed
        zip_files = glob.glob(os.path.join(zip_path, "AudioSet_SL", "AudioSet_SL.z*"))
        if os.path.exists(os.path.join(zip_path, "AudioSet_SL", "mnt")):
            if len(zip_files):
                for zip_file in zip_files:
                    try:
                        os.remove(zip_file)
                        print(f"removed: {zip_file}")
                    except OSError:
                        pass

        if not os.path.exists(os.path.join(zip_path, "AudioSet_SL", "mnt")):
            # fix broken zip files
            cmd = ["zip", "-FF", "AudioSet_SL.zip", "--out", "fixed.zip"]
            subprocess.call(cmd, cwd=os.path.join(zip_path, "AudioSet_SL"))
            extract_all(os.path.join(zip_path, "AudioSet_SL", "fixed.zip"), remove_zip=True)

    return local_dir
