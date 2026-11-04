# Adapted from vlm2vec_eval/src/utils/av_extraction_utils.py; the extraction logic is unchanged.
import math
import multiprocessing
import os
import shutil
import subprocess

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional

import av
import numpy as np
import soundfile as sf

from tqdm import tqdm

# ============================================================================
# Core extraction logic (ffmpeg frame sampling + PyAV audio decode/resample).
# ============================================================================


def get_video_fps_and_total_frames(video_path: str) -> tuple[float, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,nb_read_frames",
        "-count_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2:
        raise ValueError(f"Unexpected ffprobe output: {lines}")

    r_frame_rate = lines[0].strip()
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(r_frame_rate)

    n_frames_str = lines[1].strip()
    if not n_frames_str.isdigit():
        raise ValueError(f"Invalid frame count: {n_frames_str}")
    total_frames = int(n_frames_str)

    return fps, total_frames


def save_frames_uniform_ffmpeg(
    video_path: str,
    frame_dir: str,
    max_frames_saved: int,
) -> tuple[list[int], float, float]:
    """
    Save up to `max_frames_saved` frames uniformly (by frame index) using ffmpeg.
    Returns: (frame_indices, fps, duration_seconds_covered_by_saved_frames)
    duration is computed as last_selected_frame_idx / fps.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File {video_path} does not exist")

    Path(frame_dir).mkdir(parents=True, exist_ok=True)

    fps, total_frames = get_video_fps_and_total_frames(video_path)
    if total_frames <= 0:
        raise RuntimeError("No frames found in video")
    if total_frames < 14:
        raise RuntimeError("Short frames")

    if fps < 2 or fps > 1000:
        raise RuntimeError("fps error")

    if total_frames <= max_frames_saved:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / max_frames_saved
        frame_indices = [math.floor(i * step) for i in range(max_frames_saved)]

    duration_s = frame_indices[-1] / fps

    if duration_s < 2:
        raise RuntimeError("Short duration")

    select_expr = "+".join([f"eq(n\\,{i})" for i in frame_indices])
    output_pattern = os.path.join(frame_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "vfr",
        output_pattern,
    ]
    subprocess.run(cmd, check=True)

    return frame_indices, fps, duration_s


def extract_audio_mp4_to_wav_16k_clamped(
    video_path: str,
    output_wav_path: str,
    clamp_duration_s: float,
    target_sr: int = 16000,
):
    """
    Extract audio from MP4, resample to 16kHz mono, clamp to clamp_duration_s, save wav.
    """
    container = av.open(video_path)
    audio_stream = next(s for s in container.streams if s.type == "audio")

    resampler = av.audio.resampler.AudioResampler(
        format="fltp",  # float32
        layout="mono",
        rate=target_sr,
    )

    chunks = []
    for frame in container.decode(audio_stream):
        resampled = resampler.resample(frame)
        if resampled is None:
            continue

        if isinstance(resampled, list):
            for f in resampled:
                arr = f.to_ndarray()
                chunks.append(arr)
        else:
            arr = resampled.to_ndarray()
            chunks.append(arr)

    if not chunks:
        raise RuntimeError(f"No audio frames decoded: {video_path}")

    audio = np.concatenate(chunks, axis=1)[0]  # mono

    max_len = int(clamp_duration_s * target_sr)
    if audio.shape[0] > max_len:
        audio = audio[:max_len]

    Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
    audio = audio.astype(np.float32, copy=False)
    sf.write(output_wav_path, audio, target_sr)


def save_frames_and_audio(
    video_path: str,
    frame_dir: str,
    audio_wav_path: str,
    max_frames_saved: int,
) -> float:
    """
    1) Save uniformly sampled frames (ffmpeg)
    2) Compute duration from saved-frame coverage (last_idx / fps)
    3) Extract audio -> 16kHz mono -> clamp to that duration -> save wav

    Returns: duration_seconds_used_for_audio_clamp
    """
    _, _, duration_s = save_frames_uniform_ffmpeg(
        video_path=video_path,
        frame_dir=frame_dir,
        max_frames_saved=max_frames_saved,
    )

    extract_audio_mp4_to_wav_16k_clamped(
        video_path=video_path,
        output_wav_path=audio_wav_path,
        clamp_duration_s=duration_s,
        target_sr=16000,
    )

    return duration_s


# ============================================================================
# Download + extraction driver.
# ============================================================================


def outputs_valid(frame_dir: str, audio_path: Optional[str] = None) -> bool:
    """Frames are non-empty jpgs and, if given, the wav has at least one sample."""
    if not os.path.isdir(frame_dir):
        return False
    frames = [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
    if not frames or any(os.path.getsize(os.path.join(frame_dir, f)) == 0 for f in frames):
        return False
    if audio_path is None:
        return True
    if not os.path.isfile(audio_path):
        return False
    try:
        return sf.info(audio_path).frames > 0
    except RuntimeError:
        return False


def extract_one(video_id, video_path, frame_dir, audio_path, max_frames_saved):
    """Extract frames (+ audio unless `audio_path` is None). Returns (video_id, error or None)."""
    if outputs_valid(frame_dir, audio_path):
        return video_id, None
    # extract into temporary paths and move them in place only after validation,
    # so an interrupted run never leaves partial outputs behind
    tmp_frame_dir = f"{frame_dir}.tmp"
    tmp_audio_path = None if audio_path is None else f"{audio_path[:-len('.wav')]}.tmp.wav"
    shutil.rmtree(tmp_frame_dir, ignore_errors=True)
    try:
        if audio_path is None:
            save_frames_uniform_ffmpeg(video_path, tmp_frame_dir, max_frames_saved)
        else:
            save_frames_and_audio(video_path, tmp_frame_dir, tmp_audio_path, max_frames_saved)
        if not outputs_valid(tmp_frame_dir, tmp_audio_path):
            raise RuntimeError("empty frames or audio")
        shutil.rmtree(frame_dir, ignore_errors=True)
        os.replace(tmp_frame_dir, frame_dir)
        if audio_path is not None:
            os.replace(tmp_audio_path, audio_path)
        return video_id, None
    except Exception as e:
        return video_id, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp_frame_dir, ignore_errors=True)
        if tmp_audio_path is not None and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def make_extract_pool(workers: int = 16) -> ProcessPoolExecutor:
    # spawn: avoid forking while download threads are running
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


def download_and_extract(
    video_ids: Iterable[str],
    download_fn: Callable[[str], None],
    video_dir: str,
    frame_root: str,
    audio_root: str,
    max_frames_saved: int = 32,
    download_workers: int = 8,
    extract_workers: int = 16,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    For each video_id, download `{video_dir}/{video_id}.mp4` via `download_fn(video_id)` if
    missing, then extract frames + 16kHz mono audio into `{frame_root}/{video_id}/` and
    `{audio_root}/{video_id}.wav`. Each video is extracted as soon as its download finishes;
    already-extracted ids are skipped.

    Returns (succeeded video_ids, [(failed video_id, reason), ...]).
    """

    def paths(vid):
        return (
            os.path.join(video_dir, f"{vid}.mp4"),
            os.path.join(frame_root, vid),
            os.path.join(audio_root, f"{vid}.wav"),
        )

    succeeded, failed = [], []
    with ThreadPoolExecutor(download_workers
                            ) as dl_pool, make_extract_pool(extract_workers) as ex_pool:
        dl_futures, ex_futures = {}, []
        for vid in sorted(set(video_ids)):
            video_path, frame_dir, audio_path = paths(vid)
            if outputs_valid(frame_dir, audio_path):
                succeeded.append(vid)
            elif os.path.exists(video_path):
                ex_futures.append(ex_pool.submit(extract_one, vid, *paths(vid), max_frames_saved))
            else:
                dl_futures[dl_pool.submit(download_fn, vid)] = vid

        for fut in tqdm(as_completed(dl_futures), total=len(dl_futures), desc="download"):
            vid = dl_futures[fut]
            try:
                fut.result()
            except Exception as e:
                failed.append((vid, f"download {type(e).__name__}: {e}"))
                continue
            ex_futures.append(ex_pool.submit(extract_one, vid, *paths(vid), max_frames_saved))

        for fut in tqdm(as_completed(ex_futures), total=len(ex_futures), desc="extract"):
            vid, err = fut.result()
            if err is None:
                succeeded.append(vid)
            else:
                failed.append((vid, err))

    return sorted(succeeded), sorted(failed)
