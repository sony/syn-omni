import json
import math
import os
import subprocess

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Set

import av
import numpy as np
import soundfile as sf

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
# Reusable batch driver.
# ============================================================================


def _process_one(video_id, video_path, frame_dir, audio_path, max_frames_saved):
    if os.path.exists(frame_dir) and os.path.exists(audio_path):
        return video_id, True, None
    try:
        save_frames_and_audio(video_path, frame_dir, audio_path, max_frames_saved)
        return video_id, True, None
    except Exception as e:
        return video_id, False, f"{type(e).__name__}: {e}"


def extract_frames_and_audio(
    video_dir: str,
    frame_root: str,
    audio_root: str,
    video_ids: Iterable[str],
    max_frames_saved: int = 32,
    workers: int = 16,
    manifest_path: Optional[str] = None,
) -> Set[str]:
    """
    For each video_id, extract frames + 16kHz mono audio from
    `{video_dir}/{video_id}.mp4` into `{frame_root}/{video_id}/` and
    `{audio_root}/{video_id}.wav`. Already-extracted ids are skipped.

    Returns the set of video_ids that have valid (extracted or pre-existing)
    frames+audio. If `manifest_path` is given, the result is cached there
    (as a JSON list) and reused on subsequent calls without re-scanning.
    """
    video_ids = sorted(set(video_ids))

    if manifest_path is not None and os.path.exists(manifest_path):
        return set(json.load(open(manifest_path)))

    jobs = []
    for vid in video_ids:
        video_path = os.path.join(video_dir, f"{vid}.mp4")
        frame_dir = os.path.join(frame_root, vid)
        audio_path = os.path.join(audio_root, f"{vid}.wav")
        jobs.append((vid, video_path, frame_dir, audio_path))

    succeeded = []
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_one, vid, vp, fd, ap, max_frames_saved): vid
            for vid, vp, fd, ap in jobs
        }
        for fut in as_completed(futures):
            vid, ok, err = fut.result()
            if ok:
                succeeded.append(vid)
            else:
                failed.append((vid, err))

    if manifest_path is not None:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(sorted(succeeded), f, indent=2)
        failed_log = os.path.join(os.path.dirname(manifest_path), "failed_list.txt")
        with open(failed_log, "w") as f:
            for vid, err in failed:
                f.write(f"{vid}\t{err}\n")

    return set(succeeded)
