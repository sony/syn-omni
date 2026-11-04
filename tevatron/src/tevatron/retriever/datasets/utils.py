import os

import numpy as np


def load_video_frames(frame_path: str, num_frames: int):
    frames = sorted(os.listdir(frame_path))
    assert len(frames), f"no frames found: {frame_path}"

    if len(frames) < num_frames:
        idxs = list(range(len(frames))) + [len(frames) - 1] * (num_frames - len(frames))
    else:
        idxs = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
    frames = [os.path.join(frame_path, frames[i]) for i in idxs]
    return frames
