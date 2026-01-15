import subprocess

import torch
import numpy as np


def save_video_tensor(frames_tensor: torch.Tensor, output_path: str, fps: int):
    frames_tensor = (frames_tensor * 255).clamp_(0, 255).to(torch.uint8)
    frames = frames_tensor.cpu().numpy()  # (N, H, W, 3), RGB

    N, H, W, _ = frames.shape
    H2 = H + (H % 2)
    W2 = W + (W % 2)

    if (H2 != H) or (W2 != W):
        padded = np.zeros((N, H2, W2, 3), dtype=np.uint8)
        padded[:, :H, :W] = frames
        frames = padded

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W2}x{H2}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.stdin is None:
        raise BrokenPipeError("No stdin buffer received.")

    process.stdin.write(frames.tobytes())
    process.stdin.close()
    process.wait()

    if process.returncode != 0:
        error_output = (
            process.stderr.read().decode() if process.stderr else "Unknown error"
        )
        raise RuntimeError(f"FFmpeg failed with error: {error_output}")
