import subprocess

import cv2
import numpy as np

from .ffmpeg import find_ffmpeg

def frame_video(path: str):
    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        good, frame = cap.read()
        if not good:
            break
        frames.append(frame)

    cap.release()
    return frames

def iter_video_frames(path: str, scale: float = 1.0, step: int = 1):
    """
    Yield frames sequentially without storing the whole video in memory.

    Yields:
      (processed_index, frame)
    Where processed_index is the index after applying `step` (0..N-1).
    """
    if step < 1:
        raise ValueError("step must be >= 1")

    cap = cv2.VideoCapture(path)
    original_idx = 0
    processed_idx = 0

    while True:
        good, frame = cap.read()
        if not good:
            break

        if original_idx % step != 0:
            original_idx += 1
            continue

        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        yield processed_idx, frame
        processed_idx += 1
        original_idx += 1

    cap.release()

def iter_video_frames_adaptive(
    path: str,
    scale: float = 1.0,
    base_step: int = 2,
    motion_threshold: float = 12.0,
    motion_burst_frames: int = 3,
):
    """
    Yield frames for inference with reduced skipping during fast motion bursts.

    Yields:
      (track_idx, original_frame_idx, resized_frame)

    Motion heuristic is computed on resized frames:
      motion_score = mean(absdiff(gray_t, gray_{t-1}))
    When motion_score exceeds `motion_threshold`, the next
    `motion_burst_frames` frames are yielded densely (step=1).
    """
    if base_step < 1:
        raise ValueError("base_step must be >= 1")
    if motion_burst_frames < 1:
        raise ValueError("motion_burst_frames must be >= 1")

    cap = cv2.VideoCapture(path)
    prev_gray = None
    read_idx = 0
    track_idx = 0
    burst_remaining = 0

    while True:
        good, frame = cap.read()
        if not good:
            break

        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            motion_score = 0.0
        else:
            motion_score = float(np.mean(cv2.absdiff(gray, prev_gray)))
        prev_gray = gray

        if motion_score >= motion_threshold:
            burst_remaining = max(burst_remaining, motion_burst_frames)

        # Choose whether to yield this frame:
        # - During burst: yield every frame
        # - Otherwise: yield only when base_step aligns
        should_yield = burst_remaining > 0 or (read_idx % base_step == 0)
        if should_yield:
            yield track_idx, read_idx, frame
            track_idx += 1

        if burst_remaining > 0:
            burst_remaining -= 1

        read_idx += 1

    cap.release()


def get_video_fps(path: str, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fallback if fps is None or fps <= 0 else float(fps)

def save_video(frames, path: str):
    codec = cv2.VideoWriter_fourcc(*'XVID')

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, codec, 30.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def iter_video_frames_ffmpeg_hwaccel(
    path: str,
    scale: float = 1.0,
    step: int = 1,
    hwaccel: str = "auto",
):
    """
    Yield frames decoded via an ffmpeg subprocess with optional hardware decode.

    Mirrors the API of `iter_video_frames`: yields (processed_idx, frame_bgr).
    Decode runs in a subprocess so it executes in parallel with the caller's
    per-frame Python work — that's the main perf win versus cv2.VideoCapture.

    `hwaccel` selects the ffmpeg `-hwaccel` mode:
      - "auto":     omit the flag, let ffmpeg pick a backend
      - "cuda":     `-hwaccel cuda -hwaccel_output_format nv12`
      - "d3d11va":  `-hwaccel d3d11va`
      - "none":     no hardware acceleration

    Output is forced to bgr24 raw frames so the caller receives the same
    numpy layout as `iter_video_frames` (no torch tensor variant — keeps
    the CPU drawing path identical).
    """
    if step < 1:
        raise ValueError("step must be >= 1")

    ffmpeg_exe = find_ffmpeg()
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg not found (install system ffmpeg or imageio-ffmpeg)")

    cap = cv2.VideoCapture(path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"could not read video dimensions from {path}")

    if scale != 1.0:
        out_w = max(2, int(round(src_w * scale)) & ~1)  # even dims for yuv420p chains
        out_h = max(2, int(round(src_h * scale)) & ~1)
    else:
        out_w, out_h = src_w, src_h

    cmd = [ffmpeg_exe, "-nostdin", "-loglevel", "error"]
    mode = (hwaccel or "auto").lower()
    if mode == "cuda":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "nv12"]
    elif mode == "d3d11va":
        cmd += ["-hwaccel", "d3d11va"]
    elif mode == "none" or mode == "auto":
        # "auto" intentionally omits -hwaccel and lets ffmpeg pick.
        pass
    else:
        cmd += ["-hwaccel", mode]
    cmd += ["-i", path]
    if (out_w, out_h) != (src_w, src_h):
        cmd += ["-vf", f"scale={out_w}:{out_h}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-"]

    nbytes = out_w * out_h * 3
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=nbytes * 4,
    )

    processed_idx = 0
    raw_idx = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            if raw_idx % step != 0:
                raw_idx += 1
                continue
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3)
            yield processed_idx, frame
            processed_idx += 1
            raw_idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass