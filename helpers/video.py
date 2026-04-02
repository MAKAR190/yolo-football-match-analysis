import cv2
import numpy as np

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