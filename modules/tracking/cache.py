import hashlib
import os

CACHE_FOLDER = "cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)


def cache_key_path(
    *,
    model_path: str,
    pipeline_version: int,
    video_path: str,
    scale: float,
    step: int,
    batch: int,
    motion_threshold: float,
    motion_burst_frames: int,
    ball_tracking_mode: str,
    ball_min_candidate_confidence: float = 0.0,
    ball_aspect_ratio_min: float = 0.25,
    ball_aspect_ratio_max: float = 3.0,
) -> str:
    """
    Build the on-disk cache path for a track-pass result.

    The hash inputs MUST cover everything that influences track output —
    model identity, pipeline version, video identity, all preprocessing
    and ball-prefilter params. Bumping ByteTracker.PIPELINE_VERSION is
    the kill-switch that invalidates every existing cache.
    """
    raw = (
        f"{model_path}|v{pipeline_version}|{video_path}|"
        f"scale={scale}|base_step={step}|batch={batch}|"
        f"motion_threshold={motion_threshold}|motion_burst_frames={motion_burst_frames}|"
        f"ball_tracking_mode={ball_tracking_mode}|"
        f"ball_min_conf={ball_min_candidate_confidence}|"
        f"ball_ar={ball_aspect_ratio_min}-{ball_aspect_ratio_max}"
    )
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_FOLDER, f"bytetracker_cache_{h}.pkl")
