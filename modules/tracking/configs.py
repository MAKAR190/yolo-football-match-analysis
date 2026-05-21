from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """YOLO inference + streaming pass settings (track-pass)."""
    device: str = "cuda:0"
    use_half: bool = False
    batch: int = 32
    conf: float = 0.4
    imgsz: int = 1024
    run_quality_checks: bool = True
    scale: float = 1.0
    step: int = 1
    motion_threshold: float = 12.0
    motion_burst_frames: int = 3
    # Streaming-only: run YOLO+ByteTrack every Nth frame and reuse cached
    # tracks/owner on the in-between frames. 1 = detect every frame (default).
    # 2 = ~2x speedup with one frame of staleness (~33ms at 30fps source).
    inference_every_n_frames: int = 1
    # Streaming-only: run detection (producer) and rendering (consumer) on
    # separate threads joined by a 1-deep queue. YOLO releases the GIL during
    # CUDA inference, so per-frame time collapses toward max(yolo, rest).
    threaded: bool = True


@dataclass(frozen=True)
class BallDetectionConfig:
    """Ball candidate selection + temporal-guard settings shared by both passes."""
    # raw_candidates | byte_track | hybrid
    ball_tracking_mode: str = "hybrid"
    ball_min_candidate_confidence: float = 0.55
    ball_aspect_ratio_min: float = 0.45
    ball_aspect_ratio_max: float = 2.2
    interpolate_ball_positions: bool = False
    ball_prefilter_debug: bool = False


@dataclass(frozen=True)
class RenderOwnershipStatsConfig:
    # Render output format
    codec: str = "XVID"
    codec_preset: str = "balanced"
    use_hw_encode: bool = True
    hw_encoder: str = "h264_mf"
    team_assign_debug: bool = False

    # Ball ownership / assignment
    ball_owner_hold_frames: int = 10
    ball_owner_lock_enabled: bool = True
    ball_owner_switch_confirm_frames: int = 2
    ball_owner_switch_margin_px: float = 20.0
    ball_owner_switch_margin_ratio: float = 0.70
    ball_owner_release_distance_px: float = 105.0
    ball_assign_max_player_ball_distance_px: float = 72.0
    ball_assign_ambiguity_margin_px: float = 12.0
    # max_player_ball_distance is bounded by min(max(base, scale*h), cap) instead of max(base, h).
    ball_max_distance_height_scale: float = 0.38
    ball_max_distance_cap_px: float = 110.0

    # Require nearest candidate to persist N frames when ball center is inside player bbox.
    ball_owner_primary_confirm_frames: int = 2
    # Shorter hold when ball detection confidence is below ball_owner_hold_min_confidence.
    ball_owner_hold_min_confidence: float = 0.35
    ball_owner_hold_frames_weak_evidence: int = 5
    # Require minimum selected-ball confidence before any owner is assigned.
    ball_owner_min_confidence: float = 0.5
    # Extra confirmation when switching possession to a player on another team.
    ball_owner_switch_confirm_frames_cross_team: int = 3

    # Stats: rolling majority vote on raw owner id (0 = disabled, use raw).
    stats_owner_smoothing_window: int = 5

    # Motion-change touch (ownership only on ball motion change near player)
    motion_touch_enabled: bool = True
    # Smooth ball centers before computing motion (EMA alpha in [0,1]).
    ball_center_smoothing_alpha: float = 0.55
    # Compensate ball motion by camera movement (if enabled).
    camera_motion_compensation_enabled: bool = True
    # After ball is re-detected (following missing frames), ignore motion-touch
    # for a short cooldown window to avoid false "speed drop" touches.
    motion_touch_redetect_cooldown_frames: int = 2
    # Reset EMA/motion history on re-detection so velocities aren't computed
    # across a detection gap.
    motion_touch_reset_history_on_redetect: bool = True
    # Require a meaningful previous speed before we consider a "touch"
    motion_touch_min_prev_speed_px_per_frame: float = 10.0
    # Touch if speed drops sharply: speed_curr <= ratio * speed_prev
    motion_touch_speed_drop_ratio: float = 0.82
    # Or touch if direction changes sharply (degrees) while speed is non-trivial
    motion_touch_angle_change_deg: float = 70.0
    motion_touch_use_angle_change: bool = False
    # Candidate must be close and unambiguous to count as a "touch"
    motion_touch_max_candidate_distance_px: float = 52.0
    motion_touch_min_second_best_margin_px: float = 14.0
    # Require persistence of the motion-touch signal
    motion_touch_confirm_frames: int = 2
    # Require the same nearest candidate to persist
    motion_touch_candidate_confirm_frames: int = 2

    # Fast-ball "no owner"
    ball_in_transit_velocity_threshold_px_per_frame: float = 20.0
    ball_in_transit_confirm_frames: int = 1
    ball_in_transit_freeze_owner: bool = False
    ball_in_transit_grace_frames_after_reappear: int = 2

    # Optional overlays / computed signals
    camera_movement_enabled: bool = True
    camera_movement_overlay_enabled: bool = False
    # Sample optical flow every Nth render frame; hold the last (dx, dy) on
    # skipped frames. 1 = run every frame (original behavior).
    camera_movement_sample_every_n_frames: int = 3
    # Stats output
    stats_enabled: bool = True

    # GPU render path: ffmpeg hw decode (best effort) + torch CUDA overlays + ffmpeg hw encode.
    render_gpu_pipeline: bool = False
    # auto | cuda | d3d11va | none — passed to iter_video_frames_ffmpeg_hwaccel when pipeline on.
    render_gpu_decode_hwaccel: str = "auto"
    render_gpu_overlay: bool = True
    render_gpu_device: str = "cuda:0"
