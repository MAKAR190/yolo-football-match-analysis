import supervision as sv
import pickle
import os
import cv2
import time
import numpy as np
import torch
import shutil
import subprocess
import math
import re
from collections import defaultdict
from .annotations import Annotations
from .team_assigner import TeamAssigner
from .player_ball_assigner import PlayerBallAssigner
from .camera_movement_estimator import CameraMovementEstimator
from .stats import StatsManager
from .tracking.ball_prefilter import (
    BallOutputOutlierRejection,
    _ball_bbox_aspect_ratio,
    ball_candidate_prefilter,
)
from .tracking.cache import CACHE_FOLDER, cache_key_path
from .tracking.configs import (
    BallDetectionConfig,
    InferenceConfig,
    RenderOwnershipStatsConfig,
)
from ultralytics import YOLO
from helpers import (
    iter_video_frames,
    iter_video_frames_adaptive,
    get_video_fps,
    find_ffmpeg,
)

shared_annotations = Annotations()


class ByteTracker:
    # Bump this when detection/postprocessing logic changes so old caches aren't reused.
    PIPELINE_VERSION = 12

    def __init__(self, model: str):
        self.model_path = model
        self.model = YOLO(model)
        self._tracker_cfg = {
            "track_activation_threshold": 0.25,
            "lost_track_buffer": 45,
            "minimum_matching_threshold": 0.80,
            "minimum_consecutive_frames": 3,
        }
        # Stricter association settings reduce ID swaps when players cross.
        self._set_tracker_fps(fps=30.0)
        self.device = "cpu"
        self.use_half = False

    def _set_tracker_fps(self, fps: float) -> float:
        resolved_fps = float(fps) if fps and float(fps) > 0 else 30.0
        self.tracker = sv.ByteTrack(
            frame_rate=resolved_fps,
            **self._tracker_cfg,
        )
        return resolved_fps

    def configure_inference(
        self,
        device: str = "auto",
        use_half: bool = True,
    ):
        """
        Configure runtime inference mode (GPU/CPU and fp16/fp32).
        """
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if str(device).startswith("cuda") and torch.cuda.is_available():
            self.device = str(device)
            self.use_half = bool(use_half)
        else:
            self.device = "cpu"
            self.use_half = False

        print(
            f"[ByteTracker] Inference device={self.device}, half={self.use_half}"
        )

    def _predict_batch(self, frames, conf: float, batch: int, imgsz: int):
        return self.model.predict(
            frames,
            conf=conf,
            verbose=False,
            device=self.device,
            half=self.use_half,
            batch=batch,
            imgsz=imgsz,
        )

    def _cache_key_path(
        self,
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
        return cache_key_path(
            model_path=self.model_path,
            pipeline_version=self.PIPELINE_VERSION,
            video_path=video_path,
            scale=scale,
            step=step,
            batch=batch,
            motion_threshold=motion_threshold,
            motion_burst_frames=motion_burst_frames,
            ball_tracking_mode=ball_tracking_mode,
            ball_min_candidate_confidence=ball_min_candidate_confidence,
            ball_aspect_ratio_min=ball_aspect_ratio_min,
            ball_aspect_ratio_max=ball_aspect_ratio_max,
        )

    def get_tracks_from_video(
        self,
        video_path: str,
        cache: bool = True,
        scale: float = 1.0,
        step: int = 1,
        batch: int = 30,
        motion_threshold: float = 12.0,
        motion_burst_frames: int = 3,
        conf: float = 0.1,
        imgsz: int = 512,
        # "raw_candidates": old behavior (pick ball from raw YOLO dets)
        # "byte_track": use ByteTrack output for ball (more consistent IDs when it works)
        # "hybrid": prefer ByteTrack ball; fall back to raw YOLO ball when ByteTrack misses
        ball_tracking_mode: str = "raw_candidates",
        ball_min_candidate_confidence: float = 0.0,
        ball_aspect_ratio_min: float = 0.25,
        ball_aspect_ratio_max: float = 3.0,
        ball_prefilter_debug: bool = False,
    ):
        """
        Memory-safe tracking:
        - Does NOT store all frames.
        - Streams frames, runs YOLO in batches, updates ByteTrack per frame.
        - Only keeps jersey crops + track metadata in memory.
        """
        cache_path = self._cache_key_path(
            video_path=video_path,
            scale=scale,
            step=step,
            batch=batch,
            motion_threshold=motion_threshold,
            motion_burst_frames=motion_burst_frames,
            ball_tracking_mode=ball_tracking_mode,
            ball_min_candidate_confidence=ball_min_candidate_confidence,
            ball_aspect_ratio_min=ball_aspect_ratio_min,
            ball_aspect_ratio_max=ball_aspect_ratio_max,
        )

        if cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        # Ball-specific postprocessing:
        # 1) choose a single best ball candidate per frame
        # 2) reject temporal outliers so we don't "latch" onto the wrong blob
        ball_outlier_rejection = BallOutputOutlierRejection(
            max_jump_dist_norm=0.13,
            enabled=True,
        )
        ball_outlier_rejection.reset()

        prefilter_reject_counts: dict[str, int] = defaultdict(int)

        # Class ids produced by your trained model.
        BALL_CLASS_ID = 1 # 0
        PLAYER_CLASS_ID = 0 # 2
        REFEREE_CLASS_ID = 2 # 3

        class_map = {
            BALL_CLASS_ID: "ball",
            PLAYER_CLASS_ID: "player",
            REFEREE_CLASS_ID: "referee",
        }

        class_to_track_key = {
            "player": "players",
            "referee": "referees",
            "ball": "ball",
        }
        locked_referee_ids = set()

        def process_prediction(current_frame_idx: int, frame_for_crop, prediction):
            # Ensure per-frame containers exist.
            for key in tracks:
                tracks[key].append({})

            sv_det = sv.Detections.from_ultralytics(prediction)
            detection_with_tracks = self.tracker.update_with_detections(sv_det)

            det_xyxy = detection_with_tracks.xyxy
            det_class_ids = detection_with_tracks.class_id
            det_tracker_ids = detection_with_tracks.tracker_id
            det_confs = getattr(detection_with_tracks, "confidence", None)

            ball_candidates = []
            byte_ball_candidates = []

            if ball_tracking_mode in ("raw_candidates", "hybrid"):
                raw_xyxy = sv_det.xyxy
                raw_class_ids = sv_det.class_id
                raw_confs = getattr(sv_det, "confidence", None)

                for j in range(len(sv_det)):
                    raw_bbox = raw_xyxy[j].tolist()
                    raw_class_id = int(raw_class_ids[j])
                    raw_class_name = class_map.get(raw_class_id)
                    raw_track_key = class_to_track_key.get(raw_class_name)

                    if raw_class_name == "ball" and raw_track_key == "ball":
                        x1, y1, x2, y2 = map(int, raw_bbox)
                        height, width = frame_for_crop.shape[:2]
                        x1 = max(0, min(x1, width - 1))
                        x2 = max(0, min(x2, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        y2 = max(0, min(y2, height - 1))

                        area = float(max(0, x2 - x1) * max(0, y2 - y1))
                        area_frac = (
                            float(area / float(width * height))
                            if width > 0 and height > 0
                            else 0.0
                        )

                        conf_val = None
                        if raw_confs is not None:
                            try:
                                conf_val = float(raw_confs[j])
                            except Exception:
                                conf_val = None

                        score = conf_val if conf_val is not None else area_frac
                        ok_pf, pf_reason = ball_candidate_prefilter(
                            [x1, y1, x2, y2],
                            frame_for_crop.shape,
                            conf_val,
                            min_confidence=ball_min_candidate_confidence,
                            aspect_ratio_min=ball_aspect_ratio_min,
                            aspect_ratio_max=ball_aspect_ratio_max,
                        )
                        if not ok_pf:
                            if pf_reason:
                                prefilter_reject_counts[str(pf_reason)] += 1
                            continue

                        cx = float((x1 + x2) / 2.0)
                        cy = float((y1 + y2) / 2.0)
                        conf_for_track = float(conf_val) if conf_val is not None else float(score)
                        ball_candidates.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "score": float(score),
                                "confidence": conf_for_track,
                                "cx": cx,
                                "cy": cy,
                                "area_frac": float(area_frac),
                            }
                        )

            for i in range(len(detection_with_tracks)):
                bbox = det_xyxy[i].tolist()
                tracker_id = int(det_tracker_ids[i])
                class_id = int(det_class_ids[i])
                class_name = class_map.get(class_id)
                if class_name == "referee":
                    locked_referee_ids.add(tracker_id)
                if tracker_id in locked_referee_ids:
                    class_name = "referee"
                track_key = class_to_track_key.get(class_name)

                if class_name == "player":
                    x1, y1, x2, y2 = map(int, bbox)
                    height, width = frame_for_crop.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height - 1))
                    player_bbox = [x1, y1, x2, y2]
                    player_entry = {"bbox": player_bbox}
                    tracks["players"][current_frame_idx][tracker_id] = player_entry
                elif class_name == "referee" and track_key == "referees":
                    tracks["referees"][current_frame_idx][tracker_id] = {
                        "bbox": bbox
                    }
                elif (
                    ball_tracking_mode in ("byte_track", "hybrid")
                    and class_name == "ball"
                    and track_key == "ball"
                ):
                    x1, y1, x2, y2 = map(int, bbox)
                    height, width = frame_for_crop.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height - 1))

                    area = float(max(0, x2 - x1) * max(0, y2 - y1))
                    area_frac = (
                        float(area / float(width * height))
                        if width > 0 and height > 0
                        else 0.0
                    )

                    conf_val = None
                    if det_confs is not None:
                        try:
                            conf_val = float(det_confs[i])
                        except Exception:
                            conf_val = None

                    score = conf_val if conf_val is not None else area_frac
                    ok_pf, pf_reason = ball_candidate_prefilter(
                        [x1, y1, x2, y2],
                        frame_for_crop.shape,
                        conf_val,
                        min_confidence=ball_min_candidate_confidence,
                        aspect_ratio_min=ball_aspect_ratio_min,
                        aspect_ratio_max=ball_aspect_ratio_max,
                    )
                    if not ok_pf:
                        if pf_reason:
                            prefilter_reject_counts[str(pf_reason)] += 1
                        continue

                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)
                    conf_for_track = float(conf_val) if conf_val is not None else float(score)

                    byte_ball_candidates.append(
                        {
                            "tracker_id": tracker_id,
                            "bbox": [x1, y1, x2, y2],
                            "score": float(score),
                            "confidence": conf_for_track,
                            "cx": cx,
                            "cy": cy,
                        }
                    )

            # Choose a single best ball bbox and apply temporal outlier rejection.
            if ball_tracking_mode == "raw_candidates":
                if ball_candidates:
                    height, width = frame_for_crop.shape[:2]
                    diag = float(np.sqrt(width * width + height * height))
                    prev_center = ball_outlier_rejection.prev_center
                    prev_prev_center = ball_outlier_rejection.prev_prev_center

                    if prev_center is None or diag <= 0:
                        best = max(ball_candidates, key=lambda x: x["score"])
                    else:
                        prev_cx, prev_cy = prev_center
                        if prev_prev_center is not None:
                            prev_prev_cx, prev_prev_cy = prev_prev_center
                            expected_cx = (2.0 * prev_cx) - prev_prev_cx
                            expected_cy = (2.0 * prev_cy) - prev_prev_cy
                        else:
                            expected_cx, expected_cy = prev_cx, prev_cy

                        def dist_key(c):
                            dist_norm = float(
                                np.hypot(c["cx"] - expected_cx, c["cy"] - expected_cy)
                                / diag
                            )
                            return (dist_norm, -float(c["score"]))

                        best = min(ball_candidates, key=dist_key)

                    best_bbox = best["bbox"]
                    best_bbox = ball_outlier_rejection.filter_bbox(
                        bbox=best_bbox,
                        frame_shape=frame_for_crop.shape,
                    )
                    if best_bbox is not None:
                        c_out = float(best.get("confidence", best["score"]))
                        tracks["ball"][current_frame_idx][1] = {
                            "bbox": best_bbox,
                            "confidence": c_out,
                        }
            elif ball_tracking_mode == "byte_track":
                if byte_ball_candidates:
                    height, width = frame_for_crop.shape[:2]
                    diag = float(np.sqrt(width * width + height * height))
                    prev_center = ball_outlier_rejection.prev_center
                    prev_prev_center = ball_outlier_rejection.prev_prev_center

                    if prev_center is None or diag <= 0:
                        best = max(byte_ball_candidates, key=lambda x: x["score"])
                    else:
                        prev_cx, prev_cy = prev_center
                        if prev_prev_center is not None:
                            prev_prev_cx, prev_prev_cy = prev_prev_center
                            expected_cx = (2.0 * prev_cx) - prev_prev_cx
                            expected_cy = (2.0 * prev_cy) - prev_prev_cy
                        else:
                            expected_cx, expected_cy = prev_cx, prev_cy

                        def dist_key(c):
                            dist_norm = float(
                                np.hypot(c["cx"] - expected_cx, c["cy"] - expected_cy)
                                / diag
                            )
                            return (dist_norm, -float(c["score"]))

                        best = min(byte_ball_candidates, key=dist_key)

                    best_bbox = best["bbox"]
                    best_bbox = ball_outlier_rejection.filter_bbox(
                        bbox=best_bbox,
                        frame_shape=frame_for_crop.shape,
                    )
                    if best_bbox is not None:
                        c_out = float(best.get("confidence", best["score"]))
                        tracks["ball"][current_frame_idx][best["tracker_id"]] = {
                            "bbox": best_bbox,
                            "confidence": c_out,
                        }
            else:
                preferred = byte_ball_candidates if byte_ball_candidates else ball_candidates
                if preferred:
                    height, width = frame_for_crop.shape[:2]
                    diag = float(np.sqrt(width * width + height * height))
                    prev_center = ball_outlier_rejection.prev_center
                    prev_prev_center = ball_outlier_rejection.prev_prev_center

                    if prev_center is None or diag <= 0:
                        best = max(preferred, key=lambda x: x["score"])
                    else:
                        prev_cx, prev_cy = prev_center
                        if prev_prev_center is not None:
                            prev_prev_cx, prev_prev_cy = prev_prev_center
                            expected_cx = (2.0 * prev_cx) - prev_prev_cx
                            expected_cy = (2.0 * prev_cy) - prev_prev_cy
                        else:
                            expected_cx, expected_cy = prev_cx, prev_cy

                        def dist_key(c):
                            dist_norm = float(
                                np.hypot(c["cx"] - expected_cx, c["cy"] - expected_cy)
                                / diag
                            )
                            return (dist_norm, -float(c["score"]))

                        best = min(preferred, key=dist_key)

                    best_bbox = best["bbox"]
                    best_bbox = ball_outlier_rejection.filter_bbox(
                        bbox=best_bbox,
                        frame_shape=frame_for_crop.shape,
                    )
                    if best_bbox is not None:
                        key = best.get("tracker_id", 1)
                        c_out = float(best.get("confidence", best["score"]))
                        tracks["ball"][current_frame_idx][key] = {
                            "bbox": best_bbox,
                            "confidence": c_out,
                        }

        # Stream frames, run YOLO on bounded batches.
        track_original_frame_indices = []
        batch_frames = []
        batch_indices = []

        # We rely on YOLO+ByteTrack output, but do ball candidate selection +
        # temporal outlier rejection to reduce "wrong ball" frames.

        for track_idx, original_frame_idx, frame in iter_video_frames_adaptive(
            video_path,
            scale=scale,
            base_step=step,
            motion_threshold=motion_threshold,
            motion_burst_frames=motion_burst_frames,
        ):
            batch_frames.append(frame)
            batch_indices.append(track_idx)
            track_original_frame_indices.append(original_frame_idx)

            if len(batch_frames) < batch:
                continue

            objects = self._predict_batch(
                frames=batch_frames,
                conf=conf,
                batch=batch,
                imgsz=imgsz,
            )

            for local_i, prediction in enumerate(objects):
                frame_for_crop = batch_frames[local_i]
                current_frame_idx = batch_indices[local_i]
                process_prediction(
                    current_frame_idx=current_frame_idx,
                    frame_for_crop=frame_for_crop,
                    prediction=prediction,
                )

            batch_frames.clear()
            batch_indices.clear()

        # Flush remaining frames (last partial batch).
        if batch_frames:
            objects = self._predict_batch(
                frames=batch_frames,
                conf=conf,
                batch=len(batch_frames),
                imgsz=imgsz,
            )
            for local_i, prediction in enumerate(objects):
                frame_for_crop = batch_frames[local_i]
                current_frame_idx = batch_indices[local_i]
                process_prediction(
                    current_frame_idx=current_frame_idx,
                    frame_for_crop=frame_for_crop,
                    prediction=prediction,
                )

        if ball_prefilter_debug and prefilter_reject_counts:
            print(
                "[ByteTracker] Ball prefilter reject counts:",
                dict(prefilter_reject_counts),
            )

        if cache:
            with open(cache_path, "wb") as f:
                pickle.dump((tracks, track_original_frame_indices), f)

        return tracks, track_original_frame_indices

    @staticmethod
    def interpolate_ball_positions(ball_tracks):
        """
        Fill missing ball bboxes by linear interpolation (and edge fill).

        This mirrors the idea from the referenced repo's pandas-based approach, but uses
        NumPy only.

        Input:
          ball_tracks: List[Dict[track_id, {"bbox": [x1,y1,x2,y2]}]]

        Output:
          List[Dict[int, {"bbox": [x1,y1,x2,y2]}]] with a single key=1 when any ball
          position is recoverable; empty dicts if the ball is never seen.
        """
        if not ball_tracks:
            return ball_tracks

        n = len(ball_tracks)
        bboxes = [None] * n

        for i, frame_dict in enumerate(ball_tracks):
            if not frame_dict:
                continue
            # If multiple entries exist, take the first one (your pipeline is already
            # selecting a single "best" ball per frame).
            first_ball = next(iter(frame_dict.values()), None)
            if isinstance(first_ball, dict):
                bb = first_ball.get("bbox")
                if bb and len(bb) >= 4:
                    bboxes[i] = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]

        known_idx = [i for i, bb in enumerate(bboxes) if bb is not None]
        if not known_idx:
            return [{} for _ in range(n)]

        known = np.asarray([bboxes[i] for i in known_idx], dtype=np.float64)  # (k,4)
        x_known = np.asarray(known_idx, dtype=np.float64)
        x_all = np.arange(n, dtype=np.float64)

        out = np.zeros((n, 4), dtype=np.float64)
        for d in range(4):
            out[:, d] = np.interp(x_all, x_known, known[:, d])

        # Rebuild in the "legacy" shape expected by downstream code.
        return [{1: {"bbox": out[i, :].tolist()}} for i in range(n)]

    def render_video_from_tracks(
        self,
        video_path: str,
        output_path: str,
        tracks,
        track_original_frame_indices,
        scale: float = 1.0,
        fps=None,
        render_cfg: RenderOwnershipStatsConfig = RenderOwnershipStatsConfig(),
    ):
        """
        Render annotations to disk without building `output_frames` in memory.

        Output video is rendered with `step=1` to keep the same timeline as the input.
        Since tracking may skip frames adaptively, we map each rendered frame
        to the most recent tracked frame using `track_original_frame_indices`.

        Ball-owner highlight can **lag** the visible ball-on-foot moment because
        ``ball_owner_switch_confirm_frames`` requires consecutive frames before a
        switch; lock/margin rules resist changes; and the assigner may return no
        owner when two players are within ``ball_assign_ambiguity_margin_px``.
        Fast ball motion above ``ball_in_transit_velocity_threshold_px_per_frame`` for
        ``ball_in_transit_confirm_frames`` consecutive frames clears assignment
        unless you set ``ball_in_transit_freeze_owner=True``.
        """
        fps = fps if fps is not None else get_video_fps(video_path)
        if fps <= 0:
            fps = 30.0

        cfg = render_cfg
        use_gpu_overlay = bool(
            cfg.render_gpu_pipeline and cfg.render_gpu_overlay and torch.cuda.is_available()
        )
        draw_render_overlays_gpu = None
        if use_gpu_overlay:
            try:
                from .gpu_overlay import draw_render_overlays_gpu as draw_render_overlays_gpu
            except ImportError:
                # gpu_overlay module not implemented yet — fall back to CPU
                # drawing while still using GPU decode/encode if requested.
                use_gpu_overlay = False

        # Alias config fields so the rest of the function logic stays unchanged.
        codec = cfg.codec
        codec_preset = cfg.codec_preset
        use_hw_encode = cfg.use_hw_encode
        hw_encoder = cfg.hw_encoder
        team_assign_debug = cfg.team_assign_debug

        ball_owner_hold_frames = cfg.ball_owner_hold_frames
        ball_owner_lock_enabled = cfg.ball_owner_lock_enabled
        ball_owner_switch_confirm_frames = cfg.ball_owner_switch_confirm_frames
        ball_owner_switch_margin_px = cfg.ball_owner_switch_margin_px
        ball_owner_switch_margin_ratio = cfg.ball_owner_switch_margin_ratio
        ball_owner_release_distance_px = cfg.ball_owner_release_distance_px
        ball_assign_max_player_ball_distance_px = cfg.ball_assign_max_player_ball_distance_px
        ball_assign_ambiguity_margin_px = cfg.ball_assign_ambiguity_margin_px

        ball_in_transit_velocity_threshold_px_per_frame = (
            cfg.ball_in_transit_velocity_threshold_px_per_frame
        )
        ball_in_transit_confirm_frames = cfg.ball_in_transit_confirm_frames
        ball_in_transit_freeze_owner = cfg.ball_in_transit_freeze_owner
        ball_in_transit_grace_frames_after_reappear = (
            cfg.ball_in_transit_grace_frames_after_reappear
        )

        camera_movement_enabled = cfg.camera_movement_enabled
        camera_movement_overlay_enabled = cfg.camera_movement_overlay_enabled

        stats_enabled = cfg.stats_enabled
        # No foot-based debug in this pipeline.

        stats_manager = None
        if stats_enabled:
            stats_manager = StatsManager(
                fps=float(fps),
                stats_owner_smoothing_window=int(cfg.stats_owner_smoothing_window),
            )

        ffmpeg_exe = find_ffmpeg()
        frames_iter = None
        if cfg.render_gpu_pipeline and ffmpeg_exe:
            try:
                from helpers.video import iter_video_frames_ffmpeg_hwaccel
                frames_iter = iter_video_frames_ffmpeg_hwaccel(
                    video_path,
                    scale=scale,
                    step=1,
                    hwaccel=cfg.render_gpu_decode_hwaccel,
                )
            except Exception as e:
                print(f"[ByteTracker] Render decode: ffmpeg failed ({e}); using OpenCV VideoCapture")
                frames_iter = None
        if frames_iter is None:
            frames_iter = iter_video_frames(video_path, scale=scale, step=1)
        first = next(frames_iter, None)
        if first is None:
            return None
        first_frame_idx, first_frame = first

        height, width = first_frame.shape[:2]
        preset_to_codec = {
            # Usually fastest to encode on many Windows setups.
            "fast": "MJPG",
            # Existing default behavior.
            "balanced": codec,
            # Lower CPU load alternative (can vary by environment).
            "compat": "XVID",
        }
        writer = None
        ffmpeg_proc = None
        ffmpeg_stdin = None
        use_ffmpeg_writer = False
        output_ext = os.path.splitext(output_path)[1].lower()

        def get_fallback_codec_name():
            codec_name = preset_to_codec.get(codec_preset, codec)
            # OpenCV+MP4 only supports a subset of fourccs; avoid MJPG/XVID in .mp4.
            if output_ext == ".mp4":
                u = codec_name.upper()
                if u in ("XVID", "MJPG", "MJPEG"):
                    return "mp4v"
            return codec_name

        selected_encoder = None
        if use_hw_encode and ffmpeg_exe:
            encoders_text = ""
            try:
                encoders_probe = subprocess.run(
                    [ffmpeg_exe, "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                encoders_text = (encoders_probe.stdout or "") + (encoders_probe.stderr or "")
            except Exception:
                encoders_text = ""

            def _ff_encoder_available(name: str) -> bool:
                return re.search(rf"\b{re.escape(name)}\b", encoders_text) is not None

            priority = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_mf", "libx264"]
            available_encoders = [c for c in priority if _ff_encoder_available(c)]

            if hw_encoder != "auto":
                if _ff_encoder_available(hw_encoder):
                    selected_encoder = hw_encoder
            elif available_encoders:
                selected_encoder = available_encoders[0]

            if selected_encoder is not None:
                ffmpeg_cmd = [
                    ffmpeg_exe,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    f"{float(fps)}",
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    selected_encoder,
                ]
                if selected_encoder == "h264_nvenc":
                    ffmpeg_cmd += [
                        "-preset",
                        "p5",
                        "-rc",
                        "vbr",
                        "-cq",
                        "19",
                        "-b:v",
                        "0",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                elif selected_encoder == "h264_qsv":
                    ffmpeg_cmd += ["-preset", "medium", "-global_quality", "20", "-pix_fmt", "nv12"]
                elif selected_encoder == "h264_amf":
                    ffmpeg_cmd += ["-quality", "balanced", "-rc", "cqp", "-qp_i", "19", "-qp_p", "20", "-pix_fmt", "yuv420p"]
                elif selected_encoder == "h264_mf":
                    # h264_mf quality scale: lower is better (~18-22 for high quality on motion content)
                    ffmpeg_cmd += ["-rate_control", "quality", "-quality", "85", "-pix_fmt", "yuv420p"]
                elif selected_encoder == "libx264":
                    ffmpeg_cmd += ["-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p"]
                else:
                    ffmpeg_cmd += ["-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p"]

                # Browser-friendly mp4: move moov atom to the start so playback can begin
                # before the file finishes downloading.
                if output_ext == ".mp4":
                    ffmpeg_cmd += ["-movflags", "+faststart"]

                ffmpeg_cmd += [output_path]
                try:
                    # stderr must not be PIPE without a reader: ffmpeg fills the buffer and
                    # blocks, which then breaks stdin and triggers BrokenPipeError in Python.
                    ffmpeg_proc = subprocess.Popen(
                        ffmpeg_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    ffmpeg_stdin = ffmpeg_proc.stdin
                    use_ffmpeg_writer = ffmpeg_stdin is not None
                except Exception:
                    ffmpeg_proc = None
                    ffmpeg_stdin = None
                    use_ffmpeg_writer = False

        if not use_ffmpeg_writer:
            codec_name = get_fallback_codec_name()
            codec_tag = cv2.VideoWriter_fourcc(*codec_name)
            writer = cv2.VideoWriter(output_path, codec_tag, float(fps), (width, height))
        elif use_ffmpeg_writer and selected_encoder is not None:
            print(f"[ByteTracker] Render encode: ffmpeg -vcodec {selected_encoder}")

        def write_output_frame(frame_to_write):
            nonlocal use_ffmpeg_writer, ffmpeg_proc, ffmpeg_stdin, writer
            if use_ffmpeg_writer:
                try:
                    ffmpeg_stdin.write(frame_to_write.tobytes())
                    return
                except (BrokenPipeError, OSError):
                    # HW encoder failed mid-run (unsupported encoder/device/session).
                    # Fall back to OpenCV writer to avoid crashing the whole analysis.
                    dead = ffmpeg_proc
                    try:
                        if ffmpeg_stdin is not None:
                            ffmpeg_stdin.close()
                    except Exception:
                        pass
                    try:
                        if dead is not None:
                            dead.wait(timeout=2)
                    except Exception:
                        pass

                    use_ffmpeg_writer = False
                    ffmpeg_stdin = None
                    ffmpeg_proc = None

                    if writer is None:
                        codec_name = get_fallback_codec_name()
                        codec_tag = cv2.VideoWriter_fourcc(*codec_name)
                        writer = cv2.VideoWriter(
                            output_path, codec_tag, float(fps), (width, height)
                        )
                    print(
                        "[ByteTracker] Warning: ffmpeg hw encode failed, "
                        "falling back to OpenCV writer."
                    )

            writer.write(frame_to_write)

        # Pointer into track_original_frame_indices (track idx).
        track_ptr = 0
        while (
            track_ptr + 1 < len(track_original_frame_indices)
            and track_original_frame_indices[track_ptr + 1] <= first_frame_idx
        ):
            track_ptr += 1

        idx0 = track_ptr
        frame = first_frame.copy()
        player_tracks = tracks["players"][idx0]
        ball_tracks = tracks["ball"][idx0]
        referee_tracks = tracks["referees"][idx0]
        team_assigner = TeamAssigner()
        ball_assigner = PlayerBallAssigner(
            max_player_ball_distance=ball_assign_max_player_ball_distance_px,
            ambiguity_margin_px=float(ball_assign_ambiguity_margin_px),
            max_distance_height_scale=float(cfg.ball_max_distance_height_scale),
            max_distance_cap_px=float(cfg.ball_max_distance_cap_px),
        )
        cam_estimator = CameraMovementEstimator(first_frame) if camera_movement_enabled else None

        if camera_movement_enabled and "camera_movement" not in tracks:
            tracks["camera_movement"] = [(0.0, 0.0)] * len(tracks["players"])

        # Store which player is currently closest to the ball for each tracked frame.
        if "ball_owner" not in tracks:
            tracks["ball_owner"] = [-1] * len(tracks["players"])

        if len(player_tracks) > 0:
            team_assigner.assign_team_color(frame, player_tracks)
        prev_frame_player_states = []
        canonical_team_by_track = {}

        def bbox_iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            denom = area_a + area_b - inter
            return 0.0 if denom <= 0 else float(inter / denom)

        def apply_player_team_colors(current_frame, current_player_tracks):
            nonlocal prev_frame_player_states
            nonlocal canonical_team_by_track
            if team_assigner.kmeans is None and len(current_player_tracks) > 0:
                team_assigner.assign_team_color(current_frame, current_player_tracks)
            elif team_assigner.should_retry_team_fit(track_ptr) and len(current_player_tracks) > 0:
                team_assigner.assign_team_color(current_frame, current_player_tracks)
            current_states = []
            for player_id, player in current_player_tracks.items():
                classified_team_id = team_assigner.get_player_team(
                    current_frame,
                    player["bbox"],
                    player_id,
                )
                team_id = classified_team_id

                # Visual continuity guard: when IDs switch at intersections,
                # inherit previous-frame team if bbox overlap is strong.
                best_iou = 0.0
                inherited_team_id = None
                inherited_track_id = None
                for prev_state in prev_frame_player_states:
                    iou = bbox_iou(player["bbox"], prev_state["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        inherited_team_id = prev_state["team_id"]
                        inherited_track_id = prev_state["track_id"]
                if (
                    best_iou >= 0.5
                    and inherited_team_id is not None
                    and classified_team_id is None
                ):
                    team_id = inherited_team_id
                elif (
                    best_iou >= 0.6
                    and inherited_team_id is not None
                    and classified_team_id is not None
                    and team_assigner.player_team_dict.get(player_id) is None
                ):
                    # Weakly classified new IDs can still borrow continuity.
                    team_id = inherited_team_id

                if team_id is None:
                    team_id = canonical_team_by_track.get(player_id)

                if team_id is not None:
                    canonical_team_by_track[player_id] = team_id
                    if inherited_track_id is not None and best_iou >= 0.6:
                        canonical_team_by_track[player_id] = canonical_team_by_track.get(
                            inherited_track_id,
                            team_id,
                        )
                        canonical_team_by_track[inherited_track_id] = canonical_team_by_track[player_id]

                player["team_id"] = team_id
                player["team_color"] = team_assigner.team_colors.get(
                    team_id,
                    team_assigner.unknown_color,
                )
                if team_assign_debug:
                    player["team_roi_bbox"] = team_assigner.get_torso_roi_bbox(
                        current_frame.shape,
                        player["bbox"],
                    )
                current_states.append(
                    {"bbox": player["bbox"], "team_id": team_id, "track_id": player_id}
                )
            prev_frame_player_states = current_states

        apply_player_team_colors(frame, player_tracks)

        # Camera motion compensation state (cumulative shift in pixels).
        cam_cum_dx = 0.0
        cam_cum_dy = 0.0
        # Throttle optical flow: sample every Nth frame, hold last value otherwise.
        cam_sample_n = max(1, int(cfg.camera_movement_sample_every_n_frames))
        last_cam_movement = (0.0, 0.0)

        # Ball possession hold:
        # If the ball track disappears (camera blur / detector miss), keep the last
        # owner for a short window so possession doesn't instantly drop to -1.
        last_ball_owner_id = -1
        last_ball_center = None  # (cx, cy) in pixels
        last_ball_center_prev = None  # (cx, cy) in pixels (previous previous)
        ball_missing_frames = 0
        last_ball_owner_bbox = None  # [x1,y1,x2,y2] of last known owner in pixels
        switch_candidate_id = -1
        switch_candidate_streak = 0
        ball_in_transit_grace_remaining = 0
        ball_in_transit_fast_streak_frames = 0
        # Smoothed (EMA) ball centers (after optional camera compensation)
        ball_center_ema = None
        ball_center_prev_ema = None
        ball_center_prev_prev_ema = None
        # Motion-touch persistence and candidate persistence
        motion_touch_streak = 0
        candidate_streak = 0
        last_candidate_id = -1
        # Consecutive frames where assigner picks the same player (before bbox gates).
        prev_raw_assign_id = -1
        prev_raw_assign_streak = 0
        # Ball re-detection handling for motion-touch
        ball_detected_prev = False
        redetect_cooldown_remaining = 0

        def get_bbox_center(bbox):
            return (
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                (float(bbox[1]) + float(bbox[3])) / 2.0,
            )

        def bbox_iou_quick(a, b):
            ax1, ay1, ax2, ay2 = map(float, a[:4])
            bx1, by1, bx2, by2 = map(float, b[:4])
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            denom = area_a + area_b - inter
            return 0.0 if denom <= 0 else float(inter / denom)

        def remap_owner_if_missing(owner_id, player_tracks_local):
            nonlocal last_ball_owner_bbox
            if owner_id == -1:
                return -1
            if owner_id in player_tracks_local:
                last_ball_owner_bbox = player_tracks_local[owner_id].get("bbox")
                return owner_id

            # Player track-id swapped/changed: map by bbox overlap to keep highlighting.
            if not last_ball_owner_bbox:
                return -1
            best_iou = 0.0
            best_id = -1
            for pid, p in player_tracks_local.items():
                pb = p.get("bbox")
                if not pb:
                    continue
                iou = bbox_iou_quick(last_ball_owner_bbox, pb)
                if iou > best_iou:
                    best_iou = iou
                    best_id = pid
            if best_iou >= 0.5 and best_id != -1:
                last_ball_owner_bbox = player_tracks_local[best_id].get("bbox")
                return best_id
            return -1

        def select_ball_bbox(ball_tracks_local, prev_center):
            if not ball_tracks_local:
                return None, None, None

            best_bbox = None
            best_center = None
            best_conf = None

            if prev_center is None:
                best_area = -1.0
                for _, ball in ball_tracks_local.items():
                    bbox = ball.get("bbox") if isinstance(ball, dict) else None
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = map(float, bbox[:4])
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_bbox = bbox
                        best_center = get_bbox_center(bbox)
                        if isinstance(ball, dict):
                            c = ball.get("confidence")
                            if c is not None:
                                try:
                                    best_conf = float(c)
                                except (TypeError, ValueError):
                                    best_conf = None
                return best_bbox, best_center, best_conf

            prev_cx, prev_cy = prev_center
            best_dist = float("inf")
            for _, ball in ball_tracks_local.items():
                bbox = ball.get("bbox") if isinstance(ball, dict) else None
                if not bbox:
                    continue
                cx, cy = get_bbox_center(bbox)
                dist = float(np.hypot(cx - prev_cx, cy - prev_cy))
                if dist < best_dist:
                    best_dist = dist
                    best_bbox = bbox
                    best_center = (cx, cy)
                    if isinstance(ball, dict):
                        c = ball.get("confidence")
                        if c is not None:
                            try:
                                best_conf = float(c)
                            except (TypeError, ValueError):
                                best_conf = None
            return best_bbox, best_center, best_conf

        def compute_ball_owner(player_tracks_local, ball_tracks_local):
            nonlocal last_ball_owner_id
            nonlocal last_ball_center
            nonlocal last_ball_center_prev
            nonlocal cam_cum_dx, cam_cum_dy
            nonlocal ball_missing_frames
            nonlocal last_ball_owner_bbox
            nonlocal switch_candidate_id
            nonlocal switch_candidate_streak
            nonlocal ball_in_transit_grace_remaining
            nonlocal ball_in_transit_fast_streak_frames
            nonlocal ball_center_ema, ball_center_prev_ema
            nonlocal motion_touch_streak, candidate_streak, last_candidate_id
            nonlocal ball_center_prev_prev_ema
            nonlocal ball_detected_prev, redetect_cooldown_remaining
            nonlocal prev_raw_assign_id, prev_raw_assign_streak

            def _ball_track_confidence():
                for b in ball_tracks_local.values():
                    if isinstance(b, dict):
                        c = b.get("confidence")
                        if c is not None:
                            try:
                                return float(c)
                            except (TypeError, ValueError):
                                pass
                return None

            def _effective_hold_limit_with_detection():
                """Shorter hold when ball bbox has low detector confidence."""
                c = _ball_track_confidence()
                if c is None:
                    return ball_owner_hold_frames
                if c >= float(render_cfg.ball_owner_hold_min_confidence):
                    return ball_owner_hold_frames
                return max(1, int(render_cfg.ball_owner_hold_frames_weak_evidence))

            prev_ball_center = last_ball_center
            prev_prev_ball_center = last_ball_center_prev
            ball_bbox, ball_center, ball_confidence = select_ball_bbox(ball_tracks_local, last_ball_center)
            if ball_bbox is None:
                ball_detected_prev = False
                ball_missing_frames += 1
                ball_in_transit_grace_remaining = 0
                # Drop last center so when the ball reappears we do not treat the jump
                # vs a pre-gap position as high-speed in-transit (fixes missed owner
                # after brief non-detection).
                last_ball_center = None
                last_ball_center_prev = None
                ball_center_ema = None
                ball_center_prev_ema = None
                ball_center_prev_prev_ema = None
                motion_touch_streak = 0
                candidate_streak = 0
                last_candidate_id = -1
                redetect_cooldown_remaining = 0
                prev_raw_assign_id = -1
                prev_raw_assign_streak = 0
                if ball_missing_frames <= ball_owner_hold_frames:
                    return remap_owner_if_missing(last_ball_owner_id, player_tracks_local)
                return -1

            # Ball is detected this frame. If it was missing on the previous frame,
            # start a short cooldown window for motion-touch.
            if not ball_detected_prev:
                ball_detected_prev = True
                redetect_cooldown_remaining = max(
                    0, int(render_cfg.motion_touch_redetect_cooldown_frames)
                )
                if render_cfg.motion_touch_reset_history_on_redetect:
                    ball_center_ema = None
                    ball_center_prev_ema = None
                    ball_center_prev_prev_ema = None
                    motion_touch_streak = 0

            if ball_missing_frames > 0 and int(ball_in_transit_grace_frames_after_reappear) > 0:
                ball_in_transit_grace_remaining = max(
                    ball_in_transit_grace_remaining,
                    int(ball_in_transit_grace_frames_after_reappear),
                )

            assignment = ball_assigner.assign_ball_to_player_with_metrics(
                player_tracks_local, ball_bbox
            )

            if assignment is not None:
                owner_candidate = assignment.player_id
                candidate_dist = float(assignment.distance_px)
                second_best_dist = float(assignment.second_best_distance_px)
                this_raw_id = int(assignment.player_id)
            else:
                owner_candidate = -1
                candidate_dist = float("inf")
                second_best_dist = float("inf")
                this_raw_id = -1

            min_owner_conf = float(render_cfg.ball_owner_min_confidence)
            if (
                min_owner_conf > 0.0
                and ball_confidence is not None
                and float(ball_confidence) < min_owner_conf
            ):
                owner_candidate = -1
                candidate_dist = float("inf")
                second_best_dist = float("inf")
                this_raw_id = -1

            if this_raw_id != -1:
                if this_raw_id == prev_raw_assign_id:
                    raw_assign_streak_this = prev_raw_assign_streak + 1
                else:
                    raw_assign_streak_this = 1
            else:
                raw_assign_streak_this = 0

            def _commit_raw_assign_state():
                nonlocal prev_raw_assign_id, prev_raw_assign_streak
                prev_raw_assign_id = this_raw_id
                prev_raw_assign_streak = (
                    raw_assign_streak_this if this_raw_id != -1 else 0
                )

            candidate_streak_if_kept = 0
            if owner_candidate != -1:
                if owner_candidate == last_candidate_id:
                    candidate_streak_if_kept = candidate_streak + 1
                else:
                    candidate_streak_if_kept = 1

            # Primary strict gate: ball center inside candidate bbox.
            # Tight fallback (existing config only): allow near-foot ownership when
            # candidate is close, unambiguous, and persists briefly.
            if owner_candidate != -1 and ball_center is not None:
                cand = player_tracks_local.get(owner_candidate, {})
                pb = cand.get("bbox") if isinstance(cand, dict) else None
                if not pb or len(pb) < 4:
                    owner_candidate = -1
                    candidate_dist = float("inf")
                else:
                    x1, y1, x2, y2 = map(float, pb[:4])
                    cx, cy = float(ball_center[0]), float(ball_center[1])
                    inside = (x1 <= cx <= x2) and (y1 <= cy <= y2)
                    if inside:
                        need_pf = max(1, int(render_cfg.ball_owner_primary_confirm_frames))
                        if raw_assign_streak_this < need_pf:
                            owner_candidate = -1
                            candidate_dist = float("inf")
                    elif not inside:
                        fallback_max_dist = min(
                            float(ball_assign_max_player_ball_distance_px),
                            float(render_cfg.motion_touch_max_candidate_distance_px),
                        )
                        fallback_min_margin = max(
                            float(ball_assign_ambiguity_margin_px),
                            float(render_cfg.motion_touch_min_second_best_margin_px),
                        )
                        margin = float(second_best_dist - candidate_dist)
                        fallback_confirm = max(
                            1, int(render_cfg.motion_touch_candidate_confirm_frames)
                        )
                        # Short re-detect grace using existing transit grace window.
                        if ball_missing_frames > 0 and int(ball_in_transit_grace_frames_after_reappear) > 0:
                            fallback_confirm = 1

                        fallback_ok = (
                            candidate_dist <= fallback_max_dist
                            and margin >= fallback_min_margin
                            and candidate_streak_if_kept >= fallback_confirm
                        )
                        if not fallback_ok:
                            owner_candidate = -1
                            candidate_dist = float("inf")

            # Candidate persistence gate: require nearest candidate to persist for N frames.
            if owner_candidate != -1:
                if owner_candidate == last_candidate_id:
                    candidate_streak += 1
                else:
                    last_candidate_id = owner_candidate
                    candidate_streak = 1
            else:
                last_candidate_id = -1
                candidate_streak = 0

            # Motion-change touch rule:
            # A "touch" is detected when ball motion changes sharply between frames
            # while the ball is close enough to the best player candidate.
            motion_touch = False
            # Camera-motion compensated, smoothed ball center for motion estimation.
            ball_center_for_motion = None
            if ball_center is not None:
                cx, cy = float(ball_center[0]), float(ball_center[1])
                if render_cfg.camera_motion_compensation_enabled:
                    cx -= float(cam_cum_dx)
                    cy -= float(cam_cum_dy)
                raw = (cx, cy)
                alpha = float(render_cfg.ball_center_smoothing_alpha)
                alpha = max(0.0, min(1.0, alpha))
                if ball_center_ema is None:
                    ball_center_ema = raw
                else:
                    ball_center_ema = (
                        alpha * raw[0] + (1.0 - alpha) * float(ball_center_ema[0]),
                        alpha * raw[1] + (1.0 - alpha) * float(ball_center_ema[1]),
                    )
                ball_center_for_motion = ball_center_ema

            if (
                render_cfg.motion_touch_enabled
                and redetect_cooldown_remaining <= 0
                and ball_center_prev_prev_ema is not None
                and ball_center_prev_ema is not None
                and ball_center_for_motion is not None
                and owner_candidate != -1
                and assignment is not None
            ):
                # Use EMA centers for motion estimation:
                # v1 = c_{t-1} - c_{t-2}, v2 = c_t - c_{t-1}
                v1x = float(ball_center_prev_ema[0] - ball_center_prev_prev_ema[0])
                v1y = float(ball_center_prev_ema[1] - ball_center_prev_prev_ema[1])
                v2x = float(ball_center_for_motion[0] - ball_center_prev_ema[0])
                v2y = float(ball_center_for_motion[1] - ball_center_prev_ema[1])
                s1 = float(math.hypot(v1x, v1y))
                s2 = float(math.hypot(v2x, v2y))

                min_prev = float(render_cfg.motion_touch_min_prev_speed_px_per_frame)
                drop_ratio = float(render_cfg.motion_touch_speed_drop_ratio)
                angle_deg = float(render_cfg.motion_touch_angle_change_deg)
                use_angle = bool(render_cfg.motion_touch_use_angle_change)
                max_dist = float(render_cfg.motion_touch_max_candidate_distance_px)
                min_margin = float(render_cfg.motion_touch_min_second_best_margin_px)
                touch_confirm = max(1, int(render_cfg.motion_touch_confirm_frames))
                cand_confirm = max(1, int(render_cfg.motion_touch_candidate_confirm_frames))

                angle_change_ok = False
                if use_angle and s1 > 1e-6 and s2 > 1e-6:
                    dot = (v1x * v2x) + (v1y * v2y)
                    denom = s1 * s2
                    c = max(-1.0, min(1.0, dot / denom))
                    ang = float(math.degrees(math.acos(c)))
                    angle_change_ok = ang >= angle_deg

                speed_drop_ok = s1 >= min_prev and s2 <= (drop_ratio * s1)
                # Candidate proximity + ambiguity gate.
                second_best = float(assignment.second_best_distance_px)
                margin = second_best - candidate_dist
                candidate_ok = candidate_dist <= max_dist and margin >= min_margin
                candidate_ok = candidate_ok and (candidate_streak >= cand_confirm)

                # Ignore touch detection when ball is "in transit" fast in image space.
                v_thresh = float(ball_in_transit_velocity_threshold_px_per_frame)
                if v_thresh > 0.0 and (s1 > v_thresh or s2 > v_thresh):
                    candidate_ok = False

                instant_touch = candidate_ok and (speed_drop_ok or angle_change_ok)
                if instant_touch:
                    motion_touch_streak += 1
                else:
                    motion_touch_streak = 0
                motion_touch = motion_touch_streak >= touch_confirm
            else:
                motion_touch_streak = 0

            if redetect_cooldown_remaining > 0:
                redetect_cooldown_remaining -= 1

            # Advance EMA history (c_{t-2} <- c_{t-1} <- c_t)
            if ball_center_for_motion is not None:
                ball_center_prev_prev_ema = ball_center_prev_ema
                ball_center_prev_ema = ball_center_for_motion

            velocity_reject_hold = False
            v_thresh = float(ball_in_transit_velocity_threshold_px_per_frame)
            confirm_frames = max(1, int(ball_in_transit_confirm_frames))
            skip_in_transit = ball_in_transit_grace_remaining > 0
            if skip_in_transit:
                ball_in_transit_grace_remaining -= 1
            # Always apply the high-speed rejection logic when we have two
            # consecutive ball centers; the grace window only affects how we
            # *count* streaks, not whether speed-high frames clear ownership.
            if v_thresh > 0.0 and prev_ball_center is not None and ball_center is not None:
                dpp = float(
                    np.hypot(
                        ball_center[0] - prev_ball_center[0],
                        ball_center[1] - prev_ball_center[1],
                    )
                )
                if dpp > v_thresh:
                    ball_in_transit_fast_streak_frames += 1
                    if ball_in_transit_fast_streak_frames >= confirm_frames:
                        margin_now = float(second_best_dist - candidate_dist)
                        velocity_exception_ok = (
                            owner_candidate != -1
                            and candidate_dist <= float(render_cfg.motion_touch_max_candidate_distance_px)
                            and margin_now >= max(
                                float(ball_assign_ambiguity_margin_px),
                                float(render_cfg.motion_touch_min_second_best_margin_px),
                            )
                            and candidate_streak >= max(
                                1, int(render_cfg.motion_touch_candidate_confirm_frames)
                            )
                            and ball_missing_frames <= int(ball_in_transit_grace_frames_after_reappear)
                        )
                        if velocity_exception_ok:
                            pass
                        elif ball_in_transit_freeze_owner:
                            frozen = remap_owner_if_missing(last_ball_owner_id, player_tracks_local)
                            if frozen != -1:
                                owner_candidate = frozen
                                oj = ball_assigner.assign_ball_to_player_with_metrics(
                                    {frozen: player_tracks_local.get(frozen, {})},
                                    ball_bbox,
                                )
                                candidate_dist = (
                                    float(oj.distance_px) if oj is not None else float("inf")
                                )
                            else:
                                owner_candidate = -1
                                candidate_dist = float("inf")
                                velocity_reject_hold = True
                        else:
                            owner_candidate = -1
                            candidate_dist = float("inf")
                            velocity_reject_hold = True
                else:
                    ball_in_transit_fast_streak_frames = 0
            else:
                # Can't evaluate (no prev center).
                ball_in_transit_fast_streak_frames = 0

            # Update motion history.
            last_ball_center_prev = prev_ball_center
            last_ball_center = ball_center

            # If ball is present but we couldn't assign, treat like a short "missing" segment.
            if owner_candidate == -1:
                if velocity_reject_hold:
                    ball_missing_frames = 0
                    switch_candidate_id = -1
                    switch_candidate_streak = 0
                    _commit_raw_assign_state()
                    return -1
                ball_missing_frames += 1
                if ball_missing_frames <= _effective_hold_limit_with_detection():
                    _commit_raw_assign_state()
                    return remap_owner_if_missing(last_ball_owner_id, player_tracks_local)
                # Reset switch candidate when we have no evidence.
                switch_candidate_id = -1
                switch_candidate_streak = 0
                _commit_raw_assign_state()
                return -1

            # We have an assignment candidate this frame.
            ball_missing_frames = 0

            if not ball_owner_lock_enabled or last_ball_owner_id == -1:
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                _commit_raw_assign_state()
                return remap_owner_if_missing(owner_candidate, player_tracks_local)

            current_owner_id = remap_owner_if_missing(last_ball_owner_id, player_tracks_local)
            if current_owner_id == -1:
                # Owner ID vanished and couldn't be remapped: accept the new candidate.
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                _commit_raw_assign_state()
                return remap_owner_if_missing(owner_candidate, player_tracks_local)

            # If candidate equals current owner, keep it and clear any switch attempt.
            if owner_candidate == current_owner_id:
                last_ball_owner_id = current_owner_id
                last_ball_owner_bbox = player_tracks_local.get(current_owner_id, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                _commit_raw_assign_state()
                return current_owner_id

            # If motion-touch is detected, assign ownership immediately to the
            # closest candidate for this frame.
            if motion_touch:
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                _commit_raw_assign_state()
                return remap_owner_if_missing(owner_candidate, player_tracks_local)

            # Compute distance of current owner to ball (for release/margin decisions).
            owner_assignment = ball_assigner.assign_ball_to_player_with_metrics(
                {current_owner_id: player_tracks_local.get(current_owner_id, {})},
                ball_bbox,
            )
            owner_dist = float(owner_assignment.distance_px) if owner_assignment is not None else float("inf")

            # Rule A: if current owner is clearly no longer close to the ball, allow faster switch.
            owner_released = owner_dist > float(ball_owner_release_distance_px)

            # Rule B: candidate must be meaningfully better than current owner.
            margin_px = owner_dist - candidate_dist
            margin_ratio_ok = candidate_dist <= float(ball_owner_switch_margin_ratio) * owner_dist
            margin_ok = (margin_px >= float(ball_owner_switch_margin_px)) or margin_ratio_ok

            if not owner_released and not margin_ok:
                # Not convincing enough to start/continue a switch.
                switch_candidate_id = -1
                switch_candidate_streak = 0
                _commit_raw_assign_state()
                return current_owner_id

            # Candidate looks plausible; require a short streak to confirm.
            if switch_candidate_id == owner_candidate:
                switch_candidate_streak += 1
            else:
                switch_candidate_id = owner_candidate
                switch_candidate_streak = 1

            need_sw = int(ball_owner_switch_confirm_frames)
            ctid = player_tracks_local.get(current_owner_id, {}).get("team_id")
            ntid = player_tracks_local.get(owner_candidate, {}).get("team_id")
            if (
                ctid is not None
                and ntid is not None
                and int(ctid) != int(ntid)
            ):
                need_sw = max(
                    need_sw,
                    int(render_cfg.ball_owner_switch_confirm_frames_cross_team),
                )

            if switch_candidate_streak >= need_sw:
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                _commit_raw_assign_state()
                return remap_owner_if_missing(owner_candidate, player_tracks_local)

            # Not confirmed yet: keep current owner.
            _commit_raw_assign_state()
            return current_owner_id

        # Update camera movement *before* computing ball ownership, so motion
        # estimation can compensate ball centers by camera shift.
        if cam_estimator is not None:
            # First frame always seeds the estimator regardless of throttle.
            movement = (0.0, 0.0)
            try:
                movement = cam_estimator.update(frame)
            except Exception:
                movement = (0.0, 0.0)
            last_cam_movement = movement
            dx, dy = movement
            cam_cum_dx += float(dx)
            cam_cum_dy += float(dy)
            tracks["camera_movement"][idx0] = movement
            if camera_movement_overlay_enabled:
                frame = cam_estimator.draw_camera_movement_overlay(frame, movement)

        ball_owner_id = compute_ball_owner(player_tracks, ball_tracks)
        tracks["ball_owner"][idx0] = ball_owner_id
        if stats_manager is not None:
            stats_manager.update(
                player_tracks=player_tracks,
                ball_tracks=ball_tracks,
                ball_owner_id=ball_owner_id,
            )

        if use_gpu_overlay and draw_render_overlays_gpu is not None:
            frame = draw_render_overlays_gpu(
                frame,
                player_tracks,
                referee_tracks,
                ball_tracks,
                ball_owner_id,
                team_assign_debug,
                device_str=cfg.render_gpu_device,
            )
        else:
            for player_id, player in player_tracks.items():
                player_color = (0, 255, 0) if player_id == ball_owner_id else player.get(
                    "team_color", (255, 0, 255)
                )
                frame = shared_annotations.draw_player_ellipse(
                    frame,
                    player["bbox"],
                    player_color,
                    player_id,
                )
                if team_assign_debug:
                    roi = player.get("team_roi_bbox")
                    if roi is not None:
                        x1, y1, x2, y2 = map(int, roi)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), player.get("team_color", (128, 128, 128)), 1)
                        label_team = player.get("team_id")
                        label = "UNK" if label_team is None else f"T{label_team}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            player.get("team_color", (128, 128, 128)),
                            1,
                        )

            for _, referee in referee_tracks.items():
                frame = shared_annotations.draw_player_ellipse(
                    frame,
                    referee["bbox"],
                    (0, 165, 255),
                    None,
                )

            for _, ball in ball_tracks.items():
                ball_color = (0, 255, 0) if ball_owner_id != -1 else (0, 255, 255)
                frame = shared_annotations.draw_ball_marker(
                    frame,
                    ball["bbox"],
                    color=ball_color,
                )

        write_output_frame(frame)

        for render_frame_idx, frame in frames_iter:
            while (
                track_ptr + 1 < len(track_original_frame_indices)
                and track_original_frame_indices[track_ptr + 1] <= render_frame_idx
            ):
                track_ptr += 1
            idx = track_ptr

            frame = frame.copy()
            player_tracks = tracks["players"][idx]
            ball_tracks = tracks["ball"][idx]
            referee_tracks = tracks["referees"][idx]
            apply_player_team_colors(frame, player_tracks)

            # Update camera movement *before* computing ball ownership.
            if cam_estimator is not None:
                if render_frame_idx % cam_sample_n == 0:
                    movement = cam_estimator.update(frame)
                    last_cam_movement = movement
                else:
                    movement = last_cam_movement
                tracks["camera_movement"][idx] = movement
                dx, dy = movement
                cam_cum_dx += float(dx)
                cam_cum_dy += float(dy)
                if camera_movement_overlay_enabled:
                    frame = cam_estimator.draw_camera_movement_overlay(frame, movement)

            ball_owner_id = compute_ball_owner(player_tracks, ball_tracks)
            tracks["ball_owner"][idx] = ball_owner_id
            if stats_manager is not None:
                stats_manager.update(
                    player_tracks=player_tracks,
                    ball_tracks=ball_tracks,
                    ball_owner_id=ball_owner_id,
                )

            if use_gpu_overlay and draw_render_overlays_gpu is not None:
                frame = draw_render_overlays_gpu(
                    frame,
                    player_tracks,
                    referee_tracks,
                    ball_tracks,
                    ball_owner_id,
                    team_assign_debug,
                    device_str=cfg.render_gpu_device,
                )
            else:
                for player_id, player in player_tracks.items():
                    player_color = (0, 255, 0) if player_id == ball_owner_id else player.get(
                        "team_color", (255, 0, 255)
                    )
                    frame = shared_annotations.draw_player_ellipse(
                        frame,
                        player["bbox"],
                        player_color,
                        player_id,
                    )
                    if team_assign_debug:
                        roi = player.get("team_roi_bbox")
                        if roi is not None:
                            x1, y1, x2, y2 = map(int, roi)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), player.get("team_color", (128, 128, 128)), 1)
                            label_team = player.get("team_id")
                            label = "UNK" if label_team is None else f"T{label_team}"
                            cv2.putText(
                                frame,
                                label,
                                (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                player.get("team_color", (128, 128, 128)),
                                1,
                            )

                for _, referee in referee_tracks.items():
                    frame = shared_annotations.draw_player_ellipse(
                        frame,
                        referee["bbox"],
                        (0, 165, 255),
                        None,
                    )

                for ball_id, ball in ball_tracks.items():
                    ball_color = (0, 255, 0) if ball_owner_id != -1 else (0, 255, 255)
                    frame = shared_annotations.draw_ball_marker(
                        frame,
                        ball["bbox"],
                        color=ball_color,
                    )

            write_output_frame(frame)

        if use_ffmpeg_writer:
            try:
                if ffmpeg_stdin is not None:
                    ffmpeg_stdin.close()
            except (BrokenPipeError, OSError):
                # Encoder already exited; ignore to avoid crashing shutdown path.
                pass
            try:
                if ffmpeg_proc is not None:
                    ffmpeg_proc.wait()
            except Exception:
                pass
        elif writer is not None:
            writer.release()

        if stats_manager is None:
            return None

        stats_payload = stats_manager.build_payload(
            video_path=video_path,
            output_video_path=output_path,
        )
        stats_path = stats_manager.save(payload=stats_payload, output_video_path=output_path)
        print(f"[ByteTracker] Stats saved to: {stats_path}")
        return {"path": stats_path, "payload": stats_payload}

    def process_video(
        self,
        input_path: str,
        output_path: str,
        *,
        cache: bool = True,
        overwrite_output: bool = True,
        inference: InferenceConfig = InferenceConfig(),
        ball: BallDetectionConfig = BallDetectionConfig(),
        render: RenderOwnershipStatsConfig = RenderOwnershipStatsConfig(),
    ):
        """
        End-to-end pipeline:
        1) Track streaming without storing all frames
        2) Render annotations streaming without storing output frames
        """
        if overwrite_output and os.path.exists(output_path):
            os.remove(output_path)

        input_fps = get_video_fps(input_path)
        input_fps = self._set_tracker_fps(input_fps)
        print(f"[ByteTracker] Input FPS detected: {input_fps:.3f}")

        self.configure_inference(device=inference.device, use_half=inference.use_half)
        t0 = time.perf_counter()
        t_track_start = time.perf_counter()
        tracks, track_original_frame_indices = self.get_tracks_from_video(
            video_path=input_path,
            cache=cache,
            scale=inference.scale,
            step=inference.step,
            batch=inference.batch,
            motion_threshold=inference.motion_threshold,
            motion_burst_frames=inference.motion_burst_frames,
            conf=inference.conf,
            imgsz=inference.imgsz,
            ball_tracking_mode=ball.ball_tracking_mode,
            ball_min_candidate_confidence=ball.ball_min_candidate_confidence,
            ball_aspect_ratio_min=ball.ball_aspect_ratio_min,
            ball_aspect_ratio_max=ball.ball_aspect_ratio_max,
            ball_prefilter_debug=ball.ball_prefilter_debug,
        )
        track_time = time.perf_counter() - t_track_start

        if ball.interpolate_ball_positions:
            tracks["ball"] = self.interpolate_ball_positions(tracks.get("ball", []))

        t_render_start = time.perf_counter()
        stats_result = self.render_video_from_tracks(
            video_path=input_path,
            output_path=output_path,
            tracks=tracks,
            track_original_frame_indices=track_original_frame_indices,
            scale=inference.scale,
            fps=input_fps,
            render_cfg=render,
        )
        render_time = time.perf_counter() - t_render_start
        elapsed = time.perf_counter() - t0

        if isinstance(stats_result, dict):
            tracks["stats"] = stats_result.get("payload", {})
            tracks["stats_path"] = stats_result.get("path")

        if inference.run_quality_checks:
            n_frames = len(tracks["players"])
            ball_seen_frames = sum(1 for frame in tracks["ball"] if len(frame) > 0)
            player_seen_frames = sum(1 for frame in tracks["players"] if len(frame) > 0)
            mean_players = float(np.mean([len(frame) for frame in tracks["players"]])) if n_frames > 0 else 0.0
            mean_referees = float(np.mean([len(frame) for frame in tracks["referees"]])) if n_frames > 0 else 0.0
            quality = {
                "frames_tracked": n_frames,
                "ball_seen_frame_ratio": (ball_seen_frames / n_frames) if n_frames > 0 else 0.0,
                "player_seen_frame_ratio": (player_seen_frames / n_frames) if n_frames > 0 else 0.0,
                "mean_players_per_tracked_frame": mean_players,
                "mean_referees_per_tracked_frame": mean_referees,
            }
            track_share = (track_time / elapsed * 100.0) if elapsed > 0 else 0.0
            render_share = (render_time / elapsed * 100.0) if elapsed > 0 else 0.0

            # Estimate real-time factor using source video duration.
            cap = cv2.VideoCapture(input_path)
            input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            input_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            cap.release()
            video_seconds = (input_frames / input_fps) if input_fps > 0 else 0.0
            realtime_factor = (video_seconds / elapsed) if elapsed > 0 and video_seconds > 0 else 0.0
            print(
                "[ByteTracker] Perf:"
                f" elapsed={elapsed:.2f}s"
                f" track={track_time:.2f}s({track_share:.1f}%)"
                f" render={render_time:.2f}s({render_share:.1f}%)"
                f" step={inference.step}"
                f" batch={inference.batch}"
                f" imgsz={inference.imgsz}"
                f" render_codec_preset={render.codec_preset}"
                f" render_hw_encode={render.use_hw_encode}"
                f" render_hw_encoder={render.hw_encoder}"
                f" rt_factor={realtime_factor:.2f}x"
                f" quality={quality}"
            )

        return tracks