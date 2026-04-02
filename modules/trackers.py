import supervision as sv
import pickle
import os
import cv2
import hashlib
import time
import numpy as np
import torch
import shutil
import subprocess
import math
import importlib
from typing import Optional
from dataclasses import dataclass
from .annotations import Annotations
from .team_assigner import TeamAssigner
from .player_ball_assigner import PlayerBallAssigner
from .camera_movement_estimator import CameraMovementEstimator
from .speed_distance_estimator import SpeedDistanceEstimator, SpeedDistanceConfig
from .view_transformer import ViewTransformer
from .stats import StatsManager
from ultralytics import YOLO
from helpers import (
    iter_video_frames,
    iter_video_frames_adaptive,
    get_video_fps,
)

CACHE_FOLDER = "cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)
shared_annotations = Annotations()


@dataclass(frozen=True)
class RenderOwnershipStatsConfig:
    # Render output format
    codec: str = "XVID"
    codec_preset: str = "balanced"
    use_hw_encode: bool = False
    hw_encoder: str = "auto"
    team_assign_debug: bool = False

    # Ball ownership / assignment
    ball_owner_hold_frames: int = 8
    ball_owner_lock_enabled: bool = True
    ball_owner_switch_confirm_frames: int = 1
    ball_owner_switch_margin_px: float = 18.0
    ball_owner_switch_margin_ratio: float = 0.75
    ball_owner_release_distance_px: float = 110.0
    ball_assign_max_player_ball_distance_px: float = 75.0
    ball_assign_ambiguity_margin_px: float = 10.0

    # Motion-change touch (ownership only on ball motion change near player)
    motion_touch_enabled: bool = True
    # Smooth ball centers before computing motion (EMA alpha in [0,1]).
    ball_center_smoothing_alpha: float = 0.55
    # Compensate ball motion by camera movement (if enabled).
    camera_motion_compensation_enabled: bool = True
    # After ball is re-detected (following missing frames), ignore motion-touch
    # for a short cooldown window to avoid false "speed drop" touches.
    motion_touch_redetect_cooldown_frames: int = 3
    # Reset EMA/motion history on re-detection so velocities aren't computed
    # across a detection gap.
    motion_touch_reset_history_on_redetect: bool = True
    # Require a meaningful previous speed before we consider a "touch"
    motion_touch_min_prev_speed_px_per_frame: float = 12.0
    # Touch if speed drops sharply: speed_curr <= ratio * speed_prev
    motion_touch_speed_drop_ratio: float = 0.75
    # Or touch if direction changes sharply (degrees) while speed is non-trivial
    motion_touch_angle_change_deg: float = 70.0
    motion_touch_use_angle_change: bool = False
    # Candidate must be close and unambiguous to count as a "touch"
    motion_touch_max_candidate_distance_px: float = 55.0
    motion_touch_min_second_best_margin_px: float = 12.0
    # Require persistence of the motion-touch signal
    motion_touch_confirm_frames: int = 3
    # Require the same nearest candidate to persist
    motion_touch_candidate_confirm_frames: int = 3


    # Fast-ball "no owner"
    ball_in_transit_velocity_threshold_px_per_frame: float = 20.0
    ball_in_transit_confirm_frames: int = 1
    ball_in_transit_freeze_owner: bool = False
    ball_in_transit_grace_frames_after_reappear: int = 2

    # Optional overlays / computed signals
    camera_movement_enabled: bool = False
    speed_distance_enabled: bool = False
    # Stats output
    stats_enabled: bool = True

class BallOutputOutlierRejection:
    """
    Optional temporal guard for the *rendered ball bbox*.

    When enabled, if the ball center jumps too far compared to the previously
    accepted ball bbox, we reuse the last good bbox instead of latching onto
    an outlier.
    """

    def __init__(
        self,
        max_jump_dist_norm: float = 0.1,
        enabled: bool = False,
        max_missed_frames: int = 2,
        min_area_ratio: float = 0.5,
        max_area_ratio: float = 1.8,
        min_area_frac_abs: float = 0.00005,
        max_area_frac_abs: float = 0.008,
    ):
        self.max_jump_dist_norm = float(max_jump_dist_norm)
        self.enabled = bool(enabled)
        self.prev_center = None  # (cx, cy) in pixels
        self.prev_prev_center = None  # (cx, cy) in pixels (velocity estimate)
        self.prev_bbox = None  # [x1,y1,x2,y2] in pixels
        self.prev_area_frac = None  # bbox area / frame area
        self.max_missed_frames = int(max_missed_frames)
        self.missed_frames = 0
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_area_frac_abs = float(min_area_frac_abs)
        self.max_area_frac_abs = float(max_area_frac_abs)

    def reset(self):
        self.prev_center = None
        self.prev_prev_center = None
        self.prev_bbox = None
        self.prev_area_frac = None
        self.missed_frames = 0

    def filter_bbox(self, bbox, frame_shape):
        if not self.enabled:
            return bbox

        height, width = frame_shape[:2]
        diag = float(np.sqrt(width * width + height * height))
        if diag <= 0:
            return bbox

        cx_out = float((bbox[0] + bbox[2]) / 2.0)
        cy_out = float((bbox[1] + bbox[3]) / 2.0)
        area = float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))
        area_frac = float(area / float(width * height)) if width > 0 and height > 0 else 0.0

        # Absolute size gate: reject ball candidates that are implausibly small/large
        # even before temporal checks (helps suppress first-frame false detections).
        if (
            area_frac < self.min_area_frac_abs
            or area_frac > self.max_area_frac_abs
        ):
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.reset()
            return None

        bbox_to_store = bbox
        if self.prev_center is not None and self.prev_bbox is not None:
            prev_cx, prev_cy = self.prev_center
            # Constant-velocity expected center.
            if self.prev_prev_center is not None:
                prev_prev_cx, prev_prev_cy = self.prev_prev_center
                expected_cx = (2.0 * prev_cx) - prev_prev_cx
                expected_cy = (2.0 * prev_cy) - prev_prev_cy
            else:
                expected_cx, expected_cy = prev_cx, prev_cy

            dist_norm_out = float(
                np.hypot(cx_out - expected_cx, cy_out - expected_cy) / diag
            )
            area_outlier = False
            if self.prev_area_frac is not None and self.prev_area_frac > 0:
                area_ratio = area_frac / self.prev_area_frac
                area_outlier = (
                    area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio
                )

            if dist_norm_out > self.max_jump_dist_norm or area_outlier:
                # Candidate is inconsistent with the last accepted state.
                # Prefer returning "no ball" over freezing on a wrong bbox.
                self.missed_frames += 1
                if self.missed_frames > self.max_missed_frames:
                    self.reset()
                return None

            # Candidate accepted: clear miss counter and update motion state.
            self.missed_frames = 0
            self.prev_prev_center = self.prev_center
            self.prev_center = (cx_out, cy_out)
            self.prev_bbox = bbox
            self.prev_area_frac = area_frac
        else:
            self.prev_center = (cx_out, cy_out)
            self.prev_prev_center = None
            self.prev_bbox = bbox
            self.missed_frames = 0
            self.prev_area_frac = area_frac

        return bbox_to_store


class FastReIDEmbedder:
    """
    Minimal FastReID wrapper with lazy initialization.
    """

    def __init__(self):
        self._predictor = None
        self._device = "cpu"
        self._is_ready = False
        self._warned = False

    @property
    def is_ready(self) -> bool:
        return bool(self._is_ready and self._predictor is not None)

    def _warn_once(self, message: str):
        if not self._warned:
            print(message)
            self._warned = True

    def setup(
        self,
        config_path: str,
        weights_path: str,
        device: str = "cpu",
    ) -> bool:
        if self.is_ready:
            return True
        try:
            fastreid_config = importlib.import_module("fastreid.config")
            fastreid_engine = importlib.import_module("fastreid.engine")
            get_cfg = getattr(fastreid_config, "get_cfg")
            DefaultPredictor = getattr(fastreid_engine, "DefaultPredictor")

            cfg = get_cfg()
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.WEIGHTS = str(weights_path)
            cfg.MODEL.DEVICE = str(device)
            self._predictor = DefaultPredictor(cfg)
            self._device = str(device)
            self._is_ready = True
            return True
        except Exception as exc:
            self._predictor = None
            self._is_ready = False
            self._warn_once(
                "[FastReID] Disabled: failed to initialize FastReID predictor "
                f"(config={config_path}, weights={weights_path}): {exc}"
            )
            return False

    def extract(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        if not self.is_ready:
            return None
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # Use upper torso-heavy region to reduce leg/grass/background noise.
        ch, cw = crop.shape[:2]
        if ch >= 24 and cw >= 12:
            y_top = int(0.10 * ch)
            y_bottom = int(0.72 * ch)
            x_left = int(0.15 * cw)
            x_right = int(0.85 * cw)
            if y_bottom > y_top and x_right > x_left:
                crop = crop[y_top:y_bottom, x_left:x_right]

        try:
            with torch.no_grad():
                features = self._predictor(crop)
                if isinstance(features, (list, tuple)) and len(features) > 0:
                    features = features[0]
                if hasattr(features, "detach"):
                    vec = features.detach().cpu().numpy().reshape(-1).astype(np.float32)
                else:
                    vec = np.asarray(features, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vec))
                if norm <= 1e-8:
                    return None
                return (vec / norm).astype(np.float32)
        except Exception as exc:
            self._warn_once(f"[FastReID] Disabled after inference error: {exc}")
            self._predictor = None
            self._is_ready = False
            return None

class ByteTracker:
    # Bump this when detection/postprocessing logic changes so old caches aren't reused.
    PIPELINE_VERSION = 10

    def __init__(self, model: str):
        self.model_path = model
        self.model = YOLO(model)
        # Stricter association settings reduce ID swaps when players cross.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.35,
            lost_track_buffer=20,
            minimum_matching_threshold=0.92,
            frame_rate=30,
            minimum_consecutive_frames=2,
        )
        self.device = "cpu"
        self.use_half = False
        self._fastreid = FastReIDEmbedder()

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
        reid_enabled: bool = False,
        reid_model_config: str = "",
        reid_model_weights: str = "",
        reid_device: str = "cpu",
        reid_cosine_thresh: float = 0.70,
        reid_max_age_frames: int = 30,
    ) -> str:
        raw = (
            f"{self.model_path}|v{self.PIPELINE_VERSION}|{video_path}|"
            f"scale={scale}|base_step={step}|batch={batch}|"
            f"motion_threshold={motion_threshold}|motion_burst_frames={motion_burst_frames}|"
            f"ball_tracking_mode={ball_tracking_mode}|"
            f"reid_enabled={reid_enabled}|reid_cfg={reid_model_config}|"
            f"reid_w={reid_model_weights}|reid_dev={reid_device}|"
            f"reid_thr={reid_cosine_thresh}|reid_age={reid_max_age_frames}"
        )
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return os.path.join(CACHE_FOLDER, f"bytetracker_cache_{h}.pkl")

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
        reid_enabled: bool = False,
        reid_model_config: str = "",
        reid_model_weights: str = "",
        reid_device: str = "cpu",
        reid_cosine_thresh: float = 0.70,
        reid_max_age_frames: int = 30,
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
            reid_enabled=reid_enabled,
            reid_model_config=reid_model_config,
            reid_model_weights=reid_model_weights,
            reid_device=reid_device,
            reid_cosine_thresh=reid_cosine_thresh,
            reid_max_age_frames=reid_max_age_frames,
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

        use_reid = bool(reid_enabled)
        if use_reid:
            if not reid_model_config or not reid_model_weights:
                print(
                    "[FastReID] Disabled: config/weights path missing. "
                    "Set reid_model_config and reid_model_weights."
                )
                use_reid = False
            elif not os.path.exists(reid_model_config) or not os.path.exists(reid_model_weights):
                print(
                    "[FastReID] Disabled: config/weights file not found. "
                    f"config={reid_model_config}, weights={reid_model_weights}"
                )
                use_reid = False
            else:
                use_reid = self._fastreid.setup(
                    config_path=reid_model_config,
                    weights_path=reid_model_weights,
                    device=reid_device,
                )

        # ReID fallback memory.
        tracker_to_stable_player_id = {}
        stable_id_gallery = {}  # stable_id -> {"embedding","bbox","last_seen"}
        used_stable_ids = set()
        reid_switch_margin = 0.20
        reid_mismatch_thresh = max(0.05, float(reid_cosine_thresh) - 0.20)
        reid_force_switch_thresh = max(float(reid_cosine_thresh) + 0.12, 0.88)
        reid_min_iou_for_switch = 0.05
        reid_switch_cooldown = 20
        tracker_switch_cooldown_until = {}

        def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
            if a is None or b is None:
                return -1.0
            return float(np.dot(a, b))

        def bbox_iou_quick(a, b) -> float:
            if a is None or b is None:
                return 0.0
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

        def select_stable_player_id(
            tracker_id: int,
            bbox,
            embedding: Optional[np.ndarray],
            frame_idx: int,
        ) -> int:
            if not use_reid:
                return tracker_id

            if tracker_id in tracker_to_stable_player_id:
                stable_id = int(tracker_to_stable_player_id[tracker_id])
                curr_state = stable_id_gallery.get(stable_id, {})
                curr_emb = curr_state.get("embedding")
                curr_bbox = curr_state.get("bbox")
                curr_sim = cosine_similarity(embedding, curr_emb)

                # If an existing mapping becomes inconsistent, allow one-step remap.
                cooldown_until = int(tracker_switch_cooldown_until.get(int(tracker_id), -1))
                can_switch_now = int(frame_idx) >= cooldown_until
                if (
                    can_switch_now
                    and embedding is not None
                    and curr_emb is not None
                    and curr_sim < reid_mismatch_thresh
                ):
                    best_id = stable_id
                    best_sim = curr_sim
                    second_best = -1.0
                    for candidate_id, state in stable_id_gallery.items():
                        age = int(frame_idx) - int(state.get("last_seen", frame_idx))
                        if age < 0 or age > int(reid_max_age_frames):
                            continue
                        if candidate_id in used_stable_ids and int(candidate_id) != stable_id:
                            continue
                        sim = cosine_similarity(embedding, state.get("embedding"))
                        iou = bbox_iou_quick(bbox, state.get("bbox"))
                        if int(candidate_id) != stable_id and iou < reid_min_iou_for_switch:
                            continue
                        if sim > best_sim:
                            second_best = best_sim
                            best_sim = sim
                            best_id = int(candidate_id)
                        elif sim > second_best:
                            second_best = sim
                    if (
                        best_id != stable_id
                        and best_sim >= float(reid_force_switch_thresh)
                        and (best_sim - second_best) >= float(reid_switch_margin)
                    ):
                        stable_id = int(best_id)
                        tracker_to_stable_player_id[int(tracker_id)] = int(stable_id)
                        tracker_switch_cooldown_until[int(tracker_id)] = int(frame_idx) + int(
                            reid_switch_cooldown
                        )

                # EMA update keeps gallery embedding stable over time.
                prev_emb = stable_id_gallery.get(stable_id, {}).get("embedding")
                if embedding is not None and prev_emb is not None:
                    alpha = 0.35
                    merged = (alpha * embedding) + ((1.0 - alpha) * prev_emb)
                    denom = float(np.linalg.norm(merged))
                    embedding = (merged / denom).astype(np.float32) if denom > 1e-8 else embedding

                stable_id_gallery[stable_id] = {
                    "embedding": embedding if embedding is not None else prev_emb,
                    "bbox": bbox,
                    "last_seen": int(frame_idx),
                }
                used_stable_ids.add(stable_id)
                return stable_id

            best_id = None
            best_sim = -1.0
            for candidate_id, state in stable_id_gallery.items():
                age = int(frame_idx) - int(state.get("last_seen", frame_idx))
                if age < 0 or age > int(reid_max_age_frames):
                    continue
                if candidate_id in used_stable_ids:
                    continue
                sim = cosine_similarity(embedding, state.get("embedding"))
                iou = bbox_iou_quick(bbox, state.get("bbox"))
                if iou < reid_min_iou_for_switch:
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best_id = int(candidate_id)

            if best_id is not None and best_sim >= float(reid_force_switch_thresh):
                stable_id = int(best_id)
            else:
                stable_id = int(tracker_id)
                while stable_id in used_stable_ids:
                    stable_id += 100000

            tracker_to_stable_player_id[int(tracker_id)] = int(stable_id)
            stable_id_gallery[int(stable_id)] = {
                "embedding": embedding,
                "bbox": bbox,
                "last_seen": int(frame_idx),
            }
            used_stable_ids.add(int(stable_id))
            return int(stable_id)

        def process_prediction(current_frame_idx: int, frame_for_crop, prediction):
            # Ensure per-frame containers exist.
            for key in tracks:
                tracks[key].append({})

            used_stable_ids.clear()

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
                        cx = float((x1 + x2) / 2.0)
                        cy = float((y1 + y2) / 2.0)
                        ball_candidates.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "score": float(score),
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
                track_key = class_to_track_key.get(class_name)

                if class_name == "player":
                    x1, y1, x2, y2 = map(int, bbox)
                    height, width = frame_for_crop.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height - 1))
                    player_bbox = [x1, y1, x2, y2]

                    embedding = None
                    if use_reid:
                        embedding = self._fastreid.extract(frame_for_crop, player_bbox)
                    stable_player_id = select_stable_player_id(
                        tracker_id=tracker_id,
                        bbox=player_bbox,
                        embedding=embedding,
                        frame_idx=current_frame_idx,
                    )
                    player_entry = {"bbox": player_bbox}
                    if embedding is not None:
                        player_entry["reid_embedding"] = embedding.tolist()
                    tracks["players"][current_frame_idx][stable_player_id] = player_entry
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
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)

                    byte_ball_candidates.append(
                        {
                            "tracker_id": tracker_id,
                            "bbox": [x1, y1, x2, y2],
                            "score": float(score),
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
                        tracks["ball"][current_frame_idx][1] = {
                            "bbox": best_bbox
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
                        tracks["ball"][current_frame_idx][best["tracker_id"]] = {
                            "bbox": best_bbox
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
                        tracks["ball"][current_frame_idx][key] = {"bbox": best_bbox}

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
        speed_distance_enabled = cfg.speed_distance_enabled

        stats_enabled = cfg.stats_enabled
        # No foot-based debug in this pipeline.

        stats_manager = None
        if stats_enabled:
            stats_manager = StatsManager(
                fps=float(fps),
            )

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
            # Avoid XVID+MP4 mismatch warning/fallback in OpenCV.
            if output_ext == ".mp4" and codec_name.upper() == "XVID":
                return "mp4v"
            return codec_name

        if use_hw_encode and shutil.which("ffmpeg"):
            encoders_text = ""
            try:
                encoders_probe = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                encoders_text = (encoders_probe.stdout or "") + (encoders_probe.stderr or "")
            except Exception:
                encoders_text = ""

            available_encoders = []
            for candidate in ["h264_nvenc", "h264_qsv", "h264_amf", "libx264"]:
                if candidate in encoders_text:
                    available_encoders.append(candidate)

            selected_encoder = None
            if hw_encoder != "auto":
                selected_encoder = hw_encoder if hw_encoder in available_encoders else None
            elif available_encoders:
                selected_encoder = available_encoders[0]

            if selected_encoder is not None:
                ffmpeg_cmd = [
                    "ffmpeg",
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
                    ffmpeg_cmd += ["-preset", "p1", "-tune", "ll", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
                elif selected_encoder == "h264_qsv":
                    ffmpeg_cmd += ["-preset", "veryfast", "-global_quality", "25"]
                elif selected_encoder == "h264_amf":
                    ffmpeg_cmd += ["-quality", "speed"]
                else:
                    ffmpeg_cmd += ["-preset", "veryfast", "-crf", "23"]

                ffmpeg_cmd += [output_path]
                try:
                    ffmpeg_proc = subprocess.Popen(
                        ffmpeg_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
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

        def write_output_frame(frame_to_write):
            nonlocal use_ffmpeg_writer, ffmpeg_proc, ffmpeg_stdin, writer
            if use_ffmpeg_writer:
                try:
                    ffmpeg_stdin.write(frame_to_write.tobytes())
                    return
                except (BrokenPipeError, OSError):
                    # HW encoder failed mid-run (unsupported encoder/device/session).
                    # Fall back to OpenCV writer to avoid crashing the whole analysis.
                    try:
                        if ffmpeg_stdin is not None:
                            ffmpeg_stdin.close()
                    except Exception:
                        pass
                    try:
                        if ffmpeg_proc is not None:
                            ffmpeg_proc.wait(timeout=1)
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
                    ffmpeg_err = ""
                    try:
                        if ffmpeg_proc is not None and ffmpeg_proc.stderr is not None:
                            ffmpeg_err = ffmpeg_proc.stderr.read().decode(
                                "utf-8", errors="ignore"
                            )
                    except Exception:
                        ffmpeg_err = ""
                    print(
                        "[ByteTracker] Warning: ffmpeg hw encode failed, "
                        "falling back to OpenCV writer."
                    )
                    if ffmpeg_err:
                        err_tail = "\n".join(ffmpeg_err.strip().splitlines()[-4:])
                        if err_tail:
                            print(f"[ByteTracker] ffmpeg error tail:\n{err_tail}")

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
        )
        cam_estimator = CameraMovementEstimator(first_frame) if camera_movement_enabled else None
        speed_distance_drawer = SpeedDistanceEstimator(SpeedDistanceConfig(frame_window=1, fps=float(fps)))

        if camera_movement_enabled and "camera_movement" not in tracks:
            tracks["camera_movement"] = [(0.0, 0.0)] * len(tracks["players"])

        # Store which player is currently closest to the ball for each tracked frame.
        if "ball_owner" not in tracks:
            tracks["ball_owner"] = [-1] * len(tracks["players"])

        if len(player_tracks) > 0:
            team_assigner.assign_team_color(frame, player_tracks)
        prev_frame_player_states = []

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
            if team_assigner.kmeans is None and len(current_player_tracks) > 0:
                team_assigner.assign_team_color(current_frame, current_player_tracks)
            current_states = []
            for player_id, player in current_player_tracks.items():
                team_id = team_assigner.get_player_team(
                    current_frame,
                    player["bbox"],
                    player_id,
                )

                # Visual continuity guard: when IDs switch at intersections,
                # inherit previous-frame team if bbox overlap is strong.
                best_iou = 0.0
                inherited_team_id = None
                for prev_state in prev_frame_player_states:
                    iou = bbox_iou(player["bbox"], prev_state["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        inherited_team_id = prev_state["team_id"]
                if best_iou >= 0.5 and inherited_team_id is not None:
                    team_id = inherited_team_id

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
                current_states.append({"bbox": player["bbox"], "team_id": team_id})
            prev_frame_player_states = current_states

        apply_player_team_colors(frame, player_tracks)

        # Camera motion compensation state (cumulative shift in pixels).
        cam_cum_dx = 0.0
        cam_cum_dy = 0.0

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
                return None, None

            best_bbox = None
            best_center = None

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
                return best_bbox, best_center

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
            return best_bbox, best_center

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

            prev_ball_center = last_ball_center
            prev_prev_ball_center = last_ball_center_prev
            ball_bbox, ball_center = select_ball_bbox(ball_tracks_local, last_ball_center)
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
            else:
                owner_candidate = -1
                candidate_dist = float("inf")

            # Strict ownership gate: ball center must be inside the candidate player's bbox.
            # This is more conservative than "nearby" distance gating.
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
                    if not inside:
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
                        if ball_in_transit_freeze_owner:
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
                    return -1
                ball_missing_frames += 1
                if ball_missing_frames <= ball_owner_hold_frames:
                    return remap_owner_if_missing(last_ball_owner_id, player_tracks_local)
                # Reset switch candidate when we have no evidence.
                switch_candidate_id = -1
                switch_candidate_streak = 0
                return -1

            # We have an assignment candidate this frame.
            ball_missing_frames = 0

            if not ball_owner_lock_enabled or last_ball_owner_id == -1:
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                return remap_owner_if_missing(owner_candidate, player_tracks_local)

            current_owner_id = remap_owner_if_missing(last_ball_owner_id, player_tracks_local)
            if current_owner_id == -1:
                # Owner ID vanished and couldn't be remapped: accept the new candidate.
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                return remap_owner_if_missing(owner_candidate, player_tracks_local)

            # If candidate equals current owner, keep it and clear any switch attempt.
            if owner_candidate == current_owner_id:
                last_ball_owner_id = current_owner_id
                last_ball_owner_bbox = player_tracks_local.get(current_owner_id, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                return current_owner_id

            # If motion-touch is detected, assign ownership immediately to the
            # closest candidate for this frame.
            if motion_touch:
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
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
                return current_owner_id

            # Candidate looks plausible; require a short streak to confirm.
            if switch_candidate_id == owner_candidate:
                switch_candidate_streak += 1
            else:
                switch_candidate_id = owner_candidate
                switch_candidate_streak = 1

            if switch_candidate_streak >= int(ball_owner_switch_confirm_frames):
                last_ball_owner_id = owner_candidate
                last_ball_owner_bbox = player_tracks_local.get(owner_candidate, {}).get("bbox")
                switch_candidate_id = -1
                switch_candidate_streak = 0
                return remap_owner_if_missing(owner_candidate, player_tracks_local)

            # Not confirmed yet: keep current owner.
            return current_owner_id

        # Update camera movement *before* computing ball ownership, so motion
        # estimation can compensate ball centers by camera shift.
        if cam_estimator is not None:
            movement = (0.0, 0.0)
            try:
                movement = cam_estimator.update(frame)
            except Exception:
                movement = (0.0, 0.0)
            dx, dy = movement
            cam_cum_dx += float(dx)
            cam_cum_dy += float(dy)
            tracks["camera_movement"][idx0] = movement
            frame = cam_estimator.draw_camera_movement_overlay(frame, movement)

        ball_owner_id = compute_ball_owner(player_tracks, ball_tracks)
        tracks["ball_owner"][idx0] = ball_owner_id
        if stats_manager is not None:
            stats_manager.update(
                player_tracks=player_tracks,
                ball_tracks=ball_tracks,
                ball_owner_id=ball_owner_id,
            )

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

        if speed_distance_enabled:
            frame = speed_distance_drawer.draw_speed_and_distance(frame, player_tracks)

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
                movement = cam_estimator.update(frame)
                tracks["camera_movement"][idx] = movement
                dx, dy = movement
                cam_cum_dx += float(dx)
                cam_cum_dy += float(dy)
                frame = cam_estimator.draw_camera_movement_overlay(frame, movement)

            ball_owner_id = compute_ball_owner(player_tracks, ball_tracks)
            tracks["ball_owner"][idx] = ball_owner_id
            if stats_manager is not None:
                stats_manager.update(
                    player_tracks=player_tracks,
                    ball_tracks=ball_tracks,
                    ball_owner_id=ball_owner_id,
                )

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

            if speed_distance_enabled:
                frame = speed_distance_drawer.draw_speed_and_distance(frame, player_tracks)

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
        cache: bool = True,
        scale: float = 1.0,
        step: int = 1,
        batch: int = 30,
        overwrite_output: bool = True,
        conf: float = 0.1,
        imgsz: int = 512,
        device: str = "auto",
        use_half: bool = True,
        run_quality_checks: bool = False,
        ball_tracking_mode: str = "raw_candidates",
        render_cfg: RenderOwnershipStatsConfig = RenderOwnershipStatsConfig(),
        interpolate_ball_positions: bool = False,
        perspective_transform_enabled: bool = False,
        speed_distance_frame_window: int = 5,
        reid_enabled: bool = False,
        reid_model_config: str = "",
        reid_model_weights: str = "",
        reid_device: str = "cpu",
        reid_cosine_thresh: float = 0.70,
        reid_max_age_frames: int = 30,
    ):          
        """
        End-to-end pipeline:
        1) Track streaming without storing all frames
        2) Render annotations streaming without storing output frames
        """
        if overwrite_output and os.path.exists(output_path):
            os.remove(output_path)

        self.configure_inference(device=device, use_half=use_half)
        t0 = time.perf_counter()
        t_track_start = time.perf_counter()
        tracks, track_original_frame_indices = self.get_tracks_from_video(
            video_path=input_path,
            cache=cache,
            scale=scale,
            step=step,
            batch=batch,
            conf=conf,
            imgsz=imgsz,
            ball_tracking_mode=ball_tracking_mode,
            reid_enabled=reid_enabled,
            reid_model_config=reid_model_config,
            reid_model_weights=reid_model_weights,
            reid_device=reid_device,
            reid_cosine_thresh=reid_cosine_thresh,
            reid_max_age_frames=reid_max_age_frames,
        )
        track_time = time.perf_counter() - t_track_start

        if interpolate_ball_positions:
            tracks["ball"] = self.interpolate_ball_positions(tracks.get("ball", []))

        if perspective_transform_enabled:
            view_transformer = ViewTransformer(video_path=input_path)
            view_transformer.add_transformed_position_to_tracks(
                tracks=tracks,
                object_keys=("players", "ball"),
                video_path=input_path,
                track_original_frame_indices=track_original_frame_indices,
                dynamic_homography_enabled=True,
            )

        if render_cfg.speed_distance_enabled:
            # Speeds are meaningful only if we have transformed coordinates.
            # If perspective transform is disabled, we still attempt to compute using
            # whatever `position_transformed` exists (likely none).
            fps_val = float(get_video_fps(input_path))
            sde = SpeedDistanceEstimator(
                SpeedDistanceConfig(frame_window=int(speed_distance_frame_window), fps=fps_val)
            )
            sde.add_speed_and_distance_to_tracks(tracks)

        t_render_start = time.perf_counter()
        stats_result = self.render_video_from_tracks(
            video_path=input_path,
            output_path=output_path,
            tracks=tracks,
            track_original_frame_indices=track_original_frame_indices,
            scale=scale,
            render_cfg=render_cfg,
        )
        render_time = time.perf_counter() - t_render_start
        elapsed = time.perf_counter() - t0

        if isinstance(stats_result, dict):
            tracks["stats"] = stats_result.get("payload", {})
            tracks["stats_path"] = stats_result.get("path")

        if run_quality_checks:
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
                f" step={step}"
                f" batch={batch}"
                f" imgsz={imgsz}"
                f" render_codec_preset={render_cfg.codec_preset}"
                f" render_hw_encode={render_cfg.use_hw_encode}"
                f" render_hw_encoder={render_cfg.hw_encoder}"
                f" rt_factor={realtime_factor:.2f}x"
                f" quality={quality}"
            )

        return tracks