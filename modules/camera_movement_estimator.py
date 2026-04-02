from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMovementConfig:
    minimum_distance_px: float = 5.0
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: int = 3
    block_size: int = 7
    win_size: Tuple[int, int] = (15, 15)
    max_level: int = 2
    # Downscale before GFTT / LK so OpenCV pyramid buffers stay small (avoids OOM on 4K+).
    max_analysis_side: int = 960
    criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03,
    )
    # Feature mask strips (fractions of width). We look at borders to better
    # capture camera motion rather than player motion in the field center.
    left_strip_frac: float = 0.02
    right_strip_frac: float = 0.15
    # Minimum absolute pixel width for strips to avoid zero-width masks.
    min_strip_px: int = 20


class CameraMovementEstimator:
    """
    Streaming camera movement estimation using sparse optical flow (Lucas–Kanade).
    """

    def __init__(self, first_frame_bgr: np.ndarray, config: CameraMovementConfig | None = None):
        self.cfg = config or CameraMovementConfig()

        self.lk_params = dict(
            winSize=self.cfg.win_size,
            maxLevel=self.cfg.max_level,
            criteria=self.cfg.criteria,
        )

        first_gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)
        self._motion_scale_to_full, analysis_gray = CameraMovementEstimator._analysis_gray_and_scale(
            first_gray, self.cfg.max_analysis_side
        )

        self._feature_mask = self._build_feature_mask(analysis_gray)

        self._gftt_params = dict(
            maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
            blockSize=self.cfg.block_size,
            mask=self._feature_mask,
        )

        self.old_gray = analysis_gray
        self.old_features = self._good_features_to_track_safe(self.old_gray)
        # Flow distances are in analysis pixels; compare to threshold in the same space.
        self._min_dist_analysis = float(self.cfg.minimum_distance_px) / max(
            float(self._motion_scale_to_full), 1e-6
        )

    @staticmethod
    def _analysis_gray_and_scale(
        gray: np.ndarray, max_analysis_side: int
    ) -> Tuple[float, np.ndarray]:
        """Scale factor to convert analysis-space motion to full-frame pixels; downscaled gray."""
        h, w = gray.shape[:2]
        max_side = max(h, w)
        if max_analysis_side <= 0 or max_side <= max_analysis_side:
            return 1.0, gray
        s = max_analysis_side / float(max_side)
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return 1.0 / s, small

    def _to_analysis_gray(self, gray: np.ndarray) -> np.ndarray:
        if self._motion_scale_to_full == 1.0:
            return gray
        s = 1.0 / self._motion_scale_to_full
        h, w = gray.shape[:2]
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _good_features_to_track_safe(self, gray: np.ndarray):
        try:
            return cv2.goodFeaturesToTrack(gray, **self._gftt_params)
        except cv2.error:
            return None

    def _build_feature_mask(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        left_w = max(self.cfg.min_strip_px, int(self.cfg.left_strip_frac * w))
        right_w = max(self.cfg.min_strip_px, int(self.cfg.right_strip_frac * w))

        mask[:, :left_w] = 1
        mask[:, max(0, w - right_w) : w] = 1
        return mask

    @staticmethod
    def _measure_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
        dx = float(a_xy[0] - b_xy[0])
        dy = float(a_xy[1] - b_xy[1])
        return float(np.hypot(dx, dy))

    @staticmethod
    def _measure_xy_distance(old_xy: np.ndarray, new_xy: np.ndarray) -> Tuple[float, float]:
        return (float(new_xy[0] - old_xy[0]), float(new_xy[1] - old_xy[1]))

    def update(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """
        Compute camera movement (dx, dy) for the current frame relative to previous.
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small_gray = self._to_analysis_gray(frame_gray)

        if self.old_features is None or len(self.old_features) == 0:
            self.old_features = self._good_features_to_track_safe(small_gray)
            self.old_gray = small_gray
            return (0.0, 0.0)

        try:
            new_features, status, _err = cv2.calcOpticalFlowPyrLK(
                self.old_gray, small_gray, self.old_features, None, **self.lk_params
            )
        except cv2.error:
            self.old_features = self._good_features_to_track_safe(small_gray)
            self.old_gray = small_gray
            return (0.0, 0.0)

        if new_features is None or status is None:
            self.old_features = self._good_features_to_track_safe(small_gray)
            self.old_gray = small_gray
            return (0.0, 0.0)

        max_distance = 0.0
        cam_dx, cam_dy = 0.0, 0.0

        # status is (N,1) with 1 for valid points
        for new, old, st in zip(new_features, self.old_features, status):
            if int(st[0]) != 1:
                continue
            new_xy = new.ravel()
            old_xy = old.ravel()
            dist = self._measure_distance(new_xy, old_xy)
            if dist > max_distance:
                max_distance = dist
                cam_dx, cam_dy = self._measure_xy_distance(old_xy, new_xy)

        if max_distance > self._min_dist_analysis:
            # Re-seed features when motion is meaningful (matches repo behavior).
            self.old_features = self._good_features_to_track_safe(small_gray)
        # else: keep existing features to accumulate stable flow

        self.old_gray = small_gray
        scale = self._motion_scale_to_full
        return (cam_dx * scale, cam_dy * scale)

    @staticmethod
    def draw_camera_movement_overlay(
        frame_bgr: np.ndarray, movement_xy: Tuple[float, float]
    ) -> np.ndarray:
        frame = frame_bgr
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        dx, dy = movement_xy
        cv2.putText(
            frame,
            f"Camera Movement X: {dx:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Camera Movement Y: {dy:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        return frame

