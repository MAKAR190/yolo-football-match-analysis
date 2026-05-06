from typing import Optional, Tuple

import numpy as np


def _ball_bbox_aspect_ratio(bbox) -> float:
    x1, y1, x2, y2 = map(float, bbox[:4])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if h <= 1e-6:
        return 0.0
    return float(w / h)


def ball_candidate_prefilter(
    bbox,
    frame_shape,
    confidence: Optional[float],
    *,
    min_confidence: float,
    aspect_ratio_min: float,
    aspect_ratio_max: float,
) -> Tuple[bool, Optional[str]]:
    """
    Reject obvious non-ball blobs before temporal tracking (feet, glare, etc.).

    Returns (ok, reject_reason). If min_confidence <= 0, confidence is not gated.
    """
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return False, "bad_frame_shape"

    x1, y1, x2, y2 = map(int, bbox[:4])
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    area = float(max(0, x2 - x1) * max(0, y2 - y1))
    area_frac = float(area / float(width * height)) if width > 0 and height > 0 else 0.0

    # Match BallOutputOutlierRejection absolute size bounds (catch tiny/large false dets early).
    if area_frac < 0.00005 or area_frac > 0.008:
        return False, "area_frac_abs"

    ar = _ball_bbox_aspect_ratio([x1, y1, x2, y2])
    if ar > 0 and (ar < aspect_ratio_min or ar > aspect_ratio_max):
        return False, "aspect_ratio"

    if min_confidence > 0.0:
        if confidence is None:
            return False, "missing_conf"
        if float(confidence) < float(min_confidence):
            return False, "low_confidence"

    return True, None


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
