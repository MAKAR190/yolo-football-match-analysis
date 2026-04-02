"""
Foot regions of interest from player bounding boxes (image space).

Two small axis-aligned boxes anchored at the bottom-left and bottom-right corners,
used for ball–foot contact (overlap with ball bbox), matching the idea of
bottom-corner proximity used in PlayerBallAssigner.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

# Width as fraction of player bbox width; height as fraction of bbox height.
_FOOT_W_FRAC = 0.16
_FOOT_H_FRAC = 0.12
_FOOT_W_MIN, _FOOT_W_MAX = 8.0, 48.0
_FOOT_H_MIN, _FOOT_H_MAX = 6.0, 36.0

FootBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def _clamp_foot_size(w: float, h: float) -> Tuple[float, float]:
    fw = max(_FOOT_W_MIN, min(_FOOT_W_MAX, w * _FOOT_W_FRAC))
    fh = max(_FOOT_H_MIN, min(_FOOT_H_MAX, h * _FOOT_H_FRAC))
    return fw, fh


def foot_rois_from_player_bbox(player_bbox: Sequence[float]) -> List[FootBox]:
    """
    Return two foot ROIs (left bottom corner, right bottom corner) in xyxy order.
    """
    x1, y1, x2, y2 = map(float, player_bbox[:4])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    fw, fh = _clamp_foot_size(w, h)

    # Left foot: anchored at bottom-left (x1, y2).
    l_x1, l_x2 = x1, x1 + fw
    l_y2, l_y1 = y2, y2 - fh

    # Right foot: anchored at bottom-right (x2, y2).
    r_x1, r_x2 = x2 - fw, x2
    r_y2, r_y1 = y2, y2 - fh

    return [(l_x1, l_y1, l_x2, l_y2), (r_x1, r_y1, r_x2, r_y2)]


def axis_aligned_boxes_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if two xyxy boxes have positive-area intersection."""
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return False
    return not (ax1 >= bx2 or ax2 <= bx1 or ay1 >= by2 or ay2 <= by1)


def ball_center_in_vertical_control_zone(
    player_bbox: Sequence[float],
    ball_center_xy: Union[Sequence[float], Tuple[float, float]],
    frac_from_top: float = 0.42,
) -> bool:
    """
    True if the ball center lies in the lower portion of the player bbox (image space).

    Requires ball cy >= y1 + frac_from_top * (y2 - y1), so balls clearly above the
    torso (typical pass arc) are rejected. frac_from_top=0.5 means the bottom half.
    """
    x1, y1, x2, y2 = map(float, player_bbox[:4])
    h = max(0.0, y2 - y1)
    if h <= 0:
        return False
    cy = float(ball_center_xy[1])
    cutoff = y1 + float(frac_from_top) * h
    return cy >= cutoff


def ball_overlaps_any_foot(ball_bbox: Sequence[float], player_bbox: Sequence[float]) -> bool:
    """True if ball bbox intersects either foot ROI."""
    for foot in foot_rois_from_player_bbox(player_bbox):
        if axis_aligned_boxes_overlap(ball_bbox, foot):
            return True
    return False


def foot_rois_int_clipped(
    player_bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> List[Tuple[int, int, int, int]]:
    """Integer xyxy foot boxes clipped to frame bounds (for drawing)."""
    out: List[Tuple[int, int, int, int]] = []
    fw, fh = frame_width, frame_height
    for x1, y1, x2, y2 in foot_rois_from_player_bbox(player_bbox):
        xi1 = int(max(0, min(fw - 1, round(x1))))
        yi1 = int(max(0, min(fh - 1, round(y1))))
        xi2 = int(max(0, min(fw, round(x2))))
        yi2 = int(max(0, min(fh, round(y2))))
        if xi2 > xi1 and yi2 > yi1:
            out.append((xi1, yi1, xi2, yi2))
    return out
