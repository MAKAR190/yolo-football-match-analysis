from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from helpers import annotation_center, calculate_distance
from helpers.foot_roi import axis_aligned_boxes_overlap, ball_overlaps_any_foot


class TouchesAnalysis:
    """
    Counts touches when the ball interacts with a player (see ``_player_contact``)
    and image-space speed changes between consecutive frames (|v_t| vs |v_{t-1}|).

    Contact is not only full bbox overlap: foot ROIs, foot-point proximity (ball
    often sits just outside the player box in detections), and the frame's
    ``ball_owner_id`` from the tracker are included so missed in-box detections
    still count when paired with a speed change.

    Per-player cooldown suppresses duplicate counts during sustained control.
    """

    def __init__(
        self,
        touch_cooldown_frames: int = 12,
        vertical_control_enabled: bool = True,
        vertical_control_frac_from_top: float = 0.42,
        in_transit_velocity_threshold_px_per_frame: float = 28.0,
        speed_change_abs_min_px_per_frame: float = 3.5,
        speed_change_rel_min: float = 0.11,
        speed_change_min_prev_speed_px_per_frame: float = 8.0,
        speed_change_drop_ratio: float = 0.84,
        speed_change_spike_ratio: float = 1.22,
        use_ball_bottom_for_proximity: bool = True,
        touch_max_player_ball_distance_px: float = 78.0,
        touch_max_distance_height_scale: float = 0.38,
        touch_max_distance_cap_px: Optional[float] = 110.0,
    ):
        self.touch_cooldown_frames = max(1, int(touch_cooldown_frames))
        # Kept for backward compatibility with StatsManager / older stats JSON.
        self.vertical_control_enabled = bool(vertical_control_enabled)
        self.vertical_control_frac_from_top = float(vertical_control_frac_from_top)
        self.in_transit_velocity_threshold_px_per_frame = float(
            in_transit_velocity_threshold_px_per_frame
        )
        self.speed_change_abs_min_px_per_frame = float(speed_change_abs_min_px_per_frame)
        self.speed_change_rel_min = float(speed_change_rel_min)
        self.speed_change_min_prev_speed_px_per_frame = float(
            speed_change_min_prev_speed_px_per_frame
        )
        self.speed_change_drop_ratio = float(speed_change_drop_ratio)
        self.speed_change_spike_ratio = float(speed_change_spike_ratio)
        self.use_ball_bottom_for_proximity = bool(use_ball_bottom_for_proximity)
        self.touch_max_player_ball_distance_px = float(touch_max_player_ball_distance_px)
        self.touch_max_distance_height_scale = float(touch_max_distance_height_scale)
        self.touch_max_distance_cap_px = touch_max_distance_cap_px
        self._frame = 0
        self._last_ball_center: Optional[Tuple[float, float]] = None
        self._ball_center_prev_prev: Optional[Tuple[float, float]] = None
        self._last_touch_frame: Dict[int, int] = {}
        self.player_touches: Dict[int, int] = {}
        self.team_touches: Dict[int, int] = {}

    @staticmethod
    def _first_ball_bbox(ball_tracks: dict) -> Optional[list]:
        if not ball_tracks:
            return None
        ball = next(iter(ball_tracks.values()), None)
        if not isinstance(ball, dict):
            return None
        bbox = ball.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        return bbox

    def _ball_position_for_touch(self, ball_bbox: list) -> Tuple[float, float]:
        if self.use_ball_bottom_for_proximity and len(ball_bbox) >= 4:
            x1, _y1, x2, y2 = map(float, ball_bbox[:4])
            return ((x1 + x2) / 2.0, y2)
        c = annotation_center(ball_bbox)
        return (float(c[0]), float(c[1]))

    @staticmethod
    def _foot_distance_to_ball(
        player_bbox: list, ball_position: Tuple[float, float]
    ) -> float:
        x1, _y1, x2, y2 = map(float, player_bbox[:4])
        d_left = calculate_distance((x1, y2), ball_position)
        d_right = calculate_distance((x2, y2), ball_position)
        return float(min(d_left, d_right))

    def _effective_max_touch_distance_px(self, player_bbox: list) -> float:
        base = self.touch_max_player_ball_distance_px
        h = max(0.0, float(player_bbox[3]) - float(player_bbox[1]))
        eff = max(float(base), self.touch_max_distance_height_scale * h)
        cap = self.touch_max_distance_cap_px
        if cap is not None:
            eff = min(eff, float(cap))
        return float(eff)

    def _player_contact(
        self,
        ball_bbox: list,
        player_bbox: list,
        player_id: int,
        ball_owner_id: int,
    ) -> bool:
        if axis_aligned_boxes_overlap(ball_bbox, player_bbox):
            return True
        if ball_overlaps_any_foot(ball_bbox, player_bbox):
            return True
        ball_pos = self._ball_position_for_touch(ball_bbox)
        dist = self._foot_distance_to_ball(player_bbox, ball_pos)
        if dist <= self._effective_max_touch_distance_px(player_bbox):
            return True
        if ball_owner_id != -1 and int(ball_owner_id) == int(player_id):
            return True
        return False

    def _can_register_touch(self, player_id: int) -> bool:
        last = self._last_touch_frame.get(player_id, -10**9)
        return self._frame - last >= self.touch_cooldown_frames

    def _ball_speed_changed(self, s_prev: Optional[float], s_curr: Optional[float]) -> bool:
        """True if consecutive per-frame speeds differ meaningfully (same ball, global)."""
        if s_prev is None or s_curr is None:
            return False
        abs_diff = abs(s_curr - s_prev)
        mx = max(s_curr, s_prev, 1e-6)
        rel = abs_diff / mx
        if abs_diff >= self.speed_change_abs_min_px_per_frame:
            return True
        if rel >= self.speed_change_rel_min:
            return True
        mp = self.speed_change_min_prev_speed_px_per_frame
        if s_prev >= mp and s_curr <= self.speed_change_drop_ratio * s_prev:
            return True
        if s_prev >= mp and s_curr >= self.speed_change_spike_ratio * s_prev:
            return True
        return False

    def update(
        self,
        player_tracks: dict,
        ball_tracks: dict,
        ball_owner_id: int,
    ) -> None:
        """
        ``ball_owner_id`` supplements geometry when the ball bbox misses the player
        box; StatsManager passes the same smoothed owner used for possession when
        ``stats_owner_smoothing_window`` is enabled.
        """
        self._frame += 1
        ball_bbox = self._first_ball_bbox(ball_tracks)
        if ball_bbox is None:
            self._last_ball_center = None
            self._ball_center_prev_prev = None
            return

        ball_center = annotation_center(ball_bbox)
        s_prev: Optional[float] = None
        s_curr: Optional[float] = None
        if self._last_ball_center is not None:
            s_curr = math.hypot(
                ball_center[0] - self._last_ball_center[0],
                ball_center[1] - self._last_ball_center[1],
            )
        if self._ball_center_prev_prev is not None and self._last_ball_center is not None:
            s_prev = math.hypot(
                self._last_ball_center[0] - self._ball_center_prev_prev[0],
                self._last_ball_center[1] - self._ball_center_prev_prev[1],
            )

        speed_changed = self._ball_speed_changed(s_prev, s_curr)
        self._ball_center_prev_prev = self._last_ball_center
        self._last_ball_center = ball_center

        for player_id, player in player_tracks.items():
            pid = int(player_id)
            p_bbox = player.get("bbox")
            if not p_bbox or len(p_bbox) < 4:
                continue

            contact = self._player_contact(
                ball_bbox, p_bbox, pid, int(ball_owner_id)
            )
            team_id = player.get("team_id")

            if contact and speed_changed and team_id is not None and self._can_register_touch(pid):
                self.player_touches[pid] = self.player_touches.get(pid, 0) + 1
                t = int(team_id)
                self.team_touches[t] = self.team_touches.get(t, 0) + 1
                self._last_touch_frame[pid] = self._frame

    def finalize(self) -> dict:
        return {
            "method": "contact_or_owner_ball_speed_change",
            "touch_cooldown_frames": int(self.touch_cooldown_frames),
            "in_transit_velocity_threshold_px_per_frame": float(
                self.in_transit_velocity_threshold_px_per_frame
            ),
            "use_ball_bottom_for_proximity": bool(self.use_ball_bottom_for_proximity),
            "touch_max_player_ball_distance_px": float(
                self.touch_max_player_ball_distance_px
            ),
            "touch_max_distance_height_scale": float(self.touch_max_distance_height_scale),
            "touch_max_distance_cap_px": (
                None
                if self.touch_max_distance_cap_px is None
                else float(self.touch_max_distance_cap_px)
            ),
            "speed_change_abs_min_px_per_frame": float(
                self.speed_change_abs_min_px_per_frame
            ),
            "speed_change_rel_min": float(self.speed_change_rel_min),
            "speed_change_min_prev_speed_px_per_frame": float(
                self.speed_change_min_prev_speed_px_per_frame
            ),
            "speed_change_drop_ratio": float(self.speed_change_drop_ratio),
            "speed_change_spike_ratio": float(self.speed_change_spike_ratio),
            "player_touches": {str(k): int(v) for k, v in self.player_touches.items()},
            "team_touches": {str(k): int(v) for k, v in self.team_touches.items()},
        }

