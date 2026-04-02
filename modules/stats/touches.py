from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from helpers import annotation_center, ball_overlaps_any_foot


class TouchesAnalysis:
    """
    Counts foot–ball contacts: ball bbox overlaps a small foot ROI derived from
    the player bbox. One touch per edge (not in contact → in contact), with a
    per-player cooldown to suppress flicker during dribbles.

    When the ball moves fast between frames (same threshold as possession
    ``ball_in_transit`` in the tracker), contact is ignored for counting so
    through-balls and passes do not register as touches.
    """

    def __init__(
        self,
        touch_cooldown_frames: int = 12,
        vertical_control_enabled: bool = True,
        vertical_control_frac_from_top: float = 0.42,
        in_transit_velocity_threshold_px_per_frame: float = 28.0,
    ):
        self.touch_cooldown_frames = max(1, int(touch_cooldown_frames))
        # Kept for backward compatibility with older config; vertical gating
        # is intentionally disabled to keep touch counting consistent with
        # the "velocity-only" possession approach.
        self.vertical_control_enabled = bool(vertical_control_enabled)
        self.vertical_control_frac_from_top = float(vertical_control_frac_from_top)
        self.in_transit_velocity_threshold_px_per_frame = float(
            in_transit_velocity_threshold_px_per_frame
        )
        self._frame = 0
        self._last_ball_center: Optional[Tuple[float, float]] = None
        # player_id -> was ball overlapping foot ROI last frame we evaluated
        self._in_contact: Dict[int, bool] = {}
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

    def _can_register_touch(self, player_id: int) -> bool:
        last = self._last_touch_frame.get(player_id, -10**9)
        return self._frame - last >= self.touch_cooldown_frames

    def update(
        self,
        player_tracks: dict,
        ball_tracks: dict,
        ball_owner_id: int,
    ) -> None:
        """
        `ball_owner_id` is retained for API compatibility with StatsManager
        (not used for contact counting).
        """
        _ = ball_owner_id
        self._frame += 1
        ball_bbox = self._first_ball_bbox(ball_tracks)
        if ball_bbox is None:
            self._last_ball_center = None
            self._in_contact.clear()
            return

        ball_center = annotation_center(ball_bbox)
        ball_in_transit = False
        v_thresh = self.in_transit_velocity_threshold_px_per_frame
        if v_thresh > 0.0 and self._last_ball_center is not None:
            dpp = math.hypot(
                ball_center[0] - self._last_ball_center[0],
                ball_center[1] - self._last_ball_center[1],
            )
            ball_in_transit = dpp > v_thresh
        self._last_ball_center = ball_center

        present = set(player_tracks.keys())
        for pid in list(self._in_contact.keys()):
            if pid not in present:
                del self._in_contact[pid]

        for player_id, player in player_tracks.items():
            pid = int(player_id)
            p_bbox = player.get("bbox")
            if not p_bbox or len(p_bbox) < 4:
                if pid in self._in_contact:
                    del self._in_contact[pid]
                continue

            contact = ball_overlaps_any_foot(ball_bbox, p_bbox)
            # During fast ball motion, ignore overlap for edges (matches possession in-transit).
            contact_for_touch = contact and not ball_in_transit
            was = self._in_contact.get(pid, False)
            team_id = player.get("team_id")

            if contact_for_touch and not was:
                if team_id is not None and self._can_register_touch(pid):
                    self.player_touches[pid] = self.player_touches.get(pid, 0) + 1
                    t = int(team_id)
                    self.team_touches[t] = self.team_touches.get(t, 0) + 1
                    self._last_touch_frame[pid] = self._frame

            self._in_contact[pid] = contact_for_touch

    def finalize(self) -> dict:
        return {
            "method": "foot_bbox_overlap",
            "touch_cooldown_frames": int(self.touch_cooldown_frames),
            # vertical_control_* intentionally omitted because vertical gating is disabled.
            "in_transit_velocity_threshold_px_per_frame": float(
                self.in_transit_velocity_threshold_px_per_frame
            ),
            "player_touches": {str(k): int(v) for k, v in self.player_touches.items()},
            "team_touches": {str(k): int(v) for k, v in self.team_touches.items()},
        }

