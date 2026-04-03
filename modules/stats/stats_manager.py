import json
import os
from collections import Counter, deque
from datetime import datetime, timezone

from .ball_visibility import BallVisibilityAnalysis
from .possession import PossessionAnalysis
from .touches import TouchesAnalysis
from .pass_entry import PassEntryAnalysis


class StatsManager:
    def __init__(
        self,
        fps: float,
        touch_cooldown_frames: int = 12,
        touches_in_transit_velocity_threshold_px_per_frame: float = 28.0,
        pass_confirm_frames: int = 3,
        stats_owner_smoothing_window: int = 0,
    ):
        self.fps = float(fps) if fps else 0.0
        self.stats_owner_smoothing_window = max(0, int(stats_owner_smoothing_window))
        maxlen = self.stats_owner_smoothing_window if self.stats_owner_smoothing_window > 0 else 1
        self._owner_history: deque = deque(maxlen=maxlen)
        self.ball_visibility = BallVisibilityAnalysis()
        self.possession = PossessionAnalysis()
        self.touches = TouchesAnalysis(
            touch_cooldown_frames=touch_cooldown_frames,
            in_transit_velocity_threshold_px_per_frame=touches_in_transit_velocity_threshold_px_per_frame,
        )
        self.pass_entry = PassEntryAnalysis(
            confirm_frames=pass_confirm_frames,
        )
        self.team_colors = {}
        self.total_frames = 0

    def _smoothed_owner_id_for_stats(self, raw_owner_id: int) -> int:
        """Majority vote over recent frames; ties resolve to unknown (-1)."""
        if self.stats_owner_smoothing_window <= 0:
            return raw_owner_id
        self._owner_history.append(int(raw_owner_id))
        items = list(self._owner_history)
        if not items:
            return raw_owner_id
        c = Counter(items)
        best_n = max(c.values())
        candidates = [k for k, v in c.items() if v == best_n]
        if len(candidates) > 1:
            return -1
        return int(candidates[0])

    def _collect_team_colors(self, player_tracks: dict) -> None:
        for player in player_tracks.values():
            if not isinstance(player, dict):
                continue
            team_id = player.get("team_id")
            team_color = player.get("team_color")
            if team_id is None or team_color is None:
                continue
            try:
                bgr = [int(team_color[0]), int(team_color[1]), int(team_color[2])]
            except Exception:
                continue
            key = str(int(team_id))
            if key not in self.team_colors:
                self.team_colors[key] = bgr

    @staticmethod
    def _first_ball_bbox(ball_tracks: dict):
        if not ball_tracks:
            return None
        ball = next(iter(ball_tracks.values()), None)
        if not isinstance(ball, dict):
            return None
        bbox = ball.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        return bbox

    @staticmethod
    def _owner_team_id(ball_owner_id: int, player_tracks: dict):
        if ball_owner_id == -1:
            return None
        owner = player_tracks.get(ball_owner_id, {})
        return owner.get("team_id")

    @staticmethod
    def _first_ball_position_transformed(ball_tracks: dict):
        if not ball_tracks:
            return None
        ball = next(iter(ball_tracks.values()), None)
        if not isinstance(ball, dict):
            return None
        pos = ball.get("position_transformed")
        if not pos or len(pos) < 2:
            return None
        return pos

    def update(self, player_tracks: dict, ball_tracks: dict, ball_owner_id: int) -> None:
        self.total_frames += 1
        ball_bbox = self._first_ball_bbox(ball_tracks)
        has_ball_detected = ball_bbox is not None
        owner_for_stats = self._smoothed_owner_id_for_stats(ball_owner_id)
        owner_team_id = self._owner_team_id(owner_for_stats, player_tracks)
        # Possession / passes use smoothed owner; visibility uses detection + raw owner
        # so brief unknown frames do not over-count inferred possession.
        has_ball_evidence = bool(has_ball_detected or ball_owner_id != -1)
        has_ball_for_possession = bool(has_ball_detected or owner_for_stats != -1)
        ball_pos_transformed = self._first_ball_position_transformed(ball_tracks)

        self.ball_visibility.update(has_ball=has_ball_evidence)
        self.possession.update(
            has_ball=has_ball_for_possession,
            ball_owner_id=owner_for_stats,
            owner_team_id=owner_team_id,
        )
        # Touches: bbox/foot/proximity/owner contact + ball speed change vs prior frame.
        # Use smoothed owner (same as possession) for the owner fallback when geometry is ambiguous.
        if has_ball_detected:
            self.touches.update(
                player_tracks=player_tracks,
                ball_tracks=ball_tracks,
                ball_owner_id=owner_for_stats,
            )
        else:
            self.touches.update(
                player_tracks=player_tracks,
                ball_tracks={},
                ball_owner_id=owner_for_stats,
            )
        self._collect_team_colors(player_tracks=player_tracks)
        self.pass_entry.update(
            has_ball=has_ball_for_possession,
            owner_id=owner_for_stats,
            owner_team_id=owner_team_id,
            ball_position_transformed=ball_pos_transformed,
        )

    def build_payload(self, video_path: str, output_video_path: str) -> dict:
        return {
            "meta": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "video_path": video_path,
                "output_video_path": output_video_path,
                "fps": self.fps,
                "total_frames": int(self.total_frames),
                "stats_owner_smoothing_window": int(self.stats_owner_smoothing_window),
            },
            "ball_visibility": self.ball_visibility.finalize(),
            "possession": self.possession.finalize(fps=self.fps),
            "touches": self.touches.finalize(),
            "pass_entry": self.pass_entry.finalize(),
            "team_colors_bgr": self.team_colors,
        }

    def save(self, payload: dict, output_video_path: str) -> str:
        output_dir = os.path.dirname(output_video_path) or "."
        stem = os.path.splitext(os.path.basename(output_video_path))[0]
        stats_dir = os.path.join(output_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"{stem}.stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return stats_path

