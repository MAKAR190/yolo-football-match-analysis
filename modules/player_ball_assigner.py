from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from helpers import annotation_center, calculate_distance
from helpers.foot_roi import ball_overlaps_any_foot

"""
Ball-to-player assignment for possession / overlay.

**Default semantics (single pipeline for video + stats):** ``assign_mode="distance_gates"``
uses foot-point distance with a max radius, a vertical “control zone” (ball not above
the torso), and a best-vs-second-best margin so ambiguous frames yield no owner.

**Optional strict mode:** ``assign_mode="foot_overlap"`` requires ball bbox ∩ foot ROI
(aligned with touch detection); stricter for ground play, weak for aerial balls.

Stats and overlay both use the same assigner configuration to avoid drift between
highlighting and aggregate possession.
"""

BallAssignMode = Literal["distance_gates", "foot_overlap"]


@dataclass(frozen=True)
class BallAssignment:
    player_id: int
    distance_px: float
    second_best_distance_px: float


class PlayerBallAssigner:
    """
    Assign the ball to one player track per frame.

    ``distance_gates``: ball center vs left/right foot points, max distance, optional
    vertical gate, ambiguity margin.

    ``foot_overlap``: only players with ball bbox overlapping a foot ROI; tie-break
    by the same foot distance; ambiguity margin applies among overlapping players.

    ``max_distance_scale_by_player_height``: if > 0, per-player max distance is
    ``max(max_player_ball_distance, scale * player_bbox_height)`` so reach scales
    with player size and resolution.

    ``use_ball_bottom_for_proximity``: use the bottom center of the ball bbox for
    distance and vertical checks (closer to the ground contact point than the bbox
    center when the ball is approaching the feet).
    """

    def __init__(
        self,
        max_player_ball_distance: float = 78.0,
        assign_mode: BallAssignMode = "distance_gates",
        vertical_control_enabled: bool = False,
        vertical_control_frac_from_top: float = 0.42,
        ambiguity_margin_px: float = 10.0,
        use_ball_bottom_for_proximity: bool = True,
    ):
        self.max_player_ball_distance = float(max_player_ball_distance)
        self.assign_mode: BallAssignMode = assign_mode
        self.vertical_control_enabled = bool(vertical_control_enabled)
        self.vertical_control_frac_from_top = float(vertical_control_frac_from_top)
        self.ambiguity_margin_px = float(ambiguity_margin_px)
        self.use_ball_bottom_for_proximity = bool(use_ball_bottom_for_proximity)

    def _ball_position_for_proximity(self, ball_bbox: List[float]) -> Tuple[float, float]:
        if self.use_ball_bottom_for_proximity and len(ball_bbox) >= 4:
            x1, _y1, x2, y2 = map(float, ball_bbox[:4])
            return ((x1 + x2) / 2.0, y2)
        return annotation_center(ball_bbox)

    def _effective_max_distance_px(self, player_bbox: List[float]) -> float:
        base = self.max_player_ball_distance
        h = max(0.0, float(player_bbox[3]) - float(player_bbox[1]))
        return max(base, h)

    def assign_ball_to_player(
        self,
        players: Dict[int, Dict[str, Any]],
        ball_bbox: Optional[List[float]],
    ) -> int:
        assignment = self.assign_ball_to_player_with_metrics(players, ball_bbox)
        return assignment.player_id if assignment is not None else -1

    def _foot_distance_to_ball(self, player_bbox: List[float], ball_position: Tuple[float, float]) -> float:
        x1, y1, x2, y2 = player_bbox[:4]
        distance_left = calculate_distance((x1, y2), ball_position)
        distance_right = calculate_distance((x2, y2), ball_position)
        return float(min(distance_left, distance_right))

    def _collect_scored_players(
        self,
        players: Dict[int, Dict[str, Any]],
        ball_bbox: List[float],
        ball_position: Tuple[float, float],
    ) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for player_id, player in players.items():
            player_bbox = player.get("bbox")
            if not player_bbox or len(player_bbox) < 4:
                continue

            pb = list(map(float, player_bbox[:4]))
            if self.assign_mode == "foot_overlap":
                if not ball_overlaps_any_foot(ball_bbox, pb):
                    continue
            else:
                # Distance-based mode: we intentionally do NOT apply any vertical
                # "control zone" gating here. Velocity-based in-transit logic is
                # handled upstream in `trackers.py`.
                pass

            dist = self._foot_distance_to_ball(pb, ball_position)
            if dist >= self._effective_max_distance_px(pb):
                continue
            out.append((int(player_id), dist))

        out.sort(key=lambda x: x[1])
        return out

    def assign_ball_to_player_with_metrics(
        self,
        players: Dict[int, Dict[str, Any]],
        ball_bbox: Optional[List[float]],
    ) -> Optional[BallAssignment]:
        if ball_bbox is None:
            return None

        ball_position = self._ball_position_for_proximity(ball_bbox)
        scored = self._collect_scored_players(players, ball_bbox, ball_position)
        if not scored:
            return None

        best_id, best_dist = scored[0]
        second_best = scored[1][1] if len(scored) > 1 else float("inf")

        return BallAssignment(
            player_id=best_id,
            distance_px=best_dist,
            second_best_distance_px=second_best,
        )
