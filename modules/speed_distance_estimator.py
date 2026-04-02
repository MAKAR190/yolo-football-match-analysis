from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from helpers import player_foot_position


@dataclass(frozen=True)
class SpeedDistanceConfig:
    # How many tracked frames per measurement window (like the repo's frame_window)
    frame_window: int = 5
    # Use the actual video FPS (passed in by the pipeline)
    fps: float = 30.0


class SpeedDistanceEstimator:
    """
    Compute speed (km/h) and cumulative distance for player tracks using
    `position_transformed` (field-plane coordinates).
    """

    def __init__(self, config: SpeedDistanceConfig):
        self.cfg = config

    @staticmethod
    def _euclidean(a: List[float], b: List[float]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return float(np.hypot(dx, dy))

    def add_speed_and_distance_to_tracks(
        self,
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
        object_keys: Tuple[str, ...] = ("players",),
    ) -> None:
        total_distance: Dict[str, Dict[int, float]] = {}

        for object_key in object_keys:
            if object_key not in tracks:
                continue
            object_tracks = tracks[object_key]
            n_frames = len(object_tracks)

            for frame_num in range(0, n_frames, int(self.cfg.frame_window)):
                last_frame = min(frame_num + int(self.cfg.frame_window), n_frames - 1)

                for track_id in object_tracks[frame_num].keys():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = object_tracks[frame_num][track_id].get("position_transformed")
                    end_pos = object_tracks[last_frame][track_id].get("position_transformed")
                    if start_pos is None or end_pos is None:
                        continue

                    distance_covered = self._euclidean(start_pos, end_pos)
                    time_elapsed = float(last_frame - frame_num) / float(self.cfg.fps or 1.0)
                    if time_elapsed <= 0:
                        continue

                    speed_mps = distance_covered / time_elapsed
                    speed_kmh = speed_mps * 3.6

                    total_distance.setdefault(object_key, {}).setdefault(track_id, 0.0)
                    total_distance[object_key][track_id] += float(distance_covered)

                    for k in range(frame_num, last_frame):
                        if track_id not in tracks[object_key][k]:
                            continue
                        tracks[object_key][k][track_id]["speed"] = float(speed_kmh)
                        tracks[object_key][k][track_id]["distance"] = float(
                            total_distance[object_key][track_id]
                        )

    @staticmethod
    def draw_speed_and_distance(
        frame_bgr: np.ndarray,
        object_tracks_for_frame: Dict[int, Dict[str, Any]],
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Draw speed+distance for all tracks in a single rendered frame.
        """
        frame = frame_bgr
        for _, track_info in object_tracks_for_frame.items():
            speed = track_info.get("speed")
            distance = track_info.get("distance")
            if speed is None or distance is None:
                continue

            bbox = track_info.get("bbox")
            if not bbox:
                continue

            x, y = player_foot_position(bbox)
            y = int(y + 40)
            cv2.putText(
                frame,
                f"{float(speed):.2f} km/h",
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2,
            )
            cv2.putText(
                frame,
                f"{float(distance):.2f} m",
                (int(x), int(y + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2,
            )
        return frame

