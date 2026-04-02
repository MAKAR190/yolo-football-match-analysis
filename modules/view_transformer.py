from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from helpers import annotation_center, player_foot_position
from .camera_movement_estimator import CameraMovementEstimator


class ViewTransformer:
    """
    Perspective transformer for mapping field points from image coordinates
    into a normalized (bird's-eye) coordinate system.
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        source_polygon_margin_px: float = 35.0,
        auto_calibration_enabled: bool = False,
        auto_calibration_sample_frames: int = 12,
        auto_calibration_frame_stride: int = 15,
        auto_calibration_min_confidence: float = 0.12,
    ):
        self.pixel_vertices = np.array(
            [[110, 1035], [265, 275], [910, 260], [1640, 915]], dtype=np.float32
        )

        # Target rectangle dimensions (field units, full football pitch).
        # Keep these consistent with downstream stats/heatmap modules.
        court_width = 68.0
        court_length = 105.0
        self.pitch_width = float(court_width)
        self.pitch_length = float(court_length)

        self.target_vertices = np.array(
            [
                [0, court_width],
                [0, 0],
                [court_length, 0],
                [court_length, court_width],
            ],
            dtype=np.float32,
        )

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )
        self.source_polygon_margin_px = float(source_polygon_margin_px)
        if auto_calibration_enabled and video_path is not None:
            self._auto_calibrate(
                video_path=video_path,
                sample_frames=int(auto_calibration_sample_frames),
                frame_stride=int(auto_calibration_frame_stride),
                min_confidence=float(auto_calibration_min_confidence),
            )

    def _set_pixel_vertices(self, pixel_vertices: np.ndarray) -> None:
        arr = np.asarray(pixel_vertices, dtype=np.float32)
        if arr.shape != (4, 2):
            return
        self.pixel_vertices = arr
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def _auto_calibrate(
        self,
        video_path: str,
        sample_frames: int,
        frame_stride: int,
        min_confidence: float,
    ) -> None:
        try:
            # Optional dependency: this repo may run without auto-calibration.
            from .auto_field_calibrator import (  # type: ignore
                AutoFieldCalibrator,
                AutoFieldCalibratorConfig,
            )

            calibrator = AutoFieldCalibrator(
                AutoFieldCalibratorConfig(
                    sample_frames=max(4, int(sample_frames)),
                    frame_stride=max(1, int(frame_stride)),
                    min_confidence=max(0.01, float(min_confidence)),
                )
            )
            result = calibrator.calibrate(video_path)
            if result is None:
                return
            verts = result.get("pixel_vertices")
            if verts is None:
                return
            self._set_pixel_vertices(verts)
            self.source_polygon_margin_px = min(float(self.source_polygon_margin_px), 25.0)
        except Exception:
            return

    def transform_point(
        self,
        point: Tuple[float, float] | List[float] | np.ndarray,
        pixel_vertices: Optional[np.ndarray] = None,
        perspective_transformer: Optional[np.ndarray] = None,
    ) -> Optional[List[float]]:
        """
        Transform a single point (x,y) from pixel space to field space.
        Returns None if the point is outside the quadrilateral.
        """
        poly = pixel_vertices if pixel_vertices is not None else self.pixel_vertices
        transformer = (
            perspective_transformer
            if perspective_transformer is not None
            else self.perspective_transformer
        )

        p = (float(point[0]), float(point[1]))
        signed_dist = cv2.pointPolygonTest(poly, p, True)
        if signed_dist < -float(self.source_polygon_margin_px):
            return None

        reshaped_point = np.asarray(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped_point, transformer)
        x_t, y_t = transformed.reshape(-1, 2)[0].tolist()

        # Keep transformed positions bounded to the tactical plane.
        x_t = max(0.0, min(float(self.pitch_length), float(x_t)))
        y_t = max(0.0, min(float(self.pitch_width), float(y_t)))
        return [x_t, y_t]

    @staticmethod
    def _source_point_from_track(
        object_key: str, track_info: Dict[str, Any]
    ) -> Optional[Tuple[float, float]]:
        # Preferred: something already adjusted (repo uses this name).
        if "position_adjusted" in track_info and track_info["position_adjusted"] is not None:
            x, y = track_info["position_adjusted"]
            return float(x), float(y)

        if "position" in track_info and track_info["position"] is not None:
            x, y = track_info["position"]
            return float(x), float(y)

        bbox = track_info.get("bbox")
        if not bbox:
            return None

        # For players: transform the foot position (lower part of bbox).
        # For ball/referees: use bbox center.
        if object_key == "players":
            x, y = player_foot_position(bbox)
            return float(x), float(y)

        # Default: use center of bbox.
        x, y = annotation_center(bbox)
        return float(x), float(y)

    @staticmethod
    def _fallback_player_source_points(track_info: Dict[str, Any]) -> List[Tuple[float, float]]:
        """
        Candidate fallback points for player transform when foot point is rejected.
        Ordered from most to least semantically useful.
        """
        bbox = track_info.get("bbox")
        if not bbox or len(bbox) < 4:
            return []
        x1, y1, x2, y2 = map(float, bbox[:4])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        h = max(1.0, y2 - y1)
        # Slightly above the foot line is often more stable for distant players.
        lower_center = (cx, y2 - 0.15 * h)
        return [
            lower_center,
            (cx, cy),
            (cx, y1 + 0.65 * h),
        ]

    def add_transformed_position_to_tracks(
        self,
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
        object_keys: Tuple[str, ...] = ("players", "ball", "referees"),
        video_path: Optional[str] = None,
        track_original_frame_indices: Optional[List[int]] = None,
        dynamic_homography_enabled: bool = True,
        dynamic_update_interval: int = 20,
    ) -> None:
        """
        Mutates `tracks` in place by adding `position_transformed` to each track_info.
        """
        per_track_pixel_vertices = [self.pixel_vertices] * len(tracks.get("players", []))
        per_track_transformers = [self.perspective_transformer] * len(tracks.get("players", []))

        if (
            dynamic_homography_enabled
            and video_path is not None
            and track_original_frame_indices is not None
            and len(track_original_frame_indices) > 0
        ):
            try:
                frame_to_shift = {}
                cap = cv2.VideoCapture(video_path)
                ok, first_frame = cap.read()
                if ok and first_frame is not None:
                    estimator = CameraMovementEstimator(first_frame)
                    frame_idx = 0
                    cum_dx = 0.0
                    cum_dy = 0.0
                    frame_to_shift[frame_idx] = (cum_dx, cum_dy)
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        frame_idx += 1
                        dx, dy = estimator.update(frame)
                        cum_dx += float(dx)
                        cum_dy += float(dy)
                        frame_to_shift[frame_idx] = (cum_dx, cum_dy)
                cap.release()

                interval = max(1, int(dynamic_update_interval))
                last_shift = (0.0, 0.0)
                last_vertices = self.pixel_vertices
                last_transformer = self.perspective_transformer
                for track_idx, original_idx in enumerate(track_original_frame_indices):
                    if track_idx >= len(per_track_pixel_vertices):
                        break
                    shift = frame_to_shift.get(int(original_idx), last_shift)
                    if track_idx % interval == 0 or track_idx == 0:
                        sx, sy = shift
                        shifted_vertices = self.pixel_vertices.copy()
                        shifted_vertices[:, 0] = shifted_vertices[:, 0] + float(sx)
                        shifted_vertices[:, 1] = shifted_vertices[:, 1] + float(sy)
                        last_vertices = shifted_vertices
                        last_transformer = cv2.getPerspectiveTransform(
                            shifted_vertices.astype(np.float32),
                            self.target_vertices.astype(np.float32),
                        )
                        last_shift = shift
                    per_track_pixel_vertices[track_idx] = last_vertices
                    per_track_transformers[track_idx] = last_transformer
            except Exception:
                # Fall back to static homography if dynamic update fails.
                pass

        for object_key in object_keys:
            if object_key not in tracks:
                continue

            for frame_num, object_tracks in enumerate(tracks[object_key]):
                for track_id, track_info in object_tracks.items():
                    if not isinstance(track_info, dict):
                        continue

                    src_point = self._source_point_from_track(object_key, track_info)
                    if src_point is None:
                        continue

                    pix = (
                        per_track_pixel_vertices[frame_num]
                        if frame_num < len(per_track_pixel_vertices)
                        else self.pixel_vertices
                    )
                    trf = (
                        per_track_transformers[frame_num]
                        if frame_num < len(per_track_transformers)
                        else self.perspective_transformer
                    )
                    dst_point = self.transform_point(
                        src_point,
                        pixel_vertices=pix,
                        perspective_transformer=trf,
                    )
                    if dst_point is None and object_key == "players":
                        # Far/top players can have noisy foot points; try robust
                        # fallback anchors before discarding this sample.
                        for alt_src in self._fallback_player_source_points(track_info):
                            dst_point = self.transform_point(
                                alt_src,
                                pixel_vertices=pix,
                                perspective_transformer=trf,
                            )
                            if dst_point is not None:
                                break
                    if dst_point is None:
                        continue

                    tracks[object_key][frame_num][track_id]["position_transformed"] = dst_point

