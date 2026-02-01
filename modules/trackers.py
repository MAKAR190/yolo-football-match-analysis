import supervision as sv
import pickle
import os
import cv2
from .annotations import Annotations
from ultralytics import YOLO

CACHE_FOLDER = "cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)
shared_annotations = Annotations()

class ByteTracker:
    CACHE_PATH = os.path.join(CACHE_FOLDER, "bytetracker_cache.pkl")

    def __init__(self, model: str, jersey_model: str = None):
        self.model = YOLO(model)
        self.tracker = sv.ByteTrack()
        self.jersey_model = YOLO(jersey_model) if jersey_model else None

    def detect_frames(self, frames):
         objects = []
         batch = 30

         for i in range(0, len(frames), batch):
             objects += self.model.predict(frames[i:i+batch], conf=0.1)

         return objects

    def get_tracks(self, frames, cache=True, debug_crop=False):
        cache_path = self.CACHE_PATH

        if cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        objects = self.detect_frames(frames)

        tracks = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": []
        }

        class_map = {
            0: "ball",
            1: "goalkeeper",
            2: "player",
            3: "referee"
        }

        class_to_track_key = {
            "player": "players",
            "goalkeeper": "goalkeepers",
            "referee": "referees",
            "ball": "ball"
        }

        all_player_crops = []
        crop_locations = []

        for frame_idx, prediction in enumerate(objects):
            sv_det = sv.Detections.from_ultralytics(prediction)
            detection_with_tracks = self.tracker.update_with_detections(sv_det)

            for key in tracks:
                tracks[key].append({})

            for t in detection_with_tracks:
                bbox = t[0].tolist()
                tracker_id = t[4]
                class_id = t[3]
                class_name = class_map.get(class_id)
                track_key = class_to_track_key.get(class_name)

                if class_name == "player":
                    x1, y1, x2, y2 = map(int, bbox)

                    tracks["players"][frame_idx][tracker_id] = {"bbox": bbox, "jersey": None}

                    if frame_idx % 30 == 0:
                        jersey_height = (y2 - y1) // 2
                        jersey_x1, jersey_y1, jersey_x2, jersey_y2 = x1, y1, x2, y1 + jersey_height

                        jersey_crop = frames[frame_idx][jersey_y1:jersey_y2, jersey_x1:jersey_x2]
                        gray_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
                        rgb_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)

                        all_player_crops.append(rgb_crop)
                        crop_locations.append(
                            (frame_idx, tracker_id, tuple(bbox), (jersey_x1, jersey_y1, jersey_x2, jersey_y2)))

                        tracks["players"][frame_idx][tracker_id]["jersey_bbox"] = [jersey_x1, jersey_y1, jersey_x2,
                                                                                   jersey_y2]

                elif track_key:
                    tracks[track_key][frame_idx][1 if track_key == "ball" else tracker_id] = {"bbox": bbox}

        jersey_numbers = [None] * len(all_player_crops)
        if self.jersey_model and all_player_crops:
            batch_size = 30
            for i in range(0, len(all_player_crops), batch_size):
                batch_crops = all_player_crops[i:i + batch_size]
                preds = self.jersey_model.predict(batch_crops, conf=0.2)

                for j, pred in enumerate(preds):
                    if pred.boxes is not None and len(pred.boxes) > 0:
                        cls_id = int(pred.boxes.cls[0].cpu().numpy())
                        jersey_numbers[i + j] = cls_id

        jersey_mapping = dict(zip(crop_locations, jersey_numbers))
        for (pred_frame_idx, tracker_id, bbox_tuple, jersey_bbox), jersey_number in jersey_mapping.items():
            for frame_offset in range(30):
                target_frame = pred_frame_idx + frame_offset
                if target_frame < len(tracks["players"]):
                    if tracker_id in tracks["players"][target_frame]:
                        tracks["players"][target_frame][tracker_id]["jersey"] = jersey_number
                        tracks["players"][target_frame][tracker_id]["jersey_bbox"] = list(jersey_bbox)

        if cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    @staticmethod
    def draw_annotations(frames, tracks, show_jersey_crop=False):
        output_frames = []

        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            player_tracks = tracks["players"][frame_idx]
            ball_tracks = tracks["ball"][frame_idx]
            referee_tracks = tracks["referees"][frame_idx]

            for _, player in player_tracks.items():
                frame = shared_annotations.draw_player_ellipse(
                    frame,
                    player["bbox"],
                    (255, 0, 255),
                    player["jersey"]
                )
            for referee_id, referee in referee_tracks.items():
                frame = shared_annotations.draw_player_ellipse(
                    frame,
                    referee["bbox"],
                    (255, 0, 255),
                    referee_id
                    )


                if show_jersey_crop and player.get("jersey_bbox") is not None:
                    x1, y1, x2, y2 = player["jersey_bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for ball_id, ball in ball_tracks.items():
                frame = shared_annotations.draw_ball_marker(
                    frame,
                    ball["bbox"],
                    color=(0, 255, 255)
                )

            output_frames.append(frame)

        return output_frames