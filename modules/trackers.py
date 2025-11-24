from ultralytics import YOLO
from collections import defaultdict
import supervision as sv

class ByteTracker:
    def __init__(self, model: str):
        self.model = YOLO(model)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
         objects = []
         batch = 20

         for i in range(0, len(frames), batch):
             objects += self.model.predict(frames[i:i+batch], conf=0.1)

         return objects

    def get_tracks(self, frames):
        objects = self.detect_frames(frames)
        all_tracks = defaultdict(list)

        class_map = {
            0: "player",
            1: "goalkeeper",
            2: "referee",
        }

        for frame_idx, prediction in enumerate(objects):
            sv_det = sv.Detections.from_ultralytics(prediction)
            tracks = self.tracker.update_with_detections(sv_det)

            for t in tracks:
                bbox = t[0].tolist()
                track_id = t[4]
                class_id = t[3]
                class_name = class_map.get(class_id, f"class_{class_id}")

                all_tracks[class_name].append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "bbox": bbox
                })

                # The ball ...

                return tracks




