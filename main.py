from pathlib import Path

from modules import ByteTracker
from modules.trackers import RenderOwnershipStatsConfig

# model = YOLO('models/best.pt')
# result = model.predict("inputs/football_input.mp4", save=True)

# print(f'{result[0]}\n\n')
# for box in result[0].boxes:
#     print(box)

# Info for thesis
# CHOOSING THE MODEL
# Yolo has different models for each version (f.e. yolov8) defined by its accuracy and resource usage (from s to x)
# So if you want a high accuracy, you choose X, for the first one I've chosen medium one
# When predicting and saving to your machine, you need to be accurate, as it is using your own RAM, so for heavy inputs and high accuracy you can run out of it. In order to avoid this, use stream=True as an argument when predicting

# OBJECT DETECTION (YOLO CNN)
# 1. Bounding boxes representation (either by pointing to the center of the object (x,y coordinates) and choosing width and height or by two pointe (x1, y1 - top left, x2, y2 - bottom right = xyxy)
# Player detection is good now, but ball detection is not so well as I want
# And I also need to ignore the people out of the field and classify (like players, coaches, etc.)

# TRAINING
# So I will train it using a public dataset of annotated images from football clips, where players, ball and referee were mentioned and 'boxed'
# I chose v5 as it seems to have the most accuracy for the ball from all other ones (for now)

# Installed all the necessary libraries, and roboflow for dataset training, I trained the yolov5 model based on this dataset input and labaels.\
# Config with all training params can be found in train/data.yaml and all validation images and labels (first col is an object class) can be found in train/valid

# Trained using google collab, as I don't have good GPU. Saved trained models to models folder
# Now it is annotating people as players/referees and ball detection is better than before; also it doesn't detect anything outside the field

# BYTE TRACKER
# ByteTracker is a tracking algorithm that uses a combination of YOLO and ByteTrack to track objects in a video
# It is a memory-safe tracking algorithm that does not store all frames in memory, but rather streams them and runs YOLO on bounded batches
# It is a very efficient tracking algorithm that is able to track objects in a video with a high FPS
# It is a very accurate tracking algorithm that is able to track objects in a video with a high accuracy
# It is a very fast tracking algorithm that is able to track objects in a video with a high speed
# It is a very easy to use tracking algorithm that is able to track objects in a video with a high ease of use
# It is a very reliable tracking algorithm that is able to track objects in a video with a high reliability
# It is a very accurate tracking algorithm that is able to track objects in a video with a high accuracy

# BALL DETECTION
# I need to improve the ball detection by using the best model for it, and also I need to ignore the people out of the field and classify (like players, coaches, etc.)
# I can use the ByteTracker to track the ball and then use the best model for it to detect the ball
# I can also use the ByteTracker to track the people and then use the best model for it to detect the people
# I can also use the ByteTracker to track the people and then use the best model for it to detect the people

# LARGE DATASET ISSUES
# When training on a large dataset, the model may not be able to fit in memory, so we need to use a smaller batch size
# Security people counts as referees, so I need to improve the classification of people and referees

def run_analysis(
    input_path: str,
    model_path: str = "models/best_soccanna.pt",
    output_video_path: str | None = None,
    reid_enabled: bool = False,
    reid_model_config: str = "",
    reid_model_weights: str = "",
    reid_device: str = "cuda:0",
    reid_cosine_thresh: float = 0.70,
    reid_max_age_frames: int = 30,
) -> dict:
    """
    Run full analysis on a single video and return basic metadata.

    This is used both from CLI and from the Streamlit UI.
    """
    input_path = str(input_path)
    input_stem = Path(input_path).stem
    if output_video_path is None:
        output_video_path = f"outputs/{input_stem}_output.mp4"

    tracker = ByteTracker(model_path)
    tracks = tracker.process_video(
        input_path=input_path,
        output_path=output_video_path,
        cache=False,
        device="cuda:0",
        use_half=False,
        run_quality_checks=True,
        batch=32,
        conf=0.6,
        imgsz=1024,
        ball_tracking_mode="hybrid",
        interpolate_ball_positions=False,
        perspective_transform_enabled=True,
        reid_enabled=reid_enabled,
        reid_model_config=reid_model_config,
        reid_model_weights=reid_model_weights,
        reid_device=reid_device,
        reid_cosine_thresh=reid_cosine_thresh,
        reid_max_age_frames=reid_max_age_frames,
        render_cfg=RenderOwnershipStatsConfig(
            use_hw_encode=True,
            hw_encoder="h264_mf",
            team_assign_debug=True,
            ball_owner_hold_frames=12,
            ball_owner_lock_enabled=True,
            ball_owner_release_distance_px=110.0,
            camera_movement_enabled=True,
            speed_distance_enabled=False,
        ),
    )

    stats_path = None
    if isinstance(tracks, dict) and tracks.get("stats_path"):
        stats_path = tracks["stats_path"]
        print(f"[Main] Stats file: {stats_path}")

    if isinstance(tracks, dict) and "ball_owner" in tracks:
        ball_owner = tracks["ball_owner"]
        assigned_frames = sum(1 for owner in ball_owner if owner != -1)
        print(f"[Main] Ball assigned in {assigned_frames}/{len(ball_owner)} tracked frames")

    return {
        "input_path": input_path,
        "output_video_path": output_video_path,
        "stats_path": stats_path,
    }


def main():
    # Simple CLI entrypoint: runs analysis on the default demo video.
    default_input = "inputs/football_input.mp4"
    default_reid_config = "models/reid/osnet.yaml"
    default_reid_weights = "models/reid/osnet_ibn_x1_0_imagenet.pth"

    reid_ready = Path(default_reid_config).exists() and Path(default_reid_weights).exists()
    if Path(default_reid_config).exists() and not Path(default_reid_weights).exists():
        print(
            "[Main] ReID config found but weights missing. "
            f"Expected weights at: {default_reid_weights}. "
            "Running with ReID disabled."
        )

    result = run_analysis(
        default_input,
        reid_enabled=False,
        reid_model_config=default_reid_config if reid_ready else "",
        reid_model_weights=default_reid_weights if reid_ready else "",
    )
    print(f"[Main] Done. Output video: {result['output_video_path']}")


if __name__ == "__main__":
    main()

# Tracking explained
# So when we want to detect, we predict xyxy for each bounding box at each frame, not really caring about the previous state of xyxy for each b. box, so it looks rapid.
# So tracking is basically assigning the same entity box for multiple frames for each object we want to modules, so it will look smoothly
# I can apply some logic for it as well, for example, we modules a player, and thanks to tracking approach, he has a bounding box for each state, so if in previous state he wears white shirt, and now also white shirt, it means he plays for the same team. etc.

# ByteTracker
# So using ready provided ByteTracker from supervision and giving it our predictions for each frame using YOLO (ultralytics) we detect objects for each batch (depending on FPS)
# And then, we are gathering tracks info for each class of objects for each frame and labeling it with id in order to have on bbox for each object across multiple frames
# And lastly, we cache it for better future performance

# JERSEY NUMBER RECOGNITION + ID REASSIGNMENT
# Tried OCR - fails in most cases because of unclear bbox coordinates, each jersey needs its own
