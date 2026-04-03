from pathlib import Path

from helpers import get_video_fps
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
) -> dict:
    """
    Run full analysis on a single video and return basic metadata.

    This is used both from CLI and from the Streamlit UI.
    """
    input_path = str(input_path)
    input_stem = Path(input_path).stem
    input_fps = get_video_fps(input_path)
    print(f"[Main] Detected input FPS: {input_fps:.3f}")
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
        conf=0.4,
        imgsz=1024,
        ball_tracking_mode="hybrid",
        interpolate_ball_positions=False,
        ball_min_candidate_confidence=0.55,
        ball_aspect_ratio_min=0.45,
        ball_aspect_ratio_max=2.2,
        ball_prefilter_debug=False,
        render_cfg=RenderOwnershipStatsConfig(
            use_hw_encode=True,
            hw_encoder="h264_mf",
            team_assign_debug=False,
            ball_owner_hold_frames=10,
            ball_owner_hold_min_confidence=0.35,
            ball_owner_hold_frames_weak_evidence=5,
            ball_owner_min_confidence=0.5,
            ball_owner_lock_enabled=True,
            ball_owner_switch_confirm_frames=2,
            ball_owner_switch_confirm_frames_cross_team=3,
            ball_owner_switch_margin_px=20.0,
            ball_owner_switch_margin_ratio=0.70,
            ball_owner_release_distance_px=105.0,
            ball_assign_max_player_ball_distance_px=72.0,
            ball_assign_ambiguity_margin_px=12.0,
            ball_bbox_containment_margin_px=6.0,
            ball_max_distance_height_scale=0.38,
            ball_max_distance_cap_px=110.0,
            ball_owner_primary_confirm_frames=2,
            stats_owner_smoothing_window=5,
            motion_touch_redetect_cooldown_frames=2,
            motion_touch_min_prev_speed_px_per_frame=10.0,
            motion_touch_speed_drop_ratio=0.82,
            motion_touch_max_candidate_distance_px=52.0,
            motion_touch_min_second_best_margin_px=14.0,
            motion_touch_confirm_frames=2,
            motion_touch_candidate_confirm_frames=2,
            camera_movement_enabled=True,
            camera_movement_overlay_enabled=False,
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
        "input_fps": float(input_fps),
        "output_video_path": output_video_path,
        "stats_path": stats_path,
    }


def main():
    # Simple CLI entrypoint: runs analysis on the default demo video.
    default_input = "inputs/football_input.mp4"

    result = run_analysis(
        default_input,
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
