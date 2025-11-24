from ultralytics import YOLO
from helpers import frame_video, save_video
from modules import ByteTracker

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

def main():
    frames = frame_video('inputs/football_input.mp4')

    tracker = ByteTracker('models/best.pt')
    tracks = tracker.get_tracks(frames)

    # save_video(frames, 'outputs/football_output.avi')

if __name__ == '__main__':
    main()

# Tracking explained
# So when we want to detect, we predict xyxy for each bounding box at each frame, not really caring about the previous state of xyxy for each b. box, so it looks rapid.
# So tracking is basically assigning the same entity box for multiple frames for each object we want to modules, so it will look smoothly
# I can apply some logic for it as well, for example, we modules a player, and thanks to tracking approach, he has a bounding box for each state, so if in previous state he wears white shirt, and now also white shirt, it means he plays for the same team. etc.

# ByteTracker


