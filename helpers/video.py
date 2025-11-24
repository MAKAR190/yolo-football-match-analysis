import cv2

def frame_video(path: str):
    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        good, frame = cap.read()
        if not good:
            break
        frames.append(frame)

    cap.release()
    return frames

def save_video(frames, path: str):
    codec = cv2.VideoWriter_fourcc(*'XVID')

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, codec, 30.0, (width, height))
    for frame in frames:
        writer.write(frame)

    writer.release()