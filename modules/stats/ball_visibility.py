class BallVisibilityAnalysis:
    def __init__(self):
        self.total_frames = 0
        self.frames_with_ball = 0

    def update(self, has_ball: bool) -> None:
        self.total_frames += 1
        if has_ball:
            self.frames_with_ball += 1

    def finalize(self) -> dict:
        ratio = 0.0
        if self.total_frames > 0:
            ratio = float(self.frames_with_ball) / float(self.total_frames)
        return {
            "total_frames": int(self.total_frames),
            "frames_with_ball": int(self.frames_with_ball),
            "ball_visibility_ratio": float(ratio),
        }

