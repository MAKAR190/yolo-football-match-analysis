class PossessionAnalysis:
    def __init__(self):
        self.visible_ball_frames = 0
        self.known_owner_frames = 0
        self.unknown_owner_frames = 0
        self.team_frames = {}
        self.player_frames = {}

    def update(self, has_ball: bool, ball_owner_id: int, owner_team_id) -> None:
        if not has_ball:
            return

        self.visible_ball_frames += 1
        if ball_owner_id == -1:
            self.unknown_owner_frames += 1
            return

        self.known_owner_frames += 1
        self.player_frames[ball_owner_id] = self.player_frames.get(ball_owner_id, 0) + 1

        if owner_team_id is None:
            self.unknown_owner_frames += 1
            return
        self.team_frames[int(owner_team_id)] = self.team_frames.get(int(owner_team_id), 0) + 1

    def finalize(self, fps: float) -> dict:
        dt = (1.0 / fps) if fps > 0 else 0.0
        visible = float(self.visible_ball_frames)

        team_seconds = {str(k): float(v) * dt for k, v in self.team_frames.items()}
        player_seconds = {str(k): float(v) * dt for k, v in self.player_frames.items()}

        team_ratio = {}
        if visible > 0:
            for k, v in self.team_frames.items():
                team_ratio[str(k)] = float(v) / visible

        return {
            "visible_ball_frames": int(self.visible_ball_frames),
            "known_owner_frames": int(self.known_owner_frames),
            "unknown_owner_frames": int(self.unknown_owner_frames),
            "team_possession_seconds": team_seconds,
            "team_possession_ratio_on_visible_ball": team_ratio,
            "player_possession_seconds": player_seconds,
        }

