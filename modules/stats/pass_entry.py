class PassEntryAnalysis:
    """
    Pass heuristics with confidence gating.

    Confidence gate:
    - ball visible
    - owner known
    - owner stable for at least `confirm_frames`
    """

    def __init__(
        self,
        confirm_frames: int = 3,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.confirm_frames = max(1, int(confirm_frames))
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)

        self.owner_streak = 0
        self.last_owner_raw = -1
        self.last_confirmed_owner = -1
        self.last_confirmed_team = None

        self.team_pass_attempts = {}
        self.team_pass_completed = {}
        self.team_pass_intercepted = {}
        self.player_pass_attempts = {}
        self.player_pass_completed = {}

    @staticmethod
    def _inc(bucket: dict, key) -> None:
        if key is None:
            return
        bucket[key] = bucket.get(key, 0) + 1

    def update(
        self,
        has_ball: bool,
        owner_id: int,
        owner_team_id,
        ball_position_transformed,
    ) -> None:
        if not has_ball or owner_id == -1:
            self.owner_streak = 0
            self.last_owner_raw = -1
            return

        if owner_id == self.last_owner_raw:
            self.owner_streak += 1
        else:
            self.last_owner_raw = owner_id
            self.owner_streak = 1

        high_confidence = self.owner_streak >= self.confirm_frames
        if not high_confidence:
            return

        # Pass heuristic: confirmed owner changed and remained stable.
        if self.last_confirmed_owner != -1 and owner_id != self.last_confirmed_owner:
            prev_player = int(self.last_confirmed_owner)
            prev_team = int(self.last_confirmed_team) if self.last_confirmed_team is not None else None
            new_team = int(owner_team_id) if owner_team_id is not None else None

            self._inc(self.player_pass_attempts, prev_player)
            self._inc(self.team_pass_attempts, prev_team)

            if prev_team is not None and new_team is not None and prev_team == new_team:
                self._inc(self.player_pass_completed, prev_player)
                self._inc(self.team_pass_completed, prev_team)
            else:
                self._inc(self.team_pass_intercepted, prev_team)

        self.last_confirmed_owner = int(owner_id)
        self.last_confirmed_team = int(owner_team_id) if owner_team_id is not None else None

    def finalize(self) -> dict:
        return {
            "confirm_frames": int(self.confirm_frames),
            "team_pass_attempts": {str(k): int(v) for k, v in self.team_pass_attempts.items()},
            "team_pass_completed": {str(k): int(v) for k, v in self.team_pass_completed.items()},
            "team_pass_intercepted": {str(k): int(v) for k, v in self.team_pass_intercepted.items()},
            "player_pass_attempts": {str(k): int(v) for k, v in self.player_pass_attempts.items()},
            "player_pass_completed": {str(k): int(v) for k, v in self.player_pass_completed.items()},
        }

