from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple


class SpatialActivityAnalysis:
    """
    Aggregates homography-transformed player positions into:
    - per-team and all-player heatmap grids
    - per-team and all-player zone occupancy counts
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        heatmap_bins_x: int = 24,
        heatmap_bins_y: int = 16,
        zone_cols: int = 6,
        zone_rows: int = 3,
    ):
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.heatmap_bins_x = max(2, int(heatmap_bins_x))
        self.heatmap_bins_y = max(2, int(heatmap_bins_y))
        self.zone_cols = max(1, int(zone_cols))
        self.zone_rows = max(1, int(zone_rows))

        self._groups = ("all_players", "team_1", "team_2")
        self._heatmaps = {
            g: [[0 for _ in range(self.heatmap_bins_x)] for _ in range(self.heatmap_bins_y)]
            for g in self._groups
        }
        self._zones = {
            g: [[0 for _ in range(self.zone_cols)] for _ in range(self.zone_rows)]
            for g in self._groups
        }
        self._samples = {g: 0 for g in self._groups}

    def _group_keys_for_team(self, team_id: Optional[int]) -> Iterable[str]:
        if team_id == 1:
            return ("all_players", "team_1")
        if team_id == 2:
            return ("all_players", "team_2")
        return ("all_players",)

    def _clamp_point(self, x: float, y: float) -> Tuple[float, float]:
        x = max(0.0, min(self.pitch_length, float(x)))
        y = max(0.0, min(self.pitch_width, float(y)))
        return x, y

    def _heatmap_cell(self, x: float, y: float) -> Tuple[int, int]:
        x_norm = 0.0 if self.pitch_length <= 0 else x / self.pitch_length
        y_norm = 0.0 if self.pitch_width <= 0 else y / self.pitch_width
        col = min(self.heatmap_bins_x - 1, max(0, int(x_norm * self.heatmap_bins_x)))
        row = min(self.heatmap_bins_y - 1, max(0, int(y_norm * self.heatmap_bins_y)))
        return row, col

    def _zone_cell(self, x: float, y: float) -> Tuple[int, int]:
        x_norm = 0.0 if self.pitch_length <= 0 else x / self.pitch_length
        y_norm = 0.0 if self.pitch_width <= 0 else y / self.pitch_width
        col = min(self.zone_cols - 1, max(0, int(x_norm * self.zone_cols)))
        row = min(self.zone_rows - 1, max(0, int(y_norm * self.zone_rows)))
        return row, col

    def update(self, player_tracks: Dict[int, dict]) -> None:
        for player in player_tracks.values():
            if not isinstance(player, dict):
                continue
            pos = player.get("position_transformed")
            if not pos or len(pos) < 2:
                continue
            x, y = self._clamp_point(float(pos[0]), float(pos[1]))
            team_id = player.get("team_id")
            for group in self._group_keys_for_team(team_id):
                h_row, h_col = self._heatmap_cell(x, y)
                z_row, z_col = self._zone_cell(x, y)
                self._heatmaps[group][h_row][h_col] += 1
                self._zones[group][z_row][z_col] += 1
                self._samples[group] += 1

    def finalize(self) -> dict:
        return {
            "enabled": True,
            "pitch": {
                "length_m": self.pitch_length,
                "width_m": self.pitch_width,
            },
            "heatmap_bins": {
                "x": self.heatmap_bins_x,
                "y": self.heatmap_bins_y,
            },
            "zones": {
                "cols": self.zone_cols,
                "rows": self.zone_rows,
            },
            "samples": self._samples,
            "heatmaps": self._heatmaps,
            "zone_activity": self._zones,
        }
