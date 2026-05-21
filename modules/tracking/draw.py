"""
Per-frame overlay drawing shared between the streaming pipeline and the
batch render pass. Mirrors the inline drawing block in
render_video_from_tracks (player ellipses → referees → ball marker).
"""
import cv2

from ..annotations import Annotations


_REFEREE_COLOR = (0, 165, 255)
_OWNER_COLOR = (0, 255, 0)
_FALLBACK_PLAYER_COLOR = (255, 0, 255)
_BALL_COLOR_NO_OWNER = (0, 255, 255)
_BALL_COLOR_OWNED = (0, 255, 0)


def draw_frame_overlays(
    frame,
    player_tracks: dict,
    referee_tracks: dict,
    ball_tracks: dict,
    ball_owner_id: int,
    annotations: Annotations,
    *,
    team_assign_debug: bool = False,
):
    """
    Draw player ellipses (highlighting the ball owner), referees, and the
    ball marker. Caller passes a frame they own (no copy here — copy upstream
    if the source frame must be preserved).
    """
    for player_id, player in player_tracks.items():
        color = _OWNER_COLOR if player_id == ball_owner_id else player.get(
            "team_color", _FALLBACK_PLAYER_COLOR
        )
        frame = annotations.draw_player_ellipse(frame, player["bbox"], color, player_id)

        if team_assign_debug:
            roi = player.get("team_roi_bbox")
            if roi is not None:
                x1, y1, x2, y2 = map(int, roi)
                team_color = player.get("team_color", (128, 128, 128))
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 1)
                team_id = player.get("team_id")
                label = "UNK" if team_id is None else f"T{team_id}"
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, team_color, 1,
                )

    for _, referee in referee_tracks.items():
        frame = annotations.draw_player_ellipse(frame, referee["bbox"], _REFEREE_COLOR, None)

    for _, ball in ball_tracks.items():
        bcolor = _BALL_COLOR_OWNED if ball_owner_id != -1 else _BALL_COLOR_NO_OWNER
        frame = annotations.draw_ball_marker(frame, ball["bbox"], color=bcolor)

    return frame
