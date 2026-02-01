import cv2
from helpers import annotation_center, annotation_width

class Annotations:
    @staticmethod
    def draw_player_ellipse(frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = annotation_center(bbox)
        width = annotation_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            track_id_str = str(track_id)

            x1_text = x1_rect + 12

            if track_id_str.isdigit() and int(track_id_str) > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                track_id_str,
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    @staticmethod
    def draw_ball_marker(frame, bbox, color=(0, 255, 255), radius_factor=0.25, track_id=None):
        y_center = int((bbox[1] + bbox[3]) / 2)
        x_center, _ = annotation_center(bbox)
        width = annotation_width(bbox)
        outer_radius = int(int(width * 0.35) * 2.5)

        cv2.circle(
            frame,
            center=(x_center, y_center),
            radius=outer_radius,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame

    @staticmethod
    def draw_team_ball_control(frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame
