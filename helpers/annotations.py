import math

def annotation_width(bounding_box):
    return bounding_box[2] - bounding_box[0]

def annotation_center(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def player_foot_position(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int(y2)

def calculate_distance(player1, player2):
    dx, dy = player1[0] - player2[0], player1[1] - player2[1]
    return math.sqrt(dx ** 2 + dy ** 2)