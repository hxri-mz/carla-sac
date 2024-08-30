from enum import IntEnum
import numpy as np

class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    LEFT = 0
    RIGHT = 1
    STRAIGHT = 2

def onehot(x):
    cmd = [0, 0, 0]
    
    if x == 0:
        cmd[0] = 1
    elif x == 1:
        cmd[1] = 1
    elif x == 2:
        cmd[2] = 1

    return cmd

def road_option(current_waypoint, next_waypoint):
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    n -= 180.0
    c -= 180.0
    assert -180 <= n < 180
    assert -180 <= c < 180
    diff_angle = n - c
    diff_angle += -360 if diff_angle > 180 else (360 if diff_angle < -180 else 0)

    angle_threshold = 5

    if abs(diff_angle) < angle_threshold:
        return RoadOption.STRAIGHT
    elif diff_angle < 0.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT