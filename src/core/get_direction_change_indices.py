from typing import Optional
from src.core.utils import get_slope


def get_direction_change_indices(
    ball_track: list[tuple[Optional[float], Optional[float]]],
    buffer_length: int = 15,
    slope_threshold: float = 1e-3,
):
    """
    Analyze `buffer_length` frames before and after the current frame to determine if the ball is changing direction
    """
    direction_change_indices = []
    for i in range(buffer_length, len(ball_track) - buffer_length):
        prev_track = [
            ball_track[i - buffer_length + j]
            for j in range(0, buffer_length)
            if ball_track[i - buffer_length + j][1] is not None
        ]
        next_track = [
            ball_track[i + j]
            for j in range(0, buffer_length)
            if ball_track[i + j][1] is not None
        ]

        if len(prev_track) == 0 or len(next_track) == 0:
            continue

        y_prev = [float(val[1]) if val[1] > 0 else 0 for val in prev_track]
        y_next = [float(val[1]) if val[1] > 0 else 0 for val in next_track]

        slope_prev = get_slope(y_prev)
        slope_next = get_slope(y_next)

        changed = (
            (slope_prev * slope_next) < 0
            and (abs(slope_prev) > slope_threshold)
            and (abs(slope_next) > slope_threshold)
        )
        if changed:
            direction_change_indices.append(i)

    return set(direction_change_indices)
