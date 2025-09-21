from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import cv2
from scipy.spatial import distance
import torch
import numpy as np

from ball_detector import BallDetector
from bounce_detector import BounceDetector
from court_detection_net import CourtDetectorNet
from court_reference import CourtReference
from person_detector import PersonDetector
from utils import scene_detect

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SpeedAt:
    speed: float
    time_diff: float
    timestamp: float
    distance: float


def load_ball_detector(path_model: str):
    ball_detector = BallDetector(path_model, device)
    return ball_detector


def load_court_detector(path_model: str):
    court_detector = CourtDetectorNet(path_model, device)
    return court_detector


def load_person_detector():
    person_detector = PersonDetector(device)
    return person_detector


def load_bounce_detector(path_model: str):
    bounce_detector = BounceDetector(path_model)
    return bounce_detector


def process_frames(
    ball_detector: BallDetector,
    court_detector: CourtDetectorNet,
    person_detector: PersonDetector,
    bounce_detector: BounceDetector,
    frames_in_one_second: list,
    fps: int,
):
    ball_track = ball_detector.infer_model(frames_in_one_second)
    homography_matrices, kps_court = court_detector.infer_model(frames_in_one_second)
    # persons_top, persons_bottom = person_detector.track_players(
    #     frames_in_one_second, homography_matrices, filter_players=False
    # )
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    return ball_track, bounces, homography_matrices, kps_court


def get_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0

    p = np.polyfit(range(len(values)), values, 1)
    return p[0]


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
        prev_frames = [
            ball_track[i - buffer_length + j]
            for j in range(0, buffer_length)
            if ball_track[i - buffer_length + j][1] is not None
        ]
        next_frames = [
            ball_track[i + j]
            for j in range(0, buffer_length)
            if ball_track[i + j][1] is not None
        ]

        if len(prev_frames) == 0 or len(next_frames) == 0:
            continue

        y_prev = [float(val[1]) if val[1] > 0 else 0 for val in prev_frames]
        y_next = [float(val[1]) if val[1] > 0 else 0 for val in next_frames]

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


def prespective_transform_point(
    point: tuple[Optional[float], Optional[float]],
    homography_matrix: Optional[np.ndarray],
):
    if point[0] is None or homography_matrix is None:
        return point

    point = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    point = cv2.perspectiveTransform(point, homography_matrix)
    return point[0, 0, 0], point[0, 0, 1]


def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img


def main():
    ball_detector = load_ball_detector("./track_net_weights.pt")
    court_detector = load_court_detector("./model_tennis_court_det.pt")
    person_detector = load_person_detector()
    bounce_detector = load_bounce_detector("./ctb_regr_bounce.cbm")
    print("[INFO]: Loaded models")

    PIXEL_TO_METER_RATIO = 1 / 101.5
    video_path = "./test.mp4"

    scenes = scene_detect(video_path)
    print("[INFO]:", scenes)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # move 2 seconds forward
    cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 2)
    frames = []
    print("[INFO]: video loaded", cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frames.append(frame)

    cap.release()

    ball_track, bounces, homography_matrices, kps_court = process_frames(
        ball_detector,
        court_detector,
        person_detector,
        bounce_detector,
        frames,
        fps,
    )

    transformed_track = [
        prespective_transform_point(point, homography_matrices[i])
        for i, point in enumerate(ball_track)
    ]

    direction_change_indices = get_direction_change_indices(ball_track)

    # combine indices that have distance less than 10
    test_1 = []
    for i, ind in enumerate(sorted(direction_change_indices)):
        if i == 0:
            test_1.append(i)
            continue
        if ind - test_1[-1] < 6:
            # test_1[-1] = ind
            pass
        else:
            test_1.append(ind)

    change_before_bounce = defaultdict(list)
    outer = 0
    for i in bounces:
        for j in test_1[outer:]:
            if j < i:
                frame_diff = i - j
                if frame_diff >= 15 and frame_diff <= int(2 * fps):
                    change_before_bounce[i].append((j, transformed_track[j]))
        outer += 1

    direction_change_indices = test_1

    speed_before_bounce = dict()
    for bounce_index, source_indices in change_before_bounce.items():
        destination = transformed_track[bounce_index]
        sources = []
        for index, source in source_indices:
            if source[0] is not None:
                sources.append(source)
                continue

            # take previous and next not None points, take their average and use it as source
            previous_index = index - 1
            next_index = index + 1
            while previous_index >= 0 and transformed_track[previous_index][0] is None:
                previous_index -= 1
            while (
                next_index < len(transformed_track)
                and transformed_track[next_index][0] is None
            ):
                next_index += 1

            if previous_index < 0 or next_index >= len(transformed_track):
                continue

            source = np.mean(
                [transformed_track[previous_index], transformed_track[next_index]],
                axis=0,
            )
            sources.append(source)

        pixel_distance = np.mean(
            [distance.euclidean(source, destination) for source in sources]
        )
        meter_distance = pixel_distance * PIXEL_TO_METER_RATIO
        time_difference = (
            bounce_index - max(source_indices, key=lambda x: x[0])[0]
        ) / float(fps)
        speed_before_bounce[bounce_index] = SpeedAt(
            speed=meter_distance / time_difference,
            time_diff=time_difference,
            timestamp=bounce_index / float(fps),
            distance=meter_distance,
        )

    speed_indices = sorted(speed_before_bounce.keys(), reverse=True)

    print(speed_before_bounce)

    minimap = get_court_img()

    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720),
    )

    # Minimap dimensions
    width_minimap = 166
    height_minimap = 350

    for i in range(len(frames)):
        frame = frames[i].copy()

        # Draw ball on main frame
        if ball_track[i][0] is not None:
            if i in direction_change_indices:
                frame = cv2.circle(
                    frame,
                    (int(ball_track[i][0]), int(ball_track[i][1])),
                    10,
                    (0, 0, 255),  # Red for direction changes
                    2,
                )
            else:
                frame = cv2.circle(
                    frame,
                    (int(ball_track[i][0]), int(ball_track[i][1])),
                    5,
                    (0, 255, 0),  # Green for normal ball tracking
                    2,
                )

        # Create minimap with ball tracking points
        minimap_frame = minimap.copy()

        # Draw ball tracking points on minimap
        if ball_track[i][0] is not None and homography_matrices[i] is not None:
            ball_point = transformed_track[i]
            minimap_frame = cv2.circle(
                minimap_frame,
                (int(ball_point[0]), int(ball_point[1])),
                radius=0,
                color=(0, 255, 0),  # Green color for ball tracking points
                thickness=30,
            )

        # Draw bounces on minimap as they occur (progressive)
        if (
            i in bounces
            and homography_matrices[i] is not None
            and ball_track[i][0] is not None
        ):
            ball_point = transformed_track[i]
            minimap_frame = cv2.circle(
                minimap_frame,
                (int(ball_point[0]), int(ball_point[1])),
                radius=0,
                color=(0, 255, 255),  # Yellow for bounces
                thickness=50,
            )
            # Update the base minimap to include this bounce permanently
            minimap = cv2.circle(
                minimap,
                (int(ball_point[0]), int(ball_point[1])),
                radius=0,
                color=(0, 255, 255),  # Yellow for bounces
                thickness=50,
            )

        # Resize minimap and add to frame
        minimap_resized = cv2.resize(minimap_frame, (width_minimap, height_minimap))
        height, width = frame.shape[:2]
        frame[
            30 : (30 + height_minimap),
            (width - 30 - width_minimap) : (width - 30),
            :,
        ] = minimap_resized

        frame = cv2.putText(
            frame,
            f"Speed: {speed_before_bounce[speed_indices[-1]].speed:.2f} m/s, Time: {speed_before_bounce[speed_indices[-1]].time_diff:.2f} s, Distance: {speed_before_bounce[speed_indices[-1]].distance:.2f} m",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if i > speed_indices[-1] and len(speed_indices) > 1:
            speed_indices.pop()

        out.write(frame)

    with open("time_speed.json", "w") as f:
        f.write(str(speed_before_bounce))

    out.release()


if __name__ == "__main__":
    main()
