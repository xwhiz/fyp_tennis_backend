from math import ceil
import time
from collections import defaultdict

import cv2
import numpy as np
from typing import Literal
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from core.get_direction_change_indices import get_direction_change_indices
from core.utils import (
    get_court_img,
    perspective_transform_point,
    scene_detect,
)
from db.utils import (
    save_ball_track_in_db,
    save_bounces_in_db,
    save_direction_change_indices_in_db,
    save_speed_in_db,
    save_thumbnail_in_db,
    save_video_paths_in_db,
    update_task_status,
)
from models.speed_at import SpeedAt
from core.court_reference import CourtReference


court_ref = CourtReference()
ref_top_court = court_ref.get_court_mask(2)
ref_bottom_court = court_ref.get_court_mask(1)


def get_detections_from_frames(app, frames):
    batch_ball_track = app.ball_detector.infer_model(frames)
    batch_homography_matrices, batch_kps_court = app.court_detector.infer_model(frames)
    batch_players_top_unfiltered, batch_players_bottom_unfiltered = (
        app.person_detector.track_players(
            frames, batch_homography_matrices, filter_players=False
        )
    )
    batch_players_top = []
    batch_players_bottom = []
    batch_matrix = batch_homography_matrices[0]
    for i in range(len(batch_players_top_unfiltered)):
        top_player, bottom_player = app.person_detector.filter_players(
            batch_players_top_unfiltered[i],
            batch_players_bottom_unfiltered[i],
            batch_matrix,
        )
        if len(top_player) > 0:
            batch_players_top.append(top_player[0])
        else:
            batch_players_top.append(None)
        if len(bottom_player) > 0:
            batch_players_bottom.append(bottom_player[0])
        else:
            batch_players_bottom.append(None)

    return (
        batch_ball_track,
        batch_homography_matrices,
        batch_kps_court,
        batch_players_top,
        batch_players_bottom,
    )


def get_detections_from_video(
    app,
    task_id: int,
    video_path: str,
):
    ball_track = []
    homography_matrices = []
    kps_court = []
    player_top = []
    player_bottom = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch_size = fps

    print(f"[INFO]: number of batches: {ceil(total_frames / batch_size)}")
    frames = []
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if len(frames) == batch_size:
            (
                batch_ball_track,
                batch_homography_matrices,
                batch_kps_court,
                batch_players_top,
                batch_players_bottom,
            ) = get_detections_from_frames(app, frames)
            ball_track.extend(batch_ball_track)
            homography_matrices.extend(batch_homography_matrices)
            kps_court.extend(batch_kps_court)
            player_top.extend(batch_players_top)
            player_bottom.extend(batch_players_bottom)
            frames = []

    if frames:
        (
            batch_ball_track,
            batch_homography_matrices,
            batch_kps_court,
            batch_players_top,
            batch_players_bottom,
        ) = get_detections_from_frames(app, frames)

        ball_track.extend(batch_ball_track[: len(frames)])
        homography_matrices.extend(batch_homography_matrices[: len(frames)])
        kps_court.extend(batch_kps_court[: len(frames)])
        player_top.extend(batch_players_top[: len(frames)])
        player_bottom.extend(batch_players_bottom[: len(frames)])

    cap.release()

    print(f"[INFO]: {len(ball_track)} ball track points detected")
    print(f"[INFO]: {len(homography_matrices)} homography matrices detected")
    print(f"[INFO]: {len(kps_court)} kps court detected")

    update_task_status(task_id, "processing", 3, "Detecting bounces")
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = app.bounce_detector.predict(x_ball, y_ball)

    return (
        ball_track,
        bounces,
        homography_matrices,
        kps_court,
        player_top,
        player_bottom,
    )


def get_sources_from_source_indices(transformed_track, source_indices):
    sources = []
    indices = []
    for index, source in source_indices:
        if source[0] is not None:
            sources.append(source)
            indices.append(index)
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
        if source[0] is None:
            continue
        sources.append(source)
        indices.append(index)

    return sources, indices


def get_shot_type(
    sources, destination, player_top, player_bottom
) -> Literal["forehand", "backhand", "unknown"]:
    net_y = court_ref.net[0][1]

    src_in_top_court = sum([s[1] < net_y for s in sources]) > len(sources) / 2
    src_in_bottom_court = sum([s[1] > net_y for s in sources]) > len(sources) / 2

    dst_in_top_court = destination[1] < net_y
    dst_in_bottom_court = destination[1] > net_y

    if (
        src_in_top_court
        and dst_in_top_court
        or src_in_bottom_court
        and dst_in_bottom_court
    ):
        return "unknown"

    # ASSUME: both players are right handed
    if src_in_top_court and dst_in_bottom_court:  # top court to bottom court
        reference_x = np.mean([player[1][0] for player in player_top])
        source_x = np.mean([source[0] for source in sources])

        if source_x < reference_x:
            return "forehand"
        else:
            return "backhand"
    elif src_in_bottom_court and dst_in_top_court:  # bottom court to top court
        reference_x = np.mean([player[1][0] for player in player_bottom])
        source_x = np.mean([source[0] for source in sources])
        if source_x < reference_x:
            return "backhand"
        else:
            return "forehand"
    else:
        return "unknown"


def process_video(app, video_path: str, task_id: int, name: str):
    print(f"[INFO]: Processing video {task_id}")
    update_task_status(task_id, "processing", 0, "Loading models")

    PIXEL_TO_METER_RATIO = 1 / 101.5
    scenes = scene_detect(video_path)
    print("[INFO]:", scenes)
    max_diff = max(scenes, key=lambda x: x[1] - x[0])
    thumbnail_index = max_diff[0]
    thumbnail_path = f"output/output_{task_id}_thumbnail_{time.time()}.jpg"

    update_task_status(task_id, "processing", 1, "Loading video")

    input_video_capture = cv2.VideoCapture(video_path)
    fps = input_video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(input_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO]: number of frames: {total_frames}")
    # frames = []
    # print("[INFO]: video loaded", cap.isOpened())
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.resize(frame, (1280, 720))
    #     frames.append(frame)

    # cap.release()

    update_task_status(task_id, "processing", 2, "Detecting court and ball track")
    ball_track, bounces, homography_matrices, kps_court, player_top, player_bottom = (
        get_detections_from_video(
            app,
            task_id,
            video_path,
        )
    )

    # ball_track, bounces, homography_matrices, kps_court = process_frames(
    #     app,
    #     task_id,
    #     frames,
    #     fps,
    # )
    update_task_status(task_id, "processing", 4, "Processing ball track")
    transformed_track = [
        perspective_transform_point(point, homography_matrices[i])
        for i, point in enumerate(ball_track)
    ]

    save_ball_track_in_db(task_id, transformed_track)
    save_bounces_in_db(task_id, {index: transformed_track[index] for index in bounces})
    update_task_status(task_id, "processing", 5, "Finding ball hits")
    direction_change_indices = list(
        get_direction_change_indices(ball_track, buffer_length=8)
    )

    # combine indices that have distance less than 10
    indices = []
    for i, ind in enumerate(sorted(direction_change_indices)):
        if i == 0:
            indices.append(ind)
            continue
        if ind - indices[-1] < 6:
            # test_1[-1] = ind
            pass
        else:
            indices.append(ind)

    change_before_bounce = defaultdict(list)
    outer = 0
    for i in bounces:
        for j in indices[outer:]:
            if j < i:
                frame_diff = i - j
                if frame_diff >= 15 and frame_diff <= int(2 * fps):
                    change_before_bounce[i].append((j, transformed_track[j]))
        outer += 1

    direction_change_indices = indices
    save_direction_change_indices_in_db(
        task_id, {index: ball_track[index] for index in direction_change_indices}
    )

    update_task_status(task_id, "processing", 6, "Calculating speed")

    speed_before_bounce = dict()
    for bounce_index, source_indices in change_before_bounce.items():
        destination = transformed_track[bounce_index]
        if destination[0] is None:
            continue
        sources, indices = get_sources_from_source_indices(
            transformed_track, source_indices
        )

        shot_type = get_shot_type(
            sources,
            destination,
            [player_top[index] for index in indices],
            [player_bottom[index] for index in indices],
        )
        pixel_distance = np.mean([euclidean(source, destination) for source in sources])
        meter_distance = pixel_distance * PIXEL_TO_METER_RATIO
        time_difference = (
            bounce_index - max(source_indices, key=lambda x: x[0])[0]
        ) / float(fps)
        speed_before_bounce[bounce_index] = SpeedAt(
            speed=(meter_distance / time_difference) * 3.6,
            time_diff=time_difference,
            timestamp=bounce_index / float(fps),
            distance=meter_distance,
            shot_type=shot_type,
        )

    speed_indices = sorted(speed_before_bounce.keys(), reverse=True)
    save_speed_in_db(task_id, speed_before_bounce)

    minimap = get_court_img()

    output_path = f"output/output_{task_id}_{time.time()}.mp4"
    minimap_path = f"output/output_{task_id}_minimap_{time.time()}.mp4"

    output_video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720),
    )

    # Minimap dimensions
    width_minimap = 166
    height_minimap = 350

    update_task_status(task_id, "processing", 7, "Creating annotated video")

    for i in range(total_frames):
        ret, frame = input_video_capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))

        if i == thumbnail_index:
            cv2.imwrite(thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            save_thumbnail_in_db(task_id, thumbnail_path)

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

        if speed_indices:
            speed_index = speed_indices[-1]
            speed = speed_before_bounce[speed_index].speed
            time_diff = speed_before_bounce[speed_index].time_diff
            distance = speed_before_bounce[speed_index].distance
            shot_type = speed_before_bounce[speed_index].shot_type
            text = f"Speed: {speed:.2f} km/hr Time: {time_diff:.2f} s Distance: {distance:.2f} m Shot Type: {shot_type}"

            frame = cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if i > speed_indices[-1] and len(speed_indices) > 1:
                speed_indices.pop()

        output_video_writer.write(frame)

    output_video_writer.release()

    minimap = get_court_img()
    h, w, _ = minimap.shape
    minimap_out = cv2.VideoWriter(
        minimap_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    update_task_status(task_id, "processing", 8, "Creating minimap video")
    for i in range(len(transformed_track) - 1):
        minimap_copy = minimap.copy()
        if (
            transformed_track[i][0] is not None
            and transformed_track[i + 1][0] is not None
        ):
            color = (0, 255, 0)
            if i in direction_change_indices:
                color = (0, 0, 255)
            minimap_copy = cv2.circle(
                minimap_copy,
                (int(transformed_track[i][0]), int(transformed_track[i][1])),
                radius=0,
                color=color,
                thickness=10,
            )
            minimap_copy = cv2.line(
                minimap_copy,
                (int(transformed_track[i][0]), int(transformed_track[i][1])),
                (int(transformed_track[i + 1][0]), int(transformed_track[i + 1][1])),
                color,
                2,
            )
            minimap_out.write(minimap_copy)

        minimap = minimap_copy.copy()
    minimap_out.release()

    save_video_paths_in_db(task_id, name, output_path, minimap_path)
    update_task_status(task_id, "completed", 9, "Video processed successfully")
