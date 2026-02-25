from math import ceil
import time
import gc
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from typing import Literal
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from src.core.get_direction_change_indices import get_direction_change_indices
from src.core.utils import (
    get_court_img,
    perspective_transform_point,
    scene_detect,
)
from src.db.utils import (
    save_ball_track_in_db,
    save_bounces_in_db,
    save_direction_change_indices_in_db,
    save_player_positions_in_db,
    save_speed_in_db,
    save_thumbnail_in_db,
    save_video_paths_in_db,
    save_shot_annotation_in_db,
    update_task_status,
)
from src.core.shot_classifier import ShotClassifier
from src.schemas.speed_at import SpeedAt
from src.core.court_reference import CourtReference
from src.config import settings


court_ref = CourtReference()
ref_top_court = court_ref.get_court_mask(2)
ref_bottom_court = court_ref.get_court_mask(1)


def cleanup_memory(device):
    """Explicitly clean up GPU and CPU memory"""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_detections_from_frames(ball_detector, court_detector, person_detector, frames, task_id=None, current_frame=0, total_frames=1):
    # Calculate progress range for this batch (0.2 to 0.8)
    batch_start_progress = 0.2 + (current_frame / total_frames) * 0.6
    batch_end_progress = 0.2 + ((current_frame + len(frames)) / total_frames) * 0.6
    
    # Step 1: Ball detection (takes ~40% of batch time)
    batch_ball_track = ball_detector.infer_model(frames)
    if task_id:
        progress = round(batch_start_progress + (batch_end_progress - batch_start_progress) * 0.4, 3)
        update_task_status(task_id, "processing", progress, "Detecting ball track")
    
    # Step 2: Court detection (takes ~30% of batch time)
    batch_homography_matrices, batch_kps_court = court_detector.infer_model(frames)
    if task_id:
        progress = round(batch_start_progress + (batch_end_progress - batch_start_progress) * 0.7, 3)
        update_task_status(task_id, "processing", progress, "Detecting court")
    
    # Step 3: Person detection (takes ~20% of batch time)
    batch_players_top_unfiltered, batch_players_bottom_unfiltered = (
        person_detector.track_players(
            frames, batch_homography_matrices, filter_players=False
        )
    )
    if task_id:
        progress = round(batch_start_progress + (batch_end_progress - batch_start_progress) * 0.9, 3)
        update_task_status(task_id, "processing", progress, "Detecting players")
    batch_players_top = []
    batch_players_bottom = []
    for i in range(len(batch_players_top_unfiltered)):
        # Use the matrix for this specific frame, not always the first one
        batch_matrix = batch_homography_matrices[i]
        
        # Validate matrix before using it
        if batch_matrix is not None and isinstance(batch_matrix, np.ndarray):
            # Check if matrix has correct shape (3x3 for homography)
            if batch_matrix.shape == (3, 3):
                try:
                    top_player, bottom_player = person_detector.filter_players(
                        batch_players_top_unfiltered[i],
                        batch_players_bottom_unfiltered[i],
                        batch_matrix,
                    )
                except cv2.error:
                    # If perspectiveTransform fails, use unfiltered results
                    top_player = batch_players_top_unfiltered[i]
                    bottom_player = batch_players_bottom_unfiltered[i]
            else:
                # Invalid matrix shape, use unfiltered results
                top_player = batch_players_top_unfiltered[i]
                bottom_player = batch_players_bottom_unfiltered[i]
        else:
            # Matrix is None or invalid, use unfiltered results
            top_player = batch_players_top_unfiltered[i]
            bottom_player = batch_players_bottom_unfiltered[i]
        
        if len(top_player) > 0:
            batch_players_top.append(top_player[0])
        else:
            batch_players_top.append(None)
        if len(bottom_player) > 0:
            batch_players_bottom.append(bottom_player[0])
        else:
            batch_players_bottom.append(None)
    
    # Clean up intermediate results
    del batch_players_top_unfiltered, batch_players_bottom_unfiltered
    
    # Step 4: Filtering complete (100% of batch)
    if task_id:
        progress = round(batch_end_progress, 3)
        update_task_status(task_id, "processing", progress, "Processing detections")

    return (
        batch_ball_track,
        batch_homography_matrices,
        batch_kps_court,
        batch_players_top,
        batch_players_bottom,
    )


def get_detections_from_video(
    ball_detector,
    court_detector,
    person_detector,
    bounce_detector,
    task_id: int,
    video_path: str,
    cap=None,
):
    ball_track = []
    homography_matrices = []
    kps_court = []
    player_top = []
    player_bottom = []
    
    # Reuse provided video capture or create new one
    should_release_cap = False
    if cap is None:
        cap = cv2.VideoCapture(video_path)
        should_release_cap = True
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adaptive batch sizing for large videos
    min_batch_size = settings.video_batch_size
    batch_size = max(fps, min_batch_size)
    if total_frames > 50000:
        batch_size = max(int(fps // 2), min_batch_size)  # Ensure minimum batch size even for large videos
        print(f"[INFO]: Large video detected ({total_frames} frames), using batch size: {batch_size}")

    print(f"[INFO]: number of batches: {ceil(total_frames / batch_size)}")
    
    # Get device from ball_detector for memory cleanup
    device = getattr(ball_detector, 'device', 'cpu')
    
    frames = []
    batch_count = 0
    
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
            ) = get_detections_from_frames(
                ball_detector, court_detector, person_detector, frames,
                task_id=task_id, current_frame=i - batch_size + 1, total_frames=total_frames
            )
            ball_track.extend(batch_ball_track)
            homography_matrices.extend(batch_homography_matrices)
            kps_court.extend(batch_kps_court)
            player_top.extend(batch_players_top)
            player_bottom.extend(batch_players_bottom)
            
            # Clean up batch data
            del frames
            del batch_ball_track, batch_homography_matrices, batch_kps_court
            del batch_players_top, batch_players_bottom
            
            # Memory cleanup every batch for large videos, every 5 batches for smaller ones
            batch_count += 1
            if total_frames > 50000 or batch_count % 5 == 0:
                cleanup_memory(device)
            
            frames = []

    if frames:
        (
            batch_ball_track,
            batch_homography_matrices,
            batch_kps_court,
            batch_players_top,
            batch_players_bottom,
        ) = get_detections_from_frames(
            ball_detector, court_detector, person_detector, frames,
            task_id=task_id, current_frame=total_frames - len(frames), total_frames=total_frames
        )

        ball_track.extend(batch_ball_track[: len(frames)])
        homography_matrices.extend(batch_homography_matrices[: len(frames)])
        kps_court.extend(batch_kps_court[: len(frames)])
        player_top.extend(batch_players_top[: len(frames)])
        player_bottom.extend(batch_players_bottom[: len(frames)])
        
        # Clean up remaining batch data
        del frames
        del batch_ball_track, batch_homography_matrices, batch_kps_court
        del batch_players_top, batch_players_bottom
        cleanup_memory(device)

    # Only release if we created the capture object
    if should_release_cap:
        cap.release()

    print(f"[INFO]: {len(ball_track)} ball track points detected")
    print(f"[INFO]: {len(homography_matrices)} homography matrices detected")
    print(f"[INFO]: {len(kps_court)} kps court detected")

    # Update progress to 0.8 after frame detection is complete
    update_task_status(task_id, "processing", round(0.8, 3), "Detecting bounces")
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)
    
    # Clean up temporary lists used for bounce detection
    del x_ball, y_ball
    cleanup_memory(device)

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
    sources, destination, player_top, player_bottom, classifier: ShotClassifier = None
) -> Literal["forehand", "backhand", "unknown"]:
    # Try to use ML model if available
    if classifier is not None and classifier.is_trained():
        try:
            # Get player positions for feature extraction
            player_top_pos = player_top[0] if len(player_top) > 0 and player_top[0] is not None else None
            player_bottom_pos = player_bottom[0] if len(player_bottom) > 0 and player_bottom[0] is not None else None
            
            # Prepare position data for feature extraction
            player_pos_top = None
            player_pos_bottom = None
            
            if player_top_pos is not None:
                bbox = player_top_pos[0] if isinstance(player_top_pos, tuple) else None
                if bbox is not None:
                    player_pos_top = {"bbox": bbox.tolist() if isinstance(bbox, np.ndarray) else bbox}
            
            if player_bottom_pos is not None:
                bbox = player_bottom_pos[0] if isinstance(player_bottom_pos, tuple) else None
                if bbox is not None:
                    player_pos_bottom = {"bbox": bbox.tolist() if isinstance(bbox, np.ndarray) else bbox}
            
            # Prepare ball position
            ball_pos = {"x": float(destination[0]), "y": float(destination[1])} if destination[0] is not None else None
            
            # Extract features and predict
            net_y = court_ref.net[0][1]
            features = classifier.extract_features(
                player_pos_top, player_pos_bottom, ball_pos, net_y
            )
            prediction = classifier.predict(features)
            return prediction
        except Exception as e:
            print(f"[SHOT TYPE]: Error using ML model, falling back to rule-based: {str(e)}")
    
    # Fallback to rule-based logic
    # Filter out None values from player lists
    player_top_filtered = [player for player in player_top if player is not None]
    player_bottom_filtered = [player for player in player_bottom if player is not None]

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
        # Return "unknown" if no valid top players available
        if len(player_top_filtered) == 0:
            return "unknown"
        reference_x = np.mean([player[1][0] for player in player_top_filtered])
        source_x = np.mean([source[0] for source in sources])

        if source_x < reference_x:
            return "forehand"
        else:
            return "backhand"
    elif src_in_bottom_court and dst_in_top_court:  # bottom court to top court
        # Return "unknown" if no valid bottom players available
        if len(player_bottom_filtered) == 0:
            return "unknown"
        reference_x = np.mean([player[1][0] for player in player_bottom_filtered])
        source_x = np.mean([source[0] for source in sources])
        if source_x < reference_x:
            return "backhand"
        else:
            return "forehand"
    else:
        return "unknown"


def process_video(
    ball_detector,
    court_detector,
    person_detector,
    bounce_detector,
    video_path: str,
    task_id: int,
    name: str,
):
    print(f"[INFO]: Processing video {task_id}")
    update_task_status(task_id, "processing", round(0.0, 3), "Loading models")

    PIXEL_TO_METER_RATIO = 1 / 101.5
    scenes = scene_detect(video_path)
    print("[INFO]:", scenes)
    max_diff = max(scenes, key=lambda x: x[1] - x[0])
    thumbnail_index = max_diff[0]
    thumbnail_path = f"output/output_{task_id}_thumbnail_{time.time()}.jpg"

    update_task_status(task_id, "processing", round(0.1, 3), "Loading video")

    input_video_capture = cv2.VideoCapture(video_path)
    fps = input_video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(input_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO]: number of frames: {total_frames}")
    
    # Get device for memory cleanup
    device = getattr(ball_detector, 'device', 'cpu')

    update_task_status(task_id, "processing", round(0.2, 3), "Detecting court and ball track")
    # Reuse video capture object to avoid reopening
    ball_track, bounces, homography_matrices, kps_court, player_top, player_bottom = (
        get_detections_from_video(
            ball_detector,
            court_detector,
            person_detector,
            bounce_detector,
            task_id,
            video_path,
            cap=input_video_capture,  # Reuse the capture object
        )
    )
    
    # Memory cleanup after detection
    cleanup_memory(device)

    # ball_track, bounces, homography_matrices, kps_court = process_frames(
    #     app,
    #     task_id,
    #     frames,
    #     fps,
    # )
    update_task_status(task_id, "processing", round(0.85, 3), "Processing ball track")
    transformed_track = [
        perspective_transform_point(point, homography_matrices[i])
        for i, point in enumerate(ball_track)
    ]
    
    # Memory cleanup after transformation
    cleanup_memory(device)

    save_ball_track_in_db(task_id, transformed_track)
    save_bounces_in_db(task_id, {index: transformed_track[index] for index in bounces})
    save_player_positions_in_db(task_id, player_top, player_bottom)
    
    # Clean up after saving to DB
    cleanup_memory(device)
    update_task_status(task_id, "processing", round(0.87, 3), "Finding ball hits")
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
    
    # Memory cleanup after direction change detection
    cleanup_memory(device)

    # Update progress to 0.9 after finding ball hits is complete
    update_task_status(task_id, "processing", round(0.9, 3), "Calculating speed")

    # Initialize shot classifier
    shot_classifier = ShotClassifier()
    
    # Create directory for shot images
    shot_images_dir = f"./output/shot_images"
    os.makedirs(shot_images_dir, exist_ok=True)

    # Capture shot annotations at direction change frames (where shots actually happen)
    # Create a mapping from direction change to bounce for shot type calculation
    direction_change_to_bounce = {}
    for bounce_index, source_indices in change_before_bounce.items():
        for source_index, _ in source_indices:
            if source_index in direction_change_indices:
                direction_change_to_bounce[source_index] = bounce_index

    # Process each direction change to capture shot annotations
    for direction_change_index in direction_change_indices:
        # Only process if this direction change leads to a bounce
        if direction_change_index not in direction_change_to_bounce:
            continue
        
        bounce_index = direction_change_to_bounce[direction_change_index]
        destination = transformed_track[bounce_index]
        if destination[0] is None:
            continue
        
        # Get source indices for this bounce to calculate shot type
        source_indices = change_before_bounce.get(bounce_index, [])
        if not source_indices:
            continue
        
        sources, indices = get_sources_from_source_indices(
            transformed_track, source_indices
        )
        
        # Get player positions for shot type calculation (use bounce frame players)
        player_top_at_bounce = [player_top[index] for index in indices]
        player_bottom_at_bounce = [player_bottom[index] for index in indices]
        
        shot_type = get_shot_type(
            sources,
            destination,
            player_top_at_bounce,
            player_bottom_at_bounce,
            classifier=shot_classifier,
        )
        
        # Capture shot annotation at direction change frame (when shot happens)
        try:
            # Get frame at direction_change_index (not bounce_index)
            input_video_capture.set(cv2.CAP_PROP_POS_FRAMES, direction_change_index)
            ret, frame = input_video_capture.read()
            
            if ret and frame is not None and frame.size > 0:
                # Get homography matrix for this frame to re-detect all players
                if direction_change_index < len(homography_matrices) and homography_matrices[direction_change_index] is not None:
                    inv_matrix = homography_matrices[direction_change_index]
                    
                    # Re-detect ALL players at this frame (without filtering)
                    all_top_players, all_bottom_players = person_detector.detect_top_and_bottom_players(
                        frame, inv_matrix, filter_players=False
                    )
                    
                    # Prepare data structures for all players
                    player_image_paths = {"top": [], "bottom": []}
                    player_positions_top = {"players": []}
                    player_positions_bottom = {"players": []}
                    
                    # Save images and positions for ALL top players
                    for idx, player in enumerate(all_top_players):
                        if player is not None:
                            bbox = player[0] if isinstance(player, tuple) else None
                            point = player[1] if isinstance(player, tuple) and len(player) > 1 else None
                            
                            if bbox is not None:
                                # Store position
                                bbox_list = bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)
                                point_list = point if point is None else (point.tolist() if isinstance(point, np.ndarray) else list(point))
                                player_positions_top["players"].append({
                                    "bbox": bbox_list,
                                    "point": point_list
                                })
                                
                                # Crop and save image
                                x1, y1, x2, y2 = map(int, bbox[:4])
                                h, w = frame.shape[:2]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    player_crop = frame[y1:y2, x1:x2]
                                    
                                    # Check if crop is not empty
                                    if player_crop.size > 0:
                                        image_filename = f"shot_{task_id}_{direction_change_index}_top_{idx}_{int(time.time() * 1000)}.jpg"
                                        image_path = os.path.join(shot_images_dir, image_filename)
                                        cv2.imwrite(image_path, player_crop)
                                        player_image_paths["top"].append(f"shot_images/{image_filename}")
                    
                    # Save images and positions for ALL bottom players
                    for idx, player in enumerate(all_bottom_players):
                        if player is not None:
                            bbox = player[0] if isinstance(player, tuple) else None
                            point = player[1] if isinstance(player, tuple) and len(player) > 1 else None
                            
                            if bbox is not None:
                                # Store position
                                bbox_list = bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)
                                point_list = point if point is None else (point.tolist() if isinstance(point, np.ndarray) else list(point))
                                player_positions_bottom["players"].append({
                                    "bbox": bbox_list,
                                    "point": point_list
                                })
                                
                                # Crop and save image
                                x1, y1, x2, y2 = map(int, bbox[:4])
                                h, w = frame.shape[:2]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    player_crop = frame[y1:y2, x1:x2]
                                    
                                    # Check if crop is not empty
                                    if player_crop.size > 0:
                                        image_filename = f"shot_{task_id}_{direction_change_index}_bottom_{idx}_{int(time.time() * 1000)}.jpg"
                                        image_path = os.path.join(shot_images_dir, image_filename)
                                        cv2.imwrite(image_path, player_crop)
                                        player_image_paths["bottom"].append(f"shot_images/{image_filename}")
                    
                    # Prepare ball position at direction change
                    ball_at_direction_change = transformed_track[direction_change_index]
                    ball_pos = None
                    if ball_at_direction_change[0] is not None:
                        ball_pos = {"x": float(ball_at_direction_change[0]), "y": float(ball_at_direction_change[1])}
                    
                    # Only save if we captured at least one player image
                    if player_image_paths["top"] or player_image_paths["bottom"]:
                        save_shot_annotation_in_db(
                            task_id=task_id,
                            frame_index=direction_change_index,
                            player_position_top=player_positions_top if player_positions_top["players"] else None,
                            player_position_bottom=player_positions_bottom if player_positions_bottom["players"] else None,
                            ball_position=ball_pos,
                            player_image_paths=player_image_paths if (player_image_paths["top"] or player_image_paths["bottom"]) else None,
                            predicted_shot_type=shot_type,
                        )
        except Exception as e:
            print(f"[SHOT ANNOTATION]: Error saving shot annotation for direction change frame {direction_change_index}: {str(e)}")

    speed_before_bounce = dict()
    for bounce_index, source_indices in change_before_bounce.items():
        destination = transformed_track[bounce_index]
        if destination[0] is None:
            continue
        sources, indices = get_sources_from_source_indices(
            transformed_track, source_indices
        )

        # Get player positions at the bounce frame
        player_top_at_bounce = [player_top[index] for index in indices]
        player_bottom_at_bounce = [player_bottom[index] for index in indices]
        
        shot_type = get_shot_type(
            sources,
            destination,
            player_top_at_bounce,
            player_bottom_at_bounce,
            classifier=shot_classifier,
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
    
    # Memory cleanup after speed calculations
    cleanup_memory(device)

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

    update_task_status(task_id, "processing", round(0.95, 3), "Creating annotated video")
    
    # Reset video capture to beginning for output generation
    input_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(total_frames):
        ret, frame = input_video_capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        
        # Periodic memory cleanup during video generation
        if i > 0 and i % 1000 == 0:
            cleanup_memory(device)

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

        # Draw player positions on minimap
        inv_mat = homography_matrices[i]
        if inv_mat is not None:
            for player, color in [
                (player_top[i], (0, 0, 255)),
                (player_bottom[i], (255, 0, 0)),
            ]:
                if player is not None:
                    foot_point = np.array(player[1], dtype=np.float32).reshape(1, 1, 2)
                    court_point = cv2.perspectiveTransform(foot_point, inv_mat)
                    minimap_frame = cv2.circle(
                        minimap_frame,
                        (int(court_point[0, 0, 0]), int(court_point[0, 0, 1])),
                        radius=0,
                        color=color,
                        thickness=60,
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
    
    # Memory cleanup after main video generation
    cleanup_memory(device)

    minimap = get_court_img()
    h, w, _ = minimap.shape
    minimap_out = cv2.VideoWriter(
        minimap_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    update_task_status(task_id, "processing", round(0.98, 3), "Creating minimap video")
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
        
        # Periodic memory cleanup during minimap generation
        if i > 0 and i % 1000 == 0:
            cleanup_memory(device)
    
    minimap_out.release()
    
    # Final memory cleanup
    cleanup_memory(device)
    
    # Release video capture
    input_video_capture.release()

    save_video_paths_in_db(task_id, name, output_path, minimap_path)
    update_task_status(task_id, "completed", round(1.0, 3), "Video processed successfully")
