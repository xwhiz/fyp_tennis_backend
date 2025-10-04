import asyncio
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

import cv2
import numpy as np
import torch
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.spatial.distance import euclidean
from sqlmodel import Field, Session, SQLModel, create_engine, select

from core.ball_detector import BallDetector
from core.bounce_detector import BounceDetector
from core.court_detection_net import CourtDetectorNet
from core.get_direction_change_indices import get_direction_change_indices
from core.person_detector import PersonDetector
from core.stream_infer import (
    court_detector_stream_infer,
    get_ball_track_and_bounces_stream_infer,
)
from core.utils import get_court_img, perspective_transform_point, scene_detect
from db.engine import Engine
from db.utils import (
    save_ball_track_in_db,
    save_direction_change_indices_in_db,
    save_speed_in_db,
    save_video_paths_in_db,
)
from models.background_task_model import BackgroundTask
from models.ball_track_model import BallTrackModel
from models.bounces_model import BouncesModel
from models.direction_change_indices_model import DirectionChangeIndicesModel
from models.process_video_response import ProcessVideoResponse
from db.utils import create_all, save_bounces_in_db, update_task_status, SessionDep
from models.speed_at import SpeedAt
from models.speed_model import SpeedModel
from models.video_paths_model import VideoPathsModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_all()

    # Load models
    app.ball_detector = BallDetector("./track_net_weights.pt", device)
    app.court_detector = CourtDetectorNet("./model_tennis_court_det.pt", device)
    app.person_detector = PersonDetector(device)
    app.bounce_detector = BounceDetector("./ctb_regr_bounce.cbm")
    print("All models loaded successfully")

    yield

    # Shutdown
    executor.shutdown(wait=True)
    app.ball_detector = None
    app.court_detector = None
    app.person_detector = None
    app.bounce_detector = None


app = FastAPI(lifespan=lifespan)

os.makedirs("output", exist_ok=True)

# static files
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def process_video_background(task_id: int, video_path: str, name: str):
    """Background function to process video"""
    try:
        update_task_status(task_id, "processing", 0, "Processing video")

        process_request(video_path, task_id, name)

        update_task_status(task_id, "completed", 9, "Video processed successfully")

    except Exception as e:
        print(f"Error processing video {task_id}: {str(e)}")
        update_task_status(task_id, "failed", 0, "Error processing video")


def process_frames(
    task_id: int,
    frames: list,
    fps: int,
):
    update_task_status(task_id, "processing", 2, "Detecting ball")
    ball_track = app.ball_detector.infer_model(frames)
    update_task_status(task_id, "processing", 3, "Detecting court")
    homography_matrices, kps_court = app.court_detector.infer_model(frames)
    # persons_top, persons_bottom = person_detector.track_players(
    #     frames_in_one_second, homography_matrices, filter_players=False
    update_task_status(task_id, "processing", 4, "Detecting bounces")
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = app.bounce_detector.predict(x_ball, y_ball)

    return ball_track, bounces, homography_matrices, kps_court


def process_request(video_path: str, task_id: int, name: str):
    update_task_status(task_id, "processing", 0, "Loading models")

    PIXEL_TO_METER_RATIO = 1 / 101.5
    # scenes = scene_detect(video_path)
    # print("[INFO]:", scenes)

    update_task_status(task_id, "processing", 1, "Loading video")

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
        task_id,
        frames,
        fps,
    )
    transformed_track = [
        perspective_transform_point(point, homography_matrices[i])
        for i, point in enumerate(ball_track)
    ]

    save_ball_track_in_db(task_id, transformed_track)
    save_bounces_in_db(task_id, bounces)

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
    save_direction_change_indices_in_db(task_id, direction_change_indices)

    update_task_status(task_id, "processing", 6, "Calculating speed")

    speed_before_bounce = dict()
    for bounce_index, source_indices in change_before_bounce.items():
        destination = transformed_track[bounce_index]
        if destination[0] is None:
            continue
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
            if source[0] is None:
                continue
            sources.append(source)

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
        )

    speed_indices = sorted(speed_before_bounce.keys(), reverse=True)
    save_speed_in_db(task_id, speed_before_bounce)

    minimap = get_court_img()

    output_path = f"output/output_{task_id}_{time.time()}.mp4"
    minimap_path = f"output/output_{task_id}_minimap_{time.time()}.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720),
    )

    # Minimap dimensions
    width_minimap = 166
    height_minimap = 350

    update_task_status(task_id, "processing", 7, "Creating annotated video")

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

        if speed_indices:
            speed_index = speed_indices[-1]
            speed = speed_before_bounce[speed_index].speed
            time_diff = speed_before_bounce[speed_index].time_diff
            distance = speed_before_bounce[speed_index].distance
            text = f"Speed: {speed:.2f} km/hr Time: {time_diff:.2f} s Distance: {distance:.2f} m"

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

        out.write(frame)

    out.release()

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


@app.get("/all_tasks")
def get_all_tasks(session: SessionDep):
    statement = select(BackgroundTask)
    tasks = session.exec(statement).all()
    return {
        "success": True,
        "data": tasks,
    }


@app.get("/task_progress/{process_id}")
def get_task_progress(process_id: int, session: SessionDep, is_api: bool = True):
    # Query the database for the process
    statement = select(BackgroundTask).where(BackgroundTask.id == process_id)
    task = session.exec(statement).first()

    if not task:
        raise HTTPException(status_code=404, detail="Process not found")

    if is_api:
        return {
            "success": True,
            "data": {
                "process_id": task.id,
                "progress": task.progress,
                "total_steps": task.total_steps,
                "status": task.status,
                "description": task.description,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            },
        }
    else:
        return task


@app.post("/process_video")
async def process_video(
    name: str = Form(...), video_file: UploadFile = File(...)
) -> ProcessVideoResponse:
    # Validate file type
    if not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Create a unique filename to avoid conflicts
    import uuid

    file_extension = (
        video_file.filename.split(".")[-1] if "." in video_file.filename else "mp4"
    )
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"./uploads/{unique_filename}"

    # Create uploads directory if it doesn't exist
    import os

    os.makedirs("./uploads", exist_ok=True)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        content = await video_file.read()
        buffer.write(content)

    # Generate a process ID
    process_id = str(uuid.uuid4())

    # Store the process in database
    with Session(Engine.instance()) as session:
        task = BackgroundTask(
            id=None,  # Let SQLModel auto-generate
            progress=0,
            status="pending",
            video_path=file_path,
            description=f"Processing video: {name}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        process_id = str(task.id)

    # Start background processing
    executor.submit(process_video_background, int(process_id), file_path, name)

    return {
        "success": True,
        "message": "Video uploaded and queued for processing",
        "data": {
            "process_id": process_id,
            "filename": video_file.filename,
            "name": name,
            "file_path": file_path,
        },
    }


@app.get("/get_video_paths/{task_id}")
async def get_video_paths(task_id: int, session: SessionDep, is_api: bool = True):
    statement = select(VideoPathsModel).where(VideoPathsModel.task_id == task_id)
    video_paths = session.exec(statement).first()
    if is_api:
        return {
            "success": True,
            "data": video_paths,
        }
    else:
        return video_paths


@app.get("/get_speed_stats/{task_id}")
async def get_speed_stats(task_id: int, session: SessionDep, is_api: bool = True):
    statement = select(SpeedModel).where(SpeedModel.task_id == task_id)
    speed_stats = session.exec(statement).first()
    speed_stats.speed = json.loads(speed_stats.speed)

    if is_api:
        return {
            "success": True,
            "data": speed_stats,
        }
    else:
        return speed_stats


@app.get("/get_ball_track/{task_id}")
async def get_ball_track(task_id: int, session: SessionDep, is_api: bool = True):
    statement = select(BallTrackModel).where(BallTrackModel.task_id == task_id)
    ball_track = session.exec(statement).first()
    ball_track.ball_track = json.loads(ball_track.ball_track)
    if is_api:
        return {
            "success": True,
            "data": ball_track,
        }
    else:
        return ball_track


@app.get("/get_bounces/{task_id}")
async def get_bounces(task_id: int, session: SessionDep, is_api: bool = True):
    statement = select(BouncesModel).where(BouncesModel.task_id == task_id)
    bounces = session.exec(statement).first()
    bounces.bounces = json.loads(bounces.bounces)
    if is_api:
        return {
            "success": True,
            "data": bounces,
        }
    else:
        return bounces


@app.get("/get_direction_change_indices/{task_id}")
async def get_direction_change_indices_api(
    task_id: int, session: SessionDep, is_api: bool = True
):
    statement = select(DirectionChangeIndicesModel).where(
        DirectionChangeIndicesModel.task_id == task_id
    )
    direction_change_indices = session.exec(statement).first()
    direction_change_indices.direction_change_indices = json.loads(
        direction_change_indices.direction_change_indices
    )
    if is_api:
        return {
            "success": True,
            "data": direction_change_indices,
        }
    else:
        return direction_change_indices


@app.get("/all-stats/{task_id}")
async def get_all_stats(task_id: int, session: SessionDep):
    return {
        "success": True,
        "data": {
            "video_paths": await get_video_paths(task_id, session, is_api=False),
            "speed_stats": await get_speed_stats(task_id, session, is_api=False),
            "ball_track": await get_ball_track(task_id, session, is_api=False),
            "bounces": await get_bounces(task_id, session, is_api=False),
            "direction_change_indices": await get_direction_change_indices_api(
                task_id, session, is_api=False
            ),
        },
        "progress": get_task_progress(task_id, session, is_api=False),
    }


@app.get("/")
def test_hello_world():
    return {
        "success": True,
        "message": "Hello world",
    }


@app.get("/check-health")
def check_health():
    return {
        "success": True,
        "message": "OK",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)
