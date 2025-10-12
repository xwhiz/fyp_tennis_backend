import asyncio
import json
import mimetypes
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

import cv2
from fastapi.responses import StreamingResponse
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
from core.court_reference import CourtReference
from core.event_loop import EventLoop
from core.get_direction_change_indices import get_direction_change_indices
from core.person_detector import PersonDetector
from core.process_video import process_video
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
from models.thumbnail_model import ThumbnailModel
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

    app.event_loop = EventLoop(app)
    executor.submit(app.event_loop.run)

    yield

    # Shutdown
    app.event_loop.stop()
    executor.shutdown(wait=True)
    app.ball_detector = None
    app.court_detector = None
    app.person_detector = None
    app.bounce_detector = None


openapi_tags = [
    {"name": "tasks", "description": "Tasks"},
    {"name": "stats", "description": "Stats"},
    {"name": "misc", "description": "Misc"},
]
app = FastAPI(lifespan=lifespan, openapi_tags=openapi_tags)

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

# static files
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def process_video_background(task_id: int, video_path: str, name: str):
    """Background function to process video"""
    try:
        update_task_status(task_id, "pending", 0, "Waiting in queue to be processed")
        app.event_loop.add_task(
            {
                "id": task_id,
                "video_path": video_path,
                "name": name,
            }
        )

    except Exception as e:
        print(f"Error processing video {task_id}: {str(e)}")
        update_task_status(task_id, "failed", 0, "Error processing video")


@app.get("/all_tasks", tags=["tasks"])
def get_all_tasks(session: SessionDep):
    statement = select(BackgroundTask)
    tasks = session.exec(statement).all()
    return {
        "success": True,
        "data": tasks,
    }


@app.get("/task_progress/{process_id}", tags=["tasks"])
def get_task_progress(process_id: int, session: SessionDep, is_api: bool = True):
    # Query the database for the process
    statement = select(BackgroundTask).where(BackgroundTask.id == process_id)
    task = session.exec(statement).first()

    if not task:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Process not found",
            }
        )

    return (
        task
        if not is_api
        else {
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
    )


@app.post("/process_video", tags=["tasks"])
async def handle_process_video_request(
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
            progress=0,
            name=name,
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


@app.get("/get_video_paths/{task_id}", tags=["stats"])
async def get_video_paths(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(VideoPathsModel).where(VideoPathsModel.task_id == task_id)
    video_paths = session.exec(statement).first()
    if video_paths is None:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Video paths not found",
            }
        )

    return (
        video_paths
        if not is_api
        else {
            "success": True,
            "data": video_paths,
        }
    )


@app.get("/get_speed_stats/{task_id}", tags=["stats"])
async def get_speed_stats(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(SpeedModel).where(SpeedModel.task_id == task_id)
    speed_stats = session.exec(statement).first()

    if speed_stats is None:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Speed stats not found",
            }
        )

    return (
        json.loads(speed_stats.speed)
        if not is_api
        else {
            "success": True,
            "data": json.loads(speed_stats.speed),
        }
    )


@app.get("/get_ball_track/{task_id}", tags=["stats"])
async def get_ball_track(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(BallTrackModel).where(BallTrackModel.task_id == task_id)
    ball_track = session.exec(statement).first()
    if ball_track is None:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Ball track not found",
            }
        )

    ball_track.ball_track = json.loads(ball_track.ball_track)

    return (
        ball_track
        if not is_api
        else {
            "success": True,
            "data": ball_track,
        }
    )


@app.get("/get_bounces/{task_id}", tags=["stats"])
async def get_bounces(task_id: int, session: SessionDep, is_api: bool = True) -> object:
    statement = select(BouncesModel).where(BouncesModel.task_id == task_id)
    bounces = session.exec(statement).first()
    if bounces is None:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Bounces not found",
            }
        )

    bounces.bounces = json.loads(bounces.bounces)

    return (
        bounces
        if not is_api
        else {
            "success": True,
            "data": bounces,
        }
    )


@app.get("/get_direction_change_indices/{task_id}", tags=["stats"])
async def get_direction_change_indices_api(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
) -> object:
    statement = select(DirectionChangeIndicesModel).where(
        DirectionChangeIndicesModel.task_id == task_id
    )
    direction_change_indices = session.exec(statement).first()
    if direction_change_indices is None:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Direction change indices not found",
            }
        )

    direction_change_indices.direction_change_indices = json.loads(
        direction_change_indices.direction_change_indices
    )

    return (
        direction_change_indices
        if not is_api
        else {
            "success": True,
            "data": direction_change_indices,
        }
    )


@app.get("/thumbnail/{task_id}", tags=["stats"])
async def get_thumbnail(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(ThumbnailModel).where(ThumbnailModel.task_id == task_id)
    thumbnail = session.exec(statement).first()
    if thumbnail is None:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Thumbnail not found",
            }
        )

    return (
        thumbnail
        if not is_api
        else {
            "success": True,
            "data": thumbnail,
        }
    )


@app.get("/all-stats/{task_id}", tags=["stats"])
async def get_all_stats(task_id: int, session: SessionDep) -> object:
    video_paths = await get_video_paths(task_id, session, is_api=False)
    speed_stats = await get_speed_stats(task_id, session, is_api=False)
    ball_track = await get_ball_track(task_id, session, is_api=False)
    bounces = await get_bounces(task_id, session, is_api=False)
    direction_change_indices = await get_direction_change_indices_api(
        task_id, session, is_api=False
    )
    thumbnail = await get_thumbnail(task_id, session, is_api=False)
    progress = get_task_progress(task_id, session, is_api=False)

    return {
        "success": True,
        "data": {
            "video_paths": video_paths,
            "speed_stats": speed_stats,
            "ball_track": ball_track,
            "bounces": bounces,
            "direction_change_indices": direction_change_indices,
            "thumbnail": thumbnail,
        },
        "progress": progress,
    }


@app.get("/court_reference", tags=["misc"])
def get_court_reference():
    court_reference = CourtReference()
    return {"success": True, "data": court_reference.to_dict()}


@app.get("/", tags=["misc"])
def test_hello_world():
    return {
        "success": True,
        "message": "Hello world",
    }


@app.get("/check-health", tags=["misc"])
def check_health():
    return {
        "success": True,
        "message": "OK",
    }


@app.get("/stream/output/{filename}", tags=["stream"])
async def stream_output_file(filename: str, request: Request):
    """Stream large video files from output directory with range support"""
    file_path = f"./output/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Only stream video files
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Only video files can be streamed")

    # Fallback to full file streaming
    def iterfile(file_path: str):
        with open(file_path, mode="rb") as file_like:
            yield from file_like

    media_type, _ = mimetypes.guess_type(file_path)
    return StreamingResponse(iterfile(file_path), media_type=media_type)


@app.get("/stream/uploads/{filename}", tags=["stream"])
async def stream_uploads_file(filename: str, request: Request):
    """Stream large video files from output directory with range support"""
    file_path = f"./uploads/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Only stream video files
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Only video files can be streamed")

    # Fallback to full file streaming
    def iterfile(file_path: str):
        with open(file_path, mode="rb") as file_like:
            yield from file_like

    media_type, _ = mimetypes.guess_type(file_path)
    return StreamingResponse(iterfile(file_path), media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)
