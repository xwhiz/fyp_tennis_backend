import asyncio
import json
import mimetypes
import os
import re
import threading
import time
from collections import defaultdict
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
from sqlmodel import Field, Session, SQLModel, create_engine, select

from src.config import settings
from src.core.court_reference import CourtReference
from src.core.get_direction_change_indices import get_direction_change_indices
from src.core.stream_infer import (
    court_detector_stream_infer,
    get_ball_track_and_bounces_stream_infer,
)
from src.core.utils import get_court_img, perspective_transform_point, scene_detect
from src.db.engine import Engine
from src.db.utils import (
    save_ball_track_in_db,
    save_direction_change_indices_in_db,
    save_speed_in_db,
    save_video_paths_in_db,
)
from src.models.background_task import BackgroundTask
from src.models.ball_track import BallTrack
from src.models.bounces import Bounces
from src.models.direction_change_indices import DirectionChangeIndices
from src.schemas.process_video_response import ProcessVideoResponse
from src.db.utils import update_task_status, SessionDep
from src.schemas.speed_at import SpeedAt
from src.schemas.video_paths import VideoPathsSchema
from src.schemas.ball_track import BallTrackSchema
from src.schemas.bounces import BouncesSchema
from src.schemas.direction_change_indices import DirectionChangeIndicesSchema
from src.schemas.thumbnail import ThumbnailSchema
from src.models.speed import Speed
from src.models.thumbnail import Thumbnail
from src.models.video_paths import VideoPaths
from src.celery.worker import process_video_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are now loaded in Celery worker, not in FastAPI app
    yield


openapi_tags = [
    {"name": "tasks", "description": "Tasks"},
    {"name": "stats", "description": "Stats"},
    {"name": "misc", "description": "Misc"},
]
app = FastAPI(
    title=settings.app_name,
    description="A Self Hosted Tennis Analytics Platform",
    lifespan=lifespan,
    openapi_tags=openapi_tags,
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

# static files
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")




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

    task_dict = {
        "process_id": task.id,
        "progress": task.progress,
        "total_steps": task.total_steps,
        "status": task.status,
        "description": task.description,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }
    return (
        task_dict
        if not is_api
        else {
            "success": True,
            "data": task_dict,
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

    # Start background processing via Celery
    process_video_task.delay(int(process_id), file_path, name)

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
    statement = select(VideoPaths).where(VideoPaths.task_id == task_id)
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

    video_paths_dict = VideoPathsSchema.model_validate(video_paths).model_dump()
    return (
        video_paths_dict
        if not is_api
        else {
            "success": True,
            "data": video_paths_dict,
        }
    )


@app.get("/get_speed_stats/{task_id}", tags=["stats"])
async def get_speed_stats(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(Speed).where(Speed.task_id == task_id)
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
        json.loads(speed_stats.speeds)
        if not is_api
        else {
            "success": True,
            "data": json.loads(speed_stats.speeds),
        }
    )


@app.get("/get_ball_track/{task_id}", tags=["stats"])
async def get_ball_track(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(BallTrack).where(BallTrack.task_id == task_id)
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

    ball_track_dict = BallTrackSchema.model_validate(ball_track).model_dump()
    return (
        ball_track_dict
        if not is_api
        else {
            "success": True,
            "data": ball_track_dict,
        }
    )


@app.get("/get_bounces/{task_id}", tags=["stats"])
async def get_bounces(task_id: int, session: SessionDep, is_api: bool = True) -> object:
    statement = select(Bounces).where(Bounces.task_id == task_id)
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

    bounces_dict = BouncesSchema.model_validate(bounces).model_dump()
    return (
        bounces_dict
        if not is_api
        else {
            "success": True,
            "data": bounces_dict,
        }
    )


@app.get("/get_direction_change_indices/{task_id}", tags=["stats"])
async def get_direction_change_indices_api(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
) -> object:
    statement = select(DirectionChangeIndices).where(
        DirectionChangeIndices.task_id == task_id
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

    direction_change_indices_dict = DirectionChangeIndicesSchema.model_validate(direction_change_indices).model_dump()
    return (
        direction_change_indices_dict
        if not is_api
        else {
            "success": True,
            "data": direction_change_indices_dict,
        }
    )


@app.get("/thumbnail/{task_id}", tags=["stats"])
async def get_thumbnail(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(Thumbnail).where(Thumbnail.task_id == task_id)
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

    thumbnail_dict = ThumbnailSchema.model_validate(thumbnail).model_dump()
    return (
        thumbnail_dict
        if not is_api
        else {
            "success": True,
            "data": thumbnail_dict,
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
