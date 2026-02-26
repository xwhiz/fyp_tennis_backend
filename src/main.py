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
from typing import Annotated, Optional


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
from src.core.utils import classify_serve_type, get_court_img, perspective_transform_point, scene_detect
from src.db.engine import Engine
from src.db.utils import (
    save_ball_track_in_db,
    save_direction_change_indices_in_db,
    save_speed_in_db,
    save_video_paths_in_db,
    update_upload_progress,
)
from src.models.background_task import BackgroundTask
from src.models.ball_track import BallTrack
from src.models.bounces import Bounces
from src.models.direction_change_indices import DirectionChangeIndices
from src.models.player_positions import PlayerPositions
from src.schemas.process_video_response import ProcessVideoResponse
from src.db.utils import update_task_status, SessionDep
from src.schemas.speed_at import SpeedAt
from src.schemas.video_paths import VideoPathsSchema
from src.schemas.ball_track import BallTrackSchema
from src.schemas.bounces import BouncesSchema
from src.schemas.direction_change_indices import DirectionChangeIndicesSchema
from src.schemas.player_positions import PlayerPositionsSchema
from src.schemas.thumbnail import ThumbnailSchema
from src.models.speed import Speed
from src.models.thumbnail import Thumbnail
from src.models.video_paths import VideoPaths
from src.celery.worker import process_video_task, get_celery


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are now loaded in Celery worker, not in FastAPI app
    
    # Re-queue unprocessed tasks on startup
    try:
        with Session(Engine.instance()) as session:
            # Query for tasks that need to be re-queued
            # Exclude tasks that are not fully uploaded (is_uploaded_fully=False)
            statement = select(BackgroundTask).where(
                BackgroundTask.status.in_(["pending", "processing", "failed"]),
                BackgroundTask.is_uploaded_fully == True
            )
            unprocessed_tasks = session.exec(statement).all()
            
            requeued_count = 0
            for task in unprocessed_tasks:
                # Re-queue the task
                try:
                    process_video_task.delay(int(task.id), task.video_path, task.name)
                    
                    # Reset status to pending for tasks that were processing or failed
                    if task.status in ["processing", "failed"]:
                        task.status = "pending"
                        task.description = "Re-queued after API restart"
                        session.add(task)
                    
                    requeued_count += 1
                except Exception as e:
                    print(f"[STARTUP ERROR]: Failed to re-queue task {task.id}: {str(e)}")
            
            if requeued_count > 0:
                session.commit()
                print(f"[STARTUP]: Re-queued {requeued_count} unprocessed task(s)")
            else:
                print("[STARTUP]: No unprocessed tasks to re-queue")
    except Exception as e:
        print(f"[STARTUP ERROR]: Failed to re-queue tasks: {str(e)}")
    
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
    # make the id a string
    tasks = [{"id": str(task.id), "name": task.name, "status": task.status, "description": task.description, "created_at": task.created_at, "updated_at": task.updated_at, "total_upload_size": task.total_upload_size, "uploaded_size": task.uploaded_size, "is_uploaded_fully": task.is_uploaded_fully, "progress": task.progress} for task in tasks]
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
        "status": task.status,
        "description": task.description,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "total_upload_size": task.total_upload_size,
        "uploaded_size": task.uploaded_size,
        "is_uploaded_fully": task.is_uploaded_fully,
        "upload_progress_percent": (
            (task.uploaded_size / task.total_upload_size * 100)
            if task.total_upload_size > 0
            else 0
        ),
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
    name: str = Form(...),
    video_file: UploadFile = File(...),
    total_size: int = Form(int),
    duplicate_task: bool = Form(False),
    task_id: Optional[int] = Form(None),
) -> ProcessVideoResponse:
    # Validate duplicate request
    if duplicate_task and task_id is None:
        raise HTTPException(
            status_code=400,
            detail="task_id required when duplicate_task is True",
        )

    # Handle duplicate task request
    if duplicate_task and task_id is not None:
        with Session(Engine.instance()) as session:
            statement = select(BackgroundTask).where(BackgroundTask.id == task_id)
            existing_task = session.exec(statement).first()

            if not existing_task:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task with id {task_id} not found",
                )

            import os
            os.makedirs("./uploads", exist_ok=True)
            os.makedirs("./uploads/temp", exist_ok=True)
            chunk_size = settings.upload_chunk_size
            file_path = existing_task.video_path
            file_name = os.path.basename(file_path) if file_path else video_file.filename

            if existing_task.is_uploaded_fully:
                # Create new task reusing the already-uploaded video; start processing
                new_task = BackgroundTask(
                    progress=0.0,
                    name=name,
                    status="pending",
                    video_path=existing_task.video_path,
                    description=f"Processing video: {name}",
                    total_upload_size=existing_task.total_upload_size,
                    uploaded_size=existing_task.total_upload_size,
                    is_uploaded_fully=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(new_task)
                session.commit()
                session.refresh(new_task)
                process_video_task.delay(
                    int(new_task.id), new_task.video_path, new_task.name
                )
                return {
                    "success": True,
                    "message": "Duplicate task created. Processing started.",
                    "data": {
                        "process_id": str(new_task.id),
                        "filename": video_file.filename,
                        "name": name,
                        "file_path": new_task.video_path,
                        "file_name": file_name,
                        "total_size": new_task.total_upload_size,
                        "chunk_size": chunk_size,
                        "requires_multipart": False,
                    },
                }
            else:
                # Return existing task for client to resume chunk uploads
                needs_multipart = (
                    not existing_task.is_uploaded_fully
                    and existing_task.total_upload_size >= chunk_size
                )
                return {
                    "success": True,
                    "message": "Duplicate task found. Using existing task.",
                    "data": {
                        "process_id": str(existing_task.id),
                        "filename": video_file.filename,
                        "name": existing_task.name,
                        "file_path": file_path,
                        "file_name": file_name,
                        "total_size": existing_task.total_upload_size,
                        "chunk_size": chunk_size,
                        "requires_multipart": needs_multipart,
                    },
                }

    # Normal flow: validate file type
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
    os.makedirs("./uploads/temp", exist_ok=True)

    # Determine file size - use total_size parameter if provided, otherwise read file
    if total_size is None:
        # Read file to get size
        content = await video_file.read()
        total_size = len(content)
        # Reset file pointer for saving
        await video_file.seek(0)
    else:
        content = await video_file.read()

    # Check if multipart upload is needed
    chunk_size = settings.upload_chunk_size
    needs_multipart = total_size >= chunk_size

    # Store the process in database
    with Session(Engine.instance()) as session:
        if needs_multipart:
            # For multipart uploads, create task but don't save file yet
            # Set is_uploaded_fully=False and track upload progress
            task = BackgroundTask(
                progress=0.0,
                name=name,
                status="uploading",
                video_path=file_path,
                description=f"Uploading video: {name}",
                total_upload_size=total_size,
                uploaded_size=0,
                is_uploaded_fully=False,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            process_id = str(task.id)

            # Create temp directory for chunks
            temp_dir = f"./uploads/temp/{task.id}"
            os.makedirs(temp_dir, exist_ok=True)

            # Don't trigger Celery task yet - wait for all chunks
            return {
                "success": True,
                "message": "Task created. Please upload chunks using /upload_chunk/{task_id}",
                "data": {
                    "process_id": process_id,
                    "filename": video_file.filename,
                    "name": name,
                    "file_path": file_path,
                    "file_name": unique_filename,
                    "total_size": total_size,
                    "chunk_size": chunk_size,
                    "requires_multipart": True,
                },
            }
        else:
            # For small files, upload in one go (existing behavior)
            with open(file_path, "wb") as buffer:
                buffer.write(content)

            task = BackgroundTask(
                progress=0.0,
                name=name,
                status="pending",
                video_path=file_path,
                description=f"Processing video: {name}",
                total_upload_size=total_size,
                uploaded_size=total_size,
                is_uploaded_fully=True,
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
                    "file_name": unique_filename,
                    "total_size": total_size,
                    "chunk_size": chunk_size,
                    "requires_multipart": False,
                },
            }


@app.post("/upload_chunk/{task_id}", tags=["tasks"])
async def upload_chunk(
    task_id: int,
    session: SessionDep,
    chunk_number: int = Form(...),
    chunk_data: UploadFile = File(...),
    total_chunks: int = Form(None),
):
    """Upload a chunk of a multipart file upload."""
    # Query the task
    statement = select(BackgroundTask).where(BackgroundTask.id == task_id)
    task = session.exec(statement).first()

    if not task:
        raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")

    if task.is_uploaded_fully:
        raise HTTPException(
            status_code=400, detail="Task upload already completed. Cannot upload more chunks."
        )

    # Read chunk data
    chunk_content = await chunk_data.read()
    chunk_size = len(chunk_content)

    # Create temp directory for chunks if it doesn't exist
    temp_dir = f"./uploads/temp/{task_id}"
    os.makedirs(temp_dir, exist_ok=True)

    # Save chunk to temp directory
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_number}.part")
    
    # Check if chunk already exists (retry scenario)
    chunk_already_exists = os.path.exists(chunk_path)
    
    # Save chunk file (overwrite if exists)
    with open(chunk_path, "wb") as f:
        f.write(chunk_content)
    
    # Update uploaded_size only if this is a new chunk
    if not chunk_already_exists:
        new_uploaded_size = task.uploaded_size + chunk_size
        task.uploaded_size = new_uploaded_size
        task.updated_at = datetime.now()
        session.add(task)
        session.commit()
    else:
        # Chunk was retried, don't double-count but refresh to get current size
        session.refresh(task)
        new_uploaded_size = task.uploaded_size

    # Check if all chunks have been received
    # We check if uploaded_size >= total_upload_size (allowing for small rounding differences)
    upload_complete = new_uploaded_size >= task.total_upload_size

    if upload_complete:
        # Reassemble file from chunks
        # Get all chunk files and sort by chunk number
        chunk_files = []
        for filename in os.listdir(temp_dir):
            if filename.startswith("chunk_") and filename.endswith(".part"):
                chunk_num = int(filename.split("_")[1].split(".")[0])
                chunk_files.append((chunk_num, os.path.join(temp_dir, filename)))

        # Sort by chunk number to ensure correct order
        chunk_files.sort(key=lambda x: x[0])

        # Reassemble file
        with open(task.video_path, "wb") as output_file:
            for chunk_num, chunk_path in chunk_files:
                with open(chunk_path, "rb") as chunk_file:
                    output_file.write(chunk_file.read())

        # Clean up temp chunks
        for chunk_num, chunk_path in chunk_files:
            try:
                os.remove(chunk_path)
            except Exception as e:
                print(f"[UPLOAD CHUNK WARNING]: Failed to delete chunk {chunk_path}: {str(e)}")

        # Try to remove temp directory
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"[UPLOAD CHUNK WARNING]: Failed to remove temp directory {temp_dir}: {str(e)}")

        # Mark upload as complete and trigger processing
        task.is_uploaded_fully = True
        task.status = "pending"
        task.description = f"Upload complete. Processing video: {task.name}"
        task.updated_at = datetime.now()
        session.add(task)
        session.commit()

        # Start background processing via Celery
        process_video_task.delay(int(task_id), task.video_path, task.name)

        return {
            "success": True,
            "message": "All chunks uploaded. Video processing started.",
            "data": {
                "task_id": task_id,
                "uploaded_size": new_uploaded_size,
                "total_size": task.total_upload_size,
                "upload_complete": True,
                "chunks_received": len(chunk_files),
            },
        }
    else:
        # Upload still in progress
        progress_percent = (new_uploaded_size / task.total_upload_size) * 100 if task.total_upload_size > 0 else 0
        return {
            "success": True,
            "message": f"Chunk {chunk_number} uploaded successfully",
            "data": {
                "task_id": task_id,
                "chunk_number": chunk_number,
                "chunk_size": chunk_size,
                "uploaded_size": new_uploaded_size,
                "total_size": task.total_upload_size,
                "upload_complete": False,
                "progress_percent": round(progress_percent, 2),
            },
        }


@app.delete("/delete_task/{task_id}", tags=["tasks"])
def delete_task(task_id: int, session: SessionDep):
    """Delete a task and all its associated data and files."""
    # Query the task first
    statement = select(BackgroundTask).where(BackgroundTask.id == task_id)
    task = session.exec(statement).first()
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")
    
    files_deleted = []
    records_deleted = 0
    
    try:
        # 1. Attempt to revoke Celery task if it's pending or processing
        if task.status in ["pending", "processing"]:
            try:
                celery_app = get_celery()
                inspect = celery_app.control.inspect()
                
                # Get active and reserved tasks
                active_tasks = inspect.active() or {}
                reserved_tasks = inspect.reserved() or {}
                
                # Find and revoke tasks matching this task_id
                for worker_name, tasks in {**active_tasks, **reserved_tasks}.items():
                    for celery_task in tasks:
                        # Check if this is our process_video_task with matching task_id
                        if celery_task.get("name") == "process_video_task":
                            args = celery_task.get("args", [])
                            # args should be [task_id, video_path, name]
                            if args and len(args) > 0 and args[0] == task_id:
                                celery_task_id = celery_task.get("id")
                                if celery_task_id:
                                    celery_app.control.revoke(celery_task_id, terminate=True)
                                    print(f"[DELETE TASK]: Revoked Celery task {celery_task_id} for task_id {task_id}")
            except Exception as e:
                print(f"[DELETE TASK WARNING]: Failed to revoke Celery task: {str(e)}")
                # Continue with deletion even if revocation fails
        
        # 2. Collect file paths before deleting records
        file_paths_to_delete = []
        
        # Input video from BackgroundTask
        if task.video_path:
            # only delete the video if not referenced by any other BackgroundTask
            video_paths_stmt = select(BackgroundTask).where(BackgroundTask.video_path == task.video_path)
            video_paths = session.exec(video_paths_stmt).all()
            if len(video_paths) == 0:
                file_paths_to_delete.append(task.video_path)
            else:
                print(f"[DELETE TASK WARNING]: Video {task.video_path} is referenced by other BackgroundTasks. Not deleting.")
        
        # Get VideoPaths if exists
        video_paths_stmt = select(VideoPaths).where(VideoPaths.task_id == task_id)
        video_paths = session.exec(video_paths_stmt).first()
        if video_paths:
            if video_paths.output_path:
                file_paths_to_delete.append(video_paths.output_path)
            if video_paths.minimap_path:
                file_paths_to_delete.append(video_paths.minimap_path)
        
        # Get Thumbnail if exists
        thumbnail_stmt = select(Thumbnail).where(Thumbnail.task_id == task_id)
        thumbnail = session.exec(thumbnail_stmt).first()
        if thumbnail and thumbnail.thumbnail_path:
            file_paths_to_delete.append(thumbnail.thumbnail_path)
        
        # 3. Delete database records
        # Delete in order: related records first, then main task
        
        # Delete VideoPaths
        if video_paths:
            session.delete(video_paths)
            records_deleted += 1
        
        # Delete Thumbnail
        if thumbnail:
            session.delete(thumbnail)
            records_deleted += 1
        
        # Delete BallTrack
        ball_track_stmt = select(BallTrack).where(BallTrack.task_id == task_id)
        ball_track = session.exec(ball_track_stmt).first()
        if ball_track:
            session.delete(ball_track)
            records_deleted += 1
        
        # Delete Bounces
        bounces_stmt = select(Bounces).where(Bounces.task_id == task_id)
        bounces = session.exec(bounces_stmt).first()
        if bounces:
            session.delete(bounces)
            records_deleted += 1
        
        # Delete DirectionChangeIndices
        direction_change_stmt = select(DirectionChangeIndices).where(DirectionChangeIndices.task_id == task_id)
        direction_change = session.exec(direction_change_stmt).first()
        if direction_change:
            session.delete(direction_change)
            records_deleted += 1
        
        # Delete Speed
        speed_stmt = select(Speed).where(Speed.task_id == task_id)
        speed = session.exec(speed_stmt).first()
        if speed:
            session.delete(speed)
            records_deleted += 1
        
        # Delete PlayerPositions
        player_positions_stmt = select(PlayerPositions).where(PlayerPositions.task_id == task_id)
        player_positions = session.exec(player_positions_stmt).first()
        if player_positions:
            session.delete(player_positions)
            records_deleted += 1
        
        # Delete main BackgroundTask
        session.delete(task)
        records_deleted += 1
        
        # Commit all database deletions
        session.commit()
        
        # 4. Delete files
        for file_path in file_paths_to_delete:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    files_deleted.append(file_path)
                    print(f"[DELETE TASK]: Deleted file {file_path}")
            except Exception as e:
                print(f"[DELETE TASK WARNING]: Failed to delete file {file_path}: {str(e)}")
                # Continue with other files even if one fails
        
        return {
            "success": True,
            "message": f"Task {task_id} deleted successfully",
            "data": {
                "task_id": str(task_id),
                "files_deleted": files_deleted,
                "records_deleted": records_deleted,
            },
        }
    
    except Exception as e:
        # Rollback on error
        session.rollback()
        error_msg = f"Error deleting task {task_id}: {str(e)}"
        print(f"[DELETE TASK ERROR]: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


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


@app.get("/get_player_positions/{task_id}", tags=["stats"])
async def get_player_positions(
    task_id: int, session: SessionDep, is_api: bool = True
) -> object:
    statement = select(PlayerPositions).where(PlayerPositions.task_id == task_id)
    player_positions = session.exec(statement).first()
    if player_positions is None:
        return (
            None
            if not is_api
            else {
                "success": True,
                "message": "Player positions not found",
            }
        )

    player_positions.positions = json.loads(player_positions.positions)

    player_positions_dict = PlayerPositionsSchema.model_validate(player_positions).model_dump()
    return (
        player_positions_dict
        if not is_api
        else {
            "success": True,
            "data": player_positions_dict,
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
    # make the id a string
    thumbnail_dict = {"id": str(thumbnail.id), "thumbnail_path": thumbnail.thumbnail_path, "created_at": thumbnail.created_at, "updated_at": thumbnail.updated_at}
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
    player_positions = await get_player_positions(task_id, session, is_api=False)
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
            "player_positions": player_positions,
            "thumbnail": thumbnail,
        },
        "progress": progress,
    }


@app.get("/serve_stats/{task_id}", tags=["stats"])
async def get_serve_stats(task_id: int, session: SessionDep) -> object:
    # 1. Load bounces
    bounces_row = session.exec(
        select(Bounces).where(Bounces.task_id == task_id)
    ).first()
    if bounces_row is None:
        return {"success": True, "message": "Bounces not found", "data": {"serves": []}}
    bounces_data = json.loads(bounces_row.bounces)

    # 2. Load direction change indices
    dci_row = session.exec(
        select(DirectionChangeIndices).where(DirectionChangeIndices.task_id == task_id)
    ).first()
    if dci_row is None:
        return {"success": True, "message": "Direction change indices not found", "data": {"serves": []}}
    dci_data = json.loads(dci_row.direction_change_indices)
    dc_frames_sorted = sorted(int(k) for k in dci_data.keys())

    # 3. Load ball track
    ball_track_row = session.exec(
        select(BallTrack).where(BallTrack.task_id == task_id)
    ).first()
    if ball_track_row is None:
        return {"success": True, "message": "Ball track not found", "data": {"serves": []}}
    ball_track_data = json.loads(ball_track_row.ball_track)

    # 4. Find serve bounces and build response
    serves = []
    for frame_str, bounce_info in bounces_data.items():
        if not bounce_info.get("serve", False):
            continue

        bounce_frame = int(frame_str)
        bounce_position = bounce_info["position"]

        # Find the last direction change frame before this bounce
        origin_frame = None
        for dc_frame in reversed(dc_frames_sorted):
            if dc_frame < bounce_frame:
                origin_frame = dc_frame
                break

        if origin_frame is None:
            continue

        # Look up origin position from ball_track (court coordinates)
        origin_position = ball_track_data.get(str(origin_frame))
        if origin_position is None:
            continue

        # Collect ball track from origin to bounce (inclusive)
        track_segment = []
        for f in range(origin_frame, bounce_frame + 1):
            point = ball_track_data.get(str(f))
            if point is not None:
                track_segment.append(point)

        # Classify serve type
        serve_type = "unknown"
        if bounce_position and bounce_position[0] is not None and bounce_position[1] is not None:
            serve_type = classify_serve_type(bounce_position[0], bounce_position[1])

        serves.append({
            "bounce_frame": bounce_frame,
            "bounce_position": bounce_position,
            "origin_frame": origin_frame,
            "origin_position": origin_position,
            "ball_track": track_segment,
            "serve_type": serve_type,
        })

    # Sort by bounce frame
    serves.sort(key=lambda s: s["bounce_frame"])

    return {
        "success": True,
        "data": {"serves": serves},
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
