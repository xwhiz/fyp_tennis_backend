from datetime import datetime
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, or_, select

from src.celery.worker import get_celery, process_video_task
from src.config import settings
from src.db.engine import Engine
from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.dependencies.ownership import is_admin, require_task_access
from src.models.background_task import BackgroundTask
from src.models.ball_track import BallTrack
from src.models.bounces import Bounces
from src.models.direction_change_indices import DirectionChangeIndices
from src.models.player_positions import PlayerPositions
from src.models.rally_stats import RallyStats
from src.models.speed import Speed
from src.models.thumbnail import Thumbnail
from src.models.user import User
from src.models.video_paths import VideoPaths
from src.schemas.process_video_response import ProcessVideoResponse
from src.utils.at_tag import display_at_tag, normalize_at_tag_input

router = APIRouter(tags=["tasks"])


def _resolve_opponent_id(session: Session, owner_id: str, opponent_at_tag: Optional[str]) -> Optional[str]:
    if opponent_at_tag is None or not str(opponent_at_tag).strip():
        return None
    tag = normalize_at_tag_input(opponent_at_tag)
    if not tag:
        return None
    opp = session.exec(select(User).where(User.at_tag == tag)).first()
    if opp is None:
        raise HTTPException(status_code=404, detail="Opponent not found")
    if opp.id == owner_id:
        raise HTTPException(status_code=400, detail="Cannot set yourself as the opponent")
    return opp.id


def _opponent_payload(session: Session, opponent_id: Optional[str]) -> Optional[dict]:
    if not opponent_id:
        return None
    u = session.exec(select(User).where(User.id == opponent_id)).first()
    if u is None:
        return None
    return {
        "id": u.id,
        "atTag": display_at_tag(u.at_tag),
        "firstName": u.first_name,
        "lastName": u.last_name,
    }


@router.get("/all_tasks")
def get_all_tasks(
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    statement = select(BackgroundTask)
    if not is_admin(auth_ctx):
        statement = statement.where(
            or_(
                BackgroundTask.owner_id == auth_ctx.user_id,
                BackgroundTask.opponent_id == auth_ctx.user_id,
            ),
        )
    tasks = session.exec(statement).all()
    opp_ids = {t.opponent_id for t in tasks if t.opponent_id}
    opp_users = {}
    if opp_ids:
        for u in session.exec(select(User).where(User.id.in_(opp_ids))).all():
            opp_users[u.id] = u
    out = []
    for task in tasks:
        o = None
        if task.opponent_id and task.opponent_id in opp_users:
            ou = opp_users[task.opponent_id]
            o = {
                "id": ou.id,
                "atTag": display_at_tag(ou.at_tag),
                "firstName": ou.first_name,
                "lastName": ou.last_name,
            }
        out.append(
            {
                "id": str(task.id),
                "name": task.name,
                "status": task.status,
                "description": task.description,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "total_upload_size": task.total_upload_size,
                "uploaded_size": task.uploaded_size,
                "is_uploaded_fully": task.is_uploaded_fully,
                "progress": task.progress,
                "opponent": o,
            },
        )
    return {"success": True, "data": out}


@router.get("/task_progress/{process_id}")
def get_task_progress(
    process_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    statement = select(BackgroundTask).where(BackgroundTask.id == process_id)
    task = session.exec(statement).first()

    if not task:
        return None if not is_api else {"success": True, "message": "Process not found"}

    if not is_admin(auth_ctx) and task.owner_id != auth_ctx.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

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
        "opponent": _opponent_payload(session, task.opponent_id),
    }
    return task_dict if not is_api else {"success": True, "data": task_dict}


@router.post("/process_video")
async def handle_process_video_request(
    name: str = Form(...),
    video_file: UploadFile = File(...),
    total_size: int = Form(int),
    duplicate_task: bool = Form(False),
    task_id: Optional[int] = Form(None),
    opponent_at_tag: Optional[str] = Form(None),
    opponentAtTag: Optional[str] = Form(None),
    opponent_tag: Optional[str] = Form(None),
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> ProcessVideoResponse:
    if duplicate_task and task_id is None:
        raise HTTPException(status_code=400, detail="task_id required when duplicate_task is True")

    # Accept snake_case and camelCase aliases from different clients.
    opponent_tag_value = opponent_at_tag or opponentAtTag or opponent_tag
    with Session(Engine.instance()) as _op_session:
        resolved_opponent_id = _resolve_opponent_id(_op_session, auth_ctx.user_id, opponent_tag_value)

    if duplicate_task and task_id is not None:
        with Session(Engine.instance()) as session:
            statement = select(BackgroundTask).where(BackgroundTask.id == task_id)
            existing_task = session.exec(statement).first()

            if not existing_task:
                raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")

            if not is_admin(auth_ctx) and existing_task.owner_id != auth_ctx.user_id:
                raise HTTPException(status_code=403, detail="Access denied")

            os.makedirs("./uploads", exist_ok=True)
            os.makedirs("./uploads/temp", exist_ok=True)
            chunk_size = settings.upload_chunk_size
            file_path = existing_task.video_path
            file_name = os.path.basename(file_path) if file_path else video_file.filename

            if existing_task.is_uploaded_fully:
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
                    owner_id=auth_ctx.user_id,
                    opponent_id=resolved_opponent_id,
                )
                session.add(new_task)
                session.commit()
                session.refresh(new_task)
                process_video_task.delay(int(new_task.id), new_task.video_path, new_task.name)
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

            needs_multipart = (
                not existing_task.is_uploaded_fully
                and existing_task.total_upload_size >= chunk_size
            )
            # If caller provided/changed opponent for an existing multipart task, persist it.
            if opponent_tag_value is not None:
                existing_task.opponent_id = resolved_opponent_id
                session.add(existing_task)
                session.commit()
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

    if not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    import uuid

    file_extension = video_file.filename.split(".")[-1] if "." in video_file.filename else "mp4"
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"./uploads/{unique_filename}"

    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./uploads/temp", exist_ok=True)

    if total_size is None:
        content = await video_file.read()
        total_size = len(content)
        await video_file.seek(0)
    else:
        content = await video_file.read()

    chunk_size = settings.upload_chunk_size
    needs_multipart = total_size >= chunk_size

    with Session(Engine.instance()) as session:
        if needs_multipart:
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
                owner_id=auth_ctx.user_id,
                opponent_id=resolved_opponent_id,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            process_id = str(task.id)
            temp_dir = f"./uploads/temp/{task.id}"
            os.makedirs(temp_dir, exist_ok=True)
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
            owner_id=auth_ctx.user_id,
            opponent_id=resolved_opponent_id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        process_id = str(task.id)
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


@router.post("/upload_chunk/{task_id}")
async def upload_chunk(
    task_id: int,
    session: SessionDep,
    chunk_number: int = Form(...),
    chunk_data: UploadFile = File(...),
    total_chunks: int = Form(None),
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    task = require_task_access(session, task_id, auth_ctx)

    if task.is_uploaded_fully:
        raise HTTPException(status_code=400, detail="Task upload already completed. Cannot upload more chunks.")

    chunk_content = await chunk_data.read()
    chunk_size = len(chunk_content)
    temp_dir = f"./uploads/temp/{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_number}.part")
    chunk_already_exists = os.path.exists(chunk_path)

    with open(chunk_path, "wb") as f:
        f.write(chunk_content)

    if not chunk_already_exists:
        new_uploaded_size = task.uploaded_size + chunk_size
        task.uploaded_size = new_uploaded_size
        task.updated_at = datetime.now()
        session.add(task)
        session.commit()
    else:
        session.refresh(task)
        new_uploaded_size = task.uploaded_size

    upload_complete = new_uploaded_size >= task.total_upload_size

    if upload_complete:
        chunk_files = []
        for filename in os.listdir(temp_dir):
            if filename.startswith("chunk_") and filename.endswith(".part"):
                chunk_num = int(filename.split("_")[1].split(".")[0])
                chunk_files.append((chunk_num, os.path.join(temp_dir, filename)))

        chunk_files.sort(key=lambda x: x[0])
        with open(task.video_path, "wb") as output_file:
            for _, cpath in chunk_files:
                with open(cpath, "rb") as chunk_file:
                    output_file.write(chunk_file.read())

        for _, cpath in chunk_files:
            try:
                os.remove(cpath)
            except Exception as e:
                print(f"[UPLOAD CHUNK WARNING]: Failed to delete chunk {cpath}: {str(e)}")

        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"[UPLOAD CHUNK WARNING]: Failed to remove temp directory {temp_dir}: {str(e)}")

        task.is_uploaded_fully = True
        task.status = "pending"
        task.description = f"Upload complete. Processing video: {task.name}"
        task.updated_at = datetime.now()
        session.add(task)
        session.commit()

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


@router.delete("/delete_task/{task_id}")
def delete_task(
    task_id: int,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    task = require_task_access(session, task_id, auth_ctx)

    files_deleted = []
    records_deleted = 0

    try:
        if task.status in ["pending", "processing"]:
            try:
                celery_app = get_celery()
                inspect = celery_app.control.inspect()
                active_tasks = inspect.active() or {}
                reserved_tasks = inspect.reserved() or {}
                for _, tasks in {**active_tasks, **reserved_tasks}.items():
                    for celery_task in tasks:
                        if celery_task.get("name") == "process_video_task":
                            args = celery_task.get("args", [])
                            if args and len(args) > 0 and args[0] == task_id:
                                celery_task_id = celery_task.get("id")
                                if celery_task_id:
                                    celery_app.control.revoke(celery_task_id, terminate=True)
            except Exception as e:
                print(f"[DELETE TASK WARNING]: Failed to revoke Celery task: {str(e)}")

        file_paths_to_delete = []
        if task.video_path:
            video_paths_stmt = select(BackgroundTask).where(BackgroundTask.video_path == task.video_path)
            video_paths = session.exec(video_paths_stmt).all()
            if len(video_paths) == 0:
                file_paths_to_delete.append(task.video_path)

        video_paths_stmt = select(VideoPaths).where(VideoPaths.task_id == task_id)
        video_paths = session.exec(video_paths_stmt).first()
        if video_paths:
            if video_paths.output_path:
                file_paths_to_delete.append(video_paths.output_path)
            if video_paths.minimap_path:
                file_paths_to_delete.append(video_paths.minimap_path)

        thumbnail_stmt = select(Thumbnail).where(Thumbnail.task_id == task_id)
        thumbnail = session.exec(thumbnail_stmt).first()
        if thumbnail and thumbnail.thumbnail_path:
            file_paths_to_delete.append(thumbnail.thumbnail_path)

        if video_paths:
            session.delete(video_paths)
            records_deleted += 1
        if thumbnail:
            session.delete(thumbnail)
            records_deleted += 1

        ball_track = session.exec(select(BallTrack).where(BallTrack.task_id == task_id)).first()
        if ball_track:
            session.delete(ball_track)
            records_deleted += 1

        bounces = session.exec(select(Bounces).where(Bounces.task_id == task_id)).first()
        if bounces:
            session.delete(bounces)
            records_deleted += 1

        direction_change = session.exec(
            select(DirectionChangeIndices).where(DirectionChangeIndices.task_id == task_id),
        ).first()
        if direction_change:
            session.delete(direction_change)
            records_deleted += 1

        speed = session.exec(select(Speed).where(Speed.task_id == task_id)).first()
        if speed:
            session.delete(speed)
            records_deleted += 1

        player_positions = session.exec(select(PlayerPositions).where(PlayerPositions.task_id == task_id)).first()
        if player_positions:
            session.delete(player_positions)
            records_deleted += 1

        rally_stats = session.exec(select(RallyStats).where(RallyStats.task_id == task_id)).first()
        if rally_stats:
            session.delete(rally_stats)
            records_deleted += 1

        session.delete(task)
        records_deleted += 1
        session.commit()

        for file_path in file_paths_to_delete:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    files_deleted.append(file_path)
            except Exception as e:
                print(f"[DELETE TASK WARNING]: Failed to delete file {file_path}: {str(e)}")

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
        session.rollback()
        error_msg = f"Error deleting task {task_id}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
