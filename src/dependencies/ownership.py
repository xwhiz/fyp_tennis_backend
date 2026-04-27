import os

from fastapi import HTTPException
from sqlmodel import Session, select

from src.dependencies.auth import AuthContext
from src.config import settings
from src.models.background_task import BackgroundTask
from src.models.chat_session import ChatSession
from src.models.chat_stream import ChatStream
from src.models.thumbnail import Thumbnail
from src.models.user import UserRole
from src.models.video_paths import VideoPaths
from src.utils.profile_image import PROFILE_IMAGE_DIR


def is_admin(auth_ctx: AuthContext) -> bool:
    return auth_ctx.role == UserRole.ADMIN.value


def require_task_access(session: Session, task_id: int, auth_ctx: AuthContext) -> BackgroundTask:
    task = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if (
        not is_admin(auth_ctx)
        and task.owner_id != auth_ctx.user_id
        and task.opponent_id != auth_ctx.user_id
    ):
        raise HTTPException(status_code=403, detail="Access denied")
    return task


def require_chat_session_access(
    session: Session,
    session_id: str,
    auth_ctx: AuthContext,
) -> ChatSession:
    chat_session = session.exec(
        select(ChatSession).where(ChatSession.id == session_id),
    ).first()
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if not is_admin(auth_ctx) and chat_session.user_id != auth_ctx.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if chat_session.task_id is not None:
        require_task_access(session, chat_session.task_id, auth_ctx)
    return chat_session


def require_chat_stream_access(
    session: Session,
    stream_id: str,
    auth_ctx: AuthContext,
) -> ChatStream:
    chat_stream = session.exec(
        select(ChatStream).where(ChatStream.id == stream_id),
    ).first()
    if chat_stream is None:
        raise HTTPException(status_code=404, detail="Chat stream not found")
    if not is_admin(auth_ctx) and chat_stream.user_id != auth_ctx.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return chat_stream


def _normalize_disk_path(path: str) -> str:
    if path.startswith("./"):
        return path
    if path.startswith("output/") or path.startswith("uploads/"):
        return "./" + path
    return path


def require_output_stream_path(session: Session, filename: str, auth_ctx: AuthContext) -> str:
    """Return normalized disk path under ./output for filename if allowed."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if is_admin(auth_ctx):
        return f"./output/{filename}"

    for vp in session.exec(select(VideoPaths)).all():
        for path in (vp.output_path, vp.minimap_path):
            if path and os.path.basename(path) == filename:
                task = session.exec(
                    select(BackgroundTask).where(BackgroundTask.id == vp.task_id),
                ).first()
                if task and (
                    task.owner_id == auth_ctx.user_id or task.opponent_id == auth_ctx.user_id
                ):
                    return _normalize_disk_path(path)

    for th in session.exec(select(Thumbnail)).all():
        if th.thumbnail_path and os.path.basename(th.thumbnail_path) == filename:
            task = session.exec(
                select(BackgroundTask).where(BackgroundTask.id == th.task_id),
            ).first()
            if task and (
                task.owner_id == auth_ctx.user_id or task.opponent_id == auth_ctx.user_id
            ):
                return _normalize_disk_path(th.thumbnail_path)

    raise HTTPException(status_code=403, detail="Access denied")


def require_upload_stream_path(session: Session, filename: str, auth_ctx: AuthContext) -> str:
    """Return normalized disk path under ./uploads for filename if allowed."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if is_admin(auth_ctx):
        return os.path.join(settings.upload_root_dir, filename)

    for bt in session.exec(
        select(BackgroundTask).where(
            (BackgroundTask.owner_id == auth_ctx.user_id)
            | (BackgroundTask.opponent_id == auth_ctx.user_id),
        ),
    ).all():
        if bt.video_path and os.path.basename(bt.video_path) == filename:
            return _normalize_disk_path(bt.video_path)

    raise HTTPException(status_code=403, detail="Access denied")


def require_profile_image_stream_path(filename: str, auth_ctx: AuthContext) -> str:
    """Any authenticated user can view profile images."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return os.path.join(PROFILE_IMAGE_DIR, filename)
