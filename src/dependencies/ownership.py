import os

from fastapi import HTTPException
from sqlmodel import Session, select

from src.dependencies.auth import AuthContext
from src.models.background_task import BackgroundTask
from src.models.thumbnail import Thumbnail
from src.models.user import UserRole
from src.models.video_paths import VideoPaths


def is_admin(auth_ctx: AuthContext) -> bool:
    return auth_ctx.role == UserRole.ADMIN.value


def require_task_access(session: Session, task_id: int, auth_ctx: AuthContext) -> BackgroundTask:
    task = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if not is_admin(auth_ctx) and task.owner_id != auth_ctx.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return task


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

    for vp in session.exec(select(VideoPaths).where(VideoPaths.owner_id == auth_ctx.user_id)).all():
        for path in (vp.output_path, vp.minimap_path):
            if path and os.path.basename(path) == filename:
                return _normalize_disk_path(path)

    for th in session.exec(select(Thumbnail).where(Thumbnail.owner_id == auth_ctx.user_id)).all():
        if th.thumbnail_path and os.path.basename(th.thumbnail_path) == filename:
            return _normalize_disk_path(th.thumbnail_path)

    raise HTTPException(status_code=403, detail="Access denied")


def require_upload_stream_path(session: Session, filename: str, auth_ctx: AuthContext) -> str:
    """Return normalized disk path under ./uploads for filename if allowed."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if is_admin(auth_ctx):
        return f"./uploads/{filename}"

    for bt in session.exec(select(BackgroundTask).where(BackgroundTask.owner_id == auth_ctx.user_id)).all():
        if bt.video_path and os.path.basename(bt.video_path) == filename:
            return _normalize_disk_path(bt.video_path)

    raise HTTPException(status_code=403, detail="Access denied")
