import mimetypes
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.dependencies.auth import AuthContext, get_auth_context
from src.models.user import UserRole

router = APIRouter(tags=["stream"])


def _ensure_admin(auth_ctx: AuthContext) -> None:
    if auth_ctx.role != UserRole.ADMIN.value:
        raise HTTPException(status_code=403, detail="Access denied")


@router.get("/stream/output/{filename}")
async def stream_output_file(
    filename: str,
    request: Request,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    _ensure_admin(auth_ctx)
    file_path = f"./output/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Only video files can be streamed")

    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            yield from file_like

    media_type, _ = mimetypes.guess_type(file_path)
    return StreamingResponse(iterfile(file_path), media_type=media_type)


@router.get("/stream/uploads/{filename}")
async def stream_uploads_file(
    filename: str,
    request: Request,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    _ensure_admin(auth_ctx)
    file_path = f"./uploads/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Only video files can be streamed")

    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            yield from file_like

    media_type, _ = mimetypes.guess_type(file_path)
    return StreamingResponse(iterfile(file_path), media_type=media_type)
