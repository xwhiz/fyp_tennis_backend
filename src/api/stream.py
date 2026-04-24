import mimetypes
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.dependencies.ownership import (
    require_output_stream_path,
    require_profile_image_stream_path,
    require_upload_stream_path,
)

router = APIRouter(tags=["stream"])


@router.get("/stream/output/{filename}")
async def stream_output_file(
    filename: str,
    request: Request,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    file_path = require_output_stream_path(session, filename, auth_ctx)

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
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    file_path = require_upload_stream_path(session, filename, auth_ctx)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Only video files can be streamed")

    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            yield from file_like

    media_type, _ = mimetypes.guess_type(file_path)
    return StreamingResponse(iterfile(file_path), media_type=media_type)


@router.get("/stream/profile-image/{filename}")
async def stream_profile_image(
    filename: str,
    request: Request,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    file_path = require_profile_image_stream_path(filename, auth_ctx)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            yield from file_like

    media_type, _ = mimetypes.guess_type(file_path)
    return StreamingResponse(iterfile(file_path), media_type=media_type or "image/jpeg")
