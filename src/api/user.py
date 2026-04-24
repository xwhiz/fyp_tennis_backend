import os
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.dependencies.auth import AuthContext, get_auth_context
from src.db.utils import SessionDep
from src.schemas.user import UpdateProfileRequest
from src.utils.profile_image import PROFILE_IMAGE_DIR, ensure_profile_image_dir, profile_image_url
from src.utils.response import success_response

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/profile")
def get_user_profile(
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    return success_response("Profile fetched", auth_ctx.user.to_profile_dict())


@router.put("/profile")
def update_user_profile(
    payload: UpdateProfileRequest,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    user = auth_ctx.user
    user.first_name = payload.firstName.strip()
    user.last_name = payload.lastName.strip()
    user.player_height = payload.playerHeight
    user.dominant_hand = payload.dominantHand.strip().lower()
    session.add(user)
    session.commit()
    session.refresh(user)
    return success_response("Profile updated successfully", user.to_profile_dict())


@router.post("/profile/photo")
async def upload_profile_photo(
    session: SessionDep,
    photo: UploadFile = File(...),
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    if photo.content_type is None or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await photo.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = img[y0 : y0 + side, x0 : x0 + side]
    resized = cv2.resize(cropped, (300, 300), interpolation=cv2.INTER_AREA)

    ensure_profile_image_dir()
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(PROFILE_IMAGE_DIR, filename)
    ok = cv2.imwrite(path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save image")

    user = auth_ctx.user
    user.profile_image_path = path
    session.add(user)
    session.commit()
    session.refresh(user)

    return success_response(
        "Profile photo uploaded successfully",
        {
            "profileImageUrl": profile_image_url(user.profile_image_path),
            "width": 300,
            "height": 300,
        },
    )
