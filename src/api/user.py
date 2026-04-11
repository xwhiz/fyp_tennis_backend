from fastapi import APIRouter, Depends

from src.dependencies.auth import AuthContext, get_auth_context
from src.db.utils import SessionDep
from src.schemas.user import UpdateProfileRequest
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
