from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select

from src.dependencies.auth import AuthContext, get_auth_context
from src.db.utils import SessionDep
from src.models.user import User
from src.utils.at_tag import allocate_unique_at_tag, display_at_tag
from src.schemas.auth import (
    ForgotPasswordRequest,
    RefreshTokenRequest,
    ResetPasswordRequest,
    SignInRequest,
    SignUpRequest,
)
from src.services.jwt_service import create_access_token, verify_access_token
from src.utils.response import success_response

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/sign-up")
def sign_up(payload: SignUpRequest, session: SessionDep):
    if payload.consent is not True:
        raise HTTPException(status_code=400, detail="Consent must be true")

    normalized_email = payload.email.strip().lower()
    existing_user = session.exec(select(User).where(User.email == normalized_email)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")

    new_user = User(
        first_name=payload.firstName.strip(),
        last_name=payload.lastName.strip(),
        player_height=payload.playerHeight,
        dominant_hand=payload.dominantHand.strip().lower(),
        email=normalized_email,
        consent=payload.consent,
    )
    new_user.set_password(payload.password)
    new_user.at_tag = allocate_unique_at_tag(session, new_user.email)

    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return success_response(
        "Account created successfully",
        {"atTag": display_at_tag(new_user.at_tag)},
    )


@router.post("/sign-in")
def sign_in(payload: SignInRequest, session: SessionDep):
    normalized_email = payload.email.strip().lower()
    user = session.exec(select(User).where(User.email == normalized_email)).first()
    if user is None or not user.verify_password(payload.password):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    token = create_access_token(
        user_id=user.id,
        role=user.role.value,
        email=user.email,
    )
    return success_response("Sign in successful", {"token": token})


@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, session: SessionDep):
    return success_response("Password reset link sent")


@router.post("/refresh-token")
def refresh_token(
    payload: RefreshTokenRequest,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    try:
        token_payload = verify_access_token(payload.token)
    except Exception:
        raise HTTPException(status_code=401, detail="Session expired")

    if token_payload.get("userId") != auth_ctx.user_id:
        raise HTTPException(status_code=401, detail="Session expired")

    token = create_access_token(
        user_id=auth_ctx.user_id,
        role=auth_ctx.role,
        email=auth_ctx.email,
    )
    return success_response("Token refreshed", {"token": token})


@router.post("/reset-password")
def reset_password(
    payload: ResetPasswordRequest,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    user = auth_ctx.user
    if not user.verify_password(payload.currentPassword):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    user.set_password(payload.newPassword)
    session.add(user)
    session.commit()
    return success_response("Password updated successfully")
