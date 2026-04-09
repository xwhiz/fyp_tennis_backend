from fastapi import Request
from fastapi.responses import JSONResponse
from jwt import ExpiredSignatureError, InvalidTokenError
from sqlmodel import Session, select

from src.db.engine import Engine
from src.models.user import User, UserRole
from src.services.jwt_service import verify_access_token
from src.utils.response import error_response

PUBLIC_ENDPOINTS = {
    ("POST", "/auth/sign-up"),
    ("POST", "/auth/sign-in"),
    ("POST", "/auth/forgot-password"),
    ("GET", "/docs"),
}

USER_ALLOWED_ENDPOINTS = {
    ("GET", "/user/profile"),
    ("PUT", "/user/profile"),
    ("POST", "/auth/reset-password"),
    ("POST", "/auth/refresh-token"),
}


def _unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content=error_response("Session expired"),
    )


def _forbidden_response() -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content=error_response("Access denied"),
    )


def is_public_route(method: str, path: str) -> bool:
    return (method.upper(), path) in PUBLIC_ENDPOINTS


def is_user_allowed_route(method: str, path: str) -> bool:
    return (method.upper(), path) in USER_ALLOWED_ENDPOINTS


async def auth_middleware(request: Request, call_next):
    if request.method.upper() == "OPTIONS":
        return await call_next(request)

    if is_public_route(request.method, request.url.path):
        return await call_next(request)

    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return _unauthorized_response()

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return _unauthorized_response()

    try:
        payload = verify_access_token(token)
    except (ExpiredSignatureError, InvalidTokenError):
        return _unauthorized_response()

    user_id = payload.get("userId")
    if not user_id:
        return _unauthorized_response()

    with Session(Engine.instance()) as session:
        user = session.exec(select(User).where(User.id == user_id)).first()

    if user is None:
        return _unauthorized_response()

    request.state.user_id = user.id
    request.state.user_email = user.email
    request.state.user_role = user.role.value if isinstance(user.role, UserRole) else str(user.role)

    role = request.state.user_role

    if role == UserRole.ANNOTATOR.value:
        return _forbidden_response()

    if role == UserRole.USER.value and not is_user_allowed_route(
        request.method,
        request.url.path,
    ):
        return _forbidden_response()

    return await call_next(request)
