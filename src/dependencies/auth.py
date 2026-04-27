from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError
from sqlmodel import select

from src.db.utils import SessionDep
from src.models.user import User, UserRole
from src.services.jwt_service import verify_access_token

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    user_id: str
    email: str
    role: str
    user: User


def resolve_auth_context_from_token(session: SessionDep, token: str) -> AuthContext:
    if not token:
        raise HTTPException(status_code=401, detail="Session expired")

    try:
        payload = verify_access_token(token)
    except (ExpiredSignatureError, InvalidTokenError):
        raise HTTPException(status_code=401, detail="Session expired")

    user_id = payload.get("userId")
    if not user_id:
        raise HTTPException(status_code=401, detail="Session expired")

    user = session.exec(select(User).where(User.id == user_id)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Session expired")

    role_value = user.role.value if isinstance(user.role, UserRole) else str(user.role)
    return AuthContext(
        user_id=user.id,
        email=user.email,
        role=role_value,
        user=user,
    )


def get_auth_context(
    session: SessionDep,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthContext:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Session expired")
    return resolve_auth_context_from_token(session, credentials.credentials)
