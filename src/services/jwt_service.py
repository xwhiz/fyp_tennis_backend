from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from src.config import settings


def create_access_token(user_id: str, role: str, email: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "userId": user_id,
        "role": role,
        "email": email,
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expires_in_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def is_token_expired_or_invalid(token: str) -> bool:
    try:
        verify_access_token(token)
        return False
    except (ExpiredSignatureError, InvalidTokenError):
        return True
