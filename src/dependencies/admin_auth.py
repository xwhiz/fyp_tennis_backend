from __future__ import annotations

from fastapi import HTTPException, Request

from src.config import settings
from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, resolve_auth_context_from_token
from src.dependencies.ownership import is_admin


def get_admin_auth_context(request: Request, session: SessionDep) -> AuthContext:
    token = request.cookies.get(settings.admin_session_cookie_name)
    if not token:
        raise HTTPException(status_code=401, detail="Admin session required")
    auth_ctx = resolve_auth_context_from_token(session, token)
    if not is_admin(auth_ctx):
        raise HTTPException(status_code=403, detail="Access denied")
    return auth_ctx
