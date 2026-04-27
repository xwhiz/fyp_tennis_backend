from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlmodel import Session, select

from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.models.user import User
from src.utils.at_tag import display_at_tag, mask_email, normalize_at_tag_input
from src.utils.pagination import pagination_metadata
from src.utils.response import success_response

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/search")
def search_users(
    session: SessionDep,
    q: str = Query(..., min_length=2),
    start: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=50),
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    term = q.strip()
    if len(term) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    pat = f"%{term}%"
    tag_norm = normalize_at_tag_input(term)

    stmt = (
        select(User)
        .where(
            or_(
                User.email.ilike(pat),
                User.at_tag.ilike(pat),
                User.first_name.ilike(pat),
                User.last_name.ilike(pat),
            ),
        )
        .where(User.id != auth_ctx.user_id)
        .limit(limit * 3)
    )
    rows = list(session.exec(stmt).all())

    def rank(u: User) -> tuple[int, str]:
        t = 3
        if tag_norm and u.at_tag == tag_norm:
            t = 0
        elif u.at_tag.startswith(tag_norm) if tag_norm else False:
            t = 1
        elif u.at_tag.lower().startswith(term.lstrip("@").lower()):
            t = 1
        full_name = f"{u.first_name} {u.last_name}".lower()
        if term.lower() in full_name and t > 2:
            t = 2
        return (t, u.at_tag)

    rows.sort(key=rank)
    total = len(rows)
    rows = rows[start : start + limit]

    results = [
        {
            "id": u.id,
            "firstName": u.first_name,
            "lastName": u.last_name,
            "atTag": display_at_tag(u.at_tag),
            "emailMasked": mask_email(u.email),
        }
        for u in rows
    ]
    return success_response(
        "Search results",
        {
            "results": results,
            "pagination": pagination_metadata(
                start=start,
                limit=limit,
                total=total,
                returned=len(results),
            ),
        },
    )
