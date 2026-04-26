from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import or_, select

from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.models.background_task import BackgroundTask
from src.models.friend_relation import FriendRelation
from src.models.speed import Speed
from src.models.user import User
from src.utils.at_tag import display_at_tag, normalize_at_tag_input
from src.utils.profile_image import profile_image_url
from src.utils.response import success_response

router = APIRouter(prefix="/friend", tags=["friends"])


def _is_accepted_friend(session: SessionDep, me: str, other: str) -> bool:
    rel = session.exec(
        select(FriendRelation).where(
            FriendRelation.is_accepted.is_(True),
            or_(
                (FriendRelation.user_id == me) & (FriendRelation.friend_id == other),
                (FriendRelation.user_id == other) & (FriendRelation.friend_id == me),
            ),
        ),
    ).first()
    return rel is not None


def _basic_profile(u: User) -> dict:
    return {
        "id": u.id,
        "firstName": u.first_name,
        "lastName": u.last_name,
        "atTag": display_at_tag(u.at_tag),
        "profileImageUrl": profile_image_url(u.profile_image_path),
    }


def _recent_games_with_me(session: SessionDep, me: str, other: str) -> list[dict]:
    rows = session.exec(
        select(BackgroundTask).where(
            or_(
                (BackgroundTask.owner_id == me) & (BackgroundTask.opponent_id == other),
                (BackgroundTask.owner_id == other) & (BackgroundTask.opponent_id == me),
            ),
        ).order_by(BackgroundTask.created_at.desc()),
    ).all()
    out = []
    for t in rows[:10]:
        out.append(
            {
                "id": str(t.id),
                "name": t.name,
                "status": t.status,
                "description": t.description,
                "created_at": t.created_at,
                "updated_at": t.updated_at,
                "total_upload_size": t.total_upload_size,
                "uploaded_size": t.uploaded_size,
                "is_uploaded_fully": t.is_uploaded_fully,
                "progress": t.progress,
            },
        )
    return out


def _quick_stats(session: SessionDep, user_id: str) -> dict:
    tasks = session.exec(
        select(BackgroundTask).where(
            or_(BackgroundTask.owner_id == user_id, BackgroundTask.opponent_id == user_id),
        ),
    ).all()
    total_games = len(tasks)
    # Winner is not persisted yet; return safe placeholders.
    wins = 0
    losses = 0
    last_game_date = None
    if tasks:
        latest = max((t.updated_at or t.created_at for t in tasks if t.created_at is not None), default=None)
        if isinstance(latest, datetime):
            last_game_date = latest.isoformat()

    speeds = session.exec(select(Speed).where(Speed.owner_id == user_id)).all()
    speed_vals = []
    for s in speeds:
        payload = s.speeds if isinstance(s.speeds, dict) else {}
        for v in payload.values():
            if isinstance(v, dict) and isinstance(v.get("speed"), (float, int)):
                speed_vals.append(float(v["speed"]))
    avg_speed = round(sum(speed_vals) / len(speed_vals), 2) if speed_vals else 0.0
    return {
        "totalGames": total_games,
        "wins": wins,
        "losses": losses,
        "avgSpeed": avg_speed,
        "lastGameDate": last_game_date,
    }


@router.get("/profile")
def get_friend_profile_by_tag(
    session: SessionDep,
    atTag: str = Query(...),
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    tag = normalize_at_tag_input(atTag)
    target = session.exec(select(User).where(User.at_tag == tag)).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _friend_profile_payload(session, auth_ctx.user_id, target)


@router.get("/profile/{friend_user_id}")
def get_friend_profile_by_id(
    friend_user_id: str,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    target = session.exec(select(User).where(User.id == friend_user_id)).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _friend_profile_payload(session, auth_ctx.user_id, target)


def _friend_profile_payload(session: SessionDep, me: str, target: User):
    basic = _basic_profile(target)
    if target.id == me or _is_accepted_friend(session, me, target.id):
        full = {
            **basic,
            "playerHeight": target.player_height,
            "dominantHand": target.dominant_hand,
            "quickStats": _quick_stats(session, target.id),
            "recentGamesWithMe": _recent_games_with_me(session, me, target.id),
        }
        return success_response("Friend profile fetched", full)
    return success_response("Limited profile fetched", basic)

