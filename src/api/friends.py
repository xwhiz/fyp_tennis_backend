from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, or_, select

from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.models.friend_relation import FriendRelation
from src.models.user import User
from src.schemas.friends import FriendRequestCreate
from src.utils.at_tag import display_at_tag, normalize_at_tag_input
from src.utils.profile_image import profile_image_url
from src.utils.response import success_response

router = APIRouter(prefix="/friends", tags=["friends"])


def _user_card(u: User) -> dict:
    return {
        "id": u.id,
        "firstName": u.first_name,
        "lastName": u.last_name,
        "atTag": display_at_tag(u.at_tag),
        "profileImageUrl": profile_image_url(u.profile_image_path),
    }


@router.post("/requests")
def send_friend_request(
    payload: FriendRequestCreate,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    tag = normalize_at_tag_input(payload.atTag)
    if not tag:
        raise HTTPException(status_code=400, detail="Invalid atTag")

    target = session.exec(select(User).where(User.at_tag == tag)).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    if target.id == auth_ctx.user_id:
        raise HTTPException(status_code=400, detail="Cannot send a friend request to yourself")

    existing_ab = session.exec(
        select(FriendRelation).where(
            FriendRelation.user_id == auth_ctx.user_id,
            FriendRelation.friend_id == target.id,
        ),
    ).first()
    if existing_ab is not None:
        if existing_ab.is_accepted:
            raise HTTPException(status_code=409, detail="Already friends")
        raise HTTPException(status_code=409, detail="Friend request already sent")

    existing_ba = session.exec(
        select(FriendRelation).where(
            FriendRelation.user_id == target.id,
            FriendRelation.friend_id == auth_ctx.user_id,
        ),
    ).first()
    if existing_ba is not None:
        if existing_ba.is_accepted:
            raise HTTPException(status_code=409, detail="Already friends")
        raise HTTPException(status_code=409, detail="This user already sent you a request; accept it instead")

    rel = FriendRelation(
        user_id=auth_ctx.user_id,
        friend_id=target.id,
        is_accepted=False,
    )
    session.add(rel)
    session.commit()
    session.refresh(rel)

    return success_response(
        "Friend request sent",
        {
            "relationId": rel.id,
            "friend": _user_card(target),
            "isAccepted": False,
        },
    )


@router.post("/requests/{relation_id}/accept")
def accept_friend_request(
    relation_id: int,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    rel = session.exec(select(FriendRelation).where(FriendRelation.id == relation_id)).first()
    if rel is None:
        raise HTTPException(status_code=404, detail="Request not found")

    if rel.friend_id != auth_ctx.user_id:
        raise HTTPException(status_code=403, detail="Only the recipient can accept this request")

    if rel.is_accepted:
        raise HTTPException(status_code=409, detail="Request already accepted")

    rel.is_accepted = True
    session.add(rel)
    session.commit()

    requester = session.exec(select(User).where(User.id == rel.user_id)).first()
    if requester is None:
        raise HTTPException(status_code=404, detail="User not found")

    return success_response(
        "Friend request accepted",
        {
            "relationId": rel.id,
            "friend": _user_card(requester),
            "isAccepted": True,
        },
    )


@router.get("")
def list_friends(
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    me = auth_ctx.user_id

    accepted_rows = session.exec(
        select(FriendRelation).where(
            FriendRelation.is_accepted.is_(True),
            or_(FriendRelation.user_id == me, FriendRelation.friend_id == me),
        ),
    ).all()

    accepted_friends: list[dict] = []
    seen_pairs: set[frozenset[str]] = set()
    for rel in accepted_rows:
        other_id = rel.friend_id if rel.user_id == me else rel.user_id
        pair = frozenset({rel.user_id, rel.friend_id})
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        other = session.exec(select(User).where(User.id == other_id)).first()
        if other is None:
            continue
        accepted_friends.append({"relationId": rel.id, "user": _user_card(other)})

    incoming = session.exec(
        select(FriendRelation).where(
            FriendRelation.friend_id == me,
            FriendRelation.is_accepted.is_(False),
        ),
    ).all()
    pending_incoming = []
    for rel in incoming:
        u = session.exec(select(User).where(User.id == rel.user_id)).first()
        if u is None:
            continue
        pending_incoming.append(
            {
                "relationId": rel.id,
                "user": _user_card(u),
                "createdAt": rel.created_at.isoformat() if rel.created_at else None,
            },
        )

    outgoing = session.exec(
        select(FriendRelation).where(
            FriendRelation.user_id == me,
            FriendRelation.is_accepted.is_(False),
        ),
    ).all()
    pending_outgoing = []
    for rel in outgoing:
        u = session.exec(select(User).where(User.id == rel.friend_id)).first()
        if u is None:
            continue
        pending_outgoing.append(
            {
                "relationId": rel.id,
                "user": _user_card(u),
                "createdAt": rel.created_at.isoformat() if rel.created_at else None,
            },
        )

    return success_response(
        "Friends",
        {
            "acceptedFriends": accepted_friends,
            "pendingIncoming": pending_incoming,
            "pendingOutgoing": pending_outgoing,
        },
    )
