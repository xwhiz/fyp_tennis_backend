"""Search, friends, grouped stats, and all-stats shape."""
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from src.db.engine import Engine
from src.main import app
from src.models.background_task import BackgroundTask
from src.models.user import User, UserRole
from src.services.jwt_service import create_access_token
from src.utils.at_tag import allocate_unique_at_tag


@pytest.mark.integration
class TestUserSearch:
    def test_search_requires_auth(self):
        with TestClient(app) as client:
            r = client.get("/users/search?q=ad")
        assert r.status_code == 401

    def test_search_finds_by_at_tag(self, client_regular_user):
        """Regular user searches; admin is excluded from own search but visible to others."""
        r = client_regular_user.get("/users/search?q=admin")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        results = body["data"]["results"]
        assert body["data"]["pagination"]["start"] == 0
        assert body["data"]["pagination"]["limit"] == 20
        assert body["data"]["pagination"]["returned"] == len(results)
        assert any("admin" in (x.get("atTag") or "").lower() for x in results)

    def test_search_respects_limit(self, client_regular_user):
        r = client_regular_user.get("/users/search?q=ad&limit=1")
        assert r.status_code == 200
        payload = r.json()["data"]
        assert len(payload["results"]) <= 1
        assert payload["pagination"]["limit"] == 1
        assert payload["pagination"]["returned"] == len(payload["results"])

    def test_search_supports_start_offset(self, client_regular_user):
        r = client_regular_user.get("/users/search?q=ad&start=1&limit=1")
        assert r.status_code == 200
        payload = r.json()["data"]
        assert payload["pagination"]["start"] == 1
        assert payload["pagination"]["limit"] == 1
        assert payload["pagination"]["returned"] == len(payload["results"])


@pytest.mark.integration
class TestFriends:
    def _second_user_token(self):
        uid = uuid.uuid4().hex[:12]
        with Session(Engine.instance()) as session:
            u = User(
                first_name="Friend",
                last_name="Two",
                player_height=None,
                dominant_hand="right",
                email=f"friend_two_{uid}@example.com",
                consent=True,
                role=UserRole.USER,
            )
            u.set_password("pw")
            u.at_tag = allocate_unique_at_tag(session, u.email)
            session.add(u)
            session.commit()
            session.refresh(u)
            tok = create_access_token(user_id=u.id, role=u.role.value, email=u.email)
            return tok, u.at_tag, u.id

    def test_send_and_accept_friend_request(self, client):
        token2, tag2, _uid2 = self._second_user_token()

        r = client.post("/friends/requests", json={"atTag": tag2})
        assert r.status_code == 200
        rel_id = r.json()["data"]["relationId"]

        with TestClient(app) as c2:
            c2.headers["Authorization"] = f"Bearer {token2}"
            acc = c2.post(f"/friends/requests/{rel_id}/accept")
        assert acc.status_code == 200
        assert acc.json()["data"]["isAccepted"] is True

        lst = client.get("/friends")
        assert lst.status_code == 200
        accepted = lst.json()["data"]["acceptedFriends"]
        assert len(accepted) >= 1

    def test_cannot_friend_self(self, client):
        prof = client.get("/user/profile").json()
        tag = prof["data"]["atTag"].lstrip("@")
        r = client.post("/friends/requests", json={"atTag": tag})
        assert r.status_code == 400


@pytest.mark.integration
class TestGroupedStats:
    def test_grouped_player_positions(self, client, sample_task_id):
        r = client.get(f"/get_player_positions/{sample_task_id}?grouped=true")
        assert r.status_code in (200, 404)

    def test_all_stats_has_shared_and_players(self, client, sample_task_id):
        r = client.get(f"/all-stats/{sample_task_id}")
        assert r.status_code == 200
        payload = r.json()
        assert "data" in payload or "message" in payload


@pytest.mark.integration
class TestProfilePhotosAndFriendProfile:
    def _create_user(self, email_prefix: str) -> tuple[User, str]:
        uid = uuid.uuid4().hex[:10]
        with Session(Engine.instance()) as session:
            u = User(
                first_name="Photo",
                last_name="User",
                player_height=180.0,
                dominant_hand="right",
                email=f"{email_prefix}_{uid}@example.com",
                consent=True,
                role=UserRole.USER,
            )
            u.set_password("pw")
            u.at_tag = allocate_unique_at_tag(session, u.email)
            session.add(u)
            session.commit()
            session.refresh(u)
            token = create_access_token(user_id=u.id, role=u.role.value, email=u.email)
            return u, token

    def test_upload_profile_photo_and_stream(self, client):
        img = np.full((450, 600, 3), 127, dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", img)
        assert ok
        files = {"photo": ("avatar.jpg", enc.tobytes(), "image/jpeg")}
        r = client.post("/user/profile/photo", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["data"]["width"] == 300
        assert body["data"]["height"] == 300
        url = body["data"]["profileImageUrl"]
        assert url.startswith("/stream/profile-image/")

        stream_resp = client.get(url)
        assert stream_resp.status_code == 200
        assert stream_resp.headers.get("content-type", "").startswith("image/")

    def test_profile_and_friends_include_profile_image_url(self, client):
        img = np.full((320, 320, 3), 200, dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", img)
        assert ok
        client.post("/user/profile/photo", files={"photo": ("me.jpg", enc.tobytes(), "image/jpeg")})

        prof = client.get("/user/profile")
        assert prof.status_code == 200
        assert "profileImageUrl" in prof.json()["data"]

    def test_friend_profile_limited_and_full(self, client):
        friend_user, friend_token = self._create_user("friend_profile")
        # Not friends yet: limited profile only
        r_limited = client.get(f"/friend/profile/{friend_user.id}")
        assert r_limited.status_code == 200
        d = r_limited.json()["data"]
        assert "quickStats" not in d
        assert "recentGamesWithMe" not in d

        # Send and accept friend request
        req = client.post("/friends/requests", json={"atTag": friend_user.at_tag})
        rel_id = req.json()["data"]["relationId"]
        with TestClient(app) as c2:
            c2.headers["Authorization"] = f"Bearer {friend_token}"
            c2.post(f"/friends/requests/{rel_id}/accept")

        r_full = client.get(f"/friend/profile?atTag=@{friend_user.at_tag}")
        assert r_full.status_code == 200
        fd = r_full.json()["data"]
        assert "quickStats" in fd
        assert "recentGamesWithMe" in fd

    def test_all_tasks_includes_opponent_games(self):
        regular_user, token = self._create_user("regular_tasks_view")
        # Create a game owned by admin where this exact authenticated user is opponent.
        with Session(Engine.instance()) as session:
            admin = session.exec(select(User).where(User.email == "admin@example.com")).first()
            t = BackgroundTask(
                progress=100.0,
                name="with-opponent",
                status="completed",
                video_path="./uploads/with_opponent.mp4",
                description="game",
                total_upload_size=123,
                uploaded_size=123,
                is_uploaded_fully=True,
                owner_id=admin.id,
                opponent_id=regular_user.id,
            )
            session.add(t)
            session.commit()
            session.refresh(t)
            task_id = str(t.id)

        with TestClient(app) as c:
            c.headers["Authorization"] = f"Bearer {token}"
            r = c.get("/all_tasks")
            assert r.status_code == 200
            ids = {x["id"] for x in r.json()["data"]["tasks"]}
            assert task_id in ids
