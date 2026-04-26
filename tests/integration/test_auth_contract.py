import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from src.db.engine import Engine
from src.main import app
from src.models.user import User, UserRole
from src.services.jwt_service import create_access_token
from src.utils.at_tag import allocate_unique_at_tag


@pytest.mark.integration
class TestAuthContract:
    def test_sign_up_success(self):
        with TestClient(app) as client:
            response = client.post(
                "/auth/sign-up",
                json={
                    "firstName": "John",
                    "lastName": "Doe",
                    "playerHeight": 182.5,
                    "dominantHand": "right",
                    "email": "john@example.com",
                    "password": "secret123",
                    "consent": True,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["message"] == "Account created successfully"
        assert payload["data"]["atTag"] == "@john"

    def test_sign_in_success_returns_token(self):
        with TestClient(app) as client:
            client.post(
                "/auth/sign-up",
                json={
                    "firstName": "Jane",
                    "lastName": "Doe",
                    "playerHeight": 170.0,
                    "dominantHand": "left",
                    "email": "jane@example.com",
                    "password": "secret123",
                    "consent": True,
                },
            )
            response = client.post(
                "/auth/sign-in",
                json={"email": "jane@example.com", "password": "secret123"},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["message"] == "Sign in successful"
        assert payload["data"]["token"]

    def test_forgot_password_success(self):
        with TestClient(app) as client:
            response = client.post(
                "/auth/forgot-password",
                json={"email": "john@example.com"},
            )
        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "message": "Password reset link sent",
        }

    def test_refresh_token_success(self):
        with TestClient(app) as client:
            client.post(
                "/auth/sign-up",
                json={
                    "firstName": "A",
                    "lastName": "B",
                    "playerHeight": None,
                    "dominantHand": "right",
                    "email": "refresh@example.com",
                    "password": "secret123",
                    "consent": True,
                },
            )
            sign_in_response = client.post(
                "/auth/sign-in",
                json={"email": "refresh@example.com", "password": "secret123"},
            )
            token = sign_in_response.json()["data"]["token"]

            response = client.post(
                "/auth/refresh-token",
                json={"token": token},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["message"] == "Token refreshed"
        assert payload["data"]["token"]

    def test_refresh_token_invalid_returns_session_expired(self, client):
        response = client.post(
            "/auth/refresh-token",
            json={"token": "bad-token"},
        )
        assert response.status_code == 401
        assert response.json() == {
            "success": False,
            "message": "Session expired",
        }

    def test_get_profile_and_update_profile(self, client):
        profile_response = client.get("/user/profile")
        assert profile_response.status_code == 200
        profile_payload = profile_response.json()
        assert profile_payload["success"] is True
        assert profile_payload["message"] == "Profile fetched"
        assert "passwordHash" not in profile_payload.get("data", {})

        update_response = client.put(
            "/user/profile",
            json={
                "firstName": "AdminUpdated",
                "lastName": "UserUpdated",
                "playerHeight": 178.0,
                "dominantHand": "left",
            },
        )
        assert update_response.status_code == 200
        update_payload = update_response.json()
        assert update_payload["success"] is True
        assert update_payload["message"] == "Profile updated successfully"
        assert update_payload["data"]["firstName"] == "AdminUpdated"

    def test_reset_password_success(self):
        with TestClient(app) as client:
            client.post(
                "/auth/sign-up",
                json={
                    "firstName": "Reset",
                    "lastName": "User",
                    "playerHeight": 180.0,
                    "dominantHand": "right",
                    "email": "reset@example.com",
                    "password": "secret123",
                    "consent": True,
                },
            )
            sign_in_response = client.post(
                "/auth/sign-in",
                json={"email": "reset@example.com", "password": "secret123"},
            )
            token = sign_in_response.json()["data"]["token"]
            reset_response = client.post(
                "/auth/reset-password",
                json={"currentPassword": "secret123", "newPassword": "newsecret456"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert reset_response.status_code == 200
            assert reset_response.json() == {
                "success": True,
                "message": "Password updated successfully",
            }

            sign_in_new = client.post(
                "/auth/sign-in",
                json={"email": "reset@example.com", "password": "newsecret456"},
            )
            assert sign_in_new.status_code == 200


@pytest.mark.security
class TestAuthMiddlewareAndRbac:
    def test_missing_token_returns_401(self):
        with TestClient(app) as client:
            response = client.get("/user/profile")
        assert response.status_code == 401
        assert response.json() == {
            "success": False,
            "message": "Session expired",
        }

    def test_annotator_can_access_profile(self):
        with Session(Engine.instance()) as session:
            annotator = User(
                first_name="Anno",
                last_name="Tator",
                player_height=None,
                dominant_hand="right",
                email="annotator@example.com",
                consent=True,
                role=UserRole.ANNOTATOR,
            )
            annotator.set_password("secret123")
            annotator.at_tag = allocate_unique_at_tag(session, annotator.email)
            session.add(annotator)
            session.commit()
            session.refresh(annotator)
            token = create_access_token(
                user_id=annotator.id,
                role=annotator.role.value,
                email=annotator.email,
            )

        with TestClient(app) as client:
            response = client.get(
                "/user/profile",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        assert response.json().get("success") is True

    def test_regular_user_can_list_own_tasks(self):
        with Session(Engine.instance()) as session:
            regular = User(
                first_name="Reg",
                last_name="User",
                player_height=None,
                dominant_hand="right",
                email="regular@example.com",
                consent=True,
                role=UserRole.USER,
            )
            regular.set_password("secret123")
            regular.at_tag = allocate_unique_at_tag(session, regular.email)
            session.add(regular)
            session.commit()
            session.refresh(regular)
            token = create_access_token(
                user_id=regular.id,
                role=regular.role.value,
                email=regular.email,
            )

        with TestClient(app) as client:
            response = client.get(
                "/all_tasks",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload.get("success") is True
        assert isinstance(payload.get("data"), list)
