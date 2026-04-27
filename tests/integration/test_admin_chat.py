import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from src.db.engine import Engine
from src.main import app
from src.models.user import User, UserRole
from src.utils.at_tag import allocate_unique_at_tag


@pytest.mark.integration
class TestAdminDashboard:
    def test_admin_login_and_dashboard_load(self):
        with TestClient(app) as client:
            login_response = client.post(
                "/admin/login",
                data={"email": "admin@example.com", "password": "admin123"},
                follow_redirects=False,
            )
            assert login_response.status_code == 303

            dashboard_response = client.get("/admin")
            assert dashboard_response.status_code == 200
            assert "RAG Dashboard" in dashboard_response.text

    def test_non_admin_login_is_rejected(self):
        with Session(Engine.instance()) as session:
            regular = User(
                first_name="Regular",
                last_name="User",
                player_height=None,
                dominant_hand="right",
                email="nonadmin@example.com",
                consent=True,
                role=UserRole.USER,
            )
            regular.set_password("password123")
            regular.at_tag = allocate_unique_at_tag(session, regular.email)
            session.add(regular)
            session.commit()

        with TestClient(app) as client:
            response = client.post(
                "/admin/login",
                data={"email": "nonadmin@example.com", "password": "password123"},
            )
            assert response.status_code == 403
            assert "Admin access is required" in response.text

    def test_pdf_upload_queues_ingestion(self):
        fake_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        with TestClient(app) as client:
            login_response = client.post(
                "/admin/login",
                data={"email": "admin@example.com", "password": "admin123"},
                follow_redirects=False,
            )
            assert login_response.status_code == 303
            with patch("src.api.admin.ingest_document_task") as mock_task:
                mock_task.delay = MagicMock()
                response = client.post(
                    "/admin/documents",
                    data={
                        "title": "ITF Rules 2026",
                        "governing_body": "ITF",
                        "competition": "Grand Slam",
                        "season_year": "2026",
                    },
                    files={"pdf_file": ("rules.pdf", io.BytesIO(fake_pdf), "application/pdf")},
                    follow_redirects=False,
                )
                assert response.status_code == 303
                assert mock_task.delay.called


@pytest.mark.integration
class TestChatApi:
    def test_chat_start_stream_and_history(self, client):
        async def fake_stream_chat(messages, image_bytes=None):
            yield {"message": {"content": "Hello "}}
            yield {"message": {"content": "from AceVision"}}

        async def fake_retrieve_context(session, *, user, question, task_id=None):
            return {
                "system_prompt": "Be concise.",
                "context_sections": ["Context block"],
                "sources": [{"type": "document", "documentId": 1}],
            }

        async def fake_refresh_user_memory(session, user, chat_session):
            user.context_summary = "Recent tennis context"
            session.add(user)
            session.commit()

        start_response = client.post("/chat/start", data={"message": "How can I improve my serve?"})
        assert start_response.status_code == 200
        start_payload = start_response.json()
        session_id = start_payload["data"]["sessionId"]
        stream_id = start_payload["data"]["streamId"]

        with (
            patch("src.api.chat.stream_chat", fake_stream_chat),
            patch("src.api.chat.retrieve_context", fake_retrieve_context),
            patch("src.api.chat.refresh_user_memory", fake_refresh_user_memory),
        ):
            stream_response = client.get(f"/chat/{stream_id}")

        assert stream_response.status_code == 200
        assert "event: started" in stream_response.text
        assert "event: retrieval" in stream_response.text
        assert '"content": "Hello "' in stream_response.text
        assert '"content": "from AceVision"' in stream_response.text
        assert "event: completed" in stream_response.text

        history_response = client.get("/chat/history")
        assert history_response.status_code == 200
        history_payload = history_response.json()
        assert len(history_payload["data"]["sessions"]) >= 1
        assert history_payload["data"]["sessions"][0]["id"] == session_id

        detail_response = client.get(f"/chat/history/{session_id}")
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert len(detail_payload["data"]["messages"]) == 2
        assert detail_payload["data"]["messages"][1]["role"] == "assistant"
