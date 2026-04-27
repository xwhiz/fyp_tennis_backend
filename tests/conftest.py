"""
Shared pytest fixtures for FYP Tennis Backend tests.

Uses in-memory SQLite for integration tests so no PostgreSQL is required.
Set DATABASE_URL before any src imports so settings and engine use SQLite.
"""
import os

# Must set before any src import so config and DB use in-memory SQLite
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET"] = "test-jwt-secret"
os.environ["ADMIN_EMAIL"] = "admin@example.com"
os.environ["ADMIN_PASSWORD"] = "admin123"
os.environ["ADMIN_FIRST_NAME"] = "Admin"
os.environ["ADMIN_LAST_NAME"] = "User"
os.environ["UPLOAD_ROOT_DIR"] = "/tmp/acevision_test_uploads"
os.environ["PROFILE_IMAGE_DIR"] = "/tmp/acevision_test_uploads/profile_images"
os.environ["KNOWLEDGE_DOCUMENT_DIR"] = "/tmp/acevision_test_uploads/knowledge_documents"
os.environ["CHAT_ATTACHMENT_DIR"] = "/tmp/acevision_test_uploads/chat_attachments"

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.db.base import Base
from src.db.session import engine

# Import all models so they are registered with Base.metadata
from src.models import (
    background_task,  # noqa: F401
    ball_track,
    bounces,
    direction_change_indices,
    player_positions,
    speed,
    thumbnail,
    user,
    video_paths,
)
import src.models.homography_matrices  # noqa: F401
import src.models.model_metrics  # noqa: F401
import src.models.player_heatmap_data  # noqa: F401
import src.models.rally_stats  # noqa: F401
import src.models.shot_annotation  # noqa: F401
import src.models.friend_relation  # noqa: F401
import src.models.chat_attachment  # noqa: F401
import src.models.chat_message  # noqa: F401
import src.models.chat_session  # noqa: F401
import src.models.chat_stream  # noqa: F401
import src.models.document_chunk  # noqa: F401
import src.models.game_stat_embedding  # noqa: F401
import src.models.knowledge_document  # noqa: F401
import src.models.system_prompt  # noqa: F401
import src.models.user_memory_entry  # noqa: F401

# Create tables once at load so app lifespan (requeue tasks) can run
Base.metadata.create_all(engine)


def _create_tables():
    Base.metadata.create_all(engine)


@pytest.fixture(scope="session")
def _db_tables():
    """Create DB tables once per test session (SQLite in-memory)."""
    _create_tables()
    yield


@pytest.fixture
def auth_headers(_db_tables):
    from sqlmodel import Session, select
    from src.db.engine import Engine
    from src.models.user import User, UserRole
    from src.services.jwt_service import create_access_token
    from src.utils.at_tag import allocate_unique_at_tag

    with Session(Engine.instance()) as session:
        test_admin = session.exec(
            select(User).where(User.email == "admin@example.com"),
        ).first()
        if test_admin is None:
            test_admin = User(
                first_name="Admin",
                last_name="User",
                player_height=None,
                dominant_hand="right",
                email="admin@example.com",
                consent=True,
                role=UserRole.ADMIN,
            )
            test_admin.set_password("admin123")
            test_admin.at_tag = allocate_unique_at_tag(session, test_admin.email)
            session.add(test_admin)
            session.commit()
            session.refresh(test_admin)

    token = create_access_token(
        user_id=test_admin.id,
        role=test_admin.role.value,
        email=test_admin.email,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(_db_tables, auth_headers):
    """FastAPI TestClient using the test DB (in-memory SQLite)."""
    from src.main import app
    with TestClient(app) as c:
        c.headers.update(auth_headers)
        yield c


@pytest.fixture
def client_regular_user(_db_tables):
    """Authenticated non-admin user (different from seeded admin)."""
    import uuid

    from sqlmodel import Session

    from src.db.engine import Engine
    from src.main import app
    from src.models.user import User, UserRole
    from src.services.jwt_service import create_access_token
    from src.utils.at_tag import allocate_unique_at_tag

    with Session(Engine.instance()) as session:
        uid = str(uuid.uuid4())
        reg = User(
            first_name="Reg",
            last_name="User",
            player_height=None,
            dominant_hand="right",
            email=f"reg_{uid}@example.com",
            consent=True,
            role=UserRole.USER,
        )
        reg.set_password("password")
        reg.at_tag = allocate_unique_at_tag(session, reg.email)
        session.add(reg)
        session.commit()
        session.refresh(reg)

    token = create_access_token(
        user_id=reg.id,
        role=reg.role.value,
        email=reg.email,
    )
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as c:
        c.headers.update(headers)
        yield c


@pytest.fixture
def client_no_celery(_db_tables, auth_headers):
    """TestClient with Celery process_video_task.delay mocked (no broker/worker)."""
    from src.main import app
    with patch("src.api.tasks.process_video_task") as mock_task:
        mock_task.delay = MagicMock(return_value=MagicMock(id="mock-id"))
        with TestClient(app) as c:
            c.headers.update(auth_headers)
            yield c


@pytest.fixture
def sample_task_id(_db_tables):
    """Create a minimal background task and return its id for use in stats endpoints."""
    from datetime import datetime

    from sqlmodel import Session, select

    from src.db.engine import Engine
    from src.models.background_task import BackgroundTask
    from src.models.user import User, UserRole
    from src.utils.at_tag import allocate_unique_at_tag

    with Session(Engine.instance()) as session:
        admin = session.exec(select(User).where(User.email == "admin@example.com")).first()
        if admin is None:
            admin = User(
                first_name="Admin",
                last_name="User",
                player_height=None,
                dominant_hand="right",
                email="admin@example.com",
                consent=True,
                role=UserRole.ADMIN,
            )
            admin.set_password("admin123")
            admin.at_tag = allocate_unique_at_tag(session, admin.email)
            session.add(admin)
            session.commit()
            session.refresh(admin)

        task = BackgroundTask(
            progress=100.0,
            name="test-task",
            status="completed",
            video_path="./uploads/test.mp4",
            description="Test",
            total_upload_size=1000,
            uploaded_size=1000,
            is_uploaded_fully=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            owner_id=admin.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        return task.id
