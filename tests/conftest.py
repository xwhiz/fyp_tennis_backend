"""
Shared pytest fixtures for FYP Tennis Backend tests.

Uses in-memory SQLite for integration tests so no PostgreSQL is required.
Set DATABASE_URL before any src imports so settings and engine use SQLite.
"""
import os

# Must set before any src import so config and DB use in-memory SQLite
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

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
    video_paths,
)

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
def client(_db_tables):
    """FastAPI TestClient using the test DB (in-memory SQLite)."""
    from src.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_celery(_db_tables):
    """TestClient with Celery process_video_task.delay mocked (no broker/worker)."""
    from src.main import app
    with patch("src.main.process_video_task") as mock_task:
        mock_task.delay = MagicMock(return_value=MagicMock(id="mock-id"))
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_task_id(_db_tables):
    """Create a minimal background task and return its id for use in stats endpoints."""
    from sqlmodel import Session
    from src.db.engine import Engine
    from src.models.background_task import BackgroundTask
    from datetime import datetime

    with Session(Engine.instance()) as session:
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
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        return task.id
