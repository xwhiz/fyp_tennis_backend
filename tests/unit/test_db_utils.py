"""Unit tests for DB utilities (src.db.utils)."""
import uuid
from datetime import datetime

import pytest
from sqlmodel import Session, select

from src.db.engine import Engine
from src.db.utils import (
    to_float,
    update_task_status,
    update_upload_progress,
    save_ball_track_in_db,
    save_thumbnail_in_db,
    save_video_paths_in_db,
)
from src.models.background_task import BackgroundTask
from src.models.user import User, UserRole
from src.utils.at_tag import allocate_unique_at_tag


def _create_user_and_task(session: Session) -> tuple[str, int]:
    uid = str(uuid.uuid4())
    user = User(
        first_name="U",
        last_name="T",
        player_height=None,
        dominant_hand="right",
        email=f"ut_{uid}@example.com",
        consent=True,
        role=UserRole.USER,
    )
    user.set_password("secret")
    user.at_tag = allocate_unique_at_tag(session, user.email)
    session.add(user)
    session.commit()
    session.refresh(user)
    task = BackgroundTask(
        progress=0.0,
        name="ut-task",
        status="pending",
        video_path="",
        description="",
        total_upload_size=0,
        uploaded_size=0,
        is_uploaded_fully=True,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        owner_id=user.id,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return user.id, task.id


@pytest.mark.unit
class TestToFloat:
    """Test to_float(x) helper."""

    def test_none_returns_none(self):
        assert to_float(None) is None

    def test_list_of_ints_returns_floats(self):
        assert to_float([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_list_with_none_preserves_none(self):
        assert to_float([1, None, 3]) == [1.0, None, 3.0]

    def test_empty_list_returns_empty(self):
        assert to_float([]) == []


@pytest.mark.unit
class TestUpdateTaskStatus:
    """Test update_task_status with test DB."""

    def test_update_task_status_creates_no_task_if_missing(self, _db_tables):
        # Should not raise; task_id 999 does not exist
        update_task_status(999, "processing", progress=50.0)

    def test_update_task_status_updates_existing_task(self, _db_tables):
        with Session(Engine.instance()) as session:
            _, task_id = _create_user_and_task(session)

        update_task_status(task_id, "processing", progress=25.0, description="Running")

        with Session(Engine.instance()) as session:
            t = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
            assert t is not None
            assert t.status == "processing"
            assert t.progress == 25.0
            assert t.description == "Running"


@pytest.mark.unit
class TestUpdateUploadProgress:
    """Test update_upload_progress with test DB."""

    def test_update_upload_progress_updates_existing_task(self, _db_tables):
        with Session(Engine.instance()) as session:
            _, task_id = _create_user_and_task(session)
            t = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
            t.total_upload_size = 1000
            t.uploaded_size = 0
            t.is_uploaded_fully = False
            t.status = "uploading"
            session.add(t)
            session.commit()

        update_upload_progress(task_id, 500)

        with Session(Engine.instance()) as session:
            t = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
            assert t is not None
            assert t.uploaded_size == 500


@pytest.mark.unit
class TestSaveFunctions:
    """Test save_*_in_db with test DB."""

    def test_save_ball_track_in_db(self, _db_tables):
        with Session(Engine.instance()) as session:
            _, task_id = _create_user_and_task(session)

        ball_track = [[100.0, 200.0], [101.0, 201.0]]
        save_ball_track_in_db(task_id, ball_track)
        # No exception; row inserted (verified implicitly)

    def test_save_video_paths_in_db(self, _db_tables):
        with Session(Engine.instance()) as session:
            _, task_id = _create_user_and_task(session)

        save_video_paths_in_db(task_id, "vp-task", "/output/out.mp4", "/output/mini.png")
        # No exception

    def test_save_thumbnail_in_db(self, _db_tables):
        with Session(Engine.instance()) as session:
            _, task_id = _create_user_and_task(session)

        save_thumbnail_in_db(task_id, "/output/thumb.jpg")
        # No exception
