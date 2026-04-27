"""Integration tests for task endpoints: all_tasks, task_progress, process_video, upload_chunk, delete_task."""
import io
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from sqlmodel import Session, select

from src.db.engine import Engine
from src.models.background_task import BackgroundTask
from src.models.user import User


@pytest.mark.integration
class TestAllTasks:
    """Test GET /all_tasks."""

    def test_all_tasks_returns_200(self, client):
        r = client.get("/all_tasks")
        assert r.status_code == 200

    def test_all_tasks_returns_success_and_data(self, client):
        r = client.get("/all_tasks")
        data = r.json()
        assert data.get("success") is True
        assert "data" in data
        assert isinstance(data["data"]["tasks"], list)
        assert data["data"]["pagination"]["start"] == 0
        assert data["data"]["pagination"]["limit"] == 20
        assert data["data"]["pagination"]["returned"] == len(data["data"]["tasks"])

    def test_all_tasks_supports_pagination_and_latest_first_order(self, client):
        with Session(Engine.instance()) as session:
            admin = session.exec(select(User).where(User.email == "admin@example.com")).first()
            base_time = datetime.now()
            created_ids = []
            for idx in range(3):
                task = BackgroundTask(
                    progress=100.0,
                    name=f"ordered-task-{idx}",
                    status="completed",
                    video_path=f"./uploads/ordered_{idx}.mp4",
                    description="ordered",
                    total_upload_size=100,
                    uploaded_size=100,
                    is_uploaded_fully=True,
                    created_at=base_time + timedelta(minutes=idx),
                    updated_at=base_time + timedelta(minutes=idx),
                    owner_id=admin.id,
                )
                session.add(task)
                session.commit()
                session.refresh(task)
                created_ids.append(str(task.id))

        first_page = client.get("/all_tasks?start=0&limit=2")
        assert first_page.status_code == 200
        first_payload = first_page.json()["data"]
        first_ids = [task["id"] for task in first_payload["tasks"]]
        assert created_ids[-1] in first_ids
        assert created_ids[-2] in first_ids
        assert first_payload["pagination"]["start"] == 0
        assert first_payload["pagination"]["limit"] == 2
        assert first_payload["pagination"]["returned"] == len(first_payload["tasks"])
        assert first_payload["pagination"]["hasMore"] is True

        second_page = client.get("/all_tasks?start=2&limit=2")
        assert second_page.status_code == 200
        second_payload = second_page.json()["data"]
        second_ids = [task["id"] for task in second_payload["tasks"]]
        assert created_ids[-3] in second_ids
        assert second_payload["pagination"]["start"] == 2
        assert second_payload["pagination"]["limit"] == 2
        assert second_payload["pagination"]["returned"] == len(second_payload["tasks"])

    def test_all_tasks_search_supports_query_and_pagination(self, client):
        with Session(Engine.instance()) as session:
            admin = session.exec(select(User).where(User.email == "admin@example.com")).first()
            base_time = datetime.now()
            for idx in range(3):
                task = BackgroundTask(
                    progress=100.0,
                    name=f"serve-search-task-{idx}",
                    status="completed",
                    video_path=f"./uploads/search_{idx}.mp4",
                    description="searchable task",
                    total_upload_size=100,
                    uploaded_size=100,
                    is_uploaded_fully=True,
                    created_at=base_time + timedelta(minutes=idx),
                    updated_at=base_time + timedelta(minutes=idx),
                    owner_id=admin.id,
                )
                session.add(task)
            session.commit()

        response = client.get("/all_tasks/search?q=serve-search&start=0&limit=2")
        assert response.status_code == 200
        payload = response.json()["data"]
        assert len(payload["tasks"]) == 2
        assert all("serve-search" in task["name"] for task in payload["tasks"])
        assert payload["pagination"]["start"] == 0
        assert payload["pagination"]["limit"] == 2
        assert payload["pagination"]["returned"] == 2
        assert payload["pagination"]["hasMore"] is True


@pytest.mark.integration
class TestTaskProgress:
    """Test GET /task_progress/{process_id}."""

    def test_task_progress_not_found_returns_200_with_message(self, client):
        r = client.get("/task_progress/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True
        assert data.get("message") == "Process not found"

    def test_task_progress_existing_task_returns_data(self, client, sample_task_id):
        r = client.get(f"/task_progress/{sample_task_id}")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True
        assert "data" in data
        assert data["data"].get("process_id") == sample_task_id
        assert "status" in data["data"]


@pytest.mark.integration
class TestProcessVideo:
    """Test POST /process_video."""

    def test_process_video_non_video_returns_400(self, client_no_celery):
        files = {"video_file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")}
        data_form = {"name": "test", "total_size": "5", "duplicate_task": "false"}
        r = client_no_celery.post("/process_video", data=data_form, files=files)
        assert r.status_code == 400
        assert "video" in r.json().get("message", "").lower()

    def test_process_video_duplicate_task_no_task_id_returns_400(self, client_no_celery):
        files = {"video_file": ("x.mp4", io.BytesIO(b"x"), "video/mp4")}
        data_form = {"name": "test", "total_size": "1", "duplicate_task": "true"}
        r = client_no_celery.post("/process_video", data=data_form, files=files)
        assert r.status_code == 400
        assert "task_id" in r.json().get("message", "").lower()

    def test_process_video_duplicate_task_invalid_task_id_returns_404(self, client_no_celery):
        files = {"video_file": ("x.mp4", io.BytesIO(b"x"), "video/mp4")}
        data_form = {"name": "test", "total_size": "1", "duplicate_task": "true", "task_id": "999999"}
        r = client_no_celery.post("/process_video", data=data_form, files=files)
        assert r.status_code == 404

    def test_process_video_small_file_returns_200_and_process_id(self, client_no_celery):
        # Small file: single upload, no chunking
        content = b"fake mp4 content"
        files = {"video_file": ("small.mp4", io.BytesIO(content), "video/mp4")}
        data_form = {"name": "small-video", "total_size": str(len(content)), "duplicate_task": "false"}
        r = client_no_celery.post("/process_video", data=data_form, files=files)
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True
        assert "data" in data
        assert "process_id" in data["data"]
        assert data["data"].get("requires_multipart") is False

        # Avoid leaving a pending task behind; app startup requeues pending work.
        with Session(Engine.instance()) as session:
            task = session.exec(
                select(BackgroundTask).where(BackgroundTask.id == int(data["data"]["process_id"])),
            ).first()
            task.status = "completed"
            session.add(task)
            session.commit()


@pytest.mark.integration
class TestUploadChunk:
    """Test POST /upload_chunk/{task_id}."""

    def test_upload_chunk_task_not_found_returns_404(self, client):
        files = {"chunk_data": ("chunk.bin", io.BytesIO(b"x"), "application/octet-stream")}
        data_form = {"chunk_number": "0", "total_chunks": "1"}
        r = client.post("/upload_chunk/999999", data=data_form, files=files)
        assert r.status_code == 404

    def test_upload_chunk_task_already_fully_uploaded_returns_400(self, client, sample_task_id):
        files = {"chunk_data": ("chunk.bin", io.BytesIO(b"x"), "application/octet-stream")}
        data_form = {"chunk_number": "0", "total_chunks": "1"}
        r = client.post(f"/upload_chunk/{sample_task_id}", data=data_form, files=files)
        assert r.status_code == 400
        assert "already" in r.json().get("message", "").lower() or "completed" in r.json().get("message", "").lower()


@pytest.mark.integration
class TestDeleteTask:
    """Test DELETE /delete_task/{task_id}."""

    def test_delete_task_not_found_returns_404(self, client):
        r = client.delete("/delete_task/999999")
        assert r.status_code == 404

    def test_delete_task_existing_returns_200(self, client, sample_task_id):
        r = client.delete(f"/delete_task/{sample_task_id}")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True
        assert "task_id" in data.get("data", {})
