"""Integration tests for task endpoints: all_tasks, task_progress, process_video, upload_chunk, delete_task."""
import io
import pytest
from unittest.mock import patch, MagicMock


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
        assert isinstance(data["data"], list)


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
        assert "video" in r.json().get("detail", "").lower()

    def test_process_video_duplicate_task_no_task_id_returns_400(self, client_no_celery):
        files = {"video_file": ("x.mp4", io.BytesIO(b"x"), "video/mp4")}
        data_form = {"name": "test", "total_size": "1", "duplicate_task": "true"}
        r = client_no_celery.post("/process_video", data=data_form, files=files)
        assert r.status_code == 400
        assert "task_id" in r.json().get("detail", "").lower()

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
        assert "already" in r.json().get("detail", "").lower() or "completed" in r.json().get("detail", "").lower()


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
