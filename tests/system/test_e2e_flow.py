"""End-to-end system test: create task, poll progress, fetch stats (optional; may require Celery/worker)."""
import io
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.system
class TestE2EFlow:
    """E2E flow with mocked Celery (no real worker needed)."""

    def test_create_task_then_fetch_progress_then_stats(self, client_no_celery):
        """Create task via POST /process_video, then GET task_progress, then GET a stats endpoint."""
        content = b"fake video"
        files = {"video_file": ("e2e.mp4", io.BytesIO(content), "video/mp4")}
        data_form = {"name": "e2e-video", "total_size": str(len(content)), "duplicate_task": "false"}
        r = client_no_celery.post("/process_video", data=data_form, files=files)
        assert r.status_code == 200
        process_id = r.json()["data"]["process_id"]

        r2 = client_no_celery.get(f"/task_progress/{process_id}")
        assert r2.status_code == 200
        assert r2.json().get("success") is True
        assert str(r2.json()["data"]["process_id"]) == str(process_id)

        r3 = client_no_celery.get(f"/get_ball_track/{process_id}")
        assert r3.status_code == 200
        # Stats may be "not found" until worker runs; we only check response shape
        data3 = r3.json()
        assert "success" in data3
