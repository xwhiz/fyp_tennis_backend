"""Security tests: path traversal, input validation."""
import io
import pytest


@pytest.mark.security
class TestPathTraversal:
    """Prevent path traversal on stream endpoints."""

    def test_stream_output_path_traversal_returns_404_or_400(self, client):
        r = client.get("/stream/output/../../../etc/passwd")
        assert r.status_code in (400, 404)
        # Must not return 200 with sensitive content
        if r.status_code == 200:
            assert "root:" not in (r.text or "")

    def test_stream_output_encoded_traversal_returns_404_or_400(self, client):
        r = client.get("/stream/uploads/..%2F..%2F..%2Fetc%2Fpasswd")
        assert r.status_code in (400, 404)

    def test_stream_uploads_path_traversal_returns_404_or_400(self, client):
        r = client.get("/stream/uploads/../../../etc/passwd")
        assert r.status_code in (400, 404)


@pytest.mark.security
class TestInputValidation:
    """Input validation and safe status codes."""

    def test_process_video_rejects_non_video_content_type(self, client_no_celery):
        files = {"video_file": ("x.txt", io.BytesIO(b"x"), "text/plain")}
        data_form = {"name": "x", "total_size": "1", "duplicate_task": "false"}
        r = client_no_celery.post("/process_video", data=data_form, files=files)
        assert r.status_code == 400

    def test_task_progress_invalid_id_returns_200_with_message(self, client):
        # Invalid path param might 422 or 200 with "not found"
        r = client.get("/task_progress/999999")
        assert r.status_code == 200
        assert r.json().get("success") is True

    def test_delete_task_not_found_returns_404(self, client):
        r = client.delete("/delete_task/999999")
        assert r.status_code == 404

    def test_upload_chunk_not_found_returns_404(self, client):
        files = {"chunk_data": ("c.bin", io.BytesIO(b"x"), "application/octet-stream")}
        data_form = {"chunk_number": "0"}
        r = client.post("/upload_chunk/999999", data=data_form, files=files)
        assert r.status_code == 404
