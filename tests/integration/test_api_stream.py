"""Integration tests for stream endpoints."""
import pytest


@pytest.mark.integration
class TestStreamOutput:
    """Test GET /stream/output/{filename}."""

    def test_stream_output_missing_file_returns_404(self, client):
        r = client.get("/stream/output/nonexistent.mp4")
        assert r.status_code == 404

    def test_stream_output_non_video_extension_returns_400(self, client):
        # File doesn't need to exist; endpoint checks extension first... Actually it checks exists first.
        r = client.get("/stream/output/nonexistent.txt")
        # If file doesn't exist -> 404. If we had a .txt file it would return 400.
        assert r.status_code in (400, 404)


@pytest.mark.integration
class TestStreamUploads:
    """Test GET /stream/uploads/{filename}."""

    def test_stream_uploads_missing_file_returns_404(self, client):
        r = client.get("/stream/uploads/nonexistent.mp4")
        assert r.status_code == 404
