"""System smoke tests: app starts and core endpoints respond."""
import pytest


@pytest.mark.system
class TestSmoke:
    """Smoke tests using TestClient (no external services required)."""

    def test_root_responds(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_health_responds_ok(self, client):
        r = client.get("/check-health")
        assert r.status_code == 200
        assert r.json().get("message") == "OK"

    def test_all_tasks_responds(self, client):
        r = client.get("/all_tasks")
        assert r.status_code == 200
        data = r.json()
        assert "success" in data
        assert "data" in data
