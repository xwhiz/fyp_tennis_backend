"""Integration tests for misc endpoints: /, /check-health, /court_reference."""
import pytest


@pytest.mark.integration
class TestRootAndHealth:
    """Test root and health endpoints."""

    def test_get_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_check_health_returns_200_and_ok_message(self, client):
        r = client.get("/check-health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("message") == "OK"
        assert data.get("success") is True


@pytest.mark.integration
class TestCourtReference:
    """Test court reference endpoint."""

    def test_court_reference_returns_200(self, client):
        r = client.get("/court_reference")
        assert r.status_code == 200

    def test_court_reference_returns_structure(self, client):
        r = client.get("/court_reference")
        data = r.json()
        # Court reference returns some structure (list or dict)
        assert data is not None
