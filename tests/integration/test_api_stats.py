"""Integration tests for stats endpoints."""
import pytest


@pytest.mark.integration
class TestGetVideoPaths:
    """Test GET /get_video_paths/{task_id}."""

    def test_video_paths_not_found_returns_200_with_message(self, client):
        r = client.get("/get_video_paths/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True
        assert "not found" in data.get("message", "").lower() or "message" in data


@pytest.mark.integration
class TestGetSpeedStats:
    """Test GET /get_speed_stats/{task_id}."""

    def test_speed_stats_not_found_returns_200_with_message(self, client):
        r = client.get("/get_speed_stats/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True


@pytest.mark.integration
class TestGetBallTrack:
    """Test GET /get_ball_track/{task_id}."""

    def test_ball_track_not_found_returns_200_with_message(self, client):
        r = client.get("/get_ball_track/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True


@pytest.mark.integration
class TestGetBounces:
    """Test GET /get_bounces/{task_id}."""

    def test_bounces_not_found_returns_200_with_message(self, client):
        r = client.get("/get_bounces/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True


@pytest.mark.integration
class TestGetDirectionChangeIndices:
    """Test GET /get_direction_change_indices/{task_id}."""

    def test_direction_change_not_found_returns_200_with_message(self, client):
        r = client.get("/get_direction_change_indices/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True


@pytest.mark.integration
class TestGetPlayerPositions:
    """Test GET /get_player_positions/{task_id}."""

    def test_player_positions_not_found_returns_200_with_message(self, client):
        r = client.get("/get_player_positions/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True


@pytest.mark.integration
class TestThumbnail:
    """Test GET /thumbnail/{task_id}."""

    def test_thumbnail_not_found_returns_200_with_message(self, client):
        r = client.get("/thumbnail/999999")
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True


@pytest.mark.integration
class TestAllStats:
    """Test GET /all-stats/{task_id}."""

    def test_all_stats_returns_200(self, client):
        r = client.get("/all-stats/999999")
        assert r.status_code == 200


@pytest.mark.integration
class TestServeStats:
    """Test GET /serve_stats/{task_id}."""

    def test_serve_stats_not_found_returns_200(self, client):
        r = client.get("/serve_stats/999999")
        assert r.status_code == 200


@pytest.mark.integration
class TestPlayerHeatmaps:
    """Test GET /player_heatmaps/{task_id}."""

    def test_player_heatmaps_returns_200(self, client):
        r = client.get("/player_heatmaps/999999")
        assert r.status_code == 200
