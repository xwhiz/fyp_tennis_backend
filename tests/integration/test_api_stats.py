"""Integration tests for stats endpoints."""
import pytest


@pytest.mark.integration
class TestGetVideoPaths:
    """Test GET /get_video_paths/{task_id}."""

    def test_video_paths_missing_task_returns_404(self, client):
        r = client.get("/get_video_paths/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestGetSpeedStats:
    """Test GET /get_speed_stats/{task_id}."""

    def test_speed_stats_missing_task_returns_404(self, client):
        r = client.get("/get_speed_stats/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestGetBallTrack:
    """Test GET /get_ball_track/{task_id}."""

    def test_ball_track_missing_task_returns_404(self, client):
        r = client.get("/get_ball_track/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestGetBounces:
    """Test GET /get_bounces/{task_id}."""

    def test_bounces_missing_task_returns_404(self, client):
        r = client.get("/get_bounces/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestGetDirectionChangeIndices:
    """Test GET /get_direction_change_indices/{task_id}."""

    def test_direction_change_missing_task_returns_404(self, client):
        r = client.get("/get_direction_change_indices/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestGetPlayerPositions:
    """Test GET /get_player_positions/{task_id}."""

    def test_player_positions_missing_task_returns_404(self, client):
        r = client.get("/get_player_positions/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestThumbnail:
    """Test GET /thumbnail/{task_id}."""

    def test_thumbnail_missing_task_returns_404(self, client):
        r = client.get("/thumbnail/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestAllStats:
    """Test GET /all-stats/{task_id}."""

    def test_all_stats_missing_task_returns_404(self, client):
        r = client.get("/all-stats/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestServeStats:
    """Test GET /serve_stats/{task_id}."""

    def test_serve_stats_missing_task_returns_404(self, client):
        r = client.get("/serve_stats/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestPlayerHeatmaps:
    """Test GET /player_heatmaps/{task_id}."""

    def test_player_heatmaps_missing_task_returns_404(self, client):
        r = client.get("/player_heatmaps/999999")
        assert r.status_code == 404


@pytest.mark.integration
class TestStatsOwnership:
    """Non-admin users cannot read another user's task stats."""

    def test_get_video_paths_forbidden_for_other_owner(self, client_regular_user, sample_task_id):
        r = client_regular_user.get(f"/get_video_paths/{sample_task_id}")
        assert r.status_code == 403
