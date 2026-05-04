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
class TestRallyStats:
    """Test GET /rally_stats/{task_id}."""

    def test_rally_stats_missing_task_returns_404(self, client):
        r = client.get("/rally_stats/999999")
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


@pytest.mark.integration
class TestRallyFirstStatsContract:
    def test_admin_backfill_route_converts_legacy_rows(self, client, sample_legacy_stats_task_id):
        r = client.post(f"/admin/tasks/{sample_legacy_stats_task_id}/backfill-rally-analysis")
        assert r.status_code == 200
        body = r.json()["data"]
        assert body["task_id"] == sample_legacy_stats_task_id
        assert body["schema_version"] == 1
        assert body["summary"]["total_rallies"] == 1

    def test_admin_backfill_route_forbidden_for_non_admin(self, client_regular_user, sample_legacy_stats_task_id):
        r = client_regular_user.post(f"/admin/tasks/{sample_legacy_stats_task_id}/backfill-rally-analysis")
        assert r.status_code == 403

    def test_all_stats_returns_rally_first_payload(self, client, sample_legacy_stats_task_id):
        client.post(f"/admin/tasks/{sample_legacy_stats_task_id}/backfill-rally-analysis")
        r = client.get(f"/all-stats/{sample_legacy_stats_task_id}")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["task"]["id"] == sample_legacy_stats_task_id
        assert "video" in data
        assert "rallies" in data
        assert len(data["rallies"]) == 1
        rally = data["rallies"][0]
        assert rally["rally_id"] == "rally_0"
        assert "start_time_sec" in rally and "end_time_sec" in rally
        assert rally["players"]["p1"]["role"] == "opponent"
        assert rally["players"]["p2"]["role"] == "owner"
        assert "heatmap" in rally["players"]["p1"]
        assert "ball_bounces" in rally["shared"]

    def test_serve_stats_returns_per_rally_per_player_serves(self, client, sample_legacy_stats_task_id):
        client.post(f"/admin/tasks/{sample_legacy_stats_task_id}/backfill-rally-analysis")
        r = client.get(f"/serve_stats/{sample_legacy_stats_task_id}")
        assert r.status_code == 200
        rallies = r.json()["data"]["rallies"]
        assert len(rallies) == 1
        p1_serves = rallies[0]["players"]["p1"]["serves"]
        p2_serves = rallies[0]["players"]["p2"]["serves"]
        serves = p1_serves + p2_serves
        assert len(serves) == 1
        serve = serves[0]
        assert serve["serve_type"] in {"t", "body", "corner", "wide", "bucket", "fault"}
        assert "bounce_position" in serve

    def test_get_video_paths_returns_rally_playback_ranges(self, client, sample_legacy_stats_task_id):
        client.post(f"/admin/tasks/{sample_legacy_stats_task_id}/backfill-rally-analysis")
        r = client.get(f"/get_video_paths/{sample_legacy_stats_task_id}")
        assert r.status_code == 200
        data = r.json()["data"]
        assert "video" in data
        assert len(data["rally_ranges"]) == 1
        assert data["rally_ranges"][0]["start_frame"] == 10
