"""Performance tests: latency and optional throughput for key endpoints."""
import time
import pytest


# Thresholds in seconds (relaxed for CI; tighten for real perf requirements)
HEALTH_MAX_SEC = 2.0
ALL_TASKS_MAX_SEC = 3.0
TASK_PROGRESS_MAX_SEC = 2.0
STATS_MAX_SEC = 3.0


@pytest.mark.performance
class TestApiLatency:
    """Assert response times for key endpoints."""

    def test_check_health_latency(self, client):
        start = time.perf_counter()
        r = client.get("/check-health")
        elapsed = time.perf_counter() - start
        assert r.status_code == 200
        assert elapsed < HEALTH_MAX_SEC, f"/check-health took {elapsed:.2f}s (max {HEALTH_MAX_SEC}s)"

    def test_all_tasks_latency(self, client):
        start = time.perf_counter()
        r = client.get("/all_tasks")
        elapsed = time.perf_counter() - start
        assert r.status_code == 200
        assert elapsed < ALL_TASKS_MAX_SEC, f"/all_tasks took {elapsed:.2f}s (max {ALL_TASKS_MAX_SEC}s)"

    def test_task_progress_latency(self, client):
        start = time.perf_counter()
        r = client.get("/task_progress/999999")
        elapsed = time.perf_counter() - start
        assert r.status_code == 200
        assert elapsed < TASK_PROGRESS_MAX_SEC, f"/task_progress took {elapsed:.2f}s (max {TASK_PROGRESS_MAX_SEC}s)"

    def test_get_ball_track_latency(self, client):
        start = time.perf_counter()
        r = client.get("/get_ball_track/999999")
        elapsed = time.perf_counter() - start
        assert r.status_code == 200
        assert elapsed < STATS_MAX_SEC, f"/get_ball_track took {elapsed:.2f}s (max {STATS_MAX_SEC}s)"
