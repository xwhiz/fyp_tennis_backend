"""Tests for build_rally_stats (per-scene rallies)."""

from src.core.process_video import build_rally_stats


class TestBuildRallyStats:
    def test_empty_scenes(self):
        assert build_rally_stats([], {1, 2, 3}) == []

    def test_single_scene_all_bounces(self):
        scenes = [[0, 100]]
        bounces = {10, 50, 90}
        r = build_rally_stats(scenes, bounces)
        assert len(r) == 1
        assert r[0]["scene_index"] == 0
        assert r[0]["start_frame"] == 0
        assert r[0]["end_frame"] == 100
        assert r[0]["bounce_frames"] == [10, 50, 90]
        assert r[0]["serve_bounce_frame"] == 10
        assert r[0]["last_bounce_frame"] == 90
        assert r[0]["shot_count"] == 3

    def test_half_open_range_excludes_end_boundary(self):
        scenes = [[0, 50]]
        bounces = {49, 50}
        r = build_rally_stats(scenes, bounces)
        assert r[0]["bounce_frames"] == [49]
        assert r[0]["shot_count"] == 1

    def test_bounce_on_scene_start_includes(self):
        scenes = [[100, 200]]
        bounces = {100, 150}
        r = build_rally_stats(scenes, bounces)
        assert r[0]["bounce_frames"] == [100, 150]

    def test_no_bounces_in_scene(self):
        scenes = [[0, 10], [20, 30]]
        bounces = {5}
        r = build_rally_stats(scenes, bounces)
        assert r[0]["bounce_frames"] == [5]
        assert r[0]["shot_count"] == 1
        assert r[1]["bounce_frames"] == []
        assert r[1]["serve_bounce_frame"] is None
        assert r[1]["last_bounce_frame"] is None
        assert r[1]["shot_count"] == 0

    def test_multiple_scenes_splits_bounces(self):
        scenes = [[0, 100], [100, 200]]
        bounces = {10, 150, 180}
        r = build_rally_stats(scenes, bounces)
        assert r[0]["bounce_frames"] == [10]
        assert r[1]["bounce_frames"] == [150, 180]
