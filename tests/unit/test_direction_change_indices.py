"""Unit tests for get_direction_change_indices (src.core.get_direction_change_indices)."""
import pytest

from src.core.get_direction_change_indices import get_direction_change_indices


@pytest.mark.unit
class TestGetDirectionChangeIndices:
    """Test get_direction_change_indices(ball_track, buffer_length, slope_threshold)."""

    def test_empty_track_returns_empty_set(self):
        result = get_direction_change_indices([])
        assert result == set()

    def test_track_too_short_returns_empty_set(self):
        # Need at least 2 * buffer_length (default 15) points
        short = [(float(i), float(i)) for i in range(10)]
        result = get_direction_change_indices(short, buffer_length=5)
        assert result == set()

    def test_single_direction_no_change(self):
        # All y increasing
        track = [(float(i), float(i)) for i in range(50)]
        result = get_direction_change_indices(track, buffer_length=5)
        assert result == set()

    def test_one_direction_change_detected(self):
        # y goes up then down: peak in the middle
        y_vals = list(range(0, 25)) + list(range(25, -1, -1))  # 0..25..0
        track = [(float(i), float(y)) for i, y in enumerate(y_vals)]
        result = get_direction_change_indices(track, buffer_length=5, slope_threshold=1e-5)
        assert len(result) >= 1
        # Peak around index 25
        assert any(20 <= idx <= 30 for idx in result)

    def test_returns_set_type(self):
        track = [(float(i), float(i)) for i in range(50)]
        result = get_direction_change_indices(track, buffer_length=5)
        assert isinstance(result, set)

    def test_small_buffer_length(self):
        # Smaller buffer allows detection on shorter track
        y_vals = list(range(0, 15)) + list(range(15, -1, -1))
        track = [(float(i), float(y)) for i, y in enumerate(y_vals)]
        result = get_direction_change_indices(track, buffer_length=3, slope_threshold=1e-5)
        assert isinstance(result, set)
