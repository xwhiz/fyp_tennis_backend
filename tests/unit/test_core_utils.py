"""Unit tests for core utilities (src.core.utils)."""
import numpy as np
import pytest

from src.core.utils import (
    classify_serve_type,
    get_slope,
    perspective_transform_point,
)


@pytest.mark.unit
class TestClassifyServeType:
    """Test classify_serve_type(bounce_x, bounce_y)."""

    # Service box: x in [423, 1242], center_x=832, top y [1110,1748], bottom y [1748,2386]

    def test_t_serve_near_center_top_box(self):
        # Center of top service box -> T serve (normalized_dist < 0.33)
        assert classify_serve_type(832, 1400) == "t_serve"

    def test_t_serve_near_center_bottom_box(self):
        assert classify_serve_type(832, 2000) == "t_serve"

    def test_body_serve_middle_zone(self):
        # normalized_dist in [0.33, 0.67]
        assert classify_serve_type(1000, 1400) == "body_serve"

    def test_wide_serve_outer_zone(self):
        # normalized_dist > 0.67
        assert classify_serve_type(1200, 1400) == "wide_serve"

    def test_wide_serve_left_side(self):
        assert classify_serve_type(450, 1400) == "wide_serve"

    def test_unknown_outside_service_box(self):
        assert classify_serve_type(100, 100) == "unknown"

    def test_unknown_below_court(self):
        assert classify_serve_type(832, 3000) == "unknown"


@pytest.mark.unit
class TestGetSlope:
    """Test get_slope(values)."""

    def test_empty_list_returns_zero(self):
        assert get_slope([]) == 0

    def test_single_value_returns_zero(self):
        assert get_slope([1.0]) == 0

    def test_constant_values_returns_zero(self):
        slope = get_slope([5.0, 5.0, 5.0])
        assert abs(slope) < 1e-10  # np.polyfit can yield tiny non-zero due to float

    def test_positive_slope(self):
        # y = x
        slope = get_slope([0.0, 1.0, 2.0, 3.0])
        assert slope > 0
        assert abs(slope - 1.0) < 0.01

    def test_negative_slope(self):
        slope = get_slope([3.0, 2.0, 1.0, 0.0])
        assert slope < 0
        assert abs(slope - (-1.0)) < 0.01


@pytest.mark.unit
class TestPerspectiveTransformPoint:
    """Test perspective_transform_point(point, homography_matrix)."""

    def test_none_point_x_returns_point_unchanged(self):
        point = (None, 1.0)
        H = np.eye(3, dtype=np.float32)
        result = perspective_transform_point(point, H)
        assert result == point

    def test_none_matrix_returns_point_unchanged(self):
        point = (10.0, 20.0)
        result = perspective_transform_point(point, None)
        assert result == point

    def test_identity_matrix_returns_same_point(self):
        point = (100.0, 200.0)
        H = np.eye(3, dtype=np.float32)
        result = perspective_transform_point(point, H)
        np.testing.assert_array_almost_equal(result, (100.0, 200.0))
