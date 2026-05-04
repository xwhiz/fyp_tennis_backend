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

    def test_t_serve_near_center_top_box(self):
        assert classify_serve_type(800, 1180) == "t"

    def test_t_serve_near_center_bottom_box(self):
        assert classify_serve_type(860, 2320) == "t"

    def test_body_serve_along_service_line(self):
        assert classify_serve_type(700, 1150) == "body"

    def test_corner_serve_near_sideline_and_service_line(self):
        assert classify_serve_type(450, 1180) == "corner"

    def test_wide_serve_along_sideline(self):
        assert classify_serve_type(450, 1320) == "wide"

    def test_bucket_serve_inside_box_but_outside_named_targets(self):
        assert classify_serve_type(640, 1450) == "bucket"

    def test_fault_outside_service_box(self):
        assert classify_serve_type(100, 100) == "fault"

    def test_fault_for_missing_point(self):
        assert classify_serve_type(None, None) == "fault"


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
