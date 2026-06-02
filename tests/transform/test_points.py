"""Tests for point coordinate transformation functions."""

import numpy as np

from sleap_io.transform.points import crop_points, uncrop_points


def test_uncrop_points_basic():
    """Test mapping crop-local points back to source coordinates."""
    points = np.array([[50, 20], [100, 80]])
    result = uncrop_points(points, (100, 100, 300, 300))
    np.testing.assert_array_equal(result, [[150, 120], [200, 180]])


def test_uncrop_points_round_trip():
    """Test uncrop_points inverts crop_points exactly."""
    points = np.array([[150.5, 120.25], [200.0, 180.75], [0.0, 0.0]])
    crop = (100, 100, 300, 300)
    result = uncrop_points(crop_points(points, crop), crop)
    np.testing.assert_allclose(result, points)


def test_uncrop_points_round_trip_negative_origin():
    """Test round-trip with a negative crop origin."""
    points = np.array([[12.0, -7.0], [3.5, 4.5]])
    crop = (-40, -25, 60, 35)
    result = uncrop_points(crop_points(points, crop), crop)
    np.testing.assert_allclose(result, points)


def test_uncrop_points_round_trip_large_origin():
    """Test round-trip with a large crop origin."""
    points = np.array([[1500.0, 2400.0], [1234.5, 5678.25]])
    crop = (1000, 2000, 1640, 2480)
    result = uncrop_points(crop_points(points, crop), crop)
    np.testing.assert_allclose(result, points)


def test_uncrop_points_preserves_nan():
    """Test NaN coordinates are preserved through uncrop_points."""
    points = np.array([[50.0, 20.0], [np.nan, np.nan], [np.nan, 30.0]])
    result = uncrop_points(points, (100, 100, 300, 300))
    np.testing.assert_array_equal(result[0], [150.0, 120.0])
    assert np.isnan(result[1, 0])
    assert np.isnan(result[1, 1])
    assert np.isnan(result[2, 0])
    np.testing.assert_array_equal(result[2, 1], 130.0)


def test_uncrop_points_nan_survives_round_trip():
    """Test NaN stays NaN through crop_points then uncrop_points."""
    points = np.array([[150.0, 120.0], [np.nan, np.nan], [np.nan, 50.0]])
    crop = (100, 100, 300, 300)
    result = uncrop_points(crop_points(points, crop), crop)
    np.testing.assert_array_equal(result[0], points[0])
    assert np.isnan(result[1]).all()
    assert np.isnan(result[2, 0])
    np.testing.assert_array_equal(result[2, 1], points[2, 1])


def test_uncrop_points_preserves_shape():
    """Test uncrop_points preserves the input shape."""
    points = np.zeros((7, 2))
    result = uncrop_points(points, (10, 20, 110, 120))
    assert result.shape == points.shape


def test_uncrop_points_batched_shape():
    """Test uncrop_points works on a batched (..., 2) shape."""
    rng = np.random.default_rng(0)
    points = rng.random((4, 3, 2)) * 100.0
    crop = (15, 25, 215, 225)
    result = uncrop_points(points, crop)
    assert result.shape == points.shape
    np.testing.assert_allclose(result[..., 0], points[..., 0] + 15)
    np.testing.assert_allclose(result[..., 1], points[..., 1] + 25)
    # Round-trip on the batched shape.
    np.testing.assert_allclose(uncrop_points(crop_points(points, crop), crop), points)


def test_uncrop_points_copies_input():
    """Test uncrop_points does not mutate the input array."""
    points = np.array([[50.0, 20.0], [100.0, 80.0]])
    original = points.copy()
    uncrop_points(points, (100, 100, 300, 300))
    np.testing.assert_array_equal(points, original)
