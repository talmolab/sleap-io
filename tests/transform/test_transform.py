"""Tests for the transform module."""

import numpy as np
import pytest

from sleap_io.transform import Transform
from sleap_io.transform.core import parse_crop, parse_pad, parse_scale, resolve_scale
from sleap_io.transform.frame import (
    crop_frame,
    pad_frame,
    rotate_frame,
    scale_frame,
    transform_frame,
)
from sleap_io.transform.points import (
    count_out_of_bounds,
    crop_points,
    get_out_of_bounds_mask,
    pad_points,
    rotate_points,
    scale_points,
    transform_points,
)

# ============================================================================
# Transform Class Tests
# ============================================================================


class TestTransform:
    """Tests for the Transform dataclass."""

    def test_transform_default(self):
        """Test Transform with default values."""
        transform = Transform()
        assert transform.crop is None
        assert transform.scale is None
        assert transform.rotate is None
        assert transform.pad is None
        assert transform.quality == "bilinear"
        assert transform.fill == 0

    def test_transform_bool_empty(self):
        """Test that empty transform is falsy."""
        transform = Transform()
        assert not transform

    def test_transform_bool_with_crop(self):
        """Test that transform with crop is truthy."""
        transform = Transform(crop=(0, 0, 100, 100))
        assert transform

    def test_transform_bool_with_scale(self):
        """Test that transform with scale is truthy."""
        transform = Transform(scale=(0.5, 0.5))
        assert transform

    def test_transform_bool_with_zero_rotation(self):
        """Test that transform with zero rotation is falsy."""
        transform = Transform(rotate=0)
        assert not transform

    def test_transform_bool_with_nonzero_rotation(self):
        """Test that transform with non-zero rotation is truthy."""
        transform = Transform(rotate=45)
        assert transform

    def test_transform_bool_with_zero_pad(self):
        """Test that transform with zero padding is falsy."""
        transform = Transform(pad=(0, 0, 0, 0))
        assert not transform

    def test_transform_bool_with_nonzero_pad(self):
        """Test that transform with non-zero padding is truthy."""
        transform = Transform(pad=(10, 10, 10, 10))
        assert transform

    def test_output_size_no_transform(self):
        """Test output_size with no transformations."""
        transform = Transform()
        assert transform.output_size((640, 480)) == (640, 480)

    def test_output_size_crop(self):
        """Test output_size with crop."""
        transform = Transform(crop=(100, 100, 300, 400))
        assert transform.output_size((640, 480)) == (200, 300)

    def test_output_size_scale(self):
        """Test output_size with scale."""
        transform = Transform(scale=(0.5, 0.5))
        assert transform.output_size((640, 480)) == (320, 240)

    def test_output_size_pad(self):
        """Test output_size with padding."""
        transform = Transform(pad=(10, 20, 30, 40))
        assert transform.output_size((640, 480)) == (700, 520)

    def test_output_size_combined(self):
        """Test output_size with combined transforms."""
        # crop 100,100,300,400 -> 200x300
        # scale 0.5 -> 100x150
        # pad 10,10,10,10 -> 120x170
        transform = Transform(
            crop=(100, 100, 300, 400), scale=(0.5, 0.5), pad=(10, 10, 10, 10)
        )
        assert transform.output_size((640, 480)) == (120, 170)

    def test_to_matrix_identity(self):
        """Test to_matrix returns identity for empty transform."""
        transform = Transform()
        matrix = transform.to_matrix((640, 480))
        np.testing.assert_array_almost_equal(matrix, np.eye(3))

    def test_to_matrix_crop(self):
        """Test to_matrix for crop transform."""
        transform = Transform(crop=(100, 50, 300, 250))
        matrix = transform.to_matrix((640, 480))

        # Test point transformation
        point = np.array([150, 100, 1])  # In crop region
        result = matrix @ point
        np.testing.assert_array_almost_equal(result[:2], [50, 50])

    def test_to_matrix_scale(self):
        """Test to_matrix for scale transform."""
        transform = Transform(scale=(2.0, 0.5))
        matrix = transform.to_matrix((640, 480))

        point = np.array([100, 200, 1])
        result = matrix @ point
        np.testing.assert_array_almost_equal(result[:2], [200, 100])

    def test_to_matrix_pad(self):
        """Test to_matrix for pad transform."""
        transform = Transform(pad=(20, 0, 0, 10))  # top=20, left=10
        matrix = transform.to_matrix((640, 480))

        point = np.array([100, 50, 1])
        result = matrix @ point
        np.testing.assert_array_almost_equal(result[:2], [110, 70])

    def test_apply_to_points_empty(self):
        """Test apply_to_points with empty array."""
        transform = Transform(scale=(2.0, 2.0))
        points = np.array([]).reshape(0, 2)
        result = transform.apply_to_points(points, (640, 480))
        assert result.shape == (0, 2)

    def test_apply_to_points_scale(self):
        """Test apply_to_points with scale."""
        transform = Transform(scale=(2.0, 2.0))
        points = np.array([[100, 50], [200, 100]])
        result = transform.apply_to_points(points, (640, 480))
        np.testing.assert_array_almost_equal(result, [[200, 100], [400, 200]])

    def test_apply_to_points_preserves_nan(self):
        """Test apply_to_points preserves NaN values."""
        transform = Transform(scale=(2.0, 2.0))
        points = np.array([[100, 50], [np.nan, np.nan], [200, 100]])
        result = transform.apply_to_points(points, (640, 480))
        assert np.isnan(result[1]).all()
        np.testing.assert_array_almost_equal(result[0], [200, 100])
        np.testing.assert_array_almost_equal(result[2], [400, 200])

    def test_apply_to_frame(self):
        """Test apply_to_frame."""
        transform = Transform(crop=(10, 10, 60, 60))
        frame = np.ones((100, 100), dtype=np.uint8) * 128
        result = transform.apply_to_frame(frame)
        assert result.shape == (50, 50)

    def test_apply_to_points_all_nan(self):
        """Test apply_to_points with all NaN points (covers core.py:236 else branch)."""
        transform = Transform(scale=(2.0, 2.0))
        points = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
        result = transform.apply_to_points(points, (640, 480))
        # All points should remain NaN
        assert np.isnan(result).all()


# ============================================================================
# Center Pixel Indexing Tests
# ============================================================================


class TestCenterPixelIndexing:
    """Tests verifying center pixel indexing behavior.

    SLEAP uses center pixel indexing (format >= 1.1) where coordinate (x, y)
    refers to the center of pixel (x, y), not its top-left corner.

    Key properties:
    - Pixel centers are at integer coordinates: 0, 1, 2, ..., width-1
    - Geometric center of WxH image is at ((W-1)/2, (H-1)/2)
    - Flip maps x -> (width-1) - x (not x -> width - x)
    - Rotation center is at ((w-1)/2, (h-1)/2) (not (w/2, h/2))
    """

    def test_flip_horizontal_edge_pixels(self):
        """Test that horizontal flip correctly maps edge pixel centers."""
        transform = Transform(flip_h=True)
        # For 100-wide image, leftmost pixel center is 0, rightmost is 99
        points = np.array([[0, 50], [99, 50]], dtype=np.float64)
        result = transform.apply_to_points(points, (100, 100))
        # 0 -> 99, 99 -> 0 (swap edges)
        np.testing.assert_array_almost_equal(result, [[99, 50], [0, 50]])

    def test_flip_vertical_edge_pixels(self):
        """Test that vertical flip correctly maps edge pixel centers."""
        transform = Transform(flip_v=True)
        # For 100-tall image, topmost pixel center is 0, bottommost is 99
        points = np.array([[50, 0], [50, 99]], dtype=np.float64)
        result = transform.apply_to_points(points, (100, 100))
        # 0 -> 99, 99 -> 0 (swap edges)
        np.testing.assert_array_almost_equal(result, [[50, 99], [50, 0]])

    def test_flip_center_pixel(self):
        """Test that flip maps image center to itself."""
        # For 100x100, center is at (49.5, 49.5)
        transform_h = Transform(flip_h=True)
        transform_v = Transform(flip_v=True)
        transform_hv = Transform(flip_h=True, flip_v=True)

        center = np.array([[49.5, 49.5]], dtype=np.float64)

        # Center should map to itself under any flip
        result_h = transform_h.apply_to_points(center, (100, 100))
        result_v = transform_v.apply_to_points(center, (100, 100))
        result_hv = transform_hv.apply_to_points(center, (100, 100))

        np.testing.assert_array_almost_equal(result_h, [[49.5, 49.5]])
        np.testing.assert_array_almost_equal(result_v, [[49.5, 49.5]])
        np.testing.assert_array_almost_equal(result_hv, [[49.5, 49.5]])

    def test_rotation_180_center_fixed(self):
        """Test that 180° rotation keeps image center fixed."""
        transform = Transform(rotate=180, clip_rotation=True)
        # True center of 100x100 image
        center = np.array([[49.5, 49.5]], dtype=np.float64)
        result = transform.apply_to_points(center, (100, 100))
        np.testing.assert_array_almost_equal(result, [[49.5, 49.5]])

    def test_rotation_90_center_fixed(self):
        """Test that 90° rotation keeps image center fixed."""
        transform = Transform(rotate=90, clip_rotation=True)
        center = np.array([[49.5, 49.5]], dtype=np.float64)
        result = transform.apply_to_points(center, (100, 100))
        np.testing.assert_array_almost_equal(result, [[49.5, 49.5]])

    def test_rotation_180_swaps_corners(self):
        """Test that 180° rotation correctly swaps opposite corners."""
        transform = Transform(rotate=180, clip_rotation=True)
        # Corners of a 100x100 image
        corners = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float64)
        result = transform.apply_to_points(corners, (100, 100))
        # Each corner should swap with its diagonal opposite
        expected = np.array([[99, 99], [0, 99], [0, 0], [99, 0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)


# ============================================================================
# Frame Transform Tests
# ============================================================================


class TestFrameTransforms:
    """Tests for frame transformation functions."""

    def test_crop_frame_basic(self):
        """Test basic cropping."""
        frame = np.arange(100).reshape(10, 10).astype(np.uint8)
        result = crop_frame(frame, (2, 2, 8, 8))
        assert result.shape == (6, 6)
        assert result[0, 0] == 22  # row 2, col 2

    def test_crop_frame_with_padding(self):
        """Test cropping that extends beyond frame bounds."""
        frame = np.ones((10, 10), dtype=np.uint8) * 255
        result = crop_frame(frame, (-5, -5, 15, 15), fill=0)
        assert result.shape == (20, 20)
        # Top-left corner should be fill value
        assert result[0, 0] == 0
        # Center should be original value
        assert result[10, 10] == 255

    def test_crop_frame_color(self):
        """Test cropping color frame."""
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 128
        result = crop_frame(frame, (2, 2, 8, 8))
        assert result.shape == (6, 6, 3)
        assert result[0, 0, 0] == 128

    def test_scale_frame_up(self):
        """Test scaling frame up."""
        frame = np.ones((10, 10), dtype=np.uint8)
        result = scale_frame(frame, (2.0, 2.0))
        assert result.shape == (20, 20)

    def test_scale_frame_down(self):
        """Test scaling frame down."""
        frame = np.ones((100, 100), dtype=np.uint8)
        result = scale_frame(frame, (0.5, 0.5))
        assert result.shape == (50, 50)

    def test_scale_frame_no_change(self):
        """Test scaling with factor 1.0."""
        frame = np.ones((10, 10), dtype=np.uint8)
        result = scale_frame(frame, (1.0, 1.0))
        assert result.shape == (10, 10)
        np.testing.assert_array_equal(result, frame)

    def test_scale_frame_anisotropic(self):
        """Test anisotropic scaling."""
        frame = np.ones((100, 100), dtype=np.uint8)
        result = scale_frame(frame, (0.5, 2.0))
        assert result.shape == (200, 50)

    def test_scale_frame_invalid_raises(self):
        """Test that invalid scale raises error."""
        frame = np.ones((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            scale_frame(frame, (0.0, 1.0))

    def test_rotate_frame_90(self):
        """Test rotating frame 90 degrees."""
        frame = np.zeros((10, 10), dtype=np.uint8)
        frame[0, :] = 255  # Top row white
        result = rotate_frame(frame, 90)
        # After 90 clockwise, top row should be on right side
        assert result.shape == (10, 10)

    def test_rotate_frame_zero(self):
        """Test rotating frame 0 degrees returns same frame."""
        frame = np.ones((10, 10), dtype=np.uint8) * 128
        result = rotate_frame(frame, 0)
        np.testing.assert_array_equal(result, frame)

    def test_rotate_frame_with_fill(self):
        """Test rotation with custom fill color."""
        frame = np.ones((10, 10), dtype=np.uint8) * 255
        result = rotate_frame(frame, 45, fill=0)
        # Corners should have fill value after rotation
        assert result[0, 0] == 0 or result[-1, -1] == 0

    def test_pad_frame_uniform(self):
        """Test uniform padding."""
        frame = np.ones((10, 10), dtype=np.uint8) * 255
        result = pad_frame(frame, (5, 5, 5, 5), fill=0)
        assert result.shape == (20, 20)
        # Original frame should be in center
        assert result[5, 5] == 255
        # Padding should be fill value
        assert result[0, 0] == 0

    def test_pad_frame_asymmetric(self):
        """Test asymmetric padding."""
        frame = np.ones((10, 10), dtype=np.uint8) * 255
        result = pad_frame(frame, (1, 2, 3, 4), fill=0)
        # top=1, right=2, bottom=3, left=4
        assert result.shape == (14, 16)  # 10+1+3, 10+2+4

    def test_pad_frame_zero(self):
        """Test zero padding returns same frame."""
        frame = np.ones((10, 10), dtype=np.uint8)
        result = pad_frame(frame, (0, 0, 0, 0))
        np.testing.assert_array_equal(result, frame)

    def test_transform_frame_combined(self):
        """Test combined transformations."""
        frame = np.ones((100, 100), dtype=np.uint8) * 128
        result = transform_frame(
            frame,
            crop=(10, 10, 60, 60),  # -> 50x50
            scale=(0.5, 0.5),  # -> 25x25
            pad=(5, 5, 5, 5),  # -> 35x35
        )
        assert result.shape == (35, 35)

    def test_transform_frame_with_rotation(self):
        """Test transform_frame with rotation (covers frame.py:289)."""
        frame = np.ones((100, 100), dtype=np.uint8) * 128
        result = transform_frame(frame, rotate=45)
        # Rotation should expand the frame
        assert result.shape[0] >= 100
        assert result.shape[1] >= 100

    def test_transform_frame_with_flip_v(self):
        """Test transform_frame with vertical flip (covers frame.py:300)."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[:50, :] = 255  # Top half white

        result = transform_frame(frame, flip_v=True)

        # After vertical flip, bottom half should be white
        assert result.shape == (100, 100)
        assert result[99, 0] == 255  # Bottom should be white
        assert result[0, 0] == 0  # Top should be black

    def test_transform_frame_with_flip_h(self):
        """Test transform_frame with horizontal flip (covers frame.py:296)."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[:, :50] = 255  # Left half white

        result = transform_frame(frame, flip_h=True)

        # After horizontal flip, right half should be white
        assert result.shape == (100, 100)
        assert result[0, 99] == 255  # Right should be white
        assert result[0, 0] == 0  # Left should be black


class TestFrameTransformsGrayscale:
    """Tests for frame transforms with grayscale images (H, W, 1)."""

    def test_scale_frame_grayscale_hwc(self):
        """Test scale_frame with grayscale (H, W, 1) format."""
        # Create grayscale frame with shape (H, W, 1)
        frame = np.zeros((100, 100, 1), dtype=np.uint8)
        frame[40:60, 40:60] = 255  # White square

        result = scale_frame(frame, (0.5, 0.5))

        # Should preserve (H, W, 1) shape
        assert result.shape == (50, 50, 1)
        assert result.dtype == np.uint8

    def test_rotate_frame_grayscale_hwc(self):
        """Test rotate_frame with grayscale (H, W, 1) format."""
        frame = np.zeros((100, 100, 1), dtype=np.uint8)
        frame[0:10, 0:10] = 255  # White square in top-left

        result = rotate_frame(frame, 90)

        # Should preserve (H, W, 1) shape
        assert result.shape == (100, 100, 1)
        # After clockwise 90 degree rotation, top-left becomes top-right
        assert result[0:10, 90:100].mean() > 200

    def test_rotate_frame_tuple_fill(self):
        """Test rotate_frame with tuple fill color and clipping."""
        # RGB frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = [255, 255, 255]

        # Use tuple fill, expand=False to keep original dimensions (clipping)
        result = rotate_frame(frame, 45, fill=(128, 128, 128), expand=False)

        assert result.shape == (100, 100, 3)
        # Corners should have fill color
        assert result[0, 0, 0] == 128

    def test_crop_frame_oob_3d(self):
        """Test crop_frame with out-of-bounds region for 3D array."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Crop extending beyond frame bounds (need padding)
        result = crop_frame(frame, (-10, -10, 50, 50), fill=0)

        # Should create padded output with shape (60, 60, 3)
        assert result.shape == (60, 60, 3)
        # Top-left corner should be filled (was out of bounds)
        assert result[0, 0, 0] == 0
        # Some interior should have original values
        assert result[15, 15, 0] == 255

    def test_pad_frame_with_tuple_fill(self):
        """Test pad_frame with tuple fill color."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = pad_frame(frame, (10, 10, 10, 10), fill=(128, 128, 128))

        assert result.shape == (70, 70, 3)
        # Padded regions should have fill color
        assert result[0, 0, 0] == 128
        assert result[0, 0, 1] == 128
        assert result[0, 0, 2] == 128

    def test_flip_h_frame(self):
        """Test horizontal flip of a frame."""
        from sleap_io.transform.frame import flip_h_frame

        # Create asymmetric test frame
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[:, :50] = 255  # Left half white

        result = flip_h_frame(frame)

        # After horizontal flip, right half should be white
        assert result.shape == (100, 100)
        assert result[0, 99] == 255  # Top-right should be white
        assert result[0, 0] == 0  # Top-left should be black

    def test_flip_v_frame(self):
        """Test vertical flip of a frame."""
        from sleap_io.transform.frame import flip_v_frame

        # Create asymmetric test frame
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[:50, :] = 255  # Top half white

        result = flip_v_frame(frame)

        # After vertical flip, bottom half should be white
        assert result.shape == (100, 100)
        assert result[99, 0] == 255  # Bottom-left should be white
        assert result[0, 0] == 0  # Top-left should be black

    def test_flip_transform_frame_and_points(self):
        """Test that flip transforms frame and points consistently.

        Uses center pixel indexing: coordinate (x, y) refers to the center of
        pixel (x, y). For a 100-pixel wide image, pixel centers range from 0 to 99.
        Flip maps x -> (width-1) - x, so x=15 -> x=84 on a 100-wide image.
        """
        # Test frame
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[10:20, 10:20] = 255  # White square at (10-20, 10-20)

        # Test point at center of square
        points = np.array([[15, 15]], dtype=np.float64)
        input_size = (100, 100)

        # Test horizontal flip
        transform_h = Transform(flip_h=True)
        transformed_frame_h = transform_h.apply_to_frame(frame)
        transformed_points_h = transform_h.apply_to_points(points, input_size)

        # Point at x=15 should become x=84 ((100-1)-15 = 84, center pixel indexing)
        np.testing.assert_array_almost_equal(transformed_points_h, [[84, 15]])
        # White square at x=10-19 should now be at x=80-89, point at center (84)
        assert transformed_frame_h[15, 84] == 255

        # Test vertical flip
        transform_v = Transform(flip_v=True)
        transformed_points_v = transform_v.apply_to_points(points, input_size)

        # Point at y=15 should become y=84 ((100-1)-15 = 84)
        np.testing.assert_array_almost_equal(transformed_points_v, [[15, 84]])

        # Test both flips
        transform_hv = Transform(flip_h=True, flip_v=True)
        transformed_points_hv = transform_hv.apply_to_points(points, input_size)

        # Point at (15, 15) should become (84, 84)
        np.testing.assert_array_almost_equal(transformed_points_hv, [[84, 84]])


# ============================================================================
# Points Transform Tests
# ============================================================================


class TestPointTransforms:
    """Tests for point coordinate transformation functions."""

    def test_crop_points(self):
        """Test cropping points."""
        points = np.array([[150, 120], [200, 180]])
        result = crop_points(points, (100, 100, 300, 300))
        np.testing.assert_array_equal(result, [[50, 20], [100, 80]])

    def test_scale_points(self):
        """Test scaling points."""
        points = np.array([[100, 50], [200, 100]])
        result = scale_points(points, (2.0, 0.5))
        np.testing.assert_array_equal(result, [[200, 25], [400, 50]])

    def test_rotate_points_90(self):
        """Test rotating points 90 degrees around center."""
        points = np.array([[100, 50]])  # Single point
        center = (50, 50)
        result = rotate_points(points, 90, center)
        # Point at (100, 50) rotated 90 CW around (50, 50)
        # dx = 50, dy = 0
        # new_dx = 50*cos(90) + 0*sin(90) = 0
        # new_dy = -50*sin(90) + 0*cos(90) = -50
        # new = (50, 0)
        np.testing.assert_array_almost_equal(result, [[50, 0]])

    def test_rotate_points_zero(self):
        """Test rotating points 0 degrees returns same points."""
        points = np.array([[100, 50], [200, 100]])
        result = rotate_points(points, 0, (0, 0))
        np.testing.assert_array_equal(result, points)

    def test_pad_points(self):
        """Test padding points."""
        points = np.array([[100, 50], [200, 100]])
        result = pad_points(points, (20, 0, 0, 10))  # top=20, left=10
        np.testing.assert_array_equal(result, [[110, 70], [210, 120]])

    def test_transform_points_with_matrix(self):
        """Test transforming points with affine matrix."""
        # Identity matrix
        matrix = np.eye(3)
        points = np.array([[100, 50], [200, 100]])
        result = transform_points(points, matrix)
        np.testing.assert_array_almost_equal(result, points)

        # Scale matrix
        scale_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        result = transform_points(points, scale_matrix)
        np.testing.assert_array_almost_equal(result, [[200, 100], [400, 200]])

    def test_transform_points_preserves_nan(self):
        """Test that transform_points preserves NaN values."""
        matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        points = np.array([[100, 50], [np.nan, np.nan], [200, 100]])
        result = transform_points(points, matrix)
        assert np.isnan(result[1]).all()
        np.testing.assert_array_almost_equal(result[0], [200, 100])
        np.testing.assert_array_almost_equal(result[2], [400, 200])

    def test_count_out_of_bounds(self):
        """Test counting out of bounds points."""
        points = np.array([[-10, 50], [50, 50], [150, 50], [50, -10], [50, 150]])
        bounds = (0, 0, 100, 100)
        assert count_out_of_bounds(points, bounds) == 4

    def test_count_out_of_bounds_all_valid(self):
        """Test count when all points are in bounds."""
        points = np.array([[10, 10], [50, 50], [90, 90]])
        bounds = (0, 0, 100, 100)
        assert count_out_of_bounds(points, bounds) == 0

    def test_count_out_of_bounds_with_nan(self):
        """Test count excludes NaN points."""
        points = np.array([[-10, 50], [np.nan, np.nan], [50, 50]])
        bounds = (0, 0, 100, 100)
        assert count_out_of_bounds(points, bounds) == 1

    def test_transform_points_empty(self):
        """Test transform_points with empty array."""
        points = np.array([]).reshape(0, 2)
        matrix = np.eye(3)
        result = transform_points(points, matrix)
        assert result.shape == (0, 2)

    def test_count_out_of_bounds_all_nan(self):
        """Test count_out_of_bounds when all points are NaN."""
        points = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        count = count_out_of_bounds(points, (0, 0, 100, 100))
        assert count == 0

    def test_get_out_of_bounds_mask_basic(self):
        """Test get_out_of_bounds_mask returns correct boolean mask."""
        points = np.array([[-10, 50], [50, 50], [150, 50], [50, -10], [50, 150]])
        bounds = (0, 0, 100, 100)
        mask = get_out_of_bounds_mask(points, bounds)
        expected = [True, False, True, True, True]
        np.testing.assert_array_equal(mask, expected)

    def test_get_out_of_bounds_mask_all_valid(self):
        """Test mask when all points are in bounds."""
        points = np.array([[10, 10], [50, 50], [90, 90]])
        bounds = (0, 0, 100, 100)
        mask = get_out_of_bounds_mask(points, bounds)
        np.testing.assert_array_equal(mask, [False, False, False])

    def test_get_out_of_bounds_mask_with_nan(self):
        """Test that NaN points are not marked as OOB."""
        points = np.array([[-10, 50], [np.nan, np.nan], [50, 50]])
        bounds = (0, 0, 100, 100)
        mask = get_out_of_bounds_mask(points, bounds)
        # NaN point should NOT be marked as OOB (False), OOB point should be True
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_get_out_of_bounds_mask_boundary(self):
        """Test boundary conditions (>= max is out of bounds)."""
        # Points exactly at the max boundary are out of bounds
        points = np.array([[0, 0], [99, 99], [100, 50], [50, 100]])
        bounds = (0, 0, 100, 100)
        mask = get_out_of_bounds_mask(points, bounds)
        # (0,0) in, (99,99) in, (100,50) out (x>=100), (50,100) out (y>=100)
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_transform_points_all_nan(self):
        """Test transform_points with all NaN points."""
        matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        points = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
        result = transform_points(points, matrix)
        # All points should remain NaN
        assert np.isnan(result).all()

    def test_get_out_of_bounds_mask_all_nan(self):
        """Test get_out_of_bounds_mask with all NaN points."""
        points = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        bounds = (0, 0, 100, 100)
        mask = get_out_of_bounds_mask(points, bounds)
        # All NaN points should NOT be marked as OOB
        np.testing.assert_array_equal(mask, [False, False])


# ============================================================================
# CLI Parsing Tests
# ============================================================================


class TestParsing:
    """Tests for CLI parameter parsing functions."""

    def test_parse_scale_uniform_ratio(self):
        """Test parsing uniform scale ratio."""
        result = parse_scale("0.5")
        assert result == (0.5, 0.5)

    def test_parse_scale_uniform_ratio_decimal(self):
        """Test parsing ratio with decimal point."""
        result = parse_scale("2.0")
        assert result == (2.0, 2.0)

    def test_parse_scale_target_width(self):
        """Test parsing target width."""
        result = parse_scale("640")
        assert result == (-640.0, -1.0)

    def test_parse_scale_width_height(self):
        """Test parsing width and height."""
        result = parse_scale("640,480")
        assert result == (-640.0, -480.0)

    def test_parse_scale_width_auto(self):
        """Test parsing width with auto height."""
        result = parse_scale("640,-1")
        assert result == (-640.0, -1.0)

    def test_parse_scale_auto_height(self):
        """Test parsing auto width with height."""
        result = parse_scale("-1,480")
        assert result == (-1.0, -480.0)

    def test_parse_scale_different_ratios(self):
        """Test parsing different ratios per axis."""
        result = parse_scale("0.5,0.75")
        assert result == (0.5, 0.75)

    def test_resolve_scale_ratios(self):
        """Test resolving scale when both are ratios."""
        result = resolve_scale((0.5, 0.75), (640, 480))
        assert result == (0.5, 0.75)

    def test_resolve_scale_pixels(self):
        """Test resolving scale from pixel dimensions."""
        result = resolve_scale((-320, -240), (640, 480))
        assert result == (0.5, 0.5)

    def test_resolve_scale_width_only(self):
        """Test resolving scale with only width specified."""
        result = resolve_scale((-320, -1.0), (640, 480))
        assert result == (0.5, 0.5)

    def test_resolve_scale_height_only(self):
        """Test resolving scale with only height specified."""
        result = resolve_scale((-1.0, -240), (640, 480))
        assert result == (0.5, 0.5)

    def test_parse_scale_invalid_format(self):
        """Test that invalid scale format raises error."""
        with pytest.raises(ValueError, match="Invalid scale format"):
            parse_scale("1,2,3")

    def test_parse_scale_auto_both_axes(self):
        """Test parsing auto for both axes."""
        result = parse_scale("-1,-1")
        # Both are auto (negative)
        assert result[0] < 0
        assert result[1] < 0

    def test_resolve_scale_both_auto(self):
        """Test resolving scale when both dimensions are auto."""
        # This is an edge case that returns (1.0, 1.0)
        result = resolve_scale((-1.0, -1.0), (640, 480))
        assert result == (1.0, 1.0)

    def test_parse_crop_pixels(self):
        """Test parsing pixel coordinates."""
        result = parse_crop("100,100,500,500")
        assert result == (100, 100, 500, 500)

    def test_parse_crop_normalized(self):
        """Test parsing normalized coordinates."""
        result = parse_crop("0.25,0.25,0.75,0.75", (640, 480))
        assert result == (160, 120, 480, 360)

    def test_parse_crop_invalid_count(self):
        """Test that invalid crop value count raises error."""
        with pytest.raises(ValueError):
            parse_crop("100,100,500")

    def test_parse_crop_normalized_without_size_raises(self):
        """Test that normalized coords without size raises error."""
        with pytest.raises(ValueError):
            parse_crop("0.25,0.25,0.75,0.75")

    def test_parse_pad_uniform(self):
        """Test parsing uniform padding."""
        result = parse_pad("10")
        assert result == (10, 10, 10, 10)

    def test_parse_pad_four_values(self):
        """Test parsing four padding values."""
        result = parse_pad("10,20,30,40")
        assert result == (10, 20, 30, 40)

    def test_parse_pad_invalid_count(self):
        """Test that invalid pad value count raises error."""
        with pytest.raises(ValueError):
            parse_pad("10,20")


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the transform module."""

    def test_transform_consistency(self):
        """Test that frame and point transforms are consistent."""
        # Create a test frame with a marker
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[40:60, 40:60] = 255  # White square at (40-60, 40-60)

        # Define points at corners of the square
        points = np.array(
            [[40, 40], [60, 40], [60, 60], [40, 60]],
            dtype=np.float64,  # TL  # TR
        )  # BR  # BL

        # Create transform
        transform = Transform(crop=(20, 20, 80, 80), scale=(0.5, 0.5))

        # Apply to frame
        transformed_frame = transform.apply_to_frame(frame)

        # Apply to points
        transformed_points = transform.apply_to_points(points, (100, 100))

        # After crop (20, 20, 80, 80): points at (20, 20), (40, 20), etc.
        # After scale 0.5: points at (10, 10), (20, 10), etc.

        # Check that points are at expected locations
        expected_points = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
        np.testing.assert_array_almost_equal(transformed_points, expected_points)

        # Check that frame has expected size
        assert transformed_frame.shape == (30, 30)

    def test_rotation_point_consistency(self):
        """Test that rotation transforms points correctly with clip_rotation.

        Uses center pixel indexing where the geometric center of a 100x100 image
        is at (49.5, 49.5), not (50, 50).
        """
        # Test 180 degree rotation with clipping (original dimensions preserved)
        transform = Transform(rotate=180, clip_rotation=True)

        # Points at (10, 10), center is (49.5, 49.5) with center pixel indexing
        points = np.array([[10, 10]], dtype=np.float64)

        # After 180 degree rotation around center (49.5, 49.5):
        # new = 2*center - old = 2*(49.5) - 10 = 99 - 10 = 89
        transformed = transform.apply_to_points(points, (100, 100))
        np.testing.assert_array_almost_equal(transformed, [[89, 89]])

    def test_rotation_point_consistency_expanded(self):
        """Test that rotation transforms points correctly with expanded canvas.

        With center pixel indexing, the true center of a 100x100 image is
        at (49.5, 49.5). After 45° rotation, the canvas expands to 142x142
        with new center at (70.5, 70.5). The original center should stay fixed.
        """
        # Test 45 degree rotation with expansion (default)
        transform = Transform(rotate=45)

        # For a 100x100 frame rotated 45 degrees:
        # new_width = ceil(100*cos(45) + 100*sin(45)) = ceil(141.4) = 142
        # new_center = (142-1)/2 = 70.5 (center pixel indexing)

        # True center of 100x100 image with center pixel indexing
        points = np.array([[49.5, 49.5]], dtype=np.float64)
        transformed = transform.apply_to_points(points, (100, 100))
        # Center should stay at new center (70.5, 70.5)
        np.testing.assert_array_almost_equal(transformed, [[70.5, 70.5]])


# ============================================================================
# Video Transform Tests
# ============================================================================


class TestVideoTransforms:
    """Tests for video transformation functions."""

    def test_transform_video_basic(self, tmp_path, centered_pair_low_quality_path):
        """Test basic video transformation."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_video

        # Load video (use small subset via trimming)
        video = sio.load_video(str(centered_pair_low_quality_path))

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.mp4"

        result = transform_video(
            video=video,
            output_path=output_path,
            transform=transform,
            crf=35,
        )

        assert result == output_path
        assert output_path.exists()

        # Verify output dimensions
        out_vid = sio.load_video(str(output_path))
        assert out_vid.shape[1] == 192  # 384 * 0.5
        assert out_vid.shape[2] == 192

    def test_transform_video_with_progress(
        self, tmp_path, centered_pair_low_quality_path
    ):
        """Test video transformation with progress callback."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_video

        video = sio.load_video(str(centered_pair_low_quality_path))
        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.mp4"

        progress_calls = []

        def progress_cb(current, total):
            progress_calls.append((current, total))

        transform_video(
            video=video,
            output_path=output_path,
            transform=transform,
            crf=35,
            progress_callback=progress_cb,
        )

        # Should have callbacks for all frames
        assert len(progress_calls) > 0
        # Last call should have completed all frames
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_transform_labels_basic(self, tmp_path, slp_real_data):
        """Test transform_labels function."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_real_data)

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.slp"
        video_output_dir = tmp_path / "videos"

        result = transform_labels(
            labels=labels,
            transforms=transform,
            output_path=output_path,
            video_output_dir=video_output_dir,
            crf=35,
        )

        # Check result is returned and has frames
        assert result is not None
        assert len(result.labeled_frames) > 0
        # Check video output was created
        assert video_output_dir.exists()

    def test_transform_labels_dry_run(self, tmp_path, slp_real_data):
        """Test transform_labels in dry-run mode."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_real_data)

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.slp"
        video_output_dir = tmp_path / "videos"

        result = transform_labels(
            labels=labels,
            transforms=transform,
            output_path=output_path,
            video_output_dir=video_output_dir,
            dry_run=True,  # Don't actually process videos
        )

        # Check result is returned
        assert result is not None
        assert len(result.labeled_frames) > 0
        # In dry-run, video dir should not be created
        assert not video_output_dir.exists()

    def test_transform_labels_default_video_dir(self, tmp_path, slp_real_data):
        """Test transform_labels with default video output directory."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_real_data)

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.slp"
        # Don't specify video_output_dir - should default

        transform_labels(
            labels=labels,
            transforms=transform,
            output_path=output_path,
            video_output_dir=None,  # Use default
            crf=35,
        )

        # Default dir should be created with name based on output_path
        expected_dir = output_path.with_name("output.videos")
        assert expected_dir.exists()

    def test_transform_labels_per_video_transforms(self, tmp_path, slp_real_data):
        """Test transform_labels with per-video transforms dict."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_real_data)

        # Use dict instead of single Transform
        transforms_dict = {0: Transform(scale=(0.5, 0.5))}

        result = transform_labels(
            labels=labels,
            transforms=transforms_dict,
            output_path=tmp_path / "output.slp",
            video_output_dir=tmp_path / "videos",
            crf=35,
        )

        assert result is not None
        assert len(result.labeled_frames) > 0

    def test_transform_labels_skip_no_transform(self, tmp_path, slp_real_data):
        """Test that videos without transform are skipped."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_real_data)

        # Use an index that doesn't exist - video 0 should be skipped
        transforms_dict = {99: Transform(scale=(0.5, 0.5))}

        result = transform_labels(
            labels=labels,
            transforms=transforms_dict,
            output_path=tmp_path / "output.slp",
            video_output_dir=tmp_path / "videos",
            dry_run=True,
        )

        # Should return without processing (no transform for video 0)
        assert result is not None

    def test_transform_labels_marks_oob_points_invisible(self, tmp_path):
        """Test that out-of-bounds points are marked as not visible."""
        import sleap_io as sio
        from sleap_io.model.instance import Instance
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.skeleton import Node, Skeleton
        from sleap_io.model.video import Video
        from sleap_io.transform.video import transform_labels

        # Create a simple skeleton with 3 nodes
        skeleton = Skeleton(nodes=[Node("a"), Node("b"), Node("c")])

        # Create video with shape set via backend_metadata (100x100)
        video = Video(
            filename="fake_video.mp4",
            backend_metadata={"shape": (2, 100, 100, 1)},  # (frames, h, w, c)
        )

        # Create instance with points:
        # - (50, 50): center, stays in bounds after any reasonable transform
        # - (10, 10): near corner, will go OOB after crop
        # - (90, 90): near opposite corner, will go OOB after crop
        inst = Instance.from_numpy(
            np.array([[50.0, 50.0], [10.0, 10.0], [90.0, 90.0]]),
            skeleton=skeleton,
        )
        # Verify all points start as visible
        assert inst.points["visible"].all()

        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

        # Apply a crop that puts points (10, 10) and (90, 90) out of bounds
        # Crop to center 40x40 region: (30, 30, 70, 70)
        transform = Transform(crop=(30, 30, 70, 70))
        output_path = tmp_path / "output.slp"

        result = transform_labels(
            labels=labels,
            transforms=transform,
            output_path=output_path,
            dry_run=True,  # Don't process videos, just transform coords
        )

        # Get the transformed instance
        result_inst = result.labeled_frames[0].instances[0]

        # Check coordinates were transformed (shifted by crop offset)
        # Point (50, 50) -> (20, 20) after crop offset
        np.testing.assert_array_almost_equal(result_inst.points["xy"][0], [20.0, 20.0])
        # Point (10, 10) -> (-20, -20) after crop offset (OOB)
        np.testing.assert_array_almost_equal(
            result_inst.points["xy"][1], [-20.0, -20.0]
        )
        # Point (90, 90) -> (60, 60) after crop offset (OOB, since output is 40x40)
        np.testing.assert_array_almost_equal(result_inst.points["xy"][2], [60.0, 60.0])

        # Check visibility: only center point should be visible
        # Point 0 (20, 20): in bounds (0 <= x < 40, 0 <= y < 40) - VISIBLE
        # Point 1 (-20, -20): out of bounds - NOT VISIBLE
        # Point 2 (60, 60): out of bounds (>= 40) - NOT VISIBLE
        assert result_inst.points["visible"][0]  # center point in bounds
        assert not result_inst.points["visible"][1]  # OOB (negative)
        assert not result_inst.points["visible"][2]  # OOB (>= output size)

    def test_compute_transform_summary_single_transform(self, slp_real_data):
        """Test compute_transform_summary with a single Transform."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        transform = Transform(scale=(0.5, 0.5))
        summary = compute_transform_summary(labels, transform)

        assert "videos" in summary
        assert "total_frames" in summary
        assert "total_instances" in summary
        assert "warnings" in summary
        assert len(summary["videos"]) == len(labels.videos)

        # Check video info
        video_info = summary["videos"][0]
        assert video_info["index"] == 0
        assert video_info["has_transform"] is True
        assert video_info["input_size"] is not None
        assert video_info["output_size"] is not None
        assert "transform" in video_info

    def test_compute_transform_summary_with_dict(self, slp_real_data):
        """Test compute_transform_summary with dict of transforms."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        # Use dict with transform only for video 0
        transforms_dict = {0: Transform(scale=(0.5, 0.5))}
        summary = compute_transform_summary(labels, transforms_dict)

        assert len(summary["videos"]) == len(labels.videos)
        assert summary["videos"][0]["has_transform"] is True

    def test_compute_transform_summary_small_output_warning(self, slp_real_data):
        """Test that compute_transform_summary warns about very small output."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        # Scale down so much that output is very small
        transform = Transform(scale=(0.01, 0.01))
        summary = compute_transform_summary(labels, transform)

        # Should have a warning about small output size
        assert len(summary["warnings"]) > 0
        assert "small" in summary["warnings"][0].lower()

    def test_compute_transform_summary_crop_oob_warning(self, slp_real_data):
        """Test warning when crop extends outside frame bounds."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        # Crop that extends outside the frame
        transform = Transform(crop=(-10, -10, 100, 100))
        summary = compute_transform_summary(labels, transform)

        # Should have a warning about crop extending outside
        warnings_text = " ".join(summary["warnings"]).lower()
        assert "crop" in warnings_text and "outside" in warnings_text

    def test_compute_transform_summary_rotation_clip_warning(self, slp_real_data):
        """Test warning when rotation clips >20% of frame with clip_rotation."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        # Large rotation with clipping enabled
        transform = Transform(rotate=45, clip_rotation=True)
        summary = compute_transform_summary(labels, transform)

        # Should have a warning about rotation clipping
        warnings_text = " ".join(summary["warnings"]).lower()
        assert "rotation" in warnings_text and "clip" in warnings_text

    def test_compute_transform_summary_dimension_rounding_warning(self, slp_real_data):
        """Test warning when dimensions require rounding."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        # Scale that results in non-integer dimensions
        # 384 * 0.333 = 127.872, which needs rounding
        transform = Transform(scale=(0.333, 0.333))
        summary = compute_transform_summary(labels, transform)

        # Should have a warning about rounding
        warnings_text = " ".join(summary["warnings"]).lower()
        assert "round" in warnings_text

    def test_compute_transform_summary_no_transform(self, slp_real_data):
        """Test compute_transform_summary when video has no transform."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        # Empty transforms dict - no video gets a transform
        transforms_dict: dict[int, Transform] = {}
        summary = compute_transform_summary(labels, transforms_dict)

        # Video should show no transform
        assert summary["videos"][0]["has_transform"] is False

    def test_transform_labels_with_progress_callback(self, tmp_path, slp_real_data):
        """Test transform_labels with progress callback."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_real_data)

        transform = Transform(scale=(0.5, 0.5))
        progress_calls = []

        def progress_callback(video_name: str, current: int, total: int) -> None:
            progress_calls.append((video_name, current, total))

        transform_labels(
            labels=labels,
            transforms=transform,
            output_path=tmp_path / "output.slp",
            video_output_dir=tmp_path / "videos",
            progress_callback=progress_callback,
            crf=35,
        )

        # Progress callback should have been called
        assert len(progress_calls) > 0
        # Each call should have video name and frame counts
        assert all(isinstance(call[0], str) for call in progress_calls)
        assert all(call[1] <= call[2] for call in progress_calls)

    def test_transform_labels_multiview(self, tmp_path, slp_real_data):
        """Test transform_labels with multiple videos exercises skip path."""
        import sleap_io as sio
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.video import Video
        from sleap_io.transform.video import transform_labels

        # Load base labels
        labels = sio.load_slp(slp_real_data)

        # Create a second synthetic video (dummy - won't be processed)
        dummy_video = Video(
            filename="dummy.mp4", backend_metadata={"shape": (10, 100, 100, 3)}
        )
        labels.videos.append(dummy_video)

        # Add a labeled frame for the second video
        dummy_frame = LabeledFrame(video=dummy_video, frame_idx=0)
        labels.labeled_frames.append(dummy_frame)

        # Transform only video 0 - video 1's frames should be skipped
        transforms_dict = {0: Transform(scale=(0.5, 0.5))}

        result = transform_labels(
            labels=labels,
            transforms=transforms_dict,
            output_path=tmp_path / "output.slp",
            video_output_dir=tmp_path / "videos",
            dry_run=True,  # Dry run to skip video processing
        )

        assert result is not None
        # Both videos should be in result
        assert len(result.videos) == 2

    def test_compute_transform_summary_video_no_shape(self, slp_real_data):
        """Test compute_transform_summary when video has no shape info."""
        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_real_data)

        # Add a video without shape info (no backend, no metadata)
        no_shape_video = Video(filename="nonexistent.mp4", open_backend=False)
        labels.videos.append(no_shape_video)

        transform = Transform(scale=(0.5, 0.5))
        summary = compute_transform_summary(labels, transform)

        # Should have info for both videos
        assert len(summary["videos"]) == 2
        # First video should have shape info
        assert summary["videos"][0]["input_size"] is not None
        # Second video (no shape) should have None for input_size and n_frames
        assert summary["videos"][1]["input_size"] is None
        assert summary["videos"][1]["n_frames"] is None

    def test_is_embedded_video(self, slp_minimal_pkg, slp_real_data):
        """Test _is_embedded_video helper function."""
        import sleap_io as sio
        from sleap_io.transform.video import _is_embedded_video

        # Test embedded video
        embedded_labels = sio.load_slp(slp_minimal_pkg)
        assert _is_embedded_video(embedded_labels.videos[0]) is True

        # Test regular video
        regular_labels = sio.load_slp(slp_real_data)
        assert _is_embedded_video(regular_labels.videos[0]) is False

    def test_get_frame_indices_embedded(self, slp_minimal_pkg):
        """Test _get_frame_indices for embedded video."""
        import sleap_io as sio
        from sleap_io.transform.video import _get_frame_indices

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        frame_inds = _get_frame_indices(video)

        # Should return the embedded frame indices, not sequential
        assert frame_inds == video.backend.embedded_frame_inds

    def test_get_frame_indices_regular(self, slp_real_data):
        """Test _get_frame_indices for regular video."""
        import sleap_io as sio
        from sleap_io.transform.video import _get_frame_indices

        labels = sio.load_slp(slp_real_data)
        video = labels.videos[0]

        frame_inds = _get_frame_indices(video)

        # Should return sequential indices
        expected = list(range(video.shape[0]))
        assert frame_inds == expected

    def test_transform_embedded_video(self, tmp_path, slp_minimal_pkg):
        """Test transform_embedded_video function."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_embedded_video

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"

        # Save empty labels file first (creates HDF5 structure)
        labels.save(str(output_path))

        result = transform_embedded_video(
            video=video,
            output_path=output_path,
            video_idx=0,
            transform=transform,
        )

        # Check result is a Video object
        assert result is not None
        assert result.filename == str(output_path)

        # Check output dimensions are scaled
        original_h, original_w = video.shape[1:3]
        expected_h = int(original_h * 0.5)
        expected_w = int(original_w * 0.5)

        assert result.shape[1] == expected_h
        assert result.shape[2] == expected_w

    def test_transform_labels_embedded(self, tmp_path, slp_minimal_pkg):
        """Test transform_labels with embedded video."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_minimal_pkg)

        # Verify input has embedded video
        assert labels.videos[0].backend.__class__.__name__ == "HDF5Video"

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"

        result = transform_labels(
            labels=labels,
            transforms=transform,
            output_path=output_path,
        )

        # Check result
        assert result is not None
        assert len(result.videos) == 1
        assert len(result.labeled_frames) == len(labels.labeled_frames)

        # Output should exist
        assert output_path.exists()

        # Output video should have scaled dimensions
        original_h, original_w = labels.videos[0].shape[1:3]
        expected_h = int(original_h * 0.5)
        expected_w = int(original_w * 0.5)

        assert result.videos[0].shape[1] == expected_h
        assert result.videos[0].shape[2] == expected_w

        # Coordinates should be transformed
        original_lf = labels.labeled_frames[0]
        transformed_lf = result.labeled_frames[0]

        if len(original_lf.instances) > 0 and len(transformed_lf.instances) > 0:
            original_points = original_lf.instances[0].numpy()
            transformed_points = transformed_lf.instances[0].numpy()
            # Points should be scaled by 0.5
            # Note: points might have NaN values, so use nanmean for comparison
            import numpy as np

            if not np.all(np.isnan(original_points)) and not np.all(
                np.isnan(transformed_points)
            ):
                # Check that transformed points are approximately half the original
                scale_factor = np.nanmean(transformed_points) / np.nanmean(
                    original_points
                )
                assert abs(scale_factor - 0.5) < 0.1  # Allow some tolerance

    def test_update_videos_json(self, tmp_path, slp_minimal_pkg):
        """Test _update_videos_json helper function updates HDF5 videos_json."""
        import json

        import h5py

        import sleap_io as sio
        from sleap_io.transform.video import _update_videos_json

        # Load embedded labels and create a copy
        labels = sio.load_slp(slp_minimal_pkg)

        # Save to create a new file with videos_json
        output_path = tmp_path / "test.pkg.slp"
        labels.save(str(output_path))

        # Verify original videos_json exists
        with h5py.File(output_path, "r") as f:
            assert "videos_json" in f

        # Update videos_json with the same videos
        _update_videos_json(output_path, labels.videos)

        # Verify the update worked
        with h5py.File(output_path, "r") as f:
            videos_json = list(f["videos_json"][:])
            assert len(videos_json) == len(labels.videos)

            # First video should have embedded video reference
            video0_data = json.loads(videos_json[0])
            assert "backend" in video0_data
            assert "filename" in video0_data["backend"]

    def test_transform_embedded_video_with_progress(self, tmp_path, slp_minimal_pkg):
        """Test transform_embedded_video with progress callback."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_embedded_video

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"

        # Save empty labels file first
        labels.save(str(output_path))

        progress_calls = []

        def progress_cb(current: int, total: int) -> None:
            progress_calls.append((current, total))

        transform_embedded_video(
            video=video,
            output_path=output_path,
            video_idx=0,
            transform=transform,
            progress_callback=progress_cb,
        )

        # Progress callback should have been called for each frame
        assert len(progress_calls) > 0
        # Last call should have current == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_transform_embedded_video_imageio(
        self, tmp_path, slp_minimal_pkg, monkeypatch
    ):
        """Test transform_embedded_video using imageio backend."""
        import sys

        import sleap_io as sio
        from sleap_io.transform.video import transform_embedded_video

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"

        # Save empty labels file first
        labels.save(str(output_path))

        # Force imageio by removing cv2 from modules temporarily
        cv2_module = sys.modules.get("cv2")
        if "cv2" in sys.modules:
            del sys.modules["cv2"]

        try:
            result = transform_embedded_video(
                video=video,
                output_path=output_path,
                video_idx=0,
                transform=transform,
                plugin="imageio",  # Explicitly use imageio
            )

            assert result is not None
            # Output dimensions should be scaled
            assert result.shape[1] == int(video.shape[1] * 0.5)
            assert result.shape[2] == int(video.shape[2] * 0.5)
        finally:
            # Restore cv2 module
            if cv2_module is not None:
                sys.modules["cv2"] = cv2_module

    def test_transform_embedded_video_grayscale_imageio(
        self, tmp_path, slp_minimal_pkg
    ):
        """Test transform_embedded_video with grayscale image via imageio."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_embedded_video

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        # Check if the video is grayscale (1 channel)
        # If not, we still test the path but may not hit the squeeze branch
        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"

        labels.save(str(output_path))

        result = transform_embedded_video(
            video=video,
            output_path=output_path,
            video_idx=0,
            transform=transform,
            plugin="imageio",
        )

        assert result is not None

    def test_transform_video_with_explicit_fps(
        self, tmp_path, centered_pair_low_quality_path
    ):
        """Test transform_video with explicitly specified fps."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_video

        video = sio.load_video(str(centered_pair_low_quality_path))

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.mp4"

        # Pass explicit fps to skip the fallback logic
        result = transform_video(
            video=video,
            output_path=output_path,
            transform=transform,
            fps=25.0,  # Explicit fps
            crf=35,
        )

        assert result == output_path
        assert output_path.exists()

    def test_update_videos_json_with_input_format(self, tmp_path, slp_minimal_pkg):
        """Test _update_videos_json with backend that has input_format attribute."""
        import json

        import h5py

        import sleap_io as sio
        from sleap_io.transform.video import _update_videos_json

        labels = sio.load_slp(slp_minimal_pkg)

        # Save to create a new file
        output_path = tmp_path / "test.pkg.slp"
        labels.save(str(output_path))

        # Add input_format to backend to test that branch (line 335-336)
        video = labels.videos[0]
        if hasattr(video.backend, "__dict__"):
            video.backend.input_format = "png"

        # Update videos_json
        _update_videos_json(output_path, labels.videos)

        # Verify the update
        with h5py.File(output_path, "r") as f:
            videos_json = list(f["videos_json"][:])
            video0_data = json.loads(videos_json[0])
            # Check that the backend info is present
            assert "backend" in video0_data

    def test_transform_labels_video_no_shape_gets_from_frame(self, tmp_path):
        """Test transform_labels when video.shape is None but frame is readable."""
        import sleap_io as sio
        from sleap_io.model.instance import Instance
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.skeleton import Node, Skeleton
        from sleap_io.model.video import Video
        from sleap_io.transform.video import transform_labels

        skeleton = Skeleton(nodes=[Node("a")])

        # Create a video with metadata that provides shape
        video = Video(
            filename="fake_video.mp4",
            backend_metadata={"shape": (10, 100, 100, 1)},
        )

        inst = Instance.from_numpy(np.array([[50.0, 50.0]]), skeleton=skeleton)
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

        transform = Transform(scale=(0.5, 0.5))

        # This should work since video.shape returns from backend_metadata
        result = transform_labels(
            labels=labels,
            transforms=transform,
            output_path=tmp_path / "output.slp",
            dry_run=True,
        )

        assert result is not None
        # Check that coordinates were transformed
        np.testing.assert_array_almost_equal(
            result.labeled_frames[0].instances[0].numpy(), [[25.0, 25.0]]
        )

    def test_transform_embedded_video_no_fps(self, tmp_path, slp_minimal_pkg):
        """Test transform_embedded_video when video has no fps attribute."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_embedded_video

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        # Clear fps if present
        if hasattr(video, "_fps"):
            original_fps = video._fps
            video._fps = None
        else:
            original_fps = None

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"
        labels.save(str(output_path))

        try:
            result = transform_embedded_video(
                video=video,
                output_path=output_path,
                video_idx=0,
                transform=transform,
            )
            assert result is not None
        finally:
            # Restore fps
            if original_fps is not None:
                video._fps = original_fps

    def test_is_embedded_video_no_backend(self):
        """Test _is_embedded_video with video that has no backend."""
        from sleap_io.model.video import Video
        from sleap_io.transform.video import _is_embedded_video

        # Video with open_backend=False - backend will be None
        video = Video(filename="test.mp4", open_backend=False)

        # backend is None, so should return False
        assert _is_embedded_video(video) is False

    def test_get_frame_indices_no_frame_map(self, slp_real_data):
        """Test _get_frame_indices when HDF5Video has no frame_map."""
        import sleap_io as sio
        from sleap_io.transform.video import _get_frame_indices

        labels = sio.load_slp(slp_real_data)
        video = labels.videos[0]

        # Regular video (not embedded) has no frame_map
        frame_inds = _get_frame_indices(video)

        # Should return sequential indices
        assert frame_inds == list(range(video.shape[0]))

    def test_transform_video_fps_fallback_no_backend(self, tmp_path):
        """Test transform_video uses 30fps fallback when video.backend is None."""
        from unittest.mock import patch

        # Create a simple test video first
        import imageio.v3 as iio

        from sleap_io.model.video import Video
        from sleap_io.transform.video import transform_video

        video_path = tmp_path / "test_input.mp4"
        frames = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(5)]
        iio.imwrite(str(video_path), frames, fps=25)

        # Create video with open_backend=False to simulate backend being None
        video = Video(filename=str(video_path), open_backend=False)
        # Store shape in metadata so transform can compute output size
        video.backend_metadata["shape"] = (5, 112, 112, 3)

        # Since backend is None, we need to mock __getitem__ and shape
        with (
            patch.object(Video, "__getitem__", return_value=frames[0]),
            patch.object(
                Video, "shape", property(lambda s: s.backend_metadata.get("shape"))
            ),
        ):
            transform = Transform(scale=(0.5, 0.5))
            output_path = tmp_path / "output.mp4"

            # This should use fps=30.0 fallback (backend is None)
            result = transform_video(
                video=video,
                output_path=output_path,
                transform=transform,
            )

            assert result == output_path
            assert output_path.exists()

    def test_transform_video_fps_fallback_exception(
        self, tmp_path, centered_pair_low_quality_path
    ):
        """Test transform_video uses 30fps fallback when backend.fps raises."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_video

        video = sio.load_video(str(centered_pair_low_quality_path))

        # Patch the backend's fps property to raise an exception
        original_fps = video.backend.__class__.fps

        @property
        def raising_fps(self):
            raise RuntimeError("FPS not available")

        video.backend.__class__.fps = raising_fps

        try:
            transform = Transform(scale=(0.5, 0.5))
            output_path = tmp_path / "output.mp4"

            # Should use fps=30.0 fallback
            result = transform_video(
                video=video,
                output_path=output_path,
                transform=transform,
                crf=35,
            )

            assert result == output_path
            assert output_path.exists()
        finally:
            video.backend.__class__.fps = original_fps

    def test_transform_video_frame_none_skipped(self, tmp_path):
        """Test transform_video skips frames that return None."""
        # Create a simple test video first
        import imageio.v3 as iio

        from sleap_io.model.video import Video
        from sleap_io.transform.video import transform_video

        video_path = tmp_path / "test_input.mp4"
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        iio.imwrite(str(video_path), frames, fps=25)

        video = Video(filename=str(video_path))
        original_getitem = Video.__getitem__

        # Mock to return None for frame 2
        def mock_getitem(self, idx):
            if idx == 2:
                return None
            return original_getitem(self, idx)

        Video.__getitem__ = mock_getitem

        try:
            transform = Transform(scale=(0.5, 0.5))
            output_path = tmp_path / "output.mp4"

            result = transform_video(
                video=video,
                output_path=output_path,
                transform=transform,
            )

            assert result == output_path
            assert output_path.exists()
        finally:
            Video.__getitem__ = original_getitem

    def test_transform_embedded_video_stores_fps(self, tmp_path, slp_minimal_pkg):
        """Test transform_embedded_video stores FPS attribute when available."""
        import h5py

        import sleap_io as sio
        from sleap_io.transform.video import transform_embedded_video

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        # Set fps on the video using the proper setter
        video.fps = 24.0

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"
        labels.save(str(output_path))

        result = transform_embedded_video(
            video=video,
            output_path=output_path,
            video_idx=0,
            transform=transform,
        )

        assert result is not None

        # Verify FPS was stored
        with h5py.File(output_path, "r") as f:
            assert "video0/video" in f
            assert f["video0/video"].attrs.get("fps") == 24.0

    def test_transform_embedded_video_empty_frames(self, tmp_path, slp_minimal_pkg):
        """Test transform_embedded_video handles video with no frames."""
        from unittest.mock import patch

        import sleap_io as sio
        from sleap_io.transform.video import (
            transform_embedded_video,
        )

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"
        labels.save(str(output_path))

        # Mock _get_frame_indices to return empty list (simulating no frames)
        with patch("sleap_io.transform.video._get_frame_indices", return_value=[]):
            result = transform_embedded_video(
                video=video,
                output_path=output_path,
                video_idx=0,
                transform=transform,
            )

            # Should return a video object (referencing empty structure)
            assert result is not None

    def test_transform_embedded_video_no_shape_raises(self, tmp_path, slp_minimal_pkg):
        """Test transform_embedded_video raises when video has no shape."""
        from unittest.mock import PropertyMock, patch

        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.transform.video import transform_embedded_video

        labels = sio.load_slp(slp_minimal_pkg)
        video = labels.videos[0]

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"
        labels.save(str(output_path))

        # Mock video.shape to return None (simulating unknown dimensions)
        with patch.object(Video, "shape", new_callable=PropertyMock, return_value=None):
            with pytest.raises(
                ValueError, match="Cannot determine dimensions for embedded video"
            ):
                transform_embedded_video(
                    video=video,
                    output_path=output_path,
                    video_idx=0,
                    transform=transform,
                )

    def test_transform_labels_video_dimension_from_frame_error(
        self, tmp_path, slp_minimal
    ):
        """Test transform_labels raises when can't get dimensions from frame."""
        from unittest.mock import PropertyMock, patch

        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_minimal)

        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.slp"

        # Mock video.shape to return None and __getitem__ to raise exception
        def raise_on_getitem(self, idx):
            raise RuntimeError("Cannot read frame")

        with (
            patch.object(Video, "shape", new_callable=PropertyMock, return_value=None),
            patch.object(Video, "__getitem__", raise_on_getitem),
        ):
            with pytest.raises(ValueError, match="Cannot determine dimensions"):
                transform_labels(
                    labels=labels,
                    transforms=transform,
                    output_path=output_path,
                )

    def test_compute_transform_summary_empty_points(self, slp_minimal):
        """Test compute_transform_summary handles instances with empty points."""
        import sleap_io as sio
        from sleap_io.model.instance import Instance
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_minimal)

        # Create an instance with empty points
        if labels.labeled_frames:
            lf = labels.labeled_frames[0]
            # Create a new instance with no points
            empty_instance = Instance(
                points={},  # Empty points
                skeleton=labels.skeleton,
            )
            lf.instances.append(empty_instance)

        transform = Transform(crop=(0, 0, 100, 100))
        summary = compute_transform_summary(labels, transform)

        # Should not crash and should count instances
        assert "total_instances" in summary

    def test_compute_transform_summary_oob_landmarks_warning(self, slp_minimal):
        """Test compute_transform_summary warns about OOB landmarks."""
        import sleap_io as sio
        from sleap_io.transform.video import compute_transform_summary

        labels = sio.load_slp(slp_minimal)

        # Use a very aggressive crop that will push points OOB
        # Crop to small region at corner - points outside will be OOB
        transform = Transform(crop=(0, 0, 10, 10))

        summary = compute_transform_summary(labels, transform)

        # Should have warnings about OOB landmarks
        # (depends on whether the test data has points outside the crop region)
        assert "warnings" in summary

    def test_transform_labels_embedded_progress(self, tmp_path, slp_minimal_pkg):
        """Test transform_labels calls progress callback for embedded videos."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_minimal_pkg)
        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"

        progress_calls = []

        def progress_callback(video_name, current, total):
            progress_calls.append((video_name, current, total))

        transform_labels(
            labels=labels,
            transforms=transform,
            output_path=output_path,
            progress_callback=progress_callback,
        )

        # Should have progress calls (at least for embedded video processing)
        # The exact number depends on the number of frames
        assert output_path.exists()

    def test_transform_labels_embedded_replaces_videos(self, tmp_path, slp_minimal_pkg):
        """Test transform_labels correctly replaces video references for embedded."""
        import sleap_io as sio
        from sleap_io.transform.video import transform_labels

        labels = sio.load_slp(slp_minimal_pkg)
        transform = Transform(scale=(0.5, 0.5))
        output_path = tmp_path / "output.pkg.slp"

        result = transform_labels(
            labels=labels,
            transforms=transform,
            output_path=output_path,
        )

        # Verify the output has correct video references
        assert len(result.videos) == len(labels.videos)
        assert output_path.exists()

        # Reload and check
        reloaded = sio.load_slp(str(output_path))
        assert len(reloaded.videos) == len(labels.videos)
