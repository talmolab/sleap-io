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
        """Test rotate_frame with tuple fill color."""
        # RGB frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = [255, 255, 255]

        # Use tuple fill instead of int
        result = rotate_frame(frame, 45, fill=(128, 128, 128))

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
        """Test that rotation transforms points correctly."""
        # Test 180 degree rotation
        transform = Transform(rotate=180)

        # Points at (10, 10), center is (50, 50)
        points = np.array([[10, 10]], dtype=np.float64)

        # After 180 degree rotation around center (50, 50):
        # (10, 10) -> (90, 90)
        transformed = transform.apply_to_points(points, (100, 100))
        np.testing.assert_array_almost_equal(transformed, [[90, 90]])


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
