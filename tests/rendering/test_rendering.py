"""Tests for the rendering module.

These tests validate the rendering functionality including colors, shapes,
callbacks, and the core rendering functions.
"""

import numpy as np
import pytest
from shapely.geometry import box

import sleap_io as sio
from sleap_io.model.bbox import UserBoundingBox
from sleap_io.model.label_image import PredictedLabelImage, UserLabelImage
from sleap_io.model.mask import UserSegmentationMask
from sleap_io.model.roi import UserROI
from sleap_io.rendering import render_video
from sleap_io.rendering.core import render_image
from sleap_io.rendering.overlays import (
    draw_bboxes,
    draw_label_image,
    draw_masks,
    draw_rois,
    draw_trails,
)

# Skip all tests if skia-python is not installed
skia = pytest.importorskip("skia", reason="skia-python not installed")


# ============================================================================
# Colors Module Tests
# ============================================================================


class TestColors:
    """Tests for sleap_io.rendering.colors module."""

    def test_get_palette_builtin(self):
        """Test getting built-in palettes."""
        from sleap_io.rendering.colors import get_palette

        # Test all built-in palettes
        for palette_name in [
            "distinct",
            "rainbow",
            "warm",
            "cool",
            "pastel",
            "seaborn",
            "tableau10",
            "viridis",
        ]:
            colors = get_palette(palette_name, 5)
            assert len(colors) == 5
            for color in colors:
                assert len(color) == 3
                assert all(0 <= c <= 255 for c in color)

    def test_get_palette_extend(self):
        """Test palette extension when more colors needed."""
        from sleap_io.rendering.colors import get_palette

        # Request more colors than palette has
        colors = get_palette("cool", 20)
        assert len(colors) == 20

    def test_get_palette_unknown_raises(self):
        """Test that unknown palette raises ValueError."""
        from sleap_io.rendering.colors import get_palette

        with pytest.raises(ValueError, match="Unknown palette"):
            get_palette("unknown_palette", 5)

    def test_rgb_to_skia_color(self):
        """Test RGB to Skia color conversion."""
        from sleap_io.rendering.colors import rgb_to_skia_color

        color = rgb_to_skia_color((255, 0, 0), 128)
        assert color is not None

    def test_determine_color_scheme(self):
        """Test color scheme determination logic."""
        from sleap_io.rendering.colors import determine_color_scheme

        # With tracks -> track coloring
        assert determine_color_scheme(True, True, "auto") == "track"

        # No tracks, single image -> instance coloring
        assert determine_color_scheme(False, True, "auto") == "instance"

        # No tracks, video -> node coloring
        assert determine_color_scheme(False, False, "auto") == "node"

        # Explicit scheme
        assert determine_color_scheme(True, True, "node") == "node"

    def test_build_color_map(self):
        """Test color map building for different schemes."""
        from sleap_io.rendering.colors import build_color_map

        # Track scheme
        colors = build_color_map(
            scheme="track",
            n_instances=2,
            n_nodes=5,
            n_tracks=3,
            track_indices=[0, 2],
        )
        assert "instance_colors" in colors
        assert len(colors["instance_colors"]) == 2

        # Instance scheme
        colors = build_color_map(
            scheme="instance", n_instances=3, n_nodes=5, n_tracks=0
        )
        assert "instance_colors" in colors
        assert len(colors["instance_colors"]) == 3

        # Node scheme
        colors = build_color_map(scheme="node", n_instances=2, n_nodes=5, n_tracks=0)
        assert "node_colors" in colors
        assert len(colors["node_colors"]) == 5

    def test_resolve_color_rgb_int_tuple(self):
        """Test resolve_color with RGB int tuple."""
        from sleap_io.rendering.colors import resolve_color

        assert resolve_color((255, 128, 0)) == (255, 128, 0)
        assert resolve_color((0, 0, 0)) == (0, 0, 0)
        assert resolve_color((255, 255, 255)) == (255, 255, 255)

    def test_resolve_color_rgb_float_tuple(self):
        """Test resolve_color with RGB float tuple."""
        from sleap_io.rendering.colors import resolve_color

        assert resolve_color((1.0, 0.5, 0.0)) == (255, 127, 0)
        assert resolve_color((0.0, 0.0, 0.0)) == (0, 0, 0)
        assert resolve_color((1.0, 1.0, 1.0)) == (255, 255, 255)

    def test_resolve_color_grayscale_int(self):
        """Test resolve_color with grayscale int."""
        from sleap_io.rendering.colors import resolve_color

        assert resolve_color(128) == (128, 128, 128)
        assert resolve_color(0) == (0, 0, 0)
        assert resolve_color(255) == (255, 255, 255)

    def test_resolve_color_grayscale_float(self):
        """Test resolve_color with grayscale float."""
        from sleap_io.rendering.colors import resolve_color

        assert resolve_color(0.5) == (127, 127, 127)
        assert resolve_color(0.0) == (0, 0, 0)
        assert resolve_color(1.0) == (255, 255, 255)

    def test_resolve_color_named(self):
        """Test resolve_color with named colors."""
        from sleap_io.rendering.colors import resolve_color

        assert resolve_color("black") == (0, 0, 0)
        assert resolve_color("white") == (255, 255, 255)
        assert resolve_color("red") == (255, 0, 0)
        assert resolve_color("GREEN") == (0, 255, 0)  # Case insensitive
        assert resolve_color("gray") == (128, 128, 128)
        assert resolve_color("grey") == (128, 128, 128)  # British spelling

    def test_resolve_color_hex(self):
        """Test resolve_color with hex colors."""
        from sleap_io.rendering.colors import resolve_color

        assert resolve_color("#ff0000") == (255, 0, 0)
        assert resolve_color("#FF0000") == (255, 0, 0)  # Case insensitive
        assert resolve_color("#f00") == (255, 0, 0)  # 3-digit hex
        assert resolve_color("#808080") == (128, 128, 128)

    def test_resolve_color_palette_index(self):
        """Test resolve_color with palette index."""
        from sleap_io.rendering.colors import get_palette, resolve_color

        # Test tableau10 palette
        tableau_colors = get_palette("tableau10", 10)
        assert resolve_color("tableau10[0]") == tableau_colors[0]
        assert resolve_color("tableau10[2]") == tableau_colors[2]

        # Test distinct palette
        distinct_colors = get_palette("distinct", 5)
        assert resolve_color("distinct[0]") == distinct_colors[0]

    def test_resolve_color_invalid(self):
        """Test resolve_color with invalid inputs."""
        from sleap_io.rendering.colors import resolve_color

        with pytest.raises(ValueError):
            resolve_color("not_a_color")

        with pytest.raises(ValueError):
            resolve_color("#invalid")

        with pytest.raises(ValueError):
            resolve_color((1, 2))  # Wrong tuple length

        with pytest.raises(TypeError):
            resolve_color(["list", "not", "valid"])

    def test_resolve_color_unknown_palette_raises(self):
        """Test resolve_color with unknown palette raises ValueError."""
        from sleap_io.rendering.colors import resolve_color

        # Unknown palette in palette[n] format should raise ValueError
        with pytest.raises(ValueError, match="Invalid palette index"):
            resolve_color("unknown_palette[0]")

    def test_build_color_map_track_without_indices(self):
        """Test build_color_map for track scheme without track_indices."""
        from sleap_io.rendering.colors import build_color_map

        colors = build_color_map(
            scheme="track",
            n_instances=3,
            n_nodes=5,
            n_tracks=5,
            track_indices=None,  # No track indices provided
        )

        assert "instance_colors" in colors
        assert len(colors["instance_colors"]) == 3

    def test_get_palette_colorcet_names(self):
        """Test get_palette with colorcet palette names (falls back if missing)."""
        from sleap_io.rendering.colors import get_palette

        # These should work whether colorcet is installed or not
        # (falls back to distinct palette if not installed)
        colors = get_palette("glasbey", 10)
        assert len(colors) == 10

        colors = get_palette("glasbey_hv", 10)
        assert len(colors) == 10


# ============================================================================
# Shapes Module Tests
# ============================================================================


class TestShapes:
    """Tests for sleap_io.rendering.shapes module."""

    def test_get_marker_func(self):
        """Test getting marker drawing functions."""
        from sleap_io.rendering.shapes import get_marker_func

        for shape in ["circle", "square", "diamond", "triangle", "cross"]:
            func = get_marker_func(shape)
            assert callable(func)

    def test_get_marker_func_unknown(self):
        """Test error on unknown marker shape."""
        from sleap_io.rendering.shapes import get_marker_func

        with pytest.raises(ValueError, match="Unknown marker shape"):
            get_marker_func("unknown_shape")

    def test_draw_circle_marker(self):
        """Test circle marker drawing."""
        from sleap_io.rendering.shapes import draw_circle_marker

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        paint = skia.Paint(Color=skia.Color(255, 0, 0), AntiAlias=True)
        draw_circle_marker(canvas, 50, 50, 10, paint)
        surface.flushAndSubmit()

        # Check that something was drawn (center should have color)
        assert output[50, 50, 0] > 0

    def test_draw_square_marker(self):
        """Test square marker drawing."""
        from sleap_io.rendering.shapes import draw_square_marker

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        paint = skia.Paint(Color=skia.Color(0, 255, 0), AntiAlias=True)
        draw_square_marker(canvas, 50, 50, 10, paint)
        surface.flushAndSubmit()

        # Check that something was drawn
        assert output[50, 50, 1] > 0

    def test_draw_diamond_marker(self):
        """Test diamond marker drawing."""
        from sleap_io.rendering.shapes import draw_diamond_marker

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        paint = skia.Paint(Color=skia.Color(0, 0, 255), AntiAlias=True)
        draw_diamond_marker(canvas, 50, 50, 10, paint)
        surface.flushAndSubmit()

        # Check that something was drawn
        assert output[50, 50, 2] > 0

    def test_draw_triangle_marker(self):
        """Test triangle marker drawing."""
        from sleap_io.rendering.shapes import draw_triangle_marker

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        paint = skia.Paint(Color=skia.Color(255, 255, 0), AntiAlias=True)
        draw_triangle_marker(canvas, 50, 50, 10, paint)
        surface.flushAndSubmit()

        # Check that something was drawn (center should have color)
        assert output[50, 50, 0] > 0 or output[52, 50, 0] > 0

    def test_draw_cross_marker(self):
        """Test cross marker drawing."""
        from sleap_io.rendering.shapes import draw_cross_marker

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        paint = skia.Paint(Color=skia.Color(255, 0, 255), AntiAlias=True)
        draw_cross_marker(canvas, 50, 50, 10, paint)
        surface.flushAndSubmit()

        # Check that something was drawn (center should have color)
        assert output[50, 50, 0] > 0


# ============================================================================
# Callbacks Module Tests
# ============================================================================


class TestCallbacks:
    """Tests for sleap_io.rendering.callbacks module."""

    def test_render_context_world_to_canvas(self):
        """Test RenderContext coordinate transformation."""
        from sleap_io.rendering.callbacks import RenderContext

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        ctx = RenderContext(
            canvas=canvas,
            frame_idx=0,
            frame_size=(100, 100),
            instances=[],
            skeleton_edges=[],
            node_names=[],
            scale=2.0,
            offset=(10.0, 10.0),
        )

        x, y = ctx.world_to_canvas(20, 20)
        assert x == (20 - 10) * 2.0
        assert y == (20 - 10) * 2.0

    def test_instance_context_get_centroid(self):
        """Test InstanceContext centroid calculation."""
        from sleap_io.rendering.callbacks import InstanceContext

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        points = np.array([[10, 20], [30, 40], [np.nan, np.nan]])
        ctx = InstanceContext(
            canvas=canvas,
            instance_idx=0,
            points=points,
            skeleton_edges=[],
            node_names=[],
        )

        centroid = ctx.get_centroid()
        assert centroid is not None
        assert centroid[0] == pytest.approx(20.0)
        assert centroid[1] == pytest.approx(30.0)

    def test_instance_context_get_centroid_all_nan(self):
        """Test InstanceContext centroid with all NaN points."""
        from sleap_io.rendering.callbacks import InstanceContext

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        points = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        ctx = InstanceContext(
            canvas=canvas,
            instance_idx=0,
            points=points,
            skeleton_edges=[],
            node_names=[],
        )

        assert ctx.get_centroid() is None

    def test_instance_context_get_bbox(self):
        """Test InstanceContext bounding box calculation."""
        from sleap_io.rendering.callbacks import InstanceContext

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        points = np.array([[10, 20], [30, 40], [20, 10]])
        ctx = InstanceContext(
            canvas=canvas,
            instance_idx=0,
            points=points,
            skeleton_edges=[],
            node_names=[],
        )

        bbox = ctx.get_bbox()
        assert bbox is not None
        assert bbox == (10.0, 10.0, 30.0, 40.0)

    def test_instance_context_get_bbox_all_nan(self):
        """Test InstanceContext bbox with all NaN points."""
        from sleap_io.rendering.callbacks import InstanceContext

        output = np.zeros((100, 100, 4), dtype=np.uint8)
        surface = skia.Surface(output, colorType=skia.kRGBA_8888_ColorType)
        canvas = surface.getCanvas()

        points = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        ctx = InstanceContext(
            canvas=canvas,
            instance_idx=0,
            points=points,
            skeleton_edges=[],
            node_names=[],
        )

        assert ctx.get_bbox() is None

    def test_render_image_crop_passes_offset_to_callbacks(self, labels_predictions):
        """Test that crop offset is passed to RenderContext in callbacks.

        When render_image is called with a crop parameter, the RenderContext
        provided to callbacks should have its offset set to the crop origin
        (x1, y1) so that world_to_canvas() transforms correctly.
        """
        from sleap_io.rendering import render_image
        from sleap_io.rendering.callbacks import RenderContext

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        # Track the offset received in callback
        captured_offsets = []

        def pre_callback(ctx: RenderContext):
            captured_offsets.append(ctx.offset)

        # Render with explicit crop bounds: (x1, y1, x2, y2) = (100, 50, 200, 150)
        crop_bounds = (100, 50, 200, 150)
        render_image(
            lf, image=frame, crop=crop_bounds, pre_render_callback=pre_callback
        )

        assert len(captured_offsets) == 1
        # Offset should be (x1, y1) from the crop bounds
        assert captured_offsets[0] == (100.0, 50.0)

    def test_render_image_no_crop_offset_is_zero(self, labels_predictions):
        """Test that offset is (0, 0) when no crop is used."""
        from sleap_io.rendering import render_image
        from sleap_io.rendering.callbacks import RenderContext

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        captured_offsets = []

        def pre_callback(ctx: RenderContext):
            captured_offsets.append(ctx.offset)

        # Render without crop
        render_image(lf, image=frame, pre_render_callback=pre_callback)

        assert len(captured_offsets) == 1
        assert captured_offsets[0] == (0.0, 0.0)

    def test_render_image_crop_per_instance_callback_offset(self, labels_predictions):
        """Test that per_instance_callback receives correct crop offset."""
        from sleap_io.rendering import render_image
        from sleap_io.rendering.callbacks import InstanceContext

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        captured_offsets = []

        def instance_callback(ctx: InstanceContext):
            captured_offsets.append(ctx.offset)

        crop_bounds = (50, 25, 250, 225)
        render_image(
            lf, image=frame, crop=crop_bounds, per_instance_callback=instance_callback
        )

        # Should have one offset captured per instance
        assert len(captured_offsets) == len(lf.instances)
        # All offsets should match crop origin
        for offset in captured_offsets:
            assert offset == (50.0, 25.0)

    def test_render_image_crop_post_callback_offset(self, labels_predictions):
        """Test that post_render_callback receives correct crop offset."""
        from sleap_io.rendering import render_image
        from sleap_io.rendering.callbacks import RenderContext

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        captured_offsets = []

        def post_callback(ctx: RenderContext):
            captured_offsets.append(ctx.offset)

        crop_bounds = (75, 100, 275, 300)
        render_image(
            lf, image=frame, crop=crop_bounds, post_render_callback=post_callback
        )

        assert len(captured_offsets) == 1
        assert captured_offsets[0] == (75.0, 100.0)

    def test_render_image_crop_world_to_canvas_integration(self, labels_predictions):
        """Test that world_to_canvas correctly transforms coordinates with crop.

        This is an end-to-end test verifying that callbacks can use
        world_to_canvas to correctly map world coordinates to canvas space
        when a crop is applied.
        """
        from sleap_io.rendering import render_image
        from sleap_io.rendering.callbacks import RenderContext

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        # Test with a crop at (100, 50)
        crop_bounds = (100, 50, 300, 250)
        scale = 1.0
        captured_transforms = []

        def pre_callback(ctx: RenderContext):
            # A point at world (150, 100) should be at canvas (50, 50) after crop
            canvas_x, canvas_y = ctx.world_to_canvas(150, 100)
            captured_transforms.append((canvas_x, canvas_y))

        render_image(
            lf,
            image=frame,
            crop=crop_bounds,
            scale=scale,
            pre_render_callback=pre_callback,
        )

        assert len(captured_transforms) == 1
        canvas_x, canvas_y = captured_transforms[0]
        # world_to_canvas: (x - offset_x) * scale, (y - offset_y) * scale
        # (150 - 100) * 1.0 = 50, (100 - 50) * 1.0 = 50
        assert canvas_x == 50.0
        assert canvas_y == 50.0


# ============================================================================
# Core Rendering Tests
# ============================================================================


class TestPrepareFrame:
    """Tests for _prepare_frame_rgba helper function."""

    def test_prepare_frame_grayscale_with_channel(self):
        """Test grayscale with channel dimension (H, W, 1)."""
        from sleap_io.rendering.core import _prepare_frame_rgba

        frame = np.full((100, 100, 1), 128, dtype=np.uint8)
        result = _prepare_frame_rgba(frame)

        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
        assert result[50, 50, 3] == 255  # Alpha channel

    def test_prepare_frame_rgba_passthrough(self):
        """Test RGBA passthrough (H, W, 4)."""
        from sleap_io.rendering.core import _prepare_frame_rgba

        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red
        frame[:, :, 3] = 200  # Alpha
        result = _prepare_frame_rgba(frame)

        assert result.shape == (100, 100, 4)
        assert result[50, 50, 0] == 255  # Red preserved
        assert result[50, 50, 3] == 200  # Alpha preserved

    def test_prepare_frame_unsupported_shape(self):
        """Test unsupported frame shape raises error."""
        from sleap_io.rendering.core import _prepare_frame_rgba

        frame = np.zeros((100, 100, 5), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported frame shape"):
            _prepare_frame_rgba(frame)


class TestEstimateFrameSize:
    """Tests for _estimate_frame_size helper function."""

    def test_estimate_frame_size_basic(self):
        """Test frame size estimation from keypoints."""
        from sleap_io.rendering.core import _estimate_frame_size

        points = [
            np.array([[100, 100], [200, 200], [150, 150]]),
            np.array([[120, 120], [180, 180]]),
        ]

        h, w = _estimate_frame_size(points)

        # Should be at least as large as max coordinates plus padding
        assert w >= 200
        assert h >= 200

    def test_estimate_frame_size_empty_list(self):
        """Test frame size estimation with empty list."""
        from sleap_io.rendering.core import _estimate_frame_size

        h, w = _estimate_frame_size([])

        assert h == 64  # min_size
        assert w == 64

    def test_estimate_frame_size_all_nan(self):
        """Test frame size estimation when all points are NaN."""
        from sleap_io.rendering.core import _estimate_frame_size

        points = [np.array([[np.nan, np.nan], [np.nan, np.nan]])]

        h, w = _estimate_frame_size(points)

        assert h == 64  # min_size
        assert w == 64


class TestApplyCrop:
    """Tests for _apply_crop helper function."""

    def test_apply_crop_out_of_bounds(self):
        """Test crop extending beyond frame bounds adds padding."""
        from sleap_io.rendering.core import _apply_crop

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[50, 50]])]
        # Crop extends 50px outside frame on all sides
        crop = (-50, -50, 150, 150)

        cropped_frame, shifted_points, scale = _apply_crop(frame, points, crop)

        assert cropped_frame.shape == (200, 200, 3)
        # Original content should be offset by 50px
        assert cropped_frame[50:150, 50:150, 0].mean() == pytest.approx(128, abs=1)
        # Padding should be zeros
        assert cropped_frame[0:50, 0:50, 0].mean() == 0

    def test_apply_crop_with_output_size(self):
        """Test crop with output size scaling."""
        from sleap_io.rendering.core import _apply_crop

        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        points = [np.array([[100, 100]])]
        crop = (50, 50, 150, 150)  # 100x100 crop
        output_size = (200, 200)  # Scale up 2x

        cropped_frame, shifted_points, scale = _apply_crop(
            frame, points, crop, output_size
        )

        assert cropped_frame.shape == (200, 200, 3)
        assert scale == pytest.approx(2.0)
        # Point should be shifted and scaled
        assert shifted_points[0][0, 0] == pytest.approx((100 - 50) * 2)
        assert shifted_points[0][0, 1] == pytest.approx((100 - 50) * 2)


class TestCoreRendering:
    """Tests for sleap_io.rendering.core module."""

    def test_render_frame_basic(self):
        """Test basic frame rendering with synthetic data."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[20, 20], [50, 50], [80, 80]])]
        edges = [(0, 1), (1, 2)]
        node_names = ["a", "b", "c"]

        rendered = render_frame(frame, points, edges, node_names)

        assert rendered.shape == (100, 100, 3)
        assert rendered.dtype == np.uint8

    def test_render_frame_with_scale(self):
        """Test frame rendering with scaling."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[20, 20], [50, 50]])]
        edges = [(0, 1)]
        node_names = ["a", "b"]

        rendered = render_frame(frame, points, edges, node_names, scale=0.5)

        assert rendered.shape == (50, 50, 3)

    def test_render_frame_grayscale(self):
        """Test frame rendering with grayscale input."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100), 128, dtype=np.uint8)  # Grayscale
        points = [np.array([[50, 50]])]
        edges = []
        node_names = ["a"]

        rendered = render_frame(frame, points, edges, node_names)

        assert rendered.shape == (100, 100, 3)

    def test_render_frame_with_nan(self):
        """Test frame rendering handles NaN points."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[20, 20], [np.nan, np.nan], [80, 80]])]
        edges = [(0, 1), (1, 2)]
        node_names = ["a", "b", "c"]

        # Should not raise
        rendered = render_frame(frame, points, edges, node_names)
        assert rendered.shape == (100, 100, 3)

    def test_render_frame_show_options(self):
        """Test frame rendering with show_nodes/show_edges toggles."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 0, dtype=np.uint8)
        points = [np.array([[50, 50]])]
        edges = []
        node_names = ["a"]

        # With nodes
        rendered_with = render_frame(frame, points, edges, node_names, show_nodes=True)

        # Without nodes
        rendered_without = render_frame(
            frame, points, edges, node_names, show_nodes=False
        )

        # Center pixel should differ
        assert not np.array_equal(rendered_with[50, 50], rendered_without[50, 50])

    def test_render_frame_color_schemes(self):
        """Test frame rendering with different color schemes."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [
            np.array([[20, 20], [40, 40]]),
            np.array([[60, 60], [80, 80]]),
        ]
        edges = [(0, 1)]
        node_names = ["a", "b"]

        for scheme in ["instance", "node"]:
            rendered = render_frame(frame, points, edges, node_names, color_by=scheme)
            assert rendered.shape == (100, 100, 3)

    def test_render_frame_marker_shapes(self):
        """Test frame rendering with different marker shapes."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[50, 50]])]
        edges = []
        node_names = ["a"]

        for shape in ["circle", "square", "diamond", "triangle", "cross"]:
            rendered = render_frame(
                frame, points, edges, node_names, marker_shape=shape
            )
            assert rendered.shape == (100, 100, 3)

    def test_render_frame_with_callbacks(self):
        """Test frame rendering with callbacks."""
        from sleap_io.rendering.callbacks import InstanceContext, RenderContext
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[50, 50]])]
        edges = []
        node_names = ["a"]

        pre_called = []
        post_called = []
        per_instance_called = []

        def pre_callback(ctx: RenderContext):
            pre_called.append(ctx.frame_idx)

        def post_callback(ctx: RenderContext):
            post_called.append(ctx.frame_idx)

        def per_instance_callback(ctx: InstanceContext):
            per_instance_called.append(ctx.instance_idx)

        render_frame(
            frame,
            points,
            edges,
            node_names,
            frame_idx=42,
            pre_render_callback=pre_callback,
            post_render_callback=post_callback,
            per_instance_callback=per_instance_callback,
        )

        assert pre_called == [42]
        assert post_called == [42]
        assert per_instance_called == [0]

    def test_render_frame_track_coloring_without_track_indices(self):
        """Test track coloring falls back correctly when no track_indices."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[50, 50]])]
        edges = []
        node_names = ["a"]

        # Track coloring without track_indices should still work
        rendered = render_frame(
            frame,
            points,
            edges,
            node_names,
            color_by="track",
            track_indices=None,
            n_tracks=3,
        )

        assert rendered.shape == (100, 100, 3)

    def test_render_frame_empty_instances(self):
        """Test render_frame with no instances."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = []  # No instances
        edges = [(0, 1)]
        node_names = ["a", "b"]

        rendered = render_frame(frame, points, edges, node_names)

        assert rendered.shape == (100, 100, 3)
        # Should just be the original frame (no nodes/edges drawn)

    def test_render_frame_edge_out_of_bounds(self):
        """Test render_frame handles edge indices beyond points."""
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[50, 50]])]  # Only 1 point
        edges = [(0, 1), (1, 2)]  # Edges reference non-existent points
        node_names = ["a"]

        # Should not raise, just skip invalid edges
        rendered = render_frame(frame, points, edges, node_names)
        assert rendered.shape == (100, 100, 3)

    def test_render_frame_with_instance_metadata(self):
        """Test render_frame passes instance metadata to callbacks."""
        from sleap_io.rendering.callbacks import InstanceContext
        from sleap_io.rendering.core import render_frame

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        points = [np.array([[50, 50]])]
        edges = []
        node_names = ["a"]

        received_metadata = []

        def per_instance_cb(ctx: InstanceContext):
            received_metadata.append(
                {
                    "track_name": ctx.track_name,
                    "confidence": ctx.confidence,
                }
            )

        render_frame(
            frame,
            points,
            edges,
            node_names,
            per_instance_callback=per_instance_cb,
            instance_metadata=[{"track_name": "track1", "confidence": 0.95}],
        )

        assert len(received_metadata) == 1
        assert received_metadata[0]["track_name"] == "track1"
        assert received_metadata[0]["confidence"] == 0.95


# ============================================================================
# render_image Tests
# ============================================================================


class TestRenderImage:
    """Tests for render_image function."""

    def test_render_image_from_labeled_frame(self, labels_predictions):
        """Test render_image with a LabeledFrame."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]

        # Need to provide image since we're using test data
        frame = lf.video[lf.frame_idx]

        rendered = render_image(lf, image=frame)

        assert isinstance(rendered, np.ndarray)
        assert rendered.ndim == 3
        assert rendered.shape[2] == 3

    def test_render_image_from_labels(self, labels_predictions):
        """Test render_image with Labels object."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        rendered = render_image(labels_predictions, frame_idx=0, image=frame)

        assert isinstance(rendered, np.ndarray)
        assert rendered.ndim == 3

    def test_render_image_with_scale(self, labels_predictions):
        """Test render_image with scale parameter."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        rendered = render_image(lf, image=frame, scale=0.5)

        # Should be scaled
        assert rendered.shape[0] == frame.shape[0] // 2
        assert rendered.shape[1] == frame.shape[1] // 2

    def test_render_image_save_to_file(self, labels_predictions, tmp_path):
        """Test render_image saves to file."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]
        output_path = tmp_path / "rendered.png"

        rendered = render_image(lf, save_path=output_path, image=frame)

        assert output_path.exists()
        assert isinstance(rendered, np.ndarray)

    def test_render_image_background_color(self, labels_predictions):
        """Test render_image with solid color background."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]

        # Using background=<color> renders on solid color, skips video loading
        rendered = render_image(lf, background=(128, 128, 128))

        assert isinstance(rendered, np.ndarray)
        assert rendered.shape[2] == 3

        # Test named color
        rendered2 = render_image(lf, background="gray")
        assert isinstance(rendered2, np.ndarray)

        # Test hex color
        rendered3 = render_image(lf, background="#808080")
        assert isinstance(rendered3, np.ndarray)

    def test_render_image_crop_explicit(self, labels_predictions):
        """Test render_image with explicit crop bounds."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        # Crop to 100x100 region
        rendered = render_image(lf, image=frame, crop=(100, 100, 200, 200))

        assert isinstance(rendered, np.ndarray)
        assert rendered.shape[0] == 100  # height
        assert rendered.shape[1] == 100  # width
        assert rendered.shape[2] == 3

    def test_render_image_crop_with_scale(self, labels_predictions):
        """Test render_image with crop and scale combined."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        # Crop to 100x100, then scale 2x
        rendered = render_image(lf, image=frame, crop=(100, 100, 200, 200), scale=2.0)

        assert isinstance(rendered, np.ndarray)
        assert rendered.shape[0] == 200  # 100 * 2
        assert rendered.shape[1] == 200  # 100 * 2

    def test_render_image_crop_normalized(self, labels_predictions):
        """Test render_image with normalized crop coordinates (floats in 0-1)."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]
        h, w = frame.shape[:2]

        # Crop center 50% using normalized coordinates
        rendered = render_image(lf, image=frame, crop=(0.25, 0.25, 0.75, 0.75))

        assert isinstance(rendered, np.ndarray)
        # Should be approximately half the width and height
        assert rendered.shape[0] == int(0.5 * h)
        assert rendered.shape[1] == int(0.5 * w)
        assert rendered.shape[2] == 3

    def test_render_image_crop_normalized_full_frame(self, labels_predictions):
        """Test normalized crop (0.0, 0.0, 1.0, 1.0) returns full frame."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]
        h, w = frame.shape[:2]

        # Full frame in normalized coordinates
        rendered = render_image(lf, image=frame, crop=(0.0, 0.0, 1.0, 1.0))

        assert isinstance(rendered, np.ndarray)
        assert rendered.shape[0] == h
        assert rendered.shape[1] == w

    def test_render_image_crop_int_not_normalized(self, labels_predictions):
        """Test that int tuples are treated as pixel coords even if small values."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        # Small int values should be pixels, not normalized
        # (0, 0, 1, 1) with ints should give 1x1 crop, not full frame
        rendered = render_image(lf, image=frame, crop=(0, 0, 10, 10))

        assert isinstance(rendered, np.ndarray)
        assert rendered.shape[0] == 10  # 10 pixels
        assert rendered.shape[1] == 10  # 10 pixels

    def test_render_image_from_instance_list(self, labels_predictions):
        """Test render_image with list of instances requires image."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        instances = list(lf.instances)
        frame = lf.video[lf.frame_idx]

        # With image provided, should work
        rendered = render_image(instances, image=frame)
        assert isinstance(rendered, np.ndarray)
        assert rendered.shape[2] == 3

    def test_render_image_from_instance_list_no_image_error(self, labels_predictions):
        """Test render_image with instance list raises error if no image."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        instances = list(lf.instances)

        with pytest.raises(ValueError, match="image parameter required"):
            render_image(instances)

    def test_render_image_empty_instance_list_error(self, labels_predictions):
        """Test render_image with empty instance list raises error."""
        from sleap_io.rendering import render_image

        with pytest.raises(ValueError, match="Empty instances list"):
            render_image([], image=np.zeros((100, 100, 3), dtype=np.uint8))

    def test_render_image_invalid_source_type(self):
        """Test render_image with invalid source type."""
        from sleap_io.rendering import render_image

        with pytest.raises(TypeError, match="must be Labels, LabeledFrame"):
            render_image("not_valid")

    def test_render_image_labels_with_video_frame_idx(self, labels_predictions):
        """Test render_image with Labels + video + frame_idx."""
        from sleap_io.rendering import render_image

        # Get a frame that exists
        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        # Render by video + frame_idx
        rendered = render_image(
            labels_predictions,
            video=0,
            frame_idx=lf.frame_idx,
            image=frame,
        )

        assert isinstance(rendered, np.ndarray)

    def test_render_image_labels_no_frame_found_error(self, labels_predictions):
        """Test render_image raises error when frame not found."""
        from sleap_io.rendering import render_image

        with pytest.raises(ValueError, match="No labeled frame found"):
            render_image(
                labels_predictions,
                video=0,
                frame_idx=999999,  # Doesn't exist
                image=np.zeros((100, 100, 3), dtype=np.uint8),
            )

    def test_render_image_labels_default_first_frame(self, labels_predictions):
        """Test render_image defaults to first labeled frame."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]
        frame = lf.video[lf.frame_idx]

        # No lf_ind, no video/frame_idx -> should use first frame
        rendered = render_image(labels_predictions, image=frame)

        assert isinstance(rendered, np.ndarray)

    def test_render_image_background_with_no_video_metadata(self, labels_predictions):
        """Test render_image with background color when video has no shape."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]

        # Render with background color - should estimate frame size from points
        rendered = render_image(lf, background="black")

        assert isinstance(rendered, np.ndarray)
        # Should have some reasonable size based on keypoints

    def test_render_image_labeled_frame_without_instances(self):
        """``render_image(source=lf)`` works when the LabeledFrame has no instances.

        Before this was fixed, a bare ``LabeledFrame`` without instances raised
        ``ValueError("LabeledFrame has no instances with skeleton")``, forcing
        segmentation-only workflows to pass ``source=None, image=lf.image``.
        The LabeledFrame source now falls through to empty pose-rendering state
        (mirroring the centroid-only path in the ``Labels`` branch).
        """
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_image

        video = Video(filename="dummy.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[])

        # With an explicit image — should render the image as-is.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        rendered = render_image(lf, image=img)
        assert isinstance(rendered, np.ndarray)
        assert rendered.shape == (100, 100, 3)

        # With a solid-color background — should render a blank frame.
        rendered_bg = render_image(lf, background="black")
        assert isinstance(rendered_bg, np.ndarray)
        assert rendered_bg.ndim == 3 and rendered_bg.shape[-1] == 3

    def test_render_image_video_unavailable_with_background(self, labels_predictions):
        """Test frame size estimation from keypoints when video unavailable."""
        # Make a copy so we don't modify the fixture
        import copy

        from sleap_io.rendering import render_image

        labels = copy.deepcopy(labels_predictions)
        lf = labels.labeled_frames[0]

        # Replace video filename with non-existent path to make video unavailable
        labels.video.replace_filename("nonexistent_video.mp4", open=True)
        assert labels.video.backend is None

        # Also clear the cached shape metadata to force estimation from keypoints
        labels.video.backend_metadata.pop("shape", None)
        assert labels.video.shape is None

        # With background color, should estimate frame size from keypoints
        rendered = render_image(lf, background="black")

        assert isinstance(rendered, np.ndarray)
        assert rendered.ndim == 3
        assert rendered.shape[2] == 3
        # Frame size should be estimated from keypoints with padding
        assert rendered.shape[0] > 0
        assert rendered.shape[1] > 0

    def test_render_image_video_unavailable_no_background_error(
        self, labels_predictions
    ):
        """Test error raised when video unavailable and no background."""
        # Make a copy so we don't modify the fixture
        import copy

        from sleap_io.rendering import render_image

        labels = copy.deepcopy(labels_predictions)
        lf = labels.labeled_frames[0]

        # Replace video filename with non-existent path to make video unavailable
        labels.video.replace_filename("nonexistent_video.mp4", open=True)
        assert labels.video.backend is None

        # Without background color, should raise helpful error
        with pytest.raises(ValueError, match="Video unavailable.*background"):
            render_image(lf)

    def test_render_image_labels_video_unavailable_with_background(
        self, labels_predictions
    ):
        """Test render_image with Labels source when video unavailable."""
        # Make a copy so we don't modify the fixture
        import copy

        from sleap_io.rendering import render_image

        labels = copy.deepcopy(labels_predictions)

        # Replace video filename with non-existent path
        labels.video.replace_filename("nonexistent_video.mp4", open=True)
        assert labels.video.backend is None

        # Also clear the cached shape metadata to force estimation from keypoints
        labels.video.backend_metadata.pop("shape", None)
        assert labels.video.shape is None

        # With background color, should estimate frame size from keypoints
        rendered = render_image(labels, lf_ind=0, background="gray")

        assert isinstance(rendered, np.ndarray)
        assert rendered.ndim == 3
        assert rendered.shape[2] == 3

    def test_render_image_labels_video_unavailable_no_background_error(
        self, labels_predictions
    ):
        """Test render_image with Labels source raises error when video unavailable."""
        # Make a copy so we don't modify the fixture
        import copy

        from sleap_io.rendering import render_image

        labels = copy.deepcopy(labels_predictions)

        # Replace video filename with non-existent path
        labels.video.replace_filename("nonexistent_video.mp4", open=True)
        assert labels.video.backend is None

        # Without background color, should raise helpful error
        with pytest.raises(ValueError, match="Video unavailable.*background"):
            render_image(labels, lf_ind=0)


# ============================================================================
# _resolve_crop Tests
# ============================================================================


class TestResolveCrop:
    """Tests for _resolve_crop helper function."""

    def test_resolve_crop_pixel_coordinates(self):
        """Test pixel coordinates (int tuple) pass through."""
        from sleap_io.rendering.core import _resolve_crop

        result = _resolve_crop((100, 100, 300, 300), (480, 640))
        assert result == (100, 100, 300, 300)

    def test_resolve_crop_normalized_coordinates(self):
        """Test normalized coordinates (float tuple) are scaled."""
        from sleap_io.rendering.core import _resolve_crop

        # Center 50% of a 480x640 frame
        result = _resolve_crop((0.25, 0.25, 0.75, 0.75), (480, 640))
        # x: 0.25 * 640 = 160, 0.75 * 640 = 480
        # y: 0.25 * 480 = 120, 0.75 * 480 = 360
        assert result == (160, 120, 480, 360)

    def test_resolve_crop_full_frame_normalized(self):
        """Test (0.0, 0.0, 1.0, 1.0) gives full frame."""
        from sleap_io.rendering.core import _resolve_crop

        result = _resolve_crop((0.0, 0.0, 1.0, 1.0), (480, 640))
        assert result == (0, 0, 640, 480)

    def test_resolve_crop_float_out_of_range_treated_as_pixels(self):
        """Test floats outside [0, 1] are treated as pixel coordinates."""
        from sleap_io.rendering.core import _resolve_crop

        # Values > 1.0 should be treated as pixels
        result = _resolve_crop((100.0, 100.0, 300.0, 300.0), (480, 640))
        assert result == (100, 100, 300, 300)

    def test_resolve_crop_mixed_types_treated_as_pixels(self):
        """Test mixed int/float tuples are treated as pixel coordinates."""
        from sleap_io.rendering.core import _resolve_crop

        # Mixed types should be treated as pixels
        result = _resolve_crop((100, 100.0, 300, 300.0), (480, 640))
        assert result == (100, 100, 300, 300)


# ============================================================================
# render_video Tests
# ============================================================================


class TestRenderVideo:
    """Tests for render_video function."""

    def test_render_video_to_array(self, labels_predictions):
        """Test render_video returns arrays when no output specified."""
        from sleap_io.rendering import render_video

        # Render just first 2 frames
        frame_inds = [
            labels_predictions.labeled_frames[0].frame_idx,
            labels_predictions.labeled_frames[1].frame_idx,
        ]

        frames = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            show_progress=False,
        )

        assert isinstance(frames, list)
        assert len(frames) == 2
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_render_video_to_file(self, labels_predictions, tmp_path):
        """Test render_video saves to file."""
        from sleap_io.rendering import render_video

        output_path = tmp_path / "rendered.mp4"

        # Render just 2 frames
        frame_inds = [
            labels_predictions.labeled_frames[0].frame_idx,
            labels_predictions.labeled_frames[1].frame_idx,
        ]

        result = render_video(
            labels_predictions,
            output_path,
            frame_inds=frame_inds,
            show_progress=False,
        )

        assert output_path.exists()
        # Should return Video object
        from sleap_io.model.video import Video

        assert isinstance(result, Video)

    def test_render_video_with_preset(self, labels_predictions):
        """Test render_video with preset parameter."""
        from sleap_io.rendering import render_video

        frame_inds = [labels_predictions.labeled_frames[0].frame_idx]

        frames = render_video(
            labels_predictions,
            preset="preview",  # 0.25x scale
            frame_inds=frame_inds,
            show_progress=False,
        )

        assert len(frames) == 1
        # Preview is 0.25x, so frame should be smaller
        original_lf = labels_predictions.labeled_frames[0]
        original_frame = original_lf.video[original_lf.frame_idx]

        expected_h = int(original_frame.shape[0] * 0.25)
        assert frames[0].shape[0] == expected_h

    def test_render_video_with_start_end(self, labels_predictions):
        """Test render_video with start/end parameters."""
        from sleap_io.rendering import render_video

        all_indices = [lf.frame_idx for lf in labels_predictions.labeled_frames]
        start = min(all_indices)
        end = min(all_indices) + 3

        frames = render_video(
            labels_predictions,
            start=start,
            end=end,
            show_progress=False,
        )

        # Should have frames in range [start, end)
        assert len(frames) <= (end - start)

    def test_render_video_with_crop(self, labels_predictions):
        """Test render_video with static crop applied to all frames."""
        from sleap_io.rendering import render_video

        all_indices = [lf.frame_idx for lf in labels_predictions.labeled_frames]
        start = min(all_indices)
        end = min(all_indices) + 2

        # Render without crop
        frames_full = render_video(
            labels_predictions,
            start=start,
            end=end,
            show_progress=False,
        )

        # Render with pixel crop
        frames_cropped = render_video(
            labels_predictions,
            start=start,
            end=end,
            crop=(100, 100, 300, 300),
            show_progress=False,
        )

        # Cropped frames should have the crop dimensions
        assert len(frames_cropped) == len(frames_full)
        for frame in frames_cropped:
            assert frame.shape[0] == 200  # 300 - 100
            assert frame.shape[1] == 200  # 300 - 100
            assert frame.shape[2] == 3

    def test_render_video_with_crop_normalized(self, labels_predictions):
        """Test render_video with normalized crop coordinates."""
        from sleap_io.rendering import render_video

        all_indices = [lf.frame_idx for lf in labels_predictions.labeled_frames]
        start = min(all_indices)
        end = min(all_indices) + 2

        # Get original frame dimensions
        lf = labels_predictions.labeled_frames[0]
        original_frame = lf.video[lf.frame_idx]
        h, w = original_frame.shape[:2]

        # Render with normalized crop (center 50%)
        frames = render_video(
            labels_predictions,
            start=start,
            end=end,
            crop=(0.25, 0.25, 0.75, 0.75),
            show_progress=False,
        )

        # Cropped frames should be approximately half size
        for frame in frames:
            assert frame.shape[0] == int(0.5 * h)
            assert frame.shape[1] == int(0.5 * w)
            assert frame.shape[2] == 3

    def test_render_video_progress_callback(self, labels_predictions):
        """Test render_video with progress callback."""
        from sleap_io.rendering import render_video

        frame_inds = [
            labels_predictions.labeled_frames[0].frame_idx,
            labels_predictions.labeled_frames[1].frame_idx,
        ]

        progress_calls = []

        def progress_cb(current, total):
            progress_calls.append((current, total))
            return True

        render_video(
            labels_predictions,
            frame_inds=frame_inds,
            progress_callback=progress_cb,
            show_progress=False,
        )

        assert len(progress_calls) == 2
        assert progress_calls[0] == (0, 2)
        assert progress_calls[1] == (1, 2)

    def test_render_video_cancellation(self, labels_predictions):
        """Test render_video cancellation via progress callback."""
        from sleap_io.rendering import render_video

        frame_inds = [labels_predictions.labeled_frames[i].frame_idx for i in range(5)]

        def cancel_cb(current, total):
            return current < 2  # Cancel after 2 frames

        frames = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            progress_callback=cancel_cb,
            show_progress=False,
        )

        # Should have stopped early
        assert len(frames) <= 2

    def test_render_video_from_labeled_frame_list(self, labels_predictions):
        """Test render_video with list of LabeledFrames."""
        from sleap_io.rendering import render_video

        # Use first 2 labeled frames as a list
        lf_list = labels_predictions.labeled_frames[:2]

        frames = render_video(
            lf_list,
            show_progress=False,
        )

        assert isinstance(frames, list)
        assert len(frames) == 2

    def test_render_video_empty_labeled_frame_list_error(self):
        """Test render_video with empty list raises error."""
        from sleap_io.rendering import render_video

        with pytest.raises(ValueError, match="Empty labeled frames list"):
            render_video([], show_progress=False)

    def test_render_video_no_frames_to_render_error(self, labels_predictions):
        """Test render_video raises error when no frames match criteria."""
        from sleap_io.rendering import render_video

        with pytest.raises(ValueError, match="No frames to render"):
            render_video(
                labels_predictions,
                frame_inds=[],  # Empty frame list
                show_progress=False,
            )

    def test_render_video_invalid_source_type(self):
        """Test render_video with invalid source type."""
        from sleap_io.rendering import render_video

        with pytest.raises(TypeError, match="must be Labels or list of LabeledFrame"):
            render_video("not_valid", show_progress=False)

    def test_render_video_no_videos_error(self):
        """Test render_video raises error when Labels has no videos."""
        from sleap_io.model.labels import Labels
        from sleap_io.rendering import render_video

        labels = Labels()

        with pytest.raises(ValueError, match="Labels has no videos"):
            render_video(labels, show_progress=False)

    def test_render_video_no_skeleton_error(self):
        """Test render_video raises error when instances exist without skeleton.

        Frames with only annotations (no instances) should not require a
        skeleton and should not raise.
        """
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.labels import Labels
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        video = Video(filename="dummy.mp4")
        # Empty frame (no instances, no annotations) — should not raise
        lf = LabeledFrame(video=video, frame_idx=0, instances=[])
        labels = Labels(videos=[video], labeled_frames=[lf])

        # No skeleton needed for empty frames
        frames = render_video(labels, show_progress=False, background="black")
        assert len(frames) == 1

    def test_render_video_with_background_color(self, labels_predictions):
        """Test render_video with solid background color."""
        from sleap_io.rendering import render_video

        frame_inds = [labels_predictions.labeled_frames[0].frame_idx]

        frames = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            background="gray",
            show_progress=False,
        )

        assert len(frames) == 1
        assert isinstance(frames[0], np.ndarray)

    def test_render_video_labeled_frame_list_no_skeleton_error(self):
        """Test render_video with labeled frames that have no skeleton."""
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        video = Video(filename="dummy.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[])

        with pytest.raises(ValueError, match="No skeleton found"):
            render_video([lf], show_progress=False)

    def test_render_video_skeleton_from_instances(self, labels_predictions):
        """Test render_video finds skeleton from instances when not in skeletons."""
        from sleap_io.rendering import render_video

        # This tests the loop that finds skeleton from instances
        # The standard test data already tests this path since skeleton
        # is typically found via instances
        frame_inds = [labels_predictions.labeled_frames[0].frame_idx]

        frames = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            show_progress=False,
        )

        assert len(frames) == 1

    def test_render_video_start_end_range(self, labels_predictions):
        """Test render_video with start/end that filters labeled frames."""
        from sleap_io.rendering import render_video

        # Get all frame indices
        all_indices = sorted(lf.frame_idx for lf in labels_predictions.labeled_frames)

        # Render just a subset
        start = all_indices[1]
        end = all_indices[3]

        frames = render_video(
            labels_predictions,
            start=start,
            end=end,
            show_progress=False,
        )

        # Should only include frames in [start, end) range
        assert len(frames) >= 1

    def test_render_video_video_object_param(self, labels_predictions):
        """Test render_video with Video object instead of int."""
        from sleap_io.rendering import render_video

        video_obj = labels_predictions.videos[0]
        frame_inds = [labels_predictions.labeled_frames[0].frame_idx]

        frames = render_video(
            labels_predictions,
            video=video_obj,  # Pass Video object directly
            frame_inds=frame_inds,
            show_progress=False,
        )

        assert len(frames) == 1

    def test_render_video_track_index_fallback(self, labels_predictions):
        """Test render_video handles instances without tracks."""
        from sleap_io.rendering import render_video

        # Remove tracks from instances to test fallback path
        lf = labels_predictions.labeled_frames[0]
        for inst in lf.instances:
            inst.track = None

        frame_inds = [lf.frame_idx]

        frames = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            color_by="track",  # Force track coloring
            show_progress=False,
        )

        assert len(frames) == 1

    def test_render_video_include_unlabeled_to_file(self, tmp_path):
        """Test render_video with include_unlabeled streams unlabeled frames to file.

        This test ensures the streaming write path works for frames without labels.
        Creates a labels object with gaps (only frames 0 and 5 labeled) and renders
        the full range 0-6, which includes unlabeled frames 1-4.
        """
        import numpy as np

        import sleap_io as sio
        from sleap_io.rendering import render_video

        # Create skeleton
        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Create a small test video
        video_path = tmp_path / "test_video.mp4"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
        sio.save_video(frames, video_path)
        video = sio.Video.from_filename(str(video_path))

        # Create labeled frames with gaps - only frames 0 and 5 have labels
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[
                sio.Instance.from_numpy(
                    np.array([[32, 20], [32, 44]]), skeleton=skeleton
                )
            ],
        )
        lf5 = sio.LabeledFrame(
            video=video,
            frame_idx=5,
            instances=[
                sio.Instance.from_numpy(
                    np.array([[32, 20], [32, 44]]), skeleton=skeleton
                )
            ],
        )
        labels = sio.Labels(
            labeled_frames=[lf0, lf5], videos=[video], skeletons=[skeleton]
        )

        output_path = tmp_path / "unlabeled_rendered.mp4"

        # Render frames 0-6, which includes unlabeled frames 1-4
        result = render_video(
            labels,
            output_path,
            start=0,
            end=6,
            include_unlabeled=True,  # This renders frames 1-4 without poses
            show_progress=False,
        )

        assert output_path.exists()
        assert isinstance(result, sio.Video)

    def test_render_video_include_unlabeled_with_background(self, tmp_path):
        """Test include_unlabeled with background color covers unlabeled frame path."""
        import numpy as np

        import sleap_io as sio
        from sleap_io.rendering import render_video

        # Create skeleton
        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Create a small test video
        video_path = tmp_path / "test_video.mp4"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
        sio.save_video(frames, video_path)
        video = sio.Video.from_filename(str(video_path))

        # Create labeled frames with gaps - only frames 0 and 5 have labels
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[
                sio.Instance.from_numpy(
                    np.array([[32, 20], [32, 44]]), skeleton=skeleton
                )
            ],
        )
        lf5 = sio.LabeledFrame(
            video=video,
            frame_idx=5,
            instances=[
                sio.Instance.from_numpy(
                    np.array([[32, 20], [32, 44]]), skeleton=skeleton
                )
            ],
        )
        labels = sio.Labels(
            labeled_frames=[lf0, lf5], videos=[video], skeletons=[skeleton]
        )

        output_path = tmp_path / "unlabeled_bg.mp4"

        # Render with background color - exercises the background_color path
        # for unlabeled frames (frames 1-4)
        result = render_video(
            labels,
            output_path,
            start=0,
            end=6,
            include_unlabeled=True,
            background="black",  # Use solid background
            show_progress=False,
        )

        assert output_path.exists()
        assert isinstance(result, sio.Video)

    def test_render_video_include_unlabeled_return_list(self, tmp_path):
        """Test include_unlabeled=True without save_path returns frame list."""
        import numpy as np

        import sleap_io as sio
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Create a small test video
        video_path = tmp_path / "test_video.mp4"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
        sio.save_video(frames, video_path)
        video = sio.Video.from_filename(str(video_path))

        # Create labeled frames with gaps - only frames 0 and 5 have labels
        pts = np.array([[32, 20], [32, 44]])
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        lf5 = sio.LabeledFrame(
            video=video,
            frame_idx=5,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(
            labeled_frames=[lf0, lf5], videos=[video], skeletons=[skeleton]
        )

        # Render without save_path - returns list (covers line 1176)
        result = render_video(
            labels,
            save_path=None,  # No file output, return list
            start=0,
            end=6,
            include_unlabeled=True,
            show_progress=False,
        )

        # Should return list of frames (labeled + unlabeled = 6 frames: 0-5)
        assert isinstance(result, list)
        assert len(result) == 6
        assert all(isinstance(f, np.ndarray) for f in result)

    def test_render_video_skip_unlabeled(self, tmp_path):
        """Test include_unlabeled=False skips frames without labels."""
        import numpy as np

        import sleap_io as sio
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Create a small test video
        video_path = tmp_path / "test_video.mp4"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
        sio.save_video(frames, video_path)
        video = sio.Video.from_filename(str(video_path))

        # Create labeled frames with gaps - only frames 0 and 5 have labels
        pts = np.array([[32, 20], [32, 44]])
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        lf5 = sio.LabeledFrame(
            video=video,
            frame_idx=5,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(
            labeled_frames=[lf0, lf5], videos=[video], skeletons=[skeleton]
        )

        # Render with include_unlabeled=False (covers line 1118)
        result = render_video(
            labels,
            save_path=None,
            start=0,
            end=6,
            include_unlabeled=False,  # Skip unlabeled frames
            show_progress=False,
        )

        # Should only return 2 frames (the labeled ones: 0 and 5)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_render_video_unlabeled_with_crop(self, tmp_path):
        """Test include_unlabeled with crop_bounds exercises crop path."""
        import numpy as np

        import sleap_io as sio
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Create a small test video
        video_path = tmp_path / "test_video.mp4"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
        sio.save_video(frames, video_path)
        video = sio.Video.from_filename(str(video_path))

        # Create labeled frames with gaps
        pts = np.array([[32, 20], [32, 44]])
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        lf5 = sio.LabeledFrame(
            video=video,
            frame_idx=5,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(
            labeled_frames=[lf0, lf5], videos=[video], skeletons=[skeleton]
        )

        output_path = tmp_path / "crop_unlabeled.mp4"

        # Render with crop_bounds and include_unlabeled (covers line 1145)
        result = render_video(
            labels,
            output_path,
            start=0,
            end=6,
            include_unlabeled=True,
            crop=(10, 10, 50, 50),  # Crop bounds
            show_progress=False,
        )

        assert output_path.exists()
        assert isinstance(result, sio.Video)

    def test_render_video_skip_unlabeled_with_frame_inds(self):
        """Test include_unlabeled=False skips unlabeled frames via continue."""
        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Video with no backend (shape=None, no file needed)
        video = Video(filename="dummy.mp4", backend=None, open_backend=False)

        # Create labeled frames at indices 0 and 3 only
        pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        lf3 = sio.LabeledFrame(
            video=video,
            frame_idx=3,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(
            labeled_frames=[lf0, lf3], videos=[video], skeletons=[skeleton]
        )

        # Pass frame_inds that include unlabeled indices (1, 2)
        # With include_unlabeled=False, those frames hit lf is None -> continue
        result = render_video(
            labels,
            save_path=None,
            frame_inds=[0, 1, 2, 3],
            include_unlabeled=False,
            background="black",
            fps=30,
            show_progress=False,
        )

        # Only labeled frames 0 and 3 should be rendered
        assert isinstance(result, list)
        assert len(result) == 2

    def test_render_video_unlabeled_bg_color_no_video_shape(self):
        """Test unlabeled frame with background_color and video with no shape."""
        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Video with no backend -> shape is None
        video = Video(filename="dummy.mp4", backend=None, open_backend=False)

        # Only frame 0 is labeled
        pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(labeled_frames=[lf0], videos=[video], skeletons=[skeleton])

        # frame_inds includes unlabeled frame 1
        # background_color set, video has no shape -> h, w = 64, 64 fallback
        result = render_video(
            labels,
            save_path=None,
            frame_inds=[0, 1],
            include_unlabeled=True,
            background="black",
            fps=30,
            show_progress=False,
        )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_render_video_unlabeled_video_unavailable(self):
        """Test unlabeled frame raises ValueError when video can't be read."""
        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Video with no backend -> reading frames raises
        video = Video(filename="dummy.mp4", backend=None, open_backend=False)

        pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(labeled_frames=[lf0], videos=[video], skeletons=[skeleton])

        # frame_inds only includes unlabeled frame 1, no background_color
        # Video can't be read -> ValueError
        with pytest.raises(ValueError, match="Video unavailable at frame 1"):
            render_video(
                labels,
                save_path=None,
                frame_inds=[1],
                include_unlabeled=True,
                background="video",
                fps=30,
                show_progress=False,
            )

    def test_render_video_labeled_bg_color_no_video_shape(self):
        """Test labeled frame with background_color and video with no shape."""
        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Video with no backend -> shape is None
        video = Video(filename="dummy.mp4", backend=None, open_backend=False)

        pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(labeled_frames=[lf0], videos=[video], skeletons=[skeleton])

        # background_color set, video has no shape -> _estimate_frame_size fallback
        result = render_video(
            labels,
            save_path=None,
            frame_inds=[0],
            background="black",
            fps=30,
            show_progress=False,
        )

        assert isinstance(result, list)
        assert len(result) == 1

    def test_render_video_labeled_image_unavailable(self):
        """Test labeled frame raises ValueError when image can't be read."""
        import sleap_io as sio
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])

        # Video with no backend -> lf.image raises
        video = Video(filename="dummy.mp4", backend=None, open_backend=False)

        pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
        lf0 = sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labels = sio.Labels(labeled_frames=[lf0], videos=[video], skeletons=[skeleton])

        # No background_color -> tries to read lf.image -> fails
        with pytest.raises(ValueError, match="Video unavailable at frame 0"):
            render_video(
                labels,
                save_path=None,
                frame_inds=[0],
                background="video",
                fps=30,
                show_progress=False,
            )


# ============================================================================
# Labels.render() Method Tests
# ============================================================================


class TestLabelsRenderMethod:
    """Tests for Labels.render() method."""

    def test_labels_render_method(self, labels_predictions):
        """Test Labels.render() convenience method."""
        frame_inds = [labels_predictions.labeled_frames[0].frame_idx]

        frames = labels_predictions.render(
            frame_inds=frame_inds,
            show_progress=False,
        )

        assert isinstance(frames, list)
        assert len(frames) == 1


# ============================================================================
# Top-level API Tests
# ============================================================================


class TestTopLevelAPI:
    """Tests for top-level sio.render_* functions."""

    def test_render_video_import(self):
        """Test render_video is importable from sleap_io."""
        from sleap_io import render_video

        assert callable(render_video)

    def test_render_image_import(self):
        """Test render_image is importable from sleap_io."""
        from sleap_io import render_image

        assert callable(render_image)

    def test_get_palette_import(self):
        """Test get_palette is importable from sleap_io."""
        from sleap_io import get_palette

        colors = get_palette("distinct", 5)
        assert len(colors) == 5

    def test_callback_context_imports(self):
        """Test callback context classes are importable."""
        from sleap_io import InstanceContext, RenderContext

        assert RenderContext is not None
        assert InstanceContext is not None


# ============================================================================
# Overlay Drawing Tests
# ============================================================================


def test_draw_rois_outline():
    """draw_rois should draw ROI outlines on an image."""
    from sleap_io.rendering.overlays import draw_rois

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = UserROI.from_bbox(10, 10, 30, 30)
    result = draw_rois(img, [roi], color=(0, 255, 0), line_width=1)

    # The outline should have green pixels on the boundary
    assert result is img  # Modified in place
    assert result[10, 10].tolist() == [0, 255, 0]  # Top-left corner
    assert result[50, 50].tolist() == [0, 0, 0]  # Outside


def test_draw_rois_with_fill():
    """draw_rois with fill_alpha should fill the interior."""
    from sleap_io.rendering.overlays import draw_rois

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = UserROI.from_bbox(10, 10, 30, 30)
    result = draw_rois(img, [roi], color=(255, 0, 0), fill_alpha=1.0)

    # Interior should be filled
    assert result[25, 25, 0] == 255  # Red channel filled


def test_draw_masks():
    """draw_masks should draw colored overlays for masks."""
    from sleap_io.rendering.overlays import draw_masks

    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    mask_data = np.zeros((50, 50), dtype=bool)
    mask_data[10:30, 10:30] = True
    mask = UserSegmentationMask.from_numpy(mask_data)

    result = draw_masks(img, [mask], color=(255, 0, 0), alpha=0.5)
    assert result is img  # Modified in place
    # Masked region should be blended
    assert result[20, 20, 0] > 100  # Red channel increased
    # Non-masked region should be unchanged
    assert result[5, 5].tolist() == [100, 100, 100]


def test_draw_rois_empty():
    """draw_rois with empty list should be no-op."""
    from sleap_io.rendering.overlays import draw_rois

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    result = draw_rois(img, [])
    assert not result.any()


def test_draw_rois_point():
    """draw_rois should draw a marker for Point geometry ROIs."""
    from shapely.geometry import Point

    from sleap_io.rendering.overlays import draw_rois

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = UserROI(geometry=Point(50, 50))
    result = draw_rois(img, [roi], color=(0, 255, 0), line_width=4)

    # The center pixel should be colored
    assert result is img
    assert result[50, 50].tolist() == [0, 255, 0]
    # A pixel far away should be untouched
    assert result[0, 0].tolist() == [0, 0, 0]


def test_draw_rois_linestring():
    """draw_rois should draw lines for LineString geometry ROIs."""
    from shapely.geometry import LineString

    from sleap_io.rendering.overlays import draw_rois

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = UserROI(geometry=LineString([(10, 10), (10, 50)]))
    result = draw_rois(img, [roi], color=(0, 0, 255), line_width=1)

    # Pixels along the vertical line should be colored
    assert result is img
    assert result[10, 10].tolist() == [0, 0, 255]
    assert result[30, 10].tolist() == [0, 0, 255]
    assert result[50, 10].tolist() == [0, 0, 255]
    # A pixel away from the line should be untouched
    assert result[30, 50].tolist() == [0, 0, 0]


def test_draw_rois_geometry_collection():
    """draw_rois should recurse into GeometryCollection components."""
    from shapely.geometry import GeometryCollection, LineString, Point

    from sleap_io.rendering.overlays import draw_rois

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    geom = GeometryCollection([Point(20, 20), LineString([(60, 60), (60, 80)])])
    roi = UserROI(geometry=geom)
    result = draw_rois(img, [roi], color=(255, 0, 0), line_width=2)

    # Point marker should be drawn
    assert result[20, 20].tolist() == [255, 0, 0]
    # Line pixels should be drawn
    assert result[70, 60].tolist() == [255, 0, 0]
    # A pixel away from both geometries should be untouched
    assert result[0, 0].tolist() == [0, 0, 0]


def test_draw_rois_multi_polygon():
    """draw_rois should draw all polygons in a MultiPolygon."""
    from sleap_io.rendering.overlays import draw_rois

    roi = UserROI.from_multi_polygon(
        [
            [(10, 10), (20, 10), (20, 20), (10, 20)],
            [(50, 50), (60, 50), (60, 60), (50, 60)],
        ]
    )
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = draw_rois(img, [roi], color=(0, 255, 0), line_width=1)

    assert result[10, 10].tolist() == [0, 255, 0]
    assert result[50, 50].tolist() == [0, 255, 0]
    assert result[35, 35].tolist() == [0, 0, 0]


def test_draw_rois_multi_point():
    """draw_rois should draw markers for all points in a MultiPoint."""
    from shapely.geometry import MultiPoint

    from sleap_io.rendering.overlays import draw_rois

    geom = MultiPoint([(20, 20), (60, 60)])
    roi = UserROI(geometry=geom)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = draw_rois(img, [roi], color=(0, 255, 0), line_width=2)

    assert result[20, 20].tolist() == [0, 255, 0]
    assert result[60, 60].tolist() == [0, 255, 0]
    assert result[0, 0].tolist() == [0, 0, 0]


def test_draw_rois_multi_linestring():
    """draw_rois should draw all lines in a MultiLineString."""
    from shapely.geometry import MultiLineString

    from sleap_io.rendering.overlays import draw_rois

    geom = MultiLineString([[(10, 10), (10, 30)], [(50, 50), (50, 70)]])
    roi = UserROI(geometry=geom)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = draw_rois(img, [roi], color=(0, 0, 255), line_width=1)

    assert result[20, 10].tolist() == [0, 0, 255]
    assert result[60, 50].tolist() == [0, 0, 255]
    assert result[0, 0].tolist() == [0, 0, 0]


def test_draw_rois_polygon_with_hole():
    """draw_rois with fill should respect polygon holes."""
    from shapely.geometry import Polygon

    from sleap_io.rendering.overlays import draw_rois

    exterior = [(10, 10), (60, 10), (60, 60), (10, 60)]
    hole = [(25, 25), (45, 25), (45, 45), (25, 45)]
    geom = Polygon(exterior, [hole])
    roi = UserROI(geometry=geom)
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    result = draw_rois(img, [roi], color=(255, 0, 0), fill_alpha=1.0)

    # Exterior filled region should be colored
    assert result[15, 15, 0] == 255
    # Hole interior should NOT be filled (even-odd rule)
    assert result[35, 35, 0] == 0


# ============================================================================
# draw_bboxes Tests
# ============================================================================


def test_draw_bboxes_basic():
    """draw_bboxes should draw axis-aligned bbox outlines on an image."""
    from sleap_io.rendering.overlays import draw_bboxes

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = UserBoundingBox.from_xyxy(10, 10, 50, 50)
    result = draw_bboxes(img, [bbox], color=(0, 255, 0), line_width=1)

    # The outline should have green pixels on the boundary
    assert result is img  # Modified in place
    assert result[10, 10].tolist() == [0, 255, 0]  # Top-left corner
    assert result[50, 50].tolist() == [0, 255, 0]  # Bottom-right corner
    assert result[80, 80].tolist() == [0, 0, 0]  # Outside the box
    # Interior should be empty (no fill)
    assert result[30, 30].tolist() == [0, 0, 0]


def test_draw_bboxes_rotated():
    """draw_bboxes should draw rotated bboxes using polylines."""
    import math

    from sleap_io.rendering.overlays import draw_bboxes

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = UserBoundingBox(x1=70, y1=80, x2=130, y2=120, angle=math.pi / 4)
    result = draw_bboxes(img, [bbox], color=(0, 0, 255), line_width=2)

    # Some pixels around the center should be drawn (rotated outline)
    assert result is img  # Modified in place
    # The image should have non-zero pixels somewhere (the rotated outline)
    assert result.any()
    # The center should still be empty (outline only)
    assert result[100, 100].tolist() == [0, 0, 0]


def test_draw_bboxes_fill():
    """draw_bboxes with fill_alpha > 0 should fill the bbox interior."""
    from sleap_io.rendering.overlays import draw_bboxes

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = UserBoundingBox.from_xyxy(10, 10, 50, 50)
    result = draw_bboxes(img, [bbox], color=(255, 0, 0), line_width=1, fill_alpha=1.0)

    # Interior should be filled with the color
    assert result[30, 30, 0] == 255  # Red channel filled
    assert result[30, 30, 1] == 0
    assert result[30, 30, 2] == 0


def test_draw_bboxes_predicted_score():
    """draw_bboxes should render score text for PredictedBoundingBox."""
    from sleap_io.model.bbox import PredictedBoundingBox
    from sleap_io.rendering.overlays import draw_bboxes

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = PredictedBoundingBox.from_xyxy(20, 20, 60, 60, score=0.95)
    result = draw_bboxes(img, [bbox], color=(0, 255, 0), line_width=1)

    # The image should have pixels for both the outline and score text
    assert result is img
    # Check that something was drawn above the box (score text region)
    # Text is drawn at (x1, y1 - 5) = (20, 15), so check region around there
    text_region = result[5:20, 15:60]
    assert text_region.any(), "Score text should be rendered near the top-left corner"


def test_draw_bboxes_empty():
    """draw_bboxes with empty list should return the unchanged image."""
    from sleap_io.rendering.overlays import draw_bboxes

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    original = img.copy()
    result = draw_bboxes(img, [])

    assert result is img
    np.testing.assert_array_equal(result, original)


def test_draw_masks_per_mask_colors():
    """draw_masks with per-mask colors should apply different colors to each mask."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    mask1_data = np.zeros((50, 50), dtype=bool)
    mask1_data[5:15, 5:15] = True
    mask2_data = np.zeros((50, 50), dtype=bool)
    mask2_data[30:40, 30:40] = True

    mask1 = UserSegmentationMask.from_numpy(mask1_data)
    mask2 = UserSegmentationMask.from_numpy(mask2_data)

    result = draw_masks(
        img,
        [mask1, mask2],
        colors=[(255, 0, 0), (0, 0, 255)],
        alpha=0.5,
    )
    assert result is img
    # Mask 1 region should have red bias
    assert result[10, 10, 0] > result[10, 10, 2]
    # Mask 2 region should have blue bias
    assert result[35, 35, 2] > result[35, 35, 0]
    # Non-masked region should be unchanged
    assert result[0, 0].tolist() == [100, 100, 100]


def test_draw_label_image_basic():
    """draw_label_image should blend colored overlays for each label ID."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[30:40, 30:40] = 2

    result = draw_label_image(img, labels, alpha=0.5, palette="distinct")
    assert result is img
    # Labeled regions should differ from the original gray
    assert not np.array_equal(result[15, 15], [128, 128, 128])
    assert not np.array_equal(result[35, 35], [128, 128, 128])
    # The two labeled regions should have different colors
    assert not np.array_equal(result[15, 15], result[35, 35])
    # Background should remain unchanged
    np.testing.assert_array_equal(result[0, 0], [128, 128, 128])


def test_draw_label_image_empty():
    """draw_label_image with all-zero labels should leave image unchanged."""
    img = np.ones((30, 30, 3), dtype=np.uint8) * 200
    labels = np.zeros((30, 30), dtype=np.int32)
    original = img.copy()

    result = draw_label_image(img, labels, alpha=0.5)
    assert result is img
    np.testing.assert_array_equal(result, original)


def test_draw_label_image_with_outline():
    """draw_label_image with outline=True should draw boundaries."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:30, 10:30] = 1

    result = draw_label_image(img, labels, alpha=0.3, outline=True, outline_width=1)
    assert result is img
    # Boundary pixel (edge of the labeled region) should differ from interior
    interior = result[20, 20].copy()
    edge = result[10, 10].copy()
    # Both should differ from background
    assert not np.array_equal(interior, [128, 128, 128])
    assert not np.array_equal(edge, [128, 128, 128])


def test_draw_label_image_with_outline_uniform_color():
    """draw_label_image with uniform outline_color should use that color."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:30, 10:30] = 1

    result = draw_label_image(
        img,
        labels,
        alpha=0.3,
        outline=True,
        outline_width=1,
        outline_color=(255, 255, 255),
    )
    # The boundary pixels should be white
    # Top edge of the labeled region: row 10, cols 10-29 are label boundary
    assert result[10, 15].tolist() == [255, 255, 255]


def test_draw_label_image_with_thick_outline():
    """draw_label_image with outline_width > 1 should dilate boundaries inward."""
    img = np.ones((60, 60, 3), dtype=np.uint8) * 128
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[15:45, 15:45] = 1

    result = draw_label_image(
        img,
        labels,
        alpha=0.3,
        outline=True,
        outline_width=3,
        outline_color=(0, 255, 0),
    )
    # With width=3, the boundary at row 15 should dilate inward to row 16
    assert result[16, 20].tolist() == [0, 255, 0]
    # Interior should not be affected by outline
    assert result[30, 30].tolist() != [0, 255, 0]


def test_draw_label_image_size_mismatch():
    """draw_label_image should clip when labels are larger than image."""
    img = np.ones((30, 30, 3), dtype=np.uint8) * 128
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[5:25, 5:25] = 1

    # Should not raise, should clip to image bounds
    result = draw_label_image(img, labels, alpha=0.5)
    assert result is img
    assert result.shape == (30, 30, 3)
    # Labeled region within image bounds should be modified
    assert not np.array_equal(result[10, 10], [128, 128, 128])


def test_draw_masks_half_resolution():
    """draw_masks with scale=(0.5, 0.5) should cover the full image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    # Half-resolution mask (50x50) covering full 100x100 image
    mask_data = np.ones((50, 50), dtype=bool)
    mask = UserSegmentationMask.from_numpy(mask_data, scale=(0.5, 0.5))

    result = draw_masks(img, [mask], color=(255, 0, 0), alpha=0.5)
    # The entire image should be blended (mask covers full frame)
    assert result[0, 0, 0] > 100  # Top-left
    assert result[99, 99, 0] > 100  # Bottom-right
    assert result[50, 50, 0] > 100  # Center


def test_draw_masks_with_offset():
    """draw_masks with offset should place mask at the correct position."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    mask_data = np.ones((20, 20), dtype=bool)
    mask = UserSegmentationMask.from_numpy(mask_data, offset=(30.0, 40.0))

    result = draw_masks(img, [mask], color=(255, 0, 0), alpha=0.5)
    # Mask should be at (x=30, y=40) to (x=50, y=60)
    assert result[45, 35, 0] > 100  # Inside mask region
    assert result[10, 10].tolist() == [100, 100, 100]  # Outside mask region
    assert result[70, 70].tolist() == [100, 100, 100]  # Also outside


def test_draw_masks_scale_and_offset():
    """draw_masks with both scale and offset should work correctly."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    # 10x10 mask at half res, offset to (20, 30) -> covers (20,30) to (40,50)
    mask_data = np.ones((10, 10), dtype=bool)
    mask = UserSegmentationMask.from_numpy(
        mask_data, scale=(0.5, 0.5), offset=(20.0, 30.0)
    )

    result = draw_masks(img, [mask], color=(255, 0, 0), alpha=0.5)
    assert result[35, 25, 0] > 100  # Inside
    assert result[0, 0].tolist() == [100, 100, 100]  # Outside


def test_draw_label_image_half_resolution():
    """draw_label_image with scale should upscale labels to cover full image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Half-res labels
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:40, 10:40] = 1

    result = draw_label_image(img, labels, alpha=0.5, scale=(0.5, 0.5))
    # Center should be modified (label covers most of the image)
    assert not np.array_equal(result[50, 50], [128, 128, 128])
    # Corner should be unchanged (no label there in image space)
    assert result[0, 0].tolist() == [128, 128, 128]


def test_draw_label_image_with_offset():
    """draw_label_image with offset should place labels at correct position."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:15, 5:15] = 1

    result = draw_label_image(img, labels, alpha=0.5, offset=(30.0, 40.0))
    # Inside offset region should be modified
    assert not np.array_equal(result[47, 37], [128, 128, 128])
    # Origin should be unchanged
    assert result[0, 0].tolist() == [128, 128, 128]


# ============================================================================
# render_image overlay integration tests
# ============================================================================


def test_render_image_overlay_label_image():
    """render_image with overlay= numpy label image should apply overlay."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[30:40, 30:40] = 2

    result = render_image(image=img, overlay=labels, overlay_alpha=0.4)
    # Background unchanged
    assert result[0, 0].tolist() == [128, 128, 128]
    # Labeled regions modified
    assert not np.array_equal(result[15, 15], [128, 128, 128])
    assert not np.array_equal(result[35, 35], [128, 128, 128])
    # Two labels get different colors
    assert not np.array_equal(result[15, 15], result[35, 35])


def test_render_image_overlay_masks():
    """render_image with overlay= list of SegmentationMask should work."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    mask_data = np.zeros((50, 50), dtype=bool)
    mask_data[10:30, 10:30] = True
    mask = UserSegmentationMask.from_numpy(mask_data)

    result = render_image(image=img, overlay=[mask], overlay_alpha=0.5)
    # Masked region should be modified
    assert not np.array_equal(result[20, 20], [128, 128, 128])
    # Background unchanged
    assert result[0, 0].tolist() == [128, 128, 128]


def test_render_image_overlay_rois():
    """render_image with overlay= list of ROI should work."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = UserROI(geometry=box(20, 20, 60, 60))

    result = render_image(image=img, overlay=[roi], overlay_alpha=0.3)
    # ROI region should have color applied
    assert result[40, 40].any()


def test_render_image_overlay_bboxes():
    """render_image with overlay= list of BoundingBox should work."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = UserBoundingBox(x1=30, y1=30, x2=70, y2=70)

    result = render_image(image=img, overlay=[bbox], overlay_alpha=0.3)
    # BBox region should have color applied
    assert result[50, 50].any()


def test_render_image_overlay_with_scale():
    """render_image with overlay and scale should scale the output."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[20:80, 20:80] = 1

    result = render_image(image=img, overlay=labels, overlay_alpha=0.4, scale=0.5)
    assert result.shape == (50, 50, 3)


def test_render_image_labeled_frame_source_with_label_image_overlay():
    """``source=lf`` + LabelImage overlay works when lf has no instances.

    Regression: previously raised ``ValueError("LabeledFrame has no instances
    with skeleton")``, forcing callers to pass ``source=None, image=lf.image``
    for segmentation-only frames.
    """
    from sleap_io.model.label_image import LabelImage, UserLabelImage
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.video import Video

    data = np.zeros((60, 60), dtype=np.int32)
    data[10:25, 10:25] = 1
    data[35:50, 35:50] = 2
    li = UserLabelImage(
        data=data,
        objects={1: LabelImage.Info(category="a"), 2: LabelImage.Info(category="b")},
    )
    lf = LabeledFrame(video=Video(filename="dummy.mp4"), frame_idx=0)
    lf.label_images.append(li)

    # Pass an explicit image so we control the canvas size.
    base = np.zeros((60, 60, 3), dtype=np.uint8)
    result = render_image(source=lf, image=base, overlay=li, overlay_alpha=0.45)
    assert result.shape == (60, 60, 3)
    # Labeled regions are visible on the black background.
    assert result[15, 15].any()
    assert result[40, 40].any()
    # The two labels take different colors.
    assert not np.array_equal(result[15, 15], result[40, 40])


def test_render_image_labeled_frame_source_with_mask_overlay():
    """``source=lf`` + SegmentationMask overlay works with no instances."""
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.video import Video

    binary = np.zeros((50, 50), dtype=bool)
    binary[10:30, 10:30] = True
    mask = UserSegmentationMask.from_numpy(binary)
    lf = LabeledFrame(video=Video(filename="dummy.mp4"), frame_idx=0)
    lf.masks.append(mask)

    base = np.zeros((50, 50, 3), dtype=np.uint8)
    result = render_image(source=lf, image=base, overlay=[mask], overlay_alpha=0.5)
    assert result.shape == (50, 50, 3)
    assert result[20, 20].any()


def test_render_image_labeled_frame_source_with_roi_overlay():
    """``source=lf`` + ROI overlay works with no instances."""
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.video import Video

    roi = UserROI(geometry=box(20, 20, 60, 60))
    lf = LabeledFrame(video=Video(filename="dummy.mp4"), frame_idx=0)
    lf.rois.append(roi)

    result = render_image(
        source=lf, overlay=[roi], overlay_alpha=0.3, background="black"
    )
    # Rendered at the default size (64x64) since no video and no instances.
    assert result.ndim == 3 and result.shape[-1] == 3


def test_render_image_labeled_frame_source_with_bbox_overlay():
    """``source=lf`` + BoundingBox overlay works with no instances."""
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.video import Video

    bbox = UserBoundingBox(x1=30, y1=30, x2=70, y2=70)
    lf = LabeledFrame(video=Video(filename="dummy.mp4"), frame_idx=0)
    lf.bboxes.append(bbox)

    result = render_image(
        source=lf, overlay=[bbox], overlay_alpha=0.3, background="black"
    )
    assert result.ndim == 3 and result.shape[-1] == 3


def test_render_image_labeled_frame_with_instances_still_uses_skeleton(
    labels_predictions,
):
    """LabeledFrame with instances still goes through the skeleton-aware path.

    Verifies the fix did not break the classic pose-rendering code path.
    """
    lf = labels_predictions.labeled_frames[0]
    assert len(lf.instances) > 0  # sanity
    result = render_image(source=lf, background="black")
    assert result.ndim == 3 and result.shape[-1] == 3


def test_render_image_source_none_requires_image():
    """render_image with source=None and no image should raise."""
    with pytest.raises(ValueError, match="image parameter required"):
        render_image(source=None)


# ============================================================================
# draw_rois / draw_bboxes per-item colors tests
# ============================================================================


def test_draw_rois_per_roi_colors():
    """draw_rois with per-ROI colors should apply different colors to each ROI."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi1 = UserROI.from_bbox(5, 5, 20, 20)
    roi2 = UserROI.from_bbox(60, 60, 20, 20)

    result = draw_rois(
        img,
        [roi1, roi2],
        colors=[(255, 0, 0), (0, 0, 255)],
        fill_alpha=1.0,
    )

    assert result is img
    # First ROI interior should be red
    assert result[15, 15, 0] == 255
    assert result[15, 15, 2] == 0
    # Second ROI interior should be blue
    assert result[70, 70, 0] == 0
    assert result[70, 70, 2] == 255
    # Outside both ROIs should be untouched
    assert result[50, 50].tolist() == [0, 0, 0]


def test_draw_bboxes_per_bbox_colors():
    """draw_bboxes with per-bbox colors should apply different colors."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox1 = UserBoundingBox.from_xyxy(5, 5, 25, 25)
    bbox2 = UserBoundingBox.from_xyxy(60, 60, 80, 80)

    result = draw_bboxes(
        img,
        [bbox1, bbox2],
        colors=[(255, 0, 0), (0, 0, 255)],
        fill_alpha=1.0,
    )

    assert result is img
    # First bbox interior should be red
    assert result[15, 15, 0] == 255
    assert result[15, 15, 2] == 0
    # Second bbox interior should be blue
    assert result[70, 70, 0] == 0
    assert result[70, 70, 2] == 255
    # Outside both bboxes should be untouched
    assert result[45, 45].tolist() == [0, 0, 0]


def test_draw_bboxes_per_bbox_colors_predicted_score():
    """draw_bboxes per-bbox colors with PredictedBoundingBox renders score text."""
    from sleap_io.model.bbox import PredictedBoundingBox

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = PredictedBoundingBox.from_xyxy(20, 20, 60, 60, score=0.85)

    result = draw_bboxes(
        img,
        [bbox],
        colors=[(0, 255, 0)],
        fill_alpha=0.5,
    )

    assert result is img
    # Score text should be rendered near top-left of bbox
    text_region = result[5:20, 15:60]
    assert text_region.any(), "Score text should be rendered"


def test_render_image_grayscale_with_overlay():
    """render_image with a grayscale image and overlay should convert to RGB."""
    gray_img = np.ones((64, 64), dtype=np.uint8) * 128
    overlay = np.zeros((64, 64), dtype=np.int32)
    overlay[10:30, 10:30] = 1

    result = render_image(
        source=None,
        image=gray_img,
        overlay=overlay,
        overlay_alpha=0.5,
    )

    # Output should be RGB
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_render_video_grayscale_with_2d_overlay():
    """render_video with grayscale frames and 2D static overlay."""
    labels_obj = _make_synthetic_labels(n_frames=1, h=64, w=64)
    overlay = np.zeros((64, 64), dtype=np.int32)
    overlay[10:30, 10:30] = 1

    frames = render_video(
        labels_obj,
        save_path=None,
        background="black",
        overlay=overlay,
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 1
    assert frames[0].ndim == 3


# ============================================================================
# render_video overlay tests
# ============================================================================


def _make_synthetic_labels(n_frames=3, h=64, w=64):
    """Create a minimal synthetic Labels for overlay tests.

    Returns Labels with n_frames of (h, w) frames rendered on a solid black
    background (no real video needed).
    """
    skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])
    video = sio.Video(filename="dummy.mp4")
    pts = np.array([[w // 4, h // 4], [3 * w // 4, 3 * h // 4]], dtype=np.float32)

    labeled_frames = []
    for i in range(n_frames):
        lf = sio.LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[sio.Instance.from_numpy(pts, skeleton=skeleton)],
        )
        labeled_frames.append(lf)

    return sio.Labels(
        labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton]
    )


def test_render_video_overlay_3d_array():
    """render_video with 3D (T,H,W) overlay should apply per-frame labels."""
    labels = _make_synthetic_labels(n_frames=3, h=64, w=64)
    overlay = np.zeros((3, 64, 64), dtype=np.int32)
    overlay[0, 10:30, 10:30] = 1
    overlay[1, 30:50, 30:50] = 2
    # Frame 2 has no overlay (all zeros)

    frames = render_video(
        labels,
        save_path=None,
        background="black",
        overlay=overlay,
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 3
    # Frame 0 should have colored overlay in region [10:30, 10:30]
    assert not np.array_equal(frames[0][20, 20], frames[0][0, 0])
    # Frame 2 should have no overlay (uniform background)
    # (poses are still drawn, so just check it didn't crash)
    assert frames[2].shape == frames[0].shape


def test_render_video_overlay_2d_static():
    """render_video with 2D (H,W) overlay should apply same overlay to all frames."""
    labels = _make_synthetic_labels(n_frames=2, h=64, w=64)
    overlay = np.zeros((64, 64), dtype=np.int32)
    overlay[10:30, 10:30] = 1

    frames = render_video(
        labels,
        save_path=None,
        background="black",
        overlay=overlay,
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 2
    # Both frames should have the overlay in the same region
    for frame in frames:
        assert not np.array_equal(frame[20, 20], frame[0, 0])


def test_render_video_overlay_callable():
    """render_video with callable overlay should call it per-frame."""
    labels = _make_synthetic_labels(n_frames=2, h=64, w=64)

    called_with = []

    def overlay_fn(fidx):
        called_with.append(fidx)
        arr = np.zeros((64, 64), dtype=np.int32)
        arr[10:30, 10:30] = fidx + 1
        return arr

    frames = render_video(
        labels,
        save_path=None,
        background="black",
        overlay=overlay_fn,
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 2
    assert 0 in called_with
    assert 1 in called_with


def test_render_video_overlay_list_by_frame_idx():
    """render_video with list overlay applies masks to all frames (no frame_idx)."""
    labels = _make_synthetic_labels(n_frames=2, h=64, w=64)

    mask_data = np.zeros((64, 64), dtype=bool)
    mask_data[10:30, 10:30] = True
    mask = UserSegmentationMask.from_numpy(mask_data)

    frames = render_video(
        labels,
        save_path=None,
        background=(128, 128, 128),
        overlay=[mask],
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 2
    # Mask without frame_idx is applied to all frames
    assert not np.array_equal(frames[0][20, 20], frames[0][0, 0])
    assert not np.array_equal(frames[1][20, 20], frames[1][0, 0])


def test_render_video_overlay_static_objects():
    """render_video with frame_idx=None objects should render them on all frames."""
    labels = _make_synthetic_labels(n_frames=3, h=64, w=64)

    mask_data = np.zeros((64, 64), dtype=bool)
    mask_data[10:30, 10:30] = True
    mask = UserSegmentationMask.from_numpy(mask_data)
    # frame_idx defaults to None -> should appear on every frame

    frames = render_video(
        labels,
        save_path=None,
        background=(128, 128, 128),
        overlay=[mask],
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 3
    # Static mask (frame_idx=None) should be rendered on ALL frames
    for i, frame in enumerate(frames):
        assert not np.array_equal(frame[20, 20], [128, 128, 128]), (
            f"Frame {i}: static mask was not rendered"
        )


def test_render_video_overlay_out_of_bounds():
    """render_video with 3D overlay shorter than video should not crash."""
    labels = _make_synthetic_labels(n_frames=3, h=64, w=64)
    # Overlay only has 1 frame, but video has 3
    overlay = np.zeros((1, 64, 64), dtype=np.int32)
    overlay[0, 10:30, 10:30] = 1

    frames = render_video(
        labels,
        save_path=None,
        background="black",
        overlay=overlay,
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 3


# ============================================================================
# _apply_overlay TypeError for unknown list element types
# ============================================================================


def test_apply_overlay_unknown_list_type():
    """_apply_overlay should raise TypeError for unsupported list element types."""
    from sleap_io.rendering.core import _apply_overlay

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="Unsupported overlay element type"):
        _apply_overlay(img, ["not_a_mask"])


# ============================================================================
# crop + overlay alignment tests
# ============================================================================


def test_render_image_crop_with_overlay():
    """render_image with crop + overlay should align overlay to cropped region."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    labels = np.zeros((100, 100), dtype=np.int32)
    # Place label in bottom-right quadrant only
    labels[60:80, 60:80] = 1

    # Crop to bottom-right quadrant (50,50)-(100,100)
    result = render_image(
        image=img,
        overlay=labels,
        overlay_alpha=0.5,
        crop=(50, 50, 100, 100),
    )

    assert result.shape == (50, 50, 3)
    # The label region (60:80, 60:80) maps to (10:30, 10:30) in cropped coords
    # Interior of labeled region should be colored
    assert not np.array_equal(result[20, 20], [128, 128, 128])
    # Top-left of cropped image (originally 50,50) has no label -> unchanged
    assert result[0, 0].tolist() == [128, 128, 128]


def test_render_image_crop_with_overlay_no_overlap():
    """Overlay outside crop region should not affect the cropped image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    labels = np.zeros((100, 100), dtype=np.int32)
    # Place label in top-left only
    labels[0:20, 0:20] = 1

    # Crop to bottom-right (50,50)-(100,100) — no overlap with label
    result = render_image(
        image=img,
        overlay=labels,
        overlay_alpha=0.5,
        crop=(50, 50, 100, 100),
    )

    # Entire cropped region should be unchanged
    assert result.shape == (50, 50, 3)
    np.testing.assert_array_equal(result, np.ones((50, 50, 3), dtype=np.uint8) * 128)


def test_render_video_crop_with_overlay():
    """render_video with crop + overlay should align overlay to cropped region."""
    labels_obj = _make_synthetic_labels(n_frames=1, h=100, w=100)
    overlay = np.zeros((100, 100), dtype=np.int32)
    overlay[60:80, 60:80] = 1

    frames = render_video(
        labels_obj,
        save_path=None,
        background="black",
        overlay=overlay,
        overlay_alpha=0.5,
        crop=(50, 50, 100, 100),
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 1
    # Cropped to 50x50, the label at (60:80,60:80) maps to (10:30,10:30)
    assert frames[0].shape[:2] == (50, 50)


def test_apply_overlay_label_image():
    """_apply_overlay with a LabelImage input should modify labeled regions."""
    from sleap_io.rendering.core import _apply_overlay

    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    label_data = np.zeros((50, 50), dtype=np.int32)
    label_data[10:30, 10:30] = 1

    label_img = UserLabelImage(data=label_data)

    result = _apply_overlay(img, label_img, alpha=0.5)

    # The labeled region should have been modified by the overlay
    assert result.shape == (50, 50, 3)
    # Pixels in the labeled region should differ from 128
    labeled_region = result[10:30, 10:30]
    assert not np.all(labeled_region == 128)
    # Pixels outside the labeled region should be unchanged
    assert np.all(result[0:10, 0:10] == 128)


def test_render_image_overlay_label_image_object():
    """render_image with overlay=LabelImage object should work."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    label_data = np.zeros((50, 50), dtype=np.int32)
    label_data[5:25, 5:25] = 1
    label_data[25:45, 25:45] = 2

    label_img = UserLabelImage(data=label_data)

    result = render_image(image=img, overlay=label_img, overlay_alpha=0.5)

    assert result.shape == (50, 50, 3)
    # Both labeled regions should be colored (different from 200)
    assert not np.all(result[5:25, 5:25] == 200)
    assert not np.all(result[25:45, 25:45] == 200)
    # Background should be unchanged
    assert np.all(result[0:5, 0:5] == 200)


def test_render_video_overlay_label_images():
    """render_video with overlay=list[LabelImage] should apply per-frame overlays."""
    labels_obj = _make_synthetic_labels(n_frames=2, h=50, w=50)

    li0 = UserLabelImage(
        data=np.array(np.pad(np.ones((20, 20), dtype=np.int32), ((10, 20), (10, 20)))),
    )
    li1 = UserLabelImage(
        data=np.array(np.pad(np.full((15, 15), 2, dtype=np.int32), ((30, 5), (30, 5)))),
    )

    frames = render_video(
        labels_obj,
        save_path=None,
        background="black",
        overlay=[li0, li1],
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 2
    # Frame 0: region (10:30, 10:30) should be colored
    assert not np.all(frames[0][10:30, 10:30] == 0)
    # Frame 0: top-left corner should remain black
    assert np.all(frames[0][0:5, 0:5, :] == 0)
    # Frame 1: region (30:45, 30:45) should be colored
    assert not np.all(frames[1][30:45, 30:45] == 0)


def test_render_video_spatial_only_no_poses():
    """render_video with only label_images should work."""
    video = sio.Video(filename="dummy.mp4")
    li0 = UserLabelImage(
        data=np.array(np.pad(np.ones((20, 20), dtype=np.int32), ((10, 34), (10, 34)))),
    )
    li1 = UserLabelImage(
        data=np.array(np.pad(np.full((15, 15), 2, dtype=np.int32), ((40, 9), (40, 9)))),
    )
    lf0 = sio.LabeledFrame(video=video, frame_idx=0)
    lf0.label_images.append(li0)
    lf1 = sio.LabeledFrame(video=video, frame_idx=1)
    lf1.label_images.append(li1)
    labels = sio.Labels(labeled_frames=[lf0, lf1])

    frames = render_video(
        labels,
        save_path=None,
        background="black",
        overlay_alpha=0.5,
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 2
    assert frames[0].shape == (64, 64, 3)
    # Frame 0: overlay region should be colored
    assert not np.all(frames[0][10:30, 10:30] == 0)
    # Frame 0: outside overlay should remain black
    assert np.all(frames[0][0:5, 0:5, :] == 0)
    # Frame 1: overlay region should be colored
    assert not np.all(frames[1][40:55, 40:55] == 0)


def test_render_video_crop_with_user_label_image_overlay():
    """render_video with crop + UserLabelImage overlay should crop the label image."""
    labels_obj = _make_synthetic_labels(n_frames=1, h=50, w=50)

    label_data = np.zeros((50, 50), dtype=np.int32)
    label_data[20:40, 20:40] = 1
    li = UserLabelImage(data=label_data)

    frames = render_video(
        labels_obj,
        save_path=None,
        background="black",
        overlay=[li],
        overlay_alpha=0.5,
        crop=(10, 10, 40, 40),
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 1
    # Cropped to (10,10)-(40,40) -> 30x30
    assert frames[0].shape == (30, 30, 3)
    # Label region (20:40, 20:40) maps to (10:30, 10:30) in cropped coords
    assert not np.all(frames[0][10:30, 10:30] == 0)


def test_render_video_crop_with_predicted_label_image_overlay():
    """render_video with crop + PredictedLabelImage overlay should preserve score."""
    labels_obj = _make_synthetic_labels(n_frames=1, h=50, w=50)

    label_data = np.zeros((50, 50), dtype=np.int32)
    label_data[20:40, 20:40] = 1
    li = PredictedLabelImage(
        data=label_data,
        score=0.9,
        score_map=np.ones((50, 50), dtype=np.float32) * 0.8,
    )

    frames = render_video(
        labels_obj,
        save_path=None,
        background="black",
        overlay=[li],
        overlay_alpha=0.5,
        crop=(10, 10, 40, 40),
        fps=30,
        show_progress=False,
    )

    assert len(frames) == 1
    # Cropped to (10,10)-(40,40) -> 30x30
    assert frames[0].shape == (30, 30, 3)
    # Label region (20:40, 20:40) maps to (10:30, 10:30) in cropped coords
    assert not np.all(frames[0][10:30, 10:30] == 0)


# --- Centroid rendering tests ---


def test_draw_centroids_basic():
    """draw_centroids draws filled circles on the image."""
    from sleap_io.model.centroid import UserCentroid
    from sleap_io.rendering.overlays import draw_centroids

    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    c1 = UserCentroid(x=30.0, y=40.0)
    c2 = UserCentroid(x=70.0, y=60.0)
    result = draw_centroids(img, [c1, c2], color=(255, 0, 0), marker_size=8.0)
    assert result.shape == (100, 100, 3)
    red = (result[:, :, 0] > 200) & (result[:, :, 1] < 50) & (result[:, :, 2] < 50)
    assert red.sum() > 0


def test_draw_centroids_per_color():
    """draw_centroids with per-centroid colors."""
    from sleap_io.model.centroid import UserCentroid
    from sleap_io.rendering.overlays import draw_centroids

    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    c1 = UserCentroid(x=25.0, y=25.0)
    c2 = UserCentroid(x=75.0, y=75.0)
    result = draw_centroids(
        img, [c1, c2], colors=[(255, 0, 0), (0, 0, 255)], marker_size=8.0
    )
    red = (result[:, :, 0] > 200) & (result[:, :, 1] < 50)
    blue = (result[:, :, 2] > 200) & (result[:, :, 0] < 50)
    assert red.sum() > 0
    assert blue.sum() > 0


def test_draw_centroids_empty():
    """draw_centroids with empty list is a no-op."""
    from sleap_io.rendering.overlays import draw_centroids

    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    result = draw_centroids(img, [], marker_size=5.0)
    np.testing.assert_array_equal(result, img)


def test_draw_centroids_grayscale():
    """draw_centroids handles grayscale input."""
    from sleap_io.model.centroid import UserCentroid
    from sleap_io.rendering.overlays import draw_centroids

    img = np.full((50, 50), 128, dtype=np.uint8)
    c = UserCentroid(x=25.0, y=25.0)
    result = draw_centroids(img, [c], color=(255, 0, 0), marker_size=5.0)
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_draw_centroids_with_offset():
    """draw_centroids applies coordinate offset."""
    from sleap_io.model.centroid import UserCentroid
    from sleap_io.rendering.overlays import draw_centroids

    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    c = UserCentroid(x=100.0, y=100.0)
    result = draw_centroids(
        img, [c], color=(0, 255, 0), marker_size=5.0, offset=(90.0, 90.0)
    )
    green = (result[:, :, 1] > 200) & (result[:, :, 0] < 50) & (result[:, :, 2] < 50)
    assert green.sum() > 0


def test_render_image_with_centroids():
    """render_image draws centroids from Labels."""
    from sleap_io.model.centroid import UserCentroid
    from sleap_io.model.instance import Track

    track = Track(name="t1")
    c = UserCentroid(x=25.0, y=25.0, track=track)
    video = sio.Video(filename="dummy.mp4", open_backend=False)
    lf = sio.LabeledFrame(video=video, frame_idx=0)
    lf.centroids.append(c)
    labels = sio.Labels(
        labeled_frames=[lf],
        skeletons=[sio.Skeleton(["A"])],
        tracks=[track],
    )
    rendered = render_image(labels, frame_idx=0, video=video, background="white")
    assert rendered.ndim == 3
    non_white = np.any(rendered != 255, axis=-1)
    assert non_white.sum() > 0


def test_render_video_centroids_only():
    """render_video works with centroid-only Labels (no labeled frames)."""
    from sleap_io.model.centroid import UserCentroid
    from sleap_io.model.instance import Track

    video = sio.Video(filename="test.mp4", open_backend=False)
    video.backend_metadata["shape"] = (10, 64, 64, 3)
    track = Track(name="t1")
    c0 = UserCentroid(x=10.0, y=10.0, track=track)
    c1 = UserCentroid(x=20.0, y=20.0, track=track)
    c2 = UserCentroid(x=30.0, y=30.0, track=track)
    lf0 = sio.LabeledFrame(video=video, frame_idx=0)
    lf0.centroids.append(c0)
    lf1 = sio.LabeledFrame(video=video, frame_idx=1)
    lf1.centroids.append(c1)
    lf2 = sio.LabeledFrame(video=video, frame_idx=2)
    lf2.centroids.append(c2)
    labels = sio.Labels(
        labeled_frames=[lf0, lf1, lf2],
        skeletons=[sio.Skeleton(["A"])],
        tracks=[track],
    )
    frames = render_video(
        labels, background="black", show_progress=False, frame_inds=[0, 1, 2]
    )
    assert len(frames) == 3
    # Centroid should be drawn on each frame (non-zero pixels).
    for frame in frames:
        assert np.any(frame != 0)


def test_render_image_centroids_show_false():
    """render_image with show_centroids=False skips centroid markers."""
    from sleap_io.model.centroid import UserCentroid
    from sleap_io.model.instance import Track

    # Use a Labels with a labeled frame AND a centroid so the off case still
    # renders (it just skips centroid markers).
    skel = sio.Skeleton(["A"])
    track = Track(name="t1")
    inst = sio.Instance.from_numpy(np.array([[50.0, 50.0]]), skeleton=skel)
    lf = sio.LabeledFrame(
        video=sio.Video(filename="test.mp4", open_backend=False),
        frame_idx=0,
        instances=[inst],
    )
    c = UserCentroid(x=10.0, y=10.0, track=track)
    lf.centroids.append(c)
    labels = sio.Labels(
        labeled_frames=[lf],
        skeletons=[skel],
        tracks=[track],
    )
    rendered_on = render_image(
        labels, lf_ind=0, background="white", show_centroids=True
    )
    rendered_off = render_image(
        labels, lf_ind=0, background="white", show_centroids=False
    )
    on_count = np.any(rendered_on != 255, axis=-1).sum()
    off_count = np.any(rendered_off != 255, axis=-1).sum()
    assert on_count > off_count


def test_render_video_centroids_no_track():
    """render_video handles centroids without tracks."""
    from sleap_io.model.centroid import PredictedCentroid

    video = sio.Video(filename="test.mp4", open_backend=False)
    video.backend_metadata["shape"] = (10, 64, 64, 3)
    c = PredictedCentroid(x=25.0, y=25.0, score=0.9)
    lf = sio.LabeledFrame(video=video, frame_idx=0)
    lf.centroids.append(c)
    labels = sio.Labels(
        labeled_frames=[lf],
        skeletons=[sio.Skeleton(["A"])],
    )
    frames = render_video(
        labels, background="black", show_progress=False, frame_inds=[0]
    )
    assert len(frames) == 1


# ============================================================================
# Motion trail tests
# ============================================================================


def _make_trail_labels(n_frames=12, n_video_frames=None):
    """Build synthetic tracked Labels with two tracks moving on straight lines.

    Args:
        n_frames: Number of labeled frames to create (frame indices 0..n-1).
        n_video_frames: If set, attaches video shape metadata with this many
            frames (enables ``include_unlabeled`` rendering).

    Returns:
        A `Labels` object with two single-edge tracks moving rightward.
    """
    skel = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])
    video = sio.Video(filename="trail_test.mp4", open_backend=False)
    if n_video_frames is not None:
        video.backend_metadata["shape"] = (n_video_frames, 200, 200, 3)
    tracks = [sio.Track("t0"), sio.Track("t1")]
    lfs = []
    for fi in range(n_frames):
        insts = []
        for ti, tr in enumerate(tracks):
            x = 20.0 + fi * 10.0 + ti * 5.0
            y = 50.0 + ti * 60.0
            insts.append(
                sio.Instance(
                    points={"a": [x, y], "b": [x + 8, y + 8]},
                    skeleton=skel,
                    track=tr,
                )
            )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=insts))
    return sio.Labels(
        labeled_frames=lfs, videos=[video], skeletons=[skel], tracks=tracks
    )


class TestDrawTrails:
    """Tests for the draw_trails overlay function."""

    def test_draw_trails_basic(self):
        """draw_trails draws a polyline and returns the modified image."""
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        trail = np.array([[float(x), 25.0] for x in range(10, 91, 10)])
        result = draw_trails(image, [trail], color=(255, 0, 0), line_width=3.0)
        assert result.shape == (50, 100, 3)
        assert result.dtype == np.uint8
        assert result.sum() > 0

    def test_draw_trails_empty(self):
        """draw_trails with no trails returns the image unchanged."""
        image = np.full((20, 20, 3), 7, dtype=np.uint8)
        result = draw_trails(image, [])
        assert np.array_equal(result, np.full((20, 20, 3), 7, dtype=np.uint8))

    def test_draw_trails_nan_gap(self):
        """A NaN point breaks the polyline, leaving a gap."""
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        trail = np.array(
            [
                [10.0, 25.0],
                [30.0, 25.0],
                [np.nan, np.nan],
                [70.0, 25.0],
                [90.0, 25.0],
            ]
        )
        result = draw_trails(image, [trail], color=(255, 255, 255), line_width=3.0)
        # Segment within [10, 30] is drawn.
        assert result[25, 20].sum() > 0
        # Segment spanning the NaN gap [30, 70] is not drawn.
        assert result[25, 50].sum() == 0
        # Segment within [70, 90] is drawn.
        assert result[25, 80].sum() > 0

    def test_draw_trails_alpha_fade(self):
        """alpha_fade makes the oldest segment fainter than the newest."""
        trail = np.array([[float(x), 25.0] for x in range(10, 91, 10)])

        img_fade = np.zeros((50, 100, 3), dtype=np.uint8)
        draw_trails(
            img_fade, [trail], color=(255, 255, 255), line_width=3.0, alpha_fade=True
        )
        img_solid = np.zeros((50, 100, 3), dtype=np.uint8)
        draw_trails(
            img_solid,
            [trail],
            color=(255, 255, 255),
            line_width=3.0,
            alpha_fade=False,
        )

        left_fade = img_fade[22:28, 12:20].mean()
        right_fade = img_fade[22:28, 80:88].mean()
        # Oldest (left) segment is fainter than newest (right) under fade.
        assert left_fade < right_fade
        # Without fade, the oldest segment is brighter than the faded one.
        left_solid = img_solid[22:28, 12:20].mean()
        assert left_solid > left_fade

    def test_draw_trails_per_trail_colors(self):
        """Per-trail colors are applied independently."""
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        trail0 = np.array([[10.0, 20.0], [90.0, 20.0]])
        trail1 = np.array([[10.0, 60.0], [90.0, 60.0]])
        result = draw_trails(
            image,
            [trail0, trail1],
            colors=[(255, 0, 0), (0, 0, 255)],
            line_width=3.0,
            alpha_fade=False,
        )
        # First trail is red.
        assert result[20, 50, 0] > 100 and result[20, 50, 2] < 50
        # Second trail is blue.
        assert result[60, 50, 2] > 100 and result[60, 50, 0] < 50

    def test_draw_trails_global_alpha(self):
        """A lower global alpha blends the trail more faintly into the image."""
        trail = np.array([[10.0, 25.0], [90.0, 25.0]])

        img_opaque = np.zeros((50, 100, 3), dtype=np.uint8)
        draw_trails(
            img_opaque,
            [trail],
            color=(255, 255, 255),
            line_width=3.0,
            alpha_fade=False,
            alpha=1.0,
        )
        img_faint = np.zeros((50, 100, 3), dtype=np.uint8)
        draw_trails(
            img_faint,
            [trail],
            color=(255, 255, 255),
            line_width=3.0,
            alpha_fade=False,
            alpha=0.3,
        )
        # The faint trail is dimmer than the opaque one.
        assert img_faint[25, 50].mean() < img_opaque[25, 50].mean()
        assert img_faint[25, 50].sum() > 0

    def test_draw_trails_offset(self):
        """The offset argument shifts trail coordinates (for cropped images)."""
        trail = np.array([[50.0, 25.0], [70.0, 25.0]])

        img_no_offset = np.zeros((50, 100, 3), dtype=np.uint8)
        draw_trails(img_no_offset, [trail], color=(255, 255, 255), line_width=3.0)
        img_offset = np.zeros((50, 100, 3), dtype=np.uint8)
        draw_trails(
            img_offset,
            [trail],
            color=(255, 255, 255),
            line_width=3.0,
            offset=(50.0, 0.0),
        )
        # Without offset the trail is around x=60.
        assert img_no_offset[25, 60].sum() > 0
        # With offset=(50, 0) it shifts to around x=10.
        assert img_offset[25, 10].sum() > 0
        assert img_offset[25, 60].sum() == 0

    def test_draw_trails_grayscale_input(self):
        """A grayscale image is promoted to RGB."""
        image = np.zeros((40, 60), dtype=np.uint8)
        trail = np.array([[10.0, 20.0], [50.0, 20.0]])
        result = draw_trails(image, [trail], color=(255, 0, 0), line_width=3.0)
        assert result.ndim == 3
        assert result.shape == (40, 60, 3)
        assert result.sum() > 0

    def test_draw_trails_single_point(self):
        """A single-point trail has no segment and draws nothing."""
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        trail = np.array([[30.0, 20.0]])
        result = draw_trails(image, [trail], color=(255, 255, 255))
        assert result.sum() == 0


class TestTrailHelpers:
    """Tests for _resolve_trail_node and _compute_trails."""

    def test_resolve_trail_node_centroid(self):
        """The string 'centroid' resolves to a single centroid target."""
        from sleap_io.rendering.core import _resolve_trail_node

        skel = _make_trail_labels(n_frames=1).skeletons[0]
        assert _resolve_trail_node("centroid", skel) == [None]
        # Case-insensitive.
        assert _resolve_trail_node("Centroid", skel) == [None]

    def test_resolve_trail_node_name(self):
        """A node name resolves to that node's index."""
        from sleap_io.rendering.core import _resolve_trail_node

        skel = _make_trail_labels(n_frames=1).skeletons[0]
        assert _resolve_trail_node("a", skel) == [0]
        assert _resolve_trail_node("b", skel) == [1]

    def test_resolve_trail_node_list(self):
        """A list of node names resolves to one target per node."""
        from sleap_io.rendering.core import _resolve_trail_node

        skel = _make_trail_labels(n_frames=1).skeletons[0]
        assert _resolve_trail_node(["a", "b"], skel) == [0, 1]

    def test_resolve_trail_node_unknown(self):
        """An unknown node name raises ValueError."""
        from sleap_io.rendering.core import _resolve_trail_node

        skel = _make_trail_labels(n_frames=1).skeletons[0]
        with pytest.raises(ValueError, match="Unknown trail_node"):
            _resolve_trail_node("nonexistent", skel)

    def test_compute_trails_basic(self):
        """_compute_trails produces one polyline per track."""
        from sleap_io.rendering.core import _compute_trails

        labels = _make_trail_labels(n_frames=8)
        frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        track_idx_map = {id(t): i for i, t in enumerate(labels.tracks)}
        palette = [(255, 0, 0), (0, 255, 0)]
        trails, colors = _compute_trails(
            fidx=7,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=4,
            trail_targets=[None],
            track_idx_map=track_idx_map,
            palette_colors=palette,
            has_tracks=True,
        )
        assert len(trails) == 2  # One per track.
        assert all(t.shape == (5, 2) for t in trails)  # trail_length + 1.
        assert len(colors) == 2
        assert set(colors) == {(255, 0, 0), (0, 255, 0)}

    def test_compute_trails_node_list(self):
        """_compute_trails produces one polyline per (track, node) pair."""
        from sleap_io.rendering.core import _compute_trails

        labels = _make_trail_labels(n_frames=8)
        frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        track_idx_map = {id(t): i for i, t in enumerate(labels.tracks)}
        trails, _ = _compute_trails(
            fidx=7,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=4,
            trail_targets=[0, 1],
            track_idx_map=track_idx_map,
            palette_colors=[(255, 0, 0), (0, 255, 0)],
            has_tracks=True,
        )
        # 2 tracks x 2 nodes.
        assert len(trails) == 4

    def test_compute_trails_missing_frame_nan(self):
        """Missing frames leave NaN rows in the trail."""
        from sleap_io.rendering.core import _compute_trails

        labels = _make_trail_labels(n_frames=8)
        frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        del frame_idx_to_lf[3]
        track_idx_map = {id(t): i for i, t in enumerate(labels.tracks)}
        trails, _ = _compute_trails(
            fidx=5,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=5,
            trail_targets=[None],
            track_idx_map=track_idx_map,
            palette_colors=[(255, 0, 0), (0, 255, 0)],
            has_tracks=True,
        )
        # Frame range is [0, 5]; frame 3 is at row index 3.
        assert all(np.isnan(t[3]).all() for t in trails)
        assert all(np.isfinite(t[5]).all() for t in trails)

    def test_compute_trails_untracked(self):
        """Untracked data keys trails by instance index."""
        from sleap_io.rendering.core import _compute_trails

        labels = _make_trail_labels(n_frames=6)
        frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        trails, colors = _compute_trails(
            fidx=5,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=4,
            trail_targets=[None],
            track_idx_map={},
            palette_colors=[(10, 10, 10), (20, 20, 20)],
            has_tracks=False,
        )
        # One trail per instance position index.
        assert len(trails) == 2

    def test_compute_trails_skips_untracked_instance(self):
        """In tracked data, an instance with no track is skipped."""
        from sleap_io.rendering.core import _compute_trails

        labels = _make_trail_labels(n_frames=5)
        skel = labels.skeletons[0]
        # Add an instance with no track assignment.
        labels.labeled_frames[2].instances.append(
            sio.Instance(
                points={"a": [5.0, 5.0], "b": [9.0, 9.0]},
                skeleton=skel,
                track=None,
            )
        )
        frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        track_idx_map = {id(t): i for i, t in enumerate(labels.tracks)}
        trails, _ = _compute_trails(
            fidx=4,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=4,
            trail_targets=[None],
            track_idx_map=track_idx_map,
            palette_colors=[(1, 1, 1), (2, 2, 2)],
            has_tracks=True,
        )
        # Only the two tracked instances yield trails.
        assert len(trails) == 2

    def test_compute_trails_drops_all_nan_trail(self):
        """A trail that is entirely NaN (never-visible node) is dropped."""
        from sleap_io.rendering.core import _compute_trails

        skel = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])
        video = sio.Video(filename="x.mp4", open_backend=False)
        track = sio.Track("t0")
        lfs = []
        for fi in range(5):
            inst = sio.Instance(
                points={"a": [10.0 + fi, 20.0], "b": [np.nan, np.nan]},
                skeleton=skel,
                track=track,
            )
            lfs.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=[inst]))
        labels = sio.Labels(
            labeled_frames=lfs, videos=[video], skeletons=[skel], tracks=[track]
        )
        frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        trails, _ = _compute_trails(
            fidx=4,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=4,
            trail_targets=[0, 1],
            track_idx_map={id(track): 0},
            palette_colors=[(1, 1, 1)],
            has_tracks=True,
        )
        # Node "a" yields a trail; node "b" is all-NaN and dropped.
        assert len(trails) == 1


class TestRenderImageTrails:
    """Tests for show_trails in render_image."""

    def test_render_image_show_trails_centroid(self, labels_predictions):
        """show_trails with centroid target changes the rendered output."""
        without = render_image(labels_predictions, lf_ind=50, show_trails=False)
        with_trails = render_image(
            labels_predictions, lf_ind=50, show_trails=True, trail_length=20
        )
        assert with_trails.shape == without.shape
        assert not np.array_equal(with_trails, without)

    def test_render_image_show_trails_color(self, labels_predictions):
        """A uniform trail_color produces a different render than palette colors."""
        palette_colored = render_image(
            labels_predictions, lf_ind=50, show_trails=True, trail_length=20
        )
        uniform = render_image(
            labels_predictions,
            lf_ind=50,
            show_trails=True,
            trail_length=20,
            trail_color="white",
        )
        assert not np.array_equal(palette_colored, uniform)

    def test_render_image_show_trails_alpha(self, labels_predictions):
        """trail_alpha changes the rendered output."""
        opaque = render_image(
            labels_predictions, lf_ind=50, show_trails=True, trail_length=20
        )
        faint = render_image(
            labels_predictions,
            lf_ind=50,
            show_trails=True,
            trail_length=20,
            trail_alpha=0.25,
        )
        assert not np.array_equal(opaque, faint)

    def test_render_image_show_trails_node(self, labels_predictions):
        """show_trails accepts a node name as the trail target."""
        node_name = labels_predictions.skeletons[0].node_names[0]
        rendered = render_image(
            labels_predictions,
            lf_ind=50,
            show_trails=True,
            trail_node=node_name,
            trail_length=15,
        )
        assert rendered.ndim == 3

    def test_render_image_show_trails_node_list(self, labels_predictions):
        """show_trails accepts a list of node names."""
        node_names = labels_predictions.skeletons[0].node_names[:2]
        rendered = render_image(
            labels_predictions,
            lf_ind=50,
            show_trails=True,
            trail_node=node_names,
            trail_length=15,
        )
        assert rendered.ndim == 3

    def test_render_image_show_trails_unknown_node(self, labels_predictions):
        """An unknown trail_node raises ValueError."""
        with pytest.raises(ValueError, match="Unknown trail_node"):
            render_image(
                labels_predictions,
                lf_ind=50,
                show_trails=True,
                trail_node="not_a_real_node",
            )

    def test_render_image_show_trails_labeled_frame_source_noop(
        self, labels_predictions
    ):
        """show_trails on a LabeledFrame source is skipped without error."""
        lf = labels_predictions.labeled_frames[50]
        rendered = render_image(lf, show_trails=True, trail_length=10)
        assert rendered.ndim == 3

    def test_render_image_show_trails_with_crop(self, labels_predictions):
        """show_trails works together with cropping."""
        rendered = render_image(
            labels_predictions,
            lf_ind=50,
            show_trails=True,
            trail_length=15,
            crop=(50, 50, 250, 250),
        )
        assert rendered.shape == (200, 200, 3)

    def test_render_image_show_trails_first_frame(self, labels_predictions):
        """show_trails on the first frame (no history) does not error."""
        rendered = render_image(
            labels_predictions, lf_ind=0, show_trails=True, trail_length=20
        )
        assert rendered.ndim == 3


class TestRenderVideoTrails:
    """Tests for show_trails in render_video."""

    def test_render_video_show_trails(self, labels_predictions):
        """render_video with show_trails returns rendered frames."""
        frame_inds = [lf.frame_idx for lf in labels_predictions.labeled_frames[:5]]
        frames = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            show_trails=True,
            trail_length=10,
            show_progress=False,
        )
        assert len(frames) == 5
        assert all(f.ndim == 3 for f in frames)

    def test_render_video_show_trails_save(self, labels_predictions, tmp_path):
        """render_video with show_trails writes a video file."""
        output_path = tmp_path / "trails.mp4"
        frame_inds = [lf.frame_idx for lf in labels_predictions.labeled_frames[:5]]
        render_video(
            labels_predictions,
            output_path,
            frame_inds=frame_inds,
            show_trails=True,
            trail_length=10,
            show_progress=False,
        )
        assert output_path.exists()

    def test_render_video_show_trails_node(self, labels_predictions):
        """render_video trails accept a node name."""
        node_name = labels_predictions.skeletons[0].node_names[0]
        frame_inds = [lf.frame_idx for lf in labels_predictions.labeled_frames[:4]]
        frames = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            show_trails=True,
            trail_node=node_name,
            trail_length=8,
            show_progress=False,
        )
        assert len(frames) == 4

    def test_render_video_show_trails_list_source(self, labels_predictions):
        """render_video trails work for a list[LabeledFrame] source."""
        lfs = labels_predictions.labeled_frames[40:55]
        frames = render_video(
            lfs,
            show_trails=True,
            trail_length=8,
            show_progress=False,
        )
        assert len(frames) == len(lfs)

    def test_render_video_show_trails_color_and_alpha(self, labels_predictions):
        """render_video accepts a uniform trail_color and trail_alpha."""
        frame_inds = [lf.frame_idx for lf in labels_predictions.labeled_frames[:5]]
        default = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            show_trails=True,
            trail_length=10,
            show_progress=False,
        )
        styled = render_video(
            labels_predictions,
            frame_inds=frame_inds,
            show_trails=True,
            trail_length=10,
            trail_color=(255, 255, 255),
            trail_alpha=0.4,
            show_progress=False,
        )
        assert len(styled) == 5
        assert not np.array_equal(default[-1], styled[-1])

    def test_render_video_show_trails_include_unlabeled(self):
        """render_video draws trails on unlabeled frames without error.

        With a short trail and labeled frames only at 0-7, later frames have an
        empty trail window, exercising the no-trails fast path too.
        """
        labels = _make_trail_labels(n_frames=8, n_video_frames=16)
        frames = render_video(
            labels,
            background="black",
            show_trails=True,
            trail_length=3,
            include_unlabeled=True,
            start=0,
            end=16,
            show_progress=False,
        )
        assert len(frames) == 16
