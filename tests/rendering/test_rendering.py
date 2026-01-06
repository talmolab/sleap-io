"""Tests for the rendering module.

These tests validate the rendering functionality including colors, shapes,
callbacks, and the core rendering functions.
"""

import numpy as np
import pytest

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

    def test_render_image_labeled_frame_no_skeleton_error(self):
        """Test render_image with LabeledFrame that has no instances."""
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_image

        # Create empty labeled frame
        video = Video(filename="dummy.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[])

        with pytest.raises(ValueError, match="no instances with skeleton"):
            render_image(lf, image=np.zeros((100, 100, 3), dtype=np.uint8))

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
        """Test render_video raises error when no skeleton found."""
        from sleap_io.model.labeled_frame import LabeledFrame
        from sleap_io.model.labels import Labels
        from sleap_io.model.video import Video
        from sleap_io.rendering import render_video

        video = Video(filename="dummy.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[])
        labels = Labels(videos=[video], labeled_frames=[lf])

        with pytest.raises(ValueError, match="No skeleton found"):
            render_video(labels, show_progress=False)

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
