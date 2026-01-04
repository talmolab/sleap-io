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

    def test_get_palette_unknown_fallback(self):
        """Test that unknown palette falls back to distinct."""
        from sleap_io.rendering.colors import get_palette

        colors = get_palette("unknown_palette", 5)
        assert len(colors) == 5

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

        rendered = render_image(lf, output=output_path, image=frame)

        assert output_path.exists()
        assert isinstance(rendered, np.ndarray)

    def test_render_image_fallback_color(self, labels_predictions):
        """Test render_image with fallback color when video unavailable."""
        from sleap_io.rendering import render_image

        lf = labels_predictions.labeled_frames[0]

        # Without providing image and require_video=False, should use fallback
        rendered = render_image(
            lf,
            require_video=False,
            fallback_color=(128, 128, 128),
        )

        assert isinstance(rendered, np.ndarray)


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
