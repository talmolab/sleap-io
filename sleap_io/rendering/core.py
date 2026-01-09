"""Core rendering functions for pose visualization.

This module provides the main API for rendering pose data with skia-python.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, Union

import numpy as np

from sleap_io.rendering.callbacks import InstanceContext, RenderContext
from sleap_io.rendering.colors import (
    ColorScheme,
    ColorSpec,
    PaletteName,
    build_color_map,
    determine_color_scheme,
    get_palette,
    resolve_color,
    rgb_to_skia_color,
)
from sleap_io.rendering.shapes import MarkerShape, get_marker_func

if TYPE_CHECKING:
    from sleap_io.model.instance import Instance, PredictedInstance
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video

# Preset configurations
PRESETS: dict[str, dict] = {
    "preview": {"scale": 0.25},
    "draft": {"scale": 0.5},
    "final": {"scale": 1.0},
}

# Type alias for crop specification
# Note: Use Tuple (not tuple) for Python 3.8 compatibility in Union
# Supports both pixel coordinates (int tuple) and normalized coordinates (float tuple)
CropSpec = Union[
    Tuple[int, int, int, int],  # Pixel coordinates
    Tuple[float, float, float, float],  # Normalized coordinates (0.0-1.0)
    None,
]


def _resolve_crop(
    crop: Union[
        Tuple[int, int, int, int],
        Tuple[float, float, float, float],
    ],
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Resolve crop specification to pixel coordinates.

    Supports both pixel coordinates (integers) and normalized coordinates
    (floats in 0.0-1.0 range). Detection is based on Python types:

    - If all values are ``float`` type AND all are in [0.0, 1.0]: normalized
    - Otherwise: pixel coordinates

    Args:
        crop: Crop bounds as (x1, y1, x2, y2) where (x1, y1) is the top-left
            corner and (x2, y2) is the bottom-right corner (exclusive).
        frame_shape: (height, width) of the frame.

    Returns:
        Tuple of (x1, y1, x2, y2) as integer pixel coordinates.

    Examples:
        >>> _resolve_crop((100, 100, 300, 300), (480, 640))
        (100, 100, 300, 300)

        >>> _resolve_crop((0.25, 0.25, 0.75, 0.75), (480, 640))
        (160, 120, 480, 360)
    """
    h, w = frame_shape
    x1, y1, x2, y2 = crop

    # Check if normalized: all values are float type AND in [0.0, 1.0] range
    is_normalized = all(isinstance(v, float) for v in crop) and all(
        0.0 <= v <= 1.0 for v in crop
    )

    if is_normalized:
        # Convert normalized to pixels
        return (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )
    else:
        # Already pixel coordinates
        return (int(x1), int(y1), int(x2), int(y2))


def _apply_crop(
    frame: np.ndarray,
    instances_points: list[np.ndarray],
    crop: tuple[int, int, int, int],
    output_size: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Apply crop to frame and shift instance coordinates.

    Args:
        frame: Input frame array.
        instances_points: List of (n_nodes, 2) arrays.
        crop: (x1, y1, x2, y2) crop bounds.
        output_size: Optional (width, height) for output. If None, uses crop size.

    Returns:
        Tuple of (cropped_frame, shifted_points, scale_factor).
    """
    from PIL import Image

    x1, y1, x2, y2 = crop
    crop_w, crop_h = x2 - x1, y2 - y1

    # Extract crop region
    h, w = frame.shape[:2]
    # Clamp to valid bounds
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    cropped = frame[src_y1:src_y2, src_x1:src_x2]

    # Handle crop extending beyond frame bounds
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        # Create padded frame
        padded = np.zeros((crop_h, crop_w) + frame.shape[2:], dtype=frame.dtype)
        paste_x1 = src_x1 - x1
        paste_y1 = src_y1 - y1
        paste_x2 = paste_x1 + (src_x2 - src_x1)
        paste_y2 = paste_y1 + (src_y2 - src_y1)
        padded[paste_y1:paste_y2, paste_x1:paste_x2] = cropped
        cropped = padded

    # Scale if output_size specified
    scale = 1.0
    if output_size is not None:
        out_w, out_h = output_size
        scale = min(out_w / crop_w, out_h / crop_h)
        if scale != 1.0:
            pil_img = Image.fromarray(cropped)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            cropped = np.array(pil_img)

    # Shift instance coordinates
    shifted_points = []
    for pts in instances_points:
        shifted = pts.copy()
        shifted[:, 0] = (shifted[:, 0] - x1) * scale
        shifted[:, 1] = (shifted[:, 1] - y1) * scale
        shifted_points.append(shifted)

    return cropped, shifted_points, scale


def _prepare_frame_rgba(frame: np.ndarray) -> np.ndarray:
    """Convert frame to RGBA format for Skia.

    Args:
        frame: Input frame array. Can be:
            - Grayscale (H, W)
            - Grayscale with channel (H, W, 1)
            - RGB (H, W, 3)
            - RGBA (H, W, 4)

    Returns:
        RGBA uint8 array (H, W, 4).
    """
    if frame.ndim == 2:
        # Grayscale (H, W) -> RGB
        frame_rgb = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 1:
        # Grayscale with channel (H, W, 1) -> RGB
        frame_rgb = np.concatenate([frame] * 3, axis=-1)
    elif frame.shape[2] == 3:
        frame_rgb = frame
    elif frame.shape[2] == 4:
        # Already RGBA
        return frame.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    # Add alpha channel
    h, w = frame_rgb.shape[:2]
    frame_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    frame_rgba[:, :, :3] = frame_rgb.astype(np.uint8)
    frame_rgba[:, :, 3] = 255
    return frame_rgba


def _create_blank_frame(
    height: int, width: int, color: tuple[int, int, int]
) -> np.ndarray:
    """Create a blank RGBA frame with the specified color.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        color: RGB color tuple.

    Returns:
        RGBA uint8 array (H, W, 4).
    """
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[:, :, 0] = color[0]
    frame[:, :, 1] = color[1]
    frame[:, :, 2] = color[2]
    frame[:, :, 3] = 255
    return frame


def _estimate_frame_size(
    instances_points: list[np.ndarray],
    padding: float = 0.1,
    min_size: int = 64,
) -> tuple[int, int]:
    """Estimate frame dimensions from instance keypoints.

    Computes a bounding box around all valid points and adds padding.

    Args:
        instances_points: List of (n_nodes, 2) arrays of keypoint coordinates.
        padding: Padding as fraction of bounding box size (default 0.1).
        min_size: Minimum dimension size (default 64).

    Returns:
        Tuple of (height, width).
    """
    if not instances_points:
        return (min_size, min_size)

    all_points = np.concatenate(instances_points)
    valid = np.isfinite(all_points).all(axis=1)
    pts = all_points[valid]

    if len(pts) == 0:
        return (min_size, min_size)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Add padding
    pad_x = max((x_max - x_min) * padding, 10)
    pad_y = max((y_max - y_min) * padding, 10)

    # Compute dimensions (ensure non-zero)
    width = max(int(x_max + pad_x), min_size)
    height = max(int(y_max + pad_y), min_size)

    return (height, width)


def _scale_frame(frame_rgba: np.ndarray, scale: float) -> np.ndarray:
    """Scale frame using PIL (faster than Skia for image operations).

    Args:
        frame_rgba: RGBA frame array.
        scale: Scale factor.

    Returns:
        Scaled RGBA array.
    """
    if scale == 1.0:
        return frame_rgba

    from PIL import Image

    h, w = frame_rgba.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)

    pil_img = Image.fromarray(frame_rgba)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil_img)


def render_frame(
    frame: np.ndarray,
    instances_points: list[np.ndarray],
    edge_inds: list[tuple[int, int]],
    node_names: list[str],
    *,
    # Appearance
    color_by: ColorScheme = "instance",
    palette: Union[PaletteName, str] = "standard",
    marker_shape: MarkerShape = "circle",
    marker_size: float = 4.0,
    line_width: float = 2.0,
    alpha: float = 1.0,
    show_nodes: bool = True,
    show_edges: bool = True,
    scale: float = 1.0,
    # Track info for track coloring
    track_indices: Optional[list[int]] = None,
    n_tracks: int = 0,
    # Callbacks
    pre_render_callback: Optional[Callable[[RenderContext], None]] = None,
    post_render_callback: Optional[Callable[[RenderContext], None]] = None,
    per_instance_callback: Optional[Callable[[InstanceContext], None]] = None,
    # Callback context info
    frame_idx: int = 0,
    instance_metadata: Optional[list[dict]] = None,
) -> np.ndarray:
    """Render poses on a single frame.

    This is the core low-level rendering function. For most use cases, prefer
    `render_image()` which provides a higher-level interface.

    Args:
        frame: Background image as numpy array (H, W), (H, W, 1), (H, W, 3) or
            (H, W, 4).
        instances_points: List of (n_nodes, 2) arrays of keypoint coordinates.
        edge_inds: List of (src_idx, dst_idx) tuples for skeleton edges.
        node_names: List of node name strings.
        color_by: Color scheme - 'track', 'instance', or 'node'.
        palette: Color palette name.
        marker_shape: Node marker shape.
        marker_size: Node marker radius in pixels.
        line_width: Edge line width in pixels.
        alpha: Global transparency (0.0-1.0).
        show_nodes: Whether to draw node markers.
        show_edges: Whether to draw skeleton edges.
        scale: Scale factor for output.
        track_indices: Track index for each instance (for track coloring).
        n_tracks: Total number of tracks (for track coloring).
        pre_render_callback: Called before poses are drawn.
        post_render_callback: Called after poses are drawn.
        per_instance_callback: Called after each instance is drawn.
        frame_idx: Current frame index (for callbacks).
        instance_metadata: List of dicts with 'track_id', 'track_name', 'confidence'
            for each instance (for callbacks).

    Returns:
        Rendered RGBA array (H*scale, W*scale, 4) as uint8.
    """
    import skia

    # Prepare frame
    frame_rgba = _prepare_frame_rgba(frame)
    h, w = frame_rgba.shape[:2]

    # Scale frame
    frame_rgba = _scale_frame(frame_rgba, scale)
    out_h, out_w = frame_rgba.shape[:2]

    # Create Skia surface
    surface = skia.Surface(frame_rgba, colorType=skia.kRGBA_8888_ColorType)
    canvas = surface.getCanvas()

    alpha_int = int(alpha * 255)
    n_instances = len(instances_points)
    n_nodes = (
        len(node_names)
        if node_names
        else (len(instances_points[0]) if instances_points else 0)
    )

    # Build color mapping
    color_map = build_color_map(
        scheme=color_by,
        n_instances=n_instances,
        n_nodes=n_nodes,
        n_tracks=n_tracks,
        track_indices=track_indices,
        palette=palette,
    )

    # Get drawing function for marker shape
    draw_marker = get_marker_func(marker_shape)

    # Pre-render callback
    if pre_render_callback:
        ctx = RenderContext(
            canvas=canvas,
            frame_idx=frame_idx,
            frame_size=(w, h),
            instances=instances_points,
            skeleton_edges=edge_inds,
            node_names=node_names,
            scale=scale,
            offset=(0.0, 0.0),
        )
        pre_render_callback(ctx)

    # Draw instances
    for inst_idx, points in enumerate(instances_points):
        # Determine colors for this instance
        if color_by == "node" and "node_colors" in color_map:
            # Node coloring: each node has its own color
            node_colors = color_map["node_colors"]
        else:
            # Instance/track coloring: all nodes share instance color
            if "instance_colors" in color_map:
                inst_color = color_map["instance_colors"][
                    inst_idx % len(color_map["instance_colors"])
                ]
            else:
                inst_color = get_palette(palette, 1)[0]
            node_colors = [inst_color] * n_nodes

        # Draw edges
        if show_edges:
            for src_idx, dst_idx in edge_inds:
                if src_idx >= len(points) or dst_idx >= len(points):
                    continue

                x1, y1 = points[src_idx]
                x2, y2 = points[dst_idx]

                if not (
                    np.isfinite(x1)
                    and np.isfinite(y1)
                    and np.isfinite(x2)
                    and np.isfinite(y2)
                ):
                    continue

                # Edge color: use destination node color for node coloring,
                # instance color otherwise
                edge_color = node_colors[dst_idx % len(node_colors)]
                edge_paint = skia.Paint(
                    Color=rgb_to_skia_color(edge_color, alpha_int),
                    AntiAlias=True,
                    Style=skia.Paint.kStroke_Style,
                    StrokeWidth=line_width * scale,
                    StrokeCap=skia.Paint.kRound_Cap,
                )
                canvas.drawLine(
                    float(x1) * scale,
                    float(y1) * scale,
                    float(x2) * scale,
                    float(y2) * scale,
                    edge_paint,
                )

        # Draw nodes
        if show_nodes:
            for node_idx, (x, y) in enumerate(points):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue

                node_color = node_colors[node_idx % len(node_colors)]
                fill_paint = skia.Paint(
                    Color=rgb_to_skia_color(node_color, alpha_int),
                    AntiAlias=True,
                    Style=skia.Paint.kFill_Style,
                )
                draw_marker(
                    canvas,
                    float(x) * scale,
                    float(y) * scale,
                    marker_size * scale,
                    fill_paint,
                )

        # Per-instance callback
        if per_instance_callback:
            meta = (
                instance_metadata[inst_idx]
                if instance_metadata and inst_idx < len(instance_metadata)
                else {}
            )
            inst_ctx = InstanceContext(
                canvas=canvas,
                instance_idx=inst_idx,
                points=points,
                skeleton_edges=edge_inds,
                node_names=node_names,
                track_id=meta.get("track_id"),
                track_name=meta.get("track_name"),
                confidence=meta.get("confidence"),
                scale=scale,
                offset=(0.0, 0.0),
            )
            per_instance_callback(inst_ctx)

    # Post-render callback
    if post_render_callback:
        ctx = RenderContext(
            canvas=canvas,
            frame_idx=frame_idx,
            frame_size=(w, h),
            instances=instances_points,
            skeleton_edges=edge_inds,
            node_names=node_names,
            scale=scale,
            offset=(0.0, 0.0),
        )
        post_render_callback(ctx)

    surface.flushAndSubmit()

    # Convert to RGB (drop alpha) for video compatibility
    return frame_rgba[:, :, :3]


def render_image(
    source: Union[
        "Labels",
        "LabeledFrame",
        list[Union["Instance", "PredictedInstance"]],
    ],
    save_path: Optional[Union[str, Path]] = None,
    *,
    # Frame specification (for Labels input)
    lf_ind: Optional[int] = None,
    video: Optional[Union["Video", int]] = None,
    frame_idx: Optional[int] = None,
    # Image override
    image: Optional[np.ndarray] = None,
    # Cropping
    crop: CropSpec = None,
    # Appearance
    color_by: ColorScheme = "auto",
    palette: Union[PaletteName, str] = "standard",
    marker_shape: MarkerShape = "circle",
    marker_size: float = 4.0,
    line_width: float = 2.0,
    alpha: float = 1.0,
    show_nodes: bool = True,
    show_edges: bool = True,
    scale: float = 1.0,
    # Background control
    background: Union[Literal["video"], ColorSpec] = "video",
    # Callbacks
    pre_render_callback: Optional[Callable[[RenderContext], None]] = None,
    post_render_callback: Optional[Callable[[RenderContext], None]] = None,
    per_instance_callback: Optional[Callable[[InstanceContext], None]] = None,
) -> np.ndarray:
    """Render single frame with pose overlays.

    Args:
        source: LabeledFrame, Labels (with frame specifier), or list of instances.
        save_path: Output image path (PNG/JPEG). If None, only returns array.
        lf_ind: LabeledFrame index within Labels.labeled_frames (when source is Labels).
        video: Video object or video index (used with frame_idx when source is Labels).
        frame_idx: Video frame index (0-based, used with video when source is Labels).
        image: Override image array (H, W) or (H, W, C) uint8. Fetched from
            LabeledFrame if not provided.
        crop: Crop specification. Bounds are (x1, y1, x2, y2) where (x1, y1) is
            the top-left corner and (x2, y2) is the bottom-right (exclusive).
            Origin (0, 0) is at the image top-left. Can be:

            - **Pixel coordinates** (int tuple): ``(100, 100, 300, 300)`` crops
              from pixel (100, 100) to (300, 300).
            - **Normalized coordinates** (float tuple in [0.0, 1.0]):
              ``(0.25, 0.25, 0.75, 0.75)`` crops the center 50% of the frame.
              Detection is type-based: all values must be ``float`` and in range.
            - ``None``: No cropping (default).
        color_by: Color scheme - 'track', 'instance', 'node', or 'auto'.
        palette: Color palette name.
        marker_shape: Node marker shape.
        marker_size: Node marker radius in pixels.
        line_width: Edge line width in pixels.
        alpha: Global transparency (0.0-1.0).
        show_nodes: Whether to draw node markers.
        show_edges: Whether to draw skeleton edges.
        scale: Output scale factor. Applied after cropping.
        background: Background control. Can be:
            - ``"video"``: Load video frame (default). Raises error if unavailable.
            - Any color spec: Use solid color background, skip video loading entirely.
              Supports RGB tuples ``(255, 128, 0)``, float tuples ``(1.0, 0.5, 0.0)``,
              grayscale ``128`` or ``0.5``, named colors ``"black"``, hex ``"#ff8000"``,
              or palette index ``"tableau10[2]"``.
        pre_render_callback: Called before poses are drawn.
        post_render_callback: Called after poses are drawn.
        per_instance_callback: Called after each instance is drawn.

    Returns:
        Rendered numpy array (H, W, 3) uint8.

    Raises:
        ValueError: If background="video" and video unavailable.

    Examples:
        Render a single labeled frame:

        >>> import sleap_io as sio
        >>> labels = sio.load_slp("predictions.slp")
        >>> lf = labels.labeled_frames[0]
        >>> img = sio.render_image(lf)

        Render with solid color background (no video required):

        >>> img = sio.render_image(lf, background="black")
        >>> img = sio.render_image(lf, background=(40, 40, 40))
        >>> img = sio.render_image(lf, background="#404040")
        >>> img = sio.render_image(lf, background=0.25)

        Crop to a region (pixel coordinates):

        >>> img = sio.render_image(lf, crop=(100, 100, 300, 300))

        Normalized crop (center 50% of frame):

        >>> img = sio.render_image(lf, crop=(0.25, 0.25, 0.75, 0.75))

        Render and save to file:

        >>> sio.render_image(labels, lf_ind=0, save_path="frame.png")
        >>> sio.render_image(labels, video=0, frame_idx=42, save_path="frame.png")
    """
    import skia  # noqa: F401

    from sleap_io.model.instance import Instance, PredictedInstance
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.labels import Labels

    # Handle background parameter
    use_video = background == "video"
    background_color: Optional[tuple[int, int, int]] = None
    if not use_video:
        background_color = resolve_color(background)

    # Resolve source to LabeledFrame or instances
    if isinstance(source, Labels):
        if video is not None and frame_idx is not None:
            # Render by video + frame_idx
            target_video = source.videos[video] if isinstance(video, int) else video
            lf_list = source.find(target_video, frame_idx)
            if not lf_list:
                raise ValueError(
                    f"No labeled frame found for video {target_video} "
                    f"at frame {frame_idx}"
                )
            lf = lf_list[0]
        elif lf_ind is not None:
            # Render by labeled frame index
            lf = source.labeled_frames[lf_ind]
        else:
            # Default to first labeled frame
            lf = source.labeled_frames[0]

        instances = list(lf.instances)
        skeleton = instances[0].skeleton if instances else source.skeletons[0]
        edge_inds = skeleton.edge_inds
        node_names = [n.name for n in skeleton.nodes]
        fidx_for_callback = lf.frame_idx

        # Get track info
        track_indices = []
        n_tracks = len(source.tracks)
        for inst in instances:
            if inst.track is not None and inst.track in source.tracks:
                track_indices.append(source.tracks.index(inst.track))
            else:
                track_indices.append(0)

        has_tracks = n_tracks > 0

        # Convert instances to point arrays (needed for both image size and rendering)
        instances_points = [inst.numpy() for inst in instances]

        # Get image if not provided
        if image is None:
            if background_color is not None:
                # Solid color background - skip video loading entirely
                video_obj = lf.video
                if hasattr(video_obj, "shape") and video_obj.shape is not None:
                    h, w = video_obj.shape[1:3]
                else:
                    # Estimate from points
                    h, w = _estimate_frame_size(instances_points)
                image = _create_blank_frame(h, w, background_color)[:, :, :3]
            else:
                # Load video frame
                try:
                    image = lf.image
                    if image is None:
                        raise ValueError("No image available")
                except Exception:
                    raise ValueError(
                        "Video unavailable. Specify a background color to render "
                        "without video, e.g., background='black' or "
                        "background=(40, 40, 40)."
                    )

    elif isinstance(source, LabeledFrame):
        lf = source
        instances = list(lf.instances)
        skeleton = instances[0].skeleton if instances else None
        if skeleton is None:
            raise ValueError("LabeledFrame has no instances with skeleton")
        edge_inds = skeleton.edge_inds
        node_names = [n.name for n in skeleton.nodes]
        fidx_for_callback = lf.frame_idx
        track_indices = None
        n_tracks = 0
        has_tracks = False

        # Convert instances to point arrays (needed for both image size and rendering)
        instances_points = [inst.numpy() for inst in instances]

        # Get image if not provided
        if image is None:
            if background_color is not None:
                # Solid color background - skip video loading entirely
                video_obj = lf.video
                if hasattr(video_obj, "shape") and video_obj.shape is not None:
                    h, w = video_obj.shape[1:3]
                else:
                    # Estimate from points
                    h, w = _estimate_frame_size(instances_points)
                image = _create_blank_frame(h, w, background_color)[:, :, :3]
            else:
                # Load video frame
                try:
                    image = lf.image
                    if image is None:
                        raise ValueError("No image available")
                except Exception:
                    raise ValueError(
                        "Video unavailable. Specify a background color to render "
                        "without video, e.g., background='black' or "
                        "background=(40, 40, 40)."
                    )

    elif isinstance(source, list) and all(
        isinstance(x, (Instance, PredictedInstance)) for x in source
    ):
        instances = source
        if not instances:
            raise ValueError("Empty instances list")
        skeleton = instances[0].skeleton
        edge_inds = skeleton.edge_inds
        node_names = [n.name for n in skeleton.nodes]
        fidx_for_callback = 0
        track_indices = None
        n_tracks = 0
        has_tracks = False

        # Convert instances to point arrays
        instances_points = [inst.numpy() for inst in instances]

        if image is None:
            raise ValueError(
                "image parameter required when source is list of instances"
            )

    else:
        raise TypeError(
            f"source must be Labels, LabeledFrame, or list of instances, "
            f"got {type(source)}"
        )

    # Apply cropping if specified
    render_image_data = image
    render_points = instances_points
    if crop is not None:
        h, w = image.shape[:2]
        # Resolve normalized or pixel coordinates
        crop_bounds = _resolve_crop(crop, (h, w))

        render_image_data, render_points, _ = _apply_crop(
            image, instances_points, crop_bounds
        )

    # Build instance metadata for callbacks
    instance_metadata = []
    for inst in instances:
        meta = {}
        if hasattr(inst, "track") and inst.track is not None:
            meta["track_name"] = inst.track.name
        if hasattr(inst, "score"):
            meta["confidence"] = inst.score
        instance_metadata.append(meta)

    # Determine color scheme
    resolved_scheme = determine_color_scheme(
        has_tracks=has_tracks,
        is_single_image=True,
        scheme=color_by,
    )

    # Render
    rendered = render_frame(
        frame=render_image_data,
        instances_points=render_points,
        edge_inds=edge_inds,
        node_names=node_names,
        color_by=resolved_scheme,
        palette=palette,
        marker_shape=marker_shape,
        marker_size=marker_size,
        line_width=line_width,
        alpha=alpha,
        show_nodes=show_nodes,
        show_edges=show_edges,
        scale=scale,
        track_indices=track_indices,
        n_tracks=n_tracks,
        pre_render_callback=pre_render_callback,
        post_render_callback=post_render_callback,
        per_instance_callback=per_instance_callback,
        frame_idx=fidx_for_callback,
        instance_metadata=instance_metadata,
    )

    # Save if save_path provided
    if save_path is not None:
        from PIL import Image

        save_path_ = Path(save_path)
        save_path_.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rendered).save(save_path_)

    return rendered


def render_video(
    source: Union["Labels", list["LabeledFrame"]],
    save_path: Optional[Union[str, Path]] = None,
    *,
    # Video selection
    video: Optional[Union["Video", int]] = None,
    # Frame selection
    frame_inds: Optional[list[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    include_unlabeled: bool = False,
    # Cropping
    crop: CropSpec = None,
    # Quality/scale
    preset: Optional[Literal["preview", "draft", "final"]] = None,
    scale: float = 1.0,
    # Appearance
    color_by: ColorScheme = "auto",
    palette: Union[PaletteName, str] = "standard",
    marker_shape: MarkerShape = "circle",
    marker_size: float = 4.0,
    line_width: float = 2.0,
    alpha: float = 1.0,
    show_nodes: bool = True,
    show_edges: bool = True,
    # Video encoding
    fps: Optional[float] = None,
    codec: str = "libx264",
    crf: int = 25,
    x264_preset: str = "superfast",
    # Background control
    background: Union[Literal["video"], ColorSpec] = "video",
    # Callbacks
    pre_render_callback: Optional[Callable[[RenderContext], None]] = None,
    post_render_callback: Optional[Callable[[RenderContext], None]] = None,
    per_instance_callback: Optional[Callable[[InstanceContext], None]] = None,
    # Progress
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    show_progress: bool = True,
) -> Union["Video", list[np.ndarray]]:
    """Render video with pose overlays.

    Args:
        source: Labels object or list of LabeledFrames to render.
        save_path: Output video path. If None, returns list of rendered arrays.
        video: Video to render from (default: first video in Labels).
        frame_inds: Specific frame indices to render.
        start: Start frame index (inclusive).
        end: End frame index (exclusive).
        include_unlabeled: If True, render all frames in range even if they have
            no LabeledFrame (just shows video frame without poses). Default False.
        crop: Static crop applied uniformly to all frames. Bounds are
            (x1, y1, x2, y2) where (x1, y1) is the top-left corner and (x2, y2)
            is the bottom-right (exclusive). Supports:

            - **Pixel coordinates** (int tuple): ``(100, 100, 300, 300)``
            - **Normalized coordinates** (float tuple in [0.0, 1.0]):
              ``(0.25, 0.25, 0.75, 0.75)`` crops the center 50%.
            - ``None``: No cropping (default).
        preset: Quality preset ('preview'=0.25x, 'draft'=0.5x, 'final'=1.0x).
        scale: Scale factor (overrides preset if both provided).
        color_by: Color scheme - 'track', 'instance', 'node', or 'auto'.
        palette: Color palette name.
        marker_shape: Node marker shape.
        marker_size: Node marker radius in pixels.
        line_width: Edge line width in pixels.
        alpha: Global transparency (0.0-1.0).
        show_nodes: Whether to draw node markers.
        show_edges: Whether to draw skeleton edges.
        fps: Output frame rate (default: source video fps).
        codec: Video codec for encoding.
        crf: Constant rate factor for quality (2-32, lower=better). Default 25.
        x264_preset: H.264 encoding preset (ultrafast, superfast, fast, medium, slow).
        background: Background control. Can be:
            - ``"video"``: Load video frame (default). Raises error if unavailable.
            - Any color spec: Use solid color background, skip video loading entirely.
              Supports RGB tuples ``(255, 128, 0)``, float tuples ``(1.0, 0.5, 0.0)``,
              grayscale ``128`` or ``0.5``, named colors ``"black"``, hex ``"#ff8000"``,
              or palette index ``"tableau10[2]"``.
        pre_render_callback: Called before each frame's poses are drawn.
        post_render_callback: Called after each frame's poses are drawn.
        per_instance_callback: Called after each instance is drawn.
        progress_callback: Called with (current, total), return False to cancel.
        show_progress: Show tqdm progress bar.

    Returns:
        If save_path provided: Video object pointing to output file.
        If save_path is None: List of rendered numpy arrays (H, W, 3) uint8.

    Raises:
        ValueError: If background="video" and video unavailable.

    Examples:
        Render full video with pose overlays:

        >>> import sleap_io as sio
        >>> labels = sio.load_slp("predictions.slp")
        >>> sio.render_video(labels, "output.mp4")

        Fast preview at reduced resolution:

        >>> sio.render_video(labels, "preview.mp4", preset="preview")

        Get rendered frames as numpy arrays:

        >>> frames = sio.render_video(labels)
    """
    import skia  # noqa: F401

    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video as VideoModel

    # Handle background parameter
    use_video = background == "video"
    background_color: Optional[tuple[int, int, int]] = None
    if not use_video:
        background_color = resolve_color(background)

    # Handle preset
    if preset is not None and preset in PRESETS:
        scale = PRESETS[preset]["scale"]

    # Resolve source
    if isinstance(source, Labels):
        labels = source

        # Resolve video
        if video is None:
            if not labels.videos:
                raise ValueError("Labels has no videos")
            target_video = labels.videos[0]
        elif isinstance(video, int):
            target_video = labels.videos[video]
        else:
            target_video = video

        # Get labeled frames for this video
        labeled_frames = labels.find(target_video)
        if not labeled_frames:
            raise ValueError(f"No labeled frames found for video {target_video}")

        # Sort by frame index
        labeled_frames = sorted(labeled_frames, key=lambda lf: lf.frame_idx)

        # Get skeleton info
        skeleton = labels.skeletons[0] if labels.skeletons else None
        if skeleton is None and labeled_frames:
            for lf in labeled_frames:
                for inst in lf.instances:
                    skeleton = inst.skeleton
                    break
                if skeleton:
                    break

        if skeleton is None:
            raise ValueError("No skeleton found in labels")

        edge_inds = skeleton.edge_inds
        node_names = [n.name for n in skeleton.nodes]
        n_tracks = len(labels.tracks)
        has_tracks = n_tracks > 0

    elif isinstance(source, list) and all(isinstance(x, LabeledFrame) for x in source):
        labeled_frames = source
        if not labeled_frames:
            raise ValueError("Empty labeled frames list")

        target_video = labeled_frames[0].video
        skeleton = None
        for lf in labeled_frames:
            for inst in lf.instances:
                skeleton = inst.skeleton
                break
            if skeleton:
                break

        if skeleton is None:
            raise ValueError("No skeleton found in labeled frames")

        edge_inds = skeleton.edge_inds
        node_names = [n.name for n in skeleton.nodes]
        n_tracks = 0
        has_tracks = False
        labels = None

    else:
        raise TypeError(
            f"source must be Labels or list of LabeledFrame, got {type(source)}"
        )

    # Create frame index mapping
    frame_idx_to_lf = {lf.frame_idx: lf for lf in labeled_frames}

    # Get video frame count for include_unlabeled mode
    n_video_frames = None
    if include_unlabeled:
        if hasattr(target_video, "shape") and target_video.shape is not None:
            n_video_frames = target_video.shape[0]

    # Determine frame indices to render
    if frame_inds is not None:
        render_indices = frame_inds
    elif start is not None or end is not None:
        labeled_indices = sorted(frame_idx_to_lf.keys())
        if include_unlabeled and n_video_frames is not None:
            # Render all frames in range, not just labeled ones
            start_idx = start if start is not None else 0
            end_idx = end if end is not None else n_video_frames
            render_indices = list(range(start_idx, end_idx))
        else:
            # Only render labeled frames in range
            start_idx = start if start is not None else min(labeled_indices, default=0)
            end_idx = end if end is not None else max(labeled_indices, default=0) + 1
            render_indices = [i for i in labeled_indices if start_idx <= i < end_idx]
    else:
        if include_unlabeled and n_video_frames is not None:
            # Render entire video
            render_indices = list(range(n_video_frames))
        else:
            # Only render labeled frames
            render_indices = sorted(frame_idx_to_lf.keys())

    if not render_indices:
        raise ValueError("No frames to render")

    # Determine FPS
    if fps is None:
        # Try to get from video
        if hasattr(target_video, "backend") and target_video.backend is not None:
            try:
                fps = target_video.backend.fps
            except Exception:
                fps = 30.0
        else:
            fps = 30.0

    # Determine color scheme
    resolved_scheme = determine_color_scheme(
        has_tracks=has_tracks,
        is_single_image=False,
        scheme=color_by,
    )

    # Resolve crop bounds once (before the loop)
    # We need the video shape to resolve normalized coordinates
    crop_bounds: Optional[tuple[int, int, int, int]] = None
    if crop is not None:
        if hasattr(target_video, "shape") and target_video.shape is not None:
            h, w = target_video.shape[1:3]
        else:
            # Fallback: try to get from first frame
            h, w = 480, 640  # reasonable default
        crop_bounds = _resolve_crop(crop, (h, w))

    # Setup progress
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(render_indices, desc="Rendering", unit="frame")
        except ImportError:
            iterator = render_indices
    else:
        iterator = render_indices

    # Render frames
    rendered_frames = []
    total_frames = len(render_indices)

    for i, fidx in enumerate(iterator):
        # Check for cancellation
        if progress_callback is not None:
            if progress_callback(i, total_frames) is False:
                break

        lf = frame_idx_to_lf.get(fidx)

        # Handle frames without LabeledFrame
        if lf is None:
            if not include_unlabeled:
                continue
            # Render just the video frame without poses
            if background_color is not None:
                # Solid color background - skip video loading entirely
                if hasattr(target_video, "shape") and target_video.shape is not None:
                    h, w = target_video.shape[1:3]
                else:
                    # No video metadata and no points - use minimum default
                    h, w = 64, 64
                image = _create_blank_frame(h, w, background_color)[:, :, :3]
            else:
                try:
                    image = target_video[fidx]
                    if image is None:
                        raise ValueError("No image")
                except Exception:
                    raise ValueError(
                        f"Video unavailable at frame {fidx}. "
                        "Specify a background color to render without video."
                    )

            # Apply cropping if specified
            render_image_data = image
            if crop_bounds is not None:
                render_image_data, _, _ = _apply_crop(image, [], crop_bounds)

            # Render frame without poses
            rendered = render_frame(
                frame=render_image_data,
                instances_points=[],
                edge_inds=edge_inds,
                node_names=node_names,
                color_by=resolved_scheme,
                palette=palette,
                marker_shape=marker_shape,
                marker_size=marker_size,
                line_width=line_width,
                alpha=alpha,
                show_nodes=show_nodes,
                show_edges=show_edges,
                scale=scale,
                track_indices=None,
                n_tracks=n_tracks,
                pre_render_callback=pre_render_callback,
                post_render_callback=post_render_callback,
                per_instance_callback=None,
                frame_idx=fidx,
                instance_metadata=[],
            )
            rendered_frames.append(rendered)
            continue

        instances = list(lf.instances)
        instances_points = [inst.numpy() for inst in instances]

        # Get track indices
        track_indices = None
        if labels is not None and has_tracks:
            track_indices = []
            for inst in instances:
                if inst.track is not None and inst.track in labels.tracks:
                    track_indices.append(labels.tracks.index(inst.track))
                else:
                    track_indices.append(0)

        # Build instance metadata
        instance_metadata = []
        for inst in instances:
            meta = {}
            if hasattr(inst, "track") and inst.track is not None:
                meta["track_name"] = inst.track.name
            if hasattr(inst, "score"):
                meta["confidence"] = inst.score
            instance_metadata.append(meta)

        # Get image
        if background_color is not None:
            # Solid color background - skip video loading entirely
            if hasattr(target_video, "shape") and target_video.shape is not None:
                h, w = target_video.shape[1:3]
            else:
                # Estimate from points
                h, w = _estimate_frame_size(instances_points)
            image = _create_blank_frame(h, w, background_color)[:, :, :3]
        else:
            try:
                image = lf.image
                if image is None:
                    raise ValueError("No image")
            except Exception:
                raise ValueError(
                    f"Video unavailable at frame {fidx}. "
                    "Specify a background color to render without video."
                )

        # Apply cropping if specified
        render_image_data = image
        render_points = instances_points
        if crop_bounds is not None:
            render_image_data, render_points, _ = _apply_crop(
                image, instances_points, crop_bounds
            )

        # Render frame
        rendered = render_frame(
            frame=render_image_data,
            instances_points=render_points,
            edge_inds=edge_inds,
            node_names=node_names,
            color_by=resolved_scheme,
            palette=palette,
            marker_shape=marker_shape,
            marker_size=marker_size,
            line_width=line_width,
            alpha=alpha,
            show_nodes=show_nodes,
            show_edges=show_edges,
            scale=scale,
            track_indices=track_indices,
            n_tracks=n_tracks,
            pre_render_callback=pre_render_callback,
            post_render_callback=post_render_callback,
            per_instance_callback=per_instance_callback,
            frame_idx=fidx,
            instance_metadata=instance_metadata,
        )

        rendered_frames.append(rendered)

    # Write video or return frames
    if save_path is not None:
        from sleap_io.io.video_writing import VideoWriter

        save_path_ = Path(save_path)
        save_path_.parent.mkdir(parents=True, exist_ok=True)

        with VideoWriter(
            filename=save_path_,
            fps=fps,
            codec=codec,
            crf=crf,
            preset=x264_preset,
        ) as writer:
            for frame in rendered_frames:
                writer(frame)

        # Return Video object pointing to output
        return VideoModel.from_filename(str(save_path_))

    return rendered_frames
