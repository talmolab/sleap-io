"""Core rendering functions for pose visualization.

This module provides the main API for rendering pose data with skia-python.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Literal

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
    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.instance import Instance, PredictedInstance
    from sleap_io.model.label_image import LabelImage
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.labels import Labels
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.roi import ROI
    from sleap_io.model.skeleton import Skeleton
    from sleap_io.model.video import Video

# Preset configurations
PRESETS: dict[str, dict] = {
    "preview": {"scale": 0.25},
    "draft": {"scale": 0.5},
    "final": {"scale": 1.0},
}

# Type alias for crop specification
# Supports both pixel coordinates (int tuple) and normalized coordinates (float tuple)
CropSpec = (
    tuple[int, int, int, int]  # Pixel coordinates
    | tuple[float, float, float, float]  # Normalized coordinates (0.0-1.0)
    | None
)


def _resolve_crop(
    crop: tuple[int, int, int, int] | tuple[float, float, float, float],
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
    output_size: tuple[int, int] | None = None,
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


def _resolve_trail_node(
    trail_node: str | list[str],
    skeleton: "Skeleton",
) -> list[int | None]:
    """Resolve a ``trail_node`` specification to a list of trail targets.

    Args:
        trail_node: Trail target specification. One of:

            - ``"centroid"``: trail the instance centroid.
            - A node name string: trail that node.
            - A list of node name strings: trail each node separately.
        skeleton: Skeleton used to resolve node names to indices.

    Returns:
        List of targets, one per requested node. Each target is ``None``
        (centroid) or an integer node index.

    Raises:
        ValueError: If a node name is not present in the skeleton.
    """
    names = [trail_node] if isinstance(trail_node, str) else list(trail_node)

    targets: list[int | None] = []
    for name in names:
        if isinstance(name, str) and name.lower() == "centroid":
            targets.append(None)
            continue
        try:
            targets.append(skeleton.index(name))
        except (KeyError, IndexError):
            raise ValueError(
                f"Unknown trail_node {name!r}; skeleton nodes: {skeleton.node_names}"
            )
    return targets


def _compute_trails(
    fidx: int,
    frame_idx_to_lf: dict[int, "LabeledFrame"],
    trail_length: int,
    trail_targets: list[int | None],
    track_idx_map: dict[int, int],
    palette_colors: list[tuple[int, int, int]],
    has_tracks: bool,
    pts_cache: dict[int, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """Compute motion-trail polylines ending at a given frame.

    Args:
        fidx: Current frame index (the trail ends here).
        frame_idx_to_lf: Mapping from frame index to LabeledFrame.
        trail_length: Number of past frames behind the current frame. The trail
            spans frames ``[fidx - trail_length, fidx]`` inclusive.
        trail_targets: List of targets from `_resolve_trail_node`. Each is
            ``None`` (centroid) or an integer node index.
        track_idx_map: Mapping from ``id(track)`` to track index, used to key
            trails by track and to assign colors.
        palette_colors: Color palette indexed by track index (tracked data) or
            instance index (untracked data).
        has_tracks: Whether the data has track assignments. When ``False``,
            trails are keyed by instance position index instead of track.
        pts_cache: Optional cache mapping ``id(instance)`` to its extracted
            ``(n_nodes, 2)`` point array. When rendering a video, consecutive
            frames share overlapping trail windows, so passing a persistent
            cache avoids re-extracting the same instance points repeatedly.

    Returns:
        Tuple of ``(trails, colors)`` where ``trails`` is a list of ``(M, 2)``
        float arrays (``M = trail_length + 1``, oldest to newest, NaN for
        missing positions) and ``colors`` is the parallel list of RGB tuples.
    """
    frame_range = range(fidx - trail_length, fidx + 1)
    n_points = trail_length + 1

    # Map from (key index, target index) -> (M, 2) array of positions, where
    # key index is the track index (tracked) or instance index (untracked).
    trail_data: dict[tuple[int, int], np.ndarray] = {}

    for j, f in enumerate(frame_range):
        lf = frame_idx_to_lf.get(f)
        if lf is None:
            continue
        for inst_idx, inst in enumerate(lf.instances):
            if has_tracks:
                if inst.track is None:
                    continue
                key_idx = track_idx_map.get(id(inst.track))
                if key_idx is None:
                    continue
            else:
                key_idx = inst_idx

            # Extract instance points once, reusing the cache across the
            # overlapping trail windows of consecutive frames when provided.
            if pts_cache is not None:
                pts = pts_cache.get(id(inst))
                if pts is None:
                    pts = inst.numpy()
                    pts_cache[id(inst)] = pts
            else:
                pts = inst.numpy()

            for t_idx, target in enumerate(trail_targets):
                if target is None:
                    # Centroid: mean of visible points, matching
                    # `Instance.centroid_xy` (visibility keyed off column 0).
                    visible = ~np.isnan(pts[:, 0])
                    if visible.any():
                        coord = (
                            float(pts[visible, 0].mean()),
                            float(pts[visible, 1].mean()),
                        )
                    else:
                        coord = (np.nan, np.nan)
                elif target < len(pts):
                    coord = (float(pts[target][0]), float(pts[target][1]))
                else:
                    coord = (np.nan, np.nan)

                dkey = (key_idx, t_idx)
                arr = trail_data.get(dkey)
                if arr is None:
                    arr = np.full((n_points, 2), np.nan, dtype=np.float64)
                    trail_data[dkey] = arr
                arr[j] = coord

    trails: list[np.ndarray] = []
    colors: list[tuple[int, int, int]] = []
    for (key_idx, _), arr in trail_data.items():
        if not np.isfinite(arr).any():
            continue
        trails.append(arr)
        colors.append(palette_colors[key_idx % len(palette_colors)])

    return trails, colors


def _n_trail_palette_colors(
    has_tracks: bool,
    n_tracks: int,
    labeled_frames: Iterable["LabeledFrame"],
) -> int:
    """Return the number of palette colors needed for motion trails.

    Trails are colored by track when tracks are present, otherwise by instance
    position index. To keep untracked colors stable across a render, the palette
    is sized to the largest instance count over the provided frames.

    Args:
        has_tracks: Whether the data has track assignments.
        n_tracks: Total number of tracks (used when ``has_tracks`` is ``True``).
        labeled_frames: Frames to scan for the peak instance count (used when
            ``has_tracks`` is ``False``).

    Returns:
        The palette size, always at least 1.
    """
    if has_tracks:
        return max(n_tracks, 1)
    peak = max((len(lf.instances) for lf in labeled_frames), default=1)
    return max(peak, 1)


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


def _save_image(image: np.ndarray, save_path: str | Path) -> None:
    """Save a rendered image to disk.

    Args:
        image: Image array to save (any shape accepted by PIL).
        save_path: Output file path. Parent directories are created if needed.
    """
    from PIL import Image

    save_path_ = Path(save_path)
    save_path_.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(save_path_)


def _is_label_image(obj: object) -> bool:
    """Check if an object is a LabelImage."""
    from sleap_io.model.label_image import LabelImage

    return isinstance(obj, LabelImage)


def _apply_overlay(
    image: np.ndarray,
    overlay: (
        "np.ndarray | LabelImage"
        " | SegmentationMask | ROI | BoundingBox"
        " | list[SegmentationMask] | list[ROI] | list[BoundingBox]"
    ),
    alpha: float = 0.3,
    palette: PaletteName | str = "distinct",
    outline: bool = False,
    outline_width: int = 1,
    outline_color: tuple[int, int, int] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Apply an annotation overlay to an image, dispatching by type.

    Supports integer label images, ``SegmentationMask``, ``ROI``, and
    ``BoundingBox`` objects. Each item is rendered with a distinct color
    drawn from the specified palette.

    Args:
        image: Image array ``(H, W, 3)`` uint8 (modified in-place).
        overlay: Annotation data. One of:

            - ``np.ndarray``: Integer label image ``(H, W)`` with 0 =
              background and positive values as object IDs.
            - ``SegmentationMask``, ``ROI``, or ``BoundingBox``: A single
              annotation object (normalized to a one-element list).
            - ``list[SegmentationMask]``: Binary segmentation masks.
            - ``list[ROI]``: Vector geometries (polygons, points, etc.).
            - ``list[BoundingBox]``: Axis-aligned or rotated bounding boxes.
        alpha: Fill opacity (0.0 to 1.0).
        palette: Color palette name for per-item coloring.
        outline: Draw outlines (only used for label images).
        outline_width: Outline width in pixels.
        outline_color: Uniform outline color, or ``None`` for auto-darkened.
        colors: Optional per-element RGB colors for a ``list`` overlay. When
            provided, overrides the positional ``palette`` coloring (used by
            callers to color overlays by track identity). Must match the length
            of ``overlay``. Ignored for label-image overlays.

    Returns:
        The modified image array.
    """
    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.roi import ROI
    from sleap_io.rendering.overlays import (
        draw_bboxes,
        draw_label_image,
        draw_masks,
        draw_rois,
    )

    # Normalize a single annotation object to a one-element list so a bare
    # ``SegmentationMask``/``ROI``/``BoundingBox`` (or a User/Predicted subclass)
    # "just works" like a one-element list. ndarray/LabelImage are handled below.
    if isinstance(overlay, (SegmentationMask, ROI, BoundingBox)):
        overlay = [overlay]

    if isinstance(overlay, np.ndarray):
        draw_label_image(
            image,
            overlay,
            alpha=alpha,
            palette=palette,
            outline=outline,
            outline_width=outline_width,
            outline_color=outline_color,
        )
    elif _is_label_image(overlay):
        draw_label_image(
            image,
            overlay.data,
            alpha=alpha,
            palette=palette,
            outline=outline,
            outline_width=outline_width,
            outline_color=outline_color,
            scale=overlay.scale,
            offset=overlay.offset,
        )
    elif isinstance(overlay, list) and overlay:
        first = overlay[0]

        if _is_label_image(first):
            raise TypeError(
                "Pass individual LabelImage objects to _apply_overlay, not a "
                "list. Per-frame dispatch from a list[LabelImage] should "
                "happen at the render_video level."
            )

        if colors is None:
            colors = get_palette(palette, len(overlay))

        if isinstance(first, SegmentationMask):
            draw_masks(image, overlay, colors=colors, alpha=alpha)
        elif isinstance(first, ROI):
            draw_rois(image, overlay, colors=colors, fill_alpha=alpha)
        elif isinstance(first, BoundingBox):
            draw_bboxes(image, overlay, colors=colors, fill_alpha=alpha)
        else:
            raise TypeError(
                f"Unsupported overlay element type: {type(first).__name__}. "
                "Expected SegmentationMask, ROI, or BoundingBox."
            )

    return image


def _compute_identity_coloring(
    instances: Iterable,
    catalog: list | None,
) -> tuple[list[int], int]:
    """Compute per-instance identity indices for the ``identity`` color scheme.

    Mirrors the track-index plumbing. Each instance is mapped to a stable index in
    the identity catalog (the ``Labels.identities`` order when available, otherwise
    discovered from the instances in encounter order). Instances without an identity
    map to index 0. Coloring is by palette index, exactly like the ``track`` scheme.

    Args:
        instances: Instances to color (order matches the rendered points).
        catalog: Optional starting identity catalog (e.g. ``labels.identities``).

    Returns:
        Tuple of ``(identity_indices, n_identities)``.
    """
    catalog = list(catalog) if catalog else []
    idmap = {id(idn): i for i, idn in enumerate(catalog)}
    identity_indices: list[int] = []
    for inst in instances:
        idn = getattr(inst, "identity", None)
        if idn is None:
            identity_indices.append(0)
            continue
        if id(idn) not in idmap:
            idmap[id(idn)] = len(catalog)
            catalog.append(idn)
        identity_indices.append(idmap[id(idn)])
    return identity_indices, len(catalog)


def _compute_category_coloring(
    instances: Iterable,
    catalog: list | None,
) -> tuple[list[int], int]:
    """Compute per-instance category indices for the ``category`` color scheme.

    Mirrors ``_compute_identity_coloring``. Each instance is mapped to a stable
    index in the category catalog (the ``Labels.categories`` order when available,
    otherwise discovered from the instances in encounter order). Instances without a
    category map to index 0. Coloring is by palette index, exactly like the
    ``identity`` scheme.

    Args:
        instances: Instances to color (order matches the rendered points).
        catalog: Optional starting category catalog (e.g. ``labels.categories``).

    Returns:
        Tuple of ``(category_indices, n_categories)``.
    """
    catalog = list(catalog) if catalog else []
    idmap = {id(cat): i for i, cat in enumerate(catalog)}
    category_indices: list[int] = []
    for inst in instances:
        cat = getattr(inst, "category", None)
        if cat is None:
            category_indices.append(0)
            continue
        if id(cat) not in idmap:
            idmap[id(cat)] = len(catalog)
            catalog.append(cat)
        category_indices.append(idmap[id(cat)])
    return category_indices, len(catalog)


def render_frame(
    frame: np.ndarray,
    instances_points: list[np.ndarray],
    edge_inds: list[tuple[int, int]],
    node_names: list[str],
    *,
    # Appearance
    color_by: ColorScheme = "instance",
    palette: PaletteName | str = "standard",
    marker_shape: MarkerShape = "circle",
    marker_size: float = 4.0,
    line_width: float = 2.0,
    alpha: float = 1.0,
    show_nodes: bool = True,
    show_edges: bool = True,
    scale: float = 1.0,
    # Track info for track coloring
    track_indices: list[int] | None = None,
    n_tracks: int = 0,
    # Identity info for identity coloring
    identity_indices: list[int] | None = None,
    n_identities: int = 0,
    # Category info for category coloring
    category_indices: list[int] | None = None,
    n_categories: int = 0,
    # Callbacks
    pre_render_callback: Callable[[RenderContext], None] | None = None,
    post_render_callback: Callable[[RenderContext], None] | None = None,
    per_instance_callback: Callable[[InstanceContext], None] | None = None,
    # Callback context info
    frame_idx: int = 0,
    instance_metadata: list[dict] | None = None,
    crop_offset: tuple[float, float] = (0.0, 0.0),
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
        color_by: Color scheme - 'track', 'instance', 'node', 'identity', or
            'category'.
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
        identity_indices: Global identity index for each instance (for identity
            coloring; a palette index into ``Labels.identities`` order).
        n_identities: Total number of identities (for identity coloring).
        category_indices: Global category index for each instance (for category
            coloring; a palette index into ``Labels.categories`` order).
        n_categories: Total number of categories (for category coloring).
        pre_render_callback: Called before poses are drawn.
        post_render_callback: Called after poses are drawn.
        per_instance_callback: Called after each instance is drawn.
        frame_idx: Current frame index (for callbacks).
        instance_metadata: List of dicts with 'track_id', 'track_name', 'confidence'
            for each instance (for callbacks).
        crop_offset: (x, y) offset for cropped views, passed to RenderContext
            for correct world_to_canvas transformation in callbacks.

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
        identity_indices=identity_indices,
        n_identities=n_identities,
        category_indices=category_indices,
        n_categories=n_categories,
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
            offset=crop_offset,
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
                offset=crop_offset,
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
            offset=crop_offset,
        )
        post_render_callback(ctx)

    surface.flushAndSubmit()

    # Convert to RGB (drop alpha) for video compatibility
    return frame_rgba[:, :, :3]


def render_image(
    source: "Labels | LabeledFrame | list[Instance | PredictedInstance] | None" = None,
    save_path: str | Path | None = None,
    *,
    # Frame specification (for Labels input)
    lf_ind: int | None = None,
    video: "Video | int | None" = None,
    frame_idx: int | None = None,
    # Image override
    image: np.ndarray | None = None,
    # Annotation overlay
    overlay: (
        "np.ndarray | LabelImage"
        " | SegmentationMask | ROI | BoundingBox"
        " | list[SegmentationMask] | list[ROI] | list[BoundingBox]"
        " | None"
    ) = None,
    overlay_alpha: float = 0.3,
    overlay_palette: PaletteName | str = "distinct",
    overlay_outline: bool = False,
    overlay_outline_width: int = 1,
    overlay_outline_color: tuple[int, int, int] | None = None,
    # Cropping
    crop: CropSpec = None,
    # Appearance
    color_by: ColorScheme = "auto",
    palette: PaletteName | str = "standard",
    marker_shape: MarkerShape = "circle",
    marker_size: float = 4.0,
    line_width: float = 2.0,
    alpha: float = 1.0,
    show_nodes: bool = True,
    show_edges: bool = True,
    show_centroids: bool = True,
    centroid_marker_size: float = 5.0,
    scale: float = 1.0,
    # Motion trails
    show_trails: bool = False,
    trail_length: int = 10,
    trail_node: str | list[str] = "centroid",
    trail_width: float = 2.0,
    trail_alpha_fade: bool = True,
    trail_alpha: float = 1.0,
    trail_color: ColorSpec | None = None,
    # Background control
    background: Literal["video"] | ColorSpec = "video",
    # Callbacks
    pre_render_callback: Callable[[RenderContext], None] | None = None,
    post_render_callback: Callable[[RenderContext], None] | None = None,
    per_instance_callback: Callable[[InstanceContext], None] | None = None,
) -> np.ndarray:
    """Render single frame with pose and/or segmentation overlays.

    Args:
        source: LabeledFrame, Labels (with frame specifier), list of instances,
            or ``None``. When ``None``, ``image`` must be provided and only
            segmentation overlays are rendered (no poses).
        save_path: Output image path (PNG/JPEG). If None, only returns array.
        lf_ind: LabeledFrame index within Labels.labeled_frames (when source is Labels).
        video: Video object or video index (used with frame_idx when source is Labels).
        frame_idx: Video frame index (0-based, used with video when source is Labels).
        image: Override image array (H, W) or (H, W, C) uint8. Fetched from
            LabeledFrame if not provided.
        overlay: Annotation data to render on the image before poses. Accepts:

            - ``np.ndarray``: Integer label image ``(H, W)`` where 0 is
              background and positive values are object IDs.
            - ``SegmentationMask``, ``ROI``, or ``BoundingBox``: A single
              annotation object (treated like a one-element list).
            - ``list[SegmentationMask]``: Binary segmentation masks.
            - ``list[ROI]``: Vector geometries (polygons, points, etc.).
            - ``list[BoundingBox]``: Bounding boxes.

            Applied before pose rendering so poses draw on top.
        overlay_alpha: Opacity for the segmentation overlay (0.0 to 1.0).
        overlay_palette: Color palette for segmentation overlay.
        overlay_outline: Whether to draw outlines around segmented regions.
        overlay_outline_width: Outline width in pixels.
        overlay_outline_color: RGB outline color, or ``None`` for auto-darkened.
        crop: Crop specification. Bounds are (x1, y1, x2, y2) where (x1, y1) is
            the top-left corner and (x2, y2) is the bottom-right (exclusive).
            Origin (0, 0) is at the image top-left. Can be:

            - **Pixel coordinates** (int tuple): ``(100, 100, 300, 300)`` crops
              from pixel (100, 100) to (300, 300).
            - **Normalized coordinates** (float tuple in [0.0, 1.0]):
              ``(0.25, 0.25, 0.75, 0.75)`` crops the center 50% of the frame.
              Detection is type-based: all values must be ``float`` and in range.
            - ``None``: No cropping (default).
        color_by: Color scheme - 'track', 'instance', 'node', 'identity',
            'category', or 'auto'.
        palette: Color palette name.
        marker_shape: Node marker shape.
        marker_size: Node marker radius in pixels.
        line_width: Edge line width in pixels.
        alpha: Global transparency (0.0-1.0).
        show_nodes: Whether to draw node markers.
        show_edges: Whether to draw skeleton edges.
        show_centroids: Whether to draw centroid markers from
            ``Labels.centroids``. Centroids are colored by track.
        centroid_marker_size: Radius of centroid markers in pixels.
        scale: Output scale factor. Applied after cropping.
        show_trails: Whether to draw motion trails tracing node or centroid
            positions over past frames. Only takes effect when ``source`` is a
            ``Labels`` object (trails need temporal context); ignored otherwise.
        trail_length: Number of past frames behind the current frame to include
            in each trail.
        trail_node: Which point to trail. One of ``"centroid"`` (default), a
            node name, or a list of node names (one trail per node).
        trail_width: Trail line width in pixels.
        trail_alpha_fade: If ``True``, fade trails from faint (oldest) to opaque
            (newest).
        trail_alpha: Global opacity multiplier for trails (0.0 to 1.0). Combines
            with ``trail_alpha_fade``.
        trail_color: Uniform color for all trails. If ``None`` (default), trails
            are colored to match the poses (by track or instance). Accepts any
            color spec (RGB tuple, named color, hex, or palette index).
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

        Overlay a segmentation mask on a raw image (no poses):

        >>> img = sio.render_image(image=frame, overlay=label_mask)

        Overlay segmentation on a labeled frame (poses draw on top):

        >>> img = sio.render_image(lf, overlay=label_mask, overlay_alpha=0.4)
    """
    import skia  # noqa: F401

    from sleap_io.model.instance import Instance, PredictedInstance
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.labels import Labels

    # Handle background parameter
    use_video = background == "video"
    background_color: tuple[int, int, int] | None = None
    if not use_video:
        background_color = resolve_color(background)

    # Resolve source to LabeledFrame or instances
    if isinstance(source, Labels):
        lf = None
        has_centroids = show_centroids and bool(source.centroids)
        if video is not None and frame_idx is not None:
            # Render by video + frame_idx
            target_video = source.videos[video] if isinstance(video, int) else video
            lf_list = source.find(target_video, frame_idx)
            if lf_list:
                lf = lf_list[0]
            elif not has_centroids:
                raise ValueError(
                    f"No labeled frame found for video {target_video} "
                    f"at frame {frame_idx}"
                )
        elif lf_ind is not None:
            # Render by labeled frame index
            lf = source.labeled_frames[lf_ind]
        elif source.labeled_frames:
            # Default to first labeled frame
            lf = source.labeled_frames[0]
        elif not has_centroids:
            raise ValueError("No labeled frames to render")

        if lf is not None:
            instances = list(lf.instances)
            if instances:
                skeleton = instances[0].skeleton
            elif source.skeletons:
                # No instances but skeletons exist: use the first skeleton for
                # pose-rendering metadata (e.g. trail node resolution).
                skeleton = source.skeletons[0]
            else:
                # No instances and no skeletons — segmentation/overlay-only
                # frame (e.g. bottom-up mask tracking). Fall through with empty
                # pose state so the background and any overlay (masks, label
                # images, ROIs, bboxes) still render instead of crashing.
                # Mirrors the LabeledFrame branch below.
                skeleton = None
            edge_inds = skeleton.edge_inds if skeleton is not None else []
            node_names = (
                [n.name for n in skeleton.nodes] if skeleton is not None else []
            )
            fidx_for_callback = lf.frame_idx
        else:
            # Centroid-only / spatial-only mode: no labeled frames.
            instances = []
            skeleton = None
            edge_inds = []
            node_names = []
            fidx_for_callback = frame_idx if frame_idx is not None else 0

        # Get track info using O(1) lookup map
        n_tracks = len(source.tracks)
        has_tracks = n_tracks > 0
        img_track_idx_map = {id(t): i for i, t in enumerate(source.tracks)}
        track_indices = []
        for inst in instances:
            tidx = img_track_idx_map.get(id(inst.track)) if inst.track else None
            track_indices.append(tidx if tidx is not None else 0)

        # Convert instances to point arrays (needed for both image size and rendering)
        instances_points = [inst.numpy() for inst in instances]

        # Get image if not provided
        if image is None:
            video_obj = (
                lf.video
                if lf is not None
                else (source.videos[0] if source.videos else None)
            )
            if background_color is not None:
                # Solid color background - skip video loading entirely
                if (
                    video_obj is not None
                    and hasattr(video_obj, "shape")
                    and video_obj.shape is not None
                ):
                    h, w = video_obj.shape[1:3]
                else:
                    # Estimate from points or default
                    if instances_points:
                        h, w = _estimate_frame_size(instances_points)
                    else:
                        h, w = 512, 512
                image = _create_blank_frame(h, w, background_color)[:, :, :3]
            else:
                # Load video frame
                try:
                    if lf is not None:
                        image = lf.image
                    elif video_obj is not None and frame_idx is not None:
                        image = video_obj[frame_idx]
                    else:
                        image = None
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
        if instances:
            skeleton = instances[0].skeleton
            edge_inds = skeleton.edge_inds
            node_names = [n.name for n in skeleton.nodes]
        else:
            # No instances — segmentation/overlay-only mode. Fall through with
            # empty pose-rendering state so the video frame and any overlay
            # (masks, label images, ROIs, bboxes) still render. Mirrors the
            # centroid-only path in the `Labels` branch above.
            edge_inds = []
            node_names = []
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

    elif source is None:
        # No poses — overlay-only or image-only mode
        if image is None:
            raise ValueError("image parameter required when source is None")
        instances = []
        instances_points = []
        edge_inds = []
        node_names = []
        fidx_for_callback = 0
        track_indices = None
        n_tracks = 0
        has_tracks = False

    else:
        raise TypeError(
            f"source must be Labels, LabeledFrame, list of instances, "
            f"or None, got {type(source)}"
        )

    # Auto-use the frame's segmentation masks as overlay when no explicit
    # overlay is given and the frame carries masks. Mirrors render_video's
    # auto-overlay behavior so a segmentation-only frame still draws its masks.
    # Only masks are auto-resolved here (not label_images): label_image overlays
    # are a Labels/render_video-level concept and _apply_overlay does not accept a
    # list[LabelImage] in the single-frame path. An explicit overlay always wins.
    if (
        overlay is None
        and isinstance(source, (Labels, LabeledFrame))
        and lf is not None
        and lf.masks
    ):
        overlay = list(lf.masks)

    # Determine color scheme up front so track-colored overlays (masks/ROIs/
    # bboxes) can match the pose/centroid/trail track colors. Consumed below by
    # both the overlay block and the pose render_frame call.
    resolved_scheme = determine_color_scheme(
        has_tracks=has_tracks,
        is_single_image=True,
        scheme=color_by,
    )

    # Apply cropping if specified
    render_image_data = image
    render_points = instances_points
    crop_offset: tuple[float, float] = (0.0, 0.0)
    if crop is not None:
        h, w = image.shape[:2]
        # Resolve normalized or pixel coordinates
        crop_bounds = _resolve_crop(crop, (h, w))
        crop_offset = (float(crop_bounds[0]), float(crop_bounds[1]))

        render_image_data, render_points, _ = _apply_crop(
            image, instances_points, crop_bounds
        )

    # Apply annotation overlay before pose rendering
    if overlay is not None:
        # Ensure image is RGB for color blending
        if render_image_data.ndim == 2:
            render_image_data = np.stack([render_image_data] * 3, axis=-1)
        elif render_image_data.ndim == 3 and render_image_data.shape[2] == 1:
            render_image_data = np.repeat(render_image_data, 3, axis=2)
        # Crop overlay to match cropped image region
        render_overlay = overlay
        if crop is not None and isinstance(overlay, np.ndarray):
            x1, y1, x2, y2 = crop_bounds
            oh, ow = overlay.shape[:2]
            render_overlay = overlay[max(0, y1) : min(oh, y2), max(0, x1) : min(ow, x2)]
        # Color overlay elements (masks/ROIs/bboxes) by track identity when
        # color_by resolves to "track", matching poses/centroids/trails (same
        # `palette`). Otherwise fall through to positional `overlay_palette`
        # coloring. Gated on a Labels source with tracks (only that branch builds
        # the track index map; `has_tracks` mirrors render_video so track-less
        # labels stay positional). Untracked elements fall back to the first
        # color.
        overlay_colors = None
        if (
            resolved_scheme == "track"
            and isinstance(source, Labels)
            and has_tracks
            and isinstance(render_overlay, list)
            and render_overlay
            and not _is_label_image(render_overlay[0])
        ):
            ov_pal = get_palette(palette, max(len(source.tracks), 1))
            overlay_colors = []
            for el in render_overlay:
                t = getattr(el, "track", None)
                tidx = img_track_idx_map.get(id(t)) if t is not None else None
                overlay_colors.append(
                    ov_pal[tidx % len(ov_pal)] if tidx is not None else ov_pal[0]
                )
        _apply_overlay(
            render_image_data,
            render_overlay,
            alpha=overlay_alpha,
            palette=overlay_palette,
            outline=overlay_outline,
            outline_width=overlay_outline_width,
            outline_color=overlay_outline_color,
            colors=overlay_colors,
        )

    # Draw motion trails behind the poses and centroids. Trails need temporal
    # context, so they are only drawn when the source is a Labels object. They
    # are drawn even when the current frame has no instances, since past frames
    # may still contribute (matching render_video).
    if (
        show_trails
        and isinstance(source, Labels)
        and trail_length > 0
        and skeleton is not None
    ):
        from sleap_io.rendering.overlays import draw_trails as _draw_trails

        trail_targets = _resolve_trail_node(trail_node, skeleton)
        frame_idx_to_lf = {lframe.frame_idx: lframe for lframe in source.find(lf.video)}
        n_trail_colors = _n_trail_palette_colors(
            has_tracks, n_tracks, frame_idx_to_lf.values()
        )
        trail_palette = get_palette(palette, n_trail_colors)
        trails, trail_colors = _compute_trails(
            fidx=fidx_for_callback,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=trail_length,
            trail_targets=trail_targets,
            track_idx_map=img_track_idx_map,
            palette_colors=trail_palette,
            has_tracks=has_tracks,
        )
        if trails:
            # A uniform trail_color overrides the per-track palette colors.
            trail_draw_kwargs: dict = {}
            if trail_color is not None:
                trail_draw_kwargs["color"] = resolve_color(trail_color)
            else:
                trail_draw_kwargs["colors"] = trail_colors
            # trail_width is NOT pre-scaled: the trail is drawn here, then the
            # whole image is upscaled once by `scale` inside render_frame, so
            # the final width matches pose edges (line_width * scale).
            render_image_data = _draw_trails(
                render_image_data,
                trails,
                line_width=trail_width,
                alpha_fade=trail_alpha_fade,
                alpha=trail_alpha,
                offset=crop_offset,
                **trail_draw_kwargs,
            )

    # Draw centroids on the image.
    if show_centroids and isinstance(source, Labels) and source.centroids:
        from sleap_io.rendering.overlays import draw_centroids as _draw_centroids

        render_fidx = fidx_for_callback
        if lf is not None:
            # Scope to the rendered frame's own video. A centroid on a
            # *different* video that happens to share this frame index must not
            # bleed in, so read this frame's centroids directly (mirroring
            # render_video's per-frame `lf.centroids`).
            frame_centroids = list(lf.centroids)
        else:
            # No labeled frame resolved — only reachable via an explicit
            # video+frame_idx that matched no frame, so `video` is a concrete
            # spec. Scope centroids to it so a *different* video's centroid that
            # shares this frame index can't bleed in (mirrors render_video's
            # get_centroids(video=target_video)).
            target_video = source.videos[video] if isinstance(video, int) else video
            frame_centroids = source.get_centroids(
                video=target_video, frame_idx=render_fidx
            )
        if frame_centroids:
            if render_image_data.ndim == 2:
                render_image_data = np.stack([render_image_data] * 3, axis=-1)
            centroid_pal = get_palette(palette, max(len(source.tracks), 1))
            c_colors = []
            for c in frame_centroids:
                tidx = img_track_idx_map.get(id(c.track)) if c.track else None
                if tidx is not None:
                    c_colors.append(centroid_pal[tidx % len(centroid_pal)])
                else:
                    c_colors.append(centroid_pal[0] if centroid_pal else (0, 255, 0))
            # centroid_marker_size is NOT pre-scaled: the centroids are drawn
            # here, then the whole image is upscaled once by `scale` inside
            # render_frame, so the final radius matches pose nodes
            # (marker_size * scale).
            render_image_data = _draw_centroids(
                render_image_data,
                frame_centroids,
                colors=c_colors,
                marker_size=centroid_marker_size,
                offset=crop_offset,
            )

    # Short-circuit: overlay-only mode (no poses to render)
    if source is None:
        # Scale if needed
        if scale != 1.0:
            render_image_data = _scale_frame(
                _prepare_frame_rgba(render_image_data), scale
            )[:, :, :3]

        if save_path is not None:
            _save_image(render_image_data, save_path)

        return render_image_data

    # Build instance metadata for callbacks
    instance_metadata = []
    for inst in instances:
        meta = {}
        if hasattr(inst, "track") and inst.track is not None:
            meta["track_name"] = inst.track.name
        if hasattr(inst, "score"):
            meta["confidence"] = inst.score
        instance_metadata.append(meta)

    # Compute per-instance identity indices when the resolved scheme is
    # "identity", mirroring the track-index plumbing.
    identity_indices = None
    n_identities = 0
    if resolved_scheme == "identity":
        catalog = source.identities if isinstance(source, Labels) else None
        identity_indices, n_identities = _compute_identity_coloring(instances, catalog)

    # Compute per-instance category indices when the resolved scheme is
    # "category", mirroring the identity plumbing.
    category_indices = None
    n_categories = 0
    if resolved_scheme == "category":
        catalog = source.categories if isinstance(source, Labels) else None
        category_indices, n_categories = _compute_category_coloring(instances, catalog)

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
        identity_indices=identity_indices,
        n_identities=n_identities,
        category_indices=category_indices,
        n_categories=n_categories,
        pre_render_callback=pre_render_callback,
        post_render_callback=post_render_callback,
        per_instance_callback=per_instance_callback,
        frame_idx=fidx_for_callback,
        instance_metadata=instance_metadata,
        crop_offset=crop_offset,
    )

    # Save if save_path provided
    if save_path is not None:
        _save_image(rendered, save_path)

    return rendered


def render_video(
    source: "Labels | list[LabeledFrame]",
    save_path: str | Path | None = None,
    *,
    # Video selection
    video: "Video | int | None" = None,
    # Frame selection
    frame_inds: list[int] | None = None,
    start: int | None = None,
    end: int | None = None,
    include_unlabeled: bool | None = None,
    # Annotation overlay
    overlay: (
        "np.ndarray"
        " | list[LabelImage] | list[SegmentationMask] | list[ROI] | list[BoundingBox]"
        " | Callable[[int], np.ndarray] | None"
    ) = None,
    overlay_alpha: float = 0.3,
    overlay_palette: PaletteName | str = "distinct",
    overlay_outline: bool = False,
    overlay_outline_width: int = 1,
    overlay_outline_color: tuple[int, int, int] | None = None,
    # Cropping
    crop: CropSpec = None,
    # Quality/scale
    preset: Literal["preview", "draft", "final"] | None = None,
    scale: float = 1.0,
    # Appearance
    color_by: ColorScheme = "auto",
    palette: PaletteName | str = "standard",
    marker_shape: MarkerShape = "circle",
    marker_size: float = 4.0,
    line_width: float = 2.0,
    alpha: float = 1.0,
    show_nodes: bool = True,
    show_edges: bool = True,
    show_centroids: bool = True,
    centroid_marker_size: float = 5.0,
    # Motion trails
    show_trails: bool = False,
    trail_length: int = 10,
    trail_node: str | list[str] = "centroid",
    trail_width: float = 2.0,
    trail_alpha_fade: bool = True,
    trail_alpha: float = 1.0,
    trail_color: ColorSpec | None = None,
    # Video encoding
    fps: float | None = None,
    codec: str = "libx264",
    crf: int = 25,
    x264_preset: str = "superfast",
    # Background control
    background: Literal["video"] | ColorSpec = "video",
    # Callbacks
    pre_render_callback: Callable[[RenderContext], None] | None = None,
    post_render_callback: Callable[[RenderContext], None] | None = None,
    per_instance_callback: Callable[[InstanceContext], None] | None = None,
    # Progress
    progress_callback: Callable[[int, int], bool] | None = None,
    show_progress: bool = True,
) -> "Video | list[np.ndarray]":
    """Render video with pose overlays.

    Args:
        source: Labels object or list of LabeledFrames to render.
        save_path: Output video path. If None, returns list of rendered arrays.
        video: Video to render from (default: first video in Labels).
        frame_inds: Specific frame indices to render.
        start: Start frame index (inclusive).
        end: End frame index (exclusive).
        include_unlabeled: If True, render all frames in range even if they have
            no LabeledFrame (just shows video frame without poses). Default None,
            which resolves to False unless auto-overlay detection enables it (when
            label_images exist for the target video and no explicit overlay is given).
        overlay: Per-frame annotation overlay. Accepts:

            - ``np.ndarray``: 3-D array ``(T, H, W)`` of integer label images
              indexed by frame number, or 2-D ``(H, W)`` for a static overlay.
            - ``list[SegmentationMask | ROI | BoundingBox]``: Indexed by
              position — the item at list index ``i`` is applied to frame
              ``i``. One overlay per frame.
            - ``Callable[[int], np.ndarray]``: Called with the frame index,
              returns an ``(H, W)`` label image for that frame.
        overlay_alpha: Opacity for the annotation overlay (0.0 to 1.0).
        overlay_palette: Color palette for overlay coloring.
        overlay_outline: Draw outlines around segmented regions (label images).
        overlay_outline_width: Outline width in pixels.
        overlay_outline_color: RGB outline color, or ``None`` for auto-darkened.
        crop: Static crop applied uniformly to all frames. Bounds are
            (x1, y1, x2, y2) where (x1, y1) is the top-left corner and (x2, y2)
            is the bottom-right (exclusive). Supports:

            - **Pixel coordinates** (int tuple): ``(100, 100, 300, 300)``
            - **Normalized coordinates** (float tuple in [0.0, 1.0]):
              ``(0.25, 0.25, 0.75, 0.75)`` crops the center 50%.
            - ``None``: No cropping (default).
        preset: Quality preset ('preview'=0.25x, 'draft'=0.5x, 'final'=1.0x).
        scale: Scale factor (overrides preset if both provided).
        color_by: Color scheme - 'track', 'instance', 'node', 'identity',
            'category', or 'auto'.
        palette: Color palette name.
        marker_shape: Node marker shape.
        marker_size: Node marker radius in pixels.
        line_width: Edge line width in pixels.
        alpha: Global transparency (0.0-1.0).
        show_nodes: Whether to draw node markers.
        show_edges: Whether to draw skeleton edges.
        show_centroids: Whether to draw centroid markers from
            ``Labels.centroids``. Centroids are colored by track.
        centroid_marker_size: Radius of centroid markers in pixels.
        show_trails: Whether to draw motion trails tracing node or centroid
            positions over past frames.
        trail_length: Number of past frames behind each frame to include in the
            trail.
        trail_node: Which point to trail. One of ``"centroid"`` (default), a
            node name, or a list of node names (one trail per node).
        trail_width: Trail line width in pixels.
        trail_alpha_fade: If ``True``, fade trails from faint (oldest) to opaque
            (newest).
        trail_alpha: Global opacity multiplier for trails (0.0 to 1.0). Combines
            with ``trail_alpha_fade``.
        trail_color: Uniform color for all trails. If ``None`` (default), trails
            are colored to match the poses (by track or instance). Accepts any
            color spec (RGB tuple, named color, hex, or palette index).
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
    background_color: tuple[int, int, int] | None = None
    if not use_video:
        background_color = resolve_color(background)

    # Handle preset
    if preset is not None and preset in PRESETS:
        scale = PRESETS[preset]["scale"]

    # Whether this video has centroids to render (populated below).
    _has_video_centroids: bool = False

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

        # Sort by frame index
        labeled_frames = sorted(labeled_frames, key=lambda lf: lf.frame_idx)

        # Check for spatial annotations (label_images, masks, bboxes, rois,
        # centroids) that can be rendered even without labeled frames (poses).
        has_spatial = bool(
            labels.get_label_images(video=target_video)
            or labels.get_masks(video=target_video)
            or labels.get_bboxes(video=target_video)
            or labels.get_rois(video=target_video)
            or labels.get_centroids(video=target_video)
        )

        if not labeled_frames and not has_spatial:
            raise ValueError(f"No labeled frames found for video {target_video}")

        # Get skeleton info (not required when rendering only spatial
        # annotations)
        skeleton = labels.skeletons[0] if labels.skeletons else None
        if skeleton is None and labeled_frames:
            for lf in labeled_frames:
                for inst in lf.instances:
                    skeleton = inst.skeleton
                    break
                if skeleton:
                    break

        if skeleton is not None:
            edge_inds = skeleton.edge_inds
            node_names = [n.name for n in skeleton.nodes]
        else:
            # Only raise if frames have instances (which need a skeleton)
            has_instances = any(len(lf.instances) > 0 for lf in labeled_frames)
            if has_instances:
                raise ValueError("No skeleton found in labels")
            edge_inds = []
            node_names = []

        n_tracks = len(labels.tracks)
        has_tracks = n_tracks > 0

        # Auto-use label_images as overlay when no explicit overlay is
        # provided and the file has label images for this video.
        if overlay is None and labels.label_images:
            video_label_images = labels.get_label_images(video=target_video)
            if video_label_images:
                overlay = video_label_images
                if include_unlabeled is None:
                    include_unlabeled = True

        # Auto-use segmentation masks as overlay when no explicit overlay (and
        # no label images) resolved. Masks live on specific frames at arbitrary
        # frame indices, so resolve them per-frame via a callable rather than a
        # positional list. label_images take precedence (resolved above).
        if overlay is None and labels.masks:
            video_masks = labels.get_masks(video=target_video)
            if video_masks:
                _auto_labels = labels
                _auto_video = target_video

                def overlay(fidx: int) -> list["SegmentationMask"]:
                    """Resolve segmentation masks for a single frame index."""
                    return _auto_labels.get_masks(video=_auto_video, frame_idx=fidx)

                if include_unlabeled is None:
                    include_unlabeled = True

        # Check if centroids exist for this video (per-frame access via lf).
        if show_centroids and labels.centroids:
            _has_video_centroids = bool(labels.get_centroids(video=target_video))
            if _has_video_centroids and include_unlabeled is None:
                include_unlabeled = True

        # Resolve None to default after auto-overlay logic
        if include_unlabeled is None:
            include_unlabeled = False

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

    if not render_indices and isinstance(overlay, list) and overlay:
        # Derive frame indices from overlay list (use list indices as frame indices)
        render_indices = list(range(len(overlay)))

    if not render_indices and _has_video_centroids:
        # Derive frame indices from frames that have centroids
        render_indices = sorted(lf.frame_idx for lf in labeled_frames if lf.centroids)

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
    crop_bounds: tuple[int, int, int, int] | None = None
    crop_offset: tuple[float, float] = (0.0, 0.0)
    if crop is not None:
        if hasattr(target_video, "shape") and target_video.shape is not None:
            h, w = target_video.shape[1:3]
        else:
            # Fallback: try to get from first frame
            h, w = 480, 640  # reasonable default
        crop_bounds = _resolve_crop(crop, (h, w))
        crop_offset = (float(crop_bounds[0]), float(crop_bounds[1]))

    # Setup progress
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(render_indices, desc="Rendering", unit="frame")
        except ImportError:
            iterator = render_indices
    else:
        iterator = render_indices

    # Setup video writer for streaming output (memory optimization)
    # When save_path is provided, write frames directly instead of accumulating
    writer = None
    if save_path is not None:
        from sleap_io.io.video_writing import VideoWriter

        save_path_ = Path(save_path)
        save_path_.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(
            filename=save_path_,
            fps=fps,
            codec=codec,
            crf=crf,
            preset=x264_preset,
        )

    # Build centroid color palette (by track).
    _centroid_palette: list[tuple[int, int, int]] = []
    if _has_video_centroids and labels is not None:
        _centroid_palette = get_palette(palette, max(len(labels.tracks), 1))

    # Set up motion trails (drawn behind poses). Trails trace instances, and
    # every instance carries a skeleton, so a missing skeleton means there are
    # no instances to trail.
    _do_trails = show_trails and trail_length > 0 and skeleton is not None
    _trail_targets: list[int | None] = []
    _trail_palette: list[tuple[int, int, int]] = []
    _trail_pts_cache: dict[int, np.ndarray] = {}
    if _do_trails:
        _trail_targets = _resolve_trail_node(trail_node, skeleton)
        _trail_palette = get_palette(
            palette, _n_trail_palette_colors(has_tracks, n_tracks, labeled_frames)
        )
    # A uniform trail_color overrides the per-track palette colors.
    _trail_color_resolved = (
        resolve_color(trail_color) if trail_color is not None else None
    )

    # Pre-process overlay: determine type for per-frame dispatch
    _overlay_is_3d = (
        overlay is not None and isinstance(overlay, np.ndarray) and overlay.ndim == 3
    )
    _overlay_is_2d = (
        overlay is not None and isinstance(overlay, np.ndarray) and overlay.ndim == 2
    )
    _overlay_is_callable = callable(overlay) if overlay is not None else False
    _overlay_is_list = (
        overlay is not None and isinstance(overlay, list) and len(overlay) > 0
    )
    _overlay_is_label_image_list = _overlay_is_list and _is_label_image(overlay[0])

    def _get_frame_overlay(fidx: int):
        """Resolve overlay data for a single frame."""
        if overlay is None:
            return None
        if _overlay_is_3d:
            if fidx < overlay.shape[0]:
                return overlay[fidx]
            return None
        if _overlay_is_2d:
            return overlay
        if _overlay_is_callable:
            return overlay(fidx)
        if _overlay_is_label_image_list:
            # Match by list index (overlays ordered by frame sequence)
            if fidx < len(overlay):
                return overlay[fidx]
            return None
        if _overlay_is_list:
            # Match by list index
            if fidx < len(overlay):
                return [overlay[fidx]]
            return []
        return None

    def _apply_frame_overlay(image: np.ndarray, fidx: int) -> np.ndarray:
        """Resolve and apply overlay for a single frame.

        Returns the (possibly new) image array — the caller must use the
        returned value since grayscale-to-RGB conversion creates a new array.
        """
        frame_overlay = _get_frame_overlay(fidx)
        if frame_overlay is None:
            return image
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        # Crop overlay to match cropped image region
        cropped_overlay = frame_overlay
        if crop_bounds is not None:
            if isinstance(frame_overlay, np.ndarray):
                x1, y1, x2, y2 = crop_bounds
                oh, ow = frame_overlay.shape[:2]
                cropped_overlay = frame_overlay[
                    max(0, y1) : min(oh, y2), max(0, x1) : min(ow, x2)
                ]
            elif _is_label_image(frame_overlay):
                from sleap_io.model.label_image import PredictedLabelImage

                x1, y1, x2, y2 = crop_bounds
                oh, ow = frame_overlay.data.shape[:2]
                cropped_data = frame_overlay.data[
                    max(0, y1) : min(oh, y2), max(0, x1) : min(ow, x2)
                ]
                kwargs = dict(
                    data=cropped_data,
                    objects=frame_overlay.objects,
                )
                if isinstance(frame_overlay, PredictedLabelImage):
                    kwargs["score"] = frame_overlay.score
                    kwargs["score_map"] = frame_overlay.score_map
                cropped_overlay = type(frame_overlay)(**kwargs)
        # Color overlay elements (masks/ROIs/bboxes) by track identity when
        # color_by resolves to "track", matching poses/centroids/trails (same
        # `palette`). Otherwise fall through to positional `overlay_palette`.
        # Untracked elements fall back to the first palette color.
        overlay_colors = None
        if (
            resolved_scheme == "track"
            and _overlay_palette_by_track
            and isinstance(cropped_overlay, list)
            and cropped_overlay
            and not _is_label_image(cropped_overlay[0])
        ):
            overlay_colors = []
            for el in cropped_overlay:
                t = getattr(el, "track", None)
                tidx = _track_idx_map.get(id(t)) if t is not None else None
                overlay_colors.append(
                    _overlay_palette_by_track[tidx % len(_overlay_palette_by_track)]
                    if tidx is not None
                    else _overlay_palette_by_track[0]
                )
        _apply_overlay(
            image,
            cropped_overlay,
            alpha=overlay_alpha,
            palette=overlay_palette,
            outline=overlay_outline,
            outline_width=overlay_outline_width,
            outline_color=overlay_outline_color,
            colors=overlay_colors,
        )
        return image

    # Pre-build track index map for O(1) track color lookups.
    _track_idx_map: dict[int, int] = {}
    if labels is not None and has_tracks:
        _track_idx_map = {id(t): i for i, t in enumerate(labels.tracks)}

    # Identity catalog for stable identity coloring across frames (mirrors the
    # track index map). Only built when coloring by identity.
    _identity_catalog: list | None = None
    if resolved_scheme == "identity":
        _identity_catalog = list(labels.identities) if labels is not None else []

    # Category catalog for stable category coloring across frames (mirrors the
    # identity index map). Only built when coloring by category.
    _category_catalog: list | None = None
    if resolved_scheme == "category":
        _category_catalog = list(labels.categories) if labels is not None else []

    # Track-keyed palette for overlay (mask/ROI/bbox) coloring under
    # color_by="track", matching centroids/poses/trails (same `palette`).
    _overlay_palette_by_track: list[tuple[int, int, int]] = []
    if labels is not None and has_tracks:
        _overlay_palette_by_track = get_palette(palette, max(len(labels.tracks), 1))

    def _draw_frame_centroids(
        image: np.ndarray, fidx: int, crop_off: tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray:
        """Draw centroids for a single frame onto the image."""
        if not _has_video_centroids:
            return image

        # Use frame-level centroid access instead of linear scan.
        lf = frame_idx_to_lf.get(fidx)
        frame_centroids = lf.centroids if lf is not None else []
        if not frame_centroids:
            return image

        from sleap_io.rendering.overlays import draw_centroids

        # Ensure RGB.
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Assign colors by track using pre-built index map.
        centroid_colors = []
        for c in frame_centroids:
            if c.track is not None and id(c.track) in _track_idx_map:
                tidx = _track_idx_map[id(c.track)]
                centroid_colors.append(_centroid_palette[tidx % len(_centroid_palette)])
            else:
                centroid_colors.append(
                    _centroid_palette[0] if _centroid_palette else (0, 255, 0)
                )

        # centroid_marker_size is NOT pre-scaled: the centroids are drawn here,
        # then the whole image is upscaled once by `scale` inside render_frame,
        # so the final radius matches pose nodes (marker_size * scale).
        draw_centroids(
            image,
            frame_centroids,
            colors=centroid_colors,
            marker_size=centroid_marker_size,
            offset=crop_off,
        )
        return image

    def _draw_frame_trails(
        image: np.ndarray, fidx: int, crop_off: tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray:
        """Draw motion trails for a single frame onto the image."""
        if not _do_trails:
            return image

        trails, trail_colors = _compute_trails(
            fidx=fidx,
            frame_idx_to_lf=frame_idx_to_lf,
            trail_length=trail_length,
            trail_targets=_trail_targets,
            track_idx_map=_track_idx_map,
            palette_colors=_trail_palette,
            has_tracks=has_tracks,
            pts_cache=_trail_pts_cache,
        )
        if not trails:
            return image

        from sleap_io.rendering.overlays import draw_trails

        # A uniform trail_color overrides the per-track palette colors.
        if _trail_color_resolved is not None:
            color_kwargs: dict = {"color": _trail_color_resolved}
        else:
            color_kwargs = {"colors": trail_colors}

        # trail_width is NOT pre-scaled: the trail is drawn here, then the whole
        # image is upscaled once by `scale` inside render_frame, so the final
        # width matches pose edges (line_width * scale).
        return draw_trails(
            image,
            trails,
            line_width=trail_width,
            alpha_fade=trail_alpha_fade,
            alpha=trail_alpha,
            offset=crop_off,
            **color_kwargs,
        )

    # Only accumulate frames if returning as list (no save_path)
    rendered_frames: list[np.ndarray] = []
    total_frames = len(render_indices)

    try:
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
                    if (
                        hasattr(target_video, "shape")
                        and target_video.shape is not None
                    ):
                        h, w = target_video.shape[1:3]
                    else:
                        # No video metadata and no points - use minimum default
                        h, w = 64, 64
                    image = _create_blank_frame(h, w, background_color)[:, :, :3]
                else:
                    try:
                        image = target_video[fidx]
                    except Exception:
                        image = None
                    if image is None:
                        raise ValueError(
                            f"Video unavailable at frame {fidx}. "
                            "Specify a background color to render without video."
                        )

                # Ensure RGB for overlay compositing
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3 and image.shape[2] == 1:
                    image = np.concatenate([image] * 3, axis=-1)

                # Apply cropping if specified
                render_image_data = image
                if crop_bounds is not None:
                    render_image_data, _, _ = _apply_crop(image, [], crop_bounds)

                # Apply overlay
                render_image_data = _apply_frame_overlay(render_image_data, fidx)

                # Draw centroids
                crop_off = (
                    (float(crop_bounds[0]), float(crop_bounds[1]))
                    if crop_bounds is not None
                    else (0.0, 0.0)
                )
                render_image_data = _draw_frame_centroids(
                    render_image_data, fidx, crop_off
                )

                # Draw motion trails
                render_image_data = _draw_frame_trails(
                    render_image_data, fidx, crop_off
                )

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
                    n_identities=(
                        len(_identity_catalog) if _identity_catalog is not None else 0
                    ),
                    n_categories=(
                        len(_category_catalog) if _category_catalog is not None else 0
                    ),
                    pre_render_callback=pre_render_callback,
                    post_render_callback=post_render_callback,
                    per_instance_callback=None,
                    frame_idx=fidx,
                    instance_metadata=[],
                    crop_offset=crop_offset,
                )

                # Stream to file or accumulate for return
                if writer is not None:
                    writer(rendered)
                else:
                    rendered_frames.append(rendered)
                continue

            instances = list(lf.instances)
            instances_points = [inst.numpy() for inst in instances]

            # Get track indices using pre-built map
            track_indices = None
            if labels is not None and has_tracks:
                track_indices = []
                for inst in instances:
                    tidx = _track_idx_map.get(id(inst.track)) if inst.track else None
                    track_indices.append(tidx if tidx is not None else 0)

            # Get identity indices when coloring by identity
            identity_indices = None
            n_identities = 0
            if resolved_scheme == "identity":
                identity_indices, n_identities = _compute_identity_coloring(
                    instances, _identity_catalog
                )

            # Get category indices when coloring by category
            category_indices = None
            n_categories = 0
            if resolved_scheme == "category":
                category_indices, n_categories = _compute_category_coloring(
                    instances, _category_catalog
                )

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
                except Exception:
                    image = None
                if image is None:
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

            # Apply overlay
            render_image_data = _apply_frame_overlay(render_image_data, fidx)

            # Draw centroids
            crop_off = (
                (float(crop_bounds[0]), float(crop_bounds[1]))
                if crop_bounds is not None
                else (0.0, 0.0)
            )
            render_image_data = _draw_frame_centroids(render_image_data, fidx, crop_off)

            # Draw motion trails
            render_image_data = _draw_frame_trails(render_image_data, fidx, crop_off)

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
                identity_indices=identity_indices,
                n_identities=n_identities,
                category_indices=category_indices,
                n_categories=n_categories,
                pre_render_callback=pre_render_callback,
                post_render_callback=post_render_callback,
                per_instance_callback=per_instance_callback,
                frame_idx=fidx,
                instance_metadata=instance_metadata,
                crop_offset=crop_offset,
            )

            # Stream to file or accumulate for return
            if writer is not None:
                writer(rendered)
            else:
                rendered_frames.append(rendered)

    finally:
        # Ensure writer is closed even if an exception occurs
        if writer is not None:
            writer.close()

    # Return Video object or frame list
    if save_path is not None:
        return VideoModel.from_filename(str(save_path_))

    return rendered_frames
