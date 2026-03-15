"""Overlay drawing functions for ROIs, segmentation masks, and bounding boxes.

These functions draw annotations directly onto numpy image arrays using
skia-python for geometry rendering and numpy for mask blending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import skia

    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.roi import ROI


def draw_rois(
    image: np.ndarray,
    rois: list["ROI"],
    color: tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
    fill_alpha: float = 0.0,
) -> np.ndarray:
    """Draw ROI geometries on an image.

    Draws the boundary of each ROI's geometry using skia-python. Supports
    ``Polygon``, ``MultiPolygon``, ``Point``, ``MultiPoint``, ``LineString``,
    ``MultiLineString``, and ``GeometryCollection`` geometries.

    Args:
        image: Image array of shape (H, W, 3) uint8. Modified in-place and
            returned.
        rois: List of ROI objects to draw.
        color: RGB color tuple for the ROI outlines.
        line_width: Width of the outline in pixels.
        fill_alpha: If > 0, fill the ROI interior with this opacity (0.0 to
            1.0).

    Returns:
        The modified image array.
    """
    if not rois:
        return image

    import skia

    # Pad to RGBA for skia surface
    frame_rgba = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
    surface = skia.Surface(frame_rgba, colorType=skia.kRGBA_8888_ColorType)
    canvas = surface.getCanvas()

    # Stroke paint for outlines
    stroke_paint = skia.Paint(
        Color=skia.Color(*color),
        AntiAlias=False,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=float(line_width),
        StrokeCap=skia.Paint.kSquare_Cap,
    )

    # Fill paint (only created if needed)
    fill_paint = None
    if fill_alpha > 0:
        fill_paint = skia.Paint(
            Color=skia.Color4f(
                color[0] / 255.0,
                color[1] / 255.0,
                color[2] / 255.0,
                fill_alpha,
            ).toColor(),
            AntiAlias=False,
            Style=skia.Paint.kFill_Style,
        )

    for roi in rois:
        _draw_geometry(canvas, roi.geometry, stroke_paint, fill_paint)

    # Copy RGB channels back to the input image
    image[:] = frame_rgba[:, :, :3]

    return image


def draw_masks(
    image: np.ndarray,
    masks: list["SegmentationMask"],
    color: tuple[int, int, int] = (255, 0, 0),
    colors: list[tuple[int, int, int]] | None = None,
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw segmentation masks as colored overlays on an image.

    Args:
        image: Image array of shape (H, W, 3) uint8. Modified in-place and
            returned.
        masks: List of SegmentationMask objects to draw.
        color: RGB color tuple for the mask overlay. Used when ``colors`` is
            ``None``.
        colors: Per-mask RGB color tuples. If provided, must have the same
            length as ``masks``. Overrides ``color``.
        alpha: Opacity of the mask overlay (0.0 to 1.0).

    Returns:
        The modified image array.
    """
    for i, mask in enumerate(masks):
        mask_color = colors[i] if colors is not None else color
        mask_data = mask.data
        h, w = mask_data.shape
        img_h, img_w = image.shape[:2]

        # Clip to image bounds
        draw_h = min(h, img_h)
        draw_w = min(w, img_w)

        region = image[:draw_h, :draw_w]
        mask_region = mask_data[:draw_h, :draw_w]

        # Blend color into masked pixels
        overlay = np.array(mask_color, dtype=np.float32)
        region[mask_region] = (
            region[mask_region] * (1 - alpha) + overlay * alpha
        ).astype(np.uint8)

    return image


def draw_label_image(
    image: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.3,
    palette: str = "distinct",
    outline: bool = False,
    outline_width: int = 1,
    outline_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw an integer label image as a colored overlay on an image.

    This is an efficient rendering path for segmentation masks stored as
    integer label images (e.g., from instance or panoptic segmentation) where
    each pixel value represents a different object ID (0 = background).

    Args:
        image: Image array of shape ``(H, W, 3)`` uint8. Modified in-place and
            returned.
        labels: Integer label array of shape ``(H, W)`` where 0 is background
            and positive values are object IDs.
        alpha: Opacity of the mask overlay (0.0 to 1.0).
        palette: Color palette name for assigning colors to label IDs. See
            :func:`~sleap_io.rendering.colors.get_palette` for options.
        outline: If ``True``, draw outlines around each labeled region using
            skia-python.
        outline_width: Width of the outline in pixels (only used if
            ``outline=True``).
        outline_color: RGB color for outlines. If ``None``, uses a darkened
            version of each region's fill color.

    Returns:
        The modified image array.
    """
    from sleap_io.rendering.colors import get_palette

    # Get unique non-background labels
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return image

    # Build color lookup table (LUT): label_id -> RGB
    max_id = int(unique_ids.max())
    palette_colors = get_palette(palette, max_id + 1)

    # Create a LUT array: shape (max_id + 1, 3)
    lut = np.zeros((max_id + 1, 3), dtype=np.float32)
    for label_id in unique_ids:
        lut[label_id] = palette_colors[int(label_id) % len(palette_colors)]

    # Clip labels to image size
    img_h, img_w = image.shape[:2]
    lab_h, lab_w = labels.shape[:2]
    draw_h = min(lab_h, img_h)
    draw_w = min(lab_w, img_w)

    region = image[:draw_h, :draw_w]
    label_region = labels[:draw_h, :draw_w]

    # Vectorized blending: apply colored overlay where labels > 0
    fg_mask = label_region > 0
    if np.any(fg_mask):
        # Clamp label values for LUT indexing
        safe_labels = np.clip(label_region, 0, max_id)
        overlay_colors = lut[safe_labels]  # (H, W, 3)

        region_float = region.astype(np.float32)
        region_float[fg_mask] = (
            region_float[fg_mask] * (1 - alpha) + overlay_colors[fg_mask] * alpha
        )
        region[:] = region_float.astype(np.uint8)

    # Draw outlines if requested
    if outline:
        _draw_label_outlines(
            image, labels, draw_h, draw_w, outline_width, outline_color, lut
        )

    return image


def _draw_label_outlines(
    image: np.ndarray,
    labels: np.ndarray,
    draw_h: int,
    draw_w: int,
    outline_width: int,
    outline_color: tuple[int, int, int] | None,
    lut: np.ndarray,
) -> None:
    """Draw outlines around labeled regions using numpy edge detection.

    Detects boundary pixels where a foreground label differs from its neighbor
    and paints them directly. For ``outline_width > 1``, the edge mask is
    dilated with a square structuring element.

    Args:
        image: Image array of shape ``(H, W, 3)`` uint8.
        labels: Integer label array of shape ``(H, W)``.
        draw_h: Height to draw within.
        draw_w: Width to draw within.
        outline_width: Outline stroke width in pixels.
        outline_color: RGB color for outlines, or ``None`` to use a darkened
            version of each region's fill color.
        lut: Color lookup table mapping ``label_id`` to ``(R, G, B)`` as
            float32.
    """
    label_region = labels[:draw_h, :draw_w]
    region = image[:draw_h, :draw_w]

    # Find boundary pixels using shifted comparisons
    edges = np.zeros((draw_h, draw_w), dtype=bool)
    edges[:, :-1] |= label_region[:, :-1] != label_region[:, 1:]
    edges[:-1, :] |= label_region[:-1, :] != label_region[1:, :]
    edges[:, 1:] |= label_region[:, :-1] != label_region[:, 1:]
    edges[1:, :] |= label_region[:-1, :] != label_region[1:, :]

    # Only keep edges on foreground
    edges &= label_region > 0

    # Dilate edges for thicker outlines
    if outline_width > 1:
        dilated = np.zeros_like(edges)
        pad = outline_width // 2
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                shifted = np.zeros_like(edges)
                sy = max(0, dy)
                ey = draw_h + min(0, dy)
                sx = max(0, dx)
                ex = draw_w + min(0, dx)
                oy = max(0, -dy)
                ox = max(0, -dx)
                shifted[sy:ey, sx:ex] = edges[oy : oy + (ey - sy), ox : ox + (ex - sx)]
                dilated |= shifted
        edges = dilated & (label_region > 0)

    if outline_color is not None:
        # Uniform outline color
        region[edges] = np.array(outline_color, dtype=np.uint8)
    else:
        # Per-label darkened color
        max_id = lut.shape[0] - 1
        safe_labels = np.clip(label_region, 0, max_id)
        dark_lut = (lut * 0.6).astype(np.uint8)
        region[edges] = dark_lut[safe_labels[edges]]


def draw_bboxes(
    image: np.ndarray,
    bboxes: list["BoundingBox"],
    color: tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
    fill_alpha: float = 0.0,
    font: str | None = None,
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Draws bounding boxes as closed paths using skia-python. Both axis-aligned
    and rotated bounding boxes are handled uniformly via corner points. For
    ``PredictedBoundingBox`` instances, the confidence score is drawn as text
    near the top-left corner.

    Args:
        image: Image array of shape (H, W, 3) uint8. Modified in-place and
            returned.
        bboxes: List of BoundingBox objects to draw.
        color: RGB color tuple for the bounding box outlines.
        line_width: Width of the outline in pixels.
        fill_alpha: If > 0, fill the bounding box interior with this opacity
            (0.0 to 1.0).
        font: Font family name for score text (e.g., ``"Arial"``). If
            ``None``, uses the system default typeface.

    Returns:
        The modified image array.
    """
    if not bboxes:
        return image

    import skia

    from sleap_io.model.bbox import PredictedBoundingBox

    # Pad to RGBA for skia surface
    frame_rgba = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
    surface = skia.Surface(frame_rgba, colorType=skia.kRGBA_8888_ColorType)
    canvas = surface.getCanvas()

    # Stroke paint for outlines
    stroke_paint = skia.Paint(
        Color=skia.Color(*color),
        AntiAlias=False,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=float(line_width),
        StrokeCap=skia.Paint.kSquare_Cap,
    )

    # Fill paint (only created if needed)
    fill_paint = None
    if fill_alpha > 0:
        fill_paint = skia.Paint(
            Color=skia.Color4f(
                color[0] / 255.0,
                color[1] / 255.0,
                color[2] / 255.0,
                fill_alpha,
            ).toColor(),
            AntiAlias=False,
            Style=skia.Paint.kFill_Style,
        )

    for bbox in bboxes:
        corners = bbox.corners

        # Build a closed path from the 4 corners
        path = skia.Path()
        path.moveTo(float(corners[0][0]), float(corners[0][1]))
        for i in range(1, len(corners)):
            path.lineTo(float(corners[i][0]), float(corners[i][1]))
        path.close()

        # Draw fill if requested
        if fill_paint is not None:
            canvas.drawPath(path, fill_paint)

        # Draw stroke
        canvas.drawPath(path, stroke_paint)

        # Score text for predicted bboxes
        if isinstance(bbox, PredictedBoundingBox):
            text_x = float(corners[0][0])
            text_y = float(corners[0][1]) - 5
            typeface = skia.Typeface(font if font else "sans-serif")
            skia_font = skia.Font(typeface, 12)
            text_paint = skia.Paint(Color=skia.Color(*color), AntiAlias=True)
            canvas.drawString(
                f"{bbox.score:.2f}", text_x, text_y, skia_font, text_paint
            )

    # Copy RGB channels back to the input image
    image[:] = frame_rgba[:, :, :3]

    return image


def _draw_geometry(canvas, geometry, stroke_paint, fill_paint) -> None:
    """Draw a Shapely geometry on a skia canvas.

    Dispatches to the appropriate drawing method based on geometry type.
    Supports ``Polygon``, ``MultiPolygon``, ``Point``, ``MultiPoint``,
    ``LineString``, ``MultiLineString``, and ``GeometryCollection``.

    Args:
        canvas: A skia.Canvas to draw on.
        geometry: A Shapely geometry object.
        stroke_paint: A skia.Paint for outlines/strokes.
        fill_paint: A skia.Paint for fills, or None for outline only.
    """
    import skia
    from shapely.geometry import (
        GeometryCollection,
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )

    if isinstance(geometry, Polygon):
        path = _polygon_to_path(geometry)
        if fill_paint is not None:
            canvas.drawPath(path, fill_paint)
        canvas.drawPath(path, stroke_paint)

    elif isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            path = _polygon_to_path(polygon)
            if fill_paint is not None:
                canvas.drawPath(path, fill_paint)
            canvas.drawPath(path, stroke_paint)

    elif isinstance(geometry, Point):
        radius = max(float(stroke_paint.getStrokeWidth()), 2.0)
        fill = skia.Paint(
            Color=stroke_paint.getColor(),
            AntiAlias=False,
            Style=skia.Paint.kFill_Style,
        )
        canvas.drawCircle(float(geometry.x), float(geometry.y), radius, fill)

    elif isinstance(geometry, MultiPoint):
        radius = max(float(stroke_paint.getStrokeWidth()), 2.0)
        fill = skia.Paint(
            Color=stroke_paint.getColor(),
            AntiAlias=False,
            Style=skia.Paint.kFill_Style,
        )
        for point in geometry.geoms:
            canvas.drawCircle(float(point.x), float(point.y), radius, fill)

    elif isinstance(geometry, LineString):
        path = _linestring_to_path(geometry)
        canvas.drawPath(path, stroke_paint)

    elif isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            path = _linestring_to_path(line)
            canvas.drawPath(path, stroke_paint)

    elif isinstance(geometry, GeometryCollection):
        for sub_geom in geometry.geoms:
            _draw_geometry(canvas, sub_geom, stroke_paint, fill_paint)


def _polygon_to_path(polygon) -> "skia.Path":
    """Convert a Shapely Polygon to a skia.Path.

    The exterior ring is drawn as a closed sub-path. Any interior rings
    (holes) are added as additional sub-paths with opposite winding to
    create cutouts when using the even-odd fill rule.

    Args:
        polygon: A Shapely Polygon geometry.

    Returns:
        A skia.Path representing the polygon.
    """
    import skia

    path = skia.Path()

    # Exterior ring
    coords = list(polygon.exterior.coords)
    if coords:
        path.moveTo(float(coords[0][0]), float(coords[0][1]))
        for x, y in coords[1:]:
            path.lineTo(float(x), float(y))
        path.close()

    # Interior rings (holes)
    for interior in polygon.interiors:
        hole_coords = list(interior.coords)
        if hole_coords:
            path.moveTo(float(hole_coords[0][0]), float(hole_coords[0][1]))
            for x, y in hole_coords[1:]:
                path.lineTo(float(x), float(y))
            path.close()

    # Use even-odd fill rule so holes are correctly subtracted
    path.setFillType(skia.PathFillType.kEvenOdd)

    return path


def _linestring_to_path(linestring) -> "skia.Path":
    """Convert a Shapely LineString to a skia.Path.

    Args:
        linestring: A Shapely LineString geometry.

    Returns:
        A skia.Path representing the line string (not closed).
    """
    import skia

    path = skia.Path()
    coords = list(linestring.coords)
    if coords:
        path.moveTo(float(coords[0][0]), float(coords[0][1]))
        for x, y in coords[1:]:
            path.lineTo(float(x), float(y))
    return path
