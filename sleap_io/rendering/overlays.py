"""Overlay drawing functions for ROIs and segmentation masks.

These functions draw annotations directly onto numpy image arrays using
skia-python for geometry rendering and numpy for mask blending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import skia

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
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw segmentation masks as colored overlays on an image.

    Args:
        image: Image array of shape (H, W, 3) uint8. Modified in-place and
            returned.
        masks: List of SegmentationMask objects to draw.
        color: RGB color tuple for the mask overlay.
        alpha: Opacity of the mask overlay (0.0 to 1.0).

    Returns:
        The modified image array.
    """
    for mask in masks:
        mask_data = mask.data
        if mask_data is None:
            continue

        h, w = mask_data.shape
        img_h, img_w = image.shape[:2]

        # Clip to image bounds
        draw_h = min(h, img_h)
        draw_w = min(w, img_w)

        region = image[:draw_h, :draw_w]
        mask_region = mask_data[:draw_h, :draw_w]

        # Blend color into masked pixels
        overlay = np.array(color, dtype=np.float32)
        region[mask_region] = (
            region[mask_region] * (1 - alpha) + overlay * alpha
        ).astype(np.uint8)

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
