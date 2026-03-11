"""Overlay drawing functions for ROIs and segmentation masks.

These functions draw annotations directly onto numpy image arrays using
lightweight line-drawing algorithms, without requiring skia-python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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

    Draws the boundary of each ROI's geometry as lines on the image. Supports
    ``Polygon`` and ``MultiPolygon`` geometries. For bounding box ROIs, draws a
    rectangle.

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
    from shapely.geometry import MultiPolygon, Polygon

    for roi in rois:
        geom = roi.geometry
        if isinstance(geom, Polygon):
            _draw_polygon(image, geom, color, line_width, fill_alpha)
        elif isinstance(geom, MultiPolygon):
            for polygon in geom.geoms:
                _draw_polygon(image, polygon, color, line_width, fill_alpha)

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


def _draw_polygon(
    image: np.ndarray,
    polygon,
    color: tuple[int, int, int],
    line_width: int,
    fill_alpha: float,
) -> None:
    """Draw a single polygon on an image.

    Args:
        image: Image array to draw on (modified in-place).
        polygon: A Shapely Polygon geometry.
        color: RGB color tuple.
        line_width: Line width in pixels.
        fill_alpha: Fill opacity (0.0 for outline only).
    """
    coords = np.array(polygon.exterior.coords)

    # Fill interior if requested
    if fill_alpha > 0:
        from sleap_io.model.roi import _rasterize_geometry

        h, w = image.shape[:2]
        mask = _rasterize_geometry(polygon, h, w)
        overlay = np.array(color, dtype=np.float32)
        image[mask] = (image[mask] * (1 - fill_alpha) + overlay * fill_alpha).astype(
            np.uint8
        )

    # Draw outline
    _draw_polyline(image, coords, color, line_width)


def _draw_polyline(
    image: np.ndarray,
    coords: np.ndarray,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    """Draw a polyline (sequence of connected line segments) on an image.

    Uses Bresenham-style line drawing with configurable width.

    Args:
        image: Image array to draw on (modified in-place).
        coords: (N, 2) array of (x, y) coordinates.
        color: RGB color tuple.
        line_width: Line width in pixels.
    """
    h, w = image.shape[:2]
    half_w = line_width // 2

    for i in range(len(coords) - 1):
        x0, y0 = int(round(coords[i][0])), int(round(coords[i][1]))
        x1, y1 = int(round(coords[i + 1][0])), int(round(coords[i + 1][1]))

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Draw pixel with width
            for wy in range(max(0, y0 - half_w), min(h, y0 + half_w + 1)):
                for wx in range(max(0, x0 - half_w), min(w, x0 + half_w + 1)):
                    image[wy, wx] = color

            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
