"""Marker shape drawing functions for pose rendering.

This module provides functions to draw different marker shapes at keypoint
locations using Skia.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional

if TYPE_CHECKING:
    import skia

# Type alias for marker shapes
MarkerShape = Literal["circle", "square", "diamond", "triangle", "cross"]


def draw_circle_marker(
    canvas: "skia.Canvas",
    x: float,
    y: float,
    size: float,
    fill_paint: "skia.Paint",
    edge_paint: Optional["skia.Paint"] = None,
) -> None:
    """Draw a filled circle marker, optionally with edge.

    Args:
        canvas: Skia canvas to draw on.
        x: X coordinate of marker center.
        y: Y coordinate of marker center.
        size: Marker radius in pixels.
        fill_paint: Paint for filling the marker.
        edge_paint: Optional paint for drawing marker edge/stroke.
    """
    canvas.drawCircle(float(x), float(y), size, fill_paint)
    if edge_paint is not None:
        canvas.drawCircle(float(x), float(y), size, edge_paint)


def draw_square_marker(
    canvas: "skia.Canvas",
    x: float,
    y: float,
    size: float,
    fill_paint: "skia.Paint",
    edge_paint: Optional["skia.Paint"] = None,
) -> None:
    """Draw a filled square marker, optionally with edge.

    Args:
        canvas: Skia canvas to draw on.
        x: X coordinate of marker center.
        y: Y coordinate of marker center.
        size: Half-width of square in pixels.
        fill_paint: Paint for filling the marker.
        edge_paint: Optional paint for drawing marker edge/stroke.
    """
    import skia

    half = size
    rect = skia.Rect(float(x) - half, float(y) - half, float(x) + half, float(y) + half)
    canvas.drawRect(rect, fill_paint)
    if edge_paint is not None:
        canvas.drawRect(rect, edge_paint)


def draw_diamond_marker(
    canvas: "skia.Canvas",
    x: float,
    y: float,
    size: float,
    fill_paint: "skia.Paint",
    edge_paint: Optional["skia.Paint"] = None,
) -> None:
    """Draw a diamond (rotated square) marker.

    Args:
        canvas: Skia canvas to draw on.
        x: X coordinate of marker center.
        y: Y coordinate of marker center.
        size: Distance from center to vertices in pixels.
        fill_paint: Paint for filling the marker.
        edge_paint: Optional paint for drawing marker edge/stroke.
    """
    import skia

    path = skia.Path()
    path.moveTo(float(x), float(y) - size)  # Top
    path.lineTo(float(x) + size, float(y))  # Right
    path.lineTo(float(x), float(y) + size)  # Bottom
    path.lineTo(float(x) - size, float(y))  # Left
    path.close()
    canvas.drawPath(path, fill_paint)
    if edge_paint is not None:
        canvas.drawPath(path, edge_paint)


def draw_triangle_marker(
    canvas: "skia.Canvas",
    x: float,
    y: float,
    size: float,
    fill_paint: "skia.Paint",
    edge_paint: Optional["skia.Paint"] = None,
) -> None:
    """Draw a triangle marker (pointing up).

    Args:
        canvas: Skia canvas to draw on.
        x: X coordinate of marker center.
        y: Y coordinate of marker center.
        size: Size scale for the triangle in pixels.
        fill_paint: Paint for filling the marker.
        edge_paint: Optional paint for drawing marker edge/stroke.
    """
    import skia

    path = skia.Path()
    # Equilateral triangle centered at (x, y)
    h = size * 1.5  # Height
    path.moveTo(float(x), float(y) - h * 0.6)  # Top
    path.lineTo(float(x) + size, float(y) + h * 0.4)  # Bottom right
    path.lineTo(float(x) - size, float(y) + h * 0.4)  # Bottom left
    path.close()
    canvas.drawPath(path, fill_paint)
    if edge_paint is not None:
        canvas.drawPath(path, edge_paint)


def draw_cross_marker(
    canvas: "skia.Canvas",
    x: float,
    y: float,
    size: float,
    fill_paint: "skia.Paint",
    edge_paint: Optional["skia.Paint"] = None,
) -> None:
    """Draw a cross/plus marker.

    Args:
        canvas: Skia canvas to draw on.
        x: X coordinate of marker center.
        y: Y coordinate of marker center.
        size: Half-length of cross arms in pixels.
        fill_paint: Paint for filling the marker.
        edge_paint: Ignored for cross markers.
    """
    import skia

    # Use stroke style for cross (extracting color from fill_paint)
    cross_paint = skia.Paint(
        Color=fill_paint.getColor(),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=size * 0.4,
        StrokeCap=skia.Paint.kRound_Cap,
    )
    canvas.drawLine(float(x) - size, float(y), float(x) + size, float(y), cross_paint)
    canvas.drawLine(float(x), float(y) - size, float(x), float(y) + size, cross_paint)


# Mapping from shape names to drawing functions
MARKER_FUNCS: dict[
    MarkerShape,
    Callable[
        [
            "skia.Canvas",
            float,
            float,
            float,
            "skia.Paint",
            Optional["skia.Paint"],
        ],
        None,
    ],
] = {
    "circle": draw_circle_marker,
    "square": draw_square_marker,
    "diamond": draw_diamond_marker,
    "triangle": draw_triangle_marker,
    "cross": draw_cross_marker,
}


def get_marker_func(
    shape: MarkerShape,
) -> Callable[
    [
        "skia.Canvas",
        float,
        float,
        float,
        "skia.Paint",
        Optional["skia.Paint"],
    ],
    None,
]:
    """Get the drawing function for a marker shape.

    Args:
        shape: Marker shape name.

    Returns:
        Drawing function for the specified shape.

    Raises:
        ValueError: If shape is not recognized.
    """
    if shape not in MARKER_FUNCS:
        raise ValueError(
            f"Unknown marker shape: {shape}. "
            f"Available shapes: {list(MARKER_FUNCS.keys())}"
        )
    return MARKER_FUNCS[shape]
