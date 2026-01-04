"""Callback context classes for custom rendering.

This module provides context objects that are passed to user-defined callbacks
during rendering, giving access to the Skia canvas and rendering metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from attrs import define

if TYPE_CHECKING:
    import skia


@define
class RenderContext:
    """Context passed to pre/post render callbacks.

    This context provides access to the Skia canvas and frame-level metadata
    for drawing custom overlays before or after pose rendering.

    Attributes:
        canvas: Skia canvas for drawing.
        frame_idx: Current frame index.
        frame_size: (width, height) tuple of original frame dimensions.
        instances: List of instances in this frame.
        skeleton_edges: Edge connectivity as list of (src, dst) tuples.
        node_names: List of node name strings.
        scale: Current scale factor for rendering.
        offset: Current offset (x, y) for cropped/zoomed views.
    """

    canvas: "skia.Canvas"
    frame_idx: int
    frame_size: tuple[int, int]
    instances: list
    skeleton_edges: list[tuple[int, int]]
    node_names: list[str]
    scale: float = 1.0
    offset: tuple[float, float] = (0.0, 0.0)

    def world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        """Transform world coordinates to canvas coordinates.

        Args:
            x: X coordinate in world/frame space.
            y: Y coordinate in world/frame space.

        Returns:
            (x, y) coordinates in canvas space.
        """
        return (
            (x - self.offset[0]) * self.scale,
            (y - self.offset[1]) * self.scale,
        )


@define
class InstanceContext:
    """Context passed to per-instance callbacks.

    This context provides access to the Skia canvas and instance-level metadata
    for drawing custom overlays after each instance is rendered.

    Attributes:
        canvas: Skia canvas for drawing.
        instance_idx: Index of this instance within the frame.
        points: (n_nodes, 2) array of keypoint coordinates.
        track_id: Track ID if assigned, else None.
        track_name: Track name string if available.
        confidence: Instance confidence score if available.
        skeleton_edges: Edge connectivity as list of (src, dst) tuples.
        node_names: List of node name strings.
        scale: Current scale factor for rendering.
        offset: Current offset (x, y) for cropped/zoomed views.
    """

    canvas: "skia.Canvas"
    instance_idx: int
    points: np.ndarray
    skeleton_edges: list[tuple[int, int]]
    node_names: list[str]
    track_id: Optional[int] = None
    track_name: Optional[str] = None
    confidence: Optional[float] = None
    scale: float = 1.0
    offset: tuple[float, float] = (0.0, 0.0)

    def world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        """Transform world coordinates to canvas coordinates.

        Args:
            x: X coordinate in world/frame space.
            y: Y coordinate in world/frame space.

        Returns:
            (x, y) coordinates in canvas space.
        """
        return (
            (x - self.offset[0]) * self.scale,
            (y - self.offset[1]) * self.scale,
        )

    def get_centroid(self) -> Optional[tuple[float, float]]:
        """Get centroid of valid points.

        Returns:
            (x, y) mean of valid (non-NaN) points, or None if all invalid.
        """
        valid_mask = np.isfinite(self.points).all(axis=1)
        valid_points = self.points[valid_mask]
        if len(valid_points) == 0:
            return None
        mean_pt = valid_points.mean(axis=0)
        return (float(mean_pt[0]), float(mean_pt[1]))

    def get_bbox(self) -> Optional[tuple[float, float, float, float]]:
        """Get bounding box of valid points.

        Returns:
            (x1, y1, x2, y2) bounding box, or None if no valid points.
        """
        valid_mask = np.isfinite(self.points).all(axis=1)
        valid_points = self.points[valid_mask]
        if len(valid_points) == 0:
            return None
        return (
            float(valid_points[:, 0].min()),
            float(valid_points[:, 1].min()),
            float(valid_points[:, 0].max()),
            float(valid_points[:, 1].max()),
        )
