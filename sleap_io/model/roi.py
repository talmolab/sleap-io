"""Data structures for region of interest (ROI) annotations.

ROIs represent vector geometry annotations such as bounding boxes, polygons, and
arbitrary shapes. They use Shapely geometries internally for spatial operations.

The `AnnotationType` enum is shared by both `ROI` and `SegmentationMask` to indicate
the semantic meaning of an annotation.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from sleap_io.model.instance import Instance, Track
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.video import Video


class AnnotationType(IntEnum):
    """Semantic type of an annotation.

    Attributes:
        DEFAULT: General-purpose annotation with no specific semantic meaning.
        BOUNDING_BOX: Bounding box annotation for object detection.
        SEGMENTATION: Segmentation annotation (polygon or mask).
        ARENA: Arena boundary defining the region of valid tracking.
        ANCHOR: Anchor point or reference region.
    """

    DEFAULT = 0
    BOUNDING_BOX = 1
    SEGMENTATION = 2
    ARENA = 3
    ANCHOR = 4


@attrs.define(eq=False)
class ROI:
    """A region of interest defined by vector geometry.

    ROIs store Shapely geometry objects and optional metadata for associating
    annotations with videos, frames, tracks, and instances.

    Attributes:
        geometry: A Shapely geometry object (e.g., `Polygon`, `box`, `Point`).
        annotation_type: Semantic type of the annotation.
        name: Optional human-readable name for this ROI.
        category: Optional category label (e.g., class name for detection).
        score: Optional confidence score (0-1). If set, the ROI is considered
            a prediction.
        source: Optional string indicating the source of this annotation.
        video: Optional `Video` this ROI is associated with.
        frame_idx: Optional frame index. If `None`, the ROI is static (applies to
            all frames of the video).
        track: Optional `Track` this ROI is associated with.
        instance: Optional `Instance` this ROI is associated with.

    Notes:
        ROIs use identity-based equality (two ROI objects are only equal if they
        are the same object in memory).
    """

    geometry: object = attrs.field()
    annotation_type: AnnotationType = attrs.field(
        default=AnnotationType.DEFAULT, converter=AnnotationType
    )
    name: str = attrs.field(default="")
    category: str = attrs.field(default="")
    score: float | None = attrs.field(default=None)
    source: str = attrs.field(default="")
    video: "Video | None" = attrs.field(default=None)
    frame_idx: int | None = attrs.field(default=None)
    track: "Track | None" = attrs.field(default=None)
    instance: "Instance | None" = attrs.field(default=None)

    @classmethod
    def from_bbox(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        **kwargs,
    ) -> "ROI":
        """Create an ROI from a bounding box in xywh format.

        Args:
            x: Left edge x-coordinate.
            y: Top edge y-coordinate.
            width: Width of the bounding box.
            height: Height of the bounding box.
            **kwargs: Additional keyword arguments passed to the ROI constructor.

        Returns:
            An ROI with a rectangular polygon geometry.
        """
        from shapely.geometry import box

        geom = box(x, y, x + width, y + height)
        kwargs.setdefault("annotation_type", AnnotationType.BOUNDING_BOX)
        return cls(geometry=geom, **kwargs)

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        **kwargs,
    ) -> "ROI":
        """Create an ROI from a bounding box in xyxy (min/max) format.

        Args:
            x1: Left edge x-coordinate.
            y1: Top edge y-coordinate.
            x2: Right edge x-coordinate.
            y2: Bottom edge y-coordinate.
            **kwargs: Additional keyword arguments passed to the ROI constructor.

        Returns:
            An ROI with a rectangular polygon geometry.
        """
        from shapely.geometry import box

        geom = box(x1, y1, x2, y2)
        kwargs.setdefault("annotation_type", AnnotationType.BOUNDING_BOX)
        return cls(geometry=geom, **kwargs)

    @classmethod
    def from_polygon(
        cls,
        coords: list[tuple[float, float]] | np.ndarray,
        **kwargs,
    ) -> "ROI":
        """Create an ROI from polygon coordinates.

        Args:
            coords: A sequence of (x, y) coordinate pairs defining the polygon
                exterior ring. The polygon will be closed automatically.
            **kwargs: Additional keyword arguments passed to the ROI constructor.

        Returns:
            An ROI with a polygon geometry.
        """
        from shapely.geometry import Polygon

        geom = Polygon(coords)
        kwargs.setdefault("annotation_type", AnnotationType.SEGMENTATION)
        return cls(geometry=geom, **kwargs)

    @property
    def is_predicted(self) -> bool:
        """Whether this ROI is a prediction (has a confidence score)."""
        return self.score is not None

    @property
    def is_static(self) -> bool:
        """Whether this ROI is static (not tied to a specific frame)."""
        return self.frame_idx is None

    @property
    def is_bbox(self) -> bool:
        """Whether this ROI's geometry is a rectangular bounding box."""
        from shapely.geometry import Polygon

        if not isinstance(self.geometry, Polygon):
            return False
        # A rectangle has exactly 5 coordinates (closed ring) and the
        # minimum rotated rectangle has the same area.
        coords = list(self.geometry.exterior.coords)
        if len(coords) != 5:
            return False
        # Check if aligned to axes (all edges parallel to x or y axis)
        for i in range(4):
            dx = abs(coords[i + 1][0] - coords[i][0])
            dy = abs(coords[i + 1][1] - coords[i][1])
            if dx > 1e-10 and dy > 1e-10:
                return False
        return True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Bounding box as (minx, miny, maxx, maxy)."""
        return self.geometry.bounds

    @property
    def area(self) -> float:
        """Area of the geometry."""
        return self.geometry.area

    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid of the geometry as (x, y)."""
        c = self.geometry.centroid
        return (c.x, c.y)

    def to_mask(self, height: int, width: int) -> "SegmentationMask":
        """Rasterize this ROI into a binary segmentation mask.

        Args:
            height: Height of the output mask in pixels.
            width: Width of the output mask in pixels.

        Returns:
            A `SegmentationMask` with the rasterized geometry.
        """
        from sleap_io.model.mask import SegmentationMask

        # Rasterize geometry to binary mask
        mask = _rasterize_geometry(self.geometry, height, width)

        return SegmentationMask.from_numpy(
            mask,
            annotation_type=self.annotation_type,
            name=self.name,
            category=self.category,
            score=self.score,
            source=self.source,
            video=self.video,
            frame_idx=self.frame_idx,
            track=self.track,
            instance=self.instance,
        )


def _rasterize_geometry(geometry: object, height: int, width: int) -> np.ndarray:
    """Rasterize a Shapely geometry to a binary numpy array.

    Args:
        geometry: A Shapely geometry object.
        height: Height of the output array.
        width: Width of the output array.

    Returns:
        A boolean numpy array of shape (height, width).
    """
    from shapely.geometry import Polygon

    mask = np.zeros((height, width), dtype=bool)
    if not isinstance(geometry, Polygon):
        return mask

    # Get exterior coordinates
    coords = np.array(geometry.exterior.coords)
    # Use a simple scanline fill
    min_y = max(0, int(np.floor(coords[:, 1].min())))
    max_y = min(height - 1, int(np.floor(coords[:, 1].max())))

    for y in range(min_y, max_y + 1):
        # Find x intersections with edges at this y
        intersections = []
        n = len(coords) - 1  # Last coord == first coord (closed ring)
        for i in range(n):
            y0, y1 = coords[i, 1], coords[i + 1, 1]
            if y0 == y1:
                continue
            if min(y0, y1) <= y + 0.5 < max(y0, y1):
                x0, x1 = coords[i, 0], coords[i + 1, 0]
                t = (y + 0.5 - y0) / (y1 - y0)
                x_intersect = x0 + t * (x1 - x0)
                intersections.append(x_intersect)

        intersections.sort()
        # Fill between pairs of intersections
        for j in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(np.floor(intersections[j])))
            x_end = min(width, int(np.ceil(intersections[j + 1])))
            mask[y, x_start:x_end] = True

    return mask
