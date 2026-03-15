"""Data structures for bounding box annotations.

Bounding boxes are first-class annotations for object detection and tracking
workflows. They support axis-aligned and oriented (rotated) bounding boxes with
user/predicted distinction.

The class hierarchy:
    - `BoundingBox` — base with geometry, video/frame/track/instance metadata
    - `UserBoundingBox` — human-annotated bounding box
    - `PredictedBoundingBox` — model-predicted bounding box with confidence score
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from sleap_io.model.instance import Instance, Track
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.roi import ROI
    from sleap_io.model.video import Video


@attrs.define(eq=False)
class BoundingBox:
    """A bounding box annotation.

    Supports axis-aligned and oriented (rotated) bounding boxes with optional
    metadata for associating with videos, frames, tracks, and instances.

    Attributes:
        x_center: Center x-coordinate in pixels.
        y_center: Center y-coordinate in pixels.
        width: Box width in pixels.
        height: Box height in pixels.
        angle: Rotation angle in radians (0 = axis-aligned).
        video: Video this bounding box is associated with.
        frame_idx: Frame index within the video.
        track: Optional tracking identity.
        instance: Optional linked pose instance.
        category: Class label (e.g., "mouse", "fly").
        name: Human-readable name.
        source: Annotation source identifier.

    Notes:
        Bounding boxes use identity-based equality (two BoundingBox objects are
        only equal if they are the same object in memory).
    """

    x_center: float = attrs.field()
    y_center: float = attrs.field()
    width: float = attrs.field()
    height: float = attrs.field()
    angle: float = attrs.field(default=0.0)
    video: "Video | None" = attrs.field(default=None)
    frame_idx: int | None = attrs.field(default=None)
    track: "Track | None" = attrs.field(default=None)
    instance: "Instance | None" = attrs.field(default=None)
    category: str = attrs.field(default="")
    name: str = attrs.field(default="")
    source: str = attrs.field(default="")

    # Private: deferred instance index for lazy loading.
    _instance_idx: int = attrs.field(default=-1, repr=False, eq=False, init=False)

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        **kwargs,
    ) -> "BoundingBox":
        """Create a bounding box from corner coordinates.

        Args:
            x1: Left edge x-coordinate.
            y1: Top edge y-coordinate.
            x2: Right edge x-coordinate.
            y2: Bottom edge y-coordinate.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A new bounding box instance.
        """
        w = x2 - x1
        h = y2 - y1
        return cls(
            x_center=x1 + w / 2,
            y_center=y1 + h / 2,
            width=w,
            height=h,
            **kwargs,
        )

    @classmethod
    def from_xywh(
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        **kwargs,
    ) -> "BoundingBox":
        """Create a bounding box from top-left corner and dimensions.

        Args:
            x: Left edge x-coordinate.
            y: Top edge y-coordinate.
            w: Width of the bounding box.
            h: Height of the bounding box.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A new bounding box instance.
        """
        return cls(
            x_center=x + w / 2,
            y_center=y + h / 2,
            width=w,
            height=h,
            **kwargs,
        )

    @property
    def is_predicted(self) -> bool:
        """Whether this bounding box is a prediction."""
        return isinstance(self, PredictedBoundingBox)

    @property
    def is_rotated(self) -> bool:
        """Whether this bounding box is rotated (non-axis-aligned)."""
        return self.angle != 0.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Corner coordinates as (x1, y1, x2, y2).

        Returns:
            Tuple of (left, top, right, bottom) coordinates.

        Raises:
            ValueError: If the bounding box is rotated.
        """
        if self.is_rotated:
            raise ValueError(
                "xyxy is only defined for axis-aligned bounding boxes. "
                "Use `bounds` or `corners` for rotated boxes."
            )
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x_center - half_w,
            self.y_center - half_h,
            self.x_center + half_w,
            self.y_center + half_h,
        )

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Top-left corner and dimensions as (x, y, width, height).

        Returns:
            Tuple of (left, top, width, height).

        Raises:
            ValueError: If the bounding box is rotated.
        """
        if self.is_rotated:
            raise ValueError(
                "xywh is only defined for axis-aligned bounding boxes. "
                "Use `bounds` or `corners` for rotated boxes."
            )
        return (
            self.x_center - self.width / 2,
            self.y_center - self.height / 2,
            self.width,
            self.height,
        )

    @property
    def corners(self) -> np.ndarray:
        """Corner points as a (4, 2) array.

        Returns corners in order: top-left, top-right, bottom-right, bottom-left
        (before rotation). Works for both axis-aligned and rotated boxes.

        Returns:
            A (4, 2) numpy array of corner coordinates.
        """
        half_w = self.width / 2
        half_h = self.height / 2
        # Corners relative to center (TL, TR, BR, BL)
        corners = np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ]
        )
        if self.is_rotated:
            cos_a = math.cos(self.angle)
            sin_a = math.sin(self.angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = corners @ rotation.T
        corners[:, 0] += self.x_center
        corners[:, 1] += self.y_center
        return corners

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding extent as (minx, miny, maxx, maxy).

        Works for both axis-aligned and rotated bounding boxes.

        Returns:
            Tuple of (minx, miny, maxx, maxy).
        """
        if not self.is_rotated:
            half_w = self.width / 2
            half_h = self.height / 2
            return (
                self.x_center - half_w,
                self.y_center - half_h,
                self.x_center + half_w,
                self.y_center + half_h,
            )
        c = self.corners
        return (
            float(c[:, 0].min()),
            float(c[:, 1].min()),
            float(c[:, 0].max()),
            float(c[:, 1].max()),
        )

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    def to_roi(self) -> "ROI":
        """Convert to an ROI with Shapely polygon geometry.

        Returns:
            An ROI with a rectangular polygon matching this bounding box.
        """
        from shapely.geometry import Polygon

        from sleap_io.model.roi import ROI

        corners = self.corners
        # Close the ring
        coords = list(map(tuple, corners)) + [tuple(corners[0])]
        geom = Polygon(coords)
        return ROI(
            geometry=geom,
            name=self.name,
            category=self.category,
            source=self.source,
            video=self.video,
            frame_idx=self.frame_idx,
            track=self.track,
            instance=self.instance,
        )

    def to_mask(self, height: int, width: int) -> "SegmentationMask":
        """Rasterize this bounding box into a binary segmentation mask.

        Args:
            height: Height of the output mask in pixels.
            width: Width of the output mask in pixels.

        Returns:
            A SegmentationMask with the rasterized bounding box.
        """
        roi = self.to_roi()
        return roi.to_mask(height, width)


@attrs.define(eq=False)
class UserBoundingBox(BoundingBox):
    """A human-annotated bounding box.

    Inherits all fields from `BoundingBox`. Has no additional fields.

    See `BoundingBox` for attribute documentation.
    """

    pass


@attrs.define(eq=False)
class PredictedBoundingBox(BoundingBox):
    """A model-predicted bounding box with a confidence score.

    Attributes:
        score: Confidence score (0-1).

    See `BoundingBox` for other attribute documentation.
    """

    score: float = attrs.field(default=0.0)
