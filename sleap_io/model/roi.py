"""Data structures for region of interest (ROI) annotations.

ROIs represent vector geometry annotations such as polygons and arbitrary shapes.
They use Shapely geometries internally for spatial operations.

The `AnnotationType` enum is kept for backward compatibility with old file formats
but is no longer used as a field on `ROI` or `SegmentationMask`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from shapely.geometry import Polygon
    from shapely.geometry.base import BaseGeometry

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
        name: Optional human-readable name for this ROI.
        category: Optional category label (e.g., class name for detection).
        source: Optional string indicating the source of this annotation.
        video: Optional `Video` this ROI is associated with.
        frame_idx: Optional frame index. If `None`, the ROI is static (applies to
            all frames of the video).
        track: Optional `Track` this ROI is associated with.
        instance: Optional `Instance` this ROI is associated with. Persisted in
            SLP format (v1.6+) via instance index.

    Notes:
        ROIs use identity-based equality (two ROI objects are only equal if they
        are the same object in memory).
    """

    geometry: "BaseGeometry" = attrs.field()

    @geometry.validator
    def _validate_geometry(self, attribute, value):
        """Validate that geometry is a Shapely BaseGeometry instance."""
        from shapely.geometry.base import BaseGeometry

        if not isinstance(value, BaseGeometry):
            raise TypeError(
                f"geometry must be a Shapely BaseGeometry instance, "
                f"got {type(value).__name__}"
            )

    name: str = attrs.field(default="")
    category: str = attrs.field(default="")
    source: str = attrs.field(default="")
    video: "Video | None" = attrs.field(default=None)
    frame_idx: int | None = attrs.field(default=None)
    track: "Track | None" = attrs.field(default=None)
    instance: "Instance | None" = attrs.field(default=None)

    # Private: deferred instance index for lazy loading. When ROIs are read
    # from a file without materialized instances (e.g., lazy mode), this stores
    # the raw instance_idx so it can be resolved later or written back as-is.
    _instance_idx: int = attrs.field(default=-1, repr=False, eq=False, init=False)

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
        return cls(geometry=geom, **kwargs)

    @classmethod
    def from_multi_polygon(
        cls,
        polygons: list[list[tuple[float, float]] | np.ndarray],
        **kwargs,
    ) -> "ROI":
        """Create an ROI from multiple polygon coordinate sequences.

        Args:
            polygons: A list of polygon coordinate sequences. Each sequence is a
                list of (x, y) pairs defining a polygon exterior ring.
            **kwargs: Additional keyword arguments passed to the ROI constructor.

        Returns:
            An ROI with a MultiPolygon geometry.
        """
        from shapely.geometry import MultiPolygon, Polygon

        geom = MultiPolygon([Polygon(coords) for coords in polygons])
        return cls(geometry=geom, **kwargs)

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

    @property
    def __geo_interface__(self) -> dict:
        """GeoJSON-compatible Feature representation.

        Returns a GeoJSON Feature dict following the Python `__geo_interface__`
        protocol. The Feature contains the ROI's geometry and metadata properties.

        Returns:
            A dictionary with ``"type"``, ``"geometry"``, and ``"properties"`` keys.
        """
        from shapely.geometry import mapping

        return {
            "type": "Feature",
            "geometry": mapping(self.geometry),
            "properties": {
                "name": self.name,
                "category": self.category,
                "source": self.source,
                "frame_idx": self.frame_idx,
            },
        }

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
            name=self.name,
            category=self.category,
            source=self.source,
            video=self.video,
            frame_idx=self.frame_idx,
            track=self.track,
            instance=self.instance,
        )

    def explode(self) -> list["ROI"]:
        """Split a multi-geometry ROI into individual ROIs.

        For ``MultiPolygon`` or ``GeometryCollection`` geometries, creates a
        separate ROI for each component geometry, preserving all metadata
        (name, category, source, video, frame_idx, track, instance).

        For single geometries (e.g., ``Polygon``, ``Point``), returns a list
        containing only this ROI.

        Returns:
            A list of ROIs, one per component geometry. For single geometries,
            returns ``[self]``.
        """
        from shapely.geometry import GeometryCollection, MultiPolygon

        if isinstance(self.geometry, (MultiPolygon, GeometryCollection)):
            return [
                ROI(
                    geometry=geom,
                    name=self.name,
                    category=self.category,
                    source=self.source,
                    video=self.video,
                    frame_idx=self.frame_idx,
                    track=self.track,
                    instance=self.instance,
                )
                for geom in self.geometry.geoms
            ]
        return [self]


def _rasterize_geometry(
    geometry: "BaseGeometry", height: int, width: int
) -> np.ndarray:
    """Rasterize a Shapely geometry to a binary numpy array.

    Supported geometry types:
        - ``Polygon``: Rasterized using scanline fill with hole support.
        - ``MultiPolygon``: Each component polygon is rasterized individually.

    Unsupported geometry types (e.g., ``Point``, ``LineString``) will raise a
    ``TypeError``.

    Args:
        geometry: A Shapely geometry object (``Polygon`` or ``MultiPolygon``).
        height: Height of the output array.
        width: Width of the output array.

    Returns:
        A boolean numpy array of shape (height, width).

    Raises:
        TypeError: If the geometry type is not ``Polygon`` or ``MultiPolygon``.
    """
    from shapely.geometry import MultiPolygon, Polygon

    mask = np.zeros((height, width), dtype=bool)

    if isinstance(geometry, Polygon):
        _rasterize_polygon(geometry, mask, height, width)
    elif isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            _rasterize_polygon(polygon, mask, height, width)
    else:
        raise TypeError(
            f"Unsupported geometry type for rasterization: "
            f"{type(geometry).__name__}. "
            f"Supported types are Polygon and MultiPolygon."
        )

    return mask


def _rasterize_polygon(
    polygon: "Polygon", mask: np.ndarray, height: int, width: int
) -> None:
    """Rasterize a single Polygon onto an existing mask.

    Args:
        polygon: A Shapely ``Polygon`` geometry.
        mask: The mask array to modify in-place.
        height: Height of the mask.
        width: Width of the mask.
    """
    # Fill exterior ring
    _scanline_fill(np.array(polygon.exterior.coords), mask, height, width, fill=True)

    # Subtract interior rings (holes)
    for interior in polygon.interiors:
        _scanline_fill(np.array(interior.coords), mask, height, width, fill=False)


def _scanline_fill(
    coords: np.ndarray,
    mask: np.ndarray,
    height: int,
    width: int,
    fill: bool = True,
) -> None:
    """Fill or unfill a polygon ring on a mask using scanline algorithm.

    Args:
        coords: Polygon ring coordinates as an (N, 2) array (closed ring).
        mask: The mask array to modify in-place.
        height: Height of the mask.
        width: Width of the mask.
        fill: If True, set pixels to True. If False, set pixels to False.
    """
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
            mask[y, x_start:x_end] = fill
