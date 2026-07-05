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

    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.centroid import Centroid
    from sleap_io.model.embedding import Embedding
    from sleap_io.model.identity import Identity
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
        video: Optional `Video` this ROI is associated with. Used for static ROIs
            that are not tied to any specific frame.
        track: Optional `Track` this ROI is associated with.
        tracking_score: Confidence of the track identity assignment. ``None``
            if unassigned or manually assigned.
        identity: Optional global, ground-truth `Identity` for this ROI -- the
            persistent cross-video animal identity / re-identification key. ``None``
            if no global identity is assigned. Mirrors `Instance.identity`.
        identity_score: Score associated with the `identity` assignment (e.g. the
            re-ID match similarity). ``None`` if unassigned or assigned manually.
            Kept separate from `tracking_score` (short-term tracklet vs long-term
            identity).
        instance: Optional `Instance` this ROI is associated with. Persisted in
            SLP format (v1.6+) via instance index.
        identity_embedding: Optional `Embedding` describing this detection's
            appearance for re-identification. ``None`` by default.

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
    track: "Track | None" = attrs.field(default=None)
    tracking_score: float | None = attrs.field(default=None)
    identity: "Identity | None" = attrs.field(default=None)
    identity_score: float | None = attrs.field(default=None)
    instance: "Instance | None" = attrs.field(default=None)
    identity_embedding: "Embedding | None" = attrs.field(default=None, repr=False)

    # Private: deferred instance index for lazy loading. When ROIs are read
    # from a file without materialized instances (e.g., lazy mode), this stores
    # the raw instance_idx so it can be resolved later or written back as-is.
    _instance_idx: int = attrs.field(default=-1, repr=False, eq=False, init=False)

    def __attrs_post_init__(self):
        """Validate that this class is not instantiated directly."""
        if type(self) is ROI:
            raise TypeError("ROI is abstract. Use UserROI or PredictedROI.")

    @property
    def is_predicted(self) -> bool:
        """Whether this ROI is a model prediction."""
        return isinstance(self, PredictedROI)

    @property
    def is_empty(self) -> bool:
        """Whether this ROI's geometry is empty (no spatial extent)."""
        return bool(self.geometry.is_empty)

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

        Note:
            For detection bounding boxes, prefer ``BoundingBox.from_xywh()`` or
            ``BoundingBox.from_xyxy()`` which provide richer metadata support.

        .. deprecated::
            Use ``BoundingBox.from_xywh()`` for detection bounding boxes.
        """
        import warnings

        warnings.warn(
            "ROI.from_bbox() is deprecated. Use BoundingBox.from_xywh() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        Note:
            For detection bounding boxes, prefer ``BoundingBox.from_xywh()`` or
            ``BoundingBox.from_xyxy()`` which provide richer metadata support.

        .. deprecated::
            Use ``BoundingBox.from_xyxy()`` for detection bounding boxes.
        """
        import warnings

        warnings.warn(
            "ROI.from_xyxy() is deprecated. Use BoundingBox.from_xyxy() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
    def centroid_xy(self) -> tuple[float, float]:
        """Centroid of the geometry as ``(x, y)``."""
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
            },
        }

    def to_mask(self, height: int, width: int) -> "SegmentationMask":
        """Rasterize this ROI into a binary segmentation mask.

        A `PredictedROI` produces a `PredictedSegmentationMask` carrying its
        `score`; any other ROI produces a `UserSegmentationMask`. Metadata
        (name, category, source, track, instance) is inherited either way.

        Args:
            height: Height of the output mask in pixels.
            width: Width of the output mask in pixels.

        Returns:
            A `SegmentationMask` with the rasterized geometry.
        """
        from sleap_io.model.mask import (
            PredictedSegmentationMask,
            UserSegmentationMask,
        )

        # Rasterize geometry to binary mask
        mask = _rasterize_geometry(self.geometry, height, width)

        kwargs = dict(
            name=self.name,
            category=self.category,
            source=self.source,
            track=self.track,
            tracking_score=self.tracking_score,
            identity=self.identity,
            identity_score=self.identity_score,
            identity_embedding=self.identity_embedding,
            instance=self.instance,
        )
        if self.is_predicted:
            return PredictedSegmentationMask.from_numpy(
                mask, score=self.score, **kwargs
            )
        return UserSegmentationMask.from_numpy(mask, **kwargs)

    def to_centroid(
        self, representative: bool = False, error_on_empty: bool = False
    ) -> "Centroid":
        """Reduce this ROI to a single centroid point.

        A `PredictedROI` produces a `PredictedCentroid` carrying its `score`; any
        other ROI produces a `UserCentroid`. Metadata (track, tracking_score,
        identity, identity_score, category, name, source, instance) is inherited.

        Args:
            representative: If ``True``, use Shapely's ``representative_point()``
                (a point guaranteed to lie within the geometry); otherwise use the
                geometric ``centroid`` (which may fall outside concave shapes).
            error_on_empty: If ``True``, raise ``ValueError`` when the geometry is
                empty instead of returning a degenerate (NaN) centroid.

        Returns:
            A `Centroid` at the geometry's centroid (or NaN if empty).

        Raises:
            ValueError: If the geometry is empty and ``error_on_empty`` is ``True``.
        """
        from sleap_io.model.centroid import PredictedCentroid, UserCentroid

        if self.geometry.is_empty:
            if error_on_empty:
                raise ValueError("Cannot compute centroid of an empty ROI geometry.")
            x = y = float("nan")
        else:
            pt = (
                self.geometry.representative_point()
                if representative
                else self.geometry.centroid
            )
            x, y = float(pt.x), float(pt.y)

        kwargs = dict(
            x=x,
            y=y,
            track=self.track,
            tracking_score=self.tracking_score,
            identity=self.identity,
            identity_score=self.identity_score,
            identity_embedding=self.identity_embedding,
            instance=self.instance,
            category=self.category,
            name=self.name,
            source=self.source,
        )
        if self.is_predicted:
            return PredictedCentroid(score=self.score, **kwargs)
        return UserCentroid(**kwargs)

    def to_bbox(
        self,
        padding: float | tuple[float, float] = 0.0,
        rotated: bool = False,
        error_on_empty: bool = False,
    ) -> "BoundingBox":
        """Reduce this ROI to a bounding box.

        A `PredictedROI` produces a `PredictedBoundingBox` carrying its `score`;
        any other ROI produces a `UserBoundingBox`. Metadata (track,
        tracking_score, identity, identity_score, category, name, source,
        instance) is inherited.

        Args:
            padding: Amount to inflate the box outward. Scalar applies to both
                axes; a ``(px, py)`` tuple applies per-axis. Negative values
                shrink the box. For rotated boxes, padding enlarges the
                pre-rotation extent about the center while preserving the angle.
            rotated: If ``True``, fit a minimum-area oriented box (rotated). If
                ``False``, fit an axis-aligned box from the geometry bounds.
            error_on_empty: If ``True``, raise ``ValueError`` when the geometry is
                empty instead of returning a degenerate (NaN) box.

        Returns:
            A `BoundingBox` enclosing the geometry (or NaN corners if empty).

        Raises:
            ValueError: If the geometry is empty and ``error_on_empty`` is ``True``.
        """
        from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox

        if self.geometry.is_empty:
            if error_on_empty:
                raise ValueError(
                    "Cannot compute bounding box of an empty ROI geometry."
                )
            nan = float("nan")
            x1 = y1 = x2 = y2 = nan
            angle = 0.0
        else:
            x1, y1, x2, y2, angle = _geometry_to_bbox_coords(self.geometry, rotated)
            x1, y1, x2, y2 = _apply_padding(x1, y1, x2, y2, padding)

        kwargs = dict(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            angle=angle,
            track=self.track,
            tracking_score=self.tracking_score,
            identity=self.identity,
            identity_score=self.identity_score,
            identity_embedding=self.identity_embedding,
            instance=self.instance,
            category=self.category,
            name=self.name,
            source=self.source,
        )
        if self.is_predicted:
            return PredictedBoundingBox(score=self.score, **kwargs)
        return UserBoundingBox(**kwargs)

    def explode(self) -> list["ROI"]:
        """Split a multi-geometry ROI into individual ROIs.

        For ``MultiPolygon`` or ``GeometryCollection`` geometries, creates a
        separate ROI for each component geometry, preserving all metadata
        (name, category, source, video, track, instance).

        For single geometries (e.g., ``Polygon``, ``Point``), returns a list
        containing only this ROI.

        Returns:
            A list of ROIs, one per component geometry. For single geometries,
            returns ``[self]``.
        """
        from shapely.geometry import GeometryCollection, MultiPolygon

        if isinstance(self.geometry, (MultiPolygon, GeometryCollection)):
            extra = {"score": self.score} if hasattr(self, "score") else {}
            return [
                type(self)(
                    geometry=geom,
                    name=self.name,
                    category=self.category,
                    source=self.source,
                    video=self.video,
                    track=self.track,
                    tracking_score=self.tracking_score,
                    identity=self.identity,
                    identity_score=self.identity_score,
                    identity_embedding=self.identity_embedding,
                    instance=self.instance,
                    **extra,
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


def _apply_padding(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: float | tuple[float, float],
) -> tuple[float, float, float, float]:
    """Inflate an axis-aligned box by a scalar or per-axis padding.

    Shared padding logic used by centroid/bbox/instance conversions so that all
    paths inflate boxes identically.

    Args:
        x1: Left edge x-coordinate.
        y1: Top edge y-coordinate.
        x2: Right edge x-coordinate.
        y2: Bottom edge y-coordinate.
        padding: Scalar applied to both axes, or a ``(px, py)`` tuple applied
            per-axis. Negative values shrink the box; values are not clamped.

    Returns:
        The padded ``(x1, y1, x2, y2)`` tuple.
    """
    if isinstance(padding, (tuple, list)):
        px, py = padding
    else:
        px = py = padding
    return (x1 - px, y1 - py, x2 + px, y2 + py)


def _pose_to_geometry(
    points: np.ndarray,
    edges: list[tuple[int, int]],
    method: str = "shapes",
    node_radius: float = 0.0,
    edge_radius: float = 0.0,
    radius: float = 0.0,
    quad_segs: int = 8,
) -> "BaseGeometry":
    """Build a Shapely geometry from pose points (central pose->vector hub).

    Args:
        points: ``(n_nodes, 2)`` array of node coordinates; invisible nodes are
            ``NaN``.
        edges: List of ``(src, dst)`` node-index pairs (e.g.
            ``skeleton.edge_inds``).
        method: ``"shapes"`` to union buffered node points and/or edge segments,
            or ``"convex_hull"`` to take the convex hull of visible points.
        node_radius: Buffer radius around each visible node (``"shapes"`` only).
        edge_radius: Buffer radius around each fully-visible edge segment
            (``"shapes"`` only).
        radius: Optional buffer applied to the convex hull (``"convex_hull"``
            only).
        quad_segs: Number of segments used to approximate a quarter circle when
            buffering.

    Returns:
        A Shapely geometry. An empty ``Polygon`` is returned when there are no
        visible points (or no shapes were produced).

    Raises:
        ValueError: If ``method="shapes"`` with both ``node_radius`` and
            ``edge_radius`` equal to 0 (a misconfiguration), or if ``method`` is
            not recognized.
    """
    from shapely.geometry import LineString, MultiPoint, Point, Polygon
    from shapely.ops import unary_union

    points = np.asarray(points, dtype=float)
    visible = ~np.isnan(points[:, 0])

    if method == "shapes":
        if node_radius == 0 and edge_radius == 0:
            raise ValueError(
                "method='shapes' requires at least one of node_radius or "
                "edge_radius to be > 0."
            )
        if not visible.any():
            return Polygon()
        shapes: list[BaseGeometry] = []
        if node_radius > 0:
            for i in np.nonzero(visible)[0]:
                x, y = points[i]
                shapes.append(
                    Point(float(x), float(y)).buffer(node_radius, quad_segs=quad_segs)
                )
        if edge_radius > 0:
            for src, dst in edges:
                if visible[src] and visible[dst]:
                    xs, ys = points[src]
                    xd, yd = points[dst]
                    shapes.append(
                        LineString(
                            [(float(xs), float(ys)), (float(xd), float(yd))]
                        ).buffer(edge_radius, quad_segs=quad_segs)
                    )
        if shapes:
            return unary_union(shapes)
        return Polygon()

    elif method == "convex_hull":
        if not visible.any():
            return Polygon()
        visible_xy = points[visible]
        hull = MultiPoint([tuple(p) for p in visible_xy]).convex_hull
        if radius > 0:
            hull = hull.buffer(radius, quad_segs=quad_segs)
        return hull

    else:
        raise ValueError(
            f"Unknown method {method!r}. Expected 'shapes' or 'convex_hull'."
        )


def _geometry_to_bbox_coords(
    geometry: "BaseGeometry", rotated: bool = False
) -> tuple[float, float, float, float, float]:
    """Compute bounding-box ``(x1, y1, x2, y2, angle)`` from a geometry.

    Args:
        geometry: A Shapely geometry to enclose.
        rotated: If ``True``, fit a minimum-area oriented box and return its
            centered ``(x1, y1, x2, y2)`` plus rotation ``angle`` (radians),
            consistent with ``BoundingBox.corners``. If ``False``, return the
            axis-aligned bounds with ``angle=0``.

    Returns:
        A tuple ``(x1, y1, x2, y2, angle)``. For an empty geometry, all corner
        values are ``NaN`` and ``angle`` is 0.
    """
    if geometry.is_empty:
        nan = float("nan")
        return (nan, nan, nan, nan, 0.0)

    if not rotated:
        minx, miny, maxx, maxy = geometry.bounds
        return (float(minx), float(miny), float(maxx), float(maxy), 0.0)

    from shapely.geometry import Polygon

    mrr = geometry.minimum_rotated_rectangle
    if not isinstance(mrr, Polygon) or mrr.is_empty:
        # Degenerate MRR (Point/LineString): fall back to axis-aligned bounds.
        minx, miny, maxx, maxy = geometry.bounds
        return (float(minx), float(miny), float(maxx), float(maxy), 0.0)

    pts = np.asarray(mrr.exterior.coords[:4], dtype=float)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    edge0 = pts[1] - pts[0]
    edge1 = pts[2] - pts[1]
    width = float(np.hypot(edge0[0], edge0[1]))
    height = float(np.hypot(edge1[0], edge1[1]))
    angle = float(np.arctan2(edge0[1], edge0[0]))
    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2
    return (x1, y1, x2, y2, angle)


@attrs.define(eq=False)
class UserROI(ROI):
    """Human-annotated region of interest."""

    pass


@attrs.define(eq=False)
class PredictedROI(ROI):
    """Model-predicted region of interest with confidence score.

    Attributes:
        score: Confidence score (0-1).
    """

    score: float = attrs.field(default=0.0)
