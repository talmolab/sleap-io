"""Data structures for centroid annotations.

Centroids are lightweight point annotations representing the center of an object.
They support user/predicted distinction and interconversion with single-node
``Instance`` objects.

The class hierarchy:
    - ``Centroid`` — abstract base with coordinates, video/frame/track/instance metadata
    - ``UserCentroid`` — human-annotated or derived centroid
    - ``PredictedCentroid`` — model-predicted centroid with confidence score

A module-level ``CENTROID_SKELETON`` is provided for creating single-node
``Instance`` objects from centroids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

from sleap_io.model.category import to_category

if TYPE_CHECKING:
    from sleap_io.model.bbox import BoundingBox
    from sleap_io.model.category import Category
    from sleap_io.model.embedding import Embedding
    from sleap_io.model.identity import Identity
    from sleap_io.model.instance import Instance, PredictedInstance, Track
    from sleap_io.model.mask import SegmentationMask
    from sleap_io.model.roi import ROI
    from sleap_io.model.skeleton import Skeleton


def _make_centroid_skeleton():
    """Create the shared single-node skeleton for centroid instances.

    This is deferred to avoid circular imports at module load time.
    """
    from sleap_io.model.skeleton import Skeleton

    return Skeleton(["centroid"])


# Module-level shared skeleton. Lazily initialized on first access.
_CENTROID_SKELETON = None


def get_centroid_skeleton() -> "Skeleton":
    """Return the shared single-node ``Skeleton(["centroid"])`` instance.

    All centroid-to-instance conversions share this skeleton so that
    ``Labels.skeletons`` contains a single entry.
    """
    global _CENTROID_SKELETON
    if _CENTROID_SKELETON is None:
        _CENTROID_SKELETON = _make_centroid_skeleton()
    return _CENTROID_SKELETON


# Backwards-compatible module-level attribute. Accessing ``CENTROID_SKELETON``
# returns the lazily-created skeleton via ``__getattr__``.
def __getattr__(name):
    if name == "CENTROID_SKELETON":
        return get_centroid_skeleton()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _geometric_median(pts: np.ndarray) -> tuple[float, float]:
    """Compute the geometric median of 2D points via Weiszfeld's algorithm.

    The geometric median minimizes the sum of Euclidean distances to all input
    points. It is computed iteratively (pure numpy, no scipy) starting from the
    arithmetic mean and reweighting by inverse distance until convergence.

    Args:
        pts: An ``(k, 2)`` array of visible point coordinates. Must contain at
            least one point.

    Returns:
        The geometric median as an ``(x, y)`` tuple of floats.
    """
    pts = np.asarray(pts, dtype=float)
    estimate = pts.mean(axis=0)
    for _ in range(100):
        dist = np.linalg.norm(pts - estimate, axis=1)
        nonzero = dist > 1e-12
        if not nonzero.any():
            # All points coincide with the current estimate.
            break
        weights = 1.0 / dist[nonzero]
        new_estimate = (weights[:, None] * pts[nonzero]).sum(axis=0) / weights.sum()
        if np.linalg.norm(new_estimate - estimate) < 1e-6:
            estimate = new_estimate
            break
        estimate = new_estimate
    return float(estimate[0]), float(estimate[1])


@attrs.define(eq=False)
class Centroid:
    """A point representing the center of an object.

    Supports optional 3D coordinates, track/instance metadata,
    and interconversion with single-node ``Instance`` objects.

    Attributes:
        x: X-coordinate in pixel space.
        y: Y-coordinate in pixel space.
        z: Optional Z-coordinate for 3D data. ``None`` for 2D.
        track: Optional tracking identity.
        tracking_score: Confidence of the track identity assignment. ``None``
            if unassigned or manually assigned.
        identity: Optional global, ground-truth `Identity` for this centroid -- the
            persistent cross-video animal identity / re-identification key. ``None``
            if no global identity is assigned. Mirrors `Instance.identity`.
        identity_score: Score associated with the `identity` assignment (e.g. the
            re-ID match similarity). ``None`` if unassigned or assigned manually.
            Kept separate from `tracking_score` (short-term tracklet vs long-term
            identity).
        instance: Optional linked pose instance.
        category: Optional `Category` (class label, e.g. ``"lysosome"``,
            ``"cell"``) for this centroid. Promoted from the legacy free-form
            string; ``None`` if unset. Mirrors `Instance.category`.
        name: Human-readable name (e.g., ``"ID43008"``).
        source: How the centroid was computed (e.g., ``"center_of_mass"``,
            ``"trackmate"``).
        identity_embedding: Optional `Embedding` describing this detection's
            appearance for re-identification. ``None`` by default.
        category_score: Score associated with the `category` assignment (e.g. the
            classifier confidence). ``None`` if unassigned or assigned manually.
        category_embedding: Optional `Embedding` describing this detection's
            appearance for classification. ``None`` by default.

    Notes:
        Centroids use identity-based equality (two Centroid objects are only
        equal if they are the same object in memory).

        This class is abstract. Use ``UserCentroid`` or ``PredictedCentroid``
        instead.
    """

    x: float = attrs.field()
    y: float = attrs.field()
    z: float | None = attrs.field(default=None)
    track: "Track | None" = attrs.field(default=None)
    tracking_score: float | None = attrs.field(default=None)
    identity: "Identity | None" = attrs.field(default=None)
    identity_score: float | None = attrs.field(default=None)
    instance: "Instance | None" = attrs.field(default=None)
    category: "Category | None" = attrs.field(default=None, converter=to_category)
    name: str = attrs.field(default="")
    source: str = attrs.field(default="")
    identity_embedding: "Embedding | None" = attrs.field(default=None, repr=False)
    category_score: float | None = attrs.field(default=None)
    category_embedding: "Embedding | None" = attrs.field(default=None, repr=False)

    # Private: deferred instance index for lazy loading.
    _instance_idx: int = attrs.field(default=-1, repr=False, eq=False, init=False)

    def __attrs_post_init__(self):
        """Validate that this class is not instantiated directly."""
        if type(self) is Centroid:
            raise TypeError(
                "Centroid is abstract. Use UserCentroid or PredictedCentroid."
            )

    @property
    def xy(self) -> tuple[float, float]:
        """Return coordinates as ``(x, y)``."""
        return (self.x, self.y)

    @property
    def yx(self) -> tuple[float, float]:
        """Return coordinates as ``(y, x)`` (row, col order)."""
        return (self.y, self.x)

    @property
    def xyz(self) -> tuple[float, float, float | None]:
        """Return coordinates as ``(x, y, z)``."""
        return (self.x, self.y, self.z)

    @property
    def is_predicted(self) -> bool:
        """Return ``True`` if this is a ``PredictedCentroid``."""
        return isinstance(self, PredictedCentroid)

    @property
    def is_empty(self) -> bool:
        """Whether this centroid is degenerate (NaN ``x`` or ``y``)."""
        return bool(np.isnan(self.x) or np.isnan(self.y))

    def to_pose(
        self, skeleton: "Skeleton | None" = None
    ) -> "Instance | PredictedInstance":
        """Convert this centroid to a single-node ``Instance``.

        Args:
            skeleton: Skeleton to use for the instance. Must have exactly one
                node. Defaults to the shared ``CENTROID_SKELETON``.

        Returns:
            A ``PredictedInstance`` if this is a ``PredictedCentroid``,
            otherwise an ``Instance``.

        Raises:
            ValueError: If the skeleton has more than one node.
        """
        from sleap_io.model.instance import Instance, PredictedInstance

        if skeleton is None:
            skeleton = get_centroid_skeleton()

        if len(skeleton) > 1:
            raise ValueError(
                f"Skeleton must have exactly 1 node for centroid conversion, "
                f"got {len(skeleton)}."
            )

        points = np.array([[self.x, self.y]])

        if isinstance(self, PredictedCentroid):
            return PredictedInstance.from_numpy(
                points_data=points,
                skeleton=skeleton,
                score=self.score,
                track=self.track,
                tracking_score=self.tracking_score,
                identity=self.identity,
                identity_score=self.identity_score,
                identity_embedding=self.identity_embedding,
                category=self.category,
                category_score=self.category_score,
                category_embedding=self.category_embedding,
            )
        else:
            return Instance.from_numpy(
                points_data=points,
                skeleton=skeleton,
                track=self.track,
                tracking_score=self.tracking_score,
                identity=self.identity,
                identity_score=self.identity_score,
                identity_embedding=self.identity_embedding,
                category=self.category,
                category_score=self.category_score,
                category_embedding=self.category_embedding,
            )

    def to_instance(
        self, skeleton: "Skeleton | None" = None
    ) -> "Instance | PredictedInstance":
        """Convert this centroid to a single-node ``Instance`` (deprecated).

        .. deprecated::
            Use :meth:`to_pose` instead.

        Args:
            skeleton: Skeleton to use for the instance. Must have exactly one
                node. Defaults to the shared ``CENTROID_SKELETON``.

        Returns:
            A ``PredictedInstance`` if this is a ``PredictedCentroid``,
            otherwise an ``Instance``.
        """
        import warnings

        warnings.warn(
            "Centroid.to_instance() is deprecated; use Centroid.to_pose() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_pose(skeleton=skeleton)

    @classmethod
    def from_pose(
        cls,
        instance: "Instance",
        method: str = "center_of_mass",
        node: "str | int | None" = None,
        fallback: "str | None" = None,
        error_on_empty: bool = False,
        **kwargs,
    ) -> "Centroid":
        """Create a centroid from a pose ``Instance``.

        Args:
            instance: The source instance.
            method: Computation method:
                - ``"center_of_mass"``: NaN-ignoring unweighted mean of visible
                  node coordinates.
                - ``"bbox_center"``: Center of the bounding box of visible points.
                - ``"geometric_median"``: Weiszfeld geometric median of visible
                  points (robust to outliers).
                - ``"anchor"``: Coordinates of a specific node (requires ``node``).
            node: Node specification for the ``"anchor"`` method. Can be a node
                name (str) or index (int). Required for ``"anchor"``.
            fallback: For the ``"anchor"`` method, a non-anchor method
                (``"center_of_mass"``, ``"bbox_center"``, or
                ``"geometric_median"``) to fall back to when the anchor node is
                occluded. If ``None``, an occluded anchor yields a degenerate
                centroid (or raises when ``error_on_empty`` is ``True``).
            error_on_empty: If ``True``, raise ``ValueError`` when there are no
                visible points to compute the requested centroid instead of
                returning a degenerate (NaN) centroid.
            **kwargs: Additional keyword arguments passed to the centroid
                constructor (e.g., ``video``, ``frame_idx``, ``category``).

        Returns:
            A ``PredictedCentroid`` if the instance is a ``PredictedInstance``,
            otherwise a ``UserCentroid``. The ``source`` attribute records the
            computation method (e.g. ``"center_of_mass"``, ``"anchor:nose"``, or
            ``"anchor:nose->center_of_mass"`` when a fallback was used).

        Raises:
            ValueError: For an unknown ``method``, a missing ``node`` for the
                ``"anchor"`` method, an invalid ``node`` type, or (when
                ``error_on_empty`` is ``True``) when there are no visible points.
        """
        from sleap_io.model.instance import PredictedInstance

        pts = instance.numpy(invisible_as_nan=True)
        visible = ~np.isnan(pts[:, 0])
        nan = float("nan")

        def _compute(reduce_method: str) -> tuple[float, float]:
            """Compute a non-anchor centroid; returns NaN if no visible points."""
            if reduce_method == "center_of_mass":
                if not visible.any():
                    return nan, nan
                return (
                    float(pts[visible, 0].mean()),
                    float(pts[visible, 1].mean()),
                )
            elif reduce_method == "bbox_center":
                if not visible.any():
                    return nan, nan
                return (
                    float((pts[visible, 0].min() + pts[visible, 0].max()) / 2),
                    float((pts[visible, 1].min() + pts[visible, 1].max()) / 2),
                )
            elif reduce_method == "geometric_median":
                if not visible.any():
                    return nan, nan
                return _geometric_median(pts[visible])
            else:
                raise ValueError(
                    f"Unknown method {reduce_method!r}. Expected 'center_of_mass', "
                    f"'bbox_center', 'geometric_median', or 'anchor'."
                )

        if method == "anchor":
            if node is None:
                raise ValueError("Must specify 'node' for anchor method.")
            if isinstance(node, str):
                node_idx = instance.skeleton.index(node)
            elif isinstance(node, (int, np.integer)):
                node_idx = int(node)
            else:
                raise ValueError(f"node must be str or int, got {type(node).__name__}")

            if not np.isnan(pts[node_idx, 0]):
                x = float(pts[node_idx, 0])
                y = float(pts[node_idx, 1])
                source = f"anchor:{node}"
            elif fallback is not None:
                x, y = _compute(fallback)
                source = f"anchor:{node}->{fallback}"
            else:
                x = y = nan
                source = f"anchor:{node}"
        elif method in ("center_of_mass", "bbox_center", "geometric_median"):
            x, y = _compute(method)
            source = method
        else:
            raise ValueError(
                f"Unknown method {method!r}. Expected 'center_of_mass', "
                f"'bbox_center', 'geometric_median', or 'anchor'."
            )

        if (np.isnan(x) or np.isnan(y)) and error_on_empty:
            raise ValueError(
                f"No visible points to compute centroid (method={method!r})."
            )

        # Build constructor kwargs.
        centroid_kwargs = dict(
            x=x,
            y=y,
            track=instance.track,
            tracking_score=instance.tracking_score,
            identity=instance.identity,
            identity_score=instance.identity_score,
            identity_embedding=instance.identity_embedding,
            category=instance.category,
            category_score=instance.category_score,
            category_embedding=instance.category_embedding,
            instance=instance,
            source=source,
        )
        centroid_kwargs.update(kwargs)

        if isinstance(instance, PredictedInstance):
            return PredictedCentroid(score=instance.score, **centroid_kwargs)
        else:
            return UserCentroid(**centroid_kwargs)

    @classmethod
    def from_instance(
        cls,
        instance: "Instance",
        method: str = "center_of_mass",
        node: "str | int | None" = None,
        fallback: "str | None" = None,
        error_on_empty: bool = False,
        **kwargs,
    ) -> "Centroid":
        """Create a centroid from an ``Instance`` (deprecated).

        .. deprecated::
            Use :meth:`from_pose` instead.

        Args:
            instance: The source instance.
            method: Computation method (see :meth:`from_pose`).
            node: Node specification for the ``"anchor"`` method.
            fallback: Fallback method for an occluded anchor.
            error_on_empty: Whether to raise instead of returning a degenerate
                centroid.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A ``PredictedCentroid`` or ``UserCentroid`` (see :meth:`from_pose`).
        """
        import warnings

        warnings.warn(
            "Centroid.from_instance() is deprecated; use Centroid.from_pose() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_pose(
            instance,
            method=method,
            node=node,
            fallback=fallback,
            error_on_empty=error_on_empty,
            **kwargs,
        )

    def to_bbox(
        self,
        size: float | tuple[float, float],
        padding: float | tuple[float, float] = 0.0,
        error_on_empty: bool = False,
    ) -> "BoundingBox":
        """Construct a fixed-size bounding box centered on this centroid.

        A ``PredictedCentroid`` produces a ``PredictedBoundingBox`` carrying its
        ``score``; any other centroid produces a ``UserBoundingBox``. Metadata
        (track, tracking_score, identity, identity_score, category, name, source,
        instance) is inherited.

        Args:
            size: Box size centered on the centroid. A scalar yields a square box
                of that side length; a ``(w, h)`` tuple sets width and height
                independently. Required.
            padding: Amount to inflate the box outward after sizing. Scalar
                applies to both axes; a ``(px, py)`` tuple applies per-axis.
                Negative values shrink the box.
            error_on_empty: If ``True``, raise ``ValueError`` when this centroid
                is degenerate (NaN) instead of returning a degenerate box.

        Returns:
            A ``BoundingBox`` centered on the centroid (or NaN corners if empty).

        Raises:
            ValueError: If ``size`` is ``None``, or if the centroid is degenerate
                and ``error_on_empty`` is ``True``.
        """
        if size is None:
            raise ValueError("'size' is required for Centroid.to_bbox().")

        from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
        from sleap_io.model.roi import _apply_padding

        if self.is_empty:
            if error_on_empty:
                raise ValueError(
                    "Cannot compute bounding box of a degenerate (NaN) centroid."
                )
            nan = float("nan")
            x1 = y1 = x2 = y2 = nan
        else:
            if isinstance(size, (tuple, list)):
                w, h = size
            else:
                w = h = size
            x1 = self.x - w / 2
            y1 = self.y - h / 2
            x2 = self.x + w / 2
            y2 = self.y + h / 2
            x1, y1, x2, y2 = _apply_padding(x1, y1, x2, y2, padding)

        kwargs = dict(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            angle=0.0,
            track=self.track,
            tracking_score=self.tracking_score,
            identity=self.identity,
            identity_score=self.identity_score,
            identity_embedding=self.identity_embedding,
            instance=self.instance,
            category=self.category,
            category_score=self.category_score,
            category_embedding=self.category_embedding,
            name=self.name,
            source=self.source,
        )
        if self.is_predicted:
            return PredictedBoundingBox(score=self.score, **kwargs)
        return UserBoundingBox(**kwargs)

    def to_roi(self, radius: float, error_on_empty: bool = False) -> "ROI":
        """Construct a circular ROI centered on this centroid.

        A ``PredictedCentroid`` produces a ``PredictedROI`` carrying its
        ``score``; any other centroid produces a ``UserROI``. Metadata (track,
        tracking_score, identity, identity_score, category, name, source,
        instance) is inherited.

        Args:
            radius: Radius of the circular ROI. Required.
            error_on_empty: If ``True``, raise ``ValueError`` when this centroid
                is degenerate (NaN) instead of returning an empty-geometry ROI.

        Returns:
            A ``ROI`` with a buffered-point (circular) geometry, or an empty
            ``Polygon`` geometry if the centroid is degenerate.

        Raises:
            ValueError: If the centroid is degenerate and ``error_on_empty`` is
                ``True``.
        """
        from shapely.geometry import Point, Polygon

        from sleap_io.model.roi import PredictedROI, UserROI

        if self.is_empty:
            if error_on_empty:
                raise ValueError("Cannot compute ROI of a degenerate (NaN) centroid.")
            geom = Polygon()
        else:
            geom = Point(self.x, self.y).buffer(radius)

        kwargs = dict(
            geometry=geom,
            track=self.track,
            tracking_score=self.tracking_score,
            identity=self.identity,
            identity_score=self.identity_score,
            identity_embedding=self.identity_embedding,
            instance=self.instance,
            category=self.category,
            category_score=self.category_score,
            category_embedding=self.category_embedding,
            name=self.name,
            source=self.source,
        )
        if self.is_predicted:
            return PredictedROI(score=self.score, **kwargs)
        return UserROI(**kwargs)

    def to_mask(
        self,
        height: int,
        width: int,
        radius: float,
        error_on_empty: bool = False,
    ) -> "SegmentationMask":
        """Rasterize a circular ROI around this centroid into a mask.

        Equivalent to ``self.to_roi(radius).to_mask(height, width)``. A
        ``PredictedCentroid`` produces a ``PredictedSegmentationMask`` carrying
        its ``score``; any other centroid produces a ``UserSegmentationMask``.
        Metadata is inherited.

        Args:
            height: Height of the output mask in pixels.
            width: Width of the output mask in pixels.
            radius: Radius of the circular region around the centroid. Required.
            error_on_empty: If ``True``, raise ``ValueError`` when this centroid
                is degenerate (NaN) instead of returning an all-background mask.

        Returns:
            A ``SegmentationMask`` with the rasterized circular region (all
            background if the centroid is degenerate).

        Raises:
            ValueError: If the centroid is degenerate and ``error_on_empty`` is
                ``True``.
        """
        if self.is_empty:
            if error_on_empty:
                raise ValueError("Cannot compute mask of a degenerate (NaN) centroid.")
            from sleap_io.model.mask import (
                PredictedSegmentationMask,
                UserSegmentationMask,
            )

            empty = np.zeros((height, width), dtype=bool)
            kwargs = dict(
                track=self.track,
                tracking_score=self.tracking_score,
                identity=self.identity,
                identity_score=self.identity_score,
                identity_embedding=self.identity_embedding,
                instance=self.instance,
                category=self.category,
                category_score=self.category_score,
                category_embedding=self.category_embedding,
                name=self.name,
                source=self.source,
            )
            if self.is_predicted:
                return PredictedSegmentationMask.from_numpy(
                    empty, score=self.score, **kwargs
                )
            return UserSegmentationMask.from_numpy(empty, **kwargs)

        return self.to_roi(radius).to_mask(height, width)


@attrs.define(eq=False)
class UserCentroid(Centroid):
    """A human-annotated or derived centroid.

    Inherits all fields from ``Centroid``. Has no additional fields.

    See ``Centroid`` for attribute documentation.
    """

    pass


@attrs.define(eq=False)
class PredictedCentroid(Centroid):
    """A model-predicted centroid with a confidence score.

    Attributes:
        score: Detection confidence score (0-1).

    See ``Centroid`` for other attribute documentation.
    """

    score: float = attrs.field(default=0.0)
