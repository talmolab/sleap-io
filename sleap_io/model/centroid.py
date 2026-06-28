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

from sleap_io.model.embedding import EmbeddingMixin

if TYPE_CHECKING:
    from sleap_io.model.embedding import Embedding
    from sleap_io.model.identity import Identity
    from sleap_io.model.instance import Instance, PredictedInstance, Track
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


@attrs.define(eq=False)
class Centroid(EmbeddingMixin):
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
        category: Class label (e.g., ``"lysosome"``, ``"cell"``).
        name: Human-readable name (e.g., ``"ID43008"``).
        source: How the centroid was computed (e.g., ``"center_of_mass"``,
            ``"trackmate"``).
        embeddings: Mapping from embedding-space name to an `Embedding` describing
            this detection's appearance for re-identification. Empty by default.

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
    category: str = attrs.field(default="")
    name: str = attrs.field(default="")
    source: str = attrs.field(default="")
    embeddings: dict[str, Embedding] = attrs.field(factory=dict, repr=False)

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

    def to_instance(
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
            )
        else:
            return Instance.from_numpy(
                points_data=points,
                skeleton=skeleton,
                track=self.track,
                tracking_score=self.tracking_score,
                identity=self.identity,
                identity_score=self.identity_score,
            )

    @classmethod
    def from_instance(
        cls,
        instance: "Instance",
        method: str = "center_of_mass",
        node: "str | int | None" = None,
        **kwargs,
    ) -> "Centroid":
        """Create a centroid from an ``Instance``.

        Args:
            instance: The source instance.
            method: Computation method:
                - ``"center_of_mass"``: Mean of visible point coordinates.
                - ``"bbox_center"``: Center of bounding box of visible points.
                - ``"anchor"``: Coordinates of a specific node.
            node: Node specification for ``"anchor"`` method. Can be a node
                name (str) or index (int).
            **kwargs: Additional keyword arguments passed to the centroid
                constructor (e.g., ``video``, ``frame_idx``, ``category``).

        Returns:
            A ``PredictedCentroid`` if the instance is a ``PredictedInstance``,
            otherwise a ``UserCentroid``.

        Raises:
            ValueError: If no visible points, or invalid method/node.
        """
        from sleap_io.model.instance import PredictedInstance

        pts = instance.numpy(invisible_as_nan=True)
        visible = ~np.isnan(pts[:, 0])

        if method == "center_of_mass":
            if not visible.any():
                raise ValueError("No visible points for center_of_mass.")
            x = float(pts[visible, 0].mean())
            y = float(pts[visible, 1].mean())

        elif method == "bbox_center":
            if not visible.any():
                raise ValueError("No visible points for bbox_center.")
            x = float((pts[visible, 0].min() + pts[visible, 0].max()) / 2)
            y = float((pts[visible, 1].min() + pts[visible, 1].max()) / 2)

        elif method == "anchor":
            if node is None:
                raise ValueError("Must specify 'node' for anchor method.")
            if isinstance(node, str):
                node_idx = instance.skeleton.index(node)
            elif isinstance(node, int):
                node_idx = node
            else:
                raise ValueError(f"node must be str or int, got {type(node).__name__}")
            if np.isnan(pts[node_idx, 0]):
                raise ValueError(
                    f"Anchor node {node!r} is not visible in this instance."
                )
            x = float(pts[node_idx, 0])
            y = float(pts[node_idx, 1])

        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Expected 'center_of_mass', 'bbox_center', or 'anchor'."
            )

        # Build constructor kwargs.
        centroid_kwargs = dict(
            x=x,
            y=y,
            track=instance.track,
            tracking_score=instance.tracking_score,
            identity=instance.identity,
            identity_score=instance.identity_score,
            instance=instance,
            source=method if method != "anchor" else f"anchor:{node}",
        )
        centroid_kwargs.update(kwargs)

        if isinstance(instance, PredictedInstance):
            return PredictedCentroid(score=instance.score, **centroid_kwargs)
        else:
            return UserCentroid(**centroid_kwargs)


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
