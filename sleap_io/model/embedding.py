"""Embedding data structure for per-detection appearance / re-ID vectors."""

from __future__ import annotations

import attrs
import numpy as np
from attrs import define, field


def _as_vector(value) -> np.ndarray:
    """Coerce an input to a 1-D embedding vector, preserving floating dtype.

    Floating inputs (e.g. float32 from a neural network, float64 from JABS
    prototypes) keep their dtype; non-floating inputs are cast to float32.
    """
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    if arr.ndim != 1:
        raise ValueError(
            f"Embedding vector must be 1-dimensional, got shape {arr.shape}."
        )
    return arr


def _as_optional_xy(value) -> np.ndarray | None:
    """Coerce an optional source location to a ``(2,)`` float32 array or None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"centroid_xy must have shape (2,), got {arr.shape}.")
    return arr


@define(eq=False)
class Embedding:
    """A per-detection appearance / re-identification embedding vector.

    An `Embedding` wraps a single feature vector describing the visual appearance
    of one detection (e.g. the crop around an `Instance`, `Centroid`,
    `SegmentationMask`, or `BoundingBox`). Embeddings attach to any detection
    modality via its ``embeddings`` mapping (keyed by space name) and to
    `Identity` as a gallery / prototype vector. They are compared by cosine
    similarity for re-identification; when produced by a normalized model the
    vector lies on the unit hypersphere, so cosine similarity reduces to a dot
    product.

    Attributes:
        vector: 1-D feature vector of shape ``(D,)``. The dtype is preserved from
            the input when floating (e.g. float32 from a neural network, float64
            from JABS prototypes); non-floating inputs are cast to float32.
        name: Embedding-space name (e.g. ``"reid"``, ``"jabs"``). Multiple spaces
            may coexist on a single detection.
        normalized: Whether ``vector`` is L2-normalized (cosine geometry). Defaults
            to True to match typical re-ID model outputs.
        source: Optional provenance / model identifier that produced this vector.
        centroid_xy: Optional ``(2,)`` source location (e.g. the centroid the crop
            was sampled at, as recorded by centroid-driven embedding inference).
        metadata: Arbitrary metadata dictionary.
    """

    vector: np.ndarray = field(
        converter=_as_vector,
        eq=attrs.cmp_using(eq=np.array_equal),
    )
    name: str = "reid"
    normalized: bool = True
    source: str | None = None
    centroid_xy: np.ndarray | None = field(default=None, converter=_as_optional_xy)
    metadata: dict = field(factory=dict)

    @property
    def dim(self) -> int:
        """Dimensionality ``D`` of the embedding vector."""
        return int(self.vector.shape[0])

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the stored vector."""
        return self.vector.dtype

    def __repr__(self) -> str:
        """Return a readable string representation."""
        parts = [f'Embedding(name="{self.name}", dim={self.dim}']
        parts.append(f", normalized={self.normalized}")
        if self.source is not None:
            parts.append(f', source="{self.source}"')
        parts.append(")")
        return "".join(parts)


class EmbeddingMixin:
    """Mixin adding per-detection embedding storage helpers.

    Classes using this mixin must declare an ``embeddings: dict[str, Embedding]``
    attrs field. The mixin adds a convenience ``embedding`` accessor and a
    ``set_embedding`` helper without storing any state of its own (``__slots__``
    is empty so slotted attrs subclasses keep their slots).
    """

    __slots__ = ()

    @property
    def embedding(self) -> Embedding | None:
        """The primary embedding, or None.

        Returns the ``"reid"`` embedding if present; otherwise the sole embedding
        if exactly one exists; otherwise None.
        """
        embeddings = self.embeddings
        if "reid" in embeddings:
            return embeddings["reid"]
        if len(embeddings) == 1:
            return next(iter(embeddings.values()))
        return None

    @embedding.setter
    def embedding(self, value: Embedding | None) -> None:
        if value is None:
            self.embeddings.pop("reid", None)
        else:
            self.embeddings[value.name] = value

    def set_embedding(
        self,
        vector,
        name: str = "reid",
        normalized: bool = True,
        source: str | None = None,
        centroid_xy=None,
    ) -> Embedding:
        """Create and attach an `Embedding` to this detection.

        Args:
            vector: 1-D feature vector of shape ``(D,)``.
            name: Embedding-space name to store under. Defaults to ``"reid"``.
            normalized: Whether the vector is L2-normalized.
            source: Optional model/provenance identifier.
            centroid_xy: Optional ``(2,)`` source location.

        Returns:
            The created `Embedding`.
        """
        emb = Embedding(
            vector=vector,
            name=name,
            normalized=normalized,
            source=source,
            centroid_xy=centroid_xy,
        )
        self.embeddings[name] = emb
        return emb
