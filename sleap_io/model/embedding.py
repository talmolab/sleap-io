"""Embedding data structure for per-detection appearance / re-ID vectors."""

from __future__ import annotations

import attrs
import numpy as np
from attrs import define, field


def _as_vector(value) -> np.ndarray:
    """Coerce an input to a 1-D embedding vector, preserving floating dtype.

    Floating inputs (e.g. float32 from a neural network) keep their dtype;
    non-floating inputs are cast to float32.
    """
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    if arr.ndim != 1:
        raise ValueError(
            f"Embedding vector must be 1-dimensional, got shape {arr.shape}."
        )
    return arr


@define
class Embedding:
    """A per-detection appearance / re-identification embedding vector.

    An `Embedding` wraps a single feature vector describing the visual appearance
    of one detection (e.g. the crop around an `Instance`, `Centroid`,
    `SegmentationMask`, or `BoundingBox`). What the vector represents is implied by
    the slot it fills on the detection -- currently ``identity_embedding`` on every
    detection modality, used for re-identification.

    Attributes:
        vector: 1-D feature vector of shape ``(D,)``. The dtype is preserved from
            the input when floating (e.g. float32 from a neural network);
            non-floating inputs are cast to float32.

    Notes:
        `Embedding` uses value equality: two embeddings are equal when their
        vectors are element-wise equal. It is therefore unhashable (an `Embedding`
        is only ever a field value, never a set member or dict key). Detection
        objects use object-identity equality, so a (potentially large) embedding
        vector is never compared when two detections are compared.
    """

    vector: np.ndarray = field(
        converter=_as_vector,
        eq=attrs.cmp_using(eq=np.array_equal),
        repr=lambda v: f"<{v.shape[0]}-d {v.dtype}>",
    )

    @property
    def dim(self) -> int:
        """Dimensionality ``D`` of the embedding vector."""
        return int(self.vector.shape[0])
