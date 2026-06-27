"""Tests for the Embedding data structure and per-modality embedding slots."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from sleap_io.model.bbox import PredictedBoundingBox
from sleap_io.model.centroid import UserCentroid
from sleap_io.model.embedding import Embedding, EmbeddingMixin
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.roi import PredictedROI
from sleap_io.model.skeleton import Skeleton


def test_embedding_basic():
    """Test Embedding construction, dim, dtype, and defaults."""
    emb = Embedding(np.ones(128, dtype=np.float32))
    assert emb.vector.shape == (128,)
    assert emb.dim == 128
    assert emb.dtype == np.float32
    assert emb.name == "reid"
    assert emb.normalized is True
    assert emb.source is None
    assert emb.centroid_xy is None
    assert emb.metadata == {}


def test_embedding_dtype_preserved():
    """Test that floating dtype is preserved (e.g. JABS float64 prototypes)."""
    f64 = Embedding(np.ones(4, dtype=np.float64))
    assert f64.dtype == np.float64

    # Non-floating input is cast to float32.
    from_list = Embedding([1, 2, 3])
    assert from_list.dtype == np.float32
    np.testing.assert_array_equal(from_list.vector, [1.0, 2.0, 3.0])


def test_embedding_rejects_non_1d():
    """Test that a non-1-D vector raises rather than silently flattening."""
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        Embedding(np.ones((2, 3)))


def test_embedding_centroid_xy():
    """Test centroid_xy coercion and shape validation."""
    emb = Embedding(np.ones(8), centroid_xy=[5, 6])
    assert emb.centroid_xy.shape == (2,)
    assert emb.centroid_xy.dtype == np.float32
    np.testing.assert_array_equal(emb.centroid_xy, [5.0, 6.0])

    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        Embedding(np.ones(8), centroid_xy=[1, 2, 3])


def test_embedding_eq_false():
    """Test that Embedding uses object-identity equality."""
    e1 = Embedding(np.ones(4))
    e2 = Embedding(np.ones(4))
    assert e1 is not e2
    assert e1 != e2


def test_embedding_repr():
    """Test the Embedding repr."""
    assert (
        repr(Embedding(np.ones(3))) == 'Embedding(name="reid", dim=3, normalized=True)'
    )
    emb = Embedding(np.ones(3), name="jabs", normalized=False, source="model_v1")
    assert (
        repr(emb)
        == 'Embedding(name="jabs", dim=3, normalized=False, source="model_v1")'
    )


def test_embedding_accessor_on_instance():
    """Test the embedding accessor / set_embedding on an Instance."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)

    # No embeddings -> None.
    assert inst.embeddings == {}
    assert inst.embedding is None

    # set_embedding stores under name and returns the Embedding.
    emb = inst.set_embedding(np.arange(128, dtype=np.float32))
    assert isinstance(emb, Embedding)
    assert inst.embedding is emb
    assert list(inst.embeddings) == ["reid"]

    # A second named space coexists; the accessor still prefers "reid".
    inst.set_embedding(np.zeros(64), name="jabs")
    assert set(inst.embeddings) == {"reid", "jabs"}
    assert inst.embedding.name == "reid"


def test_embedding_accessor_sole_and_setter():
    """Test the accessor falls back to the sole space, and the setter."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)

    # Sole non-reid space -> returned by accessor.
    inst.set_embedding(np.ones(32), name="jabs")
    assert inst.embedding.name == "jabs"

    # Setter stores under the embedding's own name.
    inst.embedding = Embedding(np.ones(16), name="reid")
    assert inst.embeddings["reid"].dim == 16

    # Setting None removes the "reid" entry.
    inst.embedding = None
    assert "reid" not in inst.embeddings


def test_embedding_on_predicted_instance():
    """Test embeddings work on PredictedInstance."""
    skel = Skeleton(["A", "B"])
    pred = PredictedInstance.from_numpy(np.array([[0, 1], [2, 3]]), skel, score=0.5)
    pred.set_embedding(np.ones(128))
    assert pred.embedding.dim == 128


def test_embedding_on_identity():
    """Test Identity carries a prototype/gallery embedding."""
    identity = Identity(name="mouse_A")
    assert identity.embeddings == {}
    identity.set_embedding(np.ones(128), source="gallery")
    assert identity.embedding.dim == 128
    assert identity.embedding.source == "gallery"


def test_embedding_on_all_modalities():
    """Test the embedding slot exists and works on every detection modality."""
    detections = [
        UserCentroid(x=1.0, y=2.0),
        PredictedBoundingBox(x1=0, y1=0, x2=10, y2=10, score=0.5),
        PredictedROI(geometry=box(0, 0, 10, 10), score=0.5),
    ]
    for det in detections:
        assert isinstance(det, EmbeddingMixin)
        assert det.embeddings == {}
        det.set_embedding(np.ones(64))
        assert det.embedding.dim == 64
        # Slotted classes must not gain a __dict__.
        assert not hasattr(det, "__dict__")


def test_embedding_preserved_on_copy():
    """Test embeddings survive a deep copy of the owning instance."""
    from copy import deepcopy

    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    inst.set_embedding(np.arange(8, dtype=np.float32))
    copied = deepcopy(inst)
    assert copied.embedding is not None
    assert copied.embedding.dim == 8
    np.testing.assert_array_equal(copied.embedding.vector, np.arange(8))
