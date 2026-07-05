"""Tests for the `Embedding` data structure."""

import copy

import numpy as np
import pytest

from sleap_io.model.embedding import Embedding


def test_embedding_basic():
    e = Embedding(np.arange(8, dtype=np.float32))
    assert e.dim == 8
    assert e.vector.dtype == np.float32
    assert np.array_equal(e.vector, np.arange(8))


def test_embedding_dtype_preserved():
    e64 = Embedding(np.arange(4, dtype=np.float64))
    assert e64.vector.dtype == np.float64
    # non-floating inputs are cast to float32
    eint = Embedding(np.arange(4, dtype=np.int32))
    assert eint.vector.dtype == np.float32


def test_embedding_rejects_non_1d():
    with pytest.raises(ValueError):
        Embedding(np.zeros((2, 2), dtype=np.float32))


def test_embedding_value_equality():
    a = Embedding(np.arange(4, dtype=np.float32))
    b = Embedding(np.arange(4, dtype=np.float32))
    c = Embedding(np.ones(4, dtype=np.float32))
    assert a == b  # value equality on the vector
    assert a != c


def test_embedding_unhashable():
    with pytest.raises(TypeError):
        hash(Embedding(np.arange(4, dtype=np.float32)))


def test_embedding_has_no_extra_fields():
    e = Embedding(np.arange(4, dtype=np.float32))
    for attr in ("name", "normalized", "source", "centroid_xy", "metadata"):
        assert not hasattr(e, attr)


def test_embedding_repr_is_concise():
    r = repr(Embedding(np.arange(2048, dtype=np.float32)))
    assert "Embedding" in r and "2048" in r
    # The full vector must not be dumped into the repr.
    assert len(r) < 60


def test_embedding_deepcopy():
    a = Embedding(np.arange(4, dtype=np.float32))
    b = copy.deepcopy(a)
    assert b == a
    assert b.vector is not a.vector
