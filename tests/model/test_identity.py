"""Tests for the `Identity` data structure."""

import copy

import pytest

from sleap_io.model.identity import Identity


def test_identity_defaults():
    ident = Identity()
    assert ident.name == ""
    assert ident.metadata == {}


def test_identity_full():
    ident = Identity(name="mouse_A", metadata={"color": "#e6194b", "strain": "C57"})
    assert ident.name == "mouse_A"
    assert ident.metadata == {"color": "#e6194b", "strain": "C57"}


def test_identity_has_no_uuid_color_gallery_categories():
    ident = Identity(name="x")
    assert not hasattr(ident, "uuid")
    assert not hasattr(ident, "color")
    assert not hasattr(ident, "embeddings")
    assert not hasattr(ident, "categories")


def test_identity_eq_false():
    # Object-identity equality (like Track): two distinct objects are not equal.
    a = Identity(name="mouse_A")
    b = Identity(name="mouse_A")
    assert a is not b
    assert a != b
    assert a == a


def test_identity_hashable_by_object_identity():
    a = Identity(name="mouse_A")
    b = Identity(name="mouse_A")
    assert len({a, b}) == 2


def test_identity_repr():
    assert repr(Identity(name="mouse_A")) == 'Identity(name="mouse_A")'


def test_identity_matches_name_default():
    a = Identity(name="mouse_A")
    b = Identity(name="mouse_A")
    c = Identity(name="mouse_B")
    assert a.matches(b)  # default method is "name"
    assert a.matches(b, method="name")
    assert not a.matches(c)


def test_identity_matches_object_identity():
    a = Identity(name="mouse_A")
    b = Identity(name="mouse_A")
    assert a.matches(a, method="identity")
    assert not a.matches(b, method="identity")


def test_identity_matches_invalid_method():
    a = Identity(name="mouse_A")
    with pytest.raises(ValueError):
        a.matches(a, method="uuid")


def test_identity_metadata_validation():
    with pytest.raises(TypeError):
        Identity(name="x", metadata="notadict")


def test_identity_deepcopy():
    a = Identity(name="mouse_A", metadata={"color": "#fff"})
    b = copy.deepcopy(a)
    assert b is not a
    assert b.name == a.name
    assert b.metadata == a.metadata
    assert b.metadata is not a.metadata
