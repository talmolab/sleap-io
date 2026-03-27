"""Tests for Identity data structure."""

from __future__ import annotations

from sleap_io.model.identity import Identity


def test_identity_defaults():
    """Test Identity construction with defaults."""
    identity = Identity()
    assert identity.name == ""
    assert identity.color is None
    assert identity.metadata == {}


def test_identity_full():
    """Test Identity construction with all fields."""
    identity = Identity(name="mouse_A", color="#ff0000", metadata={"weight": 25.0})
    assert identity.name == "mouse_A"
    assert identity.color == "#ff0000"
    assert identity.metadata == {"weight": 25.0}


def test_identity_eq_false():
    """Test that Identity uses object identity for equality (eq=False)."""
    id1 = Identity(name="mouse_A")
    id2 = Identity(name="mouse_A")
    assert id1 is not id2
    assert id1 != id2  # eq=False means different objects are not equal


def test_identity_repr():
    """Test Identity string representation."""
    id1 = Identity(name="mouse_A")
    assert repr(id1) == 'Identity(name="mouse_A")'

    id2 = Identity(name="mouse_A", color="#ff0000")
    assert repr(id2) == 'Identity(name="mouse_A", color="#ff0000")'


def test_identity_hashable():
    """Test that Identity objects can be used in sets and as dict keys."""
    id1 = Identity(name="a")
    id2 = Identity(name="b")
    s = {id1, id2}
    assert len(s) == 2
    d = {id1: "first", id2: "second"}
    assert d[id1] == "first"
