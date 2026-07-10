"""Tests for the `Category` data structure and its converter."""

import copy

import pytest

from sleap_io.model.category import Category, to_category


def test_category_defaults():
    cat = Category()
    assert cat.name == ""
    assert cat.metadata == {}


def test_category_full():
    cat = Category(name="female_fly", metadata={"color": "#e6194b", "sex": "F"})
    assert cat.name == "female_fly"
    assert cat.metadata == {"color": "#e6194b", "sex": "F"}


def test_category_eq_false():
    # Object-identity equality (like Track/Identity): distinct objects are not equal.
    a = Category(name="female_fly")
    b = Category(name="female_fly")
    assert a is not b
    assert a != b
    assert a == a


def test_category_hashable_by_object_identity():
    a = Category(name="female_fly")
    b = Category(name="female_fly")
    assert len({a, b}) == 2


def test_category_repr():
    assert repr(Category(name="female_fly")) == 'Category(name="female_fly")'


def test_category_matches_name_default():
    a = Category(name="female_fly")
    b = Category(name="female_fly")
    c = Category(name="male_fly")
    assert a.matches(b)  # default method is "name"
    assert a.matches(b, method="name")
    assert not a.matches(c)


def test_category_matches_object_identity():
    a = Category(name="female_fly")
    b = Category(name="female_fly")
    assert a.matches(a, method="identity")
    assert not a.matches(b, method="identity")


def test_category_matches_invalid_method():
    a = Category(name="female_fly")
    with pytest.raises(ValueError):
        a.matches(a, method="uuid")


def test_category_metadata_validation():
    with pytest.raises(TypeError):
        Category(name="x", metadata="notadict")


def test_category_deepcopy():
    a = Category(name="female_fly", metadata={"color": "#fff"})
    b = copy.deepcopy(a)
    assert b is not a
    assert b.name == a.name
    assert b.metadata == a.metadata
    assert b.metadata is not a.metadata


def test_to_category_none_and_empty_string():
    # None and "" (the legacy "unset" sentinel) both map to None.
    assert to_category(None) is None
    assert to_category("") is None


def test_to_category_from_string():
    cat = to_category("mouse")
    assert isinstance(cat, Category)
    assert cat.name == "mouse"


def test_to_category_passthrough():
    cat = Category(name="mouse")
    assert to_category(cat) is cat


def test_to_category_invalid_type():
    with pytest.raises(TypeError):
        to_category(5)
