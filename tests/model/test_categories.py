"""Tests for the CategoriesMixin and per-instance / per-identity categories."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from sleap_io.model.categories import CategoriesMixin
from sleap_io.model.centroid import UserCentroid
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.skeleton import Skeleton


def test_categories_default_empty():
    """Test that categories defaults to an empty dict surfaced via `cat`."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    assert inst.categories == {}
    assert inst.cat == {}
    assert isinstance(inst, CategoriesMixin)


def test_cat_is_live_categories_dict():
    """Test the `cat` alias returns the live mapping (in-place assignment)."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)

    # `cat` is the same object as `categories`.
    assert inst.cat is inst.categories

    # Index-assigning through `cat` mutates `categories` in place.
    inst.cat["sex"] = "M"
    assert inst.categories == {"sex": "M"}


def test_set_category_is_void_mutator():
    """Test set_category stores under the dimension and returns None."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)

    assert inst.set_category("sex", "M") is None  # in-place mutator
    inst.set_category("strain", "C57BL/6")
    assert inst.categories == {"sex": "M", "strain": "C57BL/6"}


def test_set_categories_merges_and_overwrites():
    """Test set_categories merges, overwrites existing keys, and returns None."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    inst.set_category("sex", "M")

    assert inst.set_categories({"sex": "F", "age": "adult"}) is None
    assert inst.categories == {"sex": "F", "age": "adult"}


def test_cat_setter_replaces_mapping():
    """Test assigning to `cat` replaces the whole mapping."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    inst.set_category("sex", "M")

    inst.cat = {"reset": True}
    assert inst.categories == {"reset": True}


def test_categories_non_string_values():
    """Test categories accept JSON-serializable non-string values (e.g. lists)."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    inst.set_category("sex_probs", [0.2, 0.8])
    assert inst.categories["sex_probs"] == [0.2, 0.8]


def test_categories_constructor_kwarg():
    """Test categories can be supplied directly to the constructor."""
    skel = Skeleton(["A", "B"])
    points = Instance._convert_points(np.array([[0, 1], [2, 3]]), skel)
    inst = Instance(points=points, skeleton=skel, categories={"sex": "M"})
    assert inst.categories == {"sex": "M"}


def test_categories_on_predicted_instance():
    """Test categories work on PredictedInstance and keep slots intact."""
    skel = Skeleton(["A", "B"])
    pred = PredictedInstance.from_numpy(np.array([[0, 1], [2, 3]]), skel, score=0.5)
    assert isinstance(pred, CategoriesMixin)
    pred.cat["view"] = "top"
    pred.set_category("sex", "F")
    assert pred.categories == {"view": "top", "sex": "F"}
    assert not hasattr(pred, "__dict__")


def test_categories_on_identity():
    """Test Identity carries entity-level categories."""
    identity = Identity(name="mouse_A")
    assert isinstance(identity, CategoriesMixin)
    assert identity.categories == {}
    identity.set_category("species", "mouse")
    identity.set_categories({"sex": "M", "strain": "C57BL/6"})
    assert identity.categories == {"species": "mouse", "sex": "M", "strain": "C57BL/6"}


def test_categories_slotted_no_dict():
    """Test the categories slot does not add a __dict__ to slotted instances."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    inst.set_category("sex", "M")
    assert not hasattr(inst, "__dict__")


def test_categories_preserved_on_copy():
    """Test categories survive a deep copy of the owning instance."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    inst.set_categories({"sex": "M", "age": "adult"})
    copied = deepcopy(inst)
    assert copied.categories == {"sex": "M", "age": "adult"}
    # The copy is independent of the original.
    copied.set_category("sex", "F")
    assert inst.categories["sex"] == "M"


def test_categories_not_in_repr():
    """Test the categories field is excluded from the repr (repr=False)."""
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    inst.set_category("sex", "M")
    assert "categories" not in repr(inst)


def test_plural_categories_distinct_from_scalar_category():
    """Test the plural `categories` mapping is scoped to instances + Identity.

    Geometry primitives (e.g. `Centroid`) keep their singular scalar `category`
    field and do not gain the plural `categories` mapping in this iteration.
    """
    skel = Skeleton(["A", "B"])
    inst = Instance.from_numpy(np.array([[0, 1], [2, 3]]), skel)
    assert hasattr(inst, "categories")
    assert not hasattr(inst, "category")  # no singular scalar on instances

    centroid = UserCentroid(x=1.0, y=2.0)
    assert hasattr(centroid, "category")  # singular scalar retained
    assert not hasattr(centroid, "categories")  # no plural mapping (scope)
