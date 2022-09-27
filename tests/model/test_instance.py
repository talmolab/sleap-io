"""Tests for methods in the sleap_io.model.instance file."""
from pickletools import pyset
import numpy as np
from numpy.testing import assert_equal
import pytest
from sleap_io.model.instance import (
    Point,
    PredictedPoint,
    Track,
    Instance,
    PredictedInstance,
)
from sleap_io import Skeleton


def test_point():
    """Test `Point` object is initialized as expected."""
    pt = Point(x=1.2, y=3.4, visible=True, complete=True)
    assert_equal(pt.numpy(), np.array([1.2, 3.4]))

    pt2 = Point(x=1.2, y=3.4, visible=True, complete=True)
    assert pt == pt2

    pt.visible = False
    assert_equal(pt.numpy(), np.array([np.nan, np.nan]))


def test_close_points():
    """Test equality of two `Point` objects which have a floating point error."""
    pt1 = Point(
        x=135.82268970698718, y=213.22842752594835, visible=True, complete=False
    )
    pt2 = Point(x=135.82268970698718, y=213.2284275259484, visible=True, complete=False)
    assert pt1 == pt2

    pt1 = Point(x=np.nan, y=np.nan, visible=False, complete=False)
    pt2 = Point(x=np.nan, y=np.nan, visible=False, complete=False)
    assert pt1 == pt2


def test_predicted_point():
    """Test `PredictedPoint` is initialized as expected."""
    pt = PredictedPoint(x=1.2, y=3.4, visible=True, complete=False, score=0.9)
    assert pt.score == 0.9
    assert_equal(pt.numpy(), np.array([1.2, 3.4, 0.9]))


def test_track():
    """Test `Track` hashing by id."""
    assert Track("A") != Track("A")


def test_instance():
    """Test initialization and methods of `Instance` object."""
    inst = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[0, 1], [2, 3]])
    assert type(inst["A"]) == Point

    inst = Instance({"A": [0, 1]}, skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[0, 1], [np.nan, np.nan]])

    inst = Instance([[1, 2], [3, 4]], skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[1, 2], [3, 4]])
    assert len(inst) == 2
    assert inst.n_visible == 2
    assert_equal(inst[0].numpy(), [1, 2])
    assert_equal(inst[1].numpy(), [3, 4])
    assert_equal(inst["A"].numpy(), [1, 2])
    assert_equal(inst["B"].numpy(), [3, 4])
    assert_equal(inst[inst.skeleton.nodes[0]].numpy(), [1, 2])
    assert_equal(inst[inst.skeleton.nodes[1]].numpy(), [3, 4])

    inst = Instance(np.array([[1, 2], [3, 4]]), skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[1, 2], [3, 4]])

    inst = Instance.from_numpy([[1, 2], [3, 4]], skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[1, 2], [3, 4]])

    inst = Instance([[np.nan, np.nan], [3, 4]], skeleton=Skeleton(["A", "B"]))
    assert not inst[0].visible
    assert inst[1].visible
    assert inst.n_visible == 1
    assert not inst.is_empty

    inst = Instance([[np.nan, np.nan], [np.nan, np.nan]], skeleton=Skeleton(["A", "B"]))
    assert inst.n_visible == 0
    assert inst.is_empty

    with pytest.raises(ValueError):
        Instance([[1, 2]], skeleton=Skeleton(["A", "B"]))

    with pytest.raises(IndexError):
        inst[None]


def test_predicted_instance():
    """Test initialization and creation of `PredictedInstance` object."""
    inst = PredictedInstance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[0, 1, 0], [2, 3, 0]])
    assert type(inst["A"]) == PredictedPoint

    inst = PredictedInstance.from_numpy(
        [[0, 1], [2, 3]], [0.4, 0.5], instance_score=0.6, skeleton=Skeleton(["A", "B"])
    )
    assert_equal(inst.numpy(), [[0, 1, 0.4], [2, 3, 0.5]])
    assert inst[0].score == 0.4
    assert inst[1].score == 0.5
    assert inst.score == 0.6
