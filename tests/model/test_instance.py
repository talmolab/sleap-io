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

    # test points with NAN for coordinates
    pt1 = Point(x=np.nan, y=np.nan, visible=False, complete=False)
    pt2 = Point(x=np.nan, y=np.nan, visible=False, complete=False)
    assert pt1 == pt2

    # test floating point error
    pt1 = Point(x=135.82268970698718, y=213.22842752594835)
    pt2 = Point(x=135.82268970698718, y=213.2284275259484)
    assert pt1 == pt2

    # change allowed tolerance, and check we fail comparison
    Point.eq_atol = 0
    Point.eq_rtol = 0
    assert not pt1 == pt2

    # reset tolerance
    Point.eq_atol = 1e-08
    Point.eq_rtol = 0

    # test points with NAN for coordinates
    pt1 = PredictedPoint(x=np.nan, y=np.nan, visible=False, complete=False)
    pt2 = PredictedPoint(x=np.nan, y=np.nan, visible=False, complete=False)
    assert pt1 == pt2

    # test floating point error
    pt1 = PredictedPoint(x=135.82268970698718, y=213.22842752594835)
    pt2 = PredictedPoint(x=135.82268970698718, y=213.2284275259484)
    assert pt1 == pt2

    # change allowed tolerance, and check we fail comparison
    Point.eq_atol = 0
    Point.eq_rtol = 0
    assert not pt1 == pt2


def test_predicted_point():
    """Test `PredictedPoint` is initialized as expected."""
    ppt1 = PredictedPoint(x=1.2, y=3.4, visible=True, complete=False, score=0.9)
    assert ppt1.score == 0.9
    assert_equal(ppt1.numpy(), np.array([1.2, 3.4, 0.9]))

    ppt2 = PredictedPoint(x=1.2, y=3.4, visible=True, complete=False, score=0.9)
    assert ppt1 == ppt2

    # Test equivelance of Point and PredictedPoint
    pt3 = Point(x=1.2, y=3.4, visible=True, complete=False)
    assert not ppt1 == pt3  # PredictedPoint is not equivalent to Point
    assert not pt3 == ppt1  # Point is not equivalent to PredictedPoint


def test_track():
    """Test `Track` hashing by id."""
    assert Track("A") != Track("A")


def test_instance():
    """Test initialization and methods of `Instance` object."""
    inst = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[0, 1], [2, 3]])
    assert type(inst["A"]) == Point
    assert str(inst) == "Instance(points=[[0.0, 1.0], [2.0, 3.0]], track=None)"

    inst.track = Track("trk")
    assert str(inst) == 'Instance(points=[[0.0, 1.0], [2.0, 3.0]], track="trk")'

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


def test_instance_comparison():
    """Test some properties of `Instance` equality semantics"""
    # test that instances with different skeletons are not considered equal
    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    inst2 = Instance({"A": [0, 1], "C": [2, 3]}, skeleton=Skeleton(["A", "C"]))
    assert not inst1 == inst2

    # test that instances with the same skeleton but different point coordinates are not considered equal
    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    inst2 = Instance({"A": [2, 3], "B": [0, 1]}, skeleton=Skeleton(["A", "B"]))
    assert not inst1 == inst2


def test_predicted_instance():
    """Test initialization and creation of `PredictedInstance` object."""
    inst = PredictedInstance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[0, 1], [2, 3]])
    assert_equal(inst.numpy(scores=True), [[0, 1, 0], [2, 3, 0]])
    assert type(inst["A"]) == PredictedPoint

    inst = PredictedInstance.from_numpy(
        [[0, 1], [2, 3]], [0.4, 0.5], instance_score=0.6, skeleton=Skeleton(["A", "B"])
    )
    assert_equal(inst.numpy(), [[0, 1], [2, 3]])
    assert_equal(inst.numpy(scores=True), [[0, 1, 0.4], [2, 3, 0.5]])
    assert inst[0].score == 0.4
    assert inst[1].score == 0.5
    assert inst.score == 0.6

    assert (
        str(inst) == "PredictedInstance(points=[[0.0, 1.0], [2.0, 3.0]], track=None, "
        "score=0.60, tracking_score=None)"
    )


def test_instance_update_skeleton():
    skel = Skeleton(["A", "B", "C"])
    inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=skel)

    # No need to update on rename
    skel.rename_nodes({"A": "X", "B": "Y", "C": "Z"})
    assert inst["X"].x == 0
    assert inst["Y"].x == 1
    assert inst["Z"].x == 2
    assert_equal(inst.numpy(), [[0, 0], [1, 1], [2, 2]])

    # Remove a node from the skeleton
    Y = skel["Y"]
    skel.remove_node("Y")
    assert Y not in skel

    with pytest.raises(KeyError):
        inst.numpy()  # .numpy() breaks without update
    assert Y in inst.points  # and the points dict still has the old key
    inst.update_skeleton()
    assert Y not in inst.points  # after update, the old key is gone
    assert_equal(inst.numpy(), [[0, 0], [2, 2]])

    # Reorder nodes
    skel.reorder_nodes(["Z", "X"])
    assert_equal(inst.numpy(), [[2, 2], [0, 0]])  # .numpy() works without update
    assert (
        list(inst.points.keys()) != skel.nodes
    )  # but the points dict still has the old order
    inst.update_skeleton()
    assert list(inst.points.keys()) == skel.nodes  # after update, the order is correct


def test_instance_replace_skeleton():
    # Full replacement
    old_skel = Skeleton(["A", "B", "C"])
    inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=old_skel)
    new_skel = Skeleton(["X", "Y", "Z"])
    inst.replace_skeleton(new_skel, node_map={"A": "X", "B": "Y", "C": "Z"})
    assert inst.skeleton == new_skel
    assert_equal(inst.numpy(), [[0, 0], [1, 1], [2, 2]])
    assert list(inst.points.keys()) == new_skel.nodes

    # Partial replacement
    old_skel = Skeleton(["A", "B", "C"])
    inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=old_skel)
    new_skel = Skeleton(["X", "C", "Y"])
    inst.replace_skeleton(new_skel)
    assert inst.skeleton == new_skel
    assert_equal(inst.numpy(), [[np.nan, np.nan], [2, 2], [np.nan, np.nan]])
    assert new_skel["C"] in inst.points
    assert old_skel["A"] not in inst.points
    assert old_skel["C"] not in inst.points

    # Fast path with reverse node map
    old_skel = Skeleton(["A", "B", "C"])
    inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=old_skel)
    new_skel = Skeleton(["X", "Y", "Z"])
    rev_node_map = {
        new_node: old_node for new_node, old_node in zip(new_skel.nodes, old_skel.nodes)
    }
    inst.replace_skeleton(new_skel, rev_node_map=rev_node_map)
    assert inst.skeleton == new_skel
    assert_equal(inst.numpy(), [[0, 0], [1, 1], [2, 2]])
