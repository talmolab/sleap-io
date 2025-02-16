"""Tests for methods in the sleap_io.model.instance file."""

from pickletools import pyset
import numpy as np
from numpy.testing import assert_equal
import pytest
from sleap_io.model.instance import (
    Track,
    Instance,
    PredictedInstance,
)
from sleap_io import Skeleton


def test_track():
    """Test `Track` hashing by id."""
    assert Track("A") != Track("A")


def test_instance():
    """Test initialization and methods of `Instance` object."""
    inst = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[0, 1], [2, 3]])
    assert str(inst) == "Instance(points=[[0.0, 1.0], [2.0, 3.0]], track=None)"

    inst.track = Track("trk")
    assert str(inst) == 'Instance(points=[[0.0, 1.0], [2.0, 3.0]], track="trk")'

    inst = Instance({"A": [0, 1]}, skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[0, 1], [np.nan, np.nan]])

    inst = Instance([[1, 2], [3, 4]], skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[1, 2], [3, 4]])
    assert len(inst) == 2
    assert inst.n_visible == 2
    assert_equal(inst[0]["xy"], [1, 2])
    assert_equal(inst[1]["xy"], [3, 4])
    assert_equal(inst["A"]["xy"], [1, 2])
    assert_equal(inst["B"]["xy"], [3, 4])
    assert_equal(inst[inst.skeleton.nodes[0]]["xy"], [1, 2])
    assert_equal(inst[inst.skeleton.nodes[1]]["xy"], [3, 4])

    inst = Instance(np.array([[1, 2], [3, 4]]), skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[1, 2], [3, 4]])

    inst = Instance.from_numpy([[1, 2], [3, 4]], skeleton=Skeleton(["A", "B"]))
    assert_equal(inst.numpy(), [[1, 2], [3, 4]])

    inst = Instance([[np.nan, np.nan], [3, 4]], skeleton=Skeleton(["A", "B"]))
    assert not inst[0]["visible"]
    assert inst[1]["visible"]
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

    inst = PredictedInstance.from_numpy(
        [[0, 1, 0.4], [2, 3, 0.5]], skeleton=Skeleton(["A", "B"]), score=0.6
    )
    assert_equal(inst.numpy(), [[0, 1], [2, 3]])
    assert_equal(inst.numpy(scores=True), [[0, 1, 0.4], [2, 3, 0.5]])
    assert inst[0]["score"] == 0.4
    assert inst[1]["score"] == 0.5
    assert inst.score == 0.6

    assert (
        str(inst) == "PredictedInstance(points=[[0.0, 1.0], [2.0, 3.0]], track=None, "
        "score=0.60, tracking_score=None)"
    )


def test_instance_update_skeleton():
    skel = Skeleton(["A", "B", "C"])
    inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=skel)

    # Need to update on rename
    skel.rename_nodes({"A": "X", "B": "Y", "C": "Z"})
    assert inst.points["name"].tolist() == ["A", "B", "C"]
    inst.update_skeleton(names_only=True)
    assert inst.points["name"].tolist() == ["X", "Y", "Z"]
    assert inst["X"]["xy"][0] == 0
    assert inst["Y"]["xy"][0] == 1
    assert inst["Z"]["xy"][0] == 2
    assert_equal(inst.numpy(), [[0, 0], [1, 1], [2, 2]])

    # Remove a node from the skeleton
    Y = skel["Y"]
    skel.remove_node("Y")
    assert Y not in skel
    assert inst.points["name"].tolist() == ["X", "Y", "Z"]
    inst.update_skeleton()
    assert inst.points["name"].tolist() == ["X", "Z"]
    assert_equal(inst.numpy(), [[0, 0], [2, 2]])

    # Reorder nodes
    skel.reorder_nodes(["Z", "X"])
    assert_equal(inst.numpy(), [[0, 0], [2, 2]])
    assert list(inst.points["name"]) != skel.node_names
    inst.update_skeleton()
    assert (
        list(inst.points["name"]) == skel.node_names
    )  # after update, the order is correct
    assert_equal(inst.numpy(), [[2, 2], [0, 0]])


def test_instance_replace_skeleton():
    # Full replacement
    old_skel = Skeleton(["A", "B", "C"])
    inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=old_skel)
    new_skel = Skeleton(["X", "Y", "Z"])
    inst.replace_skeleton(new_skel, node_names_map={"A": "X", "B": "Y", "C": "Z"})
    assert inst.skeleton == new_skel
    assert_equal(inst.numpy(), [[0, 0], [1, 1], [2, 2]])
    assert list(inst.points["name"]) == new_skel.node_names

    # Partial replacement
    old_skel = Skeleton(["A", "B", "C"])
    inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=old_skel)
    new_skel = Skeleton(["X", "C", "Y"])
    inst.replace_skeleton(new_skel)
    assert inst.skeleton == new_skel
    assert_equal(inst.numpy(), [[np.nan, np.nan], [2, 2], [np.nan, np.nan]])
    assert inst.points["name"].tolist() == ["X", "C", "Y"]

    # # Fast path with reverse node map
    # old_skel = Skeleton(["A", "B", "C"])
    # inst = Instance.from_numpy([[0, 0], [1, 1], [2, 2]], skeleton=old_skel)
    # new_skel = Skeleton(["X", "Y", "Z"])
    # rev_node_map = {
    #     new_node: old_node for new_node, old_node in zip(new_skel.nodes, old_skel.nodes)
    # }
    # inst.replace_skeleton(new_skel, rev_node_map=rev_node_map)
    # assert inst.skeleton == new_skel
    # assert_equal(inst.numpy(), [[0, 0], [1, 1], [2, 2]])
