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
    inst["A"]["visible"] = False
    assert_equal(inst.numpy(), [[np.nan, np.nan], [3, 4]])
    assert_equal(inst.numpy(invisible_as_nan=False), [[1, 2], [3, 4]])

    inst = Instance([[np.nan, np.nan], [3, 4]], skeleton=Skeleton(["A", "B"]))
    assert not inst[0]["visible"]
    assert inst[1]["visible"]
    assert inst.n_visible == 1
    assert not inst.is_empty

    inst = Instance([[np.nan, np.nan], [np.nan, np.nan]], skeleton=Skeleton(["A", "B"]))
    assert inst.n_visible == 0
    assert inst.is_empty

    inst = Instance.empty(skeleton=Skeleton(["A", "B", "C"]))
    assert len(inst) == 3
    assert inst.n_visible == 0
    assert_equal(inst.numpy(), [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])

    with pytest.raises(ValueError):
        Instance([[1, 2]], skeleton=Skeleton(["A", "B"]))

    with pytest.raises(IndexError):
        inst[None]


def test_instance_convert_points():
    # Unequal number of points and skeleton nodes
    with pytest.raises(ValueError):
        Instance._convert_points([[1, 2], [3, 4]], skeleton=Skeleton(["A", "B", "C"]))

    # Provide xy
    points = Instance._convert_points(
        [[1, 2], [np.nan, np.nan], [4, 5]], skeleton=Skeleton(["A", "B", "C"])
    )
    assert_equal(points["xy"], [[1, 2], [np.nan, np.nan], [4, 5]])
    assert_equal(points["visible"], [True, False, True])

    # Provide xy, visibility and completion
    points = Instance._convert_points(
        [[1, 2, True, False], [3, 4, False, False], [4, 5, True, False]],
        skeleton=Skeleton(["A", "B", "C"]),
    )
    assert_equal(points["xy"], [[1, 2], [3, 4], [4, 5]])
    assert_equal(points["visible"], [True, False, True])
    assert_equal(points["complete"], [False, False, False])

    # Provide xy, visibility and completion (as dict)
    points = Instance._convert_points(
        {"A": [1, 2, True, False], "B": [3, 4, False, False], "C": [4, 5, True, False]},
        skeleton=Skeleton(["A", "B", "C"]),
    )
    assert_equal(points["xy"], [[1, 2], [3, 4], [4, 5]])
    assert_equal(points["visible"], [True, False, True])
    assert_equal(points["complete"], [False, False, False])

    # Else case
    with pytest.raises(ValueError):
        points = Instance._convert_points(None, skeleton=Skeleton(["A", "B"]))

    # Provide partial fields as structured array
    points = Instance._convert_points(
        np.array(
            [([1, 2], True), ([3, 4], False), ([4, 5], True)],
            dtype=[("xy", float, (2,)), ("visible", bool)],
        ),
        skeleton=Skeleton(["A", "B", "C"]),
    )
    assert_equal(points["xy"], [[1, 2], [3, 4], [4, 5]])
    assert_equal(points["visible"], [True, False, True])


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
    inst["A"]["visible"] = False
    assert_equal(inst.numpy(), [[np.nan, np.nan], [2, 3]])
    assert_equal(inst.numpy(invisible_as_nan=False), [[0, 1], [2, 3]])

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

    inst = PredictedInstance.empty(skeleton=Skeleton(["A", "B", "C"]))
    assert len(inst) == 3
    assert inst.n_visible == 0
    assert_equal(
        inst.numpy(scores=True),
        [[np.nan, np.nan, 0], [np.nan, np.nan, 0], [np.nan, np.nan, 0]],
    )


def test_predicted_instance_convert_points():
    # Unequal number of points and skeleton nodes
    with pytest.raises(ValueError):
        PredictedInstance._convert_points(
            [[1, 2], [3, 4]], skeleton=Skeleton(["A", "B", "C"])
        )

    # Provide xy
    points = PredictedInstance._convert_points(
        [[1, 2], [np.nan, np.nan], [4, 5]], skeleton=Skeleton(["A", "B", "C"])
    )
    assert_equal(points["xy"], [[1, 2], [np.nan, np.nan], [4, 5]])
    assert_equal(points["visible"], [True, False, True])

    # Provide xy, scores, visibility and completion
    points = PredictedInstance._convert_points(
        [
            [1, 2, 0.9, True, False],
            [3, 4, 0.8, False, False],
            [4, 5, 0.99, True, False],
        ],
        skeleton=Skeleton(["A", "B", "C"]),
    )
    assert_equal(points["xy"], [[1, 2], [3, 4], [4, 5]])
    assert_equal(points["score"], [0.9, 0.8, 0.99])
    assert_equal(points["visible"], [True, False, True])
    assert_equal(points["complete"], [False, False, False])

    # Provide xy, score, visibility and completion (as dict)
    points = PredictedInstance._convert_points(
        {
            "A": [1, 2, 0.9, True, False],
            "B": [3, 4, 0.8, False, False],
            "C": [4, 5, 0.99, True, False],
        },
        skeleton=Skeleton(["A", "B", "C"]),
    )
    assert_equal(points["xy"], [[1, 2], [3, 4], [4, 5]])
    assert_equal(points["score"], [0.9, 0.8, 0.99])
    assert_equal(points["visible"], [True, False, True])
    assert_equal(points["complete"], [False, False, False])

    # Else case
    with pytest.raises(ValueError):
        points = PredictedInstance._convert_points(None, skeleton=Skeleton(["A", "B"]))


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


def test_instance_setitem():
    """Test the __setitem__ method of the Instance class."""
    skel = Skeleton(["A", "B", "C"])
    inst = Instance.empty(skeleton=skel)

    # Set point by index
    inst[0] = [1, 2]
    assert_equal(inst[0]["xy"], [1, 2])
    assert inst[0]["visible"] == True

    # Set point by node name
    inst["B"] = [3, 4]
    assert_equal(inst["B"]["xy"], [3, 4])
    assert inst["B"]["visible"] == True

    # Set point by Node object
    node = inst.skeleton.nodes[2]
    inst[node] = [5, 6]
    assert_equal(inst[node]["xy"], [5, 6])
    assert inst[node]["visible"] == True

    # Check all points were set correctly
    assert_equal(inst.numpy(), [[1, 2], [3, 4], [5, 6]])

    # Test with value that has extra elements (should only use first two)
    inst["A"] = [7, 8, 9, 10]
    assert_equal(inst["A"]["xy"], [7, 8])

    # Test with too few elements
    with pytest.raises(ValueError):
        inst["A"] = [1]


def test_predicted_instance_setitem():
    """Test the __setitem__ method of the PredictedInstance class."""
    skel = Skeleton(["A", "B", "C"])
    inst = PredictedInstance.empty(skeleton=skel)

    # Set point by index without score (should default to 1.0)
    inst[0] = [1, 2]
    assert_equal(inst[0]["xy"], [1, 2])
    assert inst[0]["visible"] == True
    assert inst[0]["score"] == 1.0

    # Set point by node name with score
    inst["B"] = [3, 4, 0.75]
    assert_equal(inst["B"]["xy"], [3, 4])
    assert inst["B"]["score"] == 0.75
    assert inst["B"]["visible"] == True

    # Set point by Node object with score
    node = inst.skeleton.nodes[2]
    inst[node] = [5, 6, 0.9]
    assert_equal(inst[node]["xy"], [5, 6])
    assert inst[node]["score"] == 0.9
    assert inst[node]["visible"] == True

    # Check numpy output with scores
    expected = np.array([[1, 2, 1.0], [3, 4, 0.75], [5, 6, 0.9]])
    assert_equal(inst.numpy(scores=True), expected)

    # Test with value that has extra elements (should only use first three)
    inst["A"] = [7, 8, 0.6, 10]
    assert_equal(inst["A"]["xy"], [7, 8])
    assert inst["A"]["score"] == 0.6

    # Test with too few elements
    with pytest.raises(ValueError):
        inst["A"] = [1]
