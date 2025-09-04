"""Tests for methods in the sleap_io.model.instance file."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from sleap_io import Skeleton
from sleap_io.model.instance import (
    Instance,
    PredictedInstance,
    Track,
)


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
    """Test some properties of `Instance` equality semantics."""
    # test that instances with different skeletons are not considered equal
    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=Skeleton(["A", "B"]))
    inst2 = Instance({"A": [0, 1], "C": [2, 3]}, skeleton=Skeleton(["A", "C"]))
    assert not inst1 == inst2

    # test that instances with the same skeleton but different point coordinates
    # are not considered equal
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
    assert inst[0]["visible"]

    # Set point by node name
    inst["B"] = [3, 4]
    assert_equal(inst["B"]["xy"], [3, 4])
    assert inst["B"]["visible"]

    # Set point by Node object
    node = inst.skeleton.nodes[2]
    inst[node] = [5, 6]
    assert_equal(inst[node]["xy"], [5, 6])
    assert inst[node]["visible"]

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
    assert inst[0]["visible"]
    assert inst[0]["score"] == 1.0

    # Set point by node name with score
    inst["B"] = [3, 4, 0.75]
    assert_equal(inst["B"]["xy"], [3, 4])
    assert inst["B"]["score"] == 0.75
    assert inst["B"]["visible"]

    # Set point by Node object with score
    node = inst.skeleton.nodes[2]
    inst[node] = [5, 6, 0.9]
    assert_equal(inst[node]["xy"], [5, 6])
    assert inst[node]["score"] == 0.9
    assert inst[node]["visible"]

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


def test_instance_same_pose_as():
    """Test Instance.same_pose_as() method."""
    skeleton = Skeleton(["head", "tail"])

    # Create instances with similar poses
    inst1 = Instance.from_numpy(
        np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton
    )
    inst2 = Instance.from_numpy(
        np.array([[11.0, 11.0], [21.0, 21.0]]), skeleton=skeleton
    )
    inst3 = Instance.from_numpy(
        np.array([[50.0, 50.0], [60.0, 60.0]]), skeleton=skeleton
    )

    # Test with specified tolerance
    assert inst1.same_pose_as(inst2, tolerance=5.0)
    assert not inst1.same_pose_as(inst3, tolerance=5.0)

    # Test with different tolerance
    assert inst1.same_pose_as(inst2, tolerance=2.0)
    assert not inst1.same_pose_as(inst2, tolerance=1.0)

    # Test with different skeletons
    skeleton2 = Skeleton(["head", "thorax", "tail"])
    inst4 = Instance.from_numpy(
        np.array([[10.0, 10.0], [15.0, 15.0], [20.0, 20.0]]), skeleton=skeleton2
    )
    assert not inst1.same_pose_as(inst4)  # Different skeletons

    # Test with different visibility patterns
    inst5 = Instance.from_numpy(
        np.array([[10.0, 10.0], [np.nan, np.nan]]), skeleton=skeleton
    )
    inst6 = Instance.from_numpy(
        np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton
    )
    assert not inst5.same_pose_as(inst6)  # Different visibility

    # Test with both having no visible points
    inst7 = Instance.from_numpy(
        np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
    )
    inst8 = Instance.from_numpy(
        np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
    )
    assert inst7.same_pose_as(inst8)  # Both have all NaN - should be equal


def test_instance_same_pose_as_identical_with_nan():
    """Test Instance.same_pose_as() bug with identical instances containing NaN values.

    This test demonstrates a bug where identical instances with some NaN coordinates
    that are marked as visible are incorrectly identified as having different poses
    due to NaN distance calculations.
    """
    skeleton = Skeleton(["head", "thorax", "tail"])

    # Create two identical instances
    inst1 = Instance.from_numpy(
        np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]), skeleton=skeleton
    )
    inst2 = Instance.from_numpy(
        np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]), skeleton=skeleton
    )

    # Manually set identical NaN coordinates but keep them marked as visible
    # This simulates the scenario where tracking data has NaN coordinates but
    # points are still considered "visible"
    inst1.points["xy"][1] = [np.nan, np.nan]  # thorax has NaN coords
    inst2.points["xy"][1] = [np.nan, np.nan]  # thorax has NaN coords
    inst1.points["visible"][1] = True
    inst2.points["visible"][1] = True

    # These instances are identical and should return True
    # Test both exact comparison and tolerance-based comparison
    assert inst1.same_pose_as(inst2), (
        "Identical instances with NaN values should be considered the same pose (exact)"
    )
    assert inst1.same_pose_as(inst2, tolerance=5.0), (
        "Identical instances with NaN values should be considered the same pose "
        "(tolerance)"
    )


def test_instance_same_identity_as():
    """Test Instance.same_identity_as() method."""
    skeleton = Skeleton(["head", "tail"])
    track1 = Track(name="mouse1")
    track2 = Track(name="mouse2")

    inst1 = Instance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, track=track1
    )
    inst2 = Instance.from_numpy(
        np.array([[50, 50], [60, 60]]), skeleton=skeleton, track=track1
    )
    inst3 = Instance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, track=track2
    )
    inst4 = Instance.from_numpy(
        np.array([[10, 10], [20, 20]]),
        skeleton=skeleton,  # No track
    )

    # Same track object
    assert inst1.same_identity_as(inst2)

    # Different track objects
    assert not inst1.same_identity_as(inst3)

    # One or both without tracks
    assert not inst1.same_identity_as(inst4)
    assert not inst4.same_identity_as(inst1)

    # Both without tracks
    inst5 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
    assert not inst4.same_identity_as(inst5)


def test_instance_overlaps_with():
    """Test Instance.overlaps_with() method."""
    skeleton = Skeleton(["p1", "p2", "p3", "p4"])

    # Create instances with known bounding boxes
    # Box 1: (0,0) to (10,10)
    inst1 = Instance.from_numpy(
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), skeleton=skeleton
    )

    # Box 2: (5,5) to (15,15) - overlaps with box 1
    inst2 = Instance.from_numpy(
        np.array([[5, 5], [15, 5], [15, 15], [5, 15]]), skeleton=skeleton
    )

    # Box 3: (20,20) to (30,30) - no overlap with box 1
    inst3 = Instance.from_numpy(
        np.array([[20, 20], [30, 20], [30, 30], [20, 30]]), skeleton=skeleton
    )

    # Test overlapping instances
    assert inst1.overlaps_with(inst2, iou_threshold=0.1)

    # Calculate expected IoU for inst1 and inst2
    # Intersection: (5,5) to (10,10) = 25
    # Union: 100 + 100 - 25 = 175
    # IoU = 25/175 = 0.143
    assert inst1.overlaps_with(inst2, iou_threshold=0.14)
    assert not inst1.overlaps_with(inst2, iou_threshold=0.15)

    # Test non-overlapping instances
    assert not inst1.overlaps_with(inst3, iou_threshold=0.0)

    # Test with invisible points
    inst4 = Instance.from_numpy(
        np.array([[np.nan, np.nan], [10, 0], [10, 10], [0, 10]]), skeleton=skeleton
    )
    inst5 = Instance.from_numpy(
        np.array(
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
        ),
        skeleton=skeleton,
    )

    # Should still work with some invisible points
    assert inst4.overlaps_with(inst2, iou_threshold=0.1)

    # Should return False if either has no visible points
    assert not inst5.overlaps_with(inst1)
    assert not inst1.overlaps_with(inst5)


def test_instance_bounding_box():
    """Test Instance.bounding_box() method."""
    skeleton = Skeleton(["p1", "p2", "p3", "p4"])

    # Test normal case
    inst1 = Instance.from_numpy(
        np.array([[5, 10], [15, 20], [10, 30], [0, 25]]), skeleton=skeleton
    )
    bbox = inst1.bounding_box()
    assert_array_equal(bbox, np.array([[0, 10], [15, 30]]))

    # Test with some invisible points
    inst2 = Instance.from_numpy(
        np.array([[5, 10], [np.nan, np.nan], [10, 30], [0, 25]]), skeleton=skeleton
    )
    bbox = inst2.bounding_box()
    assert_array_equal(bbox, np.array([[0, 10], [10, 30]]))

    # Test with no visible points
    inst3 = Instance.from_numpy(
        np.array(
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
        ),
        skeleton=skeleton,
    )
    assert inst3.bounding_box() is None


def test_track_matches():
    """Test Track.matches() method."""
    track1 = Track(name="mouse1")
    track2 = Track(name="mouse1")  # Same name, different object
    track3 = Track(name="mouse2")

    # Test name matching
    assert track1.matches(track2, method="name")
    assert not track1.matches(track3, method="name")

    # Test identity matching
    assert not track1.matches(track2, method="identity")
    assert track1.matches(track1, method="identity")

    # Test invalid method
    with pytest.raises(ValueError):
        track1.matches(track2, method="invalid")


def test_track_similarity_to():
    """Test Track.similarity_to() method."""
    track1 = Track(name="mouse1")
    track2 = Track(name="mouse1")  # Same name, different object
    track3 = Track(name="mouse2")
    track4 = Track(name="")  # Empty name

    # Test with same names
    sim = track1.similarity_to(track2)
    assert sim["same_name"] is True
    assert sim["same_identity"] is False
    assert sim["name_similarity"] == 1.0

    # Test with different names
    sim = track1.similarity_to(track3)
    assert sim["same_name"] is False
    assert sim["same_identity"] is False
    assert 0 <= sim["name_similarity"] <= 1.0

    # Test with same object
    sim = track1.similarity_to(track1)
    assert sim["same_name"] is True
    assert sim["same_identity"] is True
    assert sim["name_similarity"] == 1.0

    # Test with empty names
    sim = track4.similarity_to(track4)
    assert sim["same_name"] is True
    assert sim["name_similarity"] == 1.0
