"""Tests for methods in the sleap_io.model.instance file."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from sleap_io import Skeleton
from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.centroid import PredictedCentroid, UserCentroid
from sleap_io.model.identity import Identity
from sleap_io.model.instance import (
    Instance,
    PredictedInstance,
    Track,
)
from sleap_io.model.mask import PredictedSegmentationMask, UserSegmentationMask
from sleap_io.model.roi import PredictedROI, UserROI


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


def test_instance_identity_fields():
    """Test that Instance/PredictedInstance carry identity and identity_score."""
    skeleton = Skeleton(["head", "tail"])
    identity = Identity(name="mouse_A")

    inst = Instance.from_numpy(
        np.array([[10, 10], [20, 20]]),
        skeleton=skeleton,
        identity=identity,
        identity_score=0.9,
    )
    assert inst.identity is identity
    assert inst.identity_score == 0.9

    pred = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]),
        skeleton=skeleton,
        score=0.8,
        identity=identity,
        identity_score=0.7,
    )
    assert pred.identity is identity
    assert pred.identity_score == 0.7

    # Defaults are None.
    bare = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
    assert bare.identity is None
    assert bare.identity_score is None

    # empty() also threads identity.
    empty = Instance.empty(skeleton, identity=identity, identity_score=0.5)
    assert empty.identity is identity
    assert empty.identity_score == 0.5


def test_instance_same_identity_as_global_identity():
    """Test that same_identity_as prefers global Identity (by uuid) over track."""
    skeleton = Skeleton(["head", "tail"])
    track1 = Track(name="t1")
    track2 = Track(name="t2")

    # Same uuid (e.g. two reloads of one animal) -> same identity even across
    # different track objects.
    idA1 = Identity(name="mouse_A", uuid="shared")
    idA2 = Identity(name="mouse_A", uuid="shared")
    inst1 = Instance.from_numpy(
        np.array([[1, 1], [2, 2]]), skeleton=skeleton, track=track1, identity=idA1
    )
    inst2 = Instance.from_numpy(
        np.array([[3, 3], [4, 4]]), skeleton=skeleton, track=track2, identity=idA2
    )
    assert inst1.same_identity_as(inst2)  # uuid matches despite different tracks

    # Different uuid -> not the same identity even if tracks are the same object.
    idB = Identity(name="mouse_B")
    inst3 = Instance.from_numpy(
        np.array([[5, 5], [6, 6]]), skeleton=skeleton, track=track1, identity=idB
    )
    assert not inst1.same_identity_as(inst3)

    # When only one has an identity, fall back to track comparison.
    inst4 = Instance.from_numpy(
        np.array([[7, 7], [8, 8]]), skeleton=skeleton, track=track1
    )
    assert inst1.same_identity_as(inst4)  # idA1 has no identity peer -> track t1
    inst5 = Instance.from_numpy(
        np.array([[9, 9], [0, 0]]), skeleton=skeleton, track=track2
    )
    assert not inst1.same_identity_as(inst5)  # falls back to track, t1 != t2


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


# ---------------------------------------------------------------------------
# Modality conversions (issue #529): to_centroid / to_bbox / to_roi / to_mask
# ---------------------------------------------------------------------------


def _tri_skeleton() -> Skeleton:
    """Return a simple 3-node skeleton with two edges."""
    skel = Skeleton(["a", "b", "c"])
    skel.add_edge("a", "b")
    skel.add_edge("a", "c")
    return skel


def test_instance_to_centroid_delegates_default():
    """Instance.to_centroid delegates to Centroid.from_pose (center_of_mass)."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [9, 0], [0, 9]]), skeleton=skel)

    cent = inst.to_centroid()

    assert isinstance(cent, UserCentroid)
    assert cent.source == "center_of_mass"
    assert cent.instance is inst
    assert cent.xy == pytest.approx((3.0, 3.0))


def test_instance_to_centroid_anchor_with_fallback():
    """Occluded anchor falls back and records the fallback in the source tag."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(
        np.array([[np.nan, np.nan], [10, 0], [0, 10]]), skeleton=skel
    )

    cent = inst.to_centroid(method="anchor", node="a", fallback="center_of_mass")

    assert cent.source == "anchor:a->center_of_mass"
    assert cent.xy == pytest.approx((5.0, 5.0))


def test_instance_to_centroid_predicted_carries_score():
    """A PredictedInstance yields a PredictedCentroid carrying its score."""
    skel = _tri_skeleton()
    inst = PredictedInstance.from_numpy(
        np.array([[0, 0], [4, 0], [0, 4]]), skeleton=skel, score=0.8
    )

    cent = inst.to_centroid()

    assert isinstance(cent, PredictedCentroid)
    assert cent.score == pytest.approx(0.8)
    assert cent.instance is inst


def test_instance_to_bbox_tight():
    """Tight axis-aligned bbox encloses all visible points."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[5, 10], [15, 20], [10, 30]]), skeleton=skel)

    bbox = inst.to_bbox()

    assert isinstance(bbox, UserBoundingBox)
    assert bbox.angle == 0.0
    assert bbox.xyxy == pytest.approx((5.0, 10.0, 15.0, 30.0))
    assert bbox.instance is inst


def test_instance_to_bbox_tight_ignores_invisible():
    """Tight bbox ignores invisible (NaN) points."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(
        np.array([[5, 10], [np.nan, np.nan], [10, 30]]), skeleton=skel
    )

    bbox = inst.to_bbox()

    assert bbox.xyxy == pytest.approx((5.0, 10.0, 10.0, 30.0))


def test_instance_to_bbox_tight_padding():
    """Padding inflates the tight bbox; tuple padding is per-axis."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel)

    bbox = inst.to_bbox(padding=(2, 3))

    assert bbox.xyxy == pytest.approx((-2.0, -3.0, 12.0, 13.0))


def test_instance_to_bbox_tight_rotated():
    """Rotated tight bbox fits an oriented box from the convex hull."""
    skel = Skeleton(["a", "b", "c", "d"])
    # Axis-aligned square rotated 45 degrees about the origin.
    inst = Instance.from_numpy(
        np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), skeleton=skel
    )

    bbox = inst.to_bbox(rotated=True)

    # The minimum-area rectangle of a diamond is a square of side sqrt(2).
    assert bbox.is_rotated
    assert bbox.width == pytest.approx(np.sqrt(2), abs=1e-6)
    assert bbox.height == pytest.approx(np.sqrt(2), abs=1e-6)


def test_instance_to_bbox_tight_rotated_padding():
    """Padding on a rotated tight bbox enlarges the pre-rotation extent."""
    skel = Skeleton(["a", "b", "c", "d"])
    inst = Instance.from_numpy(
        np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), skeleton=skel
    )

    base = inst.to_bbox(rotated=True)
    padded = inst.to_bbox(rotated=True, padding=1.0)

    assert padded.width == pytest.approx(base.width + 2.0, abs=1e-6)
    assert padded.height == pytest.approx(base.height + 2.0, abs=1e-6)
    assert padded.angle == pytest.approx(base.angle)


def test_instance_to_bbox_centered_scalar():
    """Centered bbox builds a fixed square box around the centroid."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [9, 0], [0, 9]]), skeleton=skel)

    bbox = inst.to_bbox(mode="centered", size=4)

    # Centroid is (3, 3); a size-4 box spans +/- 2.
    assert bbox.xyxy == pytest.approx((1.0, 1.0, 5.0, 5.0))


def test_instance_to_bbox_centered_size_tuple():
    """Centered bbox accepts a (w, h) tuple for size."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[2, 2], [2, 2], [2, 2]]), skeleton=skel)

    bbox = inst.to_bbox(mode="centered", size=(4, 6))

    assert bbox.xyxy == pytest.approx((0.0, -1.0, 4.0, 5.0))


def test_instance_to_bbox_centered_anchor_node():
    """Centered bbox can center on an anchor node."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[10, 20], [0, 0], [0, 0]]), skeleton=skel)

    bbox = inst.to_bbox(mode="centered", size=2, center_method="anchor", node="a")

    assert bbox.xyxy == pytest.approx((9.0, 19.0, 11.0, 21.0))


def test_instance_to_bbox_centered_requires_size():
    """mode='centered' without a size is a misconfiguration."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [9, 0], [0, 9]]), skeleton=skel)

    with pytest.raises(ValueError, match="size"):
        inst.to_bbox(mode="centered")


def test_instance_to_bbox_unknown_mode_raises():
    """An unknown bbox mode raises ValueError."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [9, 0], [0, 9]]), skeleton=skel)

    with pytest.raises(ValueError, match="Unknown mode"):
        inst.to_bbox(mode="bogus")


def test_instance_to_bbox_empty_degenerate():
    """No visible points yields a degenerate (NaN) bbox."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    bbox = inst.to_bbox()

    assert bbox.is_empty
    assert np.isnan(bbox.x1)


def test_instance_to_bbox_empty_error_on_empty():
    """error_on_empty raises for an empty tight bbox."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    with pytest.raises(ValueError, match="No visible points"):
        inst.to_bbox(error_on_empty=True)


def test_instance_to_bbox_centered_empty_degenerate():
    """Centered mode with no visible points yields a degenerate bbox."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    bbox = inst.to_bbox(mode="centered", size=4)

    assert bbox.is_empty


def test_instance_to_bbox_centered_empty_error_on_empty():
    """Centered mode propagates error_on_empty through the centroid step."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    with pytest.raises(ValueError):
        inst.to_bbox(mode="centered", size=4, error_on_empty=True)


def test_instance_to_bbox_predicted_carries_score():
    """A PredictedInstance yields a PredictedBoundingBox carrying its score."""
    skel = _tri_skeleton()
    inst = PredictedInstance.from_numpy(
        np.array([[0, 0], [9, 0], [0, 9]]), skeleton=skel, score=0.6
    )

    bbox = inst.to_bbox()

    assert isinstance(bbox, PredictedBoundingBox)
    assert bbox.score == pytest.approx(0.6)
    assert bbox.instance is inst


def test_instance_to_roi_shapes_nodes():
    """method='shapes' with node_radius unions buffered node disks."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel)

    roi = inst.to_roi(node_radius=2)

    assert isinstance(roi, UserROI)
    assert not roi.is_empty
    assert roi.instance is inst


def test_instance_to_roi_shapes_edges():
    """method='shapes' with edge_radius buffers fully-visible edge segments."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel)

    roi = inst.to_roi(edge_radius=1.0)

    assert not roi.is_empty
    # Two edges (a-b, a-c) each of length 10 buffered by 1 -> area roughly
    # the union of two capsules.
    assert roi.geometry.area > 0


def test_instance_to_roi_convex_hull():
    """method='convex_hull' returns the hull polygon of visible points."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel)

    roi = inst.to_roi(method="convex_hull")

    assert roi.geometry.geom_type == "Polygon"
    assert roi.geometry.area == pytest.approx(50.0)


def test_instance_to_roi_convex_hull_radius_quad_segs():
    """convex_hull radius buffers the hull; quad_segs controls smoothness."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel)

    coarse = inst.to_roi(method="convex_hull", radius=2.0, quad_segs=1)
    fine = inst.to_roi(method="convex_hull", radius=2.0, quad_segs=16)

    # Both enclose the bare hull (area 50); finer rounding has larger area.
    assert coarse.geometry.area > 50.0
    assert fine.geometry.area > coarse.geometry.area


def test_instance_to_roi_shapes_misconfig_always_raises():
    """Both radii zero is a misconfiguration that raises even with empty input."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel)
    empty = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    with pytest.raises(ValueError, match="at least one"):
        inst.to_roi()
    # Misconfiguration raises even when error_on_empty=True and no points.
    with pytest.raises(ValueError, match="at least one"):
        empty.to_roi(error_on_empty=True)


def test_instance_to_roi_unknown_method_raises():
    """An unknown ROI method raises ValueError."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel)

    with pytest.raises(ValueError, match="Unknown method"):
        inst.to_roi(method="bogus")


def test_instance_to_roi_empty_degenerate():
    """No visible points yields an empty-geometry ROI."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    roi = inst.to_roi(node_radius=2)

    assert roi.is_empty


def test_instance_to_roi_empty_error_on_empty():
    """error_on_empty raises for an empty ROI geometry."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    with pytest.raises(ValueError, match="No visible points"):
        inst.to_roi(node_radius=2, error_on_empty=True)


def test_instance_to_roi_predicted_carries_score():
    """A PredictedInstance yields a PredictedROI carrying its score."""
    skel = _tri_skeleton()
    inst = PredictedInstance.from_numpy(
        np.array([[0, 0], [10, 0], [0, 10]]), skeleton=skel, score=0.9
    )

    roi = inst.to_roi(node_radius=2)

    assert isinstance(roi, PredictedROI)
    assert roi.score == pytest.approx(0.9)
    assert roi.instance is inst


def test_instance_to_mask_matches_roi_to_mask():
    """to_mask equals to_roi(...).to_mask(h, w) for the same kwargs."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.array([[2, 2], [12, 2], [2, 12]]), skeleton=skel)

    via_mask = inst.to_mask(20, 20, node_radius=3)
    via_roi = inst.to_roi(node_radius=3).to_mask(20, 20)

    assert isinstance(via_mask, UserSegmentationMask)
    assert via_mask.area == via_roi.area
    np.testing.assert_array_equal(via_mask.data, via_roi.data)
    assert via_mask.instance is inst


def test_instance_to_mask_empty_all_background():
    """An empty instance rasterizes to an all-background mask."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    mask = inst.to_mask(8, 12, node_radius=2)

    assert isinstance(mask, UserSegmentationMask)
    assert mask.area == 0
    assert mask.data.shape == (8, 12)
    assert mask.instance is inst


def test_instance_to_mask_empty_error_on_empty():
    """error_on_empty raises for an empty instance to_mask."""
    skel = _tri_skeleton()
    inst = Instance.from_numpy(np.full((3, 2), np.nan), skeleton=skel)

    with pytest.raises(ValueError, match="No visible points"):
        inst.to_mask(8, 8, node_radius=2, error_on_empty=True)


def test_instance_to_mask_convex_hull_degenerate_all_background():
    """A zero-area convex hull (<3 visible points) rasterizes to all background.

    With fewer than three visible points the convex hull is a Point/LineString,
    which is non-empty but non-fillable; ``to_mask`` returns an all-background
    mask instead of leaking the rasterizer's ``TypeError``.
    """
    skel = _tri_skeleton()
    # Only two visible points -> hull is a LineString (zero area).
    inst = Instance.from_numpy(
        np.array([[0.0, 0.0], [10.0, 10.0], [np.nan, np.nan]]), skeleton=skel
    )

    mask = inst.to_mask(20, 20, method="convex_hull", radius=0.0)

    assert isinstance(mask, UserSegmentationMask)
    assert mask.is_empty
    assert mask.area == 0
    assert mask.data.shape == (20, 20)
    assert mask.instance is inst


def test_instance_to_mask_convex_hull_degenerate_predicted_carries_score():
    """The degenerate convex-hull path preserves the predicted variant + score."""
    skel = _tri_skeleton()
    inst = PredictedInstance.from_numpy(
        np.array([[0.0, 0.0], [10.0, 10.0], [np.nan, np.nan]]),
        skeleton=skel,
        score=0.8,
    )

    mask = inst.to_mask(20, 20, method="convex_hull", radius=0.0)

    assert isinstance(mask, PredictedSegmentationMask)
    assert mask.score == pytest.approx(0.8)
    assert mask.is_empty


def test_instance_to_mask_predicted_carries_score():
    """A PredictedInstance yields a PredictedSegmentationMask carrying its score."""
    skel = _tri_skeleton()
    inst = PredictedInstance.from_numpy(
        np.array([[2, 2], [12, 2], [2, 12]]), skeleton=skel, score=0.75
    )

    mask = inst.to_mask(20, 20, node_radius=3)

    assert isinstance(mask, PredictedSegmentationMask)
    assert mask.score == pytest.approx(0.75)
    assert mask.area > 0
    assert mask.instance is inst


def test_instance_to_mask_predicted_empty_carries_score():
    """An empty PredictedInstance to_mask is all-background but keeps its score."""
    skel = _tri_skeleton()
    inst = PredictedInstance.from_numpy(
        np.full((3, 2), np.nan), skeleton=skel, score=0.3
    )

    mask = inst.to_mask(6, 6, node_radius=2)

    assert isinstance(mask, PredictedSegmentationMask)
    assert mask.score == pytest.approx(0.3)
    assert mask.area == 0
