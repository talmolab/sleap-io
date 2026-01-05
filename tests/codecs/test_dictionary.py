"""Tests for dictionary codec."""

import json

import numpy as np
import pytest

from sleap_io import (
    Instance,
    LabeledFrame,
    Labels,
    PredictedInstance,
    Skeleton,
    Track,
    Video,
    load_slp,
)
from sleap_io.codecs.dictionary import from_dict, to_dict


def test_to_dict_basic(slp_typical):
    """Test basic conversion to dictionary."""
    labels = load_slp(slp_typical)

    d = to_dict(labels)

    # Check structure
    assert "version" in d
    assert d["version"] == "1.0.0"
    assert "skeletons" in d
    assert "videos" in d
    assert "tracks" in d
    assert "labeled_frames" in d
    assert "suggestions" in d
    assert "provenance" in d

    # Check skeletons
    assert len(d["skeletons"]) > 0
    skeleton = d["skeletons"][0]
    assert "name" in skeleton
    assert "nodes" in skeleton
    assert "edges" in skeleton
    assert isinstance(skeleton["nodes"], list)
    assert isinstance(skeleton["edges"], list)

    # Check videos
    assert len(d["videos"]) > 0
    video = d["videos"][0]
    assert "filename" in video

    # Check labeled frames
    assert len(d["labeled_frames"]) > 0
    frame = d["labeled_frames"][0]
    assert "frame_idx" in frame
    assert "video_idx" in frame
    assert "instances" in frame
    assert isinstance(frame["instances"], list)

    # Check instances
    if len(frame["instances"]) > 0:
        instance = frame["instances"][0]
        assert "type" in instance
        assert instance["type"] in ["instance", "predicted_instance"]
        assert "skeleton_idx" in instance
        assert "points" in instance
        assert isinstance(instance["points"], list)

        # Check points
        if len(instance["points"]) > 0:
            point = instance["points"][0]
            assert "x" in point
            assert "y" in point
            assert "visible" in point
            assert "complete" in point
            assert isinstance(point["x"], float)
            assert isinstance(point["y"], float)
            assert isinstance(point["visible"], bool)
            assert isinstance(point["complete"], bool)


def test_to_dict_json_serializable(slp_typical):
    """Test that output is JSON-serializable."""
    labels = load_slp(slp_typical)
    d = to_dict(labels)

    # Should not raise
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Round-trip through JSON should work
    d_roundtrip = json.loads(json_str)
    assert isinstance(d_roundtrip, dict)

    # Check that main structure is preserved
    assert d_roundtrip.keys() == d.keys()
    assert len(d_roundtrip["labeled_frames"]) == len(d["labeled_frames"])
    assert len(d_roundtrip["videos"]) == len(d["videos"])
    assert len(d_roundtrip["skeletons"]) == len(d["skeletons"])
    # Note: We don't check exact equality because NaN != NaN


def test_to_dict_with_video_filter(slp_typical):
    """Test filtering by video."""
    labels = load_slp(slp_typical)

    # Get first video
    video = labels.videos[0]

    # Convert with filter
    d = to_dict(labels, video=video)

    # All frames should be from the specified video
    for frame in d["labeled_frames"]:
        assert frame["video_idx"] == 0


def test_to_dict_skip_empty_frames(slp_typical):
    """Test skipping empty frames."""
    labels = load_slp(slp_typical)

    # Count non-empty frames
    non_empty_count = sum(1 for lf in labels.labeled_frames if len(lf.instances) > 0)

    d = to_dict(labels, skip_empty_frames=True)

    assert len(d["labeled_frames"]) == non_empty_count


def test_to_dict_with_tracks():
    """Test dictionary conversion with tracked instances."""
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")
    track1 = Track("track1")
    track2 = Track("track2")

    # Create instances with tracks
    instance1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    instance2 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0], [7.0, 8.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance1, instance2])
    labels = Labels([lf])

    d = to_dict(labels)

    # Check tracks
    assert len(d["tracks"]) == 2
    assert d["tracks"][0]["name"] == "track1"
    assert d["tracks"][1]["name"] == "track2"

    # Check instances have track indices
    frame = d["labeled_frames"][0]
    assert len(frame["instances"]) == 2
    assert "track_idx" in frame["instances"][0]
    assert "track_idx" in frame["instances"][1]


def test_to_dict_predicted_vs_user_instances():
    """Test that predicted and user instances are distinguished."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    # Create predicted instance
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.95,
    )

    # Create user instance
    user_inst = Instance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[pred_inst, user_inst])
    labels = Labels([lf])

    d = to_dict(labels)

    frame = d["labeled_frames"][0]
    instances = frame["instances"]

    # Find predicted and user instances
    pred = next(inst for inst in instances if inst["type"] == "predicted_instance")
    user = next(inst for inst in instances if inst["type"] == "instance")

    # Predicted instance should have score
    assert "score" in pred
    assert pred["score"] == 0.95

    # User instance should not have score
    assert "score" not in user


def test_to_dict_with_symmetries():
    """Test dictionary conversion with skeleton symmetries."""
    from sleap_io.model.skeleton import Edge, Node, Symmetry

    node1 = Node("left")
    node2 = Node("right")
    node3 = Node("center")

    skeleton = Skeleton(
        nodes=[node1, node2, node3],
        edges=[Edge(node1, node3), Edge(node2, node3)],
        symmetries=[Symmetry([node1, node2])],
    )

    video = Video(filename="test.mp4")
    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    d = to_dict(labels)

    # Check symmetries
    skeleton_dict = d["skeletons"][0]
    assert "symmetries" in skeleton_dict
    assert len(skeleton_dict["symmetries"]) == 1
    assert skeleton_dict["symmetries"][0] == [0, 1]  # Indices of left and right


def test_to_dict_empty_labels():
    """Test converting empty labels."""
    labels = Labels()

    d = to_dict(labels)

    assert d["version"] == "1.0.0"
    assert len(d["skeletons"]) == 0
    assert len(d["videos"]) == 0
    assert len(d["tracks"]) == 0
    assert len(d["labeled_frames"]) == 0
    assert len(d["suggestions"]) == 0


def test_to_dict_provenance():
    """Test that provenance is preserved."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")
    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    # Add provenance
    labels.provenance = {
        "source": "test",
        "version": "1.0",
        "metadata": {"key": "value"},
    }

    d = to_dict(labels)

    assert d["provenance"] == labels.provenance
    assert d["provenance"]["source"] == "test"
    assert d["provenance"]["metadata"]["key"] == "value"


def test_from_dict_basic(slp_typical):
    """Test basic round-trip through dictionary."""
    labels = load_slp(slp_typical)

    # Convert to dict and back
    d = to_dict(labels)
    labels2 = from_dict(d)

    # Check basic structure
    assert len(labels2.skeletons) == len(labels.skeletons)
    assert len(labels2.videos) == len(labels.videos)
    assert len(labels2.labeled_frames) == len(labels.labeled_frames)
    assert len(labels2.tracks) == len(labels.tracks)

    # Check skeleton details
    for skel1, skel2 in zip(labels.skeletons, labels2.skeletons):
        assert skel1.name == skel2.name
        assert len(skel1.nodes) == len(skel2.nodes)
        for n1, n2 in zip(skel1.nodes, skel2.nodes):
            assert n1.name == n2.name


def test_from_dict_with_instances(slp_typical):
    """Test that instances are properly restored."""
    labels = load_slp(slp_typical)

    d = to_dict(labels)
    labels2 = from_dict(d)

    # Check instances count
    orig_instances = sum(len(lf) for lf in labels)
    restored_instances = sum(len(lf) for lf in labels2)
    assert orig_instances == restored_instances

    # Check instance types are preserved
    for lf1, lf2 in zip(labels.labeled_frames, labels2.labeled_frames):
        assert lf1.frame_idx == lf2.frame_idx
        assert len(lf1) == len(lf2)


def test_from_dict_json_roundtrip(slp_typical):
    """Test that from_dict works with JSON-serialized data."""
    labels = load_slp(slp_typical)

    # Full JSON round-trip
    d = to_dict(labels)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    labels2 = from_dict(d2)

    assert len(labels2.labeled_frames) == len(labels.labeled_frames)
    assert len(labels2.skeletons) == len(labels.skeletons)


def test_from_dict_with_tracks():
    """Test that tracks are properly restored."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")
    track = Track(name="animal1")

    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        score=0.95,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    d = to_dict(labels)
    labels2 = from_dict(d)

    assert len(labels2.tracks) == 1
    assert labels2.tracks[0].name == "animal1"

    # Check instance has track
    assert labels2.labeled_frames[0][0].track is not None
    assert labels2.labeled_frames[0][0].track.name == "animal1"


def test_from_dict_with_symmetries():
    """Test that symmetries are properly restored."""
    skeleton = Skeleton(["left_eye", "right_eye", "nose"])
    skeleton.add_symmetry("left_eye", "right_eye")

    video = Video(filename="test.mp4")
    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    d = to_dict(labels)
    labels2 = from_dict(d)

    # Check symmetries restored
    assert len(labels2.skeletons[0].symmetries) == 1
    sym_nodes = {n.name for n in labels2.skeletons[0].symmetries[0].nodes}
    assert sym_nodes == {"left_eye", "right_eye"}


def test_from_dict_empty():
    """Test converting empty dictionary."""
    d = {
        "version": "1.0.0",
        "skeletons": [],
        "videos": [],
        "tracks": [],
        "labeled_frames": [],
        "suggestions": [],
        "provenance": {},
    }

    labels = from_dict(d)

    assert len(labels.skeletons) == 0
    assert len(labels.videos) == 0
    assert len(labels.labeled_frames) == 0


def test_from_dict_missing_keys():
    """Test that missing required keys raise ValueError."""
    with pytest.raises(ValueError, match="Missing required key"):
        from_dict({"skeletons": []})

    with pytest.raises(ValueError, match="Missing required key"):
        from_dict({"videos": []})
