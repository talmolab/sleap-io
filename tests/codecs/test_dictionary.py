"""Tests for dictionary codec."""

import json
from pathlib import Path

import numpy as np
import pytest

from sleap_io import Instance, Labels, LabeledFrame, PredictedInstance, Skeleton, Track, Video, load_slp
from sleap_io.codecs.dictionary import to_dict


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
    assert len(d_roundtrip['labeled_frames']) == len(d['labeled_frames'])
    assert len(d_roundtrip['videos']) == len(d['videos'])
    assert len(d_roundtrip['skeletons']) == len(d['skeletons'])
    # Note: We don't check exact equality because NaN values don't compare equal (NaN != NaN)


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
