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


def test_to_dict_with_suggestions():
    """Test that suggestions are included in dict."""
    from sleap_io.model.suggestions import SuggestionFrame

    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")
    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    suggestion = SuggestionFrame(video=video, frame_idx=5)

    labels = Labels([lf])
    labels.suggestions = [suggestion]

    d = to_dict(labels)

    assert len(d["suggestions"]) == 1
    assert d["suggestions"][0]["frame_idx"] == 5
    assert d["suggestions"][0]["video_idx"] == 0


def test_from_dict_with_suggestions():
    """Test that suggestions are restored from dict."""
    d = {
        "version": "1.0.0",
        "skeletons": [{"name": "skel", "nodes": ["a"], "edges": []}],
        "videos": [{"filename": "test.mp4"}],
        "tracks": [],
        "labeled_frames": [],
        "suggestions": [{"frame_idx": 10, "video_idx": 0}],
        "provenance": {},
    }

    labels = from_dict(d)

    assert len(labels.suggestions) == 1
    assert labels.suggestions[0].frame_idx == 10


def test_to_dict_with_tracking_score():
    """Test that tracking_score is preserved."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        tracking_score=0.85,
        score=0.95,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    d = to_dict(labels)

    inst_dict = d["labeled_frames"][0]["instances"][0]
    assert inst_dict["tracking_score"] == pytest.approx(0.85)


def test_to_dict_video_filter_with_suggestions():
    """Test that suggestions are filtered by video."""
    from sleap_io.model.suggestions import SuggestionFrame

    skeleton = Skeleton(["node1"])
    video1 = Video(filename="video1.mp4")
    video2 = Video(filename="video2.mp4")

    instance1 = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )
    instance2 = Instance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
    )

    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[instance1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[instance2])

    labels = Labels([lf1, lf2])
    labels.suggestions = [
        SuggestionFrame(video=video1, frame_idx=5),
        SuggestionFrame(video=video2, frame_idx=10),
    ]

    # Filter to video1 only
    d = to_dict(labels, video=0)

    assert len(d["labeled_frames"]) == 1
    assert len(d["suggestions"]) == 1
    assert d["suggestions"][0]["frame_idx"] == 5


def test_to_dict_track_without_spawned_on():
    """Test that tracks without spawned_on don't include it in dict.

    Note: The spawned_on serialization code in dictionary.py is defensive for
    future compatibility. Current Track class doesn't have spawned_on attribute.
    """
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    d = to_dict(labels)

    # Track should not have spawned_on since it's not set
    assert len(d["tracks"]) == 1
    assert "spawned_on" not in d["tracks"][0]


def test_to_dict_with_from_predicted():
    """Test that from_predicted link is indicated in serialization.

    Covers line 200: has_from_predicted = True serialization.
    """
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    # Create predicted instance
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track,
    )

    # Create user instance linked to predicted
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        from_predicted=pred_inst,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[pred_inst, user_inst])
    labels = Labels([lf])

    d = to_dict(labels)

    # Find the user instance (type="instance")
    instances = d["labeled_frames"][0]["instances"]
    user_dict = next(i for i in instances if i["type"] == "instance")

    # Should have has_from_predicted flag
    assert user_dict.get("has_from_predicted") is True


def test_to_dict_with_video_shape(centered_pair_low_quality_video):
    """Test that video shape is serialized when available.

    Covers line 124: video_dict["shape"] = list(vid.shape)
    """
    skeleton = Skeleton(["node1"])
    video = centered_pair_low_quality_video

    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    d = to_dict(labels)

    # Video should have shape
    assert len(d["videos"]) == 1
    video_dict = d["videos"][0]
    assert "shape" in video_dict
    assert video_dict["shape"] is not None
    assert len(video_dict["shape"]) == 4  # (frames, height, width, channels)


def test_to_dict_with_video_backend(centered_pair_low_quality_video):
    """Test that video backend info is serialized.

    Covers line 128: video_dict["backend"] = {"type": type(vid.backend).__name__}
    """
    skeleton = Skeleton(["node1"])
    video = centered_pair_low_quality_video

    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    d = to_dict(labels)

    # Video should have backend info
    video_dict = d["videos"][0]
    assert "backend" in video_dict
    assert "type" in video_dict["backend"]
    assert video_dict["backend"]["type"] == "MediaVideo"


def test_labels_to_dict_wrapper(slp_typical):
    """Test Labels.to_dict() wrapper method calls the codec correctly."""
    labels = load_slp(slp_typical)

    # Call via the Labels method
    d = labels.to_dict()

    # Verify it returns the same result as the codec function
    assert "version" in d
    assert d["version"] == "1.0.0"
    assert "skeletons" in d
    assert "videos" in d
    assert "labeled_frames" in d

    # Test with video filter
    d_filtered = labels.to_dict(video=0)
    assert len(d_filtered["videos"]) == 1


# =============================================================================
# Lazy Fast Path Tests
# =============================================================================


def test_to_dict_lazy_matches_eager(slp_typical):
    """Test that lazy fast path produces same output as eager."""
    # Load eager
    labels_eager = load_slp(slp_typical, lazy=False)
    dict_eager = to_dict(labels_eager)

    # Load lazy
    labels_lazy = load_slp(slp_typical, lazy=True)
    assert labels_lazy.is_lazy
    dict_lazy = to_dict(labels_lazy)

    # Compare top-level structure
    assert set(dict_eager.keys()) == set(dict_lazy.keys())
    assert len(dict_eager["skeletons"]) == len(dict_lazy["skeletons"])
    assert len(dict_eager["videos"]) == len(dict_lazy["videos"])
    assert len(dict_eager["tracks"]) == len(dict_lazy["tracks"])
    assert len(dict_eager["labeled_frames"]) == len(dict_lazy["labeled_frames"])

    # Compare first frame in detail
    if dict_eager["labeled_frames"]:
        frame_eager = dict_eager["labeled_frames"][0]
        frame_lazy = dict_lazy["labeled_frames"][0]
        assert frame_eager["frame_idx"] == frame_lazy["frame_idx"]
        assert frame_eager["video_idx"] == frame_lazy["video_idx"]
        assert len(frame_eager["instances"]) == len(frame_lazy["instances"])


def test_to_dict_lazy_with_video_filter(slp_typical):
    """Test lazy fast path with video filtering."""
    labels_lazy = load_slp(slp_typical, lazy=True)
    labels_eager = load_slp(slp_typical, lazy=False)

    dict_lazy = to_dict(labels_lazy, video=0)
    dict_eager = to_dict(labels_eager, video=0)

    assert len(dict_lazy["labeled_frames"]) == len(dict_eager["labeled_frames"])


def test_to_dict_lazy_skip_empty_frames(slp_typical):
    """Test lazy fast path with skip_empty_frames."""
    labels_lazy = load_slp(slp_typical, lazy=True)
    labels_eager = load_slp(slp_typical, lazy=False)

    dict_lazy = to_dict(labels_lazy, skip_empty_frames=True)
    dict_eager = to_dict(labels_eager, skip_empty_frames=True)

    assert len(dict_lazy["labeled_frames"]) == len(dict_eager["labeled_frames"])
