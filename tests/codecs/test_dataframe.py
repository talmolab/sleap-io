"""Tests for dataframe codec."""

import numpy as np
import pandas as pd
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
from sleap_io.codecs.dataframe import (
    DataFrameFormat,
    from_dataframe,
    to_dataframe,
    to_dataframe_iter,
)


def test_dataframe_format_enum():
    """Test DataFrameFormat enum values."""
    assert DataFrameFormat.POINTS == "points"
    assert DataFrameFormat.INSTANCES == "instances"
    assert DataFrameFormat.FRAMES == "frames"
    assert DataFrameFormat.MULTI_INDEX == "multi_index"


def test_to_dataframe_points_format(slp_typical):
    """Test conversion to points format."""
    labels = load_slp(slp_typical)

    df = to_dataframe(labels, format="points", include_video=True)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Check required columns (new naming: node instead of node_name)
    assert "frame_idx" in df.columns
    assert "node" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns

    # Check metadata columns (new naming: track instead of track_name)
    assert "video_path" in df.columns
    assert "track" in df.columns
    # New columns for scores
    assert "track_score" in df.columns
    assert "instance_score" in df.columns

    # Check data types
    assert df["frame_idx"].dtype == np.int64
    assert df["x"].dtype == np.float64
    assert df["y"].dtype == np.float64


def test_to_dataframe_points_no_metadata():
    """Test points format without metadata."""
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")
    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_metadata=False)

    # Should not have metadata columns
    assert "video_path" not in df.columns
    assert "track" not in df.columns
    assert "track_score" not in df.columns
    assert "instance_score" not in df.columns

    # Should still have core columns
    assert "frame_idx" in df.columns
    assert "node" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns


def test_to_dataframe_instances_format():
    """Test conversion to instances format."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")
    track = Track("track1")

    instance = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skeleton,
        track=track,
        score=0.95,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    # Check base columns (new naming: track instead of track_name)
    assert "frame_idx" in df.columns
    assert "track" in df.columns
    assert "track_score" in df.columns
    assert "score" in df.columns

    # Check node-specific columns (new format: dot separator)
    assert "nose.x" in df.columns
    assert "nose.y" in df.columns
    assert "tail.x" in df.columns
    assert "tail.y" in df.columns

    # Check values
    row = df.iloc[0]
    assert row["frame_idx"] == 0
    assert row["track"] == "track1"
    assert row["nose.x"] == 10.0
    assert row["nose.y"] == 20.0
    assert row["tail.x"] == 30.0
    assert row["tail.y"] == 40.0


def test_to_dataframe_instances_with_scores():
    """Test instances format with scores."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.95]),
        score=0.95,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances", include_score=True)

    # New format: dot separator
    assert "node1.score" in df.columns
    assert df.iloc[0]["node1.score"] == 0.95
    # Instance-level score
    assert "score" in df.columns
    assert df.iloc[0]["score"] == 0.95


def test_to_dataframe_frames_format():
    """Test conversion to frames format (wide format with multiplexed instances)."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")
    track1 = Track("track1")
    track2 = Track("track2")

    # Frame 0: two instances
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0], [7.0, 8.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )

    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: one instance
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[2.0, 3.0], [4.0, 5.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )

    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])

    # Default: instance_id="index" - one row per frame
    df = to_dataframe(labels, format="frames")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # One row per frame (wide format)

    # Check columns for index mode: inst0.track, inst0.a.x, etc.
    assert "frame_idx" in df.columns
    assert "inst0.track" in df.columns
    assert "inst0.a.x" in df.columns
    assert "inst0.a.y" in df.columns
    assert "inst0.b.x" in df.columns
    assert "inst1.track" in df.columns
    assert "inst1.a.x" in df.columns

    # Check values for frame 0
    row0 = df[df["frame_idx"] == 0].iloc[0]
    assert row0["inst0.a.x"] == 1.0
    assert row0["inst1.a.x"] == 5.0

    # Check values for frame 1 - only one instance, inst1 should be NaN
    row1 = df[df["frame_idx"] == 1].iloc[0]
    assert row1["inst0.a.x"] == 2.0
    assert pd.isna(row1["inst1.a.x"])


def test_to_dataframe_multi_index_format(slp_typical):
    """Test conversion to multi-index format."""
    labels = load_slp(slp_typical)

    df = to_dataframe(labels, format="multi_index")

    assert isinstance(df, pd.DataFrame)

    # Check that columns are a MultiIndex
    assert isinstance(df.columns, pd.MultiIndex)

    # Index should be frame_idx
    assert df.index.name == "frame_idx" or "frame_idx" in str(df.index)


def test_to_dataframe_with_video_filter():
    """Test filtering by video."""
    skeleton = Skeleton(["node1"])
    video1 = Video(filename="video1.mp4")
    video2 = Video(filename="video2.mp4")

    inst1 = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    inst2 = Instance.from_numpy(points_data=np.array([[3.0, 4.0]]), skeleton=skeleton)

    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    labels = Labels([lf1, lf2])

    # Filter to video1
    df = to_dataframe(labels, format="points", video=video1)

    # Should only have data from video1
    assert all(df["video_path"] == "video1.mp4")
    assert len(df) == 1  # Only 1 point from video1


def test_to_dataframe_user_vs_predicted():
    """Test filtering by instance type."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
    labels = Labels([lf])

    # Only user instances
    df_user = to_dataframe(
        labels,
        format="points",
        include_user_instances=True,
        include_predicted_instances=False,
    )
    assert len(df_user) == 1
    # User instances have no scores (NaN)
    assert pd.isna(df_user.iloc[0]["instance_score"])

    # Only predicted instances
    df_pred = to_dataframe(
        labels,
        format="points",
        include_user_instances=False,
        include_predicted_instances=True,
    )
    assert len(df_pred) == 1
    # Predicted instances have scores
    assert df_pred.iloc[0]["instance_score"] == 0.9

    # Both
    df_both = to_dataframe(
        labels,
        format="points",
        include_user_instances=True,
        include_predicted_instances=True,
    )
    assert len(df_both) == 2


def test_to_dataframe_invalid_format():
    """Test handling of invalid format."""
    labels = Labels()

    with pytest.raises(ValueError, match="Invalid format"):
        to_dataframe(labels, format="invalid_format")


def test_to_dataframe_polars_backend_not_installed():
    """Test error when polars is requested but not installed."""
    labels = Labels()

    # This might pass if polars is installed, so we only test the error path
    try:
        df = to_dataframe(labels, format="points", backend="polars")
        # If we get here, polars is installed
        import polars as pl

        assert isinstance(df, pl.DataFrame)
    except ValueError as e:
        # Polars not installed
        assert "polars is not installed" in str(e)


def test_to_dataframe_empty_labels():
    """Test converting empty labels."""
    labels = Labels()

    df = to_dataframe(labels, format="points")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_to_dataframe_points_format_comprehensive():
    """Comprehensive test of points format with all features."""
    from sleap_io.model.skeleton import Edge, Node

    # Create skeleton with edges
    nodes = [Node("a"), Node("b"), Node("c")]
    edges = [Edge(nodes[0], nodes[1]), Edge(nodes[1], nodes[2])]
    skeleton = Skeleton(nodes=nodes, edges=edges)

    video = Video(filename="test.mp4")
    track = Track("mouse1")

    # Create instance with specific point scores
    points_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    point_scores = np.array([0.9, 0.8, 0.95])

    instance = PredictedInstance.from_numpy(
        points_data=points_data,
        skeleton=skeleton,
        point_scores=point_scores,
        score=0.88,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=10, instances=[instance])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_score=True)

    assert len(df) == 3  # 3 points
    assert df["frame_idx"].unique()[0] == 10
    # New naming: node instead of node_name, track instead of track_name
    assert set(df["node"]) == {"a", "b", "c"}
    assert df["track"].unique()[0] == "mouse1"

    # Check scores
    assert df[df["node"] == "a"]["score"].iloc[0] == 0.9
    assert df[df["node"] == "b"]["score"].iloc[0] == 0.8
    assert df[df["node"] == "c"]["score"].iloc[0] == 0.95

    # Check instance-level score
    assert df["instance_score"].unique()[0] == 0.88


def test_to_dataframe_format_string_normalization():
    """Test that format strings are properly normalized."""
    labels = Labels()

    # All of these should work
    df1 = to_dataframe(labels, format="points")
    df2 = to_dataframe(labels, format="POINTS")
    df3 = to_dataframe(labels, format=DataFrameFormat.POINTS)

    # All should produce the same result
    assert df1.equals(df2)
    assert df2.equals(df3)


def test_to_dataframe_video_id_index():
    """Test using video index instead of path."""
    skeleton = Skeleton(["node1"])
    video1 = Video(filename="video1.mp4")
    video2 = Video(filename="video2.mp4")

    inst1 = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    inst2 = Instance.from_numpy(points_data=np.array([[3.0, 4.0]]), skeleton=skeleton)

    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    labels = Labels([lf1, lf2])

    # Use video index
    df = to_dataframe(labels, format="points", video_id="index")

    # Should have video_idx column instead of video_path
    assert "video_idx" in df.columns
    assert "video_path" not in df.columns
    assert df["video_idx"].tolist() == [0, 1]


def test_to_dataframe_video_id_name():
    """Test using just video filename (no directory)."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="/path/to/video.mp4")

    inst = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", video_id="name", include_video=True)

    # Should have just the filename
    assert "video_path" in df.columns
    assert df["video_path"].iloc[0] == "video.mp4"


def test_to_dataframe_video_id_object():
    """Test storing Video objects directly."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", video_id="object", include_video=True)

    # Should have video column with Video objects
    assert "video" in df.columns
    assert isinstance(df["video"].iloc[0], Video)
    assert df["video"].iloc[0] == video


def test_to_dataframe_include_video_false():
    """Test omitting video information entirely."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_video=False)

    # Should not have any video columns
    assert "video_path" not in df.columns
    assert "video_idx" not in df.columns
    assert "video" not in df.columns


def test_to_dataframe_single_video_auto_omit():
    """Test that single video automatically omits video column."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # With single video, should auto-omit
    df = to_dataframe(labels, format="points")

    # Should not have video column by default
    assert "video_path" not in df.columns


def test_to_dataframe_multi_video_auto_include():
    """Test that multiple videos automatically includes video column."""
    skeleton = Skeleton(["node1"])
    video1 = Video(filename="video1.mp4")
    video2 = Video(filename="video2.mp4")

    inst1 = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    inst2 = Instance.from_numpy(points_data=np.array([[3.0, 4.0]]), skeleton=skeleton)

    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    labels = Labels([lf1, lf2])

    # With multiple videos, should auto-include
    df = to_dataframe(labels, format="points")

    # Should have video column by default
    assert "video_path" in df.columns


def test_to_dataframe_instances_format_video_options():
    """Test video options work with instances format."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Test with video index
    df_idx = to_dataframe(
        labels, format="instances", video_id="index", include_video=True
    )
    assert "video_idx" in df_idx.columns

    # Test without video
    df_no_video = to_dataframe(labels, format="instances", include_video=False)
    assert "video_path" not in df_no_video.columns
    assert "video_idx" not in df_no_video.columns


def test_to_dataframe_multi_index_without_video():
    """Test multi-index format without video level."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")
    track = Track("track1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="multi_index", include_video=False)

    # Should have multi-index columns without video level
    assert isinstance(df.columns, pd.MultiIndex)
    # Column levels should be: skeleton, track, node, coord (no video)
    assert df.columns.nlevels < 5  # Less than with video


def test_from_dataframe_basic():
    """Test basic round-trip through DataFrame."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Convert to dataframe and back
    df = to_dataframe(labels, format="points", include_metadata=True)
    labels2 = from_dataframe(df, skeleton=skeleton, video=video)

    assert len(labels2.labeled_frames) == 1
    assert len(labels2.labeled_frames[0]) == 1


def test_from_dataframe_with_predicted():
    """Test that predicted instances are properly restored."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skeleton,
        score=0.95,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_metadata=True)
    labels2 = from_dataframe(df, skeleton=skeleton, video=video)

    assert len(labels2.labeled_frames[0].predicted_instances) == 1
    assert len(labels2.labeled_frames[0].user_instances) == 0


def test_from_dataframe_multiple_instances():
    """Test with multiple instances per frame."""
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")

    inst1 = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
    )
    inst2 = Instance.from_numpy(
        points_data=np.array([[5.0, 6.0], [7.0, 8.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_metadata=True)
    labels2 = from_dataframe(df, skeleton=skeleton, video=video)

    assert len(labels2.labeled_frames) == 1
    assert len(labels2.labeled_frames[0]) == 2


def test_from_dataframe_with_tracks():
    """Test that tracks are restored."""
    skeleton = Skeleton(["head"])
    video = Video(filename="test.mp4")
    track = Track(name="animal1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_metadata=True)
    labels2 = from_dataframe(df, skeleton=skeleton, video=video)

    assert len(labels2.tracks) == 1
    assert labels2.tracks[0].name == "animal1"
    assert labels2.labeled_frames[0][0].track is not None


def test_from_dataframe_multiple_frames():
    """Test with multiple frames."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    frames = []
    for i in range(5):
        inst = Instance.from_numpy(
            points_data=np.array([[float(i), float(i * 2)]]),
            skeleton=skeleton,
        )
        frames.append(LabeledFrame(video=video, frame_idx=i * 10, instances=[inst]))

    labels = Labels(frames)

    df = to_dataframe(labels, format="points", include_metadata=True)
    labels2 = from_dataframe(df, skeleton=skeleton, video=video)

    assert len(labels2.labeled_frames) == 5

    # Check frame indices
    frame_indices = sorted([lf.frame_idx for lf in labels2.labeled_frames])
    assert frame_indices == [0, 10, 20, 30, 40]


def test_from_dataframe_slp_roundtrip(slp_typical):
    """Test round-trip with real SLP file."""
    labels = load_slp(slp_typical)

    df = to_dataframe(labels, format="points", include_metadata=True)
    labels2 = from_dataframe(df, skeleton=labels.skeleton, video=labels.video)

    assert len(labels2.labeled_frames) == len(labels.labeled_frames)

    orig_instances = sum(len(lf) for lf in labels)
    restored_instances = sum(len(lf) for lf in labels2)
    assert orig_instances == restored_instances


def test_from_dataframe_infer_skeleton():
    """Test skeleton inference from DataFrame."""
    skeleton = Skeleton(["head", "body", "tail"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_metadata=True)

    # Restore without providing skeleton - should infer from node names
    labels2 = from_dataframe(df, video=video)

    assert len(labels2.skeletons) == 1
    assert len(labels2.skeletons[0].nodes) == 3

    # Node names should match
    node_names = {n.name for n in labels2.skeletons[0].nodes}
    assert node_names == {"head", "body", "tail"}


def test_from_dataframe_instances_format():
    """Test that instances format decoder works."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")

    # Create DataFrame in instances format
    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "track": ["track1"],
            "nose.x": [10.0],
            "nose.y": [20.0],
            "tail.x": [30.0],
            "tail.y": [40.0],
        }
    )

    labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1
    inst = labels.labeled_frames[0].instances[0]
    assert inst.numpy()[0, 0] == pytest.approx(10.0)
    assert inst.numpy()[1, 0] == pytest.approx(30.0)


def test_from_dataframe_missing_columns():
    """Test that missing columns raise ValueError."""
    df = pd.DataFrame({"frame_idx": [0], "x": [1.0], "y": [2.0]})  # Missing node

    with pytest.raises(ValueError, match="Missing required columns"):
        from_dataframe(df)


def test_from_dataframe_with_video_idx():
    """Test from_dataframe with video_idx column."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Create DataFrame with video_idx
    df = to_dataframe(
        labels,
        format="points",
        include_metadata=True,
        video_id="index",
        include_video=True,
    )
    assert "video_idx" in df.columns

    # Convert back
    labels2 = from_dataframe(df, skeleton=skeleton, video=video)
    assert len(labels2.labeled_frames) == 1


def test_from_dataframe_with_video_object():
    """Test from_dataframe with video object column."""
    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Create DataFrame with video object
    df = to_dataframe(
        labels,
        format="points",
        include_metadata=True,
        video_id="object",
        include_video=True,
    )
    assert "video" in df.columns
    assert isinstance(df["video"].iloc[0], Video)

    # Convert back - should handle Video objects
    labels2 = from_dataframe(df, skeleton=skeleton)
    assert len(labels2.labeled_frames) == 1


def test_from_dataframe_no_video_no_skeleton():
    """Test from_dataframe creates defaults when not provided."""
    # Create minimal DataFrame with new column naming
    df = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "node": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
        }
    )

    # Should create default video and skeleton
    labels = from_dataframe(df)
    assert len(labels.videos) == 1
    assert len(labels.skeletons) == 1
    assert labels.videos[0].filename == "video.mp4"


def test_from_dataframe_video_idx_no_video_provided():
    """Test from_dataframe with video_idx but no video object provided."""
    df = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "node": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "video_idx": [0, 0],
        }
    )

    skeleton = Skeleton(["a", "b"])
    labels = from_dataframe(df, skeleton=skeleton)

    # Should create placeholder video
    assert len(labels.videos) == 1
    assert "video_0" in labels.videos[0].filename


def test_from_dataframe_invalid_format_string():
    """Test from_dataframe with invalid format string."""
    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "node": ["a"],
            "x": [1.0],
            "y": [2.0],
        }
    )

    with pytest.raises(ValueError, match="Invalid format"):
        from_dataframe(df, format="invalid_format")


def test_to_dataframe_frames_format_basic():
    """Test frames format produces correct output."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames")
    assert len(df) == 1  # One row per frame (wide format)
    # Frames format should have node columns with dot separator
    assert "inst0.head.x" in df.columns
    assert "inst0.tail.x" in df.columns


def test_to_dataframe_frames_format_with_track():
    """Test frames format with tracked instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames")
    assert len(df) == 1  # One row per frame
    assert "inst0.track" in df.columns
    assert df.iloc[0]["inst0.track"] == "animal1"


def test_to_dataframe_multi_index_with_video_idx():
    """Test multi_index format with video_idx."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(
        labels, format="multi_index", video_id="index", include_video=True
    )
    # When include_video=True, columns may be mixed (regular + tuple)
    assert "video_idx" in df.columns
    # Check that instance columns exist as tuples
    assert any(isinstance(c, tuple) for c in df.columns)


def test_to_dataframe_video_as_int_index():
    """Test video parameter as integer index.

    Covers line 150: video = labels.videos[video] when video is int.
    """
    skeleton = Skeleton(["pt"])
    video1 = Video(filename="video1.mp4")
    video2 = Video(filename="video2.mp4")

    inst1 = Instance.from_numpy(points_data=np.array([[1.0, 2.0]]), skeleton=skeleton)
    inst2 = Instance.from_numpy(points_data=np.array([[3.0, 4.0]]), skeleton=skeleton)

    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    labels = Labels([lf1, lf2])

    # Filter by video index
    df = to_dataframe(labels, format="points", video=1)

    assert len(df) == 1
    assert df.iloc[0]["x"] == 3.0  # From video2


def test_to_dataframe_instances_video_id_object():
    """Test instances format with video_id='object'.

    Covers lines 342-343: video_id == 'object' in instances format.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances", video_id="object", include_video=True)

    assert "video" in df.columns
    assert df["video"].iloc[0] is video  # Should be the actual Video object


def test_to_dataframe_instances_video_id_index():
    """Test instances format with video_id='index'.

    Covers lines 340-341: video_id == 'index' in instances format.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances", video_id="index", include_video=True)

    assert "video_idx" in df.columns
    assert df["video_idx"].iloc[0] == 0


def test_to_dataframe_frames_video_id_object():
    """Test frames format with video_id='object'."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", video_id="object", include_video=True)

    assert "video" in df.columns
    assert df["video"].iloc[0] is video


def test_to_dataframe_frames_video_id_index():
    """Test frames format with video_id='index'."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", video_id="index", include_video=True)

    assert "video_idx" in df.columns
    assert df["video_idx"].iloc[0] == 0


def test_to_dataframe_multi_index_video_id_object():
    """Test multi_index format with video_id='object'.

    Multi-index with objects converts to string representation.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(
        labels, format="multi_index", video_id="object", include_video=True
    )

    # When include_video=True, we have mixed columns (regular + tuple)
    assert "video" in df.columns or any("video" in str(c) for c in df.columns)
    # Check that instance columns exist as tuples
    assert any(isinstance(c, tuple) for c in df.columns)


def test_to_dataframe_multi_index_empty():
    """Test multi_index format with no data returns empty DataFrame.

    Covers line 506: Return empty DataFrame when data_list is empty.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    labels = Labels()
    labels.videos.append(video)
    labels.skeletons.append(skeleton)

    df = to_dataframe(labels, format="multi_index")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_from_dataframe_minimal_columns():
    """Test from_dataframe with minimal columns (no scores, no track)."""
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")

    # Create DataFrame with minimal columns (no track, no scores)
    data = {
        "frame_idx": [0, 0],
        "node": ["node1", "node2"],
        "x": [1.0, 3.0],
        "y": [2.0, 4.0],
    }
    df = pd.DataFrame(data)

    labels = from_dataframe(df, video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1


def test_from_dataframe_unknown_node():
    """Test from_dataframe handles nodes not in skeleton gracefully."""
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")

    # Create DataFrame with an extra node not in skeleton
    data = {
        "frame_idx": [0, 0, 0],
        "node": ["node1", "node2", "unknown_node"],  # unknown_node not in skeleton
        "x": [1.0, 3.0, 99.0],
        "y": [2.0, 4.0, 99.0],
    }
    df = pd.DataFrame(data)

    labels = from_dataframe(df, video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    inst = labels.labeled_frames[0].instances[0]

    # Should have points for node1 and node2, unknown_node should be skipped
    assert inst.numpy()[0, 0] == pytest.approx(1.0)  # node1 x
    assert inst.numpy()[1, 0] == pytest.approx(3.0)  # node2 x


def test_to_dataframe_polars_backend():
    """Test polars backend for to_dataframe.

    Covers lines 32 (HAS_POLARS=True) and 209 (pl.from_pandas conversion).
    """
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", backend="polars")

    assert isinstance(df, pl.DataFrame)
    assert "frame_idx" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert df["x"][0] == 1.0
    assert df["y"][0] == 2.0


def test_to_dataframe_polars_instances_format():
    """Test polars backend with instances format."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")
    track = Track("track1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances", backend="polars")

    assert isinstance(df, pl.DataFrame)
    assert "frame_idx" in df.columns
    # New format: dot separator
    assert "a.x" in df.columns
    assert "b.y" in df.columns


def test_to_dataframe_polars_frames_format():
    """Test polars backend with frames format."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", backend="polars")

    assert isinstance(df, pl.DataFrame)
    assert "frame_idx" in df.columns
    # New format: wide with inst0 prefix
    assert "inst0.track" in df.columns


def test_to_dataframe_polars_multi_index_format():
    """Test polars backend with multi_index format."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="multi_index", backend="polars")

    # Multi-index format returns a polars DataFrame converted from pandas MultiIndex
    assert isinstance(df, pl.DataFrame)


# ============================================================================
# Tests for new Phase 1 features
# ============================================================================


def test_to_dataframe_frames_instance_id_track():
    """Test frames format with instance_id='track' (track-named columns)."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")
    track1 = Track("mouse1")
    track2 = Track("mouse2")

    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0], [7.0, 8.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", instance_id="track")

    assert len(df) == 1  # One row per frame
    # Track-named columns: mouse1.head.x, mouse2.head.x, etc.
    assert "mouse1.head.x" in df.columns
    assert "mouse1.tail.x" in df.columns
    assert "mouse2.head.x" in df.columns
    # No .track column in track mode
    assert "mouse1.track" not in df.columns
    # But track_score and score columns exist
    assert "mouse1.track_score" in df.columns
    assert "mouse1.score" in df.columns


def test_to_dataframe_frames_untracked_error():
    """Test that untracked instances raise error by default in track mode."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Instance without a track
    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    with pytest.raises(ValueError, match="has no track"):
        to_dataframe(labels, format="frames", instance_id="track", untracked="error")


def test_to_dataframe_frames_untracked_ignore():
    """Test that untracked instances are silently skipped with untracked='ignore'."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("tracked_one")

    # One tracked, one untracked
    tracked_inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )
    untracked_inst = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        score=0.8,
    )

    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[tracked_inst, untracked_inst]
    )
    labels = Labels([lf])

    # Should not raise, should only have tracked_one columns
    df = to_dataframe(labels, format="frames", instance_id="track", untracked="ignore")

    assert len(df) == 1
    assert "tracked_one.pt.x" in df.columns
    # Untracked instance is skipped


def test_to_dataframe_multi_index_instance_id_track():
    """Test multi_index format with instance_id='track'."""
    skeleton = Skeleton(["head"])
    video = Video(filename="test.mp4")
    track1 = Track("animal1")
    track2 = Track("animal2")

    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels([lf])

    df = to_dataframe(
        labels, format="multi_index", instance_id="track", include_video=False
    )

    # Should have tuple columns with track names as first level
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    assert len(tuple_cols) > 0

    # Check that track names appear in column tuples
    col_str = str(tuple_cols)
    assert "animal1" in col_str
    assert "animal2" in col_str


def test_from_dataframe_frames_format():
    """Test from_dataframe with frames format (wide format)."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0], [7.0, 8.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels([lf])

    # Encode and decode
    df = to_dataframe(labels, format="frames")
    labels2 = from_dataframe(df, format="frames", video=video, skeleton=skeleton)

    assert len(labels2.labeled_frames) == 1
    assert len(labels2.labeled_frames[0].instances) == 2


def test_from_dataframe_multi_index_format():
    """Test from_dataframe with multi_index format."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("animal")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Encode and decode
    df = to_dataframe(labels, format="multi_index", include_video=False)
    labels2 = from_dataframe(df, format="multi_index", video=video, skeleton=skeleton)

    assert len(labels2.labeled_frames) == 1
    assert len(labels2.labeled_frames[0].instances) == 1


def test_from_dataframe_instances_with_scores():
    """Test from_dataframe with instances format including scores."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    # Create DataFrame with dot-separator format
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1],
            "track": ["t1", "t1"],
            "track_score": [0.98, 0.97],
            "score": [0.9, 0.85],
            "a.x": [1.0, 2.0],
            "a.y": [2.0, 3.0],
            "a.score": [0.95, 0.92],
            "b.x": [3.0, 4.0],
            "b.y": [4.0, 5.0],
            "b.score": [0.88, 0.87],
        }
    )

    labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 2
    assert len(labels.tracks) == 1
    assert labels.tracks[0].name == "t1"

    # Check scores were preserved
    inst0 = labels.labeled_frames[0].instances[0]
    assert isinstance(inst0, PredictedInstance)
    assert inst0.score == pytest.approx(0.9)
    assert inst0.tracking_score == pytest.approx(0.98)


def test_from_dataframe_points_alternative_column_names():
    """Test that from_dataframe accepts alternative column names."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    # Alternative column names for external tool compatibility
    df = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "node_name": ["a", "b"],  # Alternative: node_name instead of node
            "x": [1.0, 3.0],
            "y": [2.0, 4.0],
            "track_name": ["t1", "t1"],  # Alternative: track_name instead of track
        }
    )

    labels = from_dataframe(df, format="points", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.tracks) == 1
    assert labels.tracks[0].name == "t1"


def test_to_dataframe_frames_empty_labels():
    """Test frames format with empty labels returns empty DataFrame."""
    labels = Labels()
    df = to_dataframe(labels, format="frames")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_to_dataframe_frames_no_skeleton():
    """Test frames format when instances have no skeleton returns empty DataFrame."""
    video = Video(filename="test.mp4")
    labels = Labels()
    labels.videos.append(video)

    df = to_dataframe(labels, format="frames")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_frames_roundtrip_with_track_mode():
    """Test roundtrip encoding/decoding with instance_id='track'."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")
    track1 = Track("mouse1")
    track2 = Track("mouse2")

    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.95,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[50.0, 60.0], [70.0, 80.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.88,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels([lf])

    # Encode with track mode
    df = to_dataframe(labels, format="frames", instance_id="track")

    # Decode - should reconstruct with tracks
    labels2 = from_dataframe(df, format="frames", video=video, skeleton=skeleton)

    assert len(labels2.labeled_frames) == 1
    assert len(labels2.labeled_frames[0].instances) == 2
    assert len(labels2.tracks) == 2

    # Check track names
    track_names = {t.name for t in labels2.tracks}
    assert track_names == {"mouse1", "mouse2"}


def test_points_format_with_instance_scores():
    """Test that points format includes both track_score and instance_score."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.95,
        tracking_score=0.98,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points")

    assert "track_score" in df.columns
    assert "instance_score" in df.columns
    assert df.iloc[0]["track_score"] == pytest.approx(0.98)
    assert df.iloc[0]["instance_score"] == pytest.approx(0.95)


def test_instances_format_with_instance_scores():
    """Test that instances format includes both track_score and instance score."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.95,
        tracking_score=0.98,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances")

    assert "track_score" in df.columns
    assert "score" in df.columns  # Instance-level score
    assert df.iloc[0]["track_score"] == pytest.approx(0.98)
    assert df.iloc[0]["score"] == pytest.approx(0.95)


def test_frames_format_padding_with_nan():
    """Test that frames format pads missing instances with NaN."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Frame 0: 2 instances
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: 1 instance only
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])
    df = to_dataframe(labels, format="frames")

    # Frame 1 should have NaN for inst1
    row1 = df[df["frame_idx"] == 1].iloc[0]
    assert pd.isna(row1["inst1.pt.x"])
    assert pd.isna(row1["inst1.pt.y"])


def test_frames_format_user_instances():
    """Test frames format with user (non-predicted) instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames")

    assert len(df) == 1
    assert df.iloc[0]["inst0.pt.x"] == 1.0
    # User instances have None for scores
    assert df.iloc[0]["inst0.track_score"] is None
    assert df.iloc[0]["inst0.score"] is None


def test_multi_index_format_user_instances():
    """Test multi_index format with user instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="multi_index", include_video=False)

    # Should have tuple columns
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    assert len(tuple_cols) > 0


def test_from_frames_track_mode_columns():
    """Test from_dataframe with track-mode columns (track names as prefixes)."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")

    # Create DataFrame with track names as prefixes
    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "mouse1.track_score": [0.98],
            "mouse1.score": [0.95],
            "mouse1.head.x": [1.0],
            "mouse1.head.y": [2.0],
            "mouse1.tail.x": [3.0],
            "mouse1.tail.y": [4.0],
            "mouse2.track_score": [0.97],
            "mouse2.score": [0.88],
            "mouse2.head.x": [5.0],
            "mouse2.head.y": [6.0],
            "mouse2.tail.x": [7.0],
            "mouse2.tail.y": [8.0],
        }
    )

    labels = from_dataframe(df, format="frames", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 2
    assert len(labels.tracks) == 2


def test_from_instances_no_scores():
    """Test from_dataframe instances format without score columns."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "a.x": [1.0],
            "a.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1


def test_to_dataframe_frames_with_video_path():
    """Test frames format with video path included."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="/path/to/test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", include_video=True, video_id="path")

    assert "video_path" in df.columns
    assert df.iloc[0]["video_path"] == "/path/to/test.mp4"


def test_to_dataframe_multi_index_untracked_error():
    """Test multi_index format raises error with untracked instances in track mode."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    with pytest.raises(ValueError, match="has no track"):
        to_dataframe(
            labels, format="multi_index", instance_id="track", untracked="error"
        )


def test_from_frames_with_video_column():
    """Test from_dataframe frames format with video column."""
    skeleton = Skeleton(["pt"])

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "video_path": ["test.mp4"],
            "inst0.track": ["t1"],
            "inst0.pt.x": [1.0],
            "inst0.pt.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="frames", skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.videos) == 1
    assert "test.mp4" in labels.videos[0].filename


def test_from_instances_with_video_idx():
    """Test from_dataframe instances format with video_idx column."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "video_idx": [0],
            "a.x": [1.0],
            "a.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1


def test_frames_format_without_metadata():
    """Test frames format without metadata."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", include_metadata=False)

    assert "frame_idx" in df.columns
    assert "inst0.pt.x" in df.columns
    # Without metadata, track/track_score/score still included per spec
    assert "inst0.track" in df.columns


def test_from_frames_missing_instance_data():
    """Test from_dataframe handles frames with missing instance data."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Frame 0 has inst0 data, Frame 1 has only inst0 data (inst1 is NaN)
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1],
            "inst0.pt.x": [1.0, 2.0],
            "inst0.pt.y": [2.0, 3.0],
            "inst1.pt.x": [3.0, np.nan],  # NaN in frame 1
            "inst1.pt.y": [4.0, np.nan],
        }
    )

    labels = from_dataframe(df, format="frames", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 2
    # Frame 0 should have 2 instances
    assert len(labels.labeled_frames[0].instances) == 2
    # Frame 1 should have 1 instance (inst1 is all NaN)
    assert len(labels.labeled_frames[1].instances) == 1


def test_multi_index_empty():
    """Test multi_index format returns empty DataFrame for empty labels."""
    labels = Labels()
    df = to_dataframe(labels, format="multi_index")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_from_instances_multiple_frames():
    """Test from_dataframe instances format with multiple frames."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "track": ["t1", "t1", "t1"],
            "a.x": [1.0, 2.0, 3.0],
            "a.y": [2.0, 3.0, 4.0],
            "b.x": [3.0, 4.0, 5.0],
            "b.y": [4.0, 5.0, 6.0],
        }
    )

    labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 3


def test_instances_format_no_video_column():
    """Test instances format decoding without video column uses provided video."""
    skeleton = Skeleton(["a"])
    video = Video(filename="my_video.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "a.x": [1.0],
            "a.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)

    assert labels.videos[0].filename == "my_video.mp4"


def test_frames_format_track_mode_with_missing_track():
    """Test frames format track mode with some frames missing a track."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Frame 0: both tracks
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: only track1
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])
    df = to_dataframe(labels, format="frames", instance_id="track")

    # t2 should have NaN in frame 1
    row1 = df[df["frame_idx"] == 1].iloc[0]
    assert pd.isna(row1["t2.pt.x"])


def test_from_points_with_skeleton_name():
    """Test from_dataframe points format with skeleton_name column."""
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "node": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            "skeleton_name": ["my_skeleton", "my_skeleton"],
        }
    )

    labels = from_dataframe(df, video=video)

    assert len(labels.skeletons) == 1
    assert labels.skeletons[0].name == "my_skeleton"


def test_from_instances_with_video_object_column():
    """Test from_dataframe instances format with Video object column."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "video": [video],
            "a.x": [1.0],
            "a.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="instances", skeleton=skeleton)

    assert len(labels.videos) == 1
    assert labels.videos[0] is video


def test_from_frames_with_video_idx_no_provided_video():
    """Test from_dataframe frames format with video_idx but no video provided."""
    skeleton = Skeleton(["pt"])

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "video_idx": [0],
            "inst0.pt.x": [1.0],
            "inst0.pt.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="frames", skeleton=skeleton)

    assert len(labels.videos) == 1
    assert "video_0" in labels.videos[0].filename


def test_multi_index_no_skeleton():
    """Test multi_index format returns empty when no skeleton can be found."""
    video = Video(filename="test.mp4")
    labels = Labels()
    labels.videos.append(video)

    df = to_dataframe(labels, format="multi_index")
    assert df.empty


def test_frames_no_track_mode_without_tracks():
    """Test frames format with track mode when labels has no tracks."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Should fail with track mode and untracked=error
    with pytest.raises(ValueError, match="has no track"):
        to_dataframe(labels, format="frames", instance_id="track")


def test_instances_format_missing_y_column():
    """Test instances format decoder validates required x and y columns."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")

    # Missing a.y column
    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "a.x": [1.0],
            # a.y is missing
        }
    )

    with pytest.raises(ValueError, match="missing x or y"):
        from_dataframe(df, format="instances", video=video, skeleton=skeleton)


def test_frames_format_no_node_columns():
    """Test frames format decoder validates node columns exist."""
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "some_column": [1.0],
        }
    )

    with pytest.raises(ValueError, match="No instance columns"):
        from_dataframe(df, format="frames", video=video)


def test_instances_format_no_node_columns():
    """Test instances format decoder validates node columns exist."""
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "track": ["t1"],
        }
    )

    with pytest.raises(ValueError, match="No node columns"):
        from_dataframe(df, format="instances", video=video)


def test_from_multi_index_with_flat_columns():
    """Test from_dataframe multi_index format with already flat columns."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Create a DataFrame with flat columns that look like they came from flattening
    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "inst0.pt.x": [1.0],
            "inst0.pt.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="multi_index", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1


def test_from_multi_index_with_true_multi_index():
    """Test from_dataframe multi_index format with actual MultiIndex columns."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Create a true multi-index DataFrame
    columns = pd.MultiIndex.from_tuples(
        [("inst0", "pt", "x"), ("inst0", "pt", "y")],
        names=["instance", "node", "coord"],
    )
    df = pd.DataFrame(
        [[1.0, 2.0]], columns=columns, index=pd.Index([0], name="frame_idx")
    )

    labels = from_dataframe(df, format="multi_index", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1


def test_points_format_without_scores():
    """Test points format when include_score=False."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.95]),
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", include_score=False)

    assert "score" not in df.columns


def test_instances_format_without_scores():
    """Test instances format when include_score=False."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.95]),
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances", include_score=False)

    assert "pt.score" not in df.columns


def test_frames_format_without_scores():
    """Test frames format when include_score=False."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", include_score=False)

    assert "inst0.pt.score" not in df.columns


def test_multi_index_track_mode_missing_tracks():
    """Test multi_index format track mode handles missing tracks in frames."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("animal1")
    track2 = Track("animal2")

    # Frame 0: both tracks
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: only track1
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])
    df = to_dataframe(
        labels, format="multi_index", instance_id="track", include_video=False
    )

    # Should have columns for both tracks, with NaN for animal2 in frame 1
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    assert len(tuple_cols) > 0
    col_str = str(tuple_cols)
    assert "animal1" in col_str
    assert "animal2" in col_str


def test_multi_index_index_mode_padding():
    """Test multi_index format index mode pads with NaN for missing instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    # Frame 0: 2 instances
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        score=0.8,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: 1 instance
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.85,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])
    df = to_dataframe(labels, format="multi_index", include_video=False)

    # Should have columns for inst0 and inst1
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    col_str = str(tuple_cols)
    assert "inst0" in col_str
    assert "inst1" in col_str


def test_from_points_with_instance_type_column():
    """Test from_dataframe points format handles instance_type column."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "node": ["a", "a"],
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            "instance_type": ["user", "predicted"],
        }
    )

    labels = from_dataframe(df, video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 2
    # One user, one predicted
    user_count = len(labels.labeled_frames[0].user_instances)
    pred_count = len(labels.labeled_frames[0].predicted_instances)
    assert user_count == 1
    assert pred_count == 1


def test_from_points_with_track_and_instance_type():
    """Test from_dataframe points with both track and instance_type columns."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0, 0, 0, 0],
            "node": ["a", "b", "a", "b"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "track": ["t1", "t1", "t2", "t2"],
            "instance_type": ["user", "user", "predicted", "predicted"],
        }
    )

    labels = from_dataframe(df, video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 2
    assert len(labels.tracks) == 2


def test_multi_index_format_video_path():
    """Test multi_index format with video_path column."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="/path/to/video.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="multi_index", include_video=True, video_id="path")

    assert "video_path" in df.columns


def test_from_frames_frame_idx_as_index():
    """Test from_dataframe frames format when frame_idx is in the index."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "inst0.pt.x": [1.0],
            "inst0.pt.y": [2.0],
        }
    )
    df.index = pd.Index([0], name="frame_idx")

    labels = from_dataframe(df, format="multi_index", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert labels.labeled_frames[0].frame_idx == 0


def test_multi_index_track_mode_user_instances():
    """Test multi_index format track mode with user instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    # User instance with track
    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(
        labels, format="multi_index", instance_id="track", include_video=False
    )

    # Should have columns for animal1
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    col_str = str(tuple_cols)
    assert "animal1" in col_str


def test_multi_index_track_mode_with_score_user():
    """Test multi_index format track mode with user instances and include_score."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    # User instance
    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(
        labels,
        format="multi_index",
        instance_id="track",
        include_video=False,
        include_score=True,
    )

    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    # Check that score columns exist (as None for user instances)
    col_str = str(tuple_cols)
    assert "score" in col_str


def test_multi_index_track_mode_partial_frames():
    """Test multi_index track mode where track appears in some frames only."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("a1")
    track2 = Track("a2")

    # Frame 0: track1 only
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1])

    # Frame 1: track2 only
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst2])

    labels = Labels([lf0, lf1])
    df = to_dataframe(
        labels, format="multi_index", instance_id="track", include_video=False
    )

    # Both tracks should be in columns, with NaN for missing frames
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    col_str = str(tuple_cols)
    assert "a1" in col_str
    assert "a2" in col_str


def test_from_instances_with_video_object_no_lookup():
    """Test instances decoder with Video object column."""
    skeleton = Skeleton(["a"])
    video = Video(filename="test.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "video": [video],
            "a.x": [1.0],
            "a.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="instances", skeleton=skeleton)

    assert len(labels.videos) == 1
    assert labels.videos[0] is video


def test_from_frames_video_idx_with_provided_video():
    """Test frames decoder with video_idx column and provided video."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="my_video.mp4")

    df = pd.DataFrame(
        {
            "frame_idx": [0],
            "video_idx": [0],
            "inst0.pt.x": [1.0],
            "inst0.pt.y": [2.0],
        }
    )

    labels = from_dataframe(df, format="frames", video=video, skeleton=skeleton)

    assert labels.videos[0].filename == "my_video.mp4"


def test_frames_format_track_mode_untracked_ignore():
    """Test frames format with track mode and untracked=ignore."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("tracked")

    # One tracked, one untracked
    tracked_inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )
    untracked_inst = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        score=0.8,
    )

    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[tracked_inst, untracked_inst]
    )
    labels = Labels([lf])

    # With ignore, untracked should be skipped
    df = to_dataframe(
        labels, format="multi_index", instance_id="track", untracked="ignore"
    )

    # Should only have tracked columns
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    col_str = str(tuple_cols)
    assert "tracked" in col_str


# =============================================================================
# Iterator Tests
# =============================================================================


def test_to_dataframe_iter_points_basic(slp_typical):
    """Test basic iteration with points format."""
    labels = load_slp(slp_typical)

    chunks = list(to_dataframe_iter(labels, format="points", chunk_size=5))

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, pd.DataFrame)
        assert len(chunk) <= 5

    # Verify concatenation equals full dataframe
    df_iter = pd.concat(chunks, ignore_index=True)
    df_full = to_dataframe(labels, format="points")
    assert len(df_iter) == len(df_full)


def test_to_dataframe_iter_all_formats(slp_typical):
    """Test that all formats work with iterator."""
    labels = load_slp(slp_typical)

    for fmt in ["points", "instances", "frames", "multi_index"]:
        chunks = list(to_dataframe_iter(labels, format=fmt, chunk_size=3))
        df_iter = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        df_full = to_dataframe(labels, format=fmt)
        assert len(df_iter) == len(df_full), f"Format {fmt} row count mismatch"


def test_to_dataframe_iter_no_chunk_size(slp_typical):
    """Test that chunk_size=None yields entire DataFrame."""
    labels = load_slp(slp_typical)

    chunks = list(to_dataframe_iter(labels, format="points", chunk_size=None))

    assert len(chunks) == 1
    df_full = to_dataframe(labels, format="points")
    assert len(chunks[0]) == len(df_full)


def test_to_dataframe_iter_chunk_larger_than_data(slp_typical):
    """Test when chunk_size is larger than total rows."""
    labels = load_slp(slp_typical)

    df_full = to_dataframe(labels, format="points")
    chunks = list(
        to_dataframe_iter(labels, format="points", chunk_size=len(df_full) * 2)
    )

    assert len(chunks) == 1
    assert len(chunks[0]) == len(df_full)


def test_to_dataframe_iter_chunk_size_one():
    """Test with chunk_size=1 yields individual rows."""
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")
    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    # points format: 1 instance * 2 nodes = 2 rows
    chunks = list(to_dataframe_iter(labels, format="points", chunk_size=1))
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(chunk) == 1


def test_to_dataframe_iter_empty_labels():
    """Test iteration with empty labels."""
    labels = Labels([])

    chunks = list(to_dataframe_iter(labels, format="points", chunk_size=10))

    # Should yield empty DataFrame
    assert len(chunks) == 1
    assert len(chunks[0]) == 0


def test_to_dataframe_iter_instances_format():
    """Test instances format iteration."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")
    instances = [
        PredictedInstance.from_numpy(
            points_data=np.array([[i * 10.0, i * 20.0], [i * 10.0 + 1, i * 20.0 + 1]]),
            skeleton=skeleton,
            score=0.9,
        )
        for i in range(5)
    ]
    lf = LabeledFrame(video=video, frame_idx=0, instances=instances)
    labels = Labels([lf])

    # 5 instances, chunk_size=2 -> 3 chunks
    chunks = list(to_dataframe_iter(labels, format="instances", chunk_size=2))
    assert len(chunks) == 3
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert len(chunks[2]) == 1


def test_to_dataframe_iter_frames_format():
    """Test frames format iteration."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")

    lfs = []
    for i in range(5):
        instance = PredictedInstance.from_numpy(
            points_data=np.array([[i * 10.0, i * 20.0]]),
            skeleton=skeleton,
            score=0.9,
        )
        lfs.append(LabeledFrame(video=video, frame_idx=i, instances=[instance]))
    labels = Labels(lfs)

    # 5 frames, chunk_size=2 -> 3 chunks
    chunks = list(to_dataframe_iter(labels, format="frames", chunk_size=2))
    assert len(chunks) == 3

    # Verify all frame_idx values are present
    df_iter = pd.concat(chunks, ignore_index=True)
    assert set(df_iter["frame_idx"]) == {0, 1, 2, 3, 4}


def test_to_dataframe_iter_polars_backend():
    """Test iterator with polars backend."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["node1"])
    video = Video(filename="test.mp4")
    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    chunks = list(
        to_dataframe_iter(labels, format="points", chunk_size=10, backend="polars")
    )
    assert len(chunks) == 1
    assert isinstance(chunks[0], pl.DataFrame)


def test_to_dataframe_iter_video_filter(slp_typical):
    """Test iterator with video filter."""
    labels = load_slp(slp_typical)

    if len(labels.videos) > 0:
        video = labels.videos[0]
        chunks = list(
            to_dataframe_iter(labels, format="points", chunk_size=5, video=video)
        )

        # Verify all rows are from the specified video
        for chunk in chunks:
            if "video_path" in chunk.columns:
                assert all(chunk["video_path"] == video.filename)


def test_to_dataframe_iter_parameter_passthrough(slp_typical):
    """Test that parameters are correctly passed to converter."""
    labels = load_slp(slp_typical)

    # Test without scores
    chunks = list(
        to_dataframe_iter(labels, format="points", chunk_size=100, include_score=False)
    )
    df = pd.concat(chunks, ignore_index=True)
    assert "score" not in df.columns

    # Test without metadata
    chunks = list(
        to_dataframe_iter(
            labels, format="points", chunk_size=100, include_metadata=False
        )
    )
    df = pd.concat(chunks, ignore_index=True)
    assert "track" not in df.columns


def test_labels_to_dataframe_iter_wrapper(slp_typical):
    """Test Labels.to_dataframe_iter() wrapper method."""
    labels = load_slp(slp_typical)

    # Test that wrapper method works
    chunks = list(labels.to_dataframe_iter(format="points", chunk_size=5))

    assert len(chunks) > 0
    df_iter = pd.concat(chunks, ignore_index=True)
    df_full = labels.to_dataframe(format="points")
    assert len(df_iter) == len(df_full)


def test_to_dataframe_iter_frames_track_mode():
    """Test frames format iteration with track mode."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")
    track = Track("mouse1")

    lfs = []
    for i in range(3):
        instance = PredictedInstance.from_numpy(
            points_data=np.array([[i * 10.0, i * 20.0]]),
            skeleton=skeleton,
            score=0.9,
            track=track,
        )
        lfs.append(LabeledFrame(video=video, frame_idx=i, instances=[instance]))
    labels = Labels(lfs)

    chunks = list(
        to_dataframe_iter(labels, format="frames", chunk_size=2, instance_id="track")
    )
    assert len(chunks) == 2

    # Check track columns exist
    df = pd.concat(chunks, ignore_index=True)
    assert "mouse1.nose.x" in df.columns


def test_to_dataframe_iter_multi_index_format():
    """Test multi_index format iteration."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")

    lfs = []
    for i in range(4):
        instance = PredictedInstance.from_numpy(
            points_data=np.array([[i * 10.0, i * 20.0], [i * 10.0 + 1, i * 20.0 + 1]]),
            skeleton=skeleton,
            score=0.9,
        )
        lfs.append(LabeledFrame(video=video, frame_idx=i, instances=[instance]))
    labels = Labels(lfs)

    # 4 frames, chunk_size=2 -> 2 chunks
    chunks = list(to_dataframe_iter(labels, format="multi_index", chunk_size=2))
    assert len(chunks) == 2

    df_iter = pd.concat(chunks, ignore_index=True)
    df_full = to_dataframe(labels, format="multi_index")
    assert len(df_iter) == len(df_full)


def test_to_dataframe_iter_multi_index_user_instances():
    """Test multi_index format iteration with user instances.

    Note: The iterator always uses flat keys (dot-separated), even for pandas.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Use user instances (not predicted) to test the non-predicted path
    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    chunks = list(
        to_dataframe_iter(
            labels, format="multi_index", chunk_size=1, include_video=False
        )
    )

    assert len(chunks) == 1
    df = chunks[0]
    # Iterator uses flat keys (dot-separated)
    assert "inst0.track_score" in df.columns
    assert "inst0.pt.x" in df.columns
    # User instances have None for track_score and score
    assert df.iloc[0]["inst0.track_score"] is None
    assert df.iloc[0]["inst0.score"] is None


def test_to_dataframe_iter_multi_index_padding():
    """Test multi_index format iteration with NaN padding for missing instances.

    Note: The iterator always uses flat keys (dot-separated), even for pandas.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Frame 0: 2 instances
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: 1 instance only (triggers padding)
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])

    chunks = list(
        to_dataframe_iter(
            labels, format="multi_index", chunk_size=1, include_video=False
        )
    )

    assert len(chunks) == 2

    # Frame 1 (second chunk) should have NaN for inst1
    df1 = chunks[1]
    # Iterator uses flat keys (dot-separated)
    assert "inst1.pt.x" in df1.columns
    assert pd.isna(df1.iloc[0]["inst1.pt.x"])
    assert pd.isna(df1.iloc[0]["inst1.pt.y"])


def test_to_dataframe_iter_frames_padding():
    """Test frames format iteration with NaN padding for missing instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Frame 0: 2 instances
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: 1 instance only (triggers padding)
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])

    # Use iterator to trigger _iter_frames_rows (not _to_frames_df)
    chunks = list(to_dataframe_iter(labels, format="frames", chunk_size=1))

    assert len(chunks) == 2

    # Frame 1 (second chunk) should have NaN for inst1
    df1 = chunks[1]
    assert pd.isna(df1.iloc[0]["inst1.pt.x"])
    assert pd.isna(df1.iloc[0]["inst1.pt.y"])
    assert pd.isna(df1.iloc[0]["inst1.pt.score"])


def test_to_dataframe_iter_frames_track_mode_padding():
    """Test frames format iteration with track mode and NaN padding."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("mouse1")
    track2 = Track("mouse2")

    # Frame 0: 2 instances with tracks
    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

    # Frame 1: Only one track present (mouse1 only, mouse2 missing)
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )
    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])

    # Use iterator with track mode
    chunks = list(
        to_dataframe_iter(labels, format="frames", chunk_size=1, instance_id="track")
    )

    assert len(chunks) == 2

    # Frame 1 (second chunk) should have NaN for mouse2
    df1 = chunks[1]
    assert pd.isna(df1.iloc[0]["mouse2.pt.x"])
    assert pd.isna(df1.iloc[0]["mouse2.pt.y"])
    assert pd.isna(df1.iloc[0]["mouse2.pt.score"])


def test_to_dataframe_iter_video_id_options():
    """Test different video_id options with iterator."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="/path/to/test.mp4")
    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    # Test video_id="index"
    chunks = list(
        to_dataframe_iter(
            labels, format="points", chunk_size=10, video_id="index", include_video=True
        )
    )
    df = chunks[0]
    assert "video_idx" in df.columns
    assert df["video_idx"].iloc[0] == 0

    # Test video_id="name"
    chunks = list(
        to_dataframe_iter(
            labels, format="points", chunk_size=10, video_id="name", include_video=True
        )
    )
    df = chunks[0]
    assert "video_path" in df.columns
    assert df["video_path"].iloc[0] == "test.mp4"

    # Test video_id="object"
    chunks = list(
        to_dataframe_iter(
            labels,
            format="points",
            chunk_size=10,
            video_id="object",
            include_video=True,
        )
    )
    df = chunks[0]
    assert "video" in df.columns
    assert df["video"].iloc[0] is video


def test_to_dataframe_iter_instances_video_id():
    """Test instances format with different video_id options."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="/path/to/test.mp4")
    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    # Test video_id="index" with instances format
    chunks = list(
        to_dataframe_iter(
            labels,
            format="instances",
            chunk_size=10,
            video_id="index",
            include_video=True,
        )
    )
    df = chunks[0]
    assert "video_idx" in df.columns


def test_to_dataframe_iter_frames_video_id():
    """Test frames format with different video_id options."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="/path/to/test.mp4")
    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    # Test video_id="index" with frames format
    chunks = list(
        to_dataframe_iter(
            labels, format="frames", chunk_size=10, video_id="index", include_video=True
        )
    )
    df = chunks[0]
    assert "video_idx" in df.columns

    # Test video_id="object" with frames format
    chunks = list(
        to_dataframe_iter(
            labels,
            format="frames",
            chunk_size=10,
            video_id="object",
            include_video=True,
        )
    )
    df = chunks[0]
    assert "video" in df.columns


def test_to_dataframe_iter_user_instances():
    """Test iterator with user instances (not predicted)."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")

    # Create user instance (not predicted)
    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    # Test points format with user instances
    chunks = list(to_dataframe_iter(labels, format="points", chunk_size=10))
    df = chunks[0]
    assert len(df) == 2
    assert df["track_score"].isna().all()  # User instances have no track_score

    # Test instances format with user instances
    chunks = list(to_dataframe_iter(labels, format="instances", chunk_size=10))
    df = chunks[0]
    assert len(df) == 1
    assert df["score"].isna().all()  # User instances have no score


def test_to_dataframe_iter_frames_user_instances():
    """Test frames format iterator with user instances."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")
    track = Track("track1")

    instance = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf])

    # Test frames format in index mode
    chunks = list(to_dataframe_iter(labels, format="frames", chunk_size=10))
    df = chunks[0]
    assert "inst0.track" in df.columns
    assert df["inst0.track"].iloc[0] == "track1"
    assert df["inst0.track_score"].isna().all()

    # Test frames format in track mode
    chunks = list(
        to_dataframe_iter(labels, format="frames", chunk_size=10, instance_id="track")
    )
    df = chunks[0]
    assert "track1.nose.x" in df.columns


def test_to_dataframe_iter_invalid_format():
    """Test that invalid format raises error."""
    labels = Labels([])

    with pytest.raises(ValueError, match="Invalid format"):
        list(to_dataframe_iter(labels, format="invalid", chunk_size=10))


# ============================================================================
# Native Polars Backend Tests (Phase 3)
# ============================================================================


def test_polars_native_points_format_values():
    """Test that native polars backend produces correct values in points format."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")
    track = Track("mouse1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.5, 20.3], [15.2, 25.1]]),
        skeleton=skeleton,
        point_scores=np.array([0.92, 0.89]),
        track=track,
        score=0.95,
        tracking_score=0.98,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="points", backend="polars")

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2  # Two nodes

    # Check values
    nose_row = df.filter(pl.col("node") == "nose")
    assert nose_row["x"][0] == 10.5
    assert nose_row["y"][0] == 20.3
    assert nose_row["score"][0] == 0.92
    assert nose_row["track"][0] == "mouse1"
    assert nose_row["instance_score"][0] == 0.95
    assert nose_row["track_score"][0] == 0.98


def test_polars_native_instances_format_values():
    """Test that native polars backend produces correct values in instances format."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")
    track = Track("mouse1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.5, 20.3], [15.2, 25.1]]),
        skeleton=skeleton,
        point_scores=np.array([0.92, 0.89]),
        track=track,
        score=0.95,
        tracking_score=0.98,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="instances", backend="polars")

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1  # One instance

    # Check values
    assert df["frame_idx"][0] == 0
    assert df["track"][0] == "mouse1"
    assert df["score"][0] == 0.95
    assert df["track_score"][0] == 0.98
    assert df["nose.x"][0] == 10.5
    assert df["nose.y"][0] == 20.3
    assert df["nose.score"][0] == 0.92
    assert df["tail.x"][0] == 15.2
    assert df["tail.y"][0] == 25.1
    assert df["tail.score"][0] == 0.89


def test_polars_native_frames_format_values():
    """Test that native polars backend produces correct values in frames format."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")
    track1 = Track("mouse1")
    track2 = Track("mouse2")

    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [15.0, 25.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.9, 0.8]),
        track=track1,
        score=0.95,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[50.0, 60.0], [55.0, 65.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.85, 0.82]),
        track=track2,
        score=0.88,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels([lf], tracks=[track1, track2])

    df = to_dataframe(labels, format="frames", backend="polars")

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1  # One frame

    # Check instance 0 values
    assert df["inst0.track"][0] == "mouse1"
    assert df["inst0.score"][0] == 0.95
    assert df["inst0.nose.x"][0] == 10.0
    assert df["inst0.nose.y"][0] == 20.0
    assert df["inst0.tail.x"][0] == 15.0

    # Check instance 1 values
    assert df["inst1.track"][0] == "mouse2"
    assert df["inst1.score"][0] == 0.88
    assert df["inst1.nose.x"][0] == 50.0


def test_polars_native_multi_index_format_values():
    """Test that native polars backend produces correct values in multi_index format."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")
    track = Track("mouse1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.5, 20.3], [15.2, 25.1]]),
        skeleton=skeleton,
        point_scores=np.array([0.92, 0.89]),
        track=track,
        score=0.95,
        tracking_score=0.98,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="multi_index", backend="polars")

    assert isinstance(df, pl.DataFrame)

    # For polars, multi_index uses flat dot-separated column names
    assert "inst0.track" in df.columns
    assert "inst0.nose.x" in df.columns
    assert "inst0.nose.y" in df.columns
    assert "inst0.nose.score" in df.columns

    # Check values
    assert df["inst0.track"][0] == "mouse1"
    assert df["inst0.nose.x"][0] == 10.5
    assert df["inst0.nose.y"][0] == 20.3
    assert df["inst0.nose.score"][0] == 0.92


def test_polars_pandas_equivalence():
    """Test that polars and pandas backends produce equivalent data."""
    pytest.importorskip("polars")

    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.9, 0.8]),
        track=track,
        score=0.95,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    for fmt in ["points", "instances", "frames"]:
        df_pandas = to_dataframe(labels, format=fmt, backend="pandas")
        df_polars = to_dataframe(labels, format=fmt, backend="polars")

        # Convert polars to pandas for comparison
        df_polars_pd = df_polars.to_pandas()

        # Sort columns for consistent comparison
        df_pandas = df_pandas[sorted(df_pandas.columns)]
        df_polars_pd = df_polars_pd[sorted(df_polars_pd.columns)]

        # Check same columns
        assert set(df_pandas.columns) == set(df_polars_pd.columns), f"Format {fmt}"

        # Check same values (with tolerance for floats)
        for col in df_pandas.columns:
            pd_vals = df_pandas[col].values
            pl_vals = df_polars_pd[col].values
            if np.issubdtype(pd_vals.dtype, np.floating):
                np.testing.assert_array_almost_equal(pd_vals, pl_vals)
            else:
                np.testing.assert_array_equal(pd_vals, pl_vals)


def test_polars_native_iter_values():
    """Test that native polars iterator produces correct values."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")

    instances = [
        PredictedInstance.from_numpy(
            points_data=np.array([[i * 10.0, i * 20.0]]),
            skeleton=skeleton,
            point_scores=np.array([0.9]),
            score=0.95,
        )
        for i in range(5)
    ]

    lfs = [
        LabeledFrame(video=video, frame_idx=i, instances=[instances[i]])
        for i in range(5)
    ]
    labels = Labels(lfs)

    # Get chunks
    chunks = list(
        to_dataframe_iter(labels, format="points", chunk_size=2, backend="polars")
    )

    # Should have 3 chunks (2+2+1)
    assert len(chunks) == 3

    # All chunks should be polars DataFrames
    for chunk in chunks:
        assert isinstance(chunk, pl.DataFrame)

    # Concatenate and verify values
    df = pl.concat(chunks)
    assert len(df) == 5

    for i in range(5):
        row = df.filter(pl.col("frame_idx") == i)
        assert row["x"][0] == i * 10.0
        assert row["y"][0] == i * 20.0


def test_polars_empty_labels():
    """Test that native polars handles empty labels correctly."""
    pytest.importorskip("polars")
    import polars as pl

    labels = Labels()

    for fmt in ["points", "instances", "frames", "multi_index"]:
        df = to_dataframe(labels, format=fmt, backend="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()


def test_polars_multi_index_track_mode():
    """Test multi_index format with track mode in polars."""
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")
    track1 = Track("mouse1")
    track2 = Track("mouse2")

    inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[50.0, 60.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.8,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels([lf], tracks=[track1, track2])

    df = to_dataframe(
        labels, format="multi_index", backend="polars", instance_id="track"
    )

    assert isinstance(df, pl.DataFrame)
    # Track mode: column names are track names
    assert "mouse1.nose.x" in df.columns
    assert "mouse2.nose.x" in df.columns
    assert df["mouse1.nose.x"][0] == 10.0
    assert df["mouse2.nose.x"][0] == 50.0


def test_polars_multi_index_user_instances():
    """Test multi_index format with user instances in polars.

    This covers the `elif include_score:` branch in the use_flat_keys=True path.
    """
    pytest.importorskip("polars")
    import polars as pl

    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")

    # Use user instances (not predicted) to test the non-predicted path
    inst = Instance.from_numpy(
        points_data=np.array([[10.0, 20.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="multi_index", backend="polars")

    assert isinstance(df, pl.DataFrame)
    # User instances have None for scores
    assert df["inst0.score"][0] is None
    assert df["inst0.nose.score"][0] is None
    assert df["inst0.nose.x"][0] == 10.0


# =============================================================================
# Additional coverage tests for uncovered branches
# =============================================================================


def test_invalid_video_id_parameter():
    """Test that invalid video_id raises ValueError."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")
    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Must set include_video=True to trigger video_id validation
    with pytest.raises(ValueError, match="Invalid video_id"):
        to_dataframe(
            labels, format="points", video_id="invalid_mode", include_video=True
        )


def test_video_id_name_with_list_filename():
    """Test video_id='name' with list filename (ImageVideo style)."""
    skeleton = Skeleton(["nose"])
    video = Video(filename=["/path/to/img_0001.png", "/path/to/img_0002.png"])
    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Must set include_video=True to include video column
    df = to_dataframe(labels, format="points", video_id="name", include_video=True)

    # Should use basename of first filename
    assert df["video_path"].iloc[0] == "img_0001.png"


def test_frames_format_user_instances_track_mode():
    """Test frames format with user instances in track mode."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("animal1")

    # Create user instance with track
    inst = Instance.from_numpy(
        points_data=np.array([[5.0, 10.0]]),
        skeleton=skeleton,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf], tracks=[track])

    df = to_dataframe(labels, format="frames", instance_id="track", include_video=False)

    # User instances have no scores (should be None)
    assert "animal1.track_score" in df.columns
    assert df["animal1.track_score"].isna().all()
    assert "animal1.score" in df.columns
    assert df["animal1.score"].isna().all()
    # But coordinates should be there
    assert df["animal1.pt.x"].iloc[0] == 5.0
    assert df["animal1.pt.y"].iloc[0] == 10.0


def test_frames_format_user_instances_index_mode():
    """Test frames format with user instances in index mode."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")

    # Create user instance (no track)
    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(labels, format="frames", instance_id="index", include_video=False)

    # User instances have None for track_score and score
    assert "inst0.track_score" in df.columns
    assert df["inst0.track_score"].isna().all()
    assert "inst0.score" in df.columns
    assert df["inst0.score"].isna().all()
    # But coordinates should be present
    assert df["inst0.nose.x"].iloc[0] == 1.0
    assert df["inst0.tail.y"].iloc[0] == 4.0


def test_multi_index_format_user_instances_index_mode():
    """Test multi_index format with user instances in index mode."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Create user instance (no track)
    inst = Instance.from_numpy(
        points_data=np.array([[1.5, 2.5]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    df = to_dataframe(
        labels, format="multi_index", instance_id="index", include_video=False
    )

    # Should have hierarchical columns with inst0
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    col_str = str(tuple_cols)
    assert "inst0" in col_str
    assert "pt" in col_str


def test_frames_format_track_mode_partial_frames():
    """Test frames format track mode where track appears in only some frames."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("a1")
    track2 = Track("a2")

    # Frame 0: only track1 present
    inst1 = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
    )
    lf1 = LabeledFrame(video=video, frame_idx=0, instances=[inst1])

    # Frame 1: both tracks present
    inst2a = Instance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track1,
    )
    inst2b = Instance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        track=track2,
    )
    lf2 = LabeledFrame(video=video, frame_idx=1, instances=[inst2a, inst2b])

    labels = Labels([lf1, lf2], tracks=[track1, track2])

    df = to_dataframe(labels, format="frames", instance_id="track", include_video=False)

    # Frame 0: track2 should have NaN
    assert df.loc[df["frame_idx"] == 0, "a1.pt.x"].iloc[0] == 1.0
    assert np.isnan(df.loc[df["frame_idx"] == 0, "a2.pt.x"].iloc[0])

    # Frame 1: both tracks should have values
    assert df.loc[df["frame_idx"] == 1, "a1.pt.x"].iloc[0] == 3.0
    assert df.loc[df["frame_idx"] == 1, "a2.pt.x"].iloc[0] == 5.0


def test_multi_index_format_index_mode_with_padding():
    """Test multi_index format index mode with padding for varying instance counts."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Frame 0: 1 instance
    inst1 = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )
    lf1 = LabeledFrame(video=video, frame_idx=0, instances=[inst1])

    # Frame 1: 2 instances
    inst2 = Instance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
    )
    inst3 = Instance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
    )
    lf2 = LabeledFrame(video=video, frame_idx=1, instances=[inst2, inst3])

    labels = Labels([lf1, lf2])

    df = to_dataframe(
        labels, format="multi_index", instance_id="index", include_video=False
    )

    # Should have columns for inst0 and inst1 (padded)
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    col_str = str(tuple_cols)
    assert "inst0" in col_str
    assert "inst1" in col_str


def test_to_dataframe_iter_invalid_format_string():
    """Test to_dataframe_iter with invalid format string."""
    labels = Labels()

    with pytest.raises(ValueError, match="Invalid format"):
        list(to_dataframe_iter(labels, format="not_a_format"))


def test_from_dataframe_frames_format_user_instances():
    """Test from_dataframe for frames format with user instances (not predicted)."""
    skeleton = Skeleton(["nose", "tail"])
    video = Video(filename="test.mp4")

    # Create a frames format DataFrame manually without scores
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1],
            "inst0.track": [None, None],
            "inst0.track_score": [None, None],
            "inst0.score": [None, None],
            "inst0.nose.x": [1.0, 3.0],
            "inst0.nose.y": [2.0, 4.0],
            "inst0.tail.x": [5.0, 7.0],
            "inst0.tail.y": [6.0, 8.0],
        }
    )

    labels = from_dataframe(df, format="frames", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 2
    assert labels.labeled_frames[0].frame_idx == 0
    # Since no score columns, instances should be user instances
    assert len(labels.labeled_frames[0].instances) == 1


def test_from_dataframe_multi_index_format_user_instances():
    """Test from_dataframe for multi_index format with user instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Create a multi_index DataFrame
    df = pd.DataFrame(
        {
            ("inst0", "track", ""): [None],
            ("inst0", "track_score", ""): [None],
            ("inst0", "score", ""): [None],
            ("inst0", "pt", "x"): [1.0],
            ("inst0", "pt", "y"): [2.0],
        }
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index = [0]
    df.index.name = "frame_idx"

    labels = from_dataframe(df, format="multi_index", video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert labels.labeled_frames[0].frame_idx == 0


def test_video_id_object_in_dataframe():
    """Test video_id='object' mode in to_dataframe."""
    skeleton = Skeleton(["nose"])
    video = Video(filename="test.mp4")
    inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    # Must set include_video=True to include video column
    df = to_dataframe(labels, format="points", video_id="object", include_video=True)

    # video_id="object" produces "video" column with filename as string
    assert "video" in df.columns


def test_multi_index_track_mode_track_not_in_frame():
    """Test multi_index track mode where a track is missing in some frames."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Only track1 in this frame
    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf], tracks=[track1, track2])

    df = to_dataframe(
        labels, format="multi_index", instance_id="track", include_video=False
    )

    # t2 should have NaN values
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    # Find t2 columns
    t2_cols = [c for c in tuple_cols if c[0] == "t2"]
    assert len(t2_cols) > 0  # Should have t2 columns
    # Values should be NaN for t2
    for col in t2_cols:
        if col[1] == "pt":  # x or y coordinate
            assert pd.isna(df[col].iloc[0])


def test_frames_format_mixed_user_and_predicted():
    """Test frames format with mix of user and predicted instances."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("animal")

    # One user instance and one predicted instance
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
    )
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 4.0]]),
        skeleton=skeleton,
        track=track,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
    labels = Labels([lf], tracks=[track])

    # Include both user and predicted
    df = to_dataframe(
        labels,
        format="frames",
        instance_id="index",
        include_video=False,
        include_user_instances=True,
        include_predicted_instances=True,
    )

    assert "inst0.pt.x" in df.columns
    assert "inst1.pt.x" in df.columns


def test_to_dataframe_iter_empty_labels_single_chunk():
    """Test to_dataframe_iter with empty labels yields single empty DataFrame."""
    labels = Labels()

    chunks = list(to_dataframe_iter(labels, format="points"))

    assert len(chunks) == 1
    assert chunks[0].empty


def test_to_dataframe_iter_frames_polars_flatten():
    """Test to_dataframe_iter with frames format and polars backend."""
    pytest.importorskip("polars")

    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    chunks = list(
        to_dataframe_iter(
            labels,
            format="frames",
            backend="polars",
            chunk_size=1,
            include_video=False,
        )
    )

    assert len(chunks) == 1
    import polars as pl

    assert isinstance(chunks[0], pl.DataFrame)


def test_to_dataframe_iter_multi_index_polars_flatten():
    """Test to_dataframe_iter with multi_index format and polars backend."""
    pytest.importorskip("polars")

    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    inst = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels([lf])

    chunks = list(
        to_dataframe_iter(
            labels,
            format="multi_index",
            backend="polars",
            chunk_size=1,
            include_video=False,
        )
    )

    assert len(chunks) == 1
    import polars as pl

    assert isinstance(chunks[0], pl.DataFrame)
    # Should have flattened column names
    assert any("." in str(col) for col in chunks[0].columns)


# =============================================================================
# Lazy Fast Path Tests
# =============================================================================


def test_to_dataframe_lazy_points_matches_eager(slp_typical):
    """Test that lazy fast path produces same output as eager for points format."""
    # Load eager
    labels_eager = load_slp(slp_typical, lazy=False)
    df_eager = to_dataframe(labels_eager, format="points")

    # Load lazy
    labels_lazy = load_slp(slp_typical, lazy=True)
    assert labels_lazy.is_lazy
    df_lazy = to_dataframe(labels_lazy, format="points")

    # Compare shapes
    assert df_eager.shape == df_lazy.shape

    # Sort both for comparison (order may differ)
    df_eager_sorted = df_eager.sort_values(["frame_idx", "node"]).reset_index(drop=True)
    df_lazy_sorted = df_lazy.sort_values(["frame_idx", "node"]).reset_index(drop=True)

    # Compare values
    for col in df_eager_sorted.columns:
        eager_vals = df_eager_sorted[col]
        lazy_vals = df_lazy_sorted[col]
        if eager_vals.dtype == "float64" or lazy_vals.dtype == "float64":
            # Numeric comparison with tolerance
            np.testing.assert_allclose(
                eager_vals.fillna(-999).values,
                lazy_vals.fillna(-999).values,
                rtol=1e-5,
                err_msg=f"Column {col} mismatch",
            )
        else:
            pd.testing.assert_series_equal(
                eager_vals, lazy_vals, check_names=False, obj=f"Column {col}"
            )


def test_to_dataframe_lazy_instances_matches_eager(slp_typical):
    """Test that lazy fast path produces same output as eager for instances format."""
    # Load eager
    labels_eager = load_slp(slp_typical, lazy=False)
    df_eager = to_dataframe(labels_eager, format="instances")

    # Load lazy
    labels_lazy = load_slp(slp_typical, lazy=True)
    assert labels_lazy.is_lazy
    df_lazy = to_dataframe(labels_lazy, format="instances")

    # Compare shapes
    assert df_eager.shape == df_lazy.shape


def test_to_dataframe_lazy_with_video_filter(slp_typical):
    """Test lazy fast path with video filtering."""
    # Load lazy
    labels_lazy = load_slp(slp_typical, lazy=True)
    labels_eager = load_slp(slp_typical, lazy=False)

    # Filter by video
    df_lazy = to_dataframe(labels_lazy, format="points", video=0)
    df_eager = to_dataframe(labels_eager, format="points", video=0)

    assert df_eager.shape == df_lazy.shape
