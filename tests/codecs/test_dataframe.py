"""Tests for dataframe codec."""

import numpy as np
import pandas as pd
import pytest

from sleap_io import Instance, Labels, LabeledFrame, PredictedInstance, Skeleton, Track, Video
from sleap_io.codecs.dataframe import DataFrameFormat, to_dataframe


def test_dataframe_format_enum():
    """Test DataFrameFormat enum values."""
    assert DataFrameFormat.POINTS == "points"
    assert DataFrameFormat.INSTANCES == "instances"
    assert DataFrameFormat.FRAMES == "frames"
    assert DataFrameFormat.MULTI_INDEX == "multi_index"


def test_to_dataframe_points_format(slp_typical):
    """Test conversion to points format."""
    labels = slp_typical

    df = to_dataframe(labels, format="points")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Check required columns
    assert "frame_idx" in df.columns
    assert "node_name" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns

    # Check metadata columns (default include_metadata=True)
    assert "video_path" in df.columns
    assert "skeleton_name" in df.columns
    assert "track_name" in df.columns

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
    assert "skeleton_name" not in df.columns
    assert "track_name" not in df.columns

    # Should still have core columns
    assert "frame_idx" in df.columns
    assert "node_name" in df.columns
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

    # Check base columns
    assert "frame_idx" in df.columns
    assert "track_name" in df.columns

    # Check node-specific columns
    assert "nose_x" in df.columns
    assert "nose_y" in df.columns
    assert "tail_x" in df.columns
    assert "tail_y" in df.columns

    # Check values
    row = df.iloc[0]
    assert row["frame_idx"] == 0
    assert row["track_name"] == "track1"
    assert row["nose_x"] == 10.0
    assert row["nose_y"] == 20.0
    assert row["tail_x"] == 30.0
    assert row["tail_y"] == 40.0


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

    assert "node1_score" in df.columns
    assert df.iloc[0]["node1_score"] == 0.95


def test_to_dataframe_frames_format():
    """Test conversion to frames format."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")
    track1 = Track("track1")
    track2 = Track("track2")

    # Frame 0
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

    # Frame 1
    inst3 = PredictedInstance.from_numpy(
        points_data=np.array([[2.0, 3.0], [4.0, 5.0]]),
        skeleton=skeleton,
        track=track1,
        score=0.85,
    )

    lf1 = LabeledFrame(video=video, frame_idx=1, instances=[inst3])

    labels = Labels([lf0, lf1])

    df = to_dataframe(labels, format="frames")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 2 instances in frame 0, 1 in frame 1

    # Check columns
    assert "frame_idx" in df.columns
    assert "track_idx" in df.columns
    assert "track_name" in df.columns
    assert "a_x" in df.columns
    assert "a_y" in df.columns
    assert "b_x" in df.columns
    assert "b_y" in df.columns

    # Check ordering (should be sorted by frame_idx, track_idx)
    assert df["frame_idx"].tolist() == [0, 0, 1]
    assert df["track_name"].tolist() == ["track1", "track2", "track1"]


def test_to_dataframe_multi_index_format(slp_typical):
    """Test conversion to multi-index format."""
    labels = slp_typical

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
    assert df_user.iloc[0]["instance_type"] == "user"

    # Only predicted instances
    df_pred = to_dataframe(
        labels,
        format="points",
        include_user_instances=False,
        include_predicted_instances=True,
    )
    assert len(df_pred) == 1
    assert df_pred.iloc[0]["instance_type"] == "predicted"

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
    assert set(df["node_name"]) == {"a", "b", "c"}
    assert df["track_name"].unique()[0] == "mouse1"

    # Check scores
    assert df[df["node_name"] == "a"]["score"].iloc[0] == 0.9
    assert df[df["node_name"] == "b"]["score"].iloc[0] == 0.8
    assert df[df["node_name"] == "c"]["score"].iloc[0] == 0.95


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
