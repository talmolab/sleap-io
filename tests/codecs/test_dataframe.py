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
from sleap_io.codecs.dataframe import DataFrameFormat, from_dataframe, to_dataframe


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


def test_from_dataframe_unsupported_format():
    """Test that unsupported formats raise NotImplementedError."""
    df = pd.DataFrame({"frame_idx": [0], "node_name": ["a"], "x": [1.0], "y": [2.0]})

    with pytest.raises(NotImplementedError):
        from_dataframe(df, format="instances")


def test_from_dataframe_missing_columns():
    """Test that missing columns raise ValueError."""
    df = pd.DataFrame({"frame_idx": [0], "x": [1.0], "y": [2.0]})  # Missing node_name

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
    # Create minimal DataFrame
    df = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "node_name": ["a", "b"],
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
            "node_name": ["a", "b"],
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
            "node_name": ["a"],
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
    assert len(df) >= 1
    # Frames format should have node columns
    assert any("head" in str(col) for col in df.columns)


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
    assert len(df) >= 1


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
    assert isinstance(df.columns, pd.MultiIndex)


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
    """Test frames format with video_id='object'.

    Covers lines 415-416: video_id == 'object' in frames format.
    """
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
    """Test frames format with video_id='index'.

    Covers lines 413-414: video_id == 'index' in frames format.
    """
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

    Covers lines 493-495: video_id == 'object' in multi_index format.
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

    assert isinstance(df.columns, pd.MultiIndex)


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
    """Test from_dataframe with minimal columns (no instance_type, no track).

    Covers lines 770-773: Fallback when no grouping columns available.
    """
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")

    # Create DataFrame with minimal columns (no instance_type, no track_name)
    data = {
        "frame_idx": [0, 0],
        "node_name": ["node1", "node2"],
        "x": [1.0, 3.0],
        "y": [2.0, 4.0],
    }
    df = pd.DataFrame(data)

    labels = from_dataframe(df, video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1


def test_from_dataframe_unknown_node():
    """Test from_dataframe handles nodes not in skeleton gracefully.

    Covers lines 815-817: Node not in skeleton is skipped.
    """
    skeleton = Skeleton(["node1", "node2"])
    video = Video(filename="test.mp4")

    # Create DataFrame with an extra node not in skeleton
    data = {
        "frame_idx": [0, 0, 0],
        "node_name": ["node1", "node2", "unknown_node"],  # unknown_node not in skeleton
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
    assert "a_x" in df.columns
    assert "b_y" in df.columns


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
    assert "track_name" in df.columns


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
