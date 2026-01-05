"""Tests for numpy codec."""

import numpy as np
import pytest

from sleap_io import (
    LabeledFrame,
    Labels,
    PredictedInstance,
    Skeleton,
    Track,
    Video,
    load_slp,
)
from sleap_io.codecs.numpy import from_numpy, to_numpy


def test_to_numpy_basic(slp_typical):
    """Test basic conversion to numpy array."""
    labels = load_slp(slp_typical)

    arr = to_numpy(labels)

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 4
    assert arr.shape[-1] == 2  # x, y coordinates
    assert arr.dtype == np.float32


def test_to_numpy_with_confidence(slp_typical):
    """Test conversion with confidence scores."""
    labels = load_slp(slp_typical)

    arr = to_numpy(labels, return_confidence=True)

    assert arr.shape[-1] == 3  # x, y, score
    assert arr.dtype == np.float32


def test_to_numpy_matches_labels_method(slp_typical):
    """Test that to_numpy matches Labels.numpy() output."""
    labels = load_slp(slp_typical)

    arr1 = to_numpy(labels)
    arr2 = labels.numpy()

    np.testing.assert_array_equal(arr1, arr2)


def test_to_numpy_with_confidence_matches_labels_method(slp_typical):
    """Test that to_numpy with confidence matches Labels.numpy()."""
    labels = load_slp(slp_typical)

    arr1 = to_numpy(labels, return_confidence=True)
    arr2 = labels.numpy(return_confidence=True)

    np.testing.assert_array_equal(arr1, arr2)


def test_from_numpy_basic():
    """Test basic conversion from numpy array."""
    # Create simple array
    arr = np.zeros((2, 1, 2, 2), dtype=np.float32)
    arr[0, 0] = [[10.0, 20.0], [30.0, 40.0]]
    arr[1, 0] = [[15.0, 25.0], [35.0, 45.0]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["node1", "node2"])

    labels = from_numpy(arr, video=video, skeleton=skeleton)

    assert len(labels.labeled_frames) == 2
    assert len(labels.videos) == 1
    assert len(labels.skeletons) == 1
    assert len(labels.tracks) == 1

    # Check first frame
    lf0 = labels.labeled_frames[0]
    assert lf0.frame_idx == 0
    assert len(lf0.instances) == 1
    inst0 = lf0.instances[0]
    assert isinstance(inst0, PredictedInstance)
    np.testing.assert_array_almost_equal(
        inst0.numpy()[:, :2], [[10.0, 20.0], [30.0, 40.0]]
    )


def test_from_numpy_with_confidence():
    """Test conversion from numpy array with confidence scores."""
    arr = np.zeros((1, 1, 2, 3), dtype=np.float32)
    arr[0, 0] = [[10.0, 20.0, 0.95], [30.0, 40.0, 0.98]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["node1", "node2"])

    labels = from_numpy(arr, video=video, skeleton=skeleton, return_confidence=True)

    assert len(labels.labeled_frames) == 1
    inst = labels.labeled_frames[0].instances[0]
    assert isinstance(inst, PredictedInstance)

    # Check scores (use approximate comparison for floating point)
    assert inst.points[0]["score"] == pytest.approx(0.95, abs=1e-6)
    assert inst.points[1]["score"] == pytest.approx(0.98, abs=1e-6)


def test_from_numpy_matches_labels_method():
    """Test that from_numpy matches Labels.from_numpy()."""
    arr = np.random.rand(5, 2, 3, 2).astype(np.float32)

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b", "c"])

    labels1 = from_numpy(arr, video=video, skeleton=skeleton)
    labels2 = Labels.from_numpy(arr, videos=[video], skeletons=skeleton)

    # Compare array outputs
    arr1 = to_numpy(labels1)
    arr2 = to_numpy(labels2)

    np.testing.assert_array_equal(arr1, arr2)


def test_from_numpy_with_track_names():
    """Test creating tracks with custom names."""
    arr = np.zeros((1, 2, 2, 2), dtype=np.float32)
    arr[0, 0] = [[1.0, 2.0], [3.0, 4.0]]
    arr[0, 1] = [[5.0, 6.0], [7.0, 8.0]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["node1", "node2"])
    track_names = ["mouse1", "mouse2"]

    labels = from_numpy(arr, video=video, skeleton=skeleton, track_names=track_names)

    assert len(labels.tracks) == 2
    assert labels.tracks[0].name == "mouse1"
    assert labels.tracks[1].name == "mouse2"


def test_from_numpy_with_first_frame():
    """Test starting from non-zero frame index."""
    arr = np.zeros((2, 1, 2, 2), dtype=np.float32)
    arr[0, 0] = [[1.0, 2.0], [3.0, 4.0]]
    arr[1, 0] = [[5.0, 6.0], [7.0, 8.0]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["node1", "node2"])

    labels = from_numpy(arr, video=video, skeleton=skeleton, first_frame=10)

    assert labels.labeled_frames[0].frame_idx == 10
    assert labels.labeled_frames[1].frame_idx == 11


def test_from_numpy_with_nan():
    """Test handling of NaN values."""
    arr = np.full((2, 2, 2, 2), np.nan, dtype=np.float32)
    # Only set some values
    arr[0, 0, 0] = [1.0, 2.0]
    arr[1, 1, 1] = [3.0, 4.0]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["node1", "node2"])

    labels = from_numpy(arr, video=video, skeleton=skeleton)

    # Should have 2 frames with data
    assert len(labels.labeled_frames) == 2


def test_from_numpy_validation():
    """Test input validation."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(["node1"])

    # Wrong number of dimensions
    with pytest.raises(ValueError, match="4 dimensions"):
        from_numpy(np.zeros((2, 2, 2)), video=video, skeleton=skeleton)

    # No video
    with pytest.raises(ValueError, match="video"):
        from_numpy(np.zeros((1, 1, 1, 2)), skeleton=skeleton)

    # No skeleton
    with pytest.raises(ValueError, match="skeleton"):
        from_numpy(np.zeros((1, 1, 1, 2)), video=video)

    # Both video and videos
    with pytest.raises(ValueError, match="both"):
        from_numpy(
            np.zeros((1, 1, 1, 2)), video=video, videos=[video], skeleton=skeleton
        )


def test_roundtrip_numpy():
    """Test round-trip conversion through numpy."""
    # Create labels
    skeleton = Skeleton(["a", "b", "c"])
    video = Video(filename="test.mp4")
    track = Track("track1")

    instance = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        skeleton=skeleton,
        track=track,
        score=0.95,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels1 = Labels([lf])

    # Convert to numpy and back
    arr = to_numpy(labels1, return_confidence=True)
    labels2 = from_numpy(
        arr, video=video, skeleton=skeleton, tracks=[track], return_confidence=True
    )

    # Convert both to numpy for comparison
    arr1 = to_numpy(labels1, return_confidence=True)
    arr2 = to_numpy(labels2, return_confidence=True)

    np.testing.assert_array_almost_equal(arr1, arr2)


def test_to_numpy_user_only(slp_typical):
    """Test getting only user instances."""
    labels = load_slp(slp_typical)

    arr = to_numpy(labels, user_instances=True, predicted_instances=False)

    # Should only include user instances
    assert isinstance(arr, np.ndarray)


def test_to_numpy_predicted_only(slp_typical):
    """Test getting only predicted instances."""
    labels = load_slp(slp_typical)

    arr = to_numpy(labels, user_instances=False, predicted_instances=True)

    # Should only include predicted instances
    assert isinstance(arr, np.ndarray)


def test_to_numpy_video_selection(slp_typical):
    """Test selecting specific video."""
    labels = load_slp(slp_typical)

    if len(labels.videos) > 0:
        # Select by index
        arr1 = to_numpy(labels, video=0)
        assert isinstance(arr1, np.ndarray)

        # Select by Video object
        arr2 = to_numpy(labels, video=labels.videos[0])
        np.testing.assert_array_equal(arr1, arr2)
