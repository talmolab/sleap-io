"""Tests for numpy codec."""

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
from sleap_io.codecs.numpy import from_numpy, to_analysis_arrays, to_numpy


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


def test_to_numpy_mixed_user_and_predicted():
    """Test to_numpy with both user and predicted instances in same frame."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="test.mp4")
    track1 = Track("track1")
    track2 = Track("track2")

    # Create user instance
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
        track=track1,
    )

    # Create predicted instance on different track
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0], [7.0, 8.0]]),
        skeleton=skeleton,
        track=track2,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
    labels = Labels([lf])

    # Get array - user instance takes precedence for its track
    arr = to_numpy(labels, user_instances=True)
    # Should have data for both tracks
    assert arr.shape[0] == 1  # 1 frame
    assert arr.shape[2] == 2  # 2 nodes


def test_to_numpy_user_instances_with_confidence():
    """Test to_numpy with user instances and return_confidence=True."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst])
    labels = Labels([lf])

    # User instances should get confidence = 1.0
    arr = to_numpy(labels, return_confidence=True, untracked=True)
    assert arr.shape[-1] == 3
    # Confidence should be 1.0 for user instances
    assert arr[0, 0, 0, 2] == pytest.approx(1.0)


def test_to_numpy_untracked_user_preferred():
    """Test to_numpy prefers user instances when untracked and user_instances=True."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Create both user and predicted instance (single instance case)
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
    )
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
    labels = Labels([lf])

    # With user_instances=True, should prefer user instance
    arr = to_numpy(labels, user_instances=True, untracked=True)
    assert arr[0, 0, 0, 0] == pytest.approx(1.0)  # User instance x coord


def test_to_numpy_linked_from_predicted():
    """Test to_numpy handles from_predicted links correctly."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    # Create predicted instance first
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

    # Should use user instance and not duplicate
    arr = to_numpy(labels, user_instances=True)
    assert arr.shape[1] == 1  # Only one track


def test_from_numpy_extend_track_names():
    """Test from_numpy extends track names if not enough provided."""
    arr = np.zeros((2, 3, 2, 2))  # 3 tracks
    arr[0, 0] = [[1, 2], [3, 4]]
    arr[0, 1] = [[5, 6], [7, 8]]
    arr[0, 2] = [[9, 10], [11, 12]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b"])

    # Only provide 1 track name for 3 tracks
    labels = from_numpy(arr, video=video, skeleton=skeleton, track_names=["mouse1"])

    assert len(labels.tracks) == 3
    assert labels.tracks[0].name == "mouse1"
    assert labels.tracks[1].name == "track_1"
    assert labels.tracks[2].name == "track_2"


def test_from_numpy_with_scores():
    """Test from_numpy properly handles score array."""
    arr = np.zeros((1, 1, 2, 3))  # Include scores
    arr[0, 0] = [[1, 2, 0.95], [3, 4, 0.90]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b"])

    labels = from_numpy(arr, video=video, skeleton=skeleton, return_confidence=True)

    inst = labels.labeled_frames[0][0]
    assert isinstance(inst, PredictedInstance)


def test_to_numpy_user_only_no_predicted():
    """Test to_numpy with only user instances (predicted_instances=False).

    Covers lines 181-182: user_instances=True, predicted_instances=False path.
    """
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    # Create only user instances
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        skeleton=skeleton,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst])
    labels = Labels([lf])

    arr = to_numpy(
        labels, user_instances=True, predicted_instances=False, untracked=True
    )

    assert arr.shape[0] == 1  # 1 frame
    assert arr[0, 0, 0, 0] == pytest.approx(1.0)


def test_to_numpy_deduplication_by_track():
    """Test that user instance is preferred when same track has both.

    Covers lines 168-174: Skip predicted if user and predicted share same track.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("shared_track")

    # Create user instance on track
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track,
    )

    # Create predicted instance on same track
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
    labels = Labels([lf])

    # With both user and predicted enabled, user should take precedence
    arr = to_numpy(labels, user_instances=True, predicted_instances=True)

    assert arr.shape[0] == 1  # 1 frame
    assert arr.shape[1] == 1  # 1 track (not duplicated)
    # Should have user instance coords
    assert arr[0, 0, 0, 0] == pytest.approx(1.0)


def test_to_numpy_deduplication_by_from_predicted():
    """Test deduplication when user instance is linked to predicted via from_predicted.

    Covers lines 161-166: Skip predicted instance if linked via from_predicted.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track = Track("t1")

    # Create predicted instance
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track,
    )

    # Create user instance linked to predicted (different track to test from_predicted)
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        from_predicted=pred_inst,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
    labels = Labels([lf])

    # With both enabled, should deduplicate via from_predicted
    arr = to_numpy(
        labels, user_instances=True, predicted_instances=True, untracked=True
    )

    # Should only have the user instance (predicted is deduplicated)
    assert arr.shape[1] == 1  # 1 instance slot
    assert arr[0, 0, 0, 0] == pytest.approx(1.0)  # User instance coords


def test_from_numpy_empty_videos_list():
    """Test from_numpy raises error with empty videos list.

    Covers line 321: Empty videos list check.
    """
    arr = np.zeros((1, 1, 2, 2))
    skeleton = Skeleton(["a", "b"])

    with pytest.raises(ValueError, match="video"):
        from_numpy(arr, videos=[], skeleton=skeleton)


def test_from_numpy_both_skeleton_and_skeletons():
    """Test from_numpy raises error when both skeleton and skeletons provided.

    Covers line 327: Cannot specify both 'skeleton' and 'skeletons'.
    """
    arr = np.zeros((1, 1, 2, 2))
    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b"])

    with pytest.raises(ValueError, match="both"):
        from_numpy(arr, video=video, skeleton=skeleton, skeletons=[skeleton])


def test_from_numpy_skeletons_as_single_skeleton():
    """Test from_numpy accepts single Skeleton for skeletons parameter.

    Covers lines 335-336: skeletons isinstance check.
    """
    arr = np.zeros((1, 1, 2, 2))
    arr[0, 0] = [[1, 2], [3, 4]]
    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b"])

    # Pass single Skeleton to skeletons (not in a list)
    labels = from_numpy(arr, video=video, skeletons=skeleton)

    assert len(labels.skeletons) == 1
    assert len(labels.labeled_frames) == 1


def test_from_numpy_empty_skeletons_list():
    """Test from_numpy raises error with empty skeletons list.

    Covers lines 337-338: Empty skeletons list check.
    """
    arr = np.zeros((1, 1, 2, 2))
    video = Video(filename="test.mp4")

    with pytest.raises(ValueError, match="skeleton"):
        from_numpy(arr, video=video, skeletons=[])


def test_from_numpy_extend_tracks():
    """Test from_numpy extends tracks when fewer provided than needed.

    Covers lines 361-365: Add missing tracks when len(tracks) < n_tracks_arr.
    """
    arr = np.zeros((1, 3, 2, 2))  # 3 tracks in data
    arr[0, 0] = [[1, 2], [3, 4]]
    arr[0, 1] = [[5, 6], [7, 8]]
    arr[0, 2] = [[9, 10], [11, 12]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b"])
    # Only provide 1 track but data has 3
    tracks = [Track("existing")]

    labels = from_numpy(arr, video=video, skeleton=skeleton, tracks=tracks)

    assert len(labels.tracks) == 3
    assert labels.tracks[0].name == "existing"
    assert labels.tracks[1].name == "track_0"
    assert labels.tracks[2].name == "track_1"


def test_from_numpy_skip_empty_frames():
    """Test from_numpy skips frames with all NaN data.

    Covers lines 387-388: Skip creating frame if no valid data.
    """
    arr = np.full((3, 1, 2, 2), np.nan, dtype=np.float32)
    # Only set data in frame 1
    arr[1, 0] = [[1.0, 2.0], [3.0, 4.0]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b"])

    labels = from_numpy(arr, video=video, skeleton=skeleton)

    # Should only have 1 frame (frame 1 with data)
    assert len(labels.labeled_frames) == 1
    assert labels.labeled_frames[0].frame_idx == 1


def test_from_numpy_return_confidence_without_scores_in_array():
    """Test from_numpy with return_confidence=True but 2D coords (no scores in array).

    Covers line 413: Default scores to 1.0 when has_confidence but array is 2D.
    """
    arr = np.zeros((1, 1, 2, 2))  # No score column
    arr[0, 0] = [[1, 2], [3, 4]]

    video = Video(filename="test.mp4")
    skeleton = Skeleton(["a", "b"])

    # Request confidence but array doesn't have it
    labels = from_numpy(arr, video=video, skeleton=skeleton, return_confidence=True)

    inst = labels.labeled_frames[0][0]
    assert isinstance(inst, PredictedInstance)
    # Scores should default to 1.0
    assert inst.points[0]["score"] == pytest.approx(1.0)
    assert inst.points[1]["score"] == pytest.approx(1.0)


def test_to_numpy_multi_track_deduplication_by_from_predicted():
    """Test deduplication in untracked mode when predicted linked via from_predicted.

    Covers lines 155-166: Deduplication when user_instances=True and predicted=True
    in untracked mode with n_instances > 1 (so is_single_instance=False).
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Create predicted instance on track1 (will be deduplicated via from_predicted)
    pred_inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track1,
    )

    # Create another predicted instance on track2 (will remain)
    pred_inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[9.0, 10.0]]),
        skeleton=skeleton,
        score=0.85,
        track=track2,
    )

    # Create user instance linked to pred_inst1 via from_predicted
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
        from_predicted=pred_inst1,
    )

    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_inst1, pred_inst2, user_inst]
    )
    labels = Labels([lf])

    # Use untracked=True to enter the untracked code path with deduplication
    # n_instances = max(1, 2) = 2, so is_single_instance = False
    arr = to_numpy(
        labels, user_instances=True, predicted_instances=True, untracked=True
    )

    # Should have user_inst and pred_inst2 (pred_inst1 deduplicated via from_predicted)
    assert arr.shape[0] == 1  # 1 frame
    assert arr.shape[1] == 2  # 2 instances (user + pred_inst2, pred_inst1 deduplicated)
    # First slot should have user instance coords
    assert arr[0, 0, 0, 0] == pytest.approx(1.0)
    # Second slot should have pred_inst2 coords
    assert arr[0, 1, 0, 0] == pytest.approx(9.0)


def test_to_numpy_multi_track_deduplication_by_track():
    """Test deduplication in untracked mode when user and predicted share same track.

    Covers lines 167-176: Deduplication by shared track in untracked mode
    with n_instances > 1 (so is_single_instance=False).
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Create user instance on track1
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
    )

    # Create predicted instance on same track1 (will be deduplicated by shared track)
    pred_inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track1,
    )

    # Create predicted instance on track2 (will remain)
    pred_inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[9.0, 10.0]]),
        skeleton=skeleton,
        score=0.85,
        track=track2,
    )

    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[user_inst, pred_inst1, pred_inst2]
    )
    labels = Labels([lf])

    # Use untracked=True to enter the untracked code path with deduplication
    # n_instances = max(1, 2) = 2, so is_single_instance = False
    arr = to_numpy(
        labels, user_instances=True, predicted_instances=True, untracked=True
    )

    # Should have user_inst and pred_inst2 (pred_inst1 deduplicated by shared track)
    assert arr.shape[0] == 1  # 1 frame
    assert arr.shape[1] == 2  # 2 instances (user + pred_inst2, pred_inst1 deduplicated)
    # First slot should have user instance coords
    assert arr[0, 0, 0, 0] == pytest.approx(1.0)
    # Second slot should have pred_inst2 coords
    assert arr[0, 1, 0, 0] == pytest.approx(9.0)


def test_to_numpy_multi_track_no_deduplication_needed():
    """Test untracked mode when no deduplication needed (predicted not skipped).

    Covers line 175-176: Predicted instance is added when not skipped.
    The key is to have n_instances > 1 so is_single_instance=False,
    and use untracked=True to enter the deduplication code path.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("t1")
    track2 = Track("t2")

    # Create user instance on track1
    user_inst = Instance.from_numpy(
        points_data=np.array([[1.0, 2.0]]),
        skeleton=skeleton,
        track=track1,
    )

    # Create predicted instances on different tracks (NOT linked and NOT same track)
    # These should all be included since no deduplication applies
    pred_inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
        track=track2,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst1])
    labels = Labels([lf])

    # Use untracked=True to enter the untracked code path
    # n_instances = max(1, 1) = 1... this would make is_single_instance = True
    # So we need more instances. Let's add another predicted.
    pred_inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[7.0, 8.0]]),
        skeleton=skeleton,
        score=0.85,
    )
    lf.instances.append(pred_inst2)
    # Now n_instances = max(1, 2) = 2, so is_single_instance = False

    arr = to_numpy(
        labels, user_instances=True, predicted_instances=True, untracked=True
    )

    # All 3 instances should be included (user, pred_inst1, pred_inst2)
    # because pred instances are not linked via from_predicted and don't share track
    assert arr.shape[1] == 2  # n_tracks = max(1,2) = 2
    assert arr[0, 0, 0, 0] == pytest.approx(1.0)  # user on track1
    assert arr[0, 1, 0, 0] == pytest.approx(
        5.0
    )  # pred_inst1 on track2  # pred on track3  # pred on track2


def test_to_numpy_user_only_frame_without_user_instances():
    """Test to_numpy with user_instances=True on frame without user instances.

    Covers lines 181-182: Enter else branch at line 177 when frame doesn't
    have user instances, then get user_instances (empty) at line 182.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    # Create frame with only predicted instances (no user instances)
    pred_inst1 = PredictedInstance.from_numpy(
        points_data=np.array([[5.0, 6.0]]),
        skeleton=skeleton,
        score=0.9,
    )
    pred_inst2 = PredictedInstance.from_numpy(
        points_data=np.array([[7.0, 8.0]]),
        skeleton=skeleton,
        score=0.85,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[pred_inst1, pred_inst2])
    labels = Labels([lf])

    # Request user instances only but frame only has predicted
    # n_instances = max(0, 2) = 2, so is_single_instance = False
    arr = to_numpy(
        labels, user_instances=True, predicted_instances=False, untracked=True
    )

    # Should have no data since there are no user instances
    # Actually the array will be all NaN since no valid instances to include
    assert arr.shape[0] == 1  # 1 frame
    assert np.all(np.isnan(arr))  # All NaN since no user instances


# =============================================================================
# Lazy Fast Path Tests
# =============================================================================


def test_to_numpy_lazy_matches_eager(slp_typical):
    """Test that lazy fast path produces same output as eager."""
    # Load eager
    labels_eager = load_slp(slp_typical, lazy=False)
    arr_eager = to_numpy(labels_eager)

    # Load lazy
    labels_lazy = load_slp(slp_typical, lazy=True)
    assert labels_lazy.is_lazy
    arr_lazy = to_numpy(labels_lazy)

    # Compare shapes
    assert arr_eager.shape == arr_lazy.shape

    # Compare values (allowing for NaN)
    np.testing.assert_allclose(
        np.nan_to_num(arr_eager, nan=-999),
        np.nan_to_num(arr_lazy, nan=-999),
        rtol=1e-5,
    )


def test_to_numpy_lazy_with_video_filter(slp_typical):
    """Test lazy fast path with video filtering."""
    labels_lazy = load_slp(slp_typical, lazy=True)
    labels_eager = load_slp(slp_typical, lazy=False)

    arr_lazy = to_numpy(labels_lazy, video=0)
    arr_eager = to_numpy(labels_eager, video=0)

    assert arr_eager.shape == arr_lazy.shape
    np.testing.assert_allclose(
        np.nan_to_num(arr_eager, nan=-999),
        np.nan_to_num(arr_lazy, nan=-999),
        rtol=1e-5,
    )


def test_to_numpy_lazy_with_confidence(slp_typical):
    """Test lazy fast path with return_confidence."""
    labels_lazy = load_slp(slp_typical, lazy=True)
    labels_eager = load_slp(slp_typical, lazy=False)

    arr_lazy = to_numpy(labels_lazy, return_confidence=True)
    arr_eager = to_numpy(labels_eager, return_confidence=True)

    # Should have 3 coordinates (x, y, score)
    assert arr_lazy.shape[-1] == 3
    assert arr_eager.shape == arr_lazy.shape


def test_to_numpy_spans_full_video(slp_real_data):
    """Test that eager to_numpy output spans the full video length."""
    labels = load_slp(slp_real_data, lazy=False)
    video = labels.videos[0]
    video_length = len(video)
    last_labeled = max(lf.frame_idx for lf in labels.find(video))
    assert video_length > last_labeled + 1, "Fixture must have a gap"

    arr = to_numpy(labels)
    assert arr.shape[0] == video_length
    # Frames beyond last labeled frame should be NaN
    assert np.all(np.isnan(arr[last_labeled + 1 :]))


def test_to_numpy_lazy_spans_full_video(slp_real_data):
    """Test that lazy to_numpy output spans the full video length."""
    labels = load_slp(slp_real_data, lazy=True)
    video = labels.videos[0]
    video_length = len(video)

    arr = to_numpy(labels)
    assert arr.shape[0] == video_length


# =============================================================================
# to_analysis_arrays
# =============================================================================


def test_to_analysis_arrays_untracked_multi():
    """Untracked multi-animal project keeps every instance (issue #430)."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    lfs = []
    for f in range(3):
        i1 = Instance.from_numpy(
            np.array([[10.0 + f, 20.0 + f], [30.0 + f, 40.0 + f]]), skeleton=skeleton
        )
        i2 = Instance.from_numpy(
            np.array([[100.0 + f, 200.0 + f], [300.0 + f, 400.0 + f]]),
            skeleton=skeleton,
        )
        lfs.append(LabeledFrame(video=video, frame_idx=f, instances=[i1, i2]))
    labels = Labels(labeled_frames=lfs)

    occupancy, locations, point_scores, instance_scores, tracking_scores, names, ff = (
        to_analysis_arrays(labels)
    )

    # Both instances survive in distinct slots (was 1 slot before the fix).
    assert occupancy.shape == (3, 2)
    assert locations.shape == (3, 2, 2, 2)
    assert names == ["track_0", "track_1"]
    assert ff == 0
    assert np.all(occupancy == 1)
    np.testing.assert_array_equal(locations[0, 0], [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(locations[0, 1], [[100.0, 200.0], [300.0, 400.0]])
    # User instances carry no scores.
    assert np.all(np.isnan(point_scores))
    assert np.all(np.isnan(instance_scores))
    assert np.all(np.isnan(tracking_scores))


def test_to_analysis_arrays_untracked_single():
    """Single-instance untracked project still exports one track slot."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")

    lfs = [
        LabeledFrame(
            video=video,
            frame_idx=f,
            instances=[
                Instance.from_numpy(
                    np.array([[float(f), float(f)], [float(f), float(f)]]),
                    skeleton=skeleton,
                )
            ],
        )
        for f in range(3)
    ]
    labels = Labels(labeled_frames=lfs)

    occupancy, locations, _, _, _, names, _ = to_analysis_arrays(labels)

    assert occupancy.shape == (3, 1)
    assert locations.shape == (3, 1, 2, 2)
    assert names == ["track_0"]


def test_to_analysis_arrays_tracked():
    """Tracked project slots instances by track and keeps real names/scores."""
    skeleton = Skeleton(["a", "b"])
    video = Video(filename="test.mp4")
    track1 = Track("animal1")
    track2 = Track("animal2")

    lfs = []
    for f in range(3):
        i1 = PredictedInstance.from_numpy(
            points_data=np.array([[1.0 + f, 1.0 + f], [2.0 + f, 2.0 + f]]),
            skeleton=skeleton,
            point_scores=np.array([0.9, 0.8]),
            score=0.95,
            track=track1,
            tracking_score=0.7,
        )
        i2 = PredictedInstance.from_numpy(
            points_data=np.array([[5.0 + f, 5.0 + f], [6.0 + f, 6.0 + f]]),
            skeleton=skeleton,
            point_scores=np.array([0.6, 0.5]),
            score=0.85,
            track=track2,
            tracking_score=0.4,
        )
        lfs.append(LabeledFrame(video=video, frame_idx=f, instances=[i1, i2]))
    labels = Labels(labeled_frames=lfs, tracks=[track1, track2])

    occupancy, locations, point_scores, instance_scores, tracking_scores, names, _ = (
        to_analysis_arrays(labels)
    )

    assert names == ["animal1", "animal2"]
    assert occupancy.shape == (3, 2)
    np.testing.assert_array_equal(locations[0, 0], [[1.0, 1.0], [2.0, 2.0]])
    np.testing.assert_array_equal(locations[0, 1], [[5.0, 5.0], [6.0, 6.0]])
    np.testing.assert_array_equal(point_scores[0, 0], [0.9, 0.8])
    assert instance_scores[0, 0] == pytest.approx(0.95)
    assert instance_scores[0, 1] == pytest.approx(0.85)
    assert tracking_scores[0, 0] == pytest.approx(0.7)


def test_to_analysis_arrays_tracked_drops_untracked_instance():
    """An untracked instance in a tracked project is dropped, not slot-0'd."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")
    track1 = Track("animal1")
    track2 = Track("animal2")

    tracked1 = PredictedInstance.from_numpy(
        points_data=np.array([[1.0, 1.0]]), skeleton=skeleton, score=0.9, track=track1
    )
    tracked2 = PredictedInstance.from_numpy(
        points_data=np.array([[2.0, 2.0]]), skeleton=skeleton, score=0.9, track=track2
    )
    stray = PredictedInstance.from_numpy(
        points_data=np.array([[9.0, 9.0]]), skeleton=skeleton, score=0.9, track=None
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[tracked1, tracked2, stray])
    labels = Labels(labeled_frames=[lf], tracks=[track1, track2])

    occupancy, locations, _, _, _, names, _ = to_analysis_arrays(labels)

    assert names == ["animal1", "animal2"]
    assert occupancy.shape == (1, 2)
    # Track 0 keeps its own instance; the stray instance is not slotted here.
    np.testing.assert_array_equal(locations[0, 0], [[1.0, 1.0]])
    np.testing.assert_array_equal(locations[0, 1], [[2.0, 2.0]])
    assert not np.any(locations == 9.0)


def test_to_analysis_arrays_min_occupancy_untracked():
    """min_occupancy filtering renumbers synthesized track names without gaps."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    lfs = []
    for f in range(10):
        # Frame 0 has 3 instances; all others have 1 -> slots 1 and 2 are sparse.
        n = 3 if f == 0 else 1
        instances = [
            Instance.from_numpy(np.array([[float(k), float(f)]]), skeleton=skeleton)
            for k in range(n)
        ]
        lfs.append(LabeledFrame(video=video, frame_idx=f, instances=instances))
    labels = Labels(labeled_frames=lfs)

    # Without filtering: 3 slots.
    occ_all, _, _, _, _, names_all, _ = to_analysis_arrays(labels)
    assert occ_all.shape[1] == 3
    assert names_all == ["track_0", "track_1", "track_2"]

    # With filtering: only the dense slot 0 survives, renumbered with no gaps.
    occ, _, _, _, _, names, _ = to_analysis_arrays(labels, min_occupancy=0.5)
    assert occ.shape[1] == 1
    assert names == ["track_0"]


def test_to_analysis_arrays_variable_instance_count():
    """n_tracks is sized to the largest per-frame instance count."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    lf0 = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[1.0, 1.0]]), skeleton=skeleton)],
    )
    lf1 = LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[
            Instance.from_numpy(np.array([[float(k), 2.0]]), skeleton=skeleton)
            for k in range(3)
        ],
    )
    labels = Labels(labeled_frames=[lf0, lf1])

    occupancy, _, _, _, _, names, _ = to_analysis_arrays(labels)

    assert occupancy.shape[1] == 3
    assert names == ["track_0", "track_1", "track_2"]
    # Frame 0 occupies only the first slot.
    np.testing.assert_array_equal(occupancy[0], [1, 0, 0])
    np.testing.assert_array_equal(occupancy[1], [1, 1, 1])


def test_to_analysis_arrays_all_frames_toggle():
    """all_frames controls whether output starts at frame 0 or first label."""
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    lfs = [
        LabeledFrame(
            video=video,
            frame_idx=f,
            instances=[Instance.from_numpy(np.array([[1.0, 1.0]]), skeleton=skeleton)],
        )
        for f in (5, 6, 7)
    ]
    labels = Labels(labeled_frames=lfs)

    occ_all, _, _, _, _, _, ff_all = to_analysis_arrays(labels, all_frames=True)
    assert ff_all == 0
    assert occ_all.shape[0] == 8

    occ_some, _, _, _, _, _, ff_some = to_analysis_arrays(labels, all_frames=False)
    assert ff_some == 5
    assert occ_some.shape[0] == 3


def test_to_analysis_arrays_no_labeled_frames():
    """A video with no labeled frames raises ValueError."""
    video = Video(filename="test.mp4")
    labels = Labels(videos=[video], skeletons=[Skeleton(["pt"])])

    with pytest.raises(ValueError, match="No labeled frames"):
        to_analysis_arrays(labels)


def test_to_analysis_arrays_video_index():
    """A video may be selected by integer index."""
    skeleton = Skeleton(["pt"])
    video0 = Video(filename="v0.mp4")
    video1 = Video(filename="v1.mp4")

    lf0 = LabeledFrame(
        video=video0,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[1.0, 1.0]]), skeleton=skeleton)],
    )
    lf1 = LabeledFrame(
        video=video1,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[2.0, 2.0]]), skeleton=skeleton)],
    )
    labels = Labels(labeled_frames=[lf0, lf1])

    _, locations, _, _, _, _, _ = to_analysis_arrays(labels, video=1)
    np.testing.assert_array_equal(locations[0, 0], [[2.0, 2.0]])


def test_to_analysis_arrays_extra_instances_dropped():
    """Instances beyond the global per-frame max are dropped, not crashed on.

    Per-frame dedup can leave more instances than `max(n_user, n_predicted)`
    when user and predicted instances are unrelated, so the track-slot guard
    must drop the overflow instead of writing out of bounds.
    """
    skeleton = Skeleton(["pt"])
    video = Video(filename="test.mp4")

    user1 = Instance.from_numpy(np.array([[1.0, 1.0]]), skeleton=skeleton)
    user2 = Instance.from_numpy(np.array([[2.0, 2.0]]), skeleton=skeleton)
    # Predicted instances unrelated to the user instances (no from_predicted
    # link, no shared track) so dedup keeps them all.
    pred1 = PredictedInstance.from_numpy(
        points_data=np.array([[3.0, 3.0]]), skeleton=skeleton, score=0.9
    )
    pred2 = PredictedInstance.from_numpy(
        points_data=np.array([[4.0, 4.0]]), skeleton=skeleton, score=0.9
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[user1, user2, pred1, pred2])
    labels = Labels(labeled_frames=[lf])

    # n_instances = max(2 user, 2 predicted) = 2, so only 2 slots exist.
    occupancy, locations, _, _, _, names, _ = to_analysis_arrays(labels)

    assert occupancy.shape[1] == 2
    assert names == ["track_0", "track_1"]
    # User instances are slotted first; the extra predicted ones are dropped.
    np.testing.assert_array_equal(locations[0, 0], [[1.0, 1.0]])
    np.testing.assert_array_equal(locations[0, 1], [[2.0, 2.0]])
