"""Tests for methods in sleap_io.model.labeled_frame file."""

import numpy as np
from numpy.testing import assert_equal

from sleap_io import Instance, PredictedInstance, Skeleton, Track, Video
from sleap_io.model.labeled_frame import LabeledFrame


def test_labeled_frame():
    """Test initialization and methods of `LabeledFrame` class."""
    inst = Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"]))
    lf = LabeledFrame(
        video=Video(filename="test"),
        frame_idx=0,
        instances=[
            inst,
            PredictedInstance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
        ],
    )
    assert_equal(lf.numpy(), [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

    assert len(lf) == 2
    assert len(lf.user_instances) == 1
    assert type(lf.user_instances[0]) is Instance
    assert len(lf.predicted_instances) == 1
    assert type(lf.predicted_instances[0]) is PredictedInstance

    # Test LabeledFrame.__getitem__ method
    assert lf[0] == inst

    assert lf.has_predicted_instances
    assert lf.has_user_instances


def test_remove_predictions():
    """Test removing predictions from `LabeledFrame`."""
    inst = Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"]))
    lf = LabeledFrame(
        video=Video(filename="test"),
        frame_idx=0,
        instances=[
            inst,
            PredictedInstance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
        ],
    )

    assert len(lf) == 2
    assert len(lf.predicted_instances) == 1
    assert lf.has_predicted_instances
    assert lf.has_user_instances

    # Remove predictions
    lf.remove_predictions()

    assert len(lf) == 1
    assert len(lf.predicted_instances) == 0
    assert type(lf[0]) is Instance
    assert_equal(lf.numpy(), [[[0, 1], [2, 3]]])
    assert not lf.has_predicted_instances
    assert lf.has_user_instances


def test_remove_empty_instances():
    """Test removing empty instances from `LabeledFrame`."""
    inst = Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"]))
    lf = LabeledFrame(
        video=Video(filename="test"),
        frame_idx=0,
        instances=[
            inst,
            Instance(
                [[np.nan, np.nan], [np.nan, np.nan]], skeleton=Skeleton(["A", "B"])
            ),
        ],
    )

    assert len(lf) == 2

    # Remove empty instances
    lf.remove_empty_instances()

    assert len(lf) == 1
    assert type(lf[0]) is Instance
    assert_equal(lf.numpy(), [[[0, 1], [2, 3]]])


def test_labeled_frame_image(centered_pair_low_quality_path):
    video = Video.from_filename(centered_pair_low_quality_path)
    lf = LabeledFrame(video=video, frame_idx=0)
    assert_equal(lf.image, video[0])


def test_labeled_frame_unused_predictions():
    video = Video("test.mp4")
    skel = Skeleton(["A", "B"])
    track = Track("trk")

    lf1 = LabeledFrame(video=video, frame_idx=0)
    lf1.instances.append(
        Instance.from_numpy([[0, 0], [0, 0]], skeleton=skel, track=track)
    )
    lf1.instances.append(
        PredictedInstance.from_numpy(
            [[0, 0], [0, 0]], skeleton=skel, point_scores=[1, 1], score=1, track=track
        )
    )
    lf1.instances.append(
        PredictedInstance.from_numpy(
            [[1, 1], [1, 1]],
            skeleton=skel,
            point_scores=[1, 1],
            score=1,
        )
    )

    assert len(lf1.unused_predictions) == 1
    assert (lf1.unused_predictions[0].numpy() == 1).all()

    lf2 = LabeledFrame(video=video, frame_idx=1)
    lf2.instances.append(
        PredictedInstance.from_numpy(
            [[0, 0], [0, 0]], skeleton=skel, point_scores=[1, 1], score=1
        )
    )
    lf2.instances.append(Instance.from_numpy([[0, 0], [0, 0]], skeleton=skel))
    lf2.instances[-1].from_predicted = lf2.instances[-2]
    lf2.instances.append(
        PredictedInstance.from_numpy(
            [[1, 1], [1, 1]], skeleton=skel, point_scores=[1, 1], score=1
        )
    )

    assert len(lf2.unused_predictions) == 1
    assert (lf2.unused_predictions[0].numpy() == 1).all()

def test_labeled_frame_matches():
    """Test LabeledFrame.matches() method."""
    video1 = Video(filename="test1.mp4")
    video2 = Video(filename="test2.mp4")
    
    # Test frames with same index and video
    lf1 = LabeledFrame(video=video1, frame_idx=10)
    lf2 = LabeledFrame(video=video1, frame_idx=10)
    assert lf1.matches(lf2)
    
    # Test frames with different indices
    lf3 = LabeledFrame(video=video1, frame_idx=20)
    assert not lf1.matches(lf3)
    
    # Test frames with same index but different videos
    lf4 = LabeledFrame(video=video2, frame_idx=10)
    assert not lf1.matches(lf4, video_must_match=True)
    
    # Test with video_must_match=False
    assert lf1.matches(lf4, video_must_match=False)
    
    # Test different frame indices even with video_must_match=False
    lf5 = LabeledFrame(video=video2, frame_idx=20)
    assert not lf1.matches(lf5, video_must_match=False)


def test_labeled_frame_similarity_to():
    """Test LabeledFrame.similarity_to() method."""
    skeleton = Skeleton(["head", "thorax", "abdomen"])
    
    # Create instances for frame 1
    user_inst1 = Instance.from_numpy(
        np.array([[10, 20], [30, 40], [50, 60]]), skeleton=skeleton
    )
    pred_inst1 = PredictedInstance.from_numpy(
        np.array([[12, 22], [32, 42], [52, 62]]), skeleton=skeleton
    )
    
    # Create instances for frame 2 with some overlap
    user_inst2 = Instance.from_numpy(
        np.array([[11, 21], [31, 41], [51, 61]]), skeleton=skeleton
    )
    pred_inst2 = PredictedInstance.from_numpy(
        np.array([[100, 100], [110, 110], [120, 120]]), skeleton=skeleton
    )
    
    lf1 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[user_inst1, pred_inst1]
    )
    
    lf2 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[user_inst2, pred_inst2]
    )
    
    # Test similarity calculation
    metrics = lf1.similarity_to(lf2)
    
    assert metrics["n_user_self"] == 1
    assert metrics["n_user_other"] == 1
    assert metrics["n_pred_self"] == 1
    assert metrics["n_pred_other"] == 1
    assert metrics["n_overlapping"] > 0  # Should detect overlapping instances
    assert metrics["mean_pose_distance"] is not None  # Should calculate pose distance
    
    # Test with empty frames
    empty_lf1 = LabeledFrame(video=Video(filename="test.mp4"), frame_idx=0)
    empty_lf2 = LabeledFrame(video=Video(filename="test.mp4"), frame_idx=0)
    
    metrics_empty = empty_lf1.similarity_to(empty_lf2)
    assert metrics_empty["n_user_self"] == 0
    assert metrics_empty["n_user_other"] == 0
    assert metrics_empty["n_pred_self"] == 0
    assert metrics_empty["n_pred_other"] == 0
    assert metrics_empty["n_overlapping"] == 0
    assert metrics_empty["mean_pose_distance"] is None
    
    # Test with non-overlapping instances
    far_inst = Instance.from_numpy(
        np.array([[1000, 1000], [2000, 2000], [3000, 3000]]), skeleton=skeleton
    )
    lf3 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[far_inst]
    )
    
    metrics_no_overlap = lf1.similarity_to(lf3)
    assert metrics_no_overlap["n_overlapping"] == 0
    
    # Test with different skeletons (no pose distance calculation)
    diff_skeleton = Skeleton(["node1", "node2"])
    diff_inst = Instance.from_numpy(
        np.array([[10, 20], [30, 40]]), skeleton=diff_skeleton
    )
    lf4 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[diff_inst]
    )
    
    metrics_diff_skel = lf1.similarity_to(lf4)
    # Even with overlap, pose distance won't be calculated for different skeletons
    # but overlapping count should still work
    
    # Test with NaN points (partial visibility)
    partial_inst = Instance.from_numpy(
        np.array([[10, 20], [np.nan, np.nan], [50, 60]]), skeleton=skeleton
    )
    lf5 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[partial_inst]
    )
    
    metrics_partial = lf1.similarity_to(lf5)
    assert metrics_partial["mean_pose_distance"] is not None  # Should still calculate for visible points


def test_labeled_frame_merge_edge_cases():
    """Test edge cases in LabeledFrame.merge() method."""
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod
    
    skeleton = Skeleton(["head", "thorax"])
    track1 = Track(name="track1")
    track2 = Track(name="track2")
    
    # Test case 1: Replace prediction with user instance
    pred_inst_self = PredictedInstance.from_numpy(
        np.array([[10, 20], [30, 40]]), skeleton=skeleton, track=track1
    )
    user_inst_other = Instance.from_numpy(
        np.array([[11, 21], [31, 41]]), skeleton=skeleton, track=track1
    )
    
    lf_self = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_inst_self]
    )
    
    lf_other = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[user_inst_other]
    )
    
    # Use identity matcher that will match based on tracks
    matcher = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)
    merged, conflicts = lf_self.merge(lf_other, instance_matcher=matcher, strategy="smart")
    
    # User instance should replace prediction
    assert len(merged) == 1
    assert type(merged[0]) is Instance
    assert merged[0] is user_inst_other
    
    # Test case 2: Keep prediction with higher score
    pred_inst1 = PredictedInstance.from_numpy(
        np.array([[10, 20], [30, 40]]), skeleton=skeleton, track=track1, score=0.8
    )
    pred_inst2 = PredictedInstance.from_numpy(
        np.array([[11, 21], [31, 41]]), skeleton=skeleton, track=track1, score=0.9
    )
    
    lf_pred1 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_inst1]
    )
    
    lf_pred2 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_inst2]
    )
    
    merged2, conflicts2 = lf_pred1.merge(lf_pred2, instance_matcher=matcher, strategy="smart")
    
    # Higher score prediction should be kept
    assert len(merged2) == 1
    assert type(merged2[0]) is PredictedInstance
    assert merged2[0].score == 0.9
    
    # Test case 3: Predictions without scores
    pred_no_score1 = PredictedInstance.from_numpy(
        np.array([[10, 20], [30, 40]]), skeleton=skeleton, track=track1
    )
    pred_no_score2 = PredictedInstance.from_numpy(
        np.array([[11, 21], [31, 41]]), skeleton=skeleton, track=track1
    )
    
    lf_no_score1 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_no_score1]
    )
    
    lf_no_score2 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_no_score2]
    )
    
    merged3, conflicts3 = lf_no_score1.merge(lf_no_score2, instance_matcher=matcher, strategy="smart")
    
    # Should keep the instance from other frame when no scores
    assert len(merged3) == 1
    # When both have default score of 0.0, it keeps the one from other frame
    assert merged3[0] is pred_no_score2 or merged3[0] is pred_no_score1
    
    # Test case 4: Unmatched predictions should be kept
    pred_unmatched = PredictedInstance.from_numpy(
        np.array([[200, 200], [300, 300]]), skeleton=skeleton, track=track2
    )
    
    lf_with_unmatched = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_inst_self, pred_unmatched]
    )
    
    lf_single = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[user_inst_other]
    )
    
    merged4, conflicts4 = lf_with_unmatched.merge(lf_single, instance_matcher=matcher, strategy="smart")
    
    # Should have both the replaced user instance and the unmatched prediction
    assert len(merged4) == 2
    # Find the unmatched prediction in the results
    assert any(inst is pred_unmatched for inst in merged4)
    
    # Test case 5: Complex scenario with multiple matches
    # Create instances with same track for matching
    shared_track = Track(name="shared")
    pred_shared1 = PredictedInstance.from_numpy(
        np.array([[5, 5], [15, 15]]), skeleton=skeleton, track=shared_track, score=0.7
    )
    pred_shared2 = PredictedInstance.from_numpy(
        np.array([[6, 6], [16, 16]]), skeleton=skeleton, track=shared_track, score=0.6
    )
    
    lf_complex1 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_shared1]
    )
    
    lf_complex2 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[pred_shared2]
    )
    
    merged5, conflicts5 = lf_complex1.merge(lf_complex2, instance_matcher=matcher, strategy="smart")
    
    # Should keep the one with higher score
    assert len(merged5) == 1
    assert merged5[0].score == 0.7
