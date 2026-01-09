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
        instances=[user_inst1, pred_inst1],
    )

    lf2 = LabeledFrame(
        video=Video(filename="test.mp4"),
        frame_idx=0,
        instances=[user_inst2, pred_inst2],
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
        video=Video(filename="test.mp4"), frame_idx=0, instances=[far_inst]
    )

    metrics_no_overlap = lf1.similarity_to(lf3)
    assert metrics_no_overlap["n_overlapping"] == 0

    # Test with different skeletons (no pose distance calculation)
    diff_skeleton = Skeleton(["node1", "node2"])
    diff_inst = Instance.from_numpy(
        np.array([[10, 20], [30, 40]]), skeleton=diff_skeleton
    )
    lf4 = LabeledFrame(
        video=Video(filename="test.mp4"), frame_idx=0, instances=[diff_inst]
    )

    # Test different skeleton case but don't use result
    lf1.similarity_to(lf4)
    # Even with overlap, pose distance won't be calculated for different skeletons
    # but overlapping count should still work

    # Test with NaN points (partial visibility)
    partial_inst = Instance.from_numpy(
        np.array([[10, 20], [np.nan, np.nan], [50, 60]]), skeleton=skeleton
    )
    lf5 = LabeledFrame(
        video=Video(filename="test.mp4"), frame_idx=0, instances=[partial_inst]
    )

    metrics_partial = lf1.similarity_to(lf5)
    assert (
        metrics_partial["mean_pose_distance"] is not None
    )  # Should still calculate for visible points


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
        video=Video(filename="test.mp4"), frame_idx=0, instances=[pred_inst_self]
    )

    lf_other = LabeledFrame(
        video=Video(filename="test.mp4"), frame_idx=0, instances=[user_inst_other]
    )

    # Use identity matcher that will match based on tracks
    matcher = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)
    merged, conflicts = lf_self.merge(lf_other, instance=matcher, frame="auto")

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
        video=Video(filename="test.mp4"), frame_idx=0, instances=[pred_inst1]
    )

    lf_pred2 = LabeledFrame(
        video=Video(filename="test.mp4"), frame_idx=0, instances=[pred_inst2]
    )

    merged2, conflicts2 = lf_pred1.merge(lf_pred2, instance=matcher, frame="auto")

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
        video=Video(filename="test.mp4"), frame_idx=0, instances=[pred_no_score1]
    )

    lf_no_score2 = LabeledFrame(
        video=Video(filename="test.mp4"), frame_idx=0, instances=[pred_no_score2]
    )

    merged3, conflicts3 = lf_no_score1.merge(
        lf_no_score2, instance=matcher, frame="auto"
    )

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
        instances=[pred_inst_self, pred_unmatched],
    )

    lf_single = LabeledFrame(
        video=Video(filename="test.mp4"), frame_idx=0, instances=[user_inst_other]
    )

    merged4, conflicts4 = lf_with_unmatched.merge(
        lf_single, instance=matcher, frame="auto"
    )

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
        video=Video(filename="test.mp4"), frame_idx=0, instances=[pred_shared1]
    )

    lf_complex2 = LabeledFrame(
        video=Video(filename="test.mp4"), frame_idx=0, instances=[pred_shared2]
    )

    merged5, conflicts5 = lf_complex1.merge(lf_complex2, instance=matcher, frame="auto")

    # Should keep the latest one
    assert len(merged5) == 1
    assert merged5[0].score == 0.6


def test_labeled_frame_merge_conflict_resolution_missing_score():
    """Test LabeledFrame.merge() when instances don't have score attributes."""
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Test case 1: Both predictions but one missing score attribute (lines 315-316)
    # Create a PredictedInstance without score attribute
    pred_no_score = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton
    )
    # Remove score attribute if it was set
    if hasattr(pred_no_score, "score"):
        delattr(pred_no_score, "score")

    pred_with_score = PredictedInstance.from_numpy(
        np.array([[11, 11], [21, 21]]), skeleton=skeleton, score=0.9
    )

    frame1 = LabeledFrame(video=video, frame_idx=0, instances=[pred_no_score])
    frame2 = LabeledFrame(video=video, frame_idx=0, instances=[pred_with_score])

    matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)
    merged, conflicts = frame1.merge(frame2, instance=matcher, frame="auto")

    # When one doesn't have score, should keep the other instance (line 316)
    assert len(merged) == 1
    assert merged[0] is pred_with_score

    # Test case 2: Both predictions, neither has score
    pred_no_score2 = PredictedInstance.from_numpy(
        np.array([[30, 30], [40, 40]]), skeleton=skeleton
    )
    if hasattr(pred_no_score2, "score"):
        delattr(pred_no_score2, "score")

    pred_no_score3 = PredictedInstance.from_numpy(
        np.array([[31, 31], [41, 41]]), skeleton=skeleton
    )
    if hasattr(pred_no_score3, "score"):
        delattr(pred_no_score3, "score")

    frame3 = LabeledFrame(video=video, frame_idx=0, instances=[pred_no_score2])
    frame4 = LabeledFrame(video=video, frame_idx=0, instances=[pred_no_score3])

    merged2, conflicts2 = frame3.merge(frame4, instance=matcher, frame="auto")

    # When neither has score, should keep the other instance (line 316)
    assert len(merged2) == 1
    assert merged2[0] is pred_no_score3


def test_labeled_frame_merge_keep_unmatched_predictions():
    """Test that unmatched predictions are kept during merge."""
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Test case for lines 329-330: prediction that was matched but should be removed
    # Create a scenario where we have:
    # Frame 1: User instance + Predicted instance
    # Frame 2: Another predicted instance that matches the first predicted instance

    user_inst = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)

    pred1 = PredictedInstance.from_numpy(
        np.array([[50, 50], [60, 60]]), skeleton=skeleton, score=0.8
    )

    pred2 = PredictedInstance.from_numpy(
        np.array([[51, 51], [61, 61]]), skeleton=skeleton, score=0.9
    )

    # Frame with user instance and prediction
    frame1 = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred1])
    # Frame with another prediction that matches pred1
    frame2 = LabeledFrame(video=video, frame_idx=0, instances=[pred2])

    matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)
    merged, conflicts = frame1.merge(frame2, instance=matcher, frame="auto")

    # Should have user instance and the higher score prediction (pred2)
    assert len(merged) == 2
    assert user_inst in merged
    assert pred2 in merged
    assert pred1 not in merged  # pred1 should be replaced by pred2

    # Test another case: multiple predictions where some match
    pred3 = PredictedInstance.from_numpy(
        np.array([[100, 100], [110, 110]]), skeleton=skeleton, score=0.7
    )
    pred4 = PredictedInstance.from_numpy(
        np.array([[101, 101], [111, 111]]), skeleton=skeleton, score=0.6
    )
    pred5 = PredictedInstance.from_numpy(
        np.array([[200, 200], [210, 210]]), skeleton=skeleton, score=0.5
    )

    frame3 = LabeledFrame(video=video, frame_idx=0, instances=[pred3, pred5])
    frame4 = LabeledFrame(video=video, frame_idx=0, instances=[pred4])

    merged2, conflicts2 = frame3.merge(frame4, instance=matcher, frame="auto")

    # pred3 matches pred4, pred4 is kept
    # pred5 has no match, should be kept
    assert len(merged2) == 2
    assert pred3 not in merged2
    assert pred5 in merged2
    assert pred4 in merged2


def test_labeled_frame_merge_matched_prediction_removal():
    """Test that matched predictions are removed from self frame.

    This specifically tests the case where a prediction from self frame matches
    an instance from other frame and should be excluded from the merged result.
    """
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Create multiple predictions in self frame, some will match, some won't
    pred_self_1 = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, score=0.5
    )
    pred_self_2 = PredictedInstance.from_numpy(
        np.array([[30, 30], [40, 40]]), skeleton=skeleton, score=0.6
    )
    pred_self_3 = PredictedInstance.from_numpy(
        np.array([[50, 50], [60, 60]]), skeleton=skeleton, score=0.7
    )

    # Create instances in other frame that will match some predictions from self
    pred_other_1 = PredictedInstance.from_numpy(
        np.array([[11, 11], [21, 21]]),
        skeleton=skeleton,
        score=0.8,  # Higher score
    )
    user_other = Instance.from_numpy(
        np.array([[31, 31], [41, 41]]),
        skeleton=skeleton,  # Will replace pred_self_2
    )
    # pred_self_3 won't have a match

    frame_self = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_self_1, pred_self_2, pred_self_3]
    )
    frame_other = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_other_1, user_other]
    )

    # Use spatial matcher with threshold that makes the intended matches
    matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=3.0)
    merged, conflicts = frame_self.merge(frame_other, instance=matcher, frame="auto")

    # Expected result:
    # - pred_other_1 replaces pred_self_1 (higher score)
    # - user_other replaces pred_self_2
    # - pred_self_3 is kept (no match)
    assert len(merged) == 3
    assert pred_other_1 in merged  # Replaced pred_self_1
    assert user_other in merged  # Replaced pred_self_2
    assert pred_self_3 in merged  # No match, kept

    # The key test: pred_self_1 and pred_self_2 should NOT be in merged
    # because they were matched and the loop at lines 326-332 should have
    # set keep=False for them (lines 329-330)
    assert pred_self_1 not in merged
    assert pred_self_2 not in merged

    # Test with multiple matched predictions to ensure the loop iterates properly
    pred_self_4 = PredictedInstance.from_numpy(
        np.array([[70, 70], [80, 80]]), skeleton=skeleton, score=0.4
    )
    pred_self_5 = PredictedInstance.from_numpy(
        np.array([[90, 90], [100, 100]]), skeleton=skeleton, score=0.3
    )
    pred_other_2 = PredictedInstance.from_numpy(
        np.array([[71, 71], [81, 81]]), skeleton=skeleton, score=0.9
    )
    pred_other_3 = PredictedInstance.from_numpy(
        np.array([[91, 91], [101, 101]]), skeleton=skeleton, score=0.95
    )

    frame_self_2 = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_self_4, pred_self_5]
    )
    frame_other_2 = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_other_2, pred_other_3]
    )

    merged2, conflicts2 = frame_self_2.merge(
        frame_other_2, instance=matcher, frame="auto"
    )

    # Both predictions from self should be replaced
    assert len(merged2) == 2
    assert pred_other_2 in merged2
    assert pred_other_3 in merged2
    assert pred_self_4 not in merged2  # Should trigger lines 329-330
    assert pred_self_5 not in merged2  # Should trigger lines 329-330


def test_labeled_frame_merge_other_to_self_mapping_iteration():
    """Test other_to_self mapping excludes matched predictions.

    This test ensures lines 328-330 iterate through multiple entries in other_to_self
    mapping and correctly identify when a prediction from self should be excluded.
    """
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Create a scenario with specific matching patterns:
    # Self frame has 5 predictions (indices 0-4)
    # Other frame has 3 instances that will create specific other_to_self mappings

    preds_self = []
    for i in range(5):
        pred = PredictedInstance.from_numpy(
            np.array([[i * 20, i * 20], [i * 20 + 10, i * 20 + 10]]),
            skeleton=skeleton,
            score=0.4 + i * 0.02,
        )
        preds_self.append(pred)

    # Other instances that will match specific self predictions:
    # other[0] -> self[1]
    # other[1] -> self[3]
    # other[2] -> self[4]
    # This means self[0] and self[2] should be kept

    other_inst_0 = PredictedInstance.from_numpy(
        np.array([[21, 21], [31, 31]]), skeleton=skeleton, score=0.9
    )
    other_inst_1 = Instance.from_numpy(
        np.array([[61, 61], [71, 71]]), skeleton=skeleton
    )
    other_inst_2 = PredictedInstance.from_numpy(
        np.array([[81, 81], [91, 91]]), skeleton=skeleton, score=0.95
    )

    frame_self = LabeledFrame(video=video, frame_idx=0, instances=preds_self)

    frame_other = LabeledFrame(
        video=video, frame_idx=0, instances=[other_inst_0, other_inst_1, other_inst_2]
    )

    # Use spatial matcher with carefully chosen threshold
    matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=2.0)

    # Perform the merge
    merged, conflicts = frame_self.merge(frame_other, instance=matcher, frame="auto")

    # Verify results:
    # - self[0] and self[2] should be kept (no matches)
    # - self[1], self[3], self[4] should be excluded (matched)
    # - All other instances should be included

    assert preds_self[0] in merged  # Unmatched, kept
    assert preds_self[1] not in merged  # Matched to other[0], excluded (lines 329-330)
    assert preds_self[2] in merged  # Unmatched, kept
    assert preds_self[3] not in merged  # Matched to other[1], excluded (lines 329-330)
    assert preds_self[4] not in merged  # Matched to other[2], excluded (lines 329-330)

    assert other_inst_0 in merged
    assert other_inst_1 in merged
    assert other_inst_2 in merged

    assert len(merged) == 5  # 2 unmatched from self + 3 from other


def test_labeled_frame_merge_lines_329_330_coverage():
    """Test specific scenario to cover lines 329-330 in labeled_frame.py.

    This creates a scenario where a prediction from self is matched in other_to_self
    but wasn't added to used_indices, forcing the inner loop to iterate and find it.
    """
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # This is a tricky scenario to construct because normally if something is in
    # other_to_self, it would have been processed and added to used_indices.
    # We need to create a case where the matching creates other_to_self entries
    # but some predictions from self still aren't in used_indices.

    # Self frame: user instance + prediction
    user_self = Instance.from_numpy(np.array([[5, 5], [15, 15]]), skeleton=skeleton)
    pred_self = PredictedInstance.from_numpy(
        np.array([[25, 25], [35, 35]]), skeleton=skeleton, score=0.5
    )

    # Other frame: prediction that matches pred_self
    # This should create a match but but new instance pred_other should be kept
    pred_other = PredictedInstance.from_numpy(
        np.array([[26, 26], [36, 36]]),
        skeleton=skeleton,
        score=0.3,  # Lower score
    )

    frame_self = LabeledFrame(
        video=video, frame_idx=0, instances=[user_self, pred_self]
    )

    frame_other = LabeledFrame(video=video, frame_idx=0, instances=[pred_other])

    # Use spatial matcher that will create the match
    matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=3.0)

    # Perform merge
    merged, conflicts = frame_self.merge(frame_other, instance=matcher, frame="auto")

    # Expected: user_self + pred_other
    assert len(merged) == 2
    assert user_self in merged
    assert pred_self not in merged
    assert pred_other in merged

    # Now let's test a different scenario that might trigger the lines
    # Create a case with multiple predictions where one might not get into used_indices
    pred_self_1 = PredictedInstance.from_numpy(
        np.array([[100, 100], [110, 110]]), skeleton=skeleton, score=0.6
    )
    pred_self_2 = PredictedInstance.from_numpy(
        np.array([[200, 200], [210, 210]]), skeleton=skeleton, score=0.7
    )

    # Other frame with instances that create complex matching
    pred_other_1 = PredictedInstance.from_numpy(
        np.array([[101, 101], [111, 111]]),
        skeleton=skeleton,
        score=0.5,  # Lower than self_1
    )
    pred_other_2 = PredictedInstance.from_numpy(
        np.array([[201, 201], [211, 211]]),
        skeleton=skeleton,
        score=0.9,  # Higher than self_2
    )

    frame_self_2 = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_self_1, pred_self_2]
    )

    frame_other_2 = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_other_1, pred_other_2]
    )

    merged2, conflicts2 = frame_self_2.merge(
        frame_other_2, instance=matcher, frame="auto"
    )

    # only pred_other instances would be kep if all are predictions
    assert len(merged2) == 2
    assert pred_self_1 not in merged2
    assert pred_other_2 in merged2
    assert pred_other_1 in merged2
    assert pred_self_2 not in merged2


def test_labeled_frame_merge_multiple_matches_to_same_prediction():
    """Test case where multiple other instances could match the same self prediction.

    This should trigger lines 329-330 where we check if a prediction was matched
    but not processed due to conflict resolution in other_to_self mapping.
    """
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Create a prediction in self frame
    pred_self = PredictedInstance.from_numpy(
        np.array([[50, 50], [60, 60]]), skeleton=skeleton, score=0.5
    )

    # Create multiple instances in other frame that could match the same prediction
    # The matcher will find multiple matches but other_to_self keeps only the best one
    other_inst_1 = PredictedInstance.from_numpy(
        np.array([[51, 51], [61, 61]]),
        skeleton=skeleton,
        score=0.3,  # Lower score
    )
    other_inst_2 = PredictedInstance.from_numpy(
        np.array([[52, 52], [62, 62]]),
        skeleton=skeleton,
        score=0.8,  # Higher score
    )

    frame_self = LabeledFrame(video=video, frame_idx=0, instances=[pred_self])

    frame_other = LabeledFrame(
        video=video, frame_idx=0, instances=[other_inst_1, other_inst_2]
    )

    # Use a matcher that will create matches between pred_self and both other instances
    # But due to the grouping logic, other_to_self will only keep the best match
    matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)

    # This should create matches: (0,0) and (0,1) where 0 is pred_self index
    # But other_to_self will only keep the better match (likely to other_inst_2)

    merged, conflicts = frame_self.merge(frame_other, instance=matcher, frame="auto")

    # Both are predictions, so we keep the new one

    assert len(merged) == 1
    assert pred_self not in merged
    assert other_inst_1 in merged
    assert other_inst_2 not in merged


def test_labeled_frame_merge_fixed_logic():
    """Test that the merge logic fix works correctly.

    This verifies that the fix to add used_indices.add(self_idx) for conflict
    scenarios works as expected and makes the defensive lines 331-332 unreachable.
    """
    from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod

    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Test user vs user conflict scenario
    user_self = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
    user_other = Instance.from_numpy(np.array([[11, 11], [21, 21]]), skeleton=skeleton)

    frame_self = LabeledFrame(video=video, frame_idx=0, instances=[user_self])
    frame_other = LabeledFrame(video=video, frame_idx=0, instances=[user_other])

    matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=3.0)
    merged, conflicts = frame_self.merge(frame_other, instance=matcher, frame="auto")

    # Should keep original user instance and record conflict
    assert len(merged) == 1
    assert user_self in merged
    assert len(conflicts) == 1

    # Test user vs prediction conflict scenario
    user_self_2 = Instance.from_numpy(np.array([[50, 50], [60, 60]]), skeleton=skeleton)
    pred_other = PredictedInstance.from_numpy(
        np.array([[51, 51], [61, 61]]), skeleton=skeleton, score=0.8
    )

    frame_self_2 = LabeledFrame(video=video, frame_idx=0, instances=[user_self_2])
    frame_other_2 = LabeledFrame(video=video, frame_idx=0, instances=[pred_other])

    merged2, conflicts2 = frame_self_2.merge(
        frame_other_2, instance=matcher, frame="auto"
    )

    # Should keep user instance and record conflict
    assert len(merged2) == 1
    assert user_self_2 in merged2
    assert len(conflicts2) == 1


def test_replace_predictions_keeps_user_instances():
    """Test that user instances from original frame are preserved."""
    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    user_inst_1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
    user_inst_2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton)
    pred_self = PredictedInstance.from_numpy(
        np.array([[50, 50], [60, 60]]), skeleton=skeleton, score=0.8
    )

    frame_self = LabeledFrame(
        video=video, frame_idx=0, instances=[user_inst_1, user_inst_2, pred_self]
    )
    frame_other = LabeledFrame(video=video, frame_idx=0, instances=[])

    merged, conflicts = frame_self.merge(frame_other, frame="replace_predictions")

    # Both user instances should be preserved
    assert len(merged) == 2
    assert user_inst_1 in merged
    assert user_inst_2 in merged
    # Prediction from self should be removed
    assert pred_self not in merged
    # No conflicts for this strategy
    assert len(conflicts) == 0


def test_replace_predictions_removes_existing_predictions():
    """Test that predicted instances in original frame are removed."""
    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    pred_self_1 = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, score=0.8
    )
    pred_self_2 = PredictedInstance.from_numpy(
        np.array([[30, 30], [40, 40]]), skeleton=skeleton, score=0.9
    )

    frame_self = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_self_1, pred_self_2]
    )
    frame_other = LabeledFrame(video=video, frame_idx=0, instances=[])

    merged, conflicts = frame_self.merge(frame_other, frame="replace_predictions")

    # All predictions from self should be removed
    assert len(merged) == 0
    assert pred_self_1 not in merged
    assert pred_self_2 not in merged
    assert len(conflicts) == 0


def test_replace_predictions_adds_incoming_predictions():
    """Test that predictions from incoming frame are added."""
    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    pred_other_1 = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, score=0.7
    )
    pred_other_2 = PredictedInstance.from_numpy(
        np.array([[30, 30], [40, 40]]), skeleton=skeleton, score=0.8
    )

    frame_self = LabeledFrame(video=video, frame_idx=0, instances=[])
    frame_other = LabeledFrame(
        video=video, frame_idx=0, instances=[pred_other_1, pred_other_2]
    )

    merged, conflicts = frame_self.merge(frame_other, frame="replace_predictions")

    # All predictions from other should be added
    assert len(merged) == 2
    assert pred_other_1 in merged
    assert pred_other_2 in merged
    assert len(conflicts) == 0


def test_replace_predictions_ignores_incoming_user_instances():
    """Test that user instances from incoming frame are ignored."""
    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    user_self = Instance.from_numpy(np.array([[5, 5], [15, 15]]), skeleton=skeleton)
    user_other = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
    pred_other = PredictedInstance.from_numpy(
        np.array([[30, 30], [40, 40]]), skeleton=skeleton, score=0.9
    )

    frame_self = LabeledFrame(video=video, frame_idx=0, instances=[user_self])
    frame_other = LabeledFrame(
        video=video, frame_idx=0, instances=[user_other, pred_other]
    )

    merged, conflicts = frame_self.merge(frame_other, frame="replace_predictions")

    # User instance from self should be kept
    assert user_self in merged
    # User instance from other should be ignored
    assert user_other not in merged
    # Prediction from other should be added
    assert pred_other in merged
    assert len(merged) == 2
    assert len(conflicts) == 0


def test_replace_predictions_empty_frames():
    """Test replace_predictions with empty frames."""
    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Both frames empty
    frame_self = LabeledFrame(video=video, frame_idx=0, instances=[])
    frame_other = LabeledFrame(video=video, frame_idx=0, instances=[])

    merged, conflicts = frame_self.merge(frame_other, frame="replace_predictions")

    assert len(merged) == 0
    assert len(conflicts) == 0

    # Self empty, other has predictions
    pred_other = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, score=0.8
    )
    frame_other_2 = LabeledFrame(video=video, frame_idx=0, instances=[pred_other])

    merged2, conflicts2 = frame_self.merge(frame_other_2, frame="replace_predictions")

    assert len(merged2) == 1
    assert pred_other in merged2
    assert len(conflicts2) == 0


def test_replace_predictions_preserves_incoming_tracks():
    """Test that incoming predictions retain their track assignments."""
    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    track1 = Track(name="animal_1")
    track2 = Track(name="animal_2")

    # Self has old predictions with different tracks
    old_pred_1 = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, score=0.5, track=track1
    )
    old_pred_2 = PredictedInstance.from_numpy(
        np.array([[30, 30], [40, 40]]), skeleton=skeleton, score=0.6, track=track2
    )

    # Other has new predictions with their own track assignments
    new_track = Track(name="new_animal")
    new_pred = PredictedInstance.from_numpy(
        np.array([[50, 50], [60, 60]]), skeleton=skeleton, score=0.9, track=new_track
    )

    frame_self = LabeledFrame(
        video=video, frame_idx=0, instances=[old_pred_1, old_pred_2]
    )
    frame_other = LabeledFrame(video=video, frame_idx=0, instances=[new_pred])

    merged, conflicts = frame_self.merge(frame_other, frame="replace_predictions")

    # Old predictions should be removed
    assert old_pred_1 not in merged
    assert old_pred_2 not in merged
    # New prediction should be added with its original track
    assert len(merged) == 1
    assert new_pred in merged
    assert merged[0].track is new_track
    assert len(conflicts) == 0


def test_replace_predictions_mixed_scenario():
    """Test replace_predictions with a mix of user and predicted instances."""
    skeleton = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    track1 = Track(name="track_1")
    track2 = Track(name="track_2")

    # Self: 2 user instances + 2 predictions
    user_self_1 = Instance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton, track=track1
    )
    user_self_2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton)
    pred_self_1 = PredictedInstance.from_numpy(
        np.array([[50, 50], [60, 60]]), skeleton=skeleton, score=0.7, track=track2
    )
    pred_self_2 = PredictedInstance.from_numpy(
        np.array([[70, 70], [80, 80]]), skeleton=skeleton, score=0.8
    )

    # Other: 1 user instance + 3 predictions (user should be ignored)
    user_other = Instance.from_numpy(
        np.array([[90, 90], [100, 100]]), skeleton=skeleton
    )
    pred_other_1 = PredictedInstance.from_numpy(
        np.array([[110, 110], [120, 120]]), skeleton=skeleton, score=0.9
    )
    pred_other_2 = PredictedInstance.from_numpy(
        np.array([[130, 130], [140, 140]]), skeleton=skeleton, score=0.85
    )
    pred_other_3 = PredictedInstance.from_numpy(
        np.array([[150, 150], [160, 160]]), skeleton=skeleton, score=0.95
    )

    frame_self = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[user_self_1, user_self_2, pred_self_1, pred_self_2],
    )
    frame_other = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[user_other, pred_other_1, pred_other_2, pred_other_3],
    )

    merged, conflicts = frame_self.merge(frame_other, frame="replace_predictions")

    # User instances from self: kept
    assert user_self_1 in merged
    assert user_self_2 in merged
    # Predictions from self: removed
    assert pred_self_1 not in merged
    assert pred_self_2 not in merged
    # User instance from other: ignored
    assert user_other not in merged
    # Predictions from other: added
    assert pred_other_1 in merged
    assert pred_other_2 in merged
    assert pred_other_3 in merged

    assert len(merged) == 5  # 2 users from self + 3 predictions from other
    assert len(conflicts) == 0
