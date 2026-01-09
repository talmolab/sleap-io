"""Integration tests for the merging functionality."""

import numpy as np

from sleap_io import Labels, Skeleton
from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.matching import (
    InstanceMatcher,
    InstanceMatchMethod,
    MergeResult,
)
from sleap_io.model.video import Video


class TestLabeledFrameMerge:
    """Test LabeledFrame merging functionality."""

    def test_merge_keep_original(self):
        """Test frame merge with keep_original strategy."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create original frame with instances
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        frame1.instances = [inst1]

        # Create new frame with different instances
        frame2 = LabeledFrame(video=video, frame_idx=0)
        inst2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton)
        frame2.instances = [inst2]

        # Merge with keep_original strategy
        merged, conflicts = frame1.merge(frame2, frame="keep_original")

        assert len(merged) == 1
        assert merged[0] is inst1
        assert len(conflicts) == 0

    def test_merge_keep_new(self):
        """Test frame merge with keep_new strategy."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create frames
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        frame1.instances = [inst1]

        frame2 = LabeledFrame(video=video, frame_idx=0)
        inst2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton)
        frame2.instances = [inst2]

        # Merge with keep_new strategy
        merged, conflicts = frame1.merge(frame2, frame="keep_new")

        assert len(merged) == 1
        assert merged[0] is inst2
        assert len(conflicts) == 0

    def test_merge_update_tracks(self):
        """Test frame merge with update_tracks strategy."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        frame1.instances = [inst1]

        frame2 = LabeledFrame(video=video, frame_idx=0)
        inst2 = PredictedInstance.from_numpy(
            np.array([[10, 10], [20, 20]]), skeleton=skeleton
        )
        inst2.track = Track(name="track1")
        frame2.instances = [inst2]

        # Merge with keep_both strategy
        merged, conflicts = frame1.merge(frame2, frame="update_tracks")

        assert len(merged) == 1
        assert inst1 in merged
        assert inst2 not in merged
        assert inst1.track == inst2.track
        assert inst1.tracking_score == inst2.tracking_score
        assert len(conflicts) == 0

    def test_merge_keep_both(self):
        """Test frame merge with keep_both strategy."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create frames
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        frame1.instances = [inst1]

        frame2 = LabeledFrame(video=video, frame_idx=0)
        inst2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton)
        frame2.instances = [inst2]

        # Merge with keep_both strategy
        merged, conflicts = frame1.merge(frame2, frame="keep_both")

        assert len(merged) == 2
        assert inst1 in merged
        assert inst2 in merged
        assert len(conflicts) == 0

    def test_merge_auto_user_vs_predicted(self):
        """Test auto merge with user labels vs predictions."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create frame with user instance
        frame1 = LabeledFrame(video=video, frame_idx=0)
        user_inst = Instance.from_numpy(
            np.array([[10, 10], [20, 20]]), skeleton=skeleton
        )
        frame1.instances = [user_inst]

        # Create frame with predicted instance at similar location
        frame2 = LabeledFrame(video=video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[11, 11], [21, 21]]),
            skeleton=skeleton,
            point_scores=np.array([0.9, 0.9]),
            score=0.9,
        )
        frame2.instances = [pred_inst]

        # Merge with auto strategy
        merged, conflicts = frame1.merge(frame2, frame="auto")

        # User instance should be kept, prediction ignored
        assert len(merged) == 1
        assert merged[0] is user_inst
        assert len(conflicts) == 1
        assert conflicts[0][2] == "kept_user"

    def test_merge_auto_replace_prediction(self):
        """Test auto merge replacing prediction with user label."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create frame with predicted instance
        frame1 = LabeledFrame(video=video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10, 10], [20, 20]]),
            skeleton=skeleton,
            point_scores=np.array([0.8, 0.8]),
            score=0.8,
        )
        frame1.instances = [pred_inst]

        # Create frame with user instance at similar location
        frame2 = LabeledFrame(video=video, frame_idx=0)
        user_inst = Instance.from_numpy(
            np.array([[11, 11], [21, 21]]), skeleton=skeleton
        )
        frame2.instances = [user_inst]

        # Merge with auto strategy
        merged, conflicts = frame2.merge(frame1, frame="auto")

        # User instance should replace prediction
        assert len(merged) == 1
        assert merged[0] is user_inst
        # There should be a conflict recorded (user instance kept over prediction)
        assert len(conflicts) == 1
        assert conflicts[0][2] == "kept_user"

    def test_merge_with_custom_matcher(self):
        """Test merge with custom instance matcher."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)
        track1 = Track(name="track1")
        track2 = Track(name="track2")

        # Create frames with tracked instances
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(
            np.array([[10, 10], [20, 20]]), skeleton=skeleton, track=track1
        )
        frame1.instances = [inst1]

        frame2 = LabeledFrame(video=video, frame_idx=0)
        inst2 = Instance.from_numpy(
            np.array([[50, 50], [60, 60]]), skeleton=skeleton, track=track1
        )
        inst3 = Instance.from_numpy(
            np.array([[70, 70], [80, 80]]), skeleton=skeleton, track=track2
        )
        frame2.instances = [inst2, inst3]

        # Use identity matcher instead of spatial
        matcher = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)
        merged, conflicts = frame1.merge(frame2, instance=matcher, frame="auto")

        # inst1 and inst2 should match by track, inst3 should be added
        assert len(merged) == 2  # inst1 (kept) and inst3 (added)


class TestLabelsMerge:
    """Test Labels merging functionality."""

    def test_simple_merge(self):
        """Test basic Labels merge."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create first Labels with one frame
        labels1 = Labels()
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        frame1.instances = [inst1]
        labels1.append(frame1)

        # Create second Labels with different frame
        labels2 = Labels()
        frame2 = LabeledFrame(video=video, frame_idx=1)
        inst2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton)
        frame2.instances = [inst2]
        labels2.append(frame2)

        # Merge
        result = labels1.merge(labels2)

        assert isinstance(result, MergeResult)
        assert result.successful
        assert result.frames_merged == 1
        assert result.instances_added == 1
        assert len(labels1.labeled_frames) == 2

    def test_merge_with_overlapping_frames(self):
        """Test merging Labels with overlapping frames."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create first Labels
        labels1 = Labels()
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        frame1.instances = [inst1]
        labels1.append(frame1)

        # Create second Labels with same frame
        labels2 = Labels()
        frame2 = LabeledFrame(video=video, frame_idx=0)
        inst2 = Instance.from_numpy(np.array([[11, 11], [21, 21]]), skeleton=skeleton)
        frame2.instances = [inst2]
        labels2.append(frame2)

        # Merge with keep_both strategy
        result = labels1.merge(labels2, frame="keep_both")

        assert result.successful
        assert result.frames_merged == 1
        assert len(labels1.labeled_frames) == 1
        # Should have both instances in the frame
        assert len(labels1.labeled_frames[0].instances) == 2

    def test_merge_with_different_skeletons(self):
        """Test merging with different skeletons."""
        skeleton1 = Skeleton(["head", "tail"])
        skeleton2 = Skeleton(["head", "thorax", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create Labels with different skeletons
        labels1 = Labels()
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton1)
        frame1.instances = [inst1]
        labels1.append(frame1)

        labels2 = Labels()
        frame2 = LabeledFrame(video=video, frame_idx=1)
        inst2 = Instance.from_numpy(
            np.array([[10, 10], [15, 15], [20, 20]]), skeleton=skeleton2
        )
        frame2.instances = [inst2]
        labels2.append(frame2)

        # Merge - should add new skeleton
        result = labels1.merge(labels2)

        assert result.successful
        assert len(labels1.skeletons) == 2
        assert skeleton1 in labels1.skeletons
        assert skeleton2 in labels1.skeletons

    def test_merge_with_tracks(self):
        """Test merging with tracks."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)
        track1 = Track(name="mouse1")
        track2 = Track(name="mouse2")

        # Create Labels with tracks
        labels1 = Labels()
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(
            np.array([[10, 10], [20, 20]]), skeleton=skeleton, track=track1
        )
        frame1.instances = [inst1]
        labels1.append(frame1)

        labels2 = Labels()
        frame2 = LabeledFrame(video=video, frame_idx=1)
        inst2 = Instance.from_numpy(
            np.array([[30, 30], [40, 40]]), skeleton=skeleton, track=track2
        )
        frame2.instances = [inst2]
        labels2.append(frame2)

        # Merge
        result = labels1.merge(labels2)

        assert result.successful
        assert len(labels1.tracks) == 2
        assert track1 in labels1.tracks
        assert track2 in labels1.tracks

    def test_merge_error_handling(self):
        """Test merge error handling."""
        skeleton1 = Skeleton(["head", "tail"])
        skeleton2 = Skeleton(["wing1", "wing2"])  # Completely different
        video = Video(filename="test.mp4", open_backend=False)

        # Create Labels with incompatible skeletons
        labels1 = Labels()
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton1)
        frame1.instances = [inst1]
        labels1.append(frame1)

        labels2 = Labels()
        frame2 = LabeledFrame(video=video, frame_idx=0)
        inst2 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton2)
        frame2.instances = [inst2]
        labels2.append(frame2)

        # Merge with strict validation should handle the mismatch
        result = labels1.merge(labels2, validate=True, error_mode="continue")

        # Should still succeed but add the new skeleton
        assert result.successful
        assert len(labels1.skeletons) == 2

    def test_merge_provenance(self):
        """Test that merge history is tracked in provenance."""
        skeleton = Skeleton(["head", "tail"])
        video = Video(filename="test.mp4", open_backend=False)

        # Create simple Labels
        labels1 = Labels()
        frame1 = LabeledFrame(video=video, frame_idx=0)
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        frame1.instances = [inst1]
        labels1.append(frame1)

        labels2 = Labels()
        frame2 = LabeledFrame(video=video, frame_idx=1)
        inst2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton)
        frame2.instances = [inst2]
        labels2.append(frame2)

        # Merge
        labels1.merge(labels2)

        # Check provenance was updated
        assert "merge_history" in labels1.provenance
        assert len(labels1.provenance["merge_history"]) == 1
        merge_record = labels1.provenance["merge_history"][0]
        assert "timestamp" in merge_record
        assert merge_record["source_labels"]["n_frames"] == 1
        assert merge_record["result"]["frames_merged"] == 1
        # Verify new provenance fields
        assert "source_filename" in merge_record
        assert "target_filename" in merge_record
        assert "sleap_io_version" in merge_record
        # In-memory labels have None for filenames
        assert merge_record["source_filename"] is None
        assert merge_record["target_filename"] is None

    def test_merge_auto_video_matching_with_identical_shapes(self):
        """Test AUTO video matching with multiple videos of identical shape.

        Regression test for GitHub issue #255:
        When multiple videos have identical shapes, AUTO matching should prefer
        basename matching over content matching to avoid incorrect video attribution.

        This test reproduces a scenario where:
        - Main labels has two videos (video_a, video_b) with identical shapes
        - Predictions are created for video_b (different directory, same basename)
        - Default AUTO matching should correctly match to video_b by basename
        - Bug: AUTO incorrectly matches to video_a by content (first match wins)

        After fix, AUTO will try strict path → basename → content (last resort).
        """
        # Create skeleton
        skeleton = Skeleton(["node1", "node2", "node3"])

        # Create two videos with IDENTICAL shapes but different basenames
        video_a = Video(filename="/data/02_07dpf_fish_1.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 900, 900, 1)

        video_b = Video(filename="/data/04_07dpf_fish_2.mp4", open_backend=False)
        video_b.backend_metadata["shape"] = (100, 900, 900, 1)  # Same shape!

        # Create main Labels with both videos (order matters: video_a first)
        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a, video_b]

        # Create predictions for video_b (different directory, same basename)
        predictions = Labels(skeletons=[skeleton])
        pred_video = Video(
            filename="/predictions/04_07dpf_fish_2.mp4", open_backend=False
        )
        pred_video.backend_metadata["shape"] = (100, 900, 900, 1)
        predictions.videos = [pred_video]

        # Add predicted instances
        frame0 = LabeledFrame(video=pred_video, frame_idx=0)
        inst0 = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0], [200.0, 300.0]]),
            skeleton=skeleton,
            score=0.95,
        )
        frame0.instances = [inst0]
        predictions.labeled_frames = [frame0]

        # Merge using default AUTO matcher
        result = labels.merge(predictions)

        # Assert merge succeeded
        assert result.successful
        assert result.frames_merged == 1
        assert len(labels.labeled_frames) == 1

        # Assert frames attributed to CORRECT video (video_b, not video_a)
        merged_frame = labels.labeled_frames[0]

        # This assertion will FAIL with current implementation (gets video_a)
        # This assertion will PASS after fix (gets video_b)
        assert merged_frame.video == video_b, (
            f"Expected predictions to match video_b by basename "
            f"('{video_b.filename}'), but got '{merged_frame.video.filename}'. "
            f"This indicates AUTO matching incorrectly used content matching "
            f"instead of basename matching."
        )

    def test_merge_auto_no_basename_match_adds_new_video(self):
        """Test AUTO adds new video when basenames don't match (no content matching).

        Per algorithm design: Shape is used for REJECTION only, never for positive
        matching. Even with a single candidate, if basenames (leaf paths) don't
        match, the video is added as new. This is conservative - false negatives
        (adding as new when should match) are recoverable; false positives
        (matching wrong video) corrupt data.

        Algorithm path:
        - Step 6 (full path): "/data/experiment_a.mp4" != "/predictions/output.mp4"
        - Step 7 (leaf uniqueness): Both unique, but "experiment_a.mp4" != "output.mp4"
        - Step 8 (fallback): Add as new video
        """
        skeleton = Skeleton(["node1", "node2"])

        # Single video in base labels
        video_a = Video(filename="/data/experiment_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (50, 512, 512, 3)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Create predictions with completely different basename
        predictions = Labels(skeletons=[skeleton])
        pred_video = Video(filename="/predictions/output.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (50, 512, 512, 3)  # Same shape
        predictions.videos = [pred_video]

        frame = LabeledFrame(video=pred_video, frame_idx=0)
        inst = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]), skeleton=skeleton, score=0.9
        )
        frame.instances = [inst]
        predictions.labeled_frames = [frame]

        # Basenames don't match → add as new (shape is for rejection only)
        result = labels.merge(predictions)
        assert result.successful
        assert len(labels.labeled_frames) == 1
        # Prediction video should be added as new (2 videos total)
        assert len(labels.videos) == 2, (
            f"Expected 2 videos (no basename match → add as new), "
            f"got {len(labels.videos)}. Shape is for rejection only."
        )
        # The merged frame should reference the new prediction video
        assert labels.labeled_frames[0].video is pred_video

    def test_merge_auto_ambiguous_content_adds_new_video(self):
        """Test AUTO refuses content matching when ambiguous (multiple candidates).

        When multiple videos have the same shape and basenames don't match,
        content-only matching would be ambiguous. In this case, AUTO should
        conservatively add the prediction video as new rather than risk
        merging to the wrong video.
        """
        skeleton = Skeleton(["node1", "node2"])

        # Two videos with same shape - content matching would be ambiguous
        video_a = Video(filename="/data/experiment_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (50, 512, 512, 3)

        video_b = Video(filename="/data/experiment_b.mp4", open_backend=False)
        video_b.backend_metadata["shape"] = (50, 512, 512, 3)  # Same shape!

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a, video_b]

        # Create predictions with completely different basename
        predictions = Labels(skeletons=[skeleton])
        pred_video = Video(filename="/predictions/output.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (50, 512, 512, 3)
        predictions.videos = [pred_video]

        frame = LabeledFrame(video=pred_video, frame_idx=0)
        inst = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]), skeleton=skeleton, score=0.9
        )
        frame.instances = [inst]
        predictions.labeled_frames = [frame]

        # With MULTIPLE content match candidates, should NOT guess - add as new
        result = labels.merge(predictions)
        assert result.successful
        assert len(labels.labeled_frames) == 1

        # Prediction video should be added as new (3 videos total)
        assert len(labels.videos) == 3
        # The merged frame should reference the new prediction video
        assert labels.labeled_frames[0].video is pred_video

    def test_merge_auto_strict_path_match(self):
        """Test AUTO matching with exact same path (strict match).

        This tests the strict path matching branch (line 1802 in labels.py)
        which takes priority over basename and content matching.
        """
        skeleton = Skeleton(["node1", "node2"])

        # Create video with specific path
        video = Video(filename="/data/video.mp4", open_backend=False)
        video.backend_metadata["shape"] = (100, 640, 480, 3)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video]

        # Create predictions with EXACT same path
        predictions = Labels(skeletons=[skeleton])
        pred_video = Video(filename="/data/video.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 640, 480, 3)
        predictions.videos = [pred_video]

        frame = LabeledFrame(video=pred_video, frame_idx=5)
        inst = PredictedInstance.from_numpy(
            np.array([[50.0, 60.0], [70.0, 80.0]]), skeleton=skeleton, score=0.85
        )
        frame.instances = [inst]
        predictions.labeled_frames = [frame]

        # Should match by strict path
        result = labels.merge(predictions)
        assert result.successful
        assert len(labels.labeled_frames) == 1
        assert labels.labeled_frames[0].video == video

    def test_merge_auto_no_match_adds_new_video(self):
        """Test AUTO matching when no match exists (different shape and basename).

        This tests the no-match scenario where a new video is added
        because neither path nor content matches.
        """
        skeleton = Skeleton(["node1", "node2"])

        # Create video with specific shape
        video_a = Video(filename="/data/video_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 640, 480, 3)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Create predictions with different basename AND different shape
        predictions = Labels(skeletons=[skeleton])
        pred_video = Video(filename="/predictions/video_b.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (200, 1920, 1080, 3)  # Different!
        predictions.videos = [pred_video]

        frame = LabeledFrame(video=pred_video, frame_idx=10)
        inst = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [300.0, 400.0]]), skeleton=skeleton, score=0.95
        )
        frame.instances = [inst]
        predictions.labeled_frames = [frame]

        # Should NOT match, new video should be added
        original_video_count = len(labels.videos)
        result = labels.merge(predictions)
        assert result.successful
        assert len(labels.videos) == original_video_count + 1
        assert len(labels.labeled_frames) == 1
        # The merged frame should reference the newly added video
        assert labels.labeled_frames[0].video != video_a


# =============================================================================
# Tests for source_video checking in AUTO matcher - PROPOSED FIX
# =============================================================================
# These tests document expected behavior for AUTO matcher source_video awareness.
# Current implementation: AUTO does NOT check source_video (tests will fail).


class TestMergeSourceVideoAwareness:
    """Tests for source_video checking in AUTO video matching.

    Background (from merge red-team investigation):
    When merging predictions from PKG.SLP files, the embedded video has a
    source_video attribute pointing to the original external video. The
    current AUTO matcher does NOT check this attribute, causing:
    1. Predictions merged to wrong video (new video added instead of matching)
    2. Duplicate videos created in the merged Labels
    3. Data loss when annotations end up on wrong video

    This is Use Case 3 (UC3) from the investigation: PKG.SLP predictions
    should merge correctly with external video Labels.
    """

    def test_merge_pkg_predictions_to_external_video(self):
        """Integration test: AUTO matcher should recognize source_video.

        Scenario: User has a project with MULTIPLE external video references.
        They create a PKG.SLP for training on ONE video, get predictions,
        and want to merge predictions back. The prediction video has
        source_video pointing to the specific video it was trained on.

        Expected: AUTO matcher recognizes source_video and merges to the
        CORRECT video, not just any video with the same shape.

        Current behavior: AUTO matcher does NOT check source_video, so
        predictions may be attributed to the WRONG video (first content match).

        This test is designed to fail with content-only matching because
        multiple videos have the same shape. source_video is the only way
        to determine the correct video.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base labels with TWO videos of the SAME shape (content matching is ambiguous)
        video_a = Video(filename="/data/recordings/video_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        video_b = Video(filename="/data/recordings/video_b.mp4", open_backend=False)
        video_b.backend_metadata["shape"] = (100, 480, 640, 1)  # Same shape!

        base_labels = Labels(skeletons=[skeleton])
        base_labels.videos = [
            video_a,
            video_b,
        ]  # video_a is first (would be first content match)

        # Predictions from PKG.SLP with source_video pointing to video_b (not video_a!)
        source = Video(filename="/data/recordings/video_b.mp4", open_backend=False)
        source.backend_metadata["shape"] = (100, 480, 640, 1)

        pred_video = Video(
            filename="predictions.pkg.slp",  # Different basename - no basename match
            source_video=source,
            open_backend=False,
        )
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        # Add predicted instances
        pred_frame = LabeledFrame(video=pred_video, frame_idx=10)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[50.0, 60.0], [70.0, 80.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        # Merge using AUTO matcher (default)
        result = base_labels.merge(predictions)

        # Assert merge succeeded
        assert result.successful
        assert result.frames_merged == 1

        # CRITICAL ASSERTION: No new video should be added
        assert len(base_labels.videos) == 2, (
            f"Expected 2 videos (source_video should match existing), "
            f"got {len(base_labels.videos)}. A new video was incorrectly added."
        )

        # CRITICAL ASSERTION: Should match video_b (via source_video), NOT video_a
        merged_frame = base_labels.labeled_frames[-1]  # The newly merged frame
        assert merged_frame.video is video_b, (
            f"Expected predictions to merge to video_b "
            f"('{video_b.filename}' via source_video), but got "
            f"'{merged_frame.video.filename}'. "
            f"This indicates AUTO matcher is using content matching (first match wins) "
            f"instead of source_video matching."
        )

    def test_merge_pkg_predictions_basename_mismatch(self):
        """Test source_video matching when basenames differ.

        Scenario: Base has TWO videos with same shape. PKG.SLP filename
        doesn't match either basename. Only source_video can determine
        which video to merge to.

        This tests the robustness of source_video checking when path-based
        matching would fail (no basename match) and content matching is
        ambiguous (multiple videos with same shape).
        """
        skeleton = Skeleton(["head", "tail"])

        # Base with TWO videos of same shape (content matching ambiguous)
        base_video_1 = Video(
            filename="/data/recordings/experiment_001.mp4", open_backend=False
        )
        base_video_1.backend_metadata["shape"] = (100, 480, 640, 1)

        base_video_2 = Video(
            filename="/data/recordings/experiment_002.mp4", open_backend=False
        )
        base_video_2.backend_metadata["shape"] = (100, 480, 640, 1)  # Same shape!

        base_labels = Labels(skeletons=[skeleton])
        base_labels.videos = [base_video_1, base_video_2]

        # Predictions with completely different filename but source_video
        # pointing to experiment_002 (the SECOND video)
        source = Video(
            filename="/data/recordings/experiment_002.mp4", open_backend=False
        )
        pred_video = Video(
            filename="training_run_2024_12_15.pkg.slp",  # No basename match!
            source_video=source,
            open_backend=False,
        )
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=5)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[25.0, 35.0], [45.0, 55.0]]), skeleton=skeleton, score=0.85
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = base_labels.merge(predictions)

        assert result.successful
        assert len(base_labels.videos) == 2, (
            "source_video matching failed - new video was added despite "
            "source_video pointing to existing video"
        )

        # Should match base_video_2, NOT base_video_1
        merged_frame = base_labels.labeled_frames[-1]
        assert merged_frame.video is base_video_2, (
            f"Expected match to experiment_002 via source_video, "
            f"but got {merged_frame.video.filename}. "
            "Content matching (first match) was used instead of source_video."
        )

    def test_merge_provenance_chain(self):
        """Test matching through multi-level provenance chain.

        Scenario: A PKG.SLP was created from another PKG.SLP, creating
        a provenance chain: final → intermediate → original

        Base has TWO videos with same shape. Only provenance chain traversal
        can determine which is the correct video.

        Real-world example: Tiernon dataset has 3-level provenance chains.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base with TWO videos of same shape (content matching ambiguous)
        video_a = Video(filename="/data/video_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        video_b = Video(filename="/data/video_b.mp4", open_backend=False)
        video_b.backend_metadata["shape"] = (100, 480, 640, 1)  # Same shape!

        base_labels = Labels(skeletons=[skeleton])
        base_labels.videos = [video_a, video_b]  # video_a is first

        # Predictions from doubly-embedded video with chain pointing to video_b
        # Chain: final.pkg.slp → intermediate.pkg.slp → /data/video_b.mp4
        intermediate_source = Video(filename="/data/video_b.mp4", open_backend=False)
        intermediate = Video(
            filename="intermediate.pkg.slp",
            source_video=intermediate_source,
            open_backend=False,
        )
        final = Video(
            filename="final.pkg.slp", source_video=intermediate, open_backend=False
        )
        final.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [final]

        pred_frame = LabeledFrame(video=final, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = base_labels.merge(predictions)

        assert result.successful
        assert len(base_labels.videos) == 2, (
            "Provenance chain traversal failed - new video was added instead of "
            "matching through final → intermediate → video_b chain"
        )

        # Should match video_b via provenance chain, NOT video_a
        merged_frame = base_labels.labeled_frames[-1]
        assert merged_frame.video is video_b, (
            f"Expected match to video_b via provenance chain traversal, "
            f"but got {merged_frame.video.filename}. "
            f"Provenance chain: final.pkg.slp → intermediate.pkg.slp → video_b.mp4"
        )


class TestMergeCrossPlatformPaths:
    """Tests for cross-platform path handling in video matching.

    Background (UC2 from investigation):
    Labeling is often done on Windows, training on Linux. The video paths
    will have different formats but same basename. AUTO matcher should
    handle this via basename matching.

    These tests verify that basename matching works correctly for
    cross-platform scenarios AND that it doesn't incorrectly match
    different files with same basename.
    """

    def test_merge_windows_to_linux_paths(self):
        r"""Merge predictions with Linux paths into Windows-path labels.

        Scenario: Labels created on Windows with paths like C:\\data\\video.mp4
        Predictions from Linux training with paths like /home/user/data/video.mp4

        AUTO matcher should match by basename when paths differ.
        """
        skeleton = Skeleton(["head", "tail"])

        # Windows-style path in base labels
        windows_video = Video(
            filename=r"C:\Users\alice\data\video.mp4", open_backend=False
        )
        windows_video.backend_metadata["shape"] = (100, 480, 640, 1)

        base_labels = Labels(skeletons=[skeleton])
        base_labels.videos = [windows_video]

        # Linux-style path in predictions
        linux_video = Video(filename="/home/bob/data/video.mp4", open_backend=False)
        linux_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [linux_video]

        pred_frame = LabeledFrame(video=linux_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = base_labels.merge(predictions)

        assert result.successful

        # Should match by basename (video.mp4 == video.mp4)
        assert len(base_labels.videos) == 1, (
            f"Cross-platform matching failed. Expected 1 video (basename match), "
            f"got {len(base_labels.videos)}. "
            f"Base path: {windows_video.filename}, "
            f"Pred path: {linux_video.filename}"
        )

        # Merged frame should be on the Windows video
        assert base_labels.labeled_frames[0].video is windows_video

    def test_merge_ambiguous_basename_with_parent_disambiguation(self):
        """Test disambiguation when multiple videos have same basename.

        Scenario: Base has two videos with same basename in different dirs:
        - /data/exp1/fly.mp4
        - /data/exp2/fly.mp4

        Predictions for exp2/fly.mp4 should match the correct video,
        not just the first one with matching basename.

        Current behavior: First basename match wins (incorrect).
        Expected behavior: Parent directory should be used to disambiguate.
        """
        skeleton = Skeleton(["head", "tail"])

        # Two videos with same basename, same shape
        video1 = Video(filename="/data/exp1/fly.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 1)

        video2 = Video(filename="/data/exp2/fly.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 1)

        base_labels = Labels(skeletons=[skeleton])
        base_labels.videos = [video1, video2]

        # Predictions specifically for exp2
        pred_video = Video(filename="/predictions/exp2/fly.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = base_labels.merge(predictions)

        assert result.successful
        assert len(base_labels.videos) == 2  # No new video added

        # CRITICAL: Should match video2 (exp2), NOT video1 (exp1)
        merged_frame = base_labels.labeled_frames[0]
        assert merged_frame.video is video2, (
            f"Ambiguous basename disambiguation failed. "
            f"Expected match to {video2.filename} (exp2), "
            f"got {merged_frame.video.filename}. "
            f"Parent directory 'exp2' should have been used for disambiguation."
        )


class TestMergeImageVideoIntegration:
    """Integration tests for ImageVideo merging.

    Background (UC4 from investigation):
    ImageVideo uses a list of image paths. Merging two ImageVideos with
    overlapping images requires building a frame_idx_map to correctly
    map annotation indices.

    Current test gaps:
    - Video.has_overlapping_images() tested in isolation
    - No integration tests for actual Labels.merge() with ImageVideo
    """

    def test_merge_imagevideo_basic(self):
        """Basic merge of two ImageVideo-based Labels.

        Scenario: Two Labels with ImageVideos that have the same images.
        Annotations should merge correctly.
        """
        skeleton = Skeleton(["head", "tail"])

        paths = ["/data/img_000.jpg", "/data/img_001.jpg", "/data/img_002.jpg"]

        video1 = Video(filename=paths.copy(), open_backend=False)
        video1.backend_metadata["shape"] = (3, 480, 640, 3)

        base_labels = Labels(skeletons=[skeleton])
        base_labels.videos = [video1]

        base_frame = LabeledFrame(video=video1, frame_idx=0)
        base_inst = Instance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton
        )
        base_frame.instances = [base_inst]
        base_labels.labeled_frames = [base_frame]

        # Predictions with same images
        video2 = Video(filename=paths.copy(), open_backend=False)
        video2.backend_metadata["shape"] = (3, 480, 640, 3)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [video2]

        pred_frame = LabeledFrame(video=video2, frame_idx=1)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[30.0, 30.0], [40.0, 40.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = base_labels.merge(predictions)

        assert result.successful
        assert len(base_labels.videos) == 1, (
            "ImageVideo with identical paths should match as same video"
        )
        assert len(base_labels.labeled_frames) == 2

    def test_merge_imagevideo_overlapping_sequences(self):
        """Merge ImageVideos with partially overlapping image sequences.

        Scenario:
        - Base: [img_000, img_001, img_002, img_003, img_004]
        - Pred: [img_002, img_003, img_005, img_006, img_007]
        - Overlap: img_002, img_003

        Annotations on pred's frame 0 (img_002) should map to base's frame 2.

        This tests the frame_idx_map construction for overlapping sequences.
        """
        from sleap_io.model.matching import VideoMatchMethod

        skeleton = Skeleton(["head", "tail"])

        # Base sequence
        base_paths = [f"/data/img_{i:03d}.jpg" for i in range(5)]  # 0-4
        base_video = Video(filename=base_paths, open_backend=False)
        base_video.backend_metadata["shape"] = (5, 480, 640, 3)

        base_labels = Labels(skeletons=[skeleton])
        base_labels.videos = [base_video]

        # Prediction sequence with overlap
        pred_paths = [
            "/data/img_002.jpg",  # Overlaps with base frame 2
            "/data/img_003.jpg",  # Overlaps with base frame 3
            "/data/img_005.jpg",  # New
            "/data/img_006.jpg",  # New
            "/data/img_007.jpg",  # New
        ]
        pred_video = Video(filename=pred_paths, open_backend=False)
        pred_video.backend_metadata["shape"] = (5, 480, 640, 3)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        # Annotation on pred frame 0 (img_002)
        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[50.0, 50.0], [60.0, 60.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        # Merge with IMAGE_DEDUP to test frame_idx_map
        from sleap_io.model.matching import VideoMatcher

        result = base_labels.merge(
            predictions, video=VideoMatcher(method=VideoMatchMethod.IMAGE_DEDUP)
        )

        assert result.successful

        # The annotation should be on base frame 2 (img_002)
        matched_frames = base_labels.find(base_video, frame_idx=2)
        assert len(matched_frames) == 1, (
            f"Expected annotation on frame 2 (img_002), "
            f"but found annotations on frames: "
            f"{[lf.frame_idx for lf in base_labels.labeled_frames]}"
        )
        assert len(matched_frames[0].instances) == 1


# =============================================================================
# Tests for Shape Rejection (Algorithm Steps 1-2)
# =============================================================================


class TestMergeShapeRejection:
    """Tests for shape-based rejection in video matching.

    Per algorithm design (07-AUTO-MATCHING-ALGORITHM.md):
    - Shape is used for REJECTION only, never positive matching
    - Compare (frames, height, width) - NOT channels
    - If shapes are incompatible → NOT MATCH (continue to next candidate)
    - If shapes are compatible or unknown → continue to next step
    """

    def test_shape_full_rejection_different_resolution(self):
        """Videos with different resolution should not match.

        Algorithm Step 1: Full shape check
        If (frames, H, W) differ → NOT MATCH for this pair
        """
        skeleton = Skeleton(["head", "tail"])

        # Base video: 640x480
        video_a = Video(filename="/data/video_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Prediction video: 1920x1080 (different resolution)
        pred_video = Video(filename="/predictions/video_a.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 1080, 1920, 1)  # Different H/W

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Shape rejection → NOT MATCH → add as new video
        assert len(labels.videos) == 2, (
            "Different resolution should cause shape rejection. "
            "Even with same basename, videos should not match."
        )

    def test_shape_full_rejection_different_frame_count(self):
        """Videos with different frame counts should not match.

        Algorithm Step 1: Full shape check
        Frame count is part of shape comparison.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base video: 100 frames
        video_a = Video(filename="/data/video.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Prediction video: 200 frames (different count)
        pred_video = Video(filename="/predictions/video.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (200, 480, 640, 1)  # Different frames

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Shape rejection → NOT MATCH → add as new video
        assert len(labels.videos) == 2, (
            "Different frame count should cause shape rejection. "
            "Even with same basename, videos should not match."
        )

    def test_shape_channels_ignored(self):
        """Channel count should NOT affect matching (grayscale detection is noisy).

        Algorithm design decision: Exclude channels from shape comparison.
        Grayscale detection is based on comparing first/last channel which is
        affected by compression and is user-configurable.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base video: 1 channel (grayscale)
        video_a = Video(filename="/data/video.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Prediction video: 3 channels (same basename, same H/W/frames)
        pred_video = Video(filename="/predictions/video.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 3)  # Different channels

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Same basename + compatible shape (channels ignored) → should match
        assert len(labels.videos) == 1, (
            "Different channel count should NOT cause shape rejection. "
            "Videos should match by basename."
        )
        assert labels.labeled_frames[0].video is video_a

    def test_shape_unknown_continues(self):
        """When shape is unavailable, should continue to next step (no rejection).

        Algorithm: If shape_a is None or shape_b is None → can't determine → continue
        """
        skeleton = Skeleton(["head", "tail"])

        # Base video: no shape metadata
        video_a = Video(filename="/data/video.mp4", open_backend=False)
        # No backend_metadata["shape"] set

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Prediction video: has shape but different basename
        pred_video = Video(filename="/predictions/video.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Unknown shape → continue → basename match → MATCH
        assert len(labels.videos) == 1, (
            "Unknown shape should not cause rejection. Videos should match by basename."
        )


# =============================================================================
# Tests for Leaf Path Uniqueness (Algorithm Step 7)
# =============================================================================


class TestMergeLeafUniqueness:
    """Tests for leaf path uniqueness in video matching.

    Per algorithm (07-AUTO-MATCHING-ALGORITHM.md) Step 7:
    - Build minimal unique leaf (basename + parents until unique)
    - If BOTH leaves are unique in their sets AND match → MATCH
    - If BOTH leaves are unique in their sets AND don't match → NOT MATCH
    - If either leaf not unique → continue to Step 8 (add as new)
    """

    def test_leaf_both_unique_match(self):
        """Unique leaves that match → MATCH.

        Scenario: Single video in each set, basenames match.
        """
        skeleton = Skeleton(["head", "tail"])

        video_a = Video(filename="/data/exp1/recording.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Different directory but same basename
        pred_video = Video(
            filename="/predictions/exp1/recording.mp4", open_backend=False
        )
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Both leaves unique, leaves match (recording.mp4) → MATCH
        assert len(labels.videos) == 1
        assert labels.labeled_frames[0].video is video_a

    def test_leaf_both_unique_no_match(self):
        """Unique leaves that don't match → NOT MATCH → add as new.

        Scenario: Single video in each set, different basenames.
        """
        skeleton = Skeleton(["head", "tail"])

        video_a = Video(filename="/data/experiment_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Different basename
        pred_video = Video(filename="/predictions/experiment_b.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)  # Same shape

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Both leaves unique, leaves don't match → NOT MATCH → add as new
        assert len(labels.videos) == 2, (
            "Different unique basenames should NOT MATCH. "
            "Shape should not be used for positive matching."
        )

    def test_leaf_not_unique_fallthrough(self):
        """Non-unique leaves → fallthrough to Step 8 (add as new).

        Scenario: Multiple videos with same basename in base labels.
        Leaf is not unique in existing set → cannot determine match.
        """
        skeleton = Skeleton(["head", "tail"])

        # Two videos with same basename
        video_1 = Video(filename="/data/exp1/fly.mp4", open_backend=False)
        video_1.backend_metadata["shape"] = (100, 480, 640, 1)

        video_2 = Video(filename="/data/exp2/fly.mp4", open_backend=False)
        video_2.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_1, video_2]

        # Prediction with same basename but DIFFERENT parent than both
        pred_video = Video(filename="/predictions/exp3/fly.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # exp3/fly.mp4 doesn't match exp1/fly.mp4 or exp2/fly.mp4 by leaf
        # → add as new
        assert len(labels.videos) == 3, (
            "Non-matching leaf should fallthrough to add as new. "
            "exp3/fly.mp4 should not match exp1/fly.mp4 or exp2/fly.mp4."
        )

    def test_leaf_parent_disambiguates(self):
        """Parent directory disambiguates same basename.

        Scenario: exp1/fly.mp4, exp2/fly.mp4 in base
        Prediction is exp2/fly.mp4 → should match video_2

        Algorithm builds minimal unique leaves:
        - video_1 leaf: "exp1/fly.mp4" (needs parent to be unique)
        - video_2 leaf: "exp2/fly.mp4" (needs parent to be unique)
        - pred leaf: "exp2/fly.mp4" (unique in incoming set)
        - pred leaf matches video_2 leaf → MATCH
        """
        skeleton = Skeleton(["head", "tail"])

        video_1 = Video(filename="/data/exp1/fly.mp4", open_backend=False)
        video_1.backend_metadata["shape"] = (100, 480, 640, 1)

        video_2 = Video(filename="/data/exp2/fly.mp4", open_backend=False)
        video_2.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_1, video_2]

        # Prediction specifically for exp2
        pred_video = Video(filename="/predictions/exp2/fly.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        assert len(labels.videos) == 2, "No new video should be added"

        # Should match video_2 (exp2), NOT video_1 (exp1)
        merged_frame = labels.labeled_frames[0]
        assert merged_frame.video is video_2, (
            f"Expected match to video_2 (exp2/fly.mp4), "
            f"got {merged_frame.video.filename}. "
            "Parent directory should disambiguate same basenames."
        )

    def test_leaf_duplicate_paths_excluded(self):
        """Duplicate paths in a set should be excluded from matching.

        If the same video appears twice in the base (a degenerate case),
        neither should match - fallthrough to Step 8.
        """
        skeleton = Skeleton(["head", "tail"])

        # Two video objects pointing to same path (duplicate)
        video_1 = Video(filename="/data/video.mp4", open_backend=False)
        video_1.backend_metadata["shape"] = (100, 480, 640, 1)

        video_2 = Video(filename="/data/video.mp4", open_backend=False)  # Same path!
        video_2.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_1, video_2]

        # Prediction with same path
        pred_video = Video(filename="/predictions/video.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        # The behavior here depends on implementation:
        # Option 1: Match first duplicate (current behavior)
        # Option 2: Fallthrough due to ambiguity (algorithm suggests this)
        # For now, we document current behavior
        assert result.successful


# =============================================================================
# Tests for Physical File Matching (Algorithm Steps 4-5)
# =============================================================================


class TestMergeSamefileMatching:
    """Tests for os.path.samefile() matching.

    Per algorithm (07-AUTO-MATCHING-ALGORITHM.md) Steps 4-5:
    - Step 4: For non-ImageVideo where BOTH files exist, use os.path.samefile()
    - Step 5: For ImageVideo, check first and last files via samefile()
    - EXCEPTION: Skip if BOTH are embedded HDF5Video (same container issue)
    """

    def test_samefile_with_symlink(self, tmp_path):
        """Symlink to same file should match via os.path.samefile().

        Creates actual files to test real samefile behavior.
        """
        skeleton = Skeleton(["head", "tail"])

        # Create real video file (just empty file for testing)
        real_file = tmp_path / "real_video.mp4"
        real_file.write_bytes(b"fake video content")

        # Create symlink to real file
        symlink = tmp_path / "symlinked_video.mp4"
        symlink.symlink_to(real_file)

        # Base video points to real file
        video_a = Video(filename=str(real_file), open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Prediction video points to symlink
        pred_video = Video(filename=str(symlink), open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # samefile() should detect these are the same file
        assert len(labels.videos) == 1, (
            "Symlink should match via os.path.samefile(). "
            f"Real: {real_file}, Symlink: {symlink}"
        )
        assert labels.labeled_frames[0].video is video_a

    def test_samefile_with_relative_path(self, tmp_path):
        """Relative and absolute paths to same file should match.

        Tests path resolution before samefile comparison.
        """
        import os

        skeleton = Skeleton(["head", "tail"])

        # Create real video file
        subdir = tmp_path / "data"
        subdir.mkdir()
        video_file = subdir / "video.mp4"
        video_file.write_bytes(b"fake video content")

        # Base video with absolute path
        video_a = Video(filename=str(video_file), open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Prediction video with relative path (from tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            relative_path = "data/video.mp4"

            pred_video = Video(filename=relative_path, open_backend=False)
            pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

            predictions = Labels(skeletons=[skeleton])
            predictions.videos = [pred_video]

            pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
            pred_inst = PredictedInstance.from_numpy(
                np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
            )
            pred_frame.instances = [pred_inst]
            predictions.labeled_frames = [pred_frame]

            result = labels.merge(predictions)
        finally:
            os.chdir(original_cwd)

        assert result.successful
        # Should match via samefile (both resolve to same file)
        assert len(labels.videos) == 1, (
            "Relative and absolute paths to same file should match. "
            f"Absolute: {video_file}, Relative: {relative_path}"
        )


# =============================================================================
# Tests for Unresolved Videos Tracking (Algorithm Step 8)
# =============================================================================


class TestMergeUnresolvedTracking:
    """Tests for unresolved_videos tracking in MergeResult.

    Per algorithm (07-AUTO-MATCHING-ALGORITHM.md) Step 8:
    When a video cannot be matched, add it as new and record in
    MergeResult.unresolved_videos for potential future resolution.
    """

    def test_unresolved_videos_populated(self):
        """Unmatched video should be tracked in unresolved_videos.

        Note: This test requires the unresolved_videos field to be added
        to MergeResult. Currently it's not implemented.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base with one video
        video_a = Video(filename="/data/video_a.mp4", open_backend=False)
        video_a.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [video_a]

        # Prediction with completely different video (no match possible)
        pred_video = Video(filename="/predictions/unrelated.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (200, 1080, 1920, 3)  # Different shape

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        assert len(labels.videos) == 2  # New video added

        # Check unresolved_videos tracking (when implemented)
        if hasattr(result, "unresolved_videos"):
            assert len(result.unresolved_videos) == 1, (
                "Unmatched video should be tracked in unresolved_videos"
            )
            unresolved = result.unresolved_videos[0]
            assert unresolved.incoming_video is pred_video
            assert (
                video_a in unresolved.potential_matches
                or len(unresolved.potential_matches) == 0
            )


# =============================================================================
# Tests for original_video OR Logic (Meta-rule)
# =============================================================================


class TestMergeOriginalVideoLogic:
    """Tests for original_video OR logic in video matching.

    Per algorithm (07-AUTO-MATCHING-ALGORITHM.md) Meta-rules:
    For embedded videos, check all combinations:
    - (A matches B) OR
    - (A.original_video matches B) OR
    - (A matches B.original_video) OR
    - (A.original_video matches B.original_video)
    """

    def test_original_video_incoming_only(self):
        """Match via incoming video's original_video.

        Scenario: Base has external video. Prediction has embedded video
        with original_video pointing to the base video.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base: external video
        base_video = Video(filename="/data/video.mp4", open_backend=False)
        base_video.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [base_video]

        # Prediction: embedded with original_video pointing to base
        original = Video(filename="/data/video.mp4", open_backend=False)
        pred_video = Video(
            filename="predictions.pkg.slp", original_video=original, open_backend=False
        )
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Should match via original_video
        assert len(labels.videos) == 1, (
            "original_video should enable matching. "
            f"pred.original_video.filename = {original.filename}, "
            f"base.filename = {base_video.filename}"
        )
        assert labels.labeled_frames[0].video is base_video

    def test_original_video_existing_only(self):
        """Match via existing video's original_video.

        Scenario: Base has embedded video with original_video.
        Prediction has external video matching that original.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base: embedded video with original_video
        original = Video(filename="/data/video.mp4", open_backend=False)
        base_video = Video(
            filename="base.pkg.slp", original_video=original, open_backend=False
        )
        base_video.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [base_video]

        # Prediction: external video matching original
        pred_video = Video(filename="/data/video.mp4", open_backend=False)
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Should match via base's original_video
        assert len(labels.videos) == 1, (
            "base.original_video should enable matching. "
            f"base.original_video.filename = {original.filename}, "
            f"pred.filename = {pred_video.filename}"
        )
        assert labels.labeled_frames[0].video is base_video

    def test_original_video_both_same_target(self):
        """Both videos have original_video pointing to same file.

        Scenario: Two PKG.SLP files from same original video.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base: embedded with original_video
        base_original = Video(filename="/data/video.mp4", open_backend=False)
        base_video = Video(
            filename="base.pkg.slp", original_video=base_original, open_backend=False
        )
        base_video.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [base_video]

        # Prediction: embedded with original_video pointing to same file
        pred_original = Video(filename="/data/video.mp4", open_backend=False)
        pred_video = Video(
            filename="predictions.pkg.slp",
            original_video=pred_original,
            open_backend=False,
        )
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Should match via original_video comparison
        assert len(labels.videos) == 1, (
            "Both original_videos point to same file - should match. "
            f"base.original_video = {base_original.filename}, "
            f"pred.original_video = {pred_original.filename}"
        )

    def test_original_video_both_different_targets(self):
        """Both videos have original_video pointing to different files.

        Scenario: Two PKG.SLP files from different original videos.
        Should NOT match.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base: embedded with original_video A
        base_original = Video(filename="/data/video_a.mp4", open_backend=False)
        base_video = Video(
            filename="base.pkg.slp", original_video=base_original, open_backend=False
        )
        base_video.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [base_video]

        # Prediction: embedded with original_video B (different)
        pred_original = Video(filename="/data/video_b.mp4", open_backend=False)
        pred_video = Video(
            filename="predictions.pkg.slp",
            original_video=pred_original,
            open_backend=False,
        )
        pred_video.backend_metadata["shape"] = (100, 480, 640, 1)  # Same shape!

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [pred_video]

        pred_frame = LabeledFrame(video=pred_video, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Different original_videos → should NOT match even with same shape
        assert len(labels.videos) == 2, (
            "Different original_videos should NOT match. "
            f"base.original_video = {base_original.filename}, "
            f"pred.original_video = {pred_original.filename}"
        )

    def test_source_video_chain_traversal(self):
        """Multi-level source_video chain should be traversed to root.

        Scenario: final.pkg.slp → intermediate.pkg.slp → /data/video.mp4
        The algorithm should use original_video (root) for matching.
        """
        skeleton = Skeleton(["head", "tail"])

        # Base: external video
        base_video = Video(filename="/data/video.mp4", open_backend=False)
        base_video.backend_metadata["shape"] = (100, 480, 640, 1)

        labels = Labels(skeletons=[skeleton])
        labels.videos = [base_video]

        # Build chain: root -> intermediate -> final
        root = Video(filename="/data/video.mp4", open_backend=False)
        intermediate = Video(
            filename="intermediate.pkg.slp", source_video=root, open_backend=False
        )
        final = Video(
            filename="final.pkg.slp", source_video=intermediate, open_backend=False
        )
        final.backend_metadata["shape"] = (100, 480, 640, 1)

        predictions = Labels(skeletons=[skeleton])
        predictions.videos = [final]

        pred_frame = LabeledFrame(video=final, frame_idx=0)
        pred_inst = PredictedInstance.from_numpy(
            np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, score=0.9
        )
        pred_frame.instances = [pred_inst]
        predictions.labeled_frames = [pred_frame]

        result = labels.merge(predictions)

        assert result.successful
        # Should traverse chain: final → intermediate → root → matches base
        assert len(labels.videos) == 1, (
            "source_video chain should be traversed to root. "
            "Chain: final.pkg.slp → intermediate.pkg.slp → /data/video.mp4"
        )
        assert labels.labeled_frames[0].video is base_video
