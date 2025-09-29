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
        merged, conflicts = frame1.merge(frame2, strategy="keep_original")

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
        merged, conflicts = frame1.merge(frame2, strategy="keep_new")

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
        merged, conflicts = frame1.merge(frame2, strategy="update_tracks")

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
        merged, conflicts = frame1.merge(frame2, strategy="keep_both")

        assert len(merged) == 2
        assert inst1 in merged
        assert inst2 in merged
        assert len(conflicts) == 0

    def test_merge_smart_user_vs_predicted(self):
        """Test smart merge with user labels vs predictions."""
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

        # Merge with smart strategy
        merged, conflicts = frame1.merge(frame2, strategy="smart")

        # User instance should be kept, prediction ignored
        assert len(merged) == 1
        assert merged[0] is user_inst
        assert len(conflicts) == 1
        assert conflicts[0][2] == "kept_user"

    def test_merge_smart_replace_prediction(self):
        """Test smart merge replacing prediction with user label."""
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

        # Merge with smart strategy
        merged, conflicts = frame2.merge(frame1, strategy="smart")

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
        merged, conflicts = frame1.merge(
            frame2, instance_matcher=matcher, strategy="smart"
        )

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
        result = labels1.merge(labels2, frame_strategy="keep_both")

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
