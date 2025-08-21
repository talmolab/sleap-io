"""Tests for the matching module."""

import numpy as np
import pytest

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.matching import (
    ConflictResolution,
    ErrorMode,
    FrameStrategy,
    InstanceMatcher,
    InstanceMatchMethod,
    MergeError,
    MergeResult,
    SkeletonMatcher,
    SkeletonMatchMethod,
    SkeletonMismatchError,
    TrackMatcher,
    TrackMatchMethod,
    VideoMatcher,
    VideoMatchMethod,
    VideoNotFoundError,
)
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


class TestSkeletonMatcher:
    """Test skeleton matching functionality."""

    def test_exact_match(self):
        """Test exact skeleton matching."""
        skel1 = Skeleton(
            nodes=["head", "thorax", "abdomen"],
            edges=[("head", "thorax"), ("thorax", "abdomen")],
        )
        skel2 = Skeleton(
            nodes=["head", "thorax", "abdomen"],
            edges=[("head", "thorax"), ("thorax", "abdomen")],
        )
        skel3 = Skeleton(
            nodes=["abdomen", "thorax", "head"],  # Different order
            edges=[("head", "thorax"), ("thorax", "abdomen")],
        )

        matcher = SkeletonMatcher(method=SkeletonMatchMethod.EXACT)
        assert matcher.match(skel1, skel2)
        assert not matcher.match(skel1, skel3)  # Different order

    def test_structure_match(self):
        """Test structure skeleton matching."""
        skel1 = Skeleton(
            nodes=["head", "thorax", "abdomen"],
            edges=[("head", "thorax"), ("thorax", "abdomen")],
        )
        skel2 = Skeleton(
            nodes=["abdomen", "thorax", "head"],  # Different order
            edges=[("head", "thorax"), ("thorax", "abdomen")],
        )
        skel3 = Skeleton(
            nodes=["head", "thorax", "tail"],  # Different node
            edges=[("head", "thorax"), ("thorax", "tail")],
        )

        matcher = SkeletonMatcher(method=SkeletonMatchMethod.STRUCTURE)
        assert matcher.match(skel1, skel2)  # Same structure, different order
        assert not matcher.match(skel1, skel3)  # Different nodes

    def test_overlap_match(self):
        """Test overlap skeleton matching."""
        skel1 = Skeleton(nodes=["head", "thorax", "abdomen"])
        skel2 = Skeleton(nodes=["head", "thorax", "tail"])  # 2/3 overlap
        skel3 = Skeleton(nodes=["wing1", "wing2", "tail"])  # 0/3 overlap

        matcher = SkeletonMatcher(method=SkeletonMatchMethod.OVERLAP, min_overlap=0.5)
        assert matcher.match(skel1, skel2)  # 66% overlap > 50%
        assert not matcher.match(skel1, skel3)  # 0% overlap < 50%

    def test_subset_match(self):
        """Test subset skeleton matching."""
        skel1 = Skeleton(nodes=["head", "thorax"])
        skel2 = Skeleton(nodes=["head", "thorax", "abdomen", "tail"])
        skel3 = Skeleton(nodes=["head", "wing"])

        matcher = SkeletonMatcher(method=SkeletonMatchMethod.SUBSET)
        assert matcher.match(skel1, skel2)  # skel1 is subset of skel2
        assert not matcher.match(skel1, skel3)  # skel1 is not subset of skel3
        assert not matcher.match(skel2, skel1)  # skel2 is not subset of skel1


class TestInstanceMatcher:
    """Test instance matching functionality."""

    def test_spatial_match(self):
        """Test spatial instance matching."""
        skeleton = Skeleton(nodes=["head", "tail"])

        # Create instances with known positions
        inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)
        inst2 = Instance.from_numpy(
            np.array([[11, 11], [21, 21]]),
            skeleton=skeleton,  # Close to inst1
        )
        inst3 = Instance.from_numpy(
            np.array([[50, 50], [60, 60]]),
            skeleton=skeleton,  # Far from inst1
        )

        matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)
        assert matcher.match(inst1, inst2)  # Within threshold
        assert not matcher.match(inst1, inst3)  # Outside threshold

    def test_identity_match(self):
        """Test identity instance matching."""
        skeleton = Skeleton(nodes=["head", "tail"])
        track1 = Track(name="track1")
        track2 = Track(name="track2")

        inst1 = Instance.from_numpy(
            np.array([[10, 10], [20, 20]]), skeleton=skeleton, track=track1
        )
        inst2 = Instance.from_numpy(
            np.array([[50, 50], [60, 60]]), skeleton=skeleton, track=track1
        )
        inst3 = Instance.from_numpy(
            np.array([[10, 10], [20, 20]]), skeleton=skeleton, track=track2
        )

        matcher = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)
        assert matcher.match(inst1, inst2)  # Same track
        assert not matcher.match(inst1, inst3)  # Different tracks

    def test_iou_match(self):
        """Test IoU instance matching."""
        skeleton = Skeleton(nodes=["p1", "p2", "p3", "p4"])

        # Create instances with overlapping bounding boxes
        inst1 = Instance.from_numpy(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), skeleton=skeleton
        )
        inst2 = Instance.from_numpy(
            np.array([[5, 5], [15, 5], [15, 15], [5, 15]]), skeleton=skeleton
        )
        inst3 = Instance.from_numpy(
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]), skeleton=skeleton
        )

        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.1)
        assert matcher.match(inst1, inst2)  # Overlapping
        assert not matcher.match(inst1, inst3)  # Not overlapping

    def test_find_matches(self):
        """Test finding matches between instance lists."""
        skeleton = Skeleton(nodes=["head", "tail"])

        instances1 = [
            Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton),
            Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton),
        ]
        instances2 = [
            Instance.from_numpy(np.array([[11, 11], [21, 21]]), skeleton=skeleton),
            Instance.from_numpy(np.array([[50, 50], [60, 60]]), skeleton=skeleton),
        ]

        matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)
        matches = matcher.find_matches(instances1, instances2)

        assert len(matches) == 1  # Only first instances match
        assert matches[0][0] == 0  # First instance in list1
        assert matches[0][1] == 0  # First instance in list2
        assert matches[0][2] > 0  # Has a score


class TestTrackMatcher:
    """Test track matching functionality."""

    def test_name_match(self):
        """Test track matching by name."""
        track1 = Track(name="mouse1")
        track2 = Track(name="mouse1")  # Same name, different object
        track3 = Track(name="mouse2")

        matcher = TrackMatcher(method=TrackMatchMethod.NAME)
        assert matcher.match(track1, track2)  # Same name
        assert not matcher.match(track1, track3)  # Different names

    def test_identity_match(self):
        """Test track matching by identity."""
        track1 = Track(name="mouse1")
        track2 = Track(name="mouse1")  # Same name, different object
        track3 = track1  # Same object

        matcher = TrackMatcher(method=TrackMatchMethod.IDENTITY)
        assert not matcher.match(track1, track2)  # Different objects
        assert matcher.match(track1, track3)  # Same object


class TestVideoMatcher:
    """Test video matching functionality."""

    def test_path_match(self):
        """Test video matching by path."""
        video1 = Video(filename="/path/to/video.mp4", open_backend=False)
        video2 = Video(filename="/path/to/video.mp4", open_backend=False)
        video3 = Video(filename="/other/path/video.mp4", open_backend=False)

        matcher = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
        assert matcher.match(video1, video2)  # Same path
        assert not matcher.match(video1, video3)  # Different path

    def test_basename_match(self):
        """Test video matching by basename."""
        video1 = Video(filename="/path/to/video.mp4", open_backend=False)
        video2 = Video(filename="/other/path/video.mp4", open_backend=False)
        video3 = Video(filename="/path/to/other.mp4", open_backend=False)

        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)
        assert matcher.match(video1, video2)  # Same basename
        assert not matcher.match(video1, video3)  # Different basename

    def test_auto_match(self):
        """Test automatic video matching."""
        video1 = Video(filename="/path/to/video.mp4", open_backend=False)
        video2 = video1  # Same object
        video3 = Video(filename="/path/to/video.mp4", open_backend=False)
        video4 = Video(filename="/other/path/video.mp4", open_backend=False)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.match(video1, video2)  # Same object
        assert matcher.match(video1, video3)  # Same path
        assert matcher.match(video1, video4)  # Same basename


class TestMergeResult:
    """Test merge result functionality."""

    def test_successful_merge(self):
        """Test successful merge result."""
        result = MergeResult(
            successful=True,
            frames_merged=10,
            instances_added=50,
            instances_updated=5,
            instances_skipped=2,
        )

        summary = result.summary()
        assert "✓ Merge completed successfully" in summary
        assert "Frames merged: 10" in summary
        assert "Instances added: 50" in summary
        assert "Instances updated: 5" in summary
        assert "Instances skipped: 2" in summary

    def test_failed_merge(self):
        """Test failed merge result."""
        error1 = MergeError(message="Error 1")
        error2 = SkeletonMismatchError(message="Skeleton mismatch")

        result = MergeResult(
            successful=False,
            frames_merged=5,
            instances_added=20,
            errors=[error1, error2],
        )

        summary = result.summary()
        assert "✗ Merge completed with errors" in summary
        assert "Errors encountered: 2" in summary
        assert "Error 1" in summary
        assert "Skeleton mismatch" in summary

    def test_merge_with_conflicts(self):
        """Test merge result with conflicts."""
        video = Video(filename="test.mp4", open_backend=False)
        frame = LabeledFrame(video=video, frame_idx=0)

        conflict = ConflictResolution(
            frame=frame,
            conflict_type="duplicate_instance",
            original_data="original",
            new_data="new",
            resolution="kept_original",
        )

        result = MergeResult(
            successful=True,
            frames_merged=1,
            conflicts=[conflict],
        )

        summary = result.summary()
        assert "Conflicts resolved: 1" in summary


class TestEnums:
    """Test enum functionality."""

    def test_skeleton_match_method(self):
        """Test SkeletonMatchMethod enum."""
        assert SkeletonMatchMethod.EXACT == "exact"
        assert SkeletonMatchMethod.STRUCTURE == "structure"
        assert SkeletonMatchMethod.OVERLAP == "overlap"
        assert SkeletonMatchMethod.SUBSET == "subset"

    def test_instance_match_method(self):
        """Test InstanceMatchMethod enum."""
        assert InstanceMatchMethod.SPATIAL == "spatial"
        assert InstanceMatchMethod.IDENTITY == "identity"
        assert InstanceMatchMethod.IOU == "iou"

    def test_frame_strategy(self):
        """Test FrameStrategy enum."""
        assert FrameStrategy.SMART == "smart"
        assert FrameStrategy.KEEP_ORIGINAL == "keep_original"
        assert FrameStrategy.KEEP_NEW == "keep_new"
        assert FrameStrategy.KEEP_BOTH == "keep_both"

    def test_error_mode(self):
        """Test ErrorMode enum."""
        assert ErrorMode.CONTINUE == "continue"
        assert ErrorMode.STRICT == "strict"
        assert ErrorMode.WARN == "warn"


class TestMergeErrors:
    """Test merge error classes."""

    def test_merge_error(self):
        """Test MergeError class."""
        error = MergeError(message="Test error", details={"key": "value"})
        assert error.message == "Test error"
        assert error.details == {"key": "value"}

    def test_skeleton_mismatch_error(self):
        """Test SkeletonMismatchError class."""
        error = SkeletonMismatchError(
            message="Skeletons don't match",
            details={"skeleton1": "skel1", "skeleton2": "skel2"},
        )
        assert isinstance(error, MergeError)
        assert error.message == "Skeletons don't match"

    def test_video_not_found_error(self):
        """Test VideoNotFoundError class."""
        error = VideoNotFoundError(
            message="Video not found",
            details={"path": "/missing/video.mp4"},
        )
        assert isinstance(error, MergeError)
        assert error.message == "Video not found"


class TestEdgeCases:
    """Test edge cases and error handling in matching."""

    def test_video_matcher_content_method(self):
        """Test VideoMatcher with CONTENT method."""
        video1 = Video(filename="test1.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)
        
        video2 = Video(filename="test2.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)
        
        video3 = Video(filename="test3.mp4", open_backend=False)
        video3.backend_metadata["shape"] = (50, 480, 640, 3)
        
        matcher = VideoMatcher(method=VideoMatchMethod.CONTENT)
        assert matcher.match(video1, video2)  # Same content
        assert not matcher.match(video1, video3)  # Different content

    def test_video_matcher_resolve_method(self):
        """Test VideoMatcher with RESOLVE method."""
        video1 = Video(filename="/path/to/video.mp4", open_backend=False)
        video2 = Video(filename="/other/path/video.mp4", open_backend=False)
        video3 = Video(filename="/path/to/different.mp4", open_backend=False)
        
        matcher = VideoMatcher(method=VideoMatchMethod.RESOLVE)
        assert matcher.match(video1, video2)  # Same basename
        assert not matcher.match(video1, video3)  # Different basename

    def test_video_matcher_auto_content_fallback(self):
        """Test VideoMatcher AUTO method falling back to content matching."""
        video1 = Video(filename="video1.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)
        
        video2 = Video(filename="video2.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)
        
        video3 = Video(filename="video3.mp4", open_backend=False)
        video3.backend_metadata["shape"] = (50, 240, 320, 3)  # Different shape
        
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        # Should match by content since paths don't match
        assert matcher.match(video1, video2)
        
        # Should not match - different paths and different content
        assert not matcher.match(video1, video3)

    def test_frame_matcher(self):
        """Test FrameMatcher functionality."""
        from sleap_io.model.matching import FrameMatcher
        
        video1 = Video(filename="test1.mp4", open_backend=False)
        video2 = Video(filename="test2.mp4", open_backend=False)
        
        frame1 = LabeledFrame(video=video1, frame_idx=0)
        frame2 = LabeledFrame(video=video1, frame_idx=0)
        frame3 = LabeledFrame(video=video2, frame_idx=0)
        
        # Test with video must match
        matcher = FrameMatcher(video_must_match=True)
        assert matcher.match(frame1, frame2)  # Same video
        assert not matcher.match(frame1, frame3)  # Different videos
        
        # Test without video must match
        matcher = FrameMatcher(video_must_match=False)
        assert matcher.match(frame1, frame2)
        assert matcher.match(frame1, frame3)  # Videos don't need to match

    def test_instance_matcher_iou_with_overlap(self):
        """Test InstanceMatcher IoU calculation with actual overlapping boxes."""
        skeleton = Skeleton(nodes=["tl", "tr", "br", "bl"])  # top-left, top-right, bottom-right, bottom-left
        
        # Create box 1: (0, 0) to (10, 10)
        inst1 = Instance.from_numpy(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), 
            skeleton=skeleton
        )
        
        # Create box 2: (5, 5) to (15, 15) - overlaps with box 1
        inst2 = Instance.from_numpy(
            np.array([[5, 5], [15, 5], [15, 15], [5, 15]]), 
            skeleton=skeleton
        )
        
        # Create box 3: (20, 20) to (30, 30) - no overlap
        inst3 = Instance.from_numpy(
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]), 
            skeleton=skeleton
        )
        
        # Create box 4: (0, 0) to (10, 10) - identical to box 1
        inst4 = Instance.from_numpy(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), 
            skeleton=skeleton
        )
        
        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.0)
        
        # Test find_matches to trigger IoU score calculation
        matches = matcher.find_matches([inst1], [inst2])
        assert len(matches) == 1
        assert matches[0][0] == 0
        assert matches[0][1] == 0
        # IoU should be 25/175 ≈ 0.143
        assert 0.14 < matches[0][2] < 0.15
        
        # Test with no overlap
        matches = matcher.find_matches([inst1], [inst3])
        assert len(matches) == 0  # No overlap, no match
        
        # Test with identical boxes
        matches = matcher.find_matches([inst1], [inst4])
        assert len(matches) == 1
        assert matches[0][2] == 1.0  # Perfect overlap

    def test_instance_matcher_iou_edge_cases(self):
        """Test InstanceMatcher IoU calculation edge cases."""
        skeleton = Skeleton(nodes=["p1", "p2"])
        
        # Instance with missing points (NaN values)
        inst1 = Instance.from_numpy(
            np.array([[10, 10], [np.nan, np.nan]]), 
            skeleton=skeleton
        )
        
        inst2 = Instance.from_numpy(
            np.array([[11, 11], [21, 21]]), 
            skeleton=skeleton
        )
        
        # Instance with all NaN points (no valid bounding box)
        inst3 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), 
            skeleton=skeleton
        )
        
        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.0)
        
        # Test with instance that has NaN points
        matches = matcher.find_matches([inst1], [inst2])
        # Should handle NaN gracefully
        assert len(matches) == 0 or matches[0][2] == 0.0
        
        # Test with instance that has no valid bounding box
        matches = matcher.find_matches([inst3], [inst2])
        assert len(matches) == 0  # No valid bounding box, no match
        
        # Test both instances with no valid bounding box
        matches = matcher.find_matches([inst3], [inst3])
        assert len(matches) == 0  # Neither has valid bounding box

    def test_merge_result_many_errors(self):
        """Test MergeResult summary with more than 5 errors."""
        errors = [
            MergeError(message=f"Error {i}") 
            for i in range(10)
        ]
        
        result = MergeResult(
            successful=False,
            errors=errors
        )
        
        summary = result.summary()
        assert "Errors encountered: 10" in summary
        assert "Error 0" in summary
        assert "Error 4" in summary
        assert "... and 5 more" in summary
        assert "Error 5" not in summary  # Should be truncated
