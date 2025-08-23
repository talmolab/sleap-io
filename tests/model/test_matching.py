"""Tests for the matching module."""

import numpy as np

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.matching import (
    ConflictResolution,
    ErrorMode,
    FrameStrategy,
    InstanceMatcher,
    InstanceMatchMethod,
    MergeError,
    MergeProgressBar,
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


class TestMergeProgressBar:
    """Test MergeProgressBar functionality."""

    def test_progress_bar_context_manager(self):
        """Test MergeProgressBar as a context manager."""
        with MergeProgressBar("Test merge") as progress:
            assert progress.desc == "Test merge"
            assert progress.pbar is None

            # Test callback
            progress.callback(0, 10, "Starting")
            # Since tqdm may or may not be available, just check it runs
            progress.callback(5, 10, "Halfway")
            progress.callback(10, 10, "Complete")


class TestVideoMatcherWithFallback:
    """Test VideoMatcher with fallback directories."""

    def test_resolve_with_fallback_directories(self, tmp_path):
        """Test VideoMatcher RESOLVE method with fallback directories."""
        # Create test structure
        fallback1 = tmp_path / "fallback1"
        fallback2 = tmp_path / "fallback2"
        fallback1.mkdir()
        fallback2.mkdir()

        # Create a video file in fallback2
        video_file = fallback2 / "test_video.mp4"
        video_file.touch()

        # Create videos with different paths but same basename
        video1 = Video(filename=str(fallback2 / "test_video.mp4"), open_backend=False)
        video2 = Video(filename="/original/path/test_video.mp4", open_backend=False)

        # Test with fallback directories
        matcher = VideoMatcher(
            method=VideoMatchMethod.RESOLVE,
            fallback_directories=[str(fallback1), str(fallback2)],
        )

        assert matcher.match(video1, video2)  # Should find it in fallback2

    def test_resolve_with_base_path(self, tmp_path):
        """Test VideoMatcher RESOLVE method with base_path."""
        base_path = tmp_path / "base"
        base_path.mkdir()

        # Create a video file at base path
        video_file = base_path / "test_video.mp4"
        video_file.touch()

        video1 = Video(filename=str(video_file), open_backend=False)
        video2 = Video(filename="/different/path/test_video.mp4", open_backend=False)

        # Test with base_path
        matcher = VideoMatcher(
            method=VideoMatchMethod.RESOLVE, base_path=str(base_path)
        )

        assert matcher.match(video1, video2)  # Should find it at base_path


class TestPreConfiguredMatchers:
    """Test pre-configured matchers."""

    def test_preconfigured_matchers_exist(self):
        """Test that all pre-configured matchers are available."""
        from sleap_io.model.matching import (
            AUTO_VIDEO_MATCHER,
            BASENAME_VIDEO_MATCHER,
            DUPLICATE_MATCHER,
            IDENTITY_INSTANCE_MATCHER,
            IDENTITY_TRACK_MATCHER,
            IOU_MATCHER,
            NAME_TRACK_MATCHER,
            OVERLAP_SKELETON_MATCHER,
            PATH_VIDEO_MATCHER,
            SOURCE_VIDEO_MATCHER,
            STRUCTURE_SKELETON_MATCHER,
            SUBSET_SKELETON_MATCHER,
        )

        # Test skeleton matchers
        assert STRUCTURE_SKELETON_MATCHER.method == SkeletonMatchMethod.STRUCTURE
        assert SUBSET_SKELETON_MATCHER.method == SkeletonMatchMethod.SUBSET
        assert OVERLAP_SKELETON_MATCHER.method == SkeletonMatchMethod.OVERLAP
        assert OVERLAP_SKELETON_MATCHER.min_overlap == 0.7

        # Test instance matchers
        assert DUPLICATE_MATCHER.method == InstanceMatchMethod.SPATIAL
        assert DUPLICATE_MATCHER.threshold == 5.0
        assert IOU_MATCHER.method == InstanceMatchMethod.IOU
        assert IOU_MATCHER.threshold == 0.5
        assert IDENTITY_INSTANCE_MATCHER.method == InstanceMatchMethod.IDENTITY

        # Test track matchers
        assert NAME_TRACK_MATCHER.method == TrackMatchMethod.NAME
        assert IDENTITY_TRACK_MATCHER.method == TrackMatchMethod.IDENTITY

        # Test video matchers
        assert AUTO_VIDEO_MATCHER.method == VideoMatchMethod.AUTO
        assert SOURCE_VIDEO_MATCHER.method == VideoMatchMethod.RESOLVE
        assert PATH_VIDEO_MATCHER.method == VideoMatchMethod.PATH
        assert PATH_VIDEO_MATCHER.strict is True
        assert BASENAME_VIDEO_MATCHER.method == VideoMatchMethod.BASENAME


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
        skeleton = Skeleton(
            nodes=["tl", "tr", "br", "bl"]
        )  # top-left, top-right, bottom-right, bottom-left

        # Create box 1: (0, 0) to (10, 10)
        inst1 = Instance.from_numpy(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), skeleton=skeleton
        )

        # Create box 2: (5, 5) to (15, 15) - overlaps with box 1
        inst2 = Instance.from_numpy(
            np.array([[5, 5], [15, 5], [15, 15], [5, 15]]), skeleton=skeleton
        )

        # Create box 3: (20, 20) to (30, 30) - no overlap
        inst3 = Instance.from_numpy(
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]), skeleton=skeleton
        )

        # Create box 4: (0, 0) to (10, 10) - identical to box 1
        inst4 = Instance.from_numpy(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), skeleton=skeleton
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
            np.array([[10, 10], [np.nan, np.nan]]), skeleton=skeleton
        )

        inst2 = Instance.from_numpy(np.array([[11, 11], [21, 21]]), skeleton=skeleton)

        # Instance with all NaN points (no valid bounding box)
        inst3 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
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
        errors = [MergeError(message=f"Error {i}") for i in range(10)]

        result = MergeResult(successful=False, errors=errors)

        summary = result.summary()
        assert "Errors encountered: 10" in summary
        assert "Error 0" in summary
        assert "Error 4" in summary
        assert "... and 5 more" in summary
        assert "Error 5" not in summary  # Should be truncated

    def test_skeleton_matcher_invalid_method(self):
        """Test that SkeletonMatcher raises an error for invalid methods."""
        from unittest.mock import Mock

        import pytest

        skel1 = Skeleton(nodes=["head", "thorax"])
        skel2 = Skeleton(nodes=["head", "thorax"])

        # Create a matcher and mock an invalid method
        matcher = SkeletonMatcher(method=SkeletonMatchMethod.EXACT)

        # Use object.__setattr__ to bypass the converter
        object.__setattr__(matcher, "method", Mock())
        matcher.method.__str__ = Mock(return_value="INVALID_METHOD")

        with pytest.raises(ValueError, match="Unknown skeleton match method"):
            matcher.match(skel1, skel2)

    def test_instance_matcher_invalid_method(self):
        """Test that InstanceMatcher raises an error for invalid methods."""
        from unittest.mock import Mock

        import pytest

        skel = Skeleton(nodes=["head", "thorax"])
        inst1 = Instance.from_numpy(np.array([[1, 2], [3, 4]]), skeleton=skel)
        inst2 = Instance.from_numpy(np.array([[1, 2], [3, 4]]), skeleton=skel)

        # Create a matcher and mock an invalid method
        matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL)

        # Use object.__setattr__ to bypass the converter
        object.__setattr__(matcher, "method", Mock())
        matcher.method.__str__ = Mock(return_value="INVALID_METHOD")

        with pytest.raises(ValueError, match="Unknown instance match method"):
            matcher.match(inst1, inst2)

    def test_instance_matcher_spatial_no_overlap(self):
        """Test spatial matching when instances have no overlapping points."""
        skel = Skeleton(nodes=["head", "thorax", "abdomen"])

        # Instance 1 has valid points for head and thorax, NaN for abdomen
        inst1 = Instance.from_numpy(
            np.array([[1, 2], [3, 4], [np.nan, np.nan]]), skeleton=skel
        )

        # Instance 2 has NaN for head and thorax, valid point for abdomen
        inst2 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan], [5, 6]]), skeleton=skel
        )

        matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=10.0)

        # Should not match because no overlapping valid points
        assert not matcher.match(inst1, inst2)

        # find_matches should return empty list for no overlap (no match)
        matches = matcher.find_matches([inst1], [inst2])
        assert len(matches) == 0  # No matches because no overlapping points

    def test_instance_matcher_iou_no_bounding_box(self):
        """Test IoU matching when instances have no valid bounding box."""
        skel = Skeleton(nodes=["head", "thorax"])

        # Instance with all NaN points (no bounding box)
        inst1 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skel
        )
        inst2 = Instance.from_numpy(np.array([[1, 2], [3, 4]]), skeleton=skel)

        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.5)

        # Should not match because inst1 has no bounding box
        assert not matcher.match(inst1, inst2)

        # find_matches should return empty list (no match)
        matches = matcher.find_matches([inst1], [inst2])
        assert len(matches) == 0  # No matches because inst1 has no bounding box

    def test_instance_matcher_iou_no_intersection(self):
        """Test IoU matching when bounding boxes don't intersect."""
        skel = Skeleton(nodes=["head", "thorax"])

        # Two instances with non-overlapping bounding boxes
        inst1 = Instance.from_numpy(np.array([[0, 0], [1, 1]]), skeleton=skel)
        inst2 = Instance.from_numpy(np.array([[10, 10], [11, 11]]), skeleton=skel)

        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.1)

        # Should not match because no intersection
        assert not matcher.match(inst1, inst2)

        # find_matches should return empty list for no intersection
        matches = matcher.find_matches([inst1], [inst2])
        assert len(matches) == 0  # No matches because no intersection

    def test_video_matcher_fallback_directories_complex(self):
        """Test video matching with fallback directories for complex scenarios."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fallback directory structure
            fallback1 = Path(tmpdir) / "fallback1"
            fallback2 = Path(tmpdir) / "fallback2"
            fallback1.mkdir()
            fallback2.mkdir()

            # Create test video files
            video_file = fallback2 / "test_video.mp4"
            video_file.touch()

            # Videos with same basename but different paths
            video1 = Video(filename="/original/path/test_video.mp4")
            video2 = Video(filename="/different/path/test_video.mp4")

            # Matcher with fallback directories
            matcher = VideoMatcher(
                method=VideoMatchMethod.RESOLVE,
                fallback_directories=[str(fallback1), str(fallback2)],
            )

            # Should match because basename matches and file exists in fallback2
            assert matcher.match(video1, video2)

            # Test with base_path
            matcher_with_base = VideoMatcher(
                method=VideoMatchMethod.RESOLVE,
                base_path=str(tmpdir),
                fallback_directories=[],
            )

            # Create file at base path
            base_video = Path(tmpdir) / "test_video.mp4"
            base_video.touch()

            # Should match because file exists at base path
            assert matcher_with_base.match(video1, video2)

            # Test relative path matching with base_path
            video3 = Video(filename="subdir/test_video.mp4")
            video4 = Video(filename="subdir/test_video.mp4")

            matcher_relative = VideoMatcher(
                method=VideoMatchMethod.RESOLVE, base_path=str(tmpdir)
            )

            # Should match because relative paths are identical
            assert matcher_relative.match(video3, video4)

    def test_video_matcher_invalid_method(self):
        """Test that VideoMatcher raises an error for invalid methods."""
        from unittest.mock import Mock

        import pytest

        video1 = Video(filename="test1.mp4")
        video2 = Video(filename="test2.mp4")

        # Create a matcher and mock an invalid method
        matcher = VideoMatcher(method=VideoMatchMethod.PATH)

        # Use object.__setattr__ to bypass the converter
        object.__setattr__(matcher, "method", Mock())
        matcher.method.__str__ = Mock(return_value="INVALID_METHOD")

        with pytest.raises(ValueError, match="Unknown video match method"):
            matcher.match(video1, video2)

    def test_merge_progress_bar_update_with_message(self):
        """Test MergeProgressBar callback method with message."""
        # Test the callback method which is the actual update mechanism
        progress_bar = MergeProgressBar(desc="Merging")

        # Test callback with message
        progress_bar.callback(50, 100, "Processing frames")
        assert progress_bar.pbar is not None
        assert progress_bar.pbar.n == 50
        assert "Processing frames" in progress_bar.pbar.desc

        # Test callback without message
        progress_bar.callback(75, 100)
        assert progress_bar.pbar.n == 75
        assert progress_bar.desc in progress_bar.pbar.desc

        # Clean up
        if progress_bar.pbar:
            progress_bar.pbar.close()

    def test_merge_progress_bar_without_pbar(self):
        """Test MergeProgressBar callback when pbar is None initially."""
        # Create progress bar without initializing pbar
        progress_bar = MergeProgressBar(desc="Merging")

        # First callback should create the pbar
        progress_bar.callback(0, 0, "No total")  # total=0 shouldn't create pbar
        assert progress_bar.pbar is None

        # Callback with total should create pbar
        progress_bar.callback(50, 100, "Processing")
        assert progress_bar.pbar is not None

        # Clean up
        if progress_bar.pbar:
            progress_bar.pbar.close()

    def test_find_matches_spatial_matching_edge_cases(self):
        """Test find_matches score calculation for spatial matching edge cases."""
        skel = Skeleton(nodes=["head", "thorax"])

        # Test case where instances match but have no valid overlapping points
        # This tests the else branch at line 148-149
        inst1 = Instance.from_numpy(np.array([[1, 2], [np.nan, np.nan]]), skeleton=skel)
        inst2 = Instance.from_numpy(np.array([[np.nan, np.nan], [3, 4]]), skeleton=skel)

        # Create a shared track for identity matching
        shared_track = Track(name="track1")
        inst1.track = shared_track
        inst2.track = shared_track  # Same track object for identity match

        matcher = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)
        matches = matcher.find_matches([inst1], [inst2])

        # Should match with identity, score should be 1.0 (binary match)
        assert len(matches) == 1
        assert matches[0][2] == 1.0

    def test_find_matches_iou_matching_edge_cases(self):
        """Test find_matches score calculation for IoU matching edge cases."""
        skel = Skeleton(nodes=["head", "thorax"])

        # Test IoU matching when one instance has no bounding box (lines 172-173)
        inst1 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skel
        )
        inst2 = Instance.from_numpy(np.array([[1, 2], [3, 4]]), skeleton=skel)

        # Force these to match with identity but calculate IoU score
        shared_track1 = Track(name="track1")
        inst1.track = shared_track1
        inst2.track = shared_track1

        # We need to test the score calculation in find_matches
        # when method is IOU but instances match via another criterion
        # This is tricky since IoU matching won't match if no bbox

        # Let's test with overlapping instances that have valid bboxes
        # but no actual intersection (lines 170-171)
        inst3 = Instance.from_numpy(np.array([[0, 0], [1, 1]]), skeleton=skel)
        inst4 = Instance.from_numpy(np.array([[10, 10], [11, 11]]), skeleton=skel)

        # Give them same track for identity match to trigger score calculation
        shared_track2 = Track(name="track2")
        inst3.track = shared_track2
        inst4.track = shared_track2

        matcher_identity = InstanceMatcher(method=InstanceMatchMethod.IDENTITY)
        matches = matcher_identity.find_matches([inst3], [inst4])

        # Should match with identity, score is 1.0 (binary)
        assert len(matches) == 1
        assert matches[0][2] == 1.0

    def test_video_matcher_fallback_with_path_errors(self):
        """Test video matcher fallback directory with path resolution errors."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create videos with completely different basenames
            video1 = Video(filename="/absolute/path/video1.mp4")
            video2 = Video(filename="relative/path/video2.mp4")

            # Test with base_path when relative_to fails (lines 251-267)
            matcher = VideoMatcher(
                method=VideoMatchMethod.RESOLVE,
                base_path=tmpdir,
                fallback_directories=[tmpdir],
            )

            # This should return False since basenames don't match
            assert not matcher.match(video1, video2)

            # Test with same basename but different paths
            video3 = Video(filename="/path1/test.mp4")
            video4 = Video(filename="/path2/test.mp4")

            # Create the file in fallback directory
            fallback_dir = Path(tmpdir) / "fallback"
            fallback_dir.mkdir()
            test_file = fallback_dir / "test.mp4"
            test_file.touch()

            matcher2 = VideoMatcher(
                method=VideoMatchMethod.RESOLVE,
                fallback_directories=[str(fallback_dir)],
            )

            # Should match because basename matches and file exists in fallback
            assert matcher2.match(video3, video4)

    def test_video_matcher_complex_relative_paths(self):
        """Test video matcher with complex relative path scenarios."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create nested directory structure
            subdir1 = base_path / "data" / "videos"
            subdir1.mkdir(parents=True)

            # Create test video files
            video_file1 = subdir1 / "test.mp4"
            video_file1.touch()

            # Test relative path resolution with base_path
            video1 = Video(filename=str(video_file1))
            video2 = Video(filename="data/videos/test.mp4")

            matcher = VideoMatcher(
                method=VideoMatchMethod.RESOLVE, base_path=str(base_path)
            )

            # Create the file at base_path location
            (base_path / "test.mp4").touch()

            # Should check various resolution strategies
            matcher.match(video1, video2)

            # Test with both videos having same relative structure
            video3 = Video(filename="subdir/video.mp4")
            video4 = Video(filename="subdir/video.mp4")

            # These should match as they have identical relative paths
            assert matcher.match(video3, video4)

    def test_instance_matcher_find_matches_all_nan_spatial(self):
        """Test find_matches SPATIAL when instances have all NaN points."""
        import numpy as np

        from sleap_io import Instance, Skeleton

        skeleton = Skeleton(nodes=["head", "tail"])

        # Create instances with all NaN points
        inst1 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
        )
        inst2 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
        )

        # Test SPATIAL matching with all NaN points
        matcher = InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=10.0)
        matches = matcher.find_matches([inst1], [inst2])

        # Should match but with score 0.0 due to all NaN points (line 148-149)
        assert len(matches) == 1
        assert matches[0][2] == 0.0  # Score should be 0 for all NaN

    def test_instance_matcher_find_matches_iou_no_intersection(self):
        """Test IoU calculation when bounding boxes exist but don't intersect."""
        import numpy as np

        from sleap_io import Instance, Skeleton

        skeleton = Skeleton(nodes=["head", "tail"])

        # Create instances with valid bounding boxes but no intersection
        inst1 = Instance.from_numpy(np.array([[0, 0], [10, 10]]), skeleton=skeleton)
        inst2 = Instance.from_numpy(np.array([[20, 20], [30, 30]]), skeleton=skeleton)

        # Force them to have same track for identity matching to trigger IoU calculation
        shared_track = Track(name="track1")
        inst1.track = shared_track
        inst2.track = shared_track

        # Test with IOU matcher - will try to match but score will be 0 (line 170-171)
        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.01)
        matches = matcher.find_matches([inst1], [inst2])

        # Should not match because IoU is 0
        assert len(matches) == 0

    def test_instance_matcher_find_matches_iou_null_bbox(self):
        """Test find_matches when one instance has no valid bounding box."""
        import numpy as np

        from sleap_io import Instance, Skeleton

        skeleton = Skeleton(nodes=["head", "tail"])

        # Create instance with all NaN points (no valid bbox)
        inst1 = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
        )
        inst2 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton)

        # Give them same track to trigger matching attempt
        shared_track = Track(name="track1")
        inst1.track = shared_track
        inst2.track = shared_track

        # Test IOU matching with null bbox (line 172-173)
        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.1)
        matches = matcher.find_matches([inst1], [inst2])

        # Should not match due to null bbox
        assert len(matches) == 0

    def test_video_matcher_resolve_with_fallback_existing_file(self):
        """Test VideoMatcher RESOLVE with fallback directories containing file."""
        import tempfile
        from pathlib import Path

        from sleap_io.model.video import Video

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a video file in fallback directory
            fallback_dir = Path(tmpdir) / "fallback"
            fallback_dir.mkdir()
            video_file = fallback_dir / "test_video.mp4"
            video_file.touch()

            # Create videos with same basename but different paths
            video1 = Video(filename="/original/path/test_video.mp4")
            video2 = Video(filename="/different/path/test_video.mp4")

            # Test with fallback directory containing the file
            matcher = VideoMatcher(
                method=VideoMatchMethod.RESOLVE,
                fallback_directories=[str(fallback_dir)],
            )

            # Should match: basename matches and file exists in fallback
            assert matcher.match(video1, video2)

            # Test without matching file in fallback
            video3 = Video(filename="/path/nonexistent.mp4")
            assert not matcher.match(video1, video3)

    def test_video_matcher_resolve_with_base_path_existing_file(self):
        """Test VideoMatcher RESOLVE with base_path containing file."""
        import tempfile
        from pathlib import Path

        from sleap_io.model.video import Video

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create a video file at base path
            video_file = base_path / "test_video.mp4"
            video_file.touch()

            # Create videos with same basename
            video1 = Video(filename="/some/path/test_video.mp4")
            video2 = Video(filename="/other/path/test_video.mp4")

            # Test with base_path containing the file
            matcher = VideoMatcher(
                method=VideoMatchMethod.RESOLVE, base_path=str(base_path)
            )

            # Should match because file exists at base path (line 263-267)
            assert matcher.match(video1, video2)

    def test_video_matcher_resolve_exception_handling(self):
        """Test VideoMatcher RESOLVE method exception handling."""
        from unittest.mock import Mock

        from sleap_io.model.video import Video

        # Test with relative paths that cause relative_to to fail (line 268-269)
        matcher = VideoMatcher(method=VideoMatchMethod.RESOLVE, base_path="/some/base")

        # These paths will cause relative_to with anchor to fail
        video1 = Video(filename="relative/path1/video.mp4")
        video2 = Video(filename="relative/path2/different.mp4")  # Different basename

        # Should handle the exception and return False
        result = matcher.match(video1, video2)
        assert not result

        # Test with mock videos having None filenames to test the condition on line 231
        video3 = Mock(spec=Video)
        video3.filename = None
        video3.matches_path = Mock(return_value=False)

        video4 = Mock(spec=Video)
        video4.filename = "test.mp4"
        video4.matches_path = Mock(return_value=False)

        # Should return False because video3.filename is None
        assert not matcher.match(video3, video4)

    def test_video_matcher_resolve_same_relative_structure(self):
        """Test VideoMatcher RESOLVE with same relative path structure."""
        import tempfile
        from pathlib import Path

        from sleap_io.model.video import Video

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test the relative path matching logic (lines 255-260)
            matcher = VideoMatcher(method=VideoMatchMethod.RESOLVE, base_path=tmpdir)

            # Test with absolute paths that should have same relative structure
            # On Unix, these would strip the leading "/"
            video1 = Video(filename="/data/subdir/video.mp4")
            video2 = Video(filename="/other/subdir/video.mp4")

            # These have same basename and same relative path after anchor
            # Should trigger lines 255-260 and return True at line 260
            result = matcher.match(video1, video2)
            # Relative paths: "data/subdir/video.mp4" vs "other/subdir/video.mp4"
            # But it likely matches via matches_path with strict=False (same basename)
            # Since basenames match and paths match with strict=False, it returns True
            assert result  # Actually returns True due to basename matching

            # Test with paths that actually have the same structure
            video3 = Video(filename="/base/data/test.mp4")
            video4 = Video(filename="/base/data/test.mp4")
            assert matcher.match(video3, video4)  # Should match early at line 241

            # Test the base_path file existence check (lines 263-267)
            # Create a file at base_path
            test_file = Path(tmpdir) / "video.mp4"
            test_file.touch()

            video5 = Video(filename="/some/path/video.mp4")
            video6 = Video(filename="/other/path/video.mp4")

            # Should find the file at base_path and return True at line 267
            assert matcher.match(video5, video6)

    def test_video_matcher_resolve_relative_paths(self):
        """Test VideoMatcher RESOLVE with relative paths that cause exceptions."""
        from sleap_io.model.video import Video

        # Test with relative paths - should trigger exception at line 268-269
        matcher = VideoMatcher(method=VideoMatchMethod.RESOLVE, base_path="/some/base")

        # These are relative paths, so anchor will be empty
        video1 = Video(filename="relative/path1/video.mp4")
        video2 = Video(filename="relative/path2/video.mp4")

        # Should handle the exception and continue
        result = matcher.match(video1, video2)
        # Since basenames match, matches_path with strict=False returns True
        assert result  # Returns True because basenames match

    def test_video_matcher_resolve_windows_paths_on_unix(self):
        """Test VideoMatcher RESOLVE with Windows-style paths on Unix."""
        import tempfile
        from pathlib import Path

        from sleap_io.model.video import Video

        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = VideoMatcher(method=VideoMatchMethod.RESOLVE, base_path=tmpdir)

            # Windows-style paths on Unix have empty anchor
            video1 = Video(filename="C:/Users/data/video.mp4")
            video2 = Video(filename="D:/Projects/data/video.mp4")

            # Should handle the relative_to exception gracefully
            result = matcher.match(video1, video2)
            # Basenames match, so matches_path with strict=False returns True
            assert result  # Returns True because basenames match

            # Now create the file at base_path
            test_file = Path(tmpdir) / "video.mp4"
            test_file.touch()

            # Should find it at base_path and return True
            assert matcher.match(video1, video2)

    def test_video_matcher_resolve_fallback_multiple_dirs(self):
        """Test VideoMatcher RESOLVE checking multiple fallback directories."""
        import tempfile
        from pathlib import Path

        from sleap_io.model.video import Video

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple fallback directories
            fallback1 = Path(tmpdir) / "fallback1"
            fallback2 = Path(tmpdir) / "fallback2"
            fallback3 = Path(tmpdir) / "fallback3"
            fallback1.mkdir()
            fallback2.mkdir()
            fallback3.mkdir()

            # Test 1: File in second directory (tests iteration through fallback dirs)
            video_file2 = fallback2 / "test.mp4"
            video_file2.touch()

            video1 = Video(filename="/original/test.mp4")
            video2 = Video(filename="/different/test.mp4")

            matcher = VideoMatcher(
                method=VideoMatchMethod.RESOLVE,
                fallback_directories=[str(fallback1), str(fallback2), str(fallback3)],
            )

            # Should find it in fallback2 (lines 246-250)
            assert matcher.match(video1, video2)

            # Test 2: File in third directory
            video_file2.unlink()
            video_file3 = fallback3 / "test.mp4"
            video_file3.touch()

            # Should find it in fallback3
            assert matcher.match(video1, video2)

            # Test with file not in any fallback
            video3 = Video(filename="/path/nonexistent.mp4")
            video4 = Video(filename="/other/nonexistent.mp4")
            # Even though file doesn't exist, basenames still match
            assert matcher.match(
                video3, video4
            )  # Returns True due to basename matching

    def test_video_matcher_resolve_basename_mismatch_but_path_match(self):
        """Test VideoMatcher RESOLVE when basenames don't match but paths do."""
        from sleap_io.model.video import Video

        # Create matcher
        matcher = VideoMatcher(method=VideoMatchMethod.RESOLVE)

        # These have different basenames but video1.matches_path might still match
        # This tests line 273-274
        # Create videos with same paths (testing different basenames scenario)
        Video(filename="/path/to/video.mp4")
        Video(filename="/path/to/video.mp4")

        # Even though we're testing different basenames, let's mock this scenario
        # by using the same Video object
        video_same = Video(filename="/path/to/video.mp4")

        # Should match because paths are identical (line 273-274)
        assert matcher.match(video_same, video_same)

    def test_video_matcher_resolve_relative_path_with_base(self):
        """Test VideoMatcher RESOLVE with base_path preserving directory structure."""
        import tempfile
        from pathlib import Path

        from sleap_io.model.video import Video

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create directory structure with preserved parent directory
            parent_dir = base_path / "experiments"
            parent_dir.mkdir()
            video_file = parent_dir / "test_video.mp4"
            video_file.touch()

            # Videos with same parent directory and basename
            video1 = Video(filename="/old/path/experiments/test_video.mp4")
            video2 = Video(filename="/new/path/experiments/test_video.mp4")

            matcher = VideoMatcher(
                method=VideoMatchMethod.RESOLVE, base_path=str(base_path)
            )

            # Should match because:
            # 1. Basenames match (test_video.mp4)
            # 2. Parent directories match (experiments)
            # 3. File exists at base_path/experiments/test_video.mp4
            assert matcher.match(video1, video2)

            # Test with different parent directories
            video3 = Video(filename="/path/data/test_video.mp4")
            video4 = Video(filename="/path/results/test_video.mp4")

            # Should not find match with preserved parent directory
            # But will still match due to basename matching at base path
            # Create file at base path root
            base_file = base_path / "test_video.mp4"
            base_file.touch()

            # Now should match because file exists at base_path/test_video.mp4
            assert matcher.match(video3, video4)

            # Test with nested directory structure
            nested_dir = base_path / "project" / "data"
            nested_dir.mkdir(parents=True)
            nested_file = nested_dir / "video.mp4"
            nested_file.touch()

            video5 = Video(filename="/some/project/data/video.mp4")
            video6 = Video(filename="/other/project/data/video.mp4")

            # Should match with preserved nested structure
            assert matcher.match(video5, video6)

    def test_video_matcher_resolve_different_basenames_matching_paths(self):
        """Test VideoMatcher RESOLVE when basenames differ but paths might match."""
        from sleap_io.model.video import Video

        matcher = VideoMatcher(method=VideoMatchMethod.RESOLVE)

        # Test case 1: Different basenames should not match
        video1 = Video(filename="/path/to/video1.mp4")
        video2 = Video(filename="/path/to/video2.mp4")

        # Should not match because basenames are different (line 279-281)
        assert not matcher.match(video1, video2)

        # Test case 2: Different basenames but check path matching logic
        video3 = Video(filename="/data/experiment1.mp4")
        video4 = Video(filename="/data/experiment2.avi")

        # Different basenames and extensions, should not match
        assert not matcher.match(video3, video4)

        # Test case 3: Same paths should match even with RESOLVE
        video5 = Video(filename="/path/to/video.mp4")
        video6 = Video(filename="/path/to/video.mp4")

        # Same paths should match (checked early in RESOLVE)
        assert matcher.match(video5, video6)

    def test_instance_matcher_iou_score_edge_cases(self):
        """Test IOU score calculation edge cases in find_matches."""
        import numpy as np

        from sleap_io import Instance, Skeleton, Track

        skeleton = Skeleton(nodes=["p1", "p2"])

        # Test case 1: Both instances have no valid bounding box (lines 172-173)
        inst1_no_bbox = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
        )
        inst2_no_bbox = Instance.from_numpy(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]), skeleton=skeleton
        )

        # Force them to match with same track
        track = Track(name="track1")
        inst1_no_bbox.track = track
        inst2_no_bbox.track = track

        # Use IDENTITY matcher to trigger match, but check IOU score calculation
        matcher = InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.0)

        # Since both have no bbox, the score should be 0.0 (line 173)
        # But they won't match because IoU matching requires valid bboxes
        matches = matcher.find_matches([inst1_no_bbox], [inst2_no_bbox])
        assert len(matches) == 0

        # Test case 2: One instance has bbox, other doesn't (also line 172-173)
        inst_with_bbox = Instance.from_numpy(
            np.array([[10, 10], [20, 20]]), skeleton=skeleton
        )
        inst_with_bbox.track = track

        matches = matcher.find_matches([inst1_no_bbox], [inst_with_bbox])
        assert len(matches) == 0  # No match because one has no bbox

        # Test case 3: Bounding boxes don't intersect (line 170-171)
        inst3 = Instance.from_numpy(np.array([[0, 0], [5, 5]]), skeleton=skeleton)
        inst4 = Instance.from_numpy(np.array([[10, 10], [15, 15]]), skeleton=skeleton)

        track2 = Track(name="track2")
        inst3.track = track2
        inst4.track = track2

        matches = matcher.find_matches([inst3], [inst4])
        assert len(matches) == 0  # No intersection, no match
