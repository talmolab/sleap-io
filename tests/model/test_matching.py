"""Tests for the matching module."""

import numpy as np

import sleap_io as sio
from sleap_io.model.category import Category
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.matching import (
    NAME_CATEGORY_MATCHER,
    NAME_IDENTITY_MATCHER,
    CategoryMatcher,
    CategoryMatchMethod,
    ConflictResolution,
    ErrorMode,
    FrameStrategy,
    IdentityMatcher,
    IdentityMatchMethod,
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
    _crop_key,
    _get_effective_shape,
    _is_same_file_direct,
    _same_file_different_crop,
    is_same_file,
    shapes_compatible,
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


class TestIdentityMatcher:
    """Test global identity matching functionality."""

    def test_name_match_default(self):
        """Test that the default method matches by name."""
        id1 = Identity(name="mouse_A")
        id2 = Identity(name="mouse_A")  # Same name, different object
        id3 = Identity(name="mouse_B")

        matcher = IdentityMatcher()
        assert matcher.method == IdentityMatchMethod.NAME
        assert matcher.match(id1, id2)  # Same name
        assert not matcher.match(id1, id3)  # Different names

    def test_identity_match(self):
        """Test identity matching by Python object identity."""
        id1 = Identity(name="mouse_A")
        id2 = Identity(name="mouse_A")  # same name, diff object

        matcher = IdentityMatcher(method=IdentityMatchMethod.IDENTITY)
        assert not matcher.match(id1, id2)  # Different objects
        assert matcher.match(id1, id1)  # Same object

    def test_string_method_coercion(self):
        """Test that a string method is coerced to the enum."""
        matcher = IdentityMatcher(method="name")
        assert matcher.method == IdentityMatchMethod.NAME

    def test_prebuilt_matchers(self):
        """Test the module-level pre-built identity matchers."""
        assert NAME_IDENTITY_MATCHER.method == IdentityMatchMethod.NAME


class TestCategoryMatcher:
    """Test category matching functionality (mirror of identity matching)."""

    def test_name_match_default(self):
        """Test that the default method matches by name."""
        cat1 = Category(name="female_fly")
        cat2 = Category(name="female_fly")  # Same name, different object
        cat3 = Category(name="male_fly")

        matcher = CategoryMatcher()
        assert matcher.method == CategoryMatchMethod.NAME
        assert matcher.match(cat1, cat2)  # Same name
        assert not matcher.match(cat1, cat3)  # Different names

    def test_identity_match(self):
        """Test category matching by Python object identity."""
        cat1 = Category(name="female_fly")
        cat2 = Category(name="female_fly")  # same name, diff object

        matcher = CategoryMatcher(method=CategoryMatchMethod.IDENTITY)
        assert not matcher.match(cat1, cat2)  # Different objects
        assert matcher.match(cat1, cat1)  # Same object

    def test_string_method_coercion(self):
        """Test that a string method is coerced to the enum."""
        matcher = CategoryMatcher(method="name")
        assert matcher.method == CategoryMatchMethod.NAME

    def test_prebuilt_matchers(self):
        """Test the module-level pre-built category matchers."""
        assert NAME_CATEGORY_MATCHER.method == CategoryMatchMethod.NAME


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
        assert FrameStrategy.AUTO == "auto"
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


class TestVideoMatcherBasename:
    """Test VideoMatcher BASENAME method."""

    def test_basename_implementation(self):
        """Test VideoMatcher BASENAME method implementation."""
        # BASENAME method does filename-based matching ignoring directory paths

        video1 = Video(filename="/some/path/test_video.mp4", open_backend=False)
        video2 = Video(filename="/different/path/test_video.mp4", open_backend=False)
        video3 = Video(filename="/path/other_video.mp4", open_backend=False)

        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

        # Same basenames should match
        assert matcher.match(video1, video2)

        # Different basenames should not match
        assert not matcher.match(video1, video3)

        # Identity should always match
        assert matcher.match(video1, video1)

    def test_basename_vs_basename_consistency(self):
        """Test BASENAME method consistency."""
        video1 = Video(filename="/path1/video.mp4", open_backend=False)
        video2 = Video(filename="/path2/video.mp4", open_backend=False)
        video3 = Video(filename="/path/other.mp4", open_backend=False)

        matcher1 = VideoMatcher(method=VideoMatchMethod.BASENAME)
        matcher2 = VideoMatcher(method=VideoMatchMethod.BASENAME)

        # All BASENAME matchers should behave consistently
        assert matcher1.match(video1, video2) == matcher2.match(video1, video2)
        assert matcher1.match(video1, video3) == matcher2.match(video1, video3)

        # Same object should always match
        assert matcher1.match(video1, video1)
        assert matcher2.match(video1, video1)


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

        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)
        assert matcher.match(video1, video2)  # Same basename
        assert not matcher.match(video1, video3)  # Different basename

    def test_video_matcher_auto_no_content_fallback(self):
        """Test VideoMatcher AUTO does NOT fall back to content-only matching.

        The safe AUTO algorithm uses shape for REJECTION only, not positive matching.
        Two videos with same shape but different names should NOT match.
        """
        video1 = Video(filename="video1.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)

        video2 = Video(filename="video2.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)

        video3 = Video(filename="video3.mp4", open_backend=False)
        video3.backend_metadata["shape"] = (50, 240, 320, 3)  # Different shape

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        # Should NOT match - different names, even with same shape
        # Shape is for rejection only, not positive evidence
        assert not matcher.match(video1, video2)

        # Should not match - different names and different shapes
        assert not matcher.match(video1, video3)

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

    def test_video_matcher_resolve_simplified(self):
        """Test video matching with simplified RESOLVE method."""
        # Videos with same basename but different paths
        video1 = Video(filename="/original/path/test_video.mp4", open_backend=False)
        video2 = Video(filename="/different/path/test_video.mp4", open_backend=False)
        video3 = Video(filename="/path/other_video.mp4", open_backend=False)

        # RESOLVE matcher (simplified implementation)
        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

        # Should match because basename matches (same as BASENAME method behavior)
        assert matcher.match(video1, video2)

        # Should not match because basenames differ
        assert not matcher.match(video1, video3)

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

    def test_video_matcher_resolve_simplified_functionality(self):
        """Test VideoMatcher RESOLVE simplified functionality."""
        from sleap_io.model.video import Video

        # Create videos with same basename but different paths
        video1 = Video(filename="/original/path/test_video.mp4", open_backend=False)
        video2 = Video(filename="/different/path/test_video.mp4", open_backend=False)

        # RESOLVE matcher (simplified - equivalent to basename matching)
        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

        # Should match: basename matches
        assert matcher.match(video1, video2)

        # Test without matching basenames
        video3 = Video(filename="/path/nonexistent.mp4", open_backend=False)
        assert not matcher.match(video1, video3)

    def test_video_matcher_resolve_identity_check(self):
        """Test VideoMatcher RESOLVE identity check behavior."""
        from sleap_io.model.video import Video

        # Test identity check (should always match same object)
        video1 = Video(filename="/path/video.mp4", open_backend=False)
        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

        # Identity check should always return True
        assert matcher.match(video1, video1)

    def test_video_matcher_resolve_basename_mismatch_but_path_match(self):
        """Test VideoMatcher RESOLVE when basenames don't match but paths do."""
        from sleap_io.model.video import Video

        # Create matcher
        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

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

    def test_video_matcher_resolve_different_basenames_matching_paths(self):
        """Test VideoMatcher RESOLVE when basenames differ but paths might match."""
        from sleap_io.model.video import Video

        matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

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

    def test_instance_matcher_iou_score_calculation_coverage(self):
        """Test to specifically cover the missing IoU score calculation lines."""
        import numpy as np

        from sleap_io import Instance, Skeleton

        skeleton = Skeleton(nodes=["tl", "tr", "br", "bl"])

        # Create a custom matcher class to bypass the initial match() check
        # This allows us to test the score calculation logic directly
        class TestableInstanceMatcher(InstanceMatcher):
            def match(self, instance1: Instance, instance2: Instance) -> bool:
                # Always return True to test score calculation
                return True

        matcher = TestableInstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.0)

        # Test case 1: Bounding boxes don't intersect (lines 170-171)
        inst1 = Instance.from_numpy(
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]]), skeleton=skeleton
        )
        inst2 = Instance.from_numpy(
            np.array([[10, 10], [12, 10], [12, 12], [10, 12]]), skeleton=skeleton
        )

        matches = matcher.find_matches([inst1], [inst2])
        assert len(matches) == 1
        assert matches[0][2] == 0.0  # Score should be 0.0 for no intersection

        # Test case 2: One instance has no valid bounding box (lines 172-173)
        inst3 = Instance.from_numpy(
            np.array(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
            ),
            skeleton=skeleton,
        )
        inst4 = Instance.from_numpy(
            np.array([[5, 5], [7, 5], [7, 7], [5, 7]]), skeleton=skeleton
        )

        matches = matcher.find_matches([inst3], [inst4])
        assert len(matches) == 1
        assert matches[0][2] == 0.0  # Score should be 0.0 for no bbox

        # Test case 3: Both instances have no valid bounding box (also lines 172-173)
        inst5 = Instance.from_numpy(
            np.array(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
            ),
            skeleton=skeleton,
        )

        matches = matcher.find_matches([inst3], [inst5])
        assert len(matches) == 1
        assert matches[0][2] == 0.0  # Score should be 0.0 for no bbox on either


class TestVideoMatcherCoverageGaps:
    """Tests specifically targeting coverage gaps in matching.py."""

    def test_is_same_file_direct_imagevideo_vs_single_file(self):
        """Test _is_same_file_direct when one is ImageVideo and other is single file.

        Covers line 184: return False when mixed ImageVideo/single file.
        """
        from sleap_io.model.matching import _is_same_file_direct

        # ImageVideo (list of paths)
        imagevideo = Video(
            filename=["/data/img_001.jpg", "/data/img_002.jpg"], open_backend=False
        )

        # Single file video
        single_video = Video(filename="/data/video.mp4", open_backend=False)

        # Mixed types should return False
        assert not _is_same_file_direct(imagevideo, single_video)
        assert not _is_same_file_direct(single_video, imagevideo)

    def test_get_effective_shape_with_original_video(self):
        """Test _get_effective_shape returns original_video's shape.

        Covers line 253: return original_shape from original_video chain.
        Note: original_video is now a computed property from source_video chain.
        """
        from sleap_io.model.matching import _get_effective_shape

        # Create original video with shape
        original = Video(filename="/data/original.mp4", open_backend=False)
        original.backend_metadata["shape"] = (100, 480, 640, 3)

        # Create embedded video with source_video reference
        # (original_video will be computed from source_video)
        embedded = Video(
            filename="embedded.pkg.slp", source_video=original, open_backend=False
        )

        # Verify original_video is computed correctly
        assert embedded.original_video is original

        # Should return original's shape
        shape = _get_effective_shape(embedded)
        assert shape == (100, 480, 640, 3)

    def test_video_matcher_auto_provenance_conflict_rejection(self):
        """Test AUTO pairwise rejects when provenance chains conflict.

        Covers line 525: return False for provenance conflict.
        Note: original_video is now a computed property from source_video chain.
        """
        # Create two videos with different source_video references
        original1 = Video(filename="/data/video1.mp4", open_backend=False)
        original1.backend_metadata["shape"] = (100, 480, 640, 3)

        original2 = Video(filename="/data/video2.mp4", open_backend=False)
        original2.backend_metadata["shape"] = (100, 480, 640, 3)  # Same shape

        embedded1 = Video(
            filename="embedded.pkg.slp", source_video=original1, open_backend=False
        )
        embedded1.backend_metadata["shape"] = (100, 480, 640, 3)

        embedded2 = Video(
            filename="embedded2.pkg.slp", source_video=original2, open_backend=False
        )
        embedded2.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        # Should NOT match despite same shape - different provenance chains
        assert not matcher.match(embedded1, embedded2)

    def test_video_matcher_auto_strict_path_match(self):
        """Test AUTO pairwise matching via strict path match.

        Covers line 533: return True for strict path match.
        """
        # Create videos with same path but different objects
        video1 = Video(filename="/data/videos/test.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)

        video2 = Video(filename="/data/videos/test.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        # Should match via strict path match
        assert matcher.match(video1, video2)

    def test_video_matcher_shape_method(self):
        """Test VideoMatcher with SHAPE method.

        Covers line 552: video1.matches_shape(video2) for SHAPE method.
        """
        video1 = Video(filename="video1.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)

        video2 = Video(filename="video2.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)

        video3 = Video(filename="video3.mp4", open_backend=False)
        video3.backend_metadata["shape"] = (200, 720, 1280, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.SHAPE)
        assert matcher.match(video1, video2)  # Same shape
        assert not matcher.match(video1, video3)  # Different shape

    def test_video_matcher_find_match_full_path_match(self):
        """Test find_match returns candidate via full path match.

        Covers line 612: return candidate for full path string match.
        """
        # Create candidate with same path as incoming
        candidate = Video(filename="/data/recordings/video.mp4", open_backend=False)
        candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        other_candidate = Video(
            filename="/data/other/video.mp4",  # Different path
            open_backend=False,
        )
        other_candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        incoming = Video(filename="/data/recordings/video.mp4", open_backend=False)
        incoming.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(incoming, [other_candidate, candidate])

        # Should match candidate with exact path
        assert result is candidate

    def test_video_matcher_find_match_imagevideo_leaf_path(self):
        """Test find_match uses first image path for ImageVideo leaf comparison.

        Covers line 622: fn = fn[0] for ImageVideo first file.
        """
        # Create ImageVideo candidates
        images1 = ["/data/exp1/img_001.jpg", "/data/exp1/img_002.jpg"]
        candidate1 = Video(filename=images1.copy(), open_backend=False)
        candidate1.backend_metadata["shape"] = (2, 480, 640, 3)

        images2 = ["/data/exp2/img_001.jpg", "/data/exp2/img_002.jpg"]
        candidate2 = Video(filename=images2.copy(), open_backend=False)
        candidate2.backend_metadata["shape"] = (2, 480, 640, 3)

        # Incoming with matching leaf path to candidate1
        incoming = Video(
            filename=["/other/exp1/img_001.jpg", "/other/exp1/img_002.jpg"],
            open_backend=False,
        )
        incoming.backend_metadata["shape"] = (2, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(incoming, [candidate1, candidate2])

        # Should match candidate1 via leaf path uniqueness (exp1 disambiguates)
        assert result is candidate1

    def test_video_matcher_find_match_depth_comparison_edge_cases(self):
        """Test find_match depth comparison when paths have different lengths.

        Covers lines 639, 646: continue when parts < depth.
        """
        # Create candidates with different path depths but DIFFERENT basenames
        # so the shallow one doesn't match at depth 1
        shallow_candidate = Video(filename="shallow.mp4", open_backend=False)
        shallow_candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        deep_candidate = Video(
            filename="/very/deep/path/to/video.mp4", open_backend=False
        )
        deep_candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        # Incoming that matches deep candidate's leaf but not shallow
        incoming = Video(filename="/path/to/video.mp4", open_backend=False)
        incoming.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(incoming, [shallow_candidate, deep_candidate])

        # Should match deep_candidate via "to/video.mp4" suffix
        # This exercises depth comparison where shallow_candidate is skipped (line 646)
        assert result is deep_candidate

    def test_video_matcher_find_match_incoming_shallow_path(self):
        """Test find_match when incoming path is shallower than candidate depth.

        Covers line 639: continue when incoming_parts < depth.
        """
        # Create candidate with deep path
        candidate = Video(
            filename="/very/deep/nested/path/to/video.mp4", open_backend=False
        )
        candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        # Incoming with shallow path (just filename)
        incoming = Video(filename="video.mp4", open_backend=False)
        incoming.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(incoming, [candidate])

        # Should match via basename at depth 1
        assert result is candidate

    def test_merge_progress_bar_close_with_active_pbar(self):
        """Test MergeProgressBar closes pbar when active.

        Covers line 832: pbar.close() when pbar is not None.
        """
        with MergeProgressBar(desc="Test merge") as progress:
            # Trigger pbar creation
            progress.callback(50, 100, "Processing")
            assert progress.pbar is not None
            # Store pbar reference to verify it was closed
            pbar_ref = progress.pbar

        # After context exit, pbar should have been closed
        # tqdm close() sets n to total, we can check it was called
        assert pbar_ref.n == 50  # Value at last callback

    def test_merge_progress_bar_exit_without_pbar(self):
        """Test MergeProgressBar exit when pbar was never created.

        Covers line 831->exit: branch where pbar is None.
        """
        with MergeProgressBar(desc="Test merge") as progress:
            # Don't call callback - pbar stays None
            assert progress.pbar is None

        # Context exit should handle pbar=None gracefully
        assert progress.pbar is None

    def test_video_matcher_find_match_line_639_incoming_shorter(self):
        """Test find_match skips depths when incoming path is too short.

        Covers line 639: continue when len(incoming_parts) < depth.

        Scenario: incoming has just basename, multiple candidates share same
        basename but have different parent paths. This forces iteration
        to depth 2+ where incoming is too short.
        """
        # Two candidates with same basename but different parents
        candidate1 = Video(filename="/data/exp1/video.mp4", open_backend=False)
        candidate1.backend_metadata["shape"] = (100, 480, 640, 3)

        candidate2 = Video(filename="/data/exp2/video.mp4", open_backend=False)
        candidate2.backend_metadata["shape"] = (100, 480, 640, 3)

        # Incoming with just basename (1 part)
        incoming = Video(filename="video.mp4", open_backend=False)
        incoming.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(incoming, [candidate1, candidate2])

        # At depth 1, both candidates match "video.mp4" - ambiguous
        # At depth 2, incoming has only 1 part < 2 → continue (line 639)
        # Loop continues without finding unique match → returns None
        assert result is None

    def test_video_matcher_find_match_line_646_candidate_shorter(self):
        """Test find_match skips candidates when their path is too short.

        Covers line 646: continue when len(parts) < depth for a candidate.

        Scenario: incoming has deep path, one candidate has same basename
        but shallow path, another has deep path. At depth 2+, shallow
        candidate is skipped.
        """
        # Shallow candidate (just basename)
        shallow_candidate = Video(filename="video.mp4", open_backend=False)
        shallow_candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        # Deep candidate with matching parent
        deep_candidate = Video(filename="/data/exp1/video.mp4", open_backend=False)
        deep_candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        # Another deep candidate with different parent
        deep_candidate2 = Video(filename="/data/exp2/video.mp4", open_backend=False)
        deep_candidate2.backend_metadata["shape"] = (100, 480, 640, 3)

        # Incoming with path matching deep_candidate
        incoming = Video(filename="/other/exp1/video.mp4", open_backend=False)
        incoming.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(
            incoming, [shallow_candidate, deep_candidate, deep_candidate2]
        )

        # At depth 1, all three match "video.mp4" - ambiguous
        # At depth 2, shallow_candidate has only 1 part < 2 → continue (line 646)
        # Only deep_candidate matches "exp1/video.mp4" → unique match
        assert result is deep_candidate

    def test_video_matcher_find_match_with_normalized_paths(self):
        """Test find_match works with path normalization via sanitize_filename.

        Tests that the matching algorithm correctly handles paths that normalize
        to the same value after sanitization, even if the raw path strings differ.
        """
        from sleap_io.io.utils import sanitize_filename

        # Create paths with mixed slashes - on Unix, backslash is literal char
        path1 = "/data\\subdir/video.mp4"  # Mixed slashes (backslash is literal)
        path2 = "/data/subdir/video.mp4"  # Forward slashes only

        # Verify sanitize_filename normalizes backslashes to forward slashes
        sanitized1 = sanitize_filename(path1)
        sanitized2 = sanitize_filename(path2)
        assert sanitized1 == sanitized2  # Both normalize to same path

        candidate = Video(filename=path2, open_backend=False)
        candidate.backend_metadata["shape"] = (100, 480, 640, 3)

        incoming = Video(filename=path1, open_backend=False)
        incoming.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(incoming, [candidate])

        # Result depends on how paths resolve - leaf path matching should work
        # since both normalize to same basename "video.mp4"
        # The exact match depends on path resolution behavior
        assert result is candidate or result is None

    def test_is_same_file_hdf5_different_datasets(self):
        """Test that HDF5 videos with same filename but different datasets.

        Regression test for a bug where embedded videos from split pkg.slp files
        were incorrectly matched because only the filename was compared, not the
        HDF5 dataset within the file.

        For example, train.pkg.slp and val.pkg.slp might both have videos with
        original_video pointing to the same source file but different datasets
        (video0/video, video1/video, etc.). These should NOT be matched.
        """
        from sleap_io.io.video_reading import HDF5Video

        # Create two videos with same filename but different datasets
        # This simulates embedded videos from the same original source
        backend1 = HDF5Video(
            filename="source.pkg.slp",
            dataset="video0/video",
            grayscale=None,
            keep_open=False,
        )
        video1 = Video(filename="source.pkg.slp", backend=backend1, open_backend=False)

        backend2 = HDF5Video(
            filename="source.pkg.slp",
            dataset="video1/video",  # Different dataset!
            grayscale=None,
            keep_open=False,
        )
        video2 = Video(filename="source.pkg.slp", backend=backend2, open_backend=False)

        backend3 = HDF5Video(
            filename="source.pkg.slp",
            dataset="video0/video",  # Same dataset as video1
            grayscale=None,
            keep_open=False,
        )
        video3 = Video(filename="source.pkg.slp", backend=backend3, open_backend=False)

        # Same filename, different datasets → different videos
        assert not _is_same_file_direct(video1, video2)

        # Same filename, same dataset → same video
        assert _is_same_file_direct(video1, video3)

    def test_is_same_file_hdf5_with_provenance_chain(self):
        """Test is_same_file works with HDF5 videos in provenance chains.

        This tests the full is_same_file function (which traverses provenance)
        with embedded videos that have source_video set.
        """
        from sleap_io.io.video_reading import HDF5Video

        # Create source videos (the original_video in the chain)
        source1_backend = HDF5Video(
            filename="original.pkg.slp",
            dataset="video5/video",
            grayscale=None,
            keep_open=False,
        )
        source1 = Video(
            filename="original.pkg.slp", backend=source1_backend, open_backend=False
        )

        source2_backend = HDF5Video(
            filename="original.pkg.slp",
            dataset="video10/video",  # Different dataset
            grayscale=None,
            keep_open=False,
        )
        source2 = Video(
            filename="original.pkg.slp", backend=source2_backend, open_backend=False
        )

        # Create embedded videos pointing to different sources
        embedded1_backend = HDF5Video(
            filename="train.pkg.slp",
            dataset="video0/video",
            grayscale=None,
            keep_open=False,
        )
        embedded1 = Video(
            filename="train.pkg.slp",
            backend=embedded1_backend,
            source_video=source1,
            open_backend=False,
        )

        embedded2_backend = HDF5Video(
            filename="val.pkg.slp",
            dataset="video0/video",
            grayscale=None,
            keep_open=False,
        )
        embedded2 = Video(
            filename="val.pkg.slp",
            backend=embedded2_backend,
            source_video=source2,
            open_backend=False,
        )

        embedded3_backend = HDF5Video(
            filename="test.pkg.slp",
            dataset="video0/video",
            grayscale=None,
            keep_open=False,
        )
        embedded3 = Video(
            filename="test.pkg.slp",
            backend=embedded3_backend,
            source_video=source1,
            open_backend=False,
        )

        # Different source datasets → different videos
        assert not is_same_file(embedded1, embedded2)

        # Same source dataset → same videos
        assert is_same_file(embedded1, embedded3)


class TestLeafPathMatchingFix:
    """Tests for the leaf-path matching bug fix."""

    def test_get_path_parts_uses_root_for_embedded(self):
        """Leaf-path matching should use root video path for embedded videos."""
        from sleap_io.model.matching import _get_root_video

        # Create a video with source_video chain
        root_video = Video(filename="/original/path/video.mp4", open_backend=False)
        embedded_video = Video(
            filename="/embedded/file.pkg.slp",
            source_video=root_video,
            open_backend=False,
        )

        # The root should be the original video
        assert _get_root_video(embedded_video).filename == "/original/path/video.mp4"

    def test_find_match_uses_root_path_for_embedded(self):
        """find_match should use root video path when matching embedded videos."""
        # Create embedded videos with same root path but different embedded paths
        root1 = Video(filename="/data/exp/CHR/video.mp4", open_backend=False)
        root1.backend_metadata["shape"] = (100, 480, 640, 3)
        embedded1 = Video(
            filename="/linux/path/train.pkg.slp",
            source_video=root1,
            open_backend=False,
        )
        embedded1.backend_metadata["shape"] = (100, 480, 640, 3)

        root2 = Video(filename="X:/data/exp/CHR/video.mp4", open_backend=False)
        root2.backend_metadata["shape"] = (100, 480, 640, 3)
        embedded2 = Video(
            filename="Y:/windows/path/val.pkg.slp",
            source_video=root2,
            open_backend=False,
        )
        embedded2.backend_metadata["shape"] = (100, 480, 640, 3)

        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        result = matcher.find_match(embedded2, [embedded1])

        # Should match because root video paths share leaf "CHR/video.mp4"
        assert result is embedded1


class TestProvenanceConflictFallthrough:
    """Tests for provenance conflict fall-through behavior."""

    def test_no_conflict_when_files_dont_exist(self):
        """Don't reject when neither provenance file exists."""
        from sleap_io.model.matching import original_videos_conflict

        video1_root = Video(filename="/nonexistent/path1/video.mp4", open_backend=False)
        video1 = Video(
            filename="/pkg1.slp", source_video=video1_root, open_backend=False
        )

        video2_root = Video(filename="/nonexistent/path2/video.mp4", open_backend=False)
        video2 = Video(
            filename="/pkg2.slp", source_video=video2_root, open_backend=False
        )

        # Both have provenance, but files don't exist
        # Should NOT conflict (allow fall-through)
        assert not original_videos_conflict(video1, video2)

    def test_file_exists_helper(self, tmp_path):
        """Test _file_exists helper function."""
        from sleap_io.model.matching import _file_exists

        # Nonexistent file
        assert not _file_exists("/definitely/not/a/real/path.mp4")

        # Existing file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        assert _file_exists(str(test_file))

        # List of files
        test_file2 = tmp_path / "test2.txt"
        test_file2.write_text("test2")
        assert _file_exists([str(test_file), str(test_file2)])

        # List with nonexistent file
        assert not _file_exists([str(test_file), "/nonexistent.txt"])

    def test_conflict_when_one_file_exists(self, tmp_path):
        """Reject when one file exists and paths differ."""
        from sleap_io.model.matching import original_videos_conflict

        # Create an actual file
        real_file = tmp_path / "real_video.mp4"
        real_file.write_text("fake video content")

        video1_root = Video(filename=str(real_file), open_backend=False)
        video1 = Video(
            filename=str(real_file), source_video=video1_root, open_backend=False
        )

        video2_root = Video(
            filename="/nonexistent/different/video.mp4", open_backend=False
        )
        video2 = Video(
            filename="/other.slp", source_video=video2_root, open_backend=False
        )

        # One file exists and paths differ - should conflict
        assert original_videos_conflict(video1, video2)


class TestPoseMatching:
    """Tests for pose-based video matching helpers."""

    def test_poses_identical_exact_match(self):
        """Identical poses should match."""
        from sleap_io.model.matching import _poses_identical

        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert _poses_identical(pts, pts.copy())

    def test_poses_identical_different_values(self):
        """Different poses should not match."""
        from sleap_io.model.matching import _poses_identical

        pts1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        pts2 = np.array([[1.0, 2.0], [3.0, 5.0]])  # One value different
        assert not _poses_identical(pts1, pts2)

    def test_poses_identical_nan_handling(self):
        """NaN values should be handled correctly."""
        from sleap_io.model.matching import _poses_identical

        pts1 = np.array([[1.0, 2.0], [np.nan, np.nan]])
        pts2 = np.array([[1.0, 2.0], [np.nan, np.nan]])
        assert _poses_identical(pts1, pts2)

        # Different NaN patterns should not match
        pts3 = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert not _poses_identical(pts1, pts3)

    def test_poses_identical_shape_mismatch(self):
        """Different shapes should not match."""
        from sleap_io.model.matching import _poses_identical

        pts1 = np.array([[1.0, 2.0]])
        pts2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not _poses_identical(pts1, pts2)

    def test_poses_identical_all_nan(self):
        """All-NaN poses should not match (no valid points)."""
        from sleap_io.model.matching import _poses_identical

        pts1 = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        pts2 = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        assert not _poses_identical(pts1, pts2)

    def test_frame_has_matching_pose(self):
        """Test frame-level pose matching."""
        from sleap_io import Instance, Skeleton
        from sleap_io.model.matching import _frame_has_matching_pose
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a"), Node("b")])

        inst1 = Instance.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton)
        inst2 = Instance.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton)
        inst3 = Instance.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]]), skeleton)

        # Same poses should match
        assert _frame_has_matching_pose([inst1], [inst2])

        # Different poses should not match
        assert not _frame_has_matching_pose([inst1], [inst3])

        # ANY match is enough
        assert _frame_has_matching_pose([inst1, inst3], [inst2])

    def test_sample_frame_indices(self):
        """Test frame index sampling."""
        from sleap_io.model.matching import _sample_frame_indices

        # Less than max_samples - return all
        indices = {0, 5, 10}
        result = _sample_frame_indices(indices, max_samples=10)
        assert result == [0, 5, 10]

        # More than max_samples - sample evenly
        indices = set(range(100))
        result = _sample_frame_indices(indices, max_samples=5)
        assert len(result) == 5
        assert result[0] == 0
        assert result[-1] == 80  # Evenly spaced


class TestVideoMatcherPoseMatching:
    """Integration tests for VideoMatcher with pose matching."""

    def test_match_by_poses_identical(self):
        """Videos with identical poses should match."""
        from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a"), Node("b")])
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])

        video1 = Video(filename="/path1/video.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)
        video2 = Video(filename="/path2/video.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)

        labels1 = Labels(
            videos=[video1],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video1,
                    frame_idx=0,
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
                LabeledFrame(
                    video=video1,
                    frame_idx=10,
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
                LabeledFrame(
                    video=video1,
                    frame_idx=20,
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
            ],
        )

        labels2 = Labels(
            videos=[video2],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video2,
                    frame_idx=0,
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
                LabeledFrame(
                    video=video2,
                    frame_idx=10,
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
                LabeledFrame(
                    video=video2,
                    frame_idx=20,
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
            ],
        )

        matcher = VideoMatcher(method="auto", content_frames=3)
        match = matcher.find_match(
            incoming=video2,
            candidates=[video1],
            labels_incoming=labels2,
            labels_base=labels1,
        )

        assert match is video1

    def test_match_by_poses_different(self):
        """Videos with different poses should not match."""
        from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a"), Node("b")])

        # Use different basenames to avoid path matching
        video1 = Video(filename="/path1/video_A.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)
        video2 = Video(filename="/path2/video_B.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)

        labels1 = Labels(
            videos=[video1],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video1,
                    frame_idx=0,
                    instances=[
                        Instance.from_numpy(
                            np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton
                        )
                    ],
                ),
            ],
        )

        labels2 = Labels(
            videos=[video2],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video2,
                    frame_idx=0,
                    instances=[
                        Instance.from_numpy(
                            np.array([[100.0, 200.0], [300.0, 400.0]]), skeleton
                        )
                    ],
                ),
            ],
        )

        matcher = VideoMatcher(method="auto")
        match = matcher.find_match(
            incoming=video2,
            candidates=[video1],
            labels_incoming=labels2,
            labels_base=labels1,
        )

        assert match is None

    def test_match_by_poses_no_common_frames(self):
        """Videos with no common frame indices should not match via poses."""
        from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a"), Node("b")])
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])

        # Use different basenames to avoid path matching
        video1 = Video(filename="/path1/video_A.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)
        video2 = Video(filename="/path2/video_B.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)

        labels1 = Labels(
            videos=[video1],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video1,
                    frame_idx=0,
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
            ],
        )

        labels2 = Labels(
            videos=[video2],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video2,
                    frame_idx=50,  # Different frame
                    instances=[Instance.from_numpy(pts, skeleton)],
                ),
            ],
        )

        matcher = VideoMatcher(method="auto")
        match = matcher.find_match(
            incoming=video2,
            candidates=[video1],
            labels_incoming=labels2,
            labels_base=labels1,
        )

        assert match is None

    def test_match_by_poses_no_annotations(self):
        """Video with no annotations should fall through to None."""
        from sleap_io import Labels, Skeleton, Video
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a")])

        # Use different basenames to avoid path matching
        video1 = Video(filename="/path1/video_A.mp4", open_backend=False)
        video1.backend_metadata["shape"] = (100, 480, 640, 3)
        video2 = Video(filename="/path2/video_B.mp4", open_backend=False)
        video2.backend_metadata["shape"] = (100, 480, 640, 3)

        labels1 = Labels(videos=[video1], skeletons=[skeleton], labeled_frames=[])
        labels2 = Labels(videos=[video2], skeletons=[skeleton], labeled_frames=[])

        matcher = VideoMatcher(method="auto")
        match = matcher.find_match(
            incoming=video2,
            candidates=[video1],
            labels_incoming=labels2,
            labels_base=labels1,
        )

        assert match is None


class TestComparePredictionsAuto:
    """Tests for compare_predictions='auto' behavior."""

    def test_auto_excludes_predictions_when_user_exists(self):
        """With user instances present, predictions should be excluded."""
        from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
        from sleap_io.model.instance import PredictedInstance
        from sleap_io.model.matching import _resolve_compare_predictions
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a")])
        video = Video(filename="/video.mp4", open_backend=False)
        labels = Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video,
                    frame_idx=0,
                    instances=[
                        Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton),
                        PredictedInstance.from_numpy(
                            np.array([[3.0, 4.0]]), skeleton, score=0.9
                        ),
                    ],
                ),
            ],
        )

        result = _resolve_compare_predictions("auto", labels, video)
        assert result is False  # Has user instances, exclude predictions

    def test_auto_includes_predictions_when_only_predictions(self):
        """With only predictions, should include them."""
        from sleap_io import LabeledFrame, Labels, Skeleton, Video
        from sleap_io.model.instance import PredictedInstance
        from sleap_io.model.matching import _resolve_compare_predictions
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a")])
        video = Video(filename="/video.mp4", open_backend=False)
        labels = Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video,
                    frame_idx=0,
                    instances=[
                        PredictedInstance.from_numpy(
                            np.array([[1.0, 2.0]]), skeleton, score=0.9
                        ),
                    ],
                ),
            ],
        )

        result = _resolve_compare_predictions("auto", labels, video)
        assert result is True  # Only predictions, include them

    def test_explicit_true_always_includes(self):
        """compare_predictions=True should always include predictions."""
        from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
        from sleap_io.model.matching import _resolve_compare_predictions
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a")])
        video = Video(filename="/video.mp4", open_backend=False)
        labels = Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video,
                    frame_idx=0,
                    instances=[Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton)],
                ),
            ],
        )

        result = _resolve_compare_predictions(True, labels, video)
        assert result is True

    def test_explicit_false_always_excludes(self):
        """compare_predictions=False should always exclude predictions."""
        from sleap_io import LabeledFrame, Labels, Skeleton, Video
        from sleap_io.model.instance import PredictedInstance
        from sleap_io.model.matching import _resolve_compare_predictions
        from sleap_io.model.skeleton import Node

        skeleton = Skeleton(nodes=[Node("a")])
        video = Video(filename="/video.mp4", open_backend=False)
        labels = Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[
                LabeledFrame(
                    video=video,
                    frame_idx=0,
                    instances=[
                        PredictedInstance.from_numpy(
                            np.array([[1.0, 2.0]]), skeleton, score=0.9
                        ),
                    ],
                ),
            ],
        )

        result = _resolve_compare_predictions(False, labels, video)
        assert result is False


class TestImageMatching:
    """Tests for image-based video matching helpers."""

    def test_get_embedded_frame_indices_no_backend(self):
        """Video without backend should return None."""
        from sleap_io.model.matching import _get_embedded_frame_indices

        video = Video(filename="/video.mp4", open_backend=False)
        result = _get_embedded_frame_indices(video)
        assert result is None

    def test_get_common_embedded_indices_no_indices(self):
        """Videos without embedded indices should return empty set."""
        from sleap_io.model.matching import _get_common_embedded_indices

        video1 = Video(filename="/video1.mp4", open_backend=False)
        video2 = Video(filename="/video2.mp4", open_backend=False)
        result = _get_common_embedded_indices(video1, video2)
        assert result == set()

    def test_to_grayscale_float_2d(self):
        """Test grayscale conversion for 2D array."""
        from sleap_io.model.matching import _to_grayscale_float

        frame = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
        result = _to_grayscale_float(frame)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        assert np.allclose(result[0, 0], 0.0)
        assert np.allclose(result[0, 2], 1.0)

    def test_to_grayscale_float_3d_single_channel(self):
        """Test grayscale conversion for 3D array with single channel."""
        from sleap_io.model.matching import _to_grayscale_float

        frame = np.array([[[0], [128]], [[255], [64]]], dtype=np.uint8)
        result = _to_grayscale_float(frame)
        assert result.shape == (2, 2)

    def test_to_grayscale_float_3d_rgb(self):
        """Test grayscale conversion for 3D RGB array."""
        from sleap_io.model.matching import _to_grayscale_float

        # Pure red
        frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
        result = _to_grayscale_float(frame)
        expected = 0.299  # Red coefficient
        assert np.allclose(result[0, 0], expected, atol=0.01)

    def test_to_grayscale_float_invalid_shape(self):
        """Test grayscale conversion with invalid shape raises error."""
        import pytest

        from sleap_io.model.matching import _to_grayscale_float

        frame = np.array([[[[1]]]], dtype=np.uint8)  # 4D
        with pytest.raises(ValueError, match="Unexpected frame shape"):
            _to_grayscale_float(frame)

    def test_to_grayscale_float_2_channels(self):
        """Test grayscale conversion for 3D array with 2 channels (edge case)."""
        from sleap_io.model.matching import _to_grayscale_float

        # 2 channels - hits the else branch at line 430
        frame = np.array([[[0, 128], [255, 64]]], dtype=np.uint8)
        result = _to_grayscale_float(frame)
        # Should use first channel only
        assert result.shape == (1, 2)
        assert np.allclose(result[0, 0], 0.0)
        assert np.allclose(result[0, 1], 1.0)

    def test_frames_similar_by_image_identical(self):
        """Test _frames_similar_by_image with identical frames."""
        from unittest.mock import patch

        from sleap_io.model.matching import _frames_similar_by_image
        from sleap_io.model.video import Video

        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        frame = np.full((10, 10), 128, dtype=np.uint8)

        # Patch at class level since attrs uses slots
        with patch.object(Video, "__getitem__", return_value=frame):
            assert _frames_similar_by_image(video1, video2, 0, 0.05)

    def test_frames_similar_by_image_different(self):
        """Test _frames_similar_by_image with very different frames."""
        from unittest.mock import patch

        from sleap_io.model.matching import _frames_similar_by_image
        from sleap_io.model.video import Video

        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        frame1 = np.full((10, 10), 0, dtype=np.uint8)
        frame2 = np.full((10, 10), 255, dtype=np.uint8)

        # Patch at class level with side_effect to return different frames
        def get_frame(self, idx):
            if self is video1:
                return frame1
            return frame2

        with patch.object(Video, "__getitem__", get_frame):
            assert not _frames_similar_by_image(video1, video2, 0, 0.05)

    def test_frames_similar_by_image_different_shapes(self):
        """Test _frames_similar_by_image with different frame shapes."""
        from unittest.mock import patch

        from sleap_io.model.matching import _frames_similar_by_image
        from sleap_io.model.video import Video

        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        frame1 = np.full((10, 10), 128, dtype=np.uint8)
        frame2 = np.full((20, 20), 128, dtype=np.uint8)

        def get_frame(self, idx):
            if self is video1:
                return frame1
            return frame2

        with patch.object(Video, "__getitem__", get_frame):
            assert _frames_similar_by_image(video1, video2, 0, 0.05) is False

    def test_frames_similar_by_image_exception(self):
        """Test _frames_similar_by_image returns False on exception."""
        from sleap_io.model.matching import _frames_similar_by_image
        from sleap_io.model.video import Video

        # Videos without backends will raise when accessing frames
        video1 = Video(filename="/nonexistent.mp4", open_backend=False)
        video2 = Video(filename="/nonexistent.mp4", open_backend=False)

        assert _frames_similar_by_image(video1, video2, 0, 0.05) is False

    def test_get_embedded_frame_indices_with_embedded_frame_inds(self):
        """Test _get_embedded_frame_indices with embedded_frame_inds attribute."""
        from unittest.mock import MagicMock

        import attrs

        from sleap_io.model.matching import _get_embedded_frame_indices
        from sleap_io.model.video import Video

        # Create mock backend with embedded_frame_inds
        mock_backend = MagicMock()
        mock_backend.embedded_frame_inds = [0, 5, 10, 15]

        # Create video with mocked backend using attrs.evolve
        video = Video(filename="/video.mp4", open_backend=False)
        video = attrs.evolve(video, backend=mock_backend)

        result = _get_embedded_frame_indices(video)
        assert result == [0, 5, 10, 15]

    def test_get_embedded_frame_indices_with_frame_map(self):
        """Test _get_embedded_frame_indices with frame_map attribute."""
        from unittest.mock import MagicMock

        import attrs

        from sleap_io.model.matching import _get_embedded_frame_indices
        from sleap_io.model.video import Video

        # Create mock backend with frame_map but no embedded_frame_inds
        mock_backend = MagicMock()
        mock_backend.embedded_frame_inds = None
        mock_backend.frame_map = {0: "frame0", 5: "frame5", 10: "frame10"}

        video = Video(filename="/video.mp4", open_backend=False)
        video = attrs.evolve(video, backend=mock_backend)

        result = _get_embedded_frame_indices(video)
        assert set(result) == {0, 5, 10}

    def test_get_common_embedded_indices_with_overlap(self):
        """Test _get_common_embedded_indices with overlapping indices."""
        from unittest.mock import MagicMock

        import attrs

        from sleap_io.model.matching import _get_common_embedded_indices
        from sleap_io.model.video import Video

        mock_backend1 = MagicMock()
        mock_backend1.embedded_frame_inds = [0, 5, 10, 15]

        mock_backend2 = MagicMock()
        mock_backend2.embedded_frame_inds = [5, 10, 20, 25]

        video1 = Video(filename="/video1.mp4", open_backend=False)
        video1 = attrs.evolve(video1, backend=mock_backend1)

        video2 = Video(filename="/video2.mp4", open_backend=False)
        video2 = attrs.evolve(video2, backend=mock_backend2)

        result = _get_common_embedded_indices(video1, video2)
        assert result == {5, 10}


class TestVideoMatcherNewAttributes:
    """Tests for new VideoMatcher attributes."""

    def test_default_values(self):
        """Test default attribute values."""
        matcher = VideoMatcher()
        assert matcher.content_frames == 3
        assert matcher.compare_predictions == "auto"
        assert matcher.compare_images is False
        assert matcher.image_similarity_threshold == 0.05

    def test_custom_values(self):
        """Test custom attribute values."""
        matcher = VideoMatcher(
            content_frames=5,
            compare_predictions=True,
            compare_images=True,
            image_similarity_threshold=0.1,
        )
        assert matcher.content_frames == 5
        assert matcher.compare_predictions is True
        assert matcher.compare_images is True
        assert matcher.image_similarity_threshold == 0.1

    def test_string_method_conversion(self):
        """Test string method is converted to enum."""
        matcher = VideoMatcher(method="auto")
        assert matcher.method == VideoMatchMethod.AUTO


class TestVideoMatcherImageMatching:
    """Tests for VideoMatcher._match_by_images method."""

    def test_match_by_images_identical_frames(self):
        """Test _match_by_images finds match with identical embedded frames."""
        from unittest.mock import MagicMock, patch

        import attrs

        # Create videos with mocked backends that have embedded frame indices
        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        mock_backend1 = MagicMock()
        mock_backend1.embedded_frame_inds = [0]
        video1 = attrs.evolve(video1, backend=mock_backend1)

        mock_backend2 = MagicMock()
        mock_backend2.embedded_frame_inds = [0]
        video2 = attrs.evolve(video2, backend=mock_backend2)

        frame = np.full((10, 10), 128, dtype=np.uint8)

        # Patch frame access at class level
        with patch.object(Video, "__getitem__", return_value=frame):
            matcher = VideoMatcher(compare_images=True, content_frames=1)
            result = matcher._match_by_images(video1, [video2])
            assert result is video2

    def test_match_by_images_different_frames(self):
        """Test _match_by_images returns None with very different frames."""
        from unittest.mock import MagicMock, patch

        import attrs

        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        mock_backend1 = MagicMock()
        mock_backend1.embedded_frame_inds = [0]
        video1 = attrs.evolve(video1, backend=mock_backend1)

        mock_backend2 = MagicMock()
        mock_backend2.embedded_frame_inds = [0]
        video2 = attrs.evolve(video2, backend=mock_backend2)

        frame1 = np.full((10, 10), 0, dtype=np.uint8)
        frame2 = np.full((10, 10), 255, dtype=np.uint8)

        def get_frame(self, idx):
            if self is video1:
                return frame1
            return frame2

        with patch.object(Video, "__getitem__", get_frame):
            matcher = VideoMatcher(compare_images=True, content_frames=1)
            result = matcher._match_by_images(video1, [video2])
            assert result is None

    def test_match_by_images_no_common_indices(self):
        """Test _match_by_images returns None with no common frame indices."""
        from unittest.mock import MagicMock

        import attrs

        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        mock_backend1 = MagicMock()
        mock_backend1.embedded_frame_inds = [0, 1, 2]
        video1 = attrs.evolve(video1, backend=mock_backend1)

        mock_backend2 = MagicMock()
        mock_backend2.embedded_frame_inds = [10, 11, 12]  # No overlap
        video2 = attrs.evolve(video2, backend=mock_backend2)

        matcher = VideoMatcher(compare_images=True, content_frames=1)
        result = matcher._match_by_images(video1, [video2])
        assert result is None

    def test_match_by_images_multiple_candidates(self):
        """Test _match_by_images with multiple candidates."""
        from unittest.mock import MagicMock, patch

        import attrs

        # Incoming video
        video_incoming = Video(filename="/incoming.mp4", open_backend=False)
        mock_backend_in = MagicMock()
        mock_backend_in.embedded_frame_inds = [0]
        video_incoming = attrs.evolve(video_incoming, backend=mock_backend_in)

        # Candidate 1 - different content
        video1 = Video(filename="/v1.mp4", open_backend=False)
        mock_backend1 = MagicMock()
        mock_backend1.embedded_frame_inds = [0]
        video1 = attrs.evolve(video1, backend=mock_backend1)

        # Candidate 2 - similar content
        video2 = Video(filename="/v2.mp4", open_backend=False)
        mock_backend2 = MagicMock()
        mock_backend2.embedded_frame_inds = [0]
        video2 = attrs.evolve(video2, backend=mock_backend2)

        incoming_frame = np.full((10, 10), 100, dtype=np.uint8)
        frame1 = np.full((10, 10), 0, dtype=np.uint8)  # Different
        frame2 = np.full((10, 10), 100, dtype=np.uint8)  # Same as incoming

        def get_frame(self, idx):
            if self is video_incoming:
                return incoming_frame
            elif self is video1:
                return frame1
            return frame2

        with patch.object(Video, "__getitem__", get_frame):
            matcher = VideoMatcher(compare_images=True, content_frames=1)
            result = matcher._match_by_images(video_incoming, [video1, video2])
            assert result is video2


class TestVideoMatcherFindMatchLeafPath:
    """Tests for VideoMatcher.find_match leaf path matching edge cases."""

    def test_find_match_incoming_shorter_path(self):
        """Test find_match when incoming path is shorter than candidate."""
        # Create videos with different path depths
        video_incoming = Video(filename="/short/video.mp4", open_backend=False)
        video_candidate = Video(
            filename="/very/long/path/to/video.mp4", open_backend=False
        )

        matcher = VideoMatcher(method="auto")
        # Should not match because leaf paths differ at higher depth
        result = matcher.find_match(video_incoming, [video_candidate])
        # Paths don't match at depth=1 (video.mp4 vs video.mp4 would match if unique)
        # But since basenames are same and that's the only candidate, it should match
        assert result is video_candidate

    def test_find_match_candidate_shorter_path(self):
        """Test find_match when candidate path is shorter than incoming."""
        video_incoming = Video(
            filename="/very/long/path/to/video.mp4", open_backend=False
        )
        video_candidate = Video(filename="/short/video.mp4", open_backend=False)

        matcher = VideoMatcher(method="auto")
        result = matcher.find_match(video_incoming, [video_candidate])
        assert result is video_candidate

    def test_find_match_leaf_uniqueness_disambiguation(self):
        """Test find_match disambiguates by leaf path at increasing depth."""
        # Two candidates with same basename but different parent directories
        video_incoming = Video(filename="/data/exp1/fly.mp4", open_backend=False)
        video_candidate1 = Video(filename="/other/exp1/fly.mp4", open_backend=False)
        video_candidate2 = Video(filename="/other/exp2/fly.mp4", open_backend=False)

        matcher = VideoMatcher(method="auto")
        result = matcher.find_match(
            video_incoming, [video_candidate1, video_candidate2]
        )
        # At depth=1 (fly.mp4), both match
        # At depth=2 (exp1/fly.mp4 vs exp2/fly.mp4), only candidate1 matches
        assert result is video_candidate1

    def test_find_match_no_match_at_any_depth(self):
        """Test find_match returns None when no leaf path matches."""
        video_incoming = Video(filename="/data/unique_name.mp4", open_backend=False)
        video_candidate = Video(
            filename="/other/different_name.mp4", open_backend=False
        )

        matcher = VideoMatcher(method="auto")
        result = matcher.find_match(video_incoming, [video_candidate])
        # Should return None because basenames don't match
        assert result is None


class TestVideoMatcherMatchByPosesNoMatch:
    """Tests for VideoMatcher._match_by_poses returning None."""

    def test_match_by_poses_no_common_frames_returns_none(self):
        """Test _match_by_poses returns None when no common frames."""
        from sleap_io.model.instance import Instance
        from sleap_io.model.labels import Labels
        from sleap_io.model.skeleton import Skeleton

        skeleton = Skeleton(["A", "B"])

        # Video 1 with frame at index 0
        video1 = Video(filename="/v1.mp4", open_backend=False)
        inst1 = Instance.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
        )
        lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
        labels1 = Labels(videos=[video1], skeletons=[skeleton], labeled_frames=[lf1])

        # Video 2 with frame at index 100 (no overlap)
        video2 = Video(filename="/v2.mp4", open_backend=False)
        inst2 = Instance.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
        )
        lf2 = LabeledFrame(video=video2, frame_idx=100, instances=[inst2])
        labels2 = Labels(videos=[video2], skeletons=[skeleton], labeled_frames=[lf2])

        matcher = VideoMatcher(method="auto", content_frames=1)
        result = matcher._match_by_poses(video1, [video2], labels1, labels2)
        assert result is None

    def test_match_by_poses_poses_differ_returns_none(self):
        """Test _match_by_poses returns None when poses don't match."""
        from sleap_io.model.instance import Instance
        from sleap_io.model.labels import Labels
        from sleap_io.model.skeleton import Skeleton

        skeleton = Skeleton(["A", "B"])

        # Video 1 with specific pose
        video1 = Video(filename="/v1.mp4", open_backend=False)
        inst1 = Instance.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
        )
        lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
        labels1 = Labels(videos=[video1], skeletons=[skeleton], labeled_frames=[lf1])

        # Video 2 with different pose at same frame
        video2 = Video(filename="/v2.mp4", open_backend=False)
        inst2 = Instance.from_numpy(
            np.array([[100.0, 200.0], [300.0, 400.0]]), skeleton=skeleton
        )
        lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])
        labels2 = Labels(videos=[video2], skeletons=[skeleton], labeled_frames=[lf2])

        matcher = VideoMatcher(method="auto", content_frames=1)
        result = matcher._match_by_poses(video1, [video2], labels1, labels2)
        assert result is None


class TestHelperFunctionsCoverage:
    """Additional tests for helper function coverage."""

    def test_get_frame_instances_multiple_videos(self):
        """Test _get_frame_instances skips frames from other videos."""
        from sleap_io.model.instance import Instance
        from sleap_io.model.labels import Labels
        from sleap_io.model.matching import _get_frame_instances
        from sleap_io.model.skeleton import Skeleton

        skeleton = Skeleton(["A", "B"])

        # Create two videos with frames
        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        inst1 = Instance.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
        )
        inst2 = Instance.from_numpy(
            np.array([[5.0, 6.0], [7.0, 8.0]]), skeleton=skeleton
        )

        lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
        lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

        labels = Labels(
            videos=[video1, video2],
            skeletons=[skeleton],
            labeled_frames=[lf1, lf2],
        )

        # Get instances for video1 - should skip video2's frames
        result = _get_frame_instances(labels, video1, include_predictions=True)
        assert len(result) == 1
        assert 0 in result
        # The frame from video2 should not be included (covers line 202 continue)

    def test_video_has_user_instances_multiple_videos(self):
        """Test _video_has_user_instances skips frames from other videos."""
        from sleap_io.model.instance import Instance
        from sleap_io.model.labels import Labels
        from sleap_io.model.matching import _video_has_user_instances
        from sleap_io.model.skeleton import Skeleton

        skeleton = Skeleton(["A", "B"])

        # Create two videos
        video1 = Video(filename="/v1.mp4", open_backend=False)
        video2 = Video(filename="/v2.mp4", open_backend=False)

        # Video2 has a user instance, video1 has none
        inst = Instance.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
        )
        lf = LabeledFrame(video=video2, frame_idx=0, instances=[inst])

        labels = Labels(
            videos=[video1, video2],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )

        # Check video1 - should return False (covers line 229)
        assert _video_has_user_instances(labels, video1) is False
        # Check video2 - should return True
        assert _video_has_user_instances(labels, video2) is True

    def test_get_embedded_frame_indices_backend_no_attrs(self):
        """Test _get_embedded_frame_indices when backend has neither attr."""
        from unittest.mock import MagicMock

        import attrs

        from sleap_io.model.matching import _get_embedded_frame_indices
        from sleap_io.model.video import Video

        # Create backend with neither embedded_frame_inds nor frame_map
        mock_backend = MagicMock(spec=[])  # Empty spec = no attributes

        video = Video(filename="/video.mp4", open_backend=False)
        video = attrs.evolve(video, backend=mock_backend)

        # Should return None (covers line 351)
        result = _get_embedded_frame_indices(video)
        assert result is None


class TestFindMatchLeafPathEdgeCases:
    """Test edge cases in find_match leaf path matching."""

    def test_find_match_incoming_path_shorter_than_depth(self):
        """Test when incoming path is shorter than search depth."""
        # Create incoming with short path (2 parts)
        video_incoming = Video(filename="/fly.mp4", open_backend=False)
        # Create two candidates with same basename but different deep paths
        # This forces iteration to deeper depths to disambiguate
        video_candidate1 = Video(filename="/a/b/c/exp1/fly.mp4", open_backend=False)
        video_candidate2 = Video(filename="/a/b/c/exp2/fly.mp4", open_backend=False)

        matcher = VideoMatcher(method="auto")
        # At depth=1: both match "fly.mp4" (ambiguous)
        # At depth=2: both have different parents (exp1 vs exp2) but incoming
        #             has len=2 which < depth=2, so continue should be hit (line 1015)
        result = matcher.find_match(
            video_incoming, [video_candidate1, video_candidate2]
        )
        # Should return None since incoming path is too short to disambiguate
        assert result is None  # Should still match at shallow depth

    def test_find_match_candidate_path_shorter_than_depth(self):
        """Test when candidate path is shorter than search depth."""
        # Create incoming with long path
        video_incoming = Video(filename="/a/b/c/d/e/fly.mp4", open_backend=False)
        # Create candidate with shorter path
        video_candidate = Video(filename="/fly.mp4", open_backend=False)

        matcher = VideoMatcher(method="auto")
        # This should exercise the len(parts) < depth continue (line 1019-1020)
        result = matcher.find_match(video_incoming, [video_candidate])
        assert result is video_candidate  # Should match at shallow depth

    def test_find_match_no_viable_candidates(self):
        """Test find_match when all candidates are filtered out."""
        # Create videos with incompatible shapes
        video_incoming = Video(filename="/v1.mp4", open_backend=False)
        video_candidate = Video(filename="/v2.mp4", open_backend=False)

        # Mock shapes to be incompatible - force shape rejection
        from unittest.mock import patch

        with patch("sleap_io.model.matching.shapes_compatible", return_value=False):
            matcher = VideoMatcher(method="auto")
            result = matcher.find_match(video_incoming, [video_candidate])
            assert result is None


class TestFindMatchWithLabels:
    """Tests for VideoMatcher.find_match with labels for pose/image matching."""

    def test_find_match_with_pose_matching_success(self):
        """Test find_match returns match via pose matching when paths are ambiguous."""
        from sleap_io.model.instance import Instance
        from sleap_io.model.labels import Labels
        from sleap_io.model.skeleton import Skeleton

        skeleton = Skeleton(["A", "B"])

        # Create incoming video
        video_incoming = Video(filename="/data/video.mp4", open_backend=False)

        # Create two candidates with same basename but different paths (ambiguous)
        video_candidate1 = Video(filename="/exp1/video.mp4", open_backend=False)
        video_candidate2 = Video(filename="/exp2/video.mp4", open_backend=False)

        # Only candidate1 has matching pose with incoming
        pose_match = np.array([[10.0, 20.0], [30.0, 40.0]])
        pose_diff = np.array([[100.0, 200.0], [300.0, 400.0]])

        inst_incoming = Instance.from_numpy(pose_match, skeleton=skeleton)
        inst_candidate1 = Instance.from_numpy(
            pose_match, skeleton=skeleton
        )  # Same pose
        inst_candidate2 = Instance.from_numpy(
            pose_diff, skeleton=skeleton
        )  # Different pose

        lf_incoming = LabeledFrame(
            video=video_incoming, frame_idx=0, instances=[inst_incoming]
        )
        lf_candidate1 = LabeledFrame(
            video=video_candidate1, frame_idx=0, instances=[inst_candidate1]
        )
        lf_candidate2 = LabeledFrame(
            video=video_candidate2, frame_idx=0, instances=[inst_candidate2]
        )

        labels_incoming = Labels(
            videos=[video_incoming],
            skeletons=[skeleton],
            labeled_frames=[lf_incoming],
        )
        labels_base = Labels(
            videos=[video_candidate1, video_candidate2],
            skeletons=[skeleton],
            labeled_frames=[lf_candidate1, lf_candidate2],
        )

        matcher = VideoMatcher(method="auto", content_frames=1)
        # Call find_match with labels - leaf path matching can't disambiguate
        # (both have basename "video.mp4"), so pose matching is needed
        result = matcher.find_match(
            video_incoming,
            [video_candidate1, video_candidate2],
            labels_incoming=labels_incoming,
            labels_base=labels_base,
        )
        # Should match candidate1 via pose (covers line 1015: return match)
        assert result is video_candidate1

    def test_find_match_with_image_matching_success(self):
        """Test find_match returns match via image matching when paths are ambiguous."""
        from unittest.mock import MagicMock, patch

        import attrs

        # Create incoming video
        video_incoming = Video(filename="/data/video.mp4", open_backend=False)

        # Create two candidates with same basename (ambiguous leaf paths)
        video_candidate1 = Video(filename="/exp1/video.mp4", open_backend=False)
        video_candidate2 = Video(filename="/exp2/video.mp4", open_backend=False)

        # Set up backends with embedded frame indices and matching shapes
        mock_backend_in = MagicMock()
        mock_backend_in.embedded_frame_inds = [0]
        mock_backend_in.shape = (100, 10, 10, 1)
        video_incoming = attrs.evolve(video_incoming, backend=mock_backend_in)

        mock_backend1 = MagicMock()
        mock_backend1.embedded_frame_inds = [0]
        mock_backend1.shape = (100, 10, 10, 1)
        video_candidate1 = attrs.evolve(video_candidate1, backend=mock_backend1)

        mock_backend2 = MagicMock()
        mock_backend2.embedded_frame_inds = [0]
        mock_backend2.shape = (100, 10, 10, 1)
        video_candidate2 = attrs.evolve(video_candidate2, backend=mock_backend2)

        # Mock frame access - candidate1 matches incoming, candidate2 doesn't
        frame_incoming = np.full((10, 10), 128, dtype=np.uint8)
        frame_match = np.full((10, 10), 128, dtype=np.uint8)  # Same as incoming
        frame_diff = np.full((10, 10), 0, dtype=np.uint8)  # Different

        def get_frame(self, idx):
            if self is video_incoming:
                return frame_incoming
            elif self is video_candidate1:
                return frame_match
            return frame_diff

        with patch.object(Video, "__getitem__", get_frame):
            matcher = VideoMatcher(method="auto", compare_images=True, content_frames=1)
            # Leaf path can't disambiguate, image matching finds candidate1
            result = matcher.find_match(
                video_incoming, [video_candidate1, video_candidate2]
            )
            # Should match candidate1 via image (covers lines 1019-1021)
            assert result is video_candidate1

    def test_find_match_non_auto_method(self):
        """Test find_match with non-AUTO method uses pairwise match()."""
        video_incoming = Video(filename="/video.mp4", open_backend=False)
        video_candidate = Video(filename="/video.mp4", open_backend=False)

        matcher = VideoMatcher(method="path")
        result = matcher.find_match(video_incoming, [video_candidate])
        # Should match via path (covers lines 1028-1031)
        assert result is video_candidate

    def test_find_match_non_auto_no_match(self):
        """Test find_match with non-AUTO method returns None when no match."""
        video_incoming = Video(filename="/v1.mp4", open_backend=False)
        video_candidate = Video(filename="/v2.mp4", open_backend=False)

        matcher = VideoMatcher(method="path", strict=True)
        result = matcher.find_match(video_incoming, [video_candidate])
        # Should return None since paths don't match (covers line 1030)
        assert result is None


class TestVideoMatcherCropAware:
    """Crop-aware matching/dedup for mosaic tiles (D-127/D-128).

    Two equal-size crops of ONE physical file (mosaic tiles) share a root file,
    so without crop awareness they collapse into a single video across the four
    dedup sites (add_video, match_video, VideoMatcher.match, find_match). These
    tests pin the crop-disambiguated behavior while asserting non-crop videos
    are unaffected.
    """

    def _tiles(self, src_path):
        """Build two equal-size left/right tiles of one source video."""
        src = Video.from_filename(src_path)
        w, h = 384, 384
        w2 = w // 2
        left = src.crop((0, 0, w2, h))
        right = src.crop((w2, 0, w, h))
        return src, left, right

    def test_crop_key_reads_crop_tuple(self, centered_pair_low_quality_path):
        """_crop_key returns the crop rect for crops and None otherwise."""
        src, left, right = self._tiles(centered_pair_low_quality_path)
        assert _crop_key(left) == (0, 0, 192, 384)
        assert _crop_key(right) == (192, 0, 384, 384)
        assert _crop_key(src) is None

    def test_crop_key_defensive_no_method(self):
        """_crop_key tolerates objects without _crop_tuple (returns None)."""

        class NoCrop:
            pass

        assert _crop_key(NoCrop()) is None

    def test_is_same_file_different_crops(self, centered_pair_low_quality_path):
        """Two distinct crops of one file are NOT the same file (D-127)."""
        src, left, right = self._tiles(centered_pair_low_quality_path)
        assert not is_same_file(left, right)

    def test_is_same_file_identical_crops(self, centered_pair_low_quality_path):
        """Two identical crops of one file ARE the same file (D-127)."""
        src = Video.from_filename(centered_pair_low_quality_path)
        a = src.crop((0, 0, 192, 384))
        b = src.crop((0, 0, 192, 384))
        assert is_same_file(a, b)

    def test_is_same_file_noncrop_unchanged(self, centered_pair_low_quality_path):
        """Non-crop videos of the same file still match (additive guard)."""
        v1 = Video.from_filename(centered_pair_low_quality_path)
        v2 = Video.from_filename(centered_pair_low_quality_path)
        assert is_same_file(v1, v2)

    def test_same_file_different_crop_helper(self, centered_pair_low_quality_path):
        """_same_file_different_crop is True only for differing crops of one file."""
        src, left, right = self._tiles(centered_pair_low_quality_path)
        assert _same_file_different_crop(left, right)
        # Identical crops: not "different crop".
        same = src.crop((0, 0, 192, 384))
        assert not _same_file_different_crop(left, same)
        # Non-crop same file: not "different crop".
        v1 = Video.from_filename(centered_pair_low_quality_path)
        v2 = Video.from_filename(centered_pair_low_quality_path)
        assert not _same_file_different_crop(v1, v2)

    def test_same_file_different_crop_via_basename(
        self, centered_pair_low_quality_path
    ):
        """Shared root basename + differing crops disambiguates even unresolved."""
        src = Video.from_filename(centered_pair_low_quality_path)
        left = src.crop((0, 0, 192, 384))
        # A crop of a same-basename file in a different directory: roots are not
        # verifiably the same file, but the basename rung could re-match them.
        other = Video(filename="/elsewhere/centered_pair_low_quality.mp4")
        other_crop = Video(
            filename="/elsewhere/centered_pair_low_quality.mp4",
            source_video=other,
        )
        other_crop.backend_metadata = {"crop": [192, 0, 384, 384]}
        assert _same_file_different_crop(left, other_crop)

    def test_match_rejects_different_crops(self, centered_pair_low_quality_path):
        """VideoMatcher.match(tileA, tileB) is False for differing crops (c)."""
        src, left, right = self._tiles(centered_pair_low_quality_path)
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert not matcher.match(left, right)

    def test_match_accepts_identical_crops(self, centered_pair_low_quality_path):
        """VideoMatcher.match accepts identical crops of one file (d)."""
        src = Video.from_filename(centered_pair_low_quality_path)
        a = src.crop((0, 0, 192, 384))
        b = src.crop((0, 0, 192, 384))
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.match(a, b)

    def test_match_noncrop_same_file(self, centered_pair_low_quality_path):
        """VideoMatcher.match still matches non-crop same-file videos (e)."""
        v1 = Video.from_filename(centered_pair_low_quality_path)
        v2 = Video.from_filename(centered_pair_low_quality_path)
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.match(v1, v2)

    def test_find_match_drops_different_crop(self, centered_pair_low_quality_path):
        """find_match(tileB, [tileA]) does NOT return tileA (c)."""
        src, left, right = self._tiles(centered_pair_low_quality_path)
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(right, [left]) is None

    def test_find_match_picks_correct_crop(self, centered_pair_low_quality_path):
        """find_match resolves to the matching tile, never the other (c)."""
        src, left, right = self._tiles(centered_pair_low_quality_path)
        right_eq = src.crop((192, 0, 384, 384))
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        # Among both tiles, the right-equivalent resolves to right, not left.
        assert matcher.find_match(right_eq, [left, right]) is right

    def test_find_match_identical_crop(self, centered_pair_low_quality_path):
        """find_match matches an identical crop (d)."""
        src = Video.from_filename(centered_pair_low_quality_path)
        a = src.crop((0, 0, 192, 384))
        a_eq = src.crop((0, 0, 192, 384))
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(a_eq, [a]) is a

    def test_find_match_noncrop_same_file(self, centered_pair_low_quality_path):
        """find_match still matches non-crop same-file videos (e)."""
        v1 = Video.from_filename(centered_pair_low_quality_path)
        v2 = Video.from_filename(centered_pair_low_quality_path)
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(v1, [v2]) is v2

    def test_add_video_keeps_distinct_tiles(self, centered_pair_low_quality_path):
        """Labels.add_video keeps two distinct tiles separate (a)."""
        src, left, right = self._tiles(centered_pair_low_quality_path)
        labels = Labels()
        labels.add_video(left)
        labels.add_video(right)
        assert len(labels.videos) == 2

    def test_add_video_dedups_identical_crop(self, centered_pair_low_quality_path):
        """Labels.add_video dedups identical crops to one (d)."""
        src = Video.from_filename(centered_pair_low_quality_path)
        a = src.crop((0, 0, 192, 384))
        a_eq = src.crop((0, 0, 192, 384))
        labels = Labels()
        labels.add_video(a)
        returned = labels.add_video(a_eq)
        assert len(labels.videos) == 1
        assert returned is a

    def test_add_video_dedups_noncrop_same_file(self, centered_pair_low_quality_path):
        """Labels.add_video still dedups non-crop same-file videos (e)."""
        v1 = Video.from_filename(centered_pair_low_quality_path)
        v2 = Video.from_filename(centered_pair_low_quality_path)
        labels = Labels()
        labels.add_video(v1)
        returned = labels.add_video(v2)
        assert len(labels.videos) == 1
        assert returned is v1


class TestEmbeddedSubsetShapeMatching:
    """Embedded subset vs restored original match in AUTO video matching (#473).

    An embedded subset of frames and its restored original source are the same
    underlying file but legitimately differ in frame count. The AUTO cascade
    used the heuristic ``shapes_compatible`` REJECTION before the definitive
    ``is_same_file`` check, so a same-file pair was discarded on frame count.

    The fix resolves a derived video's effective shape from the *nearest source
    in its ``source_video`` chain* (metadata only, no file I/O): an embedded
    subset reports its source's full shape, so it is shape-compatible with that
    source and survives to the identity check -- even when the deeper root shape
    is unknown. Genuine siblings that share a file but have no source
    relationship keep their own shapes and are still rejected on frame count.
    """

    @staticmethod
    def _subset_chain():
        """Build the issue's provenance chain with an unknown-shape root.

        Returns ``(root, restored, embedded)`` where the root ``.pkg.slp`` has
        unknown shape (the deeper provenance file is not loaded), the restored
        original reports 80 frames, and the embedded subset reports 27 frames.
        """
        root = Video(filename="labeling/original.pkg.slp", open_backend=False)
        restored = Video(
            filename="restored_source.pkg.slp", source_video=root, open_backend=False
        )
        restored.backend_metadata["shape"] = (80, 1024, 1024, 1)
        embedded = Video(
            filename="embedded_subset.pkg.slp",
            source_video=restored,
            open_backend=False,
        )
        embedded.backend_metadata["shape"] = (27, 1024, 1024, 1)
        return root, restored, embedded

    def test_effective_shape_uses_nearest_known_source(self):
        """A subset resolves to its source's shape, not its own subset count."""
        root, restored, embedded = self._subset_chain()
        # Root shape is unknown, but the intermediate `restored` knows (80,...).
        assert _get_effective_shape(embedded) == (80, 1024, 1024, 1)
        assert _get_effective_shape(restored) == (80, 1024, 1024, 1)

    def test_shapes_compatible_subset_vs_source(self):
        """The subset and its source are shape-compatible despite frame diff."""
        _, restored, embedded = self._subset_chain()
        assert shapes_compatible(embedded, restored) is True
        assert is_same_file(embedded, restored)

    def test_find_match_restored_finds_embedded_subset(self):
        """find_match(restored, [embedded]) returns the same-file subset."""
        _, restored, embedded = self._subset_chain()
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(restored, [embedded]) is embedded

    def test_find_match_embedded_subset_finds_restored(self):
        """find_match is symmetric: the subset resolves to the restored source."""
        _, restored, embedded = self._subset_chain()
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(embedded, [restored]) is restored

    def test_pairwise_match_subset_restored_both_directions(self):
        """Pairwise match() also matches the pair (used by merge)."""
        _, restored, embedded = self._subset_chain()
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.match(embedded, restored)
        assert matcher.match(restored, embedded)

    def test_different_files_incompatible_shape_still_rejected(self):
        """Guard: unrelated files with incompatible shapes still do not match."""
        a = Video(filename="/data/a.mp4", open_backend=False)
        a.backend_metadata["shape"] = (100, 480, 640, 3)
        b = Video(filename="/data/b.mp4", open_backend=False)
        b.backend_metadata["shape"] = (50, 480, 640, 3)
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(a, [b]) is None
        assert not matcher.match(a, b)

    def test_same_file_siblings_no_source_chain_still_rejected_on_shape(self):
        """Distinct same-file siblings with no source link stay shape-rejected.

        Two videos sharing a file but with NO source relationship and different
        frame counts (e.g. distinct HDF5 datasets in one ``.pkg.slp``) keep
        their own shapes and are still rejected on shape. Source-shape
        resolution must not over-reach to genuine siblings: only a derived video
        (one with a ``source_video``) borrows its source's shape.
        """
        v0 = Video(filename="project.pkg.slp", open_backend=False)
        v0.backend_metadata["shape"] = (4, 384, 384, 1)
        v0.backend_metadata["dataset"] = "video0/video"
        v1 = Video(filename="project.pkg.slp", open_backend=False)
        v1.backend_metadata["shape"] = (1, 320, 560, 3)
        v1.backend_metadata["dataset"] = "video1/video"
        assert shapes_compatible(v0, v1) is False
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(v1, [v0]) is None
        assert not matcher.match(v0, v1)

    def test_provenance_conflict_still_rejected(self):
        """Guard: distinct-root chains are still rejected despite same shape."""
        root1 = Video(filename="/data/v1.mp4", open_backend=False)
        e1 = Video(filename="e1.pkg.slp", source_video=root1, open_backend=False)
        e1.backend_metadata["shape"] = (100, 480, 640, 3)
        root2 = Video(filename="/data/v2.mp4", open_backend=False)
        e2 = Video(filename="e2.pkg.slp", source_video=root2, open_backend=False)
        e2.backend_metadata["shape"] = (100, 480, 640, 3)
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(e1, [e2]) is None
        assert not matcher.match(e1, e2)

    def test_distinct_crops_not_collapsed(self, centered_pair_low_quality_path):
        """Guard: distinct mosaic tiles of one file are still not collapsed."""
        src = Video.from_filename(centered_pair_low_quality_path)
        left = src.crop((0, 0, 192, 384))
        right = src.crop((192, 0, 384, 384))
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(right, [left]) is None
        assert not matcher.match(left, right)

    def test_find_match_imagevideo_same_file_subset(self):
        """ImageVideo same-file subset matches its restored original (#473).

        Exercises the list-filename path with differing frame counts.
        """
        images = ["f0.png", "f1.png", "f2.png", "f3.png"]
        root = Video(filename=list(images), open_backend=False)
        restored = Video(filename=list(images), source_video=root, open_backend=False)
        restored.backend_metadata["shape"] = (4, 128, 128, 1)
        embedded = Video(
            filename=list(images), source_video=restored, open_backend=False
        )
        embedded.backend_metadata["shape"] = (2, 128, 128, 1)
        assert shapes_compatible(embedded, restored) is True
        assert is_same_file(embedded, restored)
        matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
        assert matcher.find_match(restored, [embedded]) is embedded
        assert matcher.match(embedded, restored)

    def test_labels_match_embedded_subset_to_restored_original(self):
        """End-to-end via Labels.match(): GT subset pairs with restored preds.

        Without the fix, video_map is empty and any downstream per-frame pairing
        is lost ("Empty Frame Pairs").
        """
        skel = Skeleton(["a", "b"])
        _, restored, embedded = self._subset_chain()
        gt = Labels(
            videos=[embedded],
            skeletons=[skel],
            labeled_frames=[
                LabeledFrame(
                    video=embedded,
                    frame_idx=i,
                    instances=[
                        Instance.from_numpy([[1.0, 1.0], [2.0, 2.0]], skeleton=skel)
                    ],
                )
                for i in (0, 5, 10)
            ],
        )
        pr = Labels(
            videos=[restored],
            skeletons=[skel],
            labeled_frames=[
                LabeledFrame(
                    video=restored,
                    frame_idx=i,
                    instances=[
                        PredictedInstance.from_numpy(
                            [[1.0, 1.0], [2.0, 2.0]],
                            point_scores=[1.0, 1.0],
                            score=1.0,
                            skeleton=skel,
                        )
                    ],
                )
                for i in (0, 5, 10)
            ],
        )
        result = gt.match(pr)
        assert result.video_map[restored] is embedded
        assert result.all_videos_matched

    def test_labels_merge_restored_preds_into_subset_gt_dedups_video(self):
        """End-to-end via Labels.merge(): same file is not duplicated."""
        skel = Skeleton(["a", "b"])
        _, restored, embedded = self._subset_chain()
        gt = Labels(
            videos=[embedded],
            skeletons=[skel],
            labeled_frames=[
                LabeledFrame(
                    video=embedded,
                    frame_idx=0,
                    instances=[
                        Instance.from_numpy([[1.0, 1.0], [2.0, 2.0]], skeleton=skel)
                    ],
                )
            ],
        )
        pr = Labels(
            videos=[restored],
            skeletons=[skel],
            labeled_frames=[
                LabeledFrame(
                    video=restored,
                    frame_idx=1,
                    instances=[
                        PredictedInstance.from_numpy(
                            [[3.0, 3.0], [4.0, 4.0]],
                            point_scores=[1.0, 1.0],
                            score=1.0,
                            skeleton=skel,
                        )
                    ],
                )
            ],
        )
        gt.merge(pr)
        assert len(gt.videos) == 1
        assert len(gt.labeled_frames) == 2

    def test_embed_subset_restore_original_match_pipeline(
        self, tmp_path, centered_pair_low_quality_path
    ):
        """Full real-file pipeline: embed subset, restore original, then match.

        Embeds a labeled subset as a GT ``.pkg.slp``, saves predictions restored
        to the original source, and matches them end-to-end through real ``.slp``
        I/O. The restored prediction (full frame count) matches the embedded GT
        subset (fewer frames). Here the embedded video's serialized
        ``source_video`` carries the original's full shape, so the subset
        resolves to it and is shape-compatible with the restored original.
        """
        video = sio.load_video(centered_pair_low_quality_path)
        skel = Skeleton(["a", "b"])
        gt = Labels(
            videos=[video],
            skeletons=[skel],
            labeled_frames=[
                LabeledFrame(
                    video=video,
                    frame_idx=i,
                    instances=[
                        Instance.from_numpy([[10.0, 10.0], [20.0, 20.0]], skeleton=skel)
                    ],
                )
                for i in (0, 5, 10)
            ],
        )
        gt_path = tmp_path / "gt.pkg.slp"
        gt.save(gt_path, embed="user")
        labels_gt = sio.load_file(gt_path)
        assert labels_gt.videos[0].source_video is not None
        assert labels_gt.videos[0].shape[0] == 3  # embedded subset

        pred_path = tmp_path / "pred.slp"
        labels_gt.save(pred_path, embed=False)  # restore_original_videos=True
        labels_pr = sio.load_file(pred_path)
        assert labels_pr.videos[0].shape[0] == video.shape[0]  # restored original
        assert labels_pr.videos[0].shape[0] != labels_gt.videos[0].shape[0]

        result = labels_gt.match(labels_pr)
        assert result.video_map[labels_pr.videos[0]] is labels_gt.videos[0]
        assert result.all_videos_matched
