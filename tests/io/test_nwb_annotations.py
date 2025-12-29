"""Tests for NWB annotations functionality."""

from __future__ import annotations

import numpy as np
import pytest
import simplejson as json
from ndx_pose import Skeleton as NwbSkeleton
from ndx_pose import SkeletonInstance as NwbInstance
from ndx_pose import Skeletons as NwbSkeletons

import sleap_io as sio
from sleap_io import Instance as SleapInstance
from sleap_io import Labels as SleapLabels
from sleap_io import Skeleton as SleapSkeleton
from sleap_io import Video as SleapVideo
from sleap_io.io.nwb_annotations import (
    MULTISUBJECTS_AVAILABLE,
    FrameInfo,
    FrameMap,
    create_nwb_to_slp_skeleton_map,
    create_nwb_to_slp_video_map,
    create_slp_to_nwb_skeleton_map,
    create_slp_to_nwb_video_map,
    export_labeled_frames,
    export_labels,
    extract_unique_subjects,
    load_labels,
    nwb_image_series_to_sleap_video,
    nwb_pose_training_to_sleap_labels,
    nwb_skeleton_instance_to_sleap_instance,
    nwb_skeleton_to_sleap_skeleton,
    nwb_source_videos_to_sleap_videos,
    nwb_training_frame_to_sleap_labeled_frame,
    nwb_training_frames_to_sleap_labeled_frames,
    save_labels,
    sleap_instance_to_nwb_skeleton_instance,
    sleap_labeled_frame_to_nwb_training_frame,
    sleap_labeled_frames_to_nwb_training_frames,
    sleap_labels_to_nwb_pose_training,
    sleap_skeleton_to_nwb_skeleton,
    sleap_video_to_nwb_image_series,
    sleap_videos_to_nwb_source_videos,
)

# Conditionally import multisubjects functions (requires Python 3.9+)
if MULTISUBJECTS_AVAILABLE:
    from sleap_io.io.nwb_annotations import create_subjects_table

from sleap_io.io.utils import sanitize_filename
from sleap_io.model.instance import Track
from sleap_io.model.labeled_frame import LabeledFrame

# Skip marker for tests requiring multisubjects (Python 3.9+)
requires_multisubjects = pytest.mark.skipif(
    not MULTISUBJECTS_AVAILABLE,
    reason="ndx-multisubjects requires Python 3.9+",
)


def test_sleap_skeleton_to_nwb_skeleton_basic():
    """Test basic conversion from sleap-io Skeleton to ndx-pose Skeleton."""
    # Create sleap-io skeleton
    sleap_skeleton = SleapSkeleton(
        nodes=["nose", "head", "neck", "body"],
        edges=[("nose", "head"), ("head", "neck"), ("neck", "body")],
        name="test_skeleton",
    )

    # Test conversion
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(sleap_skeleton)

    # Verify nodes
    assert list(nwb_skeleton.nodes) == ["nose", "head", "neck", "body"]

    # Verify edges (should be array of indices)
    expected_edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(nwb_skeleton.edges, expected_edges)

    # Verify name
    assert nwb_skeleton.name == "test_skeleton"


def test_sleap_skeleton_to_nwb_skeleton_no_name():
    """Test conversion when sleap skeleton has no name."""
    sleap_skeleton = SleapSkeleton(nodes=["a", "b"], edges=[("a", "b")])

    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(sleap_skeleton)

    # Should use default name
    assert nwb_skeleton.name == "skeleton"


def test_sleap_skeleton_to_nwb_skeleton_empty():
    """Test conversion with empty skeleton."""
    sleap_skeleton = SleapSkeleton(nodes=[], edges=[], name="empty")

    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(sleap_skeleton)

    assert list(nwb_skeleton.nodes) == []
    assert nwb_skeleton.edges.shape == (0, 2)
    assert nwb_skeleton.name == "empty"


def test_nwb_skeleton_to_sleap_skeleton_basic():
    """Test basic conversion from ndx-pose Skeleton to sleap-io Skeleton."""
    # Create ndx-pose skeleton
    nwb_skeleton = NwbSkeleton(
        name="test_skeleton",
        nodes=["nose", "head", "neck", "body"],
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint8),
    )

    # Convert to sleap-io
    sleap_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    # Verify nodes
    assert sleap_skeleton.node_names == ["nose", "head", "neck", "body"]

    # Verify edges
    expected_edges = [(0, 1), (1, 2), (2, 3)]
    assert sleap_skeleton.edge_inds == expected_edges

    # Verify name
    assert sleap_skeleton.name == "test_skeleton"


def test_nwb_skeleton_to_sleap_skeleton_empty():
    """Test conversion with empty ndx-pose skeleton."""
    nwb_skeleton = NwbSkeleton(
        name="empty", nodes=[], edges=np.array([], dtype=np.uint8).reshape(0, 2)
    )

    sleap_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    assert sleap_skeleton.node_names == []
    assert sleap_skeleton.edge_inds == []
    assert sleap_skeleton.name == "empty"


def test_skeleton_roundtrip_conversion():
    """Test that roundtrip conversion preserves skeleton structure."""
    # Create original sleap-io skeleton
    original_skeleton = SleapSkeleton(
        nodes=["nose", "head", "neck", "left_shoulder", "right_shoulder", "body"],
        edges=[
            ("nose", "head"),
            ("head", "neck"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("neck", "body"),
        ],
        name="complex_skeleton",
    )

    # Convert to ndx-pose and back
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(original_skeleton)
    recovered_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    # Verify structure is preserved
    assert recovered_skeleton.node_names == original_skeleton.node_names
    assert recovered_skeleton.edge_inds == original_skeleton.edge_inds
    assert recovered_skeleton.name == original_skeleton.name

    # Verify skeletons match
    assert recovered_skeleton.matches(original_skeleton, require_same_order=True)


def test_skeleton_roundtrip_with_symmetries():
    """Test that skeleton conversion handles symmetries correctly."""
    # Create skeleton with symmetries
    original_skeleton = SleapSkeleton(
        nodes=["nose", "left_eye", "right_eye", "body"],
        edges=[("nose", "left_eye"), ("nose", "right_eye"), ("nose", "body")],
        symmetries=[("left_eye", "right_eye")],
        name="symmetric_skeleton",
    )

    # Convert to ndx-pose and back
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(original_skeleton)
    recovered_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    # Note: ndx-pose doesn't support symmetries, so they will be lost
    assert recovered_skeleton.node_names == original_skeleton.node_names
    assert recovered_skeleton.edge_inds == original_skeleton.edge_inds
    assert recovered_skeleton.name == original_skeleton.name
    assert len(recovered_skeleton.symmetries) == 0  # Symmetries are lost


def test_sleap_instance_to_nwb_skeleton_instance_basic():
    """Test basic conversion from sleap-io Instance to ndx-pose SkeletonInstance."""
    # Create skeleton and instance
    skeleton = SleapSkeleton(
        nodes=["nose", "head", "body"], edges=[("nose", "head"), ("head", "body")]
    )

    # Create NWB skeleton
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)

    # Create instance with some visible and invisible points
    points_array = np.array([[10.0, 20.0], [15.0, 25.0], [np.nan, np.nan]])
    sleap_instance = SleapInstance.from_numpy(points_array, skeleton)

    # Convert to NWB
    nwb_skeleton_instance = sleap_instance_to_nwb_skeleton_instance(
        sleap_instance, nwb_skeleton
    )

    # Verify node locations
    expected_locations = np.array([[10.0, 20.0], [15.0, 25.0], [np.nan, np.nan]])
    np.testing.assert_array_equal(
        nwb_skeleton_instance.node_locations, expected_locations
    )

    # Verify node visibility
    expected_visibility = np.array([True, True, False])
    np.testing.assert_array_equal(
        nwb_skeleton_instance.node_visibility, expected_visibility
    )

    # Verify defaults
    assert nwb_skeleton_instance.name == "skeleton_instance"
    assert nwb_skeleton_instance.id is None


def test_sleap_instance_to_nwb_skeleton_instance_custom_params():
    """Test conversion with custom name and id parameters."""
    skeleton = SleapSkeleton(nodes=["a", "b"], edges=[("a", "b")])
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)
    points_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    sleap_instance = SleapInstance.from_numpy(points_array, skeleton)

    nwb_skeleton_instance = sleap_instance_to_nwb_skeleton_instance(
        sleap_instance, nwb_skeleton, name="custom_instance", id=42
    )

    assert nwb_skeleton_instance.name == "custom_instance"
    assert nwb_skeleton_instance.id == 42


def test_sleap_instance_to_nwb_skeleton_instance_all_invisible():
    """Test conversion with all invisible points."""
    skeleton = SleapSkeleton(nodes=["a", "b"], edges=[])
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)
    points_array = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    sleap_instance = SleapInstance.from_numpy(points_array, skeleton)

    nwb_skeleton_instance = sleap_instance_to_nwb_skeleton_instance(
        sleap_instance, nwb_skeleton
    )

    # All locations should be NaN
    assert np.isnan(nwb_skeleton_instance.node_locations).all()

    # All visibility should be False
    expected_visibility = np.array([False, False])
    np.testing.assert_array_equal(
        nwb_skeleton_instance.node_visibility, expected_visibility
    )


def test_nwb_skeleton_instance_to_sleap_instance_basic():
    """Test basic conversion from ndx-pose SkeletonInstance to sleap-io Instance."""
    skeleton = SleapSkeleton(
        nodes=["nose", "head", "body"], edges=[("nose", "head"), ("head", "body")]
    )

    # Create NWB skeleton
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)

    # Create NWB skeleton instance
    node_locations = np.array([[10.0, 20.0], [15.0, 25.0], [30.0, 40.0]])
    node_visibility = np.array([True, True, False])

    nwb_skeleton_instance = NwbInstance(
        node_locations=node_locations,
        skeleton=nwb_skeleton,
        name="test_instance",
        id=np.uint32(123),
        node_visibility=node_visibility,
    )

    # Convert to sleap-io
    sleap_instance = nwb_skeleton_instance_to_sleap_instance(
        nwb_skeleton_instance, skeleton
    )

    # Verify points - invisible points should be NaN
    expected_points = np.array([[10.0, 20.0], [15.0, 25.0], [np.nan, np.nan]])
    np.testing.assert_array_equal(sleap_instance.numpy(), expected_points)

    # Verify visibility
    expected_visibility = np.array([True, True, False])
    np.testing.assert_array_equal(sleap_instance.points["visible"], expected_visibility)


def test_nwb_skeleton_instance_to_sleap_instance_no_visibility():
    """Test conversion when node_visibility is None (inferred from NaN)."""
    skeleton = SleapSkeleton(nodes=["a", "b", "c"], edges=[])
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)

    # Create NWB skeleton instance with NaN values but no explicit visibility
    node_locations = np.array([[1.0, 2.0], [np.nan, np.nan], [5.0, 6.0]])

    nwb_skeleton_instance = NwbInstance(
        node_locations=node_locations,
        skeleton=nwb_skeleton,
        name="test_instance",
        node_visibility=None,  # Will be inferred
    )

    sleap_instance = nwb_skeleton_instance_to_sleap_instance(
        nwb_skeleton_instance, skeleton
    )

    # Verify points
    expected_points = np.array([[1.0, 2.0], [np.nan, np.nan], [5.0, 6.0]])
    np.testing.assert_array_equal(sleap_instance.numpy(), expected_points)

    # Verify inferred visibility
    expected_visibility = np.array([True, False, True])
    np.testing.assert_array_equal(sleap_instance.points["visible"], expected_visibility)


def test_instance_roundtrip_conversion():
    """Test that roundtrip conversion preserves instance data."""
    # Create original instance
    skeleton = SleapSkeleton(
        nodes=["nose", "left_eye", "right_eye", "body"],
        edges=[("nose", "left_eye"), ("nose", "right_eye"), ("nose", "body")],
    )

    points_array = np.array(
        [
            [10.0, 20.0],  # nose - visible
            [5.0, 15.0],  # left_eye - visible
            [np.nan, np.nan],  # right_eye - invisible
            [10.0, 30.0],  # body - visible
        ]
    )
    original_instance = SleapInstance.from_numpy(points_array, skeleton)

    # Convert to NWB and back
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)
    nwb_skeleton_instance = sleap_instance_to_nwb_skeleton_instance(
        original_instance, nwb_skeleton, name="test_roundtrip", id=0
    )
    recovered_instance = nwb_skeleton_instance_to_sleap_instance(
        nwb_skeleton_instance, skeleton
    )

    # Verify points are preserved
    np.testing.assert_array_equal(recovered_instance.numpy(), original_instance.numpy())

    # Verify visibility is preserved
    np.testing.assert_array_equal(
        recovered_instance.points["visible"], original_instance.points["visible"]
    )

    # Verify skeleton is the same
    assert recovered_instance.skeleton is skeleton


def test_instance_roundtrip_all_visible():
    """Test roundtrip with all visible points."""
    skeleton = SleapSkeleton(nodes=["a", "b", "c"], edges=[])
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)
    points_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    original_instance = SleapInstance.from_numpy(points_array, skeleton)

    nwb_skeleton_instance = sleap_instance_to_nwb_skeleton_instance(
        original_instance, nwb_skeleton
    )
    recovered_instance = nwb_skeleton_instance_to_sleap_instance(
        nwb_skeleton_instance, skeleton
    )

    np.testing.assert_array_equal(recovered_instance.numpy(), original_instance.numpy())
    assert recovered_instance.n_visible == 3


def test_video_roundtrip_media_video(centered_pair_low_quality_path):
    """Test roundtrip conversion for MediaVideo backend."""
    # Create original video
    original_video = SleapVideo.from_filename(centered_pair_low_quality_path)

    # Convert to ImageSeries and back
    image_series = sleap_video_to_nwb_image_series(original_video, name="test_media")
    recovered_video = nwb_image_series_to_sleap_video(image_series)

    # Verify file path is preserved
    assert str(recovered_video.filename) == str(original_video.filename)


def test_video_roundtrip_image_video(centered_pair_frame_paths):
    """Test roundtrip conversion for ImageVideo backend."""
    # Create original video
    original_video = SleapVideo.from_filename(centered_pair_frame_paths)

    # Convert to ImageSeries and back
    image_series = sleap_video_to_nwb_image_series(original_video, name="test_images")
    recovered_video = nwb_image_series_to_sleap_video(image_series)

    # Verify file path is preserved (should be the directory path for ImageVideo)
    assert str(recovered_video.filename) == str(original_video.filename)


def test_source_videos_roundtrip_media_video(centered_pair_low_quality_path):
    """Test SourceVideos roundtrip conversion with MediaVideo backend."""
    # Create Video object
    video = SleapVideo.from_filename(centered_pair_low_quality_path)

    # Convert to NWB SourceVideos and back
    nwb_source_videos = sleap_videos_to_nwb_source_videos([video])
    recovered_videos = nwb_source_videos_to_sleap_videos(nwb_source_videos)

    # Check that we got one video back
    assert len(recovered_videos) == 1
    recovered_video = recovered_videos[0]

    # Check video properties match
    assert str(recovered_video.filename) == str(video.filename)
    assert recovered_video.shape == video.shape


def test_source_videos_roundtrip_image_video(centered_pair_frame_paths):
    """Test SourceVideos roundtrip conversion with ImageVideo backend."""
    # Create Video object
    video = SleapVideo.from_filename(centered_pair_frame_paths)

    # Convert to NWB SourceVideos and back
    nwb_source_videos = sleap_videos_to_nwb_source_videos([video])
    recovered_videos = nwb_source_videos_to_sleap_videos(nwb_source_videos)

    # Check that we got one video back
    assert len(recovered_videos) == 1
    recovered_video = recovered_videos[0]

    # Check video properties match
    assert recovered_video.filename == video.filename
    assert recovered_video.shape == video.shape


def test_source_videos_multiple_videos(
    centered_pair_low_quality_path, centered_pair_frame_paths
):
    """Test SourceVideos conversion with multiple videos."""
    # Create Video objects
    video1 = SleapVideo.from_filename(centered_pair_low_quality_path)
    video2 = SleapVideo.from_filename(centered_pair_frame_paths)

    # Convert to NWB SourceVideos and back
    nwb_source_videos = sleap_videos_to_nwb_source_videos([video1, video2])
    recovered_videos = nwb_source_videos_to_sleap_videos(nwb_source_videos)

    # Check that we got both videos back
    assert len(recovered_videos) == 2

    # Check first video
    assert str(recovered_videos[0].filename) == str(video1.filename)
    assert recovered_videos[0].shape == video1.shape

    # Check second video
    assert recovered_videos[1].filename == video2.filename
    assert recovered_videos[1].shape == video2.shape

    # Check that videos were named correctly in the container
    assert "video_0" in nwb_source_videos.image_series
    assert "video_1" in nwb_source_videos.image_series


def test_training_frame_roundtrip(slp_real_data):
    """Test TrainingFrame roundtrip conversion."""
    # Get a labeled frame with instances
    labels = sio.load_slp(slp_real_data)
    labeled_frame = labels.labeled_frames[0]

    # Convert skeleton to NWB and create mapping
    nwb_skeleton_list = [
        sleap_skeleton_to_nwb_skeleton(skeleton) for skeleton in labels.skeletons
    ]
    nwb_skeletons = NwbSkeletons(name="Skeletons", skeletons=nwb_skeleton_list)
    slp_to_nwb_skeleton_map = create_slp_to_nwb_skeleton_map(
        labels.skeletons, nwb_skeletons
    )

    # Convert video to ImageSeries
    source_video = sleap_video_to_nwb_image_series(
        labeled_frame.video, name="test_video"
    )

    # Convert to NWB TrainingFrame
    nwb_training_frame = sleap_labeled_frame_to_nwb_training_frame(
        labeled_frame,
        slp_to_nwb_skeleton_map=slp_to_nwb_skeleton_map,
        source_video=source_video,
        name="test_frame",
        annotator="test_annotator",
    )

    # Convert back to sleap-io
    nwb_to_slp_skeleton_map = create_nwb_to_slp_skeleton_map(
        nwb_skeletons, labels.skeletons
    )
    recovered_frame = nwb_training_frame_to_sleap_labeled_frame(
        nwb_training_frame, nwb_to_slp_skeleton_map, labeled_frame.video
    )

    # Check frame properties
    assert recovered_frame.frame_idx == labeled_frame.frame_idx
    assert len(recovered_frame.instances) == len(labeled_frame.instances)
    assert recovered_frame.video == labeled_frame.video

    # Check instance data
    for orig_inst, recovered_inst in zip(
        labeled_frame.instances, recovered_frame.instances
    ):
        np.testing.assert_array_equal(
            orig_inst.numpy(invisible_as_nan=True),
            recovered_inst.numpy(invisible_as_nan=True),
            err_msg="Instance points should match",
        )


def test_training_frames_roundtrip(slp_real_data):
    """Test TrainingFrames roundtrip conversion."""
    # Get labeled frames
    labels = sio.load_slp(slp_real_data)
    labeled_frames = labels.labeled_frames[:3]  # Use first 3 frames

    # Convert skeleton to NWB and create containers
    nwb_skeleton_list = [
        sleap_skeleton_to_nwb_skeleton(skeleton) for skeleton in labels.skeletons
    ]
    nwb_skeletons = NwbSkeletons(skeletons=nwb_skeleton_list)

    # Create source videos container
    source_videos = sleap_videos_to_nwb_source_videos(labels.videos)

    # Create mappings
    slp_to_nwb_video_map = create_slp_to_nwb_video_map(labels.videos, source_videos)
    nwb_to_slp_video_map = create_nwb_to_slp_video_map(
        list(source_videos.image_series.values()), labels.videos
    )
    slp_to_nwb_skeleton_map = create_slp_to_nwb_skeleton_map(
        labels.skeletons, nwb_skeletons
    )
    nwb_to_slp_skeleton_map = create_nwb_to_slp_skeleton_map(
        nwb_skeletons, labels.skeletons
    )

    # Convert to NWB TrainingFrames
    nwb_training_frames = sleap_labeled_frames_to_nwb_training_frames(
        labeled_frames,
        slp_to_nwb_skeleton_map=slp_to_nwb_skeleton_map,
        slp_to_nwb_video_map=slp_to_nwb_video_map,
        name="test_frames",
        annotator="test_annotator",
    )

    # Convert back to sleap-io
    recovered_frames = nwb_training_frames_to_sleap_labeled_frames(
        nwb_training_frames, nwb_to_slp_skeleton_map, nwb_to_slp_video_map
    )

    # Check frame count
    assert len(recovered_frames) == len(labeled_frames)

    # Check each frame
    for orig_frame, recovered_frame in zip(labeled_frames, recovered_frames):
        assert recovered_frame.frame_idx == orig_frame.frame_idx
        assert len(recovered_frame.instances) == len(orig_frame.instances)
        assert str(recovered_frame.video.filename) == str(orig_frame.video.filename)

        # Check instance data
        for orig_inst, recovered_inst in zip(
            orig_frame.instances, recovered_frame.instances
        ):
            np.testing.assert_array_equal(
                orig_inst.numpy(invisible_as_nan=True),
                recovered_inst.numpy(invisible_as_nan=True),
                err_msg="Instance points should match",
            )


def test_pose_training_roundtrip(slp_real_data):
    """Test PoseTraining roundtrip conversion."""
    # Load original labels
    original_labels = sio.load_slp(slp_real_data)

    # Use first few frames to keep test manageable
    limited_labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=original_labels.labeled_frames[:3],
    )

    # Convert to NWB PoseTraining
    nwb_pose_training, nwb_skeletons = sleap_labels_to_nwb_pose_training(
        limited_labels, name="test_pose_training", annotator="test_annotator"
    )

    # Convert back to sleap-io Labels
    recovered_labels = nwb_pose_training_to_sleap_labels(
        nwb_pose_training, nwb_skeletons
    )

    # Check skeletons
    assert len(recovered_labels.skeletons) == len(limited_labels.skeletons)
    for orig_skeleton, recovered_skeleton in zip(
        limited_labels.skeletons, recovered_labels.skeletons
    ):
        assert recovered_skeleton.node_names == orig_skeleton.node_names
        assert recovered_skeleton.edge_inds == orig_skeleton.edge_inds
        assert recovered_skeleton.name == orig_skeleton.name

    # Check videos
    assert len(recovered_labels.videos) == len(limited_labels.videos)
    for orig_video, recovered_video in zip(
        limited_labels.videos, recovered_labels.videos
    ):
        assert str(recovered_video.filename) == str(orig_video.filename)
        assert recovered_video.shape == orig_video.shape

    # Check labeled frames
    assert len(recovered_labels.labeled_frames) == len(limited_labels.labeled_frames)

    for orig_frame, recovered_frame in zip(
        limited_labels.labeled_frames, recovered_labels.labeled_frames
    ):
        assert recovered_frame.frame_idx == orig_frame.frame_idx
        assert len(recovered_frame.instances) == len(orig_frame.instances)
        assert str(recovered_frame.video.filename) == str(orig_frame.video.filename)

        # Check instance data
        for orig_inst, recovered_inst in zip(
            orig_frame.instances, recovered_frame.instances
        ):
            np.testing.assert_array_equal(
                orig_inst.numpy(invisible_as_nan=True),
                recovered_inst.numpy(invisible_as_nan=True),
                err_msg="Instance points should match",
            )


def test_save_load_labels_roundtrip(slp_real_data, tmp_path):
    """Test save_labels and load_labels roundtrip."""
    # Load original labels
    original_labels = sio.load_slp(slp_real_data)

    # Use first few frames to keep test manageable
    limited_labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=original_labels.labeled_frames[:2],
    )

    # Test with minimal parameters (required only)
    nwb_path = tmp_path / "test_minimal.nwb"
    save_labels(limited_labels, nwb_path)

    # Load back and verify
    recovered_labels = load_labels(nwb_path)

    # Check basic structure
    assert len(recovered_labels.skeletons) == len(limited_labels.skeletons)
    assert len(recovered_labels.videos) == len(limited_labels.videos)
    assert len(recovered_labels.labeled_frames) == len(limited_labels.labeled_frames)

    # Test with custom parameters
    nwb_path_custom = tmp_path / "test_custom.nwb"
    save_labels(
        limited_labels,
        nwb_path_custom,
        session_description="Custom test session",
        identifier="custom_test_id",
        annotator="test_user",
        nwb_kwargs={
            "session_id": "custom_session_001",
            "experimenter": ["Test User"],
            "lab": "Test Lab",
            "institution": "Test University",
        },
    )

    # Load back and verify custom metadata preserved in conversion
    recovered_labels_custom = load_labels(nwb_path_custom)
    assert len(recovered_labels_custom.skeletons) == len(limited_labels.skeletons)
    assert len(recovered_labels_custom.videos) == len(limited_labels.videos)
    assert len(recovered_labels_custom.labeled_frames) == len(
        limited_labels.labeled_frames
    )


def test_pose_training_structure(slp_real_data):
    """Test that PoseTraining has the expected structure."""
    # Load labels and convert
    labels = sio.load_slp(slp_real_data)
    limited_labels = SleapLabels(
        skeletons=labels.skeletons,
        videos=labels.videos,
        labeled_frames=labels.labeled_frames[:2],
    )

    nwb_pose_training, nwb_skeletons = sleap_labels_to_nwb_pose_training(
        limited_labels, name="test_structure"
    )

    # Check that PoseTraining has the expected components
    assert nwb_pose_training.name == "test_structure"
    assert hasattr(nwb_pose_training, "training_frames")
    assert hasattr(nwb_pose_training, "source_videos")

    # Check training frames structure
    assert len(nwb_pose_training.training_frames.training_frames) == 2

    # Check source videos structure
    assert len(nwb_pose_training.source_videos.image_series) == len(labels.videos)

    # Check that each training frame has a source video reference
    for training_frame in nwb_pose_training.training_frames.training_frames.values():
        assert training_frame.source_video is not None
        assert (
            training_frame.source_video
            in nwb_pose_training.source_videos.image_series.values()
        )


def test_pose_training_with_annotator(slp_real_data):
    """Test PoseTraining conversion with annotator information."""
    labels = sio.load_slp(slp_real_data)
    limited_labels = SleapLabels(
        skeletons=labels.skeletons,
        videos=labels.videos,
        labeled_frames=labels.labeled_frames[:1],
    )

    nwb_pose_training, nwb_skeletons = sleap_labels_to_nwb_pose_training(
        limited_labels, name="annotated_data", annotator="expert_annotator"
    )

    # Check that annotator information is preserved
    training_frame = next(
        iter(nwb_pose_training.training_frames.training_frames.values())
    )
    assert training_frame.annotator == "expert_annotator"


def test_export_labeled_frames(slp_real_data, tmp_path):
    """Test export_labeled_frames function."""
    # Load original labels
    original_labels = sio.load_slp(slp_real_data)

    # Use first few frames to keep test manageable
    limited_labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=original_labels.labeled_frames[:3],
    )

    # Test basic export
    export_dir = tmp_path / "export_test"
    export_dir.mkdir(parents=True, exist_ok=True)

    mjpeg_path = export_dir / "test_frames.avi"
    frame_map_path = export_dir / "test_frame_map.json"

    frame_map = export_labeled_frames(
        limited_labels,
        frame_map_path=frame_map_path,
        mjpeg_path=mjpeg_path,
    )

    # Check that files were created
    assert mjpeg_path.exists()
    assert frame_map_path.exists()

    # Check FrameMap structure
    assert isinstance(frame_map, FrameMap)
    assert len(frame_map.frames) == 3  # Should match number of labeled frames
    assert len(frame_map.videos) == len(limited_labels.videos)
    assert frame_map.mjpeg_filename == sanitize_filename(str(mjpeg_path))
    assert frame_map.frame_map_filename == sanitize_filename(str(frame_map_path))

    # Check frame mapping data
    for i, frame_info in enumerate(frame_map.frames):
        labeled_frame = limited_labels.labeled_frames[i]
        assert frame_info.video_ind == limited_labels.videos.index(labeled_frame.video)
        assert frame_info.frame_idx == labeled_frame.frame_idx

    # Check JSON file content
    with open(frame_map_path, "r") as f:
        json_data = json.load(f)

    assert "videos" in json_data
    assert "frames" in json_data
    assert len(json_data["frames"]) == 3
    assert len(json_data["videos"]) == len(limited_labels.videos)

    # Test with NWB filename parameter
    nwb_export_dir = tmp_path / "export_with_nwb"
    nwb_export_dir.mkdir(parents=True, exist_ok=True)

    nwb_path = tmp_path / "test.nwb"
    nwb_mjpeg_path = nwb_export_dir / "test_frames_nwb.avi"
    nwb_frame_map_path = nwb_export_dir / "test_frame_map_nwb.json"

    frame_map_with_nwb = export_labeled_frames(
        limited_labels,
        frame_map_path=nwb_frame_map_path,
        mjpeg_path=nwb_mjpeg_path,
        nwb_path=nwb_path,
    )

    assert frame_map_with_nwb.nwb_filename == sanitize_filename(str(nwb_path))

    # Test with labels containing empty frames
    labels_with_empty = SleapLabels(
        skeletons=limited_labels.skeletons,
        videos=limited_labels.videos,
        labeled_frames=[
            sio.LabeledFrame(video=limited_labels.videos[0], frame_idx=0, instances=[]),
            *limited_labels.labeled_frames[:2],
        ],
    )

    # Should export all frames (including empty ones)
    empty_export_dir = tmp_path / "export_with_empty"
    empty_export_dir.mkdir(parents=True, exist_ok=True)

    empty_mjpeg_path = empty_export_dir / "test_frames_empty.avi"
    empty_frame_map_path = empty_export_dir / "test_frame_map_empty.json"

    frame_map_with_empty = export_labeled_frames(
        labels_with_empty,
        frame_map_path=empty_frame_map_path,
        mjpeg_path=empty_mjpeg_path,
    )
    assert len(frame_map_with_empty.frames) == 3  # Includes empty frame

    # Test with completely empty labels (should create empty video and frame map)
    empty_labels = SleapLabels(
        skeletons=limited_labels.skeletons,
        videos=limited_labels.videos,
        labeled_frames=[],
    )

    error_export_dir = tmp_path / "export_empty"
    error_export_dir.mkdir(parents=True, exist_ok=True)

    empty_mjpeg_path = error_export_dir / "test_frames_empty.avi"
    empty_frame_map_path = error_export_dir / "test_frame_map_empty.json"

    frame_map_empty = export_labeled_frames(
        empty_labels,
        frame_map_path=empty_frame_map_path,
        mjpeg_path=empty_mjpeg_path,
    )
    assert len(frame_map_empty.frames) == 0  # No frames


def test_export_labels(slp_real_data, tmp_path):
    """Test export_labels function."""
    # Load original labels
    original_labels = sio.load_slp(slp_real_data)

    # Use first few frames to keep test manageable
    limited_labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=original_labels.labeled_frames[:3],
    )

    # Test basic export
    export_dir = tmp_path / "export_labels_test"

    export_labels(
        limited_labels,
        output_dir=export_dir,
        mjpeg_filename="annotated.avi",
        frame_map_filename="map.json",
        nwb_filename="training.nwb",
        clean=False,  # Don't clean since we have no predictions
    )

    # Check that all files were created
    assert (export_dir / "annotated.avi").exists()
    assert (export_dir / "map.json").exists()
    assert (export_dir / "training.nwb").exists()

    # Load and verify the NWB file
    loaded_labels = load_labels(export_dir / "training.nwb")

    # Check that labels were correctly exported and loaded
    assert len(loaded_labels.labeled_frames) == 3
    assert len(loaded_labels.videos) == 1  # Should have single MJPEG video
    assert loaded_labels.videos[0].filename == sanitize_filename(
        str(export_dir / "annotated.avi")
    )

    # Verify frame indices are updated to sequential
    for i, lf in enumerate(loaded_labels.labeled_frames):
        assert lf.frame_idx == i
        assert lf.video == loaded_labels.videos[0]

    # Test with custom filenames
    export_dir2 = tmp_path / "export_custom"

    export_labels(
        limited_labels,
        output_dir=export_dir2,
        mjpeg_filename="custom_video.avi",
        frame_map_filename="custom_map.json",
        nwb_filename="custom_training.nwb",
        clean=False,
    )

    assert (export_dir2 / "custom_video.avi").exists()
    assert (export_dir2 / "custom_map.json").exists()
    assert (export_dir2 / "custom_training.nwb").exists()

    # The original labels already contain predictions, so use them directly

    # Test with clean=True (should remove predictions from original data)
    export_dir3 = tmp_path / "export_clean"

    export_labels(
        original_labels,  # Use original labels which have predictions
        output_dir=export_dir3,
        clean=True,  # Should remove predicted instances
    )

    loaded_clean = load_labels(export_dir3 / "pose_training.nwb")

    # Check that predictions were removed
    # The clean=True flag should have removed predicted instances
    assert len(loaded_clean.labeled_frames) > 0  # Should have some frames left

    # Test with empty frames that should be removed when clean=True
    labels_with_empty = SleapLabels(
        skeletons=limited_labels.skeletons,
        videos=limited_labels.videos,
        labeled_frames=[
            sio.LabeledFrame(video=limited_labels.videos[0], frame_idx=0, instances=[]),
            *limited_labels.labeled_frames[:2],
        ],
    )

    export_dir4 = tmp_path / "export_no_empty"

    export_labels(
        labels_with_empty,
        output_dir=export_dir4,
        clean=True,  # Should remove empty frames
    )

    loaded_no_empty = load_labels(export_dir4 / "pose_training.nwb")
    assert len(loaded_no_empty.labeled_frames) == 2  # Empty frame removed

    # Test with clean=False to keep empty frames
    export_dir5 = tmp_path / "export_with_empty"

    export_labels(
        labels_with_empty,
        output_dir=export_dir5,
        clean=False,  # Should keep empty frames
    )

    loaded_with_empty = load_labels(export_dir5 / "pose_training.nwb")
    assert len(loaded_with_empty.labeled_frames) == 3  # Empty frame kept

    # Test error handling - completely empty labels after cleaning
    empty_labels = SleapLabels(
        skeletons=limited_labels.skeletons,
        videos=limited_labels.videos,
        labeled_frames=[
            sio.LabeledFrame(video=limited_labels.videos[0], frame_idx=0, instances=[])
        ],
    )

    try:
        export_labels(
            empty_labels,
            output_dir=tmp_path / "export_error",
            clean=True,  # Will result in no frames
        )
        assert False, "Should have raised ValueError for empty labels"
    except ValueError as e:
        assert "No labeled frames found to export" in str(e)


def test_frame_map_json_roundtrip(slp_real_data, tmp_path):
    """Test FrameMap JSON serialization and deserialization roundtrip."""
    # Load original labels
    original_labels = sio.load_slp(slp_real_data)

    # Use first few frames to keep test manageable
    limited_labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=original_labels.labeled_frames[:3],
    )

    # Export labels to create a FrameMap
    export_dir = tmp_path / "frame_map_test"

    export_labels(
        limited_labels,
        output_dir=export_dir,
        mjpeg_filename="test_video.avi",
        frame_map_filename="test_map.json",
        nwb_filename="test_training.nwb",
        clean=False,
    )

    # Check that the JSON file was created
    json_path = export_dir / "test_map.json"
    assert json_path.exists()

    # Load the FrameMap from JSON
    frame_map_loaded = FrameMap.load(json_path)

    # Verify the loaded FrameMap has correct attributes
    # Use Path objects for comparison to handle Windows path separators
    from pathlib import Path as PathLib

    assert PathLib(frame_map_loaded.frame_map_filename) == json_path
    assert PathLib(frame_map_loaded.nwb_filename) == export_dir / "test_training.nwb"
    assert PathLib(frame_map_loaded.mjpeg_filename) == export_dir / "test_video.avi"
    assert len(frame_map_loaded.videos) == len(limited_labels.videos)
    assert len(frame_map_loaded.frames) == 3

    # Verify frame info
    for i, frame_info in enumerate(frame_map_loaded.frames):
        assert frame_info.video_ind == 0  # All frames from first video
        assert frame_info.frame_idx == limited_labels.labeled_frames[i].frame_idx

    # Test roundtrip: save loaded FrameMap back to JSON
    json_path2 = tmp_path / "roundtrip_map.json"
    frame_map_loaded.save(json_path2)

    # Load again and verify
    frame_map_roundtrip = FrameMap.load(json_path2)

    assert frame_map_roundtrip.nwb_filename == frame_map_loaded.nwb_filename
    assert frame_map_roundtrip.mjpeg_filename == frame_map_loaded.mjpeg_filename
    assert len(frame_map_roundtrip.videos) == len(frame_map_loaded.videos)
    assert len(frame_map_roundtrip.frames) == len(frame_map_loaded.frames)

    # Verify videos match
    for v1, v2 in zip(frame_map_loaded.videos, frame_map_roundtrip.videos):
        assert sanitize_filename(v1.filename) == sanitize_filename(v2.filename)

    # Verify frames match
    for f1, f2 in zip(frame_map_loaded.frames, frame_map_roundtrip.frames):
        assert f1.video_ind == f2.video_ind
        assert f1.frame_idx == f2.frame_idx

    # Test with custom FrameMap creation
    custom_videos = [
        SleapVideo(filename="video1.mp4", open_backend=False),
        SleapVideo(filename="video2.mp4", open_backend=False),
    ]

    custom_frames = [
        FrameInfo(video_ind=0, frame_idx=10),
        FrameInfo(video_ind=1, frame_idx=20),
        FrameInfo(video_ind=0, frame_idx=30),
    ]

    custom_frame_map = FrameMap(
        frame_map_filename="custom_map.json",
        nwb_filename="custom.nwb",
        mjpeg_filename="custom.avi",
        videos=custom_videos,
        frames=custom_frames,
    )

    # Save to JSON
    custom_json_path = tmp_path / "custom_map.json"
    custom_frame_map.save(custom_json_path)

    # Load and verify
    custom_loaded = FrameMap.load(custom_json_path)

    assert custom_loaded.frame_map_filename == str(custom_json_path)
    assert custom_loaded.nwb_filename == "custom.nwb"
    assert custom_loaded.mjpeg_filename == "custom.avi"
    assert len(custom_loaded.videos) == 2
    assert len(custom_loaded.frames) == 3

    # Verify custom frame data preserved
    assert custom_loaded.frames[0].video_ind == 0
    assert custom_loaded.frames[0].frame_idx == 10
    assert custom_loaded.frames[1].video_ind == 1
    assert custom_loaded.frames[1].frame_idx == 20
    assert custom_loaded.frames[2].video_ind == 0
    assert custom_loaded.frames[2].frame_idx == 30


def test_unsupported_video_backend(slp_minimal_pkg):
    """Test error handling for unsupported video backend (HDF5Video)."""
    # Load labels with HDF5Video backend
    labels = sio.load_slp(slp_minimal_pkg)

    # The pkg.slp file should have HDF5Video backend
    assert len(labels.videos) > 0
    video = labels.videos[0]

    # Try to convert to NWB ImageSeries - should raise ValueError
    from sleap_io.io.video_reading import HDF5Video

    if isinstance(video.backend, HDF5Video):
        with pytest.raises(ValueError, match="Unsupported video backend"):
            sleap_video_to_nwb_image_series(video)
    else:
        # If not HDF5Video, manually create one for testing
        video.backend = HDF5Video(video.filename)
        with pytest.raises(ValueError, match="Unsupported video backend"):
            sleap_video_to_nwb_image_series(video)


def test_default_name_generation_from_video():
    """Test default name generation when name is None."""
    from sleap_io.io.video_reading import ImageVideo

    # Test with single filename
    video_single = SleapVideo(filename="test_video.mp4", open_backend=False)
    video_single.backend = ImageVideo(["dummy.png"])  # Use ImageVideo backend

    # Call without name - should generate from filename
    image_series = sleap_video_to_nwb_image_series(video_single, name=None)
    assert (
        image_series.name == "test_video.mp4"
    )  # Uses filename as-is (dots are allowed)

    # Test with list of filenames
    video_list = SleapVideo(filename=["frame1.png", "frame2.png"], open_backend=False)
    video_list.backend = ImageVideo(["frame1.png", "frame2.png"])

    # Call without name - should use first filename
    image_series_list = sleap_video_to_nwb_image_series(video_list, name=None)
    assert image_series_list.name == "frame1.png"  # Uses first filename as-is


def test_training_frame_without_source_video():
    """Test that TrainingFrame without source_video is handled correctly."""
    # Create skeleton
    skeleton = SleapSkeleton(nodes=["A", "B"], edges=[(0, 1)])
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(skeleton)

    # Create NWB Skeletons container (not used but would be in real scenario)
    _ = NwbSkeletons(name="Skeletons", skeletons=[nwb_skeleton])

    # Create a simple instance
    instance = SleapInstance.from_numpy(np.array([[0, 0], [1, 1]]), skeleton=skeleton)
    nwb_instance = sleap_instance_to_nwb_skeleton_instance(
        instance, nwb_skeleton, name="instance_0"
    )

    # Create SkeletonInstances container
    from ndx_pose import SkeletonInstances as NwbSkeletonInstances

    skeleton_instances = NwbSkeletonInstances(
        name="skeleton_instances", skeleton_instances=[nwb_instance]
    )

    # Create TrainingFrame WITHOUT source_video (set to None explicitly)
    from ndx_pose import TrainingFrame as NwbTrainingFrame
    from ndx_pose import TrainingFrames as NwbTrainingFrames

    training_frame = NwbTrainingFrame(
        name="frame_0",
        skeleton_instances=skeleton_instances,
        source_video=None,  # No source video
        source_video_frame_index=None,
    )

    # Create TrainingFrames container
    training_frames = NwbTrainingFrames(
        name="training_frames", training_frames=[training_frame]
    )

    # Create skeleton mapping
    nwb_to_slp_skeleton_map = {nwb_skeleton: skeleton}

    # Create a dummy video for the mapping (won't be used due to None source_video)
    _ = SleapVideo(filename="dummy.mp4", open_backend=False)
    nwb_to_slp_video_map = {}  # Empty map since source_video is None

    # This should raise ValueError because source_video is None
    with pytest.raises(ValueError, match="TrainingFrame must have a source_video"):
        nwb_training_frames_to_sleap_labeled_frames(
            training_frames, nwb_to_slp_skeleton_map, nwb_to_slp_video_map
        )


def test_non_external_image_series_format():
    """Test error handling for non-external ImageSeries format."""
    import numpy as np
    from pynwb.image import ImageSeries

    # Create ImageSeries with 'raw' format (embedded data, not external file)
    image_series = ImageSeries(
        name="raw_video",
        description="Test video with raw data",
        data=np.zeros((10, 100, 100)),  # 10 frames of 100x100
        unit="NA",
        format="raw",  # Not 'external' format
        rate=30.0,
    )

    # Try to convert to sleap-io Video - should raise ValueError
    with pytest.raises(ValueError, match="Unsupported ImageSeries format: raw"):
        nwb_image_series_to_sleap_video(image_series)


# ==================== Multi-Subject Tests ====================


@requires_multisubjects
def test_extract_unique_subjects_basic():
    """Test extracting subjects from labels with tracks."""
    skeleton = SleapSkeleton(nodes=["a", "b"], edges=[("a", "b")])
    video = SleapVideo(filename="test.mp4", open_backend=False)

    # Create tracks
    track1 = Track(name="mouse1")
    track2 = Track(name="mouse2")

    # Create instances with tracks
    inst1 = SleapInstance(
        skeleton=skeleton,
        points={"a": [0, 0], "b": [1, 1]},
        track=track1,
    )
    inst2 = SleapInstance(
        skeleton=skeleton,
        points={"a": [2, 2], "b": [3, 3]},
        track=track2,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = SleapLabels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    subjects_data, track_to_index = extract_unique_subjects(labels)

    assert len(subjects_data) == 2
    assert subjects_data[0]["subject_id"] == "mouse1"
    assert subjects_data[1]["subject_id"] == "mouse2"
    assert track_to_index[track1] == 0
    assert track_to_index[track2] == 1


@requires_multisubjects
def test_extract_unique_subjects_no_tracks():
    """Test with labels that have no tracks - should return empty."""
    skeleton = SleapSkeleton(nodes=["a", "b"], edges=[("a", "b")])
    video = SleapVideo(filename="test.mp4", open_backend=False)

    inst = SleapInstance(
        skeleton=skeleton,
        points={"a": [0, 0], "b": [1, 1]},
        track=None,  # No track
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = SleapLabels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    subjects_data, track_to_index = extract_unique_subjects(labels)

    assert len(subjects_data) == 0
    assert len(track_to_index) == 0


@requires_multisubjects
def test_extract_unique_subjects_unnamed_tracks():
    """Test tracks with no name get 'unknown' as subject_id."""
    skeleton = SleapSkeleton(nodes=["a", "b"], edges=[("a", "b")])
    video = SleapVideo(filename="test.mp4", open_backend=False)

    track = Track(name=None)  # Unnamed track

    inst = SleapInstance(
        skeleton=skeleton,
        points={"a": [0, 0], "b": [1, 1]},
        track=track,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = SleapLabels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    subjects_data, track_to_index = extract_unique_subjects(labels)

    assert len(subjects_data) == 1
    assert subjects_data[0]["subject_id"] == "unknown"


@requires_multisubjects
def test_extract_unique_subjects_same_track_multiple_frames():
    """Test that same Track object maps to same subject across frames."""
    skeleton = SleapSkeleton(nodes=["a", "b"], edges=[("a", "b")])
    video = SleapVideo(filename="test.mp4", open_backend=False)

    track = Track(name="mouse1")

    # Same track in multiple frames
    inst1 = SleapInstance(
        skeleton=skeleton, points={"a": [0, 0], "b": [1, 1]}, track=track
    )
    inst2 = SleapInstance(
        skeleton=skeleton, points={"a": [2, 2], "b": [3, 3]}, track=track
    )

    lf1 = LabeledFrame(video=video, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video, frame_idx=1, instances=[inst2])
    labels = SleapLabels(
        videos=[video], skeletons=[skeleton], labeled_frames=[lf1, lf2]
    )

    subjects_data, track_to_index = extract_unique_subjects(labels)

    # Should only have one subject (same track)
    assert len(subjects_data) == 1
    assert subjects_data[0]["subject_id"] == "mouse1"


@requires_multisubjects
def test_create_subjects_table_basic():
    """Test creating SubjectsTable with minimal data and default values."""
    subjects_data = [{"subject_id": "mouse1"}, {"subject_id": "mouse2"}]

    table = create_subjects_table(subjects_data)

    assert len(table) == 2
    assert table["subject_id"][0] == "mouse1"
    assert table["subject_id"][1] == "mouse2"
    # Check required fields have defaults
    assert table["sex"][0] == "U"  # Unknown
    assert table["species"][0] == "unknown"


@requires_multisubjects
def test_create_subjects_table_with_metadata():
    """Test creating SubjectsTable with full metadata."""
    subjects_data = [{"subject_id": "mouse1"}, {"subject_id": "mouse2"}]
    metadata = [
        {"sex": "M", "species": "Mus musculus", "age": "P30D"},
        {"sex": "F", "species": "Mus musculus", "age": "P45D"},
    ]

    table = create_subjects_table(subjects_data, subjects_metadata=metadata)

    assert len(table) == 2
    assert table["sex"][0] == "M"
    assert table["sex"][1] == "F"
    assert table["species"][0] == "Mus musculus"
    assert table["age"][0] == "P30D"


@requires_multisubjects
def test_save_labels_multisubjects_no_tracks(tmp_path):
    """Test error when use_multisubjects=True but no tracks exist."""
    skeleton = SleapSkeleton(nodes=["a", "b"], edges=[("a", "b")], name="skeleton")
    video = SleapVideo(filename=str(tmp_path / "test.mp4"), open_backend=False)

    # Create instance without track
    inst = SleapInstance(
        skeleton=skeleton,
        points={"a": [0, 0], "b": [1, 1]},
        track=None,
    )

    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = SleapLabels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    # Create a dummy video file
    (tmp_path / "test.mp4").touch()

    nwb_path = tmp_path / "test.nwb"

    with pytest.raises(ValueError, match="No tracked instances found"):
        save_labels(labels, str(nwb_path), use_multisubjects=True)


@requires_multisubjects
def test_save_labels_multisubjects_partial_tracks_warning(slp_real_data, tmp_path):
    """Test warning when some instances have tracks and some don't."""
    import warnings

    # Load real data with working video backends
    original_labels = sio.load_slp(slp_real_data)

    # Use first frame and add one tracked + one untracked instance
    track = Track(name="tracked_mouse")

    # Get instances from first frame, add track to first, leave second untracked
    lf = original_labels.labeled_frames[0]
    if len(lf.instances) >= 2:
        lf.instances[0].track = track
        lf.instances[1].track = None  # Ensure second is untracked
    else:
        # Create a second instance if needed
        inst2 = SleapInstance(
            skeleton=lf.instances[0].skeleton,
            points={n.name: [0, 0] for n in lf.instances[0].skeleton.nodes},
            track=None,
        )
        lf.instances[0].track = track
        lf.instances.append(inst2)

    labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=[lf],
    )

    nwb_path = tmp_path / "test.nwb"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        save_labels(labels, str(nwb_path), use_multisubjects=True)

        # Check that warning was raised about untracked instances
        warning_messages = [str(warning.message) for warning in w]
        untracked_warning = [m for m in warning_messages if "do not have tracks" in m]
        assert len(untracked_warning) == 1


@requires_multisubjects
def test_multisubjects_save_load_roundtrip(slp_real_data, tmp_path):
    """Test that pose data survives multi-subject roundtrip with correct linkage."""
    from pynwb import NWBHDF5IO

    # Load real data with working video backends
    original_labels = sio.load_slp(slp_real_data)

    # Create tracks
    track1 = Track(name="mouse1")
    track2 = Track(name="mouse2")

    # Assign tracks to instances in first two frames
    limited_frames = original_labels.labeled_frames[:2]
    for lf in limited_frames:
        for i, inst in enumerate(lf.instances):
            inst.track = track1 if i % 2 == 0 else track2

    labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=limited_frames,
    )

    nwb_path = tmp_path / "test.nwb"

    # Save with multisubjects
    save_labels(labels, str(nwb_path), use_multisubjects=True)

    # Verify the NWB file structure
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()

        # Check SubjectsTable exists
        assert hasattr(nwbfile, "subjects_table")
        assert nwbfile.subjects_table is not None
        assert len(nwbfile.subjects_table) == 2

        # Check subject IDs
        subject_ids = list(nwbfile.subjects_table["subject_id"][:])
        assert "mouse1" in subject_ids
        assert "mouse2" in subject_ids

        # Check SkeletonInstance IDs match SubjectsTable row indices
        pose_training = nwbfile.processing["behavior"]["PoseTraining"]
        training_frames = pose_training.training_frames
        frame = training_frames["frame_0"]

        # Get instance IDs
        instance_ids = []
        for inst_name in frame.skeleton_instances.skeleton_instances:
            inst = frame.skeleton_instances.skeleton_instances[inst_name]
            instance_ids.append(inst.id)

        # Instance IDs should be valid SubjectsTable indices
        for inst_id in instance_ids:
            assert 0 <= inst_id < len(subject_ids)

    # Load and verify pose data is preserved
    loaded_labels = load_labels(str(nwb_path))

    assert len(loaded_labels.labeled_frames) == len(limited_frames)
    assert len(loaded_labels.skeletons) == len(labels.skeletons)


@requires_multisubjects
def test_multisubjects_skeleton_instance_id_linkage(slp_real_data, tmp_path):
    """Test that SkeletonInstance.id correctly links to SubjectsTable row."""
    from pynwb import NWBHDF5IO

    # Load real data with working video backends
    original_labels = sio.load_slp(slp_real_data)

    # Create 3 tracks
    tracks = [Track(name=f"subject_{i}") for i in range(3)]

    # Assign tracks to instances across multiple frames
    limited_frames = original_labels.labeled_frames[:2]
    for lf in limited_frames:
        for i, inst in enumerate(lf.instances):
            inst.track = tracks[i % len(tracks)]

    labels = SleapLabels(
        skeletons=original_labels.skeletons,
        videos=original_labels.videos,
        labeled_frames=limited_frames,
    )

    nwb_path = tmp_path / "test.nwb"
    save_labels(labels, str(nwb_path), use_multisubjects=True)

    # Verify linkage
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()

        # Get subject_id to row index mapping
        subject_ids = list(nwbfile.subjects_table["subject_id"][:])

        pose_training = nwbfile.processing["behavior"]["PoseTraining"]

        # Check each frame
        for frame in pose_training.training_frames.training_frames.values():
            for inst in frame.skeleton_instances.skeleton_instances.values():
                # The instance ID should be a valid row index in SubjectsTable
                assert 0 <= inst.id < len(subject_ids)
