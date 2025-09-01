"""Tests for NWB annotations functionality."""

from __future__ import annotations

import numpy as np
from ndx_pose import Skeleton as NwbSkeleton
from ndx_pose import SkeletonInstance as NwbInstance

import sleap_io as sio
from sleap_io import Instance as SleapInstance
from sleap_io import Skeleton as SleapSkeleton
from sleap_io import Video as SleapVideo
from sleap_io.io.nwb_ann import (
    nwb_image_series_to_sleap_video,
    nwb_skeleton_instance_to_sleap_instance,
    nwb_skeleton_to_sleap_skeleton,
    nwb_source_videos_to_sleap_videos,
    nwb_training_frame_to_sleap_labeled_frame,
    sleap_instance_to_nwb_skeleton_instance,
    sleap_labeled_frame_to_nwb_training_frame,
    sleap_skeleton_to_nwb_skeleton,
    sleap_video_to_nwb_image_series,
    sleap_videos_to_nwb_source_videos,
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

    # Convert skeleton to NWB
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(labels.skeleton)

    # Convert video to ImageSeries
    source_video = sleap_video_to_nwb_image_series(
        labeled_frame.video, name="test_video"
    )

    # Convert to NWB TrainingFrame
    nwb_training_frame = sleap_labeled_frame_to_nwb_training_frame(
        labeled_frame,
        nwb_skeleton,
        source_video=source_video,
        name="test_frame",
        annotator="test_annotator",
    )

    # Convert back to sleap-io
    recovered_frame = nwb_training_frame_to_sleap_labeled_frame(
        nwb_training_frame, labels.skeleton, labeled_frame.video
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
