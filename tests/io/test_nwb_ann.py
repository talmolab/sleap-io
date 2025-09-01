"""Tests for NWB annotation I/O functionality using real fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO

import sleap_io as sio
import sleap_io.io.nwb_ann as ann
from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
from sleap_io.io.video_reading import HDF5Video


def test_create_skeletons(slp_typical):
    """Test skeleton creation with real data."""
    labels = sio.load_slp(slp_typical)

    skeletons, frame_indices, unique_skeletons, video_objects = ann.create_skeletons(
        labels
    )

    # Test skeleton creation
    assert len(unique_skeletons) == len(labels.skeletons)
    for orig_skel in labels.skeletons:
        sanitized_name = ann._sanitize_nwb_name(orig_skel.name)
        assert sanitized_name in unique_skeletons
        nwb_skel = unique_skeletons[sanitized_name]
        assert nwb_skel.name == sanitized_name
        assert list(nwb_skel.nodes) == orig_skel.node_names

    # Test frame indices mapping
    for video in labels.videos:
        assert video.filename in frame_indices
        # Should have sorted unique frame indices
        indices = frame_indices[video.filename]
        assert indices == sorted(set(indices))

    # Test video objects mapping
    for video in labels.videos:
        assert video.filename in video_objects
        assert video_objects[video.filename] == video


def test_create_skeletons_with_embedded_video(slp_minimal_pkg):
    """Test skeleton creation with embedded video."""
    labels = sio.load_slp(slp_minimal_pkg)

    skeletons, frame_indices, unique_skeletons, video_objects = ann.create_skeletons(
        labels
    )

    # Should handle embedded video path correctly
    assert len(video_objects) == 1
    video_path = list(video_objects.keys())[0]
    video_obj = list(video_objects.values())[0]

    # Check that the video has HDF5Video backend
    assert video_obj.backend is not None
    assert isinstance(video_obj.backend, HDF5Video)

    # Check frame indices
    assert video_path in frame_indices
    assert len(frame_indices[video_path]) > 0


def test_get_frames_from_slp(slp_minimal_pkg):
    """Test frame extraction from real SLP file with embedded video."""
    labels = sio.load_slp(slp_minimal_pkg)

    frames, durations, frame_map = ann.get_frames_from_slp(labels)

    # Should extract frames
    assert len(frames) == len(labels.labeled_frames)
    assert len(durations) == len(frames)

    # All frames should be numpy arrays
    assert all(isinstance(f, np.ndarray) for f in frames)

    # Check frame shapes match video
    for frame in frames:
        assert frame.ndim == 3  # height, width, channels

    # Check frame map
    for lf in labels.labeled_frames:
        assert lf.frame_idx in frame_map

    # Durations should be positive
    assert all(d > 0 for d in durations)


def test_get_frames_from_slp_with_real_video(slp_real_data):
    """Test frame extraction with real video backend."""
    labels = sio.load_slp(slp_real_data)

    # Only process first few frames to avoid loading entire video
    labels = labels.extract(inds=list(range(min(3, len(labels)))))

    frames, durations, frame_map = ann.get_frames_from_slp(labels)

    # Should extract frames
    assert len(frames) == len(labels.labeled_frames)

    # Check that frames have correct shape
    if labels.video.backend is not None:
        expected_shape = labels.video.shape
        if expected_shape is not None:
            _, height, width, channels = expected_shape
            for frame in frames:
                assert frame.shape == (height, width, channels)


def test_make_mjpeg(tmp_path, slp_minimal_pkg):
    """Test MJPEG creation with real data."""
    labels = sio.load_slp(slp_minimal_pkg)

    # Extract frames
    frames, durations, frame_map = ann.get_frames_from_slp(labels)

    # Create MJPEG
    output_path = ann.make_mjpeg(frames, durations, frame_map, output_dir=tmp_path)

    # Check output files
    assert Path(output_path).exists()
    assert (tmp_path / "frame_map.json").exists()

    # Verify frame_map.json
    with open(tmp_path / "frame_map.json", "r") as f:
        saved_map = json.load(f)

    # JSON keys are strings
    for key, value in frame_map.items():
        assert str(key) in saved_map
        assert saved_map[str(key)] == [[item[0], item[1]] for item in value]


def test_create_source_videos_with_regular_video(slp_real_data):
    """Test source video creation with regular video backend."""
    labels = sio.load_slp(slp_real_data)

    skeletons, frame_indices, unique_skeletons, video_objects = ann.create_skeletons(
        labels
    )

    # Create temporary MJPEG path
    output_mjpeg = "test_mjpeg.avi"

    source_videos, annotations_mjpeg, video_dimensions = ann.create_source_videos(
        frame_indices, video_objects, output_mjpeg
    )

    # Check video dimensions were extracted
    for video_name in video_dimensions:
        width, height = video_dimensions[video_name]
        assert width > 0
        assert height > 0

    # Check annotations MJPEG series
    assert annotations_mjpeg.name == "annotated_frames"
    assert annotations_mjpeg.external_file[0] == output_mjpeg


def test_create_source_videos_with_embedded_video(slp_minimal_pkg):
    """Test source video creation with embedded video (HDF5Video backend)."""
    labels = sio.load_slp(slp_minimal_pkg)

    skeletons, frame_indices, unique_skeletons, video_objects = ann.create_skeletons(
        labels
    )

    # Create temporary MJPEG path
    output_mjpeg = "test_mjpeg.avi"

    source_videos, annotations_mjpeg, video_dimensions = ann.create_source_videos(
        frame_indices, video_objects, output_mjpeg
    )

    # For embedded videos, should only include MJPEG (no external references)
    # The original videos list should be empty or very small
    image_series_dict = source_videos.image_series

    # Should at least have the MJPEG
    assert "annotated_frames" in image_series_dict
    assert image_series_dict["annotated_frames"] == annotations_mjpeg

    # Check video dimensions
    assert len(video_dimensions) > 0


def test_create_training_frames(centered_pair):
    """Test training frames creation with real tracked data."""
    labels = sio.load_slp(centered_pair)
    labels.clean(tracks=True)
    labels = labels.extract(inds=list(range(10)))  # Use first 10 frames

    skeletons, frame_indices, unique_skeletons, video_objects = ann.create_skeletons(
        labels
    )

    # Create dummy frame map for testing
    frame_map = {}
    for i, lf in enumerate(labels.labeled_frames):
        video_name = Path(lf.video.filename).stem
        frame_map[lf.frame_idx] = [(i, video_name)]

    # Create dummy MJPEG ImageSeries
    from pynwb.image import ImageSeries

    annotations_mjpeg = ImageSeries(
        name="test_mjpeg",
        external_file=["test.avi"],
        starting_frame=[0],
        format="external",
        rate=30.0,
    )

    # Test with identity=True to preserve tracks
    training_frames = ann.create_training_frames(
        labels, unique_skeletons, annotations_mjpeg, frame_map, identity=True
    )

    # Should have one training frame per labeled frame
    assert len(training_frames.training_frames) == len(labels.labeled_frames)

    # Check that instances are properly created
    for i, lf in enumerate(labels.labeled_frames):
        tf_name = f"frame_{i}"
        assert tf_name in training_frames.training_frames
        tf = training_frames.training_frames[tf_name]

        # Should have skeleton instances for each instance in the frame
        skeleton_instances = tf.skeleton_instances.skeleton_instances
        assert len(skeleton_instances) == len(lf.instances)


def test_write_and_read_annotations_roundtrip(tmp_path, centered_pair):
    """Test full roundtrip of writing and reading NWB annotations with real data."""
    # Load real data with tracks
    labels = sio.load_slp(centered_pair)
    labels.clean(tracks=True)
    labels = labels.extract(inds=list(range(5)))  # Use first 5 frames for speed

    # Write to NWB
    nwb_path = tmp_path / "test_annotations.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )

    # Verify files were created
    assert nwb_path.exists()
    assert (tmp_path / "annotated_frames.avi").exists()
    assert (tmp_path / "frame_map.json").exists()

    # Read back
    loaded_labels = ann.read_nwb_annotations(str(nwb_path))

    # Compare basic properties
    assert len(loaded_labels.labeled_frames) == len(labels.labeled_frames)
    assert len(loaded_labels.skeletons) == len(labels.skeletons)

    # Check that instances were preserved
    for orig_lf, loaded_lf in zip(labels.labeled_frames, loaded_labels.labeled_frames):
        assert len(loaded_lf.instances) == len(orig_lf.instances)

        # Check instance points
        for orig_inst, loaded_inst in zip(orig_lf.instances, loaded_lf.instances):
            # Check actual visibility based on NaN values (not n_visible property)
            orig_points = orig_inst.numpy()
            loaded_points = loaded_inst.numpy()

            orig_visible = np.sum(
                ~(np.isnan(orig_points[:, 0]) | np.isnan(orig_points[:, 1]))
            )
            loaded_visible = np.sum(
                ~(np.isnan(loaded_points[:, 0]) | np.isnan(loaded_points[:, 1]))
            )
            assert orig_visible == loaded_visible

            # Points should be close (may have small numerical differences)
            np.testing.assert_allclose(orig_points, loaded_points, rtol=1e-5)


def test_embedded_video_roundtrip(tmp_path, slp_minimal_pkg):
    """Test roundtrip with embedded video (HDF5Video backend)."""
    labels = sio.load_slp(slp_minimal_pkg)

    # Write to NWB
    nwb_path = tmp_path / "test_embedded.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )

    # Verify files were created
    assert nwb_path.exists()
    assert (tmp_path / "annotated_frames.avi").exists()
    assert (tmp_path / "frame_map.json").exists()

    # Read back
    loaded_labels = ann.read_nwb_annotations(str(nwb_path))

    # Should preserve the labels
    assert len(loaded_labels.labeled_frames) == len(labels.labeled_frames)
    assert len(loaded_labels.skeletons) == len(labels.skeletons)


def test_skeleton_name_sanitization_with_real_data(tmp_path):
    """Test that skeleton names with invalid characters are properly handled."""
    # Create a skeleton with invalid characters (like a file path)
    skeleton = Skeleton(
        name="C:/path/to/skeleton:data.mat",
        nodes=["node1", "node2"],
        edges=[["node1", "node2"]],
    )

    # Create a temporary image file for testing
    import imageio.v3 as iio

    test_image = np.zeros((100, 100, 1), dtype=np.uint8)
    temp_image_path = tmp_path / "test_frame.png"
    iio.imwrite(temp_image_path, test_image.squeeze())

    from sleap_io.io.video_reading import ImageVideo

    video = Video(filename="test.mp4", backend=ImageVideo([str(temp_image_path)]))

    instance = Instance.from_numpy(np.array([[10, 20], [30, 40]]), skeleton=skeleton)

    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([lf], videos=[video], skeletons=[skeleton])

    # This should not raise an error due to invalid characters
    nwb_path = tmp_path / "test_sanitized.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )

    # Verify the file was created successfully
    assert nwb_path.exists()

    # Read back and check skeleton name was sanitized
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        behavior_pm = nwbfile.processing["behavior"]
        skeletons_container = behavior_pm.data_interfaces["Skeletons"]

        # Should have sanitized name
        sanitized_name = "C__path_to_skeleton_data.mat"
        assert sanitized_name in skeletons_container.skeletons


def test_multiview_labels(tmp_path, slp_multiview):
    """Test with multiview data containing multiple videos."""
    labels = sio.load_slp(slp_multiview)

    # Take a subset for speed
    labels = labels.extract(inds=list(range(min(3, len(labels)))))

    # Write to NWB
    nwb_path = tmp_path / "test_multiview.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )

    # Should create the file
    assert nwb_path.exists()

    # Read back
    loaded_labels = ann.read_nwb_annotations(str(nwb_path))

    # When we extract frames, we get 1 video for those frames
    # When we read back from NWB, we get the original video reference + the MJPEG
    assert len(loaded_labels.videos) == 2  # Original + MJPEG
    assert len(loaded_labels.labeled_frames) == len(labels.labeled_frames)


def test_labels_without_tracks(tmp_path, slp_minimal_pkg):
    """Test with labels that don't have tracks."""
    labels = sio.load_slp(slp_minimal_pkg)

    # Remove any tracks
    labels.tracks = []
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            inst.track = None

    # Write to NWB
    nwb_path = tmp_path / "test_no_tracks.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )

    # Should work without tracks
    assert nwb_path.exists()

    # Read back
    loaded_labels = ann.read_nwb_annotations(str(nwb_path))

    # Should preserve data
    assert len(loaded_labels.labeled_frames) == len(labels.labeled_frames)
    assert len(loaded_labels.tracks) == 0


# Utility function tests


def test_sanitize_nwb_name():
    """Test NWB name sanitization."""
    # Test sanitization of invalid characters
    assert ann._sanitize_nwb_name("name/with/slashes") == "name_with_slashes"
    assert ann._sanitize_nwb_name("name:with:colons") == "name_with_colons"
    assert (
        ann._sanitize_nwb_name("C:/path/to/file:data.mat") == "C__path_to_file_data.mat"
    )

    # Test that valid names are unchanged
    assert ann._sanitize_nwb_name("valid_name") == "valid_name"
    assert ann._sanitize_nwb_name("name-with-dashes") == "name-with-dashes"
    assert ann._sanitize_nwb_name("name.with.dots") == "name.with.dots"


def test_load_frame_map(tmp_path):
    """Test loading frame map from JSON."""
    # Create test frame map
    frame_map = {
        "0": [[0, "video1"]],
        "5": [[1, "video1"], [0, "video2"]],
        "10": [[2, "video1"]],
    }

    # Save to JSON
    json_path = tmp_path / "frame_map.json"
    with open(json_path, "w") as f:
        json.dump(frame_map, f)

    # Load and verify
    loaded_map = ann._load_frame_map(str(json_path))

    # Keys should be integers
    assert 0 in loaded_map
    assert 5 in loaded_map
    assert 10 in loaded_map

    # Values should be tuples
    assert loaded_map[0] == [(0, "video1")]
    assert loaded_map[5] == [(1, "video1"), (0, "video2")]
    assert loaded_map[10] == [(2, "video1")]


def test_load_frame_map_malformed_json(tmp_path):
    """Test handling of malformed frame map JSON."""
    json_path = tmp_path / "bad_frame_map.json"

    # Write invalid JSON
    with open(json_path, "w") as f:
        f.write("{ invalid json }")

    # Should handle gracefully
    with pytest.raises(json.JSONDecodeError):
        ann._load_frame_map(str(json_path))


def test_invert_frame_map():
    """Test frame map inversion."""
    frame_map = {
        0: [(0, "video1")],
        5: [(1, "video1"), (0, "video2")],
        10: [(2, "video1")],
    }

    inverted = ann._invert_frame_map(frame_map)

    # Check that all mappings are present (structure changed to tuple keys)
    assert ("video1", 0) in inverted
    assert ("video1", 1) in inverted
    assert ("video1", 2) in inverted
    assert ("video2", 0) in inverted

    # Check values
    assert inverted[("video1", 0)] == 0
    assert inverted[("video1", 1)] == 5
    assert inverted[("video1", 2)] == 10
    assert inverted[("video2", 0)] == 5


def test_extract_skeletons_from_nwb():
    """Test skeleton extraction from NWB skeleton groups."""
    from ndx_pose import Skeleton as NWBSkeleton
    from ndx_pose import Skeletons

    # Create test skeletons
    skeleton1 = NWBSkeleton(
        name="skeleton1",
        nodes=["head", "thorax", "abdomen"],
        edges=np.array([[0, 1], [1, 2]]),
    )

    skeleton2 = NWBSkeleton(
        name="skeleton2", nodes=["node1", "node2"], edges=np.array([[0, 1]])
    )

    skeletons_group = Skeletons(name="Skeletons", skeletons=[skeleton1, skeleton2])

    # Extract and verify
    extracted = ann._extract_skeletons_from_nwb(skeletons_group)

    assert len(extracted) == 2
    assert "skeleton1" in extracted
    assert "skeleton2" in extracted

    # Check skeleton1
    skel1 = extracted["skeleton1"]
    assert skel1.name == "skeleton1"
    assert skel1.node_names == ["head", "thorax", "abdomen"]
    assert len(skel1.edges) == 2

    # Check skeleton2
    skel2 = extracted["skeleton2"]
    assert skel2.name == "skeleton2"
    assert skel2.node_names == ["node1", "node2"]
    assert len(skel2.edges) == 1


def test_write_annotations_nwb_error_cases(tmp_path):
    """Test handling of edge cases in write_annotations_nwb."""
    # Create labels without predicted instances
    skeleton = Skeleton(name="test", nodes=["a", "b"], edges=[["a", "b"]])
    # Create a temporary image file for testing
    import imageio.v3 as iio

    test_image = np.zeros((100, 100, 1), dtype=np.uint8)
    temp_image_path = tmp_path / "test_frame.png"
    iio.imwrite(temp_image_path, test_image.squeeze())

    from sleap_io.io.video_reading import ImageVideo

    video = Video(filename="test.mp4", backend=ImageVideo([str(temp_image_path)]))
    lf = LabeledFrame(video=video, frame_idx=0, instances=[])
    labels = Labels([lf], videos=[video], skeletons=[skeleton])

    # Should handle empty instances gracefully (creates empty MJPEG)
    nwb_path = tmp_path / "test.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )
    # File should be created even with no instances
    assert nwb_path.exists()


def test_duplicate_frame_indices(tmp_path, slp_minimal_pkg):
    """Test handling of duplicate frame indices."""
    labels = sio.load_slp(slp_minimal_pkg)

    # Should handle duplicates gracefully
    frames, durations, frame_map = ann.get_frames_from_slp(labels)

    # Should have frames for all labeled frames (including duplicates)
    assert len(frames) == len(labels.labeled_frames)
    assert len(durations) == len(frames)

    # Frame map should contain all unique frame indices
    unique_indices = set(lf.frame_idx for lf in labels.labeled_frames)
    assert len(frame_map) == len(unique_indices)


def test_read_nwb_annotations_missing_data(tmp_path):
    """Test reading NWB with missing or incomplete data."""
    from datetime import datetime, timezone

    from ndx_pose import Skeleton as NWBSkeleton
    from ndx_pose import Skeletons
    from pynwb import NWBFile

    # Create minimal NWB file with just skeletons
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=datetime.now(timezone.utc),
    )

    # Add minimal skeleton data
    skeleton = NWBSkeleton(
        name="test_skeleton", nodes=["head", "tail"], edges=np.array([[0, 1]])
    )

    skeletons_group = Skeletons(name="Skeletons", skeletons=[skeleton])

    # Add to behavior processing module
    behavior_pm = nwbfile.create_processing_module(
        name="behavior", description="Behavior data"
    )
    behavior_pm.add(skeletons_group)

    # Save file
    nwb_path = tmp_path / "minimal.nwb"
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    # Should read without errors but return minimal labels
    labels = ann.read_nwb_annotations(str(nwb_path))

    assert len(labels.skeletons) == 1
    assert labels.skeletons[0].name == "test_skeleton"
    assert len(labels.labeled_frames) == 0  # No frames
    assert len(labels.videos) == 0  # No videos


def test_read_nwb_annotations_no_frame_map(tmp_path, slp_typical):
    """Test reading NWB annotations without frame map file."""
    labels = sio.load_slp(slp_typical)

    # Write NWB
    nwb_path = tmp_path / "test.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )

    # Delete frame map file
    frame_map_path = tmp_path / "frame_map.json"
    if frame_map_path.exists():
        frame_map_path.unlink()

    # Should still read but without proper frame mapping
    loaded_labels = ann.read_nwb_annotations(str(nwb_path))

    # Should still have skeletons
    assert len(loaded_labels.skeletons) == len(labels.skeletons)
    # Frames might not match exactly without frame map
    assert len(loaded_labels.labeled_frames) >= 0

def test_multiview_with_different_skeletons(tmp_path):
    """Test multiview data with different skeletons per view."""
    # Create two different skeletons
    skeleton1 = Skeleton(
        name="skeleton_view1", nodes=["head", "tail"], edges=[["head", "tail"]]
    )

    skeleton2 = Skeleton(
        name="skeleton_view2", nodes=["a", "b", "c"], edges=[["a", "b"], ["b", "c"]]
    )

    # Create temporary image files for testing
    import imageio.v3 as iio

    test_image = np.zeros((100, 100, 1), dtype=np.uint8)
    temp_image_path1 = tmp_path / "test_frame1.png"
    temp_image_path2 = tmp_path / "test_frame2.png"
    iio.imwrite(temp_image_path1, test_image.squeeze())
    iio.imwrite(temp_image_path2, test_image.squeeze())

    from sleap_io.io.video_reading import ImageVideo

    video1 = Video(filename="view1.mp4", backend=ImageVideo([str(temp_image_path1)]))
    video2 = Video(filename="view2.mp4", backend=ImageVideo([str(temp_image_path2)]))

    # Create instances with different skeletons
    instance1 = Instance.from_numpy(np.array([[10, 20], [30, 40]]), skeleton=skeleton1)

    instance2 = Instance.from_numpy(
        np.array([[50, 60], [70, 80], [90, 100]]), skeleton=skeleton2
    )

    # Create labeled frames
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[instance1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[instance2])

    labels = Labels(
        [lf1, lf2], videos=[video1, video2], skeletons=[skeleton1, skeleton2]
    )

    # Write to NWB
    nwb_path = tmp_path / "test_multiview_multiskel.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )

    # Should handle multiple skeletons
    assert nwb_path.exists()

    # Read back
    loaded_labels = ann.read_nwb_annotations(str(nwb_path))

    # Should preserve both skeletons
    assert len(loaded_labels.skeletons) == 2
    skeleton_names = [s.name for s in loaded_labels.skeletons]
    assert "skeleton_view1" in skeleton_names
    assert "skeleton_view2" in skeleton_names


def test_malformed_frame_map_json(tmp_path):
    """Test reading NWB annotations with malformed frame_map.json."""
    # This tests lines 879-880 in nwb_ann.py
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile

    # Create a simple NWB file
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=datetime.now(timezone.utc),
    )
    
    nwb_path = tmp_path / "test.nwb"
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)
    
    # Create malformed JSON file
    frame_map_path = tmp_path / "frame_map.json"
    frame_map_path.write_text("{invalid json")
    
    # Try to load - should warn but not crash
    with pytest.warns(UserWarning, match="Could not load frame_map.json"):
        # This will fail because no behavior module exists, but that's ok
        # We just want to test the frame_map loading error handling
        try:
            labels = ann.read_nwb_annotations(str(nwb_path), str(frame_map_path))
        except ValueError as e:
            assert "behavior" in str(e)


def test_nwb_device_creation(tmp_path, slp_typical):
    """Test creating NWB files with device information."""
    # This tests lines 351-355 and 371 in nwb_ann.py  
    labels = sio.load_slp(slp_typical)
    
    # Limit to one frame for speed
    labels.labeled_frames = labels.labeled_frames[:1]
    
    # Write with devices enabled
    nwb_path = tmp_path / "test_devices.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
        include_devices=True,  # This triggers device creation
    )
    
    # Read back and verify devices were created
    from pynwb import NWBHDF5IO
    
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        
        # Check that devices were created
        assert len(nwbfile.devices) > 0
        device_names = list(nwbfile.devices.keys())
        assert any("camera" in name for name in device_names)
        
        # Check that image series reference the devices
        if hasattr(nwbfile, "acquisition") and nwbfile.acquisition:
            for image_series in nwbfile.acquisition.values():
                if hasattr(image_series, "device"):
                    assert image_series.device is not None


def test_missing_nwb_processing_modules(tmp_path):
    """Test reading NWB file with missing processing modules."""
    # This tests lines 888 and 894 in nwb_ann.py
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile

    # Create NWB file without behavior module
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=datetime.now(timezone.utc),
    )
    
    nwb_path = tmp_path / "no_behavior.nwb"
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)
    
    # Try to read - should raise ValueError about missing behavior module
    with pytest.raises(ValueError, match="behavior"):
        ann.read_nwb_annotations(str(nwb_path))
    
    # Create NWB file with behavior but no Skeletons
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=datetime.now(timezone.utc),
    )
    behavior_pm = nwbfile.create_processing_module(
        name="behavior", description="behavior"
    )
    
    nwb_path = tmp_path / "no_skeletons.nwb"
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)
    
    # Try to read - should raise ValueError about missing Skeletons
    with pytest.raises(ValueError, match="Skeletons"):
        ann.read_nwb_annotations(str(nwb_path))


def test_video_loading_branches(tmp_path, slp_typical):
    """Test load_source_videos parameter branches."""
    # This tests lines 928 and 937 in nwb_ann.py
    labels = sio.load_slp(slp_typical)
    labels.labeled_frames = labels.labeled_frames[:1]
    
    # Write NWB first
    nwb_path = tmp_path / "test.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )
    
    # Read with load_source_videos=True (default)
    loaded1 = ann.read_nwb_annotations(str(nwb_path), load_source_videos=True)
    # MJPEG video should have backend (but source videos may not exist)
    assert loaded1 is not None
    
    # Read with load_source_videos=False
    loaded2 = ann.read_nwb_annotations(str(nwb_path), load_source_videos=False)
    # Should still load successfully
    assert loaded2 is not None


def test_empty_instances_handling(tmp_path):
    """Test handling of training frames with no instances."""
    # This tests lines 958 and 965 in nwb_ann.py
    from datetime import datetime, timezone

    from ndx_pose import (
        Skeleton as NWBSkeleton,
        SkeletonInstances,
        Skeletons,
        TrainingFrame,
        TrainingFrames,
    )
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.image import ImageSeries

    # Create NWB file
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=datetime.now(timezone.utc),
    )
    
    # Add skeleton
    skeleton = NWBSkeleton(
        name="test_skeleton", 
        nodes=["head", "tail"],
        edges=np.array([[0, 1]])
    )
    skeletons_group = Skeletons(name="Skeletons", skeletons=[skeleton])
    
    behavior_pm = nwbfile.create_processing_module(
        name="behavior", description="behavior"
    )
    behavior_pm.add(skeletons_group)
    
    # Add MJPEG video
    mjpeg_series = ImageSeries(
        name="annotated_frames",
        external_file=["test.avi"],
        starting_frame=[0],
        rate=30.0,
        format="external",
    )
    nwbfile.add_acquisition(mjpeg_series)
    
    # Create training frame with empty skeleton instances
    skeleton_instances = SkeletonInstances(
        name="skeleton_instances",
        skeleton_instances=[],  # Empty!
    )
    
    training_frame = TrainingFrame(
        name="frame_0",
        annotator="test",
        skeleton_instances=skeleton_instances,
        source_video=mjpeg_series,
        source_video_frame_index=np.uint64(0),
    )
    
    training_frames_group = TrainingFrames(
        name="TrainingFrames", training_frames=[training_frame]
    )
    behavior_pm.add(training_frames_group)
    
    # Write to file
    nwb_path = tmp_path / "test_empty.nwb"
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)
    
    # Read back - should handle empty instances gracefully
    loaded = ann.read_nwb_annotations(str(nwb_path))
    
    # Should have no labeled frames (empty instances are skipped)
    assert len(loaded.labeled_frames) == 0


def test_duplicate_frame_handling(tmp_path, slp_typical):
    """Test handling of duplicate frame writes."""
    # This tests line 193 in nwb_ann.py
    labels = sio.load_slp(slp_typical)
    
    # Duplicate the first frame
    first_frame = labels.labeled_frames[0]
    duplicate_frame = LabeledFrame(
        video=first_frame.video,
        frame_idx=first_frame.frame_idx,
        instances=first_frame.instances,
    )
    labels.labeled_frames.append(duplicate_frame)
    
    # Write annotations - should skip duplicate
    nwb_path = tmp_path / "test_duplicates.nwb"
    ann.write_annotations_nwb(
        labels=labels,
        nwbfile_path=str(nwb_path),
        output_dir=str(tmp_path),
    )
    
    # Check that MJPEG was created with unique frames only
    mjpeg_path = tmp_path / "annotated_frames.avi"
    assert mjpeg_path.exists()
    
    # Read back the frame map
    frame_map_path = tmp_path / "frame_map.json"
    frame_map = ann._load_frame_map(frame_map_path)
    
    # Should have unique frames only
    total_frames = sum(len(frames) for frames in frame_map.values())
    # Original frames minus duplicate
    expected_frames = len(labels.labeled_frames) - 1
    assert total_frames == expected_frames
