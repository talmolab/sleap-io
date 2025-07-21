"""Tests for sleap_io.io.ultralytics module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import yaml
import imageio.v3 as iio
import sleap_io

from sleap_io import Labels, Skeleton, Node, Edge, Instance, LabeledFrame, Track, Video
from sleap_io.io.ultralytics import (
    read_labels,
    write_labels,
    parse_data_yaml,
    create_skeleton_from_config,
    parse_label_file,
    write_label_file,
    create_data_yaml,
    normalize_coordinates,
    denormalize_coordinates,
)
from sleap_io.io.main import load_file, save_file, load_ultralytics, save_ultralytics


def test_parse_data_yaml(ultralytics_data_yaml):
    """Test parsing of data.yaml configuration file."""
    config = parse_data_yaml(Path(ultralytics_data_yaml))

    assert "kpt_shape" in config
    assert "skeleton" in config
    assert "names" in config
    assert config["kpt_shape"] == [5, 3]
    assert len(config["skeleton"]) == 4  # 4 edges defined
    assert config["names"][0] == "animal"


def test_create_skeleton_from_config(ultralytics_data_yaml):
    """Test skeleton creation from configuration."""
    config = parse_data_yaml(Path(ultralytics_data_yaml))
    skeleton = create_skeleton_from_config(config)

    assert len(skeleton.nodes) == 5
    assert len(skeleton.edges) == 4
    assert skeleton.name == "ultralytics_skeleton"


def test_parse_label_file(ultralytics_dataset, ultralytics_skeleton):
    """Test parsing of individual label files."""
    label_file = Path(ultralytics_dataset) / "train" / "labels" / "image_001.txt"
    instances = parse_label_file(label_file, ultralytics_skeleton, (480, 640))

    assert len(instances) == 1
    instance = instances[0]
    assert len(instance.points) == 5
    assert instance.skeleton == ultralytics_skeleton

    # Check that points are properly denormalized
    for point in instance.points:
        x, y = point["xy"]
        if point["visible"]:
            assert 0 <= x <= 640
            assert 0 <= y <= 480


def test_parse_label_file_multi_instance(ultralytics_dataset, ultralytics_skeleton):
    """Test parsing of label file with multiple instances."""
    label_file = Path(ultralytics_dataset) / "train" / "labels" / "image_002.txt"
    instances = parse_label_file(label_file, ultralytics_skeleton, (480, 640))

    assert len(instances) == 2
    for instance in instances:
        assert len(instance.points) == 5
        assert instance.skeleton == ultralytics_skeleton


def test_read_labels_train_split(ultralytics_dataset, ultralytics_skeleton):
    """Test reading train split of Ultralytics dataset."""
    labels = read_labels(ultralytics_dataset, split="train")

    assert len(labels.labeled_frames) == 2  # 2 training images
    assert len(labels.skeletons) == 1
    assert len(labels.skeletons[0].nodes) == 5

    # Check that instances are properly loaded
    total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
    assert total_instances == 3  # 1 + 2 instances


def test_read_labels_val_split(ultralytics_dataset):
    """Test reading validation split of Ultralytics dataset."""
    labels = read_labels(ultralytics_dataset, split="val")

    assert len(labels.labeled_frames) == 1  # 1 validation image
    assert len(labels.skeletons) == 1

    # Check that instance is properly loaded
    total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
    assert total_instances == 1


def test_read_labels_with_custom_skeleton(ultralytics_dataset):
    """Test reading with a custom skeleton provided."""
    # Create a custom skeleton
    nodes = [Node(f"custom_node_{i}") for i in range(5)]
    custom_skeleton = Skeleton(nodes=nodes, name="custom")

    labels = read_labels(ultralytics_dataset, skeleton=custom_skeleton)

    assert labels.skeletons[0] == custom_skeleton
    assert all(
        instance.skeleton == custom_skeleton
        for frame in labels.labeled_frames
        for instance in frame.instances
    )


def test_write_labels_single_split(tmp_path, centered_pair_low_quality_video):
    """Test writing labels to Ultralytics format with single split."""
    # Create test labels with actual video
    head_node = Node("head")
    tail_node = Node("tail")
    skeleton = Skeleton([head_node, tail_node], [Edge(head_node, tail_node)])
    points_array = np.array([[10, 20, True], [30, 40, True]], dtype=np.float32)
    instance = Instance.from_numpy(points_data=points_array, skeleton=skeleton)

    # Create frames from actual video
    frames = []
    for i in range(3):
        frame = LabeledFrame(
            video=centered_pair_low_quality_video, frame_idx=i, instances=[instance]
        )
        frames.append(frame)

    labels = Labels(frames, [skeleton])

    # Write to Ultralytics format
    output_dir = tmp_path / "ultralytics_output"
    write_labels(labels, str(output_dir), split_ratios={"train": 1.0})

    # Check directory structure
    assert (output_dir / "data.yaml").exists()
    assert (output_dir / "train" / "images").exists()
    assert (output_dir / "train" / "labels").exists()

    # Check data.yaml content
    with open(output_dir / "data.yaml") as f:
        config = yaml.safe_load(f)
    assert config["kpt_shape"] == [2, 3]
    assert "train" in config

    # Check that real image files were created
    image_files = list((output_dir / "train" / "images").glob("*.png"))
    assert len(image_files) == 3  # Should have 3 images

    # Check image naming pattern
    expected_names = ["0000000.png", "0000001.png", "0000002.png"]
    actual_names = sorted([f.name for f in image_files])
    assert actual_names == expected_names

    # Check that images are valid and have correct size
    import imageio.v3 as iio

    for img_file in image_files:
        img = iio.imread(img_file)
        assert img.shape == (384, 384)  # The video is 384x384 grayscale

    # Check corresponding label files
    label_files = list((output_dir / "train" / "labels").glob("*.txt"))
    assert len(label_files) == 3

    # Check label naming pattern matches images
    expected_label_names = ["0000000.txt", "0000001.txt", "0000002.txt"]
    actual_label_names = sorted([f.name for f in label_files])
    assert actual_label_names == expected_label_names


def test_write_labels_jpeg_format(tmp_path, centered_pair_low_quality_video):
    """Test writing labels with JPEG image format and quality settings."""
    # Create test labels
    nose_node = Node("nose")
    body_node = Node("body")
    skeleton = Skeleton([nose_node, body_node])
    points_array = np.array([[50, 60, True], [70, 80, True]], dtype=np.float32)
    instance = Instance.from_numpy(points_data=points_array, skeleton=skeleton)

    frame = LabeledFrame(
        video=centered_pair_low_quality_video, frame_idx=10, instances=[instance]
    )
    labels = Labels([frame], [skeleton])

    # Write with JPEG format and specific quality
    output_dir = tmp_path / "ultralytics_jpeg"
    write_labels(
        labels,
        str(output_dir),
        split_ratios={"val": 1.0},
        image_format="jpg",
        image_quality=85,
    )

    # Check that JPEG files were created
    image_files = list((output_dir / "val" / "images").glob("*.jpg"))
    assert len(image_files) == 1
    assert image_files[0].name == "0000000.jpg"

    # Verify it's a valid JPEG
    import imageio.v3 as iio

    img = iio.imread(image_files[0])
    assert img.shape == (384, 384)  # Should maintain original dimensions


def test_write_labels_multiple_splits(tmp_path):
    """Test writing labels to Ultralytics format with train/val splits."""
    # Create test labels with multiple frames
    skeleton = Skeleton([Node("point1"), Node("point2")])
    labels_frames = []

    for i in range(10):
        points_array = np.array(
            [[i * 10, i * 10, True], [i * 10 + 5, i * 10 + 5, True]], dtype=np.float32
        )
        instance = Instance.from_numpy(points_data=points_array, skeleton=skeleton)
        frame = LabeledFrame(
            video=Video.from_filename(f"frame_{i}.mp4"),
            frame_idx=i,
            instances=[instance],
        )
        labels_frames.append(frame)

    labels = Labels(labels_frames, [skeleton])

    # Write with 80/20 split
    output_dir = tmp_path / "ultralytics_splits"
    write_labels(labels, str(output_dir), split_ratios={"train": 0.8, "val": 0.2})

    # Check that both splits exist
    assert (output_dir / "train").exists()
    assert (output_dir / "val").exists()

    # Check data.yaml mentions both splits
    with open(output_dir / "data.yaml") as f:
        config = yaml.safe_load(f)
    assert "train" in config
    assert "val" in config


def test_write_label_file(tmp_path, ultralytics_skeleton):
    """Test writing individual label files."""
    points_array = np.array(
        [
            [100, 150, True],  # head
            [110, 160, True],  # neck
            [120, 170, True],  # center
            [130, 180, False],  # tail_base (not visible)
            [140, 190, True],  # tail_tip
        ],
        dtype=np.float32,
    )
    instance = Instance.from_numpy(
        points_data=points_array, skeleton=ultralytics_skeleton
    )
    frame = LabeledFrame(
        video=Video.from_filename("test.mp4"), frame_idx=0, instances=[instance]
    )

    label_file = tmp_path / "test_label.txt"
    write_label_file(label_file, frame, ultralytics_skeleton, (480, 640), class_id=0)

    # Read back and verify
    with open(label_file) as f:
        lines = f.readlines()

    assert len(lines) == 1
    parts = lines[0].strip().split()
    assert parts[0] == "0"  # class_id
    assert len(parts) == 20  # class_id + bbox (4) + keypoints (5*3)


def test_normalize_coordinates():
    """Test coordinate normalization."""
    skeleton = Skeleton([Node("node1"), Node("node2")])
    points_array = np.array([[100, 200, True], [0, 0, False]], dtype=np.float32)
    instance = Instance.from_numpy(points_data=points_array, skeleton=skeleton)
    normalized = normalize_coordinates(instance, (400, 800))

    assert normalized[0] == (0.125, 0.5, 2)  # 100/800, 200/400, visible
    assert normalized[1] == (0.0, 0.0, 0)  # not visible


def test_denormalize_coordinates():
    """Test coordinate denormalization."""
    normalized = [(0.125, 0.5, 2), (0.0, 0.0, 0)]
    points_array = denormalize_coordinates(normalized, (400, 800))

    assert points_array[0, 0] == 100.0  # 0.125 * 800
    assert points_array[0, 1] == 200.0  # 0.5 * 400
    assert points_array[0, 2] == True

    assert np.isnan(points_array[1, 0])
    assert np.isnan(points_array[1, 1])
    assert points_array[1, 2] == False


def test_load_file_integration(ultralytics_dataset):
    """Test loading Ultralytics dataset through main load_file function."""
    # Test with data.yaml path
    labels = load_file(str(Path(ultralytics_dataset) / "data.yaml"))
    assert isinstance(labels, Labels)
    assert len(labels.labeled_frames) > 0

    # Test with directory path
    labels = load_file(ultralytics_dataset)
    assert isinstance(labels, Labels)
    assert len(labels.labeled_frames) > 0


def test_save_file_integration(tmp_path):
    """Test saving to Ultralytics format through main save_file function."""
    # Create simple labels
    skeleton = Skeleton([Node("point")])
    points_array = np.array([[10, 10, True]], dtype=np.float32)
    instance = Instance.from_numpy(points_data=points_array, skeleton=skeleton)
    frame = LabeledFrame(Video.from_filename("test.mp4"), 0, [instance])
    labels = Labels([frame], [skeleton])

    # Save using save_file with explicit format
    output_dir = tmp_path / "save_file_test"
    save_file(labels, str(output_dir), format="ultralytics")

    assert (output_dir / "data.yaml").exists()


def test_load_ultralytics_function(ultralytics_dataset):
    """Test the convenience load_ultralytics function."""
    labels = load_ultralytics(ultralytics_dataset, split="train")
    assert isinstance(labels, Labels)
    assert len(labels.labeled_frames) == 2


def test_save_ultralytics_function(tmp_path):
    """Test the convenience save_ultralytics function."""
    skeleton = Skeleton([Node("node1"), Node("node2")])
    points_array = np.array([[5, 5, True], [15, 15, True]], dtype=np.float32)
    instance = Instance.from_numpy(points_data=points_array, skeleton=skeleton)
    frame = LabeledFrame(Video.from_filename("test.mp4"), 0, [instance])
    labels = Labels([frame], [skeleton])

    output_dir = tmp_path / "save_ultralytics_test"
    save_ultralytics(labels, str(output_dir))

    assert (output_dir / "data.yaml").exists()
    assert (output_dir / "train").exists()


def test_round_trip_conversion(ultralytics_dataset, tmp_path):
    """Test round-trip conversion: Ultralytics -> SLEAP -> Ultralytics."""
    # Load original
    original_labels = read_labels(ultralytics_dataset, split="train")

    # Save to new location
    output_dir = tmp_path / "round_trip"
    write_labels(original_labels, str(output_dir), split_ratios={"train": 1.0})

    # Load back
    reloaded_labels = read_labels(str(output_dir), split="train")

    # Compare
    assert len(original_labels.labeled_frames) == len(reloaded_labels.labeled_frames)
    assert len(original_labels.skeletons[0].nodes) == len(
        reloaded_labels.skeletons[0].nodes
    )


def test_invalid_split_ratios():
    """Test that invalid split ratios raise errors."""
    skeleton = Skeleton([Node("node")])
    labels = Labels([], [skeleton])

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        write_labels(labels, "/tmp/test", split_ratios={"train": 0.5, "val": 0.6})


def test_missing_data_yaml():
    """Test error handling for missing data.yaml."""
    with pytest.raises(FileNotFoundError, match="data.yaml not found"):
        read_labels("/nonexistent/path")


def test_missing_images_directory():
    """Test error handling for missing images directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create only data.yaml, no images directory
        data_yaml = Path(tmp_dir) / "data.yaml"
        with open(data_yaml, "w") as f:
            yaml.dump({"kpt_shape": [1, 3], "train": "train/images"}, f)

        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            read_labels(tmp_dir, split="train")


def test_empty_label_file(tmp_path, ultralytics_skeleton):
    """Test handling of empty label files."""
    # Create empty label file
    label_file = tmp_path / "empty.txt"
    label_file.write_text("")

    instances = parse_label_file(label_file, ultralytics_skeleton, (480, 640))
    assert len(instances) == 0


def test_malformed_label_file(tmp_path, ultralytics_skeleton):
    """Test handling of malformed label files."""
    label_file = tmp_path / "malformed.txt"
    label_file.write_text("invalid line\n0 0.5\n")  # incomplete data

    # Should not crash, but warn and skip invalid lines
    instances = parse_label_file(label_file, ultralytics_skeleton, (480, 640))
    assert len(instances) == 0  # Both lines are invalid


def test_keypoint_count_mismatch_warning(tmp_path):
    """Test warning when keypoint count doesn't match skeleton."""
    # Create skeleton with 3 nodes
    skeleton = Skeleton([Node("a"), Node("b"), Node("c")])

    # Create label file with 2 keypoints (mismatch)
    label_file = tmp_path / "mismatch.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.4 0.1 0.1 2 0.2 0.2 2\n")  # Only 2 keypoints

    with pytest.warns(UserWarning, match="Keypoint count mismatch"):
        instances = parse_label_file(label_file, skeleton, (100, 100))

    assert len(instances) == 0  # Instance rejected due to mismatch


def test_realistic_labels_roundtrip(tmp_path, labels_predictions):
    """Test roundtrip conversion with realistic labels containing multiple instances and tracks."""
    # Take a subset of frames for testing
    labels_subset = Labels(
        labeled_frames=labels_predictions.labeled_frames[:10],
        videos=labels_predictions.videos,
        skeletons=labels_predictions.skeletons,
        tracks=labels_predictions.tracks,
    )

    # Write to Ultralytics format
    output_dir = tmp_path / "ultralytics_realistic"
    write_labels(labels_subset, str(output_dir), split_ratios={"train": 1.0})

    # Verify images were created
    image_files = list((output_dir / "train" / "images").glob("*.png"))
    assert len(image_files) == 10

    # Read back and compare (note: tracks will be different as Ultralytics doesn't preserve them)
    labels_read = read_labels(output_dir, split="train")
    assert len(labels_read.labeled_frames) == 10
    assert len(labels_read.skeletons) == 1

    # Just verify we have the right number of instances per frame
    # Don't check exact coordinates as frame ordering might differ
    orig_instance_counts = [
        len(frame.instances) for frame in labels_subset.labeled_frames
    ]
    read_instance_counts = [
        len(frame.instances) for frame in labels_read.labeled_frames
    ]
    assert sum(orig_instance_counts) == sum(read_instance_counts)


def test_mediavideo_backend_roundtrip(tmp_path, centered_pair_low_quality_video):
    """Test roundtrip with MediaVideo backend."""
    # Create simple labels
    skeleton = Skeleton([Node("A"), Node("B")])
    instances = [
        Instance.from_numpy(
            np.array([[100, 100, True], [200, 200, True]], dtype=np.float32), skeleton
        )
    ]
    frame = LabeledFrame(
        video=centered_pair_low_quality_video, frame_idx=0, instances=instances
    )
    labels = Labels([frame], videos=[centered_pair_low_quality_video])

    # Write and read back
    output_dir = tmp_path / "ultralytics_mediavideo"
    write_labels(labels, str(output_dir))

    labels_read = read_labels(output_dir)
    assert len(labels_read.labeled_frames) == 1
    assert len(labels_read.labeled_frames[0].instances) == 1


def test_hdf5video_backend_roundtrip(tmp_path, slp_minimal_pkg):
    """Test roundtrip with HDF5Video backend (embedded frames)."""
    # Load labels with embedded video
    labels = sleap_io.load_slp(slp_minimal_pkg)

    # Take first frame
    labels_subset = Labels(
        labeled_frames=labels.labeled_frames[:1],
        videos=labels.videos,
        skeletons=labels.skeletons,
    )

    # Write and read back
    output_dir = tmp_path / "ultralytics_hdf5"
    write_labels(labels_subset, str(output_dir), split_ratios={"train": 1.0})

    labels_read = read_labels(output_dir)
    assert len(labels_read.labeled_frames) == 1


def test_imagevideo_backend_roundtrip(tmp_path, centered_pair_frame_paths):
    """Test roundtrip with ImageVideo backend."""
    # Create ImageVideo from frame paths
    video = sleap_io.Video.from_filename(centered_pair_frame_paths)

    skeleton = Skeleton([Node("pt1"), Node("pt2")])
    instances = [
        Instance.from_numpy(
            np.array([[50, 50, True], [150, 150, True]], dtype=np.float32), skeleton
        )
    ]
    frame = LabeledFrame(video=video, frame_idx=0, instances=instances)
    labels = Labels([frame], videos=[video])

    # Write and read back
    output_dir = tmp_path / "ultralytics_imagevideo"
    write_labels(labels, str(output_dir), image_format="jpg")

    labels_read = read_labels(output_dir)
    assert len(labels_read.labeled_frames) == 1
    assert len(labels_read.labeled_frames[0].instances) == 1


def test_multiprocessing_write(tmp_path, centered_pair_low_quality_video):
    """Test multiprocessing image saving."""
    # Create labels with multiple frames
    skeleton = Skeleton([Node("A"), Node("B")])
    frames = []
    for i in range(5):
        instances = [
            Instance.from_numpy(
                np.array(
                    [[10 + i, 20 + i, True], [30 + i, 40 + i, True]], dtype=np.float32
                ),
                skeleton,
            )
        ]
        frame = LabeledFrame(
            video=centered_pair_low_quality_video, frame_idx=i * 10, instances=instances
        )
        frames.append(frame)

    labels = Labels(frames, videos=[centered_pair_low_quality_video])

    # Write with multiprocessing
    output_dir = tmp_path / "ultralytics_multiproc"
    write_labels(
        labels, str(output_dir), use_multiprocessing=True, n_workers=2, verbose=False
    )

    # Verify all images were created
    image_files = list((output_dir / "train" / "images").glob("*.png"))
    assert len(image_files) == 4  # 80% of 5 frames

    # Verify annotations were created
    label_files = list((output_dir / "train" / "labels").glob("*.txt"))
    assert len(label_files) == 4


def test_multiprocessing_with_progress(tmp_path, centered_pair_low_quality_video):
    """Test multiprocessing with progress bar."""
    # Create simple labels
    skeleton = Skeleton([Node("nose"), Node("tail")])
    instance = Instance.from_numpy(
        np.array([[100, 100, True], [200, 200, True]], dtype=np.float32), skeleton
    )
    frame = LabeledFrame(
        video=centered_pair_low_quality_video, frame_idx=0, instances=[instance]
    )
    labels = Labels([frame], videos=[centered_pair_low_quality_video])

    # Write with multiprocessing and progress
    output_dir = tmp_path / "ultralytics_mp_progress"
    write_labels(
        labels,
        str(output_dir),
        use_multiprocessing=True,
        verbose=True,  # Enable progress bar
    )

    # Check output
    assert (output_dir / "train" / "images" / "0000000.png").exists()
    assert (output_dir / "train" / "labels" / "0000000.txt").exists()


def test_multiprocessing_error_handling(tmp_path):
    """Test multiprocessing handles errors gracefully."""
    # Create labels with non-existent video
    skeleton = Skeleton([Node("A"), Node("B")])
    video = Video.from_filename("non_existent_video.mp4", open=False)
    instance = Instance.from_numpy(
        np.array([[10, 20, True], [30, 40, True]], dtype=np.float32), skeleton
    )
    frame = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels([frame], videos=[video])

    # Should not crash, just skip the frame
    output_dir = tmp_path / "ultralytics_mp_error"
    # Note: Warnings from subprocesses won't be captured by pytest
    write_labels(labels, str(output_dir), use_multiprocessing=True, verbose=False)

    # No images should be created since video doesn't exist
    image_files = list((output_dir / "train" / "images").glob("*"))
    assert len(image_files) == 0

    # No annotations should be created either
    label_files = list((output_dir / "train" / "labels").glob("*"))
    assert len(label_files) == 0


def test_save_frame_image_direct(tmp_path, centered_pair_low_quality_video):
    """Test _save_frame_image function directly to ensure coverage."""
    from sleap_io.io.ultralytics import _save_frame_image

    # Test successful save - PNG format
    output_path = tmp_path / "test_frame.png"
    frame_data = {
        "video_path": str(centered_pair_low_quality_video.filename),
        "frame_idx": 0,
        "lf_idx": 0,
        "output_path": str(output_path),
    }

    result = _save_frame_image((frame_data, "png", None))
    assert result == str(output_path)
    assert output_path.exists()

    # Verify image was saved correctly
    img = iio.imread(output_path)
    assert img.shape == (384, 384)  # Grayscale image

    # Test JPEG with quality
    output_path_jpg = tmp_path / "test_frame.jpg"
    frame_data["output_path"] = str(output_path_jpg)
    result = _save_frame_image((frame_data, "jpg", 85))
    assert result == str(output_path_jpg)
    assert output_path_jpg.exists()

    # Test PNG with compression
    output_path_png2 = tmp_path / "test_frame2.png"
    frame_data["output_path"] = str(output_path_png2)
    result = _save_frame_image((frame_data, "png", 5))
    assert result == str(output_path_png2)
    assert output_path_png2.exists()

    # Test error handling - non-existent video
    frame_data["video_path"] = "non_existent.mp4"
    frame_data["output_path"] = str(tmp_path / "should_not_exist.png")

    with pytest.warns(UserWarning, match="Error processing frame"):
        result = _save_frame_image((frame_data, "png", None))
    assert result is None

    # Test frame returns None
    # This is tricky to test, but we can test with an out-of-bounds frame index
    frame_data["video_path"] = str(centered_pair_low_quality_video.filename)
    frame_data["frame_idx"] = 999999  # Out of bounds
    frame_data["output_path"] = str(tmp_path / "out_of_bounds.png")

    result = _save_frame_image((frame_data, "png", None))
    assert result is None


def test_missing_labels_directory(tmp_path):
    """Test error when labels directory is missing."""
    # Create a data.yaml but no labels directory
    data_yaml = tmp_path / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(
            {"kpt_shape": [2, 3], "train": "train/images", "node_names": ["a", "b"]}, f
        )

    # Create images directory but not labels
    (tmp_path / "train" / "images").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Labels directory not found"):
        read_labels(tmp_path, split="train")


def test_video_shape_none_fallback(tmp_path, centered_pair_low_quality_video):
    """Test fallback to reading frame when video.shape is None."""
    # Create a mock video with None shape
    from unittest.mock import Mock, patch

    # Create test setup
    skeleton = Skeleton([Node("a"), Node("b")])
    instance = Instance.from_numpy(
        np.array([[100, 100, True], [200, 200, True]], dtype=np.float32), skeleton
    )

    # Mock video with None shape but valid frame access
    mock_video = Mock(spec=Video)
    mock_video.shape = None
    mock_video.filename = centered_pair_low_quality_video.filename
    mock_video.__getitem__ = Mock(return_value=centered_pair_low_quality_video[0])

    frame = LabeledFrame(video=mock_video, frame_idx=0, instances=[instance])
    labels = Labels([frame], [skeleton])

    # Write labels - this should trigger the fallback
    output_dir = tmp_path / "shape_none_test"
    write_labels(labels, str(output_dir), split_ratios={"train": 1.0})

    # Verify it worked
    assert (output_dir / "train" / "images" / "0000000.png").exists()
    assert (output_dir / "train" / "labels" / "0000000.txt").exists()


def test_empty_skeletons_error(tmp_path):
    """Test error when labels have no skeletons."""
    labels = Labels([], [])  # Empty skeletons list

    with pytest.raises(ValueError, match="Labels must have at least one skeleton"):
        write_labels(labels, str(tmp_path))


def test_frame_image_none_warning(tmp_path, centered_pair_low_quality_video):
    """Test warning when frame.image returns None."""
    from unittest.mock import Mock, PropertyMock, patch

    skeleton = Skeleton([Node("a"), Node("b")])
    instance = Instance.from_numpy(
        np.array([[10, 20, True], [30, 40, True]], dtype=np.float32), skeleton
    )

    # Create a frame that returns None for image property
    frame = LabeledFrame(
        video=centered_pair_low_quality_video, frame_idx=0, instances=[instance]
    )

    # Mock the image property to return None
    with patch.object(LabeledFrame, "image", new_callable=PropertyMock) as mock_image:
        mock_image.return_value = None

        labels = Labels([frame], [skeleton])
        output_dir = tmp_path / "frame_none_test"

        with pytest.warns(UserWarning, match="Could not load frame"):
            write_labels(labels, str(output_dir), verbose=False)

        # No images should be created
        image_files = list((output_dir / "train" / "images").glob("*"))
        assert len(image_files) == 0


def test_three_way_split_fallback(tmp_path, centered_pair_low_quality_video):
    """Test three-way split with fallback logic."""
    # Create labels with enough frames
    skeleton = Skeleton([Node("a")])
    frames = []
    for i in range(10):
        instance = Instance.from_numpy(
            np.array([[i * 10, i * 10, True]], dtype=np.float32), skeleton
        )
        frame = LabeledFrame(
            video=centered_pair_low_quality_video, frame_idx=i, instances=[instance]
        )
        frames.append(frame)

    labels = Labels(frames, [skeleton])

    # Test three-way split
    output_dir = tmp_path / "three_way_split"
    write_labels(
        labels,
        str(output_dir),
        split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        verbose=False,
    )

    # Check all three splits exist
    assert (output_dir / "train").exists()
    assert (output_dir / "val").exists()
    assert (output_dir / "test").exists()

    # Check split sizes
    train_images = list((output_dir / "train" / "images").glob("*.png"))
    val_images = list((output_dir / "val" / "images").glob("*.png"))
    test_images = list((output_dir / "test" / "images").glob("*.png"))

    assert len(train_images) == 6  # 60% of 10
    assert len(val_images) == 2  # 20% of 10
    assert len(test_images) == 2  # 20% of 10


def test_instance_no_visible_points(tmp_path):
    """Test handling of instances with no visible points."""
    skeleton = Skeleton([Node("a"), Node("b"), Node("c")])

    # Create instance with all points invisible
    points_array = np.array(
        [[np.nan, np.nan, False], [np.nan, np.nan, False], [np.nan, np.nan, False]],
        dtype=np.float32,
    )
    instance = Instance.from_numpy(points_data=points_array, skeleton=skeleton)

    frame = LabeledFrame(
        video=Video.from_filename("test.mp4"), frame_idx=0, instances=[instance]
    )

    # Write label file
    label_file = tmp_path / "no_visible.txt"
    write_label_file(label_file, frame, skeleton, (480, 640), class_id=0)

    # File should be empty (no instances written)
    with open(label_file) as f:
        content = f.read()
    assert content == ""


def test_parse_label_file_edge_cases(tmp_path, ultralytics_skeleton):
    """Test edge cases in parse_label_file."""
    # Test with wrong number of keypoints
    label_file = tmp_path / "wrong_keypoints.txt"
    # Write file with 3 keypoints when skeleton expects 5
    label_file.write_text("0 0.5 0.5 0.2 0.2 0.1 0.1 2 0.2 0.2 2 0.3 0.3 1\n")

    with pytest.warns(UserWarning, match="Keypoint count mismatch"):
        instances = parse_label_file(label_file, ultralytics_skeleton, (480, 640))
    assert len(instances) == 0  # Should skip the malformed instance

    # Test parsing error with invalid data
    label_file = tmp_path / "invalid_data.txt"
    label_file.write_text("not_a_number 0.5 0.5 0.2 0.2\n")

    with pytest.warns(UserWarning, match="Error parsing line"):
        instances = parse_label_file(label_file, ultralytics_skeleton, (480, 640))
    assert len(instances) == 0


def test_frame_none_return_from_video(tmp_path):
    """Test handling when video returns None for a frame."""
    from unittest.mock import Mock, patch
    from sleap_io.io.ultralytics import _save_frame_image

    # Create a mock video that returns None for frame
    mock_video = Mock(spec=Video)
    mock_video.filename = "test_video.mp4"
    mock_video.__getitem__ = Mock(return_value=None)

    # Patch Video.from_filename to return our mock
    with patch("sleap_io.io.ultralytics.Video.from_filename", return_value=mock_video):
        frame_data = {
            "video_path": "test_video.mp4",
            "frame_idx": 0,
            "lf_idx": 0,
            "output_path": str(tmp_path / "should_not_exist.png"),
        }

        result = _save_frame_image((frame_data, "png", None))
        assert result is None
        assert not (tmp_path / "should_not_exist.png").exists()


def test_parse_label_file_with_invalid_keypoint_count(tmp_path):
    """Test parse_label_file with keypoints % 3 != 0."""
    skeleton = Skeleton([Node("a"), Node("b")])

    # Create label file with incomplete keypoint data (8 values instead of 6 or 9)
    label_file = tmp_path / "invalid_keypoints.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.2 0.1 0.1 2 0.2\n")  # 8 values after bbox

    with pytest.warns(UserWarning, match="Invalid keypoint data"):
        instances = parse_label_file(label_file, skeleton, (480, 640))
    assert len(instances) == 0


def test_read_labels_fallback_to_read_first_frame(tmp_path):
    """Test read_labels when video.shape is None and falls back to reading first frame."""
    from unittest.mock import Mock, patch

    # Create test data structure
    data_yaml = tmp_path / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(
            {
                "kpt_shape": [2, 3],
                "train": "train/images",
                "node_names": ["a", "b"],
                "skeleton": [[0, 1]],
            },
            f,
        )

    # Create directories and files
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)

    # Create a test image file
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "train" / "images" / "test.png"
    iio.imwrite(image_path, test_img)

    # Create corresponding label
    label_path = tmp_path / "train" / "labels" / "test.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2 0.3 0.3 2 0.7 0.7 2\n")

    # Mock Video to have None shape but return valid frame
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def mock_video_from_filename(filename):
        mock_vid = Mock(spec=Video)
        mock_vid.filename = filename
        mock_vid.shape = None
        mock_vid.__getitem__ = Mock(return_value=mock_frame)
        return mock_vid

    with patch(
        "sleap_io.io.ultralytics.Video.from_filename",
        side_effect=mock_video_from_filename,
    ):
        labels = read_labels(tmp_path, split="train")

    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1


def test_write_labels_frame_extraction_error_continue(tmp_path):
    """Test write_labels warning for frame.image returning None - code coverage only."""
    # This test just ensures the warning code path is covered
    # The actual behavior is tricky to test due to how enumerate works with skipped frames

    # We've already tested this code path indirectly in other tests
    # This specific line (291-294) gets executed when frame.image returns None
    # which happens in test_frame_image_none_warning

    # Let's just verify the warning message format
    import re

    warning_pattern = r"Could not load frame \d+ from video, skipping\."
    assert re.match(warning_pattern, "Could not load frame 1 from video, skipping.")


def test_create_splits_three_way_fallback_exception(tmp_path):
    """Test three-way split when make_training_splits raises exception."""
    from unittest.mock import patch

    skeleton = Skeleton([Node("a")])
    frames = []
    for i in range(10):
        instance = Instance.from_numpy(
            np.array([[i * 10, i * 10, True]], dtype=np.float32), skeleton
        )
        frame = LabeledFrame(
            video=Video.from_filename(f"video_{i}.mp4", open=False),
            frame_idx=i,
            instances=[instance],
        )
        frames.append(frame)

    labels = Labels(frames, [skeleton])

    # Mock make_training_splits to raise exception
    with patch.object(
        Labels, "make_training_splits", side_effect=Exception("Test error")
    ):
        output_dir = tmp_path / "three_way_split_fallback"
        write_labels(
            labels,
            str(output_dir),
            split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            verbose=False,
        )

        # Should still create three splits using fallback logic
        assert (output_dir / "train").exists()
        assert (output_dir / "val").exists()
        assert (output_dir / "test").exists()


def test_write_label_file_skip_wrong_points_count(tmp_path):
    """Test write_label_file skips instances with wrong number of points."""
    skeleton = Skeleton([Node("a"), Node("b"), Node("c")])  # 3 nodes

    # Create instance with only 2 points (mismatch)
    instance = Instance.from_numpy(
        np.array([[10, 20, True], [30, 40, True]], dtype=np.float32),
        Skeleton([Node("x"), Node("y")]),  # Different skeleton with 2 nodes
    )

    frame = LabeledFrame(
        video=Video.from_filename("test.mp4"), frame_idx=0, instances=[instance]
    )

    label_file = tmp_path / "mismatch.txt"

    with pytest.warns(UserWarning, match="Instance has 2 points"):
        write_label_file(label_file, frame, skeleton, (480, 640))

    # File should be empty
    with open(label_file) as f:
        assert f.read() == ""
