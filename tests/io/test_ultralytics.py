"""Tests for sleap_io.io.ultralytics module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import yaml

from sleap_io import Labels, Skeleton, Node, Edge, Instance, LabeledFrame, Point, Track, Video
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
        if point.visible:
            assert 0 <= point.x <= 640
            assert 0 <= point.y <= 480


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
    assert all(instance.skeleton == custom_skeleton 
              for frame in labels.labeled_frames 
              for instance in frame.instances)


def test_write_labels_single_split(tmp_path):
    """Test writing labels to Ultralytics format with single split."""
    # Create test labels
    skeleton = Skeleton([Node("head"), Node("tail")], [Edge(Node("head"), Node("tail"))])
    points = [Point(10, 20, True), Point(30, 40, True)]
    instance = Instance(points=points, skeleton=skeleton)
    frame = LabeledFrame(video=Video.from_filename("dummy.mp4"), frame_idx=0, instances=[instance])
    labels = Labels([frame], [skeleton])
    
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


def test_write_labels_multiple_splits(tmp_path):
    """Test writing labels to Ultralytics format with train/val splits."""
    # Create test labels with multiple frames
    skeleton = Skeleton([Node("point1"), Node("point2")])
    labels_frames = []
    
    for i in range(10):
        points = [Point(i*10, i*10, True), Point(i*10+5, i*10+5, True)]
        instance = Instance(points=points, skeleton=skeleton)
        frame = LabeledFrame(video=Video.from_filename(f"frame_{i}.mp4"), frame_idx=i, instances=[instance])
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
    points = [
        Point(100, 150, True),   # head
        Point(110, 160, True),   # neck
        Point(120, 170, True),   # center
        Point(130, 180, False),  # tail_base (not visible)
        Point(140, 190, True),   # tail_tip
    ]
    instance = Instance(points=points, skeleton=ultralytics_skeleton)
    frame = LabeledFrame(video=Video.from_filename("test.mp4"), frame_idx=0, instances=[instance])
    
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
    points = [Point(100, 200, True), Point(0, 0, False)]
    normalized = normalize_coordinates(points, (400, 800))
    
    assert normalized[0] == (0.125, 0.5, 2)  # 100/800, 200/400, visible
    assert normalized[1] == (0.0, 0.0, 0)    # not visible


def test_denormalize_coordinates():
    """Test coordinate denormalization."""
    normalized = [(0.125, 0.5, 2), (0.0, 0.0, 0)]
    points = denormalize_coordinates(normalized, (400, 800))
    
    assert points[0].x == 100.0  # 0.125 * 800
    assert points[0].y == 200.0  # 0.5 * 400
    assert points[0].visible is True
    
    assert np.isnan(points[1].x)
    assert np.isnan(points[1].y)
    assert points[1].visible is False


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
    instance = Instance([Point(10, 10, True)], skeleton)
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
    instance = Instance([Point(5, 5, True), Point(15, 15, True)], skeleton)
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
    assert len(original_labels.skeletons[0].nodes) == len(reloaded_labels.skeletons[0].nodes)


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
        with open(data_yaml, 'w') as f:
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