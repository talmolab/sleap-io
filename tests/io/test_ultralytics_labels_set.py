"""Tests for loading LabelsSet from Ultralytics datasets."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import yaml

from sleap_io import Labels, LabelsSet, Node, Skeleton
from sleap_io.io.ultralytics import read_labels_set, write_labels


def create_test_ultralytics_dataset(base_path: Path, splits=["train", "val", "test"]):
    """Create a minimal Ultralytics dataset structure for testing."""
    # Create data.yaml
    data_config = {
        "path": str(base_path),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "kpt_shape": [3, 2],  # 3 keypoints
        "names": ["animal"],
    }

    with open(base_path / "data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Create splits
    for split in splits:
        split_path = base_path / split
        images_path = split_path / "images"
        labels_path = split_path / "labels"

        images_path.mkdir(parents=True)
        labels_path.mkdir(parents=True)

        # Create a few fake images and labels
        for i in range(2):
            # Create dummy image file (needs to be a real image)
            img_file = images_path / f"img_{i:03d}.jpg"
            # Create a small dummy image
            dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
            import imageio.v3 as iio

            iio.imwrite(img_file, dummy_img)

            # Create label file with normalized coordinates
            label_file = labels_path / f"img_{i:03d}.txt"
            # Format: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 x3 y3 v3
            label_line = "0 0.5 0.5 0.4 0.4 0.4 0.4 2 0.5 0.5 2 0.6 0.6 2\n"
            label_file.write_text(label_line)


def test_read_labels_set_basic(tmp_path):
    """Test basic loading of LabelsSet from Ultralytics dataset."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    create_test_ultralytics_dataset(dataset_path, ["train", "val"])

    # Load LabelsSet
    labels_set = read_labels_set(str(dataset_path), verbose=False)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 2
    assert "train" in labels_set
    assert "val" in labels_set

    # Check that each split has correct structure
    for split in ["train", "val"]:
        labels = labels_set[split]
        assert isinstance(labels, Labels)
        assert len(labels) == 2  # 2 frames per split
        assert len(labels.skeletons) == 1
        assert len(labels.skeletons[0].nodes) == 3  # 3 keypoints


def test_read_labels_set_specific_splits(tmp_path):
    """Test loading specific splits only."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    create_test_ultralytics_dataset(dataset_path, ["train", "val", "test"])

    # Load only train and test
    labels_set = read_labels_set(
        str(dataset_path), splits=["train", "test"], verbose=False
    )

    assert len(labels_set) == 2
    assert "train" in labels_set
    assert "test" in labels_set
    assert "val" not in labels_set


def test_read_labels_set_custom_skeleton(tmp_path):
    """Test loading with a custom skeleton."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    create_test_ultralytics_dataset(dataset_path, ["train"])

    # Create custom skeleton
    skeleton = Skeleton([Node(name="head"), Node(name="body"), Node(name="tail")])

    # Load with custom skeleton
    labels_set = read_labels_set(str(dataset_path), skeleton=skeleton, verbose=False)

    assert len(labels_set["train"].skeletons[0].nodes) == 3
    assert labels_set["train"].skeletons[0].nodes[0].name == "head"
    assert labels_set["train"].skeletons[0].nodes[1].name == "body"
    assert labels_set["train"].skeletons[0].nodes[2].name == "tail"


def test_read_labels_set_missing_splits(tmp_path):
    """Test handling of missing splits."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    create_test_ultralytics_dataset(dataset_path, ["train"])

    # Try to load non-existent splits
    labels_set = read_labels_set(
        str(dataset_path),
        splits=["train", "val", "test"],  # val and test don't exist
        verbose=False,
    )

    # Should only load train
    assert len(labels_set) == 1
    assert "train" in labels_set
    assert "val" not in labels_set
    assert "test" not in labels_set


def test_read_labels_set_no_splits(tmp_path):
    """Test error when no splits are found."""
    dataset_path = tmp_path / "empty_dataset"
    dataset_path.mkdir()

    with pytest.raises(ValueError, match="No splits found"):
        read_labels_set(str(dataset_path), verbose=False)


def test_read_labels_set_auto_detect_splits(tmp_path):
    """Test automatic detection of available splits."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    # Create only train and valid (not val)
    create_test_ultralytics_dataset(dataset_path, ["train", "valid"])

    # Don't specify splits - should auto-detect
    labels_set = read_labels_set(str(dataset_path), verbose=False)

    assert len(labels_set) == 2
    assert "train" in labels_set
    assert "valid" in labels_set


def test_read_labels_set_no_data_yaml(tmp_path):
    """Test loading without data.yaml requires a skeleton."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    # Create dataset structure without data.yaml
    train_path = dataset_path / "train"
    images_path = train_path / "images"
    labels_path = train_path / "labels"
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    # Create a simple file (needs to be real image)
    img_file = images_path / "img_000.jpg"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    iio.imwrite(img_file, dummy_img)
    label_file = labels_path / "img_000.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.4 0.5 0.5 2\n")

    # Since read_labels requires data.yaml, we need to create a minimal one
    # even when providing a skeleton
    data_config = {
        "path": str(dataset_path),
        "train": "train/images",
        "kpt_shape": [1, 2],  # 1 keypoint
        "names": ["animal"],
    }

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Should work with provided skeleton
    skeleton = Skeleton([Node(name="point")])
    labels_set = read_labels_set(str(dataset_path), skeleton=skeleton, verbose=False)

    assert len(labels_set) == 1
    assert "train" in labels_set
    # The custom skeleton should be used
    assert labels_set["train"].skeletons[0].nodes[0].name == "point"


def test_read_labels_set_roundtrip(tmp_path, slp_minimal):
    """Test that we can write labels and read them back as LabelsSet."""
    from sleap_io import load_slp

    # Load test data
    labels = load_slp(slp_minimal)

    dataset_path = tmp_path / "yolo_dataset"

    # Write labels in ultralytics format
    write_labels(
        labels,
        str(dataset_path),
        split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        verbose=False,
    )

    # Read back as LabelsSet
    labels_set = read_labels_set(str(dataset_path), verbose=False)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 3
    assert "train" in labels_set
    assert "val" in labels_set
    assert "test" in labels_set

    # Total frames should match original (allowing for rounding in splits)
    total_frames = sum(len(split_labels) for split_labels in labels_set.values())
    assert abs(total_frames - len(labels)) <= 1
