"""Tests for LabelsSet loading functions in main API."""

import numpy as np
import pytest
import yaml

from sleap_io import (
    LabelsSet,
    Node,
    Skeleton,
    load_labels_set,
    load_slp,
    load_slp_labels_set,
    load_ultralytics_labels_set,
)


def test_load_labels_set_slp_directory(tmp_path, slp_minimal):
    """Test load_labels_set with SLP directory."""
    labels = load_slp(slp_minimal)

    # Create test directory with SLP files
    test_dir = tmp_path / "splits"
    test_dir.mkdir()
    labels.save(test_dir / "train.slp", embed=False)
    labels.save(test_dir / "val.slp", embed=False)

    # Load without format specification (should auto-detect SLP)
    labels_set = load_labels_set(test_dir)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 2
    assert "train" in labels_set
    assert "val" in labels_set


def test_load_labels_set_slp_list(tmp_path, slp_minimal):
    """Test load_labels_set with list of SLP files."""
    labels = load_slp(slp_minimal)

    # Create test files
    file1 = tmp_path / "split1.slp"
    file2 = tmp_path / "split2.slp"
    labels.save(file1, embed=False)
    labels.save(file2, embed=False)

    # Load with list (should auto-detect SLP)
    labels_set = load_labels_set([file1, file2])

    assert len(labels_set) == 2
    assert "split1" in labels_set
    assert "split2" in labels_set


def test_load_labels_set_slp_dict(tmp_path, slp_minimal):
    """Test load_labels_set with dictionary of SLP files."""
    labels = load_slp(slp_minimal)

    # Create test files
    train_file = tmp_path / "train_data.slp"
    val_file = tmp_path / "val_data.slp"
    labels.save(train_file, embed=False)
    labels.save(val_file, embed=False)

    # Load with dict
    labels_set = load_labels_set({"training": train_file, "validation": val_file})

    assert len(labels_set) == 2
    assert "training" in labels_set
    assert "validation" in labels_set


def test_load_labels_set_ultralytics(tmp_path):
    """Test load_labels_set with Ultralytics dataset."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    # Create minimal Ultralytics structure
    data_config = {
        "path": str(dataset_path),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": [2, 2],
        "names": ["animal"],
    }

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Create train split
    train_path = dataset_path / "train"
    images_path = train_path / "images"
    labels_path = train_path / "labels"
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    # Create a dummy image and label
    import imageio.v3 as iio

    img_file = images_path / "img_000.jpg"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    iio.imwrite(img_file, dummy_img)

    label_file = labels_path / "img_000.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.4 0.4 0.4 2 0.6 0.6 2\n")

    # Load with auto-detection (should detect ultralytics)
    labels_set = load_labels_set(dataset_path, verbose=False)

    assert isinstance(labels_set, LabelsSet)
    assert "train" in labels_set


def test_load_labels_set_explicit_format(tmp_path, slp_minimal):
    """Test load_labels_set with explicit format specification."""
    labels = load_slp(slp_minimal)

    # Create test file
    test_file = tmp_path / "test.slp"
    labels.save(test_file, embed=False)

    # Load with explicit format
    labels_set = load_labels_set([test_file], format="slp")

    assert len(labels_set) == 1
    assert "test" in labels_set


def test_load_labels_set_invalid_format():
    """Test load_labels_set with invalid format."""
    with pytest.raises(ValueError, match="Unknown format"):
        load_labels_set("dummy_path", format="invalid_format")


def test_load_labels_set_ultralytics_invalid_input():
    """Test load_labels_set with invalid input for ultralytics."""
    with pytest.raises(ValueError, match="requires a directory path"):
        load_labels_set(["file1", "file2"], format="ultralytics")


def test_load_slp_labels_set(tmp_path, slp_minimal):
    """Test load_slp_labels_set function."""
    labels = load_slp(slp_minimal)

    # Create test directory
    test_dir = tmp_path / "splits"
    test_dir.mkdir()
    labels.save(test_dir / "data.slp", embed=False)

    # Load using specific function
    labels_set = load_slp_labels_set(test_dir)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 1
    assert "data" in labels_set


def test_load_ultralytics_labels_set(tmp_path):
    """Test load_ultralytics_labels_set function."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    # Create minimal dataset
    data_config = {
        "path": str(dataset_path),
        "train": "train/images",
        "kpt_shape": [3, 2],
        "names": ["animal"],
    }

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Create train split
    train_path = dataset_path / "train"
    images_path = train_path / "images"
    labels_path = train_path / "labels"
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    # Create dummy data
    import imageio.v3 as iio

    img_file = images_path / "img_000.jpg"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    iio.imwrite(img_file, dummy_img)

    label_file = labels_path / "img_000.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.4 0.4 0.4 2 0.5 0.5 2 0.6 0.6 2\n")

    # Load with custom skeleton
    skeleton = Skeleton([Node("head"), Node("body"), Node("tail")])
    labels_set = load_ultralytics_labels_set(
        str(dataset_path), skeleton=skeleton, verbose=False
    )

    assert isinstance(labels_set, LabelsSet)
    assert "train" in labels_set
    assert len(labels_set["train"].skeletons[0].nodes) == 3
    assert labels_set["train"].skeletons[0].nodes[0].name == "head"


def test_load_labels_set_kwargs_passing(tmp_path):
    """Test that kwargs are properly passed to format-specific loaders."""
    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    # Create val split only
    val_path = dataset_path / "val"
    images_path = val_path / "images"
    labels_path = val_path / "labels"
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    # Create dummy data
    import imageio.v3 as iio

    img_file = images_path / "img_000.jpg"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    iio.imwrite(img_file, dummy_img)

    label_file = labels_path / "img_000.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.4 0.5 0.5 2\n")

    # Create minimal data.yaml
    data_config = {
        "path": str(dataset_path),
        "val": "val/images",
        "kpt_shape": [1, 2],
        "names": ["animal"],
    }

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Load with specific splits and skeleton via kwargs
    skeleton = Skeleton([Node("custom_node")])
    labels_set = load_labels_set(
        dataset_path,
        format="ultralytics",
        splits=["val"],
        skeleton=skeleton,
        verbose=False,
    )

    assert len(labels_set) == 1
    assert "val" in labels_set
    assert labels_set["val"].skeletons[0].nodes[0].name == "custom_node"
