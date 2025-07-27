"""Tests for loading LabelsSet from SLP files."""

import pytest

from sleap_io import LabelsSet, load_slp
from sleap_io.io.slp import read_labels_set


def test_read_labels_set_from_directory(tmp_path, slp_minimal):
    """Test loading LabelsSet from a directory of SLP files."""
    # Load minimal labels
    labels = load_slp(slp_minimal)

    # Create test directory with multiple SLP files
    test_dir = tmp_path / "splits"
    test_dir.mkdir()

    # Save splits
    labels.save(test_dir / "train.slp", embed=False)
    labels.save(test_dir / "val.slp", embed=False)
    labels.save(test_dir / "test.slp", embed=False)

    # Load as LabelsSet
    labels_set = read_labels_set(test_dir)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 3
    assert "train" in labels_set
    assert "val" in labels_set
    assert "test" in labels_set

    # Check that each loaded Labels has correct data
    for name in ["train", "val", "test"]:
        assert len(labels_set[name]) == len(labels)


def test_read_labels_set_from_list(tmp_path, slp_minimal):
    """Test loading LabelsSet from a list of file paths."""
    labels = load_slp(slp_minimal)

    # Create test files
    file1 = tmp_path / "split1.slp"
    file2 = tmp_path / "split2.slp"
    labels.save(file1, embed=False)
    labels.save(file2, embed=False)

    # Load from list
    labels_set = read_labels_set([file1, file2])

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 2
    assert "split1" in labels_set
    assert "split2" in labels_set


def test_read_labels_set_from_dict(tmp_path, slp_minimal):
    """Test loading LabelsSet from a dictionary mapping."""
    labels = load_slp(slp_minimal)

    # Create test files
    train_file = tmp_path / "train_data.slp"
    val_file = tmp_path / "validation_data.slp"
    labels.save(train_file, embed=False)
    labels.save(val_file, embed=False)

    # Load from dictionary
    labels_set = read_labels_set({"training": train_file, "validation": val_file})

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 2
    assert "training" in labels_set
    assert "validation" in labels_set


def test_read_labels_set_empty_directory(tmp_path):
    """Test error handling for empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No .slp files found"):
        read_labels_set(empty_dir)


def test_read_labels_set_invalid_path(tmp_path):
    """Test error handling for invalid path."""
    # Non-directory path
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError, match="Path must be a directory"):
        read_labels_set(file_path)


def test_read_labels_set_without_videos(tmp_path, slp_minimal):
    """Test loading LabelsSet without opening videos."""
    labels = load_slp(slp_minimal)

    # Save a file
    test_file = tmp_path / "test.slp"
    labels.save(test_file, embed=False)

    # Load without opening videos
    labels_set = read_labels_set([test_file], open_videos=False)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 1

    # Videos should not be opened
    for lf in labels_set["test"].labeled_frames:
        assert lf.video is not None
        assert not lf.video.is_open


def test_read_labels_set_string_paths(tmp_path, slp_minimal):
    """Test that string paths work as well as Path objects."""
    labels = load_slp(slp_minimal)

    # Create test directory
    test_dir = tmp_path / "splits"
    test_dir.mkdir()
    labels.save(test_dir / "data.slp", embed=False)

    # Test with string path
    labels_set = read_labels_set(str(test_dir))
    assert len(labels_set) == 1
    assert "data" in labels_set
