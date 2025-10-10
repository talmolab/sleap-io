"""Tests for functions in the sleap_io.io.main file."""

import shutil

import pytest

from sleap_io import Labels, LabelsSet
from sleap_io.io.main import (
    load_file,
    load_jabs,
    load_labels_set,
    load_labelstudio,
    load_nwb,
    load_slp,
    load_video,
    save_file,
    save_jabs,
    save_labelstudio,
    save_nwb,
    save_video,
)


def test_load_slp(slp_typical):
    """Test `load_slp` loads a .slp to a `Labels` object."""
    assert type(load_slp(slp_typical)) is Labels
    assert type(load_file(slp_typical)) is Labels


def test_nwb(tmp_path, slp_typical, slp_real_data):
    # Test with predictions (slp_typical has predictions)
    labels_pred = load_slp(slp_typical)
    save_nwb(labels_pred, tmp_path / "test_nwb_pred.nwb")
    loaded_labels = load_nwb(tmp_path / "test_nwb_pred.nwb")
    assert type(loaded_labels) is Labels
    assert type(load_file(tmp_path / "test_nwb_pred.nwb")) is Labels

    # Test with annotations (slp_real_data has user instances)
    labels_ann = load_slp(slp_real_data)
    save_nwb(labels_ann, tmp_path / "test_nwb_ann.nwb")
    loaded_labels = load_nwb(tmp_path / "test_nwb_ann.nwb")
    assert type(loaded_labels) is Labels

    # Test overwriting (no append)
    save_nwb(labels_pred, tmp_path / "test_nwb_pred.nwb")  # Overwrites
    loaded_labels = load_nwb(tmp_path / "test_nwb_pred.nwb")
    assert type(loaded_labels) is Labels


def test_labelstudio(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    save_labelstudio(labels, tmp_path / "test_labelstudio.json")
    loaded_labels = load_labelstudio(tmp_path / "test_labelstudio.json")
    assert type(loaded_labels) is Labels
    assert type(load_file(tmp_path / "test_labelstudio.json")) is Labels
    assert len(loaded_labels) == len(labels)


def test_jabs(tmp_path, jabs_real_data_v2, jabs_real_data_v5):
    labels_single = load_jabs(jabs_real_data_v2)
    assert isinstance(labels_single, Labels)
    save_jabs(labels_single, 2, tmp_path)
    labels_single_written = load_jabs(str(tmp_path / jabs_real_data_v2))
    # Confidence field is not preserved, so just check number of labels
    assert len(labels_single) == len(labels_single_written)
    assert len(labels_single.videos) == len(labels_single_written.videos)
    assert type(load_file(jabs_real_data_v2)) is Labels

    labels_multi = load_jabs(jabs_real_data_v5)
    assert isinstance(labels_multi, Labels)
    save_jabs(labels_multi, 3, tmp_path)
    save_jabs(labels_multi, 4, tmp_path)
    save_jabs(labels_multi, 5, tmp_path)
    labels_v5_written = load_jabs(str(tmp_path / jabs_real_data_v5))
    # v5 contains all v4 and v3 data, so only need to check v5
    # Confidence field and ordering of identities is not preserved, so just check
    # number of labels
    assert len(labels_v5_written) == len(labels_multi)
    assert len(labels_v5_written.videos) == len(labels_multi.videos)


def test_load_video(centered_pair_low_quality_path):
    assert load_video(centered_pair_low_quality_path).shape == (1100, 384, 384, 1)
    assert load_file(centered_pair_low_quality_path).shape == (1100, 384, 384, 1)


def test_load_video_MP4(centered_pair_low_quality_path, tmp_path):
    """Test loading video with uppercase extension (.MP4)."""
    # Copy the existing fixture to a temp file with uppercase extension
    uppercase_video_path = tmp_path / "centered_pair.MP4"
    shutil.copy(centered_pair_low_quality_path, uppercase_video_path)

    # Test with string path
    video = load_video(str(uppercase_video_path))
    assert video.shape == (1100, 384, 384, 1)

    # Test with Path object
    video = load_video(uppercase_video_path)
    assert video.shape == (1100, 384, 384, 1)


@pytest.mark.parametrize("format", ["slp", "nwb", "labelstudio", "jabs"])
def test_load_save_file(format, tmp_path, slp_typical, jabs_real_data_v5):
    if format == "slp":
        labels = load_slp(slp_typical)
        save_file(labels, tmp_path / "test.slp")
        assert type(load_file(tmp_path / "test.slp")) is Labels
    elif format == "nwb":
        labels = load_slp(slp_typical)
        save_file(labels, tmp_path / "test.nwb")
        assert type(load_file(tmp_path / "test.nwb")) is Labels
    elif format == "labelstudio":
        labels = load_slp(slp_typical)
        save_file(labels, tmp_path / "test.json")
        assert type(load_file(tmp_path / "test.json")) is Labels
    elif format == "jabs":
        labels = load_jabs(jabs_real_data_v5)
        save_file(labels, tmp_path, pose_version=5)
        assert type(load_file(tmp_path / jabs_real_data_v5)) is Labels

        save_file(labels, tmp_path, format="jabs")
        assert type(load_file(tmp_path / jabs_real_data_v5)) is Labels


def test_load_save_file_invalid():
    with pytest.raises(ValueError):
        load_file("invalid_file.ext")

    with pytest.raises(ValueError):
        save_file(Labels(), "invalid_file.ext")


def test_save_video(centered_pair_low_quality_video, tmp_path):
    imgs = centered_pair_low_quality_video[:4]
    save_video(imgs, tmp_path / "output.mp4")
    vid = load_video(tmp_path / "output.mp4")
    assert vid.shape == (4, 384, 384, 1)
    save_video(vid, tmp_path / "output2.mp4")
    vid2 = load_video(tmp_path / "output2.mp4")
    assert vid2.shape == (4, 384, 384, 1)


def test_save_file_ultralytics_autodetect(tmp_path, slp_typical):
    """Test ultralytics format auto-detection in save_file."""
    labels = load_slp(slp_typical)

    # Test with directory path (should auto-detect ultralytics)
    output_dir = tmp_path / "ultralytics_output"
    output_dir.mkdir()
    save_file(labels, str(output_dir))  # No format specified

    # Check that files were created in ultralytics format
    assert (output_dir / "data.yaml").exists()
    # Default split creates train/val directories
    assert (output_dir / "train").exists()
    assert (output_dir / "val").exists()

    # Test with split_ratios kwarg (should auto-detect ultralytics)
    output_dir2 = tmp_path / "ultralytics_output2"
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    save_file(labels, str(output_dir2), split_ratios=splits)

    assert (output_dir2 / "data.yaml").exists()
    # With 3-way split, creates train/val/test
    assert (output_dir2 / "train").exists()
    assert (output_dir2 / "val").exists()
    assert (output_dir2 / "test").exists()


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
    import numpy as np
    import yaml

    from sleap_io import LabelsSet

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
    labels_set = load_labels_set(dataset_path)

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


def test_load_labels_set_kwargs_passing(tmp_path):
    """Test that kwargs are properly passed to format-specific loaders."""
    import numpy as np
    import yaml

    from sleap_io import Node, Skeleton

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
    )

    assert len(labels_set) == 1
    assert "val" in labels_set
    assert labels_set["val"].skeletons[0].nodes[0].name == "custom_node"


def test_load_labels_set_format_detection_edge_cases(tmp_path, slp_minimal):
    """Test edge cases in format detection for load_labels_set."""
    labels = load_slp(slp_minimal)

    # Test single SLP file wrapped in a list
    single_file = tmp_path / "single.slp"
    labels.save(single_file, embed=False)

    # Should auto-detect SLP format from file extension when in list
    labels_set = load_labels_set([str(single_file)])
    assert len(labels_set) == 1
    assert "single" in labels_set

    # Test non-directory path with explicit format
    # This tests the edge case where a single file path is provided but
    # format is explicit
    try:
        # This should fail because SLP format expects directory/list/dict
        load_labels_set(str(single_file), format="slp")
    except ValueError as e:
        assert "Path must be a directory" in str(e)
