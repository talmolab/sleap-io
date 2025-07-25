"""Tests for functions in the sleap_io.io.main file."""

import pytest

from sleap_io import Labels
from sleap_io.io.main import (
    load_file,
    load_jabs,
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


def test_nwb(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    save_nwb(labels, tmp_path / "test_nwb.nwb")
    loaded_labels = load_nwb(tmp_path / "test_nwb.nwb")
    assert type(loaded_labels) is Labels
    assert type(load_file(tmp_path / "test_nwb.nwb")) is Labels
    assert len(loaded_labels) == len(labels)

    labels2 = load_slp(slp_typical)
    labels2.videos[0].filename = "test"
    save_nwb(labels2, tmp_path / "test_nwb.nwb", append=True)
    loaded_labels = load_nwb(tmp_path / "test_nwb.nwb")
    assert type(loaded_labels) is Labels
    assert len(loaded_labels) == (len(labels) + len(labels2))
    assert len(loaded_labels.videos) == 2


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
    save_file(labels, str(output_dir2), split_ratios={"train": 0.8, "val": 0.1, "test": 0.1})
    
    assert (output_dir2 / "data.yaml").exists()
    # With 3-way split, creates train/val/test
    assert (output_dir2 / "train").exists()
    assert (output_dir2 / "val").exists()
    assert (output_dir2 / "test").exists()
