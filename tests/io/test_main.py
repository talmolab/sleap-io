"""Tests for functions in the sleap_io.io.main file."""
from sleap_io import Labels
from sleap_io.io.main import (
    load_slp,
    load_nwb,
    save_nwb,
    load_labelstudio,
    save_labelstudio,
    load_jabs,
    save_jabs,
)


def test_load_slp(slp_typical):
    """Test `load_slp` loads a .slp to a `Labels` object."""
    assert type(load_slp(slp_typical)) == Labels


def test_nwb(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    save_nwb(labels, tmp_path / "test_nwb.nwb")
    loaded_labels = load_nwb(tmp_path / "test_nwb.nwb")
    assert type(loaded_labels) == Labels
    assert len(loaded_labels) == len(labels)

    labels2 = load_slp(slp_typical)
    labels2.videos[0].filename = "test"
    save_nwb(labels2, tmp_path / "test_nwb.nwb", append=True)
    loaded_labels = load_nwb(tmp_path / "test_nwb.nwb")
    assert type(loaded_labels) == Labels
    assert len(loaded_labels) == (len(labels) + len(labels2))
    assert len(loaded_labels.videos) == 2


def test_labelstudio(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    save_labelstudio(labels, tmp_path / "test_labelstudio.json")
    loaded_labels = load_labelstudio(tmp_path / "test_labelstudio.json")
    assert type(loaded_labels) == Labels
    assert len(loaded_labels) == len(labels)


def test_jabs(tmp_path, jabs_real_data_v2, jabs_real_data_v5):
    labels_single = load_jabs(jabs_real_data_v2)
    assert isinstance(labels_single, Labels)
    save_jabs(labels_single, 2, tmp_path)

    labels_multi = load_jabs(jabs_real_data_v5)
    assert isinstance(labels_multi, Labels)
    save_jabs(labels_multi, 3, tmp_path)
    save_jabs(labels_multi, 4, tmp_path)
    save_jabs(labels_multi, 5, tmp_path)
