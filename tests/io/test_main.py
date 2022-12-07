"""Tests for functions in the sleap_io.io.main file."""
from sleap_io import Labels
from sleap_io.io.main import (
    load_slp,
    load_nwb,
    save_nwb,
    load_labelstudio,
    save_labelstudio,
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


def test_labelstudio(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    save_labelstudio(labels, tmp_path / "test_labelstudio.json")
    loaded_labels = load_labelstudio(tmp_path / "test_labelstudio.json")
    assert type(loaded_labels) == Labels
    assert len(loaded_labels) == len(labels)
