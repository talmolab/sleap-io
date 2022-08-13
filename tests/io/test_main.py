"""Tests for functions in the sleap_io.io.main file."""
from sleap_io import Labels
from sleap_io.io.main import load_slp


def test_load_slp(slp_typical):
    """Test `load_slp` loads a .slp to a `Labels` object."""
    assert type(load_slp(slp_typical)) == Labels
