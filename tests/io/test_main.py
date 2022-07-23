from sleap_io import Labels
from sleap_io.io.main import load_slp


def test_load_slp(slp_file1):
    assert type(load_slp(slp_file1)) == Labels
