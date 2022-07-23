import pytest
import h5py
import numpy as np
from sleap_io.io.utils import read_hdf5_attrs, read_hdf5_dataset, read_hdf5_group


@pytest.fixture
def hdf5_file(tmp_path):
    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("ds1", data=np.array([0, 1, 2]))
        f.create_dataset("grp/ds2", data=np.array([3, 4, 5]))
        f.attrs["attr1"] = 0
        f.attrs["attr2"] = 1
    return path


def test_read_hdf5_attrs(hdf5_file):
    assert read_hdf5_attrs(hdf5_file, dataset="/", attribute="attr1") == 0
    assert read_hdf5_attrs(hdf5_file, dataset="/", attribute="attr2") == 1
    attrs = read_hdf5_attrs(hdf5_file, dataset="/")
    assert attrs == {"attr1": 0, "attr2": 1}


def test_read_hdf5_dataset(hdf5_file):
    np.testing.assert_array_equal(read_hdf5_dataset(hdf5_file, "ds1"), [0, 1, 2])


def test_read_hdf5_group(hdf5_file):
    data = read_hdf5_group(hdf5_file, group="/")
    np.testing.assert_array_equal(data["/ds1"], [0, 1, 2])
    np.testing.assert_array_equal(data["/grp/ds2"], [3, 4, 5])
    assert len(data.keys()) == 2
