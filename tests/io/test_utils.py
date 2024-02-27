"""Tests for the sleap_io.io.test_utils file."""

import pytest
import h5py
import numpy as np
from sleap_io.io.utils import (
    read_hdf5_attrs,
    read_hdf5_dataset,
    read_hdf5_group,
    write_hdf5_dataset,
    write_hdf5_group,
    write_hdf5_attrs,
)


@pytest.fixture
def hdf5_file(tmp_path):
    """Define hdf5 fixture."""
    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("ds1", data=np.array([0, 1, 2]))
        f.create_dataset("grp/ds2", data=np.array([3, 4, 5]))
        f.attrs["attr1"] = 0
        f.attrs["attr2"] = 1
    return path


def test_read_hdf5_dataset(hdf5_file):
    """Test `read_hdf5_dataset` can read hdf5 datasets."""
    np.testing.assert_array_equal(read_hdf5_dataset(hdf5_file, "ds1"), [0, 1, 2])


def test_write_hdf5_dataset(hdf5_file):
    """Test `write_hdf5_dataset` can write hdf5 datasets."""

    def write_read_assert_dataset(file, dataset, data):
        write_hdf5_dataset(file, dataset, data)
        np.testing.assert_array_equal(read_hdf5_dataset(file, dataset), data)

    # Overwrite existing dataset
    write_read_assert_dataset(hdf5_file, dataset="ds1", data=np.array([1, 2]))
    write_read_assert_dataset(hdf5_file, dataset="grp/ds2", data=np.array([4, 5]))

    # Write new dataset
    write_read_assert_dataset(hdf5_file, dataset="ds3", data=np.array([1, 2]))
    write_read_assert_dataset(hdf5_file, dataset="grp/ds4", data=np.array([4, 5]))


def test_read_hdf5_group(hdf5_file):
    """Test `read_hdf5_group` can read hdf5 groups."""
    data = read_hdf5_group(hdf5_file, group="/")
    np.testing.assert_array_equal(data["/ds1"], [0, 1, 2])
    np.testing.assert_array_equal(data["/grp/ds2"], [3, 4, 5])
    assert len(data.keys()) == 2


def test_write_hdf5_group(hdf5_file):
    """Test `write_hdf5_group` can write hdf5 groups."""

    def write_read_assert_group(file, data, group, values):
        write_hdf5_group(file, data)
        read = read_hdf5_group if isinstance(data.values(), dict) else read_hdf5_dataset
        np.testing.assert_array_equal(read(file, group), values)

    # Overwrite existing group
    write_read_assert_group(
        hdf5_file, data={"ds1": np.array([1, 2])}, group="ds1", values=np.array([1, 2])
    )  # Expect dataset to be overwritten
    write_read_assert_group(
        hdf5_file,
        data={"grp/ds2": np.array([4, 5])},
        group="grp/ds2",
        values=np.array([4, 5]),
    )  # Expect group to be overwritten
    write_read_assert_group(
        hdf5_file,
        data={"grp": {"ds2": np.array([6, 7])}},
        group="grp/ds2",
        values=np.array([6, 7]),
    )  # Test different way of writing group

    # Write new group
    write_read_assert_group(
        hdf5_file, data={"ds3": np.array([1, 2])}, group="ds3", values=np.array([1, 2])
    )  # Expect just a dataset
    write_read_assert_group(
        hdf5_file,
        data={"grp/ds4": np.array([4, 5])},
        group="grp/ds4",
        values=np.array([4, 5]),
    )  # Expect a group and a dataset
    write_read_assert_group(
        hdf5_file,
        data={"grp": {"ds5": np.array([6, 7])}},
        group="grp/ds5",
        values=np.array([6, 7]),
    )  # Test different way of writing group with existing group, but new dataset
    write_read_assert_group(
        hdf5_file,
        data={"grp2": {"ds6": np.array([0, 3])}},
        group="grp2/ds6",
        values=np.array([0, 3]),
    )  # Test different way of writing group with new group

    # Ensure entire group is overwritten
    with pytest.raises(KeyError):
        np.testing.assert_array_equal(
            read_hdf5_group(hdf5_file, group="grp/ds4"), np.array([4, 5])
        )


def test_read_hdf5_attrs(hdf5_file):
    """Test `read_hdf5_attrs` can read hdf5 attributes."""
    assert read_hdf5_attrs(hdf5_file, dataset="/", attribute="attr1") == 0
    assert read_hdf5_attrs(hdf5_file, dataset="/", attribute="attr2") == 1
    attrs = read_hdf5_attrs(hdf5_file, dataset="/")
    assert attrs == {"attr1": 0, "attr2": 1}


def test_write_hdf5_attrs(hdf5_file):
    """Test `write_hdf5_attrs` can write hdf5 attributes."""

    def write_read_assert_attrs(file, dataset, attributes):
        existing_attrs = read_hdf5_attrs(file, dataset)
        write_hdf5_attrs(file, dataset, attributes)
        assert read_hdf5_attrs(hdf5_file, dataset) == {**existing_attrs, **attributes}

    # Add new attributes
    write_read_assert_attrs(
        hdf5_file, dataset="ds1", attributes={"attr3": 2}
    )  # Add attributes to dataset
    write_read_assert_attrs(
        hdf5_file, dataset="grp", attributes={"attr4": 3}
    )  # Add attributes to group
    write_read_assert_attrs(
        hdf5_file, dataset="grp/ds2", attributes={"attr5": 4, "attr6": 5}
    )  # Add multiple attributes to dataset inside group

    # Append attributes
    write_read_assert_attrs(hdf5_file, dataset="/", attributes={"attr7": 6})
