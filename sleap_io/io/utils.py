"""Miscellaneous utilities for working with different I/O formats."""

from __future__ import annotations
import h5py  # type: ignore[import]
import numpy as np
from typing import Any, Union, Optional
from pathlib import Path


def read_hdf5_dataset(filename: str, dataset: str) -> np.ndarray:
    """Read data from an HDF5 file.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset.

    Returns:
        The data as an array.
    """
    with h5py.File(filename, "r") as f:
        data = f[dataset][()]
    return data


def _overwrite_hdf5_dataset(
    f: Union[h5py.File, h5py.Group], dataset: str, data: np.ndarray
):
    """Overwrite dataset in HDF5 file.

    Args:
        f: Handle to HDF5 file.
        dataset: Path to a dataset.
        data: Data to write to dataset.
    """
    try:
        del f[dataset]
    except KeyError:
        pass
    f.create_dataset(dataset, data=data)


def write_hdf5_dataset(filename: str, dataset: str, data: np.ndarray):
    """Write data to an HDF5 file.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset.
        data: Data to write to dataset.
    """
    with h5py.File(filename, "a") as f:  # "a": read/write if exists, create otherwise
        _overwrite_hdf5_dataset(f, dataset, data)


def read_hdf5_group(filename: str, group: str = "/") -> dict[str, np.ndarray]:
    """Read an entire group from an HDF5 file.

    Args:
        filename: Path an HDF5 file.
        group: Path to a group within the HDF5 file. Defaults to "/" (read the entire
            file).

    Returns:
        A flat dictionary with keys corresponding to dataset paths and values
        corresponding to the datasets as arrays.
    """
    data = {}

    def read_datasets(k, v):
        if type(v) == h5py.Dataset:
            data[v.name] = v[()]

    with h5py.File(filename, "r") as f:
        f[group].visititems(read_datasets)

    return data


def write_hdf5_group(filename: str, data: dict[str, np.ndarray]):
    """Write an entire group to an HDF5 file.

    Args:
        filename: Path an HDF5 file.
        data: A dictionary with keys corresponding to dataset/group paths and values
            corresponding to either sub group paths or the datasets as arrays.
    """

    def overwrite_hdf5_group(
        file_or_group: Union[h5py.File, h5py.Group], group_name: str
    ) -> h5py.Group:
        """Overwrite group in HDF5 file.

        Args:
            file_or_group: Path to an HDF5 file or parent group.
            group_name: Path to a group.

        Return:
            group: (Sub-)group under specified file or parent group.
        """
        try:
            del file_or_group[group_name]
        except KeyError:
            pass
        group = file_or_group.create_group(group_name)
        return group

    def write_group(parent_group, data_to_write):
        for name, dataset_or_group in data_to_write.items():
            if isinstance(dataset_or_group, dict):
                # Create (sub-)group under parent group (top level being the file)
                group = overwrite_hdf5_group(parent_group, name)
                write_group(group, dataset_or_group)  # Recall with new parent
            else:
                # Create dataset if dataset_or_group is a dataset
                _overwrite_hdf5_dataset(
                    f=parent_group, dataset=name, data=dataset_or_group
                )

    with h5py.File(filename, "a") as f:  # "a": read/write if exists, create otherwise
        write_group(f, data)


def read_hdf5_attrs(
    filename: str, dataset: str = "/", attribute: Optional[str] = None
) -> Union[Any, dict[str, Any]]:
    """Read attributes from an HDF5 dataset.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset or group from which attributes will be read.
        attribute: If specified, the attribute name to read. If `None` (the default),
            all attributes for the dataset will be returned.

    Returns:
        The attributes in a dictionary, or the attribute field if `attribute` was
        provided.
    """
    with h5py.File(filename, "r") as f:
        ds = f[dataset]
        if attribute is None:
            data = dict(ds.attrs)
        else:
            data = ds.attrs[attribute]
    return data


def write_hdf5_attrs(filename: str, dataset: str, attributes: dict[str, Any]):
    """Write attributes to an HDF5 dataset.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset or group to which attributes will be written.
        attributes: The attributes in a dictionary with the keys as the attribute names.
    """

    def _overwrite_hdf5_attr(
        group_or_dataset: Union[h5py.Group, h5py.Dataset], attr_name: str, data: Any
    ):
        """Overwrite attribute for group or dataset in HDF5 file.

        Args:
            group_or_dataset: Path to group or dataset in HDF5 file.
            attr_name: Name of attribute.
            data: Data to write to attribute.
        """
        try:
            del group_or_dataset.attrs[attr_name]
        except KeyError:
            pass
        group_or_dataset.attrs.create(attr_name, data)

    with h5py.File(filename, "a") as f:  # "a": read/write if exists, create otherwise
        ds = f[dataset]
        for attr_name, attr_value in attributes.items():
            _overwrite_hdf5_attr(ds, attr_name, attr_value)


def is_file_accessible(filename: str | Path) -> bool:
    """Check if a file is accessible.

    Args:
        filename: Path to a file.

    Returns:
        `True` if the file is accessible, `False` otherwise.

    Notes:
        This checks if the file readable by the current user by reading one byte from
        the file.
    """
    filename = Path(filename)
    try:
        with open(filename, "rb") as f:
            f.read(1)
        return True
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return False
