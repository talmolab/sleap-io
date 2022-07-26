"""Miscellaneous utilities for working with different I/O formats."""

from __future__ import annotations
import h5py  # type: ignore[import]
import numpy as np
from typing import Any, Union, Optional


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


def read_hdf5_attrs(
    filename, dataset: str = "/", attribute: Optional[str] = None
) -> Union[Any, dict[str, Any]]:
    """Read attributes from an HDF5 dataset.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset or group from which attributes will be read.
        attribute: If specified, the attribute name to read. If `None` (the default),
            all attributes for the dataset will be returned.

    Returns:
        The attributes in a dictionary, or the attribute field is `attribute` was
        provided.
    """
    with h5py.File(filename, "r") as f:
        ds = f[dataset]
        if attribute is None:
            data = dict(ds.attrs)
        else:
            data = ds.attrs[attribute]
    return data
