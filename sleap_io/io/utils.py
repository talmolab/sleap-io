"""Miscellaneous utilities for working with different I/O formats."""

import h5py
import numpy as np


def read_hdf5(filename: str, dataset: str = "/") -> dict[str, np.ndarray]:
    """Read data from an HDF5 file.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset or group. If a dataset, return the entire
            dataset as an array. If group, all datasets contained within the
            group will be recursively loaded and returned in a dict keyed by
            their full path. Defaults to "/" (load everything).

    Returns:
        The data as an array (for datasets) or dictionary (for groups).
    """
    data = {}

    def read_datasets(k, v):
        if type(v) == h5py.Dataset:
            data[v.name] = v[()]

    with h5py.File(filename, "r") as f:
        if type(f[dataset]) == h5py.Group:
            f.visititems(read_datasets)
        elif type(f[dataset]) == h5py.Dataset:
            data = f[dataset][()]
    return data
