"""Miscellaneous utilities for working with different I/O formats."""

from __future__ import annotations
import errno
import os
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
        filename: Path to an HDF5 file.
        dataset: Path to a dataset.
        data: Data to write to dataset.
    """
    try:
        del f[dataset]
    except KeyError:
        pass
    f.create_dataset(dataset, data=data)


def write_hdf5_dataset(filename: str, dataset: str, data: np.ndarray):
    """Write data from an HDF5 file.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset.
        data: Data to write to dataset.
    """
    with h5py.File(filename, "r+") as f:
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

    with h5py.File(filename, "r+") as f:
        write_group(f, data)


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

    with h5py.File(filename, "r+") as f:
        ds = f[dataset]
        for attr_name, attr_value in attributes.items():
            _overwrite_hdf5_attr(ds, attr_name, attr_value)


def resolve_path(old_path_str: str, starting_points_str: Optional[list[str]] = None):
    """Find a path given an old path and a starting point.

    To resolve the path, we assume that the file/directory is uniquely named. First, we
    attempt to find the file/directory at the old_path_str. If the old_path_str no
    longer exists, then we search for the item in the list of starting points.

    If no starting points were given, then we set the current working directory as the
    starting point. First, we swap the anchor of the old path with the anchor of the
    starting point path and search for the item at this new path. Lastly,
    we search for the item at the starting point.

    Args:
        old_path_str: Path to file or directory that may no longer exist.
        starting_points_str: List of paths to file or directory to search for
            file/directorythat exists (where the search begins).
            Initialized to the current working directory if none is given.

    Returns:
        Path to file or directory that matches unique file/directory name of the
        old_path. FileNotFoundError raised if no path was found.

    Raises:
        FileNotFoundError: raised if no file/directory was found.
    """
    if starting_points_str is None:
        starting_points_str = []

    def find_item(starting_point, item_to_find):
        items_found: list[Path] = sorted(starting_point.rglob(str(item_to_find)))
        if len(items_found) > 0:
            new_path = items_found[0]
            if len(items_found) > 1:
                print(
                    f"Found multiple items named '{item_to_find}'.\n"
                    f"Returning item with shortest relative path: '{new_path}'."
                )
            print(f"Found item at {new_path}.")
            return new_path
        return None

    # Convert strings to absolute Path objects
    if len(starting_points_str) == 0:
        use_cwd = True
        starting_points_str = ["."]
    else:
        use_cwd = False
    starting_points = [Path(sp_str).resolve() for sp_str in starting_points_str]
    old_path: Path = Path(old_path_str).resolve()
    item_to_find: Path = Path(old_path.parts[-1])

    # Check if file/directory exists at old path
    if old_path.exists():
        return old_path

    if not use_cwd:
        # Use starting point to find item
        for sp in starting_points:
            new_path = find_item(sp, item_to_find)
            if new_path is not None:
                return new_path

        # TODO(LM): The starting point will likely be the path to the .slp file which is
        # usually located at the top level project directory. We should be able to find
        # any similarities in the the .slp path and any other project componenets, then
        # replace the prefix (prior to the common path) with the .slp path (which is a
        # valid path).
    else:
        # No starting point given, try to automatically resolve path using cwd

        # Get parts of each path
        old_path_parts = old_path.parts
        starting_point_parts = starting_points[0].parts

        # Change anchor of old path with starting point anchor
        new_path_parts = list(old_path_parts)
        new_path_parts[0] = starting_point_parts[0]
        new_path = Path(*new_path_parts).absolute()
        if new_path.exists():
            return new_path

        new_path = find_item(starting_points[0], item_to_find)
        if new_path is not None:
            return new_path

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(item_to_find))
