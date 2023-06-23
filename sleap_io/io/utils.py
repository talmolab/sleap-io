"""Miscellaneous utilities for working with different I/O formats."""

from __future__ import annotations
import h5py  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
from typing import Any, Union, Optional, Generator
from sleap_io import Labels, LabeledFrame, PredictedInstance


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


def convert_predictions_to_dataframe(labels: Labels) -> pd.DataFrame:
    """Convert predictions data to a Pandas dataframe.

    Args:
        labels: A general label object.

    Returns:
        pd.DataFrame: A pandas data frame with the structured data with
        hierarchical columns. The column hierarchy is:
                "video_path",
                "skeleton_name",
                "track_name",
                "node_name",
        And it is indexed by the frames.

    Raises:
        ValueError: If no frames in the label objects contain predicted instances.
    """
    # Form pairs of labeled_frames and predicted instances
    labeled_frames = labels.labeled_frames
    all_frame_instance_tuples: Generator[
        tuple[LabeledFrame, PredictedInstance], None, None
    ] = (
        (label_frame, instance)  # type: ignore
        for label_frame in labeled_frames
        for instance in label_frame.predicted_instances
    )

    # Extract the data
    data_list = list()
    for labeled_frame, instance in all_frame_instance_tuples:
        # Traverse the nodes of the instances's skeleton
        skeleton = instance.skeleton
        for node in skeleton.nodes:
            row_dict = dict(
                frame_idx=labeled_frame.frame_idx,
                x=instance.points[node].x,
                y=instance.points[node].y,
                score=instance.points[node].score,  # type: ignore[attr-defined]
                node_name=node.name,
                skeleton_name=skeleton.name,
                track_name=instance.track.name if instance.track else "untracked",
                video_path=labeled_frame.video.filename,
            )
            data_list.append(row_dict)

    if not data_list:
        raise ValueError("No predicted instances found in labels object")

    labels_df = pd.DataFrame(data_list)

    # Reformat the data with columns for dict-like hierarchical data access.
    index = [
        "skeleton_name",
        "track_name",
        "node_name",
        "video_path",
        "frame_idx",
    ]

    labels_tidy_df = (
        labels_df.set_index(index)
        .unstack(level=[0, 1, 2, 3])
        .swaplevel(0, -1, axis=1)  # video_path on top while x, y score on bottom
        .sort_index(axis=1)  # Better format for columns
        .sort_index(axis=0)  # Sorts by frames
    )

    return labels_tidy_df
