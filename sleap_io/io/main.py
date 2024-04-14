"""This module contains high-level wrappers for utilizing different I/O backends."""

from __future__ import annotations
from sleap_io import Labels, Skeleton, Video
from sleap_io.io import slp, nwb, labelstudio, jabs
from typing import Optional, Union
from pathlib import Path


def load_slp(filename: str) -> Labels:
    """Load a SLEAP dataset.

    Args:
        filename: Path to a SLEAP labels file (`.slp`).

    Returns:
        The dataset as a `Labels` object.
    """
    return slp.read_labels(filename)


def save_slp(labels: Labels, filename: str):
    """Save a SLEAP dataset to a `.slp` file.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to ending with `.slp`.
    """
    return slp.write_labels(filename, labels)


def load_nwb(filename: str) -> Labels:
    """Load an NWB dataset as a SLEAP `Labels` object.

    Args:
        filename: Path to a NWB file (`.nwb`).

    Returns:
        The dataset as a `Labels` object.
    """
    return nwb.read_nwb(filename)


def save_nwb(labels: Labels, filename: str, append: bool = True):
    """Save a SLEAP dataset to NWB format.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to NWB file to save to. Must end in `.nwb`.
        append: If `True` (the default), append to existing NWB file. File will be
            created if it does not exist.

    See also: nwb.write_nwb, nwb.append_nwb
    """
    if append and Path(filename).exists():
        nwb.append_nwb(labels, filename)
    else:
        nwb.write_nwb(labels, filename)


def load_labelstudio(
    filename: str, skeleton: Optional[Union[Skeleton, list[str]]] = None
) -> Labels:
    """Read Label Studio-style annotations from a file and return a `Labels` object.

    Args:
        labels_path: Path to the label-studio annotation file in JSON format.
        skeleton: An optional `Skeleton` object or list of node names. If not provided
            (the default), skeleton will be inferred from the data. It may be useful to
            provide this so the keypoint label types can be filtered to just the ones in
            the skeleton.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    return labelstudio.read_labels(filename, skeleton=skeleton)


def save_labelstudio(labels: Labels, filename: str):
    """Save a SLEAP dataset to Label Studio format.
    
    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to ending with `.json`.
    """
    labelstudio.write_labels(labels, filename)


def load_jabs(filename: str, skeleton: Optional[Skeleton] = None) -> Labels:
    """Read JABS-style predictions from a file and return a `Labels` object.

    Args:
        filename: Path to the jabs h5 pose file.
        skeleton: An optional `Skeleton` object.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    return jabs.read_labels(filename, skeleton=skeleton)


def save_jabs(labels: Labels, pose_version: int, root_folder: Optional[str] = None):
    """Save a SLEAP dataset to JABS pose file format.

    Args:
        labels: SLEAP `Labels` object.
        pose_version: The JABS pose version to write data out.
        root_folder: Optional root folder where the files should be saved.

    Note:
        Filenames for JABS poses are based on video filenames.
    """
    jabs.write_labels(labels, pose_version, root_folder)


def load_video(filename: str, **kwargs) -> Video:
    """Load a video file.
    
    Args:
        filename: Path to a video file.

    Returns:
        A `Video` object.
    """
    return Video.from_filename(filename, **kwargs)


def load_file(filename: str, **kwargs) -> Union[Labels, Video]:
    """Load a file and return the appropriate object.

    Args:
        filename: Path to a file.

    Returns:
        A `Labels` or `Video` object.
    """
    if filename.endswith(".slp"):
        return load_slp(filename)
    elif filename.endswith(".nwb"):
        return load_nwb(filename)
    elif filename.endswith(".json"):
        return load_labelstudio(filename)
    elif filename.endswith(".h5"):
        return load_jabs(filename)
    else:
        return load_video(filename, **kwargs)