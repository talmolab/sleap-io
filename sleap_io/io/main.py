"""This module contains high-level wrappers for utilizing different I/O backends."""

from __future__ import annotations
from sleap_io import Labels, Skeleton, Video
from sleap_io.io import slp, nwb, labelstudio, jabs, video_writing
from typing import Optional, Union
from pathlib import Path
import numpy as np


def load_slp(filename: str, open_videos: bool = True) -> Labels:
    """Load a SLEAP dataset.

    Args:
        filename: Path to a SLEAP labels file (`.slp`).
        open_videos: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).

    Returns:
        The dataset as a `Labels` object.
    """
    return slp.read_labels(filename, open_videos=open_videos)


def save_slp(
    labels: Labels,
    filename: str,
    embed: bool | str | list[tuple[Video, int]] | None = None,
):
    """Save a SLEAP dataset to a `.slp` file.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to ending with `.slp`.
        embed: Frames to embed in the saved labels file. One of `None`, `True`,
            `"all"`, `"user"`, `"suggestions"`, `"user+suggestions"`, `"source"` or list
            of tuples of `(video, frame_idx)`.

            If `None` is specified (the default) and the labels contains embedded
            frames, those embedded frames will be re-saved to the new file.

            If `True` or `"all"`, all labeled frames and suggested frames will be
            embedded.

            If `"source"` is specified, no images will be embedded and the source video
            will be restored if available.

            This argument is only valid for the SLP backend.
    """
    return slp.write_labels(filename, labels, embed=embed)


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
        filename: Path to the label-studio annotation file in JSON format.
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
        filename: The filename(s) of the video. Supported extensions: "mp4", "avi",
            "mov", "mj2", "mkv", "h5", "hdf5", "slp", "png", "jpg", "jpeg", "tif",
            "tiff", "bmp". If the filename is a list, a list of image filenames are
            expected. If filename is a folder, it will be searched for images.

    Returns:
        A `Video` object.
    """
    return Video.from_filename(filename, **kwargs)


def save_video(
    frames: np.ndarray | Video,
    filename: str | Path,
    fps: float = 30,
    pixelformat: str = "yuv420p",
    codec: str = "libx264",
    crf: int = 25,
    preset: str = "superfast",
    output_params: list | None = None,
):
    """Write a list of frames to a video file.

    Args:
        frames: Sequence of frames to write to video. Each frame should be a 2D or 3D
            numpy array with dimensions (height, width) or (height, width, channels).
        filename: Path to output video file.
        fps: Frames per second. Defaults to 30.
        pixelformat: Pixel format for video. Defaults to "yuv420p".
        codec: Codec to use for encoding. Defaults to "libx264".
        crf: Constant rate factor to control lossiness of video. Values go from 2 to 32,
            with numbers in the 18 to 30 range being most common. Lower values mean less
            compressed/higher quality. Defaults to 25. No effect if codec is not
            "libx264".
        preset: H264 encoding preset. Defaults to "superfast". No effect if codec is not
            "libx264".
        output_params: Additional output parameters for FFMPEG. This should be a list of
            strings corresponding to command line arguments for FFMPEG and libx264. Use
            `ffmpeg -h encoder=libx264` to see all options for libx264 output_params.

    See also: `sio.VideoWriter`
    """
    if output_params is None:
        output_params = []

    with video_writing.VideoWriter(
        filename,
        fps=fps,
        pixelformat=pixelformat,
        codec=codec,
        crf=crf,
        preset=preset,
        output_params=output_params,
    ) as writer:
        for frame in frames:
            writer(frame)


def load_file(
    filename: str | Path, format: Optional[str] = None, **kwargs
) -> Union[Labels, Video]:
    """Load a file and return the appropriate object.

    Args:
        filename: Path to a file.
        format: Optional format to load as. If not provided, will be inferred from the
            file extension. Available formats are: "slp", "nwb", "labelstudio", "jabs"
            and "video".

    Returns:
        A `Labels` or `Video` object.
    """
    if isinstance(filename, Path):
        filename = filename.as_posix()

    if format is None:
        if filename.endswith(".slp"):
            format = "slp"
        elif filename.endswith(".nwb"):
            format = "nwb"
        elif filename.endswith(".json"):
            format = "json"
        elif filename.endswith(".h5"):
            format = "jabs"
        else:
            for vid_ext in Video.EXTS:
                if filename.endswith(vid_ext):
                    format = "video"
                    break
        if format is None:
            raise ValueError(f"Could not infer format from filename: '{filename}'.")

    if filename.endswith(".slp"):
        return load_slp(filename, **kwargs)
    elif filename.endswith(".nwb"):
        return load_nwb(filename, **kwargs)
    elif filename.endswith(".json"):
        return load_labelstudio(filename, **kwargs)
    elif filename.endswith(".h5"):
        return load_jabs(filename, **kwargs)
    elif format == "video":
        return load_video(filename, **kwargs)


def save_file(
    labels: Labels, filename: str | Path, format: Optional[str] = None, **kwargs
):
    """Save a file based on the extension.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to.
        format: Optional format to save as. If not provided, will be inferred from the
            file extension. Available formats are: "slp", "nwb", "labelstudio" and
            "jabs".
    """
    if isinstance(filename, Path):
        filename = str(filename)

    if format is None:
        if filename.endswith(".slp"):
            format = "slp"
        elif filename.endswith(".nwb"):
            format = "nwb"
        elif filename.endswith(".json"):
            format = "labelstudio"
        elif "pose_version" in kwargs:
            format = "jabs"

    if format == "slp":
        save_slp(labels, filename, **kwargs)
    elif format == "nwb":
        save_nwb(labels, filename, **kwargs)
    elif format == "labelstudio":
        save_labelstudio(labels, filename, **kwargs)
    elif format == "jabs":
        pose_version = kwargs.pop("pose_version", 5)
        root_folder = kwargs.pop("root_folder", filename)
        save_jabs(labels, pose_version=pose_version, root_folder=root_folder)
    else:
        raise ValueError(f"Unknown format '{format}' for filename: '{filename}'.")
