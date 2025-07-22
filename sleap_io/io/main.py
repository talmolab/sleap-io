"""This module contains high-level wrappers for utilizing different I/O backends."""

from __future__ import annotations
from sleap_io import Labels, Skeleton, Video
from sleap_io.io import slp, nwb, labelstudio, jabs, video_writing, ultralytics
from sleap_io.io.skeleton import (
    SkeletonDecoder,
    SkeletonEncoder,
    SkeletonYAMLDecoder,
    SkeletonYAMLEncoder,
)
from typing import Optional, Union, List
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
    embed: bool | str | list[tuple[Video, int]] | None = False,
    restore_original_videos: bool = True,
    verbose: bool = True,
):
    """Save a SLEAP dataset to a `.slp` file.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to ending with `.slp`.
        embed: Frames to embed in the saved labels file. One of `None`, `True`,
            `"all"`, `"user"`, `"suggestions"`, `"user+suggestions"`, `"source"` or list
            of tuples of `(video, frame_idx)`.

            If `False` is specified (the default), the source video will be restored
            if available, otherwise the embedded frames will be re-saved.

            If `True` or `"all"`, all labeled frames and suggested frames will be
            embedded.

            If `"source"` is specified, no images will be embedded and the source video
            will be restored if available.

            This argument is only valid for the SLP backend.
        restore_original_videos: If `True` (default) and `embed=False`, use original
            video files. If `False` and `embed=False`, keep references to source
            `.pkg.slp` files. Only applies when `embed=False`.
        verbose: If `True` (the default), display a progress bar when embedding frames.
    """
    return slp.write_labels(
        filename,
        labels,
        embed=embed,
        restore_original_videos=restore_original_videos,
        verbose=verbose,
    )


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


def load_ultralytics(
    dataset_path: str,
    split: str = "train",
    skeleton: Optional[Skeleton] = None,
    **kwargs,
) -> Labels:
    """Load an Ultralytics YOLO pose dataset as a SLEAP `Labels` object.

    Args:
        dataset_path: Path to the Ultralytics dataset root directory containing data.yaml.
        split: Dataset split to read ('train', 'val', or 'test'). Defaults to 'train'.
        skeleton: Optional skeleton to use. If not provided, will be inferred from data.yaml.

    Returns:
        The dataset as a `Labels` object.
    """
    return ultralytics.read_labels(
        dataset_path, split=split, skeleton=skeleton, **kwargs
    )


def save_ultralytics(
    labels: Labels,
    dataset_path: str,
    split_ratios: dict = {"train": 0.8, "val": 0.2},
    **kwargs,
):
    """Save a SLEAP dataset to Ultralytics YOLO pose format.

    Args:
        labels: A SLEAP `Labels` object.
        dataset_path: Path to save the Ultralytics dataset.
        split_ratios: Dictionary mapping split names to ratios (must sum to 1.0).
                     Defaults to {"train": 0.8, "val": 0.2}.
    """
    ultralytics.write_labels(labels, dataset_path, split_ratios=split_ratios, **kwargs)


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
            file extension. Available formats are: "slp", "nwb", "labelstudio", "jabs",
            "ultralytics", and "video".

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
        elif filename.endswith("data.yaml") or (
            Path(filename).is_dir() and (Path(filename) / "data.yaml").exists()
        ):
            format = "ultralytics"
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
    elif format == "ultralytics":
        return load_ultralytics(filename, **kwargs)
    elif format == "video":
        return load_video(filename, **kwargs)


def save_file(
    labels: Labels,
    filename: str | Path,
    format: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
):
    """Save a file based on the extension.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to.
        format: Optional format to save as. If not provided, will be inferred from the
            file extension. Available formats are: "slp", "nwb", "labelstudio", "jabs",
            and "ultralytics".
        verbose: If `True` (the default), display a progress bar when embedding frames
            (only applies to the SLP format).
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
        elif "split_ratios" in kwargs or Path(filename).is_dir():
            format = "ultralytics"

    if format == "slp":
        save_slp(labels, filename, verbose=verbose, **kwargs)
    elif format == "nwb":
        save_nwb(labels, filename, **kwargs)
    elif format == "labelstudio":
        save_labelstudio(labels, filename, **kwargs)
    elif format == "jabs":
        pose_version = kwargs.pop("pose_version", 5)
        root_folder = kwargs.pop("root_folder", filename)
        save_jabs(labels, pose_version=pose_version, root_folder=root_folder)
    elif format == "ultralytics":
        save_ultralytics(labels, filename, **kwargs)
    else:
        raise ValueError(f"Unknown format '{format}' for filename: '{filename}'.")


def load_skeleton(filename: str | Path) -> Union[Skeleton, List[Skeleton]]:
    """Load skeleton(s) from a JSON, YAML, or SLP file.

    Args:
        filename: Path to a skeleton file. Supported formats:
            - JSON: Standalone skeleton or training config with embedded skeletons
            - YAML: Simplified skeleton format
            - SLP: SLEAP project file

    Returns:
        A single `Skeleton` or list of `Skeleton` objects.

    Notes:
        This function loads skeletons from various file types:
        - JSON files: Can be standalone skeleton files (jsonpickle format) or training
          config files with embedded skeletons
        - YAML files: Use a simplified human-readable format
        - SLP files: Extracts skeletons from SLEAP project files
        The format is detected based on the file extension and content.
    """
    if isinstance(filename, Path):
        filename = str(filename)

    # Detect format based on extension
    if filename.lower().endswith(".slp"):
        # SLP format - extract skeletons from SLEAP file
        from sleap_io.io.slp import read_skeletons

        return read_skeletons(filename)
    elif filename.lower().endswith((".yaml", ".yml")):
        # YAML format
        with open(filename, "r") as f:
            yaml_data = f.read()
        decoder = SkeletonYAMLDecoder()
        return decoder.decode(yaml_data)
    else:
        # JSON format (default) - could be standalone or training config
        import json

        with open(filename, "r") as f:
            json_data = f.read()

        # Try to detect if this is a training config file
        try:
            data = json.loads(json_data)
            if isinstance(data, dict) and "data" in data:
                if "labels" in data["data"] and "skeletons" in data["data"]["labels"]:
                    # This is a training config file with embedded skeletons
                    decoder = SkeletonDecoder()
                    return decoder.decode(data["data"]["labels"]["skeletons"])
        except (json.JSONDecodeError, KeyError, TypeError):
            # Not a training config or invalid JSON structure
            pass

        # Fall back to regular skeleton JSON decoding
        decoder = SkeletonDecoder()
        return decoder.decode(json_data)


def save_skeleton(skeleton: Union[Skeleton, List[Skeleton]], filename: str | Path):
    """Save skeleton(s) to a JSON or YAML file.

    Args:
        skeleton: A single `Skeleton` or list of `Skeleton` objects to save.
        filename: Path to save the skeleton file.

    Notes:
        This function saves skeletons in either JSON or YAML format based on the
        file extension. JSON files use the jsonpickle format compatible with SLEAP,
        while YAML files use a simplified human-readable format.
    """
    if isinstance(filename, Path):
        filename = str(filename)

    # Detect format based on extension
    if filename.lower().endswith((".yaml", ".yml")):
        # YAML format
        encoder = SkeletonYAMLEncoder()
        yaml_data = encoder.encode(skeleton)
        with open(filename, "w") as f:
            f.write(yaml_data)
    else:
        # JSON format (default)
        encoder = SkeletonEncoder()
        json_data = encoder.encode(skeleton)
        with open(filename, "w") as f:
            f.write(json_data)
