"""This module contains high-level wrappers for utilizing different I/O backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from sleap_io.io import (
    coco,
    dlc,
    jabs,
    labelstudio,
    nwb,
    slp,
    ultralytics,
    video_writing,
)
from sleap_io.io.skeleton import (
    SkeletonDecoder,
    SkeletonEncoder,
    SkeletonYAMLDecoder,
    SkeletonYAMLEncoder,
)
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from sleap_io.model.labels_set import LabelsSet


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


def load_dlc(
    filename: str, video_search_paths: Optional[List[Union[str, Path]]] = None, **kwargs
) -> Labels:
    """Read DeepLabCut annotations from a CSV file and return a `Labels` object.

    Args:
        filename: Path to DLC CSV file with annotations.
        video_search_paths: Optional list of paths to search for video files.
        **kwargs: Additional arguments passed to DLC loader.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    return dlc.load_dlc(filename, video_search_paths=video_search_paths, **kwargs)


def load_ultralytics(
    dataset_path: str,
    split: str = "train",
    skeleton: Optional[Skeleton] = None,
    **kwargs,
) -> Labels:
    """Load an Ultralytics YOLO pose dataset as a SLEAP `Labels` object.

    Args:
        dataset_path: Path to the Ultralytics dataset root directory containing
            data.yaml.
        split: Dataset split to read ('train', 'val', or 'test'). Defaults to 'train'.
        skeleton: Optional skeleton to use. If not provided, will be inferred from
            data.yaml.
        **kwargs: Additional arguments passed to `ultralytics.read_labels`.
            Currently supports:
            - image_size: Tuple of (height, width) for coordinate denormalization.
              Defaults to
              (480, 640). Will attempt to infer from actual images if available.

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
        **kwargs: Additional arguments passed to `ultralytics.write_labels`.
            Currently supports:
            - class_id: Class ID to use for all instances (default: 0).
            - image_format: Image format to use for saving frames. Either "png"
              (default, lossless) or "jpg".
            - image_quality: Image quality for JPEG format (1-100). For PNG, this is
              the compression
              level (0-9). If None, uses default quality settings.
            - verbose: If True (default), show progress bars during export.
            - use_multiprocessing: If True, use multiprocessing for parallel image
              saving. Default is False.
            - n_workers: Number of worker processes. If None, uses CPU count - 1.
              Only used if
              use_multiprocessing=True.
    """
    ultralytics.write_labels(labels, dataset_path, split_ratios=split_ratios, **kwargs)


def _detect_coco_format(json_path: str) -> bool:
    """Detect if a JSON file is in COCO format vs Label Studio format.

    Args:
        json_path: Path to JSON file to check.

    Returns:
        True if the file appears to be COCO format, False otherwise.
    """
    try:
        import json

        with open(json_path, "r") as f:
            data = json.load(f)

        # COCO format has specific top-level fields
        coco_fields = {"images", "annotations", "categories"}
        has_coco_fields = all(field in data for field in coco_fields)

        # Check if categories have keypoints (pose data)
        has_keypoints = False
        if "categories" in data:
            has_keypoints = any("keypoints" in cat for cat in data["categories"])

        return has_coco_fields and has_keypoints
    except Exception:
        return False


def load_coco(
    json_path: str,
    dataset_root: Optional[str] = None,
    grayscale: bool = False,
    **kwargs,
) -> Labels:
    """Load a COCO-style pose dataset and return a Labels object.

    Args:
        json_path: Path to the COCO annotation JSON file.
        dataset_root: Root directory of the dataset. If None, uses parent directory
                     of json_path.
        grayscale: If True, load images as grayscale (1 channel). If False, load as
                   RGB (3 channels). Default is False.
        **kwargs: Additional arguments (currently unused).

    Returns:
        The dataset as a `Labels` object.
    """
    return coco.read_labels(json_path, dataset_root=dataset_root, grayscale=grayscale)


def load_video(filename: str, **kwargs) -> Video:
    """Load a video file.

    Args:
        filename: The filename(s) of the video. Supported extensions: "mp4", "avi",
            "mov", "mj2", "mkv", "h5", "hdf5", "slp", "png", "jpg", "jpeg", "tif",
            "tiff", "bmp". If the filename is a list, a list of image filenames are
            expected. If filename is a folder, it will be searched for images.
        **kwargs: Additional arguments passed to `Video.from_filename`.
            Currently supports:
            - dataset: Name of dataset in HDF5 file.
            - grayscale: Whether to force grayscale. If None, autodetect on first
              frame load.
            - keep_open: Whether to keep the video reader open between calls to read
              frames.
              If False, will close the reader after each call. If True (the
              default), it will
              keep the reader open and cache it for subsequent calls which may
              enhance the
              performance of reading multiple frames.
            - source_video: Source video object if this is a proxy video. This is
              metadata
              and does not affect reading.
            - backend_metadata: Metadata to store on the video backend. This is
              useful for
              storing metadata that requires an open backend (e.g., shape
              information) without
              having to open the backend.
            - plugin: Video plugin to use for MediaVideo backend. One of "opencv",
              "FFMPEG",
              or "pyav". Also accepts aliases (case-insensitive):
              * opencv: "opencv", "cv", "cv2", "ocv"
              * FFMPEG: "FFMPEG", "ffmpeg", "imageio-ffmpeg", "imageio_ffmpeg"
              * pyav: "pyav", "av"

              If not specified, uses the following priority:
              1. Global default set via `sio.set_default_video_plugin()`
              2. Auto-detection based on available packages

              To set a global default:
              >>> import sleap_io as sio
              >>> sio.set_default_video_plugin("opencv")
              >>> video = sio.load_video("video.mp4")  # Uses opencv
            - input_format: Format of the data in HDF5 datasets. One of
              "channels_last" (the
              default) in (frames, height, width, channels) order or "channels_first" in
              (frames, channels, width, height) order.
            - frame_map: Mapping from frame indices to indices in the HDF5 dataset.
              This is
              used to translate between frame indices of images within their source
              video
              and indices of images in the dataset.
            - source_filename: Path to the source video file for HDF5 embedded videos.
            - source_inds: Indices of frames in the source video file for HDF5
              embedded videos.
            - image_format: Format of images in HDF5 embedded dataset.

    Returns:
        A `Video` object.

    See Also:
        set_default_video_plugin: Set the default video plugin globally.
        get_default_video_plugin: Get the current default video plugin.
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
            file extension. Available formats are: "slp", "nwb", "labelstudio", "coco",
            "jabs", "dlc", "ultralytics", and "video".
        **kwargs: Additional arguments passed to the format-specific loading function:
            - For "slp" format: No additional arguments.
            - For "nwb" format: No additional arguments.
            - For "labelstudio" format: skeleton (Optional[Skeleton]): Skeleton to
              use for
              the labels.
            - For "coco" format: dataset_root (Optional[str]): Root directory of the
              dataset. grayscale (bool): If True, load images as grayscale (1 channel).
              If False, load as RGB (3 channels). Default is False.
            - For "jabs" format: skeleton (Optional[Skeleton]): Skeleton to use for
              the labels.
            - For "dlc" format: video_search_paths (Optional[List[str]]): Paths to
              search for video files.
            - For "ultralytics" format: See `load_ultralytics` for supported arguments.
            - For "video" format: See `load_video` for supported arguments.

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
            # Detect if this is COCO format or Label Studio format
            if _detect_coco_format(filename):
                format = "coco"
            else:
                format = "json"
        elif filename.endswith(".h5"):
            format = "jabs"
        elif filename.endswith("data.yaml") or (
            Path(filename).is_dir() and (Path(filename) / "data.yaml").exists()
        ):
            format = "ultralytics"
        elif filename.endswith(".csv") and dlc.is_dlc_file(filename):
            format = "dlc"
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
        if format == "coco":
            return load_coco(filename, **kwargs)
        else:
            return load_labelstudio(filename, **kwargs)
    elif filename.endswith(".h5"):
        return load_jabs(filename, **kwargs)
    elif format == "dlc":
        return load_dlc(filename, **kwargs)
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
        **kwargs: Additional arguments passed to the format-specific saving function:
            - For "slp" format: embed (bool | str | list[tuple[Video, int]] |
              None): Frames
              to embed in the saved labels file. One of None, True, "all", "user",
              "suggestions", "user+suggestions", "source" or list of tuples of
              (video, frame_idx). If False (the default), no frames are embedded.
            - For "nwb" format: pose_estimation_metadata (dict): Metadata to store
              in the
              NWB file. append (bool): If True, append to existing NWB file.
            - For "labelstudio" format: No additional arguments.
            - For "jabs" format: pose_version (int): JABS pose format version (1-6).
              root_folder (Optional[str]): Root folder for JABS project structure.
            - For "ultralytics" format: See `save_ultralytics` for supported arguments.
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


def load_labels_set(
    path: Union[str, Path, list[Union[str, Path]], dict[str, Union[str, Path]]],
    format: Optional[str] = None,
    open_videos: bool = True,
    **kwargs,
) -> LabelsSet:
    """Load a LabelsSet from multiple files.

    Args:
        path: Can be one of:
            - A directory path containing label files
            - A list of file paths
            - A dictionary mapping names to file paths
        format: Optional format specification. If None, will try to infer from path.
            Supported formats: "slp", "ultralytics"
        open_videos: If `True` (the default), attempt to open video backends.
        **kwargs: Additional format-specific arguments.

    Returns:
        A LabelsSet containing the loaded Labels objects.

    Examples:
        Load from SLP directory:
        >>> labels_set = load_labels_set("path/to/splits/")

        Load from list of SLP files:
        >>> labels_set = load_labels_set(["train.slp", "val.slp"])

        Load from Ultralytics dataset:
        >>> labels_set = load_labels_set("path/to/yolo_dataset/", format="ultralytics")
    """
    # Try to infer format if not specified
    if format is None:
        if isinstance(path, (str, Path)):
            path_obj = Path(path)
            if path_obj.is_dir():
                # Check for ultralytics structure
                if (path_obj / "data.yaml").exists() or any(
                    (path_obj / split).exists() for split in ["train", "val", "test"]
                ):
                    format = "ultralytics"
                else:
                    # Default to SLP for directories
                    format = "slp"
            else:
                # Single file path - check extension
                if path_obj.suffix == ".slp":
                    format = "slp"
        elif isinstance(path, list) and len(path) > 0:
            # Check first file in list
            first_path = Path(path[0])
            if first_path.suffix == ".slp":
                format = "slp"
        elif isinstance(path, dict):
            # Dictionary input defaults to SLP
            format = "slp"

    if format == "slp":
        return slp.read_labels_set(path, open_videos=open_videos)
    elif format == "ultralytics":
        # Extract ultralytics-specific kwargs
        splits = kwargs.pop("splits", None)
        skeleton = kwargs.pop("skeleton", None)
        image_size = kwargs.pop("image_size", (480, 640))
        # Remove verbose from kwargs if present (for backward compatibility)
        kwargs.pop("verbose", None)

        if not isinstance(path, (str, Path)):
            raise ValueError(
                "Ultralytics format requires a directory path, "
                f"got {type(path).__name__}"
            )

        return ultralytics.read_labels_set(
            str(path),
            splits=splits,
            skeleton=skeleton,
            image_size=image_size,
        )
    else:
        raise ValueError(
            f"Unknown format: {format}. Supported formats: 'slp', 'ultralytics'"
        )


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
