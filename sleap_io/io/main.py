"""This module contains high-level wrappers for utilizing different I/O backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

# Format modules are now imported inside functions for lazy loading
from sleap_io.io.skeleton import (
    decode_yaml_skeleton,
    encode_skeleton,
    encode_yaml_skeleton,
    load_skeleton_from_json,
)
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from sleap_io.model.label_image import LabelImage
    from sleap_io.model.labels_set import LabelsSet


def load_slp(
    filename: str | os.PathLike,
    open_videos: bool = True,
    lazy: bool = False,
    *,
    headers: dict[str, str] | None = None,
    stream_mode: str = "auto",
    cache_storage: str | os.PathLike | None = None,
    cache_expiry: float | None = None,
    block_size: int = 1 << 20,
    max_blocks: int = 32,
    retries: int = 3,
) -> Labels:
    """Load a SLEAP dataset from a local path or HTTP/cloud URL.

    For local paths, all URL-specific keyword arguments are ignored.

    Args:
        filename: Path to a SLEAP labels file (`.slp`), or a URL. Supported URL
            schemes: `http`, `https`, `s3`, `gs`, `gcs`, `az`, `abfs`. Cloud
            schemes require `pip install 'sleap-io[cloud]'`. Google Drive share
            links (`https://drive.google.com/file/d/<ID>/view`) are also
            supported and resolved to a direct download automatically (the file
            is fully downloaded into memory; folder links are not supported).
        open_videos: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).
        lazy: If `True`, defer instance materialization for faster loading.
            Lazy-loaded Labels support read operations and fast numpy/save.
            To modify, call `labels.materialize()` first. Default is `False`.
        headers: HTTP headers (e.g. `{"Authorization": "Bearer ..."}`) forwarded
            to fsspec for URL loads. Stripped on cross-origin redirect. Ignored
            for local paths.
        stream_mode: Remote streaming strategy (ignored for local paths). One of:
            `"auto"` (default; uses fsspec `blockcache` for lazy range reads),
            `"blockcache"`, `"cache"` (full download via `simplecache`),
            `"filecache"` (download with ETag revalidation), or `"download"`
            (ephemeral full download into memory).
        cache_storage: Override fsspec's cache directory for `cache`/`filecache`
            modes. Ignored for local paths.
        cache_expiry: TTL (seconds) for `filecache` revalidation. Defaults to
            3600 (1h) when not given. Ignored for other modes and local paths.
        block_size: Range block size in bytes for `blockcache` mode. Default:
            1 MiB. Ignored for local paths.
        max_blocks: Max blocks kept in the in-memory LRU per open file. Default:
            32 (32 MiB cap per open file). Ignored for local paths.
        retries: Retry count for transient HTTP errors. Default: 3. Ignored for
            local paths.

    Returns:
        The dataset as a `Labels` object.

    Raises:
        RemoteIOError: For HTTP errors against URLs (404, 416, 5xx after
            retries, connection failures).
        ImportError: For cloud schemes when the corresponding extra is not
            installed.
        ValueError: For an unrecognized `stream_mode`.

    See Also:
        Labels.is_lazy: Check if Labels is lazy-loaded.
        Labels.materialize: Convert lazy Labels to eager.
    """
    import h5py

    from sleap_io.io import _remote, slp

    if _remote._is_url(filename):
        url = os.fspath(filename) if isinstance(filename, os.PathLike) else filename
        file_like = _remote.open_url(
            url,
            headers=headers,
            stream_mode=stream_mode,
            cache_storage=cache_storage,
            cache_expiry=cache_expiry,
            block_size=block_size,
            max_blocks=max_blocks,
            retries=retries,
        )
        resolved_mode = "blockcache" if stream_mode == "auto" else stream_mode
        try:
            with h5py.File(file_like, "r") as f:
                if lazy:
                    labels = slp._read_labels_lazy_from_open_file(
                        url,
                        f,
                        open_videos=open_videos,
                        _url_headers=headers,
                        _url_stream_mode=resolved_mode,
                    )
                else:
                    labels = slp._read_labels_from_open_file(
                        url,
                        f,
                        open_videos=open_videos,
                        _url_headers=headers,
                        _url_stream_mode=resolved_mode,
                    )
        finally:
            file_like.close()

        # Propagate URL config to embedded HDF5Video backends so they can reopen
        # the remote file lazily on frame reads.
        from sleap_io.io.video_reading import HDF5Video

        for video in labels.videos:
            if isinstance(video.backend, HDF5Video):
                object.__setattr__(video.backend, "_url_headers", headers)
                object.__setattr__(video.backend, "_url_stream_mode", resolved_mode)
        return labels

    # Local path - UNCHANGED behaviour; URL-specific kwargs are no-ops.
    if lazy:
        return slp._read_labels_lazy(filename, open_videos=open_videos)
    return slp.read_labels(filename, open_videos=open_videos)


def save_slp(
    labels: Labels,
    filename: str,
    embed: bool | str | list[tuple[Video, int]] | None = False,
    restore_original_videos: bool = True,
    embed_inplace: bool = False,
    verbose: bool = True,
    plugin: str | None = None,
    progress_callback: Callable[[int, int], bool] | None = None,
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
        embed_inplace: If `False` (default), a copy of the labels is made before
            embedding to avoid modifying the in-memory labels. If `True`, the
            labels will be modified in-place to point to the embedded videos,
            which is faster but mutates the input. Only applies when embedding.
        verbose: If `True` (the default), display a progress bar when embedding frames.
        plugin: Image plugin to use for encoding embedded frames. One of "opencv"
            or "imageio". If None, uses the global default from
            `get_default_image_plugin()`. If no global default is set, auto-detects
            based on available packages (opencv preferred, then imageio).
        progress_callback: Optional callback function called during frame embedding
            with `(current, total)` arguments. If it returns `False`, the operation
            is cancelled and `ExportCancelled` is raised. When provided, tqdm
            progress bar is disabled in favor of the callback.
    """
    from sleap_io.io import slp

    return slp.write_labels(
        filename,
        labels,
        embed=embed,
        restore_original_videos=restore_original_videos,
        embed_inplace=embed_inplace,
        verbose=verbose,
        plugin=plugin,
        progress_callback=progress_callback,
    )


def load_nwb(filename: str) -> Labels:
    """Load an NWB dataset as a SLEAP `Labels` object.

    Args:
        filename: Path to a NWB file (`.nwb`).

    Returns:
        The dataset as a `Labels` object.
    """
    from sleap_io.io import nwb

    return nwb.load_nwb(filename)


def save_nwb(
    labels: Labels,
    filename: str | Path,
    nwb_format: str = "auto",
    append: bool = False,
) -> None:
    """Save a SLEAP dataset to NWB format.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to NWB file to save to. Must end in `.nwb`.
        nwb_format: Format to use for saving. Options are:
            - "auto" (default): Automatically detect based on data
            - "annotations": Save training annotations (PoseTraining)
            - "annotations_export": Export annotations with video frames
            - "predictions": Save predictions (PoseEstimation)
        append: If True, append to existing NWB file. Only supported for
            predictions format. Defaults to False.

    Raises:
        ValueError: If an invalid format is specified.
    """
    from sleap_io.io import nwb
    from sleap_io.io.nwb import NwbFormat

    # Convert string to NwbFormat if needed
    if isinstance(nwb_format, str):
        nwb_format = NwbFormat(nwb_format)

    nwb.save_nwb(labels, filename, nwb_format, append=append)


def load_labelstudio(
    filename: str, skeleton: Skeleton | list[str] | None = None
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
    from sleap_io.io import labelstudio

    return labelstudio.read_labels(filename, skeleton=skeleton)


def save_labelstudio(labels: Labels, filename: str):
    """Save a SLEAP dataset to Label Studio format.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to ending with `.json`.
    """
    from sleap_io.io import labelstudio

    labelstudio.write_labels(labels, filename)


def load_alphatracker(filename: str) -> Labels:
    """Read AlphaTracker annotations from a file and return a `Labels` object.

    Args:
        filename: Path to the AlphaTracker annotation file in JSON format.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    from sleap_io.io import alphatracker

    return alphatracker.read_labels(filename)


def load_jabs(filename: str, skeleton: Skeleton | None = None) -> Labels:
    """Read JABS-style predictions from a file and return a `Labels` object.

    Args:
        filename: Path to the jabs h5 pose file.
        skeleton: An optional `Skeleton` object.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    from sleap_io.io import jabs

    return jabs.read_labels(filename, skeleton=skeleton)


def load_analysis_h5(
    filename: str,
    video: "Video | str | None" = None,
) -> Labels:
    """Load SLEAP Analysis HDF5 file.

    Args:
        filename: Path to Analysis HDF5 file.
        video: Video to associate with data. If None, uses video_path stored
            in the file. Can be a Video object or path string.

    Returns:
        Labels object with loaded pose data.

    Notes:
        If the file contains extended metadata (skeleton symmetries, video
        backend metadata, etc.), it will be used to reconstruct the full
        Labels context.

    See Also:
        save_analysis_h5: Save Labels to Analysis HDF5 file.
    """
    from sleap_io.io import analysis_h5

    return analysis_h5.read_labels(filename, video=video)


def save_analysis_h5(
    labels: Labels,
    filename: str,
    *,
    video: "Video | int | None" = None,
    labels_path: str | None = None,
    all_frames: bool = True,
    min_occupancy: float = 0.0,
    preset: str | None = None,
    frame_dim: int | None = None,
    track_dim: int | None = None,
    node_dim: int | None = None,
    xy_dim: int | None = None,
    save_metadata: bool = True,
) -> None:
    """Save Labels to SLEAP Analysis HDF5 file.

    Args:
        labels: Labels to export.
        filename: Output file path.
        video: Video to export. If None, uses first video. Can be a Video
            object or an integer index.
        labels_path: Source labels path (stored as metadata).
        all_frames: Include all frames from 0 to last labeled frame.
            Default True.
        min_occupancy: Minimum track occupancy ratio (0-1) to keep.
            0 = keep all non-empty tracks (SLEAP default).
            0.5 = keep tracks with >50% occupancy.
        preset: Axis ordering preset. Options:
            - "matlab" (default): SLEAP-compatible ordering for MATLAB.
              tracks shape: (n_tracks, 2, n_nodes, n_frames)
            - "standard": Intuitive Python ordering.
              tracks shape: (n_frames, n_tracks, n_nodes, 2)
            Mutually exclusive with explicit dimension parameters.
        frame_dim: Position of the frame dimension (0-3).
        track_dim: Position of the track dimension (0-3).
        node_dim: Position of the node dimension (0-3).
        xy_dim: Position of the xy dimension (0-3).
        save_metadata: Store extended metadata for full round-trip.
            Default True.

    See Also:
        load_analysis_h5: Load Labels from Analysis HDF5 file.
    """
    from sleap_io.io import analysis_h5

    analysis_h5.write_labels(
        labels,
        filename,
        video=video,
        labels_path=labels_path,
        all_frames=all_frames,
        min_occupancy=min_occupancy,
        preset=preset,
        frame_dim=frame_dim,
        track_dim=track_dim,
        node_dim=node_dim,
        xy_dim=xy_dim,
        save_metadata=save_metadata,
    )


def save_jabs(labels: Labels, pose_version: int, root_folder: str | None = None):
    """Save a SLEAP dataset to JABS pose file format.

    Args:
        labels: SLEAP `Labels` object.
        pose_version: The JABS pose version to write data out.
        root_folder: Optional root folder where the files should be saved.

    Note:
        Filenames for JABS poses are based on video filenames.
    """
    from sleap_io.io import jabs

    jabs.write_labels(labels, pose_version, root_folder)


def load_dlc(
    filename: str, video_search_paths: list[str | Path] | None = None, **kwargs
) -> Labels:
    """Read DeepLabCut annotations from a CSV file and return a `Labels` object.

    Args:
        filename: Path to DLC CSV file with annotations.
        video_search_paths: Optional list of paths to search for video files.
        **kwargs: Additional arguments passed to DLC loader.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    from sleap_io.io import dlc

    return dlc.load_dlc(filename, video_search_paths=video_search_paths, **kwargs)


def load_trackmate(
    filename: str,
    video: "Video | str | None" = None,
    **kwargs,
) -> Labels:
    """Read TrackMate CSV exports and return a ``Labels`` object.

    Loads a TrackMate ``*_spots.csv`` file and optionally the corresponding
    ``*_edges.csv`` (auto-detected if present). Spot detections are imported
    as ``PredictedCentroid`` objects.

    Args:
        filename: Path to the TrackMate spots CSV file.
        video: Video to associate with centroids. Can be a ``Video`` object,
            a string path to a video file, or ``None`` (auto-detects a
            sibling ``.tif`` file).
        **kwargs: Additional arguments passed to ``read_trackmate_csv``.

    Returns:
        Parsed labels as a ``Labels`` instance with centroids.
    """
    from sleap_io.io import trackmate

    return trackmate.read_trackmate_csv(filename, video=video, **kwargs)


def load_ultralytics(
    dataset_path: str,
    split: str = "train",
    skeleton: Skeleton | None = None,
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
    from sleap_io.io import ultralytics

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
    from sleap_io.io import ultralytics

    ultralytics.write_labels(labels, dataset_path, split_ratios=split_ratios, **kwargs)


def load_geojson(filename: str) -> list:
    """Load ROIs from a GeoJSON file.

    Args:
        filename: Path to a ``.geojson`` file containing ROI features.

    Returns:
        A list of `ROI` objects.

    See Also:
        `ROI`: Region of interest data structure.
        `save_geojson`: Write ROIs to GeoJSON.
    """
    from sleap_io.io import geojson

    return geojson.read_rois(filename)


def save_geojson(rois: list, filename: str) -> None:
    """Save ROIs to a GeoJSON file.

    Args:
        rois: A list of `ROI` objects to save.
        filename: Path to the output ``.geojson`` file.

    See Also:
        `ROI`: Region of interest data structure.
        `load_geojson`: Read ROIs from GeoJSON.
    """
    from sleap_io.io import geojson

    geojson.write_rois(rois, filename)


def _is_alphatracker_data(data: object) -> bool:
    """Classify already-parsed JSON data as AlphaTracker format.

    Args:
        data: The parsed JSON object.

    Returns:
        True if the data matches the AlphaTracker schema, False otherwise.
    """
    # AlphaTracker format is a list of frame objects
    if not isinstance(data, list):
        return False

    if len(data) == 0:
        return False

    # Check first frame structure
    first_frame = data[0]
    if not isinstance(first_frame, dict):
        return False

    # AlphaTracker frames have "filename", "class": "image", and "annotations"
    has_required = (
        "filename" in first_frame
        and first_frame.get("class") == "image"
        and "annotations" in first_frame
    )

    if not has_required:
        return False

    # Check annotations structure
    annotations = first_frame.get("annotations", [])
    if not isinstance(annotations, list):
        return False

    # Check for Face and point classes
    has_face = any(a.get("class") == "Face" for a in annotations)
    has_point = any(a.get("class") == "point" for a in annotations)

    return has_face and has_point


def _is_coco_data(data: object) -> bool:
    """Classify already-parsed JSON data as COCO (pose) format.

    Args:
        data: The parsed JSON object.

    Returns:
        True if the data matches the COCO pose schema, False otherwise.
    """
    if not isinstance(data, dict):
        return False

    # COCO format has specific top-level fields
    coco_fields = {"images", "annotations", "categories"}
    has_coco_fields = all(field in data for field in coco_fields)

    # Check if categories have keypoints (pose data)
    has_keypoints = False
    if "categories" in data:
        has_keypoints = any("keypoints" in cat for cat in data["categories"])

    return has_coco_fields and has_keypoints


def _detect_alphatracker_format(json_path: str) -> bool:
    """Detect if a JSON file is in AlphaTracker format.

    Args:
        json_path: Path to JSON file to check.

    Returns:
        True if the file appears to be AlphaTracker format, False otherwise.
    """
    try:
        import json

        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return False
    return _is_alphatracker_data(data)


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
    except Exception:
        return False
    return _is_coco_data(data)


def load_leap(
    filename: str,
    skeleton: Skeleton | None = None,
    **kwargs,
) -> Labels:
    """Load a LEAP dataset from a .mat file.

    Args:
        filename: Path to a LEAP .mat file.
        skeleton: An optional `Skeleton` object. If not provided, will be constructed
            from the data in the file.
        **kwargs: Additional arguments (currently unused).

    Returns:
        The dataset as a `Labels` object.
    """
    from sleap_io.io import leap

    return leap.read_labels(filename, skeleton=skeleton)


def load_csv(
    filename: str,
    format: str = "auto",
    video: "Video | str | None" = None,
    skeleton: "Skeleton | None" = None,
) -> "Labels":
    """Load pose data from a CSV file.

    Args:
        filename: Path to CSV file.
        format: CSV format. One of "auto", "sleap", "dlc", "points", "instances",
            "frames". Default "auto" detects format from file content.
        video: Video to associate with data. Can be Video object or path string.
        skeleton: Skeleton to use. If None, inferred from columns or metadata.

    Returns:
        Labels object.

    Notes:
        If a metadata JSON file exists alongside the CSV (same base name with
        .json extension), it will be automatically loaded to restore full
        Labels context including skeleton edges, symmetries, and provenance.

    See Also:
        save_csv: Save Labels to CSV file.
    """
    from sleap_io.io import csv

    return csv.read_labels(filename, format=format, video=video, skeleton=skeleton)


def save_csv(
    labels: "Labels",
    filename: str,
    format: str = "sleap",
    video: "Video | int | None" = None,
    include_score: bool = True,
    include_empty: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
    scorer: str = "sleap-io",
    save_metadata: bool = False,
    chunk_size: int | None = None,
    video_id: str = "path",
) -> None:
    """Save pose data to a CSV file.

    Args:
        labels: Labels to save.
        filename: Output path.
        format: CSV format. One of "sleap" (default), "dlc", "points",
            "instances", "frames".
        video: Video to filter to. Can be Video object or integer index.
            If None, includes all videos.
        include_score: Include confidence scores in output. Default True.
        include_empty: Include frames with no instances (filled with NaN values).
            Default False. Only applies to "frames" and "instances" formats.
        start_frame: Start frame index (inclusive) for output. If None, starts
            from 0 when include_empty=True, or from first labeled frame otherwise.
        end_frame: End frame index (exclusive) for output. If None, ends at
            last labeled frame + 1.
        scorer: Scorer name for DLC format. Default "sleap-io".
        save_metadata: Save JSON metadata file alongside CSV that enables
            full round-trip reconstruction. Default False.
        chunk_size: Number of rows per chunk for memory-efficient writing. If None
            (default), writes entire DataFrame at once. Useful for large datasets.
            Not supported for DLC format.
        video_id: How to represent videos in the CSV. Options: "path" (default),
            "index", or "name".

    See Also:
        load_csv: Load Labels from CSV file.
    """
    from sleap_io.io import csv

    csv.write_labels(
        labels,
        filename,
        format=format,
        video=video,
        include_score=include_score,
        include_empty=include_empty,
        start_frame=start_frame,
        end_frame=end_frame,
        scorer=scorer,
        save_metadata=save_metadata,
        chunk_size=chunk_size,
        video_id=video_id,
    )


def load_coco(
    json_path: str,
    dataset_root: str | None = None,
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
    from sleap_io.io import coco

    return coco.read_labels(json_path, dataset_root=dataset_root, grayscale=grayscale)


def save_coco(
    labels: Labels,
    json_path: str,
    image_filenames: str | list[str] | None = None,
    visibility_encoding: str = "ternary",
):
    """Save a SLEAP dataset to COCO-style JSON annotation format.

    Args:
        labels: A SLEAP `Labels` object.
        json_path: Path to save the COCO annotation JSON file.
        image_filenames: Optional image filenames to use in the COCO JSON. If
                        provided, must be a single string (for single-frame videos) or
                        a list of strings matching the number of labeled frames. If
                        None, generates filenames from video filenames and frame
                        indices.
        visibility_encoding: Visibility encoding to use. Either "binary" (0/1) or
                           "ternary" (0/1/2). Default is "ternary".

    Notes:
        - This function only writes the JSON annotation file. It does not save images.
        - The generated JSON can be used with mmpose and other COCO-compatible tools.
        - For saving images along with annotations, you would need to extract and save
          frames separately.
    """
    from sleap_io.io import coco

    coco.write_labels(labels, json_path, image_filenames, visibility_encoding)


def load_video(filename: str, **kwargs) -> Video:
    """Load a video file.

    Remote media videos can be loaded from ``http``/``https`` URLs (see the
    ``filename`` argument). Only ``http``/``https`` URLs are supported for video
    (cloud schemes are not), and the ``av`` package is required (install with
    ``pip install 'sleap-io[pyav]'``).

    Warning:
        Decoding a remote video streams bytes from the URL into FFmpeg (via
        pyav), whose demuxers/decoders are a large, historically
        vulnerability-prone attack surface. Load remote video only from trusted
        sources, and sandbox untrusted inputs (e.g. decode in an isolated
        container/VM with no credentials and a restricted network). sleap-io
        only passes ``http``/``https`` URLs through to the decoder.

    Args:
        filename: The filename(s) of the video. Supported extensions: "mp4", "avi",
            "mov", "mj2", "mkv", "h5", "hdf5", "slp", "png", "jpg", "jpeg", "tif",
            "tiff", "bmp", "seq". If the filename is a list, a list of image filenames
            are expected. If filename is a folder, it will be searched for images.
            May also be an ``http(s)://`` URL pointing to a remote media video
            (one of "mp4", "avi", "mov", "mj2", "mkv"). Remote videos are read
            with the pyav plugin, which is selected automatically for URLs; it
            requires the ``av`` package (install with
            ``pip install 'sleap-io[pyav]'``). See the security warning above.
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
    from sleap_io.io import video_writing

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
    filename: str | Path,
    format: str | None = None,
    *,
    sniff: bool | None = None,
    **kwargs,
) -> Labels | Video:
    """Load a file and return the appropriate object.

    Args:
        filename: Path to a file, or a URL (`http`, `https`, `s3`, `gs`, `gcs`,
            `az`, `abfs`). Google Drive file share links are also supported; the
            file is downloaded and its format detected from the content (pass an
            explicit `format=` to skip the detection download).
        format: Optional format to load as. If not provided, will be inferred from the
            file extension. Available formats are: "slp", "nwb", "geojson",
            "alphatracker", "labelstudio", "coco", "jabs", "analysis_h5", "dlc",
            "trackmate", "ultralytics", "leap", and "video".
        sniff: Controls magic-byte sniffing for URLs with ambiguous extensions
            (`.h5`, `.json`, `.csv`). If `True`, fetch the first bytes via a
            Range request to disambiguate. If `None` (default), sniff only for
            URLs with ambiguous extensions (never for local paths, where opening
            the file is cheap). If `False`, never sniff; raise `ValueError` on an
            ambiguous URL extension when no explicit `format` is given.
        **kwargs: Additional arguments passed to the format-specific loading function:
            - For "slp" format: No additional arguments.
            - For "nwb" format: No additional arguments.
            - For "alphatracker" format: No additional arguments.
            - For "leap" format: skeleton (Optional[Skeleton]): Skeleton to use if not
              defined in the file.
            - For "labelstudio" format: skeleton (Optional[Skeleton]): Skeleton to
              use for
              the labels.
            - For "coco" format: dataset_root (Optional[str]): Root directory of the
              dataset. grayscale (bool): If True, load images as grayscale (1 channel).
              If False, load as RGB (3 channels). Default is False.
            - For "jabs" format: skeleton (Optional[Skeleton]): Skeleton to use for
              the labels.
            - For "analysis_h5" format: video (Optional[Video | str]): Video to
              associate with data. If None, uses video_path stored in the file.
            - For "dlc" format: video_search_paths (Optional[List[str]]): Paths to
              search for video files.
            - For "ultralytics" format: See `load_ultralytics` for supported arguments.
            - For "video" format: See `load_video` for supported arguments.

    Returns:
        A `Labels` or `Video` object.
    """
    if isinstance(filename, Path):
        filename = filename.as_posix()

    from sleap_io.io import _remote

    if _remote._is_url(filename):
        return _load_file_url(filename, format=format, sniff=sniff, **kwargs)

    if format is None:
        if filename.lower().endswith(".slp"):
            format = "slp"
        elif filename.lower().endswith(".nwb"):
            format = "nwb"
        elif filename.lower().endswith(".mat"):
            format = "leap"
        elif filename.lower().endswith(".json"):
            # Detect JSON format: AlphaTracker, COCO, or Label Studio
            if _detect_alphatracker_format(filename):
                format = "alphatracker"
            elif _detect_coco_format(filename):
                format = "coco"
            else:
                format = "json"
        elif filename.lower().endswith(".h5"):
            # Check if this is Analysis HDF5 or JABS
            from sleap_io.io import analysis_h5

            if analysis_h5.is_analysis_h5_file(filename):
                format = "analysis_h5"
            else:
                format = "jabs"
        elif filename.lower().endswith(".geojson"):
            format = "geojson"
        elif filename.endswith("data.yaml") or (
            Path(filename).is_dir() and (Path(filename) / "data.yaml").exists()
        ):
            format = "ultralytics"
        elif filename.lower().endswith(".csv"):
            from sleap_io.io import dlc, trackmate

            if trackmate.is_trackmate_file(filename):
                format = "trackmate"
            elif dlc.is_dlc_file(filename):
                format = "dlc"
            else:
                format = "csv"
        else:
            for vid_ext in Video.EXTS:
                if filename.lower().endswith(vid_ext.lower()):
                    format = "video"
                    break
        if format is None:
            raise ValueError(f"Could not infer format from filename: '{filename}'.")

    if filename.lower().endswith(".slp"):
        return load_slp(filename, **kwargs)
    elif filename.lower().endswith(".nwb"):
        return load_nwb(filename, **kwargs)
    elif filename.lower().endswith(".mat"):
        return load_leap(filename, **kwargs)
    elif filename.lower().endswith(".json"):
        if format == "alphatracker":
            return load_alphatracker(filename, **kwargs)
        elif format == "coco":
            return load_coco(filename, **kwargs)
        else:
            return load_labelstudio(filename, **kwargs)
    elif filename.lower().endswith(".h5"):
        if format == "analysis_h5":
            return load_analysis_h5(filename, **kwargs)
        else:
            return load_jabs(filename, **kwargs)
    elif format == "dlc":
        return load_dlc(filename, **kwargs)
    elif format == "csv":
        return load_csv(filename, **kwargs)
    elif format == "trackmate":
        return load_trackmate(filename, **kwargs)
    elif format == "ultralytics":
        return load_ultralytics(filename, **kwargs)
    elif format == "geojson":
        return Labels(rois=load_geojson(filename))
    elif format == "video":
        return load_video(filename, **kwargs)


#: Loaders for unambiguous URL extensions that need no magic-byte sniffing.
_URL_UNAMBIGUOUS_EXTS: dict[str, str] = {
    ".slp": "slp",
    ".nwb": "nwb",
    ".mat": "leap",
    ".geojson": "geojson",
}

#: URL extensions that map to multiple formats and require sniffing.
_URL_AMBIGUOUS_EXTS = frozenset({".h5", ".json", ".csv"})


#: URL-loadable formats. Only `slp` (`.slp`/`.pkg.slp`) and `video` (remote
#: media via pyav) loading are implemented for remote URLs; every other
#: format's loader opens the path locally and would fail with a cryptic
#: `OSError` over a URL, so those are gated behind a clean `NotImplementedError`.
_URL_IMPLEMENTED_FORMATS = frozenset({"slp", "video"})


def _dispatch_url_format(filename: str, format: str, **kwargs) -> Labels | Video:
    """Dispatch a URL to a concrete loader given a resolved format string.

    Only the `slp` (labels) and `video` (media) formats are URL-aware; every
    other format's loader opens the path with a local file `open()` and would
    fail with a cryptic `OSError` over a URL, so those are gated behind a clean
    `NotImplementedError`.

    Args:
        filename: The URL to load.
        format: The resolved format name (e.g. ``"slp"``, ``"coco"``).
        **kwargs: Forwarded to the underlying loader.

    Returns:
        The loaded `Labels` or `Video` object.

    Raises:
        ValueError: If `format` is not a recognized format.
        NotImplementedError: If `format` is recognized but remote loading for it
            is not yet implemented (every format other than `slp`/`video`).
    """
    from sleap_io.io import _remote

    known_formats = {
        "slp",
        "nwb",
        "leap",
        "alphatracker",
        "coco",
        "labelstudio",
        "analysis_h5",
        "jabs",
        "dlc",
        "csv",
        "trackmate",
        "ultralytics",
        "video",
        "geojson",
    }
    if format not in known_formats:
        raise ValueError(
            f"Unsupported format '{format}' for URL: '{_remote._redact_url(filename)}'."
        )

    if format not in _URL_IMPLEMENTED_FORMATS:
        raise NotImplementedError(
            f"Remote {format} loading is not yet implemented "
            f"(URL: {_remote._redact_url(filename)}); tracked as a follow-up. "
            "Download the file locally first."
        )

    if format == "video":
        return load_video(filename, **kwargs)
    return load_slp(filename, **kwargs)


def _gdrive_format_from_bytes(data: bytes) -> str:
    """Resolve a format name from the first bytes of a Drive download.

    Drive download URLs have no extension, so the format is inferred from the
    magic bytes (and, for HDF5, the top-level group structure).

    Args:
        data: The fully-downloaded file bytes.

    Returns:
        A format name suitable for `_dispatch_url_format` (e.g. ``"slp"``,
        ``"analysis_h5"``, ``"jabs"``, ``"coco"``, ``"labelstudio"``,
        ``"alphatracker"``, ``"csv"``).

    Raises:
        ValueError: If the content type cannot be recognized.
    """
    import io
    import json

    from sleap_io.io import _remote

    family = _remote._identify_magic(data[:16])
    if family == "hdf5":
        import h5py

        with h5py.File(io.BytesIO(data), "r") as f:
            if "track_occupancy" in f:
                return "analysis_h5"
            if "metadata" in f:
                return "slp"
            return "jabs"
    if family == "json":
        try:
            parsed = json.loads(data)
        except Exception:
            parsed = None
        if _is_alphatracker_data(parsed):
            return "alphatracker"
        if _is_coco_data(parsed):
            return "coco"
        return "labelstudio"
    if family == "csv":
        return "csv"
    raise ValueError(
        f"Could not determine the format of the Google Drive file (sniffed "
        f"'{family}'). Pass an explicit format= (e.g. 'slp')."
    )


def _dispatch_gdrive_url_format(filename: str, **kwargs) -> Labels | Video:
    """Resolve + sniff a Google Drive URL, then dispatch to the right loader.

    The Drive bytes are fetched once to sniff the format; the chosen loader
    then re-resolves the URL to read it. Pass an explicit ``format=`` to skip
    this detection fetch entirely.

    Args:
        filename: The Google Drive URL.
        **kwargs: Forwarded to the underlying loader (URL streaming kwargs such
            as ``headers`` flow through to `load_slp`).

    Returns:
        The loaded `Labels` or `Video` object.

    Raises:
        ValueError: If the content type cannot be recognized.
        RemoteIOError: For HTTP / resolution failures.
    """
    from sleap_io.io._gdrive import _open_gdrive

    headers = kwargs.get("headers")
    file_like = _open_gdrive(filename, headers=headers)
    try:
        data = file_like.read()
    finally:
        file_like.close()
    fmt = _gdrive_format_from_bytes(data)
    return _dispatch_url_format(filename, fmt, **kwargs)


def _resolve_hdf5_url_format(
    url: str,
    headers: dict[str, str] | None,
    *,
    stream_mode: str = "auto",
) -> str:
    """Disambiguate an HDF5 URL into ``slp``/``analysis_h5``/``jabs``.

    Opens the remote file via fsspec and inspects its top-level group
    structure: a `track_occupancy` dataset marks an Analysis HDF5 file, a
    `metadata` group marks a `.slp` file, and anything else falls back to JABS.

    Args:
        url: The HDF5 URL to inspect.
        headers: Optional HTTP headers for the probe.
        stream_mode: Streaming strategy for the probe (mirrors the value that
            will be used for the full load).

    Returns:
        One of ``"slp"``, ``"analysis_h5"``, or ``"jabs"``.
    """
    import h5py

    from sleap_io.io import _remote

    file_like = _remote.open_url(url, headers=headers, stream_mode=stream_mode)
    try:
        with h5py.File(file_like, "r") as f:
            if "track_occupancy" in f:
                return "analysis_h5"
            if "metadata" in f:
                return "slp"
            return "jabs"
    finally:
        file_like.close()


def _load_file_url(
    filename: str,
    *,
    format: str | None = None,
    sniff: bool | None = None,
    **kwargs,
) -> Labels | Video:
    """Load a labels/video file from a URL with extension + magic-byte routing.

    The local content-detectors used by `load_file` call `open()` on the path,
    which breaks for URLs, so URL routing relies on the extension plus (for
    ambiguous extensions) a magic-byte sniff via a ranged read.

    Args:
        filename: The URL to load.
        format: Explicit format. If given, dispatch directly.
        sniff: Sniff control (see `load_file`). For ambiguous extensions,
            `True`/`None` sniff via fsspec; `False` raises without a format.
        **kwargs: Forwarded to the underlying loader (URL streaming kwargs such
            as `headers`/`stream_mode` flow through to `load_slp`).

    Returns:
        The loaded `Labels` or `Video` object.

    Raises:
        ValueError: If the format cannot be inferred and sniffing is disabled,
            or the sniffed magic bytes are unsupported/unrecognized.
    """
    import urllib.parse

    from sleap_io.io import _remote

    # Explicit format always wins, bypassing detection entirely.
    if format is not None:
        return _dispatch_url_format(filename, format, **kwargs)

    # Google Drive share links carry no usable file extension, so resolve +
    # sniff the bytes to pick a format. `sniff=False` disables this since there
    # is no extension to fall back on.
    from sleap_io.io._gdrive import _is_gdrive_url

    if _is_gdrive_url(filename):
        if sniff is False:
            raise ValueError(
                "Cannot infer format for a Google Drive URL when sniff=False "
                f"(Drive links carry no file extension): '{filename}'. Pass an "
                "explicit format= (e.g. 'slp')."
            )
        return _dispatch_gdrive_url_format(filename, **kwargs)

    parsed = urllib.parse.urlparse(filename)
    ext = "".join(Path(parsed.path).suffixes[-1:]).lower()

    # Unambiguous labels extension: dispatch by extension exactly like local.
    if ext in _URL_UNAMBIGUOUS_EXTS:
        return _dispatch_url_format(filename, _URL_UNAMBIGUOUS_EXTS[ext], **kwargs)

    # Ambiguous extension (.h5/.json/.csv): sniff unless explicitly disabled.
    # Checked before the video-extension fallback because `Video.EXTS` also
    # lists `h5`/`hdf5`, which here mean a labels file, not a video.
    if ext in _URL_AMBIGUOUS_EXTS:
        if sniff is False:
            raise ValueError(
                f"Cannot infer format for URL with ambiguous extension "
                f"'{ext}' when sniff=False: '{_remote._redact_url(filename)}'. "
                "Pass an explicit format= (e.g. 'analysis_h5', 'jabs', 'coco', "
                "'labelstudio', 'alphatracker', 'dlc', 'trackmate', 'csv')."
            )
        headers = kwargs.get("headers")
        stream_mode = kwargs.get("stream_mode", "auto")
        family = _remote._sniff_format(filename, headers=headers)
        if family == "hdf5":
            resolved = _resolve_hdf5_url_format(
                filename, headers, stream_mode=stream_mode
            )
            return _dispatch_url_format(filename, resolved, **kwargs)
        if family == "json":
            if _detect_alphatracker_url(filename, headers):
                return _dispatch_url_format(filename, "alphatracker", **kwargs)
            if _detect_coco_url(filename, headers):
                return _dispatch_url_format(filename, "coco", **kwargs)
            return _dispatch_url_format(filename, "labelstudio", **kwargs)
        if family == "csv":
            # Without downloading the full CSV we cannot cheaply distinguish
            # DLC/TrackMate from a generic SLEAP CSV; default to the generic
            # reader. Pass an explicit format= to force dlc/trackmate.
            return _dispatch_url_format(filename, "csv", **kwargs)
        raise ValueError(
            f"Unsupported or unrecognized content (sniffed '{family}') for "
            f"URL: '{_remote._redact_url(filename)}'. Pass an explicit format=."
        )

    # Video extension fallback (the genuine video extensions). Match against the
    # already-parsed, path-stripped ``ext`` (computed above, e.g. ``".mp4"``)
    # rather than the raw URL, so query-stringed/fragment URLs (e.g. presigned
    # S3/GCS links with a ``?token=`` or ``#fragment``) still match. ``ext``
    # carries a leading dot while ``Video.EXTS`` entries do not, so compare with
    # ``endswith``. This mirrors the ``_extension_token`` path-stripping in
    # ``VideoBackend.from_filename`` so that ``load_file`` and ``load_video``
    # agree on the same URL.
    if ext.endswith(tuple(vid_ext.lower() for vid_ext in Video.EXTS)):
        return _dispatch_url_format(filename, "video", **kwargs)

    raise ValueError(
        f"Could not infer format from URL: '{_remote._redact_url(filename)}'."
    )


def _detect_alphatracker_url(url: str, headers: dict[str, str] | None) -> bool:
    """Return True if a JSON URL is in AlphaTracker format.

    Args:
        url: The JSON URL to inspect.
        headers: Optional HTTP headers.

    Returns:
        True if the content matches the AlphaTracker schema, else False.
    """
    import json

    from sleap_io.io import _remote

    file_like = _remote.open_url(url, headers=headers, stream_mode="download")
    try:
        data = json.loads(file_like.read())
    except Exception:
        return False
    finally:
        file_like.close()
    return _is_alphatracker_data(data)


def _detect_coco_url(url: str, headers: dict[str, str] | None) -> bool:
    """Return True if a JSON URL is in COCO format.

    Args:
        url: The JSON URL to inspect.
        headers: Optional HTTP headers.

    Returns:
        True if the content matches the COCO schema, else False.
    """
    import json

    from sleap_io.io import _remote

    file_like = _remote.open_url(url, headers=headers, stream_mode="download")
    try:
        data = json.loads(file_like.read())
    except Exception:
        return False
    finally:
        file_like.close()
    return _is_coco_data(data)


def save_file(
    labels: Labels,
    filename: str | Path,
    format: str | None = None,
    verbose: bool = True,
    progress_callback: Callable[[int, int], bool] | None = None,
    **kwargs,
):
    """Save a file based on the extension.

    Args:
        labels: A SLEAP `Labels` object (see `load_slp`).
        filename: Path to save labels to.
        format: Optional format to save as. If not provided, will be inferred from the
            file extension. Available formats are: "slp", "nwb", "labelstudio", "coco",
            "jabs", "analysis_h5", "ultralytics", and "geojson".
        verbose: If `True` (the default), display a progress bar when embedding frames
            (only applies to the SLP format).
        progress_callback: Optional callback function called during frame embedding
            (SLP format only) with `(current, total)` arguments. If it returns `False`,
            the operation is cancelled and `ExportCancelled` is raised.
        **kwargs: Additional arguments passed to the format-specific saving function:
            - For "slp" format: embed (bool | str | list[tuple[Video, int]] |
              None): Frames
              to embed in the saved labels file. One of None, True, "all", "user",
              "suggestions", "user+suggestions", "source" or list of tuples of
              (video, frame_idx). If False (the default), no frames are embedded.
              embed_inplace (bool): If False (default), copy labels before embedding
              to avoid mutating the input. If True, modify labels in-place.
            - For "nwb" format: pose_estimation_metadata (dict): Metadata to store
              in the
              NWB file. append (bool): If True, append to existing NWB file.
            - For "labelstudio" format: No additional arguments.
            - For "coco" format: image_filenames (Optional[Union[str, List[str]]]):
              Image filenames to use. visibility_encoding (str): Either "binary" or
              "ternary" (default).
            - For "jabs" format: pose_version (int): JABS pose format version (1-6).
              root_folder (Optional[str]): Root folder for JABS project structure.
            - For "analysis_h5" format: See `save_analysis_h5` for supported arguments.
            - For "ultralytics" format: See `save_ultralytics` for supported arguments.
    """
    if isinstance(filename, Path):
        filename = str(filename)

    if format is None:
        if filename.lower().endswith(".slp"):
            format = "slp"
        elif filename.lower().endswith(".nwb"):
            format = "nwb"
        elif filename.lower().endswith(".json"):
            # Check if this should be COCO format based on kwargs
            if "visibility_encoding" in kwargs or "image_filenames" in kwargs:
                format = "coco"
            else:
                format = "labelstudio"
        elif filename.lower().endswith(".h5") or filename.lower().endswith(
            ".analysis.h5"
        ):
            # Analysis HDF5 can be detected by extension pattern or kwargs
            if "min_occupancy" in kwargs or filename.lower().endswith(".analysis.h5"):
                format = "analysis_h5"
            elif "pose_version" in kwargs:
                format = "jabs"
            else:
                # Default to analysis_h5 for .h5 extension without specific jabs kwargs
                format = "analysis_h5"
        elif filename.lower().endswith(".geojson"):
            format = "geojson"
        elif "pose_version" in kwargs:
            format = "jabs"
        elif "split_ratios" in kwargs or Path(filename).is_dir():
            format = "ultralytics"

    if format == "slp":
        save_slp(
            labels,
            filename,
            verbose=verbose,
            progress_callback=progress_callback,
            **kwargs,
        )
    elif format == "nwb":
        save_nwb(labels, filename, **kwargs)
    elif format == "labelstudio":
        save_labelstudio(labels, filename, **kwargs)
    elif format == "coco":
        save_coco(labels, filename, **kwargs)
    elif format == "jabs":
        pose_version = kwargs.pop("pose_version", 5)
        root_folder = kwargs.pop("root_folder", filename)
        save_jabs(labels, pose_version=pose_version, root_folder=root_folder)
    elif format == "analysis_h5":
        # Filter kwargs to those accepted by save_analysis_h5
        analysis_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "video",
                "labels_path",
                "all_frames",
                "min_occupancy",
                "preset",
                "frame_dim",
                "track_dim",
                "node_dim",
                "xy_dim",
                "save_metadata",
            )
        }
        save_analysis_h5(labels, filename, **analysis_kwargs)
    elif format == "ultralytics":
        save_ultralytics(labels, filename, **kwargs)
    elif format == "geojson":
        save_geojson(labels.rois, filename)
    elif format == "csv" or filename.lower().endswith(".csv"):
        csv_format = kwargs.pop("csv_format", "sleap")
        # Filter kwargs to only those accepted by save_csv
        csv_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("video", "include_score", "scorer", "save_metadata")
        }
        save_csv(labels, filename, format=csv_format, **csv_kwargs)
    else:
        raise ValueError(f"Unknown format '{format}' for filename: '{filename}'.")


def load_skeleton(filename: str | Path) -> Skeleton | list[Skeleton]:
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
        return decode_yaml_skeleton(yaml_data)
    else:
        # JSON format (default) - could be standalone or training config
        with open(filename, "r") as f:
            json_data = f.read()
        return load_skeleton_from_json(json_data)


def load_labels_set(
    path: str | Path | list[str | Path] | dict[str, str | Path],
    format: str | None = None,
    open_videos: bool = True,
    **kwargs,
) -> "LabelsSet":
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
        from sleap_io.io import slp

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

        from sleap_io.io import ultralytics

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


def load_label_images(
    path: str | Path,
    video: Video | None = None,
    tracks: dict | None = None,
    categories: list[str] | dict[int, str] | None = None,
    pages_as: str = "auto",
) -> list[LabelImage]:
    """Load label images from TIFF file(s) or directory.

    Args:
        path: Path to a TIFF file (single or multi-page stack) or a directory
            of per-frame TIFFs.
        video: Video to associate with all frames.
        tracks: Global ``{label_id: Track}`` mapping. If ``None``, auto-creates
            one Track per unique ID found across all frames. Ignored for
            class-stacked layouts.
        categories: Category strings.

            - ``dict[int, str]`` keyed by label ID (time mode).
            - ``list[str]`` positional, one per class (class mode).
            - ``None`` to read from sidecar if present.

        pages_as: How to interpret multi-page TIFFs.

            - ``"auto"`` (default): consult sidecar ``"axes"``, then TIFF
              metadata (OME-XML / ImageJ hyperstack). Falls back to
              ``"time"`` for plain multi-page files with a one-time warning.
            - ``"time"``: force each page to be one frame.
            - ``"classes"``: force pages to be per-class binary masks for a
              single frame (N pages -> 1 ``LabelImage`` with label IDs 1..N).

    Returns:
        List of ``LabelImage``, one per frame, sorted by frame index.
    """
    from sleap_io.io import tiff

    return tiff.read_label_images(
        path,
        video=video,
        tracks=tracks,
        categories=categories,
        pages_as=pages_as,
    )


def save_label_images(
    path: str | Path,
    label_images: list[LabelImage],
    stack: bool = True,
) -> None:
    """Save label images to TIFF.

    Args:
        path: Output path. If ``stack=True``, writes a single multi-page TIFF.
            If ``stack=False``, writes per-frame files to this directory.
        label_images: ``LabelImage`` objects to write.
        stack: Write as multi-page TIFF stack (``True``) or per-frame files in
            a directory (``False``).
    """
    from sleap_io.io import tiff

    tiff.write_label_images(path, label_images, stack=stack)


def merge_label_images(
    source_paths: list[str | Path],
    dest_path: str | Path,
    video: Video | None = None,
) -> Labels:
    """Merge label images from multiple SLP files into one.

    Copies compressed chunks directly (no decompression) via
    ``read_direct_chunk`` -> ``write_direct_chunk`` when possible, falling
    back to decompress + recompress for legacy blob-format sources.

    Args:
        source_paths: List of paths to source SLP files containing label
            images to merge.
        dest_path: Path to the destination SLP file to create.
        video: Optional ``Video`` to associate with all merged label images.
            If ``None``, videos are deduplicated by filename across sources.

    Returns:
        A ``Labels`` object pointing at the merged file.

    Raises:
        ValueError: If source files have label images with different
            ``(height, width)`` dimensions, or if no source files are
            provided, or if a source contains no label images.

    See also: :func:`sleap_io.io.slp.merge_label_images`
    """
    from sleap_io.io.slp import merge_label_images as _merge_label_images

    return _merge_label_images(source_paths, dest_path, video=video)


def save_skeleton(skeleton: Skeleton | list[Skeleton], filename: str | Path):
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
        yaml_data = encode_yaml_skeleton(skeleton)
        with open(filename, "w") as f:
            f.write(yaml_data)
    else:
        # JSON format (default)
        json_data = encode_skeleton(skeleton)
        with open(filename, "w") as f:
            f.write(json_data)
