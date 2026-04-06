"""This module handles direct I/O operations for working with .slp files.

Format version history:
    - 1.0: Initial format
    - 1.1: Changed coordinate system from top-left pixel at (0, 0) to center at (0, 0)
    - 1.2: Added tracking_score field to instances
    - 1.3: Added explicit handling for tracking_score
    - 1.4: Added channel_order attribute to embedded video datasets to track RGB vs BGR
    - 1.5: Added ROI and segmentation mask datasets (/rois, /roi_wkb, /masks, /mask_rle)
    - 1.6: Added instance_idx to ROI dtype for instance-level associations
    - 1.7: Added bounding box dataset (/bboxes)
    - 1.8: Added label image datasets (/label_images, /label_image_data)
    - 1.9: Added identity dataset (/identities_json), Instance3D support,
            predicted variants (is_predicted, score) for masks/ROIs/label images;
            instance association for masks; string datasets for metadata
    - 2.0: Columnar bounding box storage (/bboxes group with x1/y1/x2/y2 datasets)
"""

from __future__ import annotations

import sys
import warnings
import zlib
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import imageio.v3 as iio
import numpy as np
import simplejson as json
from tqdm import tqdm

from sleap_io.io.skeleton import SkeletonSLPDecoder, SkeletonSLPEncoder
from sleap_io.io.utils import (
    is_file_accessible,
    read_hdf5_attrs,
    read_hdf5_dataset,
    sanitize_filename,
)
from sleap_io.io.video_reading import (
    HDF5Video,
    ImageVideo,
    MediaVideo,
    TiffVideo,
    VideoBackend,
)
from sleap_io.model.bbox import BoundingBox, PredictedBoundingBox, UserBoundingBox
from sleap_io.model.camera import (
    Camera,
    CameraGroup,
    FrameGroup,
    InstanceGroup,
    RecordingSession,
)
from sleap_io.model.identity import Identity
from sleap_io.model.instance import (
    Instance,
    Instance3D,
    PredictedInstance,
    PredictedInstance3D,
    Track,
)
from sleap_io.model.label_image import LabelImage, PredictedLabelImage, UserLabelImage
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.mask import (
    PredictedSegmentationMask,
    SegmentationMask,
    UserSegmentationMask,
)
from sleap_io.model.roi import ROI, PredictedROI, UserROI
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.suggestions import SuggestionFrame
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from sleap_io.model.instance import PointsArray, PredictedPointsArray
    from sleap_io.model.labels_set import LabelsSet

try:
    import cv2
except ImportError:
    pass


class VideoReferenceMode(Enum):
    """How to handle video references when saving."""

    EMBED = "embed"  # Embed frames in the file
    RESTORE_ORIGINAL = "restore_original"  # Use original video if available
    PRESERVE_SOURCE = "preserve_source"  # Keep reference to source file (.pkg.slp)


class InstanceType(IntEnum):
    """Enumeration of instance types to integers."""

    USER = 0
    PREDICTED = 1


class ExportCancelled(Exception):
    """Raised when an export operation is cancelled by the user."""

    pass


def _is_embedded_video_metadata(video: Video) -> bool:
    """Check if a video has embedded frames based on metadata.

    This function detects embedded videos even when the video backend is not open
    (i.e., when loaded with open_backend=False). It checks the backend_metadata
    for indicators that the video contains embedded frames.

    Args:
        video: Video object to check.

    Returns:
        True if the video appears to have embedded frames based on its metadata.
    """
    meta = video.backend_metadata
    if not meta:
        return False

    # Embedded videos have filename="." and a dataset like "video0/video"
    if meta.get("filename") == ".":
        return True

    # Also check if dataset name indicates embedded video
    dataset = meta.get("dataset", "")
    if dataset and "/" in dataset and dataset.endswith("/video"):
        return True

    return False


def make_video(
    labels_path: str,
    video_json: dict,
    open_backend: bool = True,
    _hdf5_file: h5py.File | None = None,
) -> Video:
    """Create a `Video` object from a JSON dictionary.

    Args:
        labels_path: A string path to the SLEAP labels file.
        video_json: A dictionary containing the video metadata.
        open_backend: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).
        _hdf5_file: Optional already-open HDF5 file handle. For internal use to avoid
            repeatedly opening the same file when loading many embedded videos.
    """
    backend_metadata = video_json["backend"]

    # Get video path from backend metadata (fall back to top-level filename if needed).
    if "filename" in backend_metadata:
        video_path = backend_metadata["filename"]
    elif "filename" in video_json:
        video_path = video_json["filename"]
    else:
        raise ValueError("Video JSON does not contain a filename.")

    # Marker for embedded videos.
    source_video = None
    is_embedded = False
    if video_path == ".":
        video_path = labels_path
        is_embedded = True

    # Basic path resolution.
    video_path = Path(sanitize_filename(video_path))

    if is_embedded:
        # Try to recover the source video from HDF5 attrs.
        # Use provided file handle if available to avoid repeated file opens.
        # Note: original_video is now a computed property derived from source_video,
        # so we only load source_video. Legacy files with original_video but no
        # source_video are handled by using original_video as source_video.
        def _read_embedded_video_metadata(f: h5py.File):
            nonlocal source_video
            dataset = backend_metadata["dataset"]
            if dataset.endswith("/video"):
                dataset = dataset[:-6]

            # Load source_video metadata
            if dataset in f and "source_video" in f[dataset]:
                source_video_json = json.loads(
                    f[f"{dataset}/source_video"].attrs["json"]
                )
                source_video = make_video(
                    labels_path,
                    source_video_json,
                    open_backend=open_backend,
                    _hdf5_file=f,
                )

            # Legacy compatibility: if original_video exists but source_video doesn't,
            # use original_video as source_video (they're equivalent for single-level)
            if source_video is None and f"{dataset}/original_video" in f:
                original_video_json = json.loads(
                    f[f"{dataset}/original_video"].attrs["json"]
                )
                source_video = make_video(
                    labels_path,
                    original_video_json,
                    open_backend=False,  # Original videos are often not available
                    _hdf5_file=f,
                )

        if _hdf5_file is not None:
            _read_embedded_video_metadata(_hdf5_file)
        else:
            with h5py.File(labels_path, "r") as f:
                _read_embedded_video_metadata(f)
    else:
        # For non-embedded videos, check if metadata is in videos_json
        if "source_video" in video_json:
            source_video = make_video(
                labels_path,
                video_json["source_video"],
                open_backend=open_backend,
            )

        # Legacy compatibility: if original_video exists but source_video doesn't,
        # use original_video as source_video
        if source_video is None and "original_video" in video_json:
            source_video = make_video(
                labels_path,
                video_json["original_video"],
                open_backend=False,  # Original videos are often not available
            )

    # Handle ImageVideo filenames - always expand to full list regardless of
    # open_backend. This ensures Video.filename is consistently a list for image
    # sequences.
    if "filenames" in backend_metadata:
        # This is an ImageVideo.
        # TODO: Path resolution.
        video_path = backend_metadata["filenames"]
        video_path = [Path(sanitize_filename(p)) for p in video_path]

    backend = None
    if open_backend:
        try:
            if not isinstance(video_path, list) and not is_file_accessible(video_path):
                # Check for the same filename in the same directory as the labels file.
                candidate_video_path = Path(labels_path).parent / video_path.name
                if is_file_accessible(candidate_video_path):
                    video_path = candidate_video_path
                else:
                    # TODO (TP): Expand capabilities of path resolution to support more
                    # complex path finding strategies.
                    pass
        except (OSError, PermissionError, FileNotFoundError):
            pass

        # Convert video path to string (only if not already a list for ImageVideo).
        if isinstance(video_path, Path):
            video_path = video_path.as_posix()

        try:
            grayscale = None
            if "grayscale" in backend_metadata:
                grayscale = backend_metadata["grayscale"]
            elif "shape" in backend_metadata:
                grayscale = backend_metadata["shape"][-1] == 1
            backend = VideoBackend.from_filename(
                video_path,
                dataset=backend_metadata.get("dataset", None),
                grayscale=grayscale,
                input_format=backend_metadata.get("input_format", None),
                format=backend_metadata.get("format", None),
            )

            # Restore FPS from metadata for backends that don't read it from file
            # (ImageVideo, HDF5Video, TiffVideo). MediaVideo reads from container.
            fps = backend_metadata.get("fps")
            if fps is not None and not isinstance(backend, MediaVideo):
                backend._fps = fps
        except Exception:
            backend = None

    # Ensure video_path is a string or list of strings (not Path) for the Video object
    if isinstance(video_path, Path):
        video_path = sanitize_filename(video_path)
    elif isinstance(video_path, list):
        # ImageVideo: convert list of Paths to list of strings
        video_path = [
            sanitize_filename(p) if isinstance(p, Path) else p for p in video_path
        ]

    return Video(
        filename=video_path,
        backend=backend,
        backend_metadata=backend_metadata,
        source_video=source_video,
        open_backend=open_backend,
    )


def read_videos(labels_path: str, open_backend: bool = True) -> list[Video]:
    """Read `Video` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        open_backend: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).

    Returns:
        A list of `Video` objects.
    """
    videos = []
    # Open file once and pass handle to make_video to avoid repeated opens
    # for embedded videos (which would otherwise open the file per video).
    with h5py.File(labels_path, "r") as f:
        videos_metadata = f["videos_json"][:]
        for video_data in videos_metadata:
            video_json = json.loads(video_data)
            video = make_video(
                labels_path, video_json, open_backend=open_backend, _hdf5_file=f
            )
            videos.append(video)
    return videos


def video_to_dict(video: Video, labels_path: str | None = None) -> dict:
    """Convert a `Video` object to a JSON-compatible dictionary.

    Args:
        video: A `Video` object to convert.
        labels_path: Path to the labels file being written. Used to determine if the
            video should use a self-reference (".") or external reference.

    Returns:
        A dictionary containing the video metadata.
    """
    video_filename = sanitize_filename(video.filename)
    result = {"filename": video_filename}

    # Add backend metadata
    if video.backend is None:
        # Copy backend_metadata to avoid mutating the original
        result["backend"] = video.backend_metadata.copy()
        # Ensure filename is always present in backend metadata for compatibility
        # with make_video() which expects backend["filename"] to exist
        if "filename" not in result["backend"]:
            result["backend"]["filename"] = video_filename
    elif type(video.backend) is MediaVideo:
        result["backend"] = {
            "type": "MediaVideo",
            "shape": video.shape,
            "filename": video_filename,
            "grayscale": video.grayscale,
            "bgr": True,
            "dataset": "",
            "input_format": "",
            "fps": video.fps,
        }
    elif type(video.backend) is HDF5Video:
        # Determine if we should use self-reference or external reference
        use_self_reference = (
            video.backend.has_embedded_images
            and labels_path is not None
            and Path(sanitize_filename(video.filename)).resolve()
            == Path(sanitize_filename(labels_path)).resolve()
        )

        result["backend"] = {
            "type": "HDF5Video",
            "shape": video.shape,
            "filename": ("." if use_self_reference else video_filename),
            "dataset": video.backend.dataset,
            "input_format": video.backend.input_format,
            "convert_range": False,
            "has_embedded_images": video.backend.has_embedded_images,
            "grayscale": video.grayscale,
            "fps": video.fps,
        }
    elif type(video.backend) is ImageVideo:
        if video.shape is not None:
            height, width, channels = video.shape[1:4]
        else:
            height, width, channels = None, None, 3
        result["backend"] = {
            "type": "ImageVideo",
            "shape": video.shape,
            "filename": sanitize_filename(video.backend.filename[0]),
            "filenames": sanitize_filename(video.backend.filename),
            "height_": height,
            "width_": width,
            "channels_": channels,
            "grayscale": video.grayscale,
            "fps": video.fps,
        }
    elif type(video.backend) is TiffVideo:
        result["backend"] = {
            "type": "TiffVideo",
            "shape": video.shape,
            "filename": video_filename,
            "grayscale": video.grayscale,
            "keep_open": video.backend.keep_open,
            "format": video.backend.format,
            "fps": video.fps,
        }

    # Add source_video metadata if present
    if hasattr(video, "source_video") and video.source_video is not None:
        result["source_video"] = video_to_dict(video.source_video, labels_path)

    # Note: original_video is now a computed property derived from source_video,
    # so we don't store it. Legacy files with original_video are handled on load.

    return result


def prepare_frames_to_embed(
    labels_path: str,
    labels: Labels,
    frames_to_embed: list[tuple[Video, int]],
) -> list[dict]:
    """Prepare frames to embed by gathering all metadata needed for embedding.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object containing the videos.
        frames_to_embed: A list of tuples of `(video, frame_idx)` specifying the
            frames to embed.

    Returns:
        A list of dictionaries, each containing metadata for a frame to embed:
            - video: The Video object
            - frame_idx: The index of the frame to embed
            - video_ind: The index of the video in labels.videos
            - group: The HDF5 group to store the embedded data in
    """
    # First, group frames by video
    to_embed_by_video = {}
    for video, frame_idx in frames_to_embed:
        if video not in to_embed_by_video:
            to_embed_by_video[video] = []
        to_embed_by_video[video].append(frame_idx)

    # Remove duplicates and sort
    for video in to_embed_by_video:
        to_embed_by_video[video] = sorted(list(set(to_embed_by_video[video])))

    # Create a list of frame metadata for embedding
    frames_metadata = []
    for video, frame_inds in to_embed_by_video.items():
        video_ind = labels.videos.index(video)
        group = f"video{video_ind}"
        for frame_idx in frame_inds:
            frames_metadata.append(
                {
                    "video": video,
                    "frame_idx": frame_idx,
                    "video_ind": video_ind,
                    "group": group,
                }
            )

    return frames_metadata


def can_use_fast_path(video: Video, frame_idx: int, target_format: str) -> bool:
    """Check if fast path copy is possible for a frame.

    The fast path allows copying raw encoded bytes directly from an embedded
    HDF5 video without decoding and re-encoding, which is faster and avoids
    quality degradation for lossy formats like JPEG.

    Args:
        video: Video object to check.
        frame_idx: Frame index to check.
        target_format: Target image format ("png", "jpg", etc.)

    Returns:
        True if the frame can be copied directly without decode/encode cycle.
    """
    from sleap_io.io.video_reading import HDF5Video

    # Must have an HDF5Video backend
    if video.backend is None or not isinstance(video.backend, HDF5Video):
        return False

    # Must have embedded images
    if not video.backend.has_embedded_images:
        return False

    # Format must match
    if video.backend.image_format != target_format:
        return False

    # Frame must be available
    if not video.backend.has_frame(frame_idx):
        return False

    return True


def process_and_embed_frames(
    labels_path: str,
    frames_metadata: list[dict],
    image_format: str = "png",
    fixed_length: bool = True,
    verbose: bool = True,
    plugin: str | None = None,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> dict[Video, Video]:
    """Process and embed frames into a SLEAP labels file.

    This function loads, encodes, and writes frames to the HDF5 file in a single loop,
    making it easier to add progress monitoring.

    Args:
        labels_path: A string path to the SLEAP labels file.
        frames_metadata: A list of dictionaries with frame metadata from
            prepare_frames_to_embed.
        image_format: The image format to use for embedding. Valid formats are "png"
            (the default), "jpg" or "hdf5".
        fixed_length: If `True` (the default), the embedded images will be padded to the
            length of the largest image. If `False`, the images will be stored as
            variable length, which is smaller but may not be supported by all readers.
        verbose: If `True` (the default), display a progress bar for the embedding
            process.
        plugin: Image plugin to use for encoding. One of "opencv" or "imageio".
            If None, uses the global default from `get_default_image_plugin()`.
            If no global default is set, auto-detects based on available packages.
        progress_callback: Optional callback function called during frame embedding
            with `(current, total)` arguments (1-based current frame index and total
            frame count). If it returns `False`, the operation is cancelled and
            `ExportCancelled` is raised. When provided, tqdm progress bar is disabled
            in favor of the callback.

    Returns:
        A dictionary mapping original Video objects to their embedded versions.

    Raises:
        ExportCancelled: If the progress_callback returns `False`.
    """
    # Determine which plugin to use for encoding
    from sleap_io.io.video_reading import get_default_image_plugin

    if plugin is None:
        plugin = get_default_image_plugin()
    if plugin is None:
        # Auto-detect: prefer opencv, fallback to imageio
        plugin = "opencv" if "cv2" in sys.modules else "imageio"

    # Initialize a dictionary to store data by group
    data_by_group = {}
    total_frames = len(frames_metadata)

    # Use tqdm only if verbose AND no callback (CLI mode)
    use_tqdm = verbose and progress_callback is None
    frame_iter = (
        tqdm(frames_metadata, desc="Embedding frames", disable=not use_tqdm)
        if use_tqdm
        else frames_metadata
    )

    for i, frame_meta in enumerate(frame_iter):
        video = frame_meta["video"]
        frame_idx = frame_meta["frame_idx"]
        group = frame_meta["group"]

        # Initialize group data structure if this is the first frame for this group
        if group not in data_by_group:
            data_by_group[group] = {
                "video": video,  # All frames in a group are from the same video
                "frame_inds": [],
                "imgs_data": [],
                "channel_order": None,  # Track channel order: "RGB" or "BGR"
            }

        # Fast path: Copy raw bytes directly if formats match (avoids decode/encode)
        # This is faster and prevents quality degradation for lossy formats like JPEG
        if can_use_fast_path(video, frame_idx, image_format):
            raw_bytes = video.backend.get_frame_raw_bytes(frame_idx)
            if raw_bytes is not None:
                data_by_group[group]["imgs_data"].append(raw_bytes)
                data_by_group[group]["frame_inds"].append(frame_idx)
                # Preserve original channel order from source
                if data_by_group[group]["channel_order"] is None:
                    data_by_group[group]["channel_order"] = video.backend.channel_order

                # Report progress via callback
                if progress_callback is not None:
                    if not progress_callback(i + 1, total_frames):
                        raise ExportCancelled("Export cancelled by user")
                continue

        # Slow path: Load and encode the frame
        frame = video[frame_idx]

        # Encode the frame
        if image_format == "hdf5":
            img_data = frame
            channel_order = "RGB"  # HDF5 format stores as-is (RGB)
        else:
            if plugin == "opencv":
                img_data = np.squeeze(
                    cv2.imencode("." + image_format, frame)[1]
                ).astype("int8")
                channel_order = "BGR"  # OpenCV encodes in BGR
            else:  # imageio
                if frame.shape[-1] == 1:
                    frame = frame.squeeze(axis=-1)
                img_data = np.frombuffer(
                    iio.imwrite("<bytes>", frame, extension="." + image_format),
                    dtype="int8",
                )
                channel_order = "RGB"  # imageio encodes in RGB

        # Store channel order (should be consistent for all frames in a group)
        if data_by_group[group]["channel_order"] is None:
            data_by_group[group]["channel_order"] = channel_order

        # Store frame data in the appropriate group
        data_by_group[group]["imgs_data"].append(img_data)
        data_by_group[group]["frame_inds"].append(frame_idx)

        # Report progress via callback
        if progress_callback is not None:
            if not progress_callback(i + 1, total_frames):
                raise ExportCancelled("Export cancelled by user")

    # Write all frame data to the HDF5 file
    replaced_videos = {}
    with h5py.File(labels_path, "a") as f:
        for group, data in data_by_group.items():
            video = data["video"]
            frame_inds = data["frame_inds"]
            imgs_data = data["imgs_data"]

            if image_format == "hdf5":
                f.create_dataset(
                    f"{group}/video", data=imgs_data, compression="gzip", chunks=True
                )
                ds = f[f"{group}/video"]
            else:
                if fixed_length:
                    img_bytes_len = 0
                    for img in imgs_data:
                        img_bytes_len = max(img_bytes_len, len(img))
                    ds = f.create_dataset(
                        f"{group}/video",
                        shape=(len(imgs_data), img_bytes_len),
                        dtype="int8",
                        compression="gzip",
                    )
                    for i, img in enumerate(imgs_data):
                        ds[i, : len(img)] = img
                else:
                    ds = f.create_dataset(
                        f"{group}/video",
                        shape=(len(imgs_data),),
                        dtype=h5py.special_dtype(vlen=np.dtype("int8")),
                    )
                    for i, img in enumerate(imgs_data):
                        ds[i] = img

            # Store metadata
            ds.attrs["format"] = image_format
            ds.attrs["channel_order"] = data["channel_order"]
            video_shape = video.shape
            (
                ds.attrs["frames"],
                ds.attrs["height"],
                ds.attrs["width"],
                ds.attrs["channels"],
            ) = video_shape

            # Store FPS if available (inherited from source video)
            if video.fps is not None:
                ds.attrs["fps"] = video.fps

            # Store frame indices
            f.create_dataset(f"{group}/frame_numbers", data=frame_inds)

            # Store source video
            if video.source_video is not None:
                source_video = video.source_video
            else:
                source_video = video

            # Create embedded video object
            embedded_video = Video(
                filename=labels_path,
                backend=VideoBackend.from_filename(
                    labels_path,
                    dataset=f"{group}/video",
                    grayscale=video.grayscale,
                    keep_open=False,
                ),
                source_video=source_video,
            )

            # Store source video metadata
            grp = f.require_group(f"{group}/source_video")
            grp.attrs["json"] = json.dumps(
                video_to_dict(source_video, labels_path), separators=(",", ":")
            )

            # Store the embedded video for return
            replaced_videos[video] = embedded_video

    return replaced_videos


def _create_empty_embedded_video(
    labels_path: str,
    video: Video,
    video_ind: int,
) -> Video:
    """Create an empty embedded video reference for a video without frames.

    This is used when exporting package files to ensure all videos point to
    the package file rather than external paths, even if they have no frames.

    Args:
        labels_path: Path to the labels file being written.
        video: The original Video object.
        video_ind: The index of this video in labels.videos.

    Returns:
        A new Video object with an empty HDF5Video backend pointing to the
        labels file, with source_video set to the original video.
    """
    group = f"video{video_ind}"

    # Determine source video (preserve chain if already embedded)
    source_video = video.source_video if video.source_video is not None else video

    # Write empty video group with source_video metadata to HDF5
    with h5py.File(labels_path, "a") as f:
        grp = f.require_group(group)

        # Store empty frame_numbers dataset
        if "frame_numbers" not in grp:
            f.create_dataset(f"{group}/frame_numbers", data=[])

        # Create empty video dataset with metadata so HDF5Video recognizes it
        if "video" not in grp:
            ds = f.create_dataset(f"{group}/video", shape=(0,), dtype="int8")
            ds.attrs["format"] = "png"
            ds.attrs["channel_order"] = "RGB"
            # Store video shape metadata from the source video
            video_shape = video.shape
            ds.attrs["frames"] = video_shape[0]
            ds.attrs["height"] = video_shape[1]
            ds.attrs["width"] = video_shape[2]
            ds.attrs["channels"] = video_shape[3]
            # Store FPS if available
            if video.fps is not None:
                ds.attrs["fps"] = video.fps

        # Store source video metadata for restoration
        source_grp = f.require_group(f"{group}/source_video")
        source_grp.attrs["json"] = json.dumps(
            video_to_dict(source_video, labels_path), separators=(",", ":")
        )

    # Create the embedded video object using VideoBackend.from_filename
    # This ensures the HDF5Video is properly initialized with the dataset metadata
    embedded_video = Video(
        filename=labels_path,
        backend=VideoBackend.from_filename(
            labels_path,
            dataset=f"{group}/video",
            grayscale=video.grayscale if video.grayscale is not None else False,
            keep_open=False,
        ),
        source_video=source_video,
    )

    return embedded_video


def embed_frames(
    labels_path: str,
    labels: Labels,
    embed: list[tuple[Video, int]],
    image_format: str = "png",
    verbose: bool = True,
    plugin: str | None = None,
    embed_all_videos: bool = True,
    progress_callback: Callable[[int, int], bool] | None = None,
):
    """Embed frames in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to embed in the labels file.
        embed: A list of tuples of `(video, frame_idx)` specifying the frames to embed.
        image_format: The image format to use for embedding. Valid formats are "png"
            (the default), "jpg" or "hdf5".
        verbose: If `True` (the default), display a progress bar for the embedding
            process.
        plugin: Image plugin to use for encoding. One of "opencv" or "imageio".
            If None, uses the global default from `get_default_image_plugin()`.
        embed_all_videos: If `True` (the default), all videos in the labels will be
            converted to embedded references, even if they have no frames to embed.
            This ensures package files are portable. If `False`, only videos with
            frames to embed are converted.
        progress_callback: Optional callback function called during frame embedding
            with `(current, total)` arguments. If it returns `False`, the operation
            is cancelled and `ExportCancelled` is raised.

    Notes:
        This function will embed the frames in the labels file and update the `Videos`
        and `Labels` objects in place.
    """
    frames_metadata = prepare_frames_to_embed(labels_path, labels, embed)
    replaced_videos = process_and_embed_frames(
        labels_path,
        frames_metadata,
        image_format=image_format,
        verbose=verbose,
        plugin=plugin,
        progress_callback=progress_callback,
    )

    # Handle videos without any frames to embed.
    # These still need embedded references so the package is portable.
    if embed_all_videos:
        videos_with_frames = {fm["video"] for fm in frames_metadata}
        for video_ind, video in enumerate(labels.videos):
            if video not in videos_with_frames and video not in replaced_videos:
                replaced_videos[video] = _create_empty_embedded_video(
                    labels_path, video, video_ind
                )

    if len(replaced_videos) > 0:
        labels.replace_videos(video_map=replaced_videos)


def embed_videos(
    labels_path: str,
    labels: Labels,
    embed: bool | str | list[tuple[Video, int]],
    verbose: bool = True,
    plugin: str | None = None,
    embed_all_videos: bool = True,
    progress_callback: Callable[[int, int], bool] | None = None,
):
    """Embed videos in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file to save.
        labels: A `Labels` object to save.
        embed: Frames to embed in the saved labels file. One of `None`, `True`,
            `"all"`, `"user"`, `"suggestions"`, `"user+suggestions"`, `"source"` or list
            of tuples of `(video, frame_idx)`.

            If `None` is specified (the default) and the labels contains embedded
            frames, those embedded frames will be re-saved to the new file.

            If `True` or `"all"`, all labeled frames and suggested frames will be
            embedded.
        verbose: If `True` (the default), display a progress bar for the embedding
            process.
        plugin: Image plugin to use for encoding. One of "opencv" or "imageio".
            If None, uses the global default from `get_default_image_plugin()`.

            If `"source"` is specified, no images will be embedded and the source video
            will be restored if available.

            This argument is only valid for the SLP backend.
        embed_all_videos: If `True` (the default), all videos in the labels will be
            converted to embedded references, even if they have no frames to embed.
            This ensures package files are portable. If `False`, only videos with
            frames to embed are converted.
        progress_callback: Optional callback function called during frame embedding
            with `(current, total)` arguments. If it returns `False`, the operation
            is cancelled and `ExportCancelled` is raised.
    """
    if embed is True:
        embed = "all"
    if embed == "user":
        embed = [(lf.video, lf.frame_idx) for lf in labels.user_labeled_frames]
    elif embed == "suggestions":
        embed = [(sf.video, sf.frame_idx) for sf in labels.suggestions]
    elif embed == "user+suggestions":
        embed = [(lf.video, lf.frame_idx) for lf in labels.user_labeled_frames]
        embed += [(sf.video, sf.frame_idx) for sf in labels.suggestions]
    elif embed == "all":
        embed = [(lf.video, lf.frame_idx) for lf in labels]
        embed += [(sf.video, sf.frame_idx) for sf in labels.suggestions]
    elif embed == "source":
        embed = []
    elif isinstance(embed, list):
        embed = embed
    else:
        raise ValueError(f"Invalid value for embed: {embed}")

    embed_frames(
        labels_path,
        labels,
        embed,
        verbose=verbose,
        plugin=plugin,
        embed_all_videos=embed_all_videos,
        progress_callback=progress_callback,
    )


def write_videos(
    labels_path: str,
    videos: list[Video],
    restore_source: bool = False,
    reference_mode: VideoReferenceMode | None = None,
    original_videos: list[Video] | None = None,
    verbose: bool = True,
):
    """Write video metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects to store the metadata for.
        restore_source: Deprecated. Use reference_mode instead. If `True`, restore
            source videos if available and will not re-embed the embedded images.
            If `False` (the default), will re-embed images that were previously
            embedded.
        reference_mode: How to handle video references:
            - EMBED: Re-embed frames that were previously embedded
            - RESTORE_ORIGINAL: Use original video if available
            - PRESERVE_SOURCE: Keep reference to source file (e.g., .pkg.slp)
        original_videos: Optional list of original video objects before embedding.
            Used when reference_mode is EMBED to preserve metadata.
        verbose: If `True` (the default), display a progress bar when embedding frames.
    """
    # Handle backwards compatibility
    if reference_mode is None:
        if restore_source:
            reference_mode = VideoReferenceMode.RESTORE_ORIGINAL
        else:
            reference_mode = VideoReferenceMode.EMBED

    videos_to_embed = []
    videos_to_write = []
    videos_to_copy = []  # For embedded videos without backend (raw HDF5 copy)

    # First determine which videos need embedding
    for video_ind, video in enumerate(videos):
        # Check if video has an open backend with embedded images
        has_backend_with_embedded = (
            type(video.backend) is HDF5Video and video.backend.has_embedded_images
        )
        # Also detect embedded videos via metadata (when backend is None)
        has_embedded_via_metadata = (
            video.backend is None and _is_embedded_video_metadata(video)
        )

        if has_backend_with_embedded:
            if reference_mode == VideoReferenceMode.RESTORE_ORIGINAL:
                if video.source_video is None:
                    # No source video available, reference the current embedded video
                    # file
                    videos_to_write.append((video_ind, video))
                else:
                    # Use the source video
                    videos_to_write.append((video_ind, video.source_video))
            elif reference_mode == VideoReferenceMode.PRESERVE_SOURCE:
                # Keep the reference to the source .pkg.slp file
                videos_to_write.append((video_ind, video))
            else:  # EMBED mode
                # If the video has embedded images, check if we need to re-embed them
                already_embedded = False
                if Path(labels_path).exists():
                    with h5py.File(labels_path, "r") as f:
                        already_embedded = f"video{video_ind}/video" in f

                if already_embedded:
                    videos_to_write.append((video_ind, video))
                else:
                    # Collect information for embedding
                    frames_to_embed = [
                        (video, frame_idx) for frame_idx in video.backend.source_inds
                    ]
                    videos_to_embed.append((video_ind, video, frames_to_embed))
        elif has_embedded_via_metadata:
            # Video has embedded frames but backend is not open (open_videos=False)
            if reference_mode == VideoReferenceMode.RESTORE_ORIGINAL:
                if video.source_video is None:
                    videos_to_write.append((video_ind, video))
                else:
                    videos_to_write.append((video_ind, video.source_video))
            elif reference_mode == VideoReferenceMode.PRESERVE_SOURCE:
                videos_to_write.append((video_ind, video))
            else:  # EMBED mode
                # Check if already embedded in destination
                already_embedded = False
                if Path(labels_path).exists():
                    with h5py.File(labels_path, "r") as f:
                        already_embedded = f"video{video_ind}/video" in f

                if already_embedded:
                    videos_to_write.append((video_ind, video))
                else:
                    # Need to copy raw HDF5 data from source file
                    videos_to_copy.append((video_ind, video))
        else:
            videos_to_write.append((video_ind, video))

    # Process videos that need embedding
    if videos_to_embed:
        # Prepare all frames to embed
        all_frames_to_embed = []
        for video_ind, video, frames in videos_to_embed:
            for frame in frames:
                all_frames_to_embed.append(frame)

        # Create a temporary Labels object for embedding
        temp_labels = Labels(
            videos=[v for _, v, _ in videos_to_embed], labeled_frames=[]
        )

        # Prepare and embed all frames in a single process
        frames_metadata = prepare_frames_to_embed(
            labels_path, temp_labels, all_frames_to_embed
        )
        replaced_videos = process_and_embed_frames(
            labels_path,
            frames_metadata,
            image_format=[
                v.backend.image_format if hasattr(v.backend, "image_format") else "png"
                for _, v, _ in videos_to_embed
            ][0],  # Use the first video's format
            verbose=verbose,
        )

        # Add the embedded videos to the list
        for video_ind, video, _ in videos_to_embed:
            if video in replaced_videos:
                videos_to_write.append((video_ind, replaced_videos[video]))

    # Copy raw HDF5 data for embedded videos without backends
    if videos_to_copy:
        for video_ind, video in videos_to_copy:
            # Get the source file path (video.filename points to the source pkg.slp)
            source_path = video.filename
            if not Path(source_path).exists():
                # Can't copy if source doesn't exist, just write metadata
                videos_to_write.append((video_ind, video))
                continue

            # Get the source dataset name from backend_metadata
            meta = video.backend_metadata
            source_dataset = meta.get("dataset", "") if meta else ""
            if not source_dataset:
                videos_to_write.append((video_ind, video))
                continue

            # Extract the video group name (e.g., "video0" from "video0/video")
            source_group = source_dataset.split("/")[0] if "/" in source_dataset else ""
            if not source_group:
                videos_to_write.append((video_ind, video))
                continue

            # Destination group name uses the current video index
            dest_group = f"video{video_ind}"

            # Copy the entire video group from source to destination
            with h5py.File(source_path, "r") as src_f:
                if source_group not in src_f:
                    videos_to_write.append((video_ind, video))
                    continue

                with h5py.File(labels_path, "a") as dst_f:
                    # Copy the video group with all its datasets and attributes
                    src_f.copy(source_group, dst_f, name=dest_group)

            # Add to videos_to_write - the metadata will reference the copied data
            videos_to_write.append((video_ind, video))

    # Write video metadata
    video_jsons = []
    for video_ind, video in sorted(videos_to_write, key=lambda x: x[0]):
        video_json = video_to_dict(video, labels_path)
        video_jsons.append(np.bytes_(json.dumps(video_json, separators=(",", ":"))))

    with h5py.File(labels_path, "a") as f:
        if "videos_json" not in f:
            f.create_dataset("videos_json", data=video_jsons, maxshape=(None,))

    # Save source_video lineage metadata in a separate pass to ensure video groups exist
    # Note: original_video is now a computed property derived from source_video chain,
    # so we only need to store source_video (immediate parent).
    with h5py.File(labels_path, "a") as f:
        for video_ind, video in enumerate(videos):
            dataset = f"video{video_ind}"

            # If original_videos is provided (e.g., during embedding), use those
            pre_embed_video = original_videos[video_ind] if original_videos else video

            # Determine source_video to save based on reference mode
            source_to_save = None
            if reference_mode != VideoReferenceMode.PRESERVE_SOURCE:
                if reference_mode == VideoReferenceMode.EMBED and original_videos:
                    # For embed mode, save the pre-embedding video as source
                    source_to_save = pre_embed_video
                elif pre_embed_video.source_video is not None:
                    source_to_save = pre_embed_video.source_video

            # Write source_video metadata to the video group
            if dataset in f and source_to_save is not None:
                video_group = f[dataset]

                # For EMBED mode with original_videos, we need to overwrite
                # source_video because embed_videos saves the wrong metadata
                if (
                    reference_mode == VideoReferenceMode.EMBED
                    and original_videos
                    and "source_video" in video_group
                ):
                    del video_group["source_video"]

                if "source_video" not in video_group:
                    source_grp = video_group.require_group("source_video")
                    source_json = video_to_dict(source_to_save, labels_path)
                    source_grp.attrs["json"] = json.dumps(
                        source_json, separators=(",", ":")
                    )


def read_tracks(labels_path: str) -> list[Track]:
    """Read `Track` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `Track` objects.
    """
    tracks = [json.loads(x) for x in read_hdf5_dataset(labels_path, "tracks_json")]
    track_objects = []
    for track in tracks:
        track_objects.append(Track(name=track[1]))
    return track_objects


def write_tracks(labels_path: str, tracks: list[Track]):
    """Write track metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        tracks: A list of `Track` objects to store the metadata for.
    """
    # TODO: Add support for track metadata like spawned on frame.
    SPAWNED_ON = 0
    tracks_json = [
        np.bytes_(json.dumps([SPAWNED_ON, track.name], separators=(",", ":")))
        for track in tracks
    ]
    with h5py.File(labels_path, "a") as f:
        f.create_dataset("tracks_json", data=tracks_json, maxshape=(None,))


def read_identities(labels_path: str) -> list[Identity]:
    """Read Identity dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `Identity` objects.
    """
    try:
        identities = read_hdf5_dataset(labels_path, "identities_json")
    except KeyError:
        return []
    identity_objects = []
    for identity_data in identities:
        d = json.loads(identity_data)
        identity_objects.append(
            Identity(
                name=d.get("name", ""),
                color=d.get("color", None),
                metadata={k: v for k, v in d.items() if k not in ("name", "color")},
            )
        )
    return identity_objects


def write_identities(labels_path: str, identities: list[Identity]):
    """Write Identity metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        identities: A list of `Identity` objects.
    """
    if not identities:
        return

    identities_json = []
    for identity in identities:
        d = {"name": identity.name}
        if identity.color is not None:
            d["color"] = identity.color
        d.update(identity.metadata)
        identities_json.append(np.bytes_(json.dumps(d, separators=(",", ":"))))

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("identities_json", data=identities_json, maxshape=(None,))


def read_suggestions(labels_path: str, videos: list[Video]) -> list[SuggestionFrame]:
    """Read `SuggestionFrame` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects.

    Returns:
        A list of `SuggestionFrame` objects.
    """
    try:
        suggestions = read_hdf5_dataset(labels_path, "suggestions_json")
    except KeyError:
        return []
    suggestions = [json.loads(x) for x in suggestions]
    suggestions_objects = []
    for suggestion in suggestions:
        # Extract metadata (e.g., "group")
        metadata = {"group": suggestion.get("group", 0)}

        suggestions_objects.append(
            SuggestionFrame(
                video=videos[int(suggestion["video"])],
                frame_idx=suggestion["frame_idx"],
                metadata=metadata,
            )
        )
    return suggestions_objects


def write_suggestions(
    labels_path: str, suggestions: list[SuggestionFrame], videos: list[Video]
):
    """Write track metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        suggestions: A list of `SuggestionFrame` objects to store the metadata for.
        videos: A list of `Video` objects.
    """
    suggestions_json = []
    for suggestion in suggestions:
        # Get group from metadata if available, otherwise use default
        group = suggestion.metadata.get("group", 0) if suggestion.metadata else 0

        suggestion_dict = {
            "video": str(videos.index(suggestion.video)),
            "frame_idx": suggestion.frame_idx,
            "group": group,
        }
        suggestion_json = np.bytes_(json.dumps(suggestion_dict, separators=(",", ":")))
        suggestions_json.append(suggestion_json)

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("suggestions_json", data=suggestions_json, maxshape=(None,))


def write_negative_frames(labels_path: str, labels: Labels):
    """Write negative frame markers to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object containing negative frames to write.

    Notes:
        Uses sparse video IDs (same as /frames dataset) for consistency when videos
        are embedded or reordered. The /negative_frames dataset stores (video_id,
        frame_idx) tuples identifying which frames are explicitly marked as negative
        (pure background, no instances).
    """
    # Build video index to sparse ID mapping (reuse pattern from write_lfs)
    video_idx_id_map = {}
    for video_idx, video in enumerate(labels.videos):
        # Default to sequential index
        video_idx_id_map[video_idx] = video_idx

        # Check if this is an embedded video with a sparse video ID
        if (
            hasattr(video, "backend")
            and video.backend is not None
            and hasattr(video.backend, "dataset")
            and video.backend.dataset is not None
        ):
            dataset = video.backend.dataset
            # Extract video ID from dataset name (e.g., "video15/video" → 15)
            try:
                video_group = dataset.split("/")[0]
                if video_group.startswith("video"):
                    video_id = int(video_group[5:])  # Remove "video" prefix and convert
                    video_idx_id_map[video_idx] = video_id
            except (ValueError, IndexError):
                # If parsing fails, keep the default sequential index
                pass

    # Collect negative frames
    negative_data = []
    for lf in labels.labeled_frames:
        if lf.is_negative:
            video_idx = labels.videos.index(lf.video)
            sparse_video_id = video_idx_id_map[video_idx]
            negative_data.append((sparse_video_id, lf.frame_idx))

    if negative_data:
        dtype = np.dtype([("video_id", "u4"), ("frame_idx", "u8")])
        data = np.array(negative_data, dtype=dtype)
        with h5py.File(labels_path, "a") as f:
            if "negative_frames" in f:
                del f["negative_frames"]  # Replace if exists
            f.create_dataset("negative_frames", data=data)


def read_negative_frames(labels_path: str) -> set[tuple[int, int]]:
    """Read negative frame markers from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A set of (sparse_video_id, frame_idx) tuples identifying negative frames.
        Returns empty set if no negative frames dataset exists.
    """
    try:
        data = read_hdf5_dataset(labels_path, "negative_frames")
        return {(int(row["video_id"]), int(row["frame_idx"])) for row in data}
    except KeyError:
        return set()


def read_metadata(labels_path: str) -> dict:
    """Read metadata from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A dict containing the metadata from a SLEAP labels file.
    """
    md = read_hdf5_attrs(labels_path, "metadata", "json")
    if isinstance(md, bytes):
        md = md.decode()
    elif isinstance(md, np.ndarray):
        md = md.tobytes().decode()
    # If md is already a str (e.g., h5py vlen string), use as-is.
    return json.loads(md)


def read_skeletons(labels_path: str) -> list[Skeleton]:
    """Read `Skeleton` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file.

    Returns:
        A list of `Skeleton` objects.
    """
    metadata = read_metadata(labels_path)

    # Get node names. This is a superset of all nodes across all skeletons. Note that
    # node ordering is specific to each skeleton, so we'll need to fix this afterwards.
    node_names = [x["name"] for x in metadata["nodes"]]

    # Use the SLP skeleton decoder
    decoder = SkeletonSLPDecoder()
    return decoder.decode(metadata, node_names)


def serialize_skeletons(skeletons: list[Skeleton]) -> tuple[list[dict], list[dict]]:
    """Serialize a list of `Skeleton` objects to JSON-compatible dicts.

    Args:
        skeletons: A list of `Skeleton` objects.

    Returns:
        A tuple of `skeletons_dicts, nodes_dicts`.

        `nodes_dicts` is a list of dicts containing the nodes in all the skeletons.

        `skeletons_dicts` is a list of dicts containing the skeletons.

    Notes:
        This function attempts to replicate the serialization of skeletons in legacy
        SLEAP which relies on a combination of networkx's graph serialization and our
        own metadata used to store nodes and edges independent of the graph structure.

        However, because sleap-io does not currently load in the legacy metadata, this
        function will not produce byte-level compatible serialization with legacy
        formats, even though the ordering and all attributes of nodes and edges should
        match up.
    """
    # Use the SLP skeleton encoder
    encoder = SkeletonSLPEncoder()
    return encoder.encode_skeletons(skeletons)


def write_metadata(labels_path: str, labels: Labels):
    """Write metadata to a SLEAP labels file.

    This function will write the skeletons and provenance for the labels.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.

    See also: serialize_skeletons
    """
    skeletons_dicts, nodes_dicts = serialize_skeletons(labels.skeletons)

    md = {
        "version": "2.0.0",
        "skeletons": skeletons_dicts,
        "nodes": nodes_dicts,
        "videos": [],
        "tracks": [],
        "suggestions": [],  # TODO: Handle suggestions metadata.
        "negative_anchors": {},
        "provenance": labels.provenance,
    }

    # Custom encoding.
    for k in md["provenance"]:
        if isinstance(md["provenance"][k], Path):
            # Path -> str
            md["provenance"][k] = md["provenance"][k].as_posix()

    # Bump format_id based on features used
    has_instance_rois = any(
        roi.instance is not None or roi._instance_idx >= 0 for roi in labels.rois
    )
    has_predicted_annotations = (
        any(isinstance(m, PredictedSegmentationMask) for m in labels.masks)
        or any(isinstance(r, PredictedROI) for r in labels.rois)
        or any(isinstance(li, PredictedLabelImage) for li in labels.label_images)
    )
    has_mask_instances = any(
        mask.instance is not None or mask._instance_idx >= 0 for mask in labels.masks
    )
    has_identities = len(labels.identities) > 0
    if labels.bboxes:
        format_id = 2.0
    elif has_predicted_annotations or has_mask_instances:
        format_id = 1.9
    elif labels.label_images:
        format_id = 1.8
    elif has_instance_rois:
        format_id = 1.6
    elif labels.rois or labels.masks:
        format_id = 1.5
    else:
        format_id = 1.4

    # Bump for identities (new in 1.9)
    if has_identities:
        format_id = max(format_id, 1.9)

    # Bump for spatial metadata on dense annotations (new in 2.1)
    has_spatial_metadata = any(m.has_spatial_transform for m in labels.masks) or any(
        li.has_spatial_transform for li in labels.label_images
    )
    if has_spatial_metadata:
        format_id = max(format_id, 2.1)

    with h5py.File(labels_path, "a") as f:
        # Bump for chunked label image format (new in 2.2)
        if "label_image_data" in f and f["label_image_data"].ndim == 3:
            format_id = max(format_id, 2.2)

        grp = f.require_group("metadata")
        grp.attrs["format_id"] = format_id
        grp.attrs["json"] = np.bytes_(json.dumps(md, separators=(",", ":")))


def read_points(labels_path: str) -> np.ndarray:
    """Read points dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A structured array of point data.
    """
    pts = read_hdf5_dataset(labels_path, "points")
    return pts


def read_pred_points(labels_path: str) -> np.ndarray:
    """Read predicted points dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A structured array of predicted point data.
    """
    pred_pts = read_hdf5_dataset(labels_path, "pred_points")
    return pred_pts


def _points_from_hdf5_data(
    pts_data: np.ndarray,
    skeleton: Skeleton,
    is_predicted: bool = False,
) -> "PointsArray | PredictedPointsArray":
    """Build PointsArray directly from HDF5 structured array data.

    This is a fast path that avoids column_stack and intermediate array creation
    by directly constructing the target PointsArray structure from HDF5 data.

    Args:
        pts_data: Structured array from HDF5 with fields x, y, visible, complete,
            and optionally score.
        skeleton: The skeleton defining the node structure.
        is_predicted: If True, create a PredictedPointsArray with scores.

    Returns:
        A fully populated PointsArray or PredictedPointsArray.
    """
    from sleap_io.model.instance import PointsArray, PredictedPointsArray

    n = len(pts_data)

    if is_predicted:
        points = PredictedPointsArray.empty(n)
        points["score"] = pts_data["score"]
    else:
        points = PointsArray.empty(n)

    # Direct field assignment (faster than column_stack)
    points["xy"][:, 0] = pts_data["x"]
    points["xy"][:, 1] = pts_data["y"]
    points["visible"] = pts_data["visible"]
    points["complete"] = pts_data["complete"]
    points["name"] = skeleton.node_names

    return points


def read_instances(
    labels_path: str,
    skeletons: list[Skeleton],
    tracks: list[Track],
    points: np.ndarray,
    pred_points: np.ndarray,
    format_id: float,
) -> list[Instance | PredictedInstance]:
    """Read `Instance` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        skeletons: A list of `Skeleton` objects (see `read_skeletons`).
        tracks: A list of `Track` objects (see `read_tracks`).
        points: A structured array of point data (see `read_points`).
        pred_points: A structured array of predicted point data (see
            `read_pred_points`).
        format_id: The format version identifier used to specify the format of the input
            file.

    Returns:
        A list of `Instance` and/or `PredictedInstance` objects.
    """
    instances_data = read_hdf5_dataset(labels_path, "instances")

    instances = {}
    from_predicted_pairs = []
    for instance_data in instances_data:
        if format_id < 1.2:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
            ) = instance_data
            tracking_score = 0.0
        elif format_id >= 1.2:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
                tracking_score,
            ) = instance_data

        # Cast index values to int for h5wasm compatibility. h5wasm may write
        # all columns as float64, which can't be used as list indices or slice
        # bounds. Safe for compound dtypes too: int(numpy.int64(x)) -> int.
        instance_id = int(instance_id)
        skeleton_id = int(skeleton_id)
        track_id = int(track_id)
        from_predicted = int(from_predicted)
        point_id_start = int(point_id_start)
        point_id_end = int(point_id_end)

        skeleton = skeletons[skeleton_id]
        track = tracks[track_id] if track_id >= 0 else None

        if instance_type == InstanceType.USER:
            pts_data = points[point_id_start:point_id_end]
            # Fast path: Build PointsArray directly from HDF5 data
            points_array = _points_from_hdf5_data(
                pts_data, skeleton, is_predicted=False
            )
            if format_id < 1.1:
                # Legacy coordinate system: top-left of pixel is (0, 0)
                # Adjust to new system: center of pixel is (0, 0)
                points_array["xy"] -= 0.5
            inst = Instance(
                points_array,
                skeleton=skeleton,
                track=track,
                tracking_score=tracking_score,
            )
            instances[instance_id] = inst

        elif instance_type == InstanceType.PREDICTED:
            pts_data = pred_points[point_id_start:point_id_end]
            # Fast path: Build PredictedPointsArray directly from HDF5 data
            points_array = _points_from_hdf5_data(pts_data, skeleton, is_predicted=True)
            if format_id < 1.1:
                # Legacy coordinate system: top-left of pixel is (0, 0)
                # Adjust to new system: center of pixel is (0, 0)
                points_array["xy"] -= 0.5
            inst = PredictedInstance(
                points_array,
                skeleton=skeleton,
                track=track,
                score=instance_score,
                tracking_score=tracking_score,
            )
            instances[instance_id] = inst

        if from_predicted >= 0:
            from_predicted_pairs.append((instance_id, from_predicted))

    # Link instances based on from_predicted field.
    for instance_id, from_predicted in from_predicted_pairs:
        instances[instance_id].from_predicted = instances[from_predicted]

    # Convert instances back to list.
    instances = list(instances.values())

    return instances


def write_lfs(labels_path: str, labels: Labels):
    """Write labeled frames, instances and points to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.
    """
    # We store the data in structured arrays for performance, so we first define the
    # dtype fields.
    instance_dtype = np.dtype(
        [
            ("instance_id", "i8"),
            ("instance_type", "u1"),
            ("frame_id", "u8"),
            ("skeleton", "u4"),
            ("track", "i4"),
            ("from_predicted", "i8"),
            ("score", "f4"),
            ("point_id_start", "u8"),
            ("point_id_end", "u8"),
            ("tracking_score", "f4"),  # FORMAT_ID >= 1.2 (1.3 adds explicit handling)
        ]
    )
    frame_dtype = np.dtype(
        [
            ("frame_id", "u8"),
            ("video", "u4"),
            ("frame_idx", "u8"),
            ("instance_id_start", "u8"),
            ("instance_id_end", "u8"),
        ]
    )
    point_dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")]
    )
    predicted_point_dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?"), ("score", "f8")]
    )

    # Next, we extract the data from the labels object into lists with the same fields.
    frames, instances, points, predicted_points, to_link = [], [], [], [], []
    inst_to_id = {}
    # get sparse ids instead of list indices
    video_idx_id_map = {}
    for video_idx, video in enumerate(labels.videos):
        # Default to sequential index
        video_idx_id_map[video_idx] = video_idx

        # Check if this is an embedded video with a sparse video ID
        if (
            hasattr(video, "backend")
            and video.backend is not None
            and hasattr(video.backend, "dataset")
            and video.backend.dataset is not None
        ):
            dataset = video.backend.dataset
            # Extract video ID from dataset name (e.g., "video15/video" → 15)
            try:
                video_group = dataset.split("/")[0]
                if video_group.startswith("video"):
                    video_id = int(video_group[5:])  # Remove "video" prefix and convert
                    video_idx_id_map[video_idx] = video_id
            except (ValueError, IndexError):
                # If parsing fails, keep the default sequential index
                pass
    for lf in labels:
        frame_id = len(frames)
        instance_id_start = len(instances)
        for inst in lf:
            instance_id = len(instances)
            inst_to_id[id(inst)] = instance_id
            skeleton_id = labels.skeletons.index(inst.skeleton)
            track = labels.tracks.index(inst.track) if inst.track else -1
            from_predicted = -1
            if inst.from_predicted:
                to_link.append((instance_id, inst.from_predicted))
            score = 0.0

            if type(inst) is Instance:
                instance_type = InstanceType.USER
                tracking_score = inst.tracking_score
                point_id_start = len(points)

                for pt in inst.points:
                    points.append(
                        [pt["xy"][0], pt["xy"][1], pt["visible"], pt["complete"]]
                    )

                point_id_end = len(points)

            elif type(inst) is PredictedInstance:
                instance_type = InstanceType.PREDICTED
                score = inst.score
                tracking_score = inst.tracking_score
                point_id_start = len(predicted_points)

                for pt in inst.points:
                    predicted_points.append(
                        [
                            pt["xy"][0],
                            pt["xy"][1],
                            pt["visible"],
                            pt["complete"],
                            pt["score"],
                        ]
                    )

                point_id_end = len(predicted_points)

            else:
                raise ValueError(f"Unknown instance type: {type(inst)}")

            instances.append(
                [
                    instance_id,
                    int(instance_type),
                    frame_id,
                    skeleton_id,
                    track,
                    from_predicted,
                    score,
                    point_id_start,
                    point_id_end,
                    tracking_score,
                ]
            )

        instance_id_end = len(instances)

        frames.append(
            [
                frame_id,
                video_idx_id_map[labels.videos.index(lf.video)],
                lf.frame_idx,
                instance_id_start,
                instance_id_end,
            ]
        )

    # Link instances based on from_predicted field.
    for instance_id, from_predicted in to_link:
        # Source instance may be missing if predictions were removed from the labels, in
        # which case, remove the link.
        instances[instance_id][5] = inst_to_id.get(id(from_predicted), -1)

    # Create structured arrays.
    points = np.array([tuple(x) for x in points], dtype=point_dtype)
    predicted_points = np.array(
        [tuple(x) for x in predicted_points], dtype=predicted_point_dtype
    )
    instances = np.array([tuple(x) for x in instances], dtype=instance_dtype)
    frames = np.array([tuple(x) for x in frames], dtype=frame_dtype)

    # Write to file.
    with h5py.File(labels_path, "a") as f:
        f.create_dataset("points", data=points, dtype=points.dtype)
        f.create_dataset(
            "pred_points",
            data=predicted_points,
            dtype=predicted_points.dtype,
        )
        f.create_dataset(
            "instances",
            data=instances,
            dtype=instances.dtype,
        )
        f.create_dataset(
            "frames",
            data=frames,
            dtype=frames.dtype,
        )


def make_instance_group(
    instance_group_dict: dict,
    labeled_frames: list[LabeledFrame],
    camera_group: CameraGroup,
    identities: list[Identity] | None = None,
) -> InstanceGroup:
    """Creates an `InstanceGroup` object from a dictionary.

    Args:
        instance_group_dict: Dictionary with the following necessary key:
            - "camcorder_to_lf_and_inst_idx_map": Dictionary mapping `Camera` indices to
                a tuple of `LabeledFrame` index (in `labeled_frames`) and `Instance`
                index (in containing `LabeledFrame.instances`).
            and optional keys:
            - "score": A float representing the reprojection score for the
                `InstanceGroup`.
            - "points": 3D points for the `InstanceGroup`.
            - Any keys containing metadata.
        labeled_frames: List of `LabeledFrame` objects (expecting
            `Labels.labeled_frames`) used to retrieve `Instance` objects.
        camera_group: `CameraGroup` object used to retrieve `Camera` objects.
        identities: Optional list of `Identity` objects for resolving identity
            indices.

    Returns:
        `InstanceGroup` object.
    """
    # Avoid mutating the dictionary
    instance_group_dict = instance_group_dict.copy()

    # Get the `Instance` objects
    camera_to_lf_and_inst_idx_map: dict[str, tuple[str, str]] = instance_group_dict.pop(
        "camcorder_to_lf_and_inst_idx_map"
    )

    instance_by_camera: dict[Camera, Instance] = {}
    for cam_idx, (lf_idx, inst_idx) in camera_to_lf_and_inst_idx_map.items():
        # Retrieve the `Camera`
        camera = camera_group.cameras[int(cam_idx)]

        # Retrieve the `Instance` from the `LabeledFrame
        labeled_frame = labeled_frames[int(lf_idx)]
        instance = labeled_frame.instances[int(inst_idx)]

        # Link the `Instance` to the `Camera`
        instance_by_camera[camera] = instance

    # Get all optional attributes
    score = None
    if "score" in instance_group_dict:
        score = instance_group_dict.pop("score")

    # 3D points → Instance3D
    instance_3d = None
    points = instance_group_dict.pop("points", None)
    if points is not None:
        skeleton = None
        for inst in instance_by_camera.values():
            skeleton = inst.skeleton
            break
        if skeleton is not None:
            inst3d_score = instance_group_dict.pop("instance_3d_score", None)
            point_scores = instance_group_dict.pop("instance_3d_point_scores", None)
            if point_scores is not None:
                instance_3d = PredictedInstance3D(
                    points=points,
                    skeleton=skeleton,
                    score=inst3d_score,
                    point_scores=point_scores,
                )
            else:
                instance_3d = Instance3D(
                    points=points,
                    skeleton=skeleton,
                    score=inst3d_score,
                )
        else:
            warnings.warn(
                "3D points discarded for InstanceGroup: no skeleton available "
                "(all camera mappings failed)."
            )

    # Identity
    identity = None
    identity_idx = instance_group_dict.pop("identity_idx", None)
    if identity_idx is not None and identities is not None:
        idx = int(identity_idx)
        if 0 <= idx < len(identities):
            identity = identities[idx]
        else:
            warnings.warn(
                f"identity_idx {idx} out of range "
                f"(max {len(identities) - 1}); identity set to None."
            )

    # Metadata contains any information that the class does not deserialize.
    metadata = instance_group_dict  # Remaining keys are metadata.

    return InstanceGroup(
        instance_by_camera=instance_by_camera,
        score=score,
        instance_3d=instance_3d,
        identity=identity,
        metadata=metadata,
    )


def make_frame_group(
    frame_group_dict: dict,
    labeled_frames: list[LabeledFrame],
    camera_group: CameraGroup,
    identities: list[Identity] | None = None,
) -> FrameGroup:
    """Create a `FrameGroup` object from a dictionary.

    Args:
        frame_group_dict: Dictionary representing a `FrameGroup` object with the
            following necessary key:
            - "instance_groups": List of dictionaries containing `InstanceGroup`
                information (see `make_instance_group` for what each dictionary
                contains).
            and optional keys:
            - "frame_idx": Frame index.
            - Any keys containing metadata.
        labeled_frames: List of `LabeledFrame` objects (expecting
            `Labels.labeled_frames`).
        camera_group: `CameraGroup` object used to retrieve `Camera` objects.
        identities: Optional list of `Identity` objects for resolving identity
            indices.

    Returns:
        `FrameGroup` object.
    """
    # Avoid mutating the dictionary
    frame_group_dict = frame_group_dict.copy()

    frame_idx = None

    # Get `InstanceGroup` objects
    instance_groups_info = frame_group_dict.pop("instance_groups")
    instance_groups = []
    labeled_frame_by_camera = {}
    for instance_group_dict in instance_groups_info:
        instance_group = make_instance_group(
            instance_group_dict=instance_group_dict,
            labeled_frames=labeled_frames,
            camera_group=camera_group,
            identities=identities,
        )
        instance_groups.append(instance_group)

        # Also retrieve the `LabeledFrame` by `Camera`. We do this for each
        # `InstanceGroup` to ensure that we have don't miss a `LabeledFrame`.
        camera_to_lf_and_inst_idx_map = instance_group_dict[
            "camcorder_to_lf_and_inst_idx_map"
        ]
        for cam_idx, (lf_idx, _) in camera_to_lf_and_inst_idx_map.items():
            # Retrieve the `Camera`
            camera = camera_group.cameras[int(cam_idx)]

            # Retrieve the `LabeledFrame`
            labeled_frame = labeled_frames[int(lf_idx)]
            labeled_frame_by_camera[camera] = labeled_frame

            # We can get the frame index from the `LabeledFrame` if any.
            frame_idx = labeled_frame.frame_idx

    # Get the frame index explicitly from the dictionary if it exists.
    if "frame_idx" in frame_group_dict:
        frame_idx = frame_group_dict.pop("frame_idx")

    # Metadata contains any information that the class doesn't deserialize.
    metadata = frame_group_dict  # Remaining keys are metadata.

    return FrameGroup(
        frame_idx=frame_idx,
        instance_groups=instance_groups,
        labeled_frame_by_camera=labeled_frame_by_camera,
        metadata=metadata,
    )


def make_camera(camera_dict: dict) -> Camera:
    """Create `Camera` from a dictionary.

    Args:
        camera_dict: Dictionary containing camera information with the following
            necessary keys:
            - "name": Camera name.
            - "size": Image size (width, height) of camera in pixels of size (2,) and
                type int.
            - "matrix": Intrinsic camera matrix of size (3, 3) and type float64.
            - "distortions": Radial-tangential distortion coefficients
                [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
            - "rotation": Rotation vector in unnormalized axis-angle representation of
                size (3,) and type float64.
            - "translation": Translation vector of size (3,) and type float64.
            and optional keys containing metadata.

    Returns:
        `Camera` object created from dictionary.
    """
    # Avoid mutating the dictionary.
    camera_dict = camera_dict.copy()

    # Get all attributes we deserialize.
    name = camera_dict.pop("name")
    size = camera_dict.pop("size")
    camera = Camera(
        name=name if len(name) > 0 else None,
        size=size if len(size) > 0 else None,
        matrix=camera_dict.pop("matrix"),
        dist=camera_dict.pop("distortions"),
        rvec=camera_dict.pop("rotation"),
        tvec=camera_dict.pop("translation"),
    )

    # Add remaining metadata to `Camera`
    camera.metadata = camera_dict

    return camera


def make_camera_group(calibration_dict: dict) -> CameraGroup:
    """Create a `CameraGroup` from a calibration dictionary.

    Args:
        calibration_dict: Dictionary containing calibration information for cameras
            with optional keys:
            - "metadata": Dictionary containing metadata for the `CameraGroup`.
            - Arbitrary (but unique) keys for every `Camera`, each containing a
                dictionary with camera information (see `make_camera` for what each
                dictionary contains).

    Returns:
        `CameraGroup` object created from calibration dictionary.
    """
    cameras = []
    metadata = {}
    for dict_name, camera_dict in calibration_dict.items():
        if dict_name == "metadata":
            metadata = camera_dict
            continue
        camera = make_camera(camera_dict)
        cameras.append(camera)

    return CameraGroup(cameras=cameras, metadata=metadata)


def make_session(
    session_dict: dict,
    videos: list[Video],
    labeled_frames: list[LabeledFrame],
    identities: list[Identity] | None = None,
) -> RecordingSession:
    """Create a `RecordingSession` from a dictionary.

    Args:
        session_dict: Dictionary with keys:
            - "calibration": Dictionary containing calibration information for cameras.
            - "camcorder_to_video_idx_map": Dictionary mapping camera index to video
                index.
            - "frame_group_dicts": List of dictionaries containing `FrameGroup`
                information. See `make_frame_group` for what each dictionary contains.
            - Any optional keys containing metadata.
        videos: List containing `Video` objects (expected `Labels.videos`).
        labeled_frames: List containing `LabeledFrame` objects (expected
            `Labels.labeled_frames`).
        identities: Optional list of `Identity` objects for resolving identity
            indices.

    Returns:
        `RecordingSession` object.
    """
    # Avoid modifying original dictionary
    session_dict = session_dict.copy()

    # Restructure `RecordingSession` without `Video` to `Camera` mapping
    calibration_dict = session_dict.pop("calibration")
    camera_group = make_camera_group(calibration_dict)

    # Retrieve all `Camera` and `Video` objects, then add to `RecordingSession`
    camcorder_to_video_idx_map = session_dict.pop("camcorder_to_video_idx_map")
    video_by_camera = {}
    camera_by_video = {}
    for cam_idx, video_idx in camcorder_to_video_idx_map.items():
        camera = camera_group.cameras[int(cam_idx)]
        video = videos[int(video_idx)]
        video_by_camera[camera] = video
        camera_by_video[video] = camera

    # Reconstruct all `FrameGroup` objects and add to `RecordingSession`
    frame_group_dicts = []
    if "frame_group_dicts" in session_dict:
        frame_group_dicts = session_dict.pop("frame_group_dicts")
    frame_group_by_frame_idx = {}
    for frame_group_dict in frame_group_dicts:
        try:
            # Add `FrameGroup` to `RecordingSession`
            frame_group = make_frame_group(
                frame_group_dict=frame_group_dict,
                labeled_frames=labeled_frames,
                camera_group=camera_group,
                identities=identities,
            )
            frame_group_by_frame_idx[frame_group.frame_idx] = frame_group
        except ValueError as e:
            print(
                f"Error reconstructing FrameGroup: {frame_group_dict}. Skipping...\n{e}"
            )

    session = RecordingSession(
        camera_group=camera_group,
        video_by_camera=video_by_camera,
        camera_by_video=camera_by_video,
        frame_group_by_frame_idx=frame_group_by_frame_idx,
        metadata=session_dict,
    )

    return session


def read_sessions(
    labels_path: str,
    videos: list[Video],
    labeled_frames: list[LabeledFrame],
    identities: list[Identity] | None = None,
) -> list[RecordingSession]:
    """Read `RecordingSession` dataset from a SLEAP labels file.

    Expects a "sessions_json" dataset in the `labels_path` file, but will return an
    empty list if the dataset is not found.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects.
        labeled_frames: A list of `LabeledFrame` objects.
        identities: Optional list of `Identity` objects for resolving identity
            indices.

    Returns:
        A list of `RecordingSession` objects.
    """
    try:
        sessions = read_hdf5_dataset(labels_path, "sessions_json")
    except KeyError:
        return []
    sessions = [json.loads(x) for x in sessions]
    session_objects = []
    for session in sessions:
        session_objects.append(
            make_session(session, videos, labeled_frames, identities=identities)
        )
    return session_objects


def instance_group_to_dict(
    instance_group: InstanceGroup,
    instance_to_lf_and_inst_idx: dict[Instance, tuple[int, int]],
    camera_group: CameraGroup,
    identities: list[Identity] | None = None,
) -> dict:
    """Convert `instance_group` to a dictionary.

    Args:
        instance_group: `InstanceGroup` object to convert to a dictionary.
        instance_to_lf_and_inst_idx: Dictionary mapping `Instance` objects to
            `LabeledFrame` indices (in `Labels.labeled_frames`) and `Instance` indices
            (in containing `LabeledFrame.instances`).
        camera_group: `CameraGroup` object that determines the order of the `Camera`
            objects when converting to a dictionary.
        identities: Optional list of `Identity` objects for serializing identity
            indices.

    Returns:
        Dictionary of the `InstanceGroup` with keys:
            - "camcorder_to_lf_and_inst_idx_map": Dictionary mapping `Camera` indices
                (in `InstanceGroup.camera_cluster.cameras`) to a tuple of `LabeledFrame`
                and `Instance` indices (from `instance_to_lf_and_inst_idx`)
            - Any optional keys containing metadata.
    """
    camera_to_lf_and_inst_idx_map: dict[int, tuple[int, int]] = {
        camera_group.cameras.index(cam): instance_to_lf_and_inst_idx[instance]
        for cam, instance in instance_group.instance_by_camera.items()
    }

    # Only required key is camcorder_to_lf_and_inst_idx_map
    instance_group_dict = {
        "camcorder_to_lf_and_inst_idx_map": camera_to_lf_and_inst_idx_map,
    }

    # Optionally add score, points, and metadata if they are non-default values
    if instance_group.score is not None:
        instance_group_dict["score"] = instance_group.score

    # 3D points — serialize from Instance3D if present
    if instance_group.instance_3d is not None:
        inst3d = instance_group.instance_3d
        instance_group_dict["points"] = inst3d.points.tolist()
        if inst3d.score is not None:
            instance_group_dict["instance_3d_score"] = inst3d.score
        if isinstance(inst3d, PredictedInstance3D) and inst3d.point_scores is not None:
            instance_group_dict["instance_3d_point_scores"] = (
                inst3d.point_scores.tolist()
            )

    # Identity — serialize as index into Labels.identities
    if instance_group.identity is not None and identities is not None:
        try:
            identity_idx = identities.index(instance_group.identity)
            instance_group_dict["identity_idx"] = identity_idx
        except ValueError:
            warnings.warn(
                f"Identity '{instance_group.identity.name}' not found in "
                "Labels.identities; identity dropped during save."
            )

    instance_group_dict.update(instance_group.metadata)

    return instance_group_dict


def frame_group_to_dict(
    frame_group: FrameGroup,
    labeled_frame_to_idx: dict[LabeledFrame, int],
    camera_group: CameraGroup,
    identities: list[Identity] | None = None,
) -> dict:
    """Convert `frame_group` to a dictionary.

    Args:
        frame_group: `FrameGroup` object to convert to a dictionary.
        labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
            `Labels.labeled_frames`.
        camera_group: `CameraGroup` object that determines the order of the `Camera`
            objects when converting to a dictionary.
        identities: Optional list of `Identity` objects for serializing identity
            indices.

    Returns:
        Dictionary of the `FrameGroup` with keys:
            - "instance_groups": List of dictionaries for each `InstanceGroup` in the
                `FrameGroup`. See `instance_group_to_dict` for what each dictionary
                contains.
            - "frame_idx": Frame index for the `FrameGroup`.
            - Any optional keys containing metadata.
    """
    # Create dictionary of `Instance` to `LabeledFrame` index (in
    # `Labels.labeled_frames`) and `Instance` index in `LabeledFrame.instances`.
    instance_to_lf_and_inst_idx: dict[Instance, tuple[int, int]] = {
        inst: (labeled_frame_to_idx[labeled_frame], inst_idx)
        for labeled_frame in frame_group.labeled_frames
        for inst_idx, inst in enumerate(labeled_frame.instances)
    }

    frame_group_dict = {
        "instance_groups": [
            instance_group_to_dict(
                instance_group,
                instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx,
                camera_group=camera_group,
                identities=identities,
            )
            for instance_group in frame_group.instance_groups
        ],
    }
    frame_group_dict["frame_idx"] = frame_group.frame_idx
    frame_group_dict.update(frame_group.metadata)

    return frame_group_dict


def camera_to_dict(camera: Camera) -> dict:
    """Convert `camera` to dictionary.

    Args:
        camera: `Camera` object to convert to a dictionary.

    Returns:
        Dictionary containing camera information with the following keys:
            - "name": Camera name.
            - "size": Image size (width, height) of camera in pixels of size (2,) and
              type
                int.
            - "matrix": Intrinsic camera matrix of size (3, 3) and type float64.
            - "distortions": Radial-tangential distortion coefficients
                [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
            - "rotation": Rotation vector in unnormalized axis-angle representation of
                size (3,) and type float64.
            - "translation": Translation vector of size (3,) and type float64.
            - Any optional keys containing metadata.

    """
    # Handle optional attributes
    name = "" if camera.name is None else camera.name
    size = "" if camera.size is None else list(camera.size)

    camera_dict = {
        "name": name,
        "size": size,
        "matrix": camera.matrix.tolist(),
        "distortions": camera.dist.tolist(),
        "rotation": camera.rvec.tolist(),
        "translation": camera.tvec.tolist(),
    }
    camera_dict.update(camera.metadata)

    return camera_dict


def camera_group_to_dict(camera_group: CameraGroup) -> dict:
    """Convert `camera_group` to dictionary.

    Args:
        camera_group: `CameraGroup` object to convert to a dictionary.

    Returns:
        Dictionary containing camera group information with the following keys:
            - cam_n: Camera dictionary containing information for camera at index "n"
                with the following keys:
                name: Camera name.
                size: Image size (height, width) of camera in pixels of size (2,)
                    and type int.
                matrix: Intrinsic camera matrix of size (3, 3) and type float64.
                distortions: Radial-tangential distortion coefficients
                    [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
                rotation: Rotation vector in unnormalized axis-angle representation
                    of size (3,) and type float64.
                translation: Translation vector of size (3,) and type float64.
            - "metadata": Dictionary of optional metadata.
    """
    calibration_dict = {}
    for cam_idx, camera in enumerate(camera_group.cameras):
        camera_dict = camera_to_dict(camera)
        calibration_dict[f"cam_{cam_idx}"] = camera_dict

    calibration_dict["metadata"] = camera_group.metadata.copy()

    return calibration_dict


def session_to_dict(
    session: RecordingSession,
    video_to_idx: dict[Video, int],
    labeled_frame_to_idx: dict[LabeledFrame, int],
    identities: list[Identity] | None = None,
) -> dict:
    """Convert `RecordingSession` to a dictionary.

    Args:
        session: `RecordingSession` object to convert to a dictionary.
        video_to_idx: Dictionary of `Video` to index in `Labels.videos`.
        labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
            `Labels.labeled_frames`.
        identities: Optional list of `Identity` objects for serializing identity
            indices.

    Returns:
        Dictionary of `RecordingSession` with the following keys:
            - "calibration": Dictionary containing calibration information for cameras.
            - "camcorder_to_video_idx_map": Dictionary mapping camera index to video
                index.
            - "frame_group_dicts": List of dictionaries containing `FrameGroup`
                information. See `frame_group_to_dict` for what each dictionary
                contains.
            - Any optional keys containing metadata.
    """
    # Unstructure `CameraCluster` and `metadata`
    calibration_dict = camera_group_to_dict(session.camera_group)

    # Store camera-to-video indices map where key is camera index
    # and value is video index from `Labels.videos`
    camera_to_video_idx_map = {}
    for cam_idx, camera in enumerate(session.camera_group.cameras):
        # Skip if Camera is not linked to any Video

        if camera not in session.cameras:
            continue

        # Get video index from `Labels.videos`
        video = session.get_video(camera)
        video_idx = video_to_idx.get(video, None)

        if video_idx is not None:
            camera_to_video_idx_map[cam_idx] = video_idx
        else:
            print(
                f"Video {video} not found in `Labels.videos`. "
                "Not saving to `RecordingSession` serialization."
            )

    # Store frame groups by frame index
    frame_group_dicts = []
    if len(labeled_frame_to_idx) > 0:  # Don't save if skipping labeled frames
        for frame_group in session.frame_groups.values():
            # Only save `FrameGroup` if it has `InstanceGroup`s
            if len(frame_group.instance_groups) > 0:
                frame_group_dict = frame_group_to_dict(
                    frame_group,
                    labeled_frame_to_idx=labeled_frame_to_idx,
                    camera_group=session.camera_group,
                    identities=identities,
                )
                frame_group_dicts.append(frame_group_dict)

    session_dict = {
        "calibration": calibration_dict,
        "camcorder_to_video_idx_map": camera_to_video_idx_map,
        "frame_group_dicts": frame_group_dicts,
    }
    session_dict.update(session.metadata)

    return session_dict


def write_sessions(
    labels_path: str,
    sessions: list[RecordingSession],
    videos: list[Video],
    labeled_frames: list[LabeledFrame],
    identities: list[Identity] | None = None,
):
    """Write `RecordingSession` metadata to a SLEAP labels file.

    Creates a new dataset "sessions_json" in the `labels_path` file to store the
    sessions data.

    Args:
        labels_path: A string path to the SLEAP labels file.
        sessions: A list of `RecordingSession` objects to store in the `labels_path`
            file.
        videos: A list of `Video` objects referenced in the `RecordingSession`s
            (expecting `Labels.videos`).
        labeled_frames: A list of `LabeledFrame` objects referenced in the
            `RecordingSession`s (expecting `Labels.labeled_frames`).
        identities: Optional list of `Identity` objects for serializing identity
            indices.
    """
    sessions_json = []
    if len(sessions) > 0:
        labeled_frame_to_idx = {lf: i for i, lf in enumerate(labeled_frames)}
        video_to_idx = {video: i for i, video in enumerate(videos)}
    for session in sessions:
        session_json = session_to_dict(
            session=session,
            video_to_idx=video_to_idx,
            labeled_frame_to_idx=labeled_frame_to_idx,
            identities=identities,
        )
        sessions_json.append(np.bytes_(json.dumps(session_json, separators=(",", ":"))))

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("sessions_json", data=sessions_json, maxshape=(None,))


def _read_labels_lazy(labels_path: str, open_videos: bool = True) -> Labels:
    """Read SLP file with lazy loading.

    This function reads raw HDF5 arrays into memory but defers creation of
    LabeledFrame and Instance objects until they are accessed.

    Args:
        labels_path: Path to .slp file.
        open_videos: Whether to open video backends.

    Returns:
        Labels with LazyFrameList for labeled_frames.
    """
    from sleap_io.io.slp_lazy import LazyDataStore, LazyFrameList

    # Read raw arrays
    frames_data = read_hdf5_dataset(labels_path, "frames")
    instances_data = read_hdf5_dataset(labels_path, "instances")
    points_data = read_points(labels_path)
    pred_points_data = read_pred_points(labels_path)

    # Read format ID
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")

    # Read metadata eagerly (these are small and needed for lazy access)
    videos = read_videos(labels_path, open_backend=open_videos)
    skeletons = read_skeletons(labels_path)
    tracks = read_tracks(labels_path)
    suggestions = read_suggestions(labels_path, videos)
    metadata = read_metadata(labels_path)
    provenance = metadata.get("provenance", dict())
    negative_frames = read_negative_frames(labels_path)

    # Read sessions (small, no need for lazy loading)
    # Note: sessions require labeled_frames for full linking, but for lazy loading
    # we pass an empty list since we don't have materialized frames yet
    sessions = read_sessions(labels_path, videos, [])

    # Create LazyDataStore
    lazy_store = LazyDataStore(
        frames_data=frames_data,
        instances_data=instances_data,
        pred_points_data=pred_points_data,
        points_data=points_data,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        format_id=format_id,
        source_path=str(labels_path),
        negative_frames=negative_frames,
    )

    # Create LazyFrameList
    lazy_frames = LazyFrameList(lazy_store)

    # Read ROIs, masks, bboxes, and label images eagerly (typically small count)
    rois = read_rois(labels_path, videos, tracks)
    masks = read_masks(labels_path, videos, tracks)
    bboxes = read_bboxes(labels_path, videos, tracks)
    label_images = read_label_images(labels_path, videos, tracks)

    # Create Labels with lazy state
    labels = Labels(
        labeled_frames=lazy_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        suggestions=suggestions,
        sessions=sessions,
        provenance=provenance,
        rois=rois,
        masks=masks,
        bboxes=bboxes,
        label_images=label_images,
        lazy_store=lazy_store,
    )
    labels.provenance["filename"] = labels_path

    # Store the HDF5 file handle for lazy label image data (keeps it alive)
    li_file = getattr(read_label_images, "_open_file", None)
    if li_file is not None:
        labels._label_image_file = li_file
        read_label_images._open_file = None  # type: ignore[attr-defined]

    return labels


def read_rois(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> list[ROI]:
    """Read ROI annotations from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking ROI instance associations. If ``None``, instance
            associations will not be restored.

    Returns:
        A list of ROI objects. Returns an empty list if no ROIs are stored.
    """
    import shapely

    try:
        roi_data = read_hdf5_dataset(labels_path, "rois")
    except KeyError:
        return []

    if len(roi_data) == 0:
        return []

    # Read packed WKB geometry bytes
    try:
        roi_wkb_flat = read_hdf5_dataset(labels_path, "roi_wkb")
    except KeyError:
        return []

    # Read string metadata from string datasets first, fall back to JSON attributes
    with h5py.File(labels_path, "r") as f:
        roi_grp = f["rois"]
        if "roi_categories" in f:
            categories = [
                s.decode() if isinstance(s, bytes) else s
                for s in f["roi_categories"][:]
            ]
        else:
            categories = json.loads(roi_grp.attrs.get("categories", "[]"))
        if "roi_names" in f:
            names = [
                s.decode() if isinstance(s, bytes) else s for s in f["roi_names"][:]
            ]
        else:
            names = json.loads(roi_grp.attrs.get("names", "[]"))
        if "roi_sources" in f:
            sources = [
                s.decode() if isinstance(s, bytes) else s for s in f["roi_sources"][:]
            ]
        else:
            sources = json.loads(roi_grp.attrs.get("sources", "[]"))

    rois = []
    for i, row in enumerate(roi_data):
        wkb_start = int(row["wkb_start"])
        wkb_end = int(row["wkb_end"])
        wkb_bytes = bytes(roi_wkb_flat[wkb_start:wkb_end])
        geometry = shapely.from_wkb(wkb_bytes)

        video_idx = int(row["video"])
        video = videos[video_idx] if 0 <= video_idx < len(videos) else None

        frame_idx_val = int(row["frame_idx"])
        frame_idx = None if frame_idx_val == -1 else frame_idx_val

        track_idx = int(row["track"])
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None

        instance_idx = int(row["instance"]) if "instance" in row.dtype.names else -1
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )

        # Read predicted flag (v1.9+)
        is_predicted = (
            bool(row["is_predicted"]) if "is_predicted" in row.dtype.names else False
        )

        kwargs = dict(
            geometry=geometry,
            name=names[i] if i < len(names) else "",
            category=categories[i] if i < len(categories) else "",
            source=sources[i] if i < len(sources) else "",
            video=video,
            frame_idx=frame_idx,
            track=track,
            instance=instance,
        )

        if is_predicted:
            score_val = float(row["score"]) if "score" in row.dtype.names else 0.0
            roi = PredictedROI(
                score=score_val if not np.isnan(score_val) else 0.0, **kwargs
            )
        else:
            roi = UserROI(**kwargs)

        # Store raw index for deferred resolution (lazy loading)
        roi._instance_idx = instance_idx
        rois.append(roi)

    return rois


def write_rois(
    labels_path: str,
    rois: list[ROI],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> None:
    """Write ROI annotations to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        rois: A list of ROI objects to write.
        videos: List of Video objects for index mapping.
        tracks: List of Track objects for index mapping.
        instances: Optional list of Instance/PredictedInstance objects for index
            mapping. If provided, ROI instance associations will be persisted.
    """
    if not rois:
        return

    import shapely

    roi_dtype = np.dtype(
        [
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("is_predicted", "u1"),
            ("score", "f4"),
            ("wkb_start", "u8"),
            ("wkb_end", "u8"),
            ("instance", "i4"),
        ]
    )

    roi_rows = []
    wkb_chunks = []
    wkb_offset = 0
    categories = []
    names = []
    sources = []

    for roi in rois:
        wkb = shapely.to_wkb(roi.geometry)
        wkb_start = wkb_offset
        wkb_end = wkb_offset + len(wkb)
        wkb_chunks.append(np.frombuffer(wkb, dtype=np.uint8))
        wkb_offset = wkb_end

        video_idx = videos.index(roi.video) if roi.video in videos else -1
        frame_idx = roi.frame_idx if roi.frame_idx is not None else -1
        track_idx = tracks.index(roi.track) if roi.track in tracks else -1

        instance_idx = roi._instance_idx  # Use stored index as default
        if instances is not None and roi.instance is not None:
            try:
                instance_idx = instances.index(roi.instance)
            except ValueError:
                pass  # Keep stored _instance_idx

        is_predicted = isinstance(roi, PredictedROI)
        score = roi.score if is_predicted else float("nan")

        roi_rows.append(
            (
                0,  # annotation_type: write 0 (DEFAULT) for backward compat
                video_idx,
                frame_idx,
                track_idx,
                int(is_predicted),
                score,
                wkb_start,
                wkb_end,
                instance_idx,
            )
        )

        categories.append(roi.category)
        names.append(roi.name)
        sources.append(roi.source)

    roi_array = np.array(roi_rows, dtype=roi_dtype)
    wkb_flat = (
        np.concatenate(wkb_chunks) if wkb_chunks else np.array([], dtype=np.uint8)
    )

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("rois", data=roi_array, dtype=roi_dtype)
        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("roi_categories", data=categories, dtype=str_dt)
        f.create_dataset("roi_names", data=names, dtype=str_dt)
        f.create_dataset("roi_sources", data=sources, dtype=str_dt)
        f.create_dataset(
            "roi_wkb",
            data=wkb_flat,
            dtype=np.uint8,
            **({"chunks": True} if len(wkb_flat) > 0 else {}),
        )


def read_bboxes(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> list[BoundingBox]:
    """Read bounding box annotations from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking bounding box instance associations. If ``None``, instance
            associations will not be restored.

    Returns:
        A list of BoundingBox objects. Returns an empty list if no bboxes are
        stored.
    """
    with h5py.File(labels_path, "r") as f:
        if "bboxes" not in f:
            return []

        node = f["bboxes"]
        if isinstance(node, h5py.Group):
            return _read_bboxes_columnar(node, videos, tracks, instances)
        else:
            # Read data and attrs in one open, pass to legacy reader
            bbox_data = node[:]
            categories = json.loads(node.attrs.get("categories", "[]"))
            names = json.loads(node.attrs.get("names", "[]"))
            sources = json.loads(node.attrs.get("sources", "[]"))
            return _read_bboxes_legacy(
                bbox_data, categories, names, sources, videos, tracks, instances
            )


def _read_bboxes_columnar(
    grp: "h5py.Group",
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None,
) -> list[BoundingBox]:
    """Read bboxes from columnar /bboxes group (v2.0+ format)."""
    x1_arr = grp["x1"][:]
    y1_arr = grp["y1"][:]
    x2_arr = grp["x2"][:]
    y2_arr = grp["y2"][:]
    angle_arr = grp["angle"][:]
    video_arr = grp["video"][:]
    frame_idx_arr = grp["frame_idx"][:]
    track_arr = grp["track"][:]
    instance_arr = grp["instance"][:]
    is_predicted_arr = grp["is_predicted"][:]
    score_arr = grp["score"][:]
    category_arr = grp["category"][:]
    name_arr = grp["name"][:]
    source_arr = grp["source"][:]

    bboxes: list[BoundingBox] = []
    for i in range(len(x1_arr)):
        video_idx = int(video_arr[i])
        video = videos[video_idx] if 0 <= video_idx < len(videos) else None

        frame_idx_val = int(frame_idx_arr[i])
        frame_idx = None if frame_idx_val == -1 else frame_idx_val

        track_idx = int(track_arr[i])
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None

        instance_idx = int(instance_arr[i])
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )

        cat = category_arr[i]
        category = cat.decode() if isinstance(cat, bytes) else str(cat)
        nm = name_arr[i]
        name = nm.decode() if isinstance(nm, bytes) else str(nm)
        src = source_arr[i]
        source = src.decode() if isinstance(src, bytes) else str(src)

        kwargs = dict(
            x1=float(x1_arr[i]),
            y1=float(y1_arr[i]),
            x2=float(x2_arr[i]),
            y2=float(y2_arr[i]),
            angle=float(angle_arr[i]),
            video=video,
            frame_idx=frame_idx,
            track=track,
            instance=instance,
            category=category,
            name=name,
            source=source,
        )

        if bool(is_predicted_arr[i]):
            bbox = PredictedBoundingBox(score=float(score_arr[i]), **kwargs)
        else:
            bbox = UserBoundingBox(**kwargs)

        bbox._instance_idx = instance_idx
        bboxes.append(bbox)

    return bboxes


def _read_bboxes_legacy(
    bbox_data: np.ndarray,
    categories: list[str],
    names: list[str],
    sources: list[str],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None,
) -> list[BoundingBox]:
    """Read bboxes from legacy structured array format (pre-v2.0)."""
    if len(bbox_data) == 0:
        return []

    bboxes: list[BoundingBox] = []
    for i, row in enumerate(bbox_data):
        video_idx = int(row["video"])
        video = videos[video_idx] if 0 <= video_idx < len(videos) else None

        frame_idx_val = int(row["frame_idx"])
        frame_idx = None if frame_idx_val == -1 else frame_idx_val

        track_idx = int(row["track"])
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None

        instance_idx = int(row["instance"])
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )

        # Legacy format uses x_center/y_center/width/height -> convert to x1y1x2y2
        xc = float(row["x_center"])
        yc = float(row["y_center"])
        w = float(row["width"])
        h = float(row["height"])

        kwargs = dict(
            x1=xc - w / 2,
            y1=yc - h / 2,
            x2=xc + w / 2,
            y2=yc + h / 2,
            angle=float(row["angle"]),
            video=video,
            frame_idx=frame_idx,
            track=track,
            instance=instance,
            category=categories[i] if i < len(categories) else "",
            name=names[i] if i < len(names) else "",
            source=sources[i] if i < len(sources) else "",
        )

        is_predicted = bool(row["is_predicted"])
        if is_predicted:
            bbox = PredictedBoundingBox(score=float(row["score"]), **kwargs)
        else:
            bbox = UserBoundingBox(**kwargs)

        bbox._instance_idx = instance_idx
        bboxes.append(bbox)

    return bboxes


def write_bboxes(
    labels_path: str,
    bboxes: list[BoundingBox],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> None:
    """Write bounding box annotations to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        bboxes: A list of BoundingBox objects to write.
        videos: List of Video objects for index mapping.
        tracks: List of Track objects for index mapping.
        instances: Optional list of Instance/PredictedInstance objects for index
            mapping. If provided, bounding box instance associations will be
            persisted.
    """
    if not bboxes:
        return

    n = len(bboxes)
    x1_arr = np.empty(n, dtype=np.float64)
    y1_arr = np.empty(n, dtype=np.float64)
    x2_arr = np.empty(n, dtype=np.float64)
    y2_arr = np.empty(n, dtype=np.float64)
    angle_arr = np.empty(n, dtype=np.float64)
    video_arr = np.empty(n, dtype=np.int32)
    frame_idx_arr = np.empty(n, dtype=np.int64)
    track_arr = np.empty(n, dtype=np.int32)
    instance_arr = np.empty(n, dtype=np.int32)
    is_predicted_arr = np.empty(n, dtype=np.uint8)
    score_arr = np.empty(n, dtype=np.float32)
    categories = []
    names = []
    sources = []

    for i, bbox in enumerate(bboxes):
        x1_arr[i] = bbox.x1
        y1_arr[i] = bbox.y1
        x2_arr[i] = bbox.x2
        y2_arr[i] = bbox.y2
        angle_arr[i] = bbox.angle

        video_arr[i] = videos.index(bbox.video) if bbox.video in videos else -1
        frame_idx_arr[i] = bbox.frame_idx if bbox.frame_idx is not None else -1
        track_arr[i] = tracks.index(bbox.track) if bbox.track in tracks else -1

        instance_idx = bbox._instance_idx
        if instances is not None and bbox.instance is not None:
            try:
                instance_idx = instances.index(bbox.instance)
            except ValueError:
                pass
        instance_arr[i] = instance_idx

        is_predicted = isinstance(bbox, PredictedBoundingBox)
        is_predicted_arr[i] = int(is_predicted)
        score_arr[i] = bbox.score if is_predicted else float("nan")

        categories.append(bbox.category)
        names.append(bbox.name)
        sources.append(bbox.source)

    str_dt = h5py.special_dtype(vlen=str)
    with h5py.File(labels_path, "a") as f:
        grp = f.create_group("bboxes")
        grp.create_dataset("x1", data=x1_arr)
        grp.create_dataset("y1", data=y1_arr)
        grp.create_dataset("x2", data=x2_arr)
        grp.create_dataset("y2", data=y2_arr)
        grp.create_dataset("angle", data=angle_arr)
        grp.create_dataset("video", data=video_arr)
        grp.create_dataset("frame_idx", data=frame_idx_arr)
        grp.create_dataset("track", data=track_arr)
        grp.create_dataset("instance", data=instance_arr)
        grp.create_dataset("is_predicted", data=is_predicted_arr)
        grp.create_dataset("score", data=score_arr)
        grp.create_dataset("category", data=categories, dtype=str_dt)
        grp.create_dataset("name", data=names, dtype=str_dt)
        grp.create_dataset("source", data=sources, dtype=str_dt)


def read_masks(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> list[SegmentationMask]:
    """Read segmentation masks from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking mask instance associations. If ``None``, instance
            associations will not be restored.

    Returns:
        A list of SegmentationMask objects. Returns empty list if none stored.
    """
    try:
        mask_data = read_hdf5_dataset(labels_path, "masks")
    except KeyError:
        return []

    if len(mask_data) == 0:
        return []

    # Read packed RLE bytes
    try:
        mask_rle_flat = read_hdf5_dataset(labels_path, "mask_rle")
    except KeyError:
        return []

    # Read string metadata and score maps in a single file open
    score_map_by_idx: dict[int, np.ndarray] = {}
    with h5py.File(labels_path, "r") as f:
        mask_grp = f["masks"]
        if "mask_categories" in f:
            categories = [
                s.decode() if isinstance(s, bytes) else s
                for s in f["mask_categories"][:]
            ]
        else:
            categories = json.loads(mask_grp.attrs.get("categories", "[]"))
        if "mask_names" in f:
            names = [
                s.decode() if isinstance(s, bytes) else s for s in f["mask_names"][:]
            ]
        else:
            names = json.loads(mask_grp.attrs.get("names", "[]"))
        if "mask_sources" in f:
            sources = [
                s.decode() if isinstance(s, bytes) else s for s in f["mask_sources"][:]
            ]
        else:
            sources = json.loads(mask_grp.attrs.get("sources", "[]"))

        # Read and index score maps if available
        score_map_spatial_by_idx: dict[
            int, tuple[tuple[float, float], tuple[float, float]]
        ] = {}
        if "mask_score_map_index" in f:
            sm_index = f["mask_score_map_index"][:]
            sm_data = f["mask_score_maps"][:]
            for sm_row in sm_index:
                sm_start = int(sm_row["data_start"])
                sm_end = int(sm_row["data_end"])
                sm_h = int(sm_row["height"])
                sm_w = int(sm_row["width"])
                sm_compressed = sm_data[sm_start:sm_end]
                midx = int(sm_row["mask_idx"])
                score_map_by_idx[midx] = np.frombuffer(
                    zlib.decompress(sm_compressed.tobytes()), dtype=np.float32
                ).reshape(sm_h, sm_w)
                # Read score map spatial metadata (v2.1+)
                sm_scale = (
                    float(sm_row["scale_x"])
                    if "scale_x" in sm_row.dtype.names
                    else 1.0,
                    float(sm_row["scale_y"])
                    if "scale_y" in sm_row.dtype.names
                    else 1.0,
                )
                sm_offset = (
                    float(sm_row["offset_x"])
                    if "offset_x" in sm_row.dtype.names
                    else 0.0,
                    float(sm_row["offset_y"])
                    if "offset_y" in sm_row.dtype.names
                    else 0.0,
                )
                score_map_spatial_by_idx[midx] = (sm_scale, sm_offset)

    masks = []
    for i, row in enumerate(mask_data):
        rle_start = int(row["rle_start"])
        rle_end = int(row["rle_end"])
        rle_raw = mask_rle_flat[rle_start:rle_end]

        # Convert packed uint8 bytes back to uint32 array
        rle_counts = np.frombuffer(rle_raw.tobytes(), dtype=np.uint32)

        video_idx = int(row["video"])
        video = videos[video_idx] if 0 <= video_idx < len(videos) else None

        frame_idx_val = int(row["frame_idx"])
        frame_idx = None if frame_idx_val == -1 else frame_idx_val

        track_idx = int(row["track"])
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None

        # Read instance index (v1.9+)
        instance_idx = int(row["instance"]) if "instance" in row.dtype.names else -1
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )

        # Read predicted flag (v1.9+)
        is_predicted = (
            bool(row["is_predicted"]) if "is_predicted" in row.dtype.names else False
        )
        score_val = float(row["score"]) if "score" in row.dtype.names else float("nan")

        # Read spatial metadata (v2.1+)
        scale = (
            float(row["scale_x"]) if "scale_x" in row.dtype.names else 1.0,
            float(row["scale_y"]) if "scale_y" in row.dtype.names else 1.0,
        )
        offset = (
            float(row["offset_x"]) if "offset_x" in row.dtype.names else 0.0,
            float(row["offset_y"]) if "offset_y" in row.dtype.names else 0.0,
        )

        kwargs = dict(
            rle_counts=rle_counts,
            height=int(row["height"]),
            width=int(row["width"]),
            name=names[i] if i < len(names) else "",
            category=categories[i] if i < len(categories) else "",
            source=sources[i] if i < len(sources) else "",
            video=video,
            frame_idx=frame_idx,
            track=track,
            instance=instance,
            scale=scale,
            offset=offset,
        )

        if is_predicted:
            # Check for score map
            sm = score_map_by_idx.get(i)
            sm_spatial = score_map_spatial_by_idx.get(i)
            sm_scale = sm_spatial[0] if sm_spatial else (1.0, 1.0)
            sm_offset = sm_spatial[1] if sm_spatial else (0.0, 0.0)
            mask = PredictedSegmentationMask(
                score=score_val if not np.isnan(score_val) else 0.0,
                score_map=sm,
                score_map_scale=sm_scale,
                score_map_offset=sm_offset,
                **kwargs,
            )
        else:
            mask = UserSegmentationMask(**kwargs)

        mask._instance_idx = instance_idx
        masks.append(mask)

    return masks


def write_masks(
    labels_path: str,
    masks: list[SegmentationMask],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> None:
    """Write segmentation masks to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        masks: A list of SegmentationMask objects to write.
        videos: List of Video objects for index mapping.
        tracks: List of Track objects for index mapping.
        instances: Optional list of Instance/PredictedInstance objects for index
            mapping. If provided, mask instance associations will be persisted.
    """
    if not masks:
        return

    mask_dtype = np.dtype(
        [
            ("height", "u4"),
            ("width", "u4"),
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("instance", "i4"),
            ("is_predicted", "u1"),
            ("score", "f4"),
            ("rle_start", "u8"),
            ("rle_end", "u8"),
            ("scale_x", "f4"),
            ("scale_y", "f4"),
            ("offset_x", "f4"),
            ("offset_y", "f4"),
        ]
    )

    mask_rows = []
    rle_chunks = []
    rle_offset = 0
    categories = []
    names = []
    sources = []

    for mask in masks:
        # Pack uint32 RLE counts as raw bytes (uint8)
        rle_bytes = mask.rle_counts.astype(np.uint32).tobytes()
        rle_uint8 = np.frombuffer(rle_bytes, dtype=np.uint8)
        rle_start = rle_offset
        rle_end = rle_offset + len(rle_uint8)
        rle_chunks.append(rle_uint8)
        rle_offset = rle_end

        video_idx = videos.index(mask.video) if mask.video in videos else -1
        frame_idx = mask.frame_idx if mask.frame_idx is not None else -1
        track_idx = tracks.index(mask.track) if mask.track in tracks else -1

        instance_idx = mask._instance_idx  # Use stored index as default
        if instances is not None and mask.instance is not None:
            try:
                instance_idx = instances.index(mask.instance)
            except ValueError:
                pass  # Keep stored _instance_idx

        is_predicted = isinstance(mask, PredictedSegmentationMask)
        score = mask.score if is_predicted else float("nan")

        mask_rows.append(
            (
                mask.height,
                mask.width,
                2,  # annotation_type: write SEGMENTATION (2) for backward compat
                video_idx,
                frame_idx,
                track_idx,
                instance_idx,
                int(is_predicted),
                score,
                rle_start,
                rle_end,
                mask.scale[0],
                mask.scale[1],
                mask.offset[0],
                mask.offset[1],
            )
        )

        categories.append(mask.category)
        names.append(mask.name)
        sources.append(mask.source)

    mask_array = np.array(mask_rows, dtype=mask_dtype)
    rle_flat = (
        np.concatenate(rle_chunks) if rle_chunks else np.array([], dtype=np.uint8)
    )

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("masks", data=mask_array, dtype=mask_dtype)
        f.create_dataset(
            "mask_rle",
            data=rle_flat,
            dtype=np.uint8,
            **({"chunks": True} if len(rle_flat) > 0 else {}),
        )
        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("mask_categories", data=categories, dtype=str_dt)
        f.create_dataset("mask_names", data=names, dtype=str_dt)
        f.create_dataset("mask_sources", data=sources, dtype=str_dt)

    # Store dense score maps if any exist
    score_map_indices = []
    score_map_chunks = []
    score_map_offset = 0
    for i, mask in enumerate(masks):
        if isinstance(mask, PredictedSegmentationMask) and mask.score_map is not None:
            compressed = zlib.compress(mask.score_map.astype(np.float32).tobytes())
            sm_bytes = np.frombuffer(compressed, dtype=np.uint8)
            sm_end = score_map_offset + len(sm_bytes)
            sm_h, sm_w = mask.score_map.shape[:2]
            score_map_indices.append(
                (
                    i,
                    score_map_offset,
                    sm_end,
                    sm_h,
                    sm_w,
                    mask.score_map_scale[0],
                    mask.score_map_scale[1],
                    mask.score_map_offset[0],
                    mask.score_map_offset[1],
                )
            )
            score_map_chunks.append(sm_bytes)
            score_map_offset += len(sm_bytes)

    if score_map_indices:
        sm_index_dtype = np.dtype(
            [
                ("mask_idx", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("height", "u4"),
                ("width", "u4"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ]
        )
        sm_index_array = np.array(score_map_indices, dtype=sm_index_dtype)
        sm_flat = np.concatenate(score_map_chunks)
        with h5py.File(labels_path, "a") as f:
            f.create_dataset("mask_score_map_index", data=sm_index_array)
            f.create_dataset(
                "mask_score_maps",
                data=sm_flat,
                dtype=np.uint8,
                **({"chunks": True} if len(sm_flat) > 0 else {}),
            )


def read_label_images(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> list[LabelImage]:
    """Read label image annotations from a SLEAP labels file.

    Supports both the legacy blob format (v1.8-v2.1) and the chunked format
    (v2.2+). Pixel data is loaded lazily when possible: the HDF5 file handle
    is kept open and each frame's data is decompressed on first ``.data``
    access. The file handle is stored in a ``_label_image_file`` attribute on
    this function (caller is responsible for closing it or letting GC handle
    it via the ``Labels`` container).

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking label image instance associations. If ``None``, instance
            associations will not be restored.

    Returns:
        A list of LabelImage objects. Returns an empty list if none stored.
    """
    try:
        li_data = read_hdf5_dataset(labels_path, "label_images")
    except KeyError:
        return []

    if len(li_data) == 0:
        return []

    # Open file for lazy pixel data reading
    f = h5py.File(labels_path, "r")

    if "label_image_data" not in f:
        f.close()
        return []

    pixel_ds = f["label_image_data"]
    is_chunked_format = pixel_ds.ndim == 3  # (T, H, W) = v2.2+ chunked

    # Read objects table
    try:
        obj_data = read_hdf5_dataset(labels_path, "label_image_objects")
    except KeyError:
        obj_dtype = [
            ("label_id", "i4"),
            ("track", "i4"),
            ("instance", "i4"),
        ]
        obj_data = np.array([], dtype=obj_dtype)

    # Read string metadata and score maps
    if "label_image_obj_categories" in f:
        categories = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["label_image_obj_categories"][:]
        ]
    elif "label_image_objects" in f:
        obj_grp = f["label_image_objects"]
        categories = json.loads(obj_grp.attrs.get("categories", "[]"))
    else:
        categories = []

    if "label_image_obj_names" in f:
        names = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["label_image_obj_names"][:]
        ]
    elif "label_image_objects" in f:
        obj_grp = f["label_image_objects"]
        names = json.loads(obj_grp.attrs.get("names", "[]"))
    else:
        names = []

    if "label_image_sources" in f:
        sources = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["label_image_sources"][:]
        ]
    elif "label_images" in f:
        li_grp = f["label_images"]
        sources = json.loads(li_grp.attrs.get("sources", "[]"))
    else:
        sources = []

    # Read and index score maps if available
    li_score_map_by_idx: dict[int, np.ndarray] = {}
    li_score_map_spatial_by_idx: dict[
        int, tuple[tuple[float, float], tuple[float, float]]
    ] = {}
    if "label_image_score_map_index" in f:
        sm_index = f["label_image_score_map_index"][:]
        sm_data_ds = f["label_image_score_maps"]
        for sm_row in sm_index:
            sm_start = int(sm_row["data_start"])
            sm_end = int(sm_row["data_end"])
            sm_h = int(sm_row["height"])
            sm_w = int(sm_row["width"])
            sm_compressed = sm_data_ds[sm_start:sm_end]
            lidx = int(sm_row["li_idx"])
            li_score_map_by_idx[lidx] = np.frombuffer(
                zlib.decompress(sm_compressed.tobytes()), dtype=np.float32
            ).reshape(sm_h, sm_w)
            # Read score map spatial metadata (v2.1+)
            sm_scale = (
                float(sm_row["scale_x"]) if "scale_x" in sm_row.dtype.names else 1.0,
                float(sm_row["scale_y"]) if "scale_y" in sm_row.dtype.names else 1.0,
            )
            sm_offset = (
                float(sm_row["offset_x"]) if "offset_x" in sm_row.dtype.names else 0.0,
                float(sm_row["offset_y"]) if "offset_y" in sm_row.dtype.names else 0.0,
            )
            li_score_map_spatial_by_idx[lidx] = (sm_scale, sm_offset)

    # Factory functions for lazy loaders (avoid closure-over-loop-variable)
    def _make_chunked_loader(ds, idx):
        def loader():
            return ds[idx].copy()

        return loader

    def _make_blob_loader(ds, start, end, h, w):
        def loader():
            raw = zlib.decompress(ds[start:end].tobytes())
            return np.frombuffer(raw, dtype=np.int32).reshape(h, w).copy()

        return loader

    label_images: list[LabelImage] = []
    for i, row in enumerate(li_data):
        video_idx = int(row["video"])
        video = videos[video_idx] if 0 <= video_idx < len(videos) else None

        frame_idx_val = int(row["frame_idx"])
        frame_idx = None if frame_idx_val == -1 else frame_idx_val

        height = int(row["height"])
        width = int(row["width"])
        n_objects = int(row["n_objects"])
        objects_start = int(row["objects_start"])

        # Build objects dict from objects table
        objects: dict[int, LabelImage.Info] = {}
        for j in range(n_objects):
            obj_idx = objects_start + j
            if obj_idx < len(obj_data):
                obj_row = obj_data[obj_idx]
                label_id = int(obj_row["label_id"])

                track_idx = int(obj_row["track"])
                track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None

                instance_idx = int(obj_row["instance"])
                instance = (
                    instances[instance_idx]
                    if instances is not None and 0 <= instance_idx < len(instances)
                    else None
                )

                category = categories[obj_idx] if obj_idx < len(categories) else ""
                name = names[obj_idx] if obj_idx < len(names) else ""

                # Read per-object score if present (v1.9+)
                obj_score = None
                if "score" in obj_row.dtype.names:
                    sv = float(obj_row["score"])
                    if not np.isnan(sv):
                        obj_score = sv

                objects[label_id] = LabelImage.Info(
                    track=track,
                    category=category,
                    name=name,
                    instance=instance,
                    score=obj_score,
                )
                # Store raw index for deferred resolution (lazy loading)
                objects[label_id]._instance_idx = instance_idx

        source = sources[i] if i < len(sources) else ""

        # Read predicted flag (v1.9+)
        is_predicted = (
            bool(row["is_predicted"]) if "is_predicted" in row.dtype.names else False
        )
        score_val = float(row["score"]) if "score" in row.dtype.names else 0.0

        # Read spatial metadata (v2.1+)
        scale = (
            float(row["scale_x"]) if "scale_x" in row.dtype.names else 1.0,
            float(row["scale_y"]) if "scale_y" in row.dtype.names else 1.0,
        )
        offset = (
            float(row["offset_x"]) if "offset_x" in row.dtype.names else 0.0,
            float(row["offset_y"]) if "offset_y" in row.dtype.names else 0.0,
        )

        # Construct with data=None for lazy loading
        kwargs = dict(
            data=None,
            objects=objects,
            video=video,
            frame_idx=frame_idx,
            source=source,
            scale=scale,
            offset=offset,
        )

        if is_predicted:
            # Check for score map
            sm = li_score_map_by_idx.get(i)
            sm_spatial = li_score_map_spatial_by_idx.get(i)
            sm_scale = sm_spatial[0] if sm_spatial else (1.0, 1.0)
            sm_offset = sm_spatial[1] if sm_spatial else (0.0, 0.0)
            li = PredictedLabelImage(
                score=score_val if not np.isnan(score_val) else 0.0,
                score_map=sm,
                score_map_scale=sm_scale,
                score_map_offset=sm_offset,
                **kwargs,
            )
        else:
            li = UserLabelImage(**kwargs)

        # Set lazy loader for pixel data (decompresses on first .data access)
        if is_chunked_format:
            li._lazy_loader = _make_chunked_loader(pixel_ds, i)
        else:
            data_start = int(row["data_start"])
            data_end = int(row["data_end"])
            li._lazy_loader = _make_blob_loader(
                pixel_ds, data_start, data_end, height, width
            )
        li._height = height
        li._width = width

        label_images.append(li)

    # Store the open file handle so it stays alive for lazy loaders.
    # The caller (read_labels / Labels) should store this reference.
    read_label_images._open_file = f  # type: ignore[attr-defined]

    return label_images


def write_label_images(
    labels_path: str,
    label_images: list[LabelImage],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
) -> None:
    """Write label image annotations to a SLEAP labels file.

    When all label images share the same ``(height, width)``, pixel data is
    written as a chunked ``(T, H, W)`` int32 dataset with gzip compression
    and ``write_direct_chunk`` for maximum throughput (format v2.2). This
    avoids accumulating all compressed data in memory.

    When frame sizes differ, falls back to the legacy blob format (flat uint8
    array with per-frame byte-range offsets) for backward compatibility.

    Args:
        labels_path: A string path to the SLEAP labels file.
        label_images: A list of LabelImage objects to write.
        videos: List of Video objects for index mapping.
        tracks: List of Track objects for index mapping.
        instances: Optional list of Instance/PredictedInstance objects for index
            mapping. If provided, label image instance associations will be
            persisted.
    """
    if not label_images:
        return

    li_dtype = np.dtype(
        [
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("height", "u4"),
            ("width", "u4"),
            ("n_objects", "u4"),
            ("objects_start", "u4"),
            ("data_start", "u8"),
            ("data_end", "u8"),
            ("is_predicted", "u1"),
            ("score", "f4"),
            ("scale_x", "f4"),
            ("scale_y", "f4"),
            ("offset_x", "f4"),
            ("offset_y", "f4"),
        ]
    )

    obj_dtype = np.dtype(
        [
            ("label_id", "i4"),
            ("track", "i4"),
            ("instance", "i4"),
            ("score", "f4"),
        ]
    )

    # Determine if we can use chunked format (requires uniform frame sizes)
    shapes = {(li.height, li.width) for li in label_images}
    use_chunked = len(shapes) == 1

    li_rows = []
    obj_rows = []
    obj_offset = 0
    sources = []
    categories = []
    obj_names = []

    # For blob format fallback
    data_chunks: list[bytes] = []
    data_offset = 0

    with h5py.File(labels_path, "a") as f:
        # Create pixel data dataset
        if use_chunked:
            frame_h, frame_w = shapes.pop()
            n_frames = len(label_images)
            pixel_dset = f.create_dataset(
                "label_image_data",
                shape=(n_frames, frame_h, frame_w),
                chunks=(1, frame_h, frame_w),
                dtype=np.int32,
                compression="gzip",
                compression_opts=1,
            )

        for i, li in enumerate(label_images):
            video_idx = videos.index(li.video) if li.video in videos else -1
            frame_idx = li.frame_idx if li.frame_idx is not None else -1

            if use_chunked:
                # Write pixel data directly via write_direct_chunk (43x faster)
                compressed = zlib.compress(li.data.astype(np.int32).tobytes(), level=1)
                pixel_dset.id.write_direct_chunk((i, 0, 0), compressed)
                data_start = 0
                data_end = 0
            else:
                # Blob format: accumulate compressed chunks
                compressed = zlib.compress(li.data.astype(np.int32).tobytes())
                data_start = data_offset
                data_end = data_offset + len(compressed)
                data_chunks.append(compressed)
                data_offset = data_end

            # Build object rows for this frame
            n_objects = len(li.objects)
            objects_start = obj_offset

            for label_id in sorted(li.objects):
                info = li.objects[label_id]

                track_idx = tracks.index(info.track) if info.track in tracks else -1

                instance_idx = info._instance_idx  # Use stored index as default
                if instances is not None and info.instance is not None:
                    try:
                        instance_idx = instances.index(info.instance)
                    except ValueError:
                        pass  # Keep stored _instance_idx

                obj_score = info.score if info.score is not None else float("nan")
                obj_rows.append((label_id, track_idx, instance_idx, obj_score))
                categories.append(info.category)
                obj_names.append(info.name)

            obj_offset += n_objects

            is_predicted = isinstance(li, PredictedLabelImage)
            score = li.score if is_predicted else float("nan")

            li_rows.append(
                (
                    video_idx,
                    frame_idx,
                    li.height,
                    li.width,
                    n_objects,
                    objects_start,
                    data_start,
                    data_end,
                    int(is_predicted),
                    score,
                    li.scale[0],
                    li.scale[1],
                    li.offset[0],
                    li.offset[1],
                )
            )

            sources.append(li.source)

        # Write blob data if using legacy format
        if not use_chunked:
            data_flat = np.frombuffer(b"".join(data_chunks), dtype=np.uint8)
            f.create_dataset(
                "label_image_data",
                data=data_flat,
                dtype=np.uint8,
                **({"chunks": True} if len(data_flat) > 0 else {}),
            )

        # Write metadata datasets
        li_array = np.array(li_rows, dtype=li_dtype)
        obj_array = (
            np.array(obj_rows, dtype=obj_dtype)
            if obj_rows
            else np.array([], dtype=obj_dtype)
        )
        f.create_dataset("label_images", data=li_array, dtype=li_dtype)
        f.create_dataset("label_image_objects", data=obj_array, dtype=obj_dtype)
        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("label_image_sources", data=sources, dtype=str_dt)
        f.create_dataset("label_image_obj_categories", data=categories, dtype=str_dt)
        f.create_dataset("label_image_obj_names", data=obj_names, dtype=str_dt)

    # Store score maps for PredictedLabelImage if any exist
    li_sm_indices = []
    li_sm_chunks = []
    li_sm_offset = 0
    for i, li in enumerate(label_images):
        if isinstance(li, PredictedLabelImage) and li.score_map is not None:
            compressed = zlib.compress(li.score_map.astype(np.float32).tobytes())
            sm_bytes = np.frombuffer(compressed, dtype=np.uint8)
            sm_h, sm_w = li.score_map.shape[:2]
            li_sm_indices.append(
                (
                    i,
                    li_sm_offset,
                    li_sm_offset + len(sm_bytes),
                    sm_h,
                    sm_w,
                    li.score_map_scale[0],
                    li.score_map_scale[1],
                    li.score_map_offset[0],
                    li.score_map_offset[1],
                )
            )
            li_sm_chunks.append(sm_bytes)
            li_sm_offset += len(sm_bytes)

    if li_sm_indices:
        sm_index_dtype = np.dtype(
            [
                ("li_idx", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("height", "u4"),
                ("width", "u4"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ]
        )
        sm_index_array = np.array(li_sm_indices, dtype=sm_index_dtype)
        sm_flat = np.concatenate(li_sm_chunks)
        with h5py.File(labels_path, "a") as f:
            f.create_dataset("label_image_score_map_index", data=sm_index_array)
            f.create_dataset(
                "label_image_score_maps",
                data=sm_flat,
                dtype=np.uint8,
                **({"chunks": True} if len(sm_flat) > 0 else {}),
            )


def _write_metadata_standalone(
    labels_path: str,
    format_id: float = 2.2,
    skeletons: list[Skeleton] | None = None,
    provenance: dict | None = None,
) -> None:
    """Write minimal metadata group to an SLP file.

    This is used by ``LabelImageWriter`` to write format info without requiring
    a full ``Labels`` object.

    Args:
        labels_path: Path to the SLP file.
        format_id: Format version identifier.
        skeletons: Optional list of skeletons to serialize.
        provenance: Optional provenance dict.
    """
    skeletons = skeletons or []
    provenance = dict(provenance or {})

    # Custom encoding for provenance values
    for k in provenance:
        if isinstance(provenance[k], Path):
            provenance[k] = provenance[k].as_posix()

    skeletons_dicts, nodes_dicts = serialize_skeletons(skeletons)

    md = {
        "version": "2.0.0",
        "skeletons": skeletons_dicts,
        "nodes": nodes_dicts,
        "videos": [],
        "tracks": [],
        "suggestions": [],
        "negative_anchors": {},
        "provenance": provenance,
    }

    # Dtypes for empty placeholder datasets required by read_labels
    point_dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")]
    )
    pred_point_dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?"), ("score", "f8")]
    )
    instance_dtype = np.dtype(
        [
            ("instance_id", "i8"),
            ("instance_type", "u1"),
            ("frame_id", "u8"),
            ("skeleton", "u4"),
            ("track", "i4"),
            ("from_predicted", "i8"),
            ("score", "f4"),
            ("point_id_start", "u8"),
            ("point_id_end", "u8"),
            ("tracking_score", "f4"),
        ]
    )
    frame_dtype = np.dtype(
        [
            ("frame_id", "u8"),
            ("video", "u4"),
            ("frame_idx", "u8"),
            ("instance_id_start", "u8"),
            ("instance_id_end", "u8"),
        ]
    )

    with h5py.File(labels_path, "a") as f:
        # Bump for chunked label image format
        if "label_image_data" in f and f["label_image_data"].ndim == 3:
            format_id = max(format_id, 2.2)

        grp = f.require_group("metadata")
        grp.attrs["format_id"] = format_id
        grp.attrs["json"] = np.bytes_(json.dumps(md, separators=(",", ":")))

        # Write empty placeholder datasets so read_labels can parse the file
        if "points" not in f:
            f.create_dataset("points", data=np.array([], dtype=point_dtype))
        if "pred_points" not in f:
            f.create_dataset("pred_points", data=np.array([], dtype=pred_point_dtype))
        if "instances" not in f:
            f.create_dataset("instances", data=np.array([], dtype=instance_dtype))
        if "frames" not in f:
            f.create_dataset("frames", data=np.array([], dtype=frame_dtype))


class LabelImageWriter:
    """Streaming writer for label image annotations to SLP files.

    Writes label images one at a time (or in batches) to an HDF5/SLP file
    without holding all pixel data in memory simultaneously. Uses the chunked
    ``(T, H, W)`` int32 format with ``write_direct_chunk`` for maximum
    throughput.

    The HDF5 file and pixel dataset are created lazily on the first ``add()``
    call, since the frame dimensions ``(H, W)`` are needed to define the
    dataset shape. All subsequent frames must have the same dimensions.

    Attributes:
        path: Path to the output SLP file.
        video: Optional ``Video`` to associate with all label images.
        tracks: Optional list of ``Track`` objects.
        skeleton: Optional ``Skeleton`` for metadata.
        initial_capacity: Initial number of frames to allocate in the dataset.

    Example::

        with LabelImageWriter("output.slp", video=video) as writer:
            for frame_data in segmentation_results:
                li = UserLabelImage(data=frame_data, video=video, frame_idx=i)
                writer.add(li)
            labels = writer.finalize()
    """

    def __init__(
        self,
        path: str,
        video: Video | None = None,
        tracks: list[Track] | None = None,
        skeleton: Skeleton | None = None,
        initial_capacity: int = 100,
    ):
        """Initialize the streaming label image writer.

        Args:
            path: Path to the output SLP file.
            video: Optional video to associate with all label images.
            tracks: Optional list of tracks for object associations.
            skeleton: Optional skeleton for metadata.
            initial_capacity: Initial number of frames to allocate. The dataset
                grows exponentially (doubles) when capacity is exceeded.
        """
        self.path = str(path)
        self.video = video
        self.tracks = tracks or []
        self.skeleton = skeleton
        self.initial_capacity = initial_capacity

        # State
        self._file: h5py.File | None = None
        self._pixel_dset: h5py.Dataset | None = None
        self._frame_h: int = 0
        self._frame_w: int = 0
        self._capacity: int = initial_capacity
        self._count: int = 0

        # Accumulated metadata (kept in memory, ~80 bytes/frame + 16 bytes/obj)
        self._li_rows: list[tuple] = []
        self._obj_rows: list[tuple] = []
        self._obj_offset: int = 0
        self._sources: list[str] = []
        self._categories: list[str] = []
        self._obj_names: list[str] = []

        # Score map data (blob format, accumulated in memory)
        self._sm_indices: list[tuple] = []
        self._sm_chunks: list[np.ndarray] = []
        self._sm_offset: int = 0

        self._finalized: bool = False

    def _ensure_file(self, height: int, width: int) -> None:
        """Create the HDF5 file and pixel dataset on first use.

        Args:
            height: Frame height in pixels.
            width: Frame width in pixels.
        """
        if self._file is not None:
            return

        self._frame_h = height
        self._frame_w = width

        self._file = h5py.File(self.path, "w")
        self._pixel_dset = self._file.create_dataset(
            "label_image_data",
            shape=(self._capacity, height, width),
            maxshape=(None, height, width),
            chunks=(1, height, width),
            dtype=np.int32,
            compression="gzip",
            compression_opts=1,
        )

    def _grow_if_needed(self) -> None:
        """Double the pixel dataset capacity if it's full."""
        if self._count >= self._capacity:
            self._capacity *= 2
            self._pixel_dset.resize(self._capacity, axis=0)

    def add(self, label_image: LabelImage) -> None:
        """Add a single label image to the file.

        The first call creates the HDF5 file and locks the frame dimensions.
        Subsequent calls must provide frames with the same ``(H, W)``.

        Args:
            label_image: The label image to write.

        Raises:
            ValueError: If the frame dimensions don't match the first frame.
            RuntimeError: If the writer has already been finalized.
        """
        if self._finalized:
            raise RuntimeError("Writer has already been finalized.")

        h, w = label_image.height, label_image.width
        self._ensure_file(h, w)

        if h != self._frame_h or w != self._frame_w:
            raise ValueError(
                f"Frame size ({h}, {w}) does not match expected "
                f"({self._frame_h}, {self._frame_w}). All frames must have "
                f"the same dimensions."
            )

        idx = self._count
        self._grow_if_needed()

        # Write pixel data via write_direct_chunk
        compressed = zlib.compress(label_image.data.astype(np.int32).tobytes(), level=1)
        self._pixel_dset.id.write_direct_chunk((idx, 0, 0), compressed)

        # Collect video index
        videos = [self.video] if self.video is not None else []
        video_idx = (
            videos.index(label_image.video) if label_image.video in videos else -1
        )
        frame_idx = label_image.frame_idx if label_image.frame_idx is not None else -1

        # Build object rows
        n_objects = len(label_image.objects)
        objects_start = self._obj_offset

        for label_id in sorted(label_image.objects):
            info = label_image.objects[label_id]
            track_idx = (
                self.tracks.index(info.track) if info.track in self.tracks else -1
            )
            instance_idx = info._instance_idx
            obj_score = info.score if info.score is not None else float("nan")
            self._obj_rows.append((label_id, track_idx, instance_idx, obj_score))
            self._categories.append(info.category)
            self._obj_names.append(info.name)

        self._obj_offset += n_objects

        is_predicted = isinstance(label_image, PredictedLabelImage)
        score = label_image.score if is_predicted else float("nan")

        self._li_rows.append(
            (
                video_idx,
                frame_idx,
                h,
                w,
                n_objects,
                objects_start,
                0,  # data_start (unused for chunked)
                0,  # data_end (unused for chunked)
                int(is_predicted),
                score,
                label_image.scale[0],
                label_image.scale[1],
                label_image.offset[0],
                label_image.offset[1],
            )
        )

        self._sources.append(label_image.source)

        # Score map handling for PredictedLabelImage
        if is_predicted and label_image.score_map is not None:
            sm = label_image.score_map
            compressed_sm = zlib.compress(sm.astype(np.float32).tobytes())
            sm_bytes = np.frombuffer(compressed_sm, dtype=np.uint8)
            sm_h, sm_w = sm.shape[:2]
            self._sm_indices.append(
                (
                    idx,
                    self._sm_offset,
                    self._sm_offset + len(sm_bytes),
                    sm_h,
                    sm_w,
                    label_image.score_map_scale[0],
                    label_image.score_map_scale[1],
                    label_image.score_map_offset[0],
                    label_image.score_map_offset[1],
                )
            )
            self._sm_chunks.append(sm_bytes)
            self._sm_offset += len(sm_bytes)

        self._count += 1

    def add_batch(self, label_images: list[LabelImage]) -> None:
        """Add multiple label images at once.

        Convenience wrapper that calls ``add()`` for each label image.

        Args:
            label_images: List of label images to write.
        """
        for li in label_images:
            self.add(li)

    def finalize(self) -> Labels:
        """Finish writing, close the file, and return a ``Labels`` object.

        Trims the pixel dataset to the actual number of frames written, writes
        all metadata datasets, and closes the HDF5 file. The returned
        ``Labels`` object can be used directly or the file can be re-loaded
        with ``load_slp()``.

        Returns:
            A ``Labels`` object pointing at the written file.

        Raises:
            RuntimeError: If the writer has already been finalized.
        """
        if self._finalized:
            raise RuntimeError("Writer has already been finalized.")
        self._finalized = True

        # Handle empty writer (no frames added)
        if self._file is None:
            # Create minimal empty SLP file
            with h5py.File(self.path, "w"):
                pass
            videos = [self.video] if self.video is not None else []
            skeletons = [self.skeleton] if self.skeleton is not None else []
            write_videos(self.path, videos)
            write_tracks(self.path, self.tracks)
            _write_metadata_standalone(self.path, skeletons=skeletons)
            return Labels(
                videos=videos,
                skeletons=skeletons,
                tracks=self.tracks,
            )

        # Trim pixel dataset to actual count
        self._pixel_dset.resize(self._count, axis=0)

        # Write metadata datasets
        li_dtype = np.dtype(
            [
                ("video", "i4"),
                ("frame_idx", "i8"),
                ("height", "u4"),
                ("width", "u4"),
                ("n_objects", "u4"),
                ("objects_start", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("is_predicted", "u1"),
                ("score", "f4"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ]
        )

        obj_dtype = np.dtype(
            [
                ("label_id", "i4"),
                ("track", "i4"),
                ("instance", "i4"),
                ("score", "f4"),
            ]
        )

        li_array = np.array(self._li_rows, dtype=li_dtype)
        obj_array = (
            np.array(self._obj_rows, dtype=obj_dtype)
            if self._obj_rows
            else np.array([], dtype=obj_dtype)
        )

        str_dt = h5py.special_dtype(vlen=str)
        f = self._file
        f.create_dataset("label_images", data=li_array, dtype=li_dtype)
        f.create_dataset("label_image_objects", data=obj_array, dtype=obj_dtype)
        f.create_dataset("label_image_sources", data=self._sources, dtype=str_dt)
        f.create_dataset(
            "label_image_obj_categories", data=self._categories, dtype=str_dt
        )
        f.create_dataset("label_image_obj_names", data=self._obj_names, dtype=str_dt)

        # Write score maps if any
        if self._sm_indices:
            sm_index_dtype = np.dtype(
                [
                    ("li_idx", "u4"),
                    ("data_start", "u8"),
                    ("data_end", "u8"),
                    ("height", "u4"),
                    ("width", "u4"),
                    ("scale_x", "f4"),
                    ("scale_y", "f4"),
                    ("offset_x", "f4"),
                    ("offset_y", "f4"),
                ]
            )
            sm_index_array = np.array(self._sm_indices, dtype=sm_index_dtype)
            sm_flat = np.concatenate(self._sm_chunks)
            f.create_dataset("label_image_score_map_index", data=sm_index_array)
            f.create_dataset(
                "label_image_score_maps",
                data=sm_flat,
                dtype=np.uint8,
                **({"chunks": True} if len(sm_flat) > 0 else {}),
            )

        # Close HDF5 file before writing video/track/metadata
        f.close()
        self._file = None

        # Write video, track, and metadata info
        videos = [self.video] if self.video is not None else []
        skeletons = [self.skeleton] if self.skeleton is not None else []
        write_videos(self.path, videos)
        write_tracks(self.path, self.tracks)
        _write_metadata_standalone(self.path, skeletons=skeletons)

        return Labels(
            videos=videos,
            skeletons=skeletons,
            tracks=self.tracks,
            label_images=read_label_images(self.path, videos, self.tracks, []),
        )

    def __enter__(self) -> "LabelImageWriter":
        """Enter context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit context manager, finalizing if not already done."""
        if not self._finalized:
            self.finalize()
        elif self._file is not None:
            self._file.close()
            self._file = None


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
    """
    if not source_paths:
        raise ValueError("At least one source path is required.")

    source_paths = [str(p) for p in source_paths]
    dest_path = str(dest_path)

    # --- Phase 1: Open all sources, read index tables, validate dimensions ---
    source_files: list[h5py.File] = []
    source_index_tables: list[np.ndarray] = []
    source_pixel_datasets: list[h5py.Dataset] = []
    source_obj_tables: list[np.ndarray] = []
    source_categories: list[list[str]] = []
    source_names: list[list[str]] = []
    source_sources: list[list[str]] = []
    source_videos: list[list[Video]] = []
    source_tracks: list[list[Track]] = []
    source_sm_indices: list[np.ndarray | None] = []
    source_sm_data: list[h5py.Dataset | None] = []

    all_shapes: set[tuple[int, int]] = set()

    try:
        for src_path in source_paths:
            f = h5py.File(src_path, "r")
            source_files.append(f)

            # Read index table
            if "label_images" not in f or "label_image_data" not in f:
                raise ValueError(f"Source file has no label images: {src_path}")

            li_data = f["label_images"][:]
            if len(li_data) == 0:
                raise ValueError(f"Source file has no label images: {src_path}")
            source_index_tables.append(li_data)
            source_pixel_datasets.append(f["label_image_data"])

            # Collect frame dimensions
            for row in li_data:
                all_shapes.add((int(row["height"]), int(row["width"])))

            # Read objects table
            if "label_image_objects" in f:
                source_obj_tables.append(f["label_image_objects"][:])
            else:
                obj_dtype = np.dtype(
                    [("label_id", "i4"), ("track", "i4"), ("instance", "i4")]
                )
                source_obj_tables.append(np.array([], dtype=obj_dtype))

            # Read string metadata
            if "label_image_obj_categories" in f:
                cats = [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["label_image_obj_categories"][:]
                ]
            else:
                cats = []
            source_categories.append(cats)

            if "label_image_obj_names" in f:
                nms = [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["label_image_obj_names"][:]
                ]
            else:
                nms = []
            source_names.append(nms)

            if "label_image_sources" in f:
                srcs = [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["label_image_sources"][:]
                ]
            else:
                srcs = []
            source_sources.append(srcs)

            # Read videos and tracks from source
            source_videos.append(read_videos(src_path, open_backend=False))
            source_tracks.append(read_tracks(src_path))

            # Score map data
            if "label_image_score_map_index" in f:
                source_sm_indices.append(f["label_image_score_map_index"][:])
                source_sm_data.append(f["label_image_score_maps"])
            else:
                source_sm_indices.append(None)
                source_sm_data.append(None)

        # Validate uniform dimensions
        if len(all_shapes) > 1:
            raise ValueError(
                f"Cannot merge label images with different dimensions: "
                f"{all_shapes}. All sources must have the same (H, W)."
            )

        frame_h, frame_w = all_shapes.pop()

        # --- Phase 2: Deduplicate videos and tracks ---
        if video is not None:
            merged_videos = [video]
        else:
            # Deduplicate by filename
            merged_videos: list[Video] = []
            seen_filenames: dict[str, int] = {}
            for vlist in source_videos:
                for v in vlist:
                    fn = v.filename
                    if fn not in seen_filenames:
                        seen_filenames[fn] = len(merged_videos)
                        merged_videos.append(v)

        # Deduplicate tracks by name
        merged_tracks: list[Track] = []
        seen_track_names: dict[str, int] = {}
        for tlist in source_tracks:
            for t in tlist:
                if t.name not in seen_track_names:
                    seen_track_names[t.name] = len(merged_tracks)
                    merged_tracks.append(t)

        # --- Phase 3: Create destination and copy data ---
        total_frames = sum(len(t) for t in source_index_tables)

        li_dtype = np.dtype(
            [
                ("video", "i4"),
                ("frame_idx", "i8"),
                ("height", "u4"),
                ("width", "u4"),
                ("n_objects", "u4"),
                ("objects_start", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("is_predicted", "u1"),
                ("score", "f4"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ]
        )

        obj_dtype = np.dtype(
            [
                ("label_id", "i4"),
                ("track", "i4"),
                ("instance", "i4"),
                ("score", "f4"),
            ]
        )

        dest_li_rows: list[tuple] = []
        dest_obj_rows: list[tuple] = []
        dest_obj_offset = 0
        dest_sources: list[str] = []
        dest_categories: list[str] = []
        dest_obj_names: list[str] = []
        dest_sm_indices: list[tuple] = []
        dest_sm_chunks: list[np.ndarray] = []
        dest_sm_offset = 0

        with h5py.File(dest_path, "w") as dest_f:
            dest_pixel_dset = dest_f.create_dataset(
                "label_image_data",
                shape=(total_frames, frame_h, frame_w),
                chunks=(1, frame_h, frame_w),
                dtype=np.int32,
                compression="gzip",
                compression_opts=1,
            )

            dest_frame_idx = 0

            for src_idx, (li_table, pixel_ds, obj_table) in enumerate(
                zip(
                    source_index_tables,
                    source_pixel_datasets,
                    source_obj_tables,
                )
            ):
                is_chunked = pixel_ds.ndim == 3
                src_videos = source_videos[src_idx]
                src_tracks = source_tracks[src_idx]

                # Build video index remap for this source
                if video is not None:
                    video_remap = {i: 0 for i in range(len(src_videos))}
                else:
                    video_remap = {}
                    for i, v in enumerate(src_videos):
                        fn = v.filename
                        video_remap[i] = seen_filenames[fn]

                # Build track index remap for this source
                track_remap: dict[int, int] = {}
                for i, t in enumerate(src_tracks):
                    track_remap[i] = seen_track_names[t.name]

                for local_i, row in enumerate(li_table):
                    # Copy pixel data
                    if is_chunked:
                        # Raw chunk copy: read_direct_chunk -> write_direct_chunk
                        filter_mask, raw_data = pixel_ds.id.read_direct_chunk(
                            (local_i, 0, 0)
                        )
                        dest_pixel_dset.id.write_direct_chunk(
                            (dest_frame_idx, 0, 0), raw_data
                        )
                    else:
                        # Blob format: decompress then recompress for chunked
                        data_start = int(row["data_start"])
                        data_end = int(row["data_end"])
                        h = int(row["height"])
                        w = int(row["width"])
                        raw = zlib.decompress(pixel_ds[data_start:data_end].tobytes())
                        arr = np.frombuffer(raw, dtype=np.int32).reshape(h, w)
                        compressed = zlib.compress(arr.tobytes(), level=1)
                        dest_pixel_dset.id.write_direct_chunk(
                            (dest_frame_idx, 0, 0), compressed
                        )

                    # Remap video index
                    orig_video_idx = int(row["video"])
                    new_video_idx = video_remap.get(orig_video_idx, -1)

                    # Remap object rows
                    n_objects = int(row["n_objects"])
                    objects_start = int(row["objects_start"])

                    for j in range(n_objects):
                        obj_idx = objects_start + j
                        if obj_idx < len(obj_table):
                            obj_row = obj_table[obj_idx]
                            label_id = int(obj_row["label_id"])

                            orig_track_idx = int(obj_row["track"])
                            new_track_idx = track_remap.get(orig_track_idx, -1)

                            instance_idx = int(obj_row["instance"])
                            obj_score = (
                                float(obj_row["score"])
                                if "score" in obj_row.dtype.names
                                else float("nan")
                            )

                            dest_obj_rows.append(
                                (label_id, new_track_idx, instance_idx, obj_score)
                            )

                            # Copy string metadata
                            src_cats = source_categories[src_idx]
                            dest_categories.append(
                                src_cats[obj_idx] if obj_idx < len(src_cats) else ""
                            )
                            src_nms = source_names[src_idx]
                            dest_obj_names.append(
                                src_nms[obj_idx] if obj_idx < len(src_nms) else ""
                            )

                    # Read spatial metadata with defaults
                    scale_x = (
                        float(row["scale_x"]) if "scale_x" in row.dtype.names else 1.0
                    )
                    scale_y = (
                        float(row["scale_y"]) if "scale_y" in row.dtype.names else 1.0
                    )
                    offset_x = (
                        float(row["offset_x"]) if "offset_x" in row.dtype.names else 0.0
                    )
                    offset_y = (
                        float(row["offset_y"]) if "offset_y" in row.dtype.names else 0.0
                    )

                    is_predicted = (
                        bool(row["is_predicted"])
                        if "is_predicted" in row.dtype.names
                        else False
                    )
                    score = (
                        float(row["score"])
                        if "score" in row.dtype.names
                        else float("nan")
                    )

                    dest_li_rows.append(
                        (
                            new_video_idx,
                            int(row["frame_idx"]),
                            int(row["height"]),
                            int(row["width"]),
                            n_objects,
                            dest_obj_offset,
                            0,  # data_start (unused for chunked)
                            0,  # data_end (unused for chunked)
                            int(is_predicted),
                            score,
                            scale_x,
                            scale_y,
                            offset_x,
                            offset_y,
                        )
                    )

                    dest_obj_offset += n_objects

                    # Copy source string
                    src_srcs = source_sources[src_idx]
                    dest_sources.append(
                        src_srcs[local_i] if local_i < len(src_srcs) else ""
                    )

                    # Copy score maps if present
                    sm_index = source_sm_indices[src_idx]
                    sm_data = source_sm_data[src_idx]
                    if sm_index is not None and sm_data is not None:
                        for sm_row in sm_index:
                            if int(sm_row["li_idx"]) == local_i:
                                sm_start = int(sm_row["data_start"])
                                sm_end = int(sm_row["data_end"])
                                sm_bytes = sm_data[sm_start:sm_end]
                                sm_h = int(sm_row["height"])
                                sm_w = int(sm_row["width"])
                                sm_scale_x = (
                                    float(sm_row["scale_x"])
                                    if "scale_x" in sm_row.dtype.names
                                    else 1.0
                                )
                                sm_scale_y = (
                                    float(sm_row["scale_y"])
                                    if "scale_y" in sm_row.dtype.names
                                    else 1.0
                                )
                                sm_offset_x = (
                                    float(sm_row["offset_x"])
                                    if "offset_x" in sm_row.dtype.names
                                    else 0.0
                                )
                                sm_offset_y = (
                                    float(sm_row["offset_y"])
                                    if "offset_y" in sm_row.dtype.names
                                    else 0.0
                                )
                                dest_sm_indices.append(
                                    (
                                        dest_frame_idx,
                                        dest_sm_offset,
                                        dest_sm_offset + len(sm_bytes),
                                        sm_h,
                                        sm_w,
                                        sm_scale_x,
                                        sm_scale_y,
                                        sm_offset_x,
                                        sm_offset_y,
                                    )
                                )
                                sm_np = np.array(sm_bytes, dtype=np.uint8)
                                dest_sm_chunks.append(sm_np)
                                dest_sm_offset += len(sm_bytes)
                                break

                    dest_frame_idx += 1

            # Write metadata datasets
            li_array = np.array(dest_li_rows, dtype=li_dtype)
            obj_array = (
                np.array(dest_obj_rows, dtype=obj_dtype)
                if dest_obj_rows
                else np.array([], dtype=obj_dtype)
            )
            str_dt = h5py.special_dtype(vlen=str)
            dest_f.create_dataset("label_images", data=li_array, dtype=li_dtype)
            dest_f.create_dataset(
                "label_image_objects", data=obj_array, dtype=obj_dtype
            )
            dest_f.create_dataset(
                "label_image_sources", data=dest_sources, dtype=str_dt
            )
            dest_f.create_dataset(
                "label_image_obj_categories", data=dest_categories, dtype=str_dt
            )
            dest_f.create_dataset(
                "label_image_obj_names", data=dest_obj_names, dtype=str_dt
            )

            # Write score maps if any
            if dest_sm_indices:
                sm_index_dtype = np.dtype(
                    [
                        ("li_idx", "u4"),
                        ("data_start", "u8"),
                        ("data_end", "u8"),
                        ("height", "u4"),
                        ("width", "u4"),
                        ("scale_x", "f4"),
                        ("scale_y", "f4"),
                        ("offset_x", "f4"),
                        ("offset_y", "f4"),
                    ]
                )
                sm_index_array = np.array(dest_sm_indices, dtype=sm_index_dtype)
                sm_flat = np.concatenate(dest_sm_chunks)
                dest_f.create_dataset(
                    "label_image_score_map_index", data=sm_index_array
                )
                dest_f.create_dataset(
                    "label_image_score_maps",
                    data=sm_flat,
                    dtype=np.uint8,
                    **({"chunks": True} if len(sm_flat) > 0 else {}),
                )

        # Write video, track, and metadata info
        write_videos(dest_path, merged_videos)
        write_tracks(dest_path, merged_tracks)
        _write_metadata_standalone(dest_path)

        # Return Labels pointing at the merged file
        return Labels(
            videos=merged_videos,
            tracks=merged_tracks,
            label_images=read_label_images(dest_path, merged_videos, merged_tracks, []),
        )

    finally:
        for f in source_files:
            f.close()


def read_labels(labels_path: str, open_videos: bool = True) -> Labels:
    """Read a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        open_videos: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).

    Returns:
        The processed `Labels` object.
    """
    tracks = read_tracks(labels_path)
    videos = read_videos(labels_path, open_backend=open_videos)
    skeletons = read_skeletons(labels_path)
    points = read_points(labels_path)
    pred_points = read_pred_points(labels_path)
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")
    instances = read_instances(
        labels_path, skeletons, tracks, points, pred_points, format_id
    )
    suggestions = read_suggestions(labels_path, videos)
    metadata = read_metadata(labels_path)
    provenance = metadata.get("provenance", dict())

    frames = read_hdf5_dataset(labels_path, "frames")
    negative_markers = read_negative_frames(labels_path)

    # Check if video IDs in frames are sequential list indices (0, 1, 2, ..., n-1)
    # or sparse embedded IDs (e.g., 0, 15, 29, 47, ...) that need remapping
    frame_video_ids = set(int(frame[1]) for frame in frames)
    max_frame_video_id = max(frame_video_ids) if frame_video_ids else 0

    # If max video ID == len(videos) - 1 and IDs are contiguous, they're list indices
    # In this case, use identity mapping (backwards compatible behavior)
    frames_use_list_indices = (
        len(frame_video_ids) == len(videos) and max_frame_video_id == len(videos) - 1
    )

    if frames_use_list_indices:
        # Video IDs are sequential list indices - use identity mapping
        video_id_to_index = {i: i for i in range(len(videos))}
    else:
        # Build mapping from sparse video IDs to list indices
        # This handles files from old SLEAP where video IDs can be sparse
        # (e.g., 0, 15, 29, 47, ...) rather than sequential (0, 1, 2, 3, ...)
        video_id_to_index = {}
        for i, video in enumerate(videos):
            # For embedded videos, extract the video ID from backend.dataset
            if (
                hasattr(video, "backend")
                and video.backend is not None
                and hasattr(video.backend, "dataset")
                and video.backend.dataset is not None
            ):
                dataset = video.backend.dataset
                # Extract video ID from dataset name (e.g., "video15/video" → 15)
                if "/" in dataset:
                    video_group = dataset.split("/")[0]
                    if video_group.startswith("video"):
                        video_id_str = video_group[5:]  # Remove "video" prefix
                        if video_id_str.isdigit():
                            video_id = int(video_id_str)
                            video_id_to_index[video_id] = i
                            continue

            # For non-embedded videos or videos without extractable IDs,
            # assume sequential indexing (backwards compatible behavior)
            video_id_to_index[i] = i

    labeled_frames = []
    for _, video_id, frame_idx, instance_id_start, instance_id_end in frames:
        # Map sparse video_id to sequential list index
        video_index = video_id_to_index.get(video_id, video_id)

        # Check if this frame is marked as negative (using sparse video_id)
        is_negative = (int(video_id), int(frame_idx)) in negative_markers

        labeled_frames.append(
            LabeledFrame(
                video=videos[video_index],
                frame_idx=int(frame_idx),
                instances=instances[instance_id_start:instance_id_end],
                is_negative=is_negative,
            )
        )

    identities = read_identities(labels_path)
    sessions = read_sessions(labels_path, videos, labeled_frames, identities=identities)
    rois = read_rois(labels_path, videos, tracks, instances)
    masks = read_masks(labels_path, videos, tracks, instances)
    bboxes = read_bboxes(labels_path, videos, tracks, instances)
    label_images = read_label_images(labels_path, videos, tracks, instances)

    # Capture HDF5 file handle for lazy label image data
    _li_file = getattr(read_label_images, "_open_file", None)
    if _li_file is not None:
        read_label_images._open_file = None  # type: ignore[attr-defined]

    # Migrate old-style bbox ROIs to BoundingBox objects (skip predicted ROIs)
    if not bboxes:
        migrated_bboxes: list[BoundingBox] = []
        remaining_rois: list[ROI] = []
        for roi in rois:
            if roi.is_bbox and not roi.is_predicted:
                minx, miny, maxx, maxy = roi.geometry.bounds
                migrated_bboxes.append(
                    UserBoundingBox.from_xyxy(
                        minx,
                        miny,
                        maxx,
                        maxy,
                        video=roi.video,
                        frame_idx=roi.frame_idx,
                        track=roi.track,
                        instance=roi.instance,
                        category=roi.category,
                        name=roi.name,
                        source=roi.source,
                    )
                )
            else:
                remaining_rois.append(roi)
        if migrated_bboxes:
            bboxes = migrated_bboxes
            rois = remaining_rois

    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        identities=identities,
        suggestions=suggestions,
        sessions=sessions,
        provenance=provenance,
        rois=rois,
        masks=masks,
        bboxes=bboxes,
        label_images=label_images,
    )
    labels.provenance["filename"] = labels_path

    # Store the HDF5 file handle for lazy label image data (keeps it alive)
    if _li_file is not None:
        labels._label_image_file = _li_file

    return labels


def read_labels_set(
    path: str | Path | list[str | Path] | dict[str, str | Path],
    open_videos: bool = True,
) -> "LabelsSet":
    """Load a LabelsSet from multiple SLP files.

    Args:
        path: Can be one of:
            - A directory path containing .slp files
            - A list of .slp file paths
            - A dictionary mapping names to .slp file paths
        open_videos: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened.

    Returns:
        A LabelsSet containing the loaded Labels objects.

    Examples:
        Load from directory:
        >>> labels_set = read_labels_set("path/to/splits/")

        Load from list:
        >>> labels_set = read_labels_set(["train.slp", "val.slp", "test.slp"])

        Load from dictionary:
        >>> labels_set = read_labels_set({"train": "train.slp", "val": "val.slp"})
    """
    from sleap_io.model.labels_set import LabelsSet

    labels_dict = {}

    if isinstance(path, dict):
        # Dictionary of name -> path mappings
        for name, file_path in path.items():
            labels_dict[name] = read_labels(str(file_path), open_videos=open_videos)

    elif isinstance(path, list):
        # List of paths - auto-generate names
        for i, file_path in enumerate(path):
            file_path = Path(file_path)
            # Use filename without extension as key, or fall back to generic name
            name = file_path.stem if file_path.stem else f"labels_{i}"
            labels_dict[name] = read_labels(str(file_path), open_videos=open_videos)

    else:
        # Directory path - find all .slp files
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Path must be a directory, list, or dict. Got: {path}")

        slp_files = sorted(path.glob("*.slp"))
        if not slp_files:
            raise ValueError(f"No .slp files found in directory: {path}")

        for slp_file in slp_files:
            # Use filename without extension as key
            name = slp_file.stem
            labels_dict[name] = read_labels(str(slp_file), open_videos=open_videos)

    return LabelsSet(labels=labels_dict)


def _write_labels_lazy(
    labels_path: str,
    labels: Labels,
    embed: bool | str | list[tuple[Video, int]] | None = None,
    restore_original_videos: bool = True,
    verbose: bool = True,
) -> None:
    """Write lazy Labels to SLP file using fast path.

    This function copies raw HDF5 arrays directly without materializing
    LabeledFrame or Instance objects, providing significant performance
    improvement for save operations on lazy-loaded labels.

    Note:
        ROI-to-instance associations are preserved via the stored
        ``_instance_idx`` from the original file. Because instances are not
        materialized in lazy mode, any modifications to ``roi.instance``
        will not be reflected in the saved file. To persist modified
        instance associations, call ``labels.materialize()`` before saving.

    Args:
        labels_path: A string path to the SLEAP labels file to save.
        labels: A lazy `Labels` object to save (must have is_lazy=True).
        embed: Embedding mode. For lazy labels, only `None`, `False`, and
            `"source"` are supported without materialization. Other values
            will trigger materialization.
        restore_original_videos: If `True` (default) and `embed=False`, use original
            video files.
        verbose: If `True` (the default), display progress information.

    Raises:
        ValueError: If labels is not lazy.
    """
    if not labels.is_lazy:
        raise ValueError("_write_labels_lazy requires lazy Labels")

    lazy_store = labels._lazy_store

    # Delete existing file if it exists
    if Path(labels_path).exists():
        Path(labels_path).unlink()

    # Determine reference mode based on parameters
    if embed == "source" or (embed is False and restore_original_videos):
        reference_mode = VideoReferenceMode.RESTORE_ORIGINAL
    elif embed is False and not restore_original_videos:
        reference_mode = VideoReferenceMode.PRESERVE_SOURCE
    else:
        reference_mode = VideoReferenceMode.EMBED

    # Write videos metadata (uses labels.videos which may have been modified)
    write_videos(
        labels_path,
        labels.videos,
        reference_mode=reference_mode,
        original_videos=None,
        verbose=verbose,
    )

    # Write other metadata
    write_tracks(labels_path, labels.tracks)
    write_suggestions(labels_path, labels.suggestions, labels.videos)
    # For sessions, pass empty list since we don't have materialized frames
    # (consistent with lazy load behavior)
    write_sessions(labels_path, labels.sessions, labels.videos, [])
    write_metadata(labels_path, labels)

    # Write raw arrays directly from lazy store (fast path)
    with h5py.File(labels_path, "a") as f:
        f.create_dataset(
            "points",
            data=lazy_store.points_data,
            dtype=lazy_store.points_data.dtype,
        )
        f.create_dataset(
            "pred_points",
            data=lazy_store.pred_points_data,
            dtype=lazy_store.pred_points_data.dtype,
        )
        f.create_dataset(
            "instances",
            data=lazy_store.instances_data,
            dtype=lazy_store.instances_data.dtype,
        )
        f.create_dataset(
            "frames",
            data=lazy_store.frames_data,
            dtype=lazy_store.frames_data.dtype,
        )

    # Write negative frames directly from lazy store data
    if lazy_store._negative_frames:
        neg_dtype = np.dtype([("video_id", "u4"), ("frame_idx", "u8")])
        neg_data = np.array(sorted(lazy_store._negative_frames), dtype=neg_dtype)
        with h5py.File(labels_path, "a") as f:
            f.create_dataset("negative_frames", data=neg_data)

    # Write ROIs, masks, bboxes, and label images (eagerly loaded even in lazy mode)
    write_rois(labels_path, labels.rois, labels.videos, labels.tracks)
    write_masks(labels_path, labels.masks, labels.videos, labels.tracks)
    write_bboxes(labels_path, labels.bboxes, labels.videos, labels.tracks)
    # Note: instance associations are not persisted in lazy mode (no all_instances).
    write_label_images(labels_path, labels.label_images, labels.videos, labels.tracks)


def write_labels(
    labels_path: str,
    labels: Labels,
    embed: bool | str | list[tuple[Video, int]] | None = None,
    restore_original_videos: bool = True,
    embed_inplace: bool = False,
    verbose: bool = True,
    plugin: str | None = None,
    embed_all_videos: bool = True,
    progress_callback: Callable[[int, int], bool] | None = None,
):
    """Write a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file to save.
        labels: A `Labels` object to save.
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
            based on available packages.
        embed_all_videos: If `True` (the default), all videos in the labels will be
            converted to embedded references, even if they have no frames to embed.
            This ensures package files are portable. If `False`, only videos with
            frames to embed are converted.
        progress_callback: Optional callback function called during frame embedding
            with `(current, total)` arguments. If it returns `False`, the operation
            is cancelled and `ExportCancelled` is raised.
    """
    # Fast path for lazy labels (avoids materializing frames/instances)
    # Supported for simple embed modes: None, False, "source"
    if labels.is_lazy:
        # Check if embed mode requires materialization
        needs_materialization = (
            embed is True
            or embed
            in (
                "all",
                "user",
                "suggestions",
                "user+suggestions",
            )
            or isinstance(embed, list)
        )

        if needs_materialization:
            # Materialize to support embedding
            labels = labels.materialize()
        else:
            # Use fast path - copy raw arrays directly
            _write_labels_lazy(
                labels_path,
                labels,
                embed=embed,
                restore_original_videos=restore_original_videos,
                verbose=verbose,
            )
            return

    if Path(labels_path).exists():
        Path(labels_path).unlink()

    # Make a copy to avoid mutating the input labels when embedding
    if embed and not embed_inplace:
        original_labels = labels
        labels = labels.copy(open_videos=True)

        # If embed is a list of (video, frame_idx) tuples, remap videos to the copy
        if isinstance(embed, list):
            # Create mapping from original videos to copied videos
            video_map = {
                orig: copied
                for orig, copied in zip(original_labels.videos, labels.videos)
            }
            # Remap the embed list to use copied video objects
            embed = [
                (video_map.get(video, video), frame_idx) for video, frame_idx in embed
            ]

    # Store original videos before embedding modifies them
    # We need to make a copy of the actual video objects, not just the list
    original_videos = [v for v in labels.videos] if embed else None

    if embed:
        embed_videos(
            labels_path,
            labels,
            embed,
            verbose=verbose,
            plugin=plugin,
            embed_all_videos=embed_all_videos,
            progress_callback=progress_callback,
        )

    # Determine reference mode based on parameters
    if embed == "source" or (embed is False and restore_original_videos):
        reference_mode = VideoReferenceMode.RESTORE_ORIGINAL
    elif embed is False and not restore_original_videos:
        reference_mode = VideoReferenceMode.PRESERVE_SOURCE
    else:
        reference_mode = VideoReferenceMode.EMBED

    write_videos(
        labels_path,
        labels.videos,
        reference_mode=reference_mode,
        original_videos=original_videos,
        verbose=verbose,
    )
    write_tracks(labels_path, labels.tracks)
    write_identities(labels_path, labels.identities)
    write_suggestions(labels_path, labels.suggestions, labels.videos)
    write_sessions(
        labels_path,
        labels.sessions,
        labels.videos,
        labels.labeled_frames,
        identities=labels.identities,
    )
    write_metadata(labels_path, labels)
    write_lfs(labels_path, labels)
    write_negative_frames(labels_path, labels)

    # Collect all instances across all frames for ROI/bbox instance mapping
    all_instances: list[Instance | PredictedInstance] = []
    for lf in labels.labeled_frames:
        all_instances.extend(lf.instances)
    write_rois(labels_path, labels.rois, labels.videos, labels.tracks, all_instances)
    write_masks(labels_path, labels.masks, labels.videos, labels.tracks, all_instances)
    write_bboxes(
        labels_path, labels.bboxes, labels.videos, labels.tracks, all_instances
    )
    write_label_images(
        labels_path,
        labels.label_images,
        labels.videos,
        labels.tracks,
        all_instances,
    )
