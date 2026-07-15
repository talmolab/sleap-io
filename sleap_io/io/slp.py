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
    - 1.9: Added Instance3D support, predicted variants (is_predicted, score) for
            masks/ROIs/label images; instance association for masks; string
            datasets for metadata
    - 2.0: Columnar bounding box storage (/bboxes group with x1/y1/x2/y2 datasets)
    - 2.1: Spatial transform metadata on dense annotations (masks/label images)
    - 2.2: Chunked label image data format (/label_image_data as rank-3 dataset)
    - 2.3: Virtual on-read video crops (/video_crops dataset)
    - 2.4: Persisted mask from_predicted provenance (from_predicted column in
            mask_dtype storing the source prediction's index in the mask list)
    - 2.5: Added the re-ID identity subsystem. The /identity group holds the
            identity catalog (a native `name` string dataset + an optional columnar
            entity-attribute-value metadata table: meta_owner/meta_key/meta_val)
            plus the per-detection `links` dataset (owner_type, owner_id,
            identity_idx, identity_score) joining a detection to a catalog entry.
            The /embeddings group holds per-detection appearance / re-ID vectors as
            a single columnar struct-of-arrays (vectors (N, D) float +
            owner_type/owner_id join columns; vectors chunked so whole rows stay
            within a chunk). The owner_type column reuses the shared OWNER_* codes
            across both. All additive, read on group presence.
    - 2.6: Added frame-spanning events. The /event_types group holds the event
            catalog (a native `name` string dataset + an optional `description`
            dataset + an optional entity-attribute-value metadata table:
            meta_owner/meta_key/meta_val), mirroring /identity. The /events group
            holds the interval annotations as a columnar struct-of-arrays (video,
            start_frame/end_frame inclusive int64, type, subject/target kind+idx,
            is_predicted, score, name, source, per-event EAV metadata) plus a ragged
            CSR pair (scores flat float32 + score_offsets int64) for optional
            framewise PredictedEvent.scores traces. Every column is presence-guarded;
            both groups additive, read on group presence.
    - 2.7: Added the class/category subsystem, mirroring identity. The /categories
            group is a fully self-contained mirror of /identity: a native `name`
            string dataset + an optional entity-attribute-value metadata table
            (meta_owner/meta_key/meta_val) forming the category catalog, plus a
            per-detection `links` dataset (owner_type, owner_id, category_idx,
            category_score) joining a detection to a catalog entry. Category
            appearance vectors live in the SAME /embeddings group as sibling
            parallel datasets (category_vectors (M, D) + category_owner_type /
            category_owner_id join columns), independent of the identity
            vectors/owner_type/owner_id datasets so the two embedding kinds need
            not share dimensionality D. Identity-only files stay byte-identical
            (no category_* datasets, no /categories group written). All additive,
            read on group presence.
    - 2.8: Externalized RecordingSession frame-group data out of the per-session
            `sessions_json` string into the columnar `/session_data` group so large
            multi-view projects scale (previously an entire session -- every frame
            group with its inline 3D points -- was one JSON string, growing to
            hundreds of MB and becoming unreadable in JS/WASM past the ~0.45 GB
            vlen-string limit). `sessions_json` now holds only calibration +
            `camcorder_to_video_idx_map` + session-level metadata + an
            `fg_start`/`fg_end` range into `/session_data/frame_groups`. The group
            holds: `frame_groups` (frame_idx + instance-group range), `instance_groups`
            (identity_idx/score/instance_3d_score + points_3d range + predicted flag +
            member range), `instance_group_members` (the columnarized
            camcorder_to_lf_and_inst_idx_map: camera/lf/inst), `points_3d` (N,3) and
            `pred_points_3d` (N,4 = xyz+score) as chunked+gzip float matrices sliced by
            row range, and presence-guarded `frame_group_meta`/`instance_group_meta`
            per-row JSON metadata blobs. The reader dispatches on `/session_data`
            presence and still accepts pre-2.8 inline `frame_group_dicts` (with inline
            `points`). Lazy save copies the group verbatim; only present when a session
            has frame groups (files without stay byte-identical). Read on group
            presence.
"""

from __future__ import annotations

import io
import sys
import warnings
import zlib
from contextlib import nullcontext
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
    _read_dataset_from_open_file,
    is_file_accessible,
    read_hdf5_attrs,
    read_hdf5_dataset,
    sanitize_filename,
    write_hdf5_dataset,
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
from sleap_io.model.category import Category
from sleap_io.model.embedding import Embedding
from sleap_io.model.event import Event, EventType, PredictedEvent, UserEvent
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
    from sleap_io.io.slp_lazy import LazyDataStore
    from sleap_io.model.centroid import Centroid
    from sleap_io.model.instance import PointsArray, PredictedPointsArray
    from sleap_io.model.labels_set import LabelsSet

try:
    import cv2
except ImportError:
    pass


# -- Label image HDF5 dtype constants (shared by write/merge/streaming writer) --------

LI_DTYPE = np.dtype(
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

OBJ_DTYPE = np.dtype(
    [
        ("label_id", "i4"),
        ("track", "i4"),
        ("instance", "i4"),
        ("score", "f4"),
        ("tracking_score", "f4"),
    ]
)

LI_SM_INDEX_DTYPE = np.dtype(
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


# -- Columnar RecordingSession HDF5 dtype constants (SLP 2.8, /session_data group) ----
# Shared by the session write/read/lazy paths. See the 2.8 format-history entry.

# One row per FrameGroup (across all sessions, in emission order). ``ig_start`` /
# ``ig_end`` are a half-open range into ``instance_groups``.
FRAME_GROUP_DTYPE = np.dtype(
    [
        ("frame_idx", "i8"),
        ("ig_start", "u8"),
        ("ig_end", "u8"),
    ]
)

# One row per InstanceGroup. ``identity_idx`` is -1 when unset; ``score`` and
# ``instance_3d_score`` are NaN when unset. ``pts3d_start`` / ``pts3d_end`` are a
# half-open range into ``points_3d`` (``pts3d_predicted`` == 0) or ``pred_points_3d``
# (== 1), or -1 / -1 when the group has no 3D instance. ``member_start`` /
# ``member_end`` are a half-open range into ``instance_group_members``.
INSTANCE_GROUP_DTYPE = np.dtype(
    [
        ("identity_idx", "i4"),
        ("score", "f8"),
        ("instance_3d_score", "f8"),
        ("pts3d_start", "i8"),
        ("pts3d_end", "i8"),
        ("pts3d_predicted", "u1"),
        ("member_start", "u8"),
        ("member_end", "u8"),
    ]
)

# One row per (camera, instance) membership of an InstanceGroup -- the columnarized
# ``camcorder_to_lf_and_inst_idx_map``. ``camera`` indexes CameraGroup.cameras; ``lf``
# indexes Labels.labeled_frames; ``inst`` indexes LabeledFrame.instances.
INSTANCE_GROUP_MEMBER_DTYPE = np.dtype(
    [
        ("camera", "u4"),
        ("lf", "i8"),
        ("inst", "u4"),
    ]
)


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


def _write_source_video_json(source_grp: h5py.Group, source_video_dict: dict) -> None:
    """Write a source video's metadata JSON into its ``source_video`` group.

    Normally stored in the group's ``json`` *attribute*. If the serialized
    metadata would exceed HDF5's 64 KB attribute limit (e.g. a very large
    ``backend_metadata``), it is written to a ``json`` *dataset* in the same group
    instead — datasets have no such limit — and a ``UserWarning`` is emitted.
    `_read_source_video_json` reads the dataset first, falling back to the
    attribute, so the two stay interchangeable.

    Args:
        source_grp: The ``{group}/source_video`` HDF5 group to write into.
        source_video_dict: The source video metadata dict (from `video_to_dict`).
    """
    blob = json.dumps(source_video_dict, separators=(",", ":"))
    if len(blob.encode()) <= METADATA_ATTR_SIZE_LIMIT:
        source_grp.attrs["json"] = blob
        return

    warnings.warn(
        f"Source video metadata ({len(blob.encode())} bytes) exceeds the "
        f"{METADATA_ATTR_SIZE_LIMIT}-byte HDF5 attribute limit; storing it in a "
        f"'{source_grp.name}/json' dataset instead of the attribute. Readers older "
        "than this version will not find the source video metadata.",
        stacklevel=2,
    )
    if "json" in source_grp:
        del source_grp["json"]
    source_grp.create_dataset("json", data=np.bytes_(blob))


def _read_source_video_json(source_grp: h5py.Group) -> dict:
    """Read a source video's metadata JSON from its ``source_video`` group.

    Reads the ``json`` *dataset* (written by `_write_source_video_json` for
    oversized metadata) when present, otherwise the ``json`` *attribute* (the
    normal case and all legacy files).

    Args:
        source_grp: The ``{group}/source_video`` HDF5 group to read from.

    Returns:
        The parsed source video metadata dict.
    """
    # json.loads accepts the dataset's np.bytes_ scalar and the attribute's str
    # directly, so no explicit decoding is needed.
    if "json" in source_grp:
        return json.loads(source_grp["json"][()])
    return json.loads(source_grp.attrs["json"])


def make_video(
    labels_path: str,
    video_json: dict,
    open_backend: bool = True,
    _hdf5_file: h5py.File | None = None,
    _url_headers: dict[str, str] | None = None,
    _url_stream_mode: str = "blockcache",
    _crop_entry: dict | None = None,
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
        _url_headers: HTTP headers forwarded to the backend (and stored on the
            returned `Video`) when `labels_path`/the video is a remote URL, so the
            construction-time metadata probe and later existence probes are
            authenticated. Private; ignored for local files.
        _url_stream_mode: Remote streaming strategy for a URL-backed video. Private;
            ignored for local files.
        _crop_entry: Optional ``{"crop": [x1, y1, x2, y2], "fill": n}`` record from
            the ``/video_crops`` dataset (see ``read_video_crops``). When present,
            the reconstructed (uncropped) backend is re-wrapped in a
            ``CropVideoBackend`` and the crop is seeded into ``backend_metadata`` so
            both the open and closed paths report the cropped view. Private; mirrors
            the ``_url_headers`` threading pattern.
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

    # Basic path resolution. Keep URLs as strings — wrapping a URL in ``Path``
    # would collapse the ``//`` after the scheme (e.g. ``https://`` -> ``https:/``),
    # corrupting the URL for embedded videos loaded from a remote ``.pkg.slp``.
    from sleap_io.io._remote import _is_url

    video_path = sanitize_filename(video_path)
    if not _is_url(video_path):
        video_path = Path(video_path)

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
                source_video_json = _read_source_video_json(
                    f[f"{dataset}/source_video"]
                )
                source_video = make_video(
                    labels_path,
                    source_video_json,
                    open_backend=open_backend,
                    _hdf5_file=f,
                    _url_headers=_url_headers,
                    _url_stream_mode=_url_stream_mode,
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
                    _url_headers=_url_headers,
                    _url_stream_mode=_url_stream_mode,
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
                _url_headers=_url_headers,
                _url_stream_mode=_url_stream_mode,
            )

        # Legacy compatibility: if original_video exists but source_video doesn't,
        # use original_video as source_video
        if source_video is None and "original_video" in video_json:
            source_video = make_video(
                labels_path,
                video_json["original_video"],
                open_backend=False,  # Original videos are often not available
                _url_headers=_url_headers,
                _url_stream_mode=_url_stream_mode,
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
            if (
                not isinstance(video_path, list)
                and not _is_url(video_path)
                and not is_file_accessible(video_path)
            ):
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
                url_headers=_url_headers,
                url_stream_mode=_url_stream_mode,
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

    # Crop reconstruction (D-126, §6.4): the inner/source backend was just rebuilt
    # from the UNCROPPED ``videos_json`` entry; re-wrap it in a ``CropVideoBackend``
    # (open path) and ALWAYS seed the crop record on ``backend_metadata`` so the
    # closed path reports the cropped shape and ``Video.open()`` can re-wrap.
    if _crop_entry is not None:
        from sleap_io.io.video_reading import CropVideoBackend

        crop = tuple(_crop_entry["crop"])
        fill = _crop_entry.get("fill", 0)
        x1, y1, x2, y2 = crop

        if open_backend and backend is not None:
            # Wrap the freshly reconstructed (uncropped) backend. Each reloaded tile
            # gets its OWN inner decoder that its close() releases (owns_inner=True).
            # We do not reuse source_video.backend here: on reload each tile rebuilds
            # a private source_video.backend, so reusing it would share nothing across
            # tiles yet leak the unowned decoder (it would never be closed). Live
            # decoder sharing for a mosaic is created in-memory via Video.crop and is
            # intentionally not reconstructed on load.
            backend = CropVideoBackend.wrap(inner=backend, crop=crop, fill=fill)

        # Copy before overwriting so a shared dict is never mutated (D-126), then
        # record the cropped shape derived from the uncropped inner/source shape.
        backend_metadata = dict(backend_metadata)
        inner_shape = backend_metadata.get("shape")
        if inner_shape is not None:
            # Preserve the uncropped source shape so a closed re-serialize keeps
            # videos_json describing the full frame (DI-2).
            backend_metadata["source_shape"] = list(inner_shape)
            backend_metadata["shape"] = (
                inner_shape[0],
                y2 - y1,
                x2 - x1,
                inner_shape[3],
            )
        backend_metadata["crop"] = list(crop)
        backend_metadata["crop_fill"] = fill

    video = Video(
        filename=video_path,
        backend=backend,
        backend_metadata=backend_metadata,
        source_video=source_video,
        open_backend=open_backend,
    )
    # Persist the URL auth context on the Video (init=False fields) so existence
    # probes and a later Video.open() reconstruction stay authenticated even
    # after the backend is closed/rebuilt.
    if _url_headers is not None or _url_stream_mode != "blockcache":
        object.__setattr__(video, "_url_headers", _url_headers)
        object.__setattr__(video, "_url_stream_mode", _url_stream_mode)
    return video


def read_videos(
    labels_path: str,
    open_backend: bool = True,
    *,
    _hdf5_file: h5py.File | None = None,
    _url_headers: dict[str, str] | None = None,
    _url_stream_mode: str = "blockcache",
) -> list[Video]:
    """Read `Video` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        open_backend: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (which is left open for the caller
            to close) and threaded into `make_video`; otherwise `labels_path` is
            opened and closed internally. This is a private argument used to thread
            a single open handle through multiple reads.
        _url_headers: HTTP headers forwarded to each video backend when
            `labels_path` is a URL (so the construction-time probe is
            authenticated). Private; ignored for local files.
        _url_stream_mode: Remote streaming strategy for URL-backed videos.
            Private; ignored for local files.

    Returns:
        A list of `Video` objects.
    """
    videos = []

    def _read_videos(f: h5py.File) -> None:
        # Read the per-video crop records once (absent on old/uncropped files).
        video_crops = read_video_crops(labels_path, _hdf5_file=f)
        videos_metadata = f["videos_json"][:]
        for video_index, video_data in enumerate(videos_metadata):
            video_json = json.loads(video_data)
            video = make_video(
                labels_path,
                video_json,
                open_backend=open_backend,
                _hdf5_file=f,
                _url_headers=_url_headers,
                _url_stream_mode=_url_stream_mode,
                _crop_entry=video_crops.get(video_index),
            )
            videos.append(video)

    # Open file once and pass handle to make_video to avoid repeated opens
    # for embedded videos (which would otherwise open the file per video).
    if _hdf5_file is not None:
        _read_videos(_hdf5_file)
    else:
        with h5py.File(labels_path, "r") as f:
            _read_videos(f)
    return videos


def video_to_dict(
    video: Video, labels_path: str | None = None, prefer_metadata: bool = True
) -> dict:
    """Convert a `Video` object to a JSON-compatible dictionary.

    Args:
        video: A `Video` object to convert.
        labels_path: Path to the labels file being written. Used to determine if the
            video should use a self-reference (".") or external reference.
        prefer_metadata: If `True` (the default), serialize an uncropped video's
            shape/grayscale/fps from `video.backend_metadata` when recorded there (e.g.
            a video loaded from a `.slp`) instead of querying the live backend. For an
            open `MediaVideo` the backend read decodes a frame to recompute the shape
            (and with `keep_open` leaves a resident decoder), so this avoids needless
            decoding. Set to `False` to always read shape/grayscale/fps through the
            live backend.

    Returns:
        A dictionary containing the video metadata.
    """
    from sleap_io.io.video_reading import CropVideoBackend

    video_filename = sanitize_filename(video.filename)
    result = {"filename": video_filename}

    # Crop unwrap (D-123): a cropped Video serializes as its UNCROPPED source
    # backend so ``videos_json`` describes the full frame and old readers never
    # hit a KeyError on an unknown wrapper type. The crop tuple is emitted
    # separately into ``/video_crops`` by ``write_video_crops``; it must NOT
    # enter ``videos_json``. Dispatch on ``type(inner)`` through the existing
    # per-type whitelist below WITHOUT building a throwaway Video (which would
    # trigger ``exists()``/``open()`` side effects, F-ser-5). The uncropped
    # source's shape/grayscale/fps come from the inner backend, not the cropped
    # facade, so ``videos_json`` stays byte-identical to the uncropped save.
    backend = video.backend
    if isinstance(backend, CropVideoBackend):
        inner = backend.inner
        if isinstance(inner, CropVideoBackend):
            # A nested (un-flattened) crop-of-crop cannot be represented by the
            # single-crop-per-video /video_crops schema. wrap() only nests when
            # fills differ or the outer rect exceeds the inner frame; flatten it
            # (matching fills, in-bounds rect) or materialize before saving.
            raise ValueError(
                "Cannot serialize a nested crop-of-crop video: the /video_crops "
                "format stores a single crop per video. Flatten the crop (use "
                "matching fills and an in-bounds region) or materialize it with "
                "transform_labels before saving."
            )
        backend = inner
        # Mirror Video._get_shape(): never let a missing/unreadable inner file crash
        # the save. Fall back to the recorded uncropped source shape (videos_json must
        # describe the full frame); the per-type writers below tolerate shape=None,
        # exactly as the uncropped Video.shape==None path does.
        try:
            shape = inner.shape
        except Exception:
            src_shape = video.backend_metadata.get("source_shape")
            if src_shape is None and video.source_video is not None:
                src_shape = video.source_video.shape
            shape = tuple(src_shape) if src_shape is not None else None
        grayscale = inner.grayscale
        fps = inner.fps
    else:
        # Uncropped path. When ``prefer_metadata`` is set and the shape was recorded
        # in ``backend_metadata`` (e.g. a video loaded from a ``.slp``), serialize from
        # that metadata instead of querying the live backend: for an open ``MediaVideo``
        # the facade read below decodes a frame to recompute the shape (and with
        # ``keep_open`` leaves a resident decoder), while the recorded values are
        # identical and free. Otherwise read through the Video facade exactly as before
        # so uncropped serialization stays golden byte-identical.
        meta = video.backend_metadata
        if prefer_metadata and meta.get("shape") is not None:
            shape = meta["shape"]
            grayscale = meta["grayscale"] if "grayscale" in meta else shape[-1] == 1
            # The grayscale flag can be flipped after load (it updates the recorded
            # ``grayscale`` but not the recorded ``shape``). Serializing the two
            # independently would emit a self-inconsistent entry (e.g. a 3-channel
            # shape with ``grayscale=true``); on reload with the backend file
            # missing the channel count is wrong and the flip is silently lost.
            # Keep the emitted channel count consistent with the flag for the
            # unambiguous grayscale case (grayscale always reads as 1 channel).
            if grayscale and shape is not None and len(shape) >= 1 and shape[-1] != 1:
                shape = tuple(shape[:-1]) + (1,)
            fps = meta.get("fps")
        else:
            shape = video.shape
            grayscale = video.grayscale
            fps = video.fps

    # Add backend metadata
    if video.backend is None:
        # Copy backend_metadata to avoid mutating the original
        result["backend"] = video.backend_metadata.copy()
        # Ensure filename is always present in backend metadata for compatibility
        # with make_video() which expects backend["filename"] to exist
        if "filename" not in result["backend"]:
            result["backend"]["filename"] = video_filename
        # Closed cropped path: the copied metadata carries the CROPPED shape plus
        # a crop record. Restore the UNCROPPED source shape so videos_json describes
        # the full frame (old-reader guarantee), then drop the crop keys (the crop
        # rides /video_crops). Recover the source shape from a live source_video or
        # the recorded "source_shape"; refuse to emit a self-inconsistent entry
        # (uncropped frame with cropped dims) when neither is available.
        if "crop" in result["backend"]:
            src_shape = None
            if video.source_video is not None and video.source_video.shape is not None:
                src_shape = list(video.source_video.shape)
            elif result["backend"].get("source_shape") is not None:
                src_shape = list(result["backend"]["source_shape"])
            if src_shape is None:
                raise ValueError(
                    "Cannot serialize closed cropped video: the uncropped source "
                    "shape is unavailable (no source_video and no recorded "
                    "source_shape), so videos_json cannot describe the full frame."
                )
            result["backend"]["shape"] = src_shape
        for key in ("crop", "crop_fill", "source_shape"):
            result["backend"].pop(key, None)
    elif type(backend) is MediaVideo:
        result["backend"] = {
            "type": "MediaVideo",
            "shape": shape,
            "filename": video_filename,
            "grayscale": grayscale,
            "bgr": True,
            "dataset": "",
            "input_format": "",
            "fps": fps,
        }
    elif type(backend) is HDF5Video:
        # Determine if we should use self-reference or external reference
        use_self_reference = (
            backend.has_embedded_images
            and labels_path is not None
            and Path(sanitize_filename(video.filename)).resolve()
            == Path(sanitize_filename(labels_path)).resolve()
        )

        result["backend"] = {
            "type": "HDF5Video",
            "shape": shape,
            "filename": ("." if use_self_reference else video_filename),
            "dataset": backend.dataset,
            "input_format": backend.input_format,
            "convert_range": False,
            "has_embedded_images": backend.has_embedded_images,
            "grayscale": grayscale,
            "fps": fps,
        }
    elif type(backend) is ImageVideo:
        if shape is not None:
            height, width, channels = shape[1:4]
        else:
            height, width, channels = None, None, 3
        result["backend"] = {
            "type": "ImageVideo",
            "shape": shape,
            "filename": sanitize_filename(backend.filename[0]),
            "filenames": sanitize_filename(backend.filename),
            "height_": height,
            "width_": width,
            "channels_": channels,
            "grayscale": grayscale,
            "fps": fps,
        }
    elif type(backend) is TiffVideo:
        result["backend"] = {
            "type": "TiffVideo",
            "shape": shape,
            "filename": video_filename,
            "grayscale": grayscale,
            "keep_open": backend.keep_open,
            "format": backend.format,
            "fps": fps,
        }

    # Add source_video metadata if present
    if hasattr(video, "source_video") and video.source_video is not None:
        result["source_video"] = video_to_dict(
            video.source_video, labels_path, prefer_metadata=prefer_metadata
        )

    # Note: original_video is now a computed property derived from source_video,
    # so we don't store it. Legacy files with original_video are handled on load.

    return result


def read_video_crops(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> dict[int, dict]:
    """Read the top-level ``/video_crops`` dataset keyed by video index.

    The dataset is a single JSON string holding a list of
    ``{"video": i, "crop": [x1, y1, x2, y2], "fill": n}`` entries (one per
    cropped video). It is absent on old/uncropped files, in which case an empty
    mapping is returned so the loader falls back to the uncropped source.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open ``h5py.File`` handle to read from. If
            provided, the data is read from this handle (left open for the
            caller to close); otherwise ``labels_path`` is opened and closed
            internally. Private, mirrors the other ``read_*`` helpers.

    Returns:
        A mapping ``{video_index: {"crop": [...], "fill": n}}`` keyed by int.
        Empty if the ``/video_crops`` dataset is absent.
    """
    try:
        raw = read_hdf5_dataset(labels_path, "video_crops", _hdf5_file=_hdf5_file)
    except KeyError:
        return {}
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode()
    elif isinstance(raw, np.ndarray):
        raw = raw.item()
        if isinstance(raw, (bytes, np.bytes_)):
            raw = raw.decode()
    return {int(entry["video"]): entry for entry in json.loads(raw)}


def write_video_crops(labels_path: str, labels: Labels) -> None:
    """Write the top-level ``/video_crops`` JSON dataset for cropped videos.

    Emits one ``{"video": i, "crop": [x1, y1, x2, y2], "fill": n}`` entry for
    each cropped video in ``labels.videos`` (open or closed). The dataset is
    omitted entirely when no video is cropped, so uncropped files stay
    byte-identical to today and old readers never see an unknown dataset.

    When a cropped video's frames are embedded, the UNCROPPED source frames are
    stored (see ``embed_videos``) and the crop is recorded here, so the reader
    applies it exactly once over the full-frame embedded data.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A ``Labels`` whose videos may carry virtual crops.
    """
    crops = []
    for i, video in enumerate(labels.videos):
        rect = video._crop_tuple()
        if rect is None:
            continue
        crops.append({"video": i, "crop": list(rect), "fill": video._crop_fill()})

    if crops:
        write_hdf5_dataset(
            labels_path,
            "video_crops",
            np.bytes_(json.dumps(crops, separators=(",", ":"))),
        )


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


def _resolve_embed_format(video: Video, requested_format: str) -> str:
    """Determine the on-disk image format to store a video's embedded frames in.

    For an `ImageVideo` whose source files are already-compressed PNG/JPEG, the
    frames are byte-copied verbatim into the package (see
    `process_and_embed_frames`), so the stored format follows the *source* rather
    than `requested_format` -- re-encoding an already-compressed image would only
    inflate size (PNG) or add artifacts (JPEG) for no benefit. All other sources
    use `requested_format` (their frames are decoded and re-encoded).

    Args:
        video: The source `Video` whose frames will be embedded.
        requested_format: The format requested by the caller ("png", "jpg", or
            "hdf5").

    Returns:
        The image format string to store for this video's frames. This is the
        normalized source extension ("png" or "jpg") for a byte-copyable
        `ImageVideo`, otherwise `requested_format`.
    """
    from sleap_io.io.video_reading import ImageVideo

    if requested_format == "hdf5":
        return "hdf5"
    backend = video.backend
    if isinstance(backend, ImageVideo):
        fnames = backend.filename
        first = fnames[0] if isinstance(fnames, list) and fnames else fnames
        if isinstance(first, (str, Path)):
            ext = Path(first).suffix.lower().lstrip(".")
            if ext in ("png", "jpg", "jpeg"):
                return "jpg" if ext == "jpeg" else ext
    return requested_format


def process_and_embed_frames(
    labels_path: str,
    frames_metadata: list[dict],
    image_format: str = "png",
    fixed_length: bool = True,
    verbose: bool = True,
    plugin: str | None = None,
    progress_callback: Callable[[int, int, str], bool] | None = None,
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
        progress_callback: Optional callback function called during embedding with
            `(current, total, phase)` arguments (1-based current index and total
            count for the phase). ``phase`` is ``"embed"`` while frames are loaded/
            encoded/byte-copied into memory and ``"write"`` while the accumulated
            bytes are written to the HDF5 file. If it returns `False`, the operation
            is cancelled and `ExportCancelled` is raised. When provided, tqdm progress
            bars are disabled in favor of the callback.

            .. note::
                The ``phase`` argument (and the separate ``"write"`` phase) is a
                breaking change from the previous ``(current, total)`` signature.

    Returns:
        A dictionary mapping original Video objects to their embedded versions.

    Raises:
        ExportCancelled: If the progress_callback returns `False`.
    """
    # Determine which plugin to use for encoding
    from sleap_io.io.video_reading import (
        CropVideoBackend,
        ImageVideo,
        get_default_image_plugin,
    )

    if plugin is None:
        plugin = get_default_image_plugin()
    if plugin is None:
        # Auto-detect: prefer opencv, fallback to imageio
        plugin = "opencv" if "cv2" in sys.modules else "imageio"

    # Initialize a dictionary to store data by group
    data_by_group: dict[str, dict] = {}
    total_frames = len(frames_metadata)

    # Per-video (per-group) stored format: an ImageVideo of PNG/JPEG is byte-copied
    # verbatim so its group stores the source format; everything else re-encodes to
    # `image_format`. Decided once up front so a group's format is stable even when a
    # stray frame falls back to the decode path.
    group_format: dict[str, str] = {}
    for frame_meta in frames_metadata:
        grp = frame_meta["group"]
        if grp not in group_format:
            group_format[grp] = _resolve_embed_format(frame_meta["video"], image_format)

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

        # Format this group's frames are stored in (see `_resolve_embed_format`).
        grp_format = group_format[group]

        # Initialize group data structure if this is the first frame for this group
        if group not in data_by_group:
            data_by_group[group] = {
                "video": video,  # All frames in a group are from the same video
                "frame_inds": [],
                "imgs_data": [],
                "channel_order": None,  # Track channel order: "RGB" or "BGR"
                "image_format": grp_format,
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
                    if not progress_callback(i + 1, total_frames, "embed"):
                        raise ExportCancelled("Export cancelled by user")
                continue

        # Fast path 2: ImageVideo byte-copy. When embedding an image sequence whose
        # files are already PNG/JPEG, copy the encoded bytes verbatim -- no decode/
        # re-encode. This skips a lossy re-encode (JPEG) and stores the smaller
        # source bytes. Guard that this frame's on-disk format matches the group's
        # stored format (grp_format, decided from the first file); a stray frame with
        # a different extension falls through to the decode path and is re-encoded.
        if isinstance(video.backend, ImageVideo) and grp_format != "hdf5":
            fname = video.backend.filename[frame_idx]
            ext = (
                Path(fname).suffix.lower().lstrip(".")
                if isinstance(fname, (str, Path))
                else ""
            )
            frame_format = "jpg" if ext == "jpeg" else ext
            if frame_format == grp_format:
                raw_bytes = video.backend.get_frame_raw_bytes(frame_idx)
                if raw_bytes is not None:
                    data_by_group[group]["imgs_data"].append(raw_bytes)
                    data_by_group[group]["frame_inds"].append(frame_idx)
                    # Bytes copied from an ImageVideo decode back to RGB (see
                    # ImageVideo.get_frame_raw_bytes / _read_frame).
                    if data_by_group[group]["channel_order"] is None:
                        data_by_group[group]["channel_order"] = "RGB"

                    if progress_callback is not None:
                        if not progress_callback(i + 1, total_frames, "embed"):
                            raise ExportCancelled("Export cancelled by user")
                    continue

        # Slow path: Load and encode the frame. For a virtually-cropped video,
        # embed the UNCROPPED source frame; the crop is preserved separately in
        # /video_crops and re-applied on load, so it is baked exactly once. This
        # matches the crop-over-embedded path (which already embeds the uncropped
        # source) and keeps the reloaded video a CropVideoBackend over full frames.
        try:
            if isinstance(video.backend, CropVideoBackend):
                frame = video.backend.inner.get_frame(frame_idx)
            else:
                frame = video[frame_idx]
        except IndexError as e:
            # Surface which video the missing frame belongs to. The bare backend
            # error only reports the frame index, which is not actionable when a
            # project has more than one video.
            raise IndexError(
                f"Frame index {frame_idx} out of range for video "
                f"{frame_meta['video_ind']} ({video.filename}), which has "
                f"{len(video)} frames."
            ) from e

        # Encode the frame to this group's format.
        if grp_format == "hdf5":
            img_data = frame
            channel_order = "RGB"  # HDF5 format stores as-is (RGB)
        else:
            if plugin == "opencv":
                img_data = np.squeeze(cv2.imencode("." + grp_format, frame)[1]).astype(
                    "int8"
                )
                channel_order = "BGR"  # OpenCV encodes in BGR
            else:  # imageio
                if frame.shape[-1] == 1:
                    frame = frame.squeeze(axis=-1)
                img_data = np.frombuffer(
                    iio.imwrite("<bytes>", frame, extension="." + grp_format),
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
            if not progress_callback(i + 1, total_frames, "embed"):
                raise ExportCancelled("Export cancelled by user")

    # Write all frame data to the HDF5 file.
    replaced_videos = {}
    total_to_write = sum(len(d["imgs_data"]) for d in data_by_group.values())
    write_bar = (
        tqdm(total=total_to_write, desc="Writing frames", disable=not use_tqdm)
        if use_tqdm
        else None
    )
    written = 0

    def _report_write(n: int) -> None:
        """Advance the write-phase progress bar / callback by ``n`` frames."""
        nonlocal written
        written += n
        if write_bar is not None:
            write_bar.update(n)
        if progress_callback is not None:
            if not progress_callback(written, total_to_write, "write"):
                raise ExportCancelled("Export cancelled by user")

    with h5py.File(labels_path, "a") as f:
        for group, data in data_by_group.items():
            video = data["video"]
            frame_inds = data["frame_inds"]
            imgs_data = data["imgs_data"]
            grp_format = data["image_format"]

            if grp_format == "hdf5":
                # Raw pixel arrays are genuinely compressible, so keep gzip; written
                # in one shot (not row-by-row), so no read-modify-write amplification.
                f.create_dataset(
                    f"{group}/video", data=imgs_data, compression="gzip", chunks=True
                )
                ds = f[f"{group}/video"]
                _report_write(len(imgs_data))
            else:
                # Encoded image bytes (PNG/JPEG) are already entropy-coded: gzip gains
                # almost nothing on the bytes themselves (it only squeezes the
                # fixed-length zero padding) while costing significant CPU. More
                # importantly, gzip forces CHUNKED storage, and h5py auto-chunks a tall
                # block (~100+ rows); the row-by-row writes below then trigger repeated
                # read-modify-recompress of each chunk plus chunk-cache thrashing --
                # the pathology that made large embeds take tens of minutes. Storing
                # uncompressed (contiguous) makes each row a direct offset write.
                if fixed_length:
                    img_bytes_len = 0
                    for img in imgs_data:
                        img_bytes_len = max(img_bytes_len, len(img))
                    ds = f.create_dataset(
                        f"{group}/video",
                        shape=(len(imgs_data), img_bytes_len),
                        dtype="int8",
                    )
                    for i, img in enumerate(imgs_data):
                        ds[i, : len(img)] = img
                        _report_write(1)
                else:
                    ds = f.create_dataset(
                        f"{group}/video",
                        shape=(len(imgs_data),),
                        dtype=h5py.special_dtype(vlen=np.dtype("int8")),
                    )
                    for i, img in enumerate(imgs_data):
                        ds[i] = img
                        _report_write(1)

            # Store metadata
            ds.attrs["format"] = grp_format
            ds.attrs["channel_order"] = data["channel_order"]
            # Embedded attrs describe the UNCROPPED frame for a cropped video
            # (the embedded pixels are the full source frame).
            if isinstance(video.backend, CropVideoBackend):
                video_shape = video.backend.inner.shape
            else:
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
            embedded_backend = VideoBackend.from_filename(
                labels_path,
                dataset=f"{group}/video",
                grayscale=video.grayscale,
                keep_open=False,
            )
            if isinstance(video.backend, CropVideoBackend):
                # The embedded frames are UNCROPPED; re-wrap so the crop survives
                # (write_video_crops emits it from labels.videos and reload
                # re-applies it over the full-frame embedded data).
                crop = video.backend.crop
                fill = video.backend.fill
                x1, y1, x2, y2 = crop
                src_shape = embedded_backend.shape
                embedded_video = Video(
                    filename=labels_path,
                    backend=CropVideoBackend.wrap(
                        inner=embedded_backend, crop=crop, fill=fill
                    ),
                    source_video=source_video,
                )
                embedded_video.backend_metadata = {
                    "crop": list(crop),
                    "crop_fill": fill,
                    "source_shape": list(src_shape) if src_shape is not None else None,
                    "shape": (src_shape[0], y2 - y1, x2 - x1, src_shape[3])
                    if src_shape is not None
                    else None,
                }
            else:
                embedded_video = Video(
                    filename=labels_path,
                    backend=embedded_backend,
                    source_video=source_video,
                )

            # Store source video metadata
            grp = f.require_group(f"{group}/source_video")
            _write_source_video_json(grp, video_to_dict(source_video, labels_path))

            # Store the embedded video for return
            replaced_videos[video] = embedded_video

    if write_bar is not None:
        write_bar.close()

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
        _write_source_video_json(source_grp, video_to_dict(source_video, labels_path))

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
    progress_callback: Callable[[int, int, str], bool] | None = None,
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
        progress_callback: Optional callback function called during embedding with
            `(current, total, phase)` arguments, where ``phase`` is ``"embed"``
            (frames loaded/encoded/byte-copied) or ``"write"`` (bytes flushed to the
            HDF5 file). If it returns `False`, the operation is cancelled and
            `ExportCancelled` is raised. The ``phase`` argument is a breaking change
            from the previous ``(current, total)`` signature.

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
    progress_callback: Callable[[int, int, str], bool] | None = None,
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
        progress_callback: Optional callback function called during embedding with
            `(current, total, phase)` arguments, where ``phase`` is ``"embed"``
            (frames loaded/encoded/byte-copied) or ``"write"`` (bytes flushed to the
            HDF5 file). If it returns `False`, the operation is cancelled and
            `ExportCancelled` is raised. The ``phase`` argument is a breaking change
            from the previous ``(current, total)`` signature.
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
    prefer_metadata: bool = True,
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
        prefer_metadata: If `True` (the default), serialize each uncropped video's
            shape/grayscale/fps from its `backend_metadata` when recorded there instead
            of querying the live backend (avoids decoding frames just to recompute
            already-known metadata). Set to `False` to always read through the live
            backend. See `video_to_dict`.
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

    from sleap_io.io.video_reading import CropVideoBackend

    # First determine which videos need embedding
    for video_ind, video in enumerate(videos):
        # Crop-over-embedded (D-125): a virtual crop over an embedded HDF5Video
        # must route through the embed machinery, not the plain-external else
        # branch. Test embedded-ness / collect source_inds from the INNER backend;
        # the crop itself still rides /video_crops.
        inner_backend = (
            video.backend.inner
            if isinstance(video.backend, CropVideoBackend)
            else video.backend
        )
        # Check if video has an open backend with embedded images
        has_backend_with_embedded = (
            type(inner_backend) is HDF5Video and inner_backend.has_embedded_images
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
                    # Collect information for embedding (source_inds live on the
                    # inner backend for a crop-over-embedded video, D-125). Embed
                    # ``video`` itself (the crop facade): process_and_embed_frames
                    # special-cases a CropVideoBackend and reads the UNCROPPED frame
                    # straight from the embedded inner, so the embedded data + its
                    # videos_json entry describe the full frame WITHOUT touching the
                    # (possibly missing external) source_video. The crop rides
                    # /video_crops and is re-applied on read.
                    embed_target = video
                    frames_to_embed = [
                        (embed_target, frame_idx)
                        for frame_idx in inner_backend.source_inds
                    ]
                    videos_to_embed.append((video_ind, embed_target, frames_to_embed))
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
        video_json = video_to_dict(video, labels_path, prefer_metadata=prefer_metadata)
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
                elif (
                    pre_embed_video is not None
                    and pre_embed_video.source_video is not None
                ):
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
                    source_json = video_to_dict(
                        source_to_save, labels_path, prefer_metadata=prefer_metadata
                    )
                    _write_source_video_json(source_grp, source_json)


def read_tracks(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> list[Track]:
    """Read `Track` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of `Track` objects.
    """
    tracks = [
        json.loads(x)
        for x in read_hdf5_dataset(labels_path, "tracks_json", _hdf5_file=_hdf5_file)
    ]
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


def read_identities(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> list[Identity]:
    """Read the identity catalog from a SLEAP labels file.

    Identities are stored in the optional ``/identity`` group (SLP format 2.5+) as
    a native ``name`` string dataset plus an optional entity-attribute-value
    metadata table (``meta_owner`` / ``meta_key`` / ``meta_val``, omitted when no
    identity carries metadata). Absent in older files, in which case an empty list
    is returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A list of `Identity` objects in catalog order.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "identity" not in f or "name" not in f["identity"]:
            return []
        group = f["identity"]
        names = [
            n.decode() if isinstance(n, bytes) else str(n) for n in group["name"][:]
        ]
        metadata: list[dict[str, str]] = [{} for _ in names]
        if "meta_owner" in group:
            owners = group["meta_owner"][:]
            keys = group["meta_key"][:]
            vals = group["meta_val"][:]
            for owner, key, val in zip(owners, keys, vals):
                k = key.decode() if isinstance(key, bytes) else str(key)
                v = val.decode() if isinstance(val, bytes) else str(val)
                metadata[int(owner)][k] = v
    return [Identity(name=name, metadata=meta) for name, meta in zip(names, metadata)]


def write_identities(labels_path: str, identities: list[Identity]) -> None:
    """Write the identity catalog to a SLEAP labels file.

    Creates the ``/identity`` group (SLP format 2.5+) holding a native ``name``
    string dataset (one per identity) plus, only when any identity carries
    metadata, a columnar entity-attribute-value metadata table
    (``meta_owner`` / ``meta_key`` / ``meta_val``). No-op if there are no
    identities.

    Args:
        labels_path: A string path to the SLEAP labels file.
        identities: A list of `Identity` objects to store, in catalog order.
    """
    if not identities:
        return

    str_dtype = h5py.string_dtype(encoding="utf-8")
    names = np.array([ident.name for ident in identities], dtype=object)

    meta_owner: list[int] = []
    meta_key: list[str] = []
    meta_val: list[str] = []
    for idx, ident in enumerate(identities):
        for key, value in ident.metadata.items():
            meta_owner.append(idx)
            meta_key.append(str(key))
            meta_val.append(str(value))

    with h5py.File(labels_path, "a") as f:
        group = f.require_group("identity")
        group.create_dataset("name", data=names, dtype=str_dtype, maxshape=(None,))
        if meta_owner:
            group.create_dataset(
                "meta_owner",
                data=np.array(meta_owner, dtype="i4"),
                maxshape=(None,),
                compression="gzip",
            )
            group.create_dataset(
                "meta_key",
                data=np.array(meta_key, dtype=object),
                dtype=str_dtype,
                maxshape=(None,),
                compression="gzip",
            )
            group.create_dataset(
                "meta_val",
                data=np.array(meta_val, dtype=object),
                dtype=str_dtype,
                maxshape=(None,),
                compression="gzip",
            )


def read_categories(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> list[Category]:
    """Read the category catalog from a SLEAP labels file.

    Categories are stored in the optional ``/categories`` group (SLP format 2.7+)
    as a native ``name`` string dataset plus an optional entity-attribute-value
    metadata table (``meta_owner`` / ``meta_key`` / ``meta_val``, omitted when no
    category carries metadata). This is a fully self-contained mirror of the
    ``/identity`` catalog (`read_identities`). Absent in older files, in which case
    an empty list is returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A list of `Category` objects in catalog order.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "categories" not in f or "name" not in f["categories"]:
            return []
        group = f["categories"]
        names = [
            n.decode() if isinstance(n, bytes) else str(n) for n in group["name"][:]
        ]
        metadata: list[dict[str, str]] = [{} for _ in names]
        if "meta_owner" in group:
            owners = group["meta_owner"][:]
            keys = group["meta_key"][:]
            vals = group["meta_val"][:]
            for owner, key, val in zip(owners, keys, vals):
                k = key.decode() if isinstance(key, bytes) else str(key)
                v = val.decode() if isinstance(val, bytes) else str(val)
                metadata[int(owner)][k] = v
    return [Category(name=name, metadata=meta) for name, meta in zip(names, metadata)]


def write_categories(labels_path: str, categories: list[Category]) -> None:
    """Write the category catalog to a SLEAP labels file.

    Creates the ``/categories`` group (SLP format 2.7+) holding a native ``name``
    string dataset (one per category) plus, only when any category carries
    metadata, a columnar entity-attribute-value metadata table
    (``meta_owner`` / ``meta_key`` / ``meta_val``). A fully self-contained mirror
    of `write_identities`. No-op if there are no categories.

    Args:
        labels_path: A string path to the SLEAP labels file.
        categories: A list of `Category` objects to store, in catalog order.
    """
    if not categories:
        return

    str_dtype = h5py.string_dtype(encoding="utf-8")
    names = np.array([cat.name for cat in categories], dtype=object)

    meta_owner: list[int] = []
    meta_key: list[str] = []
    meta_val: list[str] = []
    for idx, cat in enumerate(categories):
        for key, value in cat.metadata.items():
            meta_owner.append(idx)
            meta_key.append(str(key))
            meta_val.append(str(value))

    with h5py.File(labels_path, "a") as f:
        group = f.require_group("categories")
        group.create_dataset("name", data=names, dtype=str_dtype, maxshape=(None,))
        if meta_owner:
            group.create_dataset(
                "meta_owner",
                data=np.array(meta_owner, dtype="i4"),
                maxshape=(None,),
                compression="gzip",
            )
            group.create_dataset(
                "meta_key",
                data=np.array(meta_key, dtype=object),
                dtype=str_dtype,
                maxshape=(None,),
                compression="gzip",
            )
            group.create_dataset(
                "meta_val",
                data=np.array(meta_val, dtype=object),
                dtype=str_dtype,
                maxshape=(None,),
                compression="gzip",
            )


def read_event_types(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> list[EventType]:
    """Read the event-type catalog from a SLEAP labels file.

    Event types are stored in the optional ``/event_types`` group (SLP format 2.6+)
    as a native ``name`` string dataset plus an optional ``description`` string
    dataset (omitted when every type's description is empty) and an optional
    entity-attribute-value metadata table (``meta_owner`` / ``meta_key`` /
    ``meta_val``, omitted when no type carries metadata). Absent in older files, in
    which case an empty list is returned. Mirrors `read_identities`.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A list of `EventType` objects in catalog order.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "event_types" not in f or "name" not in f["event_types"]:
            return []
        group = f["event_types"]
        names = [
            n.decode() if isinstance(n, bytes) else str(n) for n in group["name"][:]
        ]
        if "description" in group:
            descriptions = [
                d.decode() if isinstance(d, bytes) else str(d)
                for d in group["description"][:]
            ]
        else:
            descriptions = ["" for _ in names]
        metadata: list[dict[str, str]] = [{} for _ in names]
        if "meta_owner" in group:
            owners = group["meta_owner"][:]
            keys = group["meta_key"][:]
            vals = group["meta_val"][:]
            for owner, key, val in zip(owners, keys, vals):
                k = key.decode() if isinstance(key, bytes) else str(key)
                v = val.decode() if isinstance(val, bytes) else str(val)
                metadata[int(owner)][k] = v
    return [
        EventType(name=name, description=desc, metadata=meta)
        for name, desc, meta in zip(names, descriptions, metadata)
    ]


def write_event_types(labels_path: str, event_types: list[EventType]) -> None:
    """Write the event-type catalog to a SLEAP labels file.

    Creates the ``/event_types`` group (SLP format 2.6+) holding a native ``name``
    string dataset (one per event type), plus a ``description`` string dataset only
    when any type has a non-empty description, plus -- only when any type carries
    metadata -- a columnar entity-attribute-value metadata table (``meta_owner`` /
    ``meta_key`` / ``meta_val``). No-op if there are no event types. Mirrors
    `write_identities`.

    Args:
        labels_path: A string path to the SLEAP labels file.
        event_types: A list of `EventType` objects to store, in catalog order.
    """
    if not event_types:
        return

    str_dtype = h5py.string_dtype(encoding="utf-8")
    names = np.array([et.name for et in event_types], dtype=object)
    has_description = any(et.description for et in event_types)

    meta_owner: list[int] = []
    meta_key: list[str] = []
    meta_val: list[str] = []
    for idx, et in enumerate(event_types):
        for key, value in et.metadata.items():
            meta_owner.append(idx)
            meta_key.append(str(key))
            meta_val.append(str(value))

    with h5py.File(labels_path, "a") as f:
        group = f.require_group("event_types")
        group.create_dataset("name", data=names, dtype=str_dtype, maxshape=(None,))
        if has_description:
            descriptions = np.array(
                [et.description for et in event_types], dtype=object
            )
            group.create_dataset(
                "description",
                data=descriptions,
                dtype=str_dtype,
                maxshape=(None,),
            )
        if meta_owner:
            group.create_dataset(
                "meta_owner",
                data=np.array(meta_owner, dtype="i4"),
                maxshape=(None,),
                compression="gzip",
            )
            group.create_dataset(
                "meta_key",
                data=np.array(meta_key, dtype=object),
                dtype=str_dtype,
                maxshape=(None,),
                compression="gzip",
            )
            group.create_dataset(
                "meta_val",
                data=np.array(meta_val, dtype=object),
                dtype=str_dtype,
                maxshape=(None,),
                compression="gzip",
            )


def read_events(
    labels_path: str,
    videos: list[Video],
    event_types: list[EventType],
    tracks: list[Track],
    identities: list[Identity],
    *,
    _hdf5_file: h5py.File | None = None,
) -> list[Event]:
    """Read frame-spanning event annotations from a SLEAP labels file.

    Events are stored in the optional ``/events`` group (SLP format 2.6+) as a
    columnar struct-of-arrays (one dataset per column, mirroring ``/bboxes``) plus a
    ragged CSR pair for the optional framewise `PredictedEvent.scores` traces (a flat
    ``scores`` float32 dataset + an ``score_offsets`` int64 dataset of length
    ``n_events + 1``; row ``i``'s trace is ``scores[off[i]:off[i+1]]``, a zero-length
    slice meaning no trace). Absent in older files, in which case an empty list is
    returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of `Video` objects for relinking (indexed by the ``video``
            column; ``-1`` means none).
        event_types: List of `EventType` objects for relinking (indexed by the
            ``type`` column; ``-1`` means none).
        tracks: List of `Track` objects for relinking subject/target participants of
            kind ``1``.
        identities: List of `Identity` objects for relinking subject/target
            participants of kind ``2``.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A list of `Event` objects (``UserEvent`` / ``PredictedEvent``) in stored
        order. Empty if no events are stored.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "events" not in f:
            return []
        grp = f["events"]
        video_arr = grp["video"][:]
        start_arr = grp["start_frame"][:]
        end_arr = grp["end_frame"][:]
        type_arr = grp["type"][:]
        subject_kind_arr = grp["subject_kind"][:]
        target_kind_arr = grp["target_kind"][:]
        subject_idx_arr = grp["subject_idx"][:]
        target_idx_arr = grp["target_idx"][:]
        is_predicted_arr = grp["is_predicted"][:]
        score_arr = grp["score"][:] if "score" in grp else None
        name_arr = grp["name"][:]
        source_arr = grp["source"][:]
        scores_flat = grp["scores"][:] if "scores" in grp else None
        score_offsets = grp["score_offsets"][:] if "score_offsets" in grp else None

        n = len(video_arr)
        # Reconstruct per-event metadata from the optional EAV table.
        metadata: list[dict[str, str]] = [{} for _ in range(n)]
        if "meta_owner" in grp:
            owners = grp["meta_owner"][:]
            keys = grp["meta_key"][:]
            vals = grp["meta_val"][:]
            for owner, key, val in zip(owners, keys, vals):
                k = key.decode() if isinstance(key, bytes) else str(key)
                v = val.decode() if isinstance(val, bytes) else str(val)
                metadata[int(owner)][k] = v

    def _participant(kind: int, idx: int):
        if kind == 1 and 0 <= idx < len(tracks):
            return tracks[idx]
        if kind == 2 and 0 <= idx < len(identities):
            return identities[idx]
        return None

    events: list[Event] = []
    for i in range(n):
        video_idx = int(video_arr[i])
        video = videos[video_idx] if 0 <= video_idx < len(videos) else None

        type_idx = int(type_arr[i])
        if 0 <= type_idx < len(event_types):
            event_type = event_types[type_idx]
        else:
            # Defensive: an event with no catalog entry (should not happen, since
            # the catalog is auto-collected on save). Fall back to an empty type so
            # the (required) ``Event.type`` validator still passes.
            event_type = EventType(name="")

        subject = _participant(int(subject_kind_arr[i]), int(subject_idx_arr[i]))
        target = _participant(int(target_kind_arr[i]), int(target_idx_arr[i]))

        nm = name_arr[i]
        name = nm.decode() if isinstance(nm, bytes) else str(nm)
        src = source_arr[i]
        source = src.decode() if isinstance(src, bytes) else str(src)

        kwargs = dict(
            type=event_type,
            video=video,
            start_frame=int(start_arr[i]),
            end_frame=int(end_arr[i]),
            subject=subject,
            target=target,
            name=name,
            source=source,
            metadata=metadata[i],
        )

        if bool(is_predicted_arr[i]):
            scores = None
            if scores_flat is not None and score_offsets is not None:
                lo = int(score_offsets[i])
                hi = int(score_offsets[i + 1])
                if hi > lo:
                    scores = np.asarray(scores_flat[lo:hi], dtype=np.float32)
            score = None
            if score_arr is not None:
                s = float(score_arr[i])
                if not np.isnan(s):
                    score = s
            events.append(PredictedEvent(scores=scores, score=score, **kwargs))
        else:
            events.append(UserEvent(**kwargs))

    return events


def write_events(
    labels_path: str,
    events: list[Event],
    videos: list[Video],
    event_types: list[EventType],
    tracks: list[Track],
    identities: list[Identity],
) -> None:
    """Write frame-spanning event annotations to a SLEAP labels file.

    Creates the ``/events`` group (SLP format 2.6+) as a columnar struct-of-arrays
    (one dataset per column, built vectorized in a single pass, mirroring
    `write_bboxes`) plus, only when at least one event carries a framewise
    `PredictedEvent.scores` trace, a ragged CSR pair (a flat float32 ``scores``
    dataset + an int64 ``score_offsets`` dataset of length ``n_events + 1``). All
    columns are presence-guarded so unused features cost zero bytes: the scalar
    ``score`` column is omitted when no event sets a scalar score, both trace
    datasets are omitted when no event has a framewise trace, and the whole group is
    omitted (this function returns early) when there are no events.

    Args:
        labels_path: A string path to the SLEAP labels file.
        events: A list of `Event` objects to write.
        videos: List of `Video` objects for index mapping.
        event_types: List of `EventType` objects for index mapping (the ``type``
            column).
        tracks: List of `Track` objects for subject/target index mapping (kind 1).
        identities: List of `Identity` objects for subject/target index mapping
            (kind 2).
    """
    if not events:
        return

    n = len(events)
    video_id = {id(v): i for i, v in enumerate(videos)}
    type_id = {id(et): i for i, et in enumerate(event_types)}
    track_id = {id(t): i for i, t in enumerate(tracks)}
    identity_id = {id(idn): i for i, idn in enumerate(identities)}

    def _kind_idx(participant) -> tuple[int, int]:
        if isinstance(participant, Track):
            return 1, track_id.get(id(participant), -1)
        if isinstance(participant, Identity):
            return 2, identity_id.get(id(participant), -1)
        return 0, -1

    video_arr = np.empty(n, dtype=np.int64)
    start_arr = np.empty(n, dtype=np.int64)
    end_arr = np.empty(n, dtype=np.int64)
    type_arr = np.empty(n, dtype=np.int64)
    subject_kind_arr = np.zeros(n, dtype=np.int8)
    target_kind_arr = np.zeros(n, dtype=np.int8)
    subject_idx_arr = np.full(n, -1, dtype=np.int64)
    target_idx_arr = np.full(n, -1, dtype=np.int64)
    is_predicted_arr = np.zeros(n, dtype=bool)
    score_arr = np.full(n, np.nan, dtype=np.float64)
    names: list[str] = []
    sources: list[str] = []

    score_chunks: list[np.ndarray] = []
    score_lengths = np.zeros(n, dtype=np.int64)

    meta_owner: list[int] = []
    meta_key: list[str] = []
    meta_val: list[str] = []

    any_scalar_score = False
    for i, ev in enumerate(events):
        video_arr[i] = video_id.get(id(ev.video), -1)
        start_arr[i] = ev.start_frame
        end_arr[i] = ev.end_frame
        type_arr[i] = type_id.get(id(ev.type), -1) if ev.type is not None else -1

        subject_kind_arr[i], subject_idx_arr[i] = _kind_idx(ev.subject)
        target_kind_arr[i], target_idx_arr[i] = _kind_idx(ev.target)

        is_predicted = isinstance(ev, PredictedEvent)
        is_predicted_arr[i] = is_predicted
        if is_predicted and ev.score is not None:
            score_arr[i] = ev.score
            any_scalar_score = True
        if is_predicted and ev.scores is not None:
            trace = np.asarray(ev.scores, dtype=np.float32)
            score_chunks.append(trace)
            score_lengths[i] = trace.shape[0]

        names.append(ev.name)
        sources.append(ev.source)
        for key, value in ev.metadata.items():
            meta_owner.append(i)
            meta_key.append(str(key))
            meta_val.append(str(value))

    str_dt = h5py.special_dtype(vlen=str)
    with h5py.File(labels_path, "a") as f:
        grp = f.create_group("events")
        grp.create_dataset("video", data=video_arr)
        grp.create_dataset("start_frame", data=start_arr)
        grp.create_dataset("end_frame", data=end_arr)
        grp.create_dataset("type", data=type_arr)
        grp.create_dataset("subject_kind", data=subject_kind_arr)
        grp.create_dataset("target_kind", data=target_kind_arr)
        grp.create_dataset("subject_idx", data=subject_idx_arr)
        grp.create_dataset("target_idx", data=target_idx_arr)
        grp.create_dataset("is_predicted", data=is_predicted_arr)
        grp.create_dataset("name", data=names, dtype=str_dt)
        grp.create_dataset("source", data=sources, dtype=str_dt)
        # Presence-guarded: scalar scores only when at least one event sets one.
        if any_scalar_score:
            grp.create_dataset("score", data=score_arr)
        # Presence-guarded ragged CSR: framewise traces only when at least one
        # event has one. score_offsets = cumsum of per-event trace lengths.
        if score_chunks:
            scores_flat = np.concatenate(score_chunks)
            score_offsets = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(score_lengths, out=score_offsets[1:])
            grp.create_dataset(
                "scores",
                data=scores_flat,
                dtype=np.float32,
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )
            grp.create_dataset("score_offsets", data=score_offsets)
        # Presence-guarded EAV metadata table.
        if meta_owner:
            grp.create_dataset(
                "meta_owner", data=np.array(meta_owner, dtype="i4"), compression="gzip"
            )
            grp.create_dataset(
                "meta_key",
                data=np.array(meta_key, dtype=object),
                dtype=str_dt,
                compression="gzip",
            )
            grp.create_dataset(
                "meta_val",
                data=np.array(meta_val, dtype=object),
                dtype=str_dt,
                compression="gzip",
            )


def read_identity_links(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> dict[int, dict[int, tuple[int, float | None]]]:
    """Read per-detection identity links from a SLEAP labels file.

    Links are stored in the optional ``/identity/links`` dataset (SLP format 2.5+),
    a structured array of ``(owner_type, owner_id, identity_idx, identity_score)``
    rows joining a detection -- identified by ``owner_type`` (one of the ``OWNER_*``
    codes) and ``owner_id`` -- to an index into the file's identity catalog.
    ``owner_id`` is the per-owner-type positional id (e.g. the global
    ``instance_id`` assigned by `write_lfs` for instance owners), mirroring the
    ``/embeddings`` join. The dataset is absent in older files, in which case an
    empty mapping is returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A dict mapping each ``owner_type`` to a ``{owner_id: (identity_idx,
        identity_score)}`` mapping. ``identity_score`` is ``None`` when it was not
        recorded (stored as NaN). Empty for older files.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "identity" not in f or "links" not in f["identity"]:
            return {}
        data = f["identity"]["links"][:]
    result: dict[int, dict[int, tuple[int, float | None]]] = {}
    for row in data:
        owner_type = int(row["owner_type"])
        score = float(row["identity_score"])
        result.setdefault(owner_type, {})[int(row["owner_id"])] = (
            int(row["identity_idx"]),
            None if np.isnan(score) else score,
        )
    return result


def _write_identity_link_rows(
    labels_path: str, rows: list[tuple[int, int, int, float]]
) -> None:
    """Write ``(owner_type, owner_id, identity_idx, identity_score)`` rows.

    Shared dataset-writing core for the per-detection identity link. The rows can
    be built either by iterating a `Labels` (the eager path, see
    `write_identity_links`) or from a lazy store dict (the lazy save path), keeping
    the on-disk ``/identity/links`` layout identical for both.

    Args:
        labels_path: A string path to the SLEAP labels file.
        rows: A list of ``(owner_type, owner_id, identity_idx, identity_score)``
            tuples. ``identity_score`` is a float (NaN encodes an unrecorded score).
            An empty list is a no-op (no dataset is created).
    """
    if not rows:
        return

    identity_link_dtype = np.dtype(
        [
            ("owner_type", "u1"),
            ("owner_id", "i8"),
            ("identity_idx", "i4"),
            ("identity_score", "f4"),
        ]
    )
    arr = np.array(rows, dtype=identity_link_dtype)
    with h5py.File(labels_path, "a") as f:
        group = f.require_group("identity")
        group.create_dataset("links", data=arr, maxshape=(None,))


def write_identity_links(labels_path: str, labels: Labels) -> None:
    """Write per-detection identity links to a SLEAP labels file.

    Creates the ``/identity/links`` dataset (SLP format 2.5+) as a structured array
    of ``(owner_type, owner_id, identity_idx, identity_score)`` rows for every
    detection carrying an `Identity` registered in ``labels.identities``. Instance,
    mask, centroid, bbox, and ROI owners are written (``OWNER_*``).

    The instance ``owner_id`` matches the global id assigned by `write_lfs`, and the
    mask / centroid / bbox / ROI ``owner_id`` matches the global per-modality list
    index assigned by `write_masks` / `write_centroids` / `write_bboxes` /
    `write_rois` (each enumeration order over frames then the modality; ROIs append
    static ROIs after frame ROIs), so the link is resolved by joining on that id at
    read time. Identity is resolved to a catalog index by object identity (a single
    ``id()``-keyed lookup), so the pass stays O(number of detections).

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the identity links for.
    """
    if not labels.identities:
        return

    id_to_idx = {id(ident): i for i, ident in enumerate(labels.identities)}
    rows = []
    instance_id = 0
    masks: list = []
    centroids: list = []
    bboxes: list = []
    rois: list = []
    for lf in labels:
        for inst in lf:
            identity = getattr(inst, "identity", None)
            if identity is not None:
                idx = id_to_idx.get(id(identity))
                if idx is not None:
                    score = inst.identity_score
                    rows.append(
                        (
                            OWNER_INSTANCE,
                            instance_id,
                            idx,
                            float("nan") if score is None else float(score),
                        )
                    )
            instance_id += 1
        masks.extend(lf.masks)
        centroids.extend(lf.centroids)
        bboxes.extend(lf.bboxes)
        rois.extend(lf.rois)
    # Static ROIs are appended after frame ROIs (matching write_labels' all_rois).
    rois.extend(labels.static_rois)

    rows.extend(_identity_link_rows_for_owner(masks, id_to_idx, OWNER_MASK))
    rows.extend(_identity_link_rows_for_owner(centroids, id_to_idx, OWNER_CENTROID))
    rows.extend(_identity_link_rows_for_owner(bboxes, id_to_idx, OWNER_BBOX))
    rows.extend(_identity_link_rows_for_owner(rois, id_to_idx, OWNER_ROI))
    _write_identity_link_rows(labels_path, rows)


def _identity_link_rows_for_owner(
    anns: list, id_to_idx: dict[int, int], owner_type: int
) -> list[tuple[int, int, int, float]]:
    """Build identity-link rows for an ordered annotation list under one owner type.

    Annotations join on their global per-modality list index (the same enumeration
    as the modality's writer, e.g. `write_masks` / `write_centroids`), so ``anns``
    must be supplied in that exact order. Identity is resolved to a catalog index by
    object identity.

    Args:
        anns: The ordered global annotation list for one modality.
        id_to_idx: Mapping from an `Identity`'s object id to its catalog index.
        owner_type: The ``OWNER_*`` code for this modality (e.g. ``OWNER_MASK``).

    Returns:
        A list of ``(owner_type, owner_id, identity_idx, identity_score)`` tuples.
    """
    rows = []
    for owner_id, ann in enumerate(anns):
        identity = getattr(ann, "identity", None)
        if identity is None:
            continue
        idx = id_to_idx.get(id(identity))
        if idx is None:
            continue
        score = ann.identity_score
        rows.append(
            (
                owner_type,
                owner_id,
                idx,
                float("nan") if score is None else float(score),
            )
        )
    return rows


def _identity_link_rows_from_store(
    store: "LazyDataStore",
) -> list[tuple[int, int, int, float]]:
    """Build ``/identity/links`` rows from a lazy store without materializing.

    The lazy store already holds the per-instance identity links keyed by the
    global ``instance_id`` (the same id space written by the fast path), so the
    rows can be emitted directly as ``OWNER_INSTANCE`` owners without rebuilding
    `Instance` objects. Rows are sorted by ``instance_id`` for deterministic
    output.

    Args:
        store: The `LazyDataStore` backing a lazy `Labels`.

    Returns:
        A list of ``(owner_type, owner_id, identity_idx, identity_score)`` tuples
        matching the format expected by `_write_identity_link_rows`.
    """
    rows = []
    for instance_id in sorted(store._instance_identities):
        identity_idx, score = store._instance_identities[instance_id]
        rows.append(
            (
                OWNER_INSTANCE,
                int(instance_id),
                int(identity_idx),
                float("nan") if score is None else float(score),
            )
        )
    return rows


def read_category_links(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> dict[int, dict[int, tuple[int, float | None]]]:
    """Read per-detection category links from a SLEAP labels file.

    Links are stored in the optional ``/categories/links`` dataset (SLP format
    2.7+), a structured array of ``(owner_type, owner_id, category_idx,
    category_score)`` rows joining a detection -- identified by ``owner_type`` (one
    of the ``OWNER_*`` codes) and ``owner_id`` -- to an index into the file's
    category catalog. A fully self-contained mirror of `read_identity_links`.
    ``owner_id`` is the per-owner-type positional id (e.g. the global
    ``instance_id`` assigned by `write_lfs` for instance owners), mirroring the
    ``/embeddings`` join. The dataset is absent in older files, in which case an
    empty mapping is returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A dict mapping each ``owner_type`` to a ``{owner_id: (category_idx,
        category_score)}`` mapping. ``category_score`` is ``None`` when it was not
        recorded (stored as NaN). Empty for older files.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "categories" not in f or "links" not in f["categories"]:
            return {}
        data = f["categories"]["links"][:]
    result: dict[int, dict[int, tuple[int, float | None]]] = {}
    for row in data:
        owner_type = int(row["owner_type"])
        score = float(row["category_score"])
        result.setdefault(owner_type, {})[int(row["owner_id"])] = (
            int(row["category_idx"]),
            None if np.isnan(score) else score,
        )
    return result


def _write_category_link_rows(
    labels_path: str, rows: list[tuple[int, int, int, float]]
) -> None:
    """Write ``(owner_type, owner_id, category_idx, category_score)`` rows.

    Shared dataset-writing core for the per-detection category link, mirroring
    `_write_identity_link_rows`. The rows can be built either by iterating a
    `Labels` (the eager path, see `write_category_links`) or from a lazy store dict
    (the lazy save path), keeping the on-disk ``/categories/links`` layout identical
    for both.

    Args:
        labels_path: A string path to the SLEAP labels file.
        rows: A list of ``(owner_type, owner_id, category_idx, category_score)``
            tuples. ``category_score`` is a float (NaN encodes an unrecorded score).
            An empty list is a no-op (no dataset is created).
    """
    if not rows:
        return

    category_link_dtype = np.dtype(
        [
            ("owner_type", "u1"),
            ("owner_id", "i8"),
            ("category_idx", "i4"),
            ("category_score", "f4"),
        ]
    )
    arr = np.array(rows, dtype=category_link_dtype)
    with h5py.File(labels_path, "a") as f:
        group = f.require_group("categories")
        group.create_dataset("links", data=arr, maxshape=(None,))


def write_category_links(labels_path: str, labels: Labels) -> None:
    """Write per-detection category links to a SLEAP labels file.

    Creates the ``/categories/links`` dataset (SLP format 2.7+) as a structured
    array of ``(owner_type, owner_id, category_idx, category_score)`` rows for every
    detection carrying a `Category` registered in ``labels.categories``. Instance,
    mask, centroid, bbox, and ROI owners are written (``OWNER_*``). A fully
    self-contained mirror of `write_identity_links`.

    The instance ``owner_id`` matches the global id assigned by `write_lfs`, and the
    mask / centroid / bbox / ROI ``owner_id`` matches the global per-modality list
    index assigned by `write_masks` / `write_centroids` / `write_bboxes` /
    `write_rois` (each enumeration order over frames then the modality; ROIs append
    static ROIs after frame ROIs), so the link is resolved by joining on that id at
    read time. Category is resolved to a catalog index by object identity (a single
    ``id()``-keyed lookup), so the pass stays O(number of detections).

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the category links for.
    """
    if not labels.categories:
        return

    id_to_idx = {id(cat): i for i, cat in enumerate(labels.categories)}
    rows = []
    instance_id = 0
    masks: list = []
    centroids: list = []
    bboxes: list = []
    rois: list = []
    for lf in labels:
        for inst in lf:
            category = getattr(inst, "category", None)
            if category is not None:
                idx = id_to_idx.get(id(category))
                if idx is not None:
                    score = inst.category_score
                    rows.append(
                        (
                            OWNER_INSTANCE,
                            instance_id,
                            idx,
                            float("nan") if score is None else float(score),
                        )
                    )
            instance_id += 1
        masks.extend(lf.masks)
        centroids.extend(lf.centroids)
        bboxes.extend(lf.bboxes)
        rois.extend(lf.rois)
    # Static ROIs are appended after frame ROIs (matching write_labels' all_rois).
    rois.extend(labels.static_rois)

    rows.extend(_category_link_rows_for_owner(masks, id_to_idx, OWNER_MASK))
    rows.extend(_category_link_rows_for_owner(centroids, id_to_idx, OWNER_CENTROID))
    rows.extend(_category_link_rows_for_owner(bboxes, id_to_idx, OWNER_BBOX))
    rows.extend(_category_link_rows_for_owner(rois, id_to_idx, OWNER_ROI))
    _write_category_link_rows(labels_path, rows)


def _category_link_rows_for_owner(
    anns: list, id_to_idx: dict[int, int], owner_type: int
) -> list[tuple[int, int, int, float]]:
    """Build category-link rows for an ordered annotation list under one owner type.

    Annotations join on their global per-modality list index (the same enumeration
    as the modality's writer, e.g. `write_masks` / `write_centroids`), so ``anns``
    must be supplied in that exact order. Category is resolved to a catalog index by
    object identity. Mirrors `_identity_link_rows_for_owner`.

    Args:
        anns: The ordered global annotation list for one modality.
        id_to_idx: Mapping from a `Category`'s object id to its catalog index.
        owner_type: The ``OWNER_*`` code for this modality (e.g. ``OWNER_MASK``).

    Returns:
        A list of ``(owner_type, owner_id, category_idx, category_score)`` tuples.
    """
    rows = []
    for owner_id, ann in enumerate(anns):
        category = getattr(ann, "category", None)
        if category is None:
            continue
        idx = id_to_idx.get(id(category))
        if idx is None:
            continue
        score = ann.category_score
        rows.append(
            (
                owner_type,
                owner_id,
                idx,
                float("nan") if score is None else float(score),
            )
        )
    return rows


def _category_link_rows_from_store(
    store: "LazyDataStore",
) -> list[tuple[int, int, int, float]]:
    """Build ``/categories/links`` rows from a lazy store without materializing.

    The lazy store already holds the per-instance category links keyed by the
    global ``instance_id`` (the same id space written by the fast path), so the
    rows can be emitted directly as ``OWNER_INSTANCE`` owners without rebuilding
    `Instance` objects. Rows are sorted by ``instance_id`` for deterministic
    output. Mirrors `_identity_link_rows_from_store`.

    Args:
        store: The `LazyDataStore` backing a lazy `Labels`.

    Returns:
        A list of ``(owner_type, owner_id, category_idx, category_score)`` tuples
        matching the format expected by `_write_category_link_rows`.
    """
    rows = []
    for instance_id in sorted(store._instance_categories):
        category_idx, score = store._instance_categories[instance_id]
        rows.append(
            (
                OWNER_INSTANCE,
                int(instance_id),
                int(category_idx),
                float("nan") if score is None else float(score),
            )
        )
    return rows


# Owner-type codes for the owner_type join column shared by the /embeddings group
# and the /identity/links dataset. Both subsystems join a detection to a per-row
# payload via an (owner_type, owner_id) pair, so they share one code set: this
# guarantees the two join schemes can never drift (e.g. mask=3 in one and mask=4
# in the other).
OWNER_INSTANCE = 0
OWNER_CENTROID = 2
OWNER_MASK = 3
OWNER_BBOX = 4
OWNER_ROI = 5


def read_embeddings(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> dict[int, dict[int, "Embedding"]]:
    """Read per-detection re-ID embeddings from a SLEAP labels file.

    Embeddings are stored in the optional ``/embeddings`` group (SLP format 2.5+)
    as a single columnar struct-of-arrays: the stacked ``vectors`` ``(N, D)`` plus
    the ``owner_type`` / ``owner_id`` join columns. Row ``i`` of all three datasets
    describes the same detection. The returned `Embedding` vectors are views onto
    the one loaded ``vectors`` array (they share its buffer).

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A dict mapping each supported ``owner_type`` to a ``{owner_id: Embedding}``
        mapping. Rows with an unsupported owner type are skipped with a warning.
        Empty for older files.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "embeddings" not in f or "vectors" not in f["embeddings"]:
            return {}
        return _read_embedding_datasets(
            f["embeddings"], ("vectors", "owner_type", "owner_id")
        )


def read_category_embeddings(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> dict[int, dict[int, "Embedding"]]:
    """Read per-detection category (classification) embeddings from a SLP file.

    Category embeddings are stored in the SAME optional ``/embeddings`` group (SLP
    format 2.7+) as sibling parallel datasets: the stacked ``category_vectors``
    ``(M, D)`` plus the ``category_owner_type`` / ``category_owner_id`` join
    columns. These are independent of the identity ``vectors`` datasets read by
    `read_embeddings` (the two embedding kinds need not share dimensionality D).
    Absent in older files (or when no category embedding was written), in which
    case an empty mapping is returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A dict mapping each supported ``owner_type`` to a ``{owner_id: Embedding}``
        mapping. Rows with an unsupported owner type are skipped with a warning.
        Empty for older files.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "embeddings" not in f or "category_vectors" not in f["embeddings"]:
            return {}
        return _read_embedding_datasets(
            f["embeddings"],
            ("category_vectors", "category_owner_type", "category_owner_id"),
        )


def _read_embedding_datasets(
    group: h5py.Group, dataset_names: tuple[str, str, str]
) -> dict[int, dict[int, "Embedding"]]:
    """Read one ``(vectors, owner_type, owner_id)`` triple into an owner-keyed map.

    Shared reader core for the identity (`read_embeddings`) and category
    (`read_category_embeddings`) parallel dataset triples within the ``/embeddings``
    group. Row ``i`` of all three datasets describes the same detection; the
    returned `Embedding` vectors are views onto the one loaded vectors array.

    Args:
        group: The open ``/embeddings`` HDF5 group.
        dataset_names: The ``(vectors, owner_type, owner_id)`` dataset names to read.

    Returns:
        A dict mapping each supported ``owner_type`` to a ``{owner_id: Embedding}``
        mapping. Rows with an unsupported owner type are skipped with a warning.
    """
    supported = (OWNER_INSTANCE, OWNER_CENTROID, OWNER_MASK, OWNER_BBOX, OWNER_ROI)
    by_owner: dict[int, dict[int, Embedding]] = {}
    vectors_name, owner_type_name, owner_id_name = dataset_names
    vectors = group[vectors_name][:]
    owner_type = group[owner_type_name][:]
    owner_id = group[owner_id_name][:]
    unknown_owner_types: set[int] = set()
    for i in range(len(vectors)):
        otype = int(owner_type[i])
        if otype not in supported:
            unknown_owner_types.add(otype)
            continue
        by_owner.setdefault(otype, {})[int(owner_id[i])] = Embedding(vector=vectors[i])
    if unknown_owner_types:
        warnings.warn(
            f"Skipped /embeddings '{vectors_name}' rows with unsupported "
            f"owner_type(s) {sorted(unknown_owner_types)}: instance, mask, centroid, "
            "bbox, and ROI embeddings are attached on read. These vectors were "
            "written by a newer sleap-io and are ignored here.",
            stacklevel=2,
        )
    return by_owner


def _attach_identity_and_embeddings(
    anns: list,
    identities: list[Identity],
    identity_map: dict[int, tuple[int, float | None]],
    embedding_map: dict[int, "Embedding"],
) -> None:
    """Attach per-annotation identity links + embedding by global-list index.

    Shared by the eager and lazy read paths for any detection modality (e.g. masks,
    centroids). ``anns`` must be in the global per-modality list index order (as
    returned by `read_masks` / `read_centroids`), which is the id space the owner
    rows were written against.

    Args:
        anns: The ordered global annotation list for one modality.
        identities: The identity catalog (links resolve by catalog index).
        identity_map: ``{owner_id: (identity_idx, identity_score)}`` from
            `read_identity_links` for this modality's owner type.
        embedding_map: ``{owner_id: Embedding}`` from `read_embeddings` for this
            modality's owner type.
    """
    for owner_id, ann in enumerate(anns):
        link = identity_map.get(owner_id)
        if link is not None:
            identity_idx, identity_score = link
            if 0 <= identity_idx < len(identities):
                ann.identity = identities[identity_idx]
                ann.identity_score = identity_score
        emb = embedding_map.get(owner_id)
        if emb is not None:
            ann.identity_embedding = emb


def _attach_category_and_embeddings(
    anns: list,
    categories: list[Category],
    category_map: dict[int, tuple[int, float | None]],
    category_embedding_map: dict[int, "Embedding"],
) -> None:
    """Attach per-annotation category links + embedding by global-list index.

    Category mirror of `_attach_identity_and_embeddings`, shared by the eager and
    lazy read paths for any detection modality (e.g. masks, centroids). ``anns``
    must be in the global per-modality list index order (as returned by
    `read_masks` / `read_centroids`), which is the id space the owner rows were
    written against.

    Args:
        anns: The ordered global annotation list for one modality.
        categories: The category catalog (links resolve by catalog index).
        category_map: ``{owner_id: (category_idx, category_score)}`` from
            `read_category_links` for this modality's owner type.
        category_embedding_map: ``{owner_id: Embedding}`` from
            `read_category_embeddings` for this modality's owner type.
    """
    for owner_id, ann in enumerate(anns):
        link = category_map.get(owner_id)
        if link is not None:
            category_idx, category_score = link
            if 0 <= category_idx < len(categories):
                ann.category = categories[category_idx]
                ann.category_score = category_score
        emb = category_embedding_map.get(owner_id)
        if emb is not None:
            ann.category_embedding = emb


def write_embeddings(labels_path: str, labels: Labels) -> None:
    """Write per-detection re-ID embeddings to a SLEAP labels file.

    Creates the ``/embeddings`` group (SLP format 2.5+) as a single columnar
    struct-of-arrays: the stacked float ``vectors`` ``(N, D)`` plus the
    ``owner_type`` / ``owner_id`` join columns. Per-instance embeddings join on the
    global ``instance_id`` (matching `write_lfs`); per-mask / per-centroid /
    per-bbox / per-ROI embeddings join on the global per-modality list index
    (matching `write_masks` / `write_centroids` / `write_bboxes` / `write_rois`).
    Purely additive: old readers ignore the group.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the embeddings for.

    Raises:
        ValueError: If the embedding vectors do not all share the same
            dimensionality.
    """
    entries = _embedding_groups_from_labels(labels)
    _write_embedding_groups(labels_path, entries)


# Parallel category-embedding dataset names within the shared /embeddings group
# (SLP 2.7+). Sibling to the identity vectors/owner_type/owner_id datasets so the
# two embedding kinds are independent and need not share dimensionality D.
_CATEGORY_EMBEDDING_DATASETS = (
    "category_vectors",
    "category_owner_type",
    "category_owner_id",
)


def write_category_embeddings(labels_path: str, labels: Labels) -> None:
    """Write per-detection category (classification) embeddings to a SLP file.

    Creates sibling ``category_vectors`` / ``category_owner_type`` /
    ``category_owner_id`` datasets in the SAME ``/embeddings`` group (SLP format
    2.7+), independent of the identity ``vectors`` datasets so the two embedding
    kinds need not share dimensionality D. Category mirror of `write_embeddings`;
    the owner_id join spaces match `write_lfs` / `write_masks` / `write_centroids`
    / `write_bboxes` / `write_rois` exactly as identity does. Purely additive: a
    file with no category embeddings never gains the ``category_*`` datasets.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the category embeddings for.

    Raises:
        ValueError: If the category embedding vectors do not all share the same
            dimensionality.
    """
    entries = _category_embedding_groups_from_labels(labels)
    _write_embedding_groups(labels_path, entries, _CATEGORY_EMBEDDING_DATASETS)


def _embedding_groups_from_labels(labels: Labels) -> list:
    """Collect per-detection embeddings by iterating a `Labels` (eager path).

    Args:
        labels: A `Labels` object to collect detection embeddings from.

    Returns:
        A list of ``(owner_type, owner_id, Embedding)`` tuples where per-instance
        embeddings join on the global ``instance_id`` (enumeration order over frames
        then instances, matching `write_lfs`) and per-mask / per-centroid /
        per-bbox / per-ROI embeddings join on the global per-modality list index
        (matching `write_masks` / `write_centroids` / `write_bboxes` /
        `write_rois`; static ROIs follow frame ROIs).
    """
    entries: list = []

    instance_id = 0
    masks: list = []
    centroids: list = []
    bboxes: list = []
    rois: list = []
    for lf in labels:
        for inst in lf:
            emb = getattr(inst, "identity_embedding", None)
            if emb is not None:
                entries.append((OWNER_INSTANCE, instance_id, emb))
            instance_id += 1
        masks.extend(lf.masks)
        centroids.extend(lf.centroids)
        bboxes.extend(lf.bboxes)
        rois.extend(lf.rois)
    rois.extend(labels.static_rois)

    _add_owner_embedding_groups(entries, masks, OWNER_MASK)
    _add_owner_embedding_groups(entries, centroids, OWNER_CENTROID)
    _add_owner_embedding_groups(entries, bboxes, OWNER_BBOX)
    _add_owner_embedding_groups(entries, rois, OWNER_ROI)
    return entries


def _category_embedding_groups_from_labels(labels: Labels) -> list:
    """Collect per-detection category embeddings by iterating a `Labels`.

    Category mirror of `_embedding_groups_from_labels`, reading the
    ``category_embedding`` slot on each detection. Join spaces are identical
    (per-instance global ``instance_id``; per-mask / per-centroid / per-bbox /
    per-ROI global per-modality list index; static ROIs follow frame ROIs).

    Args:
        labels: A `Labels` object to collect detection category embeddings from.

    Returns:
        A list of ``(owner_type, owner_id, Embedding)`` tuples for the
        ``category_*`` datasets.
    """
    entries: list = []

    instance_id = 0
    masks: list = []
    centroids: list = []
    bboxes: list = []
    rois: list = []
    for lf in labels:
        for inst in lf:
            emb = getattr(inst, "category_embedding", None)
            if emb is not None:
                entries.append((OWNER_INSTANCE, instance_id, emb))
            instance_id += 1
        masks.extend(lf.masks)
        centroids.extend(lf.centroids)
        bboxes.extend(lf.bboxes)
        rois.extend(lf.rois)
    rois.extend(labels.static_rois)

    _add_owner_embedding_groups(entries, masks, OWNER_MASK, attr="category_embedding")
    _add_owner_embedding_groups(
        entries, centroids, OWNER_CENTROID, attr="category_embedding"
    )
    _add_owner_embedding_groups(entries, bboxes, OWNER_BBOX, attr="category_embedding")
    _add_owner_embedding_groups(entries, rois, OWNER_ROI, attr="category_embedding")
    return entries


def _add_owner_embedding_groups(
    entries: list, anns: list, owner_type: int, attr: str = "identity_embedding"
) -> None:
    """Append per-annotation embedding entries to ``entries`` (shared eager/lazy).

    Annotations join on their global per-modality list index (the same enumeration
    as the modality's writer, e.g. `write_masks` / `write_centroids`), so ``anns``
    must be supplied in that exact order. The ``attr`` parameter selects which
    embedding slot to read (``identity_embedding`` by default, or
    ``category_embedding`` for the parallel category vectors dataset).

    Args:
        entries: The ``[(owner_type, owner_id, Embedding), ...]`` accumulator to
            extend in place.
        anns: The ordered global annotation list for one modality.
        owner_type: The ``OWNER_*`` code for this modality (e.g. ``OWNER_MASK``).
        attr: The detection attribute holding the `Embedding` to collect
            (``identity_embedding`` or ``category_embedding``).
    """
    for owner_id, ann in enumerate(anns):
        emb = getattr(ann, attr, None)
        if emb is not None:
            entries.append((owner_type, owner_id, emb))


def _embedding_groups_from_store(store: "LazyDataStore") -> list:
    """Collect per-detection embeddings from a lazy store without materializing.

    The lazy store holds per-instance embeddings keyed by the global ``instance_id``
    (the same id space written by the fast path), so they can be emitted directly
    without rebuilding `Instance` objects. Entries are sorted by ``instance_id`` for
    deterministic output.

    Args:
        store: The `LazyDataStore` backing a lazy `Labels`.

    Returns:
        A list of ``(owner_type, owner_id, Embedding)`` tuples in the format expected
        by `_write_embedding_groups`.
    """
    entries: list = []
    for instance_id in sorted(store._instance_embeddings):
        entries.append(
            (OWNER_INSTANCE, int(instance_id), store._instance_embeddings[instance_id])
        )
    return entries


def _category_embedding_groups_from_store(store: "LazyDataStore") -> list:
    """Collect per-detection category embeddings from a lazy store.

    Category mirror of `_embedding_groups_from_store`, reading the store's
    per-instance category embedding dict keyed by the global ``instance_id``.
    Entries are sorted by ``instance_id`` for deterministic output.

    Args:
        store: The `LazyDataStore` backing a lazy `Labels`.

    Returns:
        A list of ``(owner_type, owner_id, Embedding)`` tuples for the
        ``category_*`` datasets.
    """
    entries: list = []
    for instance_id in sorted(store._instance_category_embeddings):
        entries.append(
            (
                OWNER_INSTANCE,
                int(instance_id),
                store._instance_category_embeddings[instance_id],
            )
        )
    return entries


def _write_embedding_groups(
    labels_path: str,
    entries: list,
    dataset_names: tuple[str, str, str] = ("vectors", "owner_type", "owner_id"),
) -> None:
    """Write per-detection embeddings to the ``/embeddings`` group of a SLP file.

    Shared dataset-writing core for both the eager (`Labels`-driven) and lazy
    (store-driven) paths. The large float ``vectors`` dataset is chunked so whole
    rows stay within a chunk (a single-detection read touches one chunk) and gzip
    compressed; the small ``owner_type`` / ``owner_id`` join columns are separate
    parallel datasets so more per-embedding attributes can be added later without
    re-laying-out ``vectors``. Purely additive: old readers ignore the group.

    The ``dataset_names`` parameter selects the ``(vectors, owner_type, owner_id)``
    dataset names within the shared ``/embeddings`` group. Identity vectors use the
    default names; category vectors use the parallel sibling names
    ``("category_vectors", "category_owner_type", "category_owner_id")`` (SLP 2.7+),
    so the two embedding kinds are independent and need not share dimensionality D.

    Args:
        labels_path: A string path to the SLEAP labels file.
        entries: A list of ``(owner_type, owner_id, Embedding)`` tuples as produced
            by `_embedding_groups_from_labels` or `_embedding_groups_from_store` (or
            their category counterparts). An empty list is a no-op.
        dataset_names: The ``(vectors, owner_type, owner_id)`` dataset names to
            create within ``/embeddings``.

    Raises:
        ValueError: If the embedding vectors do not all share the same
            dimensionality.
    """
    if not entries:
        return

    vectors_name, owner_type_name, owner_id_name = dataset_names
    dims = {emb.dim for _, _, emb in entries}
    if len(dims) > 1:
        raise ValueError(
            f"Embedding vectors have inconsistent dimensions {sorted(dims)}; all "
            f"vectors must share D to store in the columnar /embeddings "
            f"'{vectors_name}' dataset."
        )
    vectors = np.stack([emb.vector for _, _, emb in entries])
    owner_type = np.array([otype for otype, _, _ in entries], dtype="u1")
    owner_id = np.array([oid for _, oid, _ in entries], dtype="i8")

    n, d = vectors.shape
    # Chunk so whole rows stay within a chunk (target ~1 MiB/chunk); a single-row
    # read then touches exactly one chunk.
    chunk_rows = max(1, min(n, (1 << 20) // max(1, d * vectors.dtype.itemsize)))

    with h5py.File(labels_path, "a") as f:
        group = f.require_group("embeddings")
        group.create_dataset(
            vectors_name,
            data=vectors,
            maxshape=(None, d),
            chunks=(chunk_rows, d),
            compression="gzip",
        )
        group.create_dataset(
            owner_type_name, data=owner_type, maxshape=(None,), compression="gzip"
        )
        group.create_dataset(
            owner_id_name, data=owner_id, maxshape=(None,), compression="gzip"
        )


def read_suggestions(
    labels_path: str,
    videos: list[Video],
    *,
    _hdf5_file: h5py.File | None = None,
) -> list[SuggestionFrame]:
    """Read `SuggestionFrame` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of `SuggestionFrame` objects.
    """
    try:
        suggestions = read_hdf5_dataset(
            labels_path, "suggestions_json", _hdf5_file=_hdf5_file
        )
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


def read_negative_frames(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> set[tuple[int, int]]:
    """Read negative frame markers from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A set of (sparse_video_id, frame_idx) tuples identifying negative frames.
        Returns empty set if no negative frames dataset exists.
    """
    try:
        data = read_hdf5_dataset(labels_path, "negative_frames", _hdf5_file=_hdf5_file)
        return {(int(row["video_id"]), int(row["frame_idx"])) for row in data}
    except KeyError:
        return set()


def read_metadata(labels_path: str, *, _hdf5_file: h5py.File | None = None) -> dict:
    """Read metadata from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A dict containing the metadata from a SLEAP labels file.

    Raises:
        ValueError: If the ``metadata`` group is missing its ``json`` attribute
            (or the group itself is absent), indicating the file is likely
            corrupt.
    """
    try:
        md = read_hdf5_attrs(labels_path, "metadata", "json", _hdf5_file=_hdf5_file)
    except KeyError as e:
        raise ValueError(
            f"The SLEAP labels file {labels_path!r} is missing its required "
            "metadata JSON blob (the 'metadata' HDF5 group has no readable 'json' "
            "attribute) and is likely corrupt. If you have a working .slp file "
            "with the same skeleton, you can copy the attribute into a BACKUP "
            "COPY of the corrupt file with h5py (back up first):\n"
            "    import h5py\n"
            "    with h5py.File('working.slp', 'r') as src, "
            "h5py.File('corrupt_copy.slp', 'a') as dst:\n"
            "        dst['metadata'].attrs['json'] = src['metadata'].attrs['json']\n"
            "Only do this if the skeletons match exactly, otherwise the loaded "
            "data will be wrong."
        ) from e
    if isinstance(md, bytes):
        md = md.decode()
    elif isinstance(md, np.ndarray):
        md = md.tobytes().decode()
    # If md is already a str (e.g., h5py vlen string), use as-is.
    return json.loads(md)


def read_provenance(
    labels_path: str,
    metadata: dict,
    *,
    _hdf5_file: h5py.File | None = None,
) -> dict:
    """Read the provenance dict for a SLEAP labels file.

    Provenance is stored in a dedicated top-level ``/provenance_json`` dataset so
    it is not subject to HDF5's 64 KB per-attribute size limit. The legacy layout
    stored provenance inside the ``metadata/json`` attribute, which an unbounded
    ``merge_history`` (or other large provenance) could exceed, breaking saving
    entirely. For backward compatibility with files written before this change,
    this falls back to the ``provenance`` key inside the already-read ``metadata``
    blob when the dataset is absent.

    Args:
        labels_path: A string path to the SLEAP labels file.
        metadata: The parsed ``metadata`` dict (from `read_metadata`), used as the
            fallback source of provenance for files written in the legacy layout.
        _hdf5_file: An already-open ``h5py.File`` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise ``labels_path`` is opened and closed internally. Private,
            mirrors the other ``read_*`` helpers.

    Returns:
        The provenance dict. Empty if neither the dataset nor the metadata blob
        carry provenance.
    """
    try:
        raw = read_hdf5_dataset(labels_path, "provenance_json", _hdf5_file=_hdf5_file)
    except KeyError:
        return metadata.get("provenance", dict())
    # The dataset is a single JSON blob, read back as np.bytes_ (or a length-1
    # array from some writers). json.loads accepts bytes/np.bytes_/str directly.
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return json.loads(raw)


def read_skeletons(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> list[Skeleton]:
    """Read `Skeleton` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of `Skeleton` objects.
    """
    metadata = read_metadata(labels_path, _hdf5_file=_hdf5_file)

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


# HDF5 caps any single attribute at 64 KiB (65,536 bytes). Keep the metadata JSON
# blob comfortably under that, leaving headroom for the HDF5 object header and the
# sibling ``format_id`` attribute. Metadata that would exceed this has its large,
# safely-relocatable top-level keys dropped from the attribute (see
# ``_encode_metadata_attr``); the data itself is preserved in dedicated datasets.
METADATA_ATTR_SIZE_LIMIT = 64_000

# Top-level metadata keys that must never be dropped from the ``metadata/json``
# attribute because the reader has no other source for them. (``provenance`` is
# stored in the ``provenance_json`` dataset, and the remaining keys are empty
# placeholders whose real data lives in their own datasets, so all are droppable.)
_METADATA_PROTECTED_KEYS = ("version", "skeletons", "nodes")


def _encode_metadata_attr(md: dict) -> np.bytes_:
    """Serialize metadata to a JSON blob that fits HDF5's 64 KB attribute limit.

    Writing a single HDF5 attribute larger than 64 KB fails with a cryptic
    ``object header message is too large`` error (and on some HDF5 builds silently
    writes a corrupt file). To keep saving robust, if the serialized metadata
    exceeds `METADATA_ATTR_SIZE_LIMIT` this drops droppable top-level keys
    (everything except `_METADATA_PROTECTED_KEYS`) largest-first until it fits,
    emitting a ``UserWarning`` naming the dropped keys.

    Dropping is lossless for the file as a whole: ``provenance`` is independently
    stored in the ``provenance_json`` dataset, and ``videos``/``tracks``/
    ``suggestions``/``negative_anchors`` are empty placeholders in the blob whose
    real data lives in their own datasets.

    Args:
        md: The metadata dict to serialize. Mutated in place (offending keys are
            removed) when the blob exceeds the limit.

    Returns:
        The serialized JSON blob as ``np.bytes_``, ready to assign to an HDF5
        attribute.

    Raises:
        ValueError: If the protected keys (``skeletons``/``nodes``) alone exceed
            the limit and so cannot be made to fit by dropping other keys.
    """

    def _blob(d: dict) -> bytes:
        return json.dumps(d, separators=(",", ":")).encode()

    blob = _blob(md)
    if len(blob) <= METADATA_ATTR_SIZE_LIMIT:
        return np.bytes_(blob)

    original_size = len(blob)
    droppable = sorted(
        (k for k in md if k not in _METADATA_PROTECTED_KEYS),
        key=lambda k: len(_blob({k: md[k]})),
        reverse=True,
    )
    dropped = []
    for k in droppable:
        del md[k]
        dropped.append(k)
        blob = _blob(md)
        if len(blob) <= METADATA_ATTR_SIZE_LIMIT:
            break

    if len(blob) > METADATA_ATTR_SIZE_LIMIT:
        raise ValueError(
            "Unable to fit Labels metadata within HDF5's "
            f"{METADATA_ATTR_SIZE_LIMIT}-byte attribute limit: the required "
            f"skeletons/nodes metadata alone serializes to {len(blob)} bytes. "
            "Reduce the number or size of the skeletons/nodes."
        )

    warnings.warn(
        f"Labels metadata JSON ({original_size} bytes) exceeded the "
        f"{METADATA_ATTR_SIZE_LIMIT}-byte HDF5 attribute limit; dropped top-level "
        f"key(s) {dropped} from the 'metadata/json' attribute so the file could be "
        "saved. No data was lost: provenance is stored in the 'provenance_json' "
        "dataset and the other dropped keys are empty placeholders whose data is "
        "stored in separate datasets.",
        stacklevel=2,
    )
    return np.bytes_(blob)


def _write_provenance_dataset(f: h5py.File, provenance: dict) -> None:
    """Write provenance to the top-level ``provenance_json`` dataset.

    Provenance is stored as a dataset rather than inside the ``metadata/json``
    attribute so it is exempt from HDF5's 64 KB per-attribute limit (an unbounded
    ``merge_history`` would otherwise break saving). This is a no-op for empty
    provenance, and overwrites any existing dataset so repeated writes to the same
    open file stay idempotent.

    Args:
        f: An open ``h5py.File`` handle (opened in a writable mode).
        provenance: The JSON-serializable provenance dict to store.
    """
    if not provenance:
        return
    if "provenance_json" in f:
        del f["provenance_json"]
    f.create_dataset(
        "provenance_json",
        data=np.bytes_(json.dumps(provenance, separators=(",", ":"))),
    )


def write_metadata(labels_path: str, labels: Labels):
    """Write metadata to a SLEAP labels file.

    This function will write the skeletons and provenance for the labels.

    Provenance is written to a dedicated top-level ``provenance_json`` dataset (not
    subject to HDF5's 64 KB attribute limit) and, when small enough, also mirrored
    into the ``metadata/json`` attribute for backward compatibility with older
    readers (see `read_provenance` and `_encode_metadata_attr`).

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.

    See also: serialize_skeletons
    """
    skeletons_dicts, nodes_dicts = serialize_skeletons(labels.skeletons)

    # Encode provenance into a JSON-safe dict (Path -> str). Build a copy so we
    # never mutate the caller's ``labels.provenance``.
    provenance = {
        k: (v.as_posix() if isinstance(v, Path) else v)
        for k, v in labels.provenance.items()
    }

    md = {
        "version": "2.0.0",
        "skeletons": skeletons_dicts,
        "nodes": nodes_dicts,
        "videos": [],
        "tracks": [],
        "suggestions": [],  # TODO: Handle suggestions metadata.
        "negative_anchors": {},
        "provenance": provenance,
    }

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

    # Bump for virtual video crops (new in 2.3). 2.3 does not cross the only
    # legacy threshold (< 1.4, BGR embedded parsing), so this is purely additive
    # and only set when a crop is actually present (uncropped files stay <= 2.2).
    if any(v._crop_tuple() is not None for v in labels.videos):
        format_id = max(format_id, 2.3)

    # Bump for persisted mask from_predicted provenance links (new in 2.4). Only
    # set when a UserSegmentationMask actually records a source prediction; the
    # column is always written but reads are gated on column presence, so files
    # without any link stay <= 2.3.
    has_mask_from_predicted = any(
        getattr(m, "from_predicted", None) is not None for m in labels.masks
    )
    if has_mask_from_predicted:
        format_id = max(format_id, 2.4)

    # Bump for the re-ID identity subsystem (new in 2.5): the /identity catalog +
    # per-detection identity links, and the appearance /embeddings group. Purely
    # additive (each is a separate dataset/group read with a presence check), so
    # the bump is only applied when there is identity data to persist. For lazy
    # `Labels` the presence checks read the lazy store dicts directly, mirroring
    # what the fast-path writer persists without materializing any frames.
    if labels.is_lazy:
        store = labels._lazy_store

        def _store_dets(by_frame, undistributed):
            return [d for lst in by_frame.values() for d in lst] + list(undistributed)

        lazy_dets = (
            _store_dets(store._mask_by_frame, store._undistributed_masks)
            + _store_dets(store._centroid_by_frame, store._undistributed_centroids)
            + _store_dets(store._bbox_by_frame, store._undistributed_bboxes)
            + _store_dets(store._roi_by_frame, store._undistributed_rois)
        )
        has_identity_data = (
            bool(labels.identities)
            or bool(store._instance_identities)
            or bool(store._instance_embeddings)
            or any(
                getattr(d, "identity", None) is not None
                or getattr(d, "identity_embedding", None) is not None
                for d in lazy_dets
            )
        )
        has_category_data = (
            bool(labels.categories)
            or bool(store._instance_categories)
            or bool(store._instance_category_embeddings)
            or any(
                getattr(d, "category", None) is not None
                or getattr(d, "category_embedding", None) is not None
                for d in lazy_dets
            )
        )
    else:
        dets = [
            d for lf in labels for d in (*lf.masks, *lf.centroids, *lf.bboxes, *lf.rois)
        ] + list(labels.static_rois)
        has_identity_data = (
            bool(labels.identities)
            or any(
                getattr(inst, "identity", None) is not None
                or getattr(inst, "identity_embedding", None) is not None
                for lf in labels
                for inst in lf
            )
            or any(
                getattr(d, "identity", None) is not None
                or getattr(d, "identity_embedding", None) is not None
                for d in dets
            )
        )
        has_category_data = (
            bool(labels.categories)
            or any(
                getattr(inst, "category", None) is not None
                or getattr(inst, "category_embedding", None) is not None
                for lf in labels
                for inst in lf
            )
            or any(
                getattr(d, "category", None) is not None
                or getattr(d, "category_embedding", None) is not None
                for d in dets
            )
        )
    if has_identity_data:
        format_id = max(format_id, 2.5)

    # Bump for frame-spanning events (new in 2.6): the /event_types catalog and the
    # columnar /events group (+ ragged CSR framewise scores). Purely additive (each
    # is a separate group read with a presence check), so the bump only applies when
    # there is event data to persist. Works for lazy Labels too, since events and
    # event_types live on top-level lists (not the lazy store).
    if labels.events or labels.event_types:
        format_id = max(format_id, 2.6)

    # Bump for the class/category subsystem (new in 2.7): the /categories catalog +
    # per-detection category links, and the parallel category_* appearance datasets
    # in /embeddings. Purely additive (each is a separate dataset/group read with a
    # presence check), so the bump only applies when there is category data to
    # persist (computed above alongside has_identity_data for both eager and lazy).
    if has_category_data:
        format_id = max(format_id, 2.7)

    with h5py.File(labels_path, "a") as f:
        # Bump for chunked label image format (new in 2.2)
        if "label_image_data" in f and f["label_image_data"].ndim == 3:
            format_id = max(format_id, 2.2)

        # Bump for the columnar RecordingSession frame-group subsystem (new in 2.8).
        # write_sessions runs before write_metadata in both the eager and lazy write
        # paths, so gate on the group it just wrote (mirrors the label_image_data
        # presence check above). Files whose sessions have no frame groups never get
        # the group written and stay <= 2.7.
        if "session_data" in f:
            format_id = max(format_id, 2.8)

        grp = f.require_group("metadata")
        grp.attrs["format_id"] = format_id
        # Store provenance in a dedicated dataset (no 64 KB attribute limit) so an
        # unbounded merge_history can't break saving. A copy is also kept in the
        # metadata/json attribute (when it fits) by `_encode_metadata_attr` for
        # older readers.
        _write_provenance_dataset(f, provenance)
        grp.attrs["json"] = _encode_metadata_attr(md)


def read_points(labels_path: str, *, _hdf5_file: h5py.File | None = None) -> np.ndarray:
    """Read points dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A structured array of point data.
    """
    pts = read_hdf5_dataset(labels_path, "points", _hdf5_file=_hdf5_file)
    return pts


def read_pred_points(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> np.ndarray:
    """Read predicted points dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A structured array of predicted point data.
    """
    pred_pts = read_hdf5_dataset(labels_path, "pred_points", _hdf5_file=_hdf5_file)
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
    *,
    _hdf5_file: h5py.File | None = None,
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
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of `Instance` and/or `PredictedInstance` objects.
    """
    instances_data = read_hdf5_dataset(labels_path, "instances", _hdf5_file=_hdf5_file)

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


def _decode_meta_blob(meta_arr: np.ndarray | None, idx: int) -> dict:
    """Decode a per-row JSON metadata blob from a vlen-string dataset.

    Args:
        meta_arr: The ``frame_group_meta`` / ``instance_group_meta`` array, or `None`
            when the (presence-guarded) dataset was omitted.
        idx: Row index.

    Returns:
        The decoded metadata dict (empty when absent or blank).
    """
    if meta_arr is None or idx >= len(meta_arr):
        return {}
    raw = meta_arr[idx]
    if isinstance(raw, bytes):
        raw = raw.decode()
    raw = str(raw)
    if not raw:
        return {}
    return json.loads(raw)


def _make_instance_group_columnar(
    ig_row: np.void,
    session_data: dict,
    camera_group: CameraGroup,
    labeled_frames: list[LabeledFrame],
    identities: list[Identity] | None,
    ig_idx: int,
) -> InstanceGroup:
    """Reconstruct one `InstanceGroup` from the columnar `/session_data` tables.

    Args:
        ig_row: A single row of the ``instance_groups`` struct dataset.
        session_data: The loaded ``/session_data`` arrays (see `_read_session_data`).
        camera_group: `CameraGroup` for resolving camera indices.
        labeled_frames: `LabeledFrame` list for resolving member instances.
        identities: Optional identity catalog for resolving ``identity_idx``.
        ig_idx: This instance group's global row index (for metadata lookup).

    Returns:
        The reconstructed `InstanceGroup`.
    """
    members = session_data["instance_group_members"]
    m_start, m_end = int(ig_row["member_start"]), int(ig_row["member_end"])
    instance_by_camera: dict[Camera, Instance] = {}
    for m in range(m_start, m_end):
        row = members[m]
        camera = camera_group.cameras[int(row["camera"])]
        labeled_frame = labeled_frames[int(row["lf"])]
        instance = labeled_frame.instances[int(row["inst"])]
        instance_by_camera[camera] = instance

    score = float(ig_row["score"])
    score = None if np.isnan(score) else score

    identity = None
    identity_idx = int(ig_row["identity_idx"])
    if identity_idx >= 0 and identities is not None:
        if identity_idx < len(identities):
            identity = identities[identity_idx]
        else:
            warnings.warn(
                f"identity_idx {identity_idx} out of range "
                f"(max {len(identities) - 1}); identity set to None."
            )

    # 3D points from the points_3d / pred_points_3d row range.
    instance_3d = None
    pts3d_start = int(ig_row["pts3d_start"])
    if pts3d_start >= 0:
        pts3d_end = int(ig_row["pts3d_end"])
        predicted = bool(ig_row["pts3d_predicted"])
        skeleton = None
        for inst in instance_by_camera.values():
            skeleton = inst.skeleton
            break
        i3d_score = float(ig_row["instance_3d_score"])
        i3d_score = None if np.isnan(i3d_score) else i3d_score
        source = (
            session_data["pred_points_3d"] if predicted else session_data["points_3d"]
        )
        if skeleton is None:
            warnings.warn(
                "3D points discarded for InstanceGroup: no skeleton available "
                "(all camera mappings failed)."
            )
        elif source is None:
            warnings.warn(
                "3D points discarded for InstanceGroup: referenced points dataset "
                "is missing."
            )
        elif predicted:
            block = np.asarray(source[pts3d_start:pts3d_end], dtype="f8")
            instance_3d = PredictedInstance3D(
                points=block[:, :3],
                skeleton=skeleton,
                score=i3d_score,
                point_scores=block[:, 3],
            )
        else:
            instance_3d = Instance3D(
                points=np.asarray(source[pts3d_start:pts3d_end], dtype="f8"),
                skeleton=skeleton,
                score=i3d_score,
            )

    return InstanceGroup(
        instance_by_camera=instance_by_camera,
        score=score,
        instance_3d=instance_3d,
        identity=identity,
        metadata=_decode_meta_blob(session_data["instance_group_meta"], ig_idx),
    )


def _reconstruct_frame_groups_columnar(
    fg_start: int,
    fg_end: int,
    session_data: dict,
    camera_group: CameraGroup,
    labeled_frames: list[LabeledFrame],
    identities: list[Identity] | None,
) -> dict[int, FrameGroup]:
    """Reconstruct a session's `FrameGroup`s from the columnar `/session_data` tables.

    Args:
        fg_start: Start (inclusive) of this session's range into ``frame_groups``.
        fg_end: End (exclusive) of the range.
        session_data: The loaded ``/session_data`` arrays (see `_read_session_data`).
        camera_group: `CameraGroup` for resolving camera indices.
        labeled_frames: `LabeledFrame` list for resolving member instances.
        identities: Optional identity catalog for resolving ``identity_idx``.

    Returns:
        Mapping of ``frame_idx`` to reconstructed `FrameGroup`.
    """
    frame_groups = session_data["frame_groups"]
    instance_groups = session_data["instance_groups"]
    frame_group_by_frame_idx: dict[int, FrameGroup] = {}
    for fg in range(fg_start, fg_end):
        fg_row = frame_groups[fg]
        frame_idx = int(fg_row["frame_idx"])
        ig_start, ig_end = int(fg_row["ig_start"]), int(fg_row["ig_end"])
        try:
            groups: list[InstanceGroup] = []
            labeled_frame_by_camera: dict[Camera, LabeledFrame] = {}
            for ig_idx in range(ig_start, ig_end):
                group = _make_instance_group_columnar(
                    instance_groups[ig_idx],
                    session_data,
                    camera_group,
                    labeled_frames,
                    identities,
                    ig_idx,
                )
                groups.append(group)
                # Recover the LabeledFrame-by-camera mapping from the same members.
                m_start = int(instance_groups[ig_idx]["member_start"])
                m_end = int(instance_groups[ig_idx]["member_end"])
                members = session_data["instance_group_members"]
                for m in range(m_start, m_end):
                    row = members[m]
                    camera = camera_group.cameras[int(row["camera"])]
                    labeled_frame_by_camera[camera] = labeled_frames[int(row["lf"])]
            frame_group_by_frame_idx[frame_idx] = FrameGroup(
                frame_idx=frame_idx,
                instance_groups=groups,
                labeled_frame_by_camera=labeled_frame_by_camera,
                metadata=_decode_meta_blob(session_data["frame_group_meta"], fg),
            )
        except (ValueError, IndexError) as e:
            print(
                f"Error reconstructing FrameGroup at frame {frame_idx}. "
                f"Skipping...\n{e}"
            )
    return frame_group_by_frame_idx


def make_session(
    session_dict: dict,
    videos: list[Video],
    labeled_frames: list[LabeledFrame],
    identities: list[Identity] | None = None,
    session_data: dict | None = None,
) -> RecordingSession:
    """Create a `RecordingSession` from a dictionary.

    Args:
        session_dict: Dictionary with keys:
            - "calibration": Dictionary containing calibration information for cameras.
            - "camcorder_to_video_idx_map": Dictionary mapping camera index to video
                index.
            - Either an ``fg_start``/``fg_end`` range into the columnar
                ``/session_data`` tables (SLP 2.8+) or a legacy ``frame_group_dicts``
                list (<= 2.7; see `make_frame_group`).
            - Any optional keys containing metadata.
        videos: List containing `Video` objects (expected `Labels.videos`).
        labeled_frames: List containing `LabeledFrame` objects (expected
            `Labels.labeled_frames`). When empty (e.g. the lazy read path), frame
            groups are not reconstructed.
        identities: Optional list of `Identity` objects for resolving identity
            indices.
        session_data: The loaded ``/session_data`` columnar arrays (see
            `_read_session_data`) used for the SLP 2.8+ path. `None` for legacy files.

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

    frame_group_by_frame_idx: dict[int, FrameGroup] = {}
    fg_start = session_dict.pop("fg_start", None)
    fg_end = session_dict.pop("fg_end", None)
    if fg_start is not None and session_data is not None and labeled_frames:
        # SLP 2.8+ columnar path.
        frame_group_by_frame_idx = _reconstruct_frame_groups_columnar(
            int(fg_start),
            int(fg_end),
            session_data,
            camera_group,
            labeled_frames,
            identities,
        )
    elif fg_start is None and labeled_frames:
        # Legacy (<= 2.7) inline path.
        for frame_group_dict in session_dict.pop("frame_group_dicts", []):
            try:
                frame_group = make_frame_group(
                    frame_group_dict=frame_group_dict,
                    labeled_frames=labeled_frames,
                    camera_group=camera_group,
                    identities=identities,
                )
                frame_group_by_frame_idx[frame_group.frame_idx] = frame_group
            except ValueError as e:
                print(
                    f"Error reconstructing FrameGroup: {frame_group_dict}. "
                    f"Skipping...\n{e}"
                )
    else:
        # No labeled frames materialized (e.g. lazy read): drop the frame-group
        # scaffolding key so it does not leak into metadata, but skip reconstruction.
        session_dict.pop("frame_group_dicts", None)

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
    *,
    _hdf5_file: h5py.File | None = None,
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
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of `RecordingSession` objects.
    """
    try:
        sessions = read_hdf5_dataset(
            labels_path, "sessions_json", _hdf5_file=_hdf5_file
        )
    except KeyError:
        return []
    sessions = [json.loads(x) for x in sessions]
    session_data = _read_session_data(labels_path, _hdf5_file=_hdf5_file)
    session_objects = []
    for session in sessions:
        session_objects.append(
            make_session(
                session,
                videos,
                labeled_frames,
                identities=identities,
                session_data=session_data,
            )
        )
    return session_objects


def _read_session_data(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> dict | None:
    """Load the columnar ``/session_data`` arrays (SLP 2.8+), or `None` if absent.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A dict of the (possibly-`None`, presence-guarded) member arrays keyed by
        dataset name, or `None` when the group is absent (legacy <= 2.7 files).
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "session_data" not in f:
            return None
        grp = f["session_data"]
        names = [
            "frame_groups",
            "instance_groups",
            "instance_group_members",
            "points_3d",
            "pred_points_3d",
            "frame_group_meta",
            "instance_group_meta",
        ]
        # Read through the conversion-aware helper so the struct tables written by
        # the coordinated sleap-io.js 2.8 port (h5wasm cannot create compound
        # datasets, so it writes them as flat 2D arrays + a ``field_names``
        # attribute) are rebuilt as structured arrays, matching how
        # points/instances/frames are handled. h5py-written compound datasets and
        # the plain float points_3d matrices pass through unchanged.
        return {
            name: (
                _read_dataset_from_open_file(f, f"session_data/{name}")
                if name in grp
                else None
            )
            for name in names
        }


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


def _session_calibration_dict(
    session: RecordingSession, video_to_idx: dict[Video, int]
) -> tuple[dict, dict]:
    """Build the small calibration + camera-to-video maps for a session.

    These are the O(cameras) structural pieces that remain inline in the slim
    ``sessions_json`` blob (SLP 2.8); the O(frames) frame-group data is stored
    columnar in ``/session_data``. Extracted from the legacy `session_to_dict` so
    both the columnar writer and the legacy serializer share the exact same logic.

    Args:
        session: `RecordingSession` to serialize.
        video_to_idx: Mapping of `Video` to index in `Labels.videos`.

    Returns:
        A tuple of ``(calibration_dict, camera_to_video_idx_map)``.
    """
    calibration_dict = camera_group_to_dict(session.camera_group)

    camera_to_video_idx_map: dict[int, int] = {}
    for cam_idx, camera in enumerate(session.camera_group.cameras):
        # Skip if Camera is not linked to any Video.
        if camera not in session.cameras:
            continue
        video = session.get_video(camera)
        video_idx = video_to_idx.get(video, None)
        if video_idx is not None:
            camera_to_video_idx_map[cam_idx] = video_idx
        else:
            print(
                f"Video {video} not found in `Labels.videos`. "
                "Not saving to `RecordingSession` serialization."
            )
    return calibration_dict, camera_to_video_idx_map


def _append_rows_2d(
    grp: h5py.Group, dset: h5py.Dataset | None, name: str, block: np.ndarray
) -> h5py.Dataset:
    """Append rows to a resizable float64 2-D dataset, creating it on first use.

    Used for the incremental per-session append of ``points_3d`` / ``pred_points_3d``
    so the whole 3D point array is never held in memory at once (only one session's
    worth). The dataset is chunked + gzip so multi-view projects scale and remote
    JS/WASM consumers can range-read chunks.

    Args:
        grp: The open ``/session_data`` HDF5 group.
        dset: The dataset handle, or `None` to create it from ``block``'s width.
        name: Dataset name within the group.
        block: A ``(rows, ncols)`` float64 array to append.

    Returns:
        The dataset handle (created on first call).
    """
    ncols = block.shape[1]
    if dset is None:
        dset = grp.create_dataset(
            name,
            shape=(0, ncols),
            maxshape=(None, ncols),
            dtype="f8",
            chunks=(min(8192, max(1, len(block))), ncols),
            compression="gzip",
        )
    old = dset.shape[0]
    dset.resize(old + len(block), axis=0)
    dset[old:] = block
    return dset


def write_sessions(
    labels_path: str,
    sessions: list[RecordingSession],
    videos: list[Video],
    labeled_frames: list[LabeledFrame],
    identities: list[Identity] | None = None,
):
    """Write `RecordingSession` metadata to a SLEAP labels file (SLP 2.8 columnar).

    Always creates the ``sessions_json`` dataset holding one *slim* JSON blob per
    session: only calibration + ``camcorder_to_video_idx_map`` + session-level
    metadata + an ``fg_start``/``fg_end`` half-open range into
    ``/session_data/frame_groups``.

    When any session has frame groups (and ``labeled_frames`` are supplied), the
    per-frame numeric payload is written columnar into the ``/session_data`` group:
    ``frame_groups`` / ``instance_groups`` / ``instance_group_members`` struct
    datasets, ``points_3d`` (N,3) / ``pred_points_3d`` (N,4 = xyz+score) chunked
    float matrices (appended one session at a time), and presence-guarded per-row
    ``frame_group_meta`` / ``instance_group_meta`` JSON blobs. The group is omitted
    entirely (and the format stays <= 2.7) when there are no frame groups, so files
    without multi-view frame groups are byte-identical to before.

    Args:
        labels_path: A string path to the SLEAP labels file.
        sessions: A list of `RecordingSession` objects to store in the `labels_path`
            file.
        videos: A list of `Video` objects referenced in the `RecordingSession`s
            (expecting `Labels.videos`).
        labeled_frames: A list of `LabeledFrame` objects referenced in the
            `RecordingSession`s (expecting `Labels.labeled_frames`). An empty list
            skips all frame groups (matching the historical behavior when frames are
            not materialized).
        identities: Optional list of `Identity` objects for serializing identity
            indices.
    """
    sessions_json: list[np.bytes_] = []
    if len(sessions) > 0:
        labeled_frame_to_idx = {lf: i for i, lf in enumerate(labeled_frames)}
        video_to_idx = {video: i for i, video in enumerate(videos)}

    # Columnar accumulators (small; O(frames*instances) of fixed-width ints/floats).
    fg_rows: list[tuple] = []
    ig_rows: list[tuple] = []
    member_rows: list[tuple] = []
    fg_meta: list[str] = []
    ig_meta: list[str] = []

    save_frame_groups = len(sessions) > 0 and len(labeled_frame_to_idx) > 0

    with h5py.File(labels_path, "a") as f:
        points_dset: h5py.Dataset | None = None
        pred_points_dset: h5py.Dataset | None = None
        session_data_grp: h5py.Group | None = None
        pts3d_counter = 0
        pred_counter = 0

        for session in sessions:
            calibration_dict, camera_to_video_idx_map = _session_calibration_dict(
                session, video_to_idx
            )

            fg_start = len(fg_rows)
            session_pts_blocks: list[np.ndarray] = []
            session_pred_blocks: list[np.ndarray] = []

            if save_frame_groups:
                for frame_group in session.frame_groups.values():
                    if len(frame_group.instance_groups) == 0:
                        continue

                    instance_to_lf_and_inst_idx: dict[Instance, tuple[int, int]] = {
                        inst: (labeled_frame_to_idx[lf], inst_idx)
                        for lf in frame_group.labeled_frames
                        for inst_idx, inst in enumerate(lf.instances)
                    }

                    ig_start = len(ig_rows)
                    for ig in frame_group.instance_groups:
                        member_start = len(member_rows)
                        cameras = session.camera_group.cameras
                        for cam, inst in ig.instance_by_camera.items():
                            lf_idx, inst_idx = instance_to_lf_and_inst_idx[inst]
                            member_rows.append((cameras.index(cam), lf_idx, inst_idx))
                        member_end = len(member_rows)

                        # Identity -> catalog index (-1 when unset / not found).
                        identity_idx = -1
                        if ig.identity is not None and identities is not None:
                            try:
                                identity_idx = identities.index(ig.identity)
                            except ValueError:
                                warnings.warn(
                                    f"Identity '{ig.identity.name}' not found in "
                                    "Labels.identities; identity dropped during save."
                                )

                        score = ig.score if ig.score is not None else np.nan

                        # 3D points -> points_3d / pred_points_3d row range.
                        pts3d_start, pts3d_end, pts3d_predicted = -1, -1, 0
                        i3d_score = np.nan
                        inst3d = ig.instance_3d
                        if inst3d is not None and inst3d.points is not None:
                            pts = np.asarray(inst3d.points, dtype="f8")
                            is_pred = (
                                isinstance(inst3d, PredictedInstance3D)
                                and inst3d.point_scores is not None
                            )
                            if is_pred:
                                ps = np.asarray(
                                    inst3d.point_scores, dtype="f8"
                                ).reshape(-1, 1)
                                block = np.hstack([pts, ps])
                                pts3d_start = pred_counter
                                pred_counter += len(block)
                                pts3d_end = pred_counter
                                pts3d_predicted = 1
                                session_pred_blocks.append(block)
                            else:
                                pts3d_start = pts3d_counter
                                pts3d_counter += len(pts)
                                pts3d_end = pts3d_counter
                                session_pts_blocks.append(pts)
                            if inst3d.score is not None:
                                i3d_score = inst3d.score

                        ig_rows.append(
                            (
                                identity_idx,
                                score,
                                i3d_score,
                                pts3d_start,
                                pts3d_end,
                                pts3d_predicted,
                                member_start,
                                member_end,
                            )
                        )
                        ig_meta.append(
                            json.dumps(ig.metadata, separators=(",", ":"))
                            if ig.metadata
                            else ""
                        )
                    ig_end = len(ig_rows)

                    fg_rows.append((int(frame_group.frame_idx), ig_start, ig_end))
                    fg_meta.append(
                        json.dumps(frame_group.metadata, separators=(",", ":"))
                        if frame_group.metadata
                        else ""
                    )

            fg_end = len(fg_rows)

            # Append this session's 3D blocks (bounded to one session in memory).
            if session_pts_blocks:
                if session_data_grp is None:
                    session_data_grp = f.require_group("session_data")
                points_dset = _append_rows_2d(
                    session_data_grp,
                    points_dset,
                    "points_3d",
                    np.concatenate(session_pts_blocks),
                )
            if session_pred_blocks:
                if session_data_grp is None:
                    session_data_grp = f.require_group("session_data")
                pred_points_dset = _append_rows_2d(
                    session_data_grp,
                    pred_points_dset,
                    "pred_points_3d",
                    np.concatenate(session_pred_blocks),
                )

            session_json = {
                "calibration": calibration_dict,
                "camcorder_to_video_idx_map": camera_to_video_idx_map,
            }
            session_json.update(session.metadata)
            session_json["fg_start"] = fg_start
            session_json["fg_end"] = fg_end
            sessions_json.append(
                np.bytes_(json.dumps(session_json, separators=(",", ":")))
            )

        f.create_dataset("sessions_json", data=sessions_json, maxshape=(None,))

        # Columnar frame-group tables (only when there are frame groups).
        if fg_rows:
            grp = f.require_group("session_data")
            grp.create_dataset(
                "frame_groups",
                data=np.array(fg_rows, dtype=FRAME_GROUP_DTYPE),
                maxshape=(None,),
            )
            grp.create_dataset(
                "instance_groups",
                data=np.array(ig_rows, dtype=INSTANCE_GROUP_DTYPE),
                maxshape=(None,),
            )
            grp.create_dataset(
                "instance_group_members",
                data=np.array(member_rows, dtype=INSTANCE_GROUP_MEMBER_DTYPE),
                maxshape=(None,),
            )
            # Per-row metadata JSON blobs, presence-guarded (omitted when all empty).
            str_dt = h5py.special_dtype(vlen=str)
            if any(fg_meta):
                grp.create_dataset(
                    "frame_group_meta", data=fg_meta, dtype=str_dt, maxshape=(None,)
                )
            if any(ig_meta):
                grp.create_dataset(
                    "instance_group_meta", data=ig_meta, dtype=str_dt, maxshape=(None,)
                )


def _videos_unchanged(current: list[Video], original_ids: tuple) -> bool:
    """Whether the current video list matches the load-time identity snapshot.

    The lazy ``sessions_json`` passthrough encodes video indices
    (``camcorder_to_video_idx_map``); those indices only stay valid if the video
    list has not been reordered/added/removed since load. The comparison is against
    an immutable ``id()`` snapshot captured at load (``LazyDataStore``'s
    ``session_video_ids``), so it stays correct even when the store shares the list
    object with `Labels` (e.g. an in-place ``videos.reverse()``).

    Args:
        current: The current `Labels.videos`.
        original_ids: The ``id()`` tuple of the video list captured at load time.

    Returns:
        `True` when passthrough of the session video refs is safe.
    """
    return tuple(id(v) for v in current) == tuple(original_ids)


def write_sessions_passthrough(labels_path: str, store) -> bool:
    """Copy a lazy store's raw RecordingSession payload verbatim (SLP 2.8+).

    Writes the captured ``sessions_json`` bytes and the columnar ``/session_data``
    arrays exactly as loaded, so lazy re-saves preserve frame groups + 3D points
    losslessly without materializing frames (the eager lazy path drops them).

    Args:
        labels_path: A string path to the SLEAP labels file being written.
        store: The `LazyDataStore` backing the lazy `Labels`.

    Returns:
        `True` if a passthrough was written; `False` if the store carried no
        captured session payload (caller should fall back to `write_sessions`).
    """
    raw = getattr(store, "sessions_json_raw", None)
    if raw is None:
        return False
    session_data = getattr(store, "session_data", None)
    with h5py.File(labels_path, "a") as f:
        f.create_dataset("sessions_json", data=raw, maxshape=(None,))
        if session_data is not None:
            grp = f.require_group("session_data")
            for name, arr in session_data.items():
                if arr is None:
                    continue
                if name in ("points_3d", "pred_points_3d"):
                    grp.create_dataset(
                        name,
                        data=arr,
                        maxshape=(None, arr.shape[1]),
                        chunks=(min(8192, max(1, len(arr))), arr.shape[1]),
                        compression="gzip",
                    )
                elif name in ("frame_group_meta", "instance_group_meta"):
                    grp.create_dataset(
                        name,
                        data=arr,
                        dtype=h5py.special_dtype(vlen=str),
                        maxshape=(None,),
                    )
                else:
                    grp.create_dataset(name, data=arr, maxshape=(None,))
    return True


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
    with h5py.File(labels_path, "r") as f:
        return _read_labels_lazy_from_open_file(labels_path, f, open_videos=open_videos)


def _read_labels_lazy_from_open_file(
    labels_path: str,
    f: h5py.File,
    *,
    open_videos: bool = True,
    _url_headers: dict[str, str] | None = None,
    _url_stream_mode: str = "blockcache",
    _url_bytes: bytes | None = None,
) -> Labels:
    """Build a lazy `Labels` from an already-open `h5py.File`.

    Threads `_hdf5_file=f` through every `read_*` helper so the entire metadata
    read happens against a single open handle. Does NOT close `f` (the caller's
    `with` block owns it). The long-lived label-image handle returned by
    `read_label_images` (a separate, second handle) is attached to
    `labels._label_image_file` and intentionally outlives `f`.

    Args:
        labels_path: Path or URL to .slp file.
        f: An already-open `h5py.File` handle to read from.
        open_videos: Whether to open video backends.
        _url_headers: HTTP headers forwarded to the long-lived label-image
            handle when `labels_path` is a URL. Ignored for local paths.
        _url_stream_mode: Streaming strategy for the long-lived label-image
            handle when `labels_path` is a URL. Ignored for local paths.
        _url_bytes: Already-downloaded file bytes to reuse for the long-lived
            label-image handle instead of re-opening `labels_path` over the
            network (used for Google Drive, where a re-resolve is a re-download
            against the per-file quota). Ignored for local paths.

    Returns:
        Labels with LazyFrameList for labeled_frames.
    """
    from sleap_io.io.slp_lazy import LazyDataStore, LazyFrameList

    # Read raw arrays
    frames_data = read_hdf5_dataset(labels_path, "frames", _hdf5_file=f)
    instances_data = read_hdf5_dataset(labels_path, "instances", _hdf5_file=f)
    points_data = read_points(labels_path, _hdf5_file=f)
    pred_points_data = read_pred_points(labels_path, _hdf5_file=f)

    # Read format ID
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id", _hdf5_file=f)

    # Read metadata eagerly (these are small and needed for lazy access)
    videos = read_videos(
        labels_path,
        open_backend=open_videos,
        _hdf5_file=f,
        _url_headers=_url_headers,
        _url_stream_mode=_url_stream_mode,
    )
    skeletons = read_skeletons(labels_path, _hdf5_file=f)
    tracks = read_tracks(labels_path, _hdf5_file=f)
    identities = read_identities(labels_path, _hdf5_file=f)
    # Class/category catalog (SLP 2.7+). Self-contained mirror of /identity; a
    # top-level list read eagerly like identities.
    categories = read_categories(labels_path, _hdf5_file=f)
    # Frame-spanning events + catalog (SLP 2.6+). Top-level lists (not the lazy
    # store), read eagerly like tracks/identities/suggestions.
    event_types = read_event_types(labels_path, _hdf5_file=f)
    events = read_events(
        labels_path, videos, event_types, tracks, identities, _hdf5_file=f
    )
    # Identity links: instance owners ride on the lazy store (keyed by global
    # instance_id) and attach at materialization time; mask owners attach eagerly
    # to the masks read below.
    identity_links = read_identity_links(labels_path, _hdf5_file=f)
    instance_identities = identity_links.get(OWNER_INSTANCE, {})
    mask_identity_map = identity_links.get(OWNER_MASK, {})
    centroid_identity_map = identity_links.get(OWNER_CENTROID, {})
    bbox_identity_map = identity_links.get(OWNER_BBOX, {})
    roi_identity_map = identity_links.get(OWNER_ROI, {})
    # Category links (SLP 2.7+): same owner-keyed split as identity links.
    category_links = read_category_links(labels_path, _hdf5_file=f)
    instance_categories = category_links.get(OWNER_INSTANCE, {})
    mask_category_map = category_links.get(OWNER_MASK, {})
    centroid_category_map = category_links.get(OWNER_CENTROID, {})
    bbox_category_map = category_links.get(OWNER_BBOX, {})
    roi_category_map = category_links.get(OWNER_ROI, {})
    # Embeddings: per-instance embeddings ride on the lazy store; mask/centroid/
    # bbox/ROI embeddings attach to those (eagerly-read) annotations below.
    embeddings_by_owner = read_embeddings(labels_path, _hdf5_file=f)
    instance_embeddings = embeddings_by_owner.get(OWNER_INSTANCE, {})
    mask_embedding_map = embeddings_by_owner.get(OWNER_MASK, {})
    centroid_embedding_map = embeddings_by_owner.get(OWNER_CENTROID, {})
    bbox_embedding_map = embeddings_by_owner.get(OWNER_BBOX, {})
    roi_embedding_map = embeddings_by_owner.get(OWNER_ROI, {})
    # Category embeddings (SLP 2.7+): parallel category_* datasets in /embeddings.
    category_embeddings_by_owner = read_category_embeddings(labels_path, _hdf5_file=f)
    instance_category_embeddings = category_embeddings_by_owner.get(OWNER_INSTANCE, {})
    mask_category_embedding_map = category_embeddings_by_owner.get(OWNER_MASK, {})
    centroid_category_embedding_map = category_embeddings_by_owner.get(
        OWNER_CENTROID, {}
    )
    bbox_category_embedding_map = category_embeddings_by_owner.get(OWNER_BBOX, {})
    roi_category_embedding_map = category_embeddings_by_owner.get(OWNER_ROI, {})
    suggestions = read_suggestions(labels_path, videos, _hdf5_file=f)
    metadata = read_metadata(labels_path, _hdf5_file=f)
    provenance = read_provenance(labels_path, metadata, _hdf5_file=f)
    negative_frames = read_negative_frames(labels_path, _hdf5_file=f)

    # Read sessions (small, no need for lazy loading)
    # Note: sessions require labeled_frames for full linking, but for lazy loading
    # we pass an empty list since we don't have materialized frames yet (frame
    # groups are not reconstructed). The raw sessions_json bytes + columnar
    # /session_data arrays are captured verbatim so a lazy re-save can copy the
    # frame-group / 3D tables losslessly without materializing frames (SLP 2.8+).
    sessions = read_sessions(
        labels_path, videos, [], identities=identities, _hdf5_file=f
    )
    sessions_json_raw = f["sessions_json"][:] if "sessions_json" in f else None
    session_data_raw = _read_session_data(labels_path, _hdf5_file=f)

    # Create LazyDataStore
    lazy_store = LazyDataStore(
        sessions_json_raw=sessions_json_raw,
        session_data=session_data_raw,
        session_video_ids=tuple(id(v) for v in videos),
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
        identities=identities,
        instance_identities=instance_identities,
        instance_embeddings=instance_embeddings,
        categories=categories,
        instance_categories=instance_categories,
        instance_category_embeddings=instance_category_embeddings,
    )

    # Create LazyFrameList
    lazy_frames = LazyFrameList(lazy_store)

    # Read ROIs, masks, bboxes, and label images eagerly (typically small count).
    # Masks/centroids/bboxes/ROIs are eager even in lazy mode, so their identity +
    # embeddings attach now by global per-modality list index.
    roi_tuples = read_rois(labels_path, videos, tracks, _hdf5_file=f)
    _attach_identity_and_embeddings(
        [r for r, _v, _f in roi_tuples],
        identities,
        roi_identity_map,
        roi_embedding_map,
    )
    _attach_category_and_embeddings(
        [r for r, _v, _f in roi_tuples],
        categories,
        roi_category_map,
        roi_category_embedding_map,
    )
    mask_tuples = read_masks(labels_path, videos, tracks, _hdf5_file=f)
    _attach_identity_and_embeddings(
        [m for m, _v, _f in mask_tuples],
        identities,
        mask_identity_map,
        mask_embedding_map,
    )
    _attach_category_and_embeddings(
        [m for m, _v, _f in mask_tuples],
        categories,
        mask_category_map,
        mask_category_embedding_map,
    )
    bbox_tuples = read_bboxes(labels_path, videos, tracks, _hdf5_file=f)
    _attach_identity_and_embeddings(
        [b for b, _v, _f in bbox_tuples],
        identities,
        bbox_identity_map,
        bbox_embedding_map,
    )
    _attach_category_and_embeddings(
        [b for b, _v, _f in bbox_tuples],
        categories,
        bbox_category_map,
        bbox_category_embedding_map,
    )
    centroid_tuples = read_centroids(labels_path, videos, tracks, _hdf5_file=f)
    _attach_identity_and_embeddings(
        [c for c, _v, _f in centroid_tuples],
        identities,
        centroid_identity_map,
        centroid_embedding_map,
    )
    _attach_category_and_embeddings(
        [c for c, _v, _f in centroid_tuples],
        categories,
        centroid_category_map,
        centroid_category_embedding_map,
    )
    li_tuples, li_file = read_label_images(
        labels_path,
        videos,
        tracks,
        _hdf5_file=f,
        _url_headers=_url_headers,
        _url_stream_mode=_url_stream_mode,
        _url_bytes=_url_bytes,
    )

    # Build per-frame annotation dicts for lazy materialization
    def _build_ann_by_frame(ann_tuples):
        by_frame = {}
        undistributed = []
        for ann, vid_idx, fidx in ann_tuples:
            if vid_idx >= 0 and fidx >= 0:
                key = (vid_idx, fidx)
                by_frame.setdefault(key, []).append(ann)
            else:
                undistributed.append(ann)
        return by_frame, undistributed

    c_by_frame, c_undist = _build_ann_by_frame(centroid_tuples)
    b_by_frame, b_undist = _build_ann_by_frame(bbox_tuples)
    m_by_frame, m_undist = _build_ann_by_frame(mask_tuples)
    r_by_frame, r_undist = _build_ann_by_frame(roi_tuples)

    lazy_store._centroid_by_frame = c_by_frame
    lazy_store._bbox_by_frame = b_by_frame
    lazy_store._mask_by_frame = m_by_frame
    lazy_store._roi_by_frame = r_by_frame

    lazy_store._undistributed_centroids = c_undist
    lazy_store._undistributed_bboxes = b_undist
    lazy_store._undistributed_masks = m_undist
    lazy_store._undistributed_rois = r_undist

    # Label images use the same pattern
    li_by_frame, li_undist = _build_ann_by_frame(li_tuples)
    lazy_store._label_image_by_frame = li_by_frame
    lazy_store._undistributed_label_images = li_undist

    # Check for annotation-only frames not in /frames (e.g., old TrackMate SLPs)
    frame_keys = set()
    for row in frames_data:
        frame_keys.add((int(row[1]), int(row[2])))  # (video_id, frame_idx)

    all_ann_keys = set()
    for d in (
        lazy_store._centroid_by_frame,
        lazy_store._bbox_by_frame,
        lazy_store._mask_by_frame,
        lazy_store._label_image_by_frame,
        lazy_store._roi_by_frame,
    ):
        all_ann_keys.update(d.keys())

    # Create non-lazy frames for annotations without matching /frames entries
    annotation_only_frames = []
    for vid_idx, fidx in sorted(all_ann_keys - frame_keys):
        if 0 <= vid_idx < len(videos):
            key = (vid_idx, fidx)
            annotation_only_frames.append(
                LabeledFrame(
                    video=videos[vid_idx],
                    frame_idx=fidx,
                    centroids=lazy_store._centroid_by_frame.get(key, []),
                    bboxes=lazy_store._bbox_by_frame.get(key, []),
                    masks=lazy_store._mask_by_frame.get(key, []),
                    label_images=lazy_store._label_image_by_frame.get(key, []),
                    rois=lazy_store._roi_by_frame.get(key, []),
                )
            )

    # Create Labels with lazy state (annotations are on the lazy store)
    labels = Labels(
        labeled_frames=lazy_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        identities=identities,
        categories=categories,
        suggestions=suggestions,
        sessions=sessions,
        provenance=provenance,
        event_types=event_types,
        events=events,
        lazy_store=lazy_store,
    )

    # Add annotation-only frames as supplementary (non-lazy) frames
    if annotation_only_frames:
        lazy_frames._supplementary.extend(annotation_only_frames)
    labels.provenance["filename"] = labels_path

    # Keep the HDF5 file handle alive for lazy label image data
    if li_file is not None:
        labels._label_image_file = li_file

    return labels


def read_rois(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    *,
    _hdf5_file: h5py.File | None = None,
) -> list[tuple[ROI, int, int]]:
    """Read ROI annotations from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking ROI instance associations. If ``None``, instance
            associations will not be restored.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of ``(roi, video_idx, frame_idx)`` tuples. ``video_idx`` and
        ``frame_idx`` are the routing context read from the file (``-1``
        means undistributable). Returns an empty list if no ROIs are stored.
    """
    import shapely

    try:
        roi_data = read_hdf5_dataset(labels_path, "rois", _hdf5_file=_hdf5_file)
    except KeyError:
        return []

    if len(roi_data) == 0:
        return []

    # Read packed WKB geometry bytes
    try:
        roi_wkb_flat = read_hdf5_dataset(labels_path, "roi_wkb", _hdf5_file=_hdf5_file)
    except KeyError:
        return []

    # Read string metadata from string datasets first, fall back to JSON attributes
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
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

        tracking_score = None
        if "tracking_score" in row.dtype.names:
            ts = float(row["tracking_score"])
            if not np.isnan(ts):
                tracking_score = ts

        kwargs = dict(
            geometry=geometry,
            name=names[i] if i < len(names) else "",
            category=categories[i] if i < len(categories) else "",
            source=sources[i] if i < len(sources) else "",
            video=video,
            track=track,
            tracking_score=tracking_score,
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
        rois.append((roi, video_idx, frame_idx_val))

    return rois


def write_rois(
    labels_path: str,
    rois: list[ROI],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
) -> None:
    """Write ROI annotations to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        rois: A list of ROI objects to write.
        videos: List of Video objects for index mapping.
        tracks: List of Track objects for index mapping.
        instances: Optional list of Instance/PredictedInstance objects for index
            mapping. If provided, ROI instance associations will be persisted.
        contexts: Parallel list of ``(video_idx, frame_idx)`` routing context
            for each ROI. If ``None``, defaults to ``(-1, -1)`` for all.
    """
    if not rois:
        return

    if contexts is None:
        contexts = [(-1, -1)] * len(rois)

    import shapely

    roi_dtype = np.dtype(
        [
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("is_predicted", "u1"),
            ("score", "f4"),
            ("tracking_score", "f4"),
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

    for i_roi, roi in enumerate(rois):
        wkb = shapely.to_wkb(roi.geometry)
        wkb_start = wkb_offset
        wkb_end = wkb_offset + len(wkb)
        wkb_chunks.append(np.frombuffer(wkb, dtype=np.uint8))
        wkb_offset = wkb_end

        video_idx, frame_idx = contexts[i_roi]
        track_idx = tracks.index(roi.track) if roi.track in tracks else -1

        instance_idx = roi._instance_idx  # Use stored index as default
        if instances is not None and roi.instance is not None:
            try:
                instance_idx = instances.index(roi.instance)
            except ValueError:
                pass  # Keep stored _instance_idx

        is_predicted = isinstance(roi, PredictedROI)
        score = roi.score if is_predicted else float("nan")
        tracking_score = (
            roi.tracking_score if roi.tracking_score is not None else float("nan")
        )

        roi_rows.append(
            (
                0,  # annotation_type: write 0 (DEFAULT) for backward compat
                video_idx,
                frame_idx,
                track_idx,
                int(is_predicted),
                score,
                tracking_score,
                wkb_start,
                wkb_end,
                instance_idx,
            )
        )

        categories.append(roi.category.name if roi.category else "")
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
        # Gzip-compress the packed WKB geometry bytes (lossless, transparent on
        # read). Mirrors the gzip used for mask_rle / video / label-image data;
        # ~2x smaller on polygon-heavy ROI sets. wkb_flat is always non-empty
        # here: write_rois returns early on empty input and every geometry
        # serializes to >=9 WKB bytes (even an empty polygon is a 9-byte
        # header), so no empty-dataset guard is needed.
        f.create_dataset(
            "roi_wkb",
            data=wkb_flat,
            dtype=np.uint8,
            chunks=True,
            compression="gzip",
            compression_opts=1,
        )


def read_bboxes(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    *,
    _hdf5_file: h5py.File | None = None,
) -> list[tuple[BoundingBox, int, int]]:
    """Read bounding box annotations from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking bounding box instance associations. If ``None``, instance
            associations will not be restored.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of ``(bbox, video_idx, frame_idx)`` tuples. ``video_idx`` and
        ``frame_idx`` are the routing context read from the file (``-1``
        means undistributable). Returns an empty list if no bboxes are stored.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
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
) -> list[tuple[BoundingBox, int, int]]:
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
    tracking_score_arr = grp["tracking_score"][:] if "tracking_score" in grp else None

    bboxes: list[BoundingBox] = []
    for i in range(len(x1_arr)):
        video_idx = int(video_arr[i])

        frame_idx_val = int(frame_idx_arr[i])

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

        tracking_score = None
        if tracking_score_arr is not None:
            ts = float(tracking_score_arr[i])
            if not np.isnan(ts):
                tracking_score = ts

        kwargs = dict(
            x1=float(x1_arr[i]),
            y1=float(y1_arr[i]),
            x2=float(x2_arr[i]),
            y2=float(y2_arr[i]),
            angle=float(angle_arr[i]),
            track=track,
            tracking_score=tracking_score,
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
        bboxes.append((bbox, video_idx, frame_idx_val))

    return bboxes


def _read_bboxes_legacy(
    bbox_data: np.ndarray,
    categories: list[str],
    names: list[str],
    sources: list[str],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None,
) -> list[tuple[BoundingBox, int, int]]:
    """Read bboxes from legacy structured array format (pre-v2.0)."""
    if len(bbox_data) == 0:
        return []

    bboxes: list[BoundingBox] = []
    for i, row in enumerate(bbox_data):
        video_idx = int(row["video"])

        frame_idx_val = int(row["frame_idx"])

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
        bboxes.append((bbox, video_idx, frame_idx_val))

    return bboxes


def write_bboxes(
    labels_path: str,
    bboxes: list[BoundingBox],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
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
        contexts: Parallel list of ``(video_idx, frame_idx)`` routing context
            for each bounding box. If ``None``, defaults to ``(-1, -1)`` for all.
    """
    if not bboxes:
        return

    if contexts is None:
        contexts = [(-1, -1)] * len(bboxes)

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
    tracking_score_arr = np.empty(n, dtype=np.float32)
    categories = []
    names = []
    sources = []

    for i, bbox in enumerate(bboxes):
        x1_arr[i] = bbox.x1
        y1_arr[i] = bbox.y1
        x2_arr[i] = bbox.x2
        y2_arr[i] = bbox.y2
        angle_arr[i] = bbox.angle

        video_arr[i], frame_idx_arr[i] = contexts[i]
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
        tracking_score_arr[i] = (
            bbox.tracking_score if bbox.tracking_score is not None else float("nan")
        )

        categories.append(bbox.category.name if bbox.category else "")
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
        grp.create_dataset("tracking_score", data=tracking_score_arr)
        grp.create_dataset("category", data=categories, dtype=str_dt)
        grp.create_dataset("name", data=names, dtype=str_dt)
        grp.create_dataset("source", data=sources, dtype=str_dt)


def read_centroids(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    *,
    _hdf5_file: h5py.File | None = None,
) -> "list[tuple[Centroid, int, int]]":
    """Read centroid annotations from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking centroid instance associations.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of ``(centroid, video_idx, frame_idx)`` tuples. ``video_idx``
        and ``frame_idx`` are the routing context read from the file (``-1``
        means undistributable). Returns an empty list if no centroids are
        stored.
    """
    from sleap_io.model.centroid import PredictedCentroid, UserCentroid

    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        if "centroids" not in f:
            return []

        grp = f["centroids"]
        x_arr = grp["x"][:]
        y_arr = grp["y"][:]
        z_arr = grp["z"][:] if "z" in grp else None
        video_arr = grp["video"][:]
        frame_idx_arr = grp["frame_idx"][:]
        track_arr = grp["track"][:]
        instance_arr = grp["instance"][:]
        is_predicted_arr = grp["is_predicted"][:]
        score_arr = grp["score"][:]
        tracking_score_arr = (
            grp["tracking_score"][:] if "tracking_score" in grp else None
        )
        category_arr = grp["category"][:]
        name_arr = grp["name"][:]
        source_arr = grp["source"][:]

    centroids: list[Centroid] = []
    for i in range(len(x_arr)):
        video_idx = int(video_arr[i])

        frame_idx_val = int(frame_idx_arr[i])

        track_idx = int(track_arr[i])
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None

        instance_idx = int(instance_arr[i])
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )

        z = None
        if z_arr is not None:
            z_val = float(z_arr[i])
            if not np.isnan(z_val):
                z = z_val

        tracking_score = None
        if tracking_score_arr is not None:
            ts = float(tracking_score_arr[i])
            if not np.isnan(ts):
                tracking_score = ts

        cat = category_arr[i]
        category = cat.decode() if isinstance(cat, bytes) else str(cat)
        nm = name_arr[i]
        name = nm.decode() if isinstance(nm, bytes) else str(nm)
        src = source_arr[i]
        source = src.decode() if isinstance(src, bytes) else str(src)

        kwargs = dict(
            x=float(x_arr[i]),
            y=float(y_arr[i]),
            z=z,
            track=track,
            tracking_score=tracking_score,
            instance=instance,
            category=category,
            name=name,
            source=source,
        )

        if bool(is_predicted_arr[i]):
            centroid = PredictedCentroid(score=float(score_arr[i]), **kwargs)
        else:
            centroid = UserCentroid(**kwargs)

        centroid._instance_idx = instance_idx
        centroids.append((centroid, video_idx, frame_idx_val))

    return centroids


def write_centroids(
    labels_path: str,
    centroids: "list[Centroid]",
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
) -> None:
    """Write centroid annotations to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        centroids: A list of Centroid objects to write.
        videos: List of Video objects for index mapping.
        tracks: List of Track objects for index mapping.
        instances: Optional list of Instance/PredictedInstance objects for index
            mapping.
        contexts: Parallel list of ``(video_idx, frame_idx)`` routing context
            for each centroid. If ``None``, defaults to ``(-1, -1)`` for all.
    """
    from sleap_io.model.centroid import PredictedCentroid

    if not centroids:
        return

    if contexts is None:
        contexts = [(-1, -1)] * len(centroids)

    n = len(centroids)
    x_arr = np.empty(n, dtype=np.float64)
    y_arr = np.empty(n, dtype=np.float64)
    z_arr = np.empty(n, dtype=np.float64)
    video_arr = np.empty(n, dtype=np.int32)
    frame_idx_arr = np.empty(n, dtype=np.int64)
    track_arr = np.empty(n, dtype=np.int32)
    instance_arr = np.empty(n, dtype=np.int32)
    is_predicted_arr = np.empty(n, dtype=np.uint8)
    score_arr = np.empty(n, dtype=np.float32)
    tracking_score_arr = np.empty(n, dtype=np.float32)
    categories = []
    names = []
    sources = []

    for i, centroid in enumerate(centroids):
        x_arr[i] = centroid.x
        y_arr[i] = centroid.y
        z_arr[i] = centroid.z if centroid.z is not None else float("nan")

        video_arr[i], frame_idx_arr[i] = contexts[i]
        track_arr[i] = tracks.index(centroid.track) if centroid.track in tracks else -1

        instance_idx = centroid._instance_idx
        if instances is not None and centroid.instance is not None:
            try:
                instance_idx = instances.index(centroid.instance)
            except ValueError:
                pass
        instance_arr[i] = instance_idx

        is_predicted = isinstance(centroid, PredictedCentroid)
        is_predicted_arr[i] = int(is_predicted)
        score_arr[i] = centroid.score if is_predicted else float("nan")
        tracking_score_arr[i] = (
            centroid.tracking_score
            if centroid.tracking_score is not None
            else float("nan")
        )

        categories.append(centroid.category.name if centroid.category else "")
        names.append(centroid.name)
        sources.append(centroid.source)

    str_dt = h5py.special_dtype(vlen=str)
    with h5py.File(labels_path, "a") as f:
        grp = f.create_group("centroids")
        grp.create_dataset("x", data=x_arr)
        grp.create_dataset("y", data=y_arr)
        grp.create_dataset("z", data=z_arr)
        grp.create_dataset("video", data=video_arr)
        grp.create_dataset("frame_idx", data=frame_idx_arr)
        grp.create_dataset("track", data=track_arr)
        grp.create_dataset("instance", data=instance_arr)
        grp.create_dataset("is_predicted", data=is_predicted_arr)
        grp.create_dataset("score", data=score_arr)
        grp.create_dataset("tracking_score", data=tracking_score_arr)
        grp.create_dataset("category", data=categories, dtype=str_dt)
        grp.create_dataset("name", data=names, dtype=str_dt)
        grp.create_dataset("source", data=sources, dtype=str_dt)


def read_masks(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    *,
    _hdf5_file: h5py.File | None = None,
) -> list[tuple[SegmentationMask, int, int]]:
    """Read segmentation masks from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking mask instance associations. If ``None``, instance
            associations will not be restored.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of ``(mask, video_idx, frame_idx)`` tuples. ``video_idx`` and
        ``frame_idx`` are the routing context read from the file (``-1``
        means undistributable). Returns empty list if none stored.
    """
    try:
        mask_data = read_hdf5_dataset(labels_path, "masks", _hdf5_file=_hdf5_file)
    except KeyError:
        return []

    if len(mask_data) == 0:
        return []

    # Read packed RLE bytes
    try:
        mask_rle_flat = read_hdf5_dataset(
            labels_path, "mask_rle", _hdf5_file=_hdf5_file
        )
    except KeyError:
        return []

    # Read string metadata and score maps in a single file open
    score_map_by_idx: dict[int, np.ndarray] = {}
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
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
    from_predicted_pairs = []
    for i, row in enumerate(mask_data):
        rle_start = int(row["rle_start"])
        rle_end = int(row["rle_end"])
        rle_raw = mask_rle_flat[rle_start:rle_end]

        # Convert packed uint8 bytes back to uint32 array
        rle_counts = np.frombuffer(rle_raw.tobytes(), dtype=np.uint32)

        video_idx = int(row["video"])

        frame_idx_val = int(row["frame_idx"])

        track_idx = int(row["track"])
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None

        # Read instance index (v1.9+)
        instance_idx = int(row["instance"]) if "instance" in row.dtype.names else -1
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )

        # Read from_predicted index (v2.4+). Resolved to an object in a deferred
        # pass below, once all masks in this flat list have been constructed.
        from_predicted_idx = (
            int(row["from_predicted"]) if "from_predicted" in row.dtype.names else -1
        )

        # Read predicted flag (v1.9+)
        is_predicted = (
            bool(row["is_predicted"]) if "is_predicted" in row.dtype.names else False
        )
        score_val = float(row["score"]) if "score" in row.dtype.names else float("nan")

        tracking_score = None
        if "tracking_score" in row.dtype.names:
            ts = float(row["tracking_score"])
            if not np.isnan(ts):
                tracking_score = ts

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
            track=track,
            tracking_score=tracking_score,
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
        # Only user masks carry a from_predicted link; queue it for re-linking.
        if not is_predicted and from_predicted_idx >= 0:
            from_predicted_pairs.append((i, from_predicted_idx))
        masks.append((mask, video_idx, frame_idx_val))

    # Re-link from_predicted provenance now that every mask exists. The stored
    # value is a global index into this flat list (see write_masks); out-of-range
    # indices (corrupt/edited files) are skipped, leaving from_predicted as None.
    for i, fp_idx in from_predicted_pairs:
        if 0 <= fp_idx < len(masks):
            masks[i][0].from_predicted = masks[fp_idx][0]

    return masks


def write_masks(
    labels_path: str,
    masks: list[SegmentationMask],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
) -> None:
    """Write segmentation masks to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        masks: A list of SegmentationMask objects to write.
        videos: List of Video objects for index mapping.
        tracks: List of Track objects for index mapping.
        instances: Optional list of Instance/PredictedInstance objects for index
            mapping. If provided, mask instance associations will be persisted.
        contexts: Parallel list of ``(video_idx, frame_idx)`` routing context
            for each mask. If ``None``, defaults to ``(-1, -1)`` for all.
    """
    if not masks:
        return

    if contexts is None:
        contexts = [(-1, -1)] * len(masks)

    mask_dtype = np.dtype(
        [
            ("height", "u4"),
            ("width", "u4"),
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("instance", "i4"),
            ("from_predicted", "i4"),  # Format 2.4+: index into this mask list
            ("is_predicted", "u1"),
            ("score", "f4"),
            ("tracking_score", "f4"),
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

    # Map each mask object to its global index so a UserSegmentationMask's
    # ``from_predicted`` link can be persisted as an index into this flat list,
    # mirroring the instance ``from_predicted`` mechanism. The link resolves as
    # long as the source prediction is anywhere in this list (typically the same
    # frame); a source absent from the saved labels resolves to -1.
    mask_id_to_idx = {id(mask): i for i, mask in enumerate(masks)}

    for i_mask, mask in enumerate(masks):
        # Pack uint32 RLE counts as raw bytes (uint8)
        rle_bytes = mask.rle_counts.astype(np.uint32).tobytes()
        rle_uint8 = np.frombuffer(rle_bytes, dtype=np.uint8)
        rle_start = rle_offset
        rle_end = rle_offset + len(rle_uint8)
        rle_chunks.append(rle_uint8)
        rle_offset = rle_end

        video_idx, frame_idx = contexts[i_mask]
        track_idx = tracks.index(mask.track) if mask.track in tracks else -1

        instance_idx = mask._instance_idx  # Use stored index as default
        if instances is not None and mask.instance is not None:
            try:
                instance_idx = instances.index(mask.instance)
            except ValueError:
                pass  # Keep stored _instance_idx

        # Resolve the ``from_predicted`` provenance link to a global mask index
        # (-1 if absent, or if the source prediction was removed from the
        # labels). Only UserSegmentationMask carries this attribute.
        from_predicted_src = getattr(mask, "from_predicted", None)
        from_predicted_idx = (
            mask_id_to_idx.get(id(from_predicted_src), -1)
            if from_predicted_src is not None
            else -1
        )

        is_predicted = isinstance(mask, PredictedSegmentationMask)
        score = mask.score if is_predicted else float("nan")
        tracking_score = (
            mask.tracking_score if mask.tracking_score is not None else float("nan")
        )

        mask_rows.append(
            (
                mask.height,
                mask.width,
                2,  # annotation_type: write SEGMENTATION (2) for backward compat
                video_idx,
                frame_idx,
                track_idx,
                instance_idx,
                from_predicted_idx,
                int(is_predicted),
                score,
                tracking_score,
                rle_start,
                rle_end,
                mask.scale[0],
                mask.scale[1],
                mask.offset[0],
                mask.offset[1],
            )
        )

        categories.append(mask.category.name if mask.category else "")
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
            # Gzip-compress the RLE bytes (lossless, transparent on read).
            # RLE counts are uint32 run-lengths whose high bytes are mostly
            # zero, so they compress ~8x+ at level 1. Mirrors the gzip used for
            # video/label-image data. A degenerate empty-RLE mask (e.g. a
            # zero-height mask) keeps the contiguous, no-filter path: there are
            # no bytes to compress, so chunking + gzip would only add overhead.
            **(
                {"chunks": True, "compression": "gzip", "compression_opts": 1}
                if len(rle_flat) > 0
                else {}
            ),
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
    *,
    _hdf5_file: h5py.File | None = None,
    _url_headers: dict[str, str] | None = None,
    _url_stream_mode: str = "blockcache",
    _url_bytes: bytes | None = None,
) -> tuple[list[tuple[LabelImage, int, int]], "h5py.File | None"]:
    """Read label image annotations from a SLEAP labels file.

    Supports both the legacy blob format (v1.8-v2.1) and the chunked format
    (v2.2+). Pixel data is loaded lazily when possible: the HDF5 file handle
    is kept open and each frame's data is decompressed on first ``.data``
    access.

    Args:
        labels_path: A string path or URL to the SLEAP labels file.
        videos: List of Video objects for relinking.
        tracks: List of Track objects for relinking.
        instances: Optional list of Instance/PredictedInstance objects for
            relinking label image instance associations. If ``None``, instance
            associations will not be restored.
        _hdf5_file: An already-open `h5py.File` handle to use for the metadata
            reads (``label_images`` and ``label_image_objects``). If provided,
            those reads use this handle (left open for the caller to close).
            A separate long-lived handle is always opened for lazy pixel data
            access regardless of this argument, since that handle must outlive
            the caller's `with` block. This is a private argument used to thread
            a single open handle through reads.
        _url_headers: HTTP headers used when ``labels_path`` is a URL and the
            long-lived lazy-access handle must be opened over the network.
            Ignored for local paths.
        _url_stream_mode: Streaming strategy for the long-lived lazy-access
            handle when ``labels_path`` is a URL. Ignored for local paths.
        _url_bytes: Already-downloaded file bytes to wrap in a fresh in-memory
            handle for lazy pixel access, instead of re-opening ``labels_path``
            over the network. Used for Google Drive (where a re-resolve is a
            re-download against the per-file quota). Ignored for local paths.

    Returns:
        A tuple of ``(label_image_tuples, h5py_file)`` where
        ``label_image_tuples`` is a list of ``(li, video_idx, frame_idx)``
        tuples with routing context (``-1`` means undistributable), and
        ``h5py_file`` is the open HDF5 file handle that must be kept alive
        for lazy data access (or ``None`` if no label images were found).
        The caller is responsible for storing and eventually closing the
        file handle.
    """
    try:
        li_data = read_hdf5_dataset(labels_path, "label_images", _hdf5_file=_hdf5_file)
    except KeyError:
        return [], None

    if len(li_data) == 0:
        return [], None

    # Open a SECOND, long-lived file handle for lazy pixel data reading. This
    # handle intentionally outlives the orchestrator's `with` block (the closures
    # below capture it), so it cannot reuse `_hdf5_file`. For URL loads,
    # ``labels_path`` is the raw URL string, which ``h5py.File`` cannot open
    # directly; open a FRESH, independent fsspec file-like instead.
    import io

    from sleap_io.io import _remote

    if _url_bytes is not None:
        # Reuse already-downloaded bytes (e.g. a Google Drive prefetch) in a
        # fresh, independent in-memory handle instead of re-resolving the URL.
        f = h5py.File(io.BytesIO(_url_bytes), "r")
    elif _remote._is_url(labels_path):
        f = h5py.File(
            _remote.open_url(
                labels_path,
                headers=_url_headers,
                stream_mode=_url_stream_mode,
            ),
            "r",
        )
    else:
        f = h5py.File(labels_path, "r")

    if "label_image_data" not in f:
        f.close()
        return [], None

    pixel_ds = f["label_image_data"]
    is_chunked_format = pixel_ds.ndim == 3  # (T, H, W) = v2.2+ chunked

    # Read objects table
    try:
        obj_data = read_hdf5_dataset(
            labels_path, "label_image_objects", _hdf5_file=_hdf5_file
        )
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

        frame_idx_val = int(row["frame_idx"])

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

                obj_tracking_score = None
                if "tracking_score" in obj_row.dtype.names:
                    ts = float(obj_row["tracking_score"])
                    if not np.isnan(ts):
                        obj_tracking_score = ts

                objects[label_id] = LabelImage.Info(
                    track=track,
                    tracking_score=obj_tracking_score,
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

        label_images.append((li, video_idx, frame_idx_val))

    return label_images, f


def write_label_images(
    labels_path: str,
    label_images: list[LabelImage],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
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
        contexts: Parallel list of ``(video_idx, frame_idx)`` routing context
            for each label image. If ``None``, defaults to ``(-1, -1)`` for all.
    """
    if not label_images:
        return

    if contexts is None:
        contexts = [(-1, -1)] * len(label_images)

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
            video_idx, frame_idx = contexts[i]

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
                obj_tracking_score = (
                    info.tracking_score
                    if info.tracking_score is not None
                    else float("nan")
                )
                obj_rows.append(
                    (label_id, track_idx, instance_idx, obj_score, obj_tracking_score)
                )
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
        li_array = np.array(li_rows, dtype=LI_DTYPE)
        obj_array = (
            np.array(obj_rows, dtype=OBJ_DTYPE)
            if obj_rows
            else np.array([], dtype=OBJ_DTYPE)
        )
        f.create_dataset("label_images", data=li_array, dtype=LI_DTYPE)
        f.create_dataset("label_image_objects", data=obj_array, dtype=OBJ_DTYPE)
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
        sm_index_array = np.array(li_sm_indices, dtype=LI_SM_INDEX_DTYPE)
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
    videos: list[Video] | None = None,
) -> None:
    """Write minimal metadata group to an SLP file.

    This is used by ``LabelImageWriter`` to write format info without requiring
    a full ``Labels`` object.

    Args:
        labels_path: Path to the SLP file.
        format_id: Format version identifier.
        skeletons: Optional list of skeletons to serialize.
        provenance: Optional provenance dict.
        videos: Optional list of videos; if any carries a virtual crop the
            ``format_id`` is bumped to 2.3 (matching ``write_metadata``).
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

    if videos and any(v._crop_tuple() is not None for v in videos):
        format_id = max(format_id, 2.3)

    with h5py.File(labels_path, "a") as f:
        # Bump for chunked label image format
        if "label_image_data" in f and f["label_image_data"].ndim == 3:
            format_id = max(format_id, 2.2)

        grp = f.require_group("metadata")
        grp.attrs["format_id"] = format_id
        # Mirror write_metadata: provenance goes to a dedicated dataset (no 64 KB
        # attribute limit), kept in the metadata/json attribute too when it fits.
        _write_provenance_dataset(f, provenance)
        grp.attrs["json"] = _encode_metadata_attr(md)

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
            tracks: Optional initial list of tracks for object associations.
                New tracks encountered in ``add()`` calls are appended
                automatically.
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

    def add(
        self,
        label_image: LabelImage,
        video_idx: int = -1,
        frame_idx: int = -1,
    ) -> None:
        """Add a single label image to the file.

        The first call creates the HDF5 file and locks the frame dimensions.
        Subsequent calls must provide frames with the same ``(H, W)``.

        Args:
            label_image: The label image to write.
            video_idx: Video index for routing context. Defaults to ``-1``.
            frame_idx: Frame index for routing context. Defaults to ``-1``.

        Raises:
            ValueError: If the frame dimensions don't match the first frame.
            RuntimeError: If the writer has already been finalized.
        """
        if self._finalized:
            raise RuntimeError("Writer has already been finalized.")

        # Auto-resolve video_idx from self.video when not explicitly provided
        if video_idx == -1 and self.video is not None:
            video_idx = 0
        # Auto-resolve frame_idx from write count when not explicitly provided
        if frame_idx == -1:
            frame_idx = self._count

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

        # Use provided routing context (video_idx, frame_idx are parameters)

        # Build object rows (auto-collect new tracks)
        n_objects = len(label_image.objects)
        objects_start = self._obj_offset
        _track_set = set(self.tracks)

        for label_id in sorted(label_image.objects):
            info = label_image.objects[label_id]
            if info.track is not None and info.track not in _track_set:
                self.tracks.append(info.track)
                _track_set.add(info.track)
            track_idx = (
                self.tracks.index(info.track) if info.track in self.tracks else -1
            )
            instance_idx = info._instance_idx
            obj_score = info.score if info.score is not None else float("nan")
            obj_tracking_score = (
                info.tracking_score if info.tracking_score is not None else float("nan")
            )
            self._obj_rows.append(
                (label_id, track_idx, instance_idx, obj_score, obj_tracking_score)
            )
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
            write_video_crops(self.path, Labels(videos=videos))
            write_tracks(self.path, self.tracks)
            _write_metadata_standalone(self.path, skeletons=skeletons, videos=videos)
            return Labels(
                videos=videos,
                skeletons=skeletons,
                tracks=self.tracks,
            )

        # Trim pixel dataset to actual count
        self._pixel_dset.resize(self._count, axis=0)

        # Write metadata datasets
        li_array = np.array(self._li_rows, dtype=LI_DTYPE)
        obj_array = (
            np.array(self._obj_rows, dtype=OBJ_DTYPE)
            if self._obj_rows
            else np.array([], dtype=OBJ_DTYPE)
        )

        str_dt = h5py.special_dtype(vlen=str)
        f = self._file
        f.create_dataset("label_images", data=li_array, dtype=LI_DTYPE)
        f.create_dataset("label_image_objects", data=obj_array, dtype=OBJ_DTYPE)
        f.create_dataset("label_image_sources", data=self._sources, dtype=str_dt)
        f.create_dataset(
            "label_image_obj_categories", data=self._categories, dtype=str_dt
        )
        f.create_dataset("label_image_obj_names", data=self._obj_names, dtype=str_dt)

        # Write score maps if any
        if self._sm_indices:
            sm_index_array = np.array(self._sm_indices, dtype=LI_SM_INDEX_DTYPE)
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
        write_video_crops(self.path, Labels(videos=videos))
        write_tracks(self.path, self.tracks)
        _write_metadata_standalone(self.path, skeletons=skeletons, videos=videos)

        li_tuples, li_file = read_label_images(self.path, videos, self.tracks, [])
        # Distribute label images to frames
        labeled_frames = []
        frame_lookup: dict[tuple[int, int], LabeledFrame] = {}
        for li, vid_idx, fidx in li_tuples:
            key = (vid_idx, fidx)
            if key not in frame_lookup:
                video = videos[vid_idx] if 0 <= vid_idx < len(videos) else None
                if video is not None:
                    lf = LabeledFrame(video=video, frame_idx=fidx)
                    labeled_frames.append(lf)
                    frame_lookup[key] = lf
            if key in frame_lookup:
                frame_lookup[key].label_images.append(li)
        labels = Labels(
            labeled_frames=labeled_frames,
            videos=videos,
            skeletons=skeletons,
            tracks=self.tracks,
        )
        if li_file is not None:
            labels._label_image_file = li_file
        return labels

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
            seen_filenames: dict[str | tuple[str, ...], int] = {}
            for vlist in source_videos:
                for v in vlist:
                    fn = v.filename
                    # ImageVideo filenames are lists, convert to tuple for hashing
                    fn_key = tuple(fn) if isinstance(fn, list) else fn
                    if fn_key not in seen_filenames:
                        seen_filenames[fn_key] = len(merged_videos)
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
                        fn_key = tuple(fn) if isinstance(fn, list) else fn
                        video_remap[i] = seen_filenames[fn_key]

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
                            obj_tracking_score = (
                                float(obj_row["tracking_score"])
                                if "tracking_score" in obj_row.dtype.names
                                else float("nan")
                            )

                            dest_obj_rows.append(
                                (
                                    label_id,
                                    new_track_idx,
                                    instance_idx,
                                    obj_score,
                                    obj_tracking_score,
                                )
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
            li_array = np.array(dest_li_rows, dtype=LI_DTYPE)
            obj_array = (
                np.array(dest_obj_rows, dtype=OBJ_DTYPE)
                if dest_obj_rows
                else np.array([], dtype=OBJ_DTYPE)
            )
            str_dt = h5py.special_dtype(vlen=str)
            dest_f.create_dataset("label_images", data=li_array, dtype=LI_DTYPE)
            dest_f.create_dataset(
                "label_image_objects", data=obj_array, dtype=OBJ_DTYPE
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
                sm_index_array = np.array(dest_sm_indices, dtype=LI_SM_INDEX_DTYPE)
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
        write_video_crops(dest_path, Labels(videos=merged_videos))
        write_tracks(dest_path, merged_tracks)
        _write_metadata_standalone(dest_path, videos=merged_videos)

        # Return Labels pointing at the merged file
        li_tuples, li_file = read_label_images(
            dest_path, merged_videos, merged_tracks, []
        )
        # Distribute label images to frames
        labeled_frames = []
        frame_lookup: dict[tuple[int, int], LabeledFrame] = {}
        for li, vid_idx, fidx in li_tuples:
            key = (vid_idx, fidx)
            if key not in frame_lookup:
                video = (
                    merged_videos[vid_idx]
                    if 0 <= vid_idx < len(merged_videos)
                    else None
                )
                if video is not None:
                    lf = LabeledFrame(video=video, frame_idx=fidx)
                    labeled_frames.append(lf)
                    frame_lookup[key] = lf
            if key in frame_lookup:
                frame_lookup[key].label_images.append(li)
        labels = Labels(
            labeled_frames=labeled_frames,
            videos=merged_videos,
            tracks=merged_tracks,
        )
        if li_file is not None:
            labels._label_image_file = li_file
        return labels

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
    with h5py.File(labels_path, "r") as f:
        return _read_labels_from_open_file(labels_path, f, open_videos=open_videos)


def _read_labels_from_open_file(
    labels_path: str,
    f: h5py.File,
    *,
    open_videos: bool = True,
    _url_headers: dict[str, str] | None = None,
    _url_stream_mode: str = "blockcache",
    _url_bytes: bytes | None = None,
) -> Labels:
    """Build a `Labels` from an already-open `h5py.File`.

    Threads `_hdf5_file=f` through every `read_*` helper so the entire eager
    read happens against a single open handle. Does NOT close `f` (the caller's
    `with` block owns it). The long-lived label-image handle returned by
    `read_label_images` (a separate, second handle) is attached to
    `labels._label_image_file` and intentionally outlives `f`.

    Args:
        labels_path: A string path or URL to the SLEAP labels file.
        f: An already-open `h5py.File` handle to read from.
        open_videos: If `True` (the default), attempt to open the video backend
            for I/O.
        _url_headers: HTTP headers forwarded to the long-lived label-image
            handle when `labels_path` is a URL. Ignored for local paths.
        _url_stream_mode: Streaming strategy for the long-lived label-image
            handle when `labels_path` is a URL. Ignored for local paths.
        _url_bytes: Already-downloaded file bytes to reuse for the long-lived
            label-image handle instead of re-opening `labels_path` over the
            network (used for Google Drive, where a re-resolve is a re-download
            against the per-file quota). Ignored for local paths.

    Returns:
        The processed `Labels` object.
    """
    tracks = read_tracks(labels_path, _hdf5_file=f)
    videos = read_videos(
        labels_path,
        open_backend=open_videos,
        _hdf5_file=f,
        _url_headers=_url_headers,
        _url_stream_mode=_url_stream_mode,
    )
    skeletons = read_skeletons(labels_path, _hdf5_file=f)
    points = read_points(labels_path, _hdf5_file=f)
    pred_points = read_pred_points(labels_path, _hdf5_file=f)
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id", _hdf5_file=f)
    instances = read_instances(
        labels_path, skeletons, tracks, points, pred_points, format_id, _hdf5_file=f
    )
    suggestions = read_suggestions(labels_path, videos, _hdf5_file=f)
    metadata = read_metadata(labels_path, _hdf5_file=f)
    provenance = read_provenance(labels_path, metadata, _hdf5_file=f)

    frames = read_hdf5_dataset(labels_path, "frames", _hdf5_file=f)
    negative_markers = read_negative_frames(labels_path, _hdf5_file=f)

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

    identities = read_identities(labels_path, _hdf5_file=f)
    # Class/category catalog (SLP 2.7+). Self-contained mirror of /identity; a
    # top-level list read eagerly like identities.
    categories = read_categories(labels_path, _hdf5_file=f)
    # Frame-spanning events + catalog (SLP 2.6+). Top-level lists, read eagerly like
    # tracks/identities/suggestions; absent groups yield empty lists.
    event_types = read_event_types(labels_path, _hdf5_file=f)
    events = read_events(
        labels_path, videos, event_types, tracks, identities, _hdf5_file=f
    )
    # Attach identity links (SLP format 2.5+). Additive: when the dataset is absent
    # (older files) these are empty mappings and a no-op. The global instances list
    # is ordered by instance_id, so it indexes directly; mask owners are attached
    # after read_masks (mask-list index order).
    identity_links = read_identity_links(labels_path, _hdf5_file=f)
    instance_identity_map = identity_links.get(OWNER_INSTANCE, {})
    mask_identity_map = identity_links.get(OWNER_MASK, {})
    centroid_identity_map = identity_links.get(OWNER_CENTROID, {})
    bbox_identity_map = identity_links.get(OWNER_BBOX, {})
    roi_identity_map = identity_links.get(OWNER_ROI, {})
    for inst_id, (identity_idx, identity_score) in instance_identity_map.items():
        if 0 <= inst_id < len(instances) and 0 <= identity_idx < len(identities):
            instances[inst_id].identity = identities[identity_idx]
            instances[inst_id].identity_score = identity_score

    # Attach category links (SLP format 2.7+). Self-contained mirror of the identity
    # links above; absent /categories group -> empty mappings and a no-op.
    category_links = read_category_links(labels_path, _hdf5_file=f)
    instance_category_map = category_links.get(OWNER_INSTANCE, {})
    mask_category_map = category_links.get(OWNER_MASK, {})
    centroid_category_map = category_links.get(OWNER_CENTROID, {})
    bbox_category_map = category_links.get(OWNER_BBOX, {})
    roi_category_map = category_links.get(OWNER_ROI, {})
    for inst_id, (category_idx, category_score) in instance_category_map.items():
        if 0 <= inst_id < len(instances) and 0 <= category_idx < len(categories):
            instances[inst_id].category = categories[category_idx]
            instances[inst_id].category_score = category_score

    # Attach appearance / re-ID embeddings (SLP format 2.5+). Additive: absent
    # /embeddings group -> empty mappings and a no-op. Mask/centroid/bbox/ROI
    # embeddings are attached after their respective read_* calls.
    embeddings_by_owner = read_embeddings(labels_path, _hdf5_file=f)
    for inst_id, emb in embeddings_by_owner.get(OWNER_INSTANCE, {}).items():
        if 0 <= inst_id < len(instances):
            instances[inst_id].identity_embedding = emb
    mask_embedding_map = embeddings_by_owner.get(OWNER_MASK, {})
    centroid_embedding_map = embeddings_by_owner.get(OWNER_CENTROID, {})
    bbox_embedding_map = embeddings_by_owner.get(OWNER_BBOX, {})
    roi_embedding_map = embeddings_by_owner.get(OWNER_ROI, {})

    # Attach category embeddings (SLP format 2.7+): parallel category_* datasets in
    # /embeddings. Absent -> empty mappings and a no-op.
    category_embeddings_by_owner = read_category_embeddings(labels_path, _hdf5_file=f)
    for inst_id, emb in category_embeddings_by_owner.get(OWNER_INSTANCE, {}).items():
        if 0 <= inst_id < len(instances):
            instances[inst_id].category_embedding = emb
    mask_category_embedding_map = category_embeddings_by_owner.get(OWNER_MASK, {})
    centroid_category_embedding_map = category_embeddings_by_owner.get(
        OWNER_CENTROID, {}
    )
    bbox_category_embedding_map = category_embeddings_by_owner.get(OWNER_BBOX, {})
    roi_category_embedding_map = category_embeddings_by_owner.get(OWNER_ROI, {})

    sessions = read_sessions(
        labels_path, videos, labeled_frames, identities=identities, _hdf5_file=f
    )
    roi_tuples = read_rois(labels_path, videos, tracks, instances, _hdf5_file=f)
    # Attach per-ROI identity links + embeddings (global ROI-list index: frame ROIs
    # then static ROIs, the same order write_rois emits).
    _attach_identity_and_embeddings(
        [r for r, _v, _f in roi_tuples],
        identities,
        roi_identity_map,
        roi_embedding_map,
    )
    _attach_category_and_embeddings(
        [r for r, _v, _f in roi_tuples],
        categories,
        roi_category_map,
        roi_category_embedding_map,
    )
    mask_tuples = read_masks(labels_path, videos, tracks, instances, _hdf5_file=f)
    # Attach per-mask identity links + embeddings (SLP format 2.5+/2.6+). Masks
    # are returned in global mask-list index order, the same id space written by
    # write_masks / write_identity_links / write_embeddings.
    _attach_identity_and_embeddings(
        [m for m, _v, _f in mask_tuples],
        identities,
        mask_identity_map,
        mask_embedding_map,
    )
    _attach_category_and_embeddings(
        [m for m, _v, _f in mask_tuples],
        categories,
        mask_category_map,
        mask_category_embedding_map,
    )
    bbox_tuples = read_bboxes(labels_path, videos, tracks, instances, _hdf5_file=f)
    # Attach per-bbox identity links + embeddings (global bbox-list index).
    _attach_identity_and_embeddings(
        [b for b, _v, _f in bbox_tuples],
        identities,
        bbox_identity_map,
        bbox_embedding_map,
    )
    _attach_category_and_embeddings(
        [b for b, _v, _f in bbox_tuples],
        categories,
        bbox_category_map,
        bbox_category_embedding_map,
    )
    centroid_tuples = read_centroids(
        labels_path, videos, tracks, instances, _hdf5_file=f
    )
    # Attach per-centroid identity links + embeddings (global centroid-list index).
    _attach_identity_and_embeddings(
        [c for c, _v, _f in centroid_tuples],
        identities,
        centroid_identity_map,
        centroid_embedding_map,
    )
    _attach_category_and_embeddings(
        [c for c, _v, _f in centroid_tuples],
        categories,
        centroid_category_map,
        centroid_category_embedding_map,
    )
    li_tuples, _li_file = read_label_images(
        labels_path,
        videos,
        tracks,
        instances,
        _hdf5_file=f,
        _url_headers=_url_headers,
        _url_stream_mode=_url_stream_mode,
        _url_bytes=_url_bytes,
    )

    # Attach annotations to their corresponding LabeledFrames
    frame_lookup: dict[tuple[int, int], LabeledFrame] = {}
    for lf in labeled_frames:
        vid_idx = videos.index(lf.video) if lf.video in videos else -1
        frame_lookup[(vid_idx, lf.frame_idx)] = lf

    def _get_or_create_frame(vid_idx, frame_idx):
        key = (vid_idx, frame_idx)
        if key not in frame_lookup:
            video = videos[vid_idx] if 0 <= vid_idx < len(videos) else None
            if video is not None:
                lf = LabeledFrame(video=video, frame_idx=frame_idx)
                labeled_frames.append(lf)
                frame_lookup[key] = lf
                return lf
            return None
        return frame_lookup[key]

    # Distribute annotations to frames, keeping undistributable ones
    undist_centroids = []
    for c, vid_idx, fidx in centroid_tuples:
        if vid_idx >= 0 and fidx >= 0:
            lf = _get_or_create_frame(vid_idx, fidx)
            if lf is not None:
                lf.centroids.append(c)
                continue
        undist_centroids.append(c)
    undist_bboxes = []
    for b, vid_idx, fidx in bbox_tuples:
        if vid_idx >= 0 and fidx >= 0:
            lf = _get_or_create_frame(vid_idx, fidx)
            if lf is not None:
                lf.bboxes.append(b)
                continue
        undist_bboxes.append(b)
    undist_masks = []
    for m, vid_idx, fidx in mask_tuples:
        if vid_idx >= 0 and fidx >= 0:
            lf = _get_or_create_frame(vid_idx, fidx)
            if lf is not None:
                lf.masks.append(m)
                continue
        undist_masks.append(m)
    undist_label_images = []
    for li, vid_idx, fidx in li_tuples:
        if vid_idx >= 0 and fidx >= 0:
            lf = _get_or_create_frame(vid_idx, fidx)
            if lf is not None:
                lf.label_images.append(li)
                continue
        undist_label_images.append(li)
    undist_rois = []
    for r, vid_idx, fidx in roi_tuples:
        if vid_idx >= 0 and fidx >= 0:
            lf = _get_or_create_frame(vid_idx, fidx)
            if lf is not None:
                lf.rois.append(r)
                continue
        else:
            undist_rois.append(r)

    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        identities=identities,
        categories=categories,
        suggestions=suggestions,
        sessions=sessions,
        provenance=provenance,
        event_types=event_types,
        events=events,
        rois=undist_rois,
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


# Top-level HDF5 names that sleap-io itself writes (regenerated from the model on
# every save). Anything else found in a source file is treated as "unknown" and,
# when ``preserve_unknown=True``, carried across a load/save cycle so a format
# addition made by a newer sleap-io version is not silently dropped by an older
# reader (which truncates and rebuilds the file from the model). Per-video embedded
# groups (``video0``, ``video1``, ...) are also known -- see
# ``_is_known_toplevel_name``. Forgetting to list a name here is safe: it would be
# stashed and then skipped at restore because the known writer already produced it
# (only marginally wasteful), so this never causes data loss or duplication.
_KNOWN_TOPLEVEL_HDF5_NAMES = frozenset(
    {
        "metadata",
        "videos_json",
        "tracks_json",
        "suggestions_json",
        "sessions_json",
        # Columnar RecordingSession frame-group data (SLP 2.8+): frame_groups /
        # instance_groups / instance_group_members / points_3d / pred_points_3d +
        # per-row metadata JSON blobs. sessions_json keeps only calibration + video
        # map + session metadata + a range into frame_groups.
        "session_data",
        # Re-ID identity subsystem groups (SLP 2.5+): /identity holds the catalog
        # (name + EAV metadata) and per-detection links; /embeddings holds the
        # columnar appearance vectors. Class/category subsystem (SLP 2.7+):
        # /categories mirrors /identity (catalog + links); category appearance
        # vectors are parallel datasets inside the same /embeddings group.
        "identity",
        "embeddings",
        "categories",
        # Frame-spanning event subsystem groups (SLP 2.6+): /event_types holds the
        # catalog (name + optional description + EAV metadata); /events holds the
        # columnar interval annotations plus the ragged CSR framewise scores.
        "event_types",
        "events",
        "provenance_json",
        "frames",
        "instances",
        "points",
        "pred_points",
        "negative_frames",
        "video_crops",
        "bboxes",
        "centroids",
        "rois",
        "roi_wkb",
        "roi_categories",
        "roi_names",
        "roi_sources",
        "masks",
        "mask_rle",
        "mask_categories",
        "mask_names",
        "mask_sources",
        "mask_score_map_index",
        "mask_score_maps",
        "label_images",
        "label_image_objects",
        "label_image_data",
        "label_image_sources",
        "label_image_obj_categories",
        "label_image_obj_names",
        "label_image_score_map_index",
        "label_image_score_maps",
    }
)


def _is_known_toplevel_name(name: str) -> bool:
    """Return whether a top-level HDF5 object is one sleap-io writes itself.

    Args:
        name: A top-level member name in a ``.slp`` HDF5 file.

    Returns:
        ``True`` for names in `_KNOWN_TOPLEVEL_HDF5_NAMES` and for per-video
        embedded groups (``video0``, ``video1``, ...); ``False`` otherwise.
    """
    if name in _KNOWN_TOPLEVEL_HDF5_NAMES:
        return True
    # Per-video embedded groups: video0, video1, ...
    return name.startswith("video") and name[len("video") :].isdigit()


def _stash_unknown_hdf5(source_path: str | None) -> bytes | None:
    """Capture top-level HDF5 members not written by sleap-io from a source file.

    Used to carry forward unrecognized datasets/groups (e.g. additions made by a
    newer sleap-io version) across a load/save cycle, since saving truncates and
    rebuilds the file from the in-memory model and would otherwise drop them.

    Args:
        source_path: Path to the file the labels were loaded from
            (``labels.provenance["filename"]``), or ``None``.

    Returns:
        An in-memory HDF5 file image (bytes) holding the unknown members, or
        ``None`` if there is no readable HDF5 source or it has no unknown members.
    """
    if not source_path:
        return None
    try:
        if not (Path(source_path).is_file() and h5py.is_hdf5(source_path)):
            return None
        buffer = io.BytesIO()
        with h5py.File(source_path, "r") as src, h5py.File(buffer, "w") as mem:
            unknown = [k for k in src.keys() if not _is_known_toplevel_name(k)]
            for name in unknown:
                src.copy(name, mem)
        return buffer.getvalue() if unknown else None
    except (OSError, KeyError):
        # Unreadable/locked/corrupt source: skip carry-over rather than fail the
        # save. Preservation is best-effort.
        return None


def _restore_unknown_hdf5(dest_path: str, image: bytes | None) -> None:
    """Re-emit stashed unknown HDF5 members into a freshly-written file.

    Members whose names were already produced by the known writers are skipped, so
    regenerated data always wins over the stashed copy.

    Args:
        dest_path: Path to the ``.slp`` file just written.
        image: The in-memory HDF5 image from `_stash_unknown_hdf5`, or ``None``.
    """
    if not image:
        return
    with h5py.File(io.BytesIO(image), "r") as mem, h5py.File(dest_path, "a") as f:
        for name in mem.keys():
            if name not in f:
                mem.copy(name, f)


def _write_labels_lazy(
    labels_path: str,
    labels: Labels,
    embed: bool | str | list[tuple[Video, int]] | None = None,
    restore_original_videos: bool = True,
    verbose: bool = True,
    prefer_metadata: bool = True,
    preserve_unknown: bool = False,
    save_embedding_vectors: bool = False,
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
        prefer_metadata: If `True` (the default), serialize each uncropped video's
            shape/grayscale/fps from its `backend_metadata` when recorded there
            instead of querying the live backend. Set to `False` to always read
            through the live backend. See `video_to_dict`.
        preserve_unknown: If `True`, top-level HDF5 datasets/groups in the source
            file not recognized by sleap-io are carried over into the saved file.
            See `write_labels`.
        save_embedding_vectors: If `False` (the default), skip the `/embeddings`
            group while still writing identity links. Set `True` to also write the
            attached re-ID appearance embeddings. See `write_labels`.

    Raises:
        ValueError: If labels is not lazy.
    """
    if not labels.is_lazy:
        raise ValueError("_write_labels_lazy requires lazy Labels")

    lazy_store = labels._lazy_store

    # Capture unknown top-level members from the source file before it is
    # truncated, to carry them across the load/save cycle (see write_labels).
    unknown_stash = (
        _stash_unknown_hdf5(labels.provenance.get("filename"))
        if preserve_unknown
        else None
    )

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

    # Collect event catalog + participants (including each event's own video)
    # before write_videos / write_tracks so event videos/tracks/identities land in
    # the written catalogs (events live on top-level lists, not the lazy store, so
    # this is identical to the eager path).
    labels._collect_events()

    # Write videos metadata (uses labels.videos which may have been modified)
    write_videos(
        labels_path,
        labels.videos,
        reference_mode=reference_mode,
        original_videos=None,
        verbose=verbose,
        prefer_metadata=prefer_metadata,
    )
    # Emit virtual crop records (after videos_json so indices line up).
    write_video_crops(labels_path, labels)

    # Write other metadata
    write_tracks(labels_path, labels.tracks)
    # Persist identities + per-detection identity links + embeddings directly from
    # the lazy store, mirroring the eager writer (write_identities /
    # write_identity_links / write_embeddings) but driven by the store dicts so no
    # LabeledFrame/Instance is materialized. Instance owners come from the store
    # (keyed by the global instance_ids written above); mask owners come from the
    # store's mask objects in the SAME order write_masks emits below, so the mask
    # owner_id (global mask-list index) stays consistent.
    lazy_masks = [
        m for ann_list in lazy_store._mask_by_frame.values() for m in ann_list
    ] + list(lazy_store._undistributed_masks)
    lazy_centroids = [
        c for ann_list in lazy_store._centroid_by_frame.values() for c in ann_list
    ] + list(lazy_store._undistributed_centroids)
    lazy_bboxes = [
        b for ann_list in lazy_store._bbox_by_frame.values() for b in ann_list
    ] + list(lazy_store._undistributed_bboxes)
    lazy_rois = [
        r for ann_list in lazy_store._roi_by_frame.values() for r in ann_list
    ] + list(lazy_store._undistributed_rois)
    write_identities(labels_path, lazy_store.identities)
    id_to_idx = {id(ident): i for i, ident in enumerate(lazy_store.identities)}
    identity_rows = _identity_link_rows_from_store(lazy_store)
    identity_rows.extend(
        _identity_link_rows_for_owner(lazy_masks, id_to_idx, OWNER_MASK)
    )
    identity_rows.extend(
        _identity_link_rows_for_owner(lazy_centroids, id_to_idx, OWNER_CENTROID)
    )
    identity_rows.extend(
        _identity_link_rows_for_owner(lazy_bboxes, id_to_idx, OWNER_BBOX)
    )
    identity_rows.extend(_identity_link_rows_for_owner(lazy_rois, id_to_idx, OWNER_ROI))
    _write_identity_link_rows(labels_path, identity_rows)
    # Class/category catalog + per-detection category links (SLP 2.7+). Self-
    # contained mirror of the identity block above, driven by the store dicts.
    write_categories(labels_path, lazy_store.categories)
    cat_id_to_idx = {id(cat): i for i, cat in enumerate(lazy_store.categories)}
    category_rows = _category_link_rows_from_store(lazy_store)
    category_rows.extend(
        _category_link_rows_for_owner(lazy_masks, cat_id_to_idx, OWNER_MASK)
    )
    category_rows.extend(
        _category_link_rows_for_owner(lazy_centroids, cat_id_to_idx, OWNER_CENTROID)
    )
    category_rows.extend(
        _category_link_rows_for_owner(lazy_bboxes, cat_id_to_idx, OWNER_BBOX)
    )
    category_rows.extend(
        _category_link_rows_for_owner(lazy_rois, cat_id_to_idx, OWNER_ROI)
    )
    _write_category_link_rows(labels_path, category_rows)
    # Identity links always persist; the appearance vectors are gated by
    # save_embedding_vectors (mirrors the eager path).
    if save_embedding_vectors:
        embedding_entries = _embedding_groups_from_store(lazy_store)
        _add_owner_embedding_groups(embedding_entries, lazy_masks, OWNER_MASK)
        _add_owner_embedding_groups(embedding_entries, lazy_centroids, OWNER_CENTROID)
        _add_owner_embedding_groups(embedding_entries, lazy_bboxes, OWNER_BBOX)
        _add_owner_embedding_groups(embedding_entries, lazy_rois, OWNER_ROI)
        _write_embedding_groups(labels_path, embedding_entries)
        # Category appearance vectors: parallel category_* datasets in /embeddings.
        category_embedding_entries = _category_embedding_groups_from_store(lazy_store)
        _add_owner_embedding_groups(
            category_embedding_entries,
            lazy_masks,
            OWNER_MASK,
            attr="category_embedding",
        )
        _add_owner_embedding_groups(
            category_embedding_entries,
            lazy_centroids,
            OWNER_CENTROID,
            attr="category_embedding",
        )
        _add_owner_embedding_groups(
            category_embedding_entries,
            lazy_bboxes,
            OWNER_BBOX,
            attr="category_embedding",
        )
        _add_owner_embedding_groups(
            category_embedding_entries,
            lazy_rois,
            OWNER_ROI,
            attr="category_embedding",
        )
        _write_embedding_groups(
            labels_path, category_embedding_entries, _CATEGORY_EMBEDDING_DATASETS
        )
    # Frame-spanning events + their catalog (SLP 2.6+). Events live on top-level
    # lists, so the writers are identical to the eager path.
    write_event_types(labels_path, labels.event_types)
    write_events(
        labels_path,
        labels.events,
        labels.videos,
        labels.event_types,
        labels.tracks,
        labels.identities,
    )
    write_suggestions(labels_path, labels.suggestions, labels.videos)
    # Sessions: prefer a verbatim passthrough of the raw sessions_json + columnar
    # /session_data captured at load, so frame groups + 3D points survive a lazy
    # re-save losslessly (frames are never materialized here). Only safe when the
    # video list is unchanged (session refs encode video indices); otherwise fall
    # back to regenerating the slim sessions_json without frame groups (+ warn if
    # frame-group data would be dropped).
    did_passthrough = False
    if _videos_unchanged(labels.videos, lazy_store.session_video_ids):
        did_passthrough = write_sessions_passthrough(labels_path, lazy_store)
    if not did_passthrough:
        if getattr(lazy_store, "session_data", None) is not None:
            warnings.warn(
                "Videos changed since lazy load; RecordingSession frame groups "
                "(including 3D points) are not preserved on this save. Call "
                "`labels.materialize()` before saving to retain them."
            )
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

    # Write annotations from lazy store (per-frame dicts + undistributed)
    # Build annotation lists with routing contexts from the lazy store keys
    def _collect_from_lazy(by_frame, undistributed, videos):
        """Collect annotations and contexts from lazy store dicts."""
        anns = []
        ctxs = []
        for (vid_idx, fidx), ann_list in by_frame.items():
            for ann in ann_list:
                anns.append(ann)
                ctxs.append((vid_idx, fidx))
        for ann in undistributed:
            anns.append(ann)
            try:
                vid_idx = videos.index(getattr(ann, "video", None))
            except ValueError:
                vid_idx = -1
            ctxs.append((vid_idx, -1))
        return anns, ctxs

    all_centroids, centroid_ctxs = _collect_from_lazy(
        lazy_store._centroid_by_frame,
        lazy_store._undistributed_centroids,
        labels.videos,
    )
    all_bboxes, bbox_ctxs = _collect_from_lazy(
        lazy_store._bbox_by_frame, lazy_store._undistributed_bboxes, labels.videos
    )
    all_masks, mask_ctxs = _collect_from_lazy(
        lazy_store._mask_by_frame, lazy_store._undistributed_masks, labels.videos
    )
    all_rois, roi_ctxs = _collect_from_lazy(
        lazy_store._roi_by_frame, lazy_store._undistributed_rois, labels.videos
    )
    all_label_images, li_ctxs = _collect_from_lazy(
        lazy_store._label_image_by_frame,
        lazy_store._undistributed_label_images,
        labels.videos,
    )
    write_rois(labels_path, all_rois, labels.videos, labels.tracks, contexts=roi_ctxs)
    write_masks(
        labels_path, all_masks, labels.videos, labels.tracks, contexts=mask_ctxs
    )
    write_bboxes(
        labels_path, all_bboxes, labels.videos, labels.tracks, contexts=bbox_ctxs
    )
    write_centroids(
        labels_path,
        all_centroids,
        labels.videos,
        labels.tracks,
        contexts=centroid_ctxs,
    )
    # Note: instance associations are not persisted in lazy mode (no all_instances).
    write_label_images(
        labels_path,
        all_label_images,
        labels.videos,
        labels.tracks,
        contexts=li_ctxs,
    )

    # Re-emit any unknown members captured from the source file.
    _restore_unknown_hdf5(labels_path, unknown_stash)


def write_labels(
    labels_path: str,
    labels: Labels,
    embed: bool | str | list[tuple[Video, int]] | None = None,
    restore_original_videos: bool = True,
    embed_inplace: bool = False,
    verbose: bool = True,
    plugin: str | None = None,
    embed_all_videos: bool = True,
    progress_callback: Callable[[int, int, str], bool] | None = None,
    prefer_metadata: bool = True,
    preserve_unknown: bool = False,
    save_embedding_vectors: bool = False,
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
        progress_callback: Optional callback function called during embedding with
            `(current, total, phase)` arguments, where ``phase`` is ``"embed"``
            (frames loaded/encoded/byte-copied) or ``"write"`` (bytes flushed to the
            HDF5 file). If it returns `False`, the operation is cancelled and
            `ExportCancelled` is raised. The ``phase`` argument is a breaking change
            from the previous ``(current, total)`` signature.
        prefer_metadata: If `True` (the default), serialize each uncropped video's
            shape/grayscale/fps from its `backend_metadata` when recorded there
            instead of querying the live backend, avoiding frame decoding when the
            metadata is already known (e.g. saving copies of labels loaded from a
            `.slp`). Set to `False` to always read through the live backend. See
            `video_to_dict`.
        preserve_unknown: If `True`, top-level HDF5 datasets/groups present in the
            source file (``labels.provenance["filename"]``) that sleap-io does not
            recognize are copied into the saved file. This preserves additions made
            by a newer sleap-io version across a load/save cycle with an older
            version (which would otherwise drop them, since saving rebuilds the file
            from the in-memory model). Default `False`. Best-effort: requires the
            source file to still exist and be readable HDF5.
        save_embedding_vectors: If `False` (the default), skip the `/embeddings`
            group -- appearance vectors are large on disk, so only the identity
            *links* (`/identity/links`, always written) are persisted by default,
            keeping the vectors in memory. Set `True` to also write the attached
            re-ID appearance embeddings. Off by default, mirroring `embed` (which
            embeds *video frames*).
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
                prefer_metadata=prefer_metadata,
                preserve_unknown=preserve_unknown,
                save_embedding_vectors=save_embedding_vectors,
            )
            return

    # Capture unknown top-level members from the source file before it is
    # truncated, so additions made by a newer sleap-io version survive a
    # load/save cycle through this (potentially older) writer.
    unknown_stash = (
        _stash_unknown_hdf5(labels.provenance.get("filename"))
        if preserve_unknown
        else None
    )

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

    # Auto-collect event catalog entries + participants (event types into
    # labels.event_types, subject/target tracks/identities into labels.tracks/
    # labels.identities, and the event's own video into labels.videos) BEFORE the
    # original-videos snapshot and embedding, so an event-only video is embedded and
    # remapped like any other and every event reference is persisted (a post-hoc
    # `labels.events.append(...)` is not dropped). Mutates labels (eager path).
    labels._collect_events()

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
        prefer_metadata=prefer_metadata,
    )
    # Emit virtual crop records (after videos_json so indices line up). Omitted
    # entirely when no video is cropped (uncropped files stay byte-identical).
    write_video_crops(labels_path, labels)
    write_tracks(labels_path, labels.tracks)
    # Auto-collect any detection identity not yet registered in the catalog
    # (object-identity deduped) so post-hoc `inst.identity` / `mask.identity`
    # assignments are not silently dropped on save. Mutates labels.identities
    # (eager path only).
    labels._collect_identities()
    write_identities(labels_path, labels.identities)
    write_identity_links(labels_path, labels)
    # Auto-collect any detection category not yet registered in the catalog
    # (object-identity deduped), mirroring _collect_identities, then persist the
    # /categories catalog + per-detection links (SLP 2.7+). Additive: omitted
    # entirely when there are no categories (files stay byte-identical).
    labels._collect_categories()
    write_categories(labels_path, labels.categories)
    write_category_links(labels_path, labels)
    # Frame-spanning events + their catalog (SLP 2.6+). Additive groups; omitted
    # entirely when there are no events / event types (files stay byte-identical).
    write_event_types(labels_path, labels.event_types)
    write_events(
        labels_path,
        labels.events,
        labels.videos,
        labels.event_types,
        labels.tracks,
        labels.identities,
    )
    # Identity links always persist; the (large) appearance vectors are gated by
    # save_embedding_vectors so a producer can keep them in memory but off disk.
    if save_embedding_vectors:
        write_embeddings(labels_path, labels)
        # Category appearance vectors: parallel category_* datasets in /embeddings.
        write_category_embeddings(labels_path, labels)
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

    # Collect all instances and build annotation lists with routing contexts
    all_instances: list[Instance | PredictedInstance] = []
    all_centroids: list = []
    centroid_contexts: list[tuple[int, int]] = []
    all_bboxes: list = []
    bbox_contexts: list[tuple[int, int]] = []
    all_masks: list = []
    mask_contexts: list[tuple[int, int]] = []
    all_label_images: list = []
    li_contexts: list[tuple[int, int]] = []
    all_rois: list = []
    roi_contexts: list[tuple[int, int]] = []

    for lf in labels.labeled_frames:
        all_instances.extend(lf.instances)
        vid_idx = labels.videos.index(lf.video) if lf.video in labels.videos else -1
        ctx = (vid_idx, lf.frame_idx)
        for c in lf.centroids:
            all_centroids.append(c)
            centroid_contexts.append(ctx)
        for b in lf.bboxes:
            all_bboxes.append(b)
            bbox_contexts.append(ctx)
        for m in lf.masks:
            all_masks.append(m)
            mask_contexts.append(ctx)
        for li in lf.label_images:
            all_label_images.append(li)
            li_contexts.append(ctx)
        for r in lf.rois:
            all_rois.append(r)
            roi_contexts.append(ctx)

    # Add static ROIs (not tied to any frame)
    for r in labels.static_rois:
        all_rois.append(r)
        roi_contexts.append(
            (labels.videos.index(r.video) if r.video in labels.videos else -1, -1)
        )

    write_rois(
        labels_path,
        all_rois,
        labels.videos,
        labels.tracks,
        all_instances,
        contexts=roi_contexts,
    )
    write_masks(
        labels_path,
        all_masks,
        labels.videos,
        labels.tracks,
        all_instances,
        contexts=mask_contexts,
    )
    write_bboxes(
        labels_path,
        all_bboxes,
        labels.videos,
        labels.tracks,
        all_instances,
        contexts=bbox_contexts,
    )
    write_centroids(
        labels_path,
        all_centroids,
        labels.videos,
        labels.tracks,
        all_instances,
        contexts=centroid_contexts,
    )
    write_label_images(
        labels_path,
        all_label_images,
        labels.videos,
        labels.tracks,
        all_instances,
        contexts=li_contexts,
    )

    # Re-emit any unknown members captured from the source file.
    _restore_unknown_hdf5(labels_path, unknown_stash)
