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
    - 3.0: Breaking change: tables are stored one dataset per column. `/points`
            and `/pred_points` are groups of 1-D columns (`x`, `y`, `visible`,
            `verified` (renamed from `complete`), `labeled`, plus `score` for
            predictions) instead of compound datasets. A missing `labeled` column
            reads as True, and unknown columns are carried through re-saves. The
            optional `required_features` attribute on `/metadata` lists features a
            reader must understand. Older files are cleaned up on read: hidden
            points lose their (stale) coordinates and visible points without
            coordinates become hidden. `save_slp(..., legacy_format=True)` still
            writes the 2.x layout. `/instances` and `/frames` are groups of
            columns too, without the id columns (the row number is the id):
            `/frames/video` is the index into the videos, `/frames/is_negative`
            replaces `/negative_frames`, and `metadata/` subgroups hold
            per-record metadata. Every remaining table is a group of columns as
            well: `/rois` (with its `wkb` buffer), `/masks` (with its `rle` buffer
            and `score_maps/`), `/label_images` (with its pixel `data`, `objects/`
            and `score_maps/`), `/identity/links`, `/categories/links` and the
            `/session_data` tables, replacing the separate `roi_*`, `mask_*` and
            `label_image_*` datasets. Byte data stays in flat `uint8` buffers
            indexed by start/end columns. Converted tables use `f8` floats, `i8`
            integers and `bool` flags, and range checks raise for 3.0 tables and
            warn for older ones. A 3.0 file holds no compound dataset.
"""

from __future__ import annotations

import io
import sys
import warnings
import zlib
from contextlib import nullcontext
from enum import Enum, IntEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, NamedTuple

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
from sleap_io.model.record_metadata import metadata_column_kinds
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


# Files whose `/metadata` `format_id` is at or above this value use a layout this
# version of sleap-io can't read, so readers refuse them up front instead of failing
# partway through with an h5py error.
SLP_FORMAT_ID_LIMIT = 4.0

# The `format_id` written by default. Files below it use the compound (2.x) layout.
SLP_FORMAT_ID = 3.0

# Features listed in a file's `/metadata` `required_features` attribute that this
# version of sleap-io understands. A file that requires anything else is refused,
# since reading it without understanding the feature could silently misread it.
SLP_SUPPORTED_FEATURES: frozenset[str] = frozenset()


def _check_format_id(labels_path: str, f: h5py.File) -> None:
    """Raise if an SLP file is newer than this sleap-io can read.

    Files without a `/metadata` group or `format_id` attribute pass, as before.

    Args:
        labels_path: Path or URL to the SLP file (used in the error message).
        f: An already-open `h5py.File` handle for the SLP file.

    Raises:
        ValueError: If the file's `format_id` is at or above `SLP_FORMAT_ID_LIMIT`,
            or its `required_features` lists a feature this sleap-io doesn't
            support.
    """
    if "metadata" not in f:
        return
    attrs = f["metadata"].attrs
    if "format_id" in attrs:
        format_id = float(attrs["format_id"])
        if format_id >= SLP_FORMAT_ID_LIMIT:
            raise ValueError(
                f"{labels_path} uses SLP format {format_id:g}, which is newer than "
                "this version of sleap-io supports (max "
                f"{int(SLP_FORMAT_ID_LIMIT) - 1}.x); upgrade sleap-io to read it."
            )
    required = [
        r.decode() if isinstance(r, bytes) else str(r)
        for r in np.atleast_1d(attrs.get("required_features", []))
    ]
    unsupported = sorted(set(required) - SLP_SUPPORTED_FEATURES)
    if unsupported:
        raise ValueError(
            f"{labels_path} requires SLP features this version of sleap-io doesn't "
            f"support ({', '.join(unsupported)}); upgrade sleap-io to read it."
        )


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


def _identity_index(items: Iterable) -> dict[int, int]:
    """Map each item's ``id()`` to the position of its first occurrence.

    A drop-in for repeated ``items.index(x)`` lookups on lists of ``eq=False``
    objects (videos, tracks, instances, ...), which are O(len(items)) each and made
    writing large labels O(N^2). Look up with ``index.get(id(x), default)``.

    Args:
        items: The sequence to index.

    Returns:
        A dict mapping ``id(item)`` to its first index in `items`.
    """
    index: dict[int, int] = {}
    for i, item in enumerate(items):
        index.setdefault(id(item), i)
    return index


def _create_json_rows_dataset(f: h5py.File, name: str, rows: list | np.ndarray):
    """Create a 1D dataset of per-row JSON strings (e.g. ``videos_json``).

    Rows are stored as variable-length UTF-8 strings. A fixed-width string array
    pads every row to the longest one, so a single long row (e.g. an
    `ImageVideo` listing thousands of filenames) multiplies the dataset size by
    the number of rows. Readers decode both layouts identically, since h5py
    returns ``bytes`` for each row either way.

    Args:
        f: An open, writable `h5py.File`.
        name: The dataset name to create.
        rows: The JSON rows as ``str`` or ``bytes`` (including arrays read back
            from an existing fixed-width or variable-length dataset).
    """
    f.create_dataset(name, data=rows, dtype=h5py.string_dtype(), maxshape=(None,))


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

    # A grayscale video whose stored frames are multi-channel must go through a
    # decode. The channel count written to the dataset attrs comes from `video.shape`,
    # which `HDF5Video.img_shape` reports as 1 when grayscale is set, but the raw bytes
    # still carry the source's own channels. Copying them verbatim would file RGB
    # pixels under a `channels=1` attr, leaving the reloaded video stuck in grayscale:
    # its shape would say 1 channel while its frames decode to 3. Frames that are
    # already single-channel are unaffected and still copy verbatim, so a genuinely
    # grayscale package keeps its byte-exact (and, for JPEG, lossless) fast path.
    if video.backend.grayscale and video.backend.stored_channels != 1:
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
        video_jsons.append(json.dumps(video_json, separators=(",", ":")))

    with h5py.File(labels_path, "a") as f:
        if "videos_json" not in f:
            _create_json_rows_dataset(f, "videos_json", video_jsons)

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
        json.dumps([SPAWNED_ON, track.name], separators=(",", ":")) for track in tracks
    ]
    with h5py.File(labels_path, "a") as f:
        _create_json_rows_dataset(f, "tracks_json", tracks_json)


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
    labels_path: str,
    *,
    _hdf5_file: h5py.File | None = None,
    _carry: dict | None = None,
) -> dict[int, dict[int, tuple[int, float | None]]]:
    """Read per-detection identity links from a SLEAP labels file.

    Links are stored in the optional ``/identity/links`` table (SLP format 2.5+;
    a group of columns since 3.0, a compound dataset before), with
    ``(owner_type, owner_id, identity_idx, identity_score)`` rows joining a
    detection -- identified by ``owner_type`` (one of the ``OWNER_*`` codes) and
    ``owner_id`` -- to an index into the file's identity catalog. ``owner_id`` is
    the per-owner-type positional id (e.g. the global ``instance_id`` assigned by
    `write_lfs` for instance owners), mirroring the ``/embeddings`` join. The table
    is absent in older files, in which case an empty mapping is returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.
        _carry: If given, the table's raw records are added to it for carrying
            unknown columns through a later save (see `_column_carry`).

    Returns:
        A dict mapping each ``owner_type`` to a ``{owner_id: (identity_idx,
        identity_score)}`` mapping. ``identity_score`` is ``None`` when it was not
        recorded (stored as NaN). Empty for older files.

    Raises:
        ValueError: If an SLP 3.0 link points outside the identity catalog. Older
            tables warn instead.
    """
    return _read_links(
        labels_path, "identity/links", _hdf5_file=_hdf5_file, _carry=_carry
    )


def _read_links(
    labels_path: str,
    table: str,
    *,
    _hdf5_file: h5py.File | None = None,
    _carry: dict | None = None,
) -> dict[int, dict[int, tuple[int, float | None]]]:
    """Read `/identity/links` or `/categories/links` (see `read_identity_links`).

    Args:
        labels_path: A string path to the SLEAP labels file.
        table: `"identity/links"` or `"categories/links"`.
        _hdf5_file: An already-open `h5py.File` handle to read from.
        _carry: If given, the table's raw records are added to it, keyed by their
            `(owner_type, owner_id, catalog index)` values.

    Returns:
        A dict mapping each ``owner_type`` to a ``{owner_id: (catalog_idx,
        score)}`` mapping, with unrecorded (NaN) scores as ``None``.

    Raises:
        ValueError: If an SLP 3.0 link points outside the catalog (the `name`
            dataset of the same group, if there is one). Older tables warn
            instead.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        # The 2.x table may also be a flat 2-D array with a ``field_names``
        # attribute, as written by sleap-io.js (h5wasm can't create compound
        # datasets).
        links = _read_table(f, table)
        if links is None:
            return {}
        catalog = f[table.partition("/")[0]]
        n_catalog = len(catalog["name"]) if "name" in catalog else None
    rows = links.rows
    _, _, index, score_name = _TABLE_SPECS[table].raw.names
    if n_catalog is not None and _index_out_of_range(
        rows[index], n_catalog, optional=False
    ):
        _report_table_problems(
            labels_path,
            [f"/{table} {index} is not one of the {n_catalog} catalog entries"],
            strict=not links.is_legacy,
        )
    result: dict[int, dict[int, tuple[int, float | None]]] = {}
    for owner_type, owner_id, idx, score in zip(
        rows["owner_type"].tolist(),
        rows["owner_id"].tolist(),
        rows[index].tolist(),
        rows[score_name].tolist(),
    ):
        result.setdefault(owner_type, {})[owner_id] = (
            idx,
            None if np.isnan(score) else score,
        )
    if _carry is not None:
        _carry[table] = (rows, _TABLE_SPECS[table].raw, lambda: _link_keys(rows))
    return result


def _link_keys(rows: np.ndarray) -> list[tuple[int, int, int]]:
    """Return `(owner_type, owner_id, catalog index)` of each link row.

    Link rows aren't owned by one object each, so unknown link columns are carried
    by these values instead (see `_ColumnCarry`).
    """
    return list(zip(*(rows[name].tolist() for name in rows.dtype.names[:3])))


def _write_link_rows(
    labels_path: str,
    table: str,
    rows: list[tuple[int, int, int, float]],
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
) -> None:
    """Write `/identity/links` or `/categories/links` rows.

    Args:
        labels_path: A string path to the SLEAP labels file.
        table: `"identity/links"` or `"categories/links"`.
        rows: ``(owner_type, owner_id, catalog_idx, score)`` tuples, with NaN for
            an unrecorded score. An empty list is a no-op (nothing is written).
        legacy_format: If `True`, write the 2.x compound dataset. Otherwise (SLP
            3.0) a group of columns.
        compression: `None`, `"gzip"` or `"lzf"` for the columns.
        carry: Unknown columns kept from loading, appended if the rows are the
            ones loaded (see `_link_keys`).
    """
    if not rows:
        return
    arr = np.array(rows, dtype=_TABLE_SPECS[table].raw)
    if not legacy_format:
        arr = _with_carried_columns(arr, carry, table, _link_keys(arr))
    with h5py.File(labels_path, "a") as f:
        _write_table(
            f.require_group(table.partition("/")[0]),
            table,
            arr,
            legacy_format=legacy_format,
            compression=compression,
            resizable=True,
        )


def _write_identity_link_rows(
    labels_path: str,
    rows: list[tuple[int, int, int, float]],
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
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
        legacy_format: If `True`, write the 2.x compound dataset.
        compression: `None`, `"gzip"` or `"lzf"` for the columns.
        carry: Unknown columns kept from loading (see `_write_link_rows`).
    """
    _write_link_rows(
        labels_path,
        "identity/links",
        rows,
        legacy_format=legacy_format,
        compression=compression,
        carry=carry,
    )


def write_identity_links(
    labels_path: str,
    labels: Labels,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
) -> None:
    """Write per-detection identity links to a SLEAP labels file.

    Creates the ``/identity/links`` table (SLP format 2.5+) of
    ``(owner_type, owner_id, identity_idx, identity_score)`` rows for every
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
        legacy_format: If `True`, write the 2.x compound dataset. Otherwise (SLP
            3.0) a group of columns.
        compression: `None`, `"gzip"` or `"lzf"` for the columns.
        carry: Unknown columns kept from loading (see `_write_link_rows`).
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
    _write_identity_link_rows(
        labels_path,
        rows,
        legacy_format=legacy_format,
        compression=compression,
        carry=carry,
    )


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
    labels_path: str,
    *,
    _hdf5_file: h5py.File | None = None,
    _carry: dict | None = None,
) -> dict[int, dict[int, tuple[int, float | None]]]:
    """Read per-detection category links from a SLEAP labels file.

    Links are stored in the optional ``/categories/links`` table (SLP format 2.7+;
    a group of columns since 3.0, a compound dataset before), with ``(owner_type,
    owner_id, category_idx, category_score)`` rows joining a detection --
    identified by ``owner_type`` (one of the ``OWNER_*`` codes) and ``owner_id`` --
    to an index into the file's category catalog. A fully self-contained mirror of
    `read_identity_links`. ``owner_id`` is the per-owner-type positional id (e.g.
    the global ``instance_id`` assigned by `write_lfs` for instance owners),
    mirroring the ``/embeddings`` join. The table is absent in older files, in
    which case an empty mapping is returned.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.
        _carry: If given, the table's raw records are added to it for carrying
            unknown columns through a later save (see `_column_carry`).

    Returns:
        A dict mapping each ``owner_type`` to a ``{owner_id: (category_idx,
        category_score)}`` mapping. ``category_score`` is ``None`` when it was not
        recorded (stored as NaN). Empty for older files.

    Raises:
        ValueError: If an SLP 3.0 link points outside the category catalog. Older
            tables warn instead.
    """
    return _read_links(
        labels_path, "categories/links", _hdf5_file=_hdf5_file, _carry=_carry
    )


def _write_category_link_rows(
    labels_path: str,
    rows: list[tuple[int, int, int, float]],
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
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
        legacy_format: If `True`, write the 2.x compound dataset.
        compression: `None`, `"gzip"` or `"lzf"` for the columns.
        carry: Unknown columns kept from loading (see `_write_link_rows`).
    """
    _write_link_rows(
        labels_path,
        "categories/links",
        rows,
        legacy_format=legacy_format,
        compression=compression,
        carry=carry,
    )


def write_category_links(
    labels_path: str,
    labels: Labels,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
) -> None:
    """Write per-detection category links to a SLEAP labels file.

    Creates the ``/categories/links`` table (SLP format 2.7+) of ``(owner_type,
    owner_id, category_idx, category_score)`` rows for every
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
        legacy_format: If `True`, write the 2.x compound dataset. Otherwise (SLP
            3.0) a group of columns.
        compression: `None`, `"gzip"` or `"lzf"` for the columns.
        carry: Unknown columns kept from loading (see `_write_link_rows`).
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
    _write_category_link_rows(
        labels_path,
        rows,
        legacy_format=legacy_format,
        compression=compression,
        carry=carry,
    )


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
    video_ids = _identity_index(videos)
    for suggestion in suggestions:
        # Get group from metadata if available, otherwise use default
        group = suggestion.metadata.get("group", 0) if suggestion.metadata else 0

        video_idx = video_ids.get(id(suggestion.video))
        if video_idx is None:
            video_idx = videos.index(suggestion.video)  # Raises for unknown videos.
        suggestion_dict = {
            "video": str(video_idx),
            "frame_idx": suggestion.frame_idx,
            "group": group,
        }
        suggestions_json.append(json.dumps(suggestion_dict, separators=(",", ":")))

    with h5py.File(labels_path, "a") as f:
        _create_json_rows_dataset(f, "suggestions_json", suggestions_json)


def write_negative_frames(labels_path: str, labels: Labels):
    """Write negative frame markers to a SLEAP labels file in the 2.x layout.

    SLP 3.0 files store negative frames in the `/frames` `is_negative` column
    instead, so `write_labels` only writes this dataset with `legacy_format=True`.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object containing negative frames to write.

    Notes:
        Uses the video ids of the 2.x `/frames` table (each embedded video's sparse
        `videoN` number, see `_video_id_map`). The /negative_frames dataset stores
        (video_id, frame_idx) tuples identifying which frames are explicitly marked
        as negative (pure background, no instances).
    """
    with h5py.File(labels_path, "a") as f:
        video_id_map = _written_video_id_map(f, labels.videos)
    negative_data = []
    video_ids = _identity_index(labels.videos)
    for lf in labels.labeled_frames:
        if lf.is_negative:
            video_idx = video_ids.get(id(lf.video))
            if video_idx is None:
                video_idx = labels.videos.index(lf.video)  # Raises if unknown.
            negative_data.append((int(video_id_map[video_idx]), lf.frame_idx))

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


def write_metadata(labels_path: str, labels: Labels, *, legacy_format: bool = False):
    """Write metadata to a SLEAP labels file.

    This function will write the skeletons and provenance for the labels.

    Provenance is written to a dedicated top-level ``provenance_json`` dataset (not
    subject to HDF5's 64 KB attribute limit) and, when small enough, also mirrored
    into the ``metadata/json`` attribute for backward compatibility with older
    readers (see `read_provenance` and `_encode_metadata_attr`).

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.
        legacy_format: If `True`, write the 2.x `format_id` for the features used
            (for files written with `legacy_format=True`). Otherwise write
            `SLP_FORMAT_ID` (3.0).

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

        if not legacy_format:
            format_id = max(format_id, SLP_FORMAT_ID)

        grp = f.require_group("metadata")
        grp.attrs["format_id"] = format_id
        # Store provenance in a dedicated dataset (no 64 KB attribute limit) so an
        # unbounded merge_history can't break saving. A copy is also kept in the
        # metadata/json attribute (when it fits) by `_encode_metadata_attr` for
        # older readers.
        _write_provenance_dataset(f, provenance)
        grp.attrs["json"] = _encode_metadata_attr(md)


# In-memory layout of the raw point records that every reader normalises to,
# whatever the on-disk layout (2.x compound, h5wasm flat 2-D, or 3.0 columns).
# Unknown columns found in a 3.0 table follow these fields.
RAW_POINT_DTYPE = np.dtype(
    [
        ("x", "f8"),
        ("y", "f8"),
        ("visible", "?"),
        ("verified", "?"),
        ("labeled", "?"),
    ]
)
RAW_PRED_POINT_DTYPE = np.dtype(RAW_POINT_DTYPE.descr + [("score", "f8")])

# The 2.x compound layout, written by `save_slp(..., legacy_format=True)`.
LEGACY_POINT_DTYPE = np.dtype(
    [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")]
)
LEGACY_PRED_POINT_DTYPE = np.dtype(LEGACY_POINT_DTYPE.descr + [("score", "f8")])

# Defaults for optional point columns missing from a table.
_POINT_COLUMN_DEFAULTS = {
    "visible": False,
    "verified": False,
    "labeled": True,
    "score": np.nan,
}


def _read_column(ds: h5py.Dataset) -> np.ndarray:
    """Read one column, with variable-length strings as a `str` object array."""
    if h5py.check_string_dtype(ds.dtype) is not None:
        return ds.asstr()[()]
    return ds[()]


def _read_columns(
    node: h5py.Group | h5py.Dataset, *, exclude: Iterable[str] = ()
) -> tuple[dict, bool]:
    """Read every column of a table, whatever its on-disk layout.

    Args:
        node: A columnar table (a group with one 1-D dataset per column, SLP 3.0+),
            a compound dataset, or an h5wasm-style flat 2-D dataset with a
            `field_names` attribute (SLP 2.x).
        exclude: Names of datasets in a columnar table that aren't columns (for
            example a byte buffer the table indexes into). Subgroups are never
            columns.

    Returns:
        A tuple `(columns, is_legacy)` where `columns` maps column names to arrays
        (string columns as `str` object arrays) and `is_legacy` is `True` for the
        2.x (non-columnar) layouts.

    Notes:
        All of a columnar table's datasets are read in one pass, so a remote file
        is not revisited per column.
    """
    if isinstance(node, h5py.Group):
        exclude = set(exclude)
        columns = {
            name: _read_column(member)
            for name, member in node.items()
            if isinstance(member, h5py.Dataset) and name not in exclude
        }
        return columns, False
    data = _read_dataset_from_open_file(node.file, node.name)
    return {name: data[name] for name in (data.dtype.names or ())}, True


def _normalize_points(
    columns: dict[str, np.ndarray],
    *,
    is_predicted: bool,
    is_legacy: bool,
    format_id: float | None,
) -> np.ndarray:
    """Normalise a point table's columns into the raw in-memory record layout.

    Args:
        columns: The table's columns by name (see `_read_columns`).
        is_predicted: If `True`, build `RAW_PRED_POINT_DTYPE` records (with
            `score`); otherwise `RAW_POINT_DTYPE` records.
        is_legacy: If `True`, the columns come from a 2.x file and are cleaned up:
            `complete` is read as `verified`, every point is labeled, hidden points
            lose their coordinates (SLEAP keeps stale coordinates on hidden nodes),
            and visible points without coordinates become hidden.
        format_id: The file's `format_id`, or `None` if it has none. Files older
            than 1.1 used a coordinate system with the top-left of the first pixel
            at (0, 0), which is shifted here to the current pixel-center convention.

    Returns:
        A structured array with the fields of `RAW_POINT_DTYPE` (or
        `RAW_PRED_POINT_DTYPE`), followed by any unknown columns. Stored `labeled`
        values are combined with `visible`, since a visible point is always labeled.

    Raises:
        ValueError: If the `x` or `y` column is missing, or a column doesn't have
            one row per point.
    """
    base = RAW_PRED_POINT_DTYPE if is_predicted else RAW_POINT_DTYPE
    columns = dict(columns)
    if is_legacy and "complete" in columns:
        columns.setdefault("verified", columns.pop("complete"))
    if "x" not in columns or "y" not in columns:
        raise ValueError("Point table is missing its x or y column.")
    n = len(columns["x"])
    for name, col in columns.items():
        if len(col) != n:
            raise ValueError(
                f"Point table column {name!r} has {len(col)} rows, expected {n}."
            )
    extra = [
        (name, col.dtype, col.shape[1:])
        for name, col in columns.items()
        if name not in base.names
    ]
    points = np.empty(n, dtype=base.descr + extra)
    for name in base.names:
        if name in columns:
            points[name] = columns[name]
        else:
            points[name] = _POINT_COLUMN_DEFAULTS[name]
    for name, _, _ in extra:
        points[name] = columns[name]

    if is_legacy:
        points["labeled"] = True
        points["visible"] &= ~(np.isnan(points["x"]) | np.isnan(points["y"]))
        hidden = ~points["visible"]
        points["x"][hidden] = np.nan
        points["y"][hidden] = np.nan
        if format_id is not None and format_id < 1.1:
            points["x"] -= 0.5
            points["y"] -= 0.5
    points["labeled"] |= points["visible"]
    return points


def _read_point_table(
    labels_path: str,
    name: str,
    *,
    is_predicted: bool,
    _hdf5_file: h5py.File | None = None,
) -> np.ndarray:
    """Read and normalise one point table from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        name: The table name, `"points"` or `"pred_points"`.
        is_predicted: Whether the table holds predicted points (with `score`).
        _hdf5_file: An already-open `h5py.File` handle to read from (see
            `read_points`).

    Returns:
        The normalised records (see `_normalize_points`).
    """
    with (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    ) as f:
        format_id = None
        if "metadata" in f and "format_id" in f["metadata"].attrs:
            format_id = float(f["metadata"].attrs["format_id"])
        columns, is_legacy = _read_columns(f[name])
        return _normalize_points(
            columns,
            is_predicted=is_predicted,
            is_legacy=is_legacy,
            format_id=format_id,
        )


def read_points(labels_path: str, *, _hdf5_file: h5py.File | None = None) -> np.ndarray:
    """Read the user points table from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A structured array of point records with the fields of `RAW_POINT_DTYPE`
        (plus any unknown columns), whatever the file's layout. See
        `_normalize_points` for the cleanup applied to files older than 3.0.
    """
    return _read_point_table(
        labels_path, "points", is_predicted=False, _hdf5_file=_hdf5_file
    )


def read_pred_points(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> np.ndarray:
    """Read the predicted points table from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A structured array of predicted point records with the fields of
        `RAW_PRED_POINT_DTYPE` (plus any unknown columns), whatever the file's
        layout.
    """
    return _read_point_table(
        labels_path, "pred_points", is_predicted=True, _hdf5_file=_hdf5_file
    )


def _points_from_hdf5_data(
    pts_data: np.ndarray,
    skeleton: Skeleton,
    is_predicted: bool = False,
) -> "PointsArray | PredictedPointsArray":
    """Build a points array from normalised point records.

    Args:
        pts_data: Point records in the `RAW_POINT_DTYPE` (or
            `RAW_PRED_POINT_DTYPE`) layout, as returned by `read_points`.
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
    points["verified"] = pts_data["verified"]
    points["labeled"] = pts_data["labeled"]
    points["name"] = skeleton.node_names

    return points


def _write_columns(
    f: h5py.File | h5py.Group,
    name: str,
    rows: np.ndarray,
    *,
    compression: str | None = None,
    resizable: bool = False,
) -> h5py.Group:
    """Write a table as a group with one dataset per column.

    Args:
        f: The open file (or group) to write into.
        name: Name of the group to create (or to add columns to, if it exists).
        rows: A structured array of records; each field becomes a dataset of the
            same name and dtype (object fields hold `str` and become
            variable-length UTF-8 strings).
        compression: `None` (the default) writes contiguous, uncompressed columns,
            which read fastest. `"gzip"` or `"lzf"` writes chunked columns with the
            shuffle filter.
        resizable: If `True`, the columns can grow (`maxshape=(None,)`, so they
            are chunked).

    Returns:
        The created group.
    """
    grp = f.require_group(name)
    for field in rows.dtype.names or ():
        data = rows[field]
        kwargs = _compression_kwargs(compression, data.shape)
        if resizable:
            kwargs["maxshape"] = (None, *data.shape[1:])
        if data.dtype == object:
            kwargs["dtype"] = h5py.string_dtype()
        grp.create_dataset(field, data=data, **kwargs)
    return grp


# Rows per chunk for compressed columns (see `_write_columns`).
_COLUMN_CHUNK_ROWS = 65536

_COMPRESSION_MODES = (None, "gzip", "lzf")


# In-memory layout of the raw instance records that every reader normalises to,
# whatever the on-disk layout (see `_normalize_instances`). Unknown columns found in
# a 3.0 table follow these fields. `instance_id` is the row number: SLP 3.0 doesn't
# store it, and older files are remapped so that it is.
RAW_INSTANCE_DTYPE = np.dtype(
    [
        ("instance_id", "i8"),
        ("instance_type", "u1"),
        ("frame_id", "i8"),
        ("skeleton", "i8"),
        ("track", "i8"),
        ("from_predicted", "i8"),
        ("score", "f8"),
        ("point_id_start", "i8"),
        ("point_id_end", "i8"),
        ("tracking_score", "f8"),
    ]
)

# In-memory layout of the raw frame records (see `_normalize_frames`). `frame_id` is
# the row number and `video` the index into the file's videos.
RAW_FRAME_DTYPE = np.dtype(
    [
        ("frame_id", "i8"),
        ("video", "i8"),
        ("frame_idx", "i8"),
        ("instance_id_start", "i8"),
        ("instance_id_end", "i8"),
        ("is_negative", "?"),
    ]
)

# The 2.x compound layouts, written by `save_slp(..., legacy_format=True)`.
LEGACY_INSTANCE_DTYPE = np.dtype(
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
LEGACY_FRAME_DTYPE = np.dtype(
    [
        ("frame_id", "u8"),
        ("video", "u4"),
        ("frame_idx", "u8"),
        ("instance_id_start", "u8"),
        ("instance_id_end", "u8"),
    ]
)

# The id column of each table: dropped in SLP 3.0, where the row number is the id.
_ROW_ID_COLUMNS = {"instances": "instance_id", "frames": "frame_id"}

# Defaults for optional columns missing from a table. Every other column of the raw
# layout is required.
_TABLE_COLUMN_DEFAULTS = {
    "instances": {
        "track": -1,
        "from_predicted": -1,
        "score": np.nan,
        "tracking_score": np.nan,
    },
    "frames": {"is_negative": False},
}


def _table_rows(columns: dict[str, np.ndarray], table: str) -> np.ndarray:
    """Build the raw records of `/instances` or `/frames` from the table's columns.

    Args:
        columns: The table's columns by name (see `_read_columns`).
        table: `"instances"` or `"frames"`.

    Returns:
        A structured array with the fields of `RAW_INSTANCE_DTYPE` (or
        `RAW_FRAME_DTYPE`), followed by any unknown columns. The id field is the row
        number, whatever the table stores; missing optional columns take their
        defaults.

    Raises:
        ValueError: If a required column is missing or the columns don't all have
            the same length.
    """
    base = RAW_INSTANCE_DTYPE if table == "instances" else RAW_FRAME_DTYPE
    return _rows_from_columns(
        columns,
        base,
        _TABLE_COLUMN_DEFAULTS[table],
        table,
        row_id=_ROW_ID_COLUMNS[table],
    )


def _rows_from_columns(
    columns: dict[str, np.ndarray],
    base: np.dtype,
    defaults: dict,
    table: str,
    *,
    row_id: str | None = None,
) -> np.ndarray:
    """Build a table's raw records from its columns.

    Args:
        columns: The table's columns by name (see `_read_columns`).
        base: The raw record layout. Its fields come first, cast to their types.
        defaults: Defaults for optional fields missing from `columns`. Every other
            field of `base` is required.
        table: The table name, for messages.
        row_id: A field of `base` that holds the row number, whatever the table
            stores.

    Returns:
        A structured array with the fields of `base`, followed by any unknown
        columns.

    Raises:
        ValueError: If a required column is missing or the columns don't all have
            the same length.
    """
    missing = [
        name
        for name in base.names
        if name != row_id and name not in columns and name not in defaults
    ]
    if missing:
        raise ValueError(f"/{table} is missing column(s) {', '.join(missing)}.")
    lengths = {name: len(col) for name, col in columns.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name}: {n}" for name, n in lengths.items())
        raise ValueError(f"/{table} columns have different lengths ({detail}).")
    n = next(iter(lengths.values()))
    extra = [
        (name, col.dtype, col.shape[1:])
        for name, col in columns.items()
        if name not in base.names
    ]
    rows = np.empty(n, dtype=base.descr + extra)
    for name in base.names:
        if name == row_id:
            rows[name] = np.arange(n)
        elif name in columns:
            rows[name] = columns[name]
        else:
            rows[name] = defaults[name]
    for name, _, _ in extra:
        rows[name] = columns[name]
    return rows


def _row_id_map(columns: dict[str, np.ndarray], table: str) -> dict[int, int] | None:
    """Map the ids stored in a 2.x table to row numbers.

    Args:
        columns: The table's columns by name.
        table: `"instances"` or `"frames"`.

    Returns:
        A dictionary mapping each stored id to its row, or `None` if the table stores
        no ids or every id equals its row (as in every file sleap-io and SLEAP
        write).
    """
    stored = columns.get(_ROW_ID_COLUMNS[table])
    if stored is None:
        return None
    stored = np.asarray(stored).astype(np.int64)
    if np.array_equal(stored, np.arange(len(stored))):
        return None
    return {int(sid): row for row, sid in enumerate(stored.tolist())}


def _remap_ids(values: np.ndarray, id_map: dict[int, int]) -> np.ndarray:
    """Map stored ids to rows, with ids that match no row becoming -1."""
    return np.array(
        [id_map.get(v, -1) if v >= 0 else -1 for v in values.tolist()], dtype="i8"
    )


def _normalize_instances(
    columns: dict[str, np.ndarray], *, is_legacy: bool
) -> tuple[np.ndarray, dict[int, int] | None]:
    """Normalise an instance table's columns into the raw in-memory record layout.

    Args:
        columns: The table's columns by name (see `_read_columns`).
        is_legacy: If `True`, the columns come from a 2.x table: a missing
            `tracking_score` (files older than 1.2) reads as 0.0, and stored
            `instance_id` values that aren't row numbers are mapped to rows,
            remapping `from_predicted` through them.

    Returns:
        A tuple `(rows, id_map)`: the `RAW_INSTANCE_DTYPE` records followed by any
        unknown columns, and the stored id -> row mapping of a 2.x table whose ids
        aren't row numbers (`None` otherwise). Other links to instances in the file
        are remapped through `id_map` (see `_remap_instance_owners`).

    Raises:
        ValueError: If a required column is missing or the columns don't all have
            the same length.
    """
    rows = _table_rows(columns, "instances")
    id_map = None
    if is_legacy:
        if "tracking_score" not in columns:
            rows["tracking_score"] = 0.0
        id_map = _row_id_map(columns, "instances")
        if id_map is not None:
            rows["from_predicted"] = _remap_ids(rows["from_predicted"], id_map)
    return rows, id_map


def _dataset_video_id(dataset: str | None) -> int | None:
    """Return the `N` of an embedded video dataset path like `"videoN/video"`.

    Args:
        dataset: The dataset path of a video's backend, or `None`.

    Returns:
        `N`, or `None` if the path isn't a `videoN` dataset.
    """
    group = (dataset or "").split("/")[0]
    if group.startswith("video") and group[5:].isdigit():
        return int(group[5:])
    return None


def _embedded_video_id(video: Video, *, use_metadata: bool = False) -> int | None:
    """Return the `videoN` number of an embedded video's dataset, if it has one.

    Args:
        video: A video, embedded (with a backend `dataset` like `"video15/video"`)
            or not.
        use_metadata: If `True`, fall back to the `dataset` recorded in
            `video.backend_metadata` when the video has no open backend (for videos
            just read from a file with `open_videos=False`).

    Returns:
        The embedded number (15 above), or `None` for other videos.
    """
    if video.backend is not None:
        return _dataset_video_id(getattr(video.backend, "dataset", None))
    if use_metadata:
        return _dataset_video_id(video.backend_metadata.get("dataset"))
    return None


def _video_id_map(videos: list[Video], *, use_metadata: bool = False) -> np.ndarray:
    """Return the id each video is stored under in a 2.x `/frames` table.

    Args:
        videos: The videos, in order.
        use_metadata: Whether to fall back to the recorded backend metadata of
            videos that aren't open (see `_embedded_video_id`).

    Returns:
        An array with each video's embedded `videoN` number, or its list index for
        videos that aren't embedded.
    """
    ids = [_embedded_video_id(video, use_metadata=use_metadata) for video in videos]
    return np.array(
        [i if vid is None else vid for i, vid in enumerate(ids)], dtype="i8"
    )


def _written_video_id_map(f: h5py.File, videos: list[Video]) -> np.ndarray:
    """Return the 2.x `/frames` id of each video as written to a file.

    The ids follow the `videos_json` already written to `f`, since a save may
    write videos differently from how they are held in memory (for example
    restoring the source video of an embedded one).

    Args:
        f: The file being written, with its `videos_json`.
        videos: The videos written, in order.

    Returns:
        An array with each written video's embedded `videoN` number, or its list
        index for videos that aren't embedded.
    """
    if "videos_json" not in f:
        return _video_id_map(videos)
    ids = [
        _dataset_video_id(json.loads(raw).get("backend", {}).get("dataset"))
        for raw in f["videos_json"][()]
    ]
    return np.array(
        [i if vid is None else vid for i, vid in enumerate(ids)], dtype="i8"
    )


def _video_list_indices(stored: np.ndarray, videos: list[Video]) -> np.ndarray:
    """Map the video ids of a 2.x `/frames` table to list indices.

    Older SLEAP files store embedded videos under their sparse `videoN` numbers
    (for example 0, 15, 29) rather than list indices. Ids that are exactly the list
    indices `0..n-1` are kept; otherwise each embedded video's number maps to its
    index, and other ids are kept as they are.

    Args:
        stored: The stored video id of each frame.
        videos: The file's videos.

    Returns:
        The list index of each frame's video.
    """
    if np.array_equal(np.unique(stored), np.arange(len(videos))):
        return stored
    id_map = {
        vid: i
        for i, vid in enumerate(_video_id_map(videos, use_metadata=True).tolist())
    }
    return np.array([id_map.get(v, v) for v in stored.tolist()], dtype="i8")


def _normalize_frames(
    columns: dict[str, np.ndarray],
    *,
    is_legacy: bool,
    videos: list[Video],
    negative_frames: set[tuple[int, int]],
) -> tuple[np.ndarray, dict[int, int] | None]:
    """Normalise a frame table's columns into the raw in-memory record layout.

    Args:
        columns: The table's columns by name (see `_read_columns`).
        is_legacy: If `True`, the columns come from a 2.x table: stored video ids
            are mapped to list indices (see `_video_list_indices`), `is_negative`
            comes from `negative_frames`, and stored `frame_id` values that aren't
            row numbers are mapped to rows.
        videos: The file's videos.
        negative_frames: The `(stored video id, frame_idx)` pairs of the 2.x
            `/negative_frames` dataset (see `read_negative_frames`).

    Returns:
        A tuple `(rows, id_map)`: the `RAW_FRAME_DTYPE` records followed by any
        unknown columns, and the stored id -> row mapping of a 2.x table whose ids
        aren't row numbers (`None` otherwise).

    Raises:
        ValueError: If a required column is missing or the columns don't all have
            the same length.
    """
    rows = _table_rows(columns, "frames")
    id_map = None
    if is_legacy:
        id_map = _row_id_map(columns, "frames")
        stored = rows["video"].copy()
        if negative_frames:
            rows["is_negative"] = [
                key in negative_frames
                for key in zip(stored.tolist(), rows["frame_idx"].tolist())
            ]
        rows["video"] = _video_list_indices(stored, videos)
    return rows, id_map


def _check_pose_tables(
    labels_path: str,
    instances: np.ndarray,
    frames: np.ndarray,
    *,
    n_points: int,
    n_pred_points: int,
    n_videos: int,
    strict: bool,
) -> None:
    """Check the ranges and links between the pose tables, in one pass.

    Checks that the frames' instance ranges never decrease and end at the instance
    count, that each instance's `frame_id` is the frame whose range holds it, that
    point ranges fit inside their point table, that `from_predicted` is a row (or
    -1) and that `video` indexes a video. Empty tables are valid.

    Args:
        labels_path: Path or URL of the file (used in the message).
        instances: Raw instance records (see `RAW_INSTANCE_DTYPE`).
        frames: Raw frame records (see `RAW_FRAME_DTYPE`).
        n_points: Number of rows in `/points`.
        n_pred_points: Number of rows in `/pred_points`.
        n_videos: Number of videos in the file.
        strict: If `True` (SLP 3.0 tables), raise on a problem. Otherwise (2.x
            tables, which have always been read leniently) warn.

    Raises:
        ValueError: If `strict` and a check fails. The message names the table.
    """
    problems = []
    n_instances = len(instances)
    start = frames["instance_id_start"]
    end = frames["instance_id_end"]
    ranges_ok = not (
        np.any(end < start) or np.any(start[1:] < end[:-1]) or np.any(start < 0)
    )
    if not ranges_ok:
        problems.append("/frames instance ranges decrease or overlap")
    last = int(end[-1]) if len(frames) else 0
    if last != n_instances:
        problems.append(
            f"/frames instance ranges end at {last}, not at the instance count "
            f"({n_instances})"
        )
    if ranges_ok:
        # The frame whose range holds each instance row, or -1.
        rows = np.arange(n_instances)
        owner = np.searchsorted(end, rows, side="right")
        held = owner < len(frames)
        held[held] &= start[owner[held]] <= rows[held]
        owner[~held] = -1
        if np.any(instances["frame_id"] != owner):
            problems.append("/instances frame_id disagrees with the /frames ranges")
    for inst_type, n, table in (
        (InstanceType.USER, n_points, "points"),
        (InstanceType.PREDICTED, n_pred_points, "pred_points"),
    ):
        of_type = instances["instance_type"] == inst_type
        p_start = instances["point_id_start"][of_type]
        p_end = instances["point_id_end"][of_type]
        if np.any((p_start < 0) | (p_end < p_start) | (p_end > n)):
            problems.append(f"/instances point ranges fall outside /{table} ({n} rows)")
    from_predicted = instances["from_predicted"]
    if np.any((from_predicted < -1) | (from_predicted >= n_instances)):
        problems.append("/instances from_predicted is not an instance row")
    video = frames["video"]
    if np.any((video < 0) | (video >= n_videos)):
        problems.append(f"/frames video is not one of the {n_videos} videos")
    if not problems:
        return
    message = f"{labels_path} has inconsistent pose tables: {'; '.join(problems)}."
    if strict:
        raise ValueError(message)
    warnings.warn(f"{message} Reading it leniently, as an SLP 2.x file.", stacklevel=4)


# Metadata column kinds: the in-memory dtype and the placeholder for records that
# lack the key.
_METADATA_DTYPES = {"bool": "?", "int": "i8", "float": "f8", "str": object}
_METADATA_PLACEHOLDERS = {"bool": False, "int": 0, "float": np.nan, "str": ""}


def _read_metadata_columns(
    node: h5py.Group | h5py.Dataset, table: str, n: int
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Read the per-record metadata columns of a 3.0 table.

    Args:
        node: The table (`/instances` or `/frames`). Compound datasets (2.x) have
            no metadata.
        table: The table name, for messages.
        n: The number of records in the table.

    Returns:
        A dictionary mapping each metadata key to `(values, present)`: one value per
        record (strings as a `str` object array) and a `bool` array that is `False`
        for records lacking the key, or `None` if every record has it. Members of
        `metadata/` that aren't a group with a 1-D `values` dataset of a supported
        kind (bool, integer, float or string) are ignored.

    Raises:
        ValueError: If a key's `values` or `present` doesn't have `n` rows.
    """
    if not isinstance(node, h5py.Group) or "metadata" not in node:
        return {}
    columns = {}
    for key, member in node["metadata"].items():
        if not isinstance(member, h5py.Group) or "values" not in member:
            continue
        ds = member["values"]
        if h5py.check_string_dtype(ds.dtype) is not None:
            values = ds.asstr()[()]
        elif ds.dtype.kind in "biuf":
            values = ds[()]
        else:
            continue
        if values.ndim != 1:
            continue
        present = member["present"][()].astype(bool) if "present" in member else None
        if len(values) != n or (present is not None and len(present) != n):
            raise ValueError(
                f"/{table}/metadata/{key} doesn't have one row per record ({n})."
            )
        columns[key] = (values, present)
    return columns


def _metadata_records(
    columns: dict[str, tuple[np.ndarray, np.ndarray | None]], n: int
) -> list[dict]:
    """Turn metadata columns into one metadata dictionary per record.

    Args:
        columns: Metadata columns (see `_read_metadata_columns`).
        n: The number of records.

    Returns:
        A list of `n` dictionaries holding the keys each record has.
    """
    records: list[dict] = [{} for _ in range(n)]
    for key, (values, present) in columns.items():
        vals = values.tolist()
        rows = range(n) if present is None else np.flatnonzero(present).tolist()
        for row in rows:
            records[row][key] = vals[row]
    return records


def _metadata_record(
    columns: dict[str, tuple[np.ndarray, np.ndarray | None]], row: int
) -> dict:
    """Return the metadata dictionary of one record (see `_metadata_records`)."""
    return {
        key: values[row : row + 1].tolist()[0]
        for key, (values, present) in columns.items()
        if present is None or present[row]
    }


def _metadata_columns(
    records: list, kinds: dict[str, str]
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Build metadata columns from records' `metadata` dictionaries.

    Args:
        records: Objects with a `metadata` attribute, one per table row.
        kinds: The kind of each key (see `Labels.check_metadata`).

    Returns:
        Metadata columns (see `_read_metadata_columns`). Records lacking a key get
        a placeholder value (False, 0, NaN or "").
    """
    n = len(records)
    columns = {
        key: (
            np.full(n, _METADATA_PLACEHOLDERS[kind], dtype=_METADATA_DTYPES[kind]),
            np.zeros(n, dtype=bool),
        )
        for key, kind in kinds.items()
    }
    if columns:
        for row, record in enumerate(records):
            for key, value in record.metadata.items():
                if value is not None:
                    values, present = columns[key]
                    values[row] = value
                    present[row] = True
    return {
        key: (values, None if present.all() else present)
        for key, (values, present) in columns.items()
    }


def _write_metadata_columns(
    grp: h5py.Group,
    columns: dict[str, tuple[np.ndarray, np.ndarray | None]],
    *,
    compression: str | None = None,
) -> None:
    """Write metadata columns into a table's `metadata/` group.

    Args:
        grp: The table group (`/instances` or `/frames`).
        columns: Metadata columns (see `_read_metadata_columns`). Nothing is written
            if empty.
        compression: `None`, `"gzip"` or `"lzf"` (see `_write_columns`).
    """
    if not columns:
        return
    md = grp.create_group("metadata")
    for key, (values, present) in columns.items():
        col = md.create_group(key)
        kwargs = _compression_kwargs(compression, values.shape)
        if values.dtype == object:
            kwargs["dtype"] = h5py.string_dtype()
        col.create_dataset("values", data=values, **kwargs)
        if present is not None:
            col.create_dataset(
                "present",
                data=present,
                **_compression_kwargs(compression, present.shape),
            )


class _PoseTables(NamedTuple):
    """The normalised pose tables of an SLP file (see `_read_pose_tables`).

    Attributes:
        points: Raw user point records (see `read_points`).
        pred_points: Raw predicted point records (see `read_pred_points`).
        instances: Raw instance records (see `_normalize_instances`).
        frames: Raw frame records (see `_normalize_frames`).
        instance_id_map: Stored instance id -> row for a 2.x file whose ids aren't
            row numbers, or `None`.
        instance_metadata: Metadata columns of `/instances`.
        frame_metadata: Metadata columns of `/frames`.
    """

    points: np.ndarray
    pred_points: np.ndarray
    instances: np.ndarray
    frames: np.ndarray
    instance_id_map: dict[int, int] | None
    instance_metadata: dict
    frame_metadata: dict


def _read_pose_tables(
    labels_path: str, f: h5py.File, videos: list[Video]
) -> _PoseTables:
    """Read, normalise and check `/points`, `/pred_points`, `/instances`, `/frames`.

    Each table is read in its own layout (3.0 columns or 2.x compound), so files
    mixing the two read too. 2.x tables are cleaned up by the normalisers, and
    `/negative_frames` is read into `is_negative` for 2.x frame tables. The range
    checks (`_check_pose_tables`) raise for 3.0 tables and warn for 2.x ones.

    Args:
        labels_path: Path or URL of the file.
        f: The open file.
        videos: The file's videos (see `read_videos`).

    Returns:
        The normalised tables.
    """
    points = read_points(labels_path, _hdf5_file=f)
    pred_points = read_pred_points(labels_path, _hdf5_file=f)
    inst_columns, inst_legacy = _read_columns(f["instances"])
    instances, instance_id_map = _normalize_instances(
        inst_columns, is_legacy=inst_legacy
    )
    frame_columns, frame_legacy = _read_columns(f["frames"])
    frames, frame_id_map = _normalize_frames(
        frame_columns,
        is_legacy=frame_legacy,
        videos=videos,
        negative_frames=(
            read_negative_frames(labels_path, _hdf5_file=f) if frame_legacy else set()
        ),
    )
    if frame_id_map is not None:
        instances["frame_id"] = _remap_ids(instances["frame_id"], frame_id_map)
    _check_pose_tables(
        labels_path,
        instances,
        frames,
        n_points=len(points),
        n_pred_points=len(pred_points),
        n_videos=len(videos),
        strict=not (inst_legacy or frame_legacy),
    )
    return _PoseTables(
        points=points,
        pred_points=pred_points,
        instances=instances,
        frames=frames,
        instance_id_map=instance_id_map,
        instance_metadata=_read_metadata_columns(
            f["instances"], "instances", len(instances)
        ),
        frame_metadata=_read_metadata_columns(f["frames"], "frames", len(frames)),
    )


def _remap_instance_owners(
    id_map: dict[int, int] | None,
    by_owner: Iterable[dict],
    annotations: Iterable,
    instances: list | None = None,
) -> None:
    """Remap links to instances from stored 2.x instance ids to rows.

    Args:
        id_map: Stored instance id -> row (see `_normalize_instances`). Nothing is
            done if `None`.
        by_owner: Owner-type-keyed link dictionaries (identity and category links,
            embeddings), whose `OWNER_INSTANCE` entries are rekeyed in place.
        annotations: ROIs, masks, bounding boxes, centroids and label images whose
            instance links (`_instance_idx`) are remapped in place.
        instances: The instances by row, to resolve each annotation's `instance`
            (eager reads), or `None` to only remap the stored index (lazy reads).
    """
    if id_map is None:
        return
    for links in by_owner:
        if OWNER_INSTANCE in links:
            links[OWNER_INSTANCE] = {
                id_map[sid]: value
                for sid, value in links[OWNER_INSTANCE].items()
                if sid in id_map
            }
    for ann in annotations:
        targets = ann.objects.values() if isinstance(ann, LabelImage) else (ann,)
        for target in targets:
            if target._instance_idx < 0:
                continue
            row = id_map.get(target._instance_idx, -1)
            target._instance_idx = row
            if instances is not None:
                target.instance = instances[row] if row >= 0 else None


def _instances_from_rows(
    rows: np.ndarray,
    skeletons: list[Skeleton],
    tracks: list[Track],
    points: np.ndarray,
    pred_points: np.ndarray,
    metadata: dict,
    *,
    point_owners: list | None = None,
) -> list[Instance | PredictedInstance | None]:
    """Build instances from raw instance records.

    Args:
        rows: Raw instance records (see `RAW_INSTANCE_DTYPE`).
        skeletons: The file's skeletons.
        tracks: The file's tracks.
        points: Raw user point records.
        pred_points: Raw predicted point records.
        metadata: The table's metadata columns (see `_read_metadata_columns`).
        point_owners: If given, a list that each instance is appended to as a
            `(point_id_start, point_id_end, instance)` tuple (see `_column_carry`).

    Returns:
        One instance per row, or `None` for rows of an unknown instance type.
        `from_predicted` links are resolved, and a NaN `tracking_score` reads as
        `None`.
    """
    n = len(rows)
    records = _metadata_records(metadata, n) if metadata else None
    types = rows["instance_type"].tolist()
    skeleton_ids = rows["skeleton"].tolist()
    track_ids = rows["track"].tolist()
    scores = rows["score"].tolist()
    starts = rows["point_id_start"].tolist()
    ends = rows["point_id_end"].tolist()
    tracking_scores = rows["tracking_score"].tolist()
    instances: list[Instance | PredictedInstance | None] = [None] * n
    for row in range(n):
        skeleton = skeletons[skeleton_ids[row]]
        track = tracks[track_ids[row]] if track_ids[row] >= 0 else None
        tracking_score = tracking_scores[row]
        if np.isnan(tracking_score):
            tracking_score = None
        start, end = starts[row], ends[row]
        md = records[row] if records is not None else None
        if types[row] == InstanceType.USER:
            inst = Instance(
                _points_from_hdf5_data(points[start:end], skeleton, is_predicted=False),
                skeleton=skeleton,
                track=track,
                tracking_score=tracking_score,
                metadata=md,
            )
        elif types[row] == InstanceType.PREDICTED:
            inst = PredictedInstance(
                _points_from_hdf5_data(
                    pred_points[start:end], skeleton, is_predicted=True
                ),
                skeleton=skeleton,
                track=track,
                score=scores[row],
                tracking_score=tracking_score,
                metadata=md,
            )
        else:
            continue
        instances[row] = inst
        if point_owners is not None:
            point_owners.append((start, end, inst))

    for row, source in enumerate(rows["from_predicted"].tolist()):
        if 0 <= source < n and instances[row] is not None:
            instances[row].from_predicted = instances[source]
    return instances


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
        points: Normalised user point records (see `read_points`).
        pred_points: Normalised predicted point records (see `read_pred_points`).
        format_id: The format version identifier used to specify the format of the input
            file.
        _hdf5_file: An already-open `h5py.File` handle to read from. If provided,
            the data is read from this handle (left open for the caller to close);
            otherwise `labels_path` is opened and closed internally. This is a
            private argument used to thread a single open handle through reads.

    Returns:
        A list of `Instance` and/or `PredictedInstance` objects, in row order, with
        their metadata and `from_predicted` links. Rows of an unknown instance type
        are skipped.
    """
    with (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    ) as f:
        columns, is_legacy = _read_columns(f["instances"])
        rows, _ = _normalize_instances(columns, is_legacy=is_legacy)
        metadata = _read_metadata_columns(f["instances"], "instances", len(rows))
    instances = _instances_from_rows(
        rows, skeletons, tracks, points, pred_points, metadata
    )
    return [inst for inst in instances if inst is not None]


_UNLABELED_AS_MODES = ("error", "absent")


def _check_unlabeled_points(labels: Labels, unlabeled_as: str) -> None:
    """Refuse or downgrade unlabeled points before writing the compound layout.

    The compound `/points` and `/pred_points` datasets have no `labeled` column, so an
    unlabeled point would be read back as a labeled, hidden point and trained as
    absent. This check keeps that from happening silently.

    Args:
        labels: The labels about to be written.
        unlabeled_as: `"error"` raises if any point is unlabeled. `"absent"` warns;
            the legacy writer then writes unlabeled points as hidden with no
            location.

    Raises:
        ValueError: If `unlabeled_as` is `"error"` and an instance has unlabeled
            points.
    """
    n_unlabeled = 0
    for inst in labels.instances:
        points = inst.points
        n_unlabeled += int(np.count_nonzero(~(points["labeled"] | points["visible"])))
    _refuse_unlabeled_points(n_unlabeled, unlabeled_as)


def _refuse_unlabeled_points(n_unlabeled: int, unlabeled_as: str) -> None:
    """Raise or warn about unlabeled points the compound layout can't store.

    Args:
        n_unlabeled: Number of unlabeled points about to be written.
        unlabeled_as: `"error"` raises; `"absent"` warns.

    Raises:
        ValueError: If `unlabeled_as` is `"error"` and `n_unlabeled` is not zero.
    """
    if n_unlabeled == 0:
        return
    if unlabeled_as == "error":
        raise ValueError(
            f"{n_unlabeled} point(s) are unlabeled, but the legacy SLP layout can't "
            "store unlabeled points: they would be read back as labeled hidden "
            'points. Pass unlabeled_as="absent" to save them as hidden points '
            "instead, or save without legacy_format=True."
        )
    warnings.warn(
        f"Saving {n_unlabeled} unlabeled point(s) as hidden points with no location "
        "(labeled, not visible), since the legacy SLP layout can't store unlabeled "
        "points.",
        stacklevel=4,
    )


def _check_dropped_metadata(labels: Labels, *, legacy_format: bool) -> None:
    """Check record metadata kinds, then warn about metadata the layout drops.

    Args:
        labels: The labels about to be written.
        legacy_format: If `True`, the 2.x layout is written, which stores no
            metadata. Otherwise (SLP 3.0) instance and frame metadata is stored and
            only bounding box metadata is dropped.

    Raises:
        TypeError: If a metadata key mixes kinds across a table's records (see
            `Labels.check_metadata`).
    """
    kinds = labels.check_metadata()
    if legacy_format and any(kinds.values()):
        warnings.warn(
            "Instance, frame and bounding box metadata can't be stored in the legacy "
            "SLP layout and was not saved.",
            stacklevel=3,
        )
    elif not legacy_format and kinds["bboxes"]:
        warnings.warn(
            "Bounding box metadata can't be stored yet and was not saved.",
            stacklevel=3,
        )


def _raw_points_from_instances(
    instances: list[Instance | PredictedInstance], *, is_predicted: bool
) -> np.ndarray:
    """Gather the points of instances into raw point records.

    Args:
        instances: Instances whose points are concatenated, in order.
        is_predicted: Whether to build `RAW_PRED_POINT_DTYPE` records (with
            `score`) instead of `RAW_POINT_DTYPE` records.

    Returns:
        One record per point. `labeled` is stored as `labeled | visible`, and
        unlabeled points have NaN coordinates.
    """
    dtype = RAW_PRED_POINT_DTYPE if is_predicted else RAW_POINT_DTYPE
    if not instances:
        return np.empty(0, dtype=dtype)
    arrays = [inst.points for inst in instances]
    visible = np.concatenate([a["visible"] for a in arrays])
    labeled = np.concatenate([a["labeled"] for a in arrays]) | visible
    xy = np.concatenate([a["xy"] for a in arrays])
    xy[~labeled] = np.nan
    rows = np.empty(len(visible), dtype=dtype)
    rows["x"] = xy[:, 0]
    rows["y"] = xy[:, 1]
    rows["visible"] = visible
    rows["verified"] = np.concatenate([a["verified"] for a in arrays])
    rows["labeled"] = labeled
    if is_predicted:
        rows["score"] = np.concatenate([a["score"] for a in arrays])
    return rows


def _legacy_point_rows(rows: np.ndarray, *, is_predicted: bool) -> np.ndarray:
    """Convert raw point records to the 2.x compound layout.

    Args:
        rows: Raw point records (see `RAW_POINT_DTYPE`).
        is_predicted: Whether to build `LEGACY_PRED_POINT_DTYPE` records.

    Returns:
        Compound records with `verified` stored as `complete`. Unlabeled points
        become hidden points with NaN coordinates; unknown columns are dropped.
    """
    dtype = LEGACY_PRED_POINT_DTYPE if is_predicted else LEGACY_POINT_DTYPE
    legacy = np.empty(len(rows), dtype=dtype)
    unlabeled = ~(rows["labeled"] | rows["visible"])
    legacy["x"] = np.where(unlabeled, np.nan, rows["x"])
    legacy["y"] = np.where(unlabeled, np.nan, rows["y"])
    legacy["visible"] = rows["visible"]
    legacy["complete"] = rows["verified"]
    if is_predicted:
        legacy["score"] = rows["score"]
    return legacy


def _compression_kwargs(compression: str | None, shape: tuple) -> dict:
    """Return `create_dataset` keyword arguments for a table's compression.

    Args:
        compression: `None`, `"gzip"` or `"lzf"`.
        shape: Shape of the dataset to write.

    Returns:
        An empty dict for `None` or an empty dataset (contiguous and uncompressed).
        Otherwise chunked compression with the shuffle filter (gzip at level 1).
    """
    if compression is None or shape[0] == 0:
        return {}
    kwargs = dict(
        compression=compression,
        shuffle=True,
        chunks=(min(shape[0], _COLUMN_CHUNK_ROWS), *shape[1:]),
    )
    if compression == "gzip":
        kwargs["compression_opts"] = 1
    return kwargs


def _write_point_tables(
    f: h5py.File,
    points: np.ndarray,
    pred_points: np.ndarray,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
) -> None:
    """Write the `/points` and `/pred_points` tables.

    Args:
        f: The open file to write into.
        points: Raw user point records (see `RAW_POINT_DTYPE`).
        pred_points: Raw predicted point records (see `RAW_PRED_POINT_DTYPE`).
        legacy_format: If `True`, write 2.x compound datasets (see
            `_legacy_point_rows`); otherwise groups of columns.
        compression: `None`, `"gzip"` or `"lzf"`.
    """
    for name, rows, is_predicted in (
        ("points", points, False),
        ("pred_points", pred_points, True),
    ):
        if legacy_format:
            legacy = _legacy_point_rows(rows, is_predicted=is_predicted)
            f.create_dataset(
                name, data=legacy, **_compression_kwargs(compression, legacy.shape)
            )
        else:
            _write_columns(f, name, rows, compression=compression)


class _ColumnCarry:
    """Unknown table columns read from a file, kept to carry them through a save.

    A newer sleap-io may add columns to a table (for example `/points`, `/frames`
    or `/masks`) that this version doesn't know. They are kept alongside the loaded
    `Labels` and written back if the rows of their table still line up with the
    objects when saving: the same objects (for example the same instances, with the
    same points arrays), in the same order. Otherwise they are dropped with a
    warning.

    Attributes:
        tables: Maps a table name to a tuple `(owners, columns)`. `owners` is a
            tuple with one key per object owning the table's rows, in order, or
            `None` if the rows couldn't be attributed to objects. A key is
            `(instance, points)` for the point tables, the tuple itself for owners
            given as tuples (such as `(instance group, camera)` for
            `/session_data/instance_group_members` and the `(owner_type, owner_id,
            catalog index)` values of a link row, see `_link_keys`), and
            `(object,)` otherwise (see `_carry_keys`). `columns` is a structured
            array of the unknown columns, one record per row.
        consumed: Names of the tables a save has offered rows for so far (see
            `_with_carried_columns` and `_warn_unconsumed_carry`).
    """

    def __init__(self, tables: dict[str, tuple[tuple | None, np.ndarray]]):
        """Initialize from the carried tables (see class attributes)."""
        self.tables = tables
        self.consumed: set[str] = set()


def _carry_keys(name: str, objects: Iterable) -> tuple:
    """Return the keys identifying the objects that own a table's rows.

    Args:
        name: The table name.
        objects: The objects owning the table's rows, in order: instances for the
            pose tables other than `/frames`, labeled frames for `/frames`, and so
            on (see `_ColumnCarry`).

    Returns:
        A tuple with one key per object (see `_ColumnCarry`).
    """
    if name in ("points", "pred_points"):
        return tuple((inst, inst.points) for inst in objects)
    return tuple(obj if isinstance(obj, tuple) else (obj,) for obj in objects)


def _same_owner(a: object, b: object) -> bool:
    """Whether two parts of carry keys refer to the same owner.

    Objects must be identical; integers (the values keying link rows) equal.
    """
    return a is b or (type(a) is int and type(b) is int and a == b)


def _unknown_columns(rows: np.ndarray, base: np.dtype) -> list[str]:
    """Return the names of the columns of `rows` that aren't fields of `base`."""
    return [name for name in rows.dtype.names if name not in base.names]


def _point_row_owners(
    owners: list[tuple[int, int, Instance | PredictedInstance]],
    inst_type: type,
    n_rows: int,
) -> list | None:
    """Return the instances owning a point table's rows, in row order.

    Args:
        owners: `(point_id_start, point_id_end, instance)` for every instance read.
        inst_type: `Instance` for `/points`, `PredictedInstance` for `/pred_points`.
        n_rows: The number of rows in the point table.

    Returns:
        The instances of `inst_type` sorted by their rows, or `None` unless they own
        the rows back to back, from the first row to the last.
    """
    spans = sorted(
        (span for span in owners if type(span[2]) is inst_type),
        key=lambda span: span[0],
    )
    position = 0
    for start, end, _ in spans:
        if start != position:
            return None
        position = end
    return [inst for _, _, inst in spans] if position == n_rows else None


def _column_carry(
    tables: dict[str, tuple[np.ndarray, np.dtype, Callable[[], list | None]]],
) -> _ColumnCarry | None:
    """Collect unknown table columns to carry through a later save.

    Args:
        tables: Maps a table name to `(rows, base, owners)`: the normalised records
            as read, their raw dtype (for example `RAW_POINT_DTYPE`), and a
            function returning the objects owning the rows in order (or `None` if
            they can't be attributed), called only if the table has unknown
            columns.

    Returns:
        A `_ColumnCarry`, or `None` if no table has unknown columns.
    """
    carried = {}
    for name, (rows, base, owners) in tables.items():
        extra = _unknown_columns(rows, base)
        if not extra:
            continue
        columns = np.empty(
            len(rows), dtype=[(n, rows.dtype.fields[n][0]) for n in extra]
        )
        for n in extra:
            columns[n] = rows[n]
        objects = owners()
        carried[name] = (
            _carry_keys(name, objects) if objects is not None else None,
            columns,
        )
    return _ColumnCarry(carried) if carried else None


def _warn_dropped_columns(
    names: Iterable[str], columns: dict[str, Iterable[str]], reason: str
) -> None:
    """Warn that unknown table columns are not being saved.

    Args:
        names: Names of the tables whose columns are dropped.
        columns: Maps each table name to its unknown column names.
        reason: `"legacy"` (the legacy layout can't store them) or `"changed"` (the
            objects changed since loading, so the rows no longer line up).
    """
    for name in names:
        cols = ", ".join(columns[name])
        why = (
            "the legacy SLP layout can't store them"
            if reason == "legacy"
            else "its records changed since the file was loaded, so its rows no "
            "longer line up"
        )
        warnings.warn(
            f"Unknown column(s) {cols} of /{name} were not saved: {why}.",
            stacklevel=4,
        )


def _warn_dropped_carry(carry: _ColumnCarry | None) -> None:
    """Warn that the legacy layout drops every carried column (see `_ColumnCarry`)."""
    if carry is not None:
        _warn_dropped_columns(
            carry.tables,
            {name: cols.dtype.names for name, (_, cols) in carry.tables.items()},
            "legacy",
        )


def _start_carry(carry: _ColumnCarry | None) -> None:
    """Start a save that may carry unknown columns (see `_warn_unconsumed_carry`)."""
    if carry is not None:
        carry.consumed.clear()


def _warn_unconsumed_carry(carry: _ColumnCarry | None) -> None:
    """Warn about carried columns of tables a save wrote no rows for.

    A table is left out of a save when its records are all gone (for example
    every ROI was removed), so its unknown columns are dropped.
    """
    if carry is None:
        return
    unwritten = [name for name in carry.tables if name not in carry.consumed]
    _warn_dropped_columns(
        unwritten,
        {name: carry.tables[name][1].dtype.names for name in unwritten},
        "changed",
    )


def _with_carried_columns(
    rows: np.ndarray,
    carry: _ColumnCarry | None,
    name: str,
    objects: list,
) -> np.ndarray:
    """Append carried unknown columns to freshly built records.

    Args:
        rows: Raw records built from `objects`.
        carry: Unknown columns kept from loading, or `None`.
        name: The table name.
        objects: The objects owning `rows`, in order (see `_carry_keys`).

    Returns:
        `rows` with the carried columns appended when the objects are the ones
        loaded, in the same order. Otherwise `rows` unchanged, with a warning if
        columns had to be dropped.
    """
    if carry is None or name not in carry.tables:
        return rows
    carry.consumed.add(name)
    owners, columns = carry.tables[name]
    keys = _carry_keys(name, objects)
    if (
        owners is not None
        and len(owners) == len(keys)
        and all(
            len(owner) == len(key) and all(map(_same_owner, owner, key))
            for owner, key in zip(owners, keys)
        )
    ):
        merged = np.empty(len(rows), dtype=rows.dtype.descr + columns.dtype.descr)
        for n in rows.dtype.names:
            merged[n] = rows[n]
        for n in columns.dtype.names:
            merged[n] = columns[n]
        return merged
    _warn_dropped_columns([name], {name: columns.dtype.names}, "changed")
    return rows


def _legacy_table_rows(rows: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert raw instance or frame records to a 2.x compound layout.

    Args:
        rows: Raw records (see `RAW_INSTANCE_DTYPE` and `RAW_FRAME_DTYPE`).
        dtype: `LEGACY_INSTANCE_DTYPE` or `LEGACY_FRAME_DTYPE`.

    Returns:
        Compound records of `dtype`, with the row numbers as ids. Unknown columns
        are dropped. For frames, `video` is still the list index (see
        `_write_pose_tables`).
    """
    legacy = np.empty(len(rows), dtype=dtype)
    for name in dtype.names:
        legacy[name] = rows[name]
    return legacy


def _without_field(rows: np.ndarray, name: str) -> np.ndarray:
    """Return a view of structured records without one field."""
    return rows[[n for n in rows.dtype.names if n != name]]


def _write_pose_tables(
    f: h5py.File,
    points: np.ndarray,
    pred_points: np.ndarray,
    instances: np.ndarray,
    frames: np.ndarray,
    *,
    videos: list[Video],
    instance_metadata: dict,
    frame_metadata: dict,
    legacy_format: bool = False,
    compression: str | None = None,
) -> None:
    """Write `/points`, `/pred_points`, `/instances` and `/frames`.

    Args:
        f: The open file to write into.
        points: Raw user point records (see `RAW_POINT_DTYPE`).
        pred_points: Raw predicted point records (see `RAW_PRED_POINT_DTYPE`).
        instances: Raw instance records (see `RAW_INSTANCE_DTYPE`), whose
            `instance_id` is the row number.
        frames: Raw frame records (see `RAW_FRAME_DTYPE`), whose `frame_id` is the
            row number and `video` indexes `videos`.
        videos: The videos written to the file, in order.
        instance_metadata: Metadata columns of the instances (see
            `_read_metadata_columns`).
        frame_metadata: Metadata columns of the frames.
        legacy_format: If `True`, write 2.x compound datasets, with each embedded
            video's sparse `videoN` number as its id in `/frames` and negative
            frames in `/negative_frames`. Metadata and unknown columns are not
            written. Otherwise groups of columns, without the id columns.
        compression: `None`, `"gzip"` or `"lzf"`.
    """
    _write_point_tables(
        f, points, pred_points, legacy_format=legacy_format, compression=compression
    )
    if legacy_format:
        legacy_instances = _legacy_table_rows(instances, LEGACY_INSTANCE_DTYPE)
        legacy_frames = _legacy_table_rows(frames, LEGACY_FRAME_DTYPE)
        legacy_frames["video"] = _written_video_id_map(f, videos)[frames["video"]]
        for name, rows in (("instances", legacy_instances), ("frames", legacy_frames)):
            f.create_dataset(
                name, data=rows, **_compression_kwargs(compression, rows.shape)
            )
        negative = legacy_frames[frames["is_negative"]]
        if len(negative):
            neg = np.empty(
                len(negative), dtype=[("video_id", "u4"), ("frame_idx", "u8")]
            )
            neg["video_id"] = negative["video"]
            neg["frame_idx"] = negative["frame_idx"]
            f.create_dataset("negative_frames", data=neg)
        return
    for name, rows, metadata in (
        ("instances", instances, instance_metadata),
        ("frames", frames, frame_metadata),
    ):
        grp = _write_columns(
            f,
            name,
            _without_field(rows, _ROW_ID_COLUMNS[name]),
            compression=compression,
        )
        _write_metadata_columns(grp, metadata, compression=compression)


# -- Remaining tables (SLP 3.0) --------------------------------------------------------
# ROIs, masks, label images, identity/category links and the session tables are
# groups of 1-D columns in SLP 3.0, like the pose tables. Byte data (WKB geometry,
# RLE counts, score maps, label image pixels) stays in a flat buffer that the rows
# index with start/end ranges, since variable-length bytes can't hold NULs.


class _TableSpec(NamedTuple):
    """Where and how one of the remaining tables is stored.

    Attributes:
        path: Path of the table in SLP 3.0, where it is a group of columns.
        legacy_path: Path of the 2.x compound dataset.
        raw: The in-memory record layout every reader normalises to, which is also
            the column layout written in 3.0 (object fields hold strings).
        legacy: The 2.x compound layout, written by `legacy_format=True`.
        defaults: Defaults for columns that older files may lack. Every other field
            of `raw` is required.
        buffer: Name of the byte buffer the rows index into, inside the 3.0 group.
        legacy_buffer: Path of the buffer in 2.x files.
        legacy_strings: Maps each string column to the 2.x dataset holding it and
            the JSON attribute of the 2.x table that held it before (format 1.9).
    """

    path: str
    legacy_path: str
    raw: np.dtype
    legacy: np.dtype
    defaults: dict
    buffer: str | None = None
    legacy_buffer: str | None = None
    legacy_strings: dict = {}


_SPATIAL_FIELDS = [
    ("scale_x", "f8"),
    ("scale_y", "f8"),
    ("offset_x", "f8"),
    ("offset_y", "f8"),
]
_LEGACY_SPATIAL_FIELDS = [
    ("scale_x", "f4"),
    ("scale_y", "f4"),
    ("offset_x", "f4"),
    ("offset_y", "f4"),
]
_SPATIAL_DEFAULTS = {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0}
_SCORE_MAP_FIELDS = [
    ("data_start", "i8"),
    ("data_end", "i8"),
    ("height", "i8"),
    ("width", "i8"),
    *_SPATIAL_FIELDS,
]

_TABLE_SPECS = {
    "rois": _TableSpec(
        path="rois",
        legacy_path="rois",
        raw=np.dtype(
            [
                ("annotation_type", "u1"),
                ("video", "i8"),
                ("frame_idx", "i8"),
                ("track", "i8"),
                ("instance", "i8"),
                ("is_predicted", "?"),
                ("score", "f8"),
                ("tracking_score", "f8"),
                ("wkb_start", "i8"),
                ("wkb_end", "i8"),
                ("name", "O"),
                ("category", "O"),
                ("source", "O"),
            ]
        ),
        legacy=np.dtype(
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
        ),
        defaults={
            "annotation_type": 0,
            "instance": -1,
            "is_predicted": False,
            "score": np.nan,
            "tracking_score": np.nan,
            "name": "",
            "category": "",
            "source": "",
        },
        buffer="wkb",
        legacy_buffer="roi_wkb",
        legacy_strings={
            "category": ("roi_categories", "categories"),
            "name": ("roi_names", "names"),
            "source": ("roi_sources", "sources"),
        },
    ),
    "masks": _TableSpec(
        path="masks",
        legacy_path="masks",
        raw=np.dtype(
            [
                ("height", "i8"),
                ("width", "i8"),
                ("annotation_type", "u1"),
                ("video", "i8"),
                ("frame_idx", "i8"),
                ("track", "i8"),
                ("instance", "i8"),
                ("from_predicted", "i8"),
                ("is_predicted", "?"),
                ("score", "f8"),
                ("tracking_score", "f8"),
                ("rle_start", "i8"),
                ("rle_end", "i8"),
                *_SPATIAL_FIELDS,
                ("name", "O"),
                ("category", "O"),
                ("source", "O"),
            ]
        ),
        legacy=np.dtype(
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
                *_LEGACY_SPATIAL_FIELDS,
            ]
        ),
        defaults={
            "annotation_type": 2,
            "instance": -1,
            "from_predicted": -1,
            "is_predicted": False,
            "score": np.nan,
            "tracking_score": np.nan,
            **_SPATIAL_DEFAULTS,
            "name": "",
            "category": "",
            "source": "",
        },
        buffer="rle",
        legacy_buffer="mask_rle",
        legacy_strings={
            "category": ("mask_categories", "categories"),
            "name": ("mask_names", "names"),
            "source": ("mask_sources", "sources"),
        },
    ),
    "masks/score_maps": _TableSpec(
        path="masks/score_maps",
        legacy_path="mask_score_map_index",
        raw=np.dtype([("mask_idx", "i8"), *_SCORE_MAP_FIELDS]),
        legacy=np.dtype(
            [
                ("mask_idx", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("height", "u4"),
                ("width", "u4"),
                *_LEGACY_SPATIAL_FIELDS,
            ]
        ),
        defaults=dict(_SPATIAL_DEFAULTS),
        buffer="data",
        legacy_buffer="mask_score_maps",
    ),
    "label_images": _TableSpec(
        path="label_images",
        legacy_path="label_images",
        raw=np.dtype(
            [
                ("video", "i8"),
                ("frame_idx", "i8"),
                ("height", "i8"),
                ("width", "i8"),
                ("n_objects", "i8"),
                ("objects_start", "i8"),
                ("data_start", "i8"),
                ("data_end", "i8"),
                ("is_predicted", "?"),
                ("score", "f8"),
                *_SPATIAL_FIELDS,
                ("source", "O"),
            ]
        ),
        legacy=LI_DTYPE,
        defaults={
            "is_predicted": False,
            "score": np.nan,
            **_SPATIAL_DEFAULTS,
            "source": "",
        },
        buffer="data",
        legacy_buffer="label_image_data",
        legacy_strings={"source": ("label_image_sources", "sources")},
    ),
    "label_images/objects": _TableSpec(
        path="label_images/objects",
        legacy_path="label_image_objects",
        raw=np.dtype(
            [
                ("label_id", "i8"),
                ("track", "i8"),
                ("instance", "i8"),
                ("score", "f8"),
                ("tracking_score", "f8"),
                ("name", "O"),
                ("category", "O"),
            ]
        ),
        legacy=OBJ_DTYPE,
        defaults={
            "score": np.nan,
            "tracking_score": np.nan,
            "name": "",
            "category": "",
        },
        legacy_strings={
            "category": ("label_image_obj_categories", "categories"),
            "name": ("label_image_obj_names", "names"),
        },
    ),
    "label_images/score_maps": _TableSpec(
        path="label_images/score_maps",
        legacy_path="label_image_score_map_index",
        raw=np.dtype([("li_idx", "i8"), *_SCORE_MAP_FIELDS]),
        legacy=LI_SM_INDEX_DTYPE,
        defaults=dict(_SPATIAL_DEFAULTS),
        buffer="data",
        legacy_buffer="label_image_score_maps",
    ),
    "identity/links": _TableSpec(
        path="identity/links",
        legacy_path="identity/links",
        raw=np.dtype(
            [
                ("owner_type", "u1"),
                ("owner_id", "i8"),
                ("identity_idx", "i8"),
                ("identity_score", "f8"),
            ]
        ),
        legacy=np.dtype(
            [
                ("owner_type", "u1"),
                ("owner_id", "i8"),
                ("identity_idx", "i4"),
                ("identity_score", "f4"),
            ]
        ),
        defaults={},
    ),
    "categories/links": _TableSpec(
        path="categories/links",
        legacy_path="categories/links",
        raw=np.dtype(
            [
                ("owner_type", "u1"),
                ("owner_id", "i8"),
                ("category_idx", "i8"),
                ("category_score", "f8"),
            ]
        ),
        legacy=np.dtype(
            [
                ("owner_type", "u1"),
                ("owner_id", "i8"),
                ("category_idx", "i4"),
                ("category_score", "f4"),
            ]
        ),
        defaults={},
    ),
    "session_data/frame_groups": _TableSpec(
        path="session_data/frame_groups",
        legacy_path="session_data/frame_groups",
        raw=np.dtype([("frame_idx", "i8"), ("ig_start", "i8"), ("ig_end", "i8")]),
        legacy=FRAME_GROUP_DTYPE,
        defaults={},
    ),
    "session_data/instance_groups": _TableSpec(
        path="session_data/instance_groups",
        legacy_path="session_data/instance_groups",
        raw=np.dtype(
            [
                ("identity_idx", "i8"),
                ("score", "f8"),
                ("instance_3d_score", "f8"),
                ("pts3d_start", "i8"),
                ("pts3d_end", "i8"),
                ("pts3d_predicted", "?"),
                ("member_start", "i8"),
                ("member_end", "i8"),
            ]
        ),
        legacy=INSTANCE_GROUP_DTYPE,
        defaults={},
    ),
    "session_data/instance_group_members": _TableSpec(
        path="session_data/instance_group_members",
        legacy_path="session_data/instance_group_members",
        raw=np.dtype([("camera", "i8"), ("lf", "i8"), ("inst", "i8")]),
        legacy=INSTANCE_GROUP_MEMBER_DTYPE,
        defaults={},
    ),
}


class _Table(NamedTuple):
    """One of the remaining tables, read and normalised (see `_read_table`).

    Attributes:
        rows: Raw records (see `_TableSpec.raw`), followed by any unknown columns.
        is_legacy: Whether the table was stored in a 2.x layout.
        buffer: The byte buffer the rows index into, or `None` if the file has
            none (or the table has no buffer).
    """

    rows: np.ndarray
    is_legacy: bool
    buffer: h5py.Dataset | None


def _find_node(f: h5py.File, path: str) -> h5py.Group | h5py.Dataset | None:
    """Return the object at `path`, or `None` if it (or a parent group) is absent."""
    node = f
    for part in path.split("/"):
        if not isinstance(node, h5py.Group) or part not in node:
            return None
        node = node[part]
    return node


def _legacy_strings(
    f: h5py.File, dataset: str, attrs: h5py.AttributeManager, key: str, n: int
) -> np.ndarray:
    """Read a 2.x string column, stored as a dataset or (format < 1.9) an attribute.

    Args:
        f: The open file.
        dataset: Path of the 2.x string dataset.
        attrs: Attributes of the 2.x table, which may hold the strings as JSON.
        key: Name of that attribute.
        n: The number of rows in the table.

    Returns:
        A `str` object array of `n` strings, padded with `""` if fewer are stored.
    """
    if dataset in f:
        stored = _read_column(f[dataset]).tolist()
    else:
        stored = json.loads(attrs.get(key, "[]"))
    values = np.full(n, "", dtype=object)
    stored = stored[:n]
    values[: len(stored)] = stored
    return values


def _read_table(f: h5py.File, table: str) -> _Table | None:
    """Read and normalise one of the remaining tables, whatever its layout.

    Args:
        f: The open file.
        table: A key of `_TABLE_SPECS`.

    Returns:
        The normalised table, or `None` if the file doesn't have it. A 3.0 table
        (a group of columns) is looked up first, then the 2.x compound (or h5wasm
        flat) dataset, whose string columns come from their separate datasets.

    Raises:
        ValueError: If a required column is missing or the columns don't all have
            the same length.
    """
    spec = _TABLE_SPECS[table]
    node = _find_node(f, spec.path)
    if node is None or (
        not isinstance(node, h5py.Group) and spec.path != spec.legacy_path
    ):
        node = _find_node(f, spec.legacy_path)
    if node is None:
        return None
    columns, is_legacy = _read_columns(node, exclude=(spec.buffer,))
    if is_legacy:
        n = len(next(iter(columns.values()))) if columns else 0
        for name, (dataset, key) in spec.legacy_strings.items():
            columns.setdefault(name, _legacy_strings(f, dataset, node.attrs, key, n))
        buffer = _find_node(f, spec.legacy_buffer) if spec.legacy_buffer else None
    else:
        buffer = node[spec.buffer] if spec.buffer and spec.buffer in node else None
    rows = _rows_from_columns(columns, spec.raw, spec.defaults, table)
    return _Table(rows=rows, is_legacy=is_legacy, buffer=buffer)


def _write_table(
    parent: h5py.File | h5py.Group,
    table: str,
    rows: np.ndarray,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    resizable: bool = False,
) -> h5py.Group | None:
    """Write one of the remaining tables.

    Args:
        parent: The file or group to write the table into.
        table: A key of `_TABLE_SPECS`.
        rows: Raw records (see `_TableSpec.raw`), with any carried unknown columns.
        legacy_format: If `True`, write the 2.x compound dataset at the table's
            legacy path (under `parent`'s file root for sub-tables), plus its
            separate string datasets. Unknown columns are dropped.
        compression: `None`, `"gzip"` or `"lzf"` for the row columns.
        resizable: If `True`, the table can grow (`maxshape=(None,)`).

    Returns:
        The group of columns, or `None` for the legacy layout.
    """
    spec = _TABLE_SPECS[table]
    name = spec.path.rpartition("/")[2]
    if not legacy_format:
        return _write_columns(
            parent, name, rows, compression=compression, resizable=resizable
        )
    f = parent.file
    legacy = _legacy_table_rows(rows, spec.legacy)
    kwargs = _compression_kwargs(compression, legacy.shape)
    if resizable:
        kwargs["maxshape"] = (None,)
    if spec.path == spec.legacy_path:
        parent.create_dataset(name, data=legacy, **kwargs)
    else:
        f.create_dataset(spec.legacy_path, data=legacy, **kwargs)
    for column, (dataset, _) in spec.legacy_strings.items():
        f.create_dataset(
            dataset, data=rows[column].tolist(), dtype=h5py.special_dtype(vlen=str)
        )
    return None


def _write_buffer(
    f: h5py.File,
    group: h5py.Group | None,
    table: str,
    data: np.ndarray,
    *,
    gzip: bool,
) -> None:
    """Write the flat `uint8` byte buffer of one of the remaining tables.

    Args:
        f: The open file.
        group: The table's group of columns (SLP 3.0), which holds the buffer, or
            `None` to write it at its 2.x path.
        table: A key of `_TABLE_SPECS`.
        data: The bytes, as a 1-D `uint8` array.
        gzip: If `True`, compress with gzip (level 1). Otherwise the buffer is
            chunked but unfiltered (its entries are already zlib-compressed).
            An empty buffer is always contiguous.
    """
    spec = _TABLE_SPECS[table]
    kwargs: dict = {}
    if len(data) > 0:
        kwargs["chunks"] = True
        if gzip:
            kwargs.update(compression="gzip", compression_opts=1)
    parent, name = (f, spec.legacy_buffer) if group is None else (group, spec.buffer)
    parent.create_dataset(name, data=data, dtype=np.uint8, **kwargs)


def _rows_owners(owners: list, index: np.ndarray) -> list | None:
    """Return `owners[i]` for each `i` of `index`, or `None` if one is out of range."""
    index = index.tolist()
    if any(not 0 <= i < len(owners) for i in index):
        return None
    return [owners[i] for i in index]


def _span_out_of_range(rows: np.ndarray, start: str, end: str, n: int) -> bool:
    """Whether any `[start, end)` range of `rows` isn't a range inside `[0, n)`."""
    s, e = rows[start], rows[end]
    return bool(np.any((s < 0) | (e < s) | (e > n)))


def _index_out_of_range(values: np.ndarray, n: int, *, optional: bool = True) -> bool:
    """Whether any index isn't a row of a table with `n` rows (or -1 if optional)."""
    return bool(np.any((values < (-1 if optional else 0)) | (values >= n)))


def _annotation_problems(
    table: str,
    rows: np.ndarray,
    *,
    n_videos: int,
    n_tracks: int,
    n_instances: int | None,
) -> list[str]:
    """Check the links from an annotation table to videos, tracks and instances.

    Args:
        table: The table name, for messages.
        rows: Raw records with `track` and `instance` columns, and optionally a
            `video` column.
        n_videos: Number of videos in the file.
        n_tracks: Number of tracks in the file.
        n_instances: Number of rows in `/instances`, or `None` to skip checking
            instance links (2.x tables store instance ids, mapped to rows later).

    Returns:
        A description of each problem found.
    """
    problems = []
    targets = [("track", n_tracks, "tracks"), ("instance", n_instances, "instances")]
    if "video" in rows.dtype.names:
        targets.insert(0, ("video", n_videos, "videos"))
    for column, n, what in targets:
        if n is not None and _index_out_of_range(rows[column], n):
            problems.append(f"/{table} {column} is not one of the {n} {what}")
    return problems


def _report_table_problems(
    labels_path: str, problems: list[str], *, strict: bool
) -> None:
    """Raise or warn about inconsistent tables.

    Args:
        labels_path: Path or URL of the file (used in the message).
        problems: A description of each problem found. Nothing happens if empty.
        strict: If `True` (SLP 3.0 tables), raise. Otherwise (2.x tables, which
            have always been read leniently) warn.

    Raises:
        ValueError: If `strict` and there are problems.
    """
    if not problems:
        return
    message = f"{labels_path} has inconsistent tables: {'; '.join(problems)}."
    if strict:
        raise ValueError(message)
    warnings.warn(f"{message} Reading it leniently, as an SLP 2.x file.", stacklevel=3)


def write_lfs(
    labels_path: str,
    labels: Labels,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
):
    """Write labeled frames, instances and points to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.
        legacy_format: If `True`, write the pose tables as 2.x compound datasets
            (after `_check_unlabeled_points`), with negative frames in
            `/negative_frames` and without metadata. Otherwise (the default) write
            them as groups of columns (SLP 3.0), with instance and frame metadata.
        compression: Compression for the pose tables: `None`, `"gzip"` or `"lzf"`.
    """
    # The rows of each table are gathered as tuples and turned into structured
    # arrays in one go. Points are kept as the instances' own points arrays.
    frame_rows, instance_rows = [], []
    lfs = list(labels.labeled_frames)
    all_instances: list[Instance | PredictedInstance] = []
    user_points: list = []
    pred_points: list = []
    n_user_points, n_pred_points = 0, 0
    inst_to_id = {}
    video_ids = _identity_index(labels.videos)
    track_ids = _identity_index(labels.tracks)
    for frame_id, lf in enumerate(lfs):
        instance_id_start = len(instance_rows)
        for inst in lf:
            inst_to_id[id(inst)] = len(instance_rows)
            skeleton_id = labels.skeletons.index(inst.skeleton)
            track = -1
            if inst.track:
                track = track_ids.get(id(inst.track))
                if track is None:
                    track = labels.tracks.index(inst.track)  # Raises if unknown.
            score = 0.0

            if type(inst) is Instance:
                instance_type = InstanceType.USER
                point_id_start = n_user_points
                user_points.append(inst)
                n_user_points += len(inst.points)
                point_id_end = n_user_points

            elif type(inst) is PredictedInstance:
                instance_type = InstanceType.PREDICTED
                score = inst.score
                point_id_start = n_pred_points
                pred_points.append(inst)
                n_pred_points += len(inst.points)
                point_id_end = n_pred_points

            else:
                raise ValueError(f"Unknown instance type: {type(inst)}")

            tracking_score = inst.tracking_score
            instance_rows.append(
                (
                    len(instance_rows),
                    int(instance_type),
                    frame_id,
                    skeleton_id,
                    track,
                    -1,  # from_predicted, linked below
                    score,
                    point_id_start,
                    point_id_end,
                    np.nan if tracking_score is None else tracking_score,
                )
            )
            all_instances.append(inst)

        video_idx = video_ids.get(id(lf.video))
        if video_idx is None:
            video_idx = labels.videos.index(lf.video)  # Raises if unknown.
        frame_rows.append(
            (
                frame_id,
                video_idx,
                lf.frame_idx,
                instance_id_start,
                len(instance_rows),
                lf.is_negative,
            )
        )

    instances = np.array(instance_rows, dtype=RAW_INSTANCE_DTYPE)
    frames = np.array(frame_rows, dtype=RAW_FRAME_DTYPE)
    # Link instances based on from_predicted field. The source instance may be
    # missing if predictions were removed from the labels, in which case the link
    # is dropped.
    for row, inst in enumerate(all_instances):
        if inst.from_predicted is not None:
            instances["from_predicted"][row] = inst_to_id.get(
                id(inst.from_predicted), -1
            )

    points = _raw_points_from_instances(user_points, is_predicted=False)
    predicted_points = _raw_points_from_instances(pred_points, is_predicted=True)

    carry = getattr(labels, "_carried_columns", None)
    instance_metadata, frame_metadata = {}, {}
    if legacy_format:
        _warn_dropped_carry(carry)
    else:
        points = _with_carried_columns(points, carry, "points", user_points)
        predicted_points = _with_carried_columns(
            predicted_points, carry, "pred_points", pred_points
        )
        instances = _with_carried_columns(instances, carry, "instances", all_instances)
        frames = _with_carried_columns(frames, carry, "frames", lfs)
        instance_metadata = _metadata_columns(
            all_instances, metadata_column_kinds(all_instances, "instances")
        )
        frame_metadata = _metadata_columns(lfs, metadata_column_kinds(lfs, "frames"))

    with h5py.File(labels_path, "a") as f:
        _write_pose_tables(
            f,
            points,
            predicted_points,
            instances,
            frames,
            videos=labels.videos,
            instance_metadata=instance_metadata,
            frame_metadata=frame_metadata,
            legacy_format=legacy_format,
            compression=compression,
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
    owners: dict[str, dict[int, object]],
) -> dict[int, FrameGroup]:
    """Reconstruct a session's `FrameGroup`s from the columnar `/session_data` tables.

    Args:
        fg_start: Start (inclusive) of this session's range into ``frame_groups``.
        fg_end: End (exclusive) of the range.
        session_data: The loaded ``/session_data`` arrays (see `_read_session_data`).
        camera_group: `CameraGroup` for resolving camera indices.
        labeled_frames: `LabeledFrame` list for resolving member instances.
        identities: Optional identity catalog for resolving ``identity_idx``.
        owners: Maps each session table name to a dictionary that the object built
            for each row is added to, by row: the `FrameGroup`, the
            `InstanceGroup`, or `(instance group, camera)` for a member (see
            `_owners_by_row`).

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
                owners["instance_groups"][ig_idx] = group
                # Recover the LabeledFrame-by-camera mapping from the same members.
                m_start = int(instance_groups[ig_idx]["member_start"])
                m_end = int(instance_groups[ig_idx]["member_end"])
                members = session_data["instance_group_members"]
                for m in range(m_start, m_end):
                    row = members[m]
                    camera = camera_group.cameras[int(row["camera"])]
                    labeled_frame_by_camera[camera] = labeled_frames[int(row["lf"])]
                    owners["instance_group_members"][m] = (group, camera)
            frame_group = FrameGroup(
                frame_idx=frame_idx,
                instance_groups=groups,
                labeled_frame_by_camera=labeled_frame_by_camera,
                metadata=_decode_meta_blob(session_data["frame_group_meta"], fg),
            )
            frame_group_by_frame_idx[frame_idx] = frame_group
            owners["frame_groups"][fg] = frame_group
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
    *,
    _owners: dict[str, dict[int, object]] | None = None,
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
        _owners: If given, the objects built for the rows of the session tables
            are added to it (see `_reconstruct_frame_groups_columnar`).

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
            owners=_owners
            if _owners is not None
            else {name: {} for name in _SESSION_TABLES},
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
    _carry: dict | None = None,
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
        _carry: If given, the raw records of the ``/session_data`` tables are
            added to it for carrying unknown columns through a later save (see
            `_column_carry`).

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
    owners: dict[str, dict[int, object]] = {name: {} for name in _SESSION_TABLES}
    session_objects = []
    for session in sessions:
        session_objects.append(
            make_session(
                session,
                videos,
                labeled_frames,
                identities=identities,
                session_data=session_data,
                _owners=owners,
            )
        )
    if _carry is not None and session_data is not None:
        for name in _SESSION_TABLES:
            rows = session_data[name]
            _carry[f"session_data/{name}"] = (
                rows,
                _TABLE_SPECS[f"session_data/{name}"].raw,
                partial(_owners_by_row, owners[name], len(rows)),
            )
    return session_objects


def _owners_by_row(owners: dict[int, object], n: int) -> list | None:
    """Return the owner of each of `n` rows, or `None` unless every row has one."""
    if len(owners) != n:
        return None
    return [owners[row] for row in range(n)]


_SESSION_TABLES = (
    "frame_groups",
    "instance_groups",
    "instance_group_members",
)


def _read_session_data(
    labels_path: str, *, _hdf5_file: h5py.File | None = None
) -> dict | None:
    """Load the columnar ``/session_data`` arrays (SLP 2.8+), or `None` if absent.

    Args:
        labels_path: A string path to the SLEAP labels file.
        _hdf5_file: An already-open `h5py.File` handle to read from.

    Returns:
        A dict of the member arrays keyed by dataset name, or `None` when the group
        is absent (legacy <= 2.7 files). ``frame_groups``, ``instance_groups`` and
        ``instance_group_members`` are normalised raw records (see `_TABLE_SPECS`),
        followed by any unknown columns, whatever their layout (empty if missing).
        The presence-guarded ``points_3d``, ``pred_points_3d`` and metadata blobs
        are `None` when absent.

    Raises:
        ValueError: If SLP 3.0 tables have ranges outside the tables they index or
            links to rows that don't exist. Older tables warn instead.
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
        # The 2.x struct tables may also be flat 2-D arrays with a ``field_names``
        # attribute, as written by the sleap-io.js 2.8 port (h5wasm can't create
        # compound datasets). The plain float points_3d matrices and the metadata
        # JSON blobs are read as they are.
        tables = {
            name: _read_table(f, f"session_data/{name}") for name in _SESSION_TABLES
        }
        data = {
            name: np.empty(0, dtype=_TABLE_SPECS[f"session_data/{name}"].raw)
            if table is None
            else table.rows
            for name, table in tables.items()
        }
        for name in (
            "points_3d",
            "pred_points_3d",
            "frame_group_meta",
            "instance_group_meta",
        ):
            data[name] = grp[name][()] if name in grp else None
        n_identities = len(f["identity/name"]) if "identity/name" in f else 0
        frames = f["frames"] if "frames" in f else None
        if isinstance(frames, h5py.Group):
            frames = frames["frame_idx"] if "frame_idx" in frames else None
        n_frames = len(frames) if frames is not None else None
    problems = _session_problems(data, n_identities=n_identities, n_frames=n_frames)
    _report_table_problems(
        labels_path,
        problems,
        strict=not any(t.is_legacy for t in tables.values() if t is not None),
    )
    return data


def _session_problems(
    data: dict, *, n_identities: int, n_frames: int | None
) -> list[str]:
    """Check the ranges and links of the ``/session_data`` tables.

    Args:
        data: The loaded ``/session_data`` arrays (see `_read_session_data`).
        n_identities: Number of identities in the catalog.
        n_frames: Number of rows in `/frames`, or `None` if the file has no frame
            table (only sessions were written), to skip checking frame links.

    Returns:
        A description of each problem found.
    """
    frame_groups = data["frame_groups"]
    groups = data["instance_groups"]
    members = data["instance_group_members"]
    problems = []
    if _span_out_of_range(frame_groups, "ig_start", "ig_end", len(groups)):
        problems.append(
            "/session_data/frame_groups ranges fall outside instance_groups "
            f"({len(groups)} rows)"
        )
    if _span_out_of_range(groups, "member_start", "member_end", len(members)):
        problems.append(
            "/session_data/instance_groups ranges fall outside "
            f"instance_group_members ({len(members)} rows)"
        )
    with_3d = groups[groups["pts3d_start"] >= 0]
    for predicted, name in ((False, "points_3d"), (True, "pred_points_3d")):
        points = data[name]
        n = 0 if points is None else len(points)
        if _span_out_of_range(
            with_3d[with_3d["pts3d_predicted"] == predicted],
            "pts3d_start",
            "pts3d_end",
            n,
        ):
            problems.append(
                f"/session_data/instance_groups 3D ranges fall outside {name} "
                f"({n} rows)"
            )
    if _index_out_of_range(groups["identity_idx"], n_identities):
        problems.append(
            "/session_data/instance_groups identity_idx is not one of the "
            f"{n_identities} identities"
        )
    if n_frames is not None and _index_out_of_range(
        members["lf"], n_frames, optional=False
    ):
        problems.append(
            f"/session_data/instance_group_members lf is not one of the {n_frames} "
            "frames"
        )
    if np.any((members["camera"] < 0) | (members["inst"] < 0)):
        problems.append("/session_data/instance_group_members has negative indices")
    return problems


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
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
):
    """Write `RecordingSession` metadata to a SLEAP labels file (SLP 2.8 columnar).

    Always creates the ``sessions_json`` dataset holding one *slim* JSON blob per
    session: only calibration + ``camcorder_to_video_idx_map`` + session-level
    metadata + an ``fg_start``/``fg_end`` half-open range into
    ``/session_data/frame_groups``.

    When any session has frame groups (and ``labeled_frames`` are supplied), the
    per-frame numeric payload is written columnar into the ``/session_data`` group:
    ``frame_groups`` / ``instance_groups`` / ``instance_group_members`` tables
    (groups of columns in SLP 3.0, compound datasets in 2.x), ``points_3d`` (N,3) /
    ``pred_points_3d`` (N,4 = xyz+score) chunked float matrices (appended one
    session at a time), and presence-guarded per-row
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
        legacy_format: If `True`, write the three tables as 2.x compound datasets.
            Otherwise (SLP 3.0) as groups of columns.
        compression: `None`, `"gzip"` or `"lzf"` for the table columns. The 3D
            points are always gzipped.
        carry: Unknown columns kept from loading, appended to a table if its rows
            are owned by the same frame groups, instance groups and cameras, in
            the same order (see `_with_carried_columns`).
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
    # The objects owning each row, to match carried unknown columns.
    owners: dict[str, list] = {name: [] for name in _SESSION_TABLES}

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
                            owners["instance_group_members"].append((ig, cam))
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
                        owners["instance_groups"].append(ig)
                        ig_meta.append(
                            json.dumps(ig.metadata, separators=(",", ":"))
                            if ig.metadata
                            else ""
                        )
                    ig_end = len(ig_rows)

                    fg_rows.append((int(frame_group.frame_idx), ig_start, ig_end))
                    owners["frame_groups"].append(frame_group)
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

        _create_json_rows_dataset(f, "sessions_json", sessions_json)

        # Columnar frame-group tables (only when there are frame groups).
        if fg_rows:
            grp = f.require_group("session_data")
            for name, rows in (
                ("frame_groups", fg_rows),
                ("instance_groups", ig_rows),
                ("instance_group_members", member_rows),
            ):
                table = f"session_data/{name}"
                rows = np.array(rows, dtype=_TABLE_SPECS[table].raw)
                if not legacy_format:
                    rows = _with_carried_columns(rows, carry, table, owners[name])
                _write_table(
                    grp,
                    table,
                    rows,
                    legacy_format=legacy_format,
                    compression=compression,
                    resizable=True,
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


def write_sessions_passthrough(
    labels_path: str,
    store,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
) -> bool:
    """Copy a lazy store's raw RecordingSession payload verbatim (SLP 2.8+).

    Writes the captured ``sessions_json`` bytes and the ``/session_data`` arrays as
    loaded, so lazy re-saves preserve frame groups + 3D points losslessly without
    materializing frames (the eager lazy path drops them). The three tables are
    written in the layout being saved, whatever layout they were loaded from, so a
    3.0 file never gets compound session tables.

    Args:
        labels_path: A string path to the SLEAP labels file being written.
        store: The `LazyDataStore` backing the lazy `Labels`.
        legacy_format: If `True`, write the tables as 2.x compound datasets
            (dropping unknown columns, with a warning). Otherwise (SLP 3.0) as
            groups of columns, with every loaded column.
        compression: `None`, `"gzip"` or `"lzf"` for the table columns.

    Returns:
        `True` if a passthrough was written; `False` if the store carried no
        captured session payload (caller should fall back to `write_sessions`).
    """
    raw = getattr(store, "sessions_json_raw", None)
    if raw is None:
        return False
    session_data = getattr(store, "session_data", None)
    with h5py.File(labels_path, "a") as f:
        _create_json_rows_dataset(f, "sessions_json", raw)
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
                    table = f"session_data/{name}"
                    unknown = _unknown_columns(arr, _TABLE_SPECS[table].raw)
                    if legacy_format and unknown:
                        _warn_dropped_columns([table], {table: unknown}, "legacy")
                    _write_table(
                        grp,
                        table,
                        arr,
                        legacy_format=legacy_format,
                        compression=compression,
                        resizable=True,
                    )
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

    _check_format_id(labels_path, f)

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
    # Read the raw pose tables (normalised and checked, see `_read_pose_tables`).
    tables = _read_pose_tables(labels_path, f, videos)
    frames_data = tables.frames
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
    # Raw records of the tables this lazy read turns into objects, to carry their
    # unknown columns through a later save.
    carry_tables: dict = {}
    identity_links = read_identity_links(labels_path, _hdf5_file=f, _carry=carry_tables)
    _remap_instance_owners(tables.instance_id_map, [identity_links], [])
    instance_identities = identity_links.get(OWNER_INSTANCE, {})
    mask_identity_map = identity_links.get(OWNER_MASK, {})
    centroid_identity_map = identity_links.get(OWNER_CENTROID, {})
    bbox_identity_map = identity_links.get(OWNER_BBOX, {})
    roi_identity_map = identity_links.get(OWNER_ROI, {})
    # Category links (SLP 2.7+): same owner-keyed split as identity links.
    category_links = read_category_links(labels_path, _hdf5_file=f, _carry=carry_tables)
    _remap_instance_owners(tables.instance_id_map, [category_links], [])
    instance_categories = category_links.get(OWNER_INSTANCE, {})
    mask_category_map = category_links.get(OWNER_MASK, {})
    centroid_category_map = category_links.get(OWNER_CENTROID, {})
    bbox_category_map = category_links.get(OWNER_BBOX, {})
    roi_category_map = category_links.get(OWNER_ROI, {})
    # Embeddings: per-instance embeddings ride on the lazy store; mask/centroid/
    # bbox/ROI embeddings attach to those (eagerly-read) annotations below.
    embeddings_by_owner = read_embeddings(labels_path, _hdf5_file=f)
    _remap_instance_owners(tables.instance_id_map, [embeddings_by_owner], [])
    instance_embeddings = embeddings_by_owner.get(OWNER_INSTANCE, {})
    mask_embedding_map = embeddings_by_owner.get(OWNER_MASK, {})
    centroid_embedding_map = embeddings_by_owner.get(OWNER_CENTROID, {})
    bbox_embedding_map = embeddings_by_owner.get(OWNER_BBOX, {})
    roi_embedding_map = embeddings_by_owner.get(OWNER_ROI, {})
    # Category embeddings (SLP 2.7+): parallel category_* datasets in /embeddings.
    category_embeddings_by_owner = read_category_embeddings(labels_path, _hdf5_file=f)
    _remap_instance_owners(tables.instance_id_map, [category_embeddings_by_owner], [])
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
        instances_data=tables.instances,
        pred_points_data=tables.pred_points,
        points_data=tables.points,
        instance_metadata=tables.instance_metadata,
        frame_metadata=tables.frame_metadata,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        format_id=format_id,
        source_path=str(labels_path),
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
    n_instances = len(tables.instances)
    roi_tuples = read_rois(
        labels_path,
        videos,
        tracks,
        _hdf5_file=f,
        _n_instances=n_instances,
        _carry=carry_tables,
    )
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
    mask_tuples = read_masks(
        labels_path,
        videos,
        tracks,
        _hdf5_file=f,
        _n_instances=n_instances,
        _carry=carry_tables,
    )
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
        _n_instances=n_instances,
        _carry=carry_tables,
    )
    lazy_store.annotation_carry = _column_carry(carry_tables)
    _remap_instance_owners(
        tables.instance_id_map,
        [],
        [
            ann
            for tuples in (
                roi_tuples,
                mask_tuples,
                bbox_tuples,
                centroid_tuples,
                li_tuples,
            )
            for ann, _, _ in tuples
        ],
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
    frame_keys = set(
        zip(frames_data["video"].tolist(), frames_data["frame_idx"].tolist())
    )

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
    _n_instances: int | None = None,
    _carry: dict | None = None,
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
        _n_instances: Number of rows in `/instances`, to check instance links when
            `instances` isn't given (lazy reads).
        _carry: If given, the table's raw records are added to it for carrying
            unknown columns through a later save (see `_column_carry`).

    Returns:
        A list of ``(roi, video_idx, frame_idx)`` tuples. ``video_idx`` and
        ``frame_idx`` are the routing context read from the file (``-1``
        means undistributable). Returns an empty list if no ROIs are stored.

    Raises:
        ValueError: If an SLP 3.0 table has a byte range outside its `wkb` buffer
            or a link to a video, track or instance that doesn't exist. Older
            tables warn instead.
    """
    import shapely

    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        table = _read_table(f, "rois")
        if table is None or len(table.rows) == 0:
            return []
        # Files before 3.0 without a geometry buffer have never been readable.
        if table.is_legacy and table.buffer is None:
            return []
        wkb_flat = (
            table.buffer[()] if table.buffer is not None else np.empty(0, np.uint8)
        )
    rows = table.rows
    problems = _annotation_problems(
        "rois",
        rows,
        n_videos=len(videos),
        n_tracks=len(tracks),
        n_instances=None
        if table.is_legacy
        else (len(instances) if instances is not None else _n_instances),
    )
    if _span_out_of_range(rows, "wkb_start", "wkb_end", len(wkb_flat)):
        problems.append(f"/rois wkb ranges fall outside the buffer ({len(wkb_flat)})")
    _report_table_problems(labels_path, problems, strict=not table.is_legacy)

    rois = []
    for row in rows.tolist():
        record = dict(zip(rows.dtype.names, row))
        geometry = shapely.from_wkb(
            bytes(wkb_flat[record["wkb_start"] : record["wkb_end"]])
        )

        video_idx = record["video"]
        video = videos[video_idx] if 0 <= video_idx < len(videos) else None
        track_idx = record["track"]
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None
        instance_idx = record["instance"]
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )
        tracking_score = record["tracking_score"]

        kwargs = dict(
            geometry=geometry,
            name=record["name"],
            category=record["category"],
            source=record["source"],
            video=video,
            track=track,
            tracking_score=None if np.isnan(tracking_score) else tracking_score,
            instance=instance,
        )

        if record["is_predicted"]:
            score = record["score"]
            roi = PredictedROI(score=0.0 if np.isnan(score) else score, **kwargs)
        else:
            roi = UserROI(**kwargs)

        # Store raw index for deferred resolution (lazy loading)
        roi._instance_idx = instance_idx
        rois.append((roi, video_idx, record["frame_idx"]))

    if _carry is not None:
        _carry["rois"] = (rows, _TABLE_SPECS["rois"].raw, lambda: [r for r, *_ in rois])
    return rois


def write_rois(
    labels_path: str,
    rois: list[ROI],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
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
        legacy_format: If `True`, write the 2.x layout: a compound `/rois` dataset
            with `/roi_wkb`, `/roi_names`, `/roi_categories` and `/roi_sources`.
            Otherwise (SLP 3.0) a `/rois` group of columns holding the `wkb`
            buffer.
        compression: `None`, `"gzip"` or `"lzf"` for the row columns. The `wkb`
            buffer is always gzipped.
        carry: Unknown columns kept from loading, appended to the rows if the ROIs
            are the ones loaded, in the same order (see `_with_carried_columns`).
    """
    if not rois:
        return

    if contexts is None:
        contexts = [(-1, -1)] * len(rois)

    import shapely

    roi_rows = []
    wkb_chunks = []
    wkb_offset = 0

    track_ids = _identity_index(tracks)
    instance_ids = _identity_index(instances) if instances is not None else None

    for i_roi, roi in enumerate(rois):
        wkb = shapely.to_wkb(roi.geometry)
        wkb_start = wkb_offset
        wkb_end = wkb_offset + len(wkb)
        wkb_chunks.append(np.frombuffer(wkb, dtype=np.uint8))
        wkb_offset = wkb_end

        video_idx, frame_idx = contexts[i_roi]
        track_idx = track_ids.get(id(roi.track), -1)

        instance_idx = roi._instance_idx  # Use stored index as default
        if instance_ids is not None and roi.instance is not None:
            instance_idx = instance_ids.get(id(roi.instance), instance_idx)

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
                instance_idx,
                is_predicted,
                score,
                tracking_score,
                wkb_start,
                wkb_end,
                roi.name,
                roi.category.name if roi.category else "",
                roi.source,
            )
        )

    rows = np.array(roi_rows, dtype=_TABLE_SPECS["rois"].raw)
    if not legacy_format:
        rows = _with_carried_columns(rows, carry, "rois", rois)
    # Every geometry serializes to >= 9 WKB bytes (even an empty polygon has a
    # header), so the buffer is never empty here.
    wkb_flat = np.concatenate(wkb_chunks)

    with h5py.File(labels_path, "a") as f:
        grp = _write_table(
            f, "rois", rows, legacy_format=legacy_format, compression=compression
        )
        # Gzip-compress the packed WKB geometry bytes (lossless, transparent on
        # read); ~2x smaller on polygon-heavy ROI sets.
        _write_buffer(f, grp, "rois", wkb_flat, gzip=True)


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

    track_ids = _identity_index(tracks)
    instance_ids = _identity_index(instances) if instances is not None else None

    for i, bbox in enumerate(bboxes):
        x1_arr[i] = bbox.x1
        y1_arr[i] = bbox.y1
        x2_arr[i] = bbox.x2
        y2_arr[i] = bbox.y2
        angle_arr[i] = bbox.angle

        video_arr[i], frame_idx_arr[i] = contexts[i]
        track_arr[i] = track_ids.get(id(bbox.track), -1)

        instance_idx = bbox._instance_idx
        if instance_ids is not None and bbox.instance is not None:
            instance_idx = instance_ids.get(id(bbox.instance), instance_idx)
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

    track_ids = _identity_index(tracks)
    instance_ids = _identity_index(instances) if instances is not None else None

    for i, centroid in enumerate(centroids):
        x_arr[i] = centroid.x
        y_arr[i] = centroid.y
        z_arr[i] = centroid.z if centroid.z is not None else float("nan")

        video_arr[i], frame_idx_arr[i] = contexts[i]
        track_arr[i] = track_ids.get(id(centroid.track), -1)

        instance_idx = centroid._instance_idx
        if instance_ids is not None and centroid.instance is not None:
            instance_idx = instance_ids.get(id(centroid.instance), instance_idx)
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


def _read_score_map_index(
    labels_path: str, f: h5py.File, table: str, n_owners: int
) -> tuple[_Table | None, np.ndarray]:
    """Read and check the score map index of masks or label images.

    Args:
        labels_path: Path or URL of the file (used in messages).
        f: The open file.
        table: `"masks/score_maps"` or `"label_images/score_maps"`.
        n_owners: Number of rows in the owning table (`/masks` or
            `/label_images`).

    Returns:
        A tuple `(table, data)`: the score map index table (or `None` if the file
        has none) and its buffer of zlib-compressed maps.

    Raises:
        ValueError: If an SLP 3.0 index row points outside the owning table or the
            `data` buffer. Older tables warn instead.
    """
    sm = _read_table(f, table)
    if sm is None:
        return None, np.empty(0, np.uint8)
    index = _TABLE_SPECS[table].raw.names[0]
    data = sm.buffer[()] if sm.buffer is not None else np.empty(0, np.uint8)
    rows = sm.rows
    problems = []
    if _index_out_of_range(rows[index], n_owners, optional=False):
        owner = table.partition("/")[0]
        problems.append(f"/{table} {index} is not one of the {n_owners} /{owner}")
    if _span_out_of_range(rows, "data_start", "data_end", len(data)):
        problems.append(f"/{table} ranges fall outside the buffer ({len(data)})")
    _report_table_problems(labels_path, problems, strict=not sm.is_legacy)
    return sm, data


def _read_score_maps(
    labels_path: str, f: h5py.File, table: str, n_owners: int
) -> tuple[_Table | None, dict[int, tuple[np.ndarray, tuple, tuple]]]:
    """Read the dense score maps of predicted masks or label images.

    Args:
        labels_path: Path or URL of the file (used in messages).
        f: The open file.
        table: `"masks/score_maps"` or `"label_images/score_maps"`.
        n_owners: Number of rows in the owning table (`/masks` or
            `/label_images`).

    Returns:
        A tuple `(table, maps)`: the score map index table (or `None` if the file
        has none), and a dictionary mapping each owning row to its score map, scale
        and offset.

    Raises:
        ValueError: If an SLP 3.0 index row points outside the owning table or the
            `data` buffer (see `_read_score_map_index`).
    """
    sm, data = _read_score_map_index(labels_path, f, table, n_owners)
    if sm is None:
        return None, {}
    index = _TABLE_SPECS[table].raw.names[0]
    rows = sm.rows
    maps = {}
    for row in rows.tolist():
        record = dict(zip(rows.dtype.names, row))
        score_map = np.frombuffer(
            zlib.decompress(data[record["data_start"] : record["data_end"]].tobytes()),
            dtype=np.float32,
        ).reshape(record["height"], record["width"])
        maps[record[index]] = (
            score_map,
            (record["scale_x"], record["scale_y"]),
            (record["offset_x"], record["offset_y"]),
        )
    return sm, maps


def read_masks(
    labels_path: str,
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    *,
    _hdf5_file: h5py.File | None = None,
    _n_instances: int | None = None,
    _carry: dict | None = None,
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
        _n_instances: Number of rows in `/instances`, to check instance links when
            `instances` isn't given (lazy reads).
        _carry: If given, the raw records of `/masks` and its score map index are
            added to it for carrying unknown columns through a later save (see
            `_column_carry`).

    Returns:
        A list of ``(mask, video_idx, frame_idx)`` tuples. ``video_idx`` and
        ``frame_idx`` are the routing context read from the file (``-1``
        means undistributable). Returns empty list if none stored.

    Raises:
        ValueError: If an SLP 3.0 table has a range outside its buffer or a link
            to a row that doesn't exist. Older tables warn instead.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as f:
        table = _read_table(f, "masks")
        if table is None or len(table.rows) == 0:
            return []
        # Files before 3.0 without an RLE buffer have never been readable.
        if table.is_legacy and table.buffer is None:
            return []
        rle_flat = (
            table.buffer[()] if table.buffer is not None else np.empty(0, np.uint8)
        )
        rows = table.rows
        problems = _annotation_problems(
            "masks",
            rows,
            n_videos=len(videos),
            n_tracks=len(tracks),
            n_instances=None
            if table.is_legacy
            else (len(instances) if instances is not None else _n_instances),
        )
        if _index_out_of_range(rows["from_predicted"], len(rows)):
            problems.append("/masks from_predicted is not a mask row")
        if _span_out_of_range(rows, "rle_start", "rle_end", len(rle_flat)):
            problems.append(
                f"/masks rle ranges fall outside the buffer ({len(rle_flat)})"
            )
        _report_table_problems(labels_path, problems, strict=not table.is_legacy)
        sm_table, score_maps = _read_score_maps(
            labels_path, f, "masks/score_maps", len(rows)
        )

    masks = []
    from_predicted_pairs = []
    for i, row in enumerate(rows.tolist()):
        record = dict(zip(rows.dtype.names, row))
        # Convert packed uint8 bytes back to uint32 array
        rle_counts = np.frombuffer(
            rle_flat[record["rle_start"] : record["rle_end"]].tobytes(),
            dtype=np.uint32,
        )

        video_idx = record["video"]
        track_idx = record["track"]
        track = tracks[track_idx] if 0 <= track_idx < len(tracks) else None
        instance_idx = record["instance"]
        instance = (
            instances[instance_idx]
            if instances is not None and 0 <= instance_idx < len(instances)
            else None
        )
        tracking_score = record["tracking_score"]

        kwargs = dict(
            rle_counts=rle_counts,
            height=record["height"],
            width=record["width"],
            name=record["name"],
            category=record["category"],
            source=record["source"],
            track=track,
            tracking_score=None if np.isnan(tracking_score) else tracking_score,
            instance=instance,
            scale=(record["scale_x"], record["scale_y"]),
            offset=(record["offset_x"], record["offset_y"]),
        )

        if record["is_predicted"]:
            score = record["score"]
            score_map, sm_scale, sm_offset = score_maps.get(
                i, (None, (1.0, 1.0), (0.0, 0.0))
            )
            mask = PredictedSegmentationMask(
                score=0.0 if np.isnan(score) else score,
                score_map=score_map,
                score_map_scale=sm_scale,
                score_map_offset=sm_offset,
                **kwargs,
            )
        else:
            mask = UserSegmentationMask(**kwargs)

        mask._instance_idx = instance_idx
        # Only user masks carry a from_predicted link; queue it for re-linking.
        if not record["is_predicted"] and record["from_predicted"] >= 0:
            from_predicted_pairs.append((i, record["from_predicted"]))
        masks.append((mask, video_idx, record["frame_idx"]))

    # Re-link from_predicted provenance now that every mask exists. The stored
    # value is a global index into this flat list (see write_masks); out-of-range
    # indices (corrupt/edited files) are skipped, leaving from_predicted as None.
    for i, fp_idx in from_predicted_pairs:
        if 0 <= fp_idx < len(masks):
            masks[i][0].from_predicted = masks[fp_idx][0]

    if _carry is not None:
        raw = _TABLE_SPECS["masks"].raw
        _carry["masks"] = (rows, raw, lambda: [m for m, *_ in masks])
        if sm_table is not None:
            _carry["masks/score_maps"] = (
                sm_table.rows,
                _TABLE_SPECS["masks/score_maps"].raw,
                lambda: _rows_owners([m for m, *_ in masks], sm_table.rows["mask_idx"]),
            )
    return masks


def _score_map_rows(owners: list) -> tuple[list[tuple], list[np.ndarray]]:
    """Build score map index rows for the predicted masks or label images.

    Args:
        owners: The masks or label images, in table order (`None` for those that
            can't have a score map).

    Returns:
        A tuple `(rows, chunks)`: one index row per owner with a score map, and the
        zlib-compressed `float32` bytes of each map.
    """
    rows = []
    chunks = []
    offset = 0
    for i, owner in enumerate(owners):
        score_map = getattr(owner, "score_map", None)
        if score_map is None:
            continue
        compressed = zlib.compress(score_map.astype(np.float32).tobytes())
        sm_bytes = np.frombuffer(compressed, dtype=np.uint8)
        sm_h, sm_w = score_map.shape[:2]
        rows.append(
            (
                i,
                offset,
                offset + len(sm_bytes),
                sm_h,
                sm_w,
                owner.score_map_scale[0],
                owner.score_map_scale[1],
                owner.score_map_offset[0],
                owner.score_map_offset[1],
            )
        )
        chunks.append(sm_bytes)
        offset += len(sm_bytes)
    return rows, chunks


def _write_score_maps(
    f: h5py.File,
    group: h5py.Group | None,
    table: str,
    rows: np.ndarray,
    data: np.ndarray,
    *,
    legacy_format: bool,
    compression: str | None,
) -> None:
    """Write a score map index table and its `data` buffer, if there are any.

    Args:
        f: The open file.
        group: The owning table's group (`/masks` or `/label_images`) in SLP 3.0,
            which holds the `score_maps/` group, or `None` for the 2.x layout.
        table: `"masks/score_maps"` or `"label_images/score_maps"`.
        rows: Raw index records (see `_TABLE_SPECS`). Nothing is written if empty.
        data: The concatenated compressed score maps.
        legacy_format: Whether to write the 2.x layout.
        compression: `None`, `"gzip"` or `"lzf"` for the index columns. The buffer
            is chunked but not filtered, since each map is zlib-compressed.
    """
    if len(rows) == 0:
        return
    sm_group = _write_table(
        group if group is not None else f,
        table,
        rows,
        legacy_format=legacy_format,
        compression=compression,
    )
    _write_buffer(f, sm_group, table, data, gzip=False)


def write_masks(
    labels_path: str,
    masks: list[SegmentationMask],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
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
        legacy_format: If `True`, write the 2.x layout: a compound `/masks`
            dataset with `/mask_rle`, the `/mask_names`, `/mask_categories` and
            `/mask_sources` string datasets and the `/mask_score_map_index` and
            `/mask_score_maps` datasets. Otherwise (SLP 3.0) a `/masks` group of
            columns holding the `rle` buffer and a `score_maps/` group.
        compression: `None`, `"gzip"` or `"lzf"` for the row columns. The `rle`
            buffer is always gzipped.
        carry: Unknown columns kept from loading, appended to the rows if the masks
            are the ones loaded, in the same order (see `_with_carried_columns`).
    """
    if not masks:
        return

    if contexts is None:
        contexts = [(-1, -1)] * len(masks)

    mask_rows = []
    rle_chunks = []
    rle_offset = 0

    # Map each mask object to its global index so a UserSegmentationMask's
    # ``from_predicted`` link can be persisted as an index into this flat list,
    # mirroring the instance ``from_predicted`` mechanism. The link resolves as
    # long as the source prediction is anywhere in this list (typically the same
    # frame); a source absent from the saved labels resolves to -1.
    mask_id_to_idx = {id(mask): i for i, mask in enumerate(masks)}
    track_ids = _identity_index(tracks)
    instance_ids = _identity_index(instances) if instances is not None else None

    for i_mask, mask in enumerate(masks):
        # Pack uint32 RLE counts as raw bytes (uint8)
        rle_bytes = mask.rle_counts.astype(np.uint32).tobytes()
        rle_uint8 = np.frombuffer(rle_bytes, dtype=np.uint8)
        rle_start = rle_offset
        rle_end = rle_offset + len(rle_uint8)
        rle_chunks.append(rle_uint8)
        rle_offset = rle_end

        video_idx, frame_idx = contexts[i_mask]
        track_idx = track_ids.get(id(mask.track), -1)

        instance_idx = mask._instance_idx  # Use stored index as default
        if instance_ids is not None and mask.instance is not None:
            instance_idx = instance_ids.get(id(mask.instance), instance_idx)

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
                is_predicted,
                score,
                tracking_score,
                rle_start,
                rle_end,
                mask.scale[0],
                mask.scale[1],
                mask.offset[0],
                mask.offset[1],
                mask.name,
                mask.category.name if mask.category else "",
                mask.source,
            )
        )

    rows = np.array(mask_rows, dtype=_TABLE_SPECS["masks"].raw)
    sm_rows, sm_chunks = _score_map_rows(
        [m if isinstance(m, PredictedSegmentationMask) else None for m in masks]
    )
    sm_rows = np.array(sm_rows, dtype=_TABLE_SPECS["masks/score_maps"].raw)
    if not legacy_format:
        rows = _with_carried_columns(rows, carry, "masks", masks)
        sm_rows = _with_carried_columns(
            sm_rows,
            carry,
            "masks/score_maps",
            [masks[i] for i in sm_rows["mask_idx"].tolist()],
        )
    rle_flat = (
        np.concatenate(rle_chunks) if rle_chunks else np.array([], dtype=np.uint8)
    )

    with h5py.File(labels_path, "a") as f:
        grp = _write_table(
            f, "masks", rows, legacy_format=legacy_format, compression=compression
        )
        # Gzip-compress the RLE bytes (lossless, transparent on read). RLE counts
        # are uint32 run-lengths whose high bytes are mostly zero, so they compress
        # ~8x+ at level 1. A degenerate empty-RLE mask (e.g. a zero-height mask)
        # keeps the contiguous, no-filter path.
        _write_buffer(f, grp, "masks", rle_flat, gzip=True)
        _write_score_maps(
            f,
            grp,
            "masks/score_maps",
            sm_rows,
            np.concatenate(sm_chunks) if sm_chunks else np.empty(0, np.uint8),
            legacy_format=legacy_format,
            compression=compression,
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
    _n_instances: int | None = None,
    _carry: dict | None = None,
) -> tuple[list[tuple[LabelImage, int, int]], "h5py.File | None"]:
    """Read label image annotations from a SLEAP labels file.

    Supports the SLP 3.0 `/label_images` group of columns (with its `data`,
    `objects/` and `score_maps/` members) and the 2.x compound tables, with pixel
    data in the legacy blob format (v1.8-v2.1) or the chunked format (v2.2+).
    Pixel data is loaded lazily when possible: the HDF5 file handle is kept open
    and each frame's data is decompressed on first ``.data`` access.

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
        _n_instances: Number of rows in `/instances`, to check instance links when
            `instances` isn't given (lazy reads).
        _carry: If given, the raw records of `/label_images`, its objects and its
            score map index are added to it for carrying unknown columns through a
            later save (see `_column_carry`).

    Returns:
        A tuple of ``(label_image_tuples, h5py_file)`` where
        ``label_image_tuples`` is a list of ``(li, video_idx, frame_idx)``
        tuples with routing context (``-1`` means undistributable), and
        ``h5py_file`` is the open HDF5 file handle that must be kept alive
        for lazy data access (or ``None`` if no label images were found).
        The caller is responsible for storing and eventually closing the
        file handle.

    Raises:
        ValueError: If an SLP 3.0 table has a range outside the table or buffer it
            indexes, or a link to a row that doesn't exist. Older tables warn
            instead.
    """
    cm = (
        nullcontext(_hdf5_file)
        if _hdf5_file is not None
        else h5py.File(labels_path, "r")
    )
    with cm as mf:
        table = _read_table(mf, "label_images")
        if table is None or len(table.rows) == 0:
            return [], None
        rows = table.rows
        # Files before 3.0 without pixel data have never been readable.
        if table.is_legacy and table.buffer is None:
            return [], None
        objects = _read_table(mf, "label_images/objects")
        obj_rows = (
            objects.rows
            if objects is not None
            else np.empty(0, dtype=_TABLE_SPECS["label_images/objects"].raw)
        )
        problems = _label_image_problems(
            rows,
            obj_rows,
            table.buffer,
            n_videos=len(videos),
            n_tracks=len(tracks),
            n_instances=None
            if table.is_legacy
            else (len(instances) if instances is not None else _n_instances),
        )
        _report_table_problems(labels_path, problems, strict=not table.is_legacy)
        sm_table, score_maps = _read_score_maps(
            labels_path, mf, "label_images/score_maps", len(rows)
        )
        pixel_path = table.buffer.name

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

    pixel_ds = f[pixel_path]
    is_chunked_format = pixel_ds.ndim == 3  # (T, H, W) = v2.2+ chunked

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

    obj_records = [dict(zip(obj_rows.dtype.names, r)) for r in obj_rows.tolist()]
    obj_owners: list = [None] * len(obj_records)
    label_images: list[LabelImage] = []
    for i, row in enumerate(rows.tolist()):
        record = dict(zip(rows.dtype.names, row))
        height = record["height"]
        width = record["width"]
        objects_start = record["objects_start"]

        # Build objects dict from objects table
        objects: dict[int, LabelImage.Info] = {}
        for obj_idx in range(objects_start, objects_start + record["n_objects"]):
            if not 0 <= obj_idx < len(obj_records):
                continue
            obj = obj_records[obj_idx]
            track_idx = obj["track"]
            instance_idx = obj["instance"]
            info = LabelImage.Info(
                track=tracks[track_idx] if 0 <= track_idx < len(tracks) else None,
                tracking_score=None
                if np.isnan(obj["tracking_score"])
                else obj["tracking_score"],
                category=obj["category"],
                name=obj["name"],
                instance=instances[instance_idx]
                if instances is not None and 0 <= instance_idx < len(instances)
                else None,
                score=None if np.isnan(obj["score"]) else obj["score"],
            )
            # Store raw index for deferred resolution (lazy loading)
            info._instance_idx = instance_idx
            objects[obj["label_id"]] = info
            obj_owners[obj_idx] = info

        # Construct with data=None for lazy loading
        kwargs = dict(
            data=None,
            objects=objects,
            source=record["source"],
            scale=(record["scale_x"], record["scale_y"]),
            offset=(record["offset_x"], record["offset_y"]),
        )

        if record["is_predicted"]:
            score = record["score"]
            score_map, sm_scale, sm_offset = score_maps.get(
                i, (None, (1.0, 1.0), (0.0, 0.0))
            )
            li = PredictedLabelImage(
                score=0.0 if np.isnan(score) else score,
                score_map=score_map,
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
            li._lazy_loader = _make_blob_loader(
                pixel_ds, record["data_start"], record["data_end"], height, width
            )
        li._height = height
        li._width = width

        label_images.append((li, record["video"], record["frame_idx"]))

    if _carry is not None:
        lis = [li for li, *_ in label_images]
        _carry["label_images"] = (rows, _TABLE_SPECS["label_images"].raw, lambda: lis)
        _carry["label_images/objects"] = (
            obj_rows,
            _TABLE_SPECS["label_images/objects"].raw,
            lambda: None if any(o is None for o in obj_owners) else obj_owners,
        )
        if sm_table is not None:
            _carry["label_images/score_maps"] = (
                sm_table.rows,
                _TABLE_SPECS["label_images/score_maps"].raw,
                lambda: _rows_owners(lis, sm_table.rows["li_idx"]),
            )
    return label_images, f


def _label_image_problems(
    rows: np.ndarray,
    obj_rows: np.ndarray,
    data: h5py.Dataset | None,
    *,
    n_videos: int,
    n_tracks: int,
    n_instances: int | None,
) -> list[str]:
    """Check the label image tables' ranges and links.

    Args:
        rows: Raw `/label_images` records.
        obj_rows: Raw `/label_images/objects` records.
        data: The pixel data: a `(T, H, W)` array with one frame per row, or a flat
            buffer of zlib-compressed frames indexed by `data_start`/`data_end`.
        n_videos: Number of videos in the file.
        n_tracks: Number of tracks in the file.
        n_instances: Number of rows in `/instances`, or `None` to skip checking
            instance links.

    Returns:
        A description of each problem found.
    """
    problems = []
    if _index_out_of_range(rows["video"], n_videos):
        problems.append(f"/label_images video is not one of the {n_videos} videos")
    start = rows["objects_start"]
    end = start + rows["n_objects"]
    if np.any((start < 0) | (end < start) | (end > len(obj_rows))):
        problems.append(
            f"/label_images object ranges fall outside /label_images/objects "
            f"({len(obj_rows)} rows)"
        )
    if data is None:
        problems.append("/label_images has no pixel data")
    elif data.ndim == 3:
        if data.shape[0] < len(rows):
            problems.append(
                f"/label_images has {len(rows)} rows but {data.shape[0]} frames of data"
            )
    elif _span_out_of_range(rows, "data_start", "data_end", data.shape[0]):
        problems.append(
            f"/label_images data ranges fall outside the buffer ({data.shape[0]})"
        )
    return problems + _annotation_problems(
        "label_images/objects",
        obj_rows,
        n_videos=n_videos,
        n_tracks=n_tracks,
        n_instances=n_instances,
    )


def _label_image_data_path(legacy_format: bool) -> str:
    """Return the path of the label image pixel data (see `_TABLE_SPECS`)."""
    spec = _TABLE_SPECS["label_images"]
    return spec.legacy_buffer if legacy_format else f"{spec.path}/{spec.buffer}"


def _label_image_rows(
    label_image: LabelImage,
    context: tuple[int, int],
    objects_start: int,
    data_range: tuple[int, int],
    object_ids: Callable[[LabelImage.Info], tuple[int, int]],
) -> tuple[tuple, list[tuple]]:
    """Build the raw `/label_images` row and object rows of one label image.

    Args:
        label_image: The label image.
        context: Its `(video_idx, frame_idx)` routing context.
        objects_start: Row of its first object in `/label_images/objects`.
        data_range: `(data_start, data_end)` of its pixels in a flat buffer, or
            `(0, 0)` for chunked `(T, H, W)` data.
        object_ids: Returns the `(track, instance)` indices of an object.

    Returns:
        A tuple `(row, object_rows)` in the raw layouts (see `_TABLE_SPECS`), with
        objects sorted by label ID.
    """
    object_rows = []
    for label_id in sorted(label_image.objects):
        info = label_image.objects[label_id]
        track_idx, instance_idx = object_ids(info)
        object_rows.append(
            (
                label_id,
                track_idx,
                instance_idx,
                info.score if info.score is not None else float("nan"),
                info.tracking_score
                if info.tracking_score is not None
                else float("nan"),
                info.name,
                info.category,
            )
        )
    is_predicted = isinstance(label_image, PredictedLabelImage)
    row = (
        *context,
        label_image.height,
        label_image.width,
        len(object_rows),
        objects_start,
        *data_range,
        is_predicted,
        label_image.score if is_predicted else float("nan"),
        label_image.scale[0],
        label_image.scale[1],
        label_image.offset[0],
        label_image.offset[1],
        label_image.source,
    )
    return row, object_rows


def _write_label_image_tables(
    f: h5py.File,
    rows: list[tuple] | np.ndarray,
    object_rows: list[tuple] | np.ndarray,
    score_map_rows: list[tuple] | np.ndarray,
    score_map_data: np.ndarray,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
    label_images: list[LabelImage] | None = None,
) -> None:
    """Write the label image tables, next to pixel data already written.

    Args:
        f: The open file, holding the pixel data at `_label_image_data_path`.
        rows: Raw `/label_images` rows (see `_TABLE_SPECS`).
        object_rows: Raw `/label_images/objects` rows.
        score_map_rows: Raw `/label_images/score_maps` rows.
        score_map_data: The concatenated compressed score maps.
        legacy_format: If `True`, write the 2.x layout: compound `/label_images`,
            `/label_image_objects` and `/label_image_score_map_index` datasets with
            separate string datasets. Otherwise (SLP 3.0) the `/label_images` group
            of columns with `objects/` and `score_maps/` groups.
        compression: `None`, `"gzip"` or `"lzf"` for the row columns.
        carry: Unknown columns kept from loading (see `_with_carried_columns`).
        label_images: The label images written, in order, to match `carry`.
    """
    specs = {
        name: _TABLE_SPECS[name]
        for name in ("label_images", "label_images/objects", "label_images/score_maps")
    }
    rows = np.asarray(rows, dtype=specs["label_images"].raw)
    object_rows = np.asarray(object_rows, dtype=specs["label_images/objects"].raw)
    score_map_rows = np.asarray(
        score_map_rows, dtype=specs["label_images/score_maps"].raw
    )
    if carry is not None and not legacy_format:
        objects = [
            li.objects[label_id]
            for li in label_images
            for label_id in sorted(li.objects)
        ]
        rows = _with_carried_columns(rows, carry, "label_images", label_images)
        object_rows = _with_carried_columns(
            object_rows, carry, "label_images/objects", objects
        )
        score_map_rows = _with_carried_columns(
            score_map_rows,
            carry,
            "label_images/score_maps",
            [label_images[i] for i in score_map_rows["li_idx"].tolist()],
        )
    grp = _write_table(
        f, "label_images", rows, legacy_format=legacy_format, compression=compression
    )
    _write_table(
        grp if grp is not None else f,
        "label_images/objects",
        object_rows,
        legacy_format=legacy_format,
        compression=compression,
    )
    _write_score_maps(
        f,
        grp,
        "label_images/score_maps",
        score_map_rows,
        score_map_data,
        legacy_format=legacy_format,
        compression=compression,
    )


def write_label_images(
    labels_path: str,
    label_images: list[LabelImage],
    videos: list[Video],
    tracks: list[Track],
    instances: list[Instance | PredictedInstance] | None = None,
    contexts: list[tuple[int, int]] | None = None,
    *,
    legacy_format: bool = False,
    compression: str | None = None,
    carry: "_ColumnCarry | None" = None,
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
        legacy_format: If `True`, write the 2.x layout (`/label_image_data` and
            compound tables, see `_write_label_image_tables`). Otherwise (SLP 3.0)
            the `/label_images` group, with the pixels in `/label_images/data`.
        compression: `None`, `"gzip"` or `"lzf"` for the row columns. Pixel data
            is always gzipped.
        carry: Unknown columns kept from loading, appended to the rows if the
            label images are the ones loaded, in the same order (see
            `_with_carried_columns`).
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

    # For blob format fallback
    data_chunks: list[bytes] = []
    data_offset = 0

    track_ids = _identity_index(tracks)
    instance_ids = _identity_index(instances) if instances is not None else None

    def object_ids(info: LabelImage.Info) -> tuple[int, int]:
        instance_idx = info._instance_idx  # Use stored index as default
        if instance_ids is not None and info.instance is not None:
            instance_idx = instance_ids.get(id(info.instance), instance_idx)
        return track_ids.get(id(info.track), -1), instance_idx

    data_path = _label_image_data_path(legacy_format)
    with h5py.File(labels_path, "a") as f:
        # Create pixel data dataset
        if use_chunked:
            frame_h, frame_w = shapes.pop()
            n_frames = len(label_images)
            pixel_dset = f.create_dataset(
                data_path,
                shape=(n_frames, frame_h, frame_w),
                chunks=(1, frame_h, frame_w),
                dtype=np.int32,
                compression="gzip",
                compression_opts=1,
            )

        for i, li in enumerate(label_images):
            if use_chunked:
                # Write pixel data directly via write_direct_chunk (43x faster)
                compressed = zlib.compress(li.data.astype(np.int32).tobytes(), level=1)
                pixel_dset.id.write_direct_chunk((i, 0, 0), compressed)
                data_range = (0, 0)
            else:
                # Blob format: accumulate compressed chunks
                compressed = zlib.compress(li.data.astype(np.int32).tobytes())
                data_range = (data_offset, data_offset + len(compressed))
                data_chunks.append(compressed)
                data_offset += len(compressed)

            row, objects = _label_image_rows(
                li, contexts[i], len(obj_rows), data_range, object_ids
            )
            li_rows.append(row)
            obj_rows.extend(objects)

        # Write blob data if using legacy format
        if not use_chunked:
            data_flat = np.frombuffer(b"".join(data_chunks), dtype=np.uint8)
            f.create_dataset(
                data_path,
                data=data_flat,
                dtype=np.uint8,
                **({"chunks": True} if len(data_flat) > 0 else {}),
            )

        # Store score maps for PredictedLabelImage if any exist
        sm_rows, sm_chunks = _score_map_rows(
            [li if isinstance(li, PredictedLabelImage) else None for li in label_images]
        )
        _write_label_image_tables(
            f,
            li_rows,
            obj_rows,
            sm_rows,
            np.concatenate(sm_chunks) if sm_chunks else np.empty(0, np.uint8),
            legacy_format=legacy_format,
            compression=compression,
            carry=carry,
            label_images=label_images,
        )


def _write_metadata_standalone(
    labels_path: str,
    format_id: float = SLP_FORMAT_ID,
    skeletons: list[Skeleton] | None = None,
    provenance: dict | None = None,
    videos: list[Video] | None = None,
) -> None:
    """Write minimal metadata group to an SLP file.

    This is used by ``LabelImageWriter`` to write format info without requiring
    a full ``Labels`` object.

    Args:
        labels_path: Path to the SLP file.
        format_id: Format version identifier. Defaults to ``SLP_FORMAT_ID`` (3.0),
            matching the empty columnar pose tables written here.
        skeletons: Optional list of skeletons to serialize.
        provenance: Optional provenance dict.
        videos: Optional list of videos; if any carries a virtual crop the
            ``format_id`` is raised to at least 2.3 (matching ``write_metadata``).
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
            _write_columns(f, "points", np.empty(0, dtype=RAW_POINT_DTYPE))
        if "pred_points" not in f:
            _write_columns(f, "pred_points", np.empty(0, dtype=RAW_PRED_POINT_DTYPE))
        if "instances" not in f:
            _write_columns(
                f,
                "instances",
                _without_field(np.empty(0, RAW_INSTANCE_DTYPE), "instance_id"),
            )
        if "frames" not in f:
            _write_columns(
                f, "frames", _without_field(np.empty(0, RAW_FRAME_DTYPE), "frame_id")
            )


class LabelImageWriter:
    """Streaming writer for label image annotations to SLP files.

    Writes label images one at a time (or in batches) to an HDF5/SLP file
    without holding all pixel data in memory simultaneously. Uses the chunked
    ``(T, H, W)`` int32 format with ``write_direct_chunk`` for maximum
    throughput. The file is SLP 3.0: the pixels go to ``/label_images/data`` and
    the tables are written as columns by ``finalize()``. The file holds no
    instances, so label image objects link to none.

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

        # Accumulated raw table rows (kept in memory, ~100 bytes/frame + ~50
        # bytes/object), written as SLP 3.0 columns by `finalize`
        self._li_rows: list[tuple] = []
        self._obj_rows: list[tuple] = []

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
            _label_image_data_path(legacy_format=False),
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

        # Build the rows (auto-collect new tracks)
        _track_set = set(self.tracks)

        def object_ids(info: LabelImage.Info) -> tuple[int, int]:
            if info.track is not None and info.track not in _track_set:
                self.tracks.append(info.track)
                _track_set.add(info.track)
            track_idx = (
                self.tracks.index(info.track) if info.track in self.tracks else -1
            )
            # The file holds no instances, so objects link to none.
            return track_idx, -1

        row, objects = _label_image_rows(
            label_image,
            (video_idx, frame_idx),
            len(self._obj_rows),
            (0, 0),  # data range (unused for chunked)
            object_ids,
        )
        self._li_rows.append(row)
        self._obj_rows.extend(objects)

        # Score map handling for PredictedLabelImage
        if (
            isinstance(label_image, PredictedLabelImage)
            and label_image.score_map is not None
        ):
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

        # Write the label image tables (SLP 3.0 columns) next to the pixel data
        f = self._file
        _write_label_image_tables(
            f,
            self._li_rows,
            self._obj_rows,
            self._sm_indices,
            np.concatenate(self._sm_chunks)
            if self._sm_chunks
            else np.empty(0, np.uint8),
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
    back to decompress + recompress for legacy blob-format sources. Sources may
    use any SLP layout and are checked as reading them would; the merged file is
    SLP 3.0. It holds no instances, so label image objects link to none.

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
            provided, or if a source contains no label images, or if a
            source's format is newer than this sleap-io can read, or if an SLP
            3.0 source's label image tables are inconsistent.
    """
    if not source_paths:
        raise ValueError("At least one source path is required.")

    source_paths = [str(p) for p in source_paths]
    dest_path = str(dest_path)

    # --- Phase 1: Open all sources, read index tables, validate dimensions ---
    source_files: list[h5py.File] = []
    source_tables: list[_Table] = []
    source_obj_rows: list[np.ndarray] = []
    source_sm_tables: list[tuple[_Table | None, np.ndarray]] = []
    source_videos: list[list[Video]] = []
    source_tracks: list[list[Track]] = []

    all_shapes: set[tuple[int, int]] = set()

    try:
        for src_path in source_paths:
            f = h5py.File(src_path, "r")
            source_files.append(f)
            _check_format_id(src_path, f)

            # Read the index table (any layout) and its pixel data
            table = _read_table(f, "label_images")
            if table is None or table.buffer is None or len(table.rows) == 0:
                raise ValueError(f"Source file has no label images: {src_path}")
            source_tables.append(table)

            # Collect frame dimensions
            all_shapes.update(
                zip(table.rows["height"].tolist(), table.rows["width"].tolist())
            )

            # Read objects table
            objects = _read_table(f, "label_images/objects")
            obj_rows = (
                objects.rows
                if objects is not None
                else np.empty(0, dtype=_TABLE_SPECS["label_images/objects"].raw)
            )
            source_obj_rows.append(obj_rows)

            # Read videos and tracks from source
            source_videos.append(read_videos(src_path, open_backend=False))
            source_tracks.append(read_tracks(src_path))

            # Check the source's tables as reading it would
            problems = _label_image_problems(
                table.rows,
                obj_rows,
                table.buffer,
                n_videos=len(source_videos[-1]),
                n_tracks=len(source_tracks[-1]),
                n_instances=None,
            )
            _report_table_problems(src_path, problems, strict=not table.is_legacy)

            # Score map data
            source_sm_tables.append(
                _read_score_map_index(
                    src_path, f, "label_images/score_maps", len(table.rows)
                )
            )

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
        total_frames = sum(len(t.rows) for t in source_tables)

        dest_li_rows: list[tuple] = []
        dest_obj_rows: list[tuple] = []
        dest_sm_rows: list[tuple] = []
        dest_sm_chunks: list[np.ndarray] = []
        dest_sm_offset = 0

        with h5py.File(dest_path, "w") as dest_f:
            dest_pixel_dset = dest_f.create_dataset(
                _label_image_data_path(legacy_format=False),
                shape=(total_frames, frame_h, frame_w),
                chunks=(1, frame_h, frame_w),
                dtype=np.int32,
                compression="gzip",
                compression_opts=1,
            )

            dest_frame_idx = 0

            for src_idx, (table, obj_rows, (sm_table, sm_data)) in enumerate(
                zip(source_tables, source_obj_rows, source_sm_tables)
            ):
                pixel_ds = table.buffer
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

                # Score maps of this source, by label image row
                sm_rows = (
                    sm_table.rows
                    if sm_table is not None
                    else np.empty(0, _TABLE_SPECS["label_images/score_maps"].raw)
                )
                sm_by_row = {
                    rec["li_idx"]: rec
                    for rec in (
                        dict(zip(sm_rows.dtype.names, r)) for r in sm_rows.tolist()
                    )
                }

                obj_records = [
                    dict(zip(obj_rows.dtype.names, r)) for r in obj_rows.tolist()
                ]
                li_rows = table.rows
                for local_i, row in enumerate(li_rows.tolist()):
                    record = dict(zip(li_rows.dtype.names, row))
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
                        raw = zlib.decompress(
                            pixel_ds[
                                record["data_start"] : record["data_end"]
                            ].tobytes()
                        )
                        arr = np.frombuffer(raw, dtype=np.int32).reshape(
                            record["height"], record["width"]
                        )
                        compressed = zlib.compress(arr.tobytes(), level=1)
                        dest_pixel_dset.id.write_direct_chunk(
                            (dest_frame_idx, 0, 0), compressed
                        )

                    # Remap object rows
                    objects_start = len(dest_obj_rows)
                    start = record["objects_start"]
                    for obj in obj_records[start : start + record["n_objects"]]:
                        dest_obj_rows.append(
                            (
                                obj["label_id"],
                                track_remap.get(obj["track"], -1),
                                -1,  # The merged file holds no instances.
                                obj["score"],
                                obj["tracking_score"],
                                obj["name"],
                                obj["category"],
                            )
                        )

                    dest_li_rows.append(
                        (
                            video_remap.get(record["video"], -1),
                            record["frame_idx"],
                            record["height"],
                            record["width"],
                            len(dest_obj_rows) - objects_start,
                            objects_start,
                            0,  # data_start (unused for chunked)
                            0,  # data_end (unused for chunked)
                            record["is_predicted"],
                            record["score"],
                            record["scale_x"],
                            record["scale_y"],
                            record["offset_x"],
                            record["offset_y"],
                            record["source"],
                        )
                    )

                    # Copy score maps if present (still compressed)
                    sm = sm_by_row.get(local_i)
                    if sm is not None:
                        sm_bytes = sm_data[sm["data_start"] : sm["data_end"]]
                        dest_sm_rows.append(
                            (
                                dest_frame_idx,
                                dest_sm_offset,
                                dest_sm_offset + len(sm_bytes),
                                sm["height"],
                                sm["width"],
                                sm["scale_x"],
                                sm["scale_y"],
                                sm["offset_x"],
                                sm["offset_y"],
                            )
                        )
                        dest_sm_chunks.append(sm_bytes)
                        dest_sm_offset += len(sm_bytes)

                    dest_frame_idx += 1

            # Write the label image tables (SLP 3.0 columns)
            _write_label_image_tables(
                dest_f,
                dest_li_rows,
                dest_obj_rows,
                dest_sm_rows,
                np.concatenate(dest_sm_chunks)
                if dest_sm_chunks
                else np.empty(0, np.uint8),
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

    Raises:
        ValueError: If the file's format is newer than this sleap-io can read (see
            `SLP_FORMAT_ID_LIMIT`).
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
    _check_format_id(labels_path, f)
    tracks = read_tracks(labels_path, _hdf5_file=f)
    videos = read_videos(
        labels_path,
        open_backend=open_videos,
        _hdf5_file=f,
        _url_headers=_url_headers,
        _url_stream_mode=_url_stream_mode,
    )
    skeletons = read_skeletons(labels_path, _hdf5_file=f)
    tables = _read_pose_tables(labels_path, f, videos)
    point_owners: list = []
    # One entry per /instances row (None for rows of an unknown type), so links to
    # instances index it by row.
    instances = _instances_from_rows(
        tables.instances,
        skeletons,
        tracks,
        tables.points,
        tables.pred_points,
        tables.instance_metadata,
        point_owners=point_owners,
    )
    suggestions = read_suggestions(labels_path, videos, _hdf5_file=f)
    metadata = read_metadata(labels_path, _hdf5_file=f)
    provenance = read_provenance(labels_path, metadata, _hdf5_file=f)

    frames = tables.frames
    frame_records = (
        _metadata_records(tables.frame_metadata, len(frames))
        if tables.frame_metadata
        else None
    )
    labeled_frames = []
    for row, (video_idx, frame_idx, start, end, is_negative) in enumerate(
        zip(
            frames["video"].tolist(),
            frames["frame_idx"].tolist(),
            frames["instance_id_start"].tolist(),
            frames["instance_id_end"].tolist(),
            frames["is_negative"].tolist(),
        )
    ):
        labeled_frames.append(
            LabeledFrame(
                video=videos[video_idx],
                frame_idx=frame_idx,
                instances=[inst for inst in instances[start:end] if inst is not None],
                is_negative=is_negative,
                metadata=frame_records[row] if frame_records is not None else None,
            )
        )
    # Every table's raw records, to carry unknown columns through a later save. The
    # frames owning /frames rows are kept before annotation-only frames are added.
    pose_frames = list(labeled_frames)
    carry_tables = {
        "points": (
            tables.points,
            RAW_POINT_DTYPE,
            lambda: _point_row_owners(point_owners, Instance, len(tables.points)),
        ),
        "pred_points": (
            tables.pred_points,
            RAW_PRED_POINT_DTYPE,
            lambda: _point_row_owners(
                point_owners, PredictedInstance, len(tables.pred_points)
            ),
        ),
        "instances": (
            tables.instances,
            RAW_INSTANCE_DTYPE,
            lambda: None if any(i is None for i in instances) else instances,
        ),
        "frames": (tables.frames, RAW_FRAME_DTYPE, lambda: pose_frames),
    }

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
    identity_links = read_identity_links(labels_path, _hdf5_file=f, _carry=carry_tables)
    _remap_instance_owners(tables.instance_id_map, [identity_links], [])
    instance_identity_map = identity_links.get(OWNER_INSTANCE, {})
    mask_identity_map = identity_links.get(OWNER_MASK, {})
    centroid_identity_map = identity_links.get(OWNER_CENTROID, {})
    bbox_identity_map = identity_links.get(OWNER_BBOX, {})
    roi_identity_map = identity_links.get(OWNER_ROI, {})
    for inst_id, (identity_idx, identity_score) in instance_identity_map.items():
        if (
            0 <= inst_id < len(instances)
            and instances[inst_id] is not None
            and 0 <= identity_idx < len(identities)
        ):
            instances[inst_id].identity = identities[identity_idx]
            instances[inst_id].identity_score = identity_score

    # Attach category links (SLP format 2.7+). Self-contained mirror of the identity
    # links above; absent /categories group -> empty mappings and a no-op.
    category_links = read_category_links(labels_path, _hdf5_file=f, _carry=carry_tables)
    _remap_instance_owners(tables.instance_id_map, [category_links], [])
    instance_category_map = category_links.get(OWNER_INSTANCE, {})
    mask_category_map = category_links.get(OWNER_MASK, {})
    centroid_category_map = category_links.get(OWNER_CENTROID, {})
    bbox_category_map = category_links.get(OWNER_BBOX, {})
    roi_category_map = category_links.get(OWNER_ROI, {})
    for inst_id, (category_idx, category_score) in instance_category_map.items():
        if (
            0 <= inst_id < len(instances)
            and instances[inst_id] is not None
            and 0 <= category_idx < len(categories)
        ):
            instances[inst_id].category = categories[category_idx]
            instances[inst_id].category_score = category_score

    # Attach appearance / re-ID embeddings (SLP format 2.5+). Additive: absent
    # /embeddings group -> empty mappings and a no-op. Mask/centroid/bbox/ROI
    # embeddings are attached after their respective read_* calls.
    embeddings_by_owner = read_embeddings(labels_path, _hdf5_file=f)
    _remap_instance_owners(tables.instance_id_map, [embeddings_by_owner], [])
    for inst_id, emb in embeddings_by_owner.get(OWNER_INSTANCE, {}).items():
        if 0 <= inst_id < len(instances) and instances[inst_id] is not None:
            instances[inst_id].identity_embedding = emb
    mask_embedding_map = embeddings_by_owner.get(OWNER_MASK, {})
    centroid_embedding_map = embeddings_by_owner.get(OWNER_CENTROID, {})
    bbox_embedding_map = embeddings_by_owner.get(OWNER_BBOX, {})
    roi_embedding_map = embeddings_by_owner.get(OWNER_ROI, {})

    # Attach category embeddings (SLP format 2.7+): parallel category_* datasets in
    # /embeddings. Absent -> empty mappings and a no-op.
    category_embeddings_by_owner = read_category_embeddings(labels_path, _hdf5_file=f)
    _remap_instance_owners(tables.instance_id_map, [category_embeddings_by_owner], [])
    for inst_id, emb in category_embeddings_by_owner.get(OWNER_INSTANCE, {}).items():
        if 0 <= inst_id < len(instances) and instances[inst_id] is not None:
            instances[inst_id].category_embedding = emb
    mask_category_embedding_map = category_embeddings_by_owner.get(OWNER_MASK, {})
    centroid_category_embedding_map = category_embeddings_by_owner.get(
        OWNER_CENTROID, {}
    )
    bbox_category_embedding_map = category_embeddings_by_owner.get(OWNER_BBOX, {})
    roi_category_embedding_map = category_embeddings_by_owner.get(OWNER_ROI, {})

    sessions = read_sessions(
        labels_path,
        videos,
        labeled_frames,
        identities=identities,
        _hdf5_file=f,
        _carry=carry_tables,
    )
    roi_tuples = read_rois(
        labels_path, videos, tracks, instances, _hdf5_file=f, _carry=carry_tables
    )
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
    mask_tuples = read_masks(
        labels_path, videos, tracks, instances, _hdf5_file=f, _carry=carry_tables
    )
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
        _carry=carry_tables,
    )
    _remap_instance_owners(
        tables.instance_id_map,
        [],
        [
            ann
            for tuples in (
                roi_tuples,
                mask_tuples,
                bbox_tuples,
                centroid_tuples,
                li_tuples,
            )
            for ann, _, _ in tuples
        ],
        instances,
    )

    # Attach annotations to their corresponding LabeledFrames
    frame_lookup: dict[tuple[int, int], LabeledFrame] = {}
    video_ids = _identity_index(videos)
    for lf in labeled_frames:
        vid_idx = video_ids.get(id(lf.video), -1)
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
    labels._carried_columns = _column_carry(carry_tables)

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
    legacy_format: bool = False,
    compression: str | None = None,
    unlabeled_as: str = "error",
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
        legacy_format: If `True`, write the 2.x layout. See `write_labels`.
        compression: Compression for the pose tables. See `write_labels`.
        unlabeled_as: How the legacy layout handles unlabeled points. See
            `write_labels`.

    Raises:
        ValueError: If labels is not lazy.
    """
    if not labels.is_lazy:
        raise ValueError("_write_labels_lazy requires lazy Labels")

    lazy_store = labels._lazy_store

    if legacy_format:
        # Check before touching the output file so a refused save leaves it intact.
        n_unlabeled = sum(
            int(np.count_nonzero(~(rows["labeled"] | rows["visible"])))
            for rows in (lazy_store.points_data, lazy_store.pred_points_data)
        )
        _refuse_unlabeled_points(n_unlabeled, unlabeled_as)
        unknown = {
            name: _unknown_columns(rows, base)
            for name, rows, base in (
                ("points", lazy_store.points_data, RAW_POINT_DTYPE),
                ("pred_points", lazy_store.pred_points_data, RAW_PRED_POINT_DTYPE),
                ("instances", lazy_store.instances_data, RAW_INSTANCE_DTYPE),
                ("frames", lazy_store.frames_data, RAW_FRAME_DTYPE),
            )
        }
        _warn_dropped_columns(
            [n for n, cols in unknown.items() if cols], unknown, "legacy"
        )
        _warn_dropped_carry(lazy_store.annotation_carry)
        if lazy_store.instance_metadata or lazy_store.frame_metadata:
            warnings.warn(
                "Instance and frame metadata can't be stored in the legacy SLP "
                "layout and was not saved.",
                stacklevel=3,
            )

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
    # Unknown columns of the annotation and link tables, kept from loading.
    _start_carry(lazy_store.annotation_carry)
    table_kwargs = dict(
        legacy_format=legacy_format,
        compression=compression,
        carry=lazy_store.annotation_carry,
    )
    _write_identity_link_rows(labels_path, identity_rows, **table_kwargs)
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
    _write_category_link_rows(labels_path, category_rows, **table_kwargs)
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
        did_passthrough = write_sessions_passthrough(
            labels_path,
            lazy_store,
            legacy_format=legacy_format,
            compression=compression,
        )
    if not did_passthrough:
        if getattr(lazy_store, "session_data", None) is not None:
            warnings.warn(
                "Videos changed since lazy load; RecordingSession frame groups "
                "(including 3D points) are not preserved on this save. Call "
                "`labels.materialize()` before saving to retain them."
            )
        write_sessions(
            labels_path,
            labels.sessions,
            labels.videos,
            [],
            legacy_format=legacy_format,
            compression=compression,
        )
    write_metadata(labels_path, labels, legacy_format=legacy_format)

    # Write raw arrays directly from lazy store (fast path). The tables are in the
    # same row order as loaded, so unknown columns and metadata are carried through.
    with h5py.File(labels_path, "a") as f:
        _write_pose_tables(
            f,
            lazy_store.points_data,
            lazy_store.pred_points_data,
            lazy_store.instances_data,
            lazy_store.frames_data,
            videos=labels.videos,
            instance_metadata=lazy_store.instance_metadata,
            frame_metadata=lazy_store.frame_metadata,
            legacy_format=legacy_format,
            compression=compression,
        )

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
        video_ids = _identity_index(videos)
        for ann in undistributed:
            anns.append(ann)
            ctxs.append((video_ids.get(id(getattr(ann, "video", None)), -1), -1))
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
    write_rois(
        labels_path,
        all_rois,
        labels.videos,
        labels.tracks,
        contexts=roi_ctxs,
        **table_kwargs,
    )
    write_masks(
        labels_path,
        all_masks,
        labels.videos,
        labels.tracks,
        contexts=mask_ctxs,
        **table_kwargs,
    )
    write_bboxes(
        labels_path,
        all_bboxes,
        labels.videos,
        labels.tracks,
        contexts=bbox_ctxs,
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
        **table_kwargs,
    )
    if not legacy_format:
        _warn_unconsumed_carry(lazy_store.annotation_carry)

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
    unlabeled_as: str = "error",
    legacy_format: bool = False,
    compression: str | None = None,
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
        unlabeled_as: How `legacy_format=True` handles unlabeled points
            (`labeled=False`), which the 2.x layout can't store. `"error"` (the
            default) raises a `ValueError` before anything is written. `"absent"`
            saves them as hidden points (labeled, not visible, no location) and
            warns. Ignored for the default (3.0) layout, which stores them.
        legacy_format: If `True`, write the SLP 2.x layout (compound `/points`,
            `/pred_points`, `/instances` and `/frames` datasets, `/negative_frames`,
            compound ROI, mask, label image, link and session tables with their
            separate string datasets, and the 2.x `format_id`) that older sleap-io
            and SLEAP releases can read. Default `False` writes SLP format 3.0,
            where every table is a group of columns.
        compression: Compression for the table columns (the pose tables and their
            metadata columns, and the ROI, mask, label image, link and session
            tables): `None` (the default) for contiguous, uncompressed data that
            reads fastest, or `"gzip"` or `"lzf"` (chunked, with the shuffle
            filter) for smaller files. Datasets that are compressed by default
            (geometry, segmentation, label image and 3D point data) stay
            compressed either way.

    Notes:
        Instance and frame `metadata` is stored as columns of `/instances` and
        `/frames` (lazy labels write the columns they were loaded with). Bounding
        box metadata can't be stored yet, and `legacy_format=True` stores no
        metadata; either is dropped with a warning. Metadata kinds are checked
        first (`Labels.check_metadata`) for eager labels, so a key that mixes kinds
        across records raises a `TypeError` before anything is written.

        Unknown table columns read from a file written by a newer sleap-io are
        carried through the save as long as their table's rows are in their loaded
        order: always for lazy labels' pose and session tables, and otherwise
        when the objects owning the rows (instances with their points arrays,
        frames, ROIs, masks, label images and their objects, frame and instance
        groups) and the links are the ones loaded, in the same order. Otherwise
        they are dropped with a warning.
    """
    if unlabeled_as not in _UNLABELED_AS_MODES:
        raise ValueError(
            f"unlabeled_as must be one of {_UNLABELED_AS_MODES}, got {unlabeled_as!r}."
        )
    if compression not in _COMPRESSION_MODES:
        raise ValueError(
            f"compression must be one of {_COMPRESSION_MODES}, got {compression!r}."
        )

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
                legacy_format=legacy_format,
                compression=compression,
                unlabeled_as=unlabeled_as,
            )
            return

    # Check before touching the output file so a refused save leaves it intact.
    if legacy_format:
        _check_unlabeled_points(labels, unlabeled_as)
    _check_dropped_metadata(labels, legacy_format=legacy_format)

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
    # Unknown columns of every table, kept from loading (see `_ColumnCarry`).
    carry = getattr(labels, "_carried_columns", None)
    _start_carry(carry)
    table_kwargs = dict(
        legacy_format=legacy_format, compression=compression, carry=carry
    )
    write_identity_links(labels_path, labels, **table_kwargs)
    # Auto-collect any detection category not yet registered in the catalog
    # (object-identity deduped), mirroring _collect_identities, then persist the
    # /categories catalog + per-detection links (SLP 2.7+). Additive: omitted
    # entirely when there are no categories (files stay byte-identical).
    labels._collect_categories()
    write_categories(labels_path, labels.categories)
    write_category_links(labels_path, labels, **table_kwargs)
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
        **table_kwargs,
    )
    write_metadata(labels_path, labels, legacy_format=legacy_format)
    write_lfs(labels_path, labels, legacy_format=legacy_format, compression=compression)

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

    video_ids = _identity_index(labels.videos)
    for lf in labels.labeled_frames:
        all_instances.extend(lf.instances)
        vid_idx = video_ids.get(id(lf.video), -1)
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
        roi_contexts.append((video_ids.get(id(r.video), -1), -1))

    write_rois(
        labels_path,
        all_rois,
        labels.videos,
        labels.tracks,
        all_instances,
        contexts=roi_contexts,
        **table_kwargs,
    )
    write_masks(
        labels_path,
        all_masks,
        labels.videos,
        labels.tracks,
        all_instances,
        contexts=mask_contexts,
        **table_kwargs,
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
        **table_kwargs,
    )
    if not legacy_format:
        _warn_unconsumed_carry(carry)

    # Re-emit any unknown members captured from the source file.
    _restore_unknown_hdf5(labels_path, unknown_stash)
