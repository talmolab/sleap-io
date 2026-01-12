"""SLEAP Analysis HDF5 format I/O.

This module provides read/write support for the SLEAP Analysis HDF5 format,
a portable format for exporting pose tracking predictions as dense numpy arrays.

Format features:
- Configurable axis ordering via presets or explicit dimension positions
- Compressed storage with gzip
- Self-documenting: dimension names stored as dataset attributes
- Optional extended metadata for full round-trip

Dataset Shapes and Dimensions
-----------------------------

The format stores pose data in dense numpy arrays. The `preset` parameter
controls the axis ordering. Two presets are available:

**preset="standard" (Python-native, intuitive indexing):**

Arrays are stored with frame as the first axis, enabling natural indexing
like ``tracks[frame, track, node, :]`` to get (x, y) coordinates.

::

    tracks:           (n_frames, n_tracks, n_nodes, 2)
                      dims: ("frame", "track", "node", "xy")
    track_occupancy:  (n_frames, n_tracks)
                      dims: ("frame", "track")
    point_scores:     (n_frames, n_tracks, n_nodes)
                      dims: ("frame", "track", "node")
    instance_scores:  (n_frames, n_tracks)
                      dims: ("frame", "track")
    tracking_scores:  (n_frames, n_tracks)
                      dims: ("frame", "track")

**preset="matlab" (default, SLEAP-compatible):**

Arrays are stored in the order used by SLEAP's original analysis export,
which is optimized for MATLAB's column-major memory layout.

::

    tracks:           (n_tracks, 2, n_nodes, n_frames)
                      dims: ("track", "xy", "node", "frame")
    track_occupancy:  (n_frames, n_tracks)
                      dims: ("frame", "track")
    point_scores:     (n_tracks, n_nodes, n_frames)
                      dims: ("track", "node", "frame")
    instance_scores:  (n_tracks, n_frames)
                      dims: ("track", "frame")
    tracking_scores:  (n_tracks, n_frames)
                      dims: ("track", "frame")

Note: ``track_occupancy`` has different axis ordering than other 2D arrays.
This matches SLEAP's original behavior where ``track_occupancy`` was stored
with frames first, while other arrays have tracks first. This quirk is
preserved for exact MATLAB compatibility with SLEAP's analysis export.

Note: The ``xy`` dimension always has size 2, representing (x, y) coordinates.

Custom Axis Ordering
--------------------

For advanced use cases, you can specify explicit dimension positions using
the ``frame_dim``, ``track_dim``, ``node_dim``, and ``xy_dim`` parameters.
These are mutually exclusive with ``preset``.

HDF5 Attributes
---------------

Each dataset stores its dimension names in the ``dims`` attribute as a JSON
list of strings. File-level attributes include:

- ``preset``: str, the preset used ("matlab", "standard", or "custom")
- ``format``: str, always "analysis" for this format
- ``sleap_io_version``: str, format version (currently "1.0")

Example:
    >>> import sleap_io as sio
    >>> labels = sio.load_slp("predictions.slp")
    >>> sio.save_analysis_h5(labels, "predictions.analysis.h5")
    >>> labels_loaded = sio.load_analysis_h5("predictions.analysis.h5")

    To save with Python-native ordering:
    >>> sio.save_analysis_h5(labels, "output.h5", preset="standard")

    To inspect stored dimensions:
    >>> import h5py
    >>> with h5py.File("predictions.analysis.h5", "r") as f:
    ...     print(f["tracks"].attrs["dims"])
    ...     print(f.attrs["preset"])
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from sleap_io.model.instance import PredictedInstance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

# =============================================================================
# Preset Definitions
# =============================================================================

# Canonical internal shape: (frame, track, node, xy) for tracks
# This is the shape used internally before any reordering.
#
# For other arrays:
# - point_scores: (frame, track, node)
# - instance_scores, tracking_scores, track_occupancy: (frame, track)

# Presets define dimension positions in the stored array
# Format: {dim_name: position} for the tracks array (4D)
PRESETS: dict[str, dict[str, int]] = {
    # Standard: (frame, track, node, xy) - intuitive Python indexing
    "standard": {"frame": 0, "track": 1, "node": 2, "xy": 3},
    # MATLAB: (track, xy, node, frame) - SLEAP-compatible column-major
    "matlab": {"frame": 3, "track": 0, "node": 2, "xy": 1},
}


def _get_axis_order(
    preset: str | None,
    frame_dim: int | None,
    track_dim: int | None,
    node_dim: int | None,
    xy_dim: int | None,
) -> tuple[dict[str, int], str]:
    """Resolve axis ordering from preset or explicit dimensions.

    Args:
        preset: Preset name ("matlab" or "standard"), or None if using explicit dims.
        frame_dim: Position of frame dimension (0-3), or None.
        track_dim: Position of track dimension (0-3), or None.
        node_dim: Position of node dimension (0-3), or None.
        xy_dim: Position of xy dimension (0-3), or None.

    Returns:
        Tuple of (axis_order dict, preset_name string).

    Raises:
        ValueError: If both preset and explicit dims are specified, or if
            explicit dims are incomplete/invalid.
    """
    explicit_dims = [frame_dim, track_dim, node_dim, xy_dim]
    has_explicit = any(d is not None for d in explicit_dims)

    if preset is not None and has_explicit:
        raise ValueError(
            "Cannot specify both 'preset' and explicit dimension positions "
            "(frame_dim, track_dim, node_dim, xy_dim). Use one or the other."
        )

    if has_explicit:
        # Validate explicit dimensions
        if not all(d is not None for d in explicit_dims):
            raise ValueError(
                "When using explicit dimensions, all four must be specified: "
                "frame_dim, track_dim, node_dim, xy_dim"
            )
        if set(explicit_dims) != {0, 1, 2, 3}:
            raise ValueError(
                "Dimension positions must be a permutation of [0, 1, 2, 3]. "
                f"Got: frame_dim={frame_dim}, track_dim={track_dim}, "
                f"node_dim={node_dim}, xy_dim={xy_dim}"
            )
        return (
            {"frame": frame_dim, "track": track_dim, "node": node_dim, "xy": xy_dim},
            "custom",
        )

    # Use preset (default to "matlab" for backwards compatibility)
    preset = preset or "matlab"
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[preset], preset


def _get_transpose_axes(
    from_order: dict[str, int], to_order: dict[str, int], ndim: int
) -> tuple[int, ...]:
    """Compute transpose axes to convert between axis orderings.

    Args:
        from_order: Source axis ordering (dim_name -> position).
        to_order: Target axis ordering (dim_name -> position).
        ndim: Number of dimensions (4 for tracks, 3 for point_scores, 2 for others).

    Returns:
        Tuple of axis indices for np.transpose.
    """
    # For arrays with fewer dimensions, filter to applicable dims
    if ndim == 4:
        dims = ["frame", "track", "node", "xy"]
    elif ndim == 3:
        dims = ["frame", "track", "node"]
    else:  # ndim == 2
        dims = ["frame", "track"]

    # Build transpose axes: for each position in target, find source position
    axes = []
    for target_pos in range(ndim):
        # Find which dim goes to this target position
        for dim in dims:
            if to_order.get(dim) == target_pos:
                axes.append(from_order[dim])
                break
    return tuple(axes)


def _get_dims_tuple(axis_order: dict[str, int], ndim: int) -> tuple[str, ...]:
    """Get dimension names tuple for given axis ordering.

    Args:
        axis_order: Axis ordering (dim_name -> position).
        ndim: Number of dimensions.

    Returns:
        Tuple of dimension names in stored order.
    """
    if ndim == 4:
        dims = ["frame", "track", "node", "xy"]
    elif ndim == 3:
        dims = ["frame", "track", "node"]
    else:  # ndim == 2
        dims = ["frame", "track"]

    # Sort dims by their position in axis_order
    result = [""] * ndim
    for dim in dims:
        if dim in axis_order:
            result[axis_order[dim]] = dim
    return tuple(result)


# =============================================================================
# Format Detection
# =============================================================================


def is_analysis_h5_file(filename: str | Path) -> bool:
    """Check if file is a SLEAP Analysis HDF5 file.

    This distinguishes Analysis HDF5 files from JABS HDF5 files by checking
    for the presence of the `track_occupancy` dataset which is unique to
    the Analysis format.

    Args:
        filename: Path to the HDF5 file to check.

    Returns:
        True if the file is a SLEAP Analysis HDF5 file, False otherwise.
    """
    try:
        with h5py.File(filename, "r") as f:
            # Analysis HDF5 has track_occupancy dataset, JABS has poseest group
            return "track_occupancy" in f
    except Exception:
        return False


# =============================================================================
# Read Functions
# =============================================================================


def read_labels(
    filename: str | Path,
    video: Video | str | None = None,
) -> Labels:
    """Load SLEAP Analysis HDF5 file.

    Args:
        filename: Path to Analysis HDF5 file.
        video: Video to associate with data. If None, uses video_path stored
            in the file. Can be a Video object or path string.

    Returns:
        Labels object with loaded pose data.

    Notes:
        The function automatically detects the axis ordering from the stored
        ``dims`` attributes and handles all presets correctly.

        If the file contains extended metadata (skeleton symmetries, video
        backend metadata, etc.), it will be used to reconstruct the full
        Labels context.

    See Also:
        write_labels: Save Labels to Analysis HDF5 file.
    """
    filename = Path(filename)

    with h5py.File(filename, "r") as f:
        # Read dimension info from attributes
        tracks_dims = None
        if "dims" in f["tracks"].attrs:
            tracks_dims = tuple(json.loads(f["tracks"].attrs["dims"]))

        # Determine axis ordering from stored dims or legacy transpose attr
        if tracks_dims:
            # Build axis order from stored dims
            stored_order = {dim: i for i, dim in enumerate(tracks_dims)}
        else:
            # Legacy file: check transpose attribute
            was_transposed = f.attrs.get("transpose", True)
            if was_transposed:
                stored_order = PRESETS["matlab"]
            else:
                # Old "transpose=False" used a different internal shape
                # (frame, node, xy, track) - handle this legacy case
                stored_order = {"frame": 0, "node": 1, "xy": 2, "track": 3}

        # Canonical internal order for processing
        canonical_order = {"frame": 0, "track": 1, "node": 2, "xy": 3}

        # Read and reorder tracks data
        tracks_raw = f["tracks"][:]
        axes_4d = _get_transpose_axes(stored_order, canonical_order, 4)
        tracks_data = np.transpose(tracks_raw, axes_4d)
        # Now tracks_data is (frames, tracks, nodes, 2)

        # Build 3D and 2D axis orders from stored dims
        stored_order_3d = {k: v for k, v in stored_order.items() if k != "xy"}
        # Renumber positions for 3D
        positions_3d = sorted(stored_order_3d.values())
        stored_order_3d = {k: positions_3d.index(v) for k, v in stored_order_3d.items()}
        canonical_order_3d = {"frame": 0, "track": 1, "node": 2}

        stored_order_2d = {
            k: v for k, v in stored_order.items() if k in ("frame", "track")
        }
        positions_2d = sorted(stored_order_2d.values())
        stored_order_2d = {k: positions_2d.index(v) for k, v in stored_order_2d.items()}
        canonical_order_2d = {"frame": 0, "track": 1}

        # Read and reorder other arrays
        axes_3d = _get_transpose_axes(stored_order_3d, canonical_order_3d, 3)
        axes_2d = _get_transpose_axes(stored_order_2d, canonical_order_2d, 2)

        point_scores_data = np.transpose(f["point_scores"][:], axes_3d)
        instance_scores_data = np.transpose(f["instance_scores"][:], axes_2d)
        tracking_scores_data = np.transpose(f["tracking_scores"][:], axes_2d)
        # All now in canonical order: (frames, tracks, ...) or (frames, tracks)

        # Read string arrays
        if len(f["track_names"]) > 0:
            track_names = [n.decode("utf-8") for n in f["track_names"][:]]
        else:
            track_names = []

        node_names = [n.decode("utf-8") for n in f["node_names"][:]]

        # Read edges for skeleton reconstruction
        if "edge_names" in f and len(f["edge_names"]) > 0:
            edge_names = [
                (s.decode("utf-8"), d.decode("utf-8")) for s, d in f["edge_names"][:]
            ]
        else:
            edge_names = []

        # Read metadata
        video_path = ""
        if "video_path" in f:
            video_path = f["video_path"][()].decode("utf-8")

        provenance = {}
        if "provenance" in f:
            provenance = json.loads(f["provenance"][()].decode("utf-8"))

        # Read extended metadata from attributes
        skeleton_name = f.attrs.get("skeleton_name", "")
        if isinstance(skeleton_name, bytes):
            skeleton_name = skeleton_name.decode("utf-8")

        skeleton_symmetries_raw = f.attrs.get("skeleton_symmetries", "[]")
        if isinstance(skeleton_symmetries_raw, bytes):
            skeleton_symmetries_raw = skeleton_symmetries_raw.decode("utf-8")
        skeleton_symmetries = json.loads(skeleton_symmetries_raw)

        video_backend_metadata_raw = f.attrs.get("video_backend_metadata", "{}")
        if isinstance(video_backend_metadata_raw, bytes):
            video_backend_metadata_raw = video_backend_metadata_raw.decode("utf-8")
        video_backend_metadata = json.loads(video_backend_metadata_raw)

    # Create video
    if video is None:
        video = Video(video_path)
        video.backend_metadata = video_backend_metadata
    elif isinstance(video, str):
        video = Video(video)

    # Create skeleton with edges and symmetries
    skeleton = Skeleton(
        nodes=node_names,
        edges=edge_names,
        name=skeleton_name if skeleton_name else None,
    )
    for node1_name, node2_name in skeleton_symmetries:
        try:
            skeleton.add_symmetry(node1_name, node2_name)
        except (ValueError, KeyError):
            pass  # Skip invalid symmetries

    # Create tracks
    if track_names:
        tracks = [Track(name=name) for name in track_names]
    else:
        # Single-instance case (no tracks)
        tracks = [None]

    # Get dimensions from canonical shape: (frames, tracks, nodes, 2)
    n_frames, n_tracks, n_nodes, _ = tracks_data.shape

    # Create labeled frames
    labeled_frames = []
    for frame_idx in range(n_frames):
        instances = []
        for track_idx in range(n_tracks):
            # Get point data: tracks_data[frame, track, :, :] -> (nodes, 2)
            points = tracks_data[frame_idx, track_idx, :, :]

            # Skip if all points are NaN (no instance at this frame)
            if np.all(np.isnan(points)):
                continue

            # Get scores
            point_scores = point_scores_data[frame_idx, track_idx, :]
            instance_score = instance_scores_data[frame_idx, track_idx]
            tracking_score = tracking_scores_data[frame_idx, track_idx]

            # Create predicted instance
            inst = PredictedInstance.from_numpy(
                points_data=points,
                skeleton=skeleton,
                point_scores=point_scores,
                score=instance_score if not np.isnan(instance_score) else 0.0,
                track=tracks[track_idx] if tracks[track_idx] else None,
                tracking_score=tracking_score if not np.isnan(tracking_score) else None,
            )
            instances.append(inst)

        if instances:
            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
            labeled_frames.append(lf)

    # Create Labels
    labels = Labels(
        labeled_frames=labeled_frames,
        videos=[video],
        skeletons=[skeleton],
        tracks=[t for t in tracks if t is not None],
        provenance=provenance,
    )

    labels.provenance["filename"] = str(filename)

    return labels


# =============================================================================
# Write Functions
# =============================================================================


def _get_occupancy_and_points(
    labels: Labels,
    video: Video | None = None,
    all_frames: bool = True,
    min_occupancy: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    int,
]:
    """Build numpy matrices with track occupancy and point location data.

    All returned arrays are in canonical shape (frame-first ordering).

    Args:
        labels: The Labels from which to get data.
        video: Video to export. If None, uses first video.
        all_frames: If True, include all frames from 0 to last frame.
            If False, only include frames from first to last labeled frame.
        min_occupancy: Minimum occupancy ratio (0-1) to keep a track.
            0 = keep all non-empty tracks (default SLEAP behavior).

    Returns:
        Tuple of (all in canonical shape):

        - occupancy_matrix: shape (n_frames, n_tracks)
            Binary matrix indicating track presence per frame.

        - locations_matrix: shape (n_frames, n_tracks, n_nodes, 2)
            Point coordinates (x, y) for each frame, track, and node.

        - point_scores: shape (n_frames, n_tracks, n_nodes)
            Confidence score for each point.

        - instance_scores: shape (n_frames, n_tracks)
            Overall instance confidence score.

        - tracking_scores: shape (n_frames, n_tracks)
            Tracking confidence score.

        - track_names: list of track name strings, length n_tracks

        - first_frame: first frame index (int)
    """
    if video is None:
        video = labels.videos[0]

    # Get labeled frames for this video
    lfs = labels.find(video)
    if not lfs:
        raise ValueError(f"No labeled frames in video: {video.filename}")

    frame_idxs = sorted(lf.frame_idx for lf in lfs)
    first_frame = 0 if all_frames else frame_idxs[0]
    last_frame = frame_idxs[-1]
    n_frames = last_frame - first_frame + 1

    # Get track and node info
    tracks = labels.tracks or [None]  # Handle untracked case
    track_count = len(tracks)
    skeleton = labels.skeletons[0]
    node_count = len(skeleton.nodes)

    # Initialize matrices in canonical shape: (frame, track, ...)
    occupancy = np.zeros((n_frames, track_count), dtype=np.uint8)
    locations = np.full(
        (n_frames, track_count, node_count, 2), np.nan, dtype=np.float64
    )
    point_scores = np.full(
        (n_frames, track_count, node_count), np.nan, dtype=np.float64
    )
    instance_scores = np.full((n_frames, track_count), np.nan, dtype=np.float64)
    tracking_scores = np.full((n_frames, track_count), np.nan, dtype=np.float64)

    # Build lookup for frame index -> LabeledFrame
    lf_map = {lf.frame_idx: lf for lf in lfs}

    # Fill matrices
    for frame_idx in range(first_frame, last_frame + 1):
        frame_i = frame_idx - first_frame
        lf = lf_map.get(frame_idx)
        if lf is None:
            continue

        # Prefer user instances over predicted for same track
        track_instances = {}

        # Add predicted first (will be overwritten by user if exists)
        for inst in lf.predicted_instances:
            track = inst.track
            track_i = tracks.index(track) if track in tracks else 0
            track_instances[track_i] = inst

        # Add user instances (override predicted)
        for inst in lf.user_instances:
            track = inst.track
            track_i = tracks.index(track) if track in tracks else 0
            track_instances[track_i] = inst

        # Fill matrices
        for track_i, inst in track_instances.items():
            occupancy[frame_i, track_i] = 1
            locations[frame_i, track_i, :, :] = inst.numpy()

            if hasattr(inst, "tracking_score") and inst.tracking_score is not None:
                tracking_scores[frame_i, track_i] = inst.tracking_score

            if isinstance(inst, PredictedInstance):
                if "score" in inst.points.dtype.names:
                    point_scores[frame_i, track_i, :] = inst.points["score"]
                if inst.score is not None:
                    instance_scores[frame_i, track_i] = inst.score

    # Filter empty/low-occupancy tracks
    occupied_frames = np.sum(occupancy, axis=0)
    occupancy_ratio = occupied_frames / n_frames
    keep_mask = (occupied_frames > 0) & (occupancy_ratio >= min_occupancy)

    if not np.all(keep_mask):
        occupancy = occupancy[:, keep_mask]
        locations = locations[:, keep_mask, :, :]
        point_scores = point_scores[:, keep_mask, :]
        instance_scores = instance_scores[:, keep_mask]
        tracking_scores = tracking_scores[:, keep_mask]
        tracks = [t for i, t in enumerate(tracks) if keep_mask[i]]

    # Get track names
    track_names = [t.name if t else "" for t in tracks]

    return (
        occupancy,
        locations,
        point_scores,
        instance_scores,
        tracking_scores,
        track_names,
        first_frame,
    )


def write_labels(
    labels: Labels,
    filename: str | Path,
    *,
    video: Video | int | None = None,
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

            - ``"matlab"`` (default): SLEAP-compatible ordering for MATLAB.
              tracks shape: (n_tracks, 2, n_nodes, n_frames)
            - ``"standard"``: Intuitive Python ordering.
              tracks shape: (n_frames, n_tracks, n_nodes, 2)

            Mutually exclusive with explicit dimension parameters.

        frame_dim: Position of the frame dimension (0-3). Mutually exclusive
            with ``preset``.
        track_dim: Position of the track dimension (0-3). Mutually exclusive
            with ``preset``.
        node_dim: Position of the node dimension (0-3). Mutually exclusive
            with ``preset``.
        xy_dim: Position of the xy dimension (0-3). Mutually exclusive
            with ``preset``.
        save_metadata: Store extended metadata for full round-trip.
            Default True. Includes skeleton symmetries and video backend metadata.

    Notes:
        Each dataset stores its dimension names in the ``dims`` attribute as
        a JSON-encoded list of strings (e.g., '["frame", "track", "node", "xy"]').

        The file-level ``preset`` attribute indicates which preset was used,
        enabling correct loading regardless of the axis ordering.

    Examples:
        Save with default MATLAB-compatible ordering::

            >>> sio.save_analysis_h5(labels, "output.h5")

        Save with Python-native ordering::

            >>> sio.save_analysis_h5(labels, "output.h5", preset="standard")

        Save with custom axis ordering::

            >>> sio.save_analysis_h5(
            ...     labels, "output.h5",
            ...     frame_dim=0, track_dim=1, node_dim=2, xy_dim=3
            ... )

    See Also:
        read_labels: Load Labels from Analysis HDF5 file.
    """
    filename = Path(filename)

    # Resolve axis ordering
    axis_order, preset_name = _get_axis_order(
        preset, frame_dim, track_dim, node_dim, xy_dim
    )

    # Resolve video
    if video is None:
        video = labels.videos[0]
    elif isinstance(video, int):
        video = labels.videos[video]

    # Get matrices in canonical shape (frame, track, node, xy)
    (
        occupancy,
        locations,
        point_scores,
        instance_scores,
        tracking_scores,
        track_names,
        first_frame,
    ) = _get_occupancy_and_points(labels, video, all_frames, min_occupancy)

    # Define canonical order
    canonical_order_4d = {"frame": 0, "track": 1, "node": 2, "xy": 3}
    canonical_order_3d = {"frame": 0, "track": 1, "node": 2}
    canonical_order_2d = {"frame": 0, "track": 1}

    # Compute target orders for each array type
    target_order_3d = {k: v for k, v in axis_order.items() if k != "xy"}
    positions_3d = sorted(target_order_3d.values())
    target_order_3d = {k: positions_3d.index(v) for k, v in target_order_3d.items()}

    target_order_2d = {k: v for k, v in axis_order.items() if k in ("frame", "track")}
    positions_2d = sorted(target_order_2d.values())
    target_order_2d = {k: positions_2d.index(v) for k, v in target_order_2d.items()}

    # For matlab preset, track_occupancy has different ordering to match SLEAP's
    # original behavior. SLEAP stores track_occupancy as (frame, track) while
    # other 2D arrays are stored as (track, frame). This quirk is preserved for
    # backwards compatibility with MATLAB users expecting SLEAP's exact format.
    if preset_name == "matlab":
        target_order_occupancy = {"frame": 0, "track": 1}  # Same as canonical
    else:
        target_order_occupancy = target_order_2d

    # Compute transpose axes
    axes_4d = _get_transpose_axes(canonical_order_4d, axis_order, 4)
    axes_3d = _get_transpose_axes(canonical_order_3d, target_order_3d, 3)
    axes_2d = _get_transpose_axes(canonical_order_2d, target_order_2d, 2)
    axes_occupancy = _get_transpose_axes(canonical_order_2d, target_order_occupancy, 2)

    # Reorder arrays
    locations = np.transpose(locations, axes_4d)
    point_scores = np.transpose(point_scores, axes_3d)
    instance_scores = np.transpose(instance_scores, axes_2d)
    tracking_scores = np.transpose(tracking_scores, axes_2d)
    occupancy = np.transpose(occupancy, axes_occupancy)

    # Get dimension names for attributes
    dims_4d = _get_dims_tuple(axis_order, 4)
    dims_3d = _get_dims_tuple(target_order_3d, 3)
    dims_2d = _get_dims_tuple(target_order_2d, 2)
    dims_occupancy = _get_dims_tuple(target_order_occupancy, 2)

    # Get skeleton info
    skeleton = labels.skeletons[0]
    node_names = [node.name for node in skeleton.nodes]
    edge_names = [(e.source.name, e.destination.name) for e in skeleton.edges]
    edge_inds = skeleton.edge_inds

    # Write HDF5
    with h5py.File(filename, "w") as f:
        # Helper to write dataset with dimension attributes
        def write_dataset(
            name: str,
            data,
            dim_names: tuple | None = None,
        ) -> None:
            if isinstance(data, np.ndarray):
                ds = f.create_dataset(
                    name, data=data, compression="gzip", compression_opts=9
                )
                if dim_names is not None:
                    ds.attrs["dims"] = json.dumps(dim_names)
            else:
                f.create_dataset(name, data=data)

        # Core data matrices
        write_dataset("tracks", locations, dim_names=dims_4d)
        write_dataset("track_occupancy", occupancy, dim_names=dims_occupancy)
        write_dataset("point_scores", point_scores, dim_names=dims_3d)
        write_dataset("instance_scores", instance_scores, dim_names=dims_2d)
        write_dataset("tracking_scores", tracking_scores, dim_names=dims_2d)

        # String arrays
        write_dataset("track_names", np.array([n.encode("utf-8") for n in track_names]))
        write_dataset("node_names", np.array([n.encode("utf-8") for n in node_names]))
        write_dataset(
            "edge_names",
            np.array([(s.encode("utf-8"), d.encode("utf-8")) for s, d in edge_names]),
        )
        write_dataset("edge_inds", np.array(edge_inds))

        # Metadata
        write_dataset("labels_path", str(labels_path) if labels_path else "")
        write_dataset("video_path", video.filename or "")
        write_dataset("video_ind", labels.videos.index(video))
        write_dataset("provenance", json.dumps(dict(labels.provenance)))

        # File-level attributes (always written for format identification)
        f.attrs["preset"] = preset_name
        f.attrs["format"] = "analysis"
        f.attrs["sleap_io_version"] = "1.0"

        # Extended metadata for round-trip (as attributes)
        if save_metadata:
            symmetries = [
                (list(s.nodes)[0].name, list(s.nodes)[1].name)
                for s in skeleton.symmetries
            ]
            f.attrs["skeleton_name"] = skeleton.name or ""
            f.attrs["skeleton_symmetries"] = json.dumps(symmetries)
            f.attrs["video_backend_metadata"] = json.dumps(video.backend_metadata)
