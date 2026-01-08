"""DataFrame codec for SLEAP Labels objects.

This module provides conversion between Labels objects and pandas/polars DataFrames
with multiple layout formats to suit different analysis needs.

Supported formats:
- **points**: One row per point (maximally normalized, long format)
- **instances**: One row per instance (denormalized, wide format)
- **frames**: One row per frame-track combination (trajectory analysis)
- **multi_index**: Hierarchical column structure (similar to NWB format)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Generator, Iterator, Optional, Union

import numpy as np
import pandas as pd

from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io.model.labels import Labels
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from typing_extensions import Literal

    from sleap_io.io.slp_lazy import LazyDataStore

# Optional polars support
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:  # pragma: no cover
    HAS_POLARS = False
    pl = None  # type: ignore[assignment]


class DataFrameFormat(str, Enum):
    """Enumeration of supported DataFrame formats."""

    POINTS = "points"
    """One row per point (frame, instance, node). Most normalized format."""

    INSTANCES = "instances"
    """One row per instance. Columns for each node's x/y coordinates."""

    FRAMES = "frames"
    """One row per frame-track combination. For trajectory analysis."""

    MULTI_INDEX = "multi_index"
    """Hierarchical column structure. Similar to NWB format."""


def _create_dataframe_from_rows(
    rows: list[dict],
    backend: str = "pandas",
) -> "pd.DataFrame | pl.DataFrame":
    """Create a DataFrame from a list of row dictionaries.

    This helper function provides native DataFrame construction for both pandas
    and polars backends, avoiding the overhead of pandas→polars conversion.

    Args:
        rows: List of dictionaries where each dict represents a row.
        backend: "pandas" or "polars".

    Returns:
        DataFrame in the specified backend format.
    """
    if not rows:
        if backend == "polars":
            return pl.DataFrame()
        return pd.DataFrame()

    if backend == "polars":
        return pl.from_dicts(rows)
    return pd.DataFrame(rows)


def _flatten_tuple_keys(rows: list[dict]) -> list[dict]:
    """Convert tuple keys to dot-separated strings for polars compatibility.

    Args:
        rows: List of row dicts with potentially tuple keys like (inst0, nose, x).

    Returns:
        List of row dicts with flattened string keys like "inst0.nose.x".
    """
    flattened = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            if isinstance(key, tuple):  # pragma: no cover
                # Note: This branch is currently unreachable because multi_index
                # with polars backend now uses flat keys directly (use_flat_keys=True)
                parts = [str(p) for p in key if p]
                new_key = ".".join(parts)
                new_row[new_key] = value
            else:
                new_row[key] = value
        flattened.append(new_row)
    return flattened


def to_dataframe(
    labels: Labels,
    format: DataFrameFormat | str = DataFrameFormat.POINTS,
    *,
    video: Optional[Video | int] = None,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    video_id: Literal["path", "index", "name", "object"] = "path",
    include_video: Optional[bool] = None,
    instance_id: Literal["index", "track"] = "index",
    untracked: Literal["error", "ignore"] = "error",
    backend: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | "pl.DataFrame":
    """Convert Labels to a DataFrame.

    Args:
        labels: Labels object to convert.
        format: Output format. One of "points", "instances", "frames", "multi_index".
        video: Optional video filter. If specified, only frames from this video
            are included. Can be a Video object or integer index.
        include_metadata: Include track, video information in columns.
        include_score: Include confidence scores for predicted instances.
        include_user_instances: Include user-labeled instances.
        include_predicted_instances: Include predicted instances.
        video_id: How to represent videos in the DataFrame. Options:
            - "path": Full filename/path (default). Works for all video types.
            - "index": Integer video index. Compact, requires video list for decoding.
            - "name": Just the video filename (no directory). May not be unique.
            - "object": Store Video object directly. Not serializable but preserves
              all video metadata (dataset for HDF5, frame paths for ImageVideo).
        include_video: Whether to include video information. If None (default),
            automatically includes video info if there are multiple videos or if
            video metadata is needed. Set False to always omit, True to always include.
        instance_id: How to name instance columns in "frames" and "multi_index" formats.
            - "index": Use inst0, inst1, inst2, etc. (default).
            - "track": Use track names as column prefixes (e.g., mouse1, mouse2).
        untracked: Behavior for untracked instances with instance_id="track".
            - "error": Raise error if any instance lacks a track (default).
            - "ignore": Skip untracked instances silently.
        backend: "pandas" or "polars". Polars requires the polars package.
            When using polars, DataFrames are constructed natively without
            going through pandas, providing better performance for large datasets.

    Returns:
        DataFrame in the specified format. Type depends on backend parameter.

    Raises:
        ValueError: If an invalid format is specified or polars is requested but
            not installed.

    Examples:
        Basic usage:

        >>> labels = load_file("predictions.slp")
        >>> df = to_dataframe(labels, format="points")
        >>> df.head()
           frame_idx  video_path  track  node     x     y  score
        0          0  video.mp4  track0  nose  10.0  20.0   0.95
        1          0  video.mp4  track0  tail   5.0   8.0   0.92

        Wide format with instances multiplexed per frame:

        >>> df = to_dataframe(labels, format="frames")
        >>> df.columns  # inst0.track, inst0.nose.x, inst0.nose.y, ...

        Track-named columns (requires tracked instances):

        >>> df = to_dataframe(labels, format="frames", instance_id="track")
        >>> df.columns  # mouse1.nose.x, mouse1.nose.y, mouse2.nose.x, ...

        Native polars backend for better performance:

        >>> df = to_dataframe(labels, format="points", backend="polars")
        >>> type(df)
        <class 'polars.dataframe.frame.DataFrame'>

    Notes:
        The specific columns and structure depend on the format parameter.
        See the DataFrameFormat enum documentation for details on each format.

        Column naming conventions:
        - Points: frame_idx, node, x, y, track, track_score, instance_score
        - Instances: frame_idx, track, track_score, score, {node}.x/y/score
        - Frames: frame_idx, {inst}.track, {inst}.track_score, {inst}.score,
          {inst}.{node}.x, {inst}.{node}.y, {inst}.{node}.score
        - Multi-index: Hierarchical columns (inst, node, coord) with frame idx
          For polars backend, multi-index columns are flattened to dot-separated
          names (e.g., "inst0.nose.x").
    """
    # Validate backend
    if backend == "polars" and not HAS_POLARS:
        raise ValueError(
            "Polars backend requested but polars is not installed. "
            "Install with: pip install polars"
        )

    # Normalize format parameter
    if isinstance(format, str):
        try:
            format = DataFrameFormat(format.lower())
        except ValueError:
            valid_formats = ", ".join([f.value for f in DataFrameFormat])
            raise ValueError(
                f"Invalid format '{format}'. Must be one of: {valid_formats}"
            )

    # Convert video parameter to index for fast path filtering
    video_filter_idx: Optional[int] = None
    if video is not None:
        if isinstance(video, int):
            video_filter_idx = video
            video = labels.videos[video]
        else:
            video_filter_idx = labels.videos.index(video)

    # Determine whether to include video info
    if include_video is None:
        # Auto-detect: include if multiple videos, unless explicitly omitted
        include_video = len(labels.videos) > 1

    # Use lazy fast path when available (for POINTS and INSTANCES formats)
    if labels.is_lazy and format in (DataFrameFormat.POINTS, DataFrameFormat.INSTANCES):
        store = labels.labeled_frames._store

        if format == DataFrameFormat.POINTS:
            return _to_points_df_lazy(
                store,
                labels,
                video_filter=video_filter_idx,
                include_metadata=include_metadata,
                include_score=include_score,
                include_user_instances=include_user_instances,
                include_predicted_instances=include_predicted_instances,
                include_video=include_video,
                video_id=video_id,
                backend=backend,
            )
        else:  # INSTANCES
            return _to_instances_df_lazy(
                store,
                labels,
                video_filter=video_filter_idx,
                include_metadata=include_metadata,
                include_score=include_score,
                include_user_instances=include_user_instances,
                include_predicted_instances=include_predicted_instances,
                include_video=include_video,
                video_id=video_id,
                backend=backend,
            )

    # Eager path: filter labeled frames
    if video is not None:
        labeled_frames = [lf for lf in labels.labeled_frames if lf.video == video]
    else:
        labeled_frames = labels.labeled_frames

    # Route to appropriate converter based on format
    if format == DataFrameFormat.POINTS:
        df = _to_points_df(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
            backend=backend,
        )
    elif format == DataFrameFormat.INSTANCES:
        df = _to_instances_df(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
            backend=backend,
        )
    elif format == DataFrameFormat.FRAMES:
        df = _to_frames_df(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
            instance_id=instance_id,
            untracked=untracked,
            backend=backend,
        )
    elif format == DataFrameFormat.MULTI_INDEX:
        df = _to_multi_index_df(
            labels,
            labeled_frames,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
            instance_id=instance_id,
            untracked=untracked,
            backend=backend,
        )
    else:
        raise ValueError(f"Unknown format: {format}")

    return df


def to_dataframe_iter(
    labels: Labels,
    format: DataFrameFormat | str = DataFrameFormat.POINTS,
    *,
    chunk_size: int | None = None,
    video: Optional[Video | int] = None,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    video_id: Literal["path", "index", "name", "object"] = "path",
    include_video: Optional[bool] = None,
    instance_id: Literal["index", "track"] = "index",
    untracked: Literal["error", "ignore"] = "error",
    backend: Literal["pandas", "polars"] = "pandas",
) -> Iterator[pd.DataFrame | "pl.DataFrame"]:
    """Iterate over Labels data, yielding DataFrames in chunks.

    This is a memory-efficient alternative to `to_dataframe()` for large datasets.
    Instead of materializing the entire DataFrame at once, it yields smaller
    DataFrames (chunks) that can be processed incrementally.

    Args:
        labels: Labels object to convert.
        format: Output format. One of "points", "instances", "frames", "multi_index".
        chunk_size: Number of rows per chunk. If None (default), yields the entire
            DataFrame in a single chunk (equivalent to `to_dataframe()`).
            The meaning of "row" depends on the format:
            - points: One point (node) per row
            - instances: One instance per row
            - frames: One frame per row
            - multi_index: One frame per row
        video: Optional video filter. If specified, only frames from this video
            are included. Can be a Video object or integer index.
        include_metadata: Include track, video information in columns.
        include_score: Include confidence scores for predicted instances.
        include_user_instances: Include user-labeled instances.
        include_predicted_instances: Include predicted instances.
        video_id: How to represent videos in the DataFrame. Options:
            - "path": Full filename/path (default).
            - "index": Integer video index.
            - "name": Just the video filename.
            - "object": Store Video object directly.
        include_video: Whether to include video information.
        instance_id: How to name instance columns in "frames" and "multi_index" formats.
            - "index": Use inst0, inst1, inst2, etc. (default).
            - "track": Use track names as column prefixes.
        untracked: Behavior for untracked instances with instance_id="track".
            - "error": Raise error if any instance lacks a track (default).
            - "ignore": Skip untracked instances silently.
        backend: "pandas" or "polars". Polars requires the polars package.
            When using polars, DataFrames are constructed natively without
            going through pandas, providing better performance for large datasets.

    Yields:
        DataFrames, each containing up to `chunk_size` rows.

    Examples:
        Process large datasets in chunks:

        >>> for df_chunk in to_dataframe_iter(labels, chunk_size=10000):
        ...     df_chunk.to_parquet("output.parquet", append=True)

        Concatenate chunks to get full DataFrame (equivalent to to_dataframe):

        >>> import pandas as pd
        >>> df = pd.concat(list(to_dataframe_iter(labels, chunk_size=1000)))

        Memory-efficient per-video processing:

        >>> for video in labels.videos:
        ...     for chunk in to_dataframe_iter(labels, video=video, chunk_size=5000):
        ...         process_chunk(chunk)
    """
    # Validate backend
    if backend == "polars" and not HAS_POLARS:
        raise ValueError(
            "Polars backend requested but polars is not installed. "
            "Install with: pip install polars"
        )

    # Normalize format parameter
    if isinstance(format, str):
        try:
            format = DataFrameFormat(format.lower())
        except ValueError:
            valid_formats = ", ".join([f.value for f in DataFrameFormat])
            raise ValueError(
                f"Invalid format '{format}'. Must be one of: {valid_formats}"
            )

    # Filter to specific video if requested
    if video is not None:
        if isinstance(video, int):
            video = labels.videos[video]
        labeled_frames = [lf for lf in labels.labeled_frames if lf.video == video]
    else:
        labeled_frames = labels.labeled_frames

    # Determine whether to include video info
    if include_video is None:
        include_video = len(labels.videos) > 1

    # If no chunk_size specified, yield entire DataFrame at once
    if chunk_size is None:
        df = to_dataframe(
            labels,
            format=format,
            video=video,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            video_id=video_id,
            include_video=include_video,
            instance_id=instance_id,
            untracked=untracked,
            backend=backend,
        )
        yield df
        return

    # Get the appropriate row iterator and DataFrame builder
    if format == DataFrameFormat.POINTS:
        row_iter = _iter_points_rows(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
        )
    elif format == DataFrameFormat.INSTANCES:
        row_iter = _iter_instances_rows(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
        )
    elif format == DataFrameFormat.FRAMES:
        # For frames format, we need to pre-scan for max_instances and tracks
        max_instances, all_tracks, skeleton = _prescan_for_frames(
            labels,
            labeled_frames,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            instance_id=instance_id,
            untracked=untracked,
        )
        row_iter = _iter_frames_rows(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
            instance_id=instance_id,
            untracked=untracked,
            max_instances=max_instances,
            all_tracks=all_tracks,
            skeleton=skeleton,
        )
    elif format == DataFrameFormat.MULTI_INDEX:
        # For multi_index format, we also need to pre-scan
        max_instances, all_tracks, skeleton = _prescan_for_frames(
            labels,
            labeled_frames,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            instance_id=instance_id,
            untracked=untracked,
        )
        row_iter = _iter_multi_index_rows(
            labels,
            labeled_frames,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
            include_video=include_video,
            video_id=video_id,
            instance_id=instance_id,
            untracked=untracked,
            max_instances=max_instances,
            all_tracks=all_tracks,
            skeleton=skeleton,
        )
    else:
        raise ValueError(f"Unknown format: {format}")

    # Buffer rows and yield DataFrames
    buffer: list[dict] = []
    yielded_any = False
    for row in row_iter:
        buffer.append(row)
        if len(buffer) >= chunk_size:
            # For multi_index with polars, flatten tuple keys
            if format == DataFrameFormat.MULTI_INDEX and backend == "polars":
                buffer = _flatten_tuple_keys(buffer)
            df = _create_dataframe_from_rows(buffer, backend)
            yield df
            yielded_any = True
            buffer = []

    # Yield remaining rows (or empty DataFrame if no data)
    if buffer or not yielded_any:
        # For multi_index with polars, flatten tuple keys
        if format == DataFrameFormat.MULTI_INDEX and backend == "polars":
            buffer = _flatten_tuple_keys(buffer)
        df = _create_dataframe_from_rows(buffer, backend)
        yield df


def _format_video(
    video: Video, labels: Labels, video_id: str
) -> Union[str, int, Video]:
    """Format video based on video_id parameter.

    Args:
        video: Video object to format.
        labels: Labels object (for getting video index).
        video_id: How to represent the video ("path", "index", "name", "object").

    Returns:
        Formatted video representation.
    """
    if video_id == "path":
        return video.filename
    elif video_id == "index":
        return labels.videos.index(video)
    elif video_id == "name":
        # Get just the filename, handling both string and list filenames
        if isinstance(video.filename, list):
            # For ImageVideo, return first filename's basename
            from pathlib import Path

            return Path(video.filename[0]).name
        else:
            from pathlib import Path

            return Path(video.filename).name
    elif video_id == "object":
        return video
    else:
        raise ValueError(f"Invalid video_id: {video_id}")


# =============================================================================
# Row Iterators for Streaming
# =============================================================================


def _iter_points_rows(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
) -> Generator[dict, None, None]:
    """Yield row dicts for points format (one row per point)."""
    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance in instances_to_process:
            is_predicted = isinstance(instance, PredictedInstance)

            for node_idx, node in enumerate(instance.skeleton.nodes):
                point = instance.points[node_idx]

                row = {
                    "frame_idx": int(lf.frame_idx),
                    "node": node.name,
                    "x": float(point["xy"][0]),
                    "y": float(point["xy"][1]),
                }

                if include_metadata:
                    # Add video info if requested
                    if include_video:
                        video_value = _format_video(lf.video, labels, video_id)
                        if video_id == "index":
                            row["video_idx"] = video_value
                        elif video_id == "object":
                            row["video"] = video_value
                        else:
                            row["video_path"] = video_value

                    row["track"] = instance.track.name if instance.track else None

                    if is_predicted:
                        row["track_score"] = (
                            float(instance.tracking_score)
                            if instance.tracking_score is not None
                            else None
                        )
                        row["instance_score"] = (
                            float(instance.score)
                            if instance.score is not None
                            else None
                        )
                    else:
                        row["track_score"] = None
                        row["instance_score"] = None

                if include_score and is_predicted:
                    row["score"] = float(point["score"])

                yield row


def _iter_instances_rows(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
) -> Generator[dict, None, None]:
    """Yield row dicts for instances format (one row per instance)."""
    for lf in labeled_frames:
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance in instances_to_process:
            is_predicted = isinstance(instance, PredictedInstance)

            row = {"frame_idx": int(lf.frame_idx)}

            if include_metadata:
                if include_video:
                    video_value = _format_video(lf.video, labels, video_id)
                    if video_id == "index":
                        row["video_idx"] = video_value
                    elif video_id == "object":
                        row["video"] = video_value
                    else:
                        row["video_path"] = video_value

                row["track"] = instance.track.name if instance.track else None

                if is_predicted:
                    row["track_score"] = (
                        float(instance.tracking_score)
                        if instance.tracking_score is not None
                        else None
                    )
                    row["score"] = (
                        float(instance.score) if instance.score is not None else None
                    )
                else:
                    row["track_score"] = None
                    row["score"] = None

            # Add node coordinates
            for node_idx, node in enumerate(instance.skeleton.nodes):
                point = instance.points[node_idx]
                row[f"{node.name}.x"] = float(point["xy"][0])
                row[f"{node.name}.y"] = float(point["xy"][1])
                if include_score and is_predicted:
                    row[f"{node.name}.score"] = float(point["score"])
                elif include_score:
                    row[f"{node.name}.score"] = None

            yield row


def _prescan_for_frames(
    labels: Labels,
    labeled_frames: list,
    *,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    instance_id: str = "index",
    untracked: str = "error",
) -> tuple:
    """Pre-scan labeled frames to determine max_instances, tracks, and skeleton.

    This is used by frames and multi_index formats to ensure consistent column
    structure across chunks.

    Returns:
        Tuple of (max_instances, all_tracks, skeleton)
    """
    max_instances = 0

    # Use skeleton from labels (Labels always have at least one skeleton)
    skeleton = labels.skeletons[0] if labels.skeletons else None

    # Collect all track names if using track mode
    all_tracks = []
    if instance_id == "track":
        all_tracks = [t.name for t in labels.tracks]

    for lf in labeled_frames:
        # Count instances
        instances_to_process = []
        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        # Filter/validate for track mode
        if instance_id == "track":
            filtered = []
            for inst in instances_to_process:
                if inst.track is None:
                    if untracked == "error":
                        raise ValueError(
                            f"Instance in frame {lf.frame_idx} has no track. "
                            "Use instance_id='index' or untracked='ignore'."
                        )
                else:
                    filtered.append(inst)
            instances_to_process = filtered

        max_instances = max(max_instances, len(instances_to_process))

    return max_instances, all_tracks, skeleton


def _iter_frames_rows(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    instance_id: str = "index",
    untracked: str = "error",
    max_instances: int,
    all_tracks: list,
    skeleton,
) -> Generator[dict, None, None]:
    """Yield row dicts for frames format (one row per frame)."""
    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        # Filter for track mode
        if instance_id == "track":
            instances_to_process = [
                inst for inst in instances_to_process if inst.track is not None
            ]

        row = {"frame_idx": int(lf.frame_idx)}

        if include_video:
            video_value = _format_video(lf.video, labels, video_id)
            if video_id == "index":
                row["video_idx"] = video_value
            elif video_id == "object":
                row["video"] = video_value
            else:
                row["video_path"] = video_value

        if instance_id == "index":
            # Instance-indexed mode: inst0, inst1, ...
            for inst_idx in range(max_instances):
                prefix = f"inst{inst_idx}"

                if inst_idx < len(instances_to_process):
                    instance = instances_to_process[inst_idx]
                    is_predicted = isinstance(instance, PredictedInstance)

                    row[f"{prefix}.track"] = (
                        instance.track.name if instance.track else None
                    )
                    if is_predicted:
                        row[f"{prefix}.track_score"] = (
                            float(instance.tracking_score)
                            if instance.tracking_score is not None
                            else None
                        )
                        row[f"{prefix}.score"] = (
                            float(instance.score)
                            if instance.score is not None
                            else None
                        )
                    else:
                        row[f"{prefix}.track_score"] = None
                        row[f"{prefix}.score"] = None

                    for node_idx, node in enumerate(skeleton.nodes):
                        point = instance.points[node_idx]
                        row[f"{prefix}.{node.name}.x"] = float(point["xy"][0])
                        row[f"{prefix}.{node.name}.y"] = float(point["xy"][1])
                        if include_score and is_predicted:
                            row[f"{prefix}.{node.name}.score"] = float(point["score"])
                        elif include_score:
                            row[f"{prefix}.{node.name}.score"] = None
                else:
                    # Pad with NaN for missing instances
                    row[f"{prefix}.track"] = None
                    row[f"{prefix}.track_score"] = None
                    row[f"{prefix}.score"] = None
                    for node in skeleton.nodes:
                        row[f"{prefix}.{node.name}.x"] = np.nan
                        row[f"{prefix}.{node.name}.y"] = np.nan
                        if include_score:
                            row[f"{prefix}.{node.name}.score"] = np.nan

        else:  # instance_id == "track"
            # Build mapping of track_name -> instance for this frame
            track_instances = {}
            for instance in instances_to_process:
                if instance.track:
                    track_instances[instance.track.name] = instance

            for track_name in all_tracks:
                prefix = track_name

                if track_name in track_instances:
                    instance = track_instances[track_name]
                    is_predicted = isinstance(instance, PredictedInstance)

                    if is_predicted:
                        row[f"{prefix}.track_score"] = (
                            float(instance.tracking_score)
                            if instance.tracking_score is not None
                            else None
                        )
                        row[f"{prefix}.score"] = (
                            float(instance.score)
                            if instance.score is not None
                            else None
                        )
                    else:
                        row[f"{prefix}.track_score"] = None
                        row[f"{prefix}.score"] = None

                    for node_idx, node in enumerate(skeleton.nodes):
                        point = instance.points[node_idx]
                        row[f"{prefix}.{node.name}.x"] = float(point["xy"][0])
                        row[f"{prefix}.{node.name}.y"] = float(point["xy"][1])
                        if include_score and is_predicted:
                            row[f"{prefix}.{node.name}.score"] = float(point["score"])
                        elif include_score:
                            row[f"{prefix}.{node.name}.score"] = None
                else:
                    row[f"{prefix}.track_score"] = None
                    row[f"{prefix}.score"] = None
                    for node in skeleton.nodes:
                        row[f"{prefix}.{node.name}.x"] = np.nan
                        row[f"{prefix}.{node.name}.y"] = np.nan
                        if include_score:
                            row[f"{prefix}.{node.name}.score"] = np.nan

        yield row


def _iter_multi_index_rows(
    labels: Labels,
    labeled_frames: list,
    *,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    instance_id: str = "index",
    untracked: str = "error",
    max_instances: int,
    all_tracks: list,
    skeleton,
) -> Generator[dict, None, None]:
    """Yield row dicts for multi_index format.

    Note: This yields flat dicts with dot-separated column names.
    The caller is responsible for converting to MultiIndex if needed.
    """
    # Reuse frames iterator - the structure is the same
    yield from _iter_frames_rows(
        labels,
        labeled_frames,
        include_metadata=True,  # Always include for multi_index
        include_score=include_score,
        include_user_instances=include_user_instances,
        include_predicted_instances=include_predicted_instances,
        include_video=include_video,
        video_id=video_id,
        instance_id=instance_id,
        untracked=untracked,
        max_instances=max_instances,
        all_tracks=all_tracks,
        skeleton=skeleton,
    )


def _to_points_df(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    backend: str = "pandas",
) -> "pd.DataFrame | pl.DataFrame":
    """Convert to points format (one row per point).

    Column structure:
        frame_idx | video | track | track_score | instance_score | node | x | y | score
    """
    rows = []

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance in instances_to_process:
            is_predicted = isinstance(instance, PredictedInstance)

            for node_idx, node in enumerate(instance.skeleton.nodes):
                point = instance.points[node_idx]

                row = {
                    "frame_idx": int(lf.frame_idx),
                    "node": node.name,
                    "x": float(point["xy"][0]),
                    "y": float(point["xy"][1]),
                }

                if include_metadata:
                    # Add video info if requested
                    if include_video:
                        video_value = _format_video(lf.video, labels, video_id)
                        # Use appropriate column name based on video_id type
                        if video_id == "index":
                            row["video_idx"] = video_value
                        elif video_id == "object":
                            row["video"] = video_value
                        else:  # "path" or "name"
                            row["video_path"] = video_value

                    row["track"] = instance.track.name if instance.track else None

                    # Add track_score and instance_score for predicted instances
                    if is_predicted:
                        row["track_score"] = (
                            float(instance.tracking_score)
                            if instance.tracking_score is not None
                            else None
                        )
                        row["instance_score"] = (
                            float(instance.score)
                            if instance.score is not None
                            else None
                        )
                    else:
                        row["track_score"] = None
                        row["instance_score"] = None

                if include_score and is_predicted:
                    row["score"] = float(point["score"])

                rows.append(row)

    return _create_dataframe_from_rows(rows, backend)


def _to_instances_df(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    backend: str = "pandas",
) -> "pd.DataFrame | pl.DataFrame":
    """Convert to instances format (one row per instance).

    Column structure:
        frame_idx | video | track | track_score | score | {node}.x/y/score
    """
    rows = []

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance in instances_to_process:
            is_predicted = isinstance(instance, PredictedInstance)

            row = {"frame_idx": int(lf.frame_idx)}

            if include_metadata:
                # Add video info if requested
                if include_video:
                    video_value = _format_video(lf.video, labels, video_id)
                    if video_id == "index":
                        row["video_idx"] = video_value
                    elif video_id == "object":
                        row["video"] = video_value
                    else:  # "path" or "name"
                        row["video_path"] = video_value

                row["track"] = instance.track.name if instance.track else None

                # Add track_score and instance-level score for predicted instances
                if is_predicted:
                    row["track_score"] = (
                        float(instance.tracking_score)
                        if instance.tracking_score is not None
                        else None
                    )
                    row["score"] = (
                        float(instance.score) if instance.score is not None else None
                    )
                else:
                    row["track_score"] = None
                    row["score"] = None

            # Add columns for each node using dot separator
            for node_idx, node in enumerate(instance.skeleton.nodes):
                point = instance.points[node_idx]
                row[f"{node.name}.x"] = float(point["xy"][0])
                row[f"{node.name}.y"] = float(point["xy"][1])

                if include_score and is_predicted:
                    row[f"{node.name}.score"] = float(point["score"])

            rows.append(row)

    return _create_dataframe_from_rows(rows, backend)


# =============================================================================
# Lazy Fast Path Functions
# =============================================================================


def _to_points_df_lazy(
    store: "LazyDataStore",
    labels: Labels,
    *,
    video_filter: Optional[int] = None,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    backend: str = "pandas",
) -> "pd.DataFrame | pl.DataFrame":
    """Fast path for points format using raw LazyDataStore arrays.

    This builds the DataFrame directly from structured arrays without
    materializing any LabeledFrame or Instance objects.
    """
    from sleap_io.io.slp import InstanceType

    rows = []

    # Get node names from skeleton
    skeleton = store.skeletons[0] if store.skeletons else None
    if skeleton is None:
        return _create_dataframe_from_rows([], backend)
    node_names = skeleton.node_names

    # Build frame_id -> (video, frame_idx) mapping
    frame_lookup = {}
    for frame_row in store.frames_data:
        frame_id = int(frame_row["frame_id"])
        vid_id = int(frame_row["video"])
        frame_idx = int(frame_row["frame_idx"])
        frame_lookup[frame_id] = (vid_id, frame_idx)

    # Determine coordinate adjustment for legacy format
    coord_offset = 0.5 if store.format_id < 1.1 else 0.0

    # Iterate over instances
    for inst_idx, inst_row in enumerate(store.instances_data):
        instance_type = int(inst_row["instance_type"])

        # Filter by instance type
        is_user = instance_type == InstanceType.USER
        is_predicted = instance_type == InstanceType.PREDICTED

        if is_user and not include_user_instances:
            continue
        if is_predicted and not include_predicted_instances:
            continue

        # Get frame info
        frame_id = int(inst_row["frame_id"])
        if frame_id not in frame_lookup:
            continue
        vid_id, frame_idx = frame_lookup[frame_id]

        # Filter by video if requested
        if video_filter is not None and vid_id != video_filter:
            continue

        # Get instance metadata
        track_id = int(inst_row["track"])
        track_name = store.tracks[track_id].name if track_id >= 0 else None
        instance_score = float(inst_row["score"]) if is_predicted else None

        # Get tracking score
        tracking_score = float(inst_row["tracking_score"]) if is_predicted else None

        # Get points
        point_start = int(inst_row["point_id_start"])
        point_end = int(inst_row["point_id_end"])

        if is_user:
            pts_data = store.points_data[point_start:point_end]
        else:
            pts_data = store.pred_points_data[point_start:point_end]

        # Yield one row per point
        for node_idx, node_name in enumerate(node_names):
            if node_idx >= len(pts_data):
                continue  # Safety check

            pt = pts_data[node_idx]
            x = float(pt["x"]) - coord_offset
            y = float(pt["y"]) - coord_offset

            row: dict = {
                "frame_idx": frame_idx,
                "node": node_name,
                "x": x,
                "y": y,
            }

            if include_metadata:
                if include_video:
                    video_obj = store.videos[vid_id]
                    video_value = _format_video(video_obj, labels, video_id)
                    if video_id == "index":
                        row["video_idx"] = video_value
                    elif video_id == "object":
                        row["video"] = video_value
                    else:
                        row["video_path"] = video_value

                row["track"] = track_name

                if is_predicted:
                    row["track_score"] = tracking_score
                    row["instance_score"] = instance_score
                else:
                    row["track_score"] = None
                    row["instance_score"] = None

            if include_score and is_predicted:
                row["score"] = float(pt["score"])

            rows.append(row)

    return _create_dataframe_from_rows(rows, backend)


def _to_instances_df_lazy(
    store: "LazyDataStore",
    labels: Labels,
    *,
    video_filter: Optional[int] = None,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    backend: str = "pandas",
) -> "pd.DataFrame | pl.DataFrame":
    """Fast path for instances format using raw LazyDataStore arrays.

    This builds the DataFrame directly from structured arrays without
    materializing any LabeledFrame or Instance objects.
    """
    from sleap_io.io.slp import InstanceType

    rows = []

    # Get node names from skeleton
    skeleton = store.skeletons[0] if store.skeletons else None
    if skeleton is None:
        return _create_dataframe_from_rows([], backend)
    node_names = skeleton.node_names

    # Build frame_id -> (video, frame_idx) mapping
    frame_lookup = {}
    for frame_row in store.frames_data:
        frame_id = int(frame_row["frame_id"])
        vid_id = int(frame_row["video"])
        frame_idx = int(frame_row["frame_idx"])
        frame_lookup[frame_id] = (vid_id, frame_idx)

    # Determine coordinate adjustment for legacy format
    coord_offset = 0.5 if store.format_id < 1.1 else 0.0

    # Iterate over instances
    for inst_idx, inst_row in enumerate(store.instances_data):
        instance_type = int(inst_row["instance_type"])

        # Filter by instance type
        is_user = instance_type == InstanceType.USER
        is_predicted = instance_type == InstanceType.PREDICTED

        if is_user and not include_user_instances:
            continue
        if is_predicted and not include_predicted_instances:
            continue

        # Get frame info
        frame_id = int(inst_row["frame_id"])
        if frame_id not in frame_lookup:
            continue
        vid_id, frame_idx = frame_lookup[frame_id]

        # Filter by video if requested
        if video_filter is not None and vid_id != video_filter:
            continue

        # Get instance metadata
        track_id = int(inst_row["track"])
        track_name = store.tracks[track_id].name if track_id >= 0 else None
        instance_score = float(inst_row["score"]) if is_predicted else None

        # Get tracking score
        tracking_score = float(inst_row["tracking_score"]) if is_predicted else None

        # Build row
        row: dict = {"frame_idx": frame_idx}

        if include_metadata:
            if include_video:
                video_obj = store.videos[vid_id]
                video_value = _format_video(video_obj, labels, video_id)
                if video_id == "index":
                    row["video_idx"] = video_value
                elif video_id == "object":
                    row["video"] = video_value
                else:
                    row["video_path"] = video_value

            row["track"] = track_name

            if is_predicted:
                row["track_score"] = tracking_score
                row["score"] = instance_score
            else:
                row["track_score"] = None
                row["score"] = None

        # Get points and add node columns
        point_start = int(inst_row["point_id_start"])
        point_end = int(inst_row["point_id_end"])

        if is_user:
            pts_data = store.points_data[point_start:point_end]
        else:
            pts_data = store.pred_points_data[point_start:point_end]

        for node_idx, node_name in enumerate(node_names):
            if node_idx >= len(pts_data):
                row[f"{node_name}.x"] = np.nan
                row[f"{node_name}.y"] = np.nan
                if include_score:
                    row[f"{node_name}.score"] = None
                continue

            pt = pts_data[node_idx]
            row[f"{node_name}.x"] = float(pt["x"]) - coord_offset
            row[f"{node_name}.y"] = float(pt["y"]) - coord_offset

            if include_score and is_predicted:
                row[f"{node_name}.score"] = float(pt["score"])
            elif include_score:
                row[f"{node_name}.score"] = None

        rows.append(row)

    return _create_dataframe_from_rows(rows, backend)


def _to_frames_df(  # noqa: D417
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    instance_id: str = "index",
    untracked: str = "error",
    backend: str = "pandas",
) -> "pd.DataFrame | pl.DataFrame":
    """Convert to frames format (one row per frame, wide format).

    This format multiplexes all instances across columns for each frame.
    Other parameters mirror to_dataframe().

    Args:
        instance_id: How to name instance column prefixes.
            - "index": Use inst0, inst1, inst2, etc.
            - "track": Use track names (e.g., mouse1, mouse2).
        untracked: Behavior when encountering untracked instances in track mode.
            - "error": Raise error if any instance lacks a track.
            - "ignore": Skip untracked instances silently.

    Column structure (instance_id="index"):
        frame_idx | video | inst0.track | inst0.track_score | inst0.score |
        inst0.{node}.x | inst0.{node}.y | inst0.{node}.score | inst1.track | ...

    Column structure (instance_id="track"):
        frame_idx | video | {track}.track_score | {track}.score |
        {track}.{node}.x | {track}.{node}.y | {track}.{node}.score | ...
    """
    if not labeled_frames:
        return _create_dataframe_from_rows([], backend)

    # Use skeleton from labels (Labels always have at least one skeleton)
    skeleton = labels.skeletons[0]

    # Collect all data into a frame-indexed structure
    frame_data = {}  # (video, frame_idx) -> list of (instance, prefix)

    # Determine max instances per frame for padding (in index mode)
    max_instances = 0

    # If using track mode, collect all track names for consistent column ordering
    all_tracks = []
    if instance_id == "track":
        all_tracks = [t.name for t in labels.tracks]

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        # Filter/validate for track mode
        if instance_id == "track":
            filtered = []
            for inst in instances_to_process:
                if inst.track is None:
                    if untracked == "error":
                        raise ValueError(
                            f"Instance in frame {lf.frame_idx} has no track. "
                            "Use instance_id='index' or untracked='ignore'."
                        )
                    # Skip untracked instances in ignore mode
                else:
                    filtered.append(inst)
            instances_to_process = filtered

        key = (lf.video, lf.frame_idx)
        frame_data[key] = instances_to_process
        max_instances = max(max_instances, len(instances_to_process))

    # Build rows
    rows = []

    for (video, frame_idx), instances in frame_data.items():
        row = {"frame_idx": int(frame_idx)}

        if include_video:
            video_value = _format_video(video, labels, video_id)
            if video_id == "index":
                row["video_idx"] = video_value
            elif video_id == "object":
                row["video"] = video_value
            else:  # "path" or "name"
                row["video_path"] = video_value

        if instance_id == "index":
            # Instance-indexed mode: inst0, inst1, ...
            for inst_idx in range(max_instances):
                prefix = f"inst{inst_idx}"

                if inst_idx < len(instances):
                    instance = instances[inst_idx]
                    is_predicted = isinstance(instance, PredictedInstance)

                    # Instance metadata
                    row[f"{prefix}.track"] = (
                        instance.track.name if instance.track else None
                    )
                    if is_predicted:
                        row[f"{prefix}.track_score"] = (
                            float(instance.tracking_score)
                            if instance.tracking_score is not None
                            else None
                        )
                        row[f"{prefix}.score"] = (
                            float(instance.score)
                            if instance.score is not None
                            else None
                        )
                    else:
                        row[f"{prefix}.track_score"] = None
                        row[f"{prefix}.score"] = None

                    # Node coordinates
                    for node_idx, node in enumerate(skeleton.nodes):
                        point = instance.points[node_idx]
                        row[f"{prefix}.{node.name}.x"] = float(point["xy"][0])
                        row[f"{prefix}.{node.name}.y"] = float(point["xy"][1])
                        if include_score and is_predicted:
                            row[f"{prefix}.{node.name}.score"] = float(point["score"])
                        elif include_score:
                            row[f"{prefix}.{node.name}.score"] = None
                else:
                    # Pad with NaN for missing instances
                    row[f"{prefix}.track"] = None
                    row[f"{prefix}.track_score"] = None
                    row[f"{prefix}.score"] = None
                    for node in skeleton.nodes:
                        row[f"{prefix}.{node.name}.x"] = np.nan
                        row[f"{prefix}.{node.name}.y"] = np.nan
                        if include_score:
                            row[f"{prefix}.{node.name}.score"] = np.nan

        else:  # instance_id == "track"
            # Track-named mode: use track names as prefixes
            # Build mapping of track_name -> instance for this frame
            track_instances = {}
            for instance in instances:
                if instance.track:
                    track_instances[instance.track.name] = instance

            # Iterate through all tracks for consistent column ordering
            for track_name in all_tracks:
                prefix = track_name

                if track_name in track_instances:
                    instance = track_instances[track_name]
                    is_predicted = isinstance(instance, PredictedInstance)

                    # Instance metadata (no .track column in track mode)
                    if is_predicted:
                        row[f"{prefix}.track_score"] = (
                            float(instance.tracking_score)
                            if instance.tracking_score is not None
                            else None
                        )
                        row[f"{prefix}.score"] = (
                            float(instance.score)
                            if instance.score is not None
                            else None
                        )
                    else:
                        row[f"{prefix}.track_score"] = None
                        row[f"{prefix}.score"] = None

                    # Node coordinates
                    for node_idx, node in enumerate(skeleton.nodes):
                        point = instance.points[node_idx]
                        row[f"{prefix}.{node.name}.x"] = float(point["xy"][0])
                        row[f"{prefix}.{node.name}.y"] = float(point["xy"][1])
                        if include_score and is_predicted:
                            row[f"{prefix}.{node.name}.score"] = float(point["score"])
                        elif include_score:
                            row[f"{prefix}.{node.name}.score"] = None
                else:
                    # Track not present in this frame
                    row[f"{prefix}.track_score"] = None
                    row[f"{prefix}.score"] = None
                    for node in skeleton.nodes:
                        row[f"{prefix}.{node.name}.x"] = np.nan
                        row[f"{prefix}.{node.name}.y"] = np.nan
                        if include_score:
                            row[f"{prefix}.{node.name}.score"] = np.nan

        rows.append(row)

    df = _create_dataframe_from_rows(rows, backend)

    # Sort by frame_idx
    if backend == "polars":
        if not df.is_empty():
            df = df.sort("frame_idx")
    else:
        if not df.empty:
            df = df.sort_values("frame_idx").reset_index(drop=True)

    return df


def _to_multi_index_df(  # noqa: D417
    labels: Labels,
    labeled_frames: list,
    *,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    include_video: bool = True,
    video_id: str = "path",
    instance_id: str = "index",
    untracked: str = "error",
    backend: str = "pandas",
) -> "pd.DataFrame | pl.DataFrame":
    """Convert to multi-index format (hierarchical columns).

    This format uses hierarchical column structure similar to NWB format.
    The hierarchy is: instance -> node -> coordinate.
    Other parameters mirror to_dataframe().

    Args:
        instance_id: How to name instance column prefixes.
            - "index": Use inst0, inst1, inst2, etc.
            - "track": Use track names (e.g., mouse1, mouse2).
        untracked: Behavior when encountering untracked instances in track mode.
            - "error": Raise error if any instance lacks a track.
            - "ignore": Skip untracked instances silently.

    Column hierarchy (instance_id="index"):
        (inst0, inst1, ...) -> (track, track_score, score, {node}, ...) ->
            for nodes: (x, y, score)

    Column hierarchy (instance_id="track"):
        ({track}, ...) -> (track_score, score, {node}, ...) ->
            for nodes: (x, y, score)

    Index: frame_idx

    Note: For polars backend, hierarchical columns are flattened to dot-separated
    names (e.g., "inst0.nose.x") since polars doesn't support MultiIndex columns.
    """
    if not labeled_frames:
        return _create_dataframe_from_rows([], backend)

    # Use skeleton from labels (Labels always have at least one skeleton)
    skeleton = labels.skeletons[0]

    # Collect all data into a frame-indexed structure
    frame_data = {}  # (video, frame_idx) -> list of instances

    # Determine max instances per frame for padding (in index mode)
    max_instances = 0

    # If using track mode, collect all track names for consistent column ordering
    all_tracks = []
    if instance_id == "track":
        all_tracks = [t.name for t in labels.tracks]

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        # Filter/validate for track mode
        if instance_id == "track":
            filtered = []
            for inst in instances_to_process:
                if inst.track is None:
                    if untracked == "error":
                        raise ValueError(
                            f"Instance in frame {lf.frame_idx} has no track. "
                            "Use instance_id='index' or untracked='ignore'."
                        )
                    # Skip untracked instances in ignore mode
                else:
                    filtered.append(inst)
            instances_to_process = filtered

        key = (lf.video, lf.frame_idx)
        frame_data[key] = instances_to_process
        max_instances = max(max_instances, len(instances_to_process))

    # For polars backend, use flat column names directly (avoid tuple key overhead)
    use_flat_keys = backend == "polars"

    # Build rows
    rows = []

    for (video, frame_idx), instances in frame_data.items():
        row_dict = {}

        if instance_id == "index":
            # Instance-indexed mode: inst0, inst1, ...
            for inst_idx in range(max_instances):
                prefix = f"inst{inst_idx}"

                if inst_idx < len(instances):
                    instance = instances[inst_idx]
                    is_predicted = isinstance(instance, PredictedInstance)

                    # Instance metadata
                    if use_flat_keys:
                        row_dict[f"{prefix}.track"] = (
                            instance.track.name if instance.track else None
                        )
                        if is_predicted:
                            row_dict[f"{prefix}.track_score"] = (
                                float(instance.tracking_score)
                                if instance.tracking_score is not None
                                else None
                            )
                            row_dict[f"{prefix}.score"] = (
                                float(instance.score)
                                if instance.score is not None
                                else None
                            )
                        else:
                            row_dict[f"{prefix}.track_score"] = None
                            row_dict[f"{prefix}.score"] = None
                    else:
                        row_dict[(prefix, "track", "")] = (
                            instance.track.name if instance.track else None
                        )
                        if is_predicted:
                            row_dict[(prefix, "track_score", "")] = (
                                float(instance.tracking_score)
                                if instance.tracking_score is not None
                                else None
                            )
                            row_dict[(prefix, "score", "")] = (
                                float(instance.score)
                                if instance.score is not None
                                else None
                            )
                        else:
                            row_dict[(prefix, "track_score", "")] = None
                            row_dict[(prefix, "score", "")] = None

                    # Node coordinates
                    for node_idx, node in enumerate(skeleton.nodes):
                        point = instance.points[node_idx]
                        if use_flat_keys:
                            row_dict[f"{prefix}.{node.name}.x"] = float(point["xy"][0])
                            row_dict[f"{prefix}.{node.name}.y"] = float(point["xy"][1])
                            if include_score and is_predicted:
                                row_dict[f"{prefix}.{node.name}.score"] = float(
                                    point["score"]
                                )
                            elif include_score:
                                row_dict[f"{prefix}.{node.name}.score"] = None
                        else:
                            row_dict[(prefix, node.name, "x")] = float(point["xy"][0])
                            row_dict[(prefix, node.name, "y")] = float(point["xy"][1])
                            if include_score and is_predicted:
                                row_dict[(prefix, node.name, "score")] = float(
                                    point["score"]
                                )
                            elif include_score:
                                row_dict[(prefix, node.name, "score")] = None
                else:
                    # Pad with NaN for missing instances
                    if use_flat_keys:
                        row_dict[f"{prefix}.track"] = None
                        row_dict[f"{prefix}.track_score"] = None
                        row_dict[f"{prefix}.score"] = None
                        for node in skeleton.nodes:
                            row_dict[f"{prefix}.{node.name}.x"] = np.nan
                            row_dict[f"{prefix}.{node.name}.y"] = np.nan
                            if include_score:
                                row_dict[f"{prefix}.{node.name}.score"] = np.nan
                    else:
                        row_dict[(prefix, "track", "")] = None
                        row_dict[(prefix, "track_score", "")] = None
                        row_dict[(prefix, "score", "")] = None
                        for node in skeleton.nodes:
                            row_dict[(prefix, node.name, "x")] = np.nan
                            row_dict[(prefix, node.name, "y")] = np.nan
                            if include_score:
                                row_dict[(prefix, node.name, "score")] = np.nan

        else:  # instance_id == "track"
            # Track-named mode: use track names as prefixes
            # Build mapping of track_name -> instance for this frame
            track_instances = {}
            for instance in instances:
                if instance.track:
                    track_instances[instance.track.name] = instance

            # Iterate through all tracks for consistent column ordering
            for track_name in all_tracks:
                prefix = track_name

                if track_name in track_instances:
                    instance = track_instances[track_name]
                    is_predicted = isinstance(instance, PredictedInstance)

                    # Instance metadata (no track column in track mode)
                    if use_flat_keys:
                        if is_predicted:
                            row_dict[f"{prefix}.track_score"] = (
                                float(instance.tracking_score)
                                if instance.tracking_score is not None
                                else None
                            )
                            row_dict[f"{prefix}.score"] = (
                                float(instance.score)
                                if instance.score is not None
                                else None
                            )
                        else:
                            row_dict[f"{prefix}.track_score"] = None
                            row_dict[f"{prefix}.score"] = None
                    else:
                        if is_predicted:
                            row_dict[(prefix, "track_score", "")] = (
                                float(instance.tracking_score)
                                if instance.tracking_score is not None
                                else None
                            )
                            row_dict[(prefix, "score", "")] = (
                                float(instance.score)
                                if instance.score is not None
                                else None
                            )
                        else:
                            row_dict[(prefix, "track_score", "")] = None
                            row_dict[(prefix, "score", "")] = None

                    # Node coordinates
                    for node_idx, node in enumerate(skeleton.nodes):
                        point = instance.points[node_idx]
                        if use_flat_keys:
                            row_dict[f"{prefix}.{node.name}.x"] = float(point["xy"][0])
                            row_dict[f"{prefix}.{node.name}.y"] = float(point["xy"][1])
                            if include_score and is_predicted:
                                row_dict[f"{prefix}.{node.name}.score"] = float(
                                    point["score"]
                                )
                            elif include_score:
                                row_dict[f"{prefix}.{node.name}.score"] = None
                        else:
                            row_dict[(prefix, node.name, "x")] = float(point["xy"][0])
                            row_dict[(prefix, node.name, "y")] = float(point["xy"][1])
                            if include_score and is_predicted:
                                row_dict[(prefix, node.name, "score")] = float(
                                    point["score"]
                                )
                            elif include_score:
                                row_dict[(prefix, node.name, "score")] = None
                else:
                    # Track not present in this frame
                    if use_flat_keys:
                        row_dict[f"{prefix}.track_score"] = None
                        row_dict[f"{prefix}.score"] = None
                        for node in skeleton.nodes:
                            row_dict[f"{prefix}.{node.name}.x"] = np.nan
                            row_dict[f"{prefix}.{node.name}.y"] = np.nan
                            if include_score:
                                row_dict[f"{prefix}.{node.name}.score"] = np.nan
                    else:
                        row_dict[(prefix, "track_score", "")] = None
                        row_dict[(prefix, "score", "")] = None
                        for node in skeleton.nodes:
                            row_dict[(prefix, node.name, "x")] = np.nan
                            row_dict[(prefix, node.name, "y")] = np.nan
                            if include_score:
                                row_dict[(prefix, node.name, "score")] = np.nan

        # Add video info if requested (as part of index or separate)
        row_dict["frame_idx"] = int(frame_idx)
        if include_video:
            video_value = _format_video(video, labels, video_id)
            if video_id == "index":
                row_dict["video_idx"] = video_value
            elif video_id == "object":
                row_dict["video"] = str(video_value.filename)
            else:
                row_dict["video_path"] = video_value

        rows.append(row_dict)

    if not rows:
        return _create_dataframe_from_rows([], backend)

    # Handle polars backend: rows already have flat keys
    if backend == "polars":
        df = _create_dataframe_from_rows(rows, backend)
        df = df.sort("frame_idx")
        return df

    # For pandas, create DataFrame and set up multi-index columns
    df = pd.DataFrame(rows)

    # Set frame_idx as index
    df = df.set_index("frame_idx")

    # Extract tuple columns and regular columns
    tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
    regular_cols = [c for c in df.columns if not isinstance(c, tuple)]

    if tuple_cols:
        # Create multi-index for tuple columns
        multi_index = pd.MultiIndex.from_tuples(tuple_cols)

        # Separate the DataFrames
        if regular_cols:
            df_regular = df[regular_cols]
            df_multi = df[tuple_cols]
            df_multi.columns = multi_index
            # For now, keep regular columns separate (video info)
            # Combine them back
            df = pd.concat([df_regular, df_multi], axis=1)
        else:
            df.columns = multi_index

    df = df.sort_index()  # Sort by frame_idx

    return df


def from_dataframe(
    df: pd.DataFrame,
    *,
    video: Optional[Video] = None,
    skeleton: Optional["Skeleton"] = None,  # noqa: F821
    format: DataFrameFormat | str = DataFrameFormat.POINTS,
) -> Labels:
    """Create a Labels object from a DataFrame.

    This function reconstructs a Labels object from a DataFrame created by
    `to_dataframe()`. Supports all formats: points, instances, frames, multi_index.

    Args:
        df: DataFrame created by to_dataframe() or compatible structure.
        video: Video object to associate with all frames. Required if the DataFrame
            does not have video information.
        skeleton: Skeleton object to use. Required if the DataFrame does not have
            skeleton information or if the skeleton needs to be provided explicitly.
        format: The format of the input DataFrame. One of "points", "instances",
            "frames", "multi_index".

    Returns:
        A Labels object reconstructed from the DataFrame.

    Raises:
        ValueError: If required columns are missing or format is invalid.

    Examples:
        >>> df = to_dataframe(labels, format="points")
        >>> labels_restored = from_dataframe(df, video=video, skeleton=skeleton)

        >>> df = to_dataframe(labels, format="instances")
        >>> labels_restored = from_dataframe(df, format="instances", skeleton=skeleton)

    Notes:
        - The DataFrame must have the expected structure for the specified format.
        - If video information is not in the DataFrame, a Video must be provided.
        - If skeleton is not provided, it will be inferred from column names where
          possible.
        - Tracks are reconstructed from track/track_name columns if present.
    """
    # Normalize format parameter
    if isinstance(format, str):
        try:
            format = DataFrameFormat(format.lower())
        except ValueError:
            valid_formats = ", ".join([f.value for f in DataFrameFormat])
            raise ValueError(
                f"Invalid format '{format}'. Must be one of: {valid_formats}"
            )

    if format == DataFrameFormat.POINTS:
        return _from_points_df(df, video=video, skeleton=skeleton)
    elif format == DataFrameFormat.INSTANCES:
        return _from_instances_df(df, video=video, skeleton=skeleton)
    elif format == DataFrameFormat.FRAMES:
        return _from_frames_df(df, video=video, skeleton=skeleton)
    elif format == DataFrameFormat.MULTI_INDEX:
        return _from_multi_index_df(df, video=video, skeleton=skeleton)
    else:
        raise ValueError(f"Unknown format: {format}")


def _from_points_df(
    df: pd.DataFrame,
    *,
    video: Optional[Video] = None,
    skeleton: Optional["Skeleton"] = None,  # noqa: F821
) -> Labels:
    """Create Labels from a points format DataFrame (one row per point)."""
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.skeleton import Node, Skeleton

    # Handle both old and new column names
    # Old: node_name, track_name, skeleton_name, instance_type
    # New: node, track, track_score, instance_score
    node_col = "node" if "node" in df.columns else "node_name"
    track_col = "track" if "track" in df.columns else "track_name"

    # Validate required columns
    required_cols = ["frame_idx", node_col, "x", "y"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Determine video column
    video_col = None
    for col in ["video_path", "video_idx", "video"]:
        if col in df.columns:
            video_col = col
            break

    # Build video lookup
    videos = []
    video_lookup = {}

    if video_col:
        unique_videos = df[video_col].unique()
        for vid_val in unique_videos:
            if video_col == "video":
                # Already a Video object
                v = vid_val
            elif video_col == "video_idx":
                # Need the provided video or create placeholder
                if video is not None:
                    v = video
                else:
                    v = Video(filename=f"video_{vid_val}")
            else:  # video_path
                v = Video(filename=vid_val)
            videos.append(v)
            video_lookup[vid_val] = v
    else:
        if video is None:
            video = Video(filename="video.mp4")
        videos = [video]

    # Build skeleton
    if skeleton is not None:
        skeletons = [skeleton]
    elif "skeleton_name" in df.columns:
        # Legacy format with skeleton_name column
        skeleton_nodes = {}
        for _, row in df.drop_duplicates(subset=["skeleton_name", node_col]).iterrows():
            skel_name = row["skeleton_name"]
            if skel_name not in skeleton_nodes:
                skeleton_nodes[skel_name] = []
            skeleton_nodes[skel_name].append(row[node_col])

        skeletons = []
        for name, node_names in skeleton_nodes.items():
            nodes = [Node(name=n) for n in node_names]
            skeletons.append(Skeleton(nodes=nodes, name=name))
    else:
        # Create skeleton from unique node names
        node_names = df[node_col].unique().tolist()
        nodes = [Node(name=n) for n in node_names]
        skeletons = [Skeleton(nodes=nodes, name="skeleton")]

    # Build skeleton lookup by name
    skeleton_lookup = {s.name: s for s in skeletons}

    # Build tracks
    tracks = []
    track_lookup = {}

    if track_col in df.columns:
        unique_tracks = df[track_col].dropna().unique()
        for track_name in unique_tracks:
            track = Track(name=track_name)
            tracks.append(track)
            track_lookup[track_name] = track

    # Determine instance type column (optional for interoperability)
    has_instance_type = "instance_type" in df.columns
    has_score = "score" in df.columns
    has_instance_score = "instance_score" in df.columns
    has_track_score = "track_score" in df.columns

    # Group by frame (and video if present) to build labeled frames
    group_cols = ["frame_idx"]
    if video_col:
        group_cols.insert(0, video_col)

    labeled_frames = []

    for group_key, frame_df in df.groupby(group_cols, sort=False):
        # Handle groupby returning tuple for list of columns
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        if video_col:
            video_val, frame_idx = group_key
            vid = video_lookup[video_val]
        else:
            (frame_idx,) = group_key
            vid = videos[0]

        # Determine if we need to compute instance indices from node repetition
        has_track_info = (
            track_col in frame_df.columns and frame_df[track_col].notna().any()
        )

        need_instance_idx = not has_track_info

        if need_instance_idx:
            # Create instance index based on node repetition within the frame
            node_counts = frame_df[node_col].value_counts()
            n_instances = node_counts.max() if len(node_counts) > 0 else 0

            if n_instances > 0:
                frame_df = frame_df.copy()

                if has_instance_type:
                    # Assign instance indices within each instance_type group
                    instance_idx = []
                    for inst_type in frame_df["instance_type"].unique():
                        type_mask = frame_df["instance_type"] == inst_type
                        type_df = frame_df[type_mask]
                        node_counter = {}
                        for _, row in type_df.iterrows():
                            node = row[node_col]
                            if node not in node_counter:
                                node_counter[node] = 0
                            instance_idx.append(node_counter[node])
                            node_counter[node] += 1
                    frame_df["_instance_idx"] = instance_idx
                else:
                    # Simple case - assign indices based on node repetition
                    instance_idx = []
                    node_counter = {}
                    for _, row in frame_df.iterrows():
                        node = row[node_col]
                        if node not in node_counter:
                            node_counter[node] = 0
                        instance_idx.append(node_counter[node])
                        node_counter[node] += 1
                    frame_df["_instance_idx"] = instance_idx

        # Build instance grouping columns
        instance_group_cols = []
        if has_instance_type:
            instance_group_cols.append("instance_type")
        if has_track_info:
            instance_group_cols.append(track_col)
        if need_instance_idx and "_instance_idx" in frame_df.columns:
            instance_group_cols.append("_instance_idx")

        # Fallback if no grouping columns
        if not instance_group_cols:
            instance_group_cols = ["_instance_idx"]
            frame_df = frame_df.copy()
            frame_df["_instance_idx"] = 0

        instances = []

        for inst_key, inst_df in frame_df.groupby(
            instance_group_cols, sort=False, dropna=False
        ):
            # Determine skeleton
            if "skeleton_name" in inst_df.columns:
                skel_name = inst_df["skeleton_name"].iloc[0]
                skel = skeleton_lookup.get(skel_name, skeletons[0])
            else:
                skel = skeletons[0]

            # Determine track
            track = None
            if track_col in inst_df.columns:
                track_name = inst_df[track_col].iloc[0]
                if pd.notna(track_name) and track_name in track_lookup:
                    track = track_lookup[track_name]

            # Determine instance type
            is_predicted = True  # Default to predicted
            if has_instance_type:
                inst_type = inst_df["instance_type"].iloc[0]
                is_predicted = inst_type == "predicted"

            # Get instance-level scores if available
            instance_score = 1.0
            tracking_score = None
            if has_instance_score and "instance_score" in inst_df.columns:
                score_val = inst_df["instance_score"].iloc[0]
                if pd.notna(score_val):
                    instance_score = float(score_val)
            if has_track_score and "track_score" in inst_df.columns:
                score_val = inst_df["track_score"].iloc[0]
                if pd.notna(score_val):
                    tracking_score = float(score_val)

            # Build points array
            n_nodes = len(skel.nodes)
            points_data = np.full((n_nodes, 2), np.nan, dtype="float64")
            scores = np.ones(n_nodes, dtype="float32") if has_score else None

            for _, row in inst_df.iterrows():
                node_name = row[node_col]
                try:
                    node_idx = skel.index(node_name)
                    points_data[node_idx, 0] = row["x"]
                    points_data[node_idx, 1] = row["y"]
                    if has_score and scores is not None and "score" in row:
                        score_val = row["score"]
                        if pd.notna(score_val):
                            scores[node_idx] = float(score_val)
                except (ValueError, KeyError):
                    pass

            # Create instance
            if is_predicted:
                instance = PredictedInstance.from_numpy(
                    points_data=points_data,
                    skeleton=skel,
                    point_scores=scores,
                    score=instance_score,
                    track=track,
                    tracking_score=tracking_score,
                )
            else:
                instance = Instance.from_numpy(
                    points_data=points_data,
                    skeleton=skel,
                    track=track,
                )

            instances.append(instance)

        labeled_frame = LabeledFrame(
            video=vid,
            frame_idx=int(frame_idx),
            instances=instances,
        )
        labeled_frames.append(labeled_frame)

    # Create Labels object
    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
    )

    return labels


def _from_instances_df(
    df: pd.DataFrame,
    *,
    video: Optional[Video] = None,
    skeleton: Optional["Skeleton"] = None,  # noqa: F821
) -> Labels:
    """Create Labels from an instances format DataFrame (one row per instance)."""
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.skeleton import Node, Skeleton

    # Validate required columns
    if "frame_idx" not in df.columns:
        raise ValueError("Missing required column: frame_idx")

    # Detect node columns - look for {node}.x patterns
    node_cols = {}  # node_name -> {"x": col, "y": col, "score": col}

    for col in df.columns:
        if "." in col:
            parts = col.rsplit(".", 1)
            if len(parts) == 2 and parts[1] in ("x", "y", "score"):
                node_name = parts[0]
                if node_name not in node_cols:
                    node_cols[node_name] = {}
                node_cols[node_name][parts[1]] = col

    if not node_cols:
        raise ValueError("No node columns found. Expected {node}.x format.")

    # Validate that we have x and y for each node
    for node_name, coords in node_cols.items():
        if "x" not in coords or "y" not in coords:
            raise ValueError(f"Node '{node_name}' missing x or y coordinate column.")

    # Handle track column (new: "track", old: "track_name")
    track_col = "track" if "track" in df.columns else "track_name"

    # Determine video column
    video_col = None
    for col in ["video_path", "video_idx", "video"]:
        if col in df.columns:
            video_col = col
            break

    # Build video lookup
    videos = []
    video_lookup = {}

    if video_col:
        unique_videos = df[video_col].unique()
        for vid_val in unique_videos:
            if video_col == "video":
                v = vid_val
            elif video_col == "video_idx":
                if video is not None:
                    v = video
                else:
                    v = Video(filename=f"video_{vid_val}")
            else:
                v = Video(filename=vid_val)
            videos.append(v)
            video_lookup[vid_val] = v
    else:
        if video is None:
            video = Video(filename="video.mp4")
        videos = [video]

    # Build skeleton
    if skeleton is not None:
        skeletons = [skeleton]
    else:
        node_names = list(node_cols.keys())
        nodes = [Node(name=n) for n in node_names]
        skeletons = [Skeleton(nodes=nodes, name="skeleton")]

    skel = skeletons[0]

    # Build tracks
    tracks = []
    track_lookup = {}

    if track_col in df.columns:
        unique_tracks = df[track_col].dropna().unique()
        for track_name in unique_tracks:
            track = Track(name=track_name)
            tracks.append(track)
            track_lookup[track_name] = track

    # Determine score columns
    has_score = any("score" in coords for coords in node_cols.values())
    has_instance_score = "score" in df.columns
    has_track_score = "track_score" in df.columns

    # Group by frame (and video if present)
    group_cols = ["frame_idx"]
    if video_col:
        group_cols.insert(0, video_col)

    labeled_frames = []

    for group_key, frame_df in df.groupby(group_cols, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        if video_col:
            video_val, frame_idx = group_key
            vid = video_lookup[video_val]
        else:
            (frame_idx,) = group_key
            vid = videos[0]

        instances = []

        for _, row in frame_df.iterrows():
            # Determine track
            track = None
            if track_col in row and pd.notna(row[track_col]):
                track_name = row[track_col]
                if track_name in track_lookup:
                    track = track_lookup[track_name]

            # Get instance-level scores
            instance_score = 1.0
            tracking_score = None
            if has_instance_score and pd.notna(row.get("score")):
                instance_score = float(row["score"])
            if has_track_score and pd.notna(row.get("track_score")):
                tracking_score = float(row["track_score"])

            # Build points array
            n_nodes = len(skel.nodes)
            points_data = np.full((n_nodes, 2), np.nan, dtype="float64")
            scores = np.ones(n_nodes, dtype="float32") if has_score else None

            for node_idx, node in enumerate(skel.nodes):
                if node.name in node_cols:
                    coords = node_cols[node.name]
                    x_val = row.get(coords["x"])
                    y_val = row.get(coords["y"])
                    if pd.notna(x_val) and pd.notna(y_val):
                        points_data[node_idx, 0] = float(x_val)
                        points_data[node_idx, 1] = float(y_val)
                    if has_score and scores is not None and "score" in coords:
                        score_val = row.get(coords["score"])
                        if pd.notna(score_val):
                            scores[node_idx] = float(score_val)

            # Determine if predicted (default to predicted if we have scores)
            is_predicted = has_score or has_instance_score

            if is_predicted:
                instance = PredictedInstance.from_numpy(
                    points_data=points_data,
                    skeleton=skel,
                    point_scores=scores,
                    score=instance_score,
                    track=track,
                    tracking_score=tracking_score,
                )
            else:
                instance = Instance.from_numpy(
                    points_data=points_data,
                    skeleton=skel,
                    track=track,
                )

            instances.append(instance)

        labeled_frame = LabeledFrame(
            video=vid,
            frame_idx=int(frame_idx),
            instances=instances,
        )
        labeled_frames.append(labeled_frame)

    return Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
    )


def _from_frames_df(
    df: pd.DataFrame,
    *,
    video: Optional[Video] = None,
    skeleton: Optional["Skeleton"] = None,  # noqa: F821
) -> Labels:
    """Create Labels from a frames format DataFrame (one row per frame, wide format)."""
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.skeleton import Node, Skeleton

    # Validate required columns
    if "frame_idx" not in df.columns:
        raise ValueError("Missing required column: frame_idx")

    # Detect instance columns - look for {inst}.{node}.x patterns or {inst}.track
    # Instance prefixes can be inst0, inst1, ... or track names like mouse1, mouse2
    # Structure: prefix -> {"track", "track_score", "score", nodes: {name: {x,y,score}}}
    instance_data: dict = {}

    for col in df.columns:
        if "." not in col:
            continue

        parts = col.split(".")
        if len(parts) == 2:
            # {inst}.track, {inst}.track_score, {inst}.score
            prefix, attr = parts
            if attr in ("track", "track_score", "score"):
                if prefix not in instance_data:
                    instance_data[prefix] = {"nodes": {}}
                instance_data[prefix][attr] = col
        elif len(parts) == 3:
            # {inst}.{node}.x, {inst}.{node}.y, {inst}.{node}.score
            prefix, node_name, coord = parts
            if coord in ("x", "y", "score"):
                if prefix not in instance_data:
                    instance_data[prefix] = {"nodes": {}}
                if node_name not in instance_data[prefix]["nodes"]:
                    instance_data[prefix]["nodes"][node_name] = {}
                instance_data[prefix]["nodes"][node_name][coord] = col

    if not instance_data:
        raise ValueError("No instance columns found. Expected {inst}.{node}.x format.")

    # Determine if this is index mode (inst0, inst1) or track mode (track names)
    is_index_mode = all(
        p.startswith("inst") and p[4:].isdigit() for p in instance_data.keys()
    )

    # Determine video column
    video_col = None
    for col in ["video_path", "video_idx", "video"]:
        if col in df.columns:
            video_col = col
            break

    # Build video lookup
    videos = []
    video_lookup = {}

    if video_col:
        unique_videos = df[video_col].unique()
        for vid_val in unique_videos:
            if video_col == "video":
                v = vid_val
            elif video_col == "video_idx":
                if video is not None:
                    v = video
                else:
                    v = Video(filename=f"video_{vid_val}")
            else:
                v = Video(filename=vid_val)
            videos.append(v)
            video_lookup[vid_val] = v
    else:
        if video is None:
            video = Video(filename="video.mp4")
        videos = [video]

    # Build skeleton from first instance's nodes
    if skeleton is not None:
        skeletons = [skeleton]
    else:
        first_instance = next(iter(instance_data.values()))
        node_names = list(first_instance["nodes"].keys())
        nodes = [Node(name=n) for n in node_names]
        skeletons = [Skeleton(nodes=nodes, name="skeleton")]

    skel = skeletons[0]

    # Build tracks
    tracks = []
    track_lookup = {}

    # In track mode, instance prefixes are track names
    if not is_index_mode:
        for prefix in instance_data.keys():
            track = Track(name=prefix)
            tracks.append(track)
            track_lookup[prefix] = track
    else:
        # In index mode, collect track names from {inst}.track columns
        for prefix, data in instance_data.items():
            if "track" in data:
                track_col_name = data["track"]
                unique_tracks = df[track_col_name].dropna().unique()
                for track_name in unique_tracks:
                    if track_name not in track_lookup:
                        track = Track(name=track_name)
                        tracks.append(track)
                        track_lookup[track_name] = track

    # Determine if we have scores
    has_score = any(
        "score" in coords
        for data in instance_data.values()
        for coords in data["nodes"].values()
    )

    labeled_frames = []

    for _, row in df.iterrows():
        frame_idx = int(row["frame_idx"])

        # Determine video
        if video_col:
            video_val = row[video_col]
            vid = video_lookup.get(video_val, videos[0])
        else:
            vid = videos[0]

        instances = []

        for prefix, data in instance_data.items():
            # Check if this instance has any valid data
            has_valid_data = False
            for node_name, coords in data["nodes"].items():
                x_col = coords.get("x")
                y_col = coords.get("y")
                if x_col and y_col:
                    x_val = row.get(x_col)
                    y_val = row.get(y_col)
                    if pd.notna(x_val) and pd.notna(y_val):
                        has_valid_data = True
                        break

            if not has_valid_data:
                continue

            # Determine track
            track = None
            if not is_index_mode:
                # Track name is the prefix
                track = track_lookup.get(prefix)
            elif "track" in data:
                track_name = row.get(data["track"])
                if pd.notna(track_name) and track_name in track_lookup:
                    track = track_lookup[track_name]

            # Get instance-level scores
            instance_score = 1.0
            tracking_score = None
            if "score" in data:
                score_val = row.get(data["score"])
                if pd.notna(score_val):
                    instance_score = float(score_val)
            if "track_score" in data:
                score_val = row.get(data["track_score"])
                if pd.notna(score_val):
                    tracking_score = float(score_val)

            # Build points array
            n_nodes = len(skel.nodes)
            points_data = np.full((n_nodes, 2), np.nan, dtype="float64")
            scores = np.ones(n_nodes, dtype="float32") if has_score else None

            for node_idx, node in enumerate(skel.nodes):
                if node.name in data["nodes"]:
                    coords = data["nodes"][node.name]
                    x_val = row.get(coords.get("x"))
                    y_val = row.get(coords.get("y"))
                    if pd.notna(x_val) and pd.notna(y_val):
                        points_data[node_idx, 0] = float(x_val)
                        points_data[node_idx, 1] = float(y_val)
                    if has_score and scores is not None and "score" in coords:
                        score_val = row.get(coords["score"])
                        if pd.notna(score_val):
                            scores[node_idx] = float(score_val)

            # Create predicted instance (frames format typically has predictions)
            instance = PredictedInstance.from_numpy(
                points_data=points_data,
                skeleton=skel,
                point_scores=scores,
                score=instance_score,
                track=track,
                tracking_score=tracking_score,
            )

            instances.append(instance)

        if instances:
            labeled_frame = LabeledFrame(
                video=vid,
                frame_idx=frame_idx,
                instances=instances,
            )
            labeled_frames.append(labeled_frame)

    return Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
    )


def _from_multi_index_df(
    df: pd.DataFrame,
    *,
    video: Optional[Video] = None,
    skeleton: Optional["Skeleton"] = None,  # noqa: F821
) -> Labels:
    """Create Labels from a multi-index format DataFrame (hierarchical columns)."""
    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten the multi-index into dot-separated column names
        df_flat = df.copy()
        df_flat.columns = [
            ".".join(str(c) for c in col if c).strip(".") for col in df.columns.values
        ]
        # Reset index to get frame_idx as a column
        df_flat = df_flat.reset_index()

        # Now delegate to frames decoder which handles the same structure
        return _from_frames_df(df_flat, video=video, skeleton=skeleton)
    else:
        # Already flat columns with frame_idx as column
        # Check if index is frame_idx
        if df.index.name == "frame_idx":
            df = df.reset_index()
        return _from_frames_df(df, video=video, skeleton=skeleton)
