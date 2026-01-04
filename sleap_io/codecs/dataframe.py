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
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labels import Labels
from sleap_io.model.video import Video

if TYPE_CHECKING:
    from typing_extensions import Literal

# Optional polars support
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
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


def to_dataframe(
    labels: Labels,
    format: DataFrameFormat | str = DataFrameFormat.POINTS,
    *,
    video: Optional[Video | int] = None,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
    backend: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | "pl.DataFrame":
    """Convert Labels to a DataFrame.

    Args:
        labels: Labels object to convert.
        format: Output format. One of "points", "instances", "frames", "multi_index".
        video: Optional video filter. If specified, only frames from this video
            are included. Can be a Video object or integer index.
        include_metadata: Include skeleton, track, video information in columns.
        include_score: Include confidence scores for predicted instances.
        include_user_instances: Include user-labeled instances.
        include_predicted_instances: Include predicted instances.
        backend: "pandas" or "polars". Polars requires the polars package.

    Returns:
        DataFrame in the specified format. Type depends on backend parameter.

    Raises:
        ValueError: If an invalid format is specified or polars is requested but
            not installed.

    Examples:
        >>> labels = load_file("predictions.slp")
        >>> df = to_dataframe(labels, format="points")
        >>> df.head()
           frame_idx  video_path  track_name  node_name     x     y  score
        0          0  video.mp4      track0       nose  10.0  20.0   0.95
        1          0  video.mp4      track0       tail   5.0   8.0   0.92

        >>> df = to_dataframe(labels, format="instances")
        >>> df.head()
           frame_idx  track_name  nose_x  nose_y  tail_x  tail_y
        0          0      track0    10.0    20.0     5.0     8.0

        >>> df = to_dataframe(labels, format="multi_index")
        >>> df.columns
        MultiIndex([('video.mp4', 'skeleton1', 'track0', 'nose', 'x'),
                    ('video.mp4', 'skeleton1', 'track0', 'nose', 'y'), ...])

    Notes:
        The specific columns and structure depend on the format parameter.
        See the DataFrameFormat enum documentation for details on each format.
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

    # Route to appropriate converter based on format
    if format == DataFrameFormat.POINTS:
        df = _to_points_df(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
        )
    elif format == DataFrameFormat.INSTANCES:
        df = _to_instances_df(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
        )
    elif format == DataFrameFormat.FRAMES:
        df = _to_frames_df(
            labels,
            labeled_frames,
            include_metadata=include_metadata,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
        )
    elif format == DataFrameFormat.MULTI_INDEX:
        df = _to_multi_index_df(
            labels,
            labeled_frames,
            include_score=include_score,
            include_user_instances=include_user_instances,
            include_predicted_instances=include_predicted_instances,
        )
    else:
        raise ValueError(f"Unknown format: {format}")

    # Convert to polars if requested
    if backend == "polars":
        df = pl.from_pandas(df)

    return df


def _to_points_df(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
) -> pd.DataFrame:
    """Convert to points format (one row per point)."""
    rows = []

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance_idx, instance in enumerate(instances_to_process):
            is_predicted = isinstance(instance, PredictedInstance)

            for node_idx, node in enumerate(instance.skeleton.nodes):
                point = instance.points[node_idx]

                row = {
                    "frame_idx": int(lf.frame_idx),
                    "node_name": node.name,
                    "x": float(point["xy"][0]),
                    "y": float(point["xy"][1]),
                }

                if include_metadata:
                    row["video_path"] = lf.video.filename
                    row["skeleton_name"] = instance.skeleton.name
                    row["track_name"] = (
                        instance.track.name if instance.track else None
                    )
                    row["instance_type"] = "predicted" if is_predicted else "user"

                if include_score and is_predicted:
                    row["score"] = float(point["score"])

                rows.append(row)

    return pd.DataFrame(rows)


def _to_instances_df(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
) -> pd.DataFrame:
    """Convert to instances format (one row per instance)."""
    rows = []

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance_idx, instance in enumerate(instances_to_process):
            is_predicted = isinstance(instance, PredictedInstance)

            row = {"frame_idx": int(lf.frame_idx)}

            if include_metadata:
                row["video_path"] = lf.video.filename
                row["skeleton_name"] = instance.skeleton.name
                row["track_name"] = instance.track.name if instance.track else None
                row["instance_idx"] = instance_idx
                row["instance_type"] = "predicted" if is_predicted else "user"

            # Add columns for each node
            for node_idx, node in enumerate(instance.skeleton.nodes):
                point = instance.points[node_idx]
                row[f"{node.name}_x"] = float(point["xy"][0])
                row[f"{node.name}_y"] = float(point["xy"][1])

                if include_score and is_predicted:
                    row[f"{node.name}_score"] = float(point["score"])

            rows.append(row)

    return pd.DataFrame(rows)


def _to_frames_df(
    labels: Labels,
    labeled_frames: list,
    *,
    include_metadata: bool = True,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
) -> pd.DataFrame:
    """Convert to frames format (one row per frame-track combination)."""
    rows = []

    # Build a mapping of (video, frame_idx, track) -> instance
    frame_track_map = {}

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance in instances_to_process:
            if instance.track is not None:
                key = (lf.video, lf.frame_idx, instance.track)
                # Prefer user instances over predicted
                if key not in frame_track_map or isinstance(instance, Instance):
                    frame_track_map[key] = (lf, instance)

    # Convert to rows
    for (video, frame_idx, track), (lf, instance) in frame_track_map.items():
        is_predicted = isinstance(instance, PredictedInstance)

        row = {
            "frame_idx": int(frame_idx),
            "track_idx": labels.tracks.index(track),
            "track_name": track.name,
        }

        if include_metadata:
            row["video_path"] = video.filename
            row["skeleton_name"] = instance.skeleton.name
            row["instance_type"] = "predicted" if is_predicted else "user"

        # Add columns for each node
        for node_idx, node in enumerate(instance.skeleton.nodes):
            point = instance.points[node_idx]
            row[f"{node.name}_x"] = float(point["xy"][0])
            row[f"{node.name}_y"] = float(point["xy"][1])

            if include_score and is_predicted:
                row[f"{node.name}_score"] = float(point["score"])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by frame_idx and track_idx
    if not df.empty:
        df = df.sort_values(["frame_idx", "track_idx"]).reset_index(drop=True)

    return df


def _to_multi_index_df(
    labels: Labels,
    labeled_frames: list,
    *,
    include_score: bool = True,
    include_user_instances: bool = True,
    include_predicted_instances: bool = True,
) -> pd.DataFrame:
    """Convert to multi-index format (hierarchical columns)."""
    # This format is similar to the NWB predictions format
    # Columns: (video_path, skeleton_name, track_name, node_name, coord)
    # Index: frame_idx

    # First collect all data
    data_list = []

    for lf in labeled_frames:
        # Collect instances to include
        instances_to_process = []

        if include_user_instances:
            instances_to_process.extend(lf.user_instances)
        if include_predicted_instances:
            instances_to_process.extend(lf.predicted_instances)

        for instance in instances_to_process:
            is_predicted = isinstance(instance, PredictedInstance)
            skeleton = instance.skeleton

            for node in skeleton.nodes:
                row_dict = {
                    "frame_idx": int(lf.frame_idx),
                    "x": float(instance[node]["xy"][0]),
                    "y": float(instance[node]["xy"][1]),
                    "node_name": node.name,
                    "skeleton_name": skeleton.name,
                    "track_name": instance.track.name if instance.track else "untracked",
                    "video_path": lf.video.filename,
                }

                if include_score and is_predicted:
                    row_dict["score"] = float(instance[node]["score"])

                data_list.append(row_dict)

    if not data_list:
        # Return empty DataFrame with expected structure
        return pd.DataFrame()

    df = pd.DataFrame(data_list)

    # Create multi-index structure
    index_cols = ["skeleton_name", "track_name", "node_name", "video_path", "frame_idx"]

    # Determine value columns
    value_cols = ["x", "y"]
    if include_score and "score" in df.columns:
        value_cols.append("score")

    df_tidy = (
        df.set_index(index_cols)
        .unstack(level=[0, 1, 2, 3])
        .swaplevel(0, -1, axis=1)  # video_path on top, coords on bottom
        .sort_index(axis=1)  # Sort columns
        .sort_index(axis=0)  # Sort by frame_idx
    )

    return df_tidy
