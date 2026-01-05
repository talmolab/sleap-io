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

from sleap_io.model.instance import Instance, PredictedInstance, Track
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
    video_id: Literal["path", "index", "name", "object"] = "path",
    include_video: Optional[bool] = None,
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
        video_id: How to represent videos in the DataFrame. Options:
            - "path": Full filename/path (default). Works for all video types.
            - "index": Integer video index. Compact, requires video list for decoding.
            - "name": Just the video filename (no directory). May not be unique.
            - "object": Store Video object directly. Not serializable but preserves
              all video metadata (dataset for HDF5, frame paths for ImageVideo).
        include_video: Whether to include video information. If None (default),
            automatically includes video info if there are multiple videos or if
            video metadata is needed. Set False to always omit, True to always include.
        backend: "pandas" or "polars". Polars requires the polars package.

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
           frame_idx  video_path  track_name  node_name     x     y  score
        0          0  video.mp4      track0       nose  10.0  20.0   0.95
        1          0  video.mp4      track0       tail   5.0   8.0   0.92

        Use video index instead of path (more compact):

        >>> df = to_dataframe(labels, format="points", video_id="index")
        >>> df.head()
           frame_idx  video_idx  track_name  node_name     x     y  score
        0          0          0      track0       nose  10.0  20.0   0.95

        Single video - omit video column entirely:

        >>> df = to_dataframe(labels, format="points", include_video=False)
        >>> df.head()
           frame_idx  track_name  node_name     x     y  score
        0          0      track0       nose  10.0  20.0   0.95

        Use video object (preserves HDF5 dataset, ImageVideo frame paths):

        >>> df = to_dataframe(labels, format="points", video_id="object")
        >>> df["video"].iloc[0]  # Full Video object with all metadata
        <Video: video.mp4>

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

    # Determine whether to include video info
    if include_video is None:
        # Auto-detect: include if multiple videos, unless explicitly omitted
        include_video = len(labels.videos) > 1

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
        )
    else:
        raise ValueError(f"Unknown format: {format}")

    # Convert to polars if requested
    if backend == "polars":
        df = pl.from_pandas(df)

    return df


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

                    row["skeleton_name"] = instance.skeleton.name
                    row["track_name"] = instance.track.name if instance.track else None
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
    include_video: bool = True,
    video_id: str = "path",
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
                # Add video info if requested
                if include_video:
                    video_value = _format_video(lf.video, labels, video_id)
                    if video_id == "index":
                        row["video_idx"] = video_value
                    elif video_id == "object":
                        row["video"] = video_value
                    else:  # "path" or "name"
                        row["video_path"] = video_value

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
    include_video: bool = True,
    video_id: str = "path",
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
            # Add video info if requested
            if include_video:
                video_value = _format_video(video, labels, video_id)
                if video_id == "index":
                    row["video_idx"] = video_value
                elif video_id == "object":
                    row["video"] = video_value
                else:  # "path" or "name"
                    row["video_path"] = video_value

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
    include_video: bool = True,
    video_id: str = "path",
) -> pd.DataFrame:
    """Convert to multi-index format (hierarchical columns)."""
    # This format is similar to the NWB predictions format
    # Columns: (video_id, skeleton_name, track_name, node_name, coord)
    # Index: frame_idx
    # Note: video level is omitted if include_video=False

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
                    "track_name": instance.track.name
                    if instance.track
                    else "untracked",
                }

                # Add video info if requested
                if include_video:
                    video_value = _format_video(lf.video, labels, video_id)
                    # For multi-index, always use consistent column name
                    if video_id == "index":
                        row_dict["video_idx"] = video_value
                    elif video_id == "object":
                        # For objects, convert to string representation for multi-index
                        row_dict["video_path"] = str(video_value.filename)
                    else:  # "path" or "name"
                        row_dict["video_path"] = video_value

                if include_score and is_predicted:
                    row_dict["score"] = float(instance[node]["score"])

                data_list.append(row_dict)

    if not data_list:
        # Return empty DataFrame with expected structure
        return pd.DataFrame()

    df = pd.DataFrame(data_list)

    # Create multi-index structure
    # Build index columns based on whether video is included
    if include_video:
        # Determine which video column was used
        video_col = "video_idx" if "video_idx" in df.columns else "video_path"
        index_cols = [
            "skeleton_name",
            "track_name",
            "node_name",
            video_col,
            "frame_idx",
        ]
        unstack_levels = [0, 1, 2, 3]  # All except frame_idx
    else:
        index_cols = ["skeleton_name", "track_name", "node_name", "frame_idx"]
        unstack_levels = [0, 1, 2]  # All except frame_idx

    # Determine value columns
    value_cols = ["x", "y"]
    if include_score and "score" in df.columns:
        value_cols.append("score")

    df_tidy = df.set_index(index_cols).unstack(level=unstack_levels)

    # For multi-level columns, put the most specific (coords) on the bottom
    if include_video:
        df_tidy = df_tidy.swaplevel(0, -1, axis=1)  # video on top, coords on bottom

    df_tidy = (
        df_tidy.sort_index(axis=1).sort_index(  # Sort columns
            axis=0
        )  # Sort by frame_idx
    )

    return df_tidy


def from_dataframe(
    df: pd.DataFrame,
    *,
    video: Optional[Video] = None,
    skeleton: Optional["Skeleton"] = None,  # noqa: F821
    format: DataFrameFormat | str = DataFrameFormat.POINTS,
) -> Labels:
    """Create a Labels object from a DataFrame.

    This function reconstructs a Labels object from a DataFrame created by
    `to_dataframe()`. Currently supports the "points" format.

    Args:
        df: DataFrame in points format with columns for frame_idx, node_name, x, y,
            and optionally video_path/video_idx, track_name, skeleton_name, score.
        video: Video object to associate with all frames. Required if the DataFrame
            does not have video information.
        skeleton: Skeleton object to use. Required if the DataFrame does not have
            skeleton_name column or if the skeleton needs to be provided explicitly.
        format: The format of the input DataFrame. Currently only "points" is supported.

    Returns:
        A Labels object reconstructed from the DataFrame.

    Raises:
        ValueError: If required columns are missing or format is not supported.
        NotImplementedError: If a format other than "points" is specified.

    Examples:
        >>> df = to_dataframe(labels, format="points")
        >>> labels_restored = from_dataframe(df, video=video, skeleton=skeleton)

        >>> # With video and skeleton inferred from DataFrame
        >>> df = to_dataframe(labels, format="points", include_metadata=True)
        >>> labels_restored = from_dataframe(df)

    Notes:
        - The DataFrame must have at minimum: frame_idx, node_name, x, y columns.
        - If video information is not in the DataFrame, a Video must be provided.
        - If skeleton is not provided and no skeleton_name column exists, a default
          skeleton will be created from the unique node names.
        - Tracks are reconstructed from track_name column if present.
        - Instance types (user vs predicted) are preserved if instance_type column
          exists.
    """
    from sleap_io.model.labeled_frame import LabeledFrame
    from sleap_io.model.skeleton import Node, Skeleton

    # Normalize format parameter
    if isinstance(format, str):
        try:
            format = DataFrameFormat(format.lower())
        except ValueError:
            valid_formats = ", ".join([f.value for f in DataFrameFormat])
            raise ValueError(
                f"Invalid format '{format}'. Must be one of: {valid_formats}"
            )

    if format != DataFrameFormat.POINTS:
        raise NotImplementedError(
            f"from_dataframe currently only supports 'points' format, "
            f"got '{format.value}'"
        )

    # Validate required columns
    required_cols = ["frame_idx", "node_name", "x", "y"]
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
        # Create skeletons from unique combinations
        # Group nodes by skeleton
        skeleton_nodes = {}
        for _, row in df.drop_duplicates(
            subset=["skeleton_name", "node_name"]
        ).iterrows():
            skel_name = row["skeleton_name"]
            if skel_name not in skeleton_nodes:
                skeleton_nodes[skel_name] = []
            skeleton_nodes[skel_name].append(row["node_name"])

        skeletons = []
        for name, node_names in skeleton_nodes.items():
            nodes = [Node(name=n) for n in node_names]
            skeletons.append(Skeleton(nodes=nodes, name=name))
    else:
        # Create skeleton from unique node names
        node_names = df["node_name"].unique().tolist()
        nodes = [Node(name=n) for n in node_names]
        skeletons = [Skeleton(nodes=nodes, name="skeleton")]

    # Build skeleton lookup by name
    skeleton_lookup = {s.name: s for s in skeletons}

    # Build tracks
    tracks = []
    track_lookup = {}

    if "track_name" in df.columns:
        unique_tracks = df["track_name"].dropna().unique()
        for track_name in unique_tracks:
            track = Track(name=track_name)
            tracks.append(track)
            track_lookup[track_name] = track

    # Determine instance type column
    has_instance_type = "instance_type" in df.columns
    has_score = "score" in df.columns

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

        # Group by track (or instance identifier) to get instances
        # Check if we have useful track information (non-null values)
        has_track_info = (
            "track_name" in frame_df.columns and frame_df["track_name"].notna().any()
        )

        # Determine if we need to compute instance indices from node repetition
        # This is needed when:
        # 1. No track info is available, or
        # 2. Track names are all null (untracked instances)
        need_instance_idx = not has_track_info

        if need_instance_idx:
            # Create instance index based on node repetition within the frame
            # For each instance type, count how many times each node appears
            node_counts = frame_df["node_name"].value_counts()
            n_instances = node_counts.max() if len(node_counts) > 0 else 0

            if n_instances > 0:
                # Assign instance indices based on order of appearance
                # Group by instance_type first if available, then assign indices
                frame_df = frame_df.copy()

                if has_instance_type:
                    # Assign instance indices within each instance_type group
                    instance_idx = []
                    for inst_type in frame_df["instance_type"].unique():
                        type_mask = frame_df["instance_type"] == inst_type
                        type_df = frame_df[type_mask]
                        node_counter = {}
                        for _, row in type_df.iterrows():
                            node = row["node_name"]
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
                        node = row["node_name"]
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
            instance_group_cols.append("track_name")
        if need_instance_idx and "_instance_idx" in frame_df.columns:
            instance_group_cols.append("_instance_idx")

        # Fallback if no grouping columns
        if not instance_group_cols:
            instance_group_cols = ["_instance_idx"]
            frame_df = frame_df.copy()
            frame_df["_instance_idx"] = 0  # All rows are one instance

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
            if "track_name" in inst_df.columns:
                track_name = inst_df["track_name"].iloc[0]
                if pd.notna(track_name) and track_name in track_lookup:
                    track = track_lookup[track_name]

            # Determine instance type
            is_predicted = True  # Default to predicted
            if has_instance_type:
                inst_type = inst_df["instance_type"].iloc[0]
                is_predicted = inst_type == "predicted"

            # Build points array
            n_nodes = len(skel.nodes)
            points_data = np.full((n_nodes, 2), np.nan, dtype="float64")
            scores = np.ones(n_nodes, dtype="float32") if has_score else None

            for _, row in inst_df.iterrows():
                node_name = row["node_name"]
                try:
                    node_idx = skel.index(node_name)
                    points_data[node_idx, 0] = row["x"]
                    points_data[node_idx, 1] = row["y"]
                    if has_score and scores is not None and "score" in row:
                        score_val = row["score"]
                        if pd.notna(score_val):
                            scores[node_idx] = float(score_val)
                except (ValueError, KeyError):
                    # Node not in skeleton, skip
                    pass

            # Create instance
            if is_predicted:
                instance = PredictedInstance.from_numpy(
                    points_data=points_data,
                    skeleton=skel,
                    point_scores=scores,
                    score=1.0,
                    track=track,
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
