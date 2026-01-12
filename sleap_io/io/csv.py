"""CSV I/O for pose tracking data.

This module provides functions for reading and writing pose tracking data
in various CSV formats, leveraging the DataFrame codec for data transformation.

Supported formats:
    sleap: SLEAP Analysis CSV format (one row per instance, simple header).
        This is the default format and matches the output of SLEAP's
        "Export Analysis CSV" functionality.
    dlc: DeepLabCut format (multi-row header, one row per frame).
        Compatible with DeepLabCut's CSV structure for labels and predictions.
    points: One row per point (most normalized form).
    instances: One row per instance with node coordinates as columns.
    frames: One row per frame with all instances multiplexed in columns.

Example:
    >>> import sleap_io as sio
    >>> labels = sio.load_csv("predictions.csv")
    >>> sio.save_csv(labels, "output.csv", format="sleap")
    >>> sio.save_csv(labels, "dlc_output.csv", format="dlc", scorer="MyModel")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sleap_io.model.labels import Labels, SuggestionFrame
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

# Type alias for supported CSV formats
CSVFormat = Literal["sleap", "dlc", "points", "instances", "frames", "auto"]


def read_labels(
    filename: str | Path,
    format: CSVFormat = "auto",
    video: Video | str | None = None,
    skeleton: Skeleton | None = None,
) -> Labels:
    """Read pose data from CSV file and return a Labels object.

    Args:
        filename: Path to CSV file.
        format: CSV format to read. One of:
            - "auto": Auto-detect format from file content (default).
            - "sleap": SLEAP Analysis CSV format.
            - "dlc": DeepLabCut format.
            - "points": Points format from DataFrame codec.
            - "instances": Instances format from DataFrame codec.
            - "frames": Frames format from DataFrame codec.
        video: Video to associate with loaded data. Can be a Video object or
            path string. Required if video information is not in the CSV or
            metadata file.
        skeleton: Skeleton to use. If None, inferred from column names or
            metadata file.

    Returns:
        Labels object with loaded pose data.

    Notes:
        If a metadata JSON file exists alongside the CSV (same name with .json
        extension), it will be automatically loaded and applied to restore
        full Labels context (skeleton edges, symmetries, video metadata,
        suggestions, provenance).

    See Also:
        write_labels: Write Labels to CSV file.
        detect_csv_format: Detect CSV format from file content.
    """
    filename = Path(filename)

    # Check for metadata JSON file (enables round-trip)
    metadata_path = filename.with_suffix(".json")
    metadata = None
    if metadata_path.exists():
        metadata = _read_metadata(metadata_path)
        # Use video/skeleton from metadata if not explicitly provided
        if video is None and metadata.get("videos"):
            video = metadata["videos"][0]["filename"]
        if skeleton is None and metadata.get("skeletons"):
            skel_data = metadata["skeletons"][0]
            skeleton = Skeleton(
                nodes=skel_data["nodes"],
                edges=skel_data.get("edges", []),
                symmetries=skel_data.get("symmetries", []),
                name=skel_data.get("name"),
            )

    # Auto-detect format if needed
    if format == "auto":
        format = detect_csv_format(filename)

    # Delegate to format-specific loaders
    if format == "dlc":
        labels = _read_dlc(filename, video=video)
    elif format == "sleap":
        labels = _read_sleap(filename, video=video, skeleton=skeleton)
    elif format in ("points", "instances", "frames"):
        labels = _read_codec_format(
            filename, format=format, video=video, skeleton=skeleton
        )
    else:
        raise ValueError(f"Unknown CSV format: {format}")

    # Apply metadata if available (restores edges, symmetries, suggestions, etc.)
    if metadata is not None:
        labels = _apply_metadata(labels, metadata)

    return labels


def write_labels(
    labels: Labels,
    filename: str | Path,
    format: CSVFormat = "sleap",
    *,
    video: Video | int | None = None,
    include_score: bool = True,
    include_empty: bool = False,
    scorer: str = "sleap-io",
    save_metadata: bool = False,
) -> None:
    """Write Labels to CSV file.

    Args:
        labels: Labels object to save.
        filename: Output file path.
        format: CSV format to use. One of:
            - "sleap": SLEAP Analysis CSV format (default).
            - "dlc": DeepLabCut format.
            - "points": One row per point.
            - "instances": One row per instance.
            - "frames": One row per frame.
        video: Video to filter to. Can be a Video object or integer index.
            If None, includes all videos.
        include_score: Include confidence scores in output. Default True.
        include_empty: Include frames with no instances. Default False.
        scorer: Scorer/model name for DLC format. Default "sleap-io".
        save_metadata: Save a JSON metadata file alongside the CSV that enables
            full round-trip reconstruction of the Labels object. The metadata
            file contains video paths, skeleton definitions (including edges
            and symmetries), track names, suggestions, and provenance.
            Default False.

    See Also:
        read_labels: Read Labels from CSV file.
    """
    filename = Path(filename)

    # Dispatch to format-specific writers
    if format == "dlc":
        _write_dlc(
            labels,
            filename,
            video=video,
            include_score=include_score,
            scorer=scorer,
        )
    elif format == "sleap":
        _write_sleap(
            labels,
            filename,
            video=video,
            include_score=include_score,
        )
    elif format in ("points", "instances", "frames"):
        _write_codec_format(
            labels,
            filename,
            format=format,
            video=video,
            include_score=include_score,
        )
    else:
        raise ValueError(f"Unknown CSV format: {format}")

    # Save optional metadata JSON (enables round-trip)
    if save_metadata:
        _write_metadata(labels, filename.with_suffix(".json"))


def detect_csv_format(filename: str | Path) -> str:
    """Detect CSV format from file content.

    Args:
        filename: Path to CSV file.

    Returns:
        Detected format name: "dlc", "sleap", "points", "instances", or "frames".

    Notes:
        Detection priority:
        1. DLC format: Look for "scorer", "bodyparts", "coords" in first few lines
        2. SLEAP format: Has "frame_idx" and "instance.score" columns
        3. Points format: Has "node" or "node_name" column
        4. Frames format: Has columns like "inst0.node.x"
        5. Instances format: Has columns like "node.x" (default fallback)
    """
    filename = Path(filename)

    # Read first few lines for header detection
    with open(filename, "r") as f:
        lines = [f.readline() for _ in range(5)]
    content = "\n".join(lines).lower()

    # Check for DLC format (multi-header with scorer/bodyparts/coords)
    if "scorer" in content and "coords" in content and "bodyparts" in content:
        return "dlc"

    # Read header row for column-based detection
    df = pd.read_csv(filename, nrows=0)
    columns = set(df.columns)

    # Check for SLEAP Analysis format (instance.score column)
    if "frame_idx" in columns and "instance.score" in columns:
        return "sleap"

    # Check for codec points format (has 'node' column)
    if "node" in columns or "node_name" in columns:
        return "points"

    # Check for codec instances/frames format (node.x pattern)
    dot_cols = [c for c in columns if "." in c and c.endswith((".x", ".y"))]
    if dot_cols:
        # Check if frames format (inst0.node.x pattern)
        if any(c.startswith("inst") for c in dot_cols):
            return "frames"
        return "instances"

    # Default to sleap format
    return "sleap"


def is_csv_file(filename: str | Path) -> bool:
    """Check if file is a supported CSV format.

    Args:
        filename: Path to file.

    Returns:
        True if file has .csv extension and appears to be a supported format.
    """
    filename = Path(filename)
    if filename.suffix.lower() != ".csv":
        return False

    try:
        detect_csv_format(filename)
        return True
    except Exception:
        return False


# =============================================================================
# SLEAP Analysis CSV Format
# =============================================================================


def _read_sleap(
    filename: Path,
    video: Video | str | None = None,
    skeleton: Skeleton | None = None,
) -> Labels:
    """Read SLEAP Analysis CSV format.

    SLEAP Analysis CSV has columns:
        track, frame_idx, instance.score, {node}.x, {node}.y, {node}.score, ...

    One row per instance.
    """
    from sleap_io.codecs import from_dataframe

    df = pd.read_csv(filename)

    # Rename columns to match codec expectations
    rename_map = {"instance.score": "score"}
    df = df.rename(columns=rename_map)

    # Infer skeleton from columns if not provided
    if skeleton is None:
        node_names = _infer_nodes_from_columns(df.columns)
        if node_names:
            skeleton = Skeleton(node_names)

    # Create video if path provided
    if isinstance(video, str):
        video = Video(video)

    # Use codec to create Labels
    labels = from_dataframe(df, format="instances", video=video, skeleton=skeleton)
    labels.provenance["filename"] = str(filename)
    return labels


def _write_sleap(
    labels: Labels,
    filename: Path,
    video: Video | int | None = None,
    include_score: bool = True,
) -> None:
    """Write SLEAP Analysis CSV format.

    Output columns:
        track, frame_idx, instance.score, {node}.x, {node}.y, {node}.score, ...
    """
    from sleap_io.codecs import to_dataframe

    # Use instances format from codec
    df = to_dataframe(
        labels,
        format="instances",
        video=video,
        include_score=include_score,
        include_metadata=True,
    )

    # Rename columns to match SLEAP Analysis format
    rename_map = {"score": "instance.score"}
    df = df.rename(columns=rename_map)

    # Reorder columns: track, frame_idx, instance.score, then node coords
    base_cols = ["track", "frame_idx", "instance.score"]
    base_cols = [c for c in base_cols if c in df.columns]

    # Get node columns (anything with .x, .y, .score pattern)
    exclude_cols = set(base_cols) | {"video_path", "track_score"}
    node_cols = [c for c in df.columns if c not in exclude_cols]
    node_cols = sorted(node_cols)  # Alphabetical order for consistency

    df = df[base_cols + node_cols]

    # Handle NaN -> empty string for SLEAP compatibility
    df.to_csv(filename, index=False, na_rep="")


# =============================================================================
# DeepLabCut CSV Format
# =============================================================================


def _read_dlc(
    filename: Path,
    video: Video | str | None = None,
) -> Labels:
    """Read DeepLabCut CSV format.

    Delegates to existing load_dlc() for parsing since DLC format is complex.
    """
    from sleap_io.io.dlc import load_dlc

    return load_dlc(str(filename), video_search_paths=None)


def _write_dlc(
    labels: Labels,
    filename: Path,
    video: Video | int | None = None,
    include_score: bool = False,
    scorer: str = "sleap-io",
) -> None:
    """Write DeepLabCut CSV format.

    Creates multi-header CSV with structure:
        Single-animal (3 header rows): scorer, bodyparts, coords
        Multi-animal (4 header rows): scorer, individuals, bodyparts, coords

    Row index is image path (DLC convention).
    """
    # Get skeleton nodes
    skeleton = labels.skeletons[0] if labels.skeletons else None
    if skeleton is None:
        raise ValueError("Cannot export DLC format without skeleton")

    node_names = [node.name for node in skeleton.nodes]

    # Determine if multi-animal based on track count
    tracks = labels.tracks
    is_multi_animal = len(tracks) > 1

    # Filter to specific video if requested
    if video is not None:
        if isinstance(video, int):
            video = labels.videos[video]
        lfs = [lf for lf in labels if lf.video == video]
    else:
        lfs = list(labels)

    # Build column MultiIndex
    if is_multi_animal:
        # 4-level: scorer, individuals, bodyparts, coords
        individuals = [t.name if t else f"ind{i}" for i, t in enumerate(tracks)]
        coord_names = ["x", "y", "likelihood"] if include_score else ["x", "y"]
        columns = pd.MultiIndex.from_tuples(
            [
                (scorer, ind, node, coord)
                for ind in individuals
                for node in node_names
                for coord in coord_names
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
    else:
        # 3-level: scorer, bodyparts, coords
        coord_names = ["x", "y", "likelihood"] if include_score else ["x", "y"]
        columns = pd.MultiIndex.from_tuples(
            [(scorer, node, coord) for node in node_names for coord in coord_names],
            names=["scorer", "bodyparts", "coords"],
        )

    # Build data rows
    rows = []
    index = []

    # Build track to individuals mapping
    track_to_ind = {}
    if is_multi_animal:
        for i, track in enumerate(tracks):
            track_to_ind[track] = track.name if track else f"ind{i}"

    for lf in sorted(lfs, key=lambda x: x.frame_idx):
        # Create image path index (DLC convention)
        video_name = Path(lf.video.filename).stem if lf.video else "video"
        img_path = f"labeled-data/{video_name}/img{lf.frame_idx:04d}.png"
        index.append(img_path)

        # Build row data
        row = {}
        for inst_idx, inst in enumerate(lf.instances):
            if is_multi_animal:
                ind_name = track_to_ind.get(inst.track, f"ind{inst_idx}")
            else:
                ind_name = None

            pts = inst.numpy()
            for node_idx, node in enumerate(skeleton.nodes):
                pt = pts[node_idx] if node_idx < len(pts) else [np.nan, np.nan]
                x, y = pt[:2] if len(pt) >= 2 else (np.nan, np.nan)
                if is_multi_animal:
                    row[(scorer, ind_name, node.name, "x")] = x
                    row[(scorer, ind_name, node.name, "y")] = y
                    if include_score:
                        score = pt[2] if len(pt) > 2 else np.nan
                        row[(scorer, ind_name, node.name, "likelihood")] = score
                else:
                    row[(scorer, node.name, "x")] = x
                    row[(scorer, node.name, "y")] = y
                    if include_score:
                        score = pt[2] if len(pt) > 2 else np.nan
                        row[(scorer, node.name, "likelihood")] = score

        rows.append(row)

    # Create DataFrame with MultiIndex columns
    df = pd.DataFrame(rows, index=index, columns=columns)

    # Write with multi-header
    df.to_csv(filename)


# =============================================================================
# Codec Format Passthrough (points, instances, frames)
# =============================================================================


def _read_codec_format(
    filename: Path,
    format: str,
    video: Video | str | None = None,
    skeleton: Skeleton | None = None,
) -> Labels:
    """Read CSV in codec format (points, instances, frames)."""
    from sleap_io.codecs import from_dataframe

    df = pd.read_csv(filename)

    # Infer skeleton if needed
    if skeleton is None:
        node_names = _infer_nodes_from_columns(df.columns, format=format)
        if node_names:
            skeleton = Skeleton(node_names)

    # Create video if path provided
    if isinstance(video, str):
        video = Video(video)

    labels = from_dataframe(df, format=format, video=video, skeleton=skeleton)
    labels.provenance["filename"] = str(filename)
    return labels


def _write_codec_format(
    labels: Labels,
    filename: Path,
    format: str,
    video: Video | int | None = None,
    include_score: bool = True,
) -> None:
    """Write CSV in codec format (points, instances, frames)."""
    from sleap_io.codecs import to_dataframe

    df = to_dataframe(
        labels,
        format=format,
        video=video,
        include_score=include_score,
        include_metadata=True,
    )
    df.to_csv(filename, index=False)


# =============================================================================
# Metadata JSON (enables round-trip)
# =============================================================================


def _write_metadata(labels: Labels, filename: Path) -> None:
    """Write JSON metadata file with all information needed for round-trip.

    The metadata file contains everything that cannot be represented in CSV,
    enabling full reconstruction of a Labels object when loading.

    Structure:
        {
            "version": "1.0",
            "videos": [...],      # Video filenames and backend metadata
            "skeletons": [...],   # Full skeleton definitions
            "tracks": [...],      # Track names
            "suggestions": [...], # Suggested frames for labeling
            "provenance": {...}   # Source and creation metadata
        }
    """
    metadata: dict = {
        "version": "1.0",
        "videos": [],
        "skeletons": [],
        "tracks": [],
        "suggestions": [],
        "provenance": dict(labels.provenance),
    }

    # Videos: filename + backend_metadata (shape, fps, etc.)
    for video in labels.videos:
        video_data = {
            "filename": video.filename,
            "backend_metadata": video.backend_metadata,
        }
        metadata["videos"].append(video_data)

    # Skeletons: full definition including symmetries
    for skeleton in labels.skeletons:
        skel_data: dict = {
            "name": skeleton.name,
            "nodes": [node.name for node in skeleton.nodes],
            "edges": [
                (edge.source.name, edge.destination.name) for edge in skeleton.edges
            ],
            "symmetries": [
                (list(sym.nodes)[0].name, list(sym.nodes)[1].name)
                for sym in skeleton.symmetries
            ],
        }
        metadata["skeletons"].append(skel_data)

    # Tracks: names only (identity reconstructed on load)
    metadata["tracks"] = [track.name for track in labels.tracks]

    # Suggestions: video index + frame indices
    for suggestion in labels.suggestions:
        # Check for None explicitly (not truthiness, as Video.__bool__ may be False)
        if suggestion.video is not None:
            video_idx = labels.videos.index(suggestion.video)
        else:
            video_idx = None
        metadata["suggestions"].append(
            {
                "video_idx": video_idx,
                "frame_idx": suggestion.frame_idx,
            }
        )

    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)


def _read_metadata(filename: Path) -> dict:
    """Read JSON metadata file.

    Returns:
        Metadata dict that can be used to reconstruct Labels context.
    """
    with open(filename, "r") as f:
        return json.load(f)


def _apply_metadata(labels: Labels, metadata: dict) -> Labels:
    """Apply metadata to a Labels object loaded from CSV.

    This restores information that was lost in CSV serialization:
    - Video backend_metadata (shape, fps)
    - Skeleton edges and symmetries
    - Suggestions
    - Provenance

    Args:
        labels: Labels object loaded from CSV (has basic structure).
        metadata: Metadata dict from _read_metadata().

    Returns:
        Labels object with full metadata restored.
    """
    # Restore video backend_metadata
    for i, video_data in enumerate(metadata.get("videos", [])):
        if i < len(labels.videos):
            labels.videos[i].backend_metadata = video_data.get("backend_metadata", {})

    # Restore skeleton edges and symmetries
    for i, skel_data in enumerate(metadata.get("skeletons", [])):
        if i < len(labels.skeletons):
            skel = labels.skeletons[i]

            # Build name-to-node mapping
            node_map = {node.name: node for node in skel.nodes}

            # Restore edges (avoid duplicates)
            for src_name, dst_name in skel_data.get("edges", []):
                src = node_map.get(src_name)
                dst = node_map.get(dst_name)
                if src is not None and dst is not None:
                    if not any(
                        e.source == src and e.destination == dst for e in skel.edges
                    ):
                        skel.add_edge(src, dst)

            # Restore symmetries
            for node1_name, node2_name in skel_data.get("symmetries", []):
                node1 = node_map.get(node1_name)
                node2 = node_map.get(node2_name)
                if node1 is not None and node2 is not None:
                    try:
                        skel.add_symmetry(node1, node2)
                    except ValueError:
                        pass  # Symmetry already exists, skip

    # Restore suggestions
    for suggestion_data in metadata.get("suggestions", []):
        video_idx = suggestion_data.get("video_idx")
        frame_idx = suggestion_data.get("frame_idx")
        if video_idx is not None and video_idx < len(labels.videos):
            labels.suggestions.append(
                SuggestionFrame(video=labels.videos[video_idx], frame_idx=frame_idx)
            )

    # Restore provenance (merge, don't overwrite)
    labels.provenance.update(metadata.get("provenance", {}))

    return labels


# =============================================================================
# Utilities
# =============================================================================


def _infer_nodes_from_columns(columns, format: str = "instances") -> list[str]:
    """Infer node names from column patterns.

    Args:
        columns: DataFrame column names.
        format: Expected format for parsing hints.

    Returns:
        Sorted list of node names found in columns.
    """
    nodes = set()

    if format == "points":
        # Points format has explicit 'node' column, nodes not in column names
        return []

    for col in columns:
        if "." not in col:
            continue

        parts = col.split(".")
        suffix = parts[-1]

        if suffix not in ("x", "y", "score"):
            continue

        if format == "frames" and len(parts) >= 3:
            # frames: inst0.nose.x -> nose
            node = parts[-2]
        else:
            # instances/sleap: nose.x -> nose
            node = ".".join(parts[:-1])

        nodes.add(node)

    return sorted(nodes)
