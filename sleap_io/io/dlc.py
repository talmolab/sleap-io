"""This module handles direct I/O operations for working with DeepLabCut (DLC) files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Node, Skeleton
from sleap_io.model.video import Video


def is_dlc_file(filename: Union[str, Path]) -> bool:
    """Check if file is a DLC CSV file.

    Args:
        filename: Path to file to check.

    Returns:
        True if file appears to be a DLC CSV file.
    """
    try:
        # Read first few lines as raw text to check for DLC structure
        with open(filename, "r") as f:
            lines = [f.readline().strip() for _ in range(4)]

        # Join all lines to search for DLC patterns
        content = "\n".join(lines).lower()

        # Check for DLC's characteristic patterns
        has_scorer = "scorer" in content
        has_coords = "coords" in content
        has_xy = "x" in content and "y" in content
        has_bodyparts = "bodyparts" in content or any(
            part in content for part in ["animal", "individual"]
        )

        return has_scorer and has_coords and has_xy and has_bodyparts

    except Exception:
        return False


def load_dlc(
    filename: Union[str, Path],
    video_search_paths: Optional[list[Union[str, Path]]] = None,
    **kwargs,
) -> Labels:
    """Load DeepLabCut annotations from CSV file.

    Args:
        filename: Path to DLC CSV file.
        video_search_paths: List of paths to search for video files.
        **kwargs: Additional arguments (unused).

    Returns:
        Labels object with loaded data.
    """
    filename = Path(filename)

    # Try reading first few rows to determine format
    try:
        # Try multi-animal format first (header rows 1-3, skipping scorer row)
        df = pd.read_csv(filename, header=[1, 2, 3], nrows=2)
        is_multianimal = df.columns[0][0] == "individuals"
    except Exception:
        # Fall back to single-animal format
        is_multianimal = False

    # Read full file with appropriate header levels
    if is_multianimal:
        # Multi-animal format: skip scorer row, use individuals/bodyparts/coords
        df = pd.read_csv(filename, header=[1, 2, 3], index_col=0)
    else:
        # Single-animal format: use scorer/bodyparts/coords
        df = pd.read_csv(filename, header=[0, 1, 2], index_col=0)

    # Parse structure based on format
    if is_multianimal:
        skeleton, tracks = _parse_multi_animal_structure(df)
    else:
        skeleton = _parse_single_animal_structure(df)
        tracks = []

    # First, group all image paths by their video directory
    video_image_paths = {}
    frame_map = {}  # Maps image path to frame index

    for idx in df.index:
        img_path = str(idx)
        frame_idx = _extract_frame_index(img_path)
        frame_map[img_path] = frame_idx

        # Extract video name from path
        # e.g., "labeled-data/video/img000.png" -> "video"
        path_parts = Path(img_path).parts
        if len(path_parts) >= 2 and path_parts[0] == "labeled-data":
            video_name = path_parts[1]
        else:
            video_name = Path(img_path).parent.name or "default"

        if video_name not in video_image_paths:
            video_image_paths[video_name] = []
        video_image_paths[video_name].append(img_path)

    # Create one Video object per video directory
    videos = {}
    for video_name, image_paths in video_image_paths.items():
        # Sort image paths to ensure consistent ordering
        sorted_paths = sorted(image_paths, key=lambda p: frame_map[p])

        # Find the actual image files
        actual_image_files = []
        for img_path in sorted_paths:
            # First try the full path from CSV
            full_path = filename.parent / img_path
            if full_path.exists():
                actual_image_files.append(str(full_path))
            else:
                # Try just the filename in the same directory as the CSV
                img_name = Path(img_path).name
                simple_path = filename.parent / img_name
                if simple_path.exists():
                    actual_image_files.append(str(simple_path))
                else:
                    # Try going up one directory from CSV location
                    # (CSV in subdir references parent/subdir/img.png)
                    parent_path = filename.parent.parent / img_path
                    if parent_path.exists():
                        actual_image_files.append(str(parent_path))

        # Only create video if we found actual images
        if actual_image_files:
            videos[video_name] = Video.from_filename(actual_image_files)

    # Parse the actual data rows and create labeled frames
    labeled_frames = []
    for idx, row in df.iterrows():
        # Get image path from index
        img_path = str(idx)
        frame_idx = _extract_frame_index(img_path)

        # Determine which video this frame belongs to
        path_parts = Path(img_path).parts
        if len(path_parts) >= 2 and path_parts[0] == "labeled-data":
            video_name = path_parts[1]
        else:
            video_name = Path(img_path).parent.name or "default"

        # Skip if we don't have a video for this frame
        if video_name not in videos:
            continue

        # Parse instances for this frame
        if is_multianimal:
            instances = _parse_multi_animal_row(row, skeleton, tracks)
        else:
            instances = _parse_single_animal_row(row, skeleton)

        if instances:
            # Get the index of this image within its video
            sorted_video_paths = sorted(
                video_image_paths[video_name], key=lambda p: frame_map[p]
            )
            video_frame_idx = sorted_video_paths.index(img_path)
            labeled_frames.append(
                LabeledFrame(
                    video=videos[video_name],
                    frame_idx=video_frame_idx,
                    instances=instances,
                )
            )

    unique_videos = list(videos.values())

    return Labels(
        labeled_frames=labeled_frames,
        videos=unique_videos,
        tracks=tracks,
        skeletons=[skeleton] if skeleton.nodes else [],
    )


def _parse_multi_animal_structure(df: pd.DataFrame) -> tuple[Skeleton, list[Track]]:
    """Parse multi-animal DLC structure to extract skeleton and tracks."""
    # Extract unique node names and track names from columns
    tracks_dict = {}
    node_names = []

    # Iterate through columns (skip coords columns)
    for col in df.columns:
        if len(col) >= 3:  # Multi-level column (individuals, bodyparts, coords)
            individual = col[0]
            bodypart = col[1]
            coord = col[2]

            if coord == "x":  # Only process x coordinates to avoid duplicates
                # Add track (skip the header row name)
                if individual not in tracks_dict and individual not in [
                    "",
                    None,
                    "individuals",
                ]:
                    tracks_dict[individual] = Track(name=individual)

                # Add node (skip the header row name)
                if bodypart not in node_names and bodypart not in [
                    "",
                    None,
                    "bodyparts",
                ]:
                    node_names.append(bodypart)

    # Create skeleton with all unique nodes
    nodes = [Node(name=name) for name in sorted(set(node_names))]
    skeleton = Skeleton(nodes=nodes)

    # Create track list
    tracks = list(tracks_dict.values())

    return skeleton, tracks


def _parse_single_animal_structure(df: pd.DataFrame) -> Skeleton:
    """Parse single-animal DLC structure to extract skeleton."""
    # Extract node names from bodyparts level
    node_names = []

    for col in df.columns:
        if len(col) >= 3:  # Multi-level column
            bodypart = col[1]
            coord = col[2]

            if (
                coord == "x"
                and bodypart not in node_names
                and bodypart not in ["", None]
            ):
                node_names.append(bodypart)

    # Create skeleton
    nodes = [Node(name=name) for name in sorted(set(node_names))]
    skeleton = Skeleton(nodes=nodes)

    return skeleton


def _parse_multi_animal_row(
    row: pd.Series, skeleton: Skeleton, tracks: list[Track]
) -> list[Instance]:
    """Parse a row of multi-animal DLC data."""
    instances_dict = {}

    # Group data by individual
    for col_tuple, value in row.items():
        if len(col_tuple) >= 3:
            individual = col_tuple[0]
            bodypart = col_tuple[1]
            coord = col_tuple[2]

            # Skip empty individuals or header names
            if not individual or individual == "" or individual == "individuals":
                continue

            # Initialize instance data if needed
            if individual not in instances_dict:
                instances_dict[individual] = {}

            # Store coordinate data
            if bodypart and bodypart != "":
                if bodypart not in instances_dict[individual]:
                    instances_dict[individual][bodypart] = {}
                instances_dict[individual][bodypart][coord] = value

    # Create instances
    instances = []
    for individual_name, bodyparts_data in instances_dict.items():
        # Find matching track
        track = next((t for t in tracks if t.name == individual_name), None)

        # Create instance
        points = np.full((len(skeleton.nodes), 2), np.nan)
        has_valid_points = False

        for node_idx, node in enumerate(skeleton.nodes):
            if node.name in bodyparts_data:
                coords = bodyparts_data[node.name]
                if "x" in coords and "y" in coords:
                    x_val = coords["x"]
                    y_val = coords["y"]
                    if pd.notna(x_val) and pd.notna(y_val):
                        points[node_idx] = [float(x_val), float(y_val)]
                        has_valid_points = True

        # Only create instance if it has at least one valid point
        if has_valid_points:
            instance = Instance.from_numpy(
                points_data=points, skeleton=skeleton, track=track
            )
            instances.append(instance)

    return instances


def _parse_single_animal_row(row: pd.Series, skeleton: Skeleton) -> list[Instance]:
    """Parse a row of single-animal DLC data."""
    # Create instance
    points = np.full((len(skeleton.nodes), 2), np.nan)
    has_valid_points = False

    # Collect coordinates for each bodypart
    bodyparts_data = {}
    for col_tuple, value in row.items():
        if len(col_tuple) >= 3:
            bodypart = col_tuple[1]
            coord = col_tuple[2]

            if bodypart and bodypart != "":
                if bodypart not in bodyparts_data:
                    bodyparts_data[bodypart] = {}
                bodyparts_data[bodypart][coord] = value

    # Fill in points
    for node_idx, node in enumerate(skeleton.nodes):
        if node.name in bodyparts_data:
            coords = bodyparts_data[node.name]
            if "x" in coords and "y" in coords:
                x_val = coords["x"]
                y_val = coords["y"]
                if pd.notna(x_val) and pd.notna(y_val):
                    points[node_idx] = [float(x_val), float(y_val)]
                    has_valid_points = True

    # Only return instance if it has at least one valid point
    if has_valid_points:
        instance = Instance.from_numpy(points_data=points, skeleton=skeleton)
        return [instance]

    return []


def _extract_frame_index(img_path: str) -> int:
    """Extract frame index from image filename."""
    # Look for numbers in filename
    matches = re.findall(r"(\d+)", Path(img_path).stem)
    if matches:
        return int(matches[-1])  # Use last number found
    return 0
