"""This module handles direct I/O operations for working with LEAP .mat files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from sleap_io.model.instance import Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Edge, Node, Skeleton
from sleap_io.model.video import Video


def read_labels(labels_path: str, skeleton: Optional[Skeleton] = None) -> Labels:
    """Read LEAP pose data from a .mat file and return a `Labels` object.

    Args:
        labels_path: Path to the LEAP .mat pose file.
        skeleton: An optional `Skeleton` object. If not provided, will be constructed
            from the data in the file.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    try:
        from pymatreader import read_mat
    except ImportError:
        raise ImportError(
            "pymatreader is required to read LEAP .mat files. "
            "Install it with: pip install sleap-io[mat]"
        )

    # Load the MATLAB data
    mat_data = read_mat(labels_path)

    # Extract video path
    video_path = mat_data.get("boxPath", None)
    if video_path is None:
        # Try to infer video path from labels path
        video_path = Path(labels_path).with_suffix(".mp4")

    # Create Video object
    video = Video.from_filename(str(video_path))

    # Parse skeleton if not provided
    if skeleton is None:
        skeleton = _parse_skeleton(mat_data)

    # Parse pose data
    labeled_frames = _parse_pose_data(mat_data, video, skeleton)

    # Create Labels object
    labels = Labels(
        labeled_frames=labeled_frames,
        videos=[video],
        skeletons=[skeleton],
    )

    return labels


def _parse_skeleton(mat_data: dict) -> Skeleton:
    """Parse skeleton structure from LEAP .mat data.

    Args:
        mat_data: Dictionary containing the loaded MATLAB data.

    Returns:
        A `Skeleton` object.
    """
    skeleton_data = mat_data.get("skeleton", {})

    # Extract node names - can be in 'nodes' or 'joints' field
    node_names = skeleton_data.get("nodes", skeleton_data.get("joints", []))
    if isinstance(node_names, np.ndarray):
        node_names = node_names.tolist()

    # Ensure node names are strings
    if node_names and isinstance(node_names[0], np.ndarray):
        node_names = [
            str(name[0]) if isinstance(name, np.ndarray) else str(name)
            for name in node_names
        ]
    elif node_names and not isinstance(node_names[0], str):
        node_names = [str(name) for name in node_names]

    # Create nodes
    nodes = [Node(name) for name in node_names]

    # Extract edges (connections between nodes)
    edges_data = skeleton_data.get("edges", [])
    edges = []

    if edges_data is not None and len(edges_data) > 0:
        # Handle both numpy arrays and lists
        if isinstance(edges_data, np.ndarray):
            if edges_data.ndim == 1:
                edges_data = edges_data.reshape(-1, 2)

        # Convert MATLAB 1-based indexing to Python 0-based
        for edge in edges_data:
            if len(edge) >= 2:
                # MATLAB uses 1-based indexing
                src_idx = int(edge[0]) - 1
                dst_idx = int(edge[1]) - 1
                if 0 <= src_idx < len(nodes) and 0 <= dst_idx < len(nodes):
                    edges.append(Edge(nodes[src_idx], nodes[dst_idx]))

    return Skeleton(nodes=nodes, edges=edges)


def _parse_pose_data(
    mat_data: dict, video: Video, skeleton: Skeleton
) -> list[LabeledFrame]:
    """Parse pose data from LEAP .mat data.

    Args:
        mat_data: Dictionary containing the loaded MATLAB data.
        video: Video object for these labels.
        skeleton: Skeleton object defining the structure.

    Returns:
        List of `LabeledFrame` objects.
    """
    labeled_frames = []

    # Extract position data
    # LEAP stores data as (nodes, 2, frames) or similar structure
    positions = mat_data.get("positions", None)
    if positions is None:
        # Try alternative field names
        positions = mat_data.get("pose", mat_data.get("posedata", None))

    if positions is None:
        return labeled_frames

    # Determine shape and transpose if needed
    # We want shape to be (frames, nodes, 2)
    if positions.ndim == 3:
        # Could be (nodes, 2, frames) or (frames, nodes, 2)
        if positions.shape[1] == 2 and positions.shape[2] != 2:
            # Shape is (nodes, 2, frames) - need to transpose
            positions = np.transpose(positions, (2, 0, 1))
        # else shape is already (frames, nodes, 2) or needs different handling
    elif positions.ndim == 2:
        # Single frame, shape is (nodes, 2)
        positions = positions[np.newaxis, :, :]

    num_frames = positions.shape[0]
    num_nodes = positions.shape[1]

    # Create labeled frames
    for frame_idx in range(num_frames):
        frame_data = positions[frame_idx]

        # Create points for this frame
        points = {}
        for node_idx in range(min(num_nodes, len(skeleton.nodes))):
            x, y = frame_data[node_idx]
            if not (np.isnan(x) or np.isnan(y)):
                points[skeleton.nodes[node_idx]] = np.array([x, y])

        # Only create frame if we have valid points
        if points:
            instance = Instance(points=points, skeleton=skeleton)
            labeled_frame = LabeledFrame(
                video=video, frame_idx=frame_idx, instances=[instance]
            )
            labeled_frames.append(labeled_frame)

    return labeled_frames
