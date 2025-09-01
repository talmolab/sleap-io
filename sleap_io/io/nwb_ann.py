"""NWB formatted annotations."""

import numpy as np
from ndx_pose import Skeleton as NWBSkeleton

from sleap_io import Skeleton as SleapSkeleton


def _sanitize_nwb_name(name: str | None) -> str:
    """Sanitize a name for use in NWB files.

    NWB names cannot contain '/' or ':' characters.

    Args:
        name: The name to sanitize.

    Returns:
        The sanitized name with invalid characters replaced.
    """
    if name is None:
        return "skeleton"
    # Replace forward slashes and colons with underscores
    sanitized = name.replace("/", "_").replace(":", "_")
    return sanitized


def sleap_skeleton_to_nwb_skeleton(
    sleap_skeleton: SleapSkeleton,
) -> NWBSkeleton:
    """Convert a sleap-io Skeleton to ndx-pose Skeleton.

    Args:
        sleap_skeleton: The sleap-io Skeleton object to convert.

    Returns:
        An ndx-pose Skeleton object with equivalent structure.
    """
    # Convert node names to list of strings
    nodes = sleap_skeleton.node_names

    # Convert edges from Edge objects to array of node indices
    edges = np.array(sleap_skeleton.edge_inds, dtype=np.uint8)
    if edges.size == 0:
        edges = edges.reshape(0, 2)

    # Use skeleton name or default
    name = _sanitize_nwb_name(sleap_skeleton.name)

    return NWBSkeleton(name=name, nodes=nodes, edges=edges)


def nwb_skeleton_to_sleap_skeleton(nwb_skeleton: NWBSkeleton) -> SleapSkeleton:
    """Convert an ndx-pose Skeleton to sleap-io Skeleton.

    Args:
        nwb_skeleton: The ndx-pose Skeleton object to convert.

    Returns:
        A sleap-io Skeleton object with equivalent structure.
    """
    # Convert nodes (already strings in ndx-pose)
    nodes = list(nwb_skeleton.nodes)

    # Convert edges from array of indices to list of tuples
    edges = [(int(edge[0]), int(edge[1])) for edge in nwb_skeleton.edges]

    # Create sleap-io skeleton
    sleap_skeleton = SleapSkeleton(nodes=nodes, edges=edges, name=nwb_skeleton.name)

    return sleap_skeleton
