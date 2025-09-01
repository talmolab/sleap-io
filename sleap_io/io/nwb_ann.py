"""NWB formatted annotations."""

import numpy as np
from ndx_pose import Skeleton as NwbSkeleton
from ndx_pose import SkeletonInstance as NwbInstance

from sleap_io import Instance as SleapInstance
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
) -> NwbSkeleton:
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

    return NwbSkeleton(name=name, nodes=nodes, edges=edges)


def nwb_skeleton_to_sleap_skeleton(nwb_skeleton: NwbSkeleton) -> SleapSkeleton:
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


def sleap_instance_to_nwb_skeleton_instance(
    sleap_instance: SleapInstance,
    nwb_skeleton: NwbSkeleton,
    name: str = "skeleton_instance",
    id: int | None = None,
) -> NwbInstance:
    """Convert a sleap-io Instance to ndx-pose SkeletonInstance.

    Args:
        sleap_instance: The sleap-io Instance object to convert.
        nwb_skeleton: The ndx-pose Skeleton object to associate with the instance.
        name: String identifier for the skeleton instance. Default: "skeleton_instance".
        id: Optional unique identifier (integer) for the instance. Default: None.

    Returns:
        An ndx-pose SkeletonInstance object with equivalent data.
    """
    # Get node locations as (n_nodes, 2) array - always use visible points as NaN for invisible
    node_locations = sleap_instance.numpy(invisible_as_nan=True)

    # Get node visibility - True where points are not NaN
    node_visibility = ~np.isnan(node_locations).any(axis=1)

    # Convert id to unsigned integer if provided
    if id is not None:
        id = np.uint8(id)

    return NwbInstance(
        node_locations=node_locations,
        skeleton=nwb_skeleton,
        name=name,
        id=id,
        node_visibility=node_visibility,
    )


def nwb_skeleton_instance_to_sleap_instance(
    nwb_skeleton_instance: NwbInstance, skeleton: SleapSkeleton
) -> SleapInstance:
    """Convert an ndx-pose SkeletonInstance to sleap-io Instance.

    Args:
        nwb_skeleton_instance: The ndx-pose SkeletonInstance object to convert.
        skeleton: The sleap-io Skeleton to associate with the instance.

    Returns:
        A sleap-io Instance object with equivalent data.
    """
    # Get node locations and visibility
    node_locations = nwb_skeleton_instance.node_locations

    # Handle visibility - use provided visibility or infer from NaN values
    if nwb_skeleton_instance.node_visibility is not None:
        node_visibility = nwb_skeleton_instance.node_visibility
    else:
        # Infer visibility from non-NaN values
        node_visibility = ~np.isnan(node_locations).any(axis=1)

    # Create instance from numpy array - will handle visibility internally
    # Set invisible points to NaN for proper handling
    points_array = node_locations.copy()
    points_array[~node_visibility] = np.nan

    return SleapInstance.from_numpy(points_array, skeleton)
