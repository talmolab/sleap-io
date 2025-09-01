"""NWB formatted annotations."""

from pathlib import Path

import numpy as np
from ndx_pose import Skeleton as NwbSkeleton
from ndx_pose import SkeletonInstance as NwbInstance
from ndx_pose import SkeletonInstances as NwbSkeletonInstances
from ndx_pose import SourceVideos as NwbSourceVideos
from ndx_pose import TrainingFrame as NwbTrainingFrame
from pynwb.image import ImageSeries

from sleap_io import Instance as SleapInstance
from sleap_io import LabeledFrame as SleapLabeledFrame
from sleap_io import Skeleton as SleapSkeleton
from sleap_io import Video as SleapVideo
from sleap_io.io.utils import sanitize_filename
from sleap_io.io.video_reading import ImageVideo, MediaVideo


def sanitize_nwb_name(name: str) -> str:
    """Sanitize a name for use in NWB files.

    NWB names cannot contain '/' or ':' characters.

    Args:
        name: The name to sanitize.

    Returns:
        The sanitized name with invalid characters replaced.
    """
    if isinstance(name, Path):
        name = sanitize_filename(name)

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
    name = sanitize_nwb_name(sleap_skeleton.name) if sleap_skeleton.name else "skeleton"

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
    # Get node locations as (n_nodes, 2) array
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


def sleap_video_to_nwb_image_series(
    sleap_video: SleapVideo,
    name: str | None = None,
    description: str = "no description",
) -> ImageSeries:
    """Convert a sleap-io Video to pynwb ImageSeries.

    Args:
        sleap_video: The sleap-io Video object to convert.
        name: String identifier for the ImageSeries. If None, uses the filename.
        description: String description for the ImageSeries.

    Returns:
        A pynwb ImageSeries object with external file references.

    Raises:
        ValueError: If the video backend is not supported for NWB export.
    """
    # Validate supported backend
    if not isinstance(sleap_video.backend, (MediaVideo, ImageVideo)):
        raise ValueError(
            f"Unsupported video backend for NWB export: {type(sleap_video.backend)}. "
            f"Supported backends: MediaVideo, ImageVideo"
        )

    # Set default name if not provided
    if name is None:
        if isinstance(sleap_video.filename, list):
            name = sanitize_nwb_name(sleap_video.filename[0])
        else:
            name = sanitize_nwb_name(sleap_video.filename)

    # Pull out filename
    filename = sanitize_filename(sleap_video.filename)
    if isinstance(filename, str):
        filename = [filename]

    # Get video metadata
    shape = sleap_video.shape
    if shape is not None:
        height, width = shape[1:3]
    else:
        height, width = 0, 0

    fps = getattr(sleap_video, "fps", 30.0)

    starting_frame = list(range(len(filename)))  # needs to match length of filenames

    # Create ImageSeries with external file reference
    image_series = ImageSeries(
        name=name,
        description=description,
        unit="NA",  # Standard for video data
        format="external",  # External file reference
        external_file=filename,
        dimension=[width, height],
        rate=fps,
        starting_frame=starting_frame,
    )

    return image_series


def nwb_image_series_to_sleap_video(
    image_series: ImageSeries,
) -> SleapVideo:
    """Convert a pynwb ImageSeries to sleap-io Video.

    Args:
        image_series: The pynwb ImageSeries object to convert.

    Returns:
        A sleap-io Video object with equivalent data.

    Raises:
        ValueError: If the ImageSeries format is not "external".
    """
    # Check that this is an external file reference
    if image_series.format != "external":
        raise ValueError(
            f"Unsupported ImageSeries format: {image_series.format}. "
            f"Only 'external' format is supported for conversion to sleap-io Video."
        )

    filename = image_series.external_file
    if len(filename) == 1:
        filename = filename[0]

    # Create sleap-io Video
    sleap_video = SleapVideo.from_filename(filename)

    return sleap_video


def sleap_videos_to_nwb_source_videos(
    sleap_videos: list[SleapVideo],
    name: str = "SourceVideos",
) -> NwbSourceVideos:
    """Convert a list of sleap-io Videos to ndx-pose SourceVideos container.

    Args:
        sleap_videos: List of sleap-io Video objects to convert.
        name: String identifier for the SourceVideos container.

    Returns:
        An ndx-pose SourceVideos container with ImageSeries for each video.

    Raises:
        ValueError: If any video backend is not supported for NWB export.
    """
    image_series_list = []
    for video_ind, sleap_video in enumerate(sleap_videos):
        video_name = f"video_{video_ind}"
        image_series = sleap_video_to_nwb_image_series(sleap_video, name=video_name)
        image_series_list.append(image_series)

    return NwbSourceVideos(name=name, image_series=image_series_list)


def nwb_source_videos_to_sleap_videos(
    nwb_source_videos: NwbSourceVideos,
) -> list[SleapVideo]:
    """Convert ndx-pose SourceVideos to a list of sleap-io Videos.

    Args:
        nwb_source_videos: The ndx-pose SourceVideos container to convert.

    Returns:
        A list of sleap-io Video objects with equivalent data.

    Raises:
        ValueError: If any ImageSeries format is not supported.
    """
    sleap_videos = []
    for image_series in nwb_source_videos.image_series.values():
        sleap_video = nwb_image_series_to_sleap_video(image_series)
        sleap_videos.append(sleap_video)

    return sleap_videos


def sleap_labeled_frame_to_nwb_training_frame(
    sleap_labeled_frame: SleapLabeledFrame,
    nwb_skeleton: NwbSkeleton,
    source_video: ImageSeries | None = None,
    name: str = "training_frame",
    annotator: str | None = None,
) -> NwbTrainingFrame:
    """Convert a sleap-io LabeledFrame to ndx-pose TrainingFrame.

    Args:
        sleap_labeled_frame: The sleap-io LabeledFrame object to convert.
        nwb_skeleton: The ndx-pose Skeleton object to associate with instances.
        source_video: Optional ImageSeries representing the source video.
        name: String identifier for the TrainingFrame.
        annotator: Optional name of annotator who labeled the frame.

    Returns:
        An ndx-pose TrainingFrame object with equivalent data.
    """
    # Convert instances to NWB SkeletonInstances
    nwb_instances = []
    for i, sleap_instance in enumerate(sleap_labeled_frame.instances):
        instance_name = f"instance_{i}"
        nwb_instance = sleap_instance_to_nwb_skeleton_instance(
            sleap_instance, nwb_skeleton, name=instance_name, id=i
        )
        nwb_instances.append(nwb_instance)

    # Create SkeletonInstances container
    skeleton_instances = NwbSkeletonInstances(
        name="skeleton_instances", skeleton_instances=nwb_instances
    )

    # Create TrainingFrame
    training_frame_kwargs = {
        "name": name,
        "skeleton_instances": skeleton_instances,
    }

    # Add optional attributes
    if annotator is not None:
        training_frame_kwargs["annotator"] = annotator

    if source_video is not None:
        training_frame_kwargs["source_video"] = source_video
        training_frame_kwargs["source_video_frame_index"] = np.uint32(
            sleap_labeled_frame.frame_idx
        )

    return NwbTrainingFrame(**training_frame_kwargs)


def nwb_training_frame_to_sleap_labeled_frame(
    nwb_training_frame: NwbTrainingFrame,
    sleap_skeleton: SleapSkeleton,
    sleap_video: SleapVideo,
) -> SleapLabeledFrame:
    """Convert an ndx-pose TrainingFrame to sleap-io LabeledFrame.

    Args:
        nwb_training_frame: The ndx-pose TrainingFrame object to convert.
        sleap_skeleton: The sleap-io Skeleton to associate with instances.
        sleap_video: The sleap-io Video to associate with the frame.

    Returns:
        A sleap-io LabeledFrame object with equivalent data.
    """
    # Convert instances from NWB SkeletonInstances
    sleap_instances = []
    for (
        nwb_instance
    ) in nwb_training_frame.skeleton_instances.skeleton_instances.values():
        sleap_instance = nwb_skeleton_instance_to_sleap_instance(
            nwb_instance, sleap_skeleton
        )
        sleap_instances.append(sleap_instance)

    # Get frame index - use source_video_frame_index if available, otherwise 0
    frame_idx = (
        int(nwb_training_frame.source_video_frame_index)
        if nwb_training_frame.source_video_frame_index is not None
        else 0
    )

    return SleapLabeledFrame(
        video=sleap_video, frame_idx=frame_idx, instances=sleap_instances
    )
