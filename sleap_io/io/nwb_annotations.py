"""NWB formatted annotations."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import attrs
import numpy as np
import simplejson as json

# ndx-multisubjects requires Python 3.9+ (uses importlib.resources.files)
try:
    from ndx_multisubjects import (
        NdxMultiSubjectsNWBFile,
        SubjectsTable,
    )

    MULTISUBJECTS_AVAILABLE = True
except ImportError:
    NdxMultiSubjectsNWBFile = None  # type: ignore[misc, assignment]
    SubjectsTable = None  # type: ignore[misc, assignment]
    MULTISUBJECTS_AVAILABLE = False

from ndx_pose import PoseTraining as NwbPoseTraining
from ndx_pose import Skeleton as NwbSkeleton
from ndx_pose import SkeletonInstance as NwbInstance
from ndx_pose import SkeletonInstances as NwbSkeletonInstances
from ndx_pose import Skeletons as NwbSkeletons
from ndx_pose import SourceVideos as NwbSourceVideos
from ndx_pose import TrainingFrame as NwbTrainingFrame
from ndx_pose import TrainingFrames as NwbTrainingFrames
from pynwb import NWBHDF5IO, NWBFile
from pynwb.image import ImageSeries

from sleap_io import Instance as SleapInstance
from sleap_io import LabeledFrame as SleapLabeledFrame
from sleap_io import Labels as SleapLabels
from sleap_io import Skeleton as SleapSkeleton
from sleap_io import Video as SleapVideo
from sleap_io.io.utils import sanitize_filename
from sleap_io.io.video_reading import ImageVideo, MediaVideo
from sleap_io.io.video_writing import MJPEGFrameWriter


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


def extract_unique_subjects(
    labels: SleapLabels,
) -> Tuple[List[Dict[str, Any]], Dict[Any, int]]:
    """Extract unique subjects from SLEAP Labels based on instance tracks.

    Args:
        labels: The sleap-io Labels object to extract subjects from.

    Returns:
        A tuple containing:
        - List of dictionaries with subject metadata (subject_id, track)
        - Dict mapping Track objects to their index in the subjects list

    Notes:
        Subjects are identified by their Track objects. Instances without tracks
        are ignored in multi-subject exports.
    """
    unique_tracks = {}
    track_to_index = {}

    for labeled_frame in labels.labeled_frames:
        for instance in labeled_frame.instances:
            if instance.track is not None:
                if instance.track not in unique_tracks:
                    track_name = (
                        instance.track.name if instance.track.name else "unknown"
                    )
                    idx = len(unique_tracks)
                    unique_tracks[instance.track] = {
                        "subject_id": track_name,
                        "track": instance.track,
                    }
                    track_to_index[instance.track] = idx

    return list(unique_tracks.values()), track_to_index


def create_subjects_table(
    subjects_data: List[Dict[str, Any]],
    description: str = "Subjects in this session",
    subjects_metadata: Optional[List[Dict[str, Any]]] = None,
) -> SubjectsTable:
    """Create an ndx-multisubjects SubjectsTable from subject data.

    Args:
        subjects_data: List of dictionaries with subject_id keys extracted from tracks.
        description: Description for the SubjectsTable.
        subjects_metadata: Optional list of dictionaries containing additional metadata
            for each subject.

            Each dict can include fields like 'species', 'sex', 'age',
            'weight', 'genotype', 'strain', 'subject_description',
            'individual_subj_link'
            Order must match subjects_data.

    Returns:
        A SubjectsTable containing all subjects with their metadata.

    Example:
        ```python
        subjects_data = [{"subject_id": "mouse1"}, {"subject_id": "mouse2"}]
        metadata = [
            {"species": "Mus musculus", "sex": "M", "age": "P90D"},
            {"species": "Mus musculus", "sex": "F", "age": "P85D"}
        ]
        subjects_table = create_subjects_table(
                            subjects_data, subjects_metadata=metadata)
        ```
    """
    subjects_table = SubjectsTable(description=description)

    for i, subject in enumerate(subjects_data):
        # Start with required fields - sex and species have defaults per NWB spec
        row_data = {
            "subject_id": subject["subject_id"],
            "sex": "U",  # U = unknown (required by NWB spec)
            "species": "unknown",  # Required by NWB spec
        }

        # Add optional metadata if provided
        if subjects_metadata and i < len(subjects_metadata):
            metadata = subjects_metadata[i]
            # Add standard NWB subject fields if present
            for field in [
                "age",
                "subject_description",
                "genotype",
                "sex",
                "species",
                "strain",
                "weight",
                "individual_subj_link",
            ]:
                if field in metadata:
                    row_data[field] = metadata[field]

        subjects_table.add_row(**row_data)

    return subjects_table


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
    id: Optional[int] = None,
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
    node_locations = np.array(nwb_skeleton_instance.node_locations)

    # Handle visibility - use provided visibility or infer from NaN values
    if nwb_skeleton_instance.node_visibility is not None:
        node_visibility = np.array(nwb_skeleton_instance.node_visibility)
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
    name: Optional[str] = None,
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
    sleap_videos: List[SleapVideo],
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
) -> List[SleapVideo]:
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


def create_slp_to_nwb_video_map(
    sleap_videos: List[SleapVideo],
    nwb_source_videos: NwbSourceVideos,
) -> Dict[SleapVideo, ImageSeries]:
    """Create mapping from sleap-io Videos to NWB ImageSeries.

    Args:
        sleap_videos: List of sleap-io Video objects.
        nwb_source_videos: NWB SourceVideos container with ImageSeries.

    Returns:
        Dictionary mapping each sleap-io Video to its corresponding ImageSeries.

    Raises:
        ValueError: If the number of videos doesn't match or mapping fails.
    """
    # Create mapping based on order (assuming videos were created with
    # sleap_videos_to_nwb_source_videos)
    video_map = {}
    for i, sleap_video in enumerate(sleap_videos):
        video_name = f"video_{i}"
        if video_name in nwb_source_videos.image_series:
            video_map[sleap_video] = nwb_source_videos.image_series[video_name]
        else:
            raise ValueError(
                f"Could not find ImageSeries with name '{video_name}' in SourceVideos"
            )

    return video_map


def create_nwb_to_slp_video_map(
    nwb_image_series: List[ImageSeries],
    sleap_videos: List[SleapVideo],
) -> Dict[ImageSeries, SleapVideo]:
    """Create mapping from NWB ImageSeries to sleap-io Videos.

    Args:
        nwb_image_series: List of NWB ImageSeries objects.
        sleap_videos: List of sleap-io Video objects.

    Returns:
        Dictionary mapping each ImageSeries to its corresponding sleap-io Video.

    Raises:
        ValueError: If the number of videos doesn't match or mapping fails.
    """
    if len(nwb_image_series) != len(sleap_videos):
        raise ValueError(
            f"Number of NWB ImageSeries ({len(nwb_image_series)}) does not match "
            f"number of sleap videos ({len(sleap_videos)})"
        )

    # Create mapping based on order (assuming consistent ordering)
    video_map = {}
    for image_series, sleap_video in zip(nwb_image_series, sleap_videos):
        video_map[image_series] = sleap_video

    return video_map


def create_slp_to_nwb_skeleton_map(
    sleap_skeletons: List[SleapSkeleton],
    nwb_skeletons: NwbSkeletons,
) -> Dict[SleapSkeleton, NwbSkeleton]:
    """Create mapping from sleap-io Skeletons to NWB Skeletons.

    Args:
        sleap_skeletons: List of sleap-io Skeleton objects.
        nwb_skeletons: NWB Skeletons container with Skeleton objects.

    Returns:
        Dictionary mapping each sleap-io Skeleton to its corresponding NWB Skeleton.

    Raises:
        ValueError: If the number of skeletons doesn't match or mapping fails.
    """
    nwb_skeleton_list = list(nwb_skeletons.skeletons.values())

    if len(sleap_skeletons) != len(nwb_skeleton_list):
        raise ValueError(
            f"Number of sleap skeletons ({len(sleap_skeletons)}) does not match "
            f"number of NWB Skeletons ({len(nwb_skeleton_list)})"
        )

    # Create mapping based on order (assuming skeletons were created consistently)
    skeleton_map = {}
    for sleap_skeleton, nwb_skeleton in zip(sleap_skeletons, nwb_skeleton_list):
        skeleton_map[sleap_skeleton] = nwb_skeleton

    return skeleton_map


def create_nwb_to_slp_skeleton_map(
    nwb_skeletons: NwbSkeletons,
    sleap_skeletons: List[SleapSkeleton],
) -> Dict[NwbSkeleton, SleapSkeleton]:
    """Create mapping from NWB Skeletons to sleap-io Skeletons.

    Args:
        nwb_skeletons: NWB Skeletons container with Skeleton objects.
        sleap_skeletons: List of sleap-io Skeleton objects.

    Returns:
        Dictionary mapping each NWB Skeleton to its corresponding sleap-io Skeleton.

    Raises:
        ValueError: If the number of skeletons doesn't match or mapping fails.
    """
    nwb_skeleton_list = list(nwb_skeletons.skeletons.values())

    if len(nwb_skeleton_list) != len(sleap_skeletons):
        raise ValueError(
            f"Number of NWB Skeletons ({len(nwb_skeleton_list)}) does not match "
            f"number of sleap skeletons ({len(sleap_skeletons)})"
        )

    # Create mapping based on order (assuming consistent ordering)
    skeleton_map = {}
    for nwb_skeleton, sleap_skeleton in zip(nwb_skeleton_list, sleap_skeletons):
        skeleton_map[nwb_skeleton] = sleap_skeleton

    return skeleton_map


def sleap_labeled_frame_to_nwb_training_frame(
    sleap_labeled_frame: SleapLabeledFrame,
    slp_to_nwb_skeleton_map: Dict[SleapSkeleton, NwbSkeleton],
    source_video: Optional[ImageSeries] = None,
    name: str = "training_frame",
    annotator: Optional[str] = None,
    track_to_index: Optional[Dict[Any, int]] = None,
) -> NwbTrainingFrame:
    """Convert a sleap-io LabeledFrame to ndx-pose TrainingFrame.

    Args:
        sleap_labeled_frame: The sleap-io LabeledFrame object to convert.
        slp_to_nwb_skeleton_map: Mapping from sleap-io Skeletons to NWB Skeletons.
        source_video: Optional ImageSeries representing the source video.
        name: String identifier for the TrainingFrame.
        annotator: Optional name of annotator who labeled the frame.
        track_to_index: Optional mapping from Track objects to SubjectsTable row
            indices. When provided, SkeletonInstance.id will be set to the subject
            index for tracked instances, enabling linkage to the SubjectsTable.

    Returns:
        An ndx-pose TrainingFrame object with equivalent data.
    """
    # Convert instances to NWB SkeletonInstances
    nwb_instances = []
    for i, sleap_instance in enumerate(sleap_labeled_frame.instances):
        instance_name = f"instance_{i}"

        # Get the appropriate NWB skeleton for this instance
        nwb_skeleton = slp_to_nwb_skeleton_map[sleap_instance.skeleton]

        # Determine instance ID: use subject index if tracked, otherwise frame-local
        if track_to_index is not None and sleap_instance.track in track_to_index:
            instance_id = track_to_index[sleap_instance.track]
        else:
            instance_id = i

        nwb_instance = sleap_instance_to_nwb_skeleton_instance(
            sleap_instance, nwb_skeleton, name=instance_name, id=instance_id
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
    nwb_to_slp_skeleton_map: Dict[NwbSkeleton, SleapSkeleton],
    sleap_video: SleapVideo,
) -> SleapLabeledFrame:
    """Convert an ndx-pose TrainingFrame to sleap-io LabeledFrame.

    Args:
        nwb_training_frame: The ndx-pose TrainingFrame object to convert.
        nwb_to_slp_skeleton_map: Required mapping from NWB Skeletons to sleap-io
            Skeletons.
        sleap_video: The sleap-io Video to associate with the frame.

    Returns:
        A sleap-io LabeledFrame object with equivalent data.
    """
    # Convert instances from NWB SkeletonInstances
    sleap_instances = []
    for (
        nwb_instance
    ) in nwb_training_frame.skeleton_instances.skeleton_instances.values():
        # Get the appropriate sleap skeleton for this instance
        sleap_skeleton = nwb_to_slp_skeleton_map[nwb_instance.skeleton]

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


def sleap_labeled_frames_to_nwb_training_frames(
    sleap_labeled_frames: List[SleapLabeledFrame],
    slp_to_nwb_skeleton_map: Dict[SleapSkeleton, NwbSkeleton],
    slp_to_nwb_video_map: Dict[SleapVideo, ImageSeries],
    name: str = "TrainingFrames",
    annotator: Optional[str] = None,
    track_to_index: Optional[Dict[Any, int]] = None,
) -> NwbTrainingFrames:
    """Convert a list of sleap-io LabeledFrames to ndx-pose TrainingFrames container.

    Args:
        sleap_labeled_frames: List of sleap-io LabeledFrame objects to convert.
        slp_to_nwb_skeleton_map: Required mapping from sleap-io Skeletons to NWB
            Skeletons.
        slp_to_nwb_video_map: Required mapping from sleap-io Videos to ImageSeries.
        name: String identifier for the TrainingFrames container.
        annotator: Optional name of annotator who labeled the frames.
        track_to_index: Optional mapping from Track objects to SubjectsTable row
            indices. When provided, SkeletonInstance.id will be set to the subject
            index for tracked instances.

    Returns:
        An ndx-pose TrainingFrames container with TrainingFrame objects.
    """
    nwb_training_frames = []

    for frame_ind, sleap_labeled_frame in enumerate(sleap_labeled_frames):
        frame_name = f"frame_{frame_ind}"

        # Get corresponding source video from the mapping
        source_video = slp_to_nwb_video_map[sleap_labeled_frame.video]

        nwb_training_frame = sleap_labeled_frame_to_nwb_training_frame(
            sleap_labeled_frame,
            slp_to_nwb_skeleton_map=slp_to_nwb_skeleton_map,
            source_video=source_video,
            name=frame_name,
            annotator=annotator,
            track_to_index=track_to_index,
        )
        nwb_training_frames.append(nwb_training_frame)

    return NwbTrainingFrames(name=name, training_frames=nwb_training_frames)


def nwb_training_frames_to_sleap_labeled_frames(
    nwb_training_frames: NwbTrainingFrames,
    nwb_to_slp_skeleton_map: Dict[NwbSkeleton, SleapSkeleton],
    nwb_to_slp_video_map: Dict[ImageSeries, SleapVideo],
) -> List[SleapLabeledFrame]:
    """Convert ndx-pose TrainingFrames to a list of sleap-io LabeledFrames.

    Args:
        nwb_training_frames: The ndx-pose TrainingFrames container to convert.
        nwb_to_slp_skeleton_map: Required mapping from NWB Skeletons to sleap-io
            Skeletons.
        nwb_to_slp_video_map: Required mapping from ImageSeries to sleap-io Videos.

    Returns:
        A list of sleap-io LabeledFrame objects with equivalent data.

    Raises:
        ValueError: If a TrainingFrame's source_video is None.
        KeyError: If a TrainingFrame's source_video is not found in the mapping.
    """
    sleap_labeled_frames = []

    for nwb_training_frame in nwb_training_frames.training_frames.values():
        # Get corresponding sleap video using the required mapping
        if nwb_training_frame.source_video is None:
            raise ValueError(
                "TrainingFrame must have a source_video to convert to sleap-io "
                "LabeledFrame"
            )

        sleap_video = nwb_to_slp_video_map[nwb_training_frame.source_video]

        sleap_labeled_frame = nwb_training_frame_to_sleap_labeled_frame(
            nwb_training_frame, nwb_to_slp_skeleton_map, sleap_video
        )
        sleap_labeled_frames.append(sleap_labeled_frame)

    return sleap_labeled_frames


def sleap_labels_to_nwb_pose_training(
    sleap_labels: SleapLabels,
    name: str = "PoseTraining",
    annotator: Optional[str] = None,
    track_to_index: Optional[Dict[Any, int]] = None,
) -> Tuple[NwbPoseTraining, NwbSkeletons]:
    """Convert sleap-io Labels to ndx-pose PoseTraining container and Skeletons.

    Args:
        sleap_labels: The sleap-io Labels object to convert.
        name: String identifier for the PoseTraining container.
        annotator: Optional name of annotator who labeled the data.
        track_to_index: Optional mapping from Track objects to SubjectsTable row
            indices. When provided and use_multisubjects is True, SkeletonInstance.id
            will be set to the subject index for tracked instances, enabling linkage
            to the SubjectsTable.

    Returns:
        A tuple containing:
        - An ndx-pose PoseTraining container with training frames and source videos
        - An ndx-pose Skeletons container with all skeletons

    Raises:
        ValueError: If any video backend is not supported for NWB export.
    """
    # Convert all skeletons
    nwb_skeleton_list = []
    for i, sleap_skeleton in enumerate(sleap_labels.skeletons):
        # Ensure skeleton will have a unique name
        skeleton_name = sleap_skeleton.name if sleap_skeleton.name else f"skeleton_{i}"
        if skeleton_name == "skeleton":  # Default name, make it unique
            skeleton_name = f"skeleton_{i}"

        # Create temporary skeleton with the desired name
        temp_skeleton = sleap_skeleton_to_nwb_skeleton(sleap_skeleton)
        # Create new skeleton with proper name
        nwb_skeleton = NwbSkeleton(
            name=skeleton_name, nodes=temp_skeleton.nodes, edges=temp_skeleton.edges
        )
        nwb_skeleton_list.append(nwb_skeleton)

    # Create NwbSkeletons container
    nwb_skeletons = NwbSkeletons(name="Skeletons", skeletons=nwb_skeleton_list)

    # Convert videos to source videos container
    source_videos = sleap_videos_to_nwb_source_videos(
        sleap_labels.videos, name="source_videos"
    )

    # Create video mapping
    slp_to_nwb_video_map = create_slp_to_nwb_video_map(
        sleap_labels.videos, source_videos
    )

    # Create skeleton mapping
    slp_to_nwb_skeleton_map = create_slp_to_nwb_skeleton_map(
        sleap_labels.skeletons, nwb_skeletons
    )

    # Convert labeled frames to training frames
    training_frames = sleap_labeled_frames_to_nwb_training_frames(
        sleap_labels.labeled_frames,
        slp_to_nwb_skeleton_map=slp_to_nwb_skeleton_map,
        slp_to_nwb_video_map=slp_to_nwb_video_map,
        name="training_frames",  # Must be named this for PoseTraining
        annotator=annotator,
        track_to_index=track_to_index,
    )

    pose_training = NwbPoseTraining(
        name=name,
        training_frames=training_frames,
        source_videos=source_videos,
    )

    return pose_training, nwb_skeletons


def nwb_pose_training_to_sleap_labels(
    nwb_pose_training: NwbPoseTraining,
    nwb_skeletons: NwbSkeletons,
) -> SleapLabels:
    """Convert ndx-pose PoseTraining and Skeletons to sleap-io Labels.

    Args:
        nwb_pose_training: The ndx-pose PoseTraining container to convert.
        nwb_skeletons: The ndx-pose Skeletons container with skeleton definitions.

    Returns:
        A sleap-io Labels object with equivalent data.

    Raises:
        ValueError: If any ImageSeries format is not supported.
        KeyError: If video or skeleton mapping fails.
    """
    # Convert source videos back to sleap videos
    sleap_videos = nwb_source_videos_to_sleap_videos(nwb_pose_training.source_videos)

    # Convert all skeletons from NwbSkeletons container
    sleap_skeletons = [
        nwb_skeleton_to_sleap_skeleton(nwb_skeleton)
        for nwb_skeleton in nwb_skeletons.skeletons.values()
    ]

    # Create video mapping for conversion back
    nwb_to_slp_video_map = create_nwb_to_slp_video_map(
        list(nwb_pose_training.source_videos.image_series.values()), sleap_videos
    )

    # Create skeleton mapping for conversion back
    nwb_to_slp_skeleton_map = create_nwb_to_slp_skeleton_map(
        nwb_skeletons,
        sleap_skeletons,
    )

    # Convert training frames back to labeled frames
    labeled_frames = nwb_training_frames_to_sleap_labeled_frames(
        nwb_pose_training.training_frames,
        nwb_to_slp_skeleton_map,
        nwb_to_slp_video_map,
    )

    return SleapLabels(
        skeletons=sleap_skeletons,
        videos=sleap_videos,
        labeled_frames=labeled_frames,
    )


def save_labels(
    labels: SleapLabels,
    path: Union[Path, str],
    session_description: str = "SLEAP pose training data",
    identifier: Optional[str] = None,
    session_start_time: Optional[str] = None,
    annotator: Optional[str] = None,
    nwb_kwargs: Optional[dict] = None,
    use_multisubjects: bool = False,
    subjects_metadata: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save sleap-io Labels to an NWB file.

    Args:
        labels: The sleap-io Labels object to save.
        path: Path to save the NWB file.
        session_description: Description of the session (required).
        identifier: Unique identifier for the NWB file. If None, auto-generated.
        session_start_time: Start time of session (ISO format string). If None,
            uses current time.
        annotator: Name of the annotator who labeled the data. Optional.
        nwb_kwargs: Additional keyword arguments to pass to NWBFile constructor.
            Can include: session_id, experimenter, lab, institution,
            experiment_description, etc.
        use_multisubjects: If True, use NdxMultiSubjectsNWBFile and create a
            SubjectsTable based on instance tracks. Defaults to False for backward
            compatibility.
        subjects_metadata: Optional list of dictionaries containing additional metadata
            for each subject. Only used when use_multisubjects=True. Each dict can
            include fields like 'species', 'sex', 'age', 'weight', 'genotype', 'strain',
            'subject_description'. Order should match the order of unique tracks
            found in the labels.
    """
    # Set defaults for required fields
    if session_start_time is None:
        session_start_time = datetime.now().astimezone()
    elif isinstance(session_start_time, str):
        session_start_time = datetime.fromisoformat(session_start_time)

    if identifier is None:
        identifier = f"sleap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create NWB file with required parameters
    nwbfile_kwargs = {
        "session_description": session_description,
        "identifier": identifier,
        "session_start_time": session_start_time,
    }

    # Add any additional kwargs provided by user
    if nwb_kwargs is not None:
        nwbfile_kwargs.update(nwb_kwargs)

    # Create appropriate NWB file type
    track_to_index = None
    if use_multisubjects:
        if not MULTISUBJECTS_AVAILABLE:
            raise ImportError(
                "Multi-subject NWB support requires Python 3.9+. "
                "The ndx-multisubjects package uses importlib.resources.files "
                "which is not available in Python 3.8."
            )
        nwbfile = NdxMultiSubjectsNWBFile(**nwbfile_kwargs)

        # Extract subjects from tracks and create SubjectsTable
        subjects_data, track_to_index = extract_unique_subjects(labels)

        if len(subjects_data) == 0:
            raise ValueError(
                "No tracked instances found in labels. Cannot create multi-subject "
                "NWB file. Either add tracks to instances or set "
                "use_multisubjects=False."
            )

        # Check for untracked instances and warn
        total_instances = sum(len(lf.instances) for lf in labels.labeled_frames)
        tracked_instances = sum(
            1
            for lf in labels.labeled_frames
            for inst in lf.instances
            if inst.track is not None
        )
        if tracked_instances < total_instances:
            import warnings

            warnings.warn(
                f"{total_instances - tracked_instances} of {total_instances} instances "
                f"do not have tracks assigned and will not be linked to subjects in "
                f"the SubjectsTable. Use use_multisubjects=False to save all "
                f"instances without subject metadata.",
                stacklevel=2,
            )

        subjects_table = create_subjects_table(
            subjects_data,
            description="Subjects tracked in this session",
            subjects_metadata=subjects_metadata,
        )
        nwbfile.subjects_table = subjects_table
    else:
        nwbfile = NWBFile(**nwbfile_kwargs)

    # Convert SLEAP labels to NWB format
    pose_training, skeletons = sleap_labels_to_nwb_pose_training(
        labels, annotator=annotator, track_to_index=track_to_index
    )

    # Create behavior processing module
    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="processed behavioral data",
    )
    behavior_pm.add(skeletons)
    behavior_pm.add(pose_training)

    # Write to file
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)


def load_labels(path: Union[Path, str]) -> SleapLabels:
    """Load sleap-io Labels from an NWB file.

    Args:
        path: Path to the NWB file to load.

    Returns:
        A sleap-io Labels object with data from the NWB file.

    Notes:
        This function supports loading both regular NWB files and NWB files
        created with ndx-multisubjects. Multi-subject information (SubjectsTable)
        is not currently preserved when loading back to SLEAP format, but the
        pose data will be loaded correctly.
    """
    with NWBHDF5IO(path, mode="r") as io:
        nwbfile = io.read()

        # Get the behavior processing module
        behavior_pm = nwbfile.processing["behavior"]

        # Get PoseTraining and Skeletons containers
        pose_training = behavior_pm["PoseTraining"]
        skeletons = behavior_pm["Skeletons"]

        # Convert back to SLEAP format
        # Note: This works for both regular and multi-subject NWB files
        # Multi-subject metadata from SubjectsTable is not currently preserved
        labels = nwb_pose_training_to_sleap_labels(pose_training, skeletons)

        return labels


@attrs.define
class FrameInfo:
    """Information about a single frame in the MJPEG video.

    Attributes:
        video_ind: Index into the videos list indicating which video this frame
            came from.
        frame_idx: Original frame index in the source video.
    """

    video_ind: int
    frame_idx: int


@attrs.define
class FrameMap:
    """Map frames in an MJPEG video back to source videos for provenance tracking.

    This class stores the mapping between frames in an exported MJPEG video and their
    original source videos. It is serialized to/from a frame_map.json file alongside
    the MJPEG video.

    Attributes:
        frame_map_filename: Path to the frame_map.json file.
        sleap_labels_filename: Path to the SLEAP labels (.slp) file.
        nwb_filename: Path to the NWB file.
        mjpeg_filename: Path to the MJPEG video file.
        videos: List of Video objects representing the source videos.
        frames: List of FrameInfo objects, one per frame in the MJPEG video,
            indicating which source video and frame index each MJPEG frame came from.
    """

    frame_map_filename: Optional[str] = None
    sleap_labels_filename: Optional[str] = None
    nwb_filename: Optional[str] = None
    mjpeg_filename: Optional[str] = None
    videos: List[SleapVideo] = attrs.field(factory=list)
    frames: List[FrameInfo] = attrs.field(factory=list)

    @classmethod
    def from_labels(cls, labels: SleapLabels) -> "FrameMap":
        """Construct a FrameMap from a Labels object.

        Args:
            labels: Labels object containing labeled frames and videos.

        Returns:
            FrameMap instance with videos and frame mappings extracted from the
                labels.
        """
        # Copy videos from labels, preserving order
        videos = []
        for video in labels.videos:
            video_copy = SleapVideo(
                filename=video.filename,
                backend_metadata=video.backend_metadata,
                open_backend=False,
            )
            videos.append(video_copy)

        # Build frames list following labeled_frames order
        frames = []
        for lf in labels.labeled_frames:
            # Find video index in the videos list
            video_ind = labels.videos.index(lf.video)
            frame_info = FrameInfo(video_ind=video_ind, frame_idx=lf.frame_idx)
            frames.append(frame_info)

        return cls(videos=videos, frames=frames)

    def to_json(self) -> Dict[str, Any]:
        """Convert the FrameMap to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the FrameMap suitable for JSON serialization.
        """
        return {
            "frame_map_filename": self.frame_map_filename,
            "nwb_filename": self.nwb_filename,
            "mjpeg_filename": self.mjpeg_filename,
            "videos": [
                {
                    "filename": video.filename,
                    "backend_metadata": video.backend_metadata,
                }
                for video in self.videos
            ],
            "frames": [
                {"video_ind": frame.video_ind, "frame_idx": frame.frame_idx}
                for frame in self.frames
            ],
        }

    def save(self, frame_map_filename: Union[str, Path]):
        """Save the frame map to a JSON file.

        Args:
            frame_map_filename: Path to save the frame_map.json file.
        """
        # Update frame map filename with specified input.
        frame_map_filename = Path(frame_map_filename)
        self.frame_map_filename = sanitize_filename(frame_map_filename)

        # Prepare data for JSON serialization.
        json_data = self.to_json()

        # Write to disk.
        with open(self.frame_map_filename, "w") as f:
            json.dump(json_data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FrameMap":
        """Load a frame map from a JSON file.

        Args:
            path: Path to the frame_map.json file.

        Returns:
            FrameMap instance reconstructed from the JSON data.

        Raises:
            FileNotFoundError: If the frame_map.json file doesn't exist.
            json.JSONDecodeError: If the JSON file is malformed.
        """
        path = Path(path)

        with open(path, "r") as f:
            json_data = json.load(f)

        # Reconstruct Video objects without opening backends
        videos = []
        for video_data in json_data["videos"]:
            video = SleapVideo(
                filename=video_data["filename"],
                backend_metadata=video_data.get("backend_metadata", {}),
                open_backend=False,
            )
            videos.append(video)

        # Reconstruct FrameInfo objects
        frames = []
        for frame_data in json_data["frames"]:
            frames.append(
                FrameInfo(
                    video_ind=frame_data["video_ind"], frame_idx=frame_data["frame_idx"]
                )
            )

        return cls(
            frame_map_filename=str(path),
            nwb_filename=json_data.get("nwb_filename", None),
            mjpeg_filename=json_data.get("mjpeg_filename", None),
            videos=videos,
            frames=frames,
        )


def export_labeled_frames(
    labels: SleapLabels,
    frame_map_path: Union[str, Path],
    mjpeg_path: Union[str, Path],
    nwb_path: Optional[Union[str, Path]] = None,
) -> FrameMap:
    """Export labeled frames to an MJPEG video with provenance tracking.

    This function exports all labeled frames from a Labels object to a seekable
    MJPEG video file, along with a JSON frame map that tracks the provenance of
    each frame back to its original source video and frame index.

    Args:
        labels: Labels object containing labeled frames and videos to export.
        frame_map_path: Path where the frame map JSON file will be saved.
        mjpeg_path: Path where the output MJPEG video file will be saved.
        nwb_path: Optional path to associated NWB file for cross-referencing.

    Returns:
        FrameMap object containing all metadata and mappings for the exported
        video, including paths to output files and frame-to-video provenance.

    Raises:
        ValueError: If labels contain no labeled frames.
        OSError: If output files cannot be written.

    Example:
        ```python
        labels = load_file("dataset.slp")
        frame_map = export_labeled_frames(
            labels,
            frame_map_path="exports/frame_map.json",
            mjpeg_path="exports/training_data.avi",
            nwb_path="exports/dataset.nwb"
        )
        print(f"Exported {len(frame_map.frames)} frames to {frame_map.mjpeg_filename}")
        ```
    """
    # Build FrameMap from labels and set metadata
    frame_map = FrameMap.from_labels(labels)
    frame_map.mjpeg_filename = sanitize_filename(mjpeg_path)
    frame_map.frame_map_filename = sanitize_filename(frame_map_path)
    frame_map.nwb_filename = (
        sanitize_filename(nwb_path) if nwb_path is not None else None
    )

    # Export frames to MJPEG using MJPEGFrameWriter
    with MJPEGFrameWriter(mjpeg_path) as writer:
        for lf in labels.labeled_frames:
            # Get frame data from the video at the specified frame index
            frame_data = lf.video[lf.frame_idx]
            writer.write_frame(frame_data)

    # Save the frame map JSON alongside the MJPEG file
    frame_map.save(frame_map_path)

    return frame_map


def export_labels(
    labels: SleapLabels,
    output_dir: Union[Path, str],
    mjpeg_filename: str = "annotated_frames.avi",
    frame_map_filename: str = "frame_map.json",
    nwb_filename: str = "pose_training.nwb",
    clean: bool = True,
    use_multisubjects: bool = False,
    subjects_metadata: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Export Labels to NWB format with MJPEG video and frame map.

    This function exports a Labels object to NWB format along with an MJPEG video
    containing all labeled frames and a JSON frame map for provenance tracking.
    The exported NWB file will reference the MJPEG video, allowing for efficient
    storage and retrieval of training data.

    Args:
        labels: Labels object containing labeled frames to export.
        output_dir: Directory path where all output files will be saved.
        mjpeg_filename: Name of the output MJPEG video file.
            Defaults to "annotated_frames.avi".
        frame_map_filename: Name of the frame map JSON file.
            Defaults to "frame_map.json".
        nwb_filename: Name of the output NWB file.
            Defaults to "pose_training.nwb".
        clean: If True, remove empty frames and predictions before export using
            `Labels.remove_predictions(clean=True)`. Defaults to True.
        use_multisubjects: If True, use NdxMultiSubjectsNWBFile and create a
            SubjectsTable based on instance tracks. Defaults to False.
        subjects_metadata: Optional list of dictionaries containing additional metadata
            for each subject. Only used when use_multisubjects=True.

    Raises:
        ValueError: If labels contain no labeled frames after cleaning.

    Example:
        ```python
        labels = load_file("dataset.slp")
        export_labels(
            labels,
            output_dir="exports",
            mjpeg_filename="training_data.avi",
            nwb_filename="dataset.nwb"
        )
        ```

    Notes:
        The function creates a copy of the labels before processing to avoid
        modifying the original data. All file paths are relative to the specified
        output directory, which will be created if it doesn't exist.
    """
    # Make a copy of the labels since we'll be mutating them
    labels = deepcopy(labels)

    # Clean labels if requested to remove empty frames and predictions
    if clean:
        labels.remove_predictions(clean=True)

    # Check that we have frames to export after cleaning
    if len(labels.labeled_frames) == 0:
        raise ValueError(
            "No labeled frames found to export (labels may be empty). "
            "Try exporting with clean=False if you want to export empty frames."
        )

    # Convert paths to Path objects and create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build full paths for output files
    mjpeg_path = output_dir / mjpeg_filename
    frame_map_path = output_dir / frame_map_filename
    nwb_path = output_dir / nwb_filename

    # Export labeled frames to MJPEG AVI file and create frame map
    frame_map = export_labeled_frames(
        labels, frame_map_path, mjpeg_path, nwb_path=nwb_path
    )

    # Update the labels to point to the new MJPEG video
    mjpeg_video = SleapVideo.from_filename(mjpeg_path)
    for lf_ind in range(len(labels)):
        lf = labels[lf_ind]

        # Sanity checks:
        video_ind, frame_idx = labels.videos.index(lf.video), lf.frame_idx
        frame_info = frame_map.frames[lf_ind]
        assert video_ind == frame_info.video_ind
        assert frame_idx == frame_info.frame_idx

        lf.video = mjpeg_video
        lf.frame_idx = lf_ind
    labels.videos = [mjpeg_video]

    # Now save the NWB file as normal
    save_labels(
        labels,
        nwb_path,
        use_multisubjects=use_multisubjects,
        subjects_metadata=subjects_metadata,
    )
