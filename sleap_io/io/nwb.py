"""Functions to write and read from the neurodata without borders (NWB) format. 
"""
from copy import deepcopy
from typing import List, Optional, Generator
from pathlib import Path
import datetime
import uuid

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from ndx_pose import PoseEstimationSeries, PoseEstimation


from sleap_io import PredictedInstance, Labels, Video
from sleap_io.model.labeled_frame import LabeledFrame


def _extract_predicted_instances_data(labels: Labels) -> pd.DataFrame:
    """Auxiliary function to structure the predicted instances' data for nwb write.

    Args:
        labels (Labels): A general label object.

    Raises:
        ValueError: If no frames in the label objects contain predicted instances.

    Returns:
        pd.DataFrame: A pandas data frame with the structured data with
        hierarchical columns. The column hierarchy is:
                "video_path",
                "skeleton_name",
                "track_name",
                "node_name",
        And it is indexed by the frames.
    """

    # Form pairs of labeled_frames and predicted instances
    labeled_frames = labels.labeled_frames
    all_frame_instance_tuples: Generator[
        tuple[LabeledFrame, PredictedInstance], None, None
    ] = (
        (label_frame, instance)  # type: ignore
        for label_frame in labeled_frames
        for instance in label_frame.predicted_instances
    )

    # Extract the data
    data_list = list()
    for labeled_frame, instance in all_frame_instance_tuples:
        # Traverse the nodes of the instances's skeleton
        skeleton = instance.skeleton
        for node in skeleton.nodes:
            row_dict = dict(
                frame_idx=labeled_frame.frame_idx,
                x=instance.points[node].x,
                y=instance.points[node].y,
                score=instance.points[node].score,  # type: ignore[attr-defined]
                node_name=node.name,
                skeleton_name=skeleton.name,
                track_name=instance.track.name if instance.track else "untracked",
                video_path=labeled_frame.video.filename,
            )
            data_list.append(row_dict)

    if not data_list:
        raise ValueError("No predicted instances found in labels object")

    labels_df = pd.DataFrame(data_list)

    # Reformat the data with columns for dict-like hierarchical data access.
    index = [
        "skeleton_name",
        "track_name",
        "node_name",
        "video_path",
        "frame_idx",
    ]

    labels_tidy_df = (
        labels_df.set_index(index)
        .unstack(level=[0, 1, 2, 3])
        .swaplevel(0, -1, axis=1)  # video_path on top while x, y score on bottom
        .sort_index(axis=1)  # Better format for columns
        .sort_index(axis=0)  # Sorts by frames
    )

    return labels_tidy_df


def write_labels_to_nwb(
    labels: Labels,
    nwbfile_path: str,
    nwb_file_kwargs: Optional[dict] = None,
    pose_estimation_metadata: Optional[dict] = None,
):
    """Write labels to an nwb file and save it to the nwbfile_path given

    Args:
        labels (Labels): A general label object
        nwbfile_path (str): The path where the nwb file is to be written
        nwb_file_kwargs (Optional[dict], optional): A dict containing metadata to
        the nwbfile. Example:
        nwb_file_kwargs = {
            'session_description: 'your_session_description',
            'identifier': 'your session_identifier',
        }
        For a full list of possible values see:
        https://pynwb.readthedocs.io/en/stable/pynwb.file.html#pynwb.file.NWBFile

        Defaults to None and default values are used to generate the nwb file.

        pose_estimation_metadata (dict): This argument has a dual purpose:

        1) It can be used to pass time information about the video which is
        necessary for synchronizing frames in pose estimation tracking to other
        modalities. Either the video timestamps can be passed to
        This can be used to pass the timestamps with the key `video_timestamps`
        or the sampling rate with key`video_sample_rate`.

        e.g. pose_estimation_metadata["video_timestamps"] = np.array(timestamps)
        or   pose_estimation_metadata["video_sample_rate] = 15  # In Hz

        2) The other use of this dictionary is to ovewrite sleap-io default
        arguments for the PoseEstimation container.
        see https://github.com/rly/ndx-pose for a full list or arguments.
    """

    nwb_file_kwargs = nwb_file_kwargs or dict()

    # Add required values for nwbfile if not present
    session_description = nwb_file_kwargs.get(
        "session_description", "Processed SLEAP pose data"
    )
    session_start_time = nwb_file_kwargs.get(
        "session_start_time", datetime.datetime.now(datetime.timezone.utc)
    )
    identifier = nwb_file_kwargs.get("identifier", str(uuid.uuid1()))

    nwb_file_kwargs.update(
        session_description=session_description,
        session_start_time=session_start_time,
        identifier=identifier,
    )

    nwbfile = NWBFile(**nwb_file_kwargs)
    nwbfile = append_labels_data_to_nwb(labels, nwbfile, pose_estimation_metadata)

    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(nwbfile)


def append_labels_data_to_nwb(
    labels: Labels, nwbfile: NWBFile, pose_estimation_metadata: Optional[dict] = None
) -> NWBFile:
    """Append data from a Labels object to an in-memory nwb file.

    Args:
        labels (Labels): A general labels object
        nwbfile (NWBFile): And in-memory nwbfile where the data is to be appended.

        pose_estimation_metadata (dict): This argument has a dual purpose:

        1) It can be used to pass time information about the video which is
        necessary for synchronizing frames in pose estimation tracking to other
        modalities. Either the video timestamps can be passed to
        This can be used to pass the timestamps with the key `video_timestamps`
        or the sampling rate with key`video_sample_rate`.

        e.g. pose_estimation_metadata["video_timestamps"] = np.array(timestamps)
        or   pose_estimation_metadata["video_sample_rate"] = 15  # In Hz

        2) The other use of this dictionary is to ovewrite sleap-io default
        arguments for the PoseEstimation container.
        see https://github.com/rly/ndx-pose for a full list or arguments.

    Returns:
        NWBFile: An in-memory nwbfile with the data from the labels object appended.
    """

    pose_estimation_metadata = pose_estimation_metadata or dict()

    # Extract default metadata
    provenance = labels.provenance
    default_metadata = dict(scorer=str(provenance))
    sleap_version = provenance.get("sleap_version", None)
    default_metadata["source_software_version"] = sleap_version

    labels_data_df = _extract_predicted_instances_data(labels)

    # For every video create a processing module
    for video_index, video in enumerate(labels.videos):

        video_path = Path(video.filename)
        processing_module_name = f"SLEAP_VIDEO_{video_index:03}_{video_path.stem}"
        nwb_processing_module = get_processing_module_for_video(
            processing_module_name, nwbfile
        )

        # Propagate video metadata
        default_metadata["original_videos"] = [f"{video.filename}"]  # type: ignore
        default_metadata["labeled_videos"] = [f"{video.filename}"]  # type: ignore

        # Overwrite default with the user provided metadata
        default_metadata.update(pose_estimation_metadata)

        # For every track in that video create a PoseEstimation container
        name_of_tracks_in_video = (
            labels_data_df[video.filename]
            .columns.get_level_values("track_name")
            .unique()
        )

        for track_index, track_name in enumerate(name_of_tracks_in_video):
            pose_estimation_container = build_pose_estimation_container_for_track(
                labels_data_df,
                labels,
                track_name,
                video,
                default_metadata,
            )
            nwb_processing_module.add(pose_estimation_container)

    return nwbfile


def get_processing_module_for_video(
    processing_module_name: str, nwbfile: NWBFile
) -> ProcessingModule:
    """Auxiliary function to create a processing module.

    Checks for the processing module existence and creates if not available.

    Args:
        processing_module_name (str): The name of the processing module.
        nwbfile (NWBFile): The nwbfile to attach the processing module to.

    Returns:
        ProcessingModule: An nwb processing module with the desired name.
    """
    description = "Processed SLEAP data"
    processing_module = (
        nwbfile.processing[processing_module_name]
        if processing_module_name in nwbfile.processing
        else nwbfile.create_processing_module(
            name=processing_module_name, description=description
        )
    )
    return processing_module


def build_pose_estimation_container_for_track(
    labels_data_df: pd.DataFrame,
    labels: Labels,
    track_name: str,
    video: Video,
    pose_estimation_metadata: dict,
) -> PoseEstimation:
    """Creates a PoseEstimation container for a track.

    Args:
        labels_data_df (pd.DataFrame): A pandas object with the data corresponding
        to the predicted instances associated to this labels object.
        labels (Labels): A general labels object
        track_name (str): The name of the track in labels.tracks
        video (Video): The video to which data belongs to

    Returns:
        PoseEstimation: A PoseEstimation multicontainer where the time series
        of all the node trajectories in the track are stored. One time series per
        node.
    """
    # Copy metadata for local use and modification
    pose_estimation_metadata_copy = deepcopy(pose_estimation_metadata)
    video_path = Path(video.filename)

    all_track_skeletons = (
        labels_data_df[video.filename]
        .columns.get_level_values("skeleton_name")
        .unique()
    )

    # Assuming only one skeleton per track
    skeleton_name = all_track_skeletons[0]
    skeleton = next(
        skeleton for skeleton in labels.skeletons if skeleton.name == skeleton_name
    )

    track_data_df = labels_data_df[
        video.filename,
        skeleton.name,
        track_name,
    ]

    # Combine each node's PoseEstimationSeries to create a PoseEstimation container
    timestamps = pose_estimation_metadata_copy.pop("video_timestamps", None)
    sample_rate = pose_estimation_metadata_copy.pop("video_sample_rate", 1.0)
    if timestamps is None:
        # Keeps backward compatbility.
        timestamps = np.arange(track_data_df.shape[0]) * sample_rate
    else:
        timestamps = np.asarray(timestamps)

    pose_estimation_series_list = build_track_pose_estimation_list(
        track_data_df, timestamps
    )

    # Arrange and mix metadata
    pose_estimation_container_kwargs = dict(
        name=f"track={track_name}",
        description=f"Estimated positions of {skeleton.name} in video {video_path.name}",
        pose_estimation_series=pose_estimation_series_list,
        nodes=skeleton.node_names,
        edges=np.array(skeleton.edge_inds).astype("uint64"),
        source_software="SLEAP",
        # dimensions=np.array([[video.backend.height, video.backend.width]]),
    )

    pose_estimation_container_kwargs.update(**pose_estimation_metadata_copy)
    pose_estimation_container = PoseEstimation(**pose_estimation_container_kwargs)

    return pose_estimation_container


def build_track_pose_estimation_list(
    track_data_df: pd.DataFrame, timestamps: ArrayLike
) -> List[PoseEstimationSeries]:
    """An auxiliar function to build a list of PoseEstimationSeries associated with
    a Track object.

    Args:
        track_data_df (pd.DataFrame): A pandas DataFrame object containing the
        trajectories for all the nodes associated with a specific track.

    Returns:
        List[PoseEstimationSeries]: The list of all the PoseEstimationSeries.
        One for each node.
    """

    name_of_nodes_in_track = track_data_df.columns.get_level_values(
        "node_name"
    ).unique()

    pose_estimation_series_list: List[PoseEstimationSeries] = []
    for node_name in name_of_nodes_in_track:

        # Drop data with missing values
        data_for_node = track_data_df[node_name].dropna(axis="index", how="any")

        node_trajectory = data_for_node[["x", "y"]].to_numpy()
        confidence = data_for_node["score"].to_numpy()

        reference_frame = (
            "The coordinates are in (x, y) relative to the top-left of the image. "
            "Coordinates refer to the midpoint of the pixel. "
            "That is, t the midpoint of the top-left pixel is at (0, 0), whereas "
            "the top-left corner of that same pixel is at (-0.5, -0.5)."
        )

        pose_estimation_kwargs = dict(
            name=f"{node_name}",
            description=f"Sequential trajectory of {node_name}.",
            data=node_trajectory,
            unit="pixels",
            reference_frame=reference_frame,
            confidence=confidence,
            confidence_definition="Point-wise confidence scores.",
        )

        # Add timestamps or only rate if the timestamps are uniform
        frames = data_for_node.index.values
        timestamps_for_data = timestamps[frames]
        sample_periods = np.diff(timestamps_for_data)
        if sample_periods.size == 0:
            rate = None  # This is the case with only one data point
        else:
            # Difference below 0.1 ms do not matter for behavior in videos
            uniform_samples = np.unique(sample_periods.round(5)).size == 1
            rate = 1 / sample_periods[0] if uniform_samples else None

        if rate:
            # Video sample rates are ints but nwb expect floats
            rate = float(int(rate))
            pose_estimation_kwargs.update(
                rate=rate, starting_time=timestamps_for_data[0]
            )
        else:
            pose_estimation_kwargs.update(timestamps=timestamps_for_data)

        # Build the pose estimation object and attach it to the list
        pose_estimation_series = PoseEstimationSeries(**pose_estimation_kwargs)
        pose_estimation_series_list.append(pose_estimation_series)

    return pose_estimation_series_list
