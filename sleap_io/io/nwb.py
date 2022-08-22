"""Functions to write and read from the neurodata without borders (NWB) format. 
"""
from tkinter import Label
from typing import List, Optional
from pathlib import Path
import datetime

import pandas as pd
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from ndx_pose import PoseEstimationSeries, PoseEstimation


from sleap_io import PredictedInstance, Labels, Video


def _extract_predicted_instances_data(labels: Labels) -> pd.DataFrame:
    """Auxiliar function to structure the predicted instances' data for nwb write.

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
    all_frame_instance_tuples = (
        (label_frame, instance)
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
                score=instance.points[node].score,
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


def write_labels_to_nwb(labels: Labels, nwbfile_path: str):
    """Write labels to an nwb file and save it to the nwbfile_path given

    Args:
        labels (Labels): A general label object
        nwbfile_path (str): The path where the nwb file is to be written
    """

    session_description: str = "Processed SLEAP pose data"
    session_start_time = datetime.datetime.now(datetime.timezone.utc)
    identifier = "identifier"

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
    )

    nwbfile = append_labels_data_to_nwb(labels, nwbfile)

    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(nwbfile)


def append_labels_data_to_nwb(labels: Labels, nwbfile: NWBFile) -> NWBFile:
    """

    Args:
        labels (Labels): A general labels object
        nwbfile (NWBFile): And in-memory nwbfile where the data is to be appended.

    Returns:
        NWBFile: An in-memory nwbfile with the data from the labels object appended.
    """

    labels_data_df = _extract_predicted_instances_data(labels)

    # For every video create a processing module
    for video_index, video in enumerate(labels.videos):

        video_path = Path(video.filename)
        processing_module_name = f"SLEAP_VIDEO_{video_index:03}_{video_path.stem}"
        nwb_processing_module = get_processing_module_for_video(
            processing_module_name, nwbfile
        )

        name_of_tracks_in_video = (
            labels_data_df[video.filename]
            .columns.get_level_values("track_name")
            .unique()
        )

        # For every track in that video create a PoseEstimation container
        for track_index, track_name in enumerate(name_of_tracks_in_video):
            pose_estimation_container = build_pose_estimation_container_for_track(
                labels_data_df, labels, track_name, video
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
    if processing_module_name in nwbfile.processing:
        nwb_processing_module = nwbfile.processing[processing_module_name]
    else:
        nwb_processing_module = nwbfile.create_processing_module(
            name=processing_module_name, description=f"Processed SLEAP data"
        )

    return nwb_processing_module


def build_pose_estimation_container_for_track(
    labels_data_df: pd.DataFrame, labels: Labels, track_name: str, video: Video
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
    video_path = Path(video.filename)

    all_track_skeletons = (
        labels_data_df[video.filename, track_name]
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
        track_name,
        skeleton.name,
    ]

    pose_estimation_series_list = build_track_pose_estimation_list(track_data_df)

    # Combine each node's PoseEstimationSeries to create a PoseEstimation container
    container_description = (
        f"Estimated positions of {skeleton.name} in video {video_path.name} "
        f"using SLEAP."
    )
    pose_estimation_container = PoseEstimation(
        name=f"track={track_name}",
        pose_estimation_series=pose_estimation_series_list,
        description=container_description,
        original_videos=[f"{video.filename}"],
        labeled_videos=[f"{video.filename}"],
        source_software="SLEAP",
        nodes=skeleton.node_names,
        edges=np.array(skeleton.edge_inds).astype("uint64"),
        # dimensions=np.array([[video.backend.height, video.backend.width]]),
        # scorer=str(labels.provenance),
        # source_software_version=f"{sleap.__version__}",
        # To-discuss this in PR.
    )

    return pose_estimation_container


def build_track_pose_estimation_list(
    track_data_df: pd.DataFrame,
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
        # Add predicted instances only

        # Should nan's be droped?
        data_for_node = track_data_df[
            node_name,
        ]

        node_trajectory = data_for_node[["x", "y"]].to_numpy()
        confidence = data_for_node["score"].to_numpy()

        # Fake data, should be extracted from video.
        timestamps = np.arange(data_for_node.shape[0]).astype("float")

        # Build the pose estimation object and attach it to the list
        pose_estimation_series_list.append(
            PoseEstimationSeries(
                name=f"{node_name}",
                description=f"Sequential trajectory of {node_name}.",
                data=node_trajectory,
                unit="pixels",
                reference_frame="No reference.",
                timestamps=timestamps,
                confidence=confidence,
                confidence_definition="Point-wise confidence scores.",
            )
        )

    return pose_estimation_series_list
