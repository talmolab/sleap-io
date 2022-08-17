from typing import List, Optional
from pathlib import Path
import datetime

import pandas as pd
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from ndx_pose import PoseEstimationSeries, PoseEstimation


from sleap_io import PredictedInstance


def _extract_labels_data(labels):
    data_list = list()

    for labeled_frame in labels.labeled_frames:

        frame_idx = labeled_frame.frame_idx
        video_path = labeled_frame.video.filename

        for instance in labeled_frame.instances:
            predicted_instance = isinstance(instance, PredictedInstance)

            skeleton = instance.skeleton
            skeleton_name = skeleton.name
            track_name = instance.track.name if instance.track else "untracked"

            for node in skeleton.nodes:
                node_name = node.name

                predicted_points = instance.points[node]
                x, y, score = (
                    predicted_points.x,
                    predicted_points.y,
                    predicted_points.score,
                )

                row_dict = dict(
                    frame_idx=frame_idx,
                    x=x,
                    y=y,
                    score=score,
                    node_name=node_name,
                    skeleton_name=skeleton_name,
                    predicted_instance=predicted_instance,  # True for predicted instance, False for instance.
                    track_name=track_name,
                    video_path=video_path,
                )
                data_list.append(row_dict)

    labels_df = pd.DataFrame(data_list)
    index = [
        "track_name",
        "skeleton_name",
        "node_name",
        "predicted_instance",
        "video_path",
        "frame_idx",
    ]

    # Reformat as columns for dict-like hierarchical data access.
    labels_tidy_df = (
        labels_df.set_index(index)
        .unstack(level=[0, 1, 2, 3, 4])
        .swaplevel(0, -1, axis=1)
        .sort_index(axis=1)  # Better format for columns
        .sort_index(axis=0)  # Sorts by frames
    )

    return labels_tidy_df


def write_labels_to_nwb(labels, nwbfile_path):

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


def append_labels_data_to_nwb(labels, nwbfile):

    labels_data_df = _extract_labels_data(labels)

    for video_index, video in enumerate(labels.videos):
        video_path = Path(video.filename)

        processing_module_name = f"SLEAP_VIDEO_{video_index:03}_{video_path.stem}"
        nwb_processing_module = get_processing_module_for_video(
            processing_module_name, nwbfile
        )

        name_of_tracks_in_video = (
            labels_data_df[str(video_path)]
            .columns.get_level_values("track_name")
            .unique()
        )

        for track_index, track_name in enumerate(name_of_tracks_in_video):
            pose_estimation_container = build_pose_estimation_container_for_track(
                labels_data_df, labels, track_name, video
            )
            nwb_processing_module.add(pose_estimation_container)

    return nwbfile

def get_processing_module_for_video(processing_module_name, nwbfile):
    if processing_module_name in nwbfile.processing:
        nwb_processing_module = nwbfile.processing[processing_module_name]
    else:
        nwb_processing_module = nwbfile.create_processing_module(
            name=processing_module_name, description=f"Processed SLEAP pose data"
        )

    return nwb_processing_module


def build_pose_estimation_container_for_track(
    labels_data_df, labels, track_name, video
):
    video_path = Path(video.filename)

    all_track_sekletons = (
        labels_data_df[str(video_path), track_name]
        .columns.get_level_values("skeleton_name")
        .unique()
    )
    # Assuming only one skeleton per track
    skeleton_name = all_track_sekletons[0]
    skeleton = next(
        skeleton for skeleton in labels.skeletons if skeleton.name == skeleton_name
    )

    track_data_df = labels_data_df[
        str(video_path),
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
        original_videos=[f"{video_path}"],
        labeled_videos=[f"{video_path}"],
        # dimensions=np.array([[video.backend.height, video.backend.width]]),
        # scorer=str(labels.provenance),
        source_software="SLEAP",
        # source_software_version=f"{sleap.__version__}",
        nodes=skeleton.node_names,
        edges=np.array(skeleton.edge_inds).astype("uint64"),
    )
    
    return pose_estimation_container


def build_track_pose_estimation_list(track_data_df):

    name_of_nodes_in_track = track_data_df.columns.get_level_values(
        "node_name"
    ).unique()

    pose_estimation_series_list: List[PoseEstimationSeries] = []
    for node_name in name_of_nodes_in_track:
        # Add predicted instances only
        predicted_instance = True

        # Should nan's be droped?
        data_for_node = track_data_df[
            node_name,
            predicted_instance,
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
