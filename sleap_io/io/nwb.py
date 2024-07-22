"""Functions to write and read from the neurodata without borders (NWB) format."""

from copy import deepcopy
from typing import List, Optional, Union
from pathlib import Path
import datetime
import uuid
import re
from imageio.v3 import imwrite

import pandas as pd
import numpy as np

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = np.ndarray

from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.file import Subject
from pynwb.image import ImageSeries

from ndx_pose import (
    PoseEstimationSeries,
    PoseEstimation,
    Skeleton as NWBSkeleton,
    Skeletons,
    SkeletonInstance,
    SkeletonInstances,
    TrainingFrame,
    TrainingFrames,
    PoseTraining,
    SourceVideos,
)

from sleap_io import (
    Labels,
    Video,
    LabeledFrame,
    Track,
    Skeleton as SLEAPSkeleton,
    Instance,
    PredictedInstance,
    Edge,
    Node,
)
from sleap_io.io.utils import convert_predictions_to_dataframe


def pose_training_to_labels(pose_training: PoseTraining) -> Labels:  # type: ignore[return]
    """Creates a Labels object from an NWB PoseTraining object.

    Args:
        pose_training: An NWB PoseTraining object.

    Returns:
        A Labels object.
    """
    labeled_frames = []
    skeletons = {}
    for training_frame in pose_training.training_frames.training_frames.values():
        source_video = training_frame.source_video
        if source_video.format == "external" and len(source_video.external_file) == 1:
            video = Video(filename=source_video.external_file[0])
        else:
            raise NotImplementedError("Only single-file external videos are supported.")

        frame_idx = training_frame.source_video_frame_index
        instances = []
        for instance in training_frame.skeleton_instances.skeleton_instances.values():
            if instance.skeleton.name not in skeletons:
                skeletons[instance.skeleton.name] = nwb_skeleton_to_sleap(
                    instance.skeleton
                )
            skeleton = skeletons[instance.skeleton.name]
            instances.append(
                Instance.from_numpy(points=instance.node_locations, skeleton=skeleton)
            ) # `track` field is not stored in `SkeletonInstance` objects
        labeled_frames.append(
            LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )
    return Labels(labeled_frames=labeled_frames)


def nwb_skeleton_to_sleap(skeleton: NWBSkeleton) -> SLEAPSkeleton:  # type: ignore[return]
    """Converts an NWB skeleton to a SLEAP skeleton.

    Args:
        skeleton: An NWB skeleton.

    Returns:
        A SLEAP skeleton.
    """
    nodes = [Node(name=node) for node in skeleton.nodes]
    edges = [Edge(source=edge[0], destination=edge[1]) for edge in skeleton.edges]
    return SLEAPSkeleton(
        nodes=nodes,
        edges=edges,
        name=skeleton.name,
    )


def labels_to_pose_training(
    labels: Labels, img_paths: Optional[list[str]] = None, **kwargs
) -> PoseTraining:  # type: ignore[return]
    """Creates an NWB PoseTraining object from a Labels object.

    Args:
        labels: A Labels object.
        img_paths: An optional list of image paths for the labeled frames.

    Returns:
        A PoseTraining object.
    """
    skeletons_list: list[NWBSkeleton] = []  # type: ignore[assignment]
    skeletons = Skeletons(skeletons=skeletons_list)  # type: ignore[assignment]
    training_frame_list = []
    for i, labeled_frame in enumerate(labels.labeled_frames):
        training_frame_name = f"training_frame_{i}"
        training_frame_annotator = "N/A"
        skeleton_instances_list = []

        for instance in labeled_frame.instances:
            if isinstance(instance, PredictedInstance):
                continue

            if instance.skeleton not in skeletons_list:
                skeletons_list.append(instance.skeleton)
                skeletons = Skeletons(skeletons=skeletons_list)
            skeleton_instance = instance_to_skeleton_instance(instance)
            skeleton_instances_list.append(skeleton_instance)

        training_frame_skeleton_instances = SkeletonInstances(
            skeleton_instances=skeleton_instances_list
        )
        training_frame_video = labeled_frame.video
        training_frame_video_index = labeled_frame.frame_idx

        source_video = ImageSeries(
            name=training_frame_name,
            description=training_frame_annotator,
            unit="NA",
            format="external",
            external_file=[training_frame_video.filename],
            dimension=[
                training_frame_video.shape[1],
                training_frame_video.shape[2],
            ],
            starting_frame=[0],
            rate=30.0,  # change to `video.backend.fps` when available
        )
        training_frame = TrainingFrame(
            name=training_frame_name,
            annotator=training_frame_annotator,
            skeleton_instances=training_frame_skeleton_instances,
            source_video=source_video,
            source_video_frame_index=training_frame_video_index,
        )
        training_frame_list.append(training_frame)

    training_frames = TrainingFrames(training_frames=training_frame_list)
    pose_training = PoseTraining(
        training_frames=training_frames,
        source_videos=videos_to_source_videos(labels.videos),
    )
    return pose_training


def slp_skeleton_to_nwb(skeleton: SLEAPSkeleton) -> NWBSkeleton:  # type: ignore[return]
    """Converts SLEAP skeleton to NWB skeleton.

    Args:
        skeleton: A SLEAP skeleton.

    Returns:
        An NWB skeleton.
    """
    skeleton_edges = {i: node for i, node in enumerate(skeleton.nodes)}
    nwb_edges = []
    for i, source in skeleton_edges.items():
        for destination in list(skeleton_edges.values())[i:]:
            if Edge(source, destination) in skeleton.edges:
                nwb_edges.append([i, list(skeleton_edges.values()).index(destination)])

    return NWBSkeleton(
        name=skeleton.name,
        nodes=skeleton.node_names,
        edges=np.array(nwb_edges, dtype=np.uint8),
    )


def instance_to_skeleton_instance(instance: Instance) -> SkeletonInstance:  # type: ignore[return]
    """Converts a SLEAP Instance to an NWB SkeletonInstance.

    Args:
        instance: A SLEAP Instance.

    Returns:
        An NWB SkeletonInstance.
    """
    skeleton = slp_skeleton_to_nwb(instance.skeleton)
    points_list = list(instance.points.values())
    node_locs = [[point.x, point.y] for point in points_list]
    np_node_locations = np.array(node_locs)
    return SkeletonInstance(
        name=f"skeleton_instance_{id(instance)}",
        id=id(instance),
        node_locations=np_node_locations,
        node_visibility=[point.visible for point in instance.points.values()],
        skeleton=skeleton,
    )


def videos_to_source_videos(videos: List[Video]) -> SourceVideos:  # type: ignore[return]
    """Converts a list of SLEAP Videos to NWB SourceVideos.

    Args:
        videos: A list of SLEAP Videos.

    Returns:
        An NWB SourceVideos object.
    """
    source_videos = []
    for video in videos:
        image_series = ImageSeries(
            name=f"video_{id(video)}",
            description="N/A",
            unit="NA",
            format="external",
            external_file=[video.filename],
            dimension=[video.backend.img_shape[0], video.backend.img_shape[1]],
            starting_frame=[0],
            rate=30.0,  # TODO - change to `video.backend.fps` when available
        )
        source_videos.append(image_series)
    return SourceVideos(image_series=source_videos)


def sleap_pkg_to_nwb(filename: str, **kwargs) -> NWBFile:
    """Write a SLEAP package to an NWB file.

    Args:
        filename: The path to the SLEAP package.

    Returns:
        An NWBFile object.
    """
    labels = load_slp(filename)

    save_path = Path(filename.replace(".slp", ".nwb"))
    save_path.mkdir(parents=True, exist_ok=True)
    img_paths = []
    for i, labeled_frame in enumerate(labels.labeled_frames):
        img_path = save_path / f"frame_{i}.png"
        if labeled_frame.image.ndim == 3 and labeled_frame.image.shape[-1] == 1:
            imwrite(img_path, labeled_frame.image[:, :, 0])
        else:
            imwrite(img_path, labeled_frame.image)
        img_paths.append(img_path)

    # then use img_paths when saving the NWB TrainingFrames with references
    # to the appropriate image files
    save_nwb(labels, save_path, img_paths=img_paths, **kwargs)
    raise NotImplementedError("This function is not yet implemented.")


def get_timestamps(series: PoseEstimationSeries) -> np.ndarray:
    """Return a vector of timestamps for a `PoseEstimationSeries`."""
    if series.timestamps is not None:
        return np.asarray(series.timestamps)
    else:
        return np.arange(series.data.shape[0]) * series.rate + series.starting_time


def read_nwb(path: str) -> Labels:
    """Read an NWB formatted file to a SLEAP `Labels` object.

    Args:
        path: Path to an NWB file (`.nwb`).

    Returns:
        A `Labels` object.
    """
    with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        nwb_file = read_nwbfile.processing

        # Get list of videos
        video_keys: List[str] = [key for key in nwb_file.keys() if "SLEAP_VIDEO" in key]
        video_tracks = dict()

        # Get track keys
        test_processing_module: ProcessingModule = nwb_file[video_keys[0]]
        track_keys: List[str] = list(test_processing_module.fields["data_interfaces"])

        # Get track
        test_pose_estimation: PoseEstimation = test_processing_module[track_keys[0]]
        node_names = test_pose_estimation.nodes[:]
        edge_inds = test_pose_estimation.edges[:]

        for processing_module in nwb_file.values():
            # Get track keys
            _track_keys: List[str] = list(processing_module.fields["data_interfaces"])
            is_tracked: bool = re.sub("[0-9]+", "", _track_keys[0]) == "track"

            # Figure out the max number of frames and the canonical timestamps
            timestamps = np.empty(())
            for track_key in _track_keys:
                for node_name in node_names:
                    pose_estimation_series = processing_module[track_key][node_name]
                    timestamps = np.union1d(
                        timestamps, get_timestamps(pose_estimation_series)
                    )
            timestamps = np.sort(timestamps)

            # Recreate Labels numpy (same as output of Labels.numpy())
            n_tracks = len(_track_keys)
            n_frames = len(timestamps)
            n_nodes = len(node_names)
            tracks_numpy = np.full((n_frames, n_tracks, n_nodes, 2), np.nan, np.float32)
            confidence = np.full((n_frames, n_tracks, n_nodes), np.nan, np.float32)
            for track_idx, track_key in enumerate(_track_keys):
                pose_estimation = processing_module[track_key]

                for node_idx, node_name in enumerate(node_names):
                    pose_estimation_series = pose_estimation[node_name]
                    frame_inds = np.searchsorted(
                        timestamps, get_timestamps(pose_estimation_series)
                    )
                    tracks_numpy[frame_inds, track_idx, node_idx, :] = (
                        pose_estimation_series.data[:]
                    )
                    confidence[frame_inds, track_idx, node_idx] = (
                        pose_estimation_series.confidence[:]
                    )

            video_tracks[Path(pose_estimation.original_videos[0]).as_posix()] = (
                tracks_numpy,
                confidence,
                is_tracked,
            )

    # Create skeleton
    skeleton = SLEAPSkeleton(
        nodes=node_names,
        edges=edge_inds,
    )

    # Add instances to labeled frames
    lfs = []
    for video_fn, (tracks_numpy, confidence, is_tracked) in video_tracks.items():
        video = Video(filename=video_fn)
        n_frames, n_tracks, n_nodes, _ = tracks_numpy.shape
        tracks = [Track(name=f"track{track_idx}") for track_idx in range(n_tracks)]
        for frame_idx, (frame_pts, frame_confs) in enumerate(
            zip(tracks_numpy, confidence)
        ):
            insts: List[Union[Instance, PredictedInstance]] = []
            for track, (inst_pts, inst_confs) in zip(
                tracks, zip(frame_pts, frame_confs)
            ):
                if np.isnan(inst_pts).all():
                    continue
                insts.append(
                    Instance.from_numpy(
                        points=inst_pts,  # (n_nodes, 2)
                        point_scores=inst_confs,  # (n_nodes,)
                        instance_score=inst_confs.mean(),  # ()
                        skeleton=skeleton,
                        track=track if is_tracked else None,
                    )
                )
            if len(insts) > 0:
                lfs.append(
                    LabeledFrame(video=video, frame_idx=frame_idx, instances=insts)
                )
    labels = Labels(lfs)
    labels.provenance["filename"] = path
    return labels


def write_nwb(
    labels: Labels,
    nwbfile_path: str,
    nwb_file_kwargs: Optional[dict] = None,
    pose_estimation_metadata: Optional[dict] = None,
    as_training: Optional[bool] = None,
):
    """Write labels to an nwb file and save it to the nwbfile_path given.

    Args:
        labels: A general `Labels` object.
        nwbfile_path: The path where the nwb file is to be written.
        nwb_file_kwargs: A dict containing metadata to the nwbfile. Example:
            nwb_file_kwargs = {
                'session_description: 'your_session_description',
                'identifier': 'your session_identifier',
            }
            For a full list of possible values see:
            https://pynwb.readthedocs.io/en/stable/pynwb.file.html#pynwb.file.NWBFile

            Defaults to None and default values are used to generate the nwb file.

        pose_estimation_metadata: This argument has a dual purpose:

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
    if as_training:
        nwbfile = append_nwb_training(labels, nwbfile, pose_estimation_metadata)
    else:
        nwbfile = append_nwb_data(labels, nwbfile, pose_estimation_metadata)

    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        # io.nwb_version = (2.0, "2.0.0") #TODO figure out how to set version
        io.write(nwbfile)


def append_nwb_data(
    labels: Labels, nwbfile: NWBFile, pose_estimation_metadata: Optional[dict] = None
) -> NWBFile:
    """Append data from a Labels object to an in-memory nwb file.

    Args:
        labels: A general labels object
        nwbfile: And in-memory nwbfile where the data is to be appended.
        pose_estimation_metadata: This argument has a dual purpose:

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
        An in-memory nwbfile with the data from the labels object appended.
    """
    pose_estimation_metadata = pose_estimation_metadata or dict()

    # Extract default metadata
    provenance = labels.provenance
    default_metadata = dict(scorer=str(provenance))
    sleap_version = provenance.get("sleap_version", None)
    default_metadata["source_software_version"] = sleap_version

    labels_data_df = convert_predictions_to_dataframe(labels)

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

        for track_name in name_of_tracks_in_video:
            pose_estimation_container = build_pose_estimation_container_for_track(
                labels_data_df,
                labels,
                track_name,
                video,
                default_metadata,
            )
            nwb_processing_module.add(pose_estimation_container)

    return nwbfile


def append_nwb_training(
    labels: Labels, nwbfile: NWBFile, pose_estimation_metadata: Optional[dict] = None
) -> NWBFile:
    """Append training data from a Labels object to an in-memory NWB file.

    Args:
        labels: A general labels object.
        nwbfile: An in-memory NWB file.
        pose_estimation_metadata: Metadata for pose estimation.

    Returns:
        An in-memory NWB file with the PoseTraining data appended.
    """
    pose_estimation_metadata = pose_estimation_metadata or dict()
    provenance = labels.provenance
    default_metadata = dict(scorer=str(provenance))
    sleap_version = provenance.get("sleap_version", None)
    default_metadata["source_software_version"] = sleap_version

    for i, video in enumerate(labels.videos):
        video_path = Path(video.filename)
        processing_module_name = f"SLEAP_VIDEO_{i:03}_{video_path.stem}"
        nwb_processing_module = get_processing_module_for_video(
            processing_module_name, nwbfile
        )
        default_metadata["original_videos"] = [f"{video.filename}"]
        default_metadata["labeled_videos"] = [f"{video.filename}"]
        default_metadata.update(pose_estimation_metadata)

    subject = Subject(subject_id="No specified id", species="No specified species")
    nwbfile.subject = subject
    pose_training = labels_to_pose_training(labels)

    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="Behavioral data",
    )
    behavior_pm.add(pose_training)

    skeletons_list = [slp_skeleton_to_nwb(skeleton) for skeleton in labels.skeletons]
    skeletons = Skeletons(skeletons=skeletons_list)
    behavior_pm.add(skeletons)

    camera = nwbfile.create_device(
        name="camera",
        description="Camera used to record the video",
        manufacturer="No specified manufacturer",
    )

    data = np.random.rand(100, 2)
    timestamps = np.linspace(0, 10, num=100)
    confidence = np.random.rand(100)
    reference_frame = (
        "The coordinates are in (x, y) relative to the top-left of the image."
    )
    confidence_definition = "Softmax output of the deep neural network"
    pose_estimation_series_list = []
    for node in skeletons_list[0].nodes:
        pose_estimation_series = PoseEstimationSeries(
            name=node,
            description=f"Sequential trajectory of {node}.",
            data=data,
            unit="pixels",
            reference_frame=reference_frame,
            timestamps=timestamps,
            confidence=confidence,
            confidence_definition=confidence_definition,
        )
        pose_estimation_series_list.append(pose_estimation_series)

    dimensions = np.array(
        [[labels.videos[0].backend.shape[1], labels.videos[0].backend.shape[2]]]
    )
    pose_estimation = PoseEstimation(
        name="pose_estimation",
        pose_estimation_series=pose_estimation_series_list,
        description="Estimated positions of the nodes in the video",
        original_videos=[video.filename for video in labels.videos],
        labeled_videos=[video.filename for video in labels.videos],
        dimensions=dimensions,
        devices=[camera],
        scorer="No specified scorer",
        source_software="SLEAP",
        source_software_version=sleap_version,
        skeleton=skeletons_list[0],
    )
    behavior_pm.add(pose_estimation)
    return nwbfile


def append_nwb(
    labels: Labels,
    filename: str,
    pose_estimation_metadata: Optional[dict] = None,
    as_training: Optional[bool] = None,
):
    """Append a SLEAP `Labels` object to an existing NWB data file.

    Args:
        labels: A general `Labels` object.
        filename: The path to the NWB file.
        pose_estimation_metadata: Metadata for pose estimation. See `append_nwb_data`
            for details.
        as_training: If `True`, append the data as training data.

    See also: append_nwb_data
    """
    if as_training:
        with NWBHDF5IO(filename, mode="a", load_namespaces=True) as io:
            nwb_file = io.read()
            nwb_file = append_nwb_training(
                labels, nwb_file, pose_estimation_metadata=pose_estimation_metadata
            )
            io.write(nwb_file)
    else:
        with NWBHDF5IO(filename, mode="a", load_namespaces=True) as io:
            nwb_file = io.read()
            nwb_file = append_nwb_data(
                labels, nwb_file, pose_estimation_metadata=pose_estimation_metadata
            )
            io.write(nwb_file)


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
    """Create a PoseEstimation container for a track.

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
    track_data_df: pd.DataFrame, timestamps: ArrayLike  # type: ignore[return]
) -> List[PoseEstimationSeries]:
    """Build a list of PoseEstimationSeries from tracks.

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
        timestamps_for_data = timestamps[frames]  # type: ignore[index]
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
