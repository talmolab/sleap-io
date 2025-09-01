"""Functions to write and read user annotated frames to/from NWB format."""

import datetime
import json
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = np.ndarray
from ndx_pose import (
    PoseTraining,
    Skeleton,
    SkeletonInstance,
    SkeletonInstances,
    Skeletons,
    SourceVideos,
    TrainingFrame,
    TrainingFrames,
)
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject
from pynwb.image import ImageSeries

from sleap_io.io.video_writing import MJPEGFrameWriter
from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton as SleapSkeleton
from sleap_io.model.video import Video


def create_skeletons(
    labels: Labels,
) -> Tuple[Skeletons, Dict[str, List[int]], Dict[str, Skeleton]]:
    """Create NWB skeleton containers and per-video frame-index mappings.

    Iterates through the provided SLEAP `Labels` objects, extracts every
    unique `Skeleton`, and builds a single NWB `Skeletons` container. While
    doing so, it records all frame indices that contain at least one
    `Instance` for each video.

    Args:
        labels: Sequence of SLEAP `Labels` objects. Each must contain at least
            one `Video` and `Instance` collection.

    Returns:
        A tuple containing:
            - Skeletons: An NWB `Skeletons` container comprising one `Skeleton`
                object for every unique skeleton encountered across all input
                labels.
            - Dict[str, List[int]]: Maps each video file path to the sorted,
                unique frame indices that contain at least one `Instance` in that
                video.
            - Dict[str, Skeleton]: Maps each skeleton name to the corresponding
                NWB `Skeleton` object included in the returned `Skeletons`
                container.
    """
    unique_skeletons = {}  # create a new NWB skeleton obj per unique skeleton in SLEAP
    frame_indices = {}

    for j, label in enumerate(labels):
        video = labels[j].video.filename

        if video not in frame_indices:
            frame_indices[video] = []

        frame_indices[video].append(labels[j].frame_idx)

        for inst in label.instances:
            skel = inst.skeleton
            skel_name = inst.skeleton.name  # assumes skels are named uniquely

            if skel_name not in unique_skeletons:
                node_names = [node.name for node in skel.nodes]
                node_name_to_index = {name: i for i, name in enumerate(node_names)}

                edge_index_pairs = []
                skipped_edges = []
                for src_node, dst_node in skel.edges:
                    try:
                        src_idx = node_name_to_index[src_node.name]
                        dst_idx = node_name_to_index[dst_node.name]
                        edge_index_pairs.append([src_idx, dst_idx])
                    except KeyError:
                        skipped_edges.append((src_node.name, dst_node.name))
                        continue  # skip edge if node name is missing

                if skipped_edges:
                    warnings.warn(
                        f"Skipped {len(skipped_edges)} edges in skeleton '{skel_name}' "
                        f"due to missing nodes: {skipped_edges}"
                    )

                edge_array = np.array(edge_index_pairs, dtype="uint8")

                unique_skeletons[skel_name] = Skeleton(
                    name=skel_name, nodes=node_names, edges=edge_array
                )

    skeletons = Skeletons(skeletons=list(unique_skeletons.values()))

    for key in frame_indices:
        frame_indices[key] = list(sorted(set(frame_indices[key])))

    return skeletons, frame_indices, unique_skeletons


def get_frames_from_slp(
    labels: Labels,
    mjpeg_frame_duration: float = 30.0,
) -> Tuple[List[np.ndarray], List[float], Dict[int, List[Tuple[int, str]]]]:
    """Extract frames from slp and get mapping to original video.

    Extract all unique video frames referenced and build
    lookup tables for MJPEG assembly and cross-video frame alignment.

    Args:
        labels: A single SLEAP `Labels` object containing `LabeledFrame`
            records with associated `Video` backends.
        mjpeg_frame_duration: Duration (in milliseconds) to associate with
            each frame when generating an MJPEG stream. Defaults to 30.0.

    Returns:
        A tuple containing:
            - frames: List of frame arrays extracted from videos.
            - durations: List of per-frame durations in seconds.
            - frame_map: Maps each `frame_idx` (key) to a list of
                `(global_idx, video_name)` tuples. `global_idx` is the zero-based
                index of the frame within the frames list; `video_name` is the stem
                of the source video file. A given `frame_idx` may map to multiple
                videos if frames share an index across videos.
    """
    frames = []
    durations = []
    frame_map = {}

    written = set()
    global_idx = 0

    video_idx = {}
    track = 0

    for lf in labels.labeled_frames:
        frame_idx = lf.frame_idx
        video_name = Path(lf.video.filename).stem

        if video_name not in video_idx:  # track how many videos there are
            track += 1
            video_idx[video_name] = track

        v_idx = video_idx[video_name]

        cache_key = (v_idx, frame_idx)

        if cache_key in written:
            continue

        frame = lf.video.backend.get_frame(frame_idx)
        frames.append(frame)
        durations.append(mjpeg_frame_duration / 1000.0)  # Convert ms to seconds

        if frame_idx not in frame_map:
            frame_map[frame_idx] = [(global_idx, video_name)]
        else:
            frame_map[frame_idx].append((global_idx, video_name))

        global_idx += 1

        written.add(cache_key)

    return frames, durations, frame_map


def make_mjpeg(
    frames: List[np.ndarray],
    durations: List[float],
    frame_map: Dict[int, List[Tuple[int, str]]],
    output_dir: Optional[Path] = None,
) -> str:
    """Encode an MJPEG (Motion-JPEG) video from extracted frames.

    Writes frames directly to an MJPEG video file using MJPEGFrameWriter
    and serializes `frame_map` to frame_map.json.

    Creates/overwrites the following files:
        - frame_map.json: JSON dump of the provided `frame_map` (in output_dir).
        - annotated_frames.avi: Resulting MJPEG container (in output_dir).

    Args:
        frames: List of frame arrays to write to the MJPEG.
        durations: List of per-frame durations in seconds for VFR encoding.
        frame_map: Mapping of `frame_idx` to list of `(global_idx, video_name)` tuples
            used for downstream alignment; dumped verbatim to JSON.
        output_dir: Directory to write output files. If None, uses current directory.

    Returns:
        Absolute or relative path to the generated MJPEG file
        (annotated_frames.avi).
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    frame_map_json_path = output_dir / "frame_map.json"
    output_mjpeg = output_dir / "annotated_frames.avi"

    # Convert tuples to lists for JSON serialization and dump frame map
    json_frame_map = {k: [list(t) for t in v] for k, v in frame_map.items()}
    with open(frame_map_json_path, "w") as f_map:
        json.dump(json_frame_map, f_map, indent=2)

    # Write MJPEG using MJPEGFrameWriter
    with MJPEGFrameWriter(
        filename=output_mjpeg,
        fps=30.0,  # Nominal FPS for the container
        quality=2,  # High quality (same as original -q:v 2)
        frame_durations=durations,  # Per-frame durations for VFR
    ) as writer:
        for frame in frames:
            writer.write_frame(frame)

    return str(output_mjpeg)


def create_source_videos(
    frame_indices: Dict[str, List[int]],
    output_mjpeg: str,
    mjpeg_frame_rate: float = 30.0,
    include_devices: bool = False,
    nwbfile: Optional[NWBFile] = None,
) -> Tuple[SourceVideos, ImageSeries, Dict[str, Tuple[int, int]]]:
    """Assemble NWB `SourceVideos` for original and annotated footage.

    Opens every path in `frame_indices` via OpenCV to obtain the original FPS.

    Args:
        frame_indices: Mapping from absolute or relative video file paths to the
            list of frame indices that were annotated in each file.
        output_mjpeg: Path to the MJPEG file containing only annotated frames
            (output from `make_mjpeg`).
        mjpeg_frame_rate: Nominal frame rate assigned to the MJPEG `ImageSeries`.
            Defaults to 30.0 Hz.
        include_devices: If True and nwbfile is provided, create camera devices.
        nwbfile: Optional NWBFile to add camera devices to.

    Returns:
        A tuple containing:
            - SourceVideos: NWB container holding one `ImageSeries` per original
                video plus an additional series for the annotated MJPEG.
            - ImageSeries: The `ImageSeries` object corresponding to the
                annotated MJPEG; also present in the returned `SourceVideos`
                container.
            - Dict[str, Tuple[int, int]]: Mapping of video names to (width, height)
                dimensions.

    Raises:
        RuntimeError: If a video cannot be opened or its FPS is unavailable.
    """
    original_videos = []
    video_dimensions = {}

    for i, video in enumerate(frame_indices):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            raise RuntimeError(f"Cannot get fps from: {video}")

        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        video_name = Path(video).name
        video_dimensions[video_name] = (width, height)

        # Create camera device if requested
        device = None
        if include_devices and nwbfile is not None:
            device = nwbfile.create_device(
                name=f"camera_{i}",
                description=f"Camera for {video_name}",
                manufacturer="unknown",
            )

        series_kwargs = {
            "name": f"original_video_{i}",  # must have unique names
            "description": "Full original video",
            "format": "external",
            "external_file": [video_name],
            "starting_frame": [0],
            "rate": orig_fps,
            "dimension": [width, height],
        }
        if device is not None:
            series_kwargs["device"] = device

        original_videos.append(ImageSeries(**series_kwargs))

    annotations_mjpeg = ImageSeries(
        name="annotated_frames",
        description="Only frames that were annotated with node points and locations",
        format="external",
        external_file=[output_mjpeg],
        starting_frame=[0],
        rate=mjpeg_frame_rate,  # mjpeg hard coded to have a 30 fps
    )

    all_videos = original_videos + [annotations_mjpeg]

    source_videos = SourceVideos(image_series=all_videos)

    return source_videos, annotations_mjpeg, video_dimensions


def create_training_frames(
    labels: Labels,
    unique_skeletons: Dict[str, Skeleton],
    annotations_mjpeg: ImageSeries,
    frame_map: Dict[int, List[Tuple[int, str]]],
    identity: bool = False,
    annotator: str = "SLEAP",
) -> TrainingFrames:
    """Convert SLEAP annotations into NWB `TrainingFrames`.

    Args:
        labels: A SLEAP `Labels` object whose `LabeledFrame` records contain the
            annotations to export.
        unique_skeletons: Mapping of skeleton name to NWB `Skeleton` produced by
            `create_skeletons`.
        annotations_mjpeg: The `ImageSeries` object pointing to the MJPEG of
            annotated frames (output from `create_source_videos`).
        frame_map: Mapping from original `frame_idx` to list of
            `(global_idx, video_name)` tuples created by `get_frames_from_slp`.
        identity: If True, append each instance's track name to its skeleton
            name to guarantee uniqueness when identity tracking is present.
            Defaults to False.
        annotator: Name of the annotator or annotation software. Defaults to "SLEAP".

    Returns:
        An NWB `TrainingFrames` container, where every `TrainingFrame` holds
        one `SkeletonInstances` composed of the per-frame
        `SkeletonInstance`(s) and references the MJPEG via
        `source_video`/`source_video_frame_index`.

    Raises:
        KeyError: If an instance's skeleton name is absent from `unique_skeletons`.
    """
    training_frames_list = []

    unique_ids = 0
    for lf_idx, lf in enumerate(labels.labeled_frames):
        skeleton_instances_list = []

        frame_idx = lf.frame_idx

        for j in range(len(lf.instances)):
            val = lf.instances[j]

            # Use skeleton name and instance index for clarity
            skeleton_base_name = val.skeleton.name

            if identity and val.track is not None:
                instance_name = f"{skeleton_base_name}.{val.track.name}.instance_{j}"
            else:
                instance_name = f"{skeleton_base_name}.instance_{j}"

            node_locations_sk1 = np.array([[pt[0][0], pt[0][1]] for pt in val.points])

            instance_sk1 = SkeletonInstance(
                name=instance_name,
                id=np.uint64(unique_ids),
                node_locations=node_locations_sk1,
                node_visibility=[pt[1] for pt in val.points],
                skeleton=unique_skeletons[val.skeleton.name],
            )
            skeleton_instances_list.append(instance_sk1)
            unique_ids += 1

        # store the skeleton instances in a SkeletonInstances object
        skeleton_instances = SkeletonInstances(
            skeleton_instances=skeleton_instances_list
        )

        # Get the global frame index from the frame map
        mapped = frame_map.get(frame_idx, [])

        # Find the correct mapping for this video
        video_name = Path(lf.video.filename).stem
        global_frame_idx = None

        for global_idx, vid_name in mapped:
            if vid_name == video_name:
                global_frame_idx = global_idx
                break

        if global_frame_idx is None:
            # If we can't find the mapping, use the first one as fallback
            if mapped:
                global_frame_idx = mapped[0][0]
            else:
                warnings.warn(f"No frame mapping found for frame {frame_idx}")
                global_frame_idx = frame_idx

        training_frame = TrainingFrame(
            name=f"frame_{lf_idx}",  # must be unique
            annotator=annotator,
            skeleton_instances=skeleton_instances,
            source_video=annotations_mjpeg,
            source_video_frame_index=np.uint64(global_frame_idx),
        )
        training_frames_list.append(training_frame)

    training_frames = TrainingFrames(training_frames=training_frames_list)

    return training_frames


def write_annotations_nwb(
    labels: Labels,
    nwbfile_path: str,
    nwb_file_kwargs: Optional[dict] = None,
    nwb_subject_kwargs: Optional[dict] = None,
    output_dir: Optional[str] = None,
    include_devices: bool = False,
    annotator: str = "SLEAP",
) -> None:
    """Export SLEAP pose data and metadata to an NWB file.

    Generates annotated_frames.avi containing all annotated frames and dumps
    frame_map.json to the specified output directory. Writes the assembled NWB file to
    nwbfile_path using pynwb.NWBHDF5IO.

    Args:
        labels: A SLEAP `Labels` object whose `LabeledFrame` entries contain the
            pose annotations to be written.
        nwbfile_path: Destination path for the NWB file. Will be overwritten if
            it already exists.
        nwb_file_kwargs: Optional keyword arguments forwarded directly to
            pynwb.NWBFile. Missing required fields are auto-filled:
            session_description (default "Processed SLEAP pose data"),
            session_start_time (UTC datetime.now), and identifier (uuid.uuid1()).
            Defaults to None.
        nwb_subject_kwargs: Optional keyword arguments forwarded to
            pynwb.file.Subject. Missing DANDI-required fields are auto-filled:
            subject_id ("subject1"), species ("Mus musculus"), sex ("F"),
            and age ("P10W/P12W"). Defaults to None.
        output_dir: Directory for intermediate output files (frame_map.json,
            annotated_frames.avi). If None, uses current directory.
        include_devices: If True, create camera device metadata in NWB file.
        annotator: Name of annotator or annotation software. Defaults to "SLEAP".

    Raises:
        RuntimeError: If an original video cannot be opened or its FPS is
            unobtainable.
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

    nwb_subject_kwargs = nwb_subject_kwargs or dict()

    # Add required subject metadata for DANDI if not present
    subject_id = nwb_subject_kwargs.get("subject_id", "subject1")
    species = nwb_subject_kwargs.get("species", "Mus musculus")

    sex = nwb_subject_kwargs.get("sex", "F")

    age = nwb_subject_kwargs.get("age", "P10W/P12W")

    nwb_subject_kwargs.update(
        subject_id=subject_id,
        species=species,
        sex=sex,
        age=age,
    )

    subject = Subject(**nwb_subject_kwargs)
    nwbfile.subject = subject

    skeletons, frame_indices, unique_skeletons = create_skeletons(labels)

    frames, durations, frame_map = get_frames_from_slp(labels)
    output_mjpeg = make_mjpeg(frames, durations, frame_map, output_dir=output_dir)

    source_videos, annotations_mjpeg, video_dimensions = create_source_videos(
        frame_indices, output_mjpeg, include_devices=include_devices, nwbfile=nwbfile
    )
    training_frames = create_training_frames(
        labels, unique_skeletons, annotations_mjpeg, frame_map, annotator=annotator
    )

    pose_training = PoseTraining(
        training_frames=training_frames,
        source_videos=source_videos,
    )

    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="processed behavioral data",
    )
    behavior_pm.add(skeletons)
    behavior_pm.add(pose_training)

    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(nwbfile)


def _extract_skeletons_from_nwb(
    skeletons_container: Skeletons,
) -> Dict[str, SleapSkeleton]:
    """Extract SLEAP Skeleton objects from NWB Skeletons container.

    Args:
        skeletons_container: NWB Skeletons container with skeleton definitions.

    Returns:
        Dictionary mapping skeleton names to SLEAP Skeleton objects.
    """
    sleap_skeletons = {}

    for skeleton_name in skeletons_container.skeletons:
        nwb_skeleton = skeletons_container.skeletons[skeleton_name]

        # Convert nodes (list of strings) and edges (numpy array) to SLEAP format
        sleap_skeleton = SleapSkeleton(
            name=skeleton_name,
            nodes=list(nwb_skeleton.nodes),
            edges=nwb_skeleton.edges.tolist() if len(nwb_skeleton.edges) > 0 else [],
        )
        sleap_skeletons[skeleton_name] = sleap_skeleton

    return sleap_skeletons


def _load_frame_map(
    frame_map_path: Union[str, Path],
) -> Dict[int, List[Tuple[int, str]]]:
    """Load frame mapping from JSON file.

    Args:
        frame_map_path: Path to frame_map.json file.

    Returns:
        Dictionary mapping MJPEG frame indices to (original_idx, video_name) tuples.

    Raises:
        FileNotFoundError: If frame_map.json doesn't exist.
        json.JSONDecodeError: If JSON file is malformed.
    """
    with open(frame_map_path, "r") as f:
        json_data = json.load(f)

    # Convert string keys back to integers and lists back to tuples
    frame_map = {}
    for key, value in json_data.items():
        frame_map[int(key)] = [tuple(item) for item in value]

    return frame_map


def _invert_frame_map(
    frame_map: Dict[int, List[Tuple[int, str]]],
) -> Dict[Tuple[str, int], int]:
    """Invert frame mapping for efficient lookup.

    Args:
        frame_map: Original frame mapping from frame_map.json.

    Returns:
        Dictionary mapping (video_name, mjpeg_idx) to original frame index.
    """
    inverted = {}
    for orig_idx, mappings in frame_map.items():
        for mjpeg_idx, video_name in mappings:
            inverted[(video_name, mjpeg_idx)] = orig_idx
    return inverted


def _reconstruct_instances_from_training(
    training_frame: TrainingFrame,
    sleap_skeletons: Dict[str, SleapSkeleton],
    tracks: Optional[Dict[str, Track]] = None,
) -> List[Instance]:
    """Reconstruct SLEAP instances from NWB TrainingFrame.

    Args:
        training_frame: NWB TrainingFrame containing SkeletonInstances.
        sleap_skeletons: Dictionary mapping skeleton names to SLEAP Skeletons.
        tracks: Optional dictionary mapping track names to Track objects.

    Returns:
        List of SLEAP Instance objects.
    """
    instances = []

    skeleton_instances = training_frame.skeleton_instances
    for skeleton_instance_name in skeleton_instances.skeleton_instances:
        skeleton_instance = skeleton_instances.skeleton_instances[
            skeleton_instance_name
        ]

        # Extract skeleton name from instance name
        # Instance names are like "skeleton_name.instance_0" or
        # "skeleton_name.track_name.instance_0"
        instance_name = skeleton_instance.name

        # Parse the instance name using '.' as delimiter
        parts = instance_name.split(".")

        # Find the skeleton by matching against known skeleton names
        skeleton_obj = None
        track = None

        # First, try to find skeleton from the NWB skeleton reference
        nwb_skeleton = skeleton_instance.skeleton
        if nwb_skeleton and nwb_skeleton.name in sleap_skeletons:
            skeleton_obj = sleap_skeletons[nwb_skeleton.name]

        # If not found, try to parse from instance name
        if skeleton_obj is None:
            # Check if the first part matches a known skeleton
            if parts[0] in sleap_skeletons:
                skeleton_obj = sleap_skeletons[parts[0]]
            else:
                # Try to find skeleton by prefix match (for backward compatibility)
                for skel_name in sleap_skeletons:
                    if instance_name.startswith(skel_name):
                        skeleton_obj = sleap_skeletons[skel_name]
                        # Re-parse with the matched skeleton name
                        # Handle old format "skeleton_name_instance_0" or
                        # "skeleton_name_track_instance_0"
                        if "_instance_" in instance_name:
                            old_parts = instance_name.split("_instance_")[0].split("_")
                            skel_parts = skel_name.split("_")
                            if len(old_parts) > len(skel_parts):
                                # Extract track from old format
                                track_parts = old_parts[len(skel_parts) :]
                                track_name = "_".join(track_parts)
                                if track_name and track_name not in tracks:
                                    tracks[track_name] = Track(name=track_name)
                                track = tracks.get(track_name)
                        break

        if skeleton_obj is None:
            warnings.warn(f"Could not find skeleton for instance {instance_name}")
            continue

        # Extract track information if present (for new format)
        if track is None and len(parts) == 3 and parts[1] != "instance":
            # Format: skeleton_name.track_name.instance_X
            track_name = parts[1]
            if track_name not in tracks:
                tracks[track_name] = Track(name=track_name)
            track = tracks[track_name]
        elif track is None and tracks is not None and len(parts) > 2:
            # Check for any middle parts that could be track names
            for i in range(1, len(parts) - 1):
                if not parts[i].startswith("instance"):
                    track_name = parts[i]
                    if track_name not in tracks:
                        tracks[track_name] = Track(name=track_name)
                    track = tracks[track_name]
                    break

        # Create SLEAP Instance from node locations and visibility
        points = np.column_stack(
            [skeleton_instance.node_locations, skeleton_instance.node_visibility]
        )

        instance = Instance.from_numpy(
            points_data=points, skeleton=skeleton_obj, track=track
        )
        instances.append(instance)

    return instances


def read_nwb_annotations(
    nwb_path: str,
    frame_map_path: Optional[str] = None,
    load_source_videos: bool = False,
) -> Labels:
    """Read NWB file with PoseTraining annotations to SLEAP Labels.

    Args:
        nwb_path: Path to NWB file containing PoseTraining data.
        frame_map_path: Optional path to frame_map.json. If None, will look
            for it in the same directory as the NWB file.
        load_source_videos: If True, attempt to load source videos from
            paths in the NWB file. If False, create Video placeholders.

    Returns:
        Labels object containing the reconstructed annotations.

    Raises:
        ValueError: If NWB file doesn't contain PoseTraining data.
        FileNotFoundError: If frame_map.json is needed but not found.
    """
    # Determine frame_map.json path
    if frame_map_path is None:
        nwb_dir = Path(nwb_path).parent
        frame_map_path = nwb_dir / "frame_map.json"
    else:
        frame_map_path = Path(frame_map_path)

    # Try to load frame map if it exists
    frame_map = None
    inverted_map = None
    if frame_map_path.exists():
        try:
            frame_map = _load_frame_map(frame_map_path)
            inverted_map = _invert_frame_map(frame_map)
        except (json.JSONDecodeError, KeyError) as e:
            warnings.warn(f"Could not load frame_map.json: {e}")

    # Open NWB file and read PoseTraining data
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()

        # Find behavior processing module
        if "behavior" not in nwbfile.processing:
            raise ValueError("NWB file does not contain 'behavior' processing module")

        behavior_pm = nwbfile.processing["behavior"]

        # Extract Skeletons
        if "Skeletons" not in behavior_pm.data_interfaces:
            raise ValueError("NWB file does not contain Skeletons data")

        skeletons_container = behavior_pm.data_interfaces["Skeletons"]
        sleap_skeletons = _extract_skeletons_from_nwb(skeletons_container)

        # Find PoseTraining container
        pose_training = None
        for data_interface_name in behavior_pm.data_interfaces:
            data_interface = behavior_pm.data_interfaces[data_interface_name]
            if isinstance(data_interface, PoseTraining):
                pose_training = data_interface
                break

        if pose_training is None:
            raise ValueError("NWB file does not contain PoseTraining data")

        # Extract source video information
        video_map = {}  # Maps video names to Video objects
        mjpeg_video = None

        if pose_training.source_videos:
            for series_name in pose_training.source_videos.image_series:
                image_series = pose_training.source_videos.image_series[series_name]

                if series_name == "annotated_frames":
                    # This is the MJPEG video
                    mjpeg_path = image_series.external_file[0]
                    if load_source_videos:
                        mjpeg_video = Video(filename=mjpeg_path)
                    else:
                        mjpeg_video = Video(filename=mjpeg_path, backend=None)
                else:
                    # Original source video
                    video_path = image_series.external_file[0]
                    video_name = Path(video_path).stem

                    if load_source_videos:
                        video_map[video_name] = Video(filename=video_path)
                    else:
                        video_map[video_name] = Video(filename=video_path, backend=None)

        # Process TrainingFrames
        labeled_frames = []
        tracks = {}  # Track objects by name

        training_frames_container = pose_training.training_frames
        for frame_name in training_frames_container.training_frames:
            training_frame = training_frames_container.training_frames[frame_name]

            # Get MJPEG frame index
            mjpeg_frame_idx = int(training_frame.source_video_frame_index)

            # Reconstruct instances
            instances = _reconstruct_instances_from_training(
                training_frame, sleap_skeletons, tracks
            )

            if not instances:
                continue

            # Determine original video and frame index
            if inverted_map:
                # Try to find in frame map
                found = False
                for video_name in video_map:
                    key = (video_name, mjpeg_frame_idx)
                    if key in inverted_map:
                        orig_frame_idx = inverted_map[key]
                        video = video_map[video_name]
                        found = True
                        break

                if not found:
                    # Fall back to MJPEG video if available
                    if mjpeg_video:
                        video = mjpeg_video
                        orig_frame_idx = mjpeg_frame_idx
                    else:
                        warnings.warn(
                            f"Could not map MJPEG frame {mjpeg_frame_idx} "
                            f"to original video"
                        )
                        continue
            else:
                # No frame map, use MJPEG video or first available video
                if mjpeg_video:
                    video = mjpeg_video
                    orig_frame_idx = mjpeg_frame_idx
                elif video_map:
                    # Use first video as fallback
                    video = next(iter(video_map.values()))
                    orig_frame_idx = mjpeg_frame_idx
                else:
                    # Create placeholder video
                    video = Video(filename="unknown_video.mp4", backend=None)
                    orig_frame_idx = mjpeg_frame_idx

            # Create LabeledFrame
            lf = LabeledFrame(
                video=video, frame_idx=orig_frame_idx, instances=instances
            )
            labeled_frames.append(lf)

    # Create Labels object
    videos = list(video_map.values())
    if mjpeg_video and mjpeg_video not in videos:
        videos.append(mjpeg_video)

    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos if videos else None,
        skeletons=list(sleap_skeletons.values()),
        tracks=list(tracks.values()) if tracks else None,
    )

    labels.provenance["filename"] = nwb_path
    labels.provenance["source_format"] = "nwb_pose_training"

    return labels
