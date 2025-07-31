"""Functions to write user annotated frames to NWB format."""

import datetime
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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

from sleap_io import (
    Labels,
)


def create_skeletons(
        labels: Labels
) -> tuple[Skeletons, Dict[str, List[int]], Dict[str, Skeleton]]:
    """Create NWB skeleton containers and per‑video frame‑index mappings.

    Iterates through the provided SLEAP `Labels` objects, extracts every
    unique `Skeleton`, and builds a single NWB `Skeletons` container. While
    doing so, it records all frame indices that contain at least one
    `Instance` for each video.

    Args:
        labels: Sequence of SLEAP `Labels` objects. Each must contain at least
            one `Video` and `Instance` collection.

    Returns:
        Skeletons: An NWB `Skeletons` container comprising one `Skeleton`
            object for every unique skeleton encountered across *all* input
            labels.
        Dict[str, List[int]]: Maps each video file path to the sorted,
            unique frame indices that contain at least one `Instance` in that
            video.
        Dict[str, Skeleton]: Maps each skeleton name to the corresponding
            NWB `Skeleton` object included in the returned `Skeletons`
            container.
    """
    unique_skeletons = {} # create a new NWB skeleton obj per unique skeleton in SLEAP
    frame_indices = {}

    for j, label in enumerate(labels):
        video = labels[j].video.filename

        if video not in frame_indices:
            frame_indices[video] = []

        frame_indices[video].append(labels[j].frame_idx)

        for inst in label.instances:

            skel = inst.skeleton
            skel_name = inst.skeleton.name # assumes skels are named uniquely

            if skel_name not in unique_skeletons:
                node_names = [node.name for node in skel.nodes]
                node_name_to_index = {name: i for i, name in enumerate(node_names)}

                edge_index_pairs = []
                for src_node, dst_node in skel.edges:
                    try:
                        src_idx = node_name_to_index[src_node.name]
                        dst_idx = node_name_to_index[dst_node.name]
                        edge_index_pairs.append([src_idx, dst_idx])
                    except KeyError:
                        continue  # skip edge if node name is missing

                edge_array = np.array(edge_index_pairs, dtype="uint8")

                unique_skeletons[skel_name] = Skeleton(
                    name=skel_name,
                    nodes=node_names,
                    edges=edge_array
                )

    skeletons = Skeletons(skeletons=list(unique_skeletons.values()))

    for key in frame_indices:
        frame_indices[key] = list(sorted(set(frame_indices[key])))

    return skeletons, frame_indices, unique_skeletons

def get_frames_from_slp(
        labels: Labels,
        mjpeg_frame_duration: float = 30.0,
    ) -> tuple[list[tuple[str, float]], dict[int, list[list[object]]]]:
    """Write individual frames from slp and get mapping to original video.

    Write all unique video frames referenced and build
    lookup tables for MJPEG assembly and cross‑video frame alignment.

    Args:
        labels: A single SLEAP `Labels` object containing `LabeledFrame`
            records with associated `Video` backends.
        mjpeg_frame_duration: Duration (in milliseconds) to associate with
            each frame when generating an MJPEG stream.

    Returns:
        list[tuple[str, float]]: `image_list`. Ordered list whose elements are
            `(filepath, mjpeg_frame_duration)` pairs. `filepath` is the PNG
            written to `frames/`, `mjpeg_frame_duration` is the per‑frame
            delay in milliseconds.
        dict[int, list[list[object]]]: `frame_map`. Maps each `frame_idx`
            (key) to a list of `[global_idx, video_name]` pairs. `global_idx`
            is the zero‑based index of the frame within `image_list`;
            `video_name` is the stem of the source video file. A given
            `frame_idx` may map to multiple videos if frames share an index
            across videos.
    """
    out_dir = Path("frames")
    out_dir.mkdir(exist_ok=True)

    image_list = []
    frame_map = {}

    written = set()
    global_idx = 0

    video_idx = {}
    track = 0

    for lf in labels.labeled_frames:

        frame_idx   = lf.frame_idx
        video_name  = Path(lf.video.filename).stem

        if video_name not in video_idx: #track how many videos there are
            track += 1
            video_idx[video_name] = track

        v_idx = video_idx[video_name]

        cache_key   = (v_idx, frame_idx)

        if cache_key in written:
            continue

        frame = lf.video.backend.get_frame(frame_idx)

        out_path = out_dir / f"v{v_idx}_f{frame_idx}.png"
        filename = str(out_path)
        cv2.imwrite(filename, frame)
        image_list.append((filename, mjpeg_frame_duration))

        if frame_idx not in frame_map:
            frame_map[frame_idx] = [global_idx, video_name]
        elif isinstance(frame_map[frame_idx][0], list):
            frame_map[frame_idx].append([global_idx, video_name])
        else:
            frame_map[frame_idx] = [frame_map[frame_idx], [global_idx, video_name]]

        global_idx += 1

        written.add(cache_key)

    return image_list, frame_map

def make_mjpeg(
    image_list: list[tuple[str, float]],
    frame_map: dict[int, list[list[object]]],
) -> str:
    """Encode an MJPEG (Motion‑JPEG) video from extracted frame images.

    Builds an ffmpeg concat script, writes it to *input.txt*, serialises
    `frame_map` to *frame_map.json*, and invokes ffmpeg to generate
    *annotated_frames.avi*.

    Parameters
    ----------
    image_list :
        List of `(filepath, duration)` tuples. `filepath` is the on‑disk image
        (PNG, JPEG, etc.); `duration` is the per‑frame display time in
        seconds for variable‑frame‑rate (VFR) encoding.
    frame_map :
        Mapping of `frame_idx` → `[global_idx, video_name]` pairs (or list of
        such pairs) used for downstream alignment; dumped verbatim to JSON.

    Returns:
    -------
    str
        Absolute or relative path to the generated MJPEG file
        (*annotated_frames.avi*).

    Side Effects
    ------------
    Creates/overwrites the following files in the current working directory:
      * *input.txt*          – ffmpeg concat descriptor.
      * *frame_map.json*     – JSON dump of the provided `frame_map`.
      * *annotated_frames.avi* – Resulting MJPEG container.

    Raises:
    ------
    subprocess.CalledProcessError
        Propagated if ffmpeg returns a non‑zero exit status.
    """
    input_txt_path = "input.txt"
    frame_map_json_path = "frame_map.json"
    output_mjpeg = "annotated_frames.avi"

    # write input.txt for ffmpeg
    with open(input_txt_path, "w") as f:
        for image_path, duration in image_list:
            f.write(f"file '{image_path}'\n")
            f.write(f"duration {duration:.6f}\n")
        f.write(f"file '{image_list[-1][0]}'\n")  # repeat last frame for mpjeg

    # dump frame map
    with open(frame_map_json_path, "w") as f_map:
        json.dump(frame_map, f_map, indent=2)

    # ffmpeg encode
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", input_txt_path,
        "-fps_mode", "vfr",
        "-c:v", "mjpeg",
        "-q:v", "2",
        "-quality", "90",
        output_mjpeg
    ]

    subprocess.run(cmd, check=True)

    return output_mjpeg

def create_source_videos(
    frame_indices: dict[str, list[int]],
    output_mjpeg: str,
    mjpeg_frame_rate: float = 30.0,
) -> tuple[SourceVideos, ImageSeries]:
    """Assemble NWB `SourceVideos` for original and annotated footage.

    Parameters
    ----------
    frame_indices :
        Mapping from absolute or relative video file paths to the list of
        frame indices that were annotated in each file.
    output_mjpeg :
        Path to the MJPEG file containing only annotated frames (output from
        :pyfunc:`make_mjpeg`).
    mjpeg_frame_rate :
        Nominal frame rate assigned to the MJPEG `ImageSeries`. Defaults to
        ``30.0`` Hz.

    Returns:
    -------
    SourceVideos
        NWB container holding one `ImageSeries` per original video plus an
        additional series for the annotated MJPEG.
    ImageSeries
        The `ImageSeries` object corresponding to the annotated MJPEG; also
        present in the returned `SourceVideos` container.

    Side Effects
    ------------
    Opens every path in `frame_indices` via OpenCV to obtain the original FPS.
    Raises :class:`RuntimeError` if a video cannot be opened or its FPS is
    unavailable.
    """
    original_videos = []

    for i, video in enumerate(frame_indices):

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            raise RuntimeError(f"Cannot get fps from: {video}")

        video_name = os.path.basename(video)

        original_videos.append(ImageSeries(
            name=f"original_video_{i}", # must have unique names
            description="Full original video",
            format="external",
            external_file=[video_name],
            starting_frame=[0],
            rate=orig_fps,
        ))

    annotations_mjpeg = ImageSeries(
        name="annotated_frames",
        description="Only frames that were annotated with node points and locations",
        format="external",
        external_file=[output_mjpeg],
        starting_frame=[0],
        rate=mjpeg_frame_rate, #mjpeg hard coded to have a 30 fps
    )

    all_videos = original_videos + [annotations_mjpeg]

    source_videos = SourceVideos(image_series=all_videos)

    return source_videos, annotations_mjpeg

def create_training_frames(
    labels: Labels,
    unique_skeletons: dict[str, Skeleton],
    annotations_mjpeg: ImageSeries,
    frame_map: dict[int, list[list[object]]],
    identity: bool = False,
) -> TrainingFrames:
    """Convert SLEAP annotations into NWB `TrainingFrames`.

    Parameters
    ----------
    labels :
        A SLEAP `Labels` object whose `LabeledFrame` records contain the
        annotations to export.
    unique_skeletons :
        Mapping of skeleton name → NWB `Skeleton` produced by
        :pyfunc:`create_skeletons`.
    annotations_mjpeg :
        The `ImageSeries` object pointing to the MJPEG of annotated frames
        (output from :pyfunc:`create_source_videos`).
    frame_map :
        Mapping from original `frame_idx` to `[global_idx, video_name]`
        entries created by :pyfunc:`get_frames_from_slp`.
    identity :
        If ``True``, append each instance’s track name to its skeleton name to
        guarantee uniqueness when identity tracking is present.

    Returns:
    -------
    TrainingFrames
        An NWB `TrainingFrames` container, where every `TrainingFrame` holds
        one `SkeletonInstances` composed of the per‑frame
        `SkeletonInstance`(s) and references the MJPEG via
        ``source_video``/``source_video_frame_index``.

    Raises:
    ------
    KeyError
        If an instance’s skeleton name is absent from `unique_skeletons`.
    """
    training_frames_list = []

    unique_ids = 0
    for lf_idx, lf in enumerate(labels.labeled_frames):
        skeleton_instances_list = []

        frame_idx  = lf.frame_idx

        for j in range(len(lf.instances)):
            val = lf.instances[j]

            skel_name = "Skeleton" + str(j)

            if identity:
                iden_string = val.track.name
                skel_name = skel_name + "_" + str(iden_string) # identity tracking

            node_locations_sk1 = np.array([[pt[0][0], pt[0][1]] for pt in val.points])

            instance_sk1 = SkeletonInstance(
                name=skel_name,
                id=np.uint64(unique_ids),
                node_locations=node_locations_sk1,
                node_visibility=[pt[1] for pt in val.points],
                skeleton=unique_skeletons[val.skeleton.name],
            )
            skeleton_instances_list.append(instance_sk1)
            unique_ids += 1

        # store the skeleton instances in a SkeletonInstances object
        skeleton_instances = SkeletonInstances(
            skeleton_instances=skeleton_instances_list)

        mapped = frame_map.get(frame_idx)

        if isinstance(mapped[0], list): #loop through if mult videos same frame idx
            video_path = val.video.filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            for vid in mapped:
                if vid[1] == video_name:
                    frame_idx = vid[0]
        else:
            frame_idx = mapped[0]

        training_frame = TrainingFrame(
            name=f"frame_{lf_idx}", # must be unique
            skeleton_instances=skeleton_instances,
            source_video=annotations_mjpeg,
            source_video_frame_index=np.uint64(frame_idx),
        )
        training_frames_list.append(training_frame)

    training_frames = TrainingFrames(training_frames=training_frames_list)

    return training_frames

def write_annotations_nwb(
    labels: Labels,
    nwbfile_path: str,
    nwb_file_kwargs: Optional[dict] = None,
    nwb_subject_kwargs: Optional[dict] = None,
) -> None:
    """Export SLEAP pose data and metadata to an NWB file.

    Parameters
    ----------
    labels :
        A SLEAP `Labels` object whose `LabeledFrame` entries contain the
        pose annotations to be written.
    nwbfile_path :
        Destination path for the NWB file. Will be overwritten if it
        already exists.
    nwb_file_kwargs :
        Optional keyword arguments forwarded directly to
        :class:`pynwb.NWBFile`. Missing required fields are auto‑filled:
        ``session_description`` (default ``"Processed SLEAP pose data"``),
        ``session_start_time`` (UTC ``datetime.now``), and ``identifier``
        (``uuid.uuid1()``).
    nwb_subject_kwargs :
        Optional keyword arguments forwarded to :class:`pynwb.file.Subject`.
        Missing DANDI‑required fields are auto‑filled: ``subject_id``
        (``"subject1"``), ``species`` (``"Mus musculus"``), ``sex`` (``"F"``),
        and ``age`` (``"P10W/P12W"``).

    Returns:
    -------
    None
        The function writes the NWB file to disk and has no return value.

    Side Effects
    ------------
    * Creates a *frames/* directory of PNGs extracted from every annotated
      video frame.
    * Generates *annotated_frames.avi* via ffmpeg and dumps
      *frame_map.json* and *input.txt* to the working directory.
    * Writes the assembled NWB file to *nwbfile_path* using
      :class:`pynwb.NWBHDF5IO`.

    Raises:
    ------
    RuntimeError
        If an original video cannot be opened or its FPS is unobtainable.
    subprocess.CalledProcessError
        If ffmpeg fails during MJPEG generation.
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
    subject_id = nwb_subject_kwargs.get(
        "subject_id", "subject1"
    )
    species = nwb_subject_kwargs.get(
        "species", "Mus musculus"
    )

    sex = nwb_subject_kwargs.get(
        "sex", "F"
    )

    age = nwb_subject_kwargs.get(
        "age", "P10W/P12W"
    )

    nwb_subject_kwargs.update(
        subject_id=subject_id,
        species=species,
        sex=sex,
        age=age,
    )

    subject = Subject(**nwb_subject_kwargs)
    nwbfile.subject = subject

    skeletons, frame_indices, unique_skeletons = create_skeletons(labels)

    image_list, frame_map = get_frames_from_slp(labels)
    output_mjpeg = make_mjpeg(image_list, frame_map)

    source_videos, annotations_mjpeg = create_source_videos(frame_indices, output_mjpeg)
    training_frames = create_training_frames(labels, unique_skeletons,
                                             annotations_mjpeg, frame_map)

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
