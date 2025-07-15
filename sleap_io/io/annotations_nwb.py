from copy import deepcopy
from typing import List, Optional, Union, Dict
from pathlib import Path
import datetime
import uuid
import re

import pandas as pd  # type: ignore[import]
import numpy as np
import cv2
import subprocess
import os

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = np.ndarray
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule  # type: ignore[import]
from pynwb.file import Subject
from pynwb.image import ImageSeries
from ndx_pose import (
    Skeleton,
    SkeletonInstance,
    TrainingFrame,
    PoseTraining,
    Skeletons,
    TrainingFrames,
    SourceVideos,
    SkeletonInstances,
)  # type: ignore[import]

from sleap_io import (
    Labels,
    Video,
    LabeledFrame,
    Track,
    Skeleton as SleapSkeleton,
    Instance,
    PredictedInstance,
)

def create_skeletons(labels):

    """Extract unique skeletons from SLEAP labels and return NWB Skeletons and indexed frames that were labeled"""

    unique_skeletons = {} # create a new NWB skeleton obj per unique skeleton in SLEAP
    frame_indices = {}

    for j, l in enumerate(labels):
        video = labels[j].video.filename
        
        if video not in frame_indices:
            frame_indices[video] = []
        
        for inst in l.instances:
            
            frame_indices[video].append(inst.frame_idx)

            skel = inst.skeleton
            skel_name = inst.skeleton.name # assumes unique skeletons in SLEAP are named uniquely 

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
        frame_indices[key] = list(sorted(set(frame_indices[key]))) # make frame indicies unique and sorted in ascending order

    return skeletons, frame_indices, unique_skeletons

def get_frames_from_slp(labels, mjpeg_frame_duration=30.0):
    
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

        frame = lf.video.get_frame(frame_idx) 

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
    

def make_mjpeg(image_list, frame_map):

    """Generate an MJPEG video from image files using ffmpeg and return the output filename"""

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

def create_source_videos(frame_indices, output_mjpeg, mjpeg_frame_rate=30.0):

    """Create SourceVideos object for NWB using frame index mapping and MJPEG output"""

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
            external_file=[video_name], #assumes all original videos will be uploaded in the same folder as nwb file
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

def create_training_frames(labels, total_frames, unique_skeletons, annotations_mjpeg, frame_map, identity=False):

    """Construct TrainingFrame and PoseTraining objects from SLEAP annotations for NWB export"""

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
                skel_name = skel_name + "_" + str(iden_string) # work around for identity tracking, append onto end of skeleetons instance name, need each skel_name to be unique for storage in SkeletonInstances
            
            node_locations_sk1 = np.array([[pt.x, pt.y] for pt in val.points])
            
            instance_sk1 = SkeletonInstance(
                name=skel_name,
                id=np.uint64(unique_ids),
                node_locations=node_locations_sk1,
                node_visibility=[pt.visible for pt in val.points],
                skeleton=unique_skeletons[val.skeleton.name],
            )
            skeleton_instances_list.append(instance_sk1)
            unique_ids += 1
        
        # store the skeleton instances in a SkeletonInstances object
        skeleton_instances = SkeletonInstances(skeleton_instances=skeleton_instances_list)

        mapped = frame_map.get(val.frame_idx)

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

    # store the training frames and source videos in their corresponding container objects
    training_frames = TrainingFrames(training_frames=training_frames_list)

    return training_frames

def write_annotations_nwb(
    labels: Labels,
    nwbfile_path: str,
    nwb_file_kwargs: Optional[dict] = None,
    nwb_subject_kwargs: Optional[dict] = None,
):
    """Write by hand annotations in non continuous frames of a video to an nwb file and save it to the nwbfile_path given.
    Creates an MJPEG of isolated annotated frames.

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

    total_frames = sum(len(v) for v in frame_indices.values()) # total annotated frames for all videos

    image_list, frame_map = get_frames_from_slp(labels)
    output_mjpeg = make_mjpeg(image_list, frame_map)

    source_videos, annotations_mjpeg = create_source_videos(frame_indices, output_mjpeg)
    training_frames = create_training_frames(labels, total_frames, unique_skeletons, annotations_mjpeg, frame_map)

    # store the skeletons group, training frames group, and source videos group in a PoseTraining object
    pose_training = PoseTraining(
        training_frames=training_frames,
        source_videos=source_videos,
    )

    # create a "behavior" processing module to store the PoseEstimation and PoseTraining objects
    behavior_pm = nwbfile.create_processing_module(
        name="behavior",                          
        description="processed behavioral data",       
    )
    behavior_pm.add(skeletons)
    behavior_pm.add(pose_training)

    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(nwbfile)