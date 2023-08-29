"""This module handles direct I/O operations for working with JABS files.

"""

import h5py
import re
import numpy as np
from typing import Dict, Iterable, List, Tuple, Optional, Union

from sleap_io import Instance, LabeledFrame, Labels, Node, Edge, Symmetry, Point, Video, Skeleton, Track

JABS_DEFAULT_KEYPOINTS = [
    Node('NOSE'),
    Node('LEFT_EAR'),
    Node('RIGHT_EAR'),
    Node('BASE_NECK'),
    Node('LEFT_FRONT_PAW'),
    Node('RIGHT_FRONT_PAW'),
    Node('CENTER_SPINE'),
    Node('LEFT_REAR_PAW'),
    Node('RIGHT_REAR_PAW'),
    Node('BASE_TAIL'),
    Node('MID_TAIL'),
    Node('TIP_TAIL')
]

# Root node is base neck (3)
JABS_DEFAULT_EDGES = [
    # Spine
    Edge(JABS_DEFAULT_KEYPOINTS[3], JABS_DEFAULT_KEYPOINTS[0]),
    Edge(JABS_DEFAULT_KEYPOINTS[3], JABS_DEFAULT_KEYPOINTS[6]),
    Edge(JABS_DEFAULT_KEYPOINTS[6], JABS_DEFAULT_KEYPOINTS[9]),
    Edge(JABS_DEFAULT_KEYPOINTS[9], JABS_DEFAULT_KEYPOINTS[10]),
    Edge(JABS_DEFAULT_KEYPOINTS[10], JABS_DEFAULT_KEYPOINTS[11]),
    # Ears
    Edge(JABS_DEFAULT_KEYPOINTS[0], JABS_DEFAULT_KEYPOINTS[1]),
    Edge(JABS_DEFAULT_KEYPOINTS[0], JABS_DEFAULT_KEYPOINTS[2]),
    # Front paws
    Edge(JABS_DEFAULT_KEYPOINTS[6], JABS_DEFAULT_KEYPOINTS[4]),
    Edge(JABS_DEFAULT_KEYPOINTS[6], JABS_DEFAULT_KEYPOINTS[5]),
    # Rear paws
    Edge(JABS_DEFAULT_KEYPOINTS[9], JABS_DEFAULT_KEYPOINTS[7]),
    Edge(JABS_DEFAULT_KEYPOINTS[9], JABS_DEFAULT_KEYPOINTS[8]),
]

JABS_DEFAULT_SYMMETRIES = [
    Symmetry([JABS_DEFAULT_KEYPOINTS[1], JABS_DEFAULT_KEYPOINTS[2]]),
    Symmetry([JABS_DEFAULT_KEYPOINTS[4], JABS_DEFAULT_KEYPOINTS[5]]),
    Symmetry([JABS_DEFAULT_KEYPOINTS[7], JABS_DEFAULT_KEYPOINTS[8]]),
]

JABS_DEFAULT_SKELETON = Skeleton(JABS_DEFAULT_KEYPOINTS, JABS_DEFAULT_EDGES, JABS_DEFAULT_SYMMETRIES, name='Mouse')

def read_labels(labels_path: str, skeleton: Optional[Skeleton] = JABS_DEFAULT_SKELETON) -> Labels:
    """Read JABS style pose from a file and return a `Labels` object.

    Args:
        labels_path: Path to the JABS pose file.
        skeleton: An optional `Skeleton` object. Defaults to JABS pose version 2-6. 

    Returns:
        Parsed labels as a `Labels` instance.
    """
    frames: List[LabeledFrame] = []
    # Video name is the pose file minus the suffix
    video_name = re.sub(r'(_pose_est_v[2-6])?\.h5', '.avi', labels_path)
    if not skeleton:
        skeleton = JABS_DEFAULT_SKELETON
    tracks = {}

    with h5py.File(labels_path, "r") as pose_file:
        num_frames = pose_file['poseest/points'].shape[0]
        try:
            pose_version = pose_file['poseest'].attrs['version'][0]
        except:
            pose_version = 2
            data_shape = pose_file['poseest/points'].shape
            assert len(data_shape)==3, f'Pose version not present and shape does not match single mouse: shape of {data_shape} for {labels_path}'
        # Change field name for newer pose formats
        if pose_version == 3:
            id_key = 'instance_track_id'
            tracks[1] = Track('Mouse 1')
        elif pose_version > 3:
            id_key = 'instance_embed_id'
            max_ids = pose_file['poseest/points'].shape[1]

        for frame_idx in range(num_frames):
            instances = []
            pose_data = pose_file['poseest/points'][frame_idx, ...]
            pose_conf = pose_file['poseest/confidence'][frame_idx, ...]
            # single animal case
            if pose_version == 2:
                new_instance = prediction_to_instance(pose_data, pose_conf, skeleton, tracks[1])
                instances.append(new_instance)
            # multi-animal case
            if pose_version > 2:
                pose_ids = pose_file['poseest/' + id_key][frame_idx, ...]
                # pose_v3 uses another field to describe the number of valid poses
                if pose_version == 3:
                    max_ids = pose_file['poseest/instance_count'][frame_idx]
                for cur_id in range(max_ids):
                    # v4+ uses reserved values for invalid/unused poses
                    # note that we're ignoring 'poseest/id_mask' to keep predictions that were not assigned an id
                    if pose_version > 3 and pose_ids[cur_id] <= 0:
                        continue
                    if cur_id not in tracks.keys():
                        tracks[cur_id] = Track(f'Mouse {pose_ids[cur_id]}')
                    new_instance = prediction_to_instance(pose_data[cur_id], pose_conf[cur_id], skeleton, tracks[cur_id])
                    if new_instance:
                        instances.append(new_instance)
            frame_label = LabeledFrame(Video(video_name), frame_idx, instances)
            frames.append(frame_label)
    return Labels(frames)


def prediction_to_instance(data: Union[np.ndarray[np.uint16], np.ndarray[np.float32]], confidence: np.ndarray[np.float32], skeleton: Skeleton, track: Track = None) -> Instance:
    """Create an `Instance` from prediction data.

    Args:
        points: keypoint locations
        confidence: confidence for keypoints

    Returns:
        Parsed `Instance`.
    """
    assert len(skeleton.nodes) == data.shape[0], f'Skeleton ({len(skeleton.nodes)}) does not match number of keypoints ({data.shape[0]})'

    points = {}
    for i, cur_node in enumerate(skeleton.nodes):
        # confidence of 0 indicates no keypoint predicted for instance
        if confidence[i] > 0.001:
            points[cur_node] = Point(
                data[i,1],
                data[i,0],
                visible=True,
            )

    if not points:
        return None
    else:
        return Instance(points, skeleton=skeleton)
        # Tracks aren't saving correctly...
        #return Instance(points, skeleton=skeleton, track=Track)


