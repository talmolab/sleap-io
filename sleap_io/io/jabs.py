"""This module handles direct I/O operations for working with JABS files.

"""

import h5py
import re
import os
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
    TODO: Currently only reads in pose data. v5 static objects are currently ignored

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
            tracks[1] = Track('1')
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
                    # Note: ignores 'poseest/id_mask' to keep predictions that were not assigned an id
                    if pose_version > 3 and pose_ids[cur_id] <= 0:
                        continue
                    if cur_id not in tracks.keys():
                        tracks[cur_id] = Track(str(pose_ids[cur_id]))
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
        return Instance(points, skeleton=skeleton, track=track)

def get_max_ids_in_video(labels: Labels) -> int:
    """Determine the maximum number of identities that exist at the same time
    
    Args:
        labels: SLEAP `Labels` to count 
    """
    max_labels = 0
    for label in labels.labeled_frames:
        n_labels = len(label.instances)
        max_labels = max(max_labels, n_labels)

    return max_labels

def convert_labels(labels: Labels) -> dict:
    """Convert a `Labels` object into JABS-formatted annotations.
    TODO: Currently assumes all data is mouse
    TODO: Identity is an unsafe str -> cast. Convert to factorize op

    Args:
        labels: SLEAP `Labels` to be converted to JABS format.

    Returns:
        Dictionary of JABS data of the `Labels` data.
    """
    # Determine shape of output
    num_frames = labels.video.shape[0]
    num_keypoints = len(labels.skeleton.nodes)
    num_mice = get_max_ids_in_video(labels)

    keypoint_mat = np.zeros([num_frames, num_mice, num_keypoints, 2], dtype=np.uint16)
    confidence_mat = np.zeros([num_frames, num_mice, num_keypoints], dtype=np.float32)
    identity_mat = np.zeros([num_frames, num_mice], dtype=np.uint32)
    instance_vector = np.zeros([num_frames], dtype=np.uint8)

    # Populate the matrices with data
    for label in labels.labeled_frames:
        assigned_instances = 0
        for instance_idx, instance in enumerate(label.instances):
            pose = instance.numpy()
            missing_points = np.isnan(pose[:,0])
            pose[np.isnan(pose)] = 0
            # JABS stores y,x
            pose = pose.astype(np.uint16)[:,::-1]
            keypoint_mat[label.frame_idx, instance_idx] = pose
            confidence_mat[label.frame_idx, instance_idx, ~missing_points] = 1.0
            identity_mat[label.frame_idx, instance_idx] = np.uint32(instance.track.name)
            assigned_instances += 1
        instance_vector[label.frame_idx] = assigned_instances

    # Return the data as a dict
    return {'keypoints': keypoint_mat, 'confidence': confidence_mat, 'identity': identity_mat, 'num_identities': instance_vector}

def write_labels(labels: Labels, filename: str, pose_version: int):
    """Convert and save a SLEAP `Labels` object to a JABS pose file.
    Only supports pose version 2 (single mouse) and 3-5 (multi mouse).

    Args:
        labels: SLEAP `Labels` to be converted to JABS pose format.
        filename: Path to save JABS annotations (`.h5`).
        pose_version: JABS pose version to use when writing data.
    """

    for video in labels.videos:
        video_labels = labels.find(video=video)
        converted_labels = convert_labels(video_labels)
        out_filename = re.sub('\.avi', f'_pose_est_v{pose_version}.h5', video.filename)
        if os.path.exists(out_filename):
            pass

def write_jabs_v2(data: dict, filename: str):
    """ Write JABS pose file v2 data to file.
    Writes single mouse pose data.

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    # Check that we're trying to write single mouse data
    assert data['keypoints'].shape[1] == 1
    out_keypoints = np.squeeze(data['keypoints'], axis=1)
    out_confidences = np.squeeze(data['confidence'], axis=1)

    with h5py.File(filename, 'w') as h5:
        pose_grp = h5.require_group('poseest')
        pose_grp.attrs.update({'version':[2,0]})
        pose_dataset = pose_grp.require_dataset('points', out_keypoints.shape, out_keypoints.dtype, data = out_keypoints)


def write_jabs_v3(data: dict, filename: str):
    """ Write JABS pose file v3 data to file.
    Writes multi-mouse pose data.

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    with h5py.File(filename, 'w') as h5:
        pose_grp = h5.require_group('poseest')
        pose_grp.attrs.update({'version':[3,0]})
        # keypoint field
        pose_dataset = pose_grp.require_dataset('points', data['keypoints'].shape, data['keypoints'].dtype, data = data['keypoints'])
        # confidence field
        conf_dataset = pose_grp.require_dataset('confidence', data['confidence'].shape, data['confidence'].dtype, data = data['confidence'])
        # id field
        id_dataset = pose_grp.require_dataset('instance_track_id', data['identity'].shape, data['identity'].dtype, data = data['identity'])
        # instance count field
        count_dataset = pose_grp.require_dataset('instance_count', data['num_identities'].shape, data['num_identities'].dtype, data = data['num_identities'])
        # extra field where we don't have data
        kp_embedding_dataset = pose_grp.require_dataset('instance_embedding', data['confidence'].shape, data['confidence'].dtype, data = np.zeros_like(data['confidence']))

def write_jabs_v4(data: dict, filename: str):
    """ Write JABS pose file v4 data to file.
    Writes multi-mouse pose and longterm identity object data.

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    pass

def write_jabs_v5(data: dict, filename: str):
    """ Write JABS pose file v5 data to file.
    Writes multi-mouse pose, longterm identity, and static object data.

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    pass
