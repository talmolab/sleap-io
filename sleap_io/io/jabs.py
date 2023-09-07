"""This module handles direct I/O operations for working with JABS files.

"""

import h5py
import re
import os
import numpy as np
from typing import Dict, Iterable, List, Tuple, Optional, Union
import warnings

from sleap_io import (
    Instance,
    LabeledFrame,
    Labels,
    Node,
    Edge,
    Symmetry,
    Point,
    Video,
    Skeleton,
    Track,
)

JABS_DEFAULT_KEYPOINTS = [
    Node("NOSE"),
    Node("LEFT_EAR"),
    Node("RIGHT_EAR"),
    Node("BASE_NECK"),
    Node("LEFT_FRONT_PAW"),
    Node("RIGHT_FRONT_PAW"),
    Node("CENTER_SPINE"),
    Node("LEFT_REAR_PAW"),
    Node("RIGHT_REAR_PAW"),
    Node("BASE_TAIL"),
    Node("MID_TAIL"),
    Node("TIP_TAIL"),
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

JABS_DEFAULT_SKELETON = Skeleton(
    JABS_DEFAULT_KEYPOINTS, JABS_DEFAULT_EDGES, JABS_DEFAULT_SYMMETRIES, name="Mouse"
)


def read_labels(
    labels_path: str, skeleton: Optional[Skeleton] = JABS_DEFAULT_SKELETON
) -> Labels:
    """Read JABS style pose from a file and return a `Labels` object.
    TODO: Attributes are ignored, including px_to_cm field.
    TODO: Segmentation data ignored in v6, but will read in pose.
    TODO: Lixit static objects currently stored as n_lixit,2 (eg 1 object). Should be converted to multiple objects

    Args:
        labels_path: Path to the JABS pose file.
        skeleton: An optional `Skeleton` object. Defaults to JABS pose version 2-6.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    frames: List[LabeledFrame] = []
    # Video name is the pose file minus the suffix
    video_name = re.sub(r"(_pose_est_v[2-6])?\.h5", ".avi", labels_path)
    if not skeleton:
        skeleton = JABS_DEFAULT_SKELETON
    tracks = {}

    with h5py.File(labels_path, "r") as pose_file:
        num_frames = pose_file["poseest/points"].shape[0]
        try:
            pose_version = pose_file["poseest"].attrs["version"][0]
        except:
            pose_version = 2
            tracks[1] = Track("1")
            data_shape = pose_file["poseest/points"].shape
            assert (
                len(data_shape) == 3
            ), f"Pose version not present and shape does not match single mouse: shape of {data_shape} for {labels_path}"
        # Change field name for newer pose formats
        if pose_version == 3:
            id_key = "instance_track_id"
        elif pose_version > 3:
            id_key = "instance_embed_id"
            max_ids = pose_file["poseest/points"].shape[1]

        for frame_idx in range(num_frames):
            instances = []
            pose_data = pose_file["poseest/points"][frame_idx, ...]
            # JABS stores y,x for poses
            pose_data = np.flip(pose_data, axis=-1)
            pose_conf = pose_file["poseest/confidence"][frame_idx, ...]
            # single animal case
            if pose_version == 2:
                new_instance = prediction_to_instance(
                    pose_data, pose_conf, skeleton, tracks[1]
                )
                instances.append(new_instance)
            # multi-animal case
            if pose_version > 2:
                pose_ids = pose_file["poseest/" + id_key][frame_idx, ...]
                # pose_v3 uses another field to describe the number of valid poses
                if pose_version == 3:
                    max_ids = pose_file["poseest/instance_count"][frame_idx]
                for cur_id in range(max_ids):
                    # v4+ uses reserved values for invalid/unused poses
                    # Note: ignores 'poseest/id_mask' to keep predictions that were not assigned an id
                    if pose_version > 3 and pose_ids[cur_id] <= 0:
                        continue
                    if pose_ids[cur_id] not in tracks.keys():
                        tracks[pose_ids[cur_id]] = Track(str(pose_ids[cur_id]))
                    new_instance = prediction_to_instance(
                        pose_data[cur_id],
                        pose_conf[cur_id],
                        skeleton,
                        tracks[pose_ids[cur_id]],
                    )
                    if new_instance:
                        instances.append(new_instance)
            # Static objects
            if (
                frame_idx == 0
                and pose_version >= 5
                and "static_objects" in pose_file.keys()
            ):
                present_objects = pose_file["static_objects"].keys()
                for cur_object in present_objects:
                    object_keypoints = pose_file["static_objects/" + cur_object][:]
                    object_skeleton = make_simple_skeleton(
                        cur_object, object_keypoints.shape[0]
                    )
                    new_instance = prediction_to_instance(
                        object_keypoints,
                        np.ones(object_keypoints.shape[:-1]),
                        object_skeleton,
                    )
                    if new_instance:
                        instances.append(new_instance)
            frame_label = LabeledFrame(Video(video_name), frame_idx, instances)
            frames.append(frame_label)
    return Labels(frames)


def make_simple_skeleton(name: str, num_points: int) -> Skeleton:
    """Create a `Skeleton` with a requested number of nodes attached in a line

    Args:
        name: name of the skeleton and prefix to nodes
        num_points: number of points to use in the skeleton

    Returns:
        Generated `Skeleton`.

    """
    nodes = [Node(name + "_kp" + str(i)) for i in range(num_points)]
    edges = [Edge(nodes[i], nodes[i + 1]) for i in range(num_points - 1)]
    return Skeleton(nodes, edges, name=name)


def prediction_to_instance(
    data: Union[np.ndarray[np.uint16], np.ndarray[np.float32]],
    confidence: np.ndarray[np.float32],
    skeleton: Skeleton,
    track: Track = None,
) -> Instance:
    """Create an `Instance` from prediction data.

    Args:
        points: keypoint locations
        confidence: confidence for keypoints

    Returns:
        Parsed `Instance`.
    """
    assert (
        len(skeleton.nodes) == data.shape[0]
    ), f"Skeleton ({len(skeleton.nodes)}) does not match number of keypoints ({data.shape[0]})"

    points = {}
    for i, cur_node in enumerate(skeleton.nodes):
        # confidence of 0 indicates no keypoint predicted for instance
        if confidence[i] > 0.001:
            points[cur_node] = Point(
                data[i, 0],
                data[i, 1],
                visible=True,
            )

    if not points:
        return None
    else:
        return Instance(points, skeleton=skeleton, track=track)


def get_max_ids_in_video(labels: List[Labels], key: str = "Mouse") -> int:
    """Determine the maximum number of identities that exist at the same time

    Args:
        labels: SLEAP `Labels` to count
    """
    max_labels = 0
    for label in labels:
        n_labels = sum([x.skeleton.name == key for x in label.instances])
        max_labels = max(max_labels, n_labels)

    return max_labels


def convert_labels(all_labels: Labels, video: str) -> dict:
    """Convert a `Labels` object into JABS-formatted annotations.

    Args:
        all_labels: SLEAP `Labels` to be converted to JABS format.
        video: name of video to be converted

    Returns:
        Dictionary of JABS data of the `Labels` data.
    """
    labels = all_labels.find(video=video)

    # Determine shape of output
    num_frames = [x.shape[0] for x in all_labels.videos if x == video][0]
    num_keypoints = [len(x.nodes) for x in all_labels.skeletons if x.name == "Mouse"][0]
    num_mice = get_max_ids_in_video(labels, key="Mouse")
    track_2_idx = {
        key: val for key, val in zip(all_labels.tracks, range(len(all_labels.tracks)))
    }
    last_unassigned_id = num_mice

    keypoint_mat = np.zeros([num_frames, num_mice, num_keypoints, 2], dtype=np.uint16)
    confidence_mat = np.zeros([num_frames, num_mice, num_keypoints], dtype=np.float32)
    identity_mat = np.zeros([num_frames, num_mice], dtype=np.uint32)
    instance_vector = np.zeros([num_frames], dtype=np.uint8)
    static_objects = {}

    # Populate the matrices with data
    for label in labels:
        assigned_instances = 0
        tracks = [x.track for x in label.instances if x.track]
        track_ids = [track_2_idx[track] for track in tracks]
        for instance_idx, instance in enumerate(label.instances):
            # Don't handle instances without skeletons
            if not instance.skeleton:
                continue
            # Static objects just get added to the object dict
            # This will clobber data if more than one frame is annotated
            elif instance.skeleton.name != "Mouse":
                static_objects[instance.skeleton.name] = instance.numpy()
                continue
            pose = instance.numpy()
            missing_points = np.isnan(pose[:, 0])
            pose[np.isnan(pose)] = 0
            # JABS stores y,x for poses
            pose = np.flip(pose.astype(np.uint16), axis=-1)
            keypoint_mat[label.frame_idx, instance_idx, :, :] = pose
            confidence_mat[label.frame_idx, instance_idx, ~missing_points] = 1.0
            if instance.track:
                identity_mat[label.frame_idx, instance_idx] = track_2_idx[
                    instance.track
                ]
            else:
                warnings.warn(
                    f"Pose with unassigned track found on {label.video.filename} frame {label.frame_idx} instance {instance_idx}. Assigning ID {last_unassigned_id}."
                )
                identity_mat[label.frame_idx, instance_idx] = last_unassigned_id
                last_unassigned_id += 1
            assigned_instances += 1
        instance_vector[label.frame_idx] = assigned_instances

    # Return the data as a dict
    return {
        "keypoints": keypoint_mat.astype(np.uint16),
        "confidence": confidence_mat.astype(np.float32),
        "identity": identity_mat.astype(np.uint32),
        "num_identities": instance_vector.astype(np.uint16),
        "static_objects": static_objects,
    }


def write_labels(labels: Labels, pose_version: int):
    """Convert and save a SLEAP `Labels` object to a JABS pose file.
    Only supports pose version 2 (single mouse) and 3-5 (multi mouse).

    Args:
        labels: SLEAP `Labels` to be converted to JABS pose format.
        pose_version: JABS pose version to use when writing data.
    """

    for video in labels.videos:
        converted_labels = convert_labels(labels, video)
        out_filename = (
            os.path.splitext(video.filename)[0] + f"_pose_est_v{pose_version}.h5"
        )
        # Do we want to overwrite?
        if os.path.exists(out_filename):
            pass
        if pose_version == 2:
            write_jabs_v2(converted_labels, out_filename)
        elif pose_version == 3:
            write_jabs_v3(converted_labels, out_filename)
        elif pose_version == 4:
            write_jabs_v4(converted_labels, out_filename)
        elif pose_version == 5:
            write_jabs_v5(converted_labels, out_filename)
        else:
            raise NotImplementedError(f"Pose format {pose_version} not supported.")


def write_jabs_v2(data: dict, filename: str):
    """Write JABS pose file v2 data to file.
    Writes single mouse pose data.

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    # Check that we're trying to write single mouse data
    assert data["keypoints"].shape[1] == 1
    out_keypoints = np.squeeze(data["keypoints"], axis=1)
    out_confidences = np.squeeze(data["confidence"], axis=1)

    with h5py.File(filename, "w") as h5:
        pose_grp = h5.require_group("poseest")
        pose_grp.attrs.update({"version": [2, 0]})
        pose_grp.require_dataset(
            "points", out_keypoints.shape, out_keypoints.dtype, data=out_keypoints
        )
        pose_grp.require_dataset(
            "confidence",
            out_confidences.shape,
            out_confidences.dtype,
            data=out_confidences,
        )


def write_jabs_v3(data: dict, filename: str):
    """Write JABS pose file v3 data to file.
    Writes multi-mouse pose data.
    TODO: v3 requires continuous tracklets (eg no gaps) IDs need to be incremented for this field

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    with h5py.File(filename, "w") as h5:
        pose_grp = h5.require_group("poseest")
        pose_grp.attrs.update({"version": [3, 0]})
        # keypoint field
        pose_grp.require_dataset(
            "points",
            data["keypoints"].shape,
            data["keypoints"].dtype,
            data=data["keypoints"],
        )
        # confidence field
        pose_grp.require_dataset(
            "confidence",
            data["confidence"].shape,
            data["confidence"].dtype,
            data=data["confidence"],
        )
        # id field
        pose_grp.require_dataset(
            "instance_track_id",
            data["identity"].shape,
            data["identity"].dtype,
            data=data["identity"],
        )
        # instance count field
        pose_grp.require_dataset(
            "instance_count",
            data["num_identities"].shape,
            data["num_identities"].dtype,
            data=data["num_identities"],
        )
        # extra field where we don't have data, so fill with default data
        pose_grp.require_dataset(
            "instance_embedding",
            data["confidence"].shape,
            data["confidence"].dtype,
            data=np.zeros_like(data["confidence"]),
        )


def write_jabs_v4(data: dict, filename: str):
    """Write JABS pose file v4 data to file.
    Writes multi-mouse pose and longterm identity object data.

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    # v4 extends v3
    write_jabs_v3(data, filename)
    with h5py.File(filename, "a") as h5:
        pose_grp = h5.require_group("poseest")
        pose_grp.attrs.update({"version": [4, 0]})
        # new fields on top of v4
        identity_mask_mat = np.all(data["confidence"] == 0, axis=-1).astype(bool)
        pose_grp.require_dataset(
            "id_mask",
            identity_mask_mat.shape,
            identity_mask_mat.dtype,
            data=identity_mask_mat,
        )
        # No identity embedding data
        # Note that since the identity information doesn't exist, this will break any functionality that relies on it
        default_id_embeds = np.zeros(
            list(identity_mask_mat.shape) + [0], dtype=np.float32
        )
        pose_grp.require_dataset(
            "identity_embeds",
            default_id_embeds.shape,
            default_id_embeds.dtype,
            data=default_id_embeds,
        )
        default_id_centers = np.zeros(default_id_embeds.shape[1:], dtype=np.float32)
        pose_grp.require_dataset(
            "instance_id_center",
            default_id_centers.shape,
            default_id_centers.dtype,
            data=default_id_centers,
        )
        # v4 uses an id field that is 1-indexed
        identities_1_indexed = np.copy(data["identity"]) + 1
        identities_1_indexed[identity_mask_mat] = 0
        pose_grp.require_dataset(
            "instance_embed_id",
            identities_1_indexed.shape,
            identities_1_indexed.dtype,
            data=identities_1_indexed,
        )


def write_jabs_v5(data: dict, filename: str):
    """Write JABS pose file v5 data to file.
    Writes multi-mouse pose, longterm identity, and static object data.

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    # v5 extends v4
    write_jabs_v4(data, filename)
    with h5py.File(filename, "a") as h5:
        pose_grp = h5.require_group("poseest")
        pose_grp.attrs.update({"version": [5, 0]})
        if "static_objects" in data.keys():
            object_grp = h5.require_group("static_objects")
            for object_key, object_keypoints in data["static_objects"].items():
                object_grp.require_dataset(
                    object_key,
                    object_keypoints.shape,
                    np.uint16,
                    data=object_keypoints.astype(np.uint16),
                )
