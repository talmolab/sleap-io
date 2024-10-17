"""This module handles direct I/O operations for working with JABS files."""

from __future__ import annotations

import h5py
import re
import os
import numpy as np
from typing import List, Optional, Union
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
    video = Video.from_filename(video_name)
    if not skeleton:
        skeleton = JABS_DEFAULT_SKELETON
    tracks = {}

    if not os.access(labels_path, os.F_OK):
        raise FileNotFoundError(f"{labels_path} doesn't exist.")
    if not os.access(labels_path, os.R_OK):
        raise PermissionError(f"{labels_path} cannot be accessed.")

    with h5py.File(labels_path, "r") as pose_file:
        num_frames = pose_file["poseest/points"].shape[0]
        try:
            pose_version = pose_file["poseest"].attrs["version"][0]
        except (KeyError, IndexError):
            pose_version = 2
            data_shape = pose_file["poseest/points"].shape
            assert (
                len(data_shape) == 3
            ), f"Pose version not present and shape does not match single mouse: shape of {data_shape} for {labels_path}"
        if pose_version == 2:
            tracks[1] = Track("1")
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
            frame_label = LabeledFrame(video, frame_idx, instances)
            frames.append(frame_label)
    labels = Labels(frames)
    labels.provenance["filename"] = labels_path
    return labels


def make_simple_skeleton(name: str, num_points: int) -> Skeleton:
    """Create a `Skeleton` with a requested number of nodes attached in a line.

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
        data: keypoint locations
        confidence: confidence for keypoints
        skeleton: `Skeleton` to use for `Instance`
        track: `Track` to assign to `Instance`

    Returns:
        Parsed `Instance`.
    """
    assert (
        len(skeleton.nodes) == data.shape[0]
    ), f"Skeleton ({len(skeleton.nodes)}) does not match number of keypoints ({data.shape[0]})"

    points = {}
    for i, cur_node in enumerate(skeleton.nodes):
        # confidence of 0 indicates no keypoint predicted for instance
        if confidence[i] > 0:
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
    """Determine the maximum number of identities that exist at the same time.

    Args:
        labels: SLEAP `Labels` to count
        key: Name of the skeleton to select for identities

    Returns:
        Count of the maximum concurrent identities in a single frame
    """
    max_labels = 0
    for label in labels:
        n_labels = sum([x.skeleton.name == key for x in label.instances])
        max_labels = max(max_labels, n_labels)

    return max_labels


def convert_labels(all_labels: Labels, video: Video) -> dict:
    """Convert a `Labels` object into JABS-formatted annotations.

    Args:
        all_labels: SLEAP `Labels` to be converted to JABS format.
        video: name of video to be converted

    Returns:
        Dictionary of JABS data of the `Labels` data.
    """
    labels = all_labels.find(video=video)

    # Determine shape of output
    # Low estimate of last frame labeled
    num_frames = max([x.frame_idx for x in labels]) + 1
    # If there is metadata available for the video, use that
    if video.shape:
        num_frames = max(num_frames, video.shape[0])
    if len(all_labels.skeletons) == 1:
        skeleton = all_labels.skeleton
    elif len(all_labels.skeletons) > 1:
        skeleton = [x for x in all_labels.skeletons if x.name == "Mouse"]
        if len(skeleton) == 0:
            raise ValueError("No mouse skeleton found in labels.")
        skeleton = skeleton[0]
    num_keypoints = len(skeleton.nodes)
    num_mice = get_max_ids_in_video(labels, key="Mouse")
    # Note that this 1-indexes identities
    track_2_idx = {
        key: val + 1
        for key, val in zip(all_labels.tracks, range(len(all_labels.tracks)))
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
        for instance_idx, instance in enumerate(label.instances):
            # Static objects just get added to the object dict
            # This will clobber data if more than one frame is annotated
            if instance.skeleton.name != "Mouse":
                static_objects[instance.skeleton.name] = instance.numpy()
                continue
            pose = instance.numpy()
            if pose.shape[0] != len(JABS_DEFAULT_KEYPOINTS):
                warnings.warn(
                    f"JABS format only supports 12 keypoints for mice. Skipping storage of instance on frame {label.frame_idx} with {len(instance.points)} keypoints."
                )
                continue
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


def write_labels(labels: Labels, pose_version: int, root_folder: str):
    """Convert and save a SLEAP `Labels` object to a JABS pose file.

    Only supports pose version 2 (single mouse) and 3-5 (multi mouse).

    Args:
        labels: SLEAP `Labels` to be converted to JABS pose format.
        pose_version: JABS pose version to use when writing data.
        root_folder: Root folder where the jabs files should be written
    """
    for video in labels.videos:
        converted_labels = convert_labels(labels, video)
        out_filename = (
            os.path.splitext(video.filename)[0] + f"_pose_est_v{pose_version}.h5"
        )
        if root_folder:
            out_filename = os.path.join(root_folder, out_filename)
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        if os.path.exists(out_filename):
            warnings.warn(f"Skipping {out_filename} because it already exists.")
            continue
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


def tracklets_to_v3(tracklet_matrix: np.ndarray) -> np.ndarray:
    """Changes identity tracklets to the v3 format specifications.

    v3 specifications require:
        (a) tracklets are 0-indexed
        (b) tracklets appear in ascending order
        (c) tracklets exist for continuous blocks of time

    Args:
        tracklet_matrix: Numpy array of shape (frame, n_animals) that contains identity values. Identities are assumed to be 1-indexed.

    Returns:
        A corrected numpy array of the same shape as input
    """
    assert tracklet_matrix.ndim == 2

    # Fragment the tracklets based on gaps
    valid_ids = np.unique(tracklet_matrix)
    valid_ids = valid_ids[valid_ids != 0]
    track_fragments = {}
    for cur_id in valid_ids:
        frame_idx, column_idx = np.where(tracklet_matrix == cur_id)
        gaps = np.nonzero(np.diff(frame_idx) - 1)[0]
        for sliced_frame, sliced_column in zip(
            np.split(frame_idx, gaps + 1), np.split(column_idx, gaps + 1)
        ):
            # The keys used here are (first frame, first column) such that sorting can be used for ascending order
            track_fragments[sliced_frame[0], sliced_column[0]] = sliced_column

    return_mat = np.zeros_like(tracklet_matrix)
    for next_id, key in enumerate(sorted(track_fragments.keys())):
        columns_to_assign = track_fragments[key]
        return_mat[
            range(key[0], key[0] + len(columns_to_assign)), columns_to_assign
        ] = next_id

    return return_mat


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

    Args:
        data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    v3_tracklets = tracklets_to_v3(data["identity"])
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
            v3_tracklets.shape,
            v3_tracklets.dtype,
            data=v3_tracklets,
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
        pose_grp.require_dataset(
            "instance_embed_id",
            data["identity"].shape,
            data["identity"].dtype,
            data=data["identity"],
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
