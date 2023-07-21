"""This module handles direct I/O operations for working with .slp files."""

from __future__ import annotations
import numpy as np
import h5py
import simplejson as json
from typing import Union
from sleap_io import (
    Video,
    Skeleton,
    Edge,
    Symmetry,
    Node,
    Track,
    Point,
    PredictedPoint,
    Instance,
    PredictedInstance,
    LabeledFrame,
    Labels,
)
from sleap_io.io.video import MediaVideo, HDF5Video
from sleap_io.io.utils import (
    read_hdf5_attrs,
    read_hdf5_dataset,
    write_hdf5_dataset,
    write_hdf5_group,
    write_hdf5_attrs,
)
from sleap_io.io.video import VideoBackend
from enum import IntEnum
from pathlib import Path


class InstanceType(IntEnum):
    """Enumeration of instance types to integers."""

    USER = 0
    PREDICTED = 1


def read_videos(labels_path: str) -> list[Video]:
    """Read `Video` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `Video` objects.
    """
    # TODO (DS) - Find shape of video
    videos = [json.loads(x) for x in read_hdf5_dataset(labels_path, "videos_json")]
    video_objects = []
    for video in videos:
        backend = video["backend"]
        video_path = backend["filename"]

        # Marker for embedded videos.
        if video_path == ".":
            video_path = labels_path

        # Basic path resolution.
        video_path = Path(video_path)
        if not video_path.exists():
            # Check for the same filename in the same directory as the labels file.
            video_path_ = Path(labels_path).parent / video_path.name
            if video_path_.exists():
                video_path = video_path_
            else:
                # TODO (TP): Expand capabilities of path resolution to support more
                # complex path finding strategies.
                pass

        try:
            backend = VideoBackend.from_filename(
                video_path.as_posix(),
                dataset=backend.get("dataset", None),
                grayscale=backend.get("grayscale", None),
                input_format=backend.get("input_format", None),
            )
        except ValueError:
            backend = None
        video_objects.append(Video(filename=video_path.as_posix(), backend=backend))
    return video_objects


def write_videos(labels_path: str, videos: list[Video]):
    """Write video metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects to store the metadata for.
    """
    video_jsons = []
    for video in videos:
        if type(video.backend) == MediaVideo:
            video_json = {
                "backend": {
                    "filename": video.filename,
                    "grayscale": video.backend.grayscale,
                    "bgr": True,
                    "dataset": "",
                    "input_format": "",
                }
            }

        elif type(video.backend) == HDF5Video:
            video_json = {
                "backend": {
                    "filename": "."
                    if video.backend.has_embedded_images
                    else video.filename,
                    "dataset": video.backend.dataset,
                    "input_format": video.backend.input_format,
                    "convert_range": False,
                }
            }
            # TODO: Handle saving embedded images or restoring source video.
            # Ref: https://github.com/talmolab/sleap/blob/fb61b6ce7a9ac9613d99303111f3daafaffc299b/sleap/io/format/hdf5.py#L246-L273

        else:
            raise NotImplementedError(
                f"Cannot serialize video backend for video: {video}"
            )
        video_jsons.append(np.string_(json.dumps(video_json, separators=(",", ":"))))

        with h5py.File(labels_path, "a") as f:
            f.create_dataset("videos_json", data=video_jsons, maxshape=(None,))


def read_tracks(labels_path: str) -> list[Track]:
    """Read `Track` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `Track` objects.
    """
    tracks = [json.loads(x) for x in read_hdf5_dataset(labels_path, "tracks_json")]
    track_objects = []
    for track in tracks:
        track_objects.append(Track(name=track[1]))
    return track_objects


def write_tracks(labels_path: str, tracks: list[Track]):
    """Write track metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        tracks: A list of `Track` objects to store the metadata for.
    """
    # TODO: Add support for track metadata like spawned on frame.
    SPAWNED_ON = 0
    tracks_json = [
        np.string_(json.dumps([SPAWNED_ON, track.name], separators=(",", ":")))
        for track in tracks
    ]
    with h5py.File(labels_path, "a") as f:
        f.create_dataset("tracks_json", data=tracks_json, maxshape=(None,))


def read_metadata(labels_path: str) -> dict:
    """Read metadata from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A dict containing the metadata from a SLEAP labels file.
    """
    md = read_hdf5_attrs(labels_path, "metadata", "json")
    return json.loads(md.decode())


def read_skeletons(labels_path: str) -> list[Skeleton]:
    """Read `Skeleton` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file.

    Returns:
        A list of `Skeleton` objects.
    """
    metadata = read_metadata(labels_path)

    # Get node names. This is a superset of all nodes across all skeletons. Note that
    # node ordering is specific to each skeleton, so we'll need to fix this afterwards.
    node_names = [x["name"] for x in metadata["nodes"]]

    skeleton_objects = []
    for skel in metadata["skeletons"]:
        # Parse out the cattr-based serialization stuff from the skeleton links.
        edge_inds, symmetry_inds = [], []
        for link in skel["links"]:
            if "py/reduce" in link["type"]:
                edge_type = link["type"]["py/reduce"][1]["py/tuple"][0]
            else:
                edge_type = link["type"]["py/id"]

            if edge_type == 1:  # 1 -> real edge, 2 -> symmetry edge
                edge_inds.append((link["source"], link["target"]))

            elif edge_type == 2:
                symmetry_inds.append((link["source"], link["target"]))

        # Re-index correctly.
        skeleton_node_inds = [node["id"] for node in skel["nodes"]]
        node_names = [node_names[i] for i in skeleton_node_inds]

        # Create nodes.
        nodes = []
        for name in node_names:
            nodes.append(Node(name=name))

        # Create edges.
        edge_inds = [
            (skeleton_node_inds.index(s), skeleton_node_inds.index(d))
            for s, d in edge_inds
        ]
        edges = []
        for edge in edge_inds:
            edges.append(Edge(source=nodes[edge[0]], destination=nodes[edge[1]]))

        # Create symmetries.
        symmetry_inds = [
            (skeleton_node_inds.index(s), skeleton_node_inds.index(d))
            for s, d in symmetry_inds
        ]
        symmetries = []
        for symmetry in symmetry_inds:
            symmetries.append(Symmetry([nodes[symmetry[0]], nodes[symmetry[1]]]))

        # Create the full skeleton.
        skel = Skeleton(
            nodes=nodes, edges=edges, symmetries=symmetries, name=skel["graph"]["name"]
        )
        skeleton_objects.append(skel)
    return skeleton_objects


def serialize_skeletons(skeletons: list[Skeleton]) -> tuple[list[dict], list[dict]]:
    """Serialize a list of `Skeleton` objects to JSON-compatible dicts.

    Args:
        skeletons: A list of `Skeleton` objects.

    Returns:
        A tuple of `nodes_dicts, skeletons_dicts`.

        `nodes_dicts` is a list of dicts containing the nodes in all the skeletons.

        `skeletons_dicts` is a list of dicts containing the skeletons.

    Notes:
        This function attempts to replicate the serialization of skeletons in legacy
        SLEAP which relies on a combination of networkx's graph serialization and our
        own metadata used to store nodes and edges independent of the graph structure.

        However, because sleap-io does not currently load in the legacy metadata, this
        function will not produce byte-level compatible serialization with legacy
        formats, even though the ordering and all attributes of nodes and edges should
        match up.
    """
    # Create global list of nodes with all nodes from all skeletons.
    nodes_dicts = []
    node_to_id = {}
    for skeleton in skeletons:
        for node in skeleton.nodes:
            if node not in node_to_id:
                # Note: This ID is not the same as the node index in the skeleton in
                # legacy SLEAP, but we do not retain this information in the labels, so
                # IDs will be different.
                #
                # The weight is also kept fixed here, but technically this is not
                # modified or used in legacy SLEAP either.
                #
                # TODO: Store legacy metadata in labels to get byte-level compatibility?
                node_to_id[node] = len(node_to_id)
                nodes_dicts.append({"name": node.name, "weight": 1.0})

    skeletons_dicts = []
    for skeleton in skeletons:
        # Build links dicts for normal edges.
        edges_dicts = []
        for edge_ind, edge in enumerate(skeleton.edges):
            if edge_ind == 0:
                edge_type = {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [1]},  # 1 = real edge, 2 = symmetry edge
                    ]
                }
            else:
                edge_type = {"py/id": 1}

            edges_dicts.append(
                {
                    # Note: Insert idx is not the same as the edge index in the skeleton
                    # in legacy SLEAP.
                    "edge_insert_idx": edge_ind,
                    "key": 0,  # Always 0.
                    "source": node_to_id[edge.source],
                    "target": node_to_id[edge.destination],
                    "type": edge_type,
                }
            )

        # Build links dicts for symmetry edges.
        for symmetry_ind, symmetry in enumerate(skeleton.symmetries):
            if symmetry_ind == 0:
                edge_type = {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [2]},  # 1 = real edge, 2 = symmetry edge
                    ]
                }
            else:
                edge_type = {"py/id": 2}

            src, dst = tuple(symmetry.nodes)
            edges_dicts.append(
                {
                    "key": 0,
                    "source": node_to_id[src],
                    "target": node_to_id[dst],
                    "type": edge_type,
                }
            )

        # Create skeleton dict.
        skeletons_dicts.append(
            {
                "directed": True,
                "graph": {
                    "name": skeleton.name,
                    "num_edges_inserted": len(skeleton.edges),
                },
                "links": edges_dicts,
                "multigraph": True,
                # In the order in Skeleton.nodes and must match up with nodes_dicts.
                "nodes": [{"id": node_to_id[node]} for node in skeleton.nodes],
            }
        )

    return skeletons_dicts, nodes_dicts


def write_metadata(labels_path: str, labels: Labels):
    """Write metadata to a SLEAP labels file.

    This function will write the skeletons and provenance for the labels.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.

    See also: serialize_skeletons
    """
    skeletons_dicts, nodes_dicts = serialize_skeletons(labels.skeletons)

    md = {
        "version": "2.0.0",
        "skeletons": skeletons_dicts,
        "nodes": nodes_dicts,
        "videos": [],
        "tracks": [],
        "suggestions": [],  # TODO: Handle suggestions metadata.
        "negative_anchors": [],
        "provenance": labels.provenance,
    }
    with h5py.File(labels_path, "a") as f:
        grp = f.require_group("metadata")
        grp.attrs["format_id"] = 1.2
        grp.attrs["json"] = np.string_(json.dumps(md, separators=(",", ":")))


def read_points(labels_path: str) -> list[Point]:
    """Read `Point` dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `Point` objects.
    """
    pts = read_hdf5_dataset(labels_path, "points")
    return [
        Point(x=x, y=y, visible=visible, complete=complete)
        for x, y, visible, complete in pts
    ]


def read_pred_points(labels_path: str) -> list[PredictedPoint]:
    """Read `PredictedPoint` dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `PredictedPoint` objects.
    """
    pred_pts = read_hdf5_dataset(labels_path, "pred_points")
    return [
        PredictedPoint(x=x, y=y, visible=visible, complete=complete, score=score)
        for x, y, visible, complete, score in pred_pts
    ]


def read_instances(
    labels_path: str,
    skeletons: list[Skeleton],
    tracks: list[Track],
    points: list[Point],
    pred_points: list[PredictedPoint],
    format_id: float,
) -> list[Union[Instance, PredictedInstance]]:
    """Read `Instance` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        skeletons: A list of `Skeleton` objects (see `read_skeletons`).
        tracks: A list of `Track` objects (see `read_tracks`).
        points: A list of `Point` objects (see `read_points`).
        pred_points: A list of `PredictedPoint` objects (see `read_pred_points`).
        format_id: The format version identifier used to specify the format of the input
            file.

    Returns:
        A list of `Instance` and/or `PredictedInstance` objects.
    """
    instances_data = read_hdf5_dataset(labels_path, "instances")

    instances = {}
    from_predicted_pairs = []
    for instance_data in instances_data:
        if format_id < 1.2:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
            ) = instance_data
            tracking_score = np.zeros_like(instance_score)
        else:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
                tracking_score,
            ) = instance_data

        if instance_type == InstanceType.USER:
            instances[instance_id] = Instance(
                points=points[point_id_start:point_id_end],  # type: ignore[arg-type]
                skeleton=skeletons[skeleton_id],
                track=tracks[track_id] if track_id >= 0 else None,
            )
            if from_predicted >= 0:
                from_predicted_pairs.append((instance_id, from_predicted))
        elif instance_type == InstanceType.PREDICTED:
            instances[instance_id] = PredictedInstance(
                points=pred_points[point_id_start:point_id_end],  # type: ignore[arg-type]
                skeleton=skeletons[skeleton_id],
                track=tracks[track_id] if track_id >= 0 else None,
                score=instance_score,
                tracking_score=tracking_score,
            )

    # Link instances based on from_predicted field.
    for instance_id, from_predicted in from_predicted_pairs:
        instances[instance_id].from_predicted = instances[from_predicted]

    # Convert instances back to list.
    instances = list(instances.values())

    return instances


def read_labels(labels_path: str) -> Labels:
    """Read a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        The processed `Labels` object.
    """
    tracks = read_tracks(labels_path)
    videos = read_videos(labels_path)
    skeletons = read_skeletons(labels_path)
    points = read_points(labels_path)
    pred_points = read_pred_points(labels_path)
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")
    assert isinstance(format_id, float)
    instances = read_instances(
        labels_path, skeletons, tracks, points, pred_points, format_id
    )
    metadata = read_metadata(labels_path)
    provenance = metadata.get("provenance", dict())

    frames = read_hdf5_dataset(labels_path, "frames")
    labeled_frames = []
    for _, video_id, frame_idx, instance_id_start, instance_id_end in frames:
        labeled_frames.append(
            LabeledFrame(
                video=videos[video_id],
                frame_idx=frame_idx,
                instances=instances[instance_id_start:instance_id_end],
            )
        )

    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        provenance=provenance,
    )

    return labels
