"""This module handles direct I/O operations for working with .slp files."""

from __future__ import annotations
import numpy as np
import simplejson as json
from typing import Union
from sleap_io import (
    Video,
    Skeleton,
    Edge,
    Node,
    Track,
    Point,
    PredictedPoint,
    Instance,
    PredictedInstance,
    LabeledFrame,
    Labels,
)
from sleap_io.io.utils import read_hdf5_attrs, read_hdf5_dataset
from sleap_io.io.video import VideoBackend
from enum import IntEnum


class InstanceType(IntEnum):
    """Enumeration of instance types to integers."""

    USER = 0
    PREDICTED = 1


def read_videos(labels_path: str) -> list[Video]:
    """Read `Video` dataset in a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Video` objects.
    """
    # TODO (DS) - Find shape of video
    videos = [json.loads(x) for x in read_hdf5_dataset(labels_path, "videos_json")]
    video_objects = []
    for video in videos:
        backend = video["backend"]
        try:
            backend = VideoBackend.from_filename(
                backend["filename"],
                dataset=backend.get("dataset", None),
                grayscale=backend.get("grayscale", None),
                input_format=backend.get("input_format", None),
            )
        except ValueError:
            backend = None
        video_objects.append(
            Video(filename=video["backend"]["filename"], backend=backend)
        )
    return video_objects


def read_tracks(labels_path: str) -> list[Track]:
    """Read `Track` dataset in a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Track` objects.
    """
    tracks = [json.loads(x) for x in read_hdf5_dataset(labels_path, "tracks_json")]
    track_objects = []
    for track in tracks:
        track_objects.append(Track(name=track[1]))
    return track_objects


def read_metadata(labels_path: str) -> dict:
    """Read metadata from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A dict containing the metadata from a SLEAP labels file.
    """
    md = read_hdf5_attrs(labels_path, "metadata", "json")
    assert type(md) == np.bytes_
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
        edge_inds = []
        for link in skel["links"]:
            if "py/reduce" in link["type"]:
                edge_type = link["type"]["py/reduce"][1]["py/tuple"][0]
            else:
                edge_type = link["type"]["py/id"]

            if edge_type == 1:  # 1 -> real edge, 2 -> symmetry edge
                edge_inds.append((link["source"], link["target"]))

        # Re-index correctly.
        skeleton_node_inds = [node["id"] for node in skel["nodes"]]
        node_names = [node_names[i] for i in skeleton_node_inds]
        edge_inds = [
            (skeleton_node_inds.index(s), skeleton_node_inds.index(d))
            for s, d in edge_inds
        ]
        nodes = []
        for name in node_names:
            nodes.append(Node(name=name))
        edges = []
        for edge in edge_inds:
            edges.append(Edge(source=nodes[edge[0]], destination=nodes[edge[1]]))
        skel = Skeleton(nodes=nodes, edges=edges, name=skel["graph"]["name"])
        skeleton_objects.append(skel)
    return skeleton_objects


def read_points(labels_path: str) -> list[Point]:
    """Read `Point` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file.

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
        labels_path: A string that contains the path to the labels file

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
        labels_path: A string that contains the path to the labels file
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

    instances = []
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
            instances.append(
                Instance(
                    points=points[point_id_start:point_id_end],  # type: ignore[arg-type]
                    skeleton=skeletons[skeleton_id],
                    track=tracks[track_id] if track_id >= 0 else None,
                )
            )
        elif instance_type == InstanceType.PREDICTED:
            instances.append(
                PredictedInstance(
                    points=pred_points[point_id_start:point_id_end],  # type: ignore[arg-type]
                    skeleton=skeletons[skeleton_id],
                    track=tracks[track_id] if track_id >= 0 else None,
                    score=instance_score,
                    tracking_score=tracking_score,
                )
            )
    return instances


def read_labels(labels_path: str) -> Labels:
    """Read a SLEAP labels file.

    Args:
        labels_path: Path to a SLEAP-formatted labels file (.slp).

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
