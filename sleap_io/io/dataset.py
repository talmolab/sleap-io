from attr import Factory
import h5py
import numpy as np
import pandas as pd
import json
from sleap_io.model.video import Video
from sleap_io.model.skeleton import Skeleton, Edge, Node
from typing import List
from sleap_io.model.instance import (
    Instance,
    LabeledFrame,
    Track,
    Point,
    PredictedInstance,
)


def read_hdf5(filename, dataset="/"):
    """Read data from an HDF5 file.

    Args:
        filename: Path to an HDF5 file.
        dataset: Path to a dataset or group. If a dataset, return the entire
            dataset as an array. If group, all datasets contained within the
            group will be recursively loaded and returned in a dict keyed by
            their full path. Defaults to "/" (load everything).

    Returns:
        The data as an array (for datasets) or dictionary (for groups).
    """
    data = {}

    def read_datasets(k, v):
        if type(v) == h5py.Dataset:
            data[v.name] = v[()]

    with h5py.File(filename, "r") as f:
        if type(f[dataset]) == h5py.Group:
            f.visititems(read_datasets)
        elif type(f[dataset]) == h5py.Dataset:
            data = f[dataset][()]
    return data


def read_videos(labels_path):
    """Read Videos dataset in a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Video` objects.
    """
    videos = [json.loads(x) for x in read_hdf5(labels_path, "videos_json")]
    video_objects = []
    for video in videos:
        video_objects.append(Video(filename=video["backend"]["filename"]))
    return video_objects


def read_tracks(labels_path):
    """Read tracks dataset in a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Track` objects.
    """
    tracks = [json.loads(x) for x in read_hdf5(labels_path, "tracks_json")]
    track_objects = []
    for track in tracks:
        track_objects.append(Track(name=track[1]))
    return track_objects


def read_metadata(labels_path):
    """Read metadata from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A dict containing the metadata from a SLEAP labels file.
    """
    with h5py.File(labels_path, "r") as f:
        attrs = dict(f["metadata"].attrs)
    metadata = json.loads(attrs["json"].decode())
    return metadata


def read_skeleton(labels_path):
    """Read skeleton dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A `Skeleton` object.
    """
    metadata = read_metadata(labels_path)

    # Get node names. This is a superset of all nodes across all skeletons. Note that
    # node ordering is specific to each skeleton, so we'll need to fix this afterwards.
    node_names = [x["name"] for x in metadata["nodes"]]

    # TODO: Support multi-skeleton?
    skeleton = metadata["skeletons"][0]
    # Parse out the cattr-based serialization stuff from the skeleton links.
    edge_inds = []
    for link in skeleton["links"]:
        if "py/reduce" in link["type"]:
            edge_type = link["type"]["py/reduce"][1]["py/tuple"][0]
        else:
            edge_type = link["type"]["py/id"]

        if edge_type == 1:  # 1 -> real edge, 2 -> symmetry edge
            edge_inds.append((link["source"], link["target"]))

    # Re-index correctly.
    skeleton_node_inds = [node["id"] for node in skeleton["nodes"]]
    node_names = [node_names[i] for i in skeleton_node_inds]
    edge_inds = [
        (skeleton_node_inds.index(s), skeleton_node_inds.index(d)) for s, d in edge_inds
    ]
    nodes = []
    for name in node_names:
        nodes.append(Node(name=name))
    edges = []
    for edge in edge_inds:
        edges.append(Edge(source=nodes[edge[0]], destination=nodes[edge[1]]))
    skeleton = Skeleton(nodes=nodes, edges=edges, name=skeleton["graph"]["name"])
    return skeleton


def read_points(labels_path):
    """Read points dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Point` objects.
    """
    points = read_hdf5(labels_path, "points")
    return points


def read_pred_points(labels_path):
    """Read predicted points dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `PredictedPoint` objects.
    """
    pred_points = read_hdf5(labels_path, "pred_points")
    return pred_points


def read_instances(labels_path):
    """Read instances dataset in a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Instance` objects.
    """
    skeleton = read_skeleton(labels_path)
    tracks = read_tracks(labels_path)
    instances = read_hdf5(labels_path, "instances")
    default_points = read_points(labels_path)
    pred_points = read_pred_points(labels_path)
    instance_objects = []
    for idx, instance in enumerate(instances):
        if instance["instance_type"] == 0:  # Normal Instance
            instance_objects.append(
                Instance.from_pointsarray(
                    skeleton=skeleton,
                    track=tracks[instance["track"]],
                    points=np.array(
                        default_points[
                            instance["point_id_start"] : instance["point_id_end"]
                        ]
                    ),
                )
            )
        if instance["instance_type"] == 1:  # Predicted Instance
            instance_objects.append(
                PredictedInstance.from_instance(
                    instance=Instance.from_pointsarray(
                        skeleton=skeleton,
                        track=tracks[instance["track"]],
                        points=np.array(
                            pred_points[
                                instance["point_id_start"] : instance["point_id_end"]
                            ]
                        ),
                    ),
                    score=instance["score"],
                    tracking_score=instance["tracking_score"],
                )
            )
    return instance_objects
