import numpy as np
import h5py
import json
from typing import Optional, Union
import attrs
from sleap_io import (
    Video,
    Skeleton,
    Edge,
    Node,
    Instance,
    LabeledFrame,
    Labels,
    Track,
    Point,
    PredictedInstance,
)
from sleap_io.io.utils import read_hdf5


def instance_from_numpy(
    points: np.ndarray, skeleton: Skeleton, track: Optional[Track] = None
) -> Instance:
    """Create an `Instance` from an array of points.

    Args:
        points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
            contains the points in (x, y) coordinates of each `Node`. Missing
            `Node` objects should be represented as `NaN`.
        skeleton: A `Skeleton` object with `n_nodes` `Node` objects to associate with
            the `Instance`.
        track: Optional `Track` object to associate with the `Instance`.

    Returns:
        A new `Instance` object.
    """
    predicted_points = dict()
    node_names: list[str] = [node.name for node in skeleton.nodes]

    for point, node_name in zip(points, node_names):
        if (len(point)) == 4:
            predicted_points[node_name] = Point(
                x=point[0],
                y=point[1],
                visible=bool(point[2]),
                complete=bool(point[3]),
            )
        else:
            predicted_points[node_name] = Point(x=point[0], y=point[1])

    return Instance(points=predicted_points, skeleton=skeleton, track=track)


def predicted_from_instance(
    instance: Instance, score: float, tracking_score: float = 0.0
) -> PredictedInstance:
    """Create a `PredictedInstance` from an `Instance`.

    The fields are copied in a shallow manner with the exception of points. For each
    `Point` in the `Instance` a `PredictedPoint` is created with score set to default
    value.

    Args:
        instance: The `Instance` object to shallow copy data from.
        score: The score for this `Instance`.

    Returns:
        A `PredictedInstance` for the given `Instance`.
    """
    kw_args = attrs.asdict(
        instance,
        recurse=False,
    )
    kw_args["score"] = score
    kw_args["tracking_score"] = tracking_score
    return PredictedInstance(**kw_args)


def read_videos(labels_path: str) -> list[Video]:
    """Read `Video` dataset in a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Video` objects.
    """

    # TODO (DS) - Find shape of video

    videos = [json.loads(x) for x in read_hdf5(labels_path, "videos_json")]
    video_objects = []
    for video in videos:
        video_objects.append(Video(filename=video["backend"]["filename"]))
    return video_objects


def read_tracks(labels_path: str) -> list[Track]:
    """Read `Track` dataset in a SLEAP labels file.

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


def read_metadata(labels_path: str) -> dict:
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


def read_skeleton(labels_path: str) -> list[Skeleton]:
    """Read `Skeleton` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file.

    Returns:
        A `Skeleton` object.
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


def read_points(labels_path: str) -> np.ndarray:
    """Read `Point` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file.

    Returns:
        A list of `Point` objects.
    """
    points = read_hdf5(labels_path, "points")
    return points


def read_pred_points(labels_path: str) -> np.ndarray:
    """Read `PredictedPoint` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `PredictedPoint` objects.
    """
    pred_points = read_hdf5(labels_path, "pred_points")
    return pred_points


def read_instances(labels_path: str) -> list[Union[Instance, PredictedInstance]]:
    """Read `Instance` dataset in a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file

    Returns:
        A list of `Instance` objects.
    """
    skeleton = read_skeleton(labels_path)[0]
    # TODO (DS) - Support multi-skeleton
    tracks = read_tracks(labels_path)
    instances = read_hdf5(labels_path, "instances")
    default_points = read_points(labels_path)
    pred_points = read_pred_points(labels_path)
    instance_objects = []
    for idx, instance in enumerate(instances):

        # Normal Instance
        if instance["instance_type"] == 0:
            tracks_default = tracks[instance["track"]] if len(tracks) > 0 else None
            instance_objects.append(
                # Params for creating `Instance` from `numpy.array`
                instance_from_numpy(
                    skeleton=skeleton,
                    track=tracks_default,
                    points=np.array(
                        default_points[
                            instance["point_id_start"] : instance["point_id_end"]
                        ]
                    ),
                )
            )

        # Predicted Instance
        if instance["instance_type"] == 1:
            instance_objects.append(
                predicted_from_instance(
                    # Params for creating `PredictedInstance` from `Instance`
                    instance=instance_from_numpy(
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


def read_labels(filepath: str) -> Labels:
    """Read a SLEAP labels file.

    Args:
        filepath: Path to a SLEAP-formatted labels file (.slp).

    Returns:
        The processed `Labels` object.
    """
    pass
