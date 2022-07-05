from pandas import read_hdf
from sleap_io import (
    instance_from_numpy,
    predicted_from_instance,
    read_hdf5,
    read_videos,
    read_skeleton,
    read_tracks,
    read_instances,
    read_metadata,
    read_points,
    read_pred_points,
    Video,
    Skeleton,
    Edge,
    Node,
    Instance,
    LabeledFrame,
    Track,
    Point,
    PredictedPoint,
    PredictedInstance,
)
import numpy as np
import h5py
from typing import List
import json


def test_read(slp_file1):

    skeleton = Skeleton(
        nodes=[Node("head"), Node("thorax"), Node("abdomen")],
        edges=[
            Edge(source=Node("head"), destination=Node("thorax")),
            Edge(source=Node("thorax"), destination=Node("abdomen")),
        ],
    )
    instance1 = Instance(skeleton=skeleton, points={"head": Point(x=0, y=0)})
    numpy_array1 = np.array([[1, 1], [2, 2], [3, 3]], dtype="float32")
    numpy_array2 = np.array(
        [[1, 1, True, False], [2, 2, True, False], [3, 3, True, False]], dtype="float32"
    )
    instance2 = instance_from_numpy(points=numpy_array1, skeleton=skeleton)
    instance3 = instance_from_numpy(points=numpy_array2, skeleton=skeleton)

    # Instances & HDF5
    assert type(predicted_from_instance(instance1, 0.0)) == PredictedInstance
    assert type(instance2) == Instance
    assert type(instance3) == Instance
    assert type(read_hdf5(slp_file1)) == dict

    # Videos
    assert type(read_videos(slp_file1)) == list
    if len(read_videos(slp_file1)) >= 1:
        assert type(read_videos(slp_file1)[0]) == Video

    # Tracks & Metadata
    assert type(read_tracks(slp_file1)) == list
    if len(read_tracks(slp_file1)) >= 1:
        assert type(read_tracks(slp_file1)[0]) == Track
    assert type(read_metadata(slp_file1)) == dict

    # Skeleton
    assert type(read_skeleton(slp_file1)) == list
    if len(read_skeleton(slp_file1)) >= 1:
        assert type(read_skeleton(slp_file1)[0]) == Skeleton
    if len(read_skeleton(slp_file1)) >= 1:
        assert type(read_skeleton(slp_file1)[0]) == Skeleton

    # Points & PredictedPoints
    assert type(read_points(slp_file1)) == np.ndarray
    if len(read_points(slp_file1)) >= 1:
        assert type(read_points(slp_file1)[0]) == np.void
    assert type(read_pred_points(slp_file1)) == np.ndarray
    if len(read_pred_points(slp_file1)) >= 1:
        assert type(read_pred_points(slp_file1)[0]) == np.void

    # Instances
    assert type(read_instances(slp_file1)) == list
    if len(read_instances(slp_file1)) >= 1:
        assert type(read_instances(slp_file1)[0]) == Instance
        assert type(read_instances(slp_file1)[2]) == PredictedInstance
