from pandas import read_hdf
from sleap_io import (
    from_pointsarray,
    from_instance,
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
from typing import List


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
instance2 = from_pointsarray(points=numpy_array1, skeleton=skeleton)
instance3 = from_pointsarray(points=numpy_array2, skeleton=skeleton)


def test_read(slp_file):
    assert type(from_instance(instance1, 0.0)) == PredictedInstance
    assert type(instance2) == Instance
    assert type(instance3) == Instance
    assert type(read_hdf5(slp_file)) == dict
    assert type(read_videos(slp_file)) == list
    if len(read_videos(slp_file)) >= 1:
        assert type(read_videos(slp_file)[0]) == Video
    assert type(read_tracks(slp_file)) == list
    if len(read_tracks(slp_file)) >= 1:
        assert type(read_tracks(slp_file)[0]) == Track
    assert type(read_metadata(slp_file)) == dict
    assert type(read_skeleton(slp_file)) == Skeleton
    assert type(read_points(slp_file)) == np.ndarray
    if len(read_points(slp_file)) >= 1:
        assert type(read_points(slp_file)[0]) == np.void
    assert type(read_pred_points(slp_file)) == np.ndarray
    if len(read_pred_points(slp_file)) >= 1:
        assert type(read_pred_points(slp_file)[0]) == np.void
    assert type(read_instances(slp_file)) == list
    if len(read_instances(slp_file)) >= 1:
        assert type(read_instances(slp_file)[0]) == Instance or PredictedInstance
