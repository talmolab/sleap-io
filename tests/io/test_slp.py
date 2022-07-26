from sleap_io import (
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
    Labels,
)
from sleap_io.io.slp import (
    read_videos,
    read_skeletons,
    read_tracks,
    read_instances,
    read_metadata,
    read_points,
    read_pred_points,
    read_instances,
    read_labels,
)
import numpy as np


def test_read_labels(slp_typical, slp_simple_skel, slp_minimal):
    labels = read_labels(slp_typical)
    assert type(labels) == Labels

    labels = read_labels(slp_simple_skel)
    assert type(labels) == Labels

    labels = read_labels(slp_minimal)
    assert type(labels) == Labels
