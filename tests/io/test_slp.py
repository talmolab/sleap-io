"""Tests for functions in the sleap_io.io.slp file."""
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
    """Test `read_labels` can read different types of .slp files."""
    labels = read_labels(slp_typical)
    assert type(labels) == Labels

    labels = read_labels(slp_simple_skel)
    assert type(labels) == Labels

    labels = read_labels(slp_minimal)
    assert type(labels) == Labels


def test_load_slp_with_provenance(slp_predictions_with_provenance):
    labels = read_labels(slp_predictions_with_provenance)
    provenance = labels.provenance
    assert type(provenance) == dict
    assert provenance["sleap_version"] == "1.2.7"


def test_read_instances_from_predicted(slp_real_data):
    labels = read_labels(slp_real_data)

    lf = labels.find(video=labels.video, frame_idx=220)[0]
    assert len(lf) == 3
    assert type(lf.instances[0]) == PredictedInstance
    assert type(lf.instances[1]) == PredictedInstance
    assert type(lf.instances[2]) == Instance
    assert lf.instances[2].from_predicted == lf.instances[1]
    assert lf.unused_predictions == [lf.instances[0]]

    lf = labels.find(video=labels.video, frame_idx=770)[0]
    assert len(lf) == 4
    assert type(lf.instances[0]) == PredictedInstance
    assert type(lf.instances[1]) == PredictedInstance
    assert type(lf.instances[2]) == Instance
    assert type(lf.instances[3]) == Instance
    assert lf.instances[2].from_predicted == lf.instances[1]
    assert lf.instances[3].from_predicted == lf.instances[0]
    assert len(lf.unused_predictions) == 0


def test_read_videos_pkg(slp_minimal_pkg):
    videos = read_videos(slp_minimal_pkg)
    assert len(videos) == 1
    video = videos[0]
    assert video.shape == (1, 384, 384, 1)
    assert video.backend.dataset == "video0/video"
