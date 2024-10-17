"""Tests for functions in the sleap_io.io.jabs file."""

import numpy as np
import h5py
import pytest
from sleap_io import Labels, Instance, Skeleton
from sleap_io.io.jabs import (
    read_labels,
    convert_labels,
    tracklets_to_v3,
    get_max_ids_in_video,
    prediction_to_instance,
    make_simple_skeleton,
    JABS_DEFAULT_SKELETON,
)


def test_label_conversions(jabs_real_data_v5):
    """Tests that `read_labels` and `convert_labels` can run on test data.

    Does not validate data was transformed correctly.

    """
    slp_labels = read_labels(jabs_real_data_v5)
    assert isinstance(slp_labels, Labels)

    jabs_labels = convert_labels(slp_labels, slp_labels.videos[0])
    assert isinstance(jabs_labels, dict)

    slp_labels.skeletons.append(Skeleton(["a", "b"], name="test"))
    jabs_labels = convert_labels(slp_labels, slp_labels.videos[0])
    assert isinstance(jabs_labels, dict)

    slp_labels.skeletons.pop(0)
    with pytest.raises(ValueError):
        convert_labels(slp_labels, slp_labels.videos[0])


def test_tracklets_to_v3(jabs_real_data_v5):
    """Test `tracklets_to_v3` meets v3 criteria.

    Args:
        jabs_real_data_v5: A JABs v4+ file
    """
    with h5py.File(jabs_real_data_v5, "r") as f:
        original_tracklets = f["poseest/instance_embed_id"][...]

    adjusted_tracklets = tracklets_to_v3(original_tracklets)

    # Criteria 1: tracklets are 0-indexed
    valid_ids = original_tracklets != 0
    masked_ids = np.ma.array(adjusted_tracklets, mask=~valid_ids)
    id_values = np.unique(masked_ids.compressed())
    assert np.all(id_values == range(len(id_values)))

    last_id_first_frame = 0
    for current_identity in id_values:
        print(current_identity)
        frames_detected, _ = np.where(masked_ids == current_identity)
        first_frame_detected = frames_detected[0]
        # Criteria 2: tracklets appear in ascending order
        assert last_id_first_frame <= first_frame_detected
        last_id_first_frame = first_frame_detected

        # Criteria 3: tracklets are all continuous in time
        assert len(frames_detected) == (frames_detected[-1] - frames_detected[0] + 1)


def test_get_max_ids_in_video(jabs_real_data_v5):
    """Test `get_max_ids_in_video`"""
    with h5py.File(jabs_real_data_v5, "r") as f:
        max_ids = max(f["poseest/instance_count"][:])

    labels = read_labels(jabs_real_data_v5)

    found_max_ids = get_max_ids_in_video(labels.labeled_frames)
    assert max_ids == found_max_ids


def test_prediction_to_instance(jabs_real_data_v5):
    with h5py.File(jabs_real_data_v5, "r") as f:
        pose = f["poseest/points"][0, 0, :, :]
        confidence = f["poseest/confidence"][0, 0, :]

    label = prediction_to_instance(pose, confidence, JABS_DEFAULT_SKELETON)
    assert type(label) == Instance


def test_make_simple_skeleton():
    skeleton = make_simple_skeleton("test", 2)
    assert len(skeleton.nodes) == 2
    assert len(skeleton.edges) == 1
    assert skeleton.name == "test"
