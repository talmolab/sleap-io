"""Tests for functions in the sleap_io.io.jabs file."""

import h5py
import numpy as np
import pytest

from sleap_io import Labels, PredictedInstance, Skeleton
from sleap_io.io.jabs import (
    JABS_DEFAULT_SKELETON,
    _roi_to_static_object_coords,
    _static_object_to_roi,
    convert_labels,
    get_max_ids_in_video,
    make_simple_skeleton,
    prediction_to_instance,
    read_labels,
    tracklets_to_v3,
)
from sleap_io.model.roi import AnnotationType


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
    """Test `get_max_ids_in_video`."""
    with h5py.File(jabs_real_data_v5, "r") as f:
        max_ids = max(f["poseest/instance_count"][:])

    labels = read_labels(jabs_real_data_v5)

    found_max_ids = get_max_ids_in_video(labels.labeled_frames)
    assert max_ids == found_max_ids


def test_prediction_to_instance(jabs_real_data_v5):
    """Test that prediction_to_instance creates PredictedInstance with scores."""
    with h5py.File(jabs_real_data_v5, "r") as f:
        pose = f["poseest/points"][0, 0, :, :]
        confidence = f["poseest/confidence"][0, 0, :]

    inst = prediction_to_instance(pose, confidence, JABS_DEFAULT_SKELETON)
    assert type(inst) is PredictedInstance
    assert inst.score > 0

    # Check per-point scores match confidence values
    visible = confidence > 0
    point_scores = inst.points["score"][visible]
    np.testing.assert_array_almost_equal(point_scores, confidence[visible])


def test_prediction_to_instance_no_confidence():
    """Test that zero confidence returns None."""
    pose = np.zeros((12, 2), dtype=np.uint16)
    confidence = np.zeros(12, dtype=np.float32)
    result = prediction_to_instance(pose, confidence, JABS_DEFAULT_SKELETON)
    assert result is None


def test_make_simple_skeleton():
    skeleton = make_simple_skeleton("test", 2)
    assert len(skeleton.nodes) == 2
    assert len(skeleton.edges) == 1
    assert skeleton.name == "test"


def test_read_labels_predicted_instances(jabs_real_data_v5):
    """Test that JABS reader creates PredictedInstance objects."""
    labels = read_labels(jabs_real_data_v5)

    for lf in labels.labeled_frames:
        for inst in lf.instances:
            assert type(inst) is PredictedInstance
            assert inst.score > 0


def test_read_labels_static_objects_as_rois(jabs_real_data_v5):
    """Test that JABS v5 static objects are loaded as ROIs."""
    labels = read_labels(jabs_real_data_v5)

    # Only the Mouse skeleton should be present (no synthetic skeletons)
    assert len(labels.skeletons) == 1
    assert labels.skeletons[0].name == "Mouse"

    # The v5 test fixture has one static object: corners
    assert len(labels.rois) == 1
    roi = labels.rois[0]
    assert roi.name == "corners"
    assert roi.annotation_type == AnnotationType.ARENA
    assert roi.category == "static_object"
    assert roi.source == "jabs"
    assert roi.frame_idx is None  # Static ROI
    assert roi.video == labels.videos[0]


def test_jabs_roundtrip_preserves_static_objects(jabs_real_data_v5):
    """Test that static objects survive the JABS read/convert round-trip."""
    # Read original coordinates
    with h5py.File(jabs_real_data_v5, "r") as f:
        original_corners = f["static_objects/corners"][:]

    # Read through sleap-io and convert back
    labels = read_labels(jabs_real_data_v5)
    converted = convert_labels(labels, labels.videos[0])

    assert "corners" in converted["static_objects"]
    roundtrip_corners = converted["static_objects"]["corners"]
    np.testing.assert_array_equal(roundtrip_corners, original_corners)


def test_jabs_roundtrip_preserves_confidence(jabs_real_data_v5):
    """Test that per-point confidence scores are preserved through round-trip."""
    labels = read_labels(jabs_real_data_v5)

    # Verify instances have per-point scores from the original data
    lf = labels.labeled_frames[0]
    inst = lf.instances[0]
    assert type(inst) is PredictedInstance

    # Scores should reflect original confidence (not all 1.0)
    scores = inst.points["score"]
    visible = inst.points["visible"]
    assert np.all(scores[visible] > 0)

    # Convert back and verify confidence is preserved (not hardcoded to 1.0)
    converted = convert_labels(labels, labels.videos[0])
    conf = converted["confidence"]
    assert conf.shape[0] >= len(labels.labeled_frames)
    # Points with 0 confidence in the original should still be 0
    assert np.any(conf == 0)


def test_static_object_to_roi_single_point():
    """Test ROI creation for single-point static objects (e.g., lixit)."""
    from shapely.geometry import Point

    from sleap_io.model.video import Video

    video = Video.from_filename("test.mp4")
    coords = np.array([[100.0, 200.0]])
    roi = _static_object_to_roi("lixit", coords, video)

    assert roi.name == "lixit"
    assert roi.annotation_type == AnnotationType.ANCHOR
    assert isinstance(roi.geometry, Point)
    assert roi.geometry.x == 100.0
    assert roi.geometry.y == 200.0

    # Round-trip back to coordinates
    result = _roi_to_static_object_coords(roi)
    np.testing.assert_array_equal(result, coords)


def test_roi_to_static_object_coords_unsupported_geometry():
    """Test that unsupported geometry types return None."""
    from shapely.geometry import LineString

    from sleap_io.model.roi import ROI

    roi = ROI(geometry=LineString([(0, 0), (1, 1)]), name="line")
    result = _roi_to_static_object_coords(roi)
    assert result is None
