"""Tests for functions in the sleap_io.io.labelstudio file."""

import numpy as np

from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
from sleap_io.io.labelstudio import (
    convert_labels,
    parse_tasks,
    read_labels,
    write_labels,
)
from sleap_io.io.slp import read_labels as slp_read_labels


def round_trip_labels(labels: Labels) -> Labels:
    ls_labels = parse_tasks(convert_labels(labels), labels.skeletons[0])
    return ls_labels


def test_labels_round_trip(slp_typical, slp_simple_skel, slp_minimal):
    """Test `read_labels` can read different types of .slp files."""
    # first on `slp_typical`
    labels = slp_read_labels(slp_typical)
    assert type(labels) is Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) is Labels

    # now on `slp_simple_skel`
    labels = slp_read_labels(slp_simple_skel)
    assert type(labels) is Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) is Labels

    # now on `slp_minimal`
    labels = slp_read_labels(slp_minimal)
    assert type(labels) is Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) is Labels


def test_read_labels(ls_multianimal):
    file_path, skeleton = ls_multianimal

    ls_labels = read_labels(file_path, skeleton)
    _ = round_trip_labels(ls_labels)
    # assert ls_labels == rt_labels # TODO(TP): Fix equality check


def test_write_labels_with_nan_points(tmp_path):
    """Test that instances with NaN points can be written without errors."""
    # Create a skeleton
    skeleton = Skeleton(["node1", "node2", "node3"])

    # Create a video with known dimensions
    video = Video(filename="test_video.mp4")
    video.backend_metadata["shape"] = (10, 100, 100, 3)

    # Create an instance with some NaN points
    points_array = np.array(
        [
            [10.0, 20.0],  # node1: valid point
            [np.nan, np.nan],  # node2: missing point
            [30.0, 40.0],  # node3: valid point
        ]
    )
    instance = Instance.from_numpy(points_array, skeleton=skeleton)

    # Create a labeled frame
    labeled_frame = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels(labeled_frames=[labeled_frame])

    # Write to file - this should not raise an error
    output_path = tmp_path / "test_output.json"
    write_labels(labels, str(output_path))

    # Verify the file was created
    assert output_path.exists()

    # Read back and verify only valid points are present
    ls_dicts = convert_labels(labels)
    assert len(ls_dicts) == 1

    # Count the keypoint annotations (should be 2, not 3, since one has NaN)
    keypoint_annots = [
        annot
        for annot in ls_dicts[0]["annotations"][0]["result"]
        if annot["type"] == "keypointlabels"
    ]
    assert len(keypoint_annots) == 2  # Only node1 and node3, node2 was skipped
