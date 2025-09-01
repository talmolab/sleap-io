"""Test AlphaTracker I/O."""

from pathlib import Path

from sleap_io import load_file
from sleap_io.io.alphatracker import read_labels


def test_load_alphatracker(alphatracker_testdata):
    """Test loading AlphaTracker format."""
    # Load using the direct reader
    labels = read_labels(alphatracker_testdata)

    # Get labeled frames
    lfs = labels.labeled_frames

    # Ensure video and frames are read correctly
    assert len(lfs) == 4
    for file_idx, file in enumerate(labels.video.backend.filename):
        f = Path(file)
        assert f.stem == f"img00{file_idx}"

    # Ensure nodes are read correctly
    nodes = labels.skeleton.node_names
    assert nodes[0] == "1"
    assert nodes[1] == "2"
    assert nodes[2] == "3"

    # Ensure points are read correctly
    for lf_idx, lf in enumerate(lfs):
        assert len(lf.instances) == 2
        for inst_idx, inst in enumerate(lf.instances):
            for point_idx, point in enumerate(inst.points):
                assert point["xy"][0] == ((lf_idx + 1) * (inst_idx + 1))
                assert point["xy"][1] == (point_idx + 2)


def test_load_file_alphatracker(alphatracker_testdata):
    """Test loading AlphaTracker format through load_file."""
    # Load using the generic load_file function
    labels = load_file(alphatracker_testdata)

    # Get labeled frames
    lfs = labels.labeled_frames

    # Ensure video and frames are read correctly
    assert len(lfs) == 4
    for file_idx, file in enumerate(labels.video.backend.filename):
        f = Path(file)
        assert f.stem == f"img00{file_idx}"

    # Ensure nodes are read correctly
    nodes = labels.skeleton.node_names
    assert nodes[0] == "1"
    assert nodes[1] == "2"
    assert nodes[2] == "3"

    # Ensure points are read correctly
    for lf_idx, lf in enumerate(lfs):
        assert len(lf.instances) == 2
        for inst_idx, inst in enumerate(lf.instances):
            for point_idx, point in enumerate(inst.points):
                assert point["xy"][0] == ((lf_idx + 1) * (inst_idx + 1))
                assert point["xy"][1] == (point_idx + 2)


def test_load_file_alphatracker_with_format(alphatracker_testdata):
    """Test loading AlphaTracker format with explicit format parameter."""
    # Load with explicit format
    labels = load_file(alphatracker_testdata, format="alphatracker")

    # Get labeled frames
    lfs = labels.labeled_frames

    # Basic checks
    assert len(lfs) == 4
    assert len(labels.skeleton.nodes) == 3


def test_detect_alphatracker_format_edge_cases(tmp_path):
    """Test AlphaTracker format detection edge cases."""
    import json

    from sleap_io.io.main import _detect_alphatracker_format

    # Test empty list
    empty_json = tmp_path / "empty.json"
    with open(empty_json, "w") as f:
        json.dump([], f)
    assert not _detect_alphatracker_format(str(empty_json))

    # Test non-dict first frame
    non_dict_json = tmp_path / "non_dict.json"
    with open(non_dict_json, "w") as f:
        json.dump(["not a dict"], f)
    assert not _detect_alphatracker_format(str(non_dict_json))

    # Test annotations not a list
    bad_annotations_json = tmp_path / "bad_annotations.json"
    with open(bad_annotations_json, "w") as f:
        json.dump(
            [{"filename": "test.png", "class": "image", "annotations": "not a list"}], f
        )
    assert not _detect_alphatracker_format(str(bad_annotations_json))

    # Test malformed JSON file
    malformed_json = tmp_path / "malformed.json"
    with open(malformed_json, "w") as f:
        f.write("{broken json")
    assert not _detect_alphatracker_format(str(malformed_json))

    # Test non-existent file
    assert not _detect_alphatracker_format(str(tmp_path / "nonexistent.json"))


def test_alphatracker_with_extra_annotations(tmp_path):
    """Test AlphaTracker reader with extra non-Face/non-point annotations."""
    import json

    from sleap_io.io.alphatracker import read_labels

    # Create test data with extra annotation types
    test_data = [
        {
            "filename": "test.png",
            "class": "image",
            "annotations": [
                {"class": "Face", "x": 10, "y": 10, "width": 5, "height": 5},
                {"class": "point", "x": 1, "y": 2},
                {"class": "point", "x": 3, "y": 4},
                {"class": "extra_annotation", "data": "ignored"},  # Should be skipped
                {"class": "Face", "x": 20, "y": 20, "width": 5, "height": 5},
                {"class": "point", "x": 5, "y": 6},
                {"class": "point", "x": 7, "y": 8},
                {"class": "another_type", "value": 123},  # Should also be skipped
            ],
        }
    ]

    # Write test data to file
    test_json = tmp_path / "test_extra.json"
    with open(test_json, "w") as f:
        json.dump(test_data, f)

    # Create dummy image file
    img_path = tmp_path / "test.png"
    img_path.touch()

    # Read the labels
    labels = read_labels(str(test_json))

    # Should have 1 frame with 2 instances
    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 2

    # Check first instance points
    inst1 = labels.labeled_frames[0].instances[0]
    assert inst1.points["xy"][0, 0] == 1
    assert inst1.points["xy"][0, 1] == 2
    assert inst1.points["xy"][1, 0] == 3
    assert inst1.points["xy"][1, 1] == 4

    # Check second instance points
    inst2 = labels.labeled_frames[0].instances[1]
    assert inst2.points["xy"][0, 0] == 5
    assert inst2.points["xy"][0, 1] == 6
    assert inst2.points["xy"][1, 0] == 7
    assert inst2.points["xy"][1, 1] == 8
