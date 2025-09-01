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
