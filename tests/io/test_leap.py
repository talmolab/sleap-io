"""Tests for LEAP .mat file I/O."""

import numpy as np
import pytest

import sleap_io as sio
from sleap_io.io import leap
from sleap_io.model.instance import Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


def test_load_leap_mat(leap_labels_mat):
    """Test loading LEAP .mat file."""
    # Check that pymatreader is available
    pytest.importorskip("pymatreader")

    # Load the file
    labels = leap.read_labels(leap_labels_mat)

    # Check that we got a Labels object
    assert isinstance(labels, Labels)

    # Check that we have some data
    assert len(labels.labeled_frames) > 0
    assert len(labels.videos) > 0
    assert len(labels.skeletons) > 0

    # Check skeleton structure
    skeleton = labels.skeletons[0]
    assert isinstance(skeleton, Skeleton)
    assert len(skeleton.nodes) > 0

    # Check labeled frames
    for lf in labels.labeled_frames:
        assert isinstance(lf, LabeledFrame)
        assert isinstance(lf.video, Video)
        assert len(lf.instances) > 0

        # Check instances
        for inst in lf.instances:
            assert isinstance(inst, Instance)
            assert inst.skeleton == skeleton
            assert len(inst.points) > 0

            # Check points in the structured array
            for point_data in inst.points:
                assert "xy" in point_data.dtype.names
                assert point_data["xy"].shape == (2,)


def test_load_leap_via_load_file(leap_labels_mat):
    """Test loading LEAP .mat file via load_file."""
    pytest.importorskip("pymatreader")

    # Load via load_file
    labels = sio.load_file(leap_labels_mat)

    # Check that we got a Labels object
    assert isinstance(labels, Labels)
    assert len(labels.labeled_frames) > 0


def test_load_leap_via_load_leap(leap_labels_mat):
    """Test loading LEAP .mat file via load_leap."""
    pytest.importorskip("pymatreader")

    # Load via load_leap
    labels = sio.load_leap(leap_labels_mat)

    # Check that we got a Labels object
    assert isinstance(labels, Labels)
    assert len(labels.labeled_frames) > 0


def test_load_leap_with_custom_skeleton(leap_labels_mat):
    """Test loading LEAP .mat file with custom skeleton."""
    pytest.importorskip("pymatreader")

    # Create a custom skeleton
    from sleap_io.model.skeleton import Edge, Node

    nodes = [Node("A"), Node("B"), Node("C")]
    edges = [Edge(nodes[0], nodes[1]), Edge(nodes[1], nodes[2])]
    custom_skeleton = Skeleton(nodes=nodes, edges=edges)

    # Load with custom skeleton
    labels = leap.read_labels(leap_labels_mat, skeleton=custom_skeleton)

    # Check that custom skeleton was used
    assert labels.skeletons[0] == custom_skeleton

    # Check instances use the custom skeleton
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            assert inst.skeleton == custom_skeleton


def test_leap_missing_pymatreader(leap_labels_mat, monkeypatch):
    """Test error when pymatreader is not installed."""

    # Mock ImportError for pymatreader
    def mock_import(name, *args, **kwargs):
        if name == "pymatreader":
            raise ImportError("No module named 'pymatreader'")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    # Should raise ImportError with helpful message
    with pytest.raises(ImportError, match="pymatreader is required"):
        leap.read_labels(leap_labels_mat)


def test_parse_skeleton():
    """Test skeleton parsing from LEAP data."""
    # Create mock LEAP data with skeleton
    mat_data = {
        "skeleton": {
            "nodes": ["node1", "node2", "node3"],
            "edges": [[1, 2], [2, 3]],  # MATLAB 1-based indexing
        }
    }

    skeleton = leap._parse_skeleton(mat_data)

    # Check nodes
    assert len(skeleton.nodes) == 3
    assert skeleton.nodes[0].name == "node1"
    assert skeleton.nodes[1].name == "node2"
    assert skeleton.nodes[2].name == "node3"

    # Check edges (converted to 0-based indexing)
    assert len(skeleton.edges) == 2
    assert skeleton.edges[0].source == skeleton.nodes[0]
    assert skeleton.edges[0].destination == skeleton.nodes[1]
    assert skeleton.edges[1].source == skeleton.nodes[1]
    assert skeleton.edges[1].destination == skeleton.nodes[2]


def test_parse_pose_data():
    """Test pose data parsing from LEAP data."""
    from sleap_io.model.skeleton import Node

    # Create skeleton
    nodes = [Node("A"), Node("B")]
    skeleton = Skeleton(nodes=nodes)

    # Create video
    video = Video.from_filename("test.mp4")

    # Create mock LEAP data with positions
    # Shape: (nodes=2, xy=2, frames=3)
    positions = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # node 0: x, y across frames
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # node 1: x, y across frames
        ]
    )
    mat_data = {"positions": positions}

    labeled_frames = leap._parse_pose_data(mat_data, video, skeleton)

    # Check we got 3 frames
    assert len(labeled_frames) == 3

    # Check frame 0
    frame0 = labeled_frames[0]
    assert frame0.frame_idx == 0
    assert len(frame0.instances) == 1
    inst0 = frame0.instances[0]
    # Access points using the structured array
    assert np.allclose(inst0.points[0]["xy"], [1.0, 4.0])
    assert np.allclose(inst0.points[1]["xy"], [7.0, 10.0])

    # Check frame 1
    frame1 = labeled_frames[1]
    assert frame1.frame_idx == 1
    inst1 = frame1.instances[0]
    assert np.allclose(inst1.points[0]["xy"], [2.0, 5.0])
    assert np.allclose(inst1.points[1]["xy"], [8.0, 11.0])

    # Check frame 2
    frame2 = labeled_frames[2]
    assert frame2.frame_idx == 2
    inst2 = frame2.instances[0]
    assert np.allclose(inst2.points[0]["xy"], [3.0, 6.0])
    assert np.allclose(inst2.points[1]["xy"], [9.0, 12.0])


def test_parse_pose_data_alternative_fields():
    """Test parsing pose data from alternative field names."""
    from sleap_io.model.skeleton import Node

    nodes = [Node("A"), Node("B")]
    skeleton = Skeleton(nodes=nodes)
    video = Video.from_filename("test.mp4")

    # Test with 'pose' field - shape (nodes=2, xy=2, frames=1)
    mat_data = {"pose": np.array([[[1.0], [2.0]], [[3.0], [4.0]]])}
    labeled_frames = leap._parse_pose_data(mat_data, video, skeleton)
    assert len(labeled_frames) == 1

    # Test with 'posedata' field
    mat_data = {"posedata": np.array([[[1.0], [2.0]], [[3.0], [4.0]]])}
    labeled_frames = leap._parse_pose_data(mat_data, video, skeleton)
    assert len(labeled_frames) == 1

    # Test with no position data
    mat_data = {}
    labeled_frames = leap._parse_pose_data(mat_data, video, skeleton)
    assert len(labeled_frames) == 0


def test_parse_pose_data_single_frame():
    """Test parsing single frame (2D array) pose data."""
    from sleap_io.model.skeleton import Node

    nodes = [Node("A"), Node("B")]
    skeleton = Skeleton(nodes=nodes)
    video = Video.from_filename("test.mp4")

    # Single frame, shape is (nodes=2, xy=2)
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat_data = {"positions": positions}

    labeled_frames = leap._parse_pose_data(mat_data, video, skeleton)

    # Check we got 1 frame
    assert len(labeled_frames) == 1
    frame0 = labeled_frames[0]
    assert frame0.frame_idx == 0
    inst0 = frame0.instances[0]
    assert np.allclose(inst0.points[0]["xy"], [1.0, 2.0])
    assert np.allclose(inst0.points[1]["xy"], [3.0, 4.0])


def test_parse_skeleton_edge_cases():
    """Test skeleton parsing edge cases."""
    # Test with numpy array node names
    mat_data = {
        "skeleton": {
            "nodes": np.array(["node1", "node2"]),
            "edges": np.array([1, 2]).reshape(-1, 2),  # 1D array that needs reshaping
        }
    }
    skeleton = leap._parse_skeleton(mat_data)
    assert len(skeleton.nodes) == 2
    assert skeleton.nodes[0].name == "node1"
    assert len(skeleton.edges) == 1

    # Test with nested numpy arrays for node names
    mat_data = {
        "skeleton": {
            "nodes": [np.array(["node1"]), np.array(["node2"])],
            "edges": [],
        }
    }
    skeleton = leap._parse_skeleton(mat_data)
    assert len(skeleton.nodes) == 2
    assert skeleton.nodes[0].name == "node1"

    # Test with non-string node names
    mat_data = {
        "skeleton": {
            "nodes": [1, 2, 3],
            "edges": [],
        }
    }
    skeleton = leap._parse_skeleton(mat_data)
    assert len(skeleton.nodes) == 3
    assert skeleton.nodes[0].name == "1"
    assert skeleton.nodes[1].name == "2"

    # Test with 1D numpy array edges that need reshaping
    mat_data = {
        "skeleton": {
            "nodes": ["A", "B", "C", "D"],
            "edges": np.array(
                [1, 2, 2, 3, 3, 4]
            ),  # 1D array: edges (1,2), (2,3), (3,4)
        }
    }
    skeleton = leap._parse_skeleton(mat_data)
    assert len(skeleton.nodes) == 4
    assert len(skeleton.edges) == 3
    # Check the edges are correctly parsed (remember 1-based to 0-based conversion)
    assert skeleton.edges[0].source == skeleton.nodes[0]  # A
    assert skeleton.edges[0].destination == skeleton.nodes[1]  # B
    assert skeleton.edges[1].source == skeleton.nodes[1]  # B
    assert skeleton.edges[1].destination == skeleton.nodes[2]  # C
    assert skeleton.edges[2].source == skeleton.nodes[2]  # C
    assert skeleton.edges[2].destination == skeleton.nodes[3]  # D


def test_load_leap_no_video_path(tmp_path, monkeypatch):
    """Test loading LEAP file without boxPath field."""
    pytest.importorskip("pymatreader")

    # Create a minimal .mat file without boxPath
    mat_file = tmp_path / "test.mat"
    mp4_file = tmp_path / "test.mp4"

    # Create a dummy video file
    mp4_file.write_text("dummy")

    # Mock pymatreader to return data without boxPath
    def mock_read_mat(path):
        return {
            "skeleton": {"nodes": ["A", "B"], "edges": []},
            "positions": np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
        }

    monkeypatch.setattr("pymatreader.read_mat", mock_read_mat)

    # This should try to infer video path from mat file path
    labels = leap.read_labels(str(mat_file))
    assert isinstance(labels, Labels)
    assert len(labels.videos) == 1
