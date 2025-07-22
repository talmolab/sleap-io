"""Fixtures that return paths to Ultralytics YOLO pose dataset."""

import pytest
from pathlib import Path
from sleap_io import Skeleton, Node, Edge


@pytest.fixture
def ultralytics_dataset():
    """Minimal Ultralytics YOLO pose dataset for testing."""
    return "tests/data/ultralytics"


@pytest.fixture
def ultralytics_skeleton():
    """Skeleton corresponding to the test Ultralytics dataset."""
    nodes = [
        Node("head"),
        Node("neck"),
        Node("center"),
        Node("tail_base"),
        Node("tail_tip"),
    ]
    edges = [
        Edge(nodes[0], nodes[1]),  # head to neck
        Edge(nodes[1], nodes[2]),  # neck to center
        Edge(nodes[2], nodes[3]),  # center to tail_base
        Edge(nodes[2], nodes[4]),  # center to tail_tip
    ]
    return Skeleton(nodes, edges, name="test_animal")


@pytest.fixture
def ultralytics_data_yaml():
    """Path to the data.yaml configuration file."""
    return "tests/data/ultralytics/data.yaml"
