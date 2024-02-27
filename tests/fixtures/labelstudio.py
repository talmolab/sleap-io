"""Fixtures that return paths to label-studio .json files."""

import pytest
from sleap_io import Skeleton, Node, Edge


@pytest.fixture
def ls_multianimal():
    """Typical label studio file from a multi-animal DLC project (mixes multi-animal
    bodyparts and unique bodyparts."""
    nodes = [
        Node("pup_snout"),
        Node("pup_neck"),
        Node("pup_body"),
        Node("pup_tailbase"),
    ]
    edges = [
        Edge(nodes[0], nodes[1]),
        Edge(nodes[1], nodes[2]),
        Edge(nodes[2], nodes[3]),
    ]
    return (
        "tests/data/labelstudio/multi_animal_from_dlc.json",
        Skeleton(nodes, edges, name="pups"),
    )
