"""Tests for standalone skeleton JSON I/O."""

import pytest
import json
import tempfile
from pathlib import Path
import sleap_io as sio
from sleap_io.io.skeleton import SkeletonDecoder, SkeletonEncoder


# Basic decoder tests
def test_decode_simple_skeleton():
    """Test decoding a simple skeleton with one edge."""
    json_data = {
        "directed": True,
        "graph": {"name": "test-skeleton", "num_edges_inserted": 1},
        "links": [
            {
                "edge_insert_idx": 0,
                "key": 0,
                "source": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["node1", 1.0]},
                },
                "target": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["node2", 1.0]},
                },
                "type": {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [1]},
                    ]
                },
            }
        ],
        "multigraph": True,
        "nodes": [{"id": {"py/id": 1}}, {"id": {"py/id": 2}}],
    }

    decoder = SkeletonDecoder()
    skeleton = decoder.decode(json_data)

    assert skeleton.name == "test-skeleton"
    assert len(skeleton.nodes) == 2
    assert skeleton.nodes[0].name == "node1"
    assert skeleton.nodes[1].name == "node2"
    assert len(skeleton.edges) == 1
    assert skeleton.edges[0].source.name == "node1"
    assert skeleton.edges[0].destination.name == "node2"


def test_decode_skeleton_with_symmetry():
    """Test decoding a skeleton with symmetry edges."""
    json_data = {
        "directed": True,
        "graph": {"name": "symmetric-skeleton", "num_edges_inserted": 0},
        "links": [
            {
                "key": 0,
                "source": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["left", 1.0]},
                },
                "target": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["right", 1.0]},
                },
                "type": {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [2]},
                    ]
                },
            }
        ],
        "multigraph": True,
        "nodes": [{"id": {"py/id": 1}}, {"id": {"py/id": 2}}],
    }

    decoder = SkeletonDecoder()
    skeleton = decoder.decode(json_data)

    assert len(skeleton.symmetries) == 1
    assert len(skeleton.symmetries[0].nodes) == 2
    node_names = {n.name for n in skeleton.symmetries[0].nodes}
    assert node_names == {"left", "right"}


def test_decode_dict_state_format():
    """Test decoding nodes with dict py/state format."""
    json_data = {
        "directed": True,
        "graph": {"name": "dict-state-skeleton", "num_edges_inserted": 0},
        "links": [
            {
                "edge_insert_idx": 0,
                "key": 0,
                "source": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"name": "node_a", "weight": 2.0},
                },
                "target": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"name": "node_b", "weight": 3.0},
                },
                "type": {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [1]},
                    ]
                },
            }
        ],
        "multigraph": True,
        "nodes": [{"id": {"py/id": 1}}, {"id": {"py/id": 2}}],
    }

    decoder = SkeletonDecoder()
    skeleton = decoder.decode(json_data)

    assert skeleton.nodes[0].name == "node_a"
    assert skeleton.nodes[1].name == "node_b"


def test_decode_from_json_string():
    """Test decoding from JSON string."""
    json_str = (
        """{"directed": true, "graph": {"name": "test"}, "links": [], "nodes": []}"""
    )

    decoder = SkeletonDecoder()
    skeleton = decoder.decode(json_str)

    assert skeleton.name == "test"
    assert len(skeleton.nodes) == 0


def test_decode_multiple_skeletons():
    """Test decoding a list of skeletons."""
    json_data = [
        {"directed": True, "graph": {"name": "skel1"}, "links": [], "nodes": []},
        {"directed": True, "graph": {"name": "skel2"}, "links": [], "nodes": []},
    ]

    decoder = SkeletonDecoder()
    skeletons = decoder.decode(json_data)

    assert isinstance(skeletons, list)
    assert len(skeletons) == 2
    assert skeletons[0].name == "skel1"
    assert skeletons[1].name == "skel2"


# Basic encoder tests
def test_encode_simple_skeleton():
    """Test encoding a simple skeleton."""
    # Create skeleton
    nodes = [sio.Node("A"), sio.Node("B")]
    edges = [sio.Edge(nodes[0], nodes[1])]
    skeleton = sio.Skeleton(nodes=nodes, edges=edges, name="test")

    # Encode
    encoder = SkeletonEncoder()
    json_str = encoder.encode(skeleton)
    data = json.loads(json_str)

    assert data["graph"]["name"] == "test"
    assert data["directed"] == True
    assert data["multigraph"] == True
    assert len(data["links"]) == 1
    assert data["links"][0]["source"]["py/object"] == "sleap.skeleton.Node"
    assert data["links"][0]["source"]["py/state"]["py/tuple"][0] == "A"
    assert data["links"][0]["target"]["py/state"]["py/tuple"][0] == "B"


def test_encode_skeleton_with_symmetry():
    """Test encoding a skeleton with symmetries."""
    # Create skeleton
    left = sio.Node("left_eye")
    right = sio.Node("right_eye")
    symmetry = sio.Symmetry([left, right])
    skeleton = sio.Skeleton(nodes=[left, right], symmetries=[symmetry])

    # Encode
    encoder = SkeletonEncoder()
    json_str = encoder.encode(skeleton)
    data = json.loads(json_str)

    assert len(data["links"]) == 1
    assert data["links"][0]["type"]["py/reduce"][1]["py/tuple"][0] == 2  # Symmetry type


def test_encode_edge_type_references():
    """Test that edge types use py/id after first occurrence."""
    # Create skeleton with multiple edges
    nodes = [sio.Node(f"node{i}") for i in range(4)]
    edges = [
        sio.Edge(nodes[0], nodes[1]),
        sio.Edge(nodes[1], nodes[2]),
        sio.Edge(nodes[2], nodes[3]),
    ]
    skeleton = sio.Skeleton(nodes=nodes, edges=edges)

    # Encode
    encoder = SkeletonEncoder()
    json_str = encoder.encode(skeleton)
    data = json.loads(json_str)

    # First edge should have py/reduce
    assert "py/reduce" in data["links"][0]["type"]
    # Subsequent edges should have py/id
    assert data["links"][1]["type"] == {"py/id": 1}
    assert data["links"][2]["type"] == {"py/id": 1}


def test_encode_multiple_skeletons():
    """Test encoding multiple skeletons."""
    skel1 = sio.Skeleton(nodes=[sio.Node("A")], name="skel1")
    skel2 = sio.Skeleton(nodes=[sio.Node("B")], name="skel2")

    encoder = SkeletonEncoder()
    json_str = encoder.encode([skel1, skel2])
    data = json.loads(json_str)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["graph"]["name"] == "skel1"
    assert data[1]["graph"]["name"] == "skel2"


def test_dictionary_sorting():
    """Test that dictionaries are recursively sorted."""
    skeleton = sio.Skeleton(
        nodes=[sio.Node("A"), sio.Node("B")],
        edges=[sio.Edge(sio.Node("A"), sio.Node("B"))],
        name="test",
    )

    encoder = SkeletonEncoder()
    json_str = encoder.encode(skeleton)

    # Keys should be in sorted order
    data = json.loads(json_str)
    assert list(data.keys()) == sorted(data.keys())
    assert list(data["graph"].keys()) == sorted(data["graph"].keys())


# Round-trip tests
def test_simple_round_trip():
    """Test round-trip with a simple skeleton."""
    # Create skeleton
    nodes = [sio.Node("head"), sio.Node("tail")]
    edges = [sio.Edge(nodes[0], nodes[1])]
    skeleton1 = sio.Skeleton(nodes=nodes, edges=edges, name="test-skeleton")

    # Encode and decode
    encoder = SkeletonEncoder()
    decoder = SkeletonDecoder()
    json_str = encoder.encode(skeleton1)
    skeleton2 = decoder.decode(json_str)

    assert skeleton1.name == skeleton2.name
    assert len(skeleton1.nodes) == len(skeleton2.nodes)
    assert skeleton1.nodes[0].name == skeleton2.nodes[0].name
    assert skeleton1.nodes[1].name == skeleton2.nodes[1].name
    assert len(skeleton1.edges) == len(skeleton2.edges)


def test_complex_round_trip():
    """Test round-trip with edges and symmetries."""
    # Create complex skeleton
    nodes = [sio.Node(name) for name in ["head", "neck", "left_hand", "right_hand"]]
    edges = [
        sio.Edge(nodes[0], nodes[1]),  # head -> neck
        sio.Edge(nodes[1], nodes[2]),  # neck -> left_hand
        sio.Edge(nodes[1], nodes[3]),  # neck -> right_hand
    ]
    symmetries = [sio.Symmetry([nodes[2], nodes[3]])]  # left_hand <-> right_hand

    skeleton1 = sio.Skeleton(
        nodes=nodes, edges=edges, symmetries=symmetries, name="complex-skeleton"
    )

    # Round trip
    encoder = SkeletonEncoder()
    decoder = SkeletonDecoder()
    json_str = encoder.encode(skeleton1)
    skeleton2 = decoder.decode(json_str)

    assert skeleton1.name == skeleton2.name
    assert len(skeleton1.edges) == len(skeleton2.edges)
    assert len(skeleton1.symmetries) == len(skeleton2.symmetries)

    # Check symmetry nodes
    sym_nodes1 = {n.name for n in skeleton1.symmetries[0].nodes}
    sym_nodes2 = {n.name for n in skeleton2.symmetries[0].nodes}
    assert sym_nodes1 == sym_nodes2


# File I/O tests using fixtures
def test_load_minimal_skeleton_fixture(skeleton_json_minimal):
    """Test loading the minimal skeleton fixture."""
    skeleton = sio.load_skeleton(skeleton_json_minimal)

    assert skeleton.name == "Skeleton-1"
    assert len(skeleton.nodes) == 2
    assert skeleton.nodes[0].name == "head"
    assert skeleton.nodes[1].name == "abdomen"
    assert len(skeleton.edges) == 1
    assert skeleton.edges[0].source.name == "head"
    assert skeleton.edges[0].destination.name == "abdomen"
    assert len(skeleton.symmetries) == 0


def test_load_flies_skeleton_fixture(skeleton_json_flies):
    """Test loading the complex flies skeleton fixture."""
    skeleton = sio.load_skeleton(skeleton_json_flies)

    assert skeleton.name == "Skeleton-0"
    assert len(skeleton.nodes) >= 10

    # Check that some expected fly nodes are present
    node_names = {node.name for node in skeleton.nodes}
    basic_nodes = {"head", "thorax"}
    assert basic_nodes.intersection(
        node_names
    ), f"Expected basic nodes not found in {node_names}"

    # Verify we have edges and symmetries
    assert len(skeleton.edges) > 0, "Should have some edges"
    assert len(skeleton.symmetries) > 0, "Should have some symmetries"

    # Basic structural validation
    assert skeleton is not None
    assert isinstance(skeleton.name, str)


def test_round_trip_minimal_fixture(skeleton_json_minimal, tmp_path):
    """Test round-trip with minimal skeleton fixture."""
    original = sio.load_skeleton(skeleton_json_minimal)

    # Save to new file
    output_path = tmp_path / "minimal_round_trip.json"
    sio.save_skeleton(original, output_path)

    # Load again
    reloaded = sio.load_skeleton(output_path)

    assert original.name == reloaded.name
    assert len(original.nodes) == len(reloaded.nodes)
    assert len(original.edges) == len(reloaded.edges)

    # Verify node names preserved
    for o_node, r_node in zip(original.nodes, reloaded.nodes):
        assert o_node.name == r_node.name


def test_round_trip_flies_fixture(skeleton_json_flies, tmp_path):
    """Test round-trip with complex flies skeleton fixture."""
    original = sio.load_skeleton(skeleton_json_flies)

    # Verify original loads correctly
    assert original.name == "Skeleton-0"
    assert len(original.nodes) > 0
    assert len(original.edges) > 0
    assert len(original.symmetries) > 0

    # Save to new file
    output_path = tmp_path / "flies_round_trip.json"
    sio.save_skeleton(original, output_path)

    # Load again
    reloaded = sio.load_skeleton(output_path)

    # Verify basic structure preserved
    assert original.name == reloaded.name

    # For complex skeletons, just verify we maintain reasonable structure
    assert len(reloaded.nodes) > 0, "Should have nodes after round-trip"
    assert len(reloaded.edges) > 0, "Should have edges after round-trip"
    assert len(reloaded.symmetries) > 0, "Should have symmetries after round-trip"


# Backward compatibility tests
def test_load_existing_skeleton_file(skeleton_json_minimal):
    """Test loading existing skeleton files for backward compatibility."""
    skeleton = sio.load_skeleton(skeleton_json_minimal)

    # Verify content
    assert skeleton.name == "Skeleton-1"
    assert len(skeleton.nodes) == 2
    assert skeleton.nodes[0].name == "head"
    assert skeleton.nodes[1].name == "abdomen"
    assert len(skeleton.edges) == 1
    assert skeleton.edges[0].source.name == "head"
    assert skeleton.edges[0].destination.name == "abdomen"


def test_json_format_preservation(skeleton_json_minimal):
    """Test that JSON format structure is preserved."""
    # Load original JSON
    with open(skeleton_json_minimal, "r") as f:
        original_data = json.loads(f.read())

    # Load and re-encode
    skeleton = sio.load_skeleton(skeleton_json_minimal)
    encoder = SkeletonEncoder()
    encoded_json = encoder.encode(skeleton)
    encoded_data = json.loads(encoded_json)

    # Check top-level keys match
    assert set(original_data.keys()) == set(encoded_data.keys())

    # Check critical structure elements
    assert original_data["directed"] == encoded_data["directed"]
    assert original_data["multigraph"] == encoded_data["multigraph"]
    assert original_data["graph"]["name"] == encoded_data["graph"]["name"]

    # Check nodes and links count
    assert len(original_data["nodes"]) == len(encoded_data["nodes"])
    assert len(original_data["links"]) == len(encoded_data["links"])


def test_decode_variations():
    """Test decoding various valid skeleton JSON formats."""
    decoder = SkeletonDecoder()

    # Test minimal skeleton
    minimal = {
        "directed": True,
        "graph": {"name": "minimal"},
        "links": [],
        "multigraph": True,
        "nodes": [],
    }

    skeleton = decoder.decode(minimal)
    assert skeleton.name == "minimal"
    assert len(skeleton.nodes) == 0

    # Test skeleton with py/id references in edge types
    with_refs = {
        "directed": True,
        "graph": {"name": "with-refs", "num_edges_inserted": 2},
        "links": [
            {
                "edge_insert_idx": 0,
                "key": 0,
                "source": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["A", 1.0]},
                },
                "target": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["B", 1.0]},
                },
                "type": {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [1]},
                    ]
                },
            },
            {
                "edge_insert_idx": 1,
                "key": 0,
                "source": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["B", 1.0]},
                },
                "target": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["C", 1.0]},
                },
                "type": {"py/id": 1},  # Reference to first edge type
            },
        ],
        "multigraph": True,
        "nodes": [{"id": {"py/id": 1}}, {"id": {"py/id": 2}}, {"id": {"py/id": 3}}],
    }

    skeleton = decoder.decode(with_refs)
    assert len(skeleton.edges) == 2
    assert skeleton.edges[0].source.name == "A"
    assert skeleton.edges[1].destination.name == "C"


def test_edge_type_handling(skeleton_json_flies):
    """Test edge type encoding with py/id references."""
    skeleton = sio.load_skeleton(skeleton_json_flies)

    # Verify we loaded a complex skeleton
    assert len(skeleton.edges) > 0
    assert len(skeleton.symmetries) > 0

    # Re-encode it
    encoder = SkeletonEncoder()
    encoded_json = encoder.encode(skeleton)
    data = json.loads(encoded_json)

    # Check edge type encoding
    edge_types_seen = set()
    py_reduce_count = 0
    py_id_count = 0

    for link in data["links"]:
        if "type" in link:
            if "py/reduce" in link["type"]:
                py_reduce_count += 1
                edge_type = link["type"]["py/reduce"][1]["py/tuple"][0]
                edge_types_seen.add(edge_type)
            elif "py/id" in link["type"]:
                py_id_count += 1

    # Should have at least one py/reduce for first occurrence
    assert py_reduce_count >= 1

    # Check that we have edge types for both regular edges and symmetries
    assert 1 in edge_types_seen  # Regular edges
    assert 2 in edge_types_seen  # Symmetry edges

    # For a skeleton with many edges/symmetries, should have some py/id references
    if len(skeleton.edges) > 1 or len(skeleton.symmetries) > 1:
        assert py_id_count > 0
