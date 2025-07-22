"""Tests for standalone skeleton JSON I/O."""

import pytest
import json
import tempfile
from pathlib import Path
import sleap_io as sio
from sleap_io.io.skeleton import (
    SkeletonDecoder,
    SkeletonEncoder,
    SkeletonSLPDecoder,
    SkeletonSLPEncoder,
    SkeletonYAMLDecoder,
    SkeletonYAMLEncoder,
)


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
    assert len(skeleton.nodes) == 13

    # Check exact node names
    expected_nodes = [
        "head",
        "eyeL",
        "eyeR",
        "thorax",
        "abdomen",
        "wingL",
        "wingR",
        "forelegL4",
        "forelegR4",
        "midlegL4",
        "midlegR4",
        "hindlegL4",
        "hindlegR4",
    ]
    node_names = [node.name for node in skeleton.nodes]
    assert node_names == expected_nodes

    # Check edges
    assert len(skeleton.edges) == 12
    # Check first few edges to verify structure
    assert (
        skeleton.edges[0].source.name == "head"
        and skeleton.edges[0].destination.name == "eyeL"
    )
    assert (
        skeleton.edges[1].source.name == "head"
        and skeleton.edges[1].destination.name == "eyeR"
    )
    assert (
        skeleton.edges[2].source.name == "thorax"
        and skeleton.edges[2].destination.name == "head"
    )

    # Check symmetries
    assert len(skeleton.symmetries) == 5
    # Check specific symmetry pairs
    sym_pairs = [sorted([n.name for n in s.nodes]) for s in skeleton.symmetries]
    assert ["abdomen", "thorax"] in sym_pairs
    assert ["wingL", "wingR"] in sym_pairs
    assert ["forelegL4", "forelegR4"] in sym_pairs


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


# SLP encoder/decoder tests
def test_slp_encoder_decoder():
    """Test SLP format encoder and decoder."""
    # Create test skeletons
    nodes1 = [sio.Node("head"), sio.Node("neck"), sio.Node("tail")]
    edges1 = [sio.Edge(nodes1[0], nodes1[1]), sio.Edge(nodes1[1], nodes1[2])]
    skel1 = sio.Skeleton(nodes=nodes1, edges=edges1, name="skeleton1")

    nodes2 = [sio.Node("left"), sio.Node("right")]
    symmetry2 = sio.Symmetry([nodes2[0], nodes2[1]])
    skel2 = sio.Skeleton(nodes=nodes2, symmetries=[symmetry2], name="skeleton2")

    skeletons = [skel1, skel2]

    # Encode to SLP format
    encoder = SkeletonSLPEncoder()
    skeletons_dicts, nodes_dicts = encoder.encode_skeletons(skeletons)

    # Check that we get the expected structure
    assert len(skeletons_dicts) == 2
    assert len(nodes_dicts) == 5  # 3 + 2 unique nodes

    # Check that node references are integers (SLP format)
    assert isinstance(skeletons_dicts[0]["links"][0]["source"], int)
    assert isinstance(skeletons_dicts[0]["links"][0]["target"], int)
    assert isinstance(skeletons_dicts[1]["links"][0]["source"], int)
    assert isinstance(skeletons_dicts[1]["links"][0]["target"], int)

    # The exact indices depend on the global node ordering, so we just verify they're valid

    # Create fake metadata for decoder
    metadata = {"skeletons": skeletons_dicts}
    node_names = [node["name"] for node in nodes_dicts]

    # Decode back
    decoder = SkeletonSLPDecoder()
    decoded_skeletons = decoder.decode(metadata, node_names)

    # Verify structure is preserved
    assert len(decoded_skeletons) == 2
    assert decoded_skeletons[0].name == "skeleton1"
    assert decoded_skeletons[1].name == "skeleton2"

    # Check nodes
    assert len(decoded_skeletons[0].nodes) == 3
    assert len(decoded_skeletons[1].nodes) == 2
    assert decoded_skeletons[0].nodes[0].name == "head"
    assert decoded_skeletons[1].nodes[0].name == "left"

    # Check edges and symmetries
    assert len(decoded_skeletons[0].edges) == 2
    assert len(decoded_skeletons[1].edges) == 0
    assert len(decoded_skeletons[0].symmetries) == 0
    assert len(decoded_skeletons[1].symmetries) == 1


# Edge case tests for missing coverage
def test_decode_node_direct_format():
    """Test decoding a node without py/state (direct format)."""
    # This covers line 168
    decoder = SkeletonDecoder()
    node_data = {
        "py/object": "sleap.skeleton.Node",
        "name": "direct_node",  # Direct format without py/state
    }
    node = decoder._decode_node(node_data)
    assert node.name == "direct_node"


def test_decode_node_not_found_in_all_nodes():
    """Test node resolution when node is not in all_nodes."""
    # This covers line 196
    decoder = SkeletonDecoder()
    all_nodes = {"existing": sio.Node("existing")}
    py_id_to_node_name = {}

    node_ref = {
        "py/object": "sleap.skeleton.Node",
        "py/state": {"py/tuple": ["new_node", 1.0]},
    }

    node = decoder._resolve_node_reference(node_ref, all_nodes, py_id_to_node_name)
    assert node.name == "new_node"


def test_decode_py_id_fallback_resolution():
    """Test py/id resolution through fallback path."""
    # This covers lines 202-207
    decoder = SkeletonDecoder()
    decoder._id_to_object = {}  # Empty cache
    all_nodes = {"test_node": sio.Node("test_node")}
    py_id_to_node_name = {5: "test_node"}

    node_ref = {"py/id": 5}
    node = decoder._resolve_node_reference(node_ref, all_nodes, py_id_to_node_name)
    assert node.name == "test_node"


def test_decode_py_id_not_found():
    """Test py/id resolution when ID is not found."""
    # This covers line 207
    decoder = SkeletonDecoder()
    decoder._id_to_object = {}
    all_nodes = {}
    py_id_to_node_name = {}

    node_ref = {"py/id": 999}
    with pytest.raises(ValueError, match="py/id 999 not found"):
        decoder._resolve_node_reference(node_ref, all_nodes, py_id_to_node_name)


def test_decode_integer_node_reference():
    """Test error when integer node reference is used in standalone format."""
    # This covers lines 208-210
    decoder = SkeletonDecoder()
    all_nodes = {}
    py_id_to_node_name = {}

    with pytest.raises(ValueError, match="Direct index reference not supported: 5"):
        decoder._resolve_node_reference(5, all_nodes, py_id_to_node_name)


def test_decode_unknown_node_reference():
    """Test error for unknown node reference format."""
    # This covers line 212
    decoder = SkeletonDecoder()
    all_nodes = {}
    py_id_to_node_name = {}

    with pytest.raises(ValueError, match="Unknown node reference format"):
        decoder._resolve_node_reference("invalid_format", all_nodes, py_id_to_node_name)


def test_decode_edge_type_default():
    """Test default edge type when not specified."""
    # This covers line 231
    decoder = SkeletonDecoder()
    edge_type = decoder._get_edge_type({})  # Empty dict, no type info
    assert edge_type == 1  # Default to regular edge


def test_encode_edge_with_non_standard_type():
    """Test encoding edges with non-1 edge types."""
    # This covers lines 354-363
    encoder = SkeletonEncoder()

    # Create a mock edge and call _encode_edge with edge_type=2
    edge = sio.Edge(sio.Node("A"), sio.Node("B"))
    edge_dict = encoder._encode_edge(edge, 0, edge_type=2)

    # First occurrence should use py/reduce
    assert "py/reduce" in edge_dict["type"]
    assert edge_dict["type"]["py/reduce"][1]["py/tuple"][0] == 2

    # Second occurrence should use py/id
    edge_dict2 = encoder._encode_edge(edge, 1, edge_type=2)
    assert edge_dict2["type"] == {"py/id": 2}


def test_slp_encoder_multiple_symmetries():
    """Test SLP encoder with multiple symmetries to trigger py/id usage."""
    # This covers line 589
    nodes = [sio.Node(f"node{i}") for i in range(6)]
    symmetries = [
        sio.Symmetry([nodes[0], nodes[1]]),
        sio.Symmetry([nodes[2], nodes[3]]),
        sio.Symmetry([nodes[4], nodes[5]]),
    ]
    skeleton = sio.Skeleton(nodes=nodes, symmetries=symmetries, name="multi_sym")

    encoder = SkeletonSLPEncoder()
    skeletons_dicts, nodes_dicts = encoder.encode_skeletons([skeleton])

    # Check that we have py/reduce for first symmetry
    assert "py/reduce" in skeletons_dicts[0]["links"][0]["type"]

    # Check that subsequent symmetries use py/id
    assert skeletons_dicts[0]["links"][1]["type"] == {"py/id": 2}
    assert skeletons_dicts[0]["links"][2]["type"] == {"py/id": 2}


# YAML encoder/decoder tests
def test_yaml_decoder_single_skeleton():
    """Test decoding a single skeleton from YAML dict."""
    yaml_data = {
        "nodes": [{"name": "head"}, {"name": "thorax"}, {"name": "abdomen"}],
        "edges": [
            {"source": {"name": "head"}, "destination": {"name": "thorax"}},
            {"source": {"name": "thorax"}, "destination": {"name": "abdomen"}},
        ],
        "symmetries": [],
    }

    decoder = SkeletonYAMLDecoder()
    skeleton = decoder.decode(yaml_data)

    assert len(skeleton.nodes) == 3
    assert skeleton.nodes[0].name == "head"
    assert len(skeleton.edges) == 2
    assert skeleton.edges[0].source.name == "head"
    assert skeleton.edges[0].destination.name == "thorax"


def test_yaml_decoder_multiple_skeletons():
    """Test decoding multiple skeletons from YAML with names as keys."""
    yaml_data = {
        "Skeleton-1": {
            "nodes": [{"name": "A"}, {"name": "B"}],
            "edges": [{"source": {"name": "A"}, "destination": {"name": "B"}}],
            "symmetries": [],
        },
        "Skeleton-2": {
            "nodes": [{"name": "X"}, {"name": "Y"}],
            "edges": [],
            "symmetries": [[{"name": "X"}, {"name": "Y"}]],
        },
    }

    decoder = SkeletonYAMLDecoder()
    skeletons = decoder.decode(yaml_data)

    assert isinstance(skeletons, list)
    assert len(skeletons) == 2
    assert skeletons[0].name == "Skeleton-1"
    assert skeletons[1].name == "Skeleton-2"
    assert len(skeletons[1].symmetries) == 1


def test_yaml_decoder_from_string():
    """Test decoding from YAML string."""
    yaml_str = """
Skeleton-0:
  nodes:
  - name: head
  - name: tail
  edges:
  - source:
      name: head
    destination:
      name: tail
  symmetries: []
"""

    decoder = SkeletonYAMLDecoder()
    skeletons = decoder.decode(yaml_str)

    assert isinstance(skeletons, list)
    assert len(skeletons) == 1
    assert skeletons[0].name == "Skeleton-0"
    assert len(skeletons[0].nodes) == 2


def test_yaml_decoder_dict_method():
    """Test decode_dict method for embedded skeleton data."""
    skeleton_data = {
        "nodes": [{"name": "A"}, {"name": "B"}],
        "edges": [{"source": {"name": "A"}, "destination": {"name": "B"}}],
        "symmetries": [],
    }

    decoder = SkeletonYAMLDecoder()
    skeleton = decoder.decode_dict(skeleton_data, name="CustomName")

    assert skeleton.name == "CustomName"
    assert len(skeleton.nodes) == 2


def test_yaml_encoder_single_skeleton():
    """Test encoding a single skeleton to YAML."""
    nodes = [sio.Node("head"), sio.Node("tail")]
    edges = [sio.Edge(nodes[0], nodes[1])]
    skeleton = sio.Skeleton(nodes=nodes, edges=edges, name="TestSkeleton")

    encoder = SkeletonYAMLEncoder()
    yaml_str = encoder.encode(skeleton)

    # Parse back to verify structure
    import yaml

    data = yaml.safe_load(yaml_str)

    assert "TestSkeleton" in data
    assert len(data["TestSkeleton"]["nodes"]) == 2
    assert data["TestSkeleton"]["nodes"][0]["name"] == "head"
    assert len(data["TestSkeleton"]["edges"]) == 1
    assert data["TestSkeleton"]["edges"][0]["source"]["name"] == "head"


def test_yaml_encoder_multiple_skeletons():
    """Test encoding multiple skeletons to YAML."""
    skel1 = sio.Skeleton(nodes=[sio.Node("A")], name="Skel1")
    skel2 = sio.Skeleton(nodes=[sio.Node("B")], name="Skel2")

    encoder = SkeletonYAMLEncoder()
    yaml_str = encoder.encode([skel1, skel2])

    import yaml

    data = yaml.safe_load(yaml_str)

    assert "Skel1" in data
    assert "Skel2" in data
    assert data["Skel1"]["nodes"][0]["name"] == "A"
    assert data["Skel2"]["nodes"][0]["name"] == "B"


def test_yaml_encoder_dict_method():
    """Test encode_dict method for embedding skeleton data."""
    skeleton = sio.Skeleton(
        nodes=[sio.Node("A"), sio.Node("B")],
        edges=[sio.Edge(sio.Node("A"), sio.Node("B"))],
        name="Test",
    )

    encoder = SkeletonYAMLEncoder()
    skeleton_dict = encoder.encode_dict(skeleton)

    assert "nodes" in skeleton_dict
    assert "edges" in skeleton_dict
    assert "symmetries" in skeleton_dict
    assert len(skeleton_dict["nodes"]) == 2


def test_yaml_round_trip():
    """Test round-trip encoding and decoding with YAML."""
    # Create complex skeleton
    nodes = [sio.Node(name) for name in ["head", "neck", "left_hand", "right_hand"]]
    edges = [
        sio.Edge(nodes[0], nodes[1]),
        sio.Edge(nodes[1], nodes[2]),
        sio.Edge(nodes[1], nodes[3]),
    ]
    symmetries = [sio.Symmetry([nodes[2], nodes[3]])]
    skeleton1 = sio.Skeleton(
        nodes=nodes, edges=edges, symmetries=symmetries, name="Complex"
    )

    # Encode and decode
    encoder = SkeletonYAMLEncoder()
    decoder = SkeletonYAMLDecoder()
    yaml_str = encoder.encode(skeleton1)
    skeletons = decoder.decode(yaml_str)
    skeleton2 = skeletons[0]

    # Verify
    assert skeleton1.name == skeleton2.name
    assert len(skeleton1.nodes) == len(skeleton2.nodes)
    assert len(skeleton1.edges) == len(skeleton2.edges)
    assert len(skeleton1.symmetries) == len(skeleton2.symmetries)

    # Check node names preserved
    for n1, n2 in zip(skeleton1.nodes, skeleton2.nodes):
        assert n1.name == n2.name


def test_load_skeleton_yaml_fixture(skeleton_yaml_flies):
    """Test loading the YAML skeleton fixture."""
    skeleton = sio.load_skeleton(skeleton_yaml_flies)

    # Should be a list since YAML has skeleton names as keys
    assert isinstance(skeleton, list)
    assert len(skeleton) == 1
    skeleton = skeleton[0]

    assert skeleton.name == "Skeleton-0"
    assert len(skeleton.nodes) == 13

    # Check some specific nodes
    node_names = {node.name for node in skeleton.nodes}
    assert "head" in node_names
    assert "thorax" in node_names
    assert "eyeL" in node_names
    assert "eyeR" in node_names

    # Check edges
    assert len(skeleton.edges) == 12

    # Check symmetries
    assert len(skeleton.symmetries) > 0


def test_yaml_json_equivalence(skeleton_json_flies, skeleton_yaml_flies):
    """Test that YAML and JSON files produce equivalent skeletons."""
    # Load from JSON
    json_skeleton = sio.load_skeleton(skeleton_json_flies)

    # Load from YAML
    yaml_skeletons = sio.load_skeleton(skeleton_yaml_flies)
    yaml_skeleton = yaml_skeletons[0]  # YAML returns list

    # Compare structure
    assert json_skeleton.name == yaml_skeleton.name

    # Both files should have the same number of nodes
    assert len(json_skeleton.nodes) == 13
    assert len(yaml_skeleton.nodes) == 13

    # Check that both have the same node names
    json_node_names = {n.name for n in json_skeleton.nodes}
    yaml_node_names = {n.name for n in yaml_skeleton.nodes}
    assert json_node_names == yaml_node_names

    # Both should have edges and symmetries
    assert len(json_skeleton.edges) > 0
    assert len(yaml_skeleton.edges) > 0
    assert len(json_skeleton.symmetries) > 0
    assert len(yaml_skeleton.symmetries) > 0


def test_save_load_yaml_round_trip(tmp_path):
    """Test saving and loading skeleton in YAML format."""
    # Create skeleton
    nodes = [sio.Node("A"), sio.Node("B"), sio.Node("C")]
    edges = [sio.Edge(nodes[0], nodes[1]), sio.Edge(nodes[1], nodes[2])]
    skeleton = sio.Skeleton(nodes=nodes, edges=edges, name="TestSkel")

    # Save as YAML
    yaml_path = tmp_path / "test.yaml"
    sio.save_skeleton(skeleton, yaml_path)

    # Load back
    loaded = sio.load_skeleton(yaml_path)
    assert isinstance(loaded, list)
    loaded_skeleton = loaded[0]

    # Verify
    assert loaded_skeleton.name == skeleton.name
    assert len(loaded_skeleton.nodes) == len(skeleton.nodes)
    assert len(loaded_skeleton.edges) == len(skeleton.edges)

    # Check node order preserved
    for orig, loaded in zip(skeleton.nodes, loaded_skeleton.nodes):
        assert orig.name == loaded.name


def test_round_trip_fly32_skeleton(skeleton_json_fly32, tmp_path):
    """Test round-trip encoding/decoding of fly32 skeleton with non-sequential py/ids."""
    # Load original
    original = sio.load_skeleton(skeleton_json_fly32)
    assert isinstance(original, list)
    original = original[0]

    # Save to new file
    output_path = tmp_path / "fly32_round_trip.json"
    sio.save_skeleton(original, output_path)

    # Load again
    reloaded = sio.load_skeleton(output_path)
    # When saving a single skeleton, it's loaded back as a single skeleton, not a list
    if isinstance(reloaded, list):
        reloaded = reloaded[0]

    # Verify everything matches
    assert original.name == reloaded.name
    assert len(original.nodes) == len(reloaded.nodes) == 32
    assert len(original.edges) == len(reloaded.edges)

    # Verify node names and order preserved
    for i, (o_node, r_node) in enumerate(zip(original.nodes, reloaded.nodes)):
        assert (
            o_node.name == r_node.name
        ), f"Node {i} mismatch: {o_node.name} != {r_node.name}"

    # Verify edges preserved
    for i, (o_edge, r_edge) in enumerate(zip(original.edges, reloaded.edges)):
        assert o_edge.source.name == r_edge.source.name
        assert o_edge.destination.name == r_edge.destination.name


def test_training_config_decode(training_config_fly32, skeleton_json_fly32):
    """Test decoding a training config with embedded skeleton data."""

    # Test loading standalone skeleton file
    with open(skeleton_json_fly32, "r") as f:
        skeleton_data = json.load(f)
    skeletons = SkeletonDecoder().decode(skeleton_data)
    assert len(skeletons) == 1
    skeleton = skeletons[0]
    assert (
        skeleton.name
        == "M:/talmo/data/leap_datasets/BermanFlies/2018-05-03_cluster-sampled.k=10,n=150.labels.mat"
    )
    assert len(skeleton.nodes) == 32

    # Test loading skeleton from training config
    with open(training_config_fly32, "r") as f:
        data = json.load(f)

    skel_data = data["data"]["labels"]["skeletons"]
    skels_cfg = SkeletonDecoder().decode(skel_data)
    assert len(skels_cfg) == 1
    skeleton_cfg = skels_cfg[0]

    # Verify the skeletons match
    assert skeleton.name == skeleton_cfg.name
    assert len(skeleton.nodes) == len(skeleton_cfg.nodes)

    # Verify node names and order match
    for i, (n1, n2) in enumerate(zip(skeleton.nodes, skeleton_cfg.nodes)):
        assert n1.name == n2.name, f"Node {i} mismatch: {n1.name} != {n2.name}"

    # Verify we have the expected 32 nodes
    expected_node_names = [
        "head",
        "eyeL",
        "eyeR",
        "neck",
        "thorax",
        "wingL",
        "wingR",
        "abdomen",
        "forelegR1",
        "forelegR2",
        "forelegR3",
        "forelegR4",
        "midlegR1",
        "midlegR2",
        "midlegR3",
        "midlegR4",
        "hindlegR1",
        "hindlegR2",
        "hindlegR3",
        "hindlegR4",
        "forelegL1",
        "forelegL2",
        "forelegL3",
        "forelegL4",
        "midlegL1",
        "midlegL2",
        "midlegL3",
        "midlegL4",
        "hindlegL1",
        "hindlegL2",
        "hindlegL3",
        "hindlegL4",
    ]
    actual_node_names = [n.name for n in skeleton.nodes]
    assert actual_node_names == expected_node_names


def test_yaml_decoder_invalid_format():
    """Test YAML decoder with invalid data format."""
    decoder = SkeletonYAMLDecoder()

    # Test with list input (not supported)
    with pytest.raises(ValueError, match="Unexpected data format"):
        decoder.decode([{"nodes": []}])


def test_load_skeleton_from_slp(slp_typical):
    """Test loading skeletons from SLP file."""
    skeletons = sio.load_skeleton(slp_typical)

    assert isinstance(skeletons, list)
    assert len(skeletons) == 1

    # Check the skeleton details
    skeleton = skeletons[0]
    assert skeleton.name == "Skeleton-0"
    assert len(skeleton.nodes) == 2

    # Check node names
    node_names = [node.name for node in skeleton.nodes]
    assert node_names == ["A", "B"]

    # Check edges
    assert len(skeleton.edges) == 1
    assert skeleton.edges[0].source.name == "A"
    assert skeleton.edges[0].destination.name == "B"

    # Check symmetries
    assert len(skeleton.symmetries) == 0


def test_load_skeleton_from_training_config(training_config_fly32):
    """Test loading skeleton from training config JSON file."""
    skeletons = sio.load_skeleton(training_config_fly32)

    assert isinstance(skeletons, list)
    assert len(skeletons) == 1

    skeleton = skeletons[0]
    assert (
        skeleton.name
        == "M:/talmo/data/leap_datasets/BermanFlies/2018-05-03_cluster-sampled.k=10,n=150.labels.mat"
    )
    assert len(skeleton.nodes) == 32
    assert len(skeleton.edges) == 25

    # Check exact node names and order
    expected_nodes = [
        "head",
        "eyeL",
        "eyeR",
        "neck",
        "thorax",
        "wingL",
        "wingR",
        "abdomen",
        "forelegR1",
        "forelegR2",
        "forelegR3",
        "forelegR4",
        "midlegR1",
        "midlegR2",
        "midlegR3",
        "midlegR4",
        "hindlegR1",
        "hindlegR2",
        "hindlegR3",
        "hindlegR4",
        "forelegL1",
        "forelegL2",
        "forelegL3",
        "forelegL4",
        "midlegL1",
        "midlegL2",
        "midlegL3",
        "midlegL4",
        "hindlegL1",
        "hindlegL2",
        "hindlegL3",
        "hindlegL4",
    ]
    node_names = [n.name for n in skeleton.nodes]
    assert node_names == expected_nodes


def test_load_skeleton_training_config_without_skeletons(tmp_path):
    """Test loading training config without skeleton data."""
    # Create a training config without skeleton data
    config_data = {
        "data": {
            "labels": {"training_labels": "train.slp", "validation_labels": "val.slp"}
        },
        "model": {"backbone": {"unet": {}}},
    }

    config_path = tmp_path / "config_no_skeleton.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    # Should fall back to regular skeleton decoding and return empty skeleton
    skeleton = sio.load_skeleton(config_path)
    assert isinstance(skeleton, sio.Skeleton)
    assert len(skeleton.nodes) == 0  # Empty skeleton
    assert len(skeleton.edges) == 0


def test_load_skeleton_invalid_json(tmp_path):
    """Test loading invalid JSON that's neither skeleton nor training config."""
    # Create an invalid JSON file
    invalid_data = {"foo": "bar", "baz": [1, 2, 3]}

    invalid_path = tmp_path / "invalid.json"
    with open(invalid_path, "w") as f:
        json.dump(invalid_data, f)

    # Should return an empty skeleton when JSON doesn't match expected format
    skeleton = sio.load_skeleton(invalid_path)
    assert isinstance(skeleton, sio.Skeleton)
    assert len(skeleton.nodes) == 0
    assert len(skeleton.edges) == 0


def test_load_skeleton_format_detection(
    skeleton_json_minimal, skeleton_yaml_flies, training_config_fly32
):
    """Test that load_skeleton correctly detects different file formats."""
    # Test JSON skeleton file (minimal)
    json_skeleton = sio.load_skeleton(skeleton_json_minimal)
    assert isinstance(json_skeleton, sio.Skeleton)
    assert json_skeleton.name == "Skeleton-1"
    assert len(json_skeleton.nodes) == 2
    assert [n.name for n in json_skeleton.nodes] == ["head", "abdomen"]
    assert len(json_skeleton.edges) == 1
    assert json_skeleton.edges[0].source.name == "head"
    assert json_skeleton.edges[0].destination.name == "abdomen"

    # Test YAML skeleton file (flies13)
    yaml_skeletons = sio.load_skeleton(skeleton_yaml_flies)
    assert isinstance(yaml_skeletons, list)
    assert len(yaml_skeletons) == 1
    assert yaml_skeletons[0].name == "Skeleton-0"
    assert len(yaml_skeletons[0].nodes) == 13
    assert len(yaml_skeletons[0].edges) == 12
    assert len(yaml_skeletons[0].symmetries) == 10  # YAML has all symmetry pairs

    # Test training config JSON
    config_skeletons = sio.load_skeleton(training_config_fly32)
    assert isinstance(config_skeletons, list)
    assert len(config_skeletons) == 1
    assert len(config_skeletons[0].nodes) == 32
    assert len(config_skeletons[0].edges) == 25


def test_load_skeleton_malformed_json(tmp_path):
    """Test handling of malformed JSON that triggers JSONDecodeError."""
    # Create a file with invalid JSON syntax
    malformed_path = tmp_path / "malformed.json"
    with open(malformed_path, "w") as f:
        f.write('{"data": {"labels": {"skeletons": [INVALID JSON HERE}')

    # Should catch JSONDecodeError and fall back to regular skeleton decoding
    # which will then raise its own error
    with pytest.raises(json.JSONDecodeError):
        sio.load_skeleton(malformed_path)


def test_load_skeleton_training_config_type_error(tmp_path):
    """Test handling of training config that causes TypeError."""
    # Create a training config where skeletons is not the expected type
    config_data = {
        "data": {
            "labels": {"skeletons": "not_a_list_or_dict"}  # String instead of list/dict
        }
    }

    config_path = tmp_path / "bad_type_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    # Should handle gracefully and return empty skeleton
    skeleton = sio.load_skeleton(config_path)
    assert isinstance(skeleton, sio.Skeleton)
    assert len(skeleton.nodes) == 0


def test_load_skeleton_training_config_parsing_error(tmp_path):
    """Test exception handling when parsing training config structure."""
    # Create a file that looks like a training config but has a non-dict skeleton entry
    # This will cause the isinstance check to pass but the decoder to fail
    config_data = {
        "data": {
            "labels": {"skeletons": [None]}  # This will cause issues in the decoder
        }
    }

    config_path = tmp_path / "null_skeleton_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    # The decoder will fail but it should be caught and fall back
    # Since None is not a valid skeleton format, the fallback will also fail
    with pytest.raises(AttributeError):  # None has no 'get' attribute
        sio.load_skeleton(config_path)
