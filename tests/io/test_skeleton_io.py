"""Tests for standalone skeleton JSON I/O."""

import pytest
import simplejson as json

import sleap_io as sio
from sleap_io.io.skeleton import (
    SkeletonDecoder,
    SkeletonEncoder,
    SkeletonSLPDecoder,
    SkeletonSLPEncoder,
    SkeletonYAMLDecoder,
    SkeletonYAMLEncoder,
    decode_skeleton,
    encode_skeleton,
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

    skeleton = decode_skeleton(json.dumps(json_data))

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

    skeleton = decode_skeleton(json.dumps(json_data))

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

    skeleton = decode_skeleton(json.dumps(json_data))

    assert skeleton.nodes[0].name == "node_a"
    assert skeleton.nodes[1].name == "node_b"


def test_decode_from_json_string():
    """Test decoding from JSON string."""
    json_str = (
        """{"directed": true, "graph": {"name": "test"}, "links": [], "nodes": []}"""
    )

    skeleton = decode_skeleton(json_str)

    assert skeleton.name == "test"
    assert len(skeleton.nodes) == 0


def test_decode_multiple_skeletons():
    """Test decoding a list of skeletons."""
    json_data = [
        {"directed": True, "graph": {"name": "skel1"}, "links": [], "nodes": []},
        {"directed": True, "graph": {"name": "skel2"}, "links": [], "nodes": []},
    ]

    skeletons = decode_skeleton(json.dumps(json_data))

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
    json_str = encode_skeleton(skeleton)
    data = json.loads(json_str)

    assert data["graph"]["name"] == "test"
    assert data["directed"] is True
    assert data["multigraph"] is True
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
    json_str = encode_skeleton(skeleton)
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
    json_str = encode_skeleton(skeleton)
    data = json.loads(json_str)

    # First edge should have py/reduce
    assert "py/reduce" in data["links"][0]["type"]
    # Check that at least one edge uses py/id reference
    has_pyid_ref = False
    for link in data["links"]:
        if "py/id" in link["type"]:
            has_pyid_ref = True
            break
    assert has_pyid_ref  # At least one edge should use py/id reference
    assert "py/id" in data["links"][1]["type"]
    assert "py/id" in data["links"][2]["type"]
    assert data["links"][1]["type"]["py/id"] == data["links"][2]["type"]["py/id"]


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
    json_str = encode_skeleton(skeleton1)
    skeleton2 = decode_skeleton(json_str)

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
    json_str = encode_skeleton(skeleton1)
    skeleton2 = decode_skeleton(json_str)

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

    # Check exact node names - using the correct fly skeleton order
    expected_nodes = [
        "head",
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
        "eyeL",
        "eyeR",
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
    assert ["eyeL", "eyeR"] in sym_pairs
    assert ["wingL", "wingR"] in sym_pairs
    assert ["forelegL4", "forelegR4"] in sym_pairs
    assert ["midlegL4", "midlegR4"] in sym_pairs
    assert ["hindlegL4", "hindlegR4"] in sym_pairs


def test_load_mice_hc_skeleton_fixture(skeleton_json_mice_hc):
    """Test loading the mice head-centered skeleton fixture."""
    skeleton = sio.load_skeleton(skeleton_json_mice_hc)

    assert skeleton.name == "Skeleton-0"
    assert len(skeleton.nodes) == 5

    # Check exact node names - mice head-centered configuration
    expected_nodes = ["nose1", "earL1", "earR1", "tailstart1", "tailend1"]
    node_names = [node.name for node in skeleton.nodes]
    assert node_names == expected_nodes

    # Check edges
    assert len(skeleton.edges) == 4
    # Verify basic connectivity structure
    edge_pairs = [(e.source.name, e.destination.name) for e in skeleton.edges]

    # Should have connections forming a simple mouse body structure
    assert ("nose1", "earL1") in edge_pairs
    assert ("nose1", "earR1") in edge_pairs
    assert ("nose1", "tailstart1") in edge_pairs
    assert ("tailstart1", "tailend1") in edge_pairs

    # Check symmetries - this simple skeleton has no symmetries defined
    assert len(skeleton.symmetries) == 0


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
    # Test minimal skeleton
    minimal = {
        "directed": True,
        "graph": {"name": "minimal"},
        "links": [],
        "multigraph": True,
        "nodes": [],
    }

    skeleton = decode_skeleton(json.dumps(minimal))
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
                "source": {"py/id": 2},  # Reference to Node B
                "target": {
                    "py/object": "sleap.skeleton.Node",
                    "py/state": {"py/tuple": ["C", 1.0]},
                },
                "type": {"py/id": 3},  # Reference to first edge type
            },
        ],
        "multigraph": True,
        "nodes": [{"id": {"py/id": 1}}, {"id": {"py/id": 2}}, {"id": {"py/id": 3}}],
    }

    skeleton = decode_skeleton(json.dumps(with_refs))
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

    # The exact indices depend on the global node ordering, so we just verify
    # they're valid

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


def test_slp_decoder_duplicate_symmetries():
    """Test that SLP decoder deduplicates symmetries from legacy files."""
    # Create a mock metadata dict with duplicate symmetries
    # (like legacy SLEAP files that store each symmetry twice)
    metadata = {
        "skeletons": [
            {
                "graph": {"name": "test_skeleton"},
                "nodes": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
                "links": [
                    # Regular edge
                    {
                        "source": 0,
                        "target": 1,
                        "type": {
                            "py/reduce": [{"py/type": "EdgeType"}, {"py/tuple": [1]}]
                        },
                    },
                    # First symmetry (0, 2)
                    {
                        "source": 0,
                        "target": 2,
                        "type": {
                            "py/reduce": [{"py/type": "EdgeType"}, {"py/tuple": [2]}]
                        },
                    },
                    # Duplicate of first symmetry (2, 0) - should be deduplicated
                    {"source": 2, "target": 0, "type": {"py/id": 2}},
                    # Second symmetry (1, 3)
                    {"source": 1, "target": 3, "type": {"py/id": 2}},
                    # Duplicate of second symmetry (3, 1) - should be deduplicated
                    {"source": 3, "target": 1, "type": {"py/id": 2}},
                ],
            }
        ],
        "nodes": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}],
    }
    node_names = ["A", "B", "C", "D"]

    decoder = SkeletonSLPDecoder()
    skeletons = decoder.decode(metadata, node_names)

    assert len(skeletons) == 1
    skeleton = skeletons[0]

    # Should have only 2 unique symmetries, not 4
    assert len(skeleton.symmetries) == 2

    # Check that the correct symmetries are present
    sym_pairs = set()
    for sym in skeleton.symmetries:
        nodes = list(sym.nodes)
        pair = tuple(sorted([nodes[0].name, nodes[1].name]))
        sym_pairs.add(pair)

    assert sym_pairs == {("A", "C"), ("B", "D")}


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
    """Test round-trip encoding/decoding of fly32 skeleton.

    Tests skeletons with non-sequential py/ids.
    """
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
        assert o_node.name == r_node.name, (
            f"Node {i} mismatch: {o_node.name} != {r_node.name}"
        )

    # Verify edges preserved - check both edge objects and edge names
    original_edge_names = original.edge_names
    reloaded_edge_names = reloaded.edge_names

    assert original_edge_names == reloaded_edge_names, (
        "Edge names don't match after round-trip"
    )

    for i, (o_edge, r_edge) in enumerate(zip(original.edges, reloaded.edges)):
        assert o_edge.source.name == r_edge.source.name, f"Edge {i} source mismatch"
        assert o_edge.destination.name == r_edge.destination.name, (
            f"Edge {i} destination mismatch"
        )

    # Verify some specific critical edges
    # For fly32, neck should connect to thorax, not eyeL or other nodes
    assert ("neck", "thorax") in original_edge_names
    assert ("head", "neck") in original_edge_names


def test_training_config_decode(training_config_fly32, skeleton_json_fly32):
    """Test decoding a training config with embedded skeleton data."""
    # Test loading standalone skeleton file
    with open(skeleton_json_fly32, "r") as f:
        skeleton_data = json.load(f)
    skeletons = decode_skeleton(json.dumps(skeleton_data))
    assert len(skeletons) == 1
    skeleton = skeletons[0]
    expected_name = (
        "M:/talmo/data/leap_datasets/BermanFlies/"
        "2018-05-03_cluster-sampled.k=10,n=150.labels.mat"
    )
    assert skeleton.name == expected_name
    assert len(skeleton.nodes) == 32

    # Test loading skeleton from training config
    with open(training_config_fly32, "r") as f:
        data = json.load(f)

    skel_data = data["data"]["labels"]["skeletons"]
    skels_cfg = decode_skeleton(json.dumps(skel_data))
    assert len(skels_cfg) == 1
    skeleton_cfg = skels_cfg[0]

    # Verify the skeletons match
    assert skeleton.name == skeleton_cfg.name
    assert len(skeleton.nodes) == len(skeleton_cfg.nodes)

    # Verify node names match (order may differ between formats)
    skeleton_node_names = set(n.name for n in skeleton.nodes)
    skeleton_cfg_node_names = set(n.name for n in skeleton_cfg.nodes)
    assert skeleton_node_names == skeleton_cfg_node_names

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

    # Verify edges match between standalone and config
    assert len(skeleton.edges) == len(skeleton_cfg.edges)
    skeleton_edge_names = skeleton.edge_names
    skeleton_cfg_edge_names = skeleton_cfg.edge_names

    # Check that all edges match
    assert skeleton_edge_names == skeleton_cfg_edge_names

    # Verify some specific edges exist
    assert ("head", "eyeL") in skeleton_edge_names
    assert ("head", "eyeR") in skeleton_edge_names
    assert ("head", "neck") in skeleton_edge_names
    assert ("neck", "thorax") in skeleton_edge_names

    # Check edges are preserved correctly
    for i, (e1, e2) in enumerate(zip(skeleton.edges, skeleton_cfg.edges)):
        assert e1.source.name == e2.source.name, f"Edge {i} source mismatch"
        assert e1.destination.name == e2.destination.name, (
            f"Edge {i} destination mismatch"
        )


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
    with pytest.raises(ValueError):  # None is not valid skeleton data
        sio.load_skeleton(config_path)


def test_skeleton_decoder_invalid_input_types():
    """Test skeleton decoder with invalid input types."""
    decoder = SkeletonDecoder()

    # Test with non-dict input (should raise TypeError)
    with pytest.raises(TypeError, match="Skeleton data must be a dictionary"):
        decoder._decode_skeleton("not a dict")

    with pytest.raises(TypeError, match="Skeleton data must be a dictionary"):
        decoder._decode_skeleton(123)

    with pytest.raises(TypeError, match="Skeleton data must be a dictionary"):
        decoder._decode_skeleton([1, 2, 3])


def test_skeleton_node_order_from_training_config(
    training_config_13pt_fly, slp_skeleton_13pt_fly
):
    """Test that skeleton node order is preserved when loading from training config.

    This test verifies the fix for the bug where node order was incorrect when
    loading skeletons from training configs with non-sequential py/ids.
    """
    # Load skeleton from training config
    skeleton_from_config = sio.load_skeleton(training_config_13pt_fly)
    if isinstance(skeleton_from_config, list):
        skeleton_from_config = skeleton_from_config[0]

    # Load skeleton from SLP file (ground truth)
    labels = sio.load_slp(slp_skeleton_13pt_fly)
    skeleton_from_slp = labels.skeletons[0]

    # Expected node order for 13-point fly skeleton
    expected_node_names = [
        "head",
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
        "eyeL",
        "eyeR",
    ]

    # Check node names and order
    assert skeleton_from_config.node_names == expected_node_names
    assert skeleton_from_slp.node_names == expected_node_names
    assert skeleton_from_config.node_names == skeleton_from_slp.node_names

    # Expected edges (thorax should connect to body parts, not eyeL)
    expected_edges = [
        ("head", "eyeL"),
        ("head", "eyeR"),
        ("thorax", "head"),
        ("thorax", "abdomen"),
        ("thorax", "wingL"),
        ("thorax", "wingR"),
        ("thorax", "forelegL4"),
        ("thorax", "forelegR4"),
        ("thorax", "midlegL4"),
        ("thorax", "midlegR4"),
        ("thorax", "hindlegL4"),
        ("thorax", "hindlegR4"),
    ]

    # Check edges
    assert skeleton_from_config.edge_names == expected_edges
    assert skeleton_from_slp.edge_names == expected_edges
    assert skeleton_from_config.edge_names == skeleton_from_slp.edge_names

    # Verify specific edge connections that were problematic
    # These edges should come from thorax, not eyeL
    problematic_edges = [
        ("thorax", "abdomen"),
        ("thorax", "wingL"),
        ("thorax", "wingR"),
        ("thorax", "forelegL4"),
        ("thorax", "forelegR4"),
        ("thorax", "midlegL4"),
        ("thorax", "midlegR4"),
        ("thorax", "hindlegL4"),
        ("thorax", "hindlegR4"),
    ]

    for edge in problematic_edges:
        assert edge in skeleton_from_config.edge_names
        assert edge in skeleton_from_slp.edge_names

    # Ensure eyeL is NOT connecting to body parts
    eyeL_edges = [
        (e.source.name, e.destination.name)
        for e in skeleton_from_config.edges
        if e.source.name == "eyeL"
    ]
    assert len(eyeL_edges) == 0  # eyeL should not be source of any edges


def test_skeleton_json_direct_decode(skeleton_json_13pt_fly):
    """Test decoding the skeleton JSON directly."""
    with open(skeleton_json_13pt_fly, "r") as f:
        skeleton_data = json.load(f)

    skeleton = decode_skeleton(json.dumps(skeleton_data))

    # Check that py/id 5 maps to thorax (not eyeL)
    # This is the core of the bug - py/id 5 should be thorax
    assert skeleton.nodes[1].name == "thorax"  # Second node should be thorax

    # Verify edges from thorax
    thorax_edges = [
        (e.source.name, e.destination.name)
        for e in skeleton.edges
        if e.source.name == "thorax"
    ]
    assert len(thorax_edges) == 10  # thorax connects to 10 body parts

    # Check specific edges
    assert ("thorax", "head") in thorax_edges
    assert ("thorax", "abdomen") in thorax_edges
    assert ("thorax", "wingL") in thorax_edges


def test_py_id_mapping_with_non_sequential_ids(skeleton_json_13pt_fly):
    """Test that non-sequential py/ids are handled correctly.

    The 13-point fly skeleton has py/ids: [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2, 4]
    which are non-sequential and demonstrate the bug.
    """
    with open(skeleton_json_13pt_fly, "r") as f:
        skeleton_data = json.load(f)

    # Extract py/ids from nodes section
    node_pyids = [node["id"]["py/id"] for node in skeleton_data["nodes"]]
    assert node_pyids == [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2, 4]

    # Decode skeleton
    skeleton = decode_skeleton(json.dumps(skeleton_data))

    # The node order should match the nodes section order, with correct mapping
    expected_mapping = {
        1: "head",
        5: "thorax",
        6: "abdomen",
        7: "wingL",
        8: "wingR",
        9: "forelegL4",
        10: "forelegR4",
        11: "midlegL4",
        12: "midlegR4",
        13: "hindlegL4",
        14: "hindlegR4",
        2: "eyeL",
        4: "eyeR",
    }

    # Verify each node is in the correct position
    for i, (py_id, expected_name) in enumerate(zip(node_pyids, skeleton.node_names)):
        assert skeleton.nodes[i].name == expected_mapping[py_id]


def test_all_skeleton_formats_consistency(
    training_config_13pt_fly, skeleton_json_13pt_fly, slp_skeleton_13pt_fly
):
    """Test that all formats produce the same skeleton."""
    # Load from training config
    skeleton_from_config = sio.load_skeleton(training_config_13pt_fly)
    if isinstance(skeleton_from_config, list):
        skeleton_from_config = skeleton_from_config[0]

    # Load from standalone JSON
    skeleton_from_json = sio.load_skeleton(skeleton_json_13pt_fly)

    # Load from SLP
    labels = sio.load_slp(slp_skeleton_13pt_fly)
    skeleton_from_slp = labels.skeletons[0]

    # All should have the same structure
    assert skeleton_from_config.node_names == skeleton_from_json.node_names
    assert skeleton_from_config.node_names == skeleton_from_slp.node_names

    assert skeleton_from_config.edge_names == skeleton_from_json.edge_names
    assert skeleton_from_config.edge_names == skeleton_from_slp.edge_names

    # Verify the number of nodes and edges
    assert len(skeleton_from_config.nodes) == 13
    assert len(skeleton_from_config.edges) == 12

    # All edges should be regular edges (no symmetries in the edge list)
    assert all(
        hasattr(edge, "source") and hasattr(edge, "destination")
        for edge in skeleton_from_config.edges
    )


def test_clip_2nodes_slp(clip_2nodes_slp):
    """Test loading the 2-node clip SLP file."""
    labels = sio.load_slp(clip_2nodes_slp)
    assert labels is not None
    assert len(labels.skeletons) == 1
    assert len(labels.skeletons[0].nodes) == 2


def test_load_single_node_training_config(single_node_training_config):
    """Test loading training config with single-node skeleton and no edges.

    This tests the fix for the decoder bug where single-node skeletons
    with no edges would fail to load because the decoder only processed
    nodes that appeared in links, but single-node skeletons have empty links.
    """
    result = sio.load_skeleton(single_node_training_config)

    # Should return a list with one skeleton (training configs return lists)
    assert isinstance(result, list)
    assert len(result) == 1

    skeleton = result[0]

    # Verify skeleton structure
    assert skeleton.name == "Skeleton-1"
    assert len(skeleton.nodes) == 1
    assert len(skeleton.edges) == 0
    assert len(skeleton.symmetries) == 0

    # Verify the single node
    assert skeleton.nodes[0].name == "r0"

    # Test that this previously would have failed (now passes with fix)
    # Single node with no edges should be properly loaded


def test_decode_training_config_invalid_format():
    """Test that decode_training_config raises ValueError for invalid input."""
    from sleap_io.io.skeleton import decode_training_config

    # Test with completely invalid data
    with pytest.raises(ValueError, match="Invalid training config format"):
        decode_training_config({"invalid": "data"})

    # Test with missing 'data' key
    with pytest.raises(ValueError, match="Invalid training config format"):
        decode_training_config({"some_other_key": {}})

    # Test with 'data' but missing 'labels'
    with pytest.raises(ValueError, match="Invalid training config format"):
        decode_training_config({"data": {"other": "stuff"}})

    # Test with 'data.labels' but missing 'skeletons'
    with pytest.raises(ValueError, match="Invalid training config format"):
        decode_training_config({"data": {"labels": {"other": "stuff"}}})

    # Test with non-dict input
    with pytest.raises(ValueError, match="Invalid training config format"):
        decode_training_config("not a dict")


def test_slp_decoder_edge_type_pyid_resolution():
    """Test that SLP decoder correctly resolves py/id references in edge types.

    This test catches a bug where py/id values were treated as direct edge type
    values instead of references to previously defined edge types.

    When a symmetry edge (EdgeType=2) is defined before a regular edge (EdgeType=1):
    - The first py/reduce creates EdgeType(2) and assigns it py/id=1
    - The second py/reduce creates EdgeType(1) and assigns it py/id=2
    - References to py/id=1 should resolve to EdgeType(2), not EdgeType(1)
    - References to py/id=2 should resolve to EdgeType(1), not EdgeType(2)

    The buggy code treats py/id values as direct edge type values, causing
    edges and symmetries to be swapped.
    """
    # Create metadata where symmetry (EdgeType=2) is defined before edge (EdgeType=1)
    skeleton_metadata = {
        "directed": True,
        "graph": {"name": "Skeleton-12", "num_edges_inserted": 5},
        "links": [
            # Link 0: Creates EdgeType(2)=SYMMETRY, gets py/id=1
            {
                "key": 0,
                "source": 6,
                "target": 1,
                "type": {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [2]},
                    ]
                },
            },
            # Link 1: References py/id=1 -> should be EdgeType(2)=SYMMETRY
            {"key": 0, "source": 1, "target": 6, "type": {"py/id": 1}},
            # Link 2: Creates EdgeType(1)=EDGE, gets py/id=2
            {
                "edge_insert_idx": 0,
                "key": 0,
                "source": 3,
                "target": 6,
                "type": {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [1]},
                    ]
                },
            },
            # Link 3: References py/id=2 -> should be EdgeType(1)=EDGE
            {
                "edge_insert_idx": 1,
                "key": 0,
                "source": 3,
                "target": 1,
                "type": {"py/id": 2},
            },
            # Link 4: References py/id=2 -> should be EdgeType(1)=EDGE
            {
                "edge_insert_idx": 2,
                "key": 0,
                "source": 3,
                "target": 4,
                "type": {"py/id": 2},
            },
        ],
        "multigraph": True,
        "nodes": [{"id": 6}, {"id": 1}, {"id": 3}, {"id": 4}, {"id": 0}],
    }

    # Create metadata structure as it appears in .slp files
    # The "nodes" list is a global list of all nodes across all skeletons
    metadata = {
        "skeletons": [skeleton_metadata],
        "nodes": [
            {"name": "tailend", "weight": 1.0},  # index 0
            {"name": "right", "weight": 1.0},  # index 1
            {"name": "tailend", "weight": 1.0},  # index 2 (duplicate, not used)
            {"name": "nose", "weight": 1.0},  # index 3
            {"name": "tailstart", "weight": 1.0},  # index 4
            {"name": "tailend", "weight": 1.0},  # index 5 (duplicate, not used)
            {"name": "left", "weight": 1.0},  # index 6
        ],
    }

    # Extract node names the same way as sleap_io.io.slp.read_skeletons()
    node_names = [x["name"] for x in metadata["nodes"]]

    # Decode using SLP decoder
    decoder = SkeletonSLPDecoder()
    skeletons = decoder.decode(metadata, node_names)
    skeleton = skeletons[0]

    # Verify skeleton structure
    assert skeleton.name == "Skeleton-12"
    assert len(skeleton.nodes) == 5

    # Verify edges - should have 3 edges from nose to left/right/tailstart
    assert len(skeleton.edges) == 3
    edge_pairs = [(e.source.name, e.destination.name) for e in skeleton.edges]
    assert ("nose", "left") in edge_pairs
    assert ("nose", "right") in edge_pairs
    assert ("nose", "tailstart") in edge_pairs

    # Verify symmetries - should have only 1 symmetry between left and right
    # (deduplicated from the two directional links)
    assert len(skeleton.symmetries) == 1
    sym_nodes = {n.name for n in skeleton.symmetries[0].nodes}
    assert sym_nodes == {"left", "right"}
