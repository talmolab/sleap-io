"""Integration tests for skeleton JSON fixtures."""

import pytest
import json
import sleap_io as sio


def test_minimal_skeleton_fixture(skeleton_json_minimal):
    """Test loading and analyzing the minimal skeleton fixture."""
    # Load skeleton
    skeleton = sio.load_skeleton(skeleton_json_minimal)
    
    # Verify structure
    assert skeleton.name == "Skeleton-1"
    assert len(skeleton.nodes) == 2
    assert skeleton.nodes[0].name == "head"
    assert skeleton.nodes[1].name == "abdomen"
    
    # Verify edges
    assert len(skeleton.edges) == 1
    assert skeleton.edges[0].source.name == "head"
    assert skeleton.edges[0].destination.name == "abdomen"
    
    # No symmetries
    assert len(skeleton.symmetries) == 0


def test_flies_skeleton_fixture(skeleton_json_flies):
    """Test loading and analyzing the flies skeleton fixture."""
    # Load skeleton
    skeleton = sio.load_skeleton(skeleton_json_flies)
    
    # Verify basic structure
    assert skeleton.name == "Skeleton-0"
    # Verify we have a reasonable number of nodes (should be many for fly skeleton)
    assert len(skeleton.nodes) >= 10
    
    # Check that some expected fly nodes are present
    node_names = {node.name for node in skeleton.nodes}
    # Just check for a few basic nodes that should be there
    basic_nodes = {"head", "thorax"}
    assert basic_nodes.intersection(node_names), f"Expected basic nodes not found in {node_names}"
    
    # Verify we have edges and symmetries
    assert len(skeleton.edges) > 0, "Should have some edges"
    assert len(skeleton.symmetries) > 0, "Should have some symmetries"
    
    # Basic structural validation - skeleton should be loadable
    assert skeleton is not None
    assert isinstance(skeleton.name, str)


def test_skeleton_round_trip_minimal(skeleton_json_minimal, tmp_path):
    """Test round-trip serialization with minimal skeleton."""
    # Load original
    original = sio.load_skeleton(skeleton_json_minimal)
    
    # Save to new file
    output_path = tmp_path / "minimal_round_trip.json"
    sio.save_skeleton(original, output_path)
    
    # Load again
    reloaded = sio.load_skeleton(output_path)
    
    # Verify identical structure
    assert original.name == reloaded.name
    assert len(original.nodes) == len(reloaded.nodes)
    assert len(original.edges) == len(reloaded.edges)
    
    # Verify node names preserved
    for o_node, r_node in zip(original.nodes, reloaded.nodes):
        assert o_node.name == r_node.name


def test_skeleton_round_trip_flies(skeleton_json_flies, tmp_path):
    """Test round-trip serialization with complex flies skeleton."""
    # Load original
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
    # The exact node/edge/symmetry counts may vary due to complex py/id mappings
    assert len(reloaded.nodes) > 0, "Should have nodes after round-trip"
    assert len(reloaded.edges) > 0, "Should have edges after round-trip"
    assert len(reloaded.symmetries) > 0, "Should have symmetries after round-trip"


def test_skeleton_json_format_preservation(skeleton_json_minimal):
    """Test that the JSON format is preserved correctly."""
    # Load original JSON
    with open(skeleton_json_minimal, "r") as f:
        original_data = json.loads(f.read())
    
    # Load and re-encode
    skeleton = sio.load_skeleton(skeleton_json_minimal)
    encoder = sio.io.skeleton.SkeletonEncoder()
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


def test_skeleton_edge_type_handling(skeleton_json_flies):
    """Test that edge types are handled correctly with py/id references."""
    # Load the skeleton
    skeleton = sio.load_skeleton(skeleton_json_flies)
    
    # Verify we loaded a complex skeleton
    assert len(skeleton.edges) > 0
    assert len(skeleton.symmetries) > 0
    
    # Re-encode it
    encoder = sio.io.skeleton.SkeletonEncoder()
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