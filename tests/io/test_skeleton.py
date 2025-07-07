"""Tests for standalone skeleton JSON I/O."""

import pytest
import json
import tempfile
from pathlib import Path
import sleap_io as sio
from sleap_io.io.skeleton import SkeletonDecoder, SkeletonEncoder


class TestSkeletonDecoder:
    """Test skeleton decoding from JSON format."""

    def test_decode_simple_skeleton(self):
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

    def test_decode_skeleton_with_symmetry(self):
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

    def test_decode_dict_state_format(self):
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

    def test_decode_from_json_string(self):
        """Test decoding from JSON string."""
        json_str = """{"directed": true, "graph": {"name": "test"}, "links": [], "nodes": []}"""

        decoder = SkeletonDecoder()
        skeleton = decoder.decode(json_str)

        assert skeleton.name == "test"
        assert len(skeleton.nodes) == 0

    def test_decode_multiple_skeletons(self):
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

    def test_decode_real_skeleton_file(self):
        """Test decoding the actual test skeleton file."""
        skeleton = sio.load_skeleton(
            "tests/data/slp/labels.v002.rel_paths.skeleton.json"
        )

        assert skeleton.name == "Skeleton-1"
        assert len(skeleton.nodes) == 2
        assert skeleton.nodes[0].name == "head"
        assert skeleton.nodes[1].name == "abdomen"
        assert len(skeleton.edges) == 1


class TestSkeletonEncoder:
    """Test skeleton encoding to JSON format."""

    def test_encode_simple_skeleton(self):
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

    def test_encode_skeleton_with_symmetry(self):
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
        assert (
            data["links"][0]["type"]["py/reduce"][1]["py/tuple"][0] == 2
        )  # Symmetry type

    def test_encode_edge_type_references(self):
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

    def test_encode_multiple_skeletons(self):
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

    def test_dictionary_sorting(self):
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


class TestRoundTrip:
    """Test round-trip encoding and decoding."""

    def test_simple_round_trip(self):
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

    def test_complex_round_trip(self):
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

    def test_file_io_round_trip(self, tmp_path):
        """Test round-trip through file I/O."""
        # Create skeleton
        nodes = [sio.Node("A"), sio.Node("B"), sio.Node("C")]
        skeleton1 = sio.Skeleton(
            nodes=nodes,
            edges=[sio.Edge(nodes[0], nodes[1]), sio.Edge(nodes[1], nodes[2])],
            name="file-test",
        )

        # Save and load
        filename = tmp_path / "test_skeleton.json"
        sio.save_skeleton(skeleton1, filename)
        skeleton2 = sio.load_skeleton(filename)

        assert skeleton1.name == skeleton2.name
        assert len(skeleton1.nodes) == len(skeleton2.nodes)
        assert len(skeleton1.edges) == len(skeleton2.edges)

    def test_real_file_round_trip(self, tmp_path):
        """Test round-trip with the real test file."""
        # Load original
        skeleton1 = sio.load_skeleton(
            "tests/data/slp/labels.v002.rel_paths.skeleton.json"
        )

        # Save to new file
        new_file = tmp_path / "round_trip_skeleton.json"
        sio.save_skeleton(skeleton1, new_file)

        # Load again
        skeleton2 = sio.load_skeleton(new_file)

        assert skeleton1.name == skeleton2.name
        assert len(skeleton1.nodes) == len(skeleton2.nodes)
        assert all(
            n1.name == n2.name for n1, n2 in zip(skeleton1.nodes, skeleton2.nodes)
        )
        assert len(skeleton1.edges) == len(skeleton2.edges)
