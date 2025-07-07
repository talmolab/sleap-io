"""Test backward compatibility with existing skeleton JSON files."""

import pytest
import json
import sleap_io as sio
from pathlib import Path


class TestBackwardCompatibility:
    """Test that we can load existing skeleton JSON files."""

    def test_load_existing_skeleton_file(self):
        """Test loading the existing test skeleton file."""
        skeleton_path = "tests/data/slp/labels.v002.rel_paths.skeleton.json"

        # Should load without errors
        skeleton = sio.load_skeleton(skeleton_path)

        # Verify content
        assert skeleton.name == "Skeleton-1"
        assert len(skeleton.nodes) == 2
        assert skeleton.nodes[0].name == "head"
        assert skeleton.nodes[1].name == "abdomen"
        assert len(skeleton.edges) == 1
        assert skeleton.edges[0].source.name == "head"
        assert skeleton.edges[0].destination.name == "abdomen"

    def test_round_trip_preserves_structure(self):
        """Test that round-trip encoding preserves the JSON structure."""
        skeleton_path = "tests/data/slp/labels.v002.rel_paths.skeleton.json"

        # Load original
        with open(skeleton_path, "r") as f:
            original_data = json.loads(f.read())

        # Load, encode, and parse
        skeleton = sio.load_skeleton(skeleton_path)
        encoder = sio.io.skeleton.SkeletonEncoder()
        encoded_json = encoder.encode(skeleton)
        encoded_data = json.loads(encoded_json)

        # Check that all top-level keys are preserved
        assert set(original_data.keys()) == set(encoded_data.keys())

        # Check graph metadata
        assert original_data["graph"] == encoded_data["graph"]
        assert original_data["directed"] == encoded_data["directed"]
        assert original_data["multigraph"] == encoded_data["multigraph"]

        # Check structure
        assert len(original_data["links"]) == len(encoded_data["links"])
        assert len(original_data["nodes"]) == len(encoded_data["nodes"])

        # Check link structure (should have same keys)
        orig_link = original_data["links"][0]
        enc_link = encoded_data["links"][0]
        assert set(orig_link.keys()) == set(enc_link.keys())

        # Check node data is preserved
        orig_source = orig_link["source"]["py/state"]["py/tuple"]
        enc_source = enc_link["source"]["py/state"]["py/tuple"]
        assert orig_source[0] == enc_source[0]  # name
        assert orig_source[1] == enc_source[1]  # weight

    def test_decode_variations(self):
        """Test decoding various valid skeleton JSON formats."""
        # Test minimal skeleton
        minimal = {
            "directed": True,
            "graph": {"name": "minimal"},
            "links": [],
            "multigraph": True,
            "nodes": [],
        }

        decoder = sio.io.skeleton.SkeletonDecoder()
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
