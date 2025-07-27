"""Tests for sleap_io.io.coco module."""

from pathlib import Path

import numpy as np
import pytest

import sleap_io as sio
from sleap_io.io import coco

# Import COCO fixtures
pytest_plugins = ["tests.fixtures.coco"]


class TestCOCOBasicLoading:
    """Test basic COCO dataset loading functionality."""

    def test_parse_coco_json(self, coco_flat_images):
        """Test parsing COCO JSON annotation file."""
        annotations_path = Path(coco_flat_images) / "annotations.json"
        data = coco.parse_coco_json(annotations_path)

        # Check required fields
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

        # Check content
        assert len(data["images"]) == 3
        assert len(data["annotations"]) == 3
        assert len(data["categories"]) == 1

        # Check category has keypoints
        category = data["categories"][0]
        assert "keypoints" in category
        assert len(category["keypoints"]) == 17
        assert category["name"] == "mouse"

    def test_create_skeleton_from_category(self, coco_flat_images):
        """Test skeleton creation from COCO category."""
        annotations_path = Path(coco_flat_images) / "annotations.json"
        data = coco.parse_coco_json(annotations_path)
        category = data["categories"][0]

        skeleton = coco.create_skeleton_from_category(category)

        assert skeleton.name == "mouse"
        assert len(skeleton.nodes) == 17
        assert len(skeleton.edges) > 0

        # Check node names
        node_names = [node.name for node in skeleton.nodes]
        assert "nose" in node_names
        assert "left_ear" in node_names
        assert "tail_tip" in node_names

    def test_decode_keypoints(self, coco_flat_images):
        """Test keypoint decoding functionality."""
        annotations_path = Path(coco_flat_images) / "annotations.json"
        data = coco.parse_coco_json(annotations_path)

        category = data["categories"][0]
        skeleton = coco.create_skeleton_from_category(category)

        annotation = data["annotations"][0]
        keypoints = annotation["keypoints"]

        points_array = coco.decode_keypoints(keypoints, len(skeleton.nodes), skeleton)

        assert points_array.shape == (17, 3)  # 17 keypoints, 3 values each
        assert points_array.dtype == np.float32

        # Check first point (should be at coordinates 100, 100 with visibility 2)
        assert points_array[0, 0] == 100.0  # x
        assert points_array[0, 1] == 100.0  # y
        assert points_array[0, 2] == 1.0  # visible (converted from visibility=2)


class TestCOCODatasetVariants:
    """Test loading different COCO dataset variants."""

    def test_flat_images(self, coco_flat_images):
        """Test loading flat images variant."""
        labels = coco.read_labels(Path(coco_flat_images) / "annotations.json")

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 1
        assert labels.skeletons[0].name == "mouse"

        # Check instances
        total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
        assert total_instances == 3

        # Check first instance
        instance = labels.labeled_frames[0].instances[0]
        assert len(instance.points) == 17
        assert instance.skeleton.name == "mouse"

    def test_category_folders(self, coco_category_folders):
        """Test loading category folders variant."""
        labels = coco.read_labels(Path(coco_category_folders) / "annotations.json")

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 2

        skeleton_names = [skel.name for skel in labels.skeletons]
        assert "mouse" in skeleton_names
        assert "fly" in skeleton_names

        # Check instances
        total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
        assert total_instances == 3

    def test_mixed_animals(self, coco_mixed_animals):
        """Test loading mixed animals variant."""
        labels = coco.read_labels(Path(coco_mixed_animals) / "annotations.json")

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 2

        # Should have 2 instances per frame (mouse + fly)
        total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
        assert total_instances == 6

        # Check that we have both skeleton types
        frame = labels.labeled_frames[0]
        assert len(frame.instances) == 2

        skeleton_names = [inst.skeleton.name for inst in frame.instances]
        assert "mouse" in skeleton_names
        assert "fly" in skeleton_names

    def test_visibility_binary(self, coco_visibility_binary):
        """Test loading binary visibility encoding."""
        labels = coco.read_labels(Path(coco_visibility_binary) / "annotations.json")

        assert len(labels.labeled_frames) == 3
        instance = labels.labeled_frames[0].instances[0]

        # Check that some points are not visible
        visible_count = sum(1 for p in instance.points if p["visible"])
        total_points = len(instance.points)

        assert visible_count < total_points  # Some points should be invisible
        assert visible_count > 0  # Some points should be visible

    def test_visibility_ternary(self, coco_visibility_ternary):
        """Test loading ternary visibility encoding."""
        labels = coco.read_labels(Path(coco_visibility_ternary) / "annotations.json")

        assert len(labels.labeled_frames) == 3
        instance = labels.labeled_frames[0].instances[0]

        # Check that some points are not visible
        visible_count = sum(1 for p in instance.points if p["visible"])
        total_points = len(instance.points)

        assert visible_count < total_points  # Some points should be invisible
        assert visible_count > 0  # Some points should be visible

    def test_nested_paths(self, coco_nested_paths):
        """Test loading nested directory structure."""
        labels = coco.read_labels(Path(coco_nested_paths) / "annotations.json")

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 1

        # Check that nested image paths are resolved correctly
        for frame in labels.labeled_frames:
            assert frame.video is not None

    def test_multi_source(self, coco_multi_source):
        """Test loading multi-source variant."""
        labels = coco.read_labels(Path(coco_multi_source) / "annotations.json")

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 1

        # Check instances
        total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
        assert total_instances == 3


class TestCOCOMainInterface:
    """Test COCO loading through main sleap_io interface."""

    def test_auto_detection(self, coco_flat_images):
        """Test automatic COCO format detection."""
        annotations_path = Path(coco_flat_images) / "annotations.json"
        labels = sio.load_file(annotations_path)

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 1
        assert labels.skeletons[0].name == "mouse"

    def test_explicit_format(self, coco_mixed_animals):
        """Test explicit COCO format specification."""
        annotations_path = Path(coco_mixed_animals) / "annotations.json"
        labels = sio.load_file(annotations_path, format="coco")

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 2

    def test_dataset_root_parameter(self, coco_flat_images):
        """Test dataset_root parameter."""
        annotations_path = Path(coco_flat_images) / "annotations.json"
        dataset_root = Path(coco_flat_images)

        labels = sio.load_file(annotations_path, dataset_root=str(dataset_root))

        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 1


class TestCOCOMultiSplit:
    """Test multi-split COCO dataset loading."""

    def test_read_labels_set_single_dir(self, coco_flat_images):
        """Test reading labels set from single directory."""
        labels_dict = coco.read_labels_set(coco_flat_images)

        assert len(labels_dict) == 1
        assert "annotations" in labels_dict

        labels = labels_dict["annotations"]
        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 1

    def test_read_labels_set_specific_files(self, tmp_path):
        """Test reading labels set with specific files."""
        # Copy two different annotation files to test directory
        import shutil

        flat_source = Path("tests/data/coco/flat_images/annotations.json")
        mixed_source = Path("tests/data/coco/mixed_animals/annotations.json")

        shutil.copy2(flat_source, tmp_path / "train.json")
        shutil.copy2(mixed_source, tmp_path / "val.json")

        # Also copy some images for the test
        (tmp_path / "images").mkdir()
        shutil.copy2(
            "tests/data/videos/imgs/img.00.jpg", tmp_path / "images" / "image_001.jpg"
        )
        shutil.copy2(
            "tests/data/videos/imgs/img.01.jpg", tmp_path / "images" / "image_002.jpg"
        )
        shutil.copy2(
            "tests/data/videos/imgs/img.02.jpg", tmp_path / "images" / "image_003.jpg"
        )

        labels_dict = coco.read_labels_set(
            tmp_path, json_files=["train.json", "val.json"]
        )

        assert len(labels_dict) == 2
        assert "train" in labels_dict
        assert "val" in labels_dict

        # Train should have 1 skeleton (mouse only)
        assert len(labels_dict["train"].skeletons) == 1

        # Val should have 2 skeletons (mouse + fly)
        assert len(labels_dict["val"].skeletons) == 2


class TestCOCOErrorHandling:
    """Test error handling in COCO loading."""

    def test_missing_file(self):
        """Test handling of missing annotation file."""
        with pytest.raises(FileNotFoundError):
            coco.read_labels("nonexistent.json")

    def test_invalid_json_structure(self, tmp_path):
        """Test handling of invalid JSON structure."""
        # Create invalid JSON file
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text('{"invalid": "structure"}')

        with pytest.raises(ValueError, match="Missing required COCO field"):
            coco.parse_coco_json(invalid_json)

    def test_non_pose_dataset(self, tmp_path):
        """Test handling of detection-only COCO dataset."""
        # Create detection-only COCO file
        detection_json = tmp_path / "detection.json"
        detection_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}],  # No keypoints field
        }

        import json

        with open(detection_json, "w") as f:
            json.dump(detection_data, f)

        with pytest.raises(ValueError, match="No keypoint definitions found"):
            coco.parse_coco_json(detection_json)

    def test_missing_images(self, tmp_path, coco_flat_images):
        """Test handling of missing image files."""
        # Copy annotations but not images
        import shutil

        annotations_src = Path(coco_flat_images) / "annotations.json"
        annotations_dst = tmp_path / "annotations.json"
        shutil.copy2(annotations_src, annotations_dst)

        # Should not raise error, but should skip missing images
        labels = coco.read_labels(annotations_dst)

        # Should have 0 frames since no images exist
        assert len(labels.labeled_frames) == 0


class TestCOCODataStructures:
    """Test COCO data structure handling."""

    def test_skeleton_edges(self, coco_flat_images):
        """Test that skeleton edges are created correctly."""
        labels = coco.read_labels(Path(coco_flat_images) / "annotations.json")
        skeleton = labels.skeletons[0]

        assert len(skeleton.edges) > 0

        # Check that edges connect valid nodes
        for edge in skeleton.edges:
            assert edge.source in skeleton.nodes
            assert edge.destination in skeleton.nodes

    def test_point_coordinates(self, coco_flat_images):
        """Test that point coordinates are loaded correctly."""
        labels = coco.read_labels(Path(coco_flat_images) / "annotations.json")
        instance = labels.labeled_frames[0].instances[0]

        # Check first point coordinates (should match fixture data)
        first_point = instance.points[0]
        assert first_point["xy"][0] == 100.0
        assert first_point["xy"][1] == 100.0
        assert first_point["visible"]
        assert first_point["name"] == "nose"

    def test_point_visibility_handling(self, coco_visibility_binary):
        """Test that point visibility is handled correctly."""
        labels = coco.read_labels(Path(coco_visibility_binary) / "annotations.json")
        instance = labels.labeled_frames[0].instances[0]

        # Should have mix of visible and invisible points
        visible_points = [p for p in instance.points if p["visible"]]
        invisible_points = [p for p in instance.points if not p["visible"]]

        assert len(visible_points) > 0
        assert len(invisible_points) > 0
        assert len(visible_points) + len(invisible_points) == len(instance.points)
