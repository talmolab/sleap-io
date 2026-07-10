"""Tests for sleap_io.io.coco module."""

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

import sleap_io as sio
from sleap_io.io import coco
from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.instance import Track
from sleap_io.model.labels import Labels
from sleap_io.model.mask import PredictedSegmentationMask, UserSegmentationMask
from sleap_io.model.matching import (
    IMAGE_DEDUP_VIDEO_MATCHER,
    SHAPE_VIDEO_MATCHER,
)
from sleap_io.model.roi import PredictedROI, UserROI


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

    @pytest.mark.parametrize(
        "json_name, expected_total_instances, expected_second_frame_instances",
        [
            ("annotations.json", 3, 1),
            ("annotations_negative_frame.json", 2, 0),
        ],
    )
    def test_flat_images(
        self,
        coco_flat_images,
        json_name,
        expected_total_instances,
        expected_second_frame_instances,
    ):
        """Test loading flat images, including when no frames have annotations."""
        labels = coco.read_labels(Path(coco_flat_images) / json_name)

        assert len(labels.labeled_frames) == 3

        # Check instances
        total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
        assert total_instances == expected_total_instances

        # Check first frame instance structure
        instance = labels.labeled_frames[0].instances[0]
        assert len(instance.points) == 17
        assert instance.skeleton.name == "mouse"

        # Check second frame instance count
        frame_instances = labels.labeled_frames[1].instances
        assert len(frame_instances) == expected_second_frame_instances

        # Check skeletons
        assert len(labels.skeletons) == 1
        assert labels.skeletons[0].name == "mouse"

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
        """Test that detection-only COCO datasets parse without error."""
        detection_json = tmp_path / "detection.json"
        detection_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}],  # No keypoints field
        }

        with open(detection_json, "w") as f:
            json.dump(detection_data, f)

        data = coco.parse_coco_json(detection_json)
        assert len(data["categories"]) == 1

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


def test_category_without_keypoints(tmp_path):
    """Test handling of category without keypoints field."""
    # Create a COCO file with a category missing keypoints
    coco_data = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                # Missing 'keypoints' field
                "skeleton": [[0, 1]],
            }
        ],
    }

    json_path = tmp_path / "bad_category.json"
    import json

    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="Category 'person' has no keypoint definitions"
    ):
        coco.create_skeleton_from_category(coco_data["categories"][0])


def test_missing_num_keypoints_field(tmp_path):
    """Test handling when num_keypoints field is missing."""
    import json

    import imageio.v3 as iio
    import numpy as np

    # Create test data with missing num_keypoints
    coco_data = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [50, 50, 2, 60, 60, 2],
                # num_keypoints field is missing
            }
        ],
        "categories": [
            {"id": 1, "name": "person", "keypoints": ["head", "tail"], "skeleton": []}
        ],
    }

    # Save annotation file
    json_path = tmp_path / "no_num_keypoints.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    # Create dummy image
    img_path = tmp_path / "test.jpg"
    iio.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))

    # Should still load successfully (uses skeleton node count as fallback)
    labels = coco.read_labels(json_path, tmp_path)
    assert len(labels) == 1
    assert len(labels[0].instances) == 1
    # Based on skeleton, not num_keypoints
    assert len(labels[0].instances[0].points) == 2


def test_keypoint_length_mismatch(tmp_path):
    """Test error handling for mismatched keypoint array length."""
    from sleap_io import Node, Skeleton

    # Create test data with wrong keypoint length
    keypoints = [100.0, 50.0, 2]  # Only 3 values instead of 6 (2 points * 3)
    skeleton = Skeleton([Node("head"), Node("tail")])

    with pytest.raises(ValueError, match="Keypoints length 3 doesn't match expected 6"):
        coco.decode_keypoints(keypoints, num_keypoints=2, skeleton=skeleton)


def test_skeleton_annotation_mismatch(tmp_path):
    """Test error handling when skeleton nodes don't match annotation keypoints."""
    from sleap_io import Node, Skeleton

    # Create skeleton with 3 nodes
    skeleton = Skeleton([Node("head"), Node("body"), Node("tail")])

    # But provide keypoints for only 2 points
    keypoints = [100.0, 50.0, 2, 150.0, 60.0, 2]  # 2 points * 3 values

    with pytest.raises(
        ValueError, match="Skeleton has 3 nodes but annotation has 2 keypoints"
    ):
        coco.decode_keypoints(keypoints, num_keypoints=2, skeleton=skeleton)


def test_unknown_visibility_value(tmp_path):
    """Test handling of unknown visibility values."""
    from sleap_io import Node, Skeleton

    # Create test data with unusual visibility value
    keypoints = [100.0, 50.0, 99]  # Visibility value 99 (not 0, 1, or 2)
    skeleton = Skeleton([Node("head")])

    # Should treat unknown visibility as visible
    points = coco.decode_keypoints(keypoints, num_keypoints=1, skeleton=skeleton)
    assert points.shape == (1, 3)
    assert points[0, 0] == 100.0
    assert points[0, 1] == 50.0
    assert points[0, 2]  # Treated as visible


def test_mixed_annotations_skip_non_pose(tmp_path):
    """Test that non-pose annotations are properly skipped."""
    import json

    import imageio.v3 as iio
    import numpy as np

    # Create COCO data with both pose and non-pose categories
    coco_data = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,  # Pose category
                "keypoints": [50, 50, 2, 60, 60, 2],
                "num_keypoints": 2,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,  # Non-pose category (no keypoints)
                "bbox": [10, 10, 20, 20],
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "keypoints": ["head", "tail"], "skeleton": []},
            {
                "id": 2,
                "name": "car",
                # No keypoints - this is a detection category
            },
        ],
    }

    json_path = tmp_path / "mixed.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    # Create dummy image
    img_path = tmp_path / "test.jpg"
    iio.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))

    # Should load only pose annotations
    labels = coco.read_labels(json_path, tmp_path)
    assert len(labels) == 1
    assert len(labels[0].instances) == 1  # Only the pose annotation


def test_auto_discover_no_json_files(tmp_path):
    """Test error when no JSON files found in auto-discovery mode."""
    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No JSON annotation files found"):
        coco.read_labels_set(str(empty_dir))


def test_shared_video_objects_for_same_shape(tmp_path):
    """Test that images with the same shape share Video objects."""
    import json

    import imageio.v3 as iio
    import numpy as np

    # Create COCO data with multiple images of the same shape
    coco_data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "img3.jpg", "width": 200, "height": 150},
            {"id": 4, "file_name": "img4.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [50, 50, 2, 60, 60, 2],
                "num_keypoints": 2,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "keypoints": [40, 40, 2, 70, 70, 2],
                "num_keypoints": 2,
            },
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "keypoints": [100, 100, 2, 120, 120, 2],
                "num_keypoints": 2,
            },
            {
                "id": 4,
                "image_id": 4,
                "category_id": 1,
                "keypoints": [30, 30, 2, 80, 80, 2],
                "num_keypoints": 2,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "keypoints": ["head", "tail"], "skeleton": []}
        ],
    }

    # Save annotation file
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    # Create dummy images
    iio.imwrite(tmp_path / "img1.jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    iio.imwrite(tmp_path / "img2.jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    iio.imwrite(tmp_path / "img3.jpg", np.zeros((150, 200, 3), dtype=np.uint8))
    iio.imwrite(tmp_path / "img4.jpg", np.zeros((100, 100, 3), dtype=np.uint8))

    # Load labels
    labels = coco.read_labels(json_path, tmp_path)

    # Check that we have 4 frames
    assert len(labels) == 4

    # Get video objects from frames
    videos = [frame.video for frame in labels]
    unique_videos = []
    for video in videos:
        if video not in unique_videos:
            unique_videos.append(video)

    # Should have 2 unique video objects (one for 100x100, one for 200x150)
    assert len(unique_videos) == 2

    # Check that frames with same shape share the same video object
    video_100x100 = None
    video_200x150 = None

    for frame in labels:
        if frame.video.shape == (3, 100, 100, 3):  # 3 frames of 100x100
            if video_100x100 is None:
                video_100x100 = frame.video
            else:
                # Same shape should use same video object
                assert frame.video is video_100x100
        elif frame.video.shape == (1, 150, 200, 3):  # 1 frame of 150x200
            if video_200x150 is None:
                video_200x150 = frame.video
            else:
                assert frame.video is video_200x150

    # Verify frame indices are correct
    frame_indices_100x100 = []
    frame_indices_200x150 = []

    for frame in labels:
        if frame.video is video_100x100:
            frame_indices_100x100.append(frame.frame_idx)
        else:
            frame_indices_200x150.append(frame.frame_idx)

    # 100x100 video should have frames at indices 0, 1, 2 (3 images)
    assert sorted(frame_indices_100x100) == [0, 1, 2]
    # 200x150 video should have frame at index 0 (1 image)
    assert frame_indices_200x150 == [0]


def test_coco_duplicate_filename_distinct_frames(tmp_path):
    """Distinct image entries that share a file_name each get their own frame."""
    img_path = tmp_path / "dup.png"
    img_path.touch()
    # Three image entries all pointing at the same file_name, with one annotation
    # each. Previously the path-keyed frame-index map collided and raised KeyError.
    data = {
        "images": [
            {"id": 10, "file_name": "dup.png", "height": 40, "width": 40},
            {"id": 11, "file_name": "dup.png", "height": 40, "width": 40},
            {"id": 12, "file_name": "dup.png", "height": 40, "width": 40},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 1,
                "segmentation": [[1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0, 10.0]],
            },
            {
                "id": 2,
                "image_id": 11,
                "category_id": 1,
                "segmentation": [[2.0, 2.0, 12.0, 2.0, 12.0, 12.0, 2.0, 12.0]],
            },
            {
                "id": 3,
                "image_id": 12,
                "category_id": 1,
                "segmentation": [[3.0, 3.0, 13.0, 3.0, 13.0, 13.0, 3.0, 13.0]],
            },
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    json_path = tmp_path / "dup.coco.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Does not raise; one frame per image entry, each at a distinct index.
    labels = coco.read_labels(json_path, dataset_root=tmp_path)
    assert len(labels.labeled_frames) == 3
    assert sorted(lf.frame_idx for lf in labels.labeled_frames) == [0, 1, 2]
    # Each frame carries exactly its own annotation's mask.
    assert all(len(lf.masks) == 1 for lf in labels.labeled_frames)
    assert sum(len(lf.masks) for lf in labels.labeled_frames) == 3


def test_grayscale_loading(tmp_path):
    """Test loading images as grayscale."""
    import json

    import imageio.v3 as iio
    import numpy as np

    # Create simple COCO data
    coco_data = {
        "images": [
            {"id": 1, "file_name": "test.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [50, 50, 2],
                "num_keypoints": 1,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "keypoints": ["head"], "skeleton": []}
        ],
    }

    # Save annotation file
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    # Create dummy image
    iio.imwrite(tmp_path / "test.jpg", np.zeros((100, 100, 3), dtype=np.uint8))

    # Load as RGB (default)
    labels_rgb = coco.read_labels(json_path, tmp_path, grayscale=False)
    assert labels_rgb[0].video.shape == (1, 100, 100, 3)

    # Load as grayscale
    labels_gray = coco.read_labels(json_path, tmp_path, grayscale=True)
    assert labels_gray[0].video.shape == (1, 100, 100, 1)


class TestCVATCompatibility:
    """Test CVAT format compatibility with COCO loader."""

    def test_cvat_tracking(self, cvat_tracking_dataset):
        """Test that CVAT object_id creates proper tracks."""
        labels = coco.read_labels(cvat_tracking_dataset / "annotations.json")

        # Check that frames and instances were loaded
        assert len(labels.labeled_frames) == 2  # Two frames
        assert len(labels.labeled_frames[0].instances) == 2  # Two mice in frame 1
        assert len(labels.labeled_frames[1].instances) == 2  # Two mice in frame 2

        # Check tracks were created
        tracks = set()
        for frame in labels.labeled_frames:
            for instance in frame.instances:
                assert instance.track is not None, (
                    "Track should be created from object_id"
                )
                tracks.add(instance.track)

        assert len(tracks) == 2  # Two unique tracks (mouse 101 and 102)

        # Check track names
        track_names = sorted([t.name for t in tracks])
        assert track_names == ["track_101", "track_102"]

        # Verify same track object is used across frames (not just same name)
        frame1_tracks = [i.track for i in labels.labeled_frames[0].instances]
        frame2_tracks = [i.track for i in labels.labeled_frames[1].instances]

        # Sort by track name to ensure consistent ordering
        frame1_tracks.sort(key=lambda t: t.name)
        frame2_tracks.sort(key=lambda t: t.name)

        # Track objects should be the same instance (identity check)
        assert frame1_tracks[0] is frame2_tracks[0], "Same track should be reused"
        assert frame1_tracks[1] is frame2_tracks[1], "Same track should be reused"

    def test_cvat_with_metadata(self, cvat_with_occluded):
        """Test loading CVAT files with additional metadata."""
        labels = coco.read_labels(cvat_with_occluded / "annotations.json")

        assert len(labels.labeled_frames) == 1
        assert len(labels.labeled_frames[0].instances) == 2

        # Check that tracks are created from object_id
        tracks = [inst.track for inst in labels.labeled_frames[0].instances]
        assert all(t is not None for t in tracks)
        assert tracks[0].name == "track_201"
        assert tracks[1].name == "track_202"

        # Check keypoint visibility (tail should be invisible for second instance)
        inst2 = labels.labeled_frames[0].instances[1]
        assert len(inst2.points) == 3
        # The third keypoint (tail) should have NaN coordinates due to visibility=0
        assert np.isnan(inst2.points[2]["xy"][0])  # x coordinate is NaN
        assert np.isnan(inst2.points[2]["xy"][1])  # y coordinate is NaN

    def test_alternative_track_fields(self, cvat_alternative_track_fields):
        """Test support for track_id and instance_id fields."""
        # Test track_id field
        track_id_path = cvat_alternative_track_fields / "track_id"
        labels_track = coco.read_labels(track_id_path / "annotations.json")
        assert len(labels_track.labeled_frames) == 1
        assert labels_track.labeled_frames[0].instances[0].track is not None
        assert labels_track.labeled_frames[0].instances[0].track.name == "track_301"

        # Test instance_id field
        instance_id_path = cvat_alternative_track_fields / "instance_id"
        labels_instance = coco.read_labels(instance_id_path / "annotations.json")
        assert len(labels_instance.labeled_frames) == 1
        assert labels_instance.labeled_frames[0].instances[0].track is not None
        assert labels_instance.labeled_frames[0].instances[0].track.name == "track_401"

    def test_backward_compatibility(self, coco_flat_images):
        """Test that standard COCO files still work without tracks."""
        labels = coco.read_labels(Path(coco_flat_images) / "annotations.json")

        # Standard COCO should load normally
        assert len(labels.labeled_frames) == 3
        assert len(labels.skeletons) == 1

        # No tracks should be created for standard COCO without track IDs
        for frame in labels.labeled_frames:
            for instance in frame.instances:
                assert instance.track is None, "No tracks should be created without IDs"


class TestImageDeduplication:
    """Test image deduplication functionality for merging COCO datasets."""

    def test_image_dedup_matcher(self, tmp_path):
        """Test IMAGE_DEDUP video matching with overlapping images."""
        # Create two datasets with overlapping images
        images1 = ["img1.jpg", "img2.jpg", "img3.jpg"]
        images2 = ["img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]

        # Create actual image files
        for img in set(images1 + images2):
            (tmp_path / img).touch()

        # Create annotations for dataset 1
        ann1 = {
            "images": [
                {"id": i, "file_name": img, "width": 100, "height": 100}
                for i, img in enumerate(images1)
            ],
            "annotations": [
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "keypoints": [10, 10, 2, 20, 20, 2],
                    "num_keypoints": 2,
                }
                for i in range(len(images1))
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }

        # Create annotations for dataset 2
        ann2 = {
            "images": [
                {"id": i, "file_name": img, "width": 100, "height": 100}
                for i, img in enumerate(images2)
            ],
            "annotations": [
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "keypoints": [30, 30, 2, 40, 40, 2],
                    "num_keypoints": 2,
                }
                for i in range(len(images2))
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }

        # Save JSON files
        with open(tmp_path / "ann1.json", "w") as f:
            json.dump(ann1, f)
        with open(tmp_path / "ann2.json", "w") as f:
            json.dump(ann2, f)

        # Load labels
        labels1 = coco.read_labels(tmp_path / "ann1.json")
        labels2 = coco.read_labels(tmp_path / "ann2.json")

        # Test video matching with IMAGE_DEDUP
        assert len(labels1.videos) == 1
        assert len(labels2.videos) == 1

        video1 = labels1.videos[0]
        video2 = labels2.videos[0]

        # Should detect overlapping images
        assert video1.has_overlapping_images(video2)
        assert video2.has_overlapping_images(video1)

        # Test deduplication
        deduped = video2.deduplicate_with(video1)
        assert deduped is not None
        assert len(deduped.filename) == 2  # Only img4.jpg and img5.jpg remain
        assert Path(deduped.filename[0]).name == "img4.jpg"
        assert Path(deduped.filename[1]).name == "img5.jpg"

        # Test merging with deduplication
        result = labels1.merge(labels2, video=IMAGE_DEDUP_VIDEO_MATCHER)
        assert result.successful

        # Should have original video plus deduplicated new images
        assert len(labels1.videos) == 2  # Original + deduplicated

    def test_complete_duplicate_handling(self, tmp_path):
        """Test when all images in new dataset are duplicates."""
        # Create two datasets with complete overlap
        images = ["img1.jpg", "img2.jpg"]

        # Create actual image files
        for img in images:
            (tmp_path / img).touch()

        # Create identical annotations
        ann = {
            "images": [
                {"id": i, "file_name": img, "width": 100, "height": 100}
                for i, img in enumerate(images)
            ],
            "annotations": [
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "keypoints": [10, 10, 2, 20, 20, 2],
                    "num_keypoints": 2,
                }
                for i in range(len(images))
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }

        # Save JSON files
        with open(tmp_path / "ann1.json", "w") as f:
            json.dump(ann, f)
        with open(tmp_path / "ann2.json", "w") as f:
            json.dump(ann, f)

        # Load labels
        labels1 = coco.read_labels(tmp_path / "ann1.json")
        labels2 = coco.read_labels(tmp_path / "ann2.json")

        video1 = labels1.videos[0]
        video2 = labels2.videos[0]

        # All images are duplicates
        deduped = video2.deduplicate_with(video1)
        assert deduped is None  # All images were duplicates

        # Test merging - should map to existing video
        result = labels1.merge(labels2, video=IMAGE_DEDUP_VIDEO_MATCHER)
        assert result.successful
        assert len(labels1.videos) == 1  # No new video added


class TestShapeBasedMerging:
    """Test shape-based video merging functionality."""

    def test_shape_matcher(self, tmp_path):
        """Test SHAPE video matching based on dimensions only."""
        # Create two datasets with same shape but different images
        images1 = ["batch1_img1.jpg", "batch1_img2.jpg"]
        images2 = ["batch2_img1.jpg", "batch2_img2.jpg"]

        # Create actual image files
        for img in images1 + images2:
            (tmp_path / img).touch()

        # Create annotations for dataset 1
        ann1 = {
            "images": [
                {"id": i, "file_name": img, "width": 640, "height": 480}
                for i, img in enumerate(images1)
            ],
            "annotations": [
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "keypoints": [10, 10, 2, 20, 20, 2],
                    "num_keypoints": 2,
                }
                for i in range(len(images1))
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }

        # Create annotations for dataset 2 (same shape)
        ann2 = {
            "images": [
                {"id": i, "file_name": img, "width": 640, "height": 480}
                for i, img in enumerate(images2)
            ],
            "annotations": [
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "keypoints": [30, 30, 2, 40, 40, 2],
                    "num_keypoints": 2,
                }
                for i in range(len(images2))
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }

        # Save JSON files
        with open(tmp_path / "ann1.json", "w") as f:
            json.dump(ann1, f)
        with open(tmp_path / "ann2.json", "w") as f:
            json.dump(ann2, f)

        # Load labels
        labels1 = coco.read_labels(tmp_path / "ann1.json")
        labels2 = coco.read_labels(tmp_path / "ann2.json")

        video1 = labels1.videos[0]
        video2 = labels2.videos[0]

        # Should match by shape
        assert video1.matches_shape(video2)
        assert video2.matches_shape(video1)

        # Test merging
        merged = video1.merge_with(video2)
        assert len(merged.filename) == 4  # All 4 images
        assert all(
            Path(f).name
            in [
                "batch1_img1.jpg",
                "batch1_img2.jpg",
                "batch2_img1.jpg",
                "batch2_img2.jpg",
            ]
            for f in merged.filename
        )

        # Test Labels merge with SHAPE matcher
        result = labels1.merge(labels2, video=SHAPE_VIDEO_MATCHER)
        assert result.successful
        assert len(labels1.videos) == 1  # Videos merged into one
        assert len(labels1.videos[0].filename) == 4  # All images combined

    def test_shape_mismatch(self, tmp_path):
        """Test that videos with different shapes don't match."""
        # Create two datasets with different shapes
        images1 = ["img1.jpg"]
        images2 = ["img2.jpg"]

        # Create actual image files
        for img in images1 + images2:
            (tmp_path / img).touch()

        # Create annotations with different dimensions
        ann1 = {
            "images": [{"id": 0, "file_name": "img1.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "keypoints": [10, 10, 2, 20, 20, 2],
                    "num_keypoints": 2,
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }

        ann2 = {
            "images": [{"id": 0, "file_name": "img2.jpg", "width": 320, "height": 240}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "keypoints": [5, 5, 2, 10, 10, 2],
                    "num_keypoints": 2,
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }

        # Save JSON files
        with open(tmp_path / "ann1.json", "w") as f:
            json.dump(ann1, f)
        with open(tmp_path / "ann2.json", "w") as f:
            json.dump(ann2, f)

        # Load labels
        labels1 = coco.read_labels(tmp_path / "ann1.json")
        labels2 = coco.read_labels(tmp_path / "ann2.json")

        video1 = labels1.videos[0]
        video2 = labels2.videos[0]

        # Should not match due to different shapes
        assert not video1.matches_shape(video2)
        assert not video2.matches_shape(video1)

        # Test merge - should keep separate videos
        result = labels1.merge(labels2, video=SHAPE_VIDEO_MATCHER)
        assert result.successful
        assert len(labels1.videos) == 2  # Videos kept separate


class TestCOCOExport:
    """Test COCO export functionality."""

    def test_encode_keypoints_ternary(self):
        """Test encoding keypoints with ternary visibility."""
        # Create test data with visible and NaN points
        points_array = np.array(
            [
                [100.0, 200.0, True],
                [150.0, 250.0, True],
                [np.nan, np.nan, False],
                [300.0, 400.0, False],
            ],
            dtype=np.float32,
        )

        keypoints = coco.encode_keypoints(points_array, visibility_encoding="ternary")

        # Should have 12 values (4 keypoints * 3 values each)
        assert len(keypoints) == 12

        # Check first keypoint (visible)
        assert keypoints[0] == 100.0
        assert keypoints[1] == 200.0
        assert keypoints[2] == 2  # Visible in ternary

        # Check third keypoint (NaN - not labeled)
        assert keypoints[6] == 0.0
        assert keypoints[7] == 0.0
        assert keypoints[8] == 0  # Not labeled

        # Check fourth keypoint (not visible)
        assert keypoints[9] == 300.0
        assert keypoints[10] == 400.0
        assert keypoints[11] == 1  # Labeled but occluded in ternary

    def test_encode_keypoints_binary(self):
        """Test encoding keypoints with binary visibility."""
        points_array = np.array(
            [
                [100.0, 200.0, True],
                [np.nan, np.nan, False],
            ],
            dtype=np.float32,
        )

        keypoints = coco.encode_keypoints(points_array, visibility_encoding="binary")

        assert len(keypoints) == 6

        # Check first keypoint (visible)
        assert keypoints[2] == 1  # Visible in binary

        # Check second keypoint (NaN)
        assert keypoints[5] == 0  # Not visible in binary

    def test_convert_labels_basic(self, coco_flat_images):
        """Test basic conversion of Labels to COCO format."""
        # Load existing COCO dataset
        labels = coco.read_labels(Path(coco_flat_images) / "annotations.json")

        # Convert back to COCO format
        coco_data = coco.convert_labels(labels)

        # Check structure
        assert "images" in coco_data
        assert "annotations" in coco_data
        assert "categories" in coco_data

        # Check counts
        assert len(coco_data["images"]) == len(labels.labeled_frames)
        total_instances = sum(len(frame.instances) for frame in labels.labeled_frames)
        total_rois = len(labels.rois)
        total_masks = len(labels.masks)
        # Only standalone bboxes (not linked to instances) are exported
        total_standalone_bboxes = sum(1 for b in labels.bboxes if b.instance is None)
        assert len(coco_data["annotations"]) == (
            total_instances + total_rois + total_masks + total_standalone_bboxes
        )

    def test_convert_labels_with_tracks(self, tmp_path):
        """Test conversion of Labels with tracking information."""
        # Create simple labels with tracks
        skeleton = sio.Skeleton(
            nodes=["node1", "node2"], edges=[("node1", "node2")], name="test_skeleton"
        )

        video = sio.Video.from_filename(["img1.png", "img2.png"])
        track1 = sio.Track(name="track_1")
        track2 = sio.Track(name="track_2")

        # Create instances with tracks
        instance1 = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
            skeleton=skeleton,
            track=track1,
        )
        instance2 = sio.Instance.from_numpy(
            points_data=np.array([[50.0, 60.0], [70.0, 80.0]]),
            skeleton=skeleton,
            track=track2,
        )

        frame1 = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance1])
        frame2 = sio.LabeledFrame(video=video, frame_idx=1, instances=[instance2])

        labels = sio.Labels(labeled_frames=[frame1, frame2])

        # Convert to COCO
        coco_data = coco.convert_labels(labels)

        # Check track IDs are present
        assert len(coco_data["annotations"]) == 2
        assert "attributes" in coco_data["annotations"][0]
        assert "object_id" in coco_data["annotations"][0]["attributes"]
        assert "attributes" in coco_data["annotations"][1]
        assert "object_id" in coco_data["annotations"][1]["attributes"]

        # Track IDs should be different
        track_id1 = coco_data["annotations"][0]["attributes"]["object_id"]
        track_id2 = coco_data["annotations"][1]["attributes"]["object_id"]
        assert track_id1 != track_id2

    def test_convert_labels_skeleton_edges(self, tmp_path):
        """Test that skeleton edges are correctly converted to 1-based indexing."""
        skeleton = sio.Skeleton(
            nodes=["a", "b", "c"], edges=[("a", "b"), ("b", "c")], name="test"
        )

        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            skeleton=skeleton,
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        coco_data = coco.convert_labels(labels)

        # Check skeleton edges are 1-based
        category = coco_data["categories"][0]
        assert category["skeleton"] == [[1, 2], [2, 3]]  # 1-based indexing

    def test_write_labels_basic(self, coco_flat_images, tmp_path):
        """Test writing Labels to COCO JSON file."""
        # Load existing COCO dataset
        labels = coco.read_labels(Path(coco_flat_images) / "annotations.json")

        # Write to new file
        output_path = tmp_path / "output_annotations.json"
        coco.write_labels(labels, output_path)

        # Check file was created
        assert output_path.exists()

        # Load and validate
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert len(data["images"]) > 0
        assert len(data["annotations"]) > 0
        assert len(data["categories"]) > 0

    def test_write_labels_custom_filenames(self, tmp_path):
        """Test writing with custom image filenames."""
        skeleton = sio.Skeleton(nodes=["a", "b"], name="test")
        video = sio.Video.from_filename(["img1.png", "img2.png"])

        instance1 = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0], [30.0, 40.0]]), skeleton=skeleton
        )
        instance2 = sio.Instance.from_numpy(
            points_data=np.array([[50.0, 60.0], [70.0, 80.0]]), skeleton=skeleton
        )

        frame1 = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance1])
        frame2 = sio.LabeledFrame(video=video, frame_idx=1, instances=[instance2])

        labels = sio.Labels(labeled_frames=[frame1, frame2])

        # Write with custom filenames
        output_path = tmp_path / "custom_annotations.json"
        custom_filenames = ["custom_img1.jpg", "custom_img2.jpg"]
        coco.write_labels(labels, output_path, image_filenames=custom_filenames)

        # Load and verify filenames
        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["images"][0]["file_name"] == "custom_img1.jpg"
        assert data["images"][1]["file_name"] == "custom_img2.jpg"

    def test_write_labels_binary_visibility(self, tmp_path):
        """Test writing with binary visibility encoding."""
        skeleton = sio.Skeleton(nodes=["a", "b"], name="test")
        video = sio.Video.from_filename(["img.png"])

        instance = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0], [np.nan, np.nan]]), skeleton=skeleton
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        # Write with binary visibility
        output_path = tmp_path / "binary_annotations.json"
        coco.write_labels(labels, output_path, visibility_encoding="binary")

        # Load and check visibility values
        with open(output_path, "r") as f:
            data = json.load(f)

        keypoints = data["annotations"][0]["keypoints"]
        # First point visible: should be 1 in binary
        assert keypoints[2] == 1
        # Second point not visible: should be 0 in binary
        assert keypoints[5] == 0

    @pytest.mark.parametrize(
        "json_name",
        ["annotations.json", "annotations_negative_frame.json"],
    )
    def test_roundtrip_conversion(self, coco_flat_images, json_name, tmp_path):
        """Test that data survives a roundtrip conversion."""
        original_labels = coco.read_labels(Path(coco_flat_images) / json_name)

        output_path = tmp_path / "roundtrip_annotations.json"
        coco.write_labels(original_labels, output_path)

        # Load again
        reloaded_labels = coco.read_labels(output_path, dataset_root=coco_flat_images)

        # Check basic structure is preserved
        assert len(reloaded_labels.labeled_frames) == len(
            original_labels.labeled_frames
        )
        assert len(reloaded_labels.skeletons) == len(original_labels.skeletons)

        # Check skeleton nodes match
        assert len(reloaded_labels.skeletons[0].nodes) == len(
            original_labels.skeletons[0].nodes
        )

    def test_save_coco_via_main_api(self, coco_flat_images, tmp_path):
        """Test saving via the main sio.save_coco API."""
        labels = sio.load_coco(Path(coco_flat_images) / "annotations.json")

        output_path = tmp_path / "api_output.json"
        sio.save_coco(labels, str(output_path))

        assert output_path.exists()

        # Verify it's valid COCO format
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_save_file_coco_format(self, coco_flat_images, tmp_path):
        """Test saving via sio.save_file with format='coco'."""
        labels = sio.load_coco(Path(coco_flat_images) / "annotations.json")

        output_path = tmp_path / "save_file_output.json"
        sio.save_file(labels, output_path, format="coco")

        assert output_path.exists()

        # Verify it's valid COCO format
        reloaded = sio.load_file(
            output_path, format="coco", dataset_root=coco_flat_images
        )
        assert len(reloaded.labeled_frames) == len(labels.labeled_frames)

    def test_export_includes_bbox_field(self, tmp_path):
        """Test that exported COCO annotations include required bbox field."""
        # Create simple labels with two visible keypoints
        skeleton = sio.Skeleton(nodes=["a", "b"], name="test")
        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0], [50.0, 60.0]]), skeleton=skeleton
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        # Export to COCO
        coco_data = coco.convert_labels(labels)

        # Verify bbox is present
        assert "bbox" in coco_data["annotations"][0]

        # Verify bbox is correct (should encompass keypoints)
        bbox = coco_data["annotations"][0]["bbox"]
        assert len(bbox) == 4
        assert bbox[0] == 10.0  # x_min
        assert bbox[1] == 20.0  # y_min
        assert bbox[2] == 40.0  # width (50 - 10)
        assert bbox[3] == 40.0  # height (60 - 20)

    def test_export_includes_area_field(self, tmp_path):
        """Test that exported COCO annotations include area field."""
        skeleton = sio.Skeleton(nodes=["a", "b"], name="test")
        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0], [50.0, 60.0]]), skeleton=skeleton
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        coco_data = coco.convert_labels(labels)

        # Verify area is present and correct
        assert "area" in coco_data["annotations"][0]
        assert coco_data["annotations"][0]["area"] == 1600.0  # 40 * 40

    def test_export_includes_iscrowd_field(self, tmp_path):
        """Test that exported COCO annotations include iscrowd field."""
        skeleton = sio.Skeleton(nodes=["a"], name="test")
        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0]]), skeleton=skeleton
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        coco_data = coco.convert_labels(labels)

        # Verify iscrowd is present and set to 0
        assert "iscrowd" in coco_data["annotations"][0]
        assert coco_data["annotations"][0]["iscrowd"] == 0

    def test_export_bbox_with_nan_points(self, tmp_path):
        """Test bbox computation excludes NaN keypoints."""
        skeleton = sio.Skeleton(nodes=["a", "b", "c"], name="test")
        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array(
                [
                    [10.0, 20.0],  # visible
                    [np.nan, np.nan],  # not labeled
                    [50.0, 60.0],  # visible
                ]
            ),
            skeleton=skeleton,
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        coco_data = coco.convert_labels(labels)

        # Bbox should only include visible points
        bbox = coco_data["annotations"][0]["bbox"]
        assert bbox[0] == 10.0  # x_min (from first point)
        assert bbox[1] == 20.0  # y_min (from first point)
        assert bbox[2] == 40.0  # width (50 - 10)
        assert bbox[3] == 40.0  # height (60 - 20)

    def test_export_bbox_all_nan_points(self, tmp_path):
        """Test bbox computation with all NaN keypoints (edge case)."""
        skeleton = sio.Skeleton(nodes=["a", "b"], name="test")
        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ]
            ),
            skeleton=skeleton,
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        coco_data = coco.convert_labels(labels)

        # Should have zero bbox when all keypoints are NaN
        bbox = coco_data["annotations"][0]["bbox"]
        assert bbox == [0.0, 0.0, 0.0, 0.0]
        assert coco_data["annotations"][0]["area"] == 0.0
        assert coco_data["annotations"][0]["num_keypoints"] == 0

    def test_export_single_string_filename(self, tmp_path):
        """Test that single string image_filenames is converted to list."""
        skeleton = sio.Skeleton(nodes=["a"], name="test")
        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0]]), skeleton=skeleton
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        # Pass single string instead of list
        coco_data = coco.convert_labels(labels, image_filenames="single_image.jpg")

        # Should work and use the provided filename
        assert len(coco_data["images"]) == 1
        assert coco_data["images"][0]["file_name"] == "single_image.jpg"

    def test_export_filename_count_mismatch(self, tmp_path):
        """Test error when image_filenames count doesn't match frames."""
        skeleton = sio.Skeleton(nodes=["a"], name="test")
        video = sio.Video.from_filename(["img1.png", "img2.png"])

        instance1 = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0]]), skeleton=skeleton
        )
        instance2 = sio.Instance.from_numpy(
            points_data=np.array([[30.0, 40.0]]), skeleton=skeleton
        )

        frame1 = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance1])
        frame2 = sio.LabeledFrame(video=video, frame_idx=1, instances=[instance2])
        labels = sio.Labels(labeled_frames=[frame1, frame2])

        # Provide wrong number of filenames (1 instead of 2)
        with pytest.raises(
            ValueError,
            match=(
                "Number of image filenames \\(1\\) must match "
                "number of labeled frames \\(2\\)"
            ),
        ):
            coco.convert_labels(labels, image_filenames=["only_one.jpg"])

    def test_export_video_file_filename_generation(self, tmp_path):
        """Test filename generation for video files (not image sequences)."""
        skeleton = sio.Skeleton(nodes=["a"], name="test")
        # Use a video file path (string) instead of image sequence (list)
        video = sio.Video.from_filename("test_video.mp4")

        instance = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0]]), skeleton=skeleton
        )
        frame = sio.LabeledFrame(video=video, frame_idx=5, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        coco_data = coco.convert_labels(labels)

        # Should generate filename from video name + frame index
        assert len(coco_data["images"]) == 1
        assert coco_data["images"][0]["file_name"] == "test_video_frame_000005.png"

    def test_save_file_coco_autodetect_from_kwargs(self, tmp_path):
        """Test that save_file auto-detects COCO format from kwargs."""
        skeleton = sio.Skeleton(nodes=["a"], name="test")
        video = sio.Video.from_filename(["img.png"])
        instance = sio.Instance.from_numpy(
            points_data=np.array([[10.0, 20.0]]), skeleton=skeleton
        )
        frame = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labels = sio.Labels(labeled_frames=[frame])

        output_path = tmp_path / "output.json"

        # Save without explicit format, but with COCO-specific kwarg
        # This should auto-detect COCO format from visibility_encoding parameter
        sio.save_file(labels, output_path, visibility_encoding="binary")

        # Verify it saved as COCO format
        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

        # Verify binary visibility encoding was used (1 instead of 2)
        assert data["annotations"][0]["keypoints"][2] == 1  # Binary visible


class TestCOCOROIMaskIO:
    """Test ROI and segmentation mask COCO I/O."""

    def test_coco_roi_bbox_roundtrip(self, tmp_path):
        """Test roundtrip of bounding box ROIs through COCO format."""
        from sleap_io.model.roi import UserROI

        video = sio.Video.from_filename(["img1.png"])
        roi1 = UserROI.from_bbox(10.0, 20.0, 50.0, 30.0, category="dog", video=video)
        roi2 = UserROI.from_bbox(
            100.0,
            200.0,
            80.0,
            60.0,
            category="cat",
            video=video,
        )

        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.rois.extend([roi1, roi2])
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "bbox_test.json"
        coco.write_labels(labels, json_path)

        # Verify JSON structure
        with open(json_path, "r") as f:
            data = json.load(f)

        assert len(data["annotations"]) == 2
        ann1 = data["annotations"][0]
        assert ann1["bbox"] == [10.0, 20.0, 50.0, 30.0]
        assert ann1["iscrowd"] == 0

        # Verify categories were created
        cat_names = {c["name"] for c in data["categories"]}
        assert "dog" in cat_names
        assert "cat" in cat_names

    def test_coco_roi_polygon_roundtrip(self, tmp_path):
        """Test roundtrip of polygon ROIs through COCO format."""
        from sleap_io.model.roi import UserROI

        coords = [(10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)]
        video = sio.Video.from_filename(["img1.png"])
        roi = UserROI.from_polygon(coords, category="region", video=video)

        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.rois.append(roi)
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "polygon_test.json"
        coco.write_labels(labels, json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        assert len(data["annotations"]) == 1
        ann = data["annotations"][0]
        assert "segmentation" in ann
        assert isinstance(ann["segmentation"], list)
        assert isinstance(ann["segmentation"][0], list)
        # Polygon should contain x,y pairs flattened
        seg = ann["segmentation"][0]
        assert len(seg) == 8  # 4 points * 2 coords

    def test_coco_mask_rle_roundtrip(self, tmp_path):
        """Test roundtrip of segmentation masks through COCO RLE format."""
        from sleap_io.model.mask import UserSegmentationMask

        # Create a simple mask
        mask_arr = np.zeros((10, 10), dtype=bool)
        mask_arr[2:5, 3:7] = True

        video = sio.Video.from_filename(["img1.png"])
        seg_mask = UserSegmentationMask.from_numpy(mask_arr, category="cell")
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.masks.append(seg_mask)
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "mask_test.json"
        coco.write_labels(labels, json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        assert len(data["annotations"]) == 1
        ann = data["annotations"][0]
        assert ann["iscrowd"] == 1
        assert "segmentation" in ann
        assert "counts" in ann["segmentation"]
        assert "size" in ann["segmentation"]
        assert ann["segmentation"]["size"] == [10, 10]

        # Verify RLE roundtrip: decode back and compare
        decoded = coco._decode_coco_rle(
            ann["segmentation"]["counts"], ann["segmentation"]["size"]
        )
        np.testing.assert_array_equal(decoded, mask_arr)

    def test_coco_mask_scaled_export(self, tmp_path):
        """Scaled mask should be resampled to image extent for COCO export."""
        from sleap_io.model.mask import UserSegmentationMask

        # Half-resolution 5x5 mask (covers 10x10 in image space)
        mask_arr = np.ones((5, 5), dtype=bool)
        video = sio.Video.from_filename(["img1.png"])
        seg_mask = UserSegmentationMask.from_numpy(
            mask_arr, category="cell", scale=(0.5, 0.5)
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.masks.append(seg_mask)
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "mask_scaled.json"
        coco.write_labels(labels, json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        ann = data["annotations"][0]
        # Resampled to image extent: 10x10
        assert ann["segmentation"]["size"] == [10, 10]

    def test_coco_mask_track_identity_roundtrip(self, tmp_path):
        """Track on a SegmentationMask is written and restored via object_id."""
        mask_arr = np.zeros((10, 10), dtype=bool)
        mask_arr[2:5, 3:7] = True

        video = sio.Video.from_filename(["img1.png"])
        track = Track("id_7")
        seg_mask = UserSegmentationMask.from_numpy(
            mask_arr, category="cell", track=track
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.masks.append(seg_mask)
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "mask_track.json"
        coco.write_labels(labels, json_path)

        # The JSON annotation carries the track identity as attributes.object_id.
        with open(json_path, "r") as f:
            data = json.load(f)
        ann = data["annotations"][0]
        assert "attributes" in ann
        assert "object_id" in ann["attributes"]

        # Read back: the mask's track is restored and registered on the Labels.
        for img_info in data["images"]:
            (tmp_path / img_info["file_name"]).touch()
        labels_rt = coco.read_labels(json_path, dataset_root=tmp_path)
        assert len(labels_rt.masks) == 1
        assert labels_rt.masks[0].track is not None
        assert len(labels_rt.tracks) > 0

    def test_coco_roi_track_identity_roundtrip(self, tmp_path):
        """Track on an ROI is written and restored via object_id."""
        from sleap_io.model.roi import UserROI

        coords = [(10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)]
        video = sio.Video.from_filename(["img1.png"])
        track = Track("id_7")
        roi = UserROI.from_polygon(coords, category="region", track=track, video=video)
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.rois.append(roi)
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "roi_track.json"
        coco.write_labels(labels, json_path)

        with open(json_path, "r") as f:
            data = json.load(f)
        ann = data["annotations"][0]
        assert ann["attributes"]["object_id"] is not None

        for img_info in data["images"]:
            (tmp_path / img_info["file_name"]).touch()
        labels_rt = coco.read_labels(
            json_path, dataset_root=tmp_path, segmentation_format="roi"
        )
        assert len(labels_rt.rois) == 1
        assert labels_rt.rois[0].track is not None
        assert len(labels_rt.tracks) > 0

    def test_coco_bbox_track_identity_roundtrip(self, tmp_path):
        """Track on a standalone BoundingBox is written and restored."""
        video = sio.Video.from_filename(["img1.png"])
        track = Track("id_7")
        bbox = UserBoundingBox.from_xywh(10, 20, 50, 30, category="dog", track=track)
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.bboxes.append(bbox)
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "bbox_track.json"
        coco.write_labels(labels, json_path)

        with open(json_path, "r") as f:
            data = json.load(f)
        ann = data["annotations"][0]
        assert ann["attributes"]["object_id"] is not None

        for img_info in data["images"]:
            (tmp_path / img_info["file_name"]).touch()
        labels_rt = coco.read_labels(json_path, dataset_root=tmp_path)
        assert len(labels_rt.bboxes) == 1
        assert labels_rt.bboxes[0].track is not None
        assert len(labels_rt.tracks) > 0

    def test_coco_detection_only_read(self, tmp_path):
        """Test reading a detection-only COCO JSON (no keypoints)."""
        # Create image files so they can be resolved
        img_path = tmp_path / "image_001.png"
        img_path.touch()

        detection_data = {
            "images": [
                {"id": 1, "file_name": "image_001.png", "height": 100, "width": 200},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "segmentation": [[5.0, 5.0, 50.0, 5.0, 30.0, 50.0, 5.0, 50.0]],
                    "bbox": [5, 5, 45, 45],
                    "area": 1012.5,
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": {
                        "counts": [0, 5, 5, 5, 5],
                        "size": [5, 5],
                    },
                    "bbox": [0, 0, 5, 5],
                    "area": 10,
                    "iscrowd": 1,
                },
            ],
            "categories": [
                {"id": 1, "name": "animal"},
                {"id": 2, "name": "plant"},
            ],
        }

        json_path = tmp_path / "detection.json"
        with open(json_path, "w") as f:
            json.dump(detection_data, f)

        labels = coco.read_labels(json_path, dataset_root=tmp_path)

        # Should have created masks and bboxes, not instances
        assert len(labels.labeled_frames) == 1
        assert len(labels.labeled_frames[0].instances) == 0

        # Default segmentation_format="mask": polygon (annotation 2) and RLE
        # (annotation 3) both become masks; no ROIs.
        assert len(labels.rois) == 0
        assert len(labels.masks) == 2
        masks_by_cat = {m.category.name: m for m in labels.masks}
        # Polygon annotation -> mask rasterized at the image resolution.
        plant_mask = masks_by_cat["plant"]
        assert plant_mask.height == 100
        assert plant_mask.width == 200
        assert plant_mask.area > 0
        # RLE annotation -> mask at its native size.
        animal_mask = masks_by_cat["animal"]
        assert animal_mask.height == 5
        assert animal_mask.width == 5

        # Bbox-only annotation -> BoundingBox (annotation 1)
        assert len(labels.bboxes) == 1
        assert labels.bboxes[0].category.name == "animal"
        x, y, w, h = labels.bboxes[0].xywh
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)
        assert w == pytest.approx(30.0)
        assert h == pytest.approx(40.0)
        assert isinstance(labels.bboxes[0], UserBoundingBox)

        # segmentation_format="roi": polygon stays a vector ROI, RLE stays a mask.
        labels_roi = coco.read_labels(
            json_path, dataset_root=tmp_path, segmentation_format="roi"
        )
        assert len(labels_roi.rois) == 1
        assert labels_roi.rois[0].category.name == "plant"
        assert len(labels_roi.masks) == 1
        assert labels_roi.masks[0].category.name == "animal"
        assert len(labels_roi.bboxes) == 1

    def test_coco_category_preservation(self, tmp_path):
        """Test that category names roundtrip correctly."""
        from sleap_io.model.roi import UserROI

        video = sio.Video.from_filename(["img1.png"])
        roi = UserROI.from_bbox(
            10.0,
            20.0,
            30.0,
            40.0,
            category="special_class",
            video=video,
        )

        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.rois.append(roi)
        labels = sio.Labels(labeled_frames=[lf])

        json_path = tmp_path / "cat_test.json"
        coco.write_labels(labels, json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Verify category name
        assert any(c["name"] == "special_class" for c in data["categories"])

        # Verify category_id references correct category
        ann = data["annotations"][0]
        cat_id = ann["category_id"]
        cat = next(c for c in data["categories"] if c["id"] == cat_id)
        assert cat["name"] == "special_class"

    def test_coco_rle_column_major_order(self):
        """Test that COCO RLE encode/decode correctly handles column-major order."""
        # Create a non-symmetric mask to verify column-major handling
        mask = np.zeros((4, 6), dtype=bool)
        mask[0, 0:3] = True  # Top-left row
        mask[1, 0] = True

        rle = coco._encode_coco_rle(mask)
        assert rle["size"] == [4, 6]

        decoded = coco._decode_coco_rle(rle["counts"], rle["size"])
        np.testing.assert_array_equal(decoded, mask)

    def test_coco_compressed_rle_known_value(self):
        """Compressed (LEB128) RLE counts decode to the expected mask.

        ``b"01;000"`` is the pycocotools-compressed encoding of a 5x5 mask with
        True pixels on the main diagonal at (0, 0), (2, 2) and (4, 4). This pins
        decode correctness against a known-good pycocotools value, not just
        round-trip symmetry.
        """
        expected = np.zeros((5, 5), dtype=bool)
        expected[0, 0] = True
        expected[2, 2] = True
        expected[4, 4] = True

        # Both bytes and str forms must decode identically.
        decoded_bytes = coco._decode_coco_rle(b"01;000", [5, 5])
        np.testing.assert_array_equal(decoded_bytes, expected)

        decoded_str = coco._decode_coco_rle("01;000", [5, 5])
        np.testing.assert_array_equal(decoded_str, expected)

    def test_coco_compressed_rle_full_column_known_value(self):
        """Compressed RLE for a full first column decodes correctly.

        ``b"05d0"`` is the pycocotools-compressed encoding of a 5x5 mask whose
        entire first column is True (column-major order).
        """
        expected = np.zeros((5, 5), dtype=bool)
        expected[:, 0] = True

        decoded = coco._decode_coco_rle(b"05d0", [5, 5])
        np.testing.assert_array_equal(decoded, expected)

    def test_coco_compressed_rle_counts_decoder(self):
        """The LEB128 counts decoder returns the expected run lengths."""
        # ``b"05d0"`` -> [0, 5, 20] (0-run of 0, then 5 True for the full first
        # column, then 20 False to fill the remaining 4 columns of a 5x5
        # column-major mask).
        assert coco._decode_compressed_rle_counts(b"05d0") == [0, 5, 20]
        # String input must match bytes input.
        assert coco._decode_compressed_rle_counts("05d0") == [0, 5, 20]

    def test_coco_compressed_rle_roundtrip(self):
        """Uncompressed counts -> compressed string -> decode equals original."""

        def encode_compressed(run_lengths: list[int]) -> bytes:
            """Inverse of ``_decode_compressed_rle_counts`` (pycocotools scheme)."""
            out = bytearray()
            for i, x in enumerate(run_lengths):
                v = int(x)
                if i > 2:
                    v -= int(run_lengths[i - 2])
                more = True
                while more:
                    c = v & 0x1F
                    v >>= 5
                    if c & 0x10:
                        more = v != -1
                    else:
                        more = v != 0
                    if more:
                        c |= 0x20
                    out.append(c + 48)
            return bytes(out)

        mask = np.zeros((4, 6), dtype=bool)
        mask[0, 0:3] = True
        mask[1, 0] = True

        rle = coco._encode_coco_rle(mask)
        compressed = encode_compressed(rle["counts"])
        decoded = coco._decode_coco_rle(compressed, rle["size"])
        np.testing.assert_array_equal(decoded, mask)

    def test_coco_load_compressed_rle_no_crash(self, tmp_path):
        """load_coco must not crash on compressed (string-counts) RLE.

        Regression for the case where a real on-disk image resolves and the
        compressed RLE annotation is decoded (previously raised
        ``TypeError: unsupported operand type(s) for +=: 'int' and 'str'``).
        """
        img_path = tmp_path / "img.png"
        img_path.touch()

        data = {
            "images": [
                {"id": 1, "file_name": "img.png", "height": 5, "width": 5},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    # pycocotools-compressed RLE for a known 5x5 diagonal mask.
                    "segmentation": {"counts": "01;000", "size": [5, 5]},
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "cell"}],
        }

        json_path = tmp_path / "compressed_rle.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        labels = sio.load_coco(json_path, dataset_root=tmp_path)

        assert len(labels.masks) == 1
        mask = labels.masks[0]
        assert mask.category.name == "cell"
        assert mask.height == 5
        assert mask.width == 5

        expected = np.zeros((5, 5), dtype=bool)
        expected[0, 0] = True
        expected[2, 2] = True
        expected[4, 4] = True
        np.testing.assert_array_equal(mask.data.astype(bool), expected)

    def test_coco_detection_empty_segmentation_fallback_to_bbox(self, tmp_path):
        """Test that annotations with segmentation=[] fall back to bbox ROI."""
        img_path = tmp_path / "img.png"
        img_path.touch()

        data = {
            "images": [
                {"id": 1, "file_name": "img.png", "height": 100, "width": 200},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [],
                    "bbox": [10, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "animal"}],
        }

        json_path = tmp_path / "empty_seg.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        labels = coco.read_labels(json_path, dataset_root=tmp_path)

        # Empty segmentation list should fall back to BoundingBox
        assert len(labels.rois) == 0
        assert len(labels.bboxes) == 1
        bbox = labels.bboxes[0]
        assert isinstance(bbox, UserBoundingBox)
        assert bbox.category.name == "animal"
        x, y, w, h = bbox.xywh
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)
        assert w == pytest.approx(30.0)
        assert h == pytest.approx(40.0)

    def test_coco_detection_with_polygon_read(self, tmp_path):
        """Polygon segmentation rasterizes to a mask by default (roi opt-out)."""
        img_path = tmp_path / "img.png"
        img_path.touch()

        data = {
            "images": [
                {"id": 1, "file_name": "img.png", "height": 100, "width": 100},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
                    "bbox": [10, 10, 40, 40],
                    "area": 1600,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }

        json_path = tmp_path / "poly.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Default segmentation_format="mask": polygon -> rasterized mask.
        labels = coco.read_labels(json_path, dataset_root=tmp_path)
        assert len(labels.rois) == 0
        assert len(labels.masks) == 1
        mask = labels.masks[0]
        assert isinstance(mask, UserSegmentationMask)
        assert mask.category.name == "obj"
        assert mask.height == 100
        assert mask.width == 100
        # Mask bbox (in image coords) matches the polygon extent within a pixel.
        bx, by, bw, bh = mask.bbox
        assert bx == pytest.approx(10.0, abs=1.0)
        assert by == pytest.approx(10.0, abs=1.0)
        assert (bx + bw) == pytest.approx(50.0, abs=1.0)
        assert (by + bh) == pytest.approx(50.0, abs=1.0)

        # segmentation_format="roi": keep native vector geometry.
        labels_roi = coco.read_labels(
            json_path, dataset_root=tmp_path, segmentation_format="roi"
        )
        assert len(labels_roi.masks) == 0
        assert len(labels_roi.rois) == 1
        roi = labels_roi.rois[0]
        assert roi.category.name == "obj"
        minx, miny, maxx, maxy = roi.bounds
        assert minx == pytest.approx(10.0)
        assert miny == pytest.approx(10.0)
        assert maxx == pytest.approx(50.0)
        assert maxy == pytest.approx(50.0)

    def test_coco_detection_degenerate_segmentation_fallback_to_bbox(self, tmp_path):
        """Degenerate (sub-polygon) segmentation falls back to bbox, not dropped."""
        img_path = tmp_path / "img.png"
        img_path.touch()

        data = {
            "images": [
                {"id": 1, "file_name": "img.png", "height": 100, "width": 200},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    # Single-point ring: truthy but cannot form a polygon, so
                    # _decode_segmentation yields no geometry.
                    "segmentation": [[5.0, 5.0]],
                    "bbox": [2, 2, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "animal"}],
        }

        json_path = tmp_path / "degenerate_bbox.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        labels = coco.read_labels(json_path, dataset_root=tmp_path)

        # No mask/roi geometry, but the bbox is preserved.
        assert len(labels.masks) == 0
        assert len(labels.rois) == 0
        assert len(labels.bboxes) == 1
        bbox = labels.bboxes[0]
        assert isinstance(bbox, UserBoundingBox)
        assert bbox.category.name == "animal"
        x, y, w, h = bbox.xywh
        assert x == pytest.approx(2.0)
        assert y == pytest.approx(2.0)
        assert w == pytest.approx(10.0)
        assert h == pytest.approx(10.0)

    def test_coco_detection_valid_polygon_no_spurious_bbox(self, tmp_path):
        """A valid polygon yields a mask and does not also emit a bbox."""
        img_path = tmp_path / "img.png"
        img_path.touch()

        data = {
            "images": [
                {"id": 1, "file_name": "img.png", "height": 100, "width": 100},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
                    "bbox": [10, 10, 40, 40],
                    "area": 1600,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }

        json_path = tmp_path / "valid_poly_no_bbox.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        labels = coco.read_labels(json_path, dataset_root=tmp_path)

        # Geometry was decoded, so the bbox fallback must not fire.
        assert len(labels.masks) == 1
        assert len(labels.rois) == 0
        assert len(labels.bboxes) == 0


def test_read_labels_keypoints_and_segmentation(tmp_path):
    """Annotations with both keypoints and segmentation should preserve both."""
    # Create a small test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "image.png"
    iio.imwrite(str(img_path), img)

    # Create COCO JSON with an annotation that has both keypoints and segmentation
    coco_data = {
        "images": [{"id": 1, "file_name": "image.png", "height": 100, "width": 100}],
        "categories": [
            {
                "id": 1,
                "name": "animal",
                "keypoints": ["nose", "tail"],
                "skeleton": [[1, 2]],
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [50, 50, 2, 70, 70, 2],
                "num_keypoints": 2,
                "segmentation": [[10, 10, 90, 10, 90, 90, 10, 90]],
                "bbox": [10, 10, 80, 80],
                "area": 6400,
            }
        ],
    }

    json_path = tmp_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    labels = coco.read_labels(str(json_path), dataset_root=str(tmp_path))

    # Should have instances from keypoints
    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1

    # Check that the instance has the correct keypoints
    instance = labels.labeled_frames[0].instances[0]
    points = instance.numpy()
    assert points.shape == (2, 2)
    assert points[0, 0] == pytest.approx(50.0)
    assert points[0, 1] == pytest.approx(50.0)
    assert points[1, 0] == pytest.approx(70.0)
    assert points[1, 1] == pytest.approx(70.0)

    # Should also have a mask from the segmentation polygon (default mask mode),
    # rasterized at the image resolution and linked to the same instance.
    assert len(labels.rois) == 0
    assert len(labels.masks) == 1
    mask = labels.masks[0]
    assert isinstance(mask, UserSegmentationMask)
    assert mask.category.name == "animal"
    assert mask.instance is instance
    assert mask.height == 100
    assert mask.width == 100
    bx, by, bw, bh = mask.bbox
    assert bx == pytest.approx(10.0, abs=1.0)
    assert by == pytest.approx(10.0, abs=1.0)
    assert (bx + bw) == pytest.approx(90.0, abs=1.0)
    assert (by + bh) == pytest.approx(90.0, abs=1.0)

    # Should also have a BoundingBox linked to the instance
    assert len(labels.bboxes) == 1
    bbox = labels.bboxes[0]
    assert isinstance(bbox, UserBoundingBox)
    assert bbox.instance is instance
    assert bbox.category.name == "animal"
    bx, by, bw, bh = bbox.xywh
    assert bx == pytest.approx(10.0)
    assert by == pytest.approx(10.0)
    assert bw == pytest.approx(80.0)
    assert bh == pytest.approx(80.0)


def test_read_labels_keypoints_and_rle_segmentation(tmp_path):
    """Annotations with both keypoints and RLE segmentation should preserve both."""
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img_path = tmp_path / "image.png"
    iio.imwrite(str(img_path), img)

    # RLE for a 5x5 mask: 5 off, 5 on, 5 off, 5 on, 5 off  (two rows filled)
    coco_data = {
        "images": [{"id": 1, "file_name": "image.png", "height": 5, "width": 5}],
        "categories": [
            {
                "id": 1,
                "name": "animal",
                "keypoints": ["nose", "tail"],
                "skeleton": [[1, 2]],
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [1, 1, 2, 3, 3, 2],
                "num_keypoints": 2,
                "segmentation": {
                    "counts": [5, 5, 5, 5, 5],
                    "size": [5, 5],
                },
                "bbox": [0, 1, 5, 2],
                "area": 10,
                "iscrowd": 1,
            }
        ],
    }

    json_path = tmp_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    labels = coco.read_labels(str(json_path), dataset_root=str(tmp_path))

    # Should have instances from keypoints
    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1

    # Should have a mask from the RLE segmentation
    assert len(labels.masks) == 1
    mask = labels.masks[0]
    assert isinstance(mask, UserSegmentationMask)
    assert mask.category.name == "animal"
    assert mask.height == 5
    assert mask.width == 5
    # Mask should be linked to the instance
    assert mask.instance is labels.labeled_frames[0].instances[0]


def test_coco_read_invalid_segmentation_format(tmp_path):
    """An unknown segmentation_format raises ValueError."""
    data = {
        "images": [{"id": 1, "file_name": "img.png", "height": 10, "width": 10}],
        "annotations": [],
        "categories": [{"id": 1, "name": "obj"}],
    }
    json_path = tmp_path / "x.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="segmentation_format"):
        coco.read_labels(json_path, dataset_root=tmp_path, segmentation_format="bad")


def test_coco_polygon_mask_falls_back_to_roi_without_dims(tmp_path):
    """In mask mode, polygons fall back to ROI when image dims are missing."""
    img_path = tmp_path / "img.png"
    img_path.touch()
    data = {
        "images": [{"id": 1, "file_name": "img.png"}],  # no height/width
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0]],
            }
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    json_path = tmp_path / "nodims.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    # Default mask mode cannot rasterize without dims, so it keeps the polygon.
    labels = coco.read_labels(json_path, dataset_root=tmp_path)
    assert len(labels.masks) == 0
    assert len(labels.rois) == 1
    assert labels.rois[0].category.name == "obj"


def test_coco_multipolygon_annotation_single_mask(tmp_path):
    """Multiple polygons in one annotation rasterize to a single mask."""
    img_path = tmp_path / "img.png"
    img_path.touch()
    data = {
        "images": [{"id": 1, "file_name": "img.png", "height": 50, "width": 50}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [
                    [1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0, 10.0],
                    [20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0],
                ],
            }
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    json_path = tmp_path / "multipoly.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    labels = coco.read_labels(json_path, dataset_root=tmp_path)
    # Two disjoint rings of one annotation collapse into a single object mask
    # spanning both squares.
    assert len(labels.masks) == 1
    m = labels.masks[0]
    assert m.area > 0
    bx, by, bw, bh = m.bbox
    assert bx == pytest.approx(1.0, abs=1.0)
    assert (bx + bw) == pytest.approx(30.0, abs=1.0)
    assert by == pytest.approx(1.0, abs=1.0)
    assert (by + bh) == pytest.approx(30.0, abs=1.0)


def test_coco_polygon_degenerate_ring_skipped(tmp_path):
    """A polygon ring with fewer than 3 vertices is skipped, not crashed on."""
    img_path = tmp_path / "img.png"
    img_path.touch()
    data = {
        "images": [{"id": 1, "file_name": "img.png", "height": 40, "width": 40}],
        "annotations": [
            # Degenerate ring (a single point): cannot form a polygon.
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[5.0, 5.0]]},
            # Valid ring alongside it on another object.
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0, 10.0]],
            },
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    json_path = tmp_path / "degenerate.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    # Does not raise; the degenerate ring yields no mask, the valid one does.
    labels = coco.read_labels(json_path, dataset_root=tmp_path)
    assert len(labels.masks) == 1
    assert labels.masks[0].area > 0


def test_coco_segmentation_load_file_and_slp_roundtrip(tmp_path):
    """Keypoint-free segmentation COCO auto-detects and round-trips via .slp."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img_path = tmp_path / "frame.png"
    iio.imwrite(str(img_path), img)
    data = {
        "images": [{"id": 1, "file_name": "frame.png", "height": 20, "width": 20}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2.0, 2.0, 15.0, 2.0, 15.0, 15.0, 2.0, 15.0]],
                "bbox": [2, 2, 13, 13],
                "area": 169,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "critter"}],
    }
    json_path = tmp_path / "seg.coco.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    # load_file auto-detects COCO despite the absence of keypoints.
    labels = sio.load_file(str(json_path))
    assert sum(len(lf.masks) for lf in labels.labeled_frames) == 1
    mask = labels.masks[0]
    assert isinstance(mask, UserSegmentationMask)
    assert mask.category.name == "critter"

    # Round-trip through .slp preserves the mask, category, and area.
    slp_path = tmp_path / "seg.slp"
    sio.save_file(labels, str(slp_path))
    reloaded = sio.load_file(str(slp_path))
    rmasks = [m for lf in reloaded.labeled_frames for m in lf.masks]
    assert len(rmasks) == 1
    assert rmasks[0].category.name == "critter"
    assert rmasks[0].area == mask.area


def test_coco_load_file_forwards_kwargs(tmp_path):
    """load_file forwards segmentation_format and category_as_track to load_coco."""
    img_path = tmp_path / "frame.png"
    img_path.touch()
    data = {
        "images": [{"id": 1, "file_name": "frame.png", "height": 30, "width": 30}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2.0, 2.0, 15.0, 2.0, 15.0, 15.0, 2.0, 15.0]],
            }
        ],
        "categories": [{"id": 1, "name": "critter"}],
    }
    json_path = tmp_path / "kw.coco.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    # segmentation_format="roi" forwarded -> polygon kept as vector ROI.
    roi_labels = sio.load_file(str(json_path), segmentation_format="roi")
    assert len(roi_labels.rois) == 1
    assert len(roi_labels.masks) == 0

    # category_as_track=True forwarded -> identity Track named after the category.
    track_labels = sio.load_file(str(json_path), category_as_track=True)
    assert {t.name for t in track_labels.tracks} == {"critter"}
    assert track_labels.masks[0].track.name == "critter"


def test_coco_category_as_track(tmp_path):
    """category_as_track assigns one shared Track per category as identity."""
    (tmp_path / "img1.png").touch()
    (tmp_path / "img2.png").touch()
    # Two categories ("a", "b") appearing across two frames.
    data = {
        "images": [
            {"id": 1, "file_name": "img1.png", "height": 40, "width": 40},
            {"id": 2, "file_name": "img2.png", "height": 40, "width": 40},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0, 10.0]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0]],
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [[2.0, 2.0, 12.0, 2.0, 12.0, 12.0, 2.0, 12.0]],
            },
            # A bbox-only annotation (no segmentation) for category "b".
            {
                "id": 4,
                "image_id": 2,
                "category_id": 2,
                "bbox": [5.0, 5.0, 8.0, 8.0],
            },
        ],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    }
    json_path = tmp_path / "ident.coco.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    labels = coco.read_labels(json_path, dataset_root=tmp_path, category_as_track=True)

    # One Track per category, collected onto the Labels.
    assert {t.name for t in labels.tracks} == {"a", "b"}
    # Every mask's track name equals its category.
    for m in labels.masks:
        assert m.track is not None
        assert m.track.name == m.category.name
    # The bbox-only annotation also gets its category track.
    assert len(labels.bboxes) == 1
    assert labels.bboxes[0].track is not None
    assert labels.bboxes[0].track.name == labels.bboxes[0].category.name == "b"
    # The bbox and the masks of category "b" share the same Track object.
    b_mask = next(m for m in labels.masks if m.category.name == "b")
    assert labels.bboxes[0].track is b_mask.track
    # The two "a" masks (different frames) share the same Track object.
    a_masks = [m for m in labels.masks if m.category.name == "a"]
    assert len(a_masks) == 2
    assert a_masks[0].track is a_masks[1].track

    # Identity tracks survive a .slp round-trip.
    slp_path = tmp_path / "ident.slp"
    sio.save_file(labels, str(slp_path))
    reloaded = sio.load_file(str(slp_path))
    assert {t.name for t in reloaded.tracks} == {"a", "b"}
    for m in reloaded.masks:
        assert m.track is not None
        assert m.track.name == m.category.name

    # roi mode also assigns category tracks to the vector ROIs.
    roi_labels = coco.read_labels(
        json_path,
        dataset_root=tmp_path,
        segmentation_format="roi",
        category_as_track=True,
    )
    assert {t.name for t in roi_labels.tracks} == {"a", "b"}
    for r in roi_labels.rois:
        assert r.track is not None
        assert r.track.name == r.category.name

    # Default (category_as_track=False) leaves masks and bboxes untracked.
    untracked = coco.read_labels(json_path, dataset_root=tmp_path)
    assert len(untracked.tracks) == 0
    assert all(m.track is None for m in untracked.masks)
    assert all(b.track is None for b in untracked.bboxes)


def test_coco_detection_reads_as_bbox(tmp_path):
    """Detection-only annotations with bbox create BoundingBox."""
    img_path = tmp_path / "det.png"
    img_path.touch()

    data = {
        "images": [
            {"id": 1, "file_name": "det.png", "height": 200, "width": 300},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [15, 25, 50, 60],
                "area": 3000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 110, 40, 30],
                "area": 1200,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }

    json_path = tmp_path / "det.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    labels = coco.read_labels(json_path, dataset_root=tmp_path)

    assert len(labels.bboxes) == 2
    assert len(labels.rois) == 0
    assert len(labels.masks) == 0

    for bbox in labels.bboxes:
        assert isinstance(bbox, UserBoundingBox)
        assert bbox.category.name == "person"

    # Check first bbox coordinates
    x, y, w, h = labels.bboxes[0].xywh
    assert x == pytest.approx(15.0)
    assert y == pytest.approx(25.0)
    assert w == pytest.approx(50.0)
    assert h == pytest.approx(60.0)

    # Check second bbox coordinates
    x, y, w, h = labels.bboxes[1].xywh
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(110.0)
    assert w == pytest.approx(40.0)
    assert h == pytest.approx(30.0)


def test_coco_predicted_bbox(tmp_path):
    """Annotations with score field should create PredictedBoundingBox."""
    img_path = tmp_path / "pred.png"
    img_path.touch()

    data = {
        "images": [
            {"id": 1, "file_name": "pred.png", "height": 100, "width": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
                "score": 0.95,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 60, 20, 15],
                "area": 300,
                "iscrowd": 0,
                "score": 0.42,
            },
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }

    json_path = tmp_path / "pred.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    labels = coco.read_labels(json_path, dataset_root=tmp_path)

    assert len(labels.bboxes) == 2
    for bbox in labels.bboxes:
        assert isinstance(bbox, PredictedBoundingBox)
        assert bbox.is_predicted

    assert labels.bboxes[0].score == pytest.approx(0.95)
    assert labels.bboxes[1].score == pytest.approx(0.42)

    x, y, w, h = labels.bboxes[0].xywh
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(20.0)
    assert w == pytest.approx(30.0)
    assert h == pytest.approx(40.0)


def test_coco_scored_segmentation_is_predicted(tmp_path):
    """A `score` on a detection annotation yields predicted mask/ROI variants."""
    img_path = tmp_path / "pred.png"
    img_path.touch()
    data = {
        "images": [{"id": 1, "file_name": "pred.png", "height": 30, "width": 30}],
        "annotations": [
            # Scored polygon -> PredictedSegmentationMask.
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "score": 0.9,
                "segmentation": [[2.0, 2.0, 15.0, 2.0, 15.0, 15.0, 2.0, 15.0]],
            },
            # Scored RLE -> PredictedSegmentationMask.
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "score": 0.7,
                "segmentation": {"counts": [0, 5, 5, 5, 5], "size": [5, 5]},
            },
            # Unscored polygon -> UserSegmentationMask.
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[20.0, 20.0, 28.0, 20.0, 28.0, 28.0, 20.0, 28.0]],
            },
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    json_path = tmp_path / "pred_seg.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Default mask mode: scored -> predicted (with score), unscored -> user.
    labels = coco.read_labels(json_path, dataset_root=tmp_path)
    masks = labels.labeled_frames[0].masks
    assert len(masks) == 3
    predicted = [m for m in masks if isinstance(m, PredictedSegmentationMask)]
    user = [m for m in masks if type(m) is UserSegmentationMask]
    assert len(predicted) == 2
    assert len(user) == 1
    assert sorted(m.score for m in predicted) == pytest.approx([0.7, 0.9])

    # roi mode: scored polygon -> PredictedROI; the scored RLE is still a mask.
    roi_labels = coco.read_labels(
        json_path, dataset_root=tmp_path, segmentation_format="roi"
    )
    rois = roi_labels.labeled_frames[0].rois
    assert [type(r) for r in rois] == [PredictedROI, UserROI]
    assert rois[0].score == pytest.approx(0.9)
    assert len(roi_labels.labeled_frames[0].masks) == 1  # RLE stays a mask

    # Predicted masks survive a .slp round-trip with class and score intact.
    slp_path = tmp_path / "pred_seg.slp"
    sio.save_file(labels, str(slp_path))
    reloaded = sio.load_file(str(slp_path))
    rmasks = reloaded.labeled_frames[0].masks
    rpred = [m for m in rmasks if isinstance(m, PredictedSegmentationMask)]
    assert len(rpred) == 2
    assert sorted(m.score for m in rpred) == pytest.approx([0.7, 0.9])
    assert sum(1 for m in rmasks if type(m) is UserSegmentationMask) == 1


def test_coco_bbox_roundtrip(tmp_path):
    """Write bboxes to COCO and read back as BoundingBox."""
    video = sio.Video.from_filename(["img1.png"])
    bbox1 = UserBoundingBox.from_xywh(10, 20, 50, 30, category="dog")
    bbox2 = PredictedBoundingBox.from_xywh(100, 200, 80, 60, category="cat", score=0.88)

    lf = sio.LabeledFrame(video=video, frame_idx=0)
    lf.bboxes.extend([bbox1, bbox2])
    labels = sio.Labels(labeled_frames=[lf])

    json_path = tmp_path / "bbox_rt.json"
    coco.write_labels(labels, json_path)

    # Verify JSON structure
    with open(json_path, "r") as f:
        data = json.load(f)

    assert len(data["annotations"]) == 2

    ann1 = data["annotations"][0]
    assert ann1["bbox"] == [10.0, 20.0, 50.0, 30.0]
    assert ann1["area"] == pytest.approx(1500.0)
    assert "score" not in ann1

    ann2 = data["annotations"][1]
    assert ann2["bbox"] == [100.0, 200.0, 80.0, 60.0]
    assert ann2["area"] == pytest.approx(4800.0)
    assert ann2["score"] == pytest.approx(0.88)

    # Verify category names
    cat_names = {c["name"] for c in data["categories"]}
    assert "dog" in cat_names
    assert "cat" in cat_names

    # Now read back and verify round-trip
    # Create image files matching auto-generated names
    for img_info in data["images"]:
        (tmp_path / img_info["file_name"]).touch()

    labels_rt = coco.read_labels(json_path, dataset_root=tmp_path)

    assert len(labels_rt.bboxes) == 2

    # First bbox should be UserBoundingBox (no score in ann1)
    assert isinstance(labels_rt.bboxes[0], UserBoundingBox)
    assert not isinstance(labels_rt.bboxes[0], PredictedBoundingBox)
    x, y, w, h = labels_rt.bboxes[0].xywh
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(20.0)
    assert w == pytest.approx(50.0)
    assert h == pytest.approx(30.0)

    # Second bbox should be PredictedBoundingBox with score
    assert isinstance(labels_rt.bboxes[1], PredictedBoundingBox)
    assert labels_rt.bboxes[1].score == pytest.approx(0.88)
    x, y, w, h = labels_rt.bboxes[1].xywh
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(200.0)
    assert w == pytest.approx(80.0)
    assert h == pytest.approx(60.0)


def test_coco_mixed_bbox_and_instances(tmp_path):
    """Keypoints + bbox annotation should create Instance + linked BoundingBox."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "mixed.png"
    iio.imwrite(str(img_path), img)

    data = {
        "images": [
            {"id": 1, "file_name": "mixed.png", "height": 100, "width": 100},
        ],
        "categories": [
            {
                "id": 1,
                "name": "animal",
                "keypoints": ["head", "tail"],
                "skeleton": [[1, 2]],
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [30, 30, 2, 70, 70, 2],
                "num_keypoints": 2,
                "bbox": [20, 20, 60, 60],
                "area": 3600,
            },
        ],
    }

    json_path = tmp_path / "mixed.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    labels = coco.read_labels(json_path, dataset_root=tmp_path)

    # Should have one instance
    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].instances) == 1
    instance = labels.labeled_frames[0].instances[0]

    # Should have one BoundingBox linked to the instance
    assert len(labels.bboxes) == 1
    bbox = labels.bboxes[0]
    assert isinstance(bbox, UserBoundingBox)
    assert bbox.instance is instance
    assert bbox.category.name == "animal"

    x, y, w, h = bbox.xywh
    assert x == pytest.approx(20.0)
    assert y == pytest.approx(20.0)
    assert w == pytest.approx(60.0)
    assert h == pytest.approx(60.0)

    # No ROIs or masks
    assert len(labels.rois) == 0
    assert len(labels.masks) == 0


class TestCOCOPanoptic:
    """Test COCO panoptic segmentation I/O."""

    def _create_panoptic_dataset(self, base_dir):
        """Helper to create a minimal COCO panoptic dataset on disk.

        Creates two frames with thing (person) and stuff (sky) segments.
        Returns (json_path, images_dir, expected segment IDs).
        """
        from PIL import Image

        images_dir = base_dir / "panoptic"
        images_dir.mkdir()

        # Categories: person (thing), sky (stuff)
        categories = [
            {"id": 1, "name": "person", "isthing": 1},
            {"id": 2, "name": "sky", "isthing": 0},
        ]

        # Frame 1: person segment (id=1) and sky segment (id=2)
        label1 = np.zeros((20, 30), dtype=np.int32)
        label1[5:15, 5:10] = 1  # person
        label1[0:5, :] = 2  # sky

        # Frame 2: same person (id=1) reappears, different sky (id=3)
        label2 = np.zeros((20, 30), dtype=np.int32)
        label2[8:18, 10:20] = 1  # same person
        label2[0:3, :] = 3  # sky (different segment id, same category)

        # Encode as RGB PNGs
        for fname, data in [("frame1.png", label1), ("frame2.png", label2)]:
            rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
            rgb[:, :, 0] = (data % 256).astype(np.uint8)
            rgb[:, :, 1] = ((data // 256) % 256).astype(np.uint8)
            rgb[:, :, 2] = ((data // 65536) % 256).astype(np.uint8)
            Image.fromarray(rgb).save(images_dir / fname)

        # Build JSON
        coco_data = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 30, "height": 20},
                {"id": 2, "file_name": "img2.jpg", "width": 30, "height": 20},
            ],
            "annotations": [
                {
                    "image_id": 1,
                    "file_name": "frame1.png",
                    "segments_info": [
                        {"id": 1, "category_id": 1, "area": 50, "iscrowd": 0},
                        {"id": 2, "category_id": 2, "area": 150, "iscrowd": 0},
                    ],
                },
                {
                    "image_id": 2,
                    "file_name": "frame2.png",
                    "segments_info": [
                        {"id": 1, "category_id": 1, "area": 100, "iscrowd": 0},
                        {"id": 3, "category_id": 2, "area": 90, "iscrowd": 0},
                    ],
                },
            ],
            "categories": categories,
        }

        json_path = base_dir / "panoptic.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        return json_path, images_dir

    def test_read_coco_panoptic(self, tmp_path):
        """Test reading a COCO panoptic dataset."""
        json_path, images_dir = self._create_panoptic_dataset(tmp_path)

        labels = coco.read_coco_panoptic(json_path, images_dir=images_dir)

        # Should have 2 label images (one per frame)
        assert len(labels.label_images) == 2

        li1 = labels.label_images[0]
        li2 = labels.label_images[1]

        # Check frame indices (positional, on labeled frames)
        assert labels.labeled_frames[0].frame_idx == 0
        assert labels.labeled_frames[1].frame_idx == 1

        # Check dimensions
        assert li1.height == 20
        assert li1.width == 30
        assert li2.height == 20
        assert li2.width == 30

        # Frame 1: 2 objects (person id=1, sky id=2)
        assert li1.n_objects == 2
        assert 1 in li1.objects
        assert 2 in li1.objects

        # Check categories
        assert li1.objects[1].category == "person"
        assert li1.objects[2].category == "sky"

        # Thing (person) should have a track, stuff (sky) should not
        assert li1.objects[1].track is not None
        assert li1.objects[2].track is None

        # Frame 2: 2 objects (person id=1, sky id=3)
        assert li2.n_objects == 2
        assert 1 in li2.objects
        assert 3 in li2.objects

        assert li2.objects[1].category == "person"
        assert li2.objects[3].category == "sky"
        assert li2.objects[1].track is not None
        assert li2.objects[3].track is None

        # Same person (id=1) should share the same Track object across frames
        assert li1.objects[1].track is li2.objects[1].track

        # Verify pixel data is correct
        assert np.sum(li1.data == 1) == 50  # person pixels in frame 1
        assert np.sum(li1.data == 2) == 150  # sky pixels in frame 1

    def test_read_coco_panoptic_infer_images_dir(self, tmp_path):
        """Test that images_dir defaults to the JSON directory."""
        from PIL import Image

        # Create a simple single-frame dataset with PNGs in the same dir as JSON
        label_data = np.zeros((10, 10), dtype=np.int32)
        label_data[2:8, 2:8] = 1

        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = (label_data % 256).astype(np.uint8)
        Image.fromarray(rgb).save(tmp_path / "seg.png")

        coco_data = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
            "annotations": [
                {
                    "image_id": 1,
                    "file_name": "seg.png",
                    "segments_info": [
                        {"id": 1, "category_id": 1, "area": 36, "iscrowd": 0},
                    ],
                },
            ],
            "categories": [{"id": 1, "name": "cell", "isthing": 1}],
        }

        json_path = tmp_path / "panoptic.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        # Should work without explicit images_dir
        labels = coco.read_coco_panoptic(json_path)
        assert len(labels.label_images) == 1
        assert labels.label_images[0].n_objects == 1

    def test_read_coco_panoptic_missing_png(self, tmp_path):
        """Test that missing PNGs are silently skipped."""
        from PIL import Image

        # Create one valid frame
        label_data = np.zeros((10, 10), dtype=np.int32)
        label_data[2:8, 2:8] = 1

        images_dir = tmp_path / "panoptic"
        images_dir.mkdir()

        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = (label_data % 256).astype(np.uint8)
        Image.fromarray(rgb).save(images_dir / "exists.png")

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "img2.jpg", "width": 10, "height": 10},
            ],
            "annotations": [
                {
                    "image_id": 1,
                    "file_name": "exists.png",
                    "segments_info": [
                        {"id": 1, "category_id": 1, "area": 36, "iscrowd": 0},
                    ],
                },
                {
                    "image_id": 2,
                    "file_name": "missing.png",
                    "segments_info": [
                        {"id": 2, "category_id": 1, "area": 10, "iscrowd": 0},
                    ],
                },
            ],
            "categories": [{"id": 1, "name": "cell", "isthing": 1}],
        }

        json_path = tmp_path / "panoptic.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        labels = coco.read_coco_panoptic(json_path, images_dir=images_dir)
        # Only the frame with existing PNG should be loaded
        assert len(labels.label_images) == 1
        assert labels.label_images[0].n_objects == 1
        # frame_idx should be contiguous (0), not the annotation index (0 with gap)
        assert labels.labeled_frames[0].frame_idx == 0

    def test_write_coco_panoptic(self, tmp_path):
        """Test writing COCO panoptic from Labels with label_images."""
        from sleap_io.model.label_image import LabelImage, UserLabelImage

        track_a = Track(name="1")
        track_b = Track(name="2")

        # Create a label image with 2 objects
        data = np.zeros((10, 15), dtype=np.int32)
        data[2:5, 3:8] = 1  # object 1
        data[6:9, 1:4] = 2  # object 2

        objects = {
            1: LabelImage.Info(track=track_a, category="cell"),
            2: LabelImage.Info(track=track_b, category="background"),
        }
        li = UserLabelImage(data=data, objects=objects)
        video = sio.Video(filename="dummy.mp4", open_backend=False)
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.label_images.append(li)
        labels = Labels(labeled_frames=[lf])

        # Write
        json_path = tmp_path / "output.json"
        coco.write_coco_panoptic(json_path, labels)

        # Verify JSON was created
        assert json_path.exists()

        with open(json_path, "r") as f:
            written = json.load(f)

        # Check structure
        assert len(written["images"]) == 1
        assert len(written["annotations"]) == 1
        assert len(written["categories"]) == 2

        # Check categories
        cat_names = {c["name"] for c in written["categories"]}
        assert "cell" in cat_names
        assert "background" in cat_names

        # Check isthing flags
        cat_by_name = {c["name"]: c for c in written["categories"]}
        assert cat_by_name["cell"]["isthing"] == 1  # has track
        assert cat_by_name["background"]["isthing"] == 1  # also has track

        # Check segments_info
        ann = written["annotations"][0]
        assert len(ann["segments_info"]) == 2
        seg_ids = {s["id"] for s in ann["segments_info"]}
        assert seg_ids == {1, 2}

        # Check bbox field is present and correct per COCO spec [x, y, w, h]
        seg_by_id = {s["id"]: s for s in ann["segments_info"]}
        # Object 1: data[2:5, 3:8] → bbox [3, 2, 5, 3]
        assert seg_by_id[1]["bbox"] == [3, 2, 5, 3]
        # Object 2: data[6:9, 1:4] → bbox [1, 6, 3, 3]
        assert seg_by_id[2]["bbox"] == [1, 6, 3, 3]

        # Verify PNG was created in the default subdirectory
        images_dir = tmp_path / "output_panoptic"
        assert images_dir.exists()
        png_files = list(images_dir.glob("*.png"))
        assert len(png_files) == 1

    def test_write_coco_panoptic_custom_images_dir(self, tmp_path):
        """Test writing PNGs to a custom directory."""
        from sleap_io.model.label_image import LabelImage, UserLabelImage

        data = np.zeros((5, 5), dtype=np.int32)
        data[1:4, 1:4] = 1
        objects = {1: LabelImage.Info(category="obj")}
        li = UserLabelImage(data=data, objects=objects)
        video = sio.Video(filename="dummy.mp4", open_backend=False)
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        lf.label_images.append(li)
        labels = Labels(labeled_frames=[lf])

        custom_dir = tmp_path / "my_pngs"
        json_path = tmp_path / "out.json"
        coco.write_coco_panoptic(json_path, labels, images_dir=custom_dir)

        assert custom_dir.exists()
        assert len(list(custom_dir.glob("*.png"))) == 1

    def test_coco_panoptic_roundtrip(self, tmp_path):
        """Test write then read back, verify data and metadata match."""
        from sleap_io.model.label_image import LabelImage, UserLabelImage

        track1 = Track(name="obj_1")
        track2 = Track(name="obj_2")

        # Frame 1: two things
        data1 = np.zeros((20, 25), dtype=np.int32)
        data1[2:8, 3:10] = 1
        data1[10:18, 5:20] = 2
        objects1 = {
            1: LabelImage.Info(track=track1, category="animal"),
            2: LabelImage.Info(track=track2, category="animal"),
        }
        li1 = UserLabelImage(data=data1, objects=objects1)

        # Frame 2: one thing, one stuff
        data2 = np.zeros((20, 25), dtype=np.int32)
        data2[0:5, 0:25] = 3  # stuff (no track)
        data2[10:15, 10:20] = 1  # same track as frame 1
        objects2 = {
            3: LabelImage.Info(track=None, category="background"),
            1: LabelImage.Info(track=track1, category="animal"),
        }
        li2 = UserLabelImage(data=data2, objects=objects2)

        video = sio.Video(filename="dummy.mp4", open_backend=False)
        lf1 = sio.LabeledFrame(video=video, frame_idx=0)
        lf1.label_images.append(li1)
        lf2 = sio.LabeledFrame(video=video, frame_idx=1)
        lf2.label_images.append(li2)
        labels = Labels(labeled_frames=[lf1, lf2])

        # Write
        json_path = tmp_path / "roundtrip.json"
        images_dir = tmp_path / "roundtrip_pngs"
        coco.write_coco_panoptic(json_path, labels, images_dir=images_dir)

        # Read back
        labels_rt = coco.read_coco_panoptic(json_path, images_dir=images_dir)

        # Same number of label images
        assert len(labels_rt.label_images) == 2

        rt1 = labels_rt.label_images[0]
        rt2 = labels_rt.label_images[1]

        # Pixel data should match exactly
        np.testing.assert_array_equal(rt1.data, data1)
        np.testing.assert_array_equal(rt2.data, data2)

        # Same number of objects
        assert len(rt1.objects) == 2
        assert len(rt2.objects) == 2

        # Categories should match
        assert rt1.objects[1].category == "animal"
        assert rt1.objects[2].category == "animal"
        assert rt2.objects[3].category == "background"
        assert rt2.objects[1].category == "animal"

        # Thing tracks should be shared across frames
        assert rt1.objects[1].track is rt2.objects[1].track

        # Stuff should have no track
        assert rt2.objects[3].track is None

        # Frame indices preserved (on labeled frames)
        assert labels_rt.labeled_frames[0].frame_idx == 0
        assert labels_rt.labeled_frames[1].frame_idx == 1

        # Dimensions preserved
        assert rt1.height == 20


def test_coco_export_roi_only_creates_image(tmp_path):
    """Export ROIs/masks/bboxes on frames without instances creates image entries."""
    from sleap_io.model.roi import UserROI

    video = sio.Video.from_filename(["img1.png"])

    # Frame with only ROI (no instances) — forces new image_id creation
    roi = UserROI.from_bbox(10.0, 20.0, 50.0, 30.0, category="region", video=video)
    mask_arr = np.zeros((10, 10), dtype=bool)
    mask_arr[2:5, 3:7] = True
    mask = UserSegmentationMask.from_numpy(mask_arr, category="cell")
    bbox = UserBoundingBox.from_xywh(0, 0, 20, 20, category="box")

    lf = sio.LabeledFrame(video=video, frame_idx=0)
    lf.rois.append(roi)
    lf.masks.append(mask)
    lf.bboxes.append(bbox)
    labels = sio.Labels(labeled_frames=[lf])

    json_path = tmp_path / "test.json"
    coco.write_labels(labels, json_path)

    with open(json_path) as f:
        data = json.load(f)

    # All three annotation types should have created entries
    assert len(data["annotations"]) == 3
    # Image entry created for the frame
    assert len(data["images"]) == 1
    # Three distinct categories
    cat_names = {c["name"] for c in data["categories"]}
    assert cat_names == {"region", "cell", "box"}
