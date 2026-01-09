"""Tests for sleap_io.io.coco module."""

import json
from pathlib import Path

import numpy as np
import pytest

import sleap_io as sio
from sleap_io.io import coco
from sleap_io.model.matching import (
    IMAGE_DEDUP_VIDEO_MATCHER,
    SHAPE_VIDEO_MATCHER,
)


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
        assert len(coco_data["annotations"]) == total_instances
        assert len(coco_data["categories"]) == len(labels.skeletons)

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

    def test_roundtrip_conversion(self, coco_flat_images, tmp_path):
        """Test that data survives a roundtrip conversion."""
        # Load original
        original_labels = coco.read_labels(Path(coco_flat_images) / "annotations.json")

        # Write to new file
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
