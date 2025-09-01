"""Fixtures that provide CVAT-format datasets for testing.

CVAT (Computer Vision Annotation Tool) exports annotations in a COCO-compatible format
with additional attributes for tracking and metadata.
"""

import json

import pytest


@pytest.fixture
def cvat_tracking_dataset(tmp_path):
    """Create a minimal CVAT dataset with object tracking.

    This dataset:
    - Has 2 frames with 2 tracked mouse instances each
    - Uses CVAT's attributes.object_id for tracking
    - Tests track continuity across frames
    - Uses a simple 3-keypoint skeleton (nose, head, tail)
    - All keypoints are visible
    """
    # Create annotation data
    annotations = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "CVAT tracking test dataset",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": [
            {
                "id": 1,
                "name": "mouse",
                "supercategory": "",
                "keypoints": ["nose", "head", "tail"],
                "skeleton": [],  # Empty skeleton connections (CVAT style)
            }
        ],
        "images": [
            {
                "id": 1,
                "width": 100,
                "height": 100,
                "file_name": "frame01.jpg",
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            },
            {
                "id": 2,
                "width": 100,
                "height": 100,
                "file_name": "frame02.jpg",
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            },
        ],
        "annotations": [
            # Frame 1: Two mice
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [],
                "area": 100.0,
                "bbox": [10, 10, 20, 20],
                "iscrowd": 0,
                "attributes": {"object_id": 101},  # Mouse 1
                "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 2],  # nose, head, tail
                "num_keypoints": 3,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [],
                "area": 100.0,
                "bbox": [40, 40, 20, 20],
                "iscrowd": 0,
                "attributes": {"object_id": 102},  # Mouse 2
                "keypoints": [40, 40, 2, 50, 50, 2, 60, 60, 2],
                "num_keypoints": 3,
            },
            # Frame 2: Same two mice tracked
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [],
                "area": 100.0,
                "bbox": [12, 12, 20, 20],
                "iscrowd": 0,
                "attributes": {"object_id": 101},  # Mouse 1 (moved slightly)
                "keypoints": [12, 12, 2, 22, 22, 2, 32, 32, 2],
                "num_keypoints": 3,
            },
            {
                "id": 4,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [],
                "area": 100.0,
                "bbox": [42, 42, 20, 20],
                "iscrowd": 0,
                "attributes": {"object_id": 102},  # Mouse 2 (moved slightly)
                "keypoints": [42, 42, 2, 52, 52, 2, 62, 62, 2],
                "num_keypoints": 3,
            },
        ],
    }

    # Write JSON file
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)

    # Create dummy image files
    for img_info in annotations["images"]:
        img_path = tmp_path / img_info["file_name"]
        # Just create empty files since the loader will be mocked
        img_path.touch()

    return tmp_path


@pytest.fixture
def cvat_with_occluded(tmp_path):
    """Create a CVAT dataset with occlusion attributes.

    This dataset:
    - Has instances with occluded=true/false attributes
    - Tests handling of CVAT-specific metadata
    - Contains partial visibility in keypoints
    """
    annotations = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "CVAT occlusion test dataset",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": [
            {
                "id": 1,
                "name": "mouse",
                "supercategory": "",
                "keypoints": ["nose", "head", "tail"],
                "skeleton": [],
            }
        ],
        "images": [
            {
                "id": 1,
                "width": 100,
                "height": 100,
                "file_name": "frame01.jpg",
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [],
                "area": 100.0,
                "bbox": [10, 10, 20, 20],
                "iscrowd": 0,
                "attributes": {
                    "object_id": 201,
                    "occluded": False,
                    "keyframe": True,
                    "rotation": 0.0,
                },
                "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 2],  # All visible
                "num_keypoints": 3,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [],
                "area": 100.0,
                "bbox": [40, 40, 20, 20],
                "iscrowd": 0,
                "attributes": {
                    "object_id": 202,
                    "occluded": True,
                    "keyframe": False,
                    "rotation": 0.0,
                },
                "keypoints": [40, 40, 2, 50, 50, 2, 0, 0, 0],  # Tail not visible
                "num_keypoints": 2,
            },
        ],
    }

    # Write JSON file
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)

    # Create dummy image file
    (tmp_path / "frame01.jpg").touch()

    return tmp_path


@pytest.fixture
def cvat_alternative_track_fields(tmp_path):
    """Create datasets with alternative track ID field names.

    Tests compatibility with:
    - track_id (some COCO extensions)
    - instance_id (alternative naming)
    """
    # Dataset with track_id field
    track_id_data = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"description": "track_id test"},
        "categories": [
            {
                "id": 1,
                "name": "mouse",
                "keypoints": ["nose", "head", "tail"],
                "skeleton": [],
            }
        ],
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "frame01.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "track_id": 301,  # Using track_id instead of attributes.object_id
                "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 2],
                "num_keypoints": 3,
                "area": 100.0,
                "bbox": [10, 10, 20, 20],
                "iscrowd": 0,
            }
        ],
    }

    # Write track_id variant
    track_id_path = tmp_path / "track_id"
    track_id_path.mkdir()
    with open(track_id_path / "annotations.json", "w") as f:
        json.dump(track_id_data, f, indent=2)
    (track_id_path / "frame01.jpg").touch()

    # Dataset with instance_id field
    instance_id_data = track_id_data.copy()
    instance_id_data["info"]["description"] = "instance_id test"
    instance_id_data["annotations"][0] = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "instance_id": 401,  # Using instance_id
        "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 2],
        "num_keypoints": 3,
        "area": 100.0,
        "bbox": [10, 10, 20, 20],
        "iscrowd": 0,
    }

    # Write instance_id variant
    instance_id_path = tmp_path / "instance_id"
    instance_id_path.mkdir()
    with open(instance_id_path / "annotations.json", "w") as f:
        json.dump(instance_id_data, f, indent=2)
    (instance_id_path / "frame01.jpg").touch()

    return tmp_path
