"""Fixtures that return paths to COCO-style pose datasets.

These fixtures provide various COCO dataset variants to test different directory
structures, image organization patterns, and keypoint visibility encodings.
"""

import pytest


@pytest.fixture
def coco_flat_images():
    """Basic COCO dataset with flat image directory structure.

    This dataset:
    - Has images in a simple 'images/' folder
    - Contains 3 images with mouse pose annotations
    - Uses standard COCO keypoint format with 17 keypoints
    - Uses ternary visibility encoding (0/1/2)
    - Has simple sequential image naming (image_001.jpg, etc.)
    """
    return "tests/data/coco/flat_images"


@pytest.fixture
def coco_category_folders():
    """COCO dataset with images organized by category subfolders.

    This dataset:
    - Has images organized in category subdirectories (mouse/, fly/)
    - Contains multiple animal categories with different skeletons
    - Mouse category: 17 keypoints, 2 images
    - Fly category: 13 keypoints, 1 image
    - Uses category-specific image naming
    """
    return "tests/data/coco/category_folders"


@pytest.fixture
def coco_multi_source():
    """COCO dataset with images from multiple source directories.

    This dataset:
    - Has images organized by source (source1/, source2/)
    - Simulates data collected from different sessions/setups
    - Contains 3 total images across 2 source directories
    - Uses consistent mouse skeleton (17 keypoints)
    - File paths include source directory prefix
    """
    return "tests/data/coco/multi_source"


@pytest.fixture
def coco_nested_paths():
    """COCO dataset with deeply nested image directory structure.

    This dataset:
    - Has images in deeply nested paths (data/experiment_1/session_a/images/)
    - Simulates hierarchical experimental organization
    - Contains 3 images with sequential frame numbering
    - Uses mouse skeleton with 17 keypoints
    - File paths reflect full nested structure
    """
    return "tests/data/coco/nested_paths"


@pytest.fixture
def coco_visibility_binary():
    """COCO dataset using binary visibility encoding (0/1).

    This dataset:
    - Uses binary visibility: 0=not visible, 1=visible
    - Contains 3 images with mouse pose annotations
    - Some keypoints marked as invisible to test visibility handling
    - Simulates partial occlusion scenarios
    - Uses 17-keypoint mouse skeleton
    """
    return "tests/data/coco/visibility_binary"


@pytest.fixture
def coco_visibility_ternary():
    """COCO dataset using ternary visibility encoding (0/1/2).

    This dataset:
    - Uses ternary visibility: 0=not labeled, 1=labeled but not visible,
      2=labeled and visible
    - Contains 3 images with mouse pose annotations
    - Mix of all three visibility states to test comprehensive visibility handling
    - Uses 17-keypoint mouse skeleton
    - Follows COCO standard visibility encoding
    """
    return "tests/data/coco/visibility_ternary"


@pytest.fixture
def coco_mixed_animals():
    """COCO dataset with multiple animal categories in the same images.

    This dataset:
    - Contains both mouse and fly annotations in each image
    - Mouse: 17 keypoints with standard mammalian skeleton
    - Fly: 13 keypoints with insect-specific skeleton
    - Each image has 2 instances (1 mouse, 1 fly)
    - Tests multi-category pose tracking scenarios
    """
    return "tests/data/coco/mixed_animals"


@pytest.fixture
def coco_annotations_flat():
    """Path to annotations file for flat_images dataset."""
    return "tests/data/coco/flat_images/annotations.json"


@pytest.fixture
def coco_annotations_categories():
    """Path to annotations file for category_folders dataset."""
    return "tests/data/coco/category_folders/annotations.json"


@pytest.fixture
def coco_annotations_multi_source():
    """Path to annotations file for multi_source dataset."""
    return "tests/data/coco/multi_source/annotations.json"


@pytest.fixture
def coco_annotations_nested():
    """Path to annotations file for nested_paths dataset."""
    return "tests/data/coco/nested_paths/annotations.json"


@pytest.fixture
def coco_annotations_visibility_binary():
    """Path to annotations file for visibility_binary dataset."""
    return "tests/data/coco/visibility_binary/annotations.json"


@pytest.fixture
def coco_annotations_visibility_ternary():
    """Path to annotations file for visibility_ternary dataset."""
    return "tests/data/coco/visibility_ternary/annotations.json"


@pytest.fixture
def coco_annotations_mixed_animals():
    """Path to annotations file for mixed_animals dataset."""
    return "tests/data/coco/mixed_animals/annotations.json"
