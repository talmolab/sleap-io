"""Fixtures that return paths to AlphaTracker .json files."""

import pytest


@pytest.fixture
def alphatracker_testdata():
    """AlphaTracker annotation file with multi-animal pose data.

    This file:
    - Contains annotations for 4 frames (img000-img003.png)
    - Has 2 animals per frame with bounding boxes and keypoints
    - Each animal has a "Face" bounding box annotation (class: "Face")
    - Each animal has 3 keypoint annotations (class: "point")
    - Uses AlphaTracker's JSON format with image-level annotations
    - Keypoints are stored as x,y coordinates
    - Bounding boxes include x,y,width,height
    - Frame 0: 2 animals, Frame 1: 2 animals, Frame 2: 2 animals, Frame 3: 2 animals
    """
    return "tests/data/alphatracker/at_testdata.json"


@pytest.fixture
def alphatracker_image_dir():
    """Directory containing AlphaTracker test images.

    This directory:
    - Contains PNG images (img000-img003.png) used in AlphaTracker annotations
    - Images correspond to frames referenced in at_testdata.json
    - Used for loading image data alongside pose annotations
    """
    return "tests/data/alphatracker"
