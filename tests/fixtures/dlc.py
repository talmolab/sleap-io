"""Fixtures that return paths to DLC (DeepLabCut) files."""

import pytest


@pytest.fixture
def dlc_maudlc_testdata():
    """Multi-animal DLC CSV file with individual tracking (MAUDLC).

    This file:
    - Contains labeled data for multi-animal pose estimation
    - Has 3 tracks: "Animal1", "Animal2", "single"
    - Has 5 bodyparts: A, B, C (multi-animal) + D, E (unique bodyparts)
    - Uses DLC's hierarchical CSV format with scorer/individuals/bodyparts/coords
    - Contains 3 labeled frames out of 4 image files (img000-img003.png)
    - Frame 0: 2 instances, Frame 1: 3 instances, Frame 2: 2 instances
    - Many points have NaN coordinates (missing annotations)
    """
    return "tests/data/dlc/labeled-data/video/maudlc_testdata.csv"


@pytest.fixture
def dlc_madlc_testdata():
    """Multi-animal DLC CSV file (MADLC).

    This file:
    - Contains labeled data for multi-animal pose estimation
    - Has 3 bodyparts: A, B, C
    - Uses DLC's hierarchical CSV format with scorer/individuals/bodyparts/coords
    - Contains 3 labeled frames out of 4 image files (img000-img003.png)
    - Frame 0: 2 instances, Frame 1: 2 instances, Frame 2: 1 instance
    - Some points have NaN coordinates (missing annotations)
    """
    return "tests/data/dlc/labeled-data/video/madlc_testdata.csv"


@pytest.fixture
def dlc_madlc_testdata_v2():
    """Multi-animal DLC CSV file - variant 2.

    This file:
    - Contains labeled data for multi-animal pose estimation
    - Version 2 variant with different annotation structure
    - Has hierarchical CSV format with scorer/individuals/bodyparts/coords headers
    """
    return "tests/data/dlc/labeled-data/video/madlc_testdata_v2.csv"


@pytest.fixture
def dlc_maudlc_testdata_v2():
    """Multi-animal DLC CSV file - mau variant 2.

    This file:
    - Contains labeled data for multi-animal pose estimation
    - Mau version 2 variant with different annotation structure
    - Has hierarchical CSV format with scorer/individuals/bodyparts/coords headers
    """
    return "tests/data/dlc/labeled-data/video/maudlc_testdata_v2.csv"


@pytest.fixture
def dlc_testdata():
    """Single-animal DLC CSV file (SADLC).

    This file:
    - Contains labeled data for single-animal pose estimation
    - Has 3 bodyparts: A, B, C
    - Has simpler structure without individual tracking
    - Uses DLC's hierarchical CSV format with scorer/bodyparts/coords headers
    - Contains 3 labeled frames out of 4 image files (img000-img003.png)
    - 1 instance per frame
    - Some points have NaN coordinates (missing annotations)
    """
    return "tests/data/dlc/labeled-data/video/dlc_testdata.csv"


@pytest.fixture
def dlc_testdata_v2():
    """Single-animal DLC CSV file - version 2.

    This file:
    - Contains labeled data for single-animal pose estimation
    - Version 2 variant with different annotation structure
    - Has hierarchical CSV format with scorer/bodyparts/coords headers
    """
    return "tests/data/dlc/labeled-data/video/dlc_testdata_v2.csv"


@pytest.fixture
def dlc_collected_data():
    """DLC CollectedData CSV file.

    This file:
    - Contains collected annotation data in DLC format
    - Has the standard DLC CollectedData naming convention
    - Used for training and evaluation in DLC pipelines
    """
    return "tests/data/dlc/labeled-data/video/CollectedData_LM.csv"


@pytest.fixture
def dlc_config():
    """DLC project configuration YAML file.

    This file:
    - Contains DLC project configuration for multi-animal tracking
    - Defines task, scorer, project paths, and video sets
    - Lists individuals, bodyparts, and unique bodyparts
    - Includes cropping parameters and other project settings
    - Version: DLC 2.3.0 format
    """
    return "tests/data/dlc/madlc_230_config.yaml"


@pytest.fixture
def dlc_multiple_datasets_video1():
    """DLC dataset 1 from multiple datasets example.

    This file:
    - Contains labeled data for video1 in multi-dataset setup
    - Has 2 animals (Animal1, Animal2) with 3 bodyparts each (A, B, C)
    - Contains 3 labeled frames (img000-img002.jpg)
    - Some annotations are missing (empty cells)
    - Part of a multi-video dataset structure
    """
    return "tests/data/dlc_multiple_datasets/video1/dlc_dataset_1.csv"


@pytest.fixture
def dlc_multiple_datasets_video2():
    """DLC dataset 2 from multiple datasets example.

    This file:
    - Contains labeled data for video2 in multi-dataset setup
    - Has 2 animals (Animal1, Animal2) with 3 bodyparts each (A, B, C)
    - Contains 3 labeled frames (img000-img002.jpg)
    - Part of a multi-video dataset structure
    """
    return "tests/data/dlc_multiple_datasets/video2/dlc_dataset_2.csv"


@pytest.fixture
def dlc_image_dir():
    """Directory containing DLC labeled images.

    This directory:
    - Contains PNG images (img000-img003.png) used in DLC annotations
    - Images correspond to frames referenced in DLC CSV files
    - Used for loading image data alongside pose annotations
    """
    return "tests/data/dlc/labeled-data/video"


@pytest.fixture
def dlc_multiple_datasets_video1_images():
    """Directory containing images for video1 dataset.

    This directory:
    - Contains JPG images (img000-img002.jpg) for video1
    - Images correspond to frames in dlc_dataset_1.csv
    - Part of multi-video dataset structure
    """
    return "tests/data/dlc_multiple_datasets/video1"


@pytest.fixture
def dlc_multiple_datasets_video2_images():
    """Directory containing images for video2 dataset.

    This directory:
    - Contains JPG images (img000-img002.jpg) for video2
    - Images correspond to frames in dlc_dataset_2.csv
    - Part of multi-video dataset structure
    """
    return "tests/data/dlc_multiple_datasets/video2"
