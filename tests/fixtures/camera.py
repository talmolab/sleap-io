"""Fixtures that return `CameraGroup` and related objects."""

import cv2
import numpy as np
import pytest

import sleap_io


@pytest.fixture
def camera_group_345():
    """`CameraGroup` with 3-4-5 triangle configuration.

    The `CameraGroup` is configured such that the world point (4, 0, 0) - transformed to
    respective camera coordinates - is triangulated to the same point (5, 0, z) in both
    camera frames (with an arbitrary z), i.e.:
    - The `CameraGroup` object contains two `Camera`s placed at (0, 3, 0) and (0, -3, 0)
    - The `Camera`s are rotated 36.87 degrees around the z-axis in opposite directions

    E.g.:

    # Transform point from world to camera frame
    point_world = np.array([b, 0, 0])
    point_cam1 = rotation_matrix_1 @ point_world + tvec_1
    point_cam2 = rotation_matrix_2 @ point_world + tvec_2
    np.testing.assert_array_almost_equal(point_cam1, np.array([c, 0, 0]), decimal=5)
    np.testing.assert_array_almost_equal(point_cam2, np.array([c, 0, 0]), decimal=5)

    c1=(0, 3, 0)
    |
    |
    ---p=(4, 0, 0)
    |
    |
    c2=(0, -3, 0)

    Returns:
        `CameraGroup`: Camera group with 3-4-5 triangle configuration.
    """
    # Define special 3-4-5 triangle
    a = 3
    b = 4
    c = 5

    # Angles opposite to sides a, b, and c in radians
    angle_a = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))  # 36.87 degrees

    # Define camera origin and world point
    camera1_origin = np.array([0, a, 0])
    camera2_origin = np.array([0, -a, 0])

    # Define rotation and translation vectors
    rvec_1 = np.array([0, 0, 1]) * angle_a  # axis-angle representation
    rvec_2 = -rvec_1  # Opposite rotation
    rotation_matrix_1 = cv2.Rodrigues(rvec_1)[0]
    rotation_matrix_2 = cv2.Rodrigues(rvec_2)[0]
    tvec_1 = -rotation_matrix_1 @ camera1_origin  # Rotated camera origin
    tvec_2 = -rotation_matrix_2 @ camera2_origin  # Rotated camera origin

    # Define camera group
    camera_1 = sleap_io.Camera(rvec=rvec_1, tvec=tvec_1)
    camera_2 = sleap_io.Camera(rvec=rvec_2, tvec=tvec_2)
    camera_group = sleap_io.CameraGroup(cameras=[camera_1, camera_2])

    return camera_group


@pytest.fixture
def calibration_toml_path():
    """Path to a TOML file containing camera calibration data."""
    return "tests/data/camera/calibration.toml"
