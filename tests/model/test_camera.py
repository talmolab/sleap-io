"""Tests for methods in the sleap_io.model.instance file."""

import cv2
import numpy as np
import pytest

from sleap_io.model.camera import Camera, CameraGroup, RecordingSession
from sleap_io.model.video import Video


def test_camera_name():
    """Test camera name converter always converts to string."""
    # During initialization
    camera = Camera(name=12)
    assert camera.name == "12"

    # After initialization
    camera = Camera()
    assert camera.name is None
    camera.name = 12
    assert camera.name == "12"
    camera.name = "12"
    assert camera.name == "12"


def test_camera_matrix():
    """Test camera matrix converter and validator."""
    matrix_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_array = np.array(matrix_list)

    # During initialization
    camera = Camera(matrix=matrix_list)
    np.testing.assert_array_equal(camera.matrix, matrix_array)
    with pytest.raises(ValueError):
        camera = Camera(matrix=[[1, 2], [3, 4]])

    # Test matrix converter
    camera = Camera()
    camera.matrix = matrix_list
    np.testing.assert_array_equal(camera.matrix, matrix_array)
    matrix_array = np.array(matrix_list)
    camera.matrix = matrix_array
    np.testing.assert_array_equal(camera.matrix, matrix_array)
    with pytest.raises(ValueError):
        camera.matrix = [[1, 2], [3, 4]]


def test_camera_distortions():
    """Test camera distortion converter and validator."""
    distortions_unraveled = [[1], [2], [3], [4], [5]]
    distortions_raveled = np.array(distortions_unraveled).ravel()

    # During initialization
    camera = Camera(dist=distortions_unraveled)
    np.testing.assert_array_equal(camera.dist, distortions_raveled)
    with pytest.raises(ValueError):
        camera = Camera(dist=distortions_raveled[:3])

    # Test distortion converter
    camera = Camera()
    camera.dist = distortions_unraveled
    np.testing.assert_array_equal(camera.dist, distortions_raveled)
    with pytest.raises(ValueError):
        camera.dist = distortions_raveled[:3]


def test_camera_size():
    """Test camera size converter and validator."""
    size = (100, 200)

    # During initialization
    camera = Camera(size=size)
    assert camera.size == size
    with pytest.raises(ValueError):
        camera = Camera(size=(100, 200, 300))

    # Test size converter
    camera = Camera()
    camera.size = size
    assert camera.size == size
    with pytest.raises(ValueError):
        camera.size = (100, 200, 300)


def construct_extrinsic_matrix(rvec, tvec):
    """Construct extrinsic matrix from rotation and translation vectors.

    Args:
        rvec: Rotation vector in unnormalized axis angle representation of size (3,) and
            type float64.
        tvec: Translation vector of size (3,) and type float64.

    Returns:
        Extrinsic matrix of camera of size (4, 4) and type float64.
    """
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = cv2.Rodrigues(np.array(rvec))[0]
    extrinsic_matrix[:3, 3] = tvec

    return extrinsic_matrix


def test_camera_rvec():
    """Test camera rotation vector converter and validator."""
    rvec = [1, 2, 3]

    # During initialization
    camera = Camera(rvec=rvec)
    np.testing.assert_array_equal(camera.rvec, rvec)
    extrinsic_matrix = construct_extrinsic_matrix(camera.rvec, camera.tvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)
    np.testing.assert_array_equal(
        camera.extrinsic_matrix[:3, :3], cv2.Rodrigues(camera.rvec)[0]
    )
    with pytest.raises(ValueError):
        camera = Camera(rvec=[1, 2])
    np.testing.assert_array_equal(camera.rvec, rvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)

    # Test rvec validator
    camera = Camera()
    camera.rvec = rvec
    np.testing.assert_array_equal(camera.rvec, rvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)
    np.testing.assert_array_equal(
        camera.extrinsic_matrix[:3, :3], cv2.Rodrigues(camera.rvec)[0]
    )
    with pytest.raises(ValueError):
        camera.rvec = [1, 2]
    np.testing.assert_array_equal(camera.rvec, rvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)


def test_camera_tvec():
    """Test camera translation vector converter and validator."""
    tvec = [1, 2, 3]

    # During initialization
    camera = Camera(tvec=tvec)
    np.testing.assert_array_equal(camera.tvec, tvec)
    extrinsic_matrix = construct_extrinsic_matrix(camera.rvec, camera.tvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)
    np.testing.assert_array_equal(camera.extrinsic_matrix[:3, 3], camera.tvec)
    with pytest.raises(ValueError):
        camera = Camera(tvec=[1, 2])
    np.testing.assert_array_equal(camera.tvec, tvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)

    # Test tvec validator
    camera = Camera()
    camera.tvec = tvec
    np.testing.assert_array_equal(camera.tvec, tvec)
    extrinsic_matrix = construct_extrinsic_matrix(camera.rvec, camera.tvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)
    np.testing.assert_array_equal(camera.extrinsic_matrix[:3, 3], camera.tvec)
    with pytest.raises(ValueError):
        camera.tvec = [1, 2]
    np.testing.assert_array_equal(camera.tvec, tvec)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)


def test_camera_extrinsic_matrix():
    """Test camera extrinsic matrix method."""
    # During initialization

    # ... with rvec and tvec
    camera = Camera(
        rvec=[1, 2, 3],
        tvec=[1, 2, 3],
    )
    extrinsic_matrix = camera.extrinsic_matrix
    np.testing.assert_array_equal(
        extrinsic_matrix[:3, :3], cv2.Rodrigues(camera.rvec)[0]
    )
    np.testing.assert_array_equal(extrinsic_matrix[:3, 3], camera.tvec)

    # ... without rvec and tvec
    camera = Camera()
    extrinsic_matrix = camera.extrinsic_matrix
    np.testing.assert_array_equal(extrinsic_matrix, np.eye(4))

    # After initialization

    # Setting extrinsic matrix updates rvec and tvec
    extrinsic_matrix = np.random.rand(4, 4)
    camera.extrinsic_matrix = extrinsic_matrix
    rvec = cv2.Rodrigues(camera.extrinsic_matrix[:3, :3])[0]
    tvec = camera.extrinsic_matrix[:3, 3]
    np.testing.assert_array_equal(camera.rvec, rvec.ravel())
    np.testing.assert_array_equal(camera.tvec, tvec)

    # Invalid extrinsic matrix doesn't update rvec and tvec or extrinsic matrix
    with pytest.raises(ValueError):
        camera.extrinsic_matrix = np.eye(3)
    np.testing.assert_array_equal(camera.extrinsic_matrix, extrinsic_matrix)
    np.testing.assert_array_equal(camera.rvec, rvec.ravel())
    np.testing.assert_array_equal(camera.tvec, tvec)


def test_camera_from_dict_to_dict():
    """Test camera from_dict method."""

    # Define camera dictionary
    name = "back"
    size = [1280, 1024]
    matrix = [
        [762.513822135494, 0.0, 639.5],
        [0.0, 762.513822135494, 511.5],
        [0.0, 0.0, 1.0],
    ]
    distortions = [-0.2868458380166852, 0.0, 0.0, 0.0, 0.0]
    rotation = [0.3571857188780474, 0.8879473292757126, 1.6832001677006176]
    translation = [-555.4577842902744, -294.43494957092884, -190.82196458369515]
    camera_dict = {
        "name": name,
        "size": size,
        "matrix": matrix,
        "distortions": distortions,
        "rotation": rotation,
        "translation": translation,
    }

    # Test camera from_dict
    camera = Camera.from_dict(camera_dict)
    assert camera.name == "back"
    assert camera.size == tuple(size)
    np.testing.assert_array_almost_equal(camera.matrix, np.array(matrix))
    np.testing.assert_array_almost_equal(camera.dist, np.array(distortions))
    np.testing.assert_array_almost_equal(camera.rvec, np.array(rotation))
    np.testing.assert_array_almost_equal(camera.tvec, np.array(translation))

    # Test camera to_dict
    assert camera.to_dict() == camera_dict


def test_camera_undistort_points():
    """Test camera undistort points method."""
    camera = Camera(
        matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dist=[[0], [0], [0], [0], [0]],
    )

    # Test with no distortion
    points = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    undistorted_points = camera.undistort_points(points)
    np.testing.assert_array_equal(points, undistorted_points)

    # Test with distortion
    camera.dist = [[1], [0], [0], [0], [0]]
    undistorted_points = camera.undistort_points(points)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(points, undistorted_points)


def test_camera_project():
    """Test camera project method."""
    camera = Camera(
        matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dist=[[0], [0], [0], [0], [0]],
    )

    # Test with incorrect input shape
    with pytest.raises(ValueError):
        camera.project(np.random.rand(10, 1, 2))

    points_dtype = np.int32
    points = np.random.rand(10, 3).astype(points_dtype)
    projected_points = camera.project(points)
    assert points.dtype == points_dtype
    assert projected_points.dtype == points_dtype
    assert projected_points.shape == (*points.shape[:-1], 2)

    points = np.random.rand(10, 1, 3)
    projected_points = camera.project(points)
    assert projected_points.shape == (*points.shape[:-1], 2)
    assert projected_points.dtype == points.dtype


def test_camera_get_video():
    """Test camera get video method."""
    camera = Camera()
    camera_group = CameraGroup(cameras=[camera])

    # Test with no video
    session = RecordingSession(camera_group=camera_group)
    video = camera.get_video(session)
    assert video is None

    # Test with video
    video = Video(filename="not/a/file.mp4")
    session.add_video(video=video, camera=camera)
    assert camera.get_video(session) is video

    # Remove video
    session.remove_video(video)
    assert camera.get_video(session) is None


def test_recording_session_videos():
    """Test `RecordingSession.videos` property."""
    camera_1 = Camera()
    camera_2 = Camera()
    camera_group = CameraGroup(cameras=[camera_1, camera_2])

    # Test with no videos
    session = RecordingSession(camera_group=camera_group)
    assert session.videos == []

    # Test with a single videos
    video = Video(filename="not/a/file.mp4")
    session.add_video(video=video, camera=camera_2)
    assert session.videos == [video]

    # Test with multiple videos
    video_2 = Video(filename="not/a/file2.mp4")
    session.add_video(video=video_2, camera=camera_1)
    assert session.videos == [video, video_2]


def test_recording_session_get_camera():
    """Test `RecordingSession.get_camera` method."""
    camera_1 = Camera(name="camera_1")
    camera_2 = Camera(name="camera_2")
    camera_group = CameraGroup(cameras=[camera_1, camera_2])

    # Test with not a `Video` object
    session = RecordingSession(camera_group=camera_group)
    assert session.get_camera("not_a_video") is None

    # Test with a `Video` object, but no videos
    video_1 = Video(filename="not/a/file.mp4")
    assert session.get_camera(video_1) is None

    # Test with a `Video` object and a video
    session.add_video(video=video_1, camera=camera_1)
    assert session.get_camera(video_1) is camera_1

    # Test with a `Video` object and multiple videos
    video_2 = Video(filename="not/a/file2.mp4")
    session.add_video(video=video_2, camera=camera_2)
    assert session.get_camera(video_1) is camera_1
    assert session.get_camera(video_2) is camera_2


def test_recording_session_add_video():
    """Test `RecordingSession.add_video` method."""
    camera_group = CameraGroup()
    session = RecordingSession(camera_group=camera_group)
    camera_1 = Camera()
    camera_2 = Camera()
    video_1 = Video(filename="not/a/file.mp4")
    video_2 = Video(filename="not/a/file2.mp4")

    # Test with `Camera` object not in `camera_group`
    with pytest.raises(ValueError):
        session.add_video(video=video_1, camera=camera_1)
    assert session._video_by_camera == {}
    assert session._camera_by_video == {}

    # Test with not isinstance(`video`, `Video`)
    with pytest.raises(ValueError):
        session.add_video(video="not_a_video", camera=camera_1)
    assert session._video_by_camera == {}
    assert session._camera_by_video == {}

    # Test with `Camera` object in `camera_group`
    camera_group.cameras.append(camera_1)
    session.add_video(video=video_1, camera=camera_1)
    assert session._video_by_camera == {camera_1: video_1}
    assert session._camera_by_video == {video_1: camera_1}

    # Test with multiple videos
    camera_group.cameras.append(camera_2)
    session.add_video(video=video_2, camera=camera_2)
    assert session._video_by_camera == {camera_1: video_1, camera_2: video_2}
    assert session._camera_by_video == {video_1: camera_1, video_2: camera_2}


# TODO: Remove when implement triangulation without aniposelib
def test_camera_aliases():
    """Test camera aliases for attributes."""
    camera = Camera(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dist=[[1], [2], [3], [4], [5]],
        size=(100, 200),
        rvec=[1, 2, 3],
        tvec=[1, 2, 3],
        name="camera",
    )

    # Test __getattr__ aliases
    assert camera.get_name() == camera.name
    np.testing.assert_array_equal(
        camera.get_extrinsic_matrix(), camera.extrinsic_matrix
    )
