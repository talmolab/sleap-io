"""Tests for methods in the sleap_io.model.instance file."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sleap_io.model.camera import (
    Camera,
    CameraGroup,
    FrameGroup,
    InstanceGroup,
    RecordingSession,
)
from sleap_io.model.instance import Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.skeleton import Skeleton
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


def test_camera_group_cameras():
    """Test camera group cameras method."""
    camera1 = Camera(name="camera1")
    camera2 = Camera(name="camera2")
    camera_group = CameraGroup(cameras=[camera1, camera2])

    assert camera_group.cameras == [camera1, camera2]

    camera_group = CameraGroup()
    assert camera_group.cameras == []


def test_camera_group_triangulation(camera_group_345: CameraGroup):
    """Test camera group triangulation using 3-4-5 triangle on xy-plane."""

    camera_group = camera_group_345

    # Define special 3-4-5 triangle
    b = 4
    c = 5

    # Test with incorrect input shape along camera axis
    points = np.random.rand(1, 1, 2)
    with pytest.raises(ValueError):
        camera_group.triangulate(points=points)

    # Test with incorrect input shape along point-dimension axis
    points = np.random.rand(len(camera_group.cameras), 1, 3)
    with pytest.raises(ValueError):
        camera_group.triangulate(points=points)

    # Triangulate point from two camera views with shape (M, N=1, 2)
    points_dtype = np.int8
    points = np.array([[[c, 0]], [[c, 0]]]).astype(points_dtype)
    points_3d = camera_group.triangulate(points=points)
    assert points_3d.shape == (1, 3)  # == (*points.shape[1:-1], 3)
    assert points.dtype == points_dtype
    assert points_3d.dtype == points_dtype
    np.testing.assert_array_almost_equal(
        points_3d[:, :-1], np.array([[b, 0]]), decimal=5
    )  # z-coordinate is ambiguous since we only define 2D points on x-y plane

    # Triangulate points with shape (M, 2)
    points_dtype = np.float32
    points = points.reshape(-1, 2).astype(points_dtype)
    points_3d = camera_group.triangulate(points=points)
    assert points_3d.shape == (3,)  # == (*points.shape[1:-1], 3)
    assert points.dtype == points_dtype
    assert points_3d.dtype == points_dtype
    np.testing.assert_array_almost_equal(
        points_3d[:-1], np.array([b, 0]), decimal=5
    )  # z-coordinate is ambiguous since we only define 2D points on x-y plane

    # Triangulate points with shape (M, L=1, N=1, 2)
    points = points.reshape(points.shape[0], 1, 1, 2)
    points_3d = camera_group.triangulate(points=points)
    assert points_3d.shape == (1, 1, 3)  # == (*points.shape[1:-1], 3)
    assert points_3d.dtype == points.dtype
    np.testing.assert_array_almost_equal(
        points_3d[:, :, :-1], np.array([[[b, 0]]]), decimal=5
    )  # z-coordinate is ambiguous since we only define 2D points on x-y plane

    # Triangulate with triangulate_func that returns incorrect shape
    def triangulation_func(points, camera_group):
        return np.random.rand(5, 5)

    with pytest.raises(ValueError):
        camera_group.triangulate(points=points, triangulation_func=triangulation_func)


def test_camera_group_project(camera_group_345: CameraGroup):
    """Test camera group project method using 3-4-5 triangle on xy-plane."""
    camera_group = camera_group_345

    # Define special 3-4-5 triangle
    b = 4
    c = 5

    # Test with incorrect input shape along point-dimension axis
    points = np.random.rand(1)
    with pytest.raises(ValueError):
        camera_group.project(points=points)

    # Define 3D point
    n_points = 1
    points_3d = np.array([[b, 0, 1]])
    assert points_3d.shape == (n_points, 3)

    # Project points from world to camera frame
    points_3d_dtype = np.int8
    points_3d = points_3d.astype(points_3d_dtype)
    points = camera_group.project(points_3d)
    assert points.shape == (len(camera_group.cameras), n_points, 2)
    assert points_3d.dtype == points_3d_dtype
    assert points.dtype == points_3d.dtype
    np.testing.assert_array_almost_equal(
        points, np.array([[[c, 0]], [[c, 0]]]), decimal=5
    )

    # Project with arbitrary points shape (1, 1, N, 3)
    points_3d_dtype = np.float32
    points_3d = points_3d.reshape(1, 1, n_points, 3).astype(points_3d_dtype)
    points = camera_group.project(points_3d)
    assert points.shape == (len(camera_group.cameras), 1, 1, n_points, 2)
    assert points_3d.dtype == points_3d_dtype
    assert points.dtype == points_3d.dtype
    np.testing.assert_array_almost_equal(
        points, np.array([[[[[c, 0]]]], [[[[c, 0]]]]]), decimal=5
    )


def test_instance_group_init(
    camera_group_345: CameraGroup,
):
    """Test instance group initialization.

    Args:
        camera_group_345: Camera group with 3-4-5 triangle configuration.
    """
    camera_group = camera_group_345

    # Test with defaults
    instance_group = InstanceGroup()
    assert instance_group._instance_by_camera == {}
    assert instance_group._score is None
    assert instance_group._points is None
    assert instance_group.metadata == {}

    # Test with non-defaults
    skeleton = Skeleton(["A", "B"])
    instance_by_camera = {
        cam: Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
        for cam in camera_group.cameras
    }
    score = 0.5
    points = np.random.rand(10, 3).astype(np.float32)
    assert points.dtype == np.float32
    metadata = {"observation": 72317}
    instance_group = InstanceGroup(
        instance_by_camera=instance_by_camera,
        score=score,
        points=points,
        metadata=metadata,
    )
    assert instance_group.instances == list(instance_by_camera.values())
    assert instance_group.cameras == list(instance_by_camera.keys())
    assert (
        instance_group.get_instance(instance_group.cameras[-1])
        == instance_group.instances[-1]
    )
    assert instance_group._score == score
    assert instance_group._points.dtype == np.float64
    assert np.array_equal(instance_group._points, points.astype(np.float64))
    assert instance_group.metadata == metadata


def test_frame_group_init(camera_group_345: CameraGroup):
    """Test frame group initialization.

    Args:
        camera_group_345: Camera group with 3-4-5 triangle configuration.
    """
    # Need frame index
    with pytest.raises(TypeError):
        frame_group = FrameGroup()

    # Test with frame index and defaults
    frame_idx = 0
    frame_group = FrameGroup(frame_idx=frame_idx)
    assert frame_group.frame_idx == 0
    assert frame_group.instance_groups == []
    assert frame_group.cameras == []
    assert frame_group.labeled_frames == []
    assert frame_group.metadata == {}

    # Test with non-defaults
    instance_groups = [InstanceGroup(), InstanceGroup()]
    labeled_frame_by_camera = {
        cam: LabeledFrame(
            frame_idx=frame_idx,
            video=Video(filename="test"),
        )
        for cam in camera_group_345.cameras
    }
    metadata = {"observation": 72317}
    frame_group = FrameGroup(
        frame_idx=frame_idx,
        instance_groups=instance_groups,
        labeled_frame_by_camera=labeled_frame_by_camera,
        metadata=metadata,
    )
    assert frame_group.frame_idx == frame_idx
    assert frame_group.instance_groups == instance_groups
    assert frame_group.cameras == list(labeled_frame_by_camera.keys())
    assert frame_group.labeled_frames == list(labeled_frame_by_camera.values())
    assert (
        frame_group.get_frame(frame_group.cameras[-1]) == frame_group.labeled_frames[-1]
    )
    assert frame_group.metadata == metadata


def test_recording_session_init(camera_group_345: CameraGroup):
    """Test recording session initialization.

    Args:
        camera_group_345: Camera group with 3-4-5 triangle configuration.
    """
    # Test with defaults
    session = RecordingSession()
    assert session._video_by_camera == {}
    assert session._camera_by_video == {}
    assert session._frame_group_by_frame_idx == {}
    assert session.metadata == {}

    # Test with non-defaults
    camera_group = camera_group_345
    video_by_camera = {cam: Video(filename="test") for cam in camera_group.cameras}
    camera_by_video = {video: cam for cam, video in video_by_camera.items()}
    frame_group_by_frame_idx = {
        frame_idx: FrameGroup(frame_idx=frame_idx) for frame_idx in range(10)
    }
    metadata = {"observation": 72317}
    session = RecordingSession(
        camera_group=camera_group,
        video_by_camera=video_by_camera,
        camera_by_video=camera_by_video,
        frame_group_by_frame_idx=frame_group_by_frame_idx,
        metadata=metadata,
    )
    assert session.camera_group == camera_group
    assert session._video_by_camera == video_by_camera
    assert session._camera_by_video == camera_by_video
    assert session.frame_groups == frame_group_by_frame_idx
    assert session.metadata == metadata
