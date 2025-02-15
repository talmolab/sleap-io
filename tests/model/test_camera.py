"""Tests for methods in the sleap_io.model.instance file."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import toml

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
    metadata = {"extra_key": "extra_value", 444: 555}
    camera_dict = {
        "name": name,
        "size": size,
        "matrix": matrix,
        "distortions": distortions,
        "rotation": rotation,
        "translation": translation,
    }
    camera_dict.update(metadata)

    # Test camera from_dict
    camera = Camera.from_dict(camera_dict)
    assert camera.name == "back"
    assert camera.size == tuple(size)
    np.testing.assert_array_almost_equal(camera.matrix, np.array(matrix))
    np.testing.assert_array_almost_equal(camera.dist, np.array(distortions))
    np.testing.assert_array_almost_equal(camera.rvec, np.array(rotation))
    np.testing.assert_array_almost_equal(camera.tvec, np.array(translation))
    assert camera.metadata == metadata

    # Test camera to_dict
    assert camera.to_dict() == camera_dict

    # Test when Camera has None for optional attributes

    camera = Camera(rvec=rotation, tvec=translation)
    assert camera.name is None
    assert camera.size is None

    # Test to_dict
    camera_dict = camera.to_dict()
    assert camera_dict["name"] == ""
    assert camera_dict["size"] == ""
    assert camera_dict["matrix"] == camera.matrix.tolist()
    assert camera_dict["distortions"] == camera.dist.tolist()
    assert camera_dict["rotation"] == camera.rvec.tolist()
    assert camera_dict["translation"] == camera.tvec.tolist()

    # Test from_dict
    camera_0 = Camera.from_dict(camera_dict)
    assert camera_0.name is None
    assert camera_0.size is None


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


def test_camera_group_from_dict_to_dict():
    """Test camera group from_dict and to_dict methods."""

    # Define template camera dictionary
    size = [1280, 1024]
    matrix = np.eye(3).tolist()
    distortions = np.zeros(5).tolist()
    rotation = np.zeros(3).tolist()
    translation = np.zeros(3).tolist()
    camera_dict_template = {
        "size": size,
        "matrix": matrix,
        "distortions": distortions,
        "rotation": rotation,
        "translation": translation,
    }

    camera_group_dict = {}
    n_cameras = 3
    for i in range(n_cameras):
        camera_dict = camera_dict_template.copy()
        camera_dict["name"] = f"camera{i}"
        camera_group_dict[f"cam_{i}"] = camera_dict
    metadata = {"extra_key": "extra_value", 444: 555}
    camera_group_dict["metadata"] = metadata

    camera_group_0 = CameraGroup.from_dict(camera_group_dict)
    assert camera_group_0.metadata == metadata
    camera_group_dict_0: dict = camera_group_0.to_dict()
    assert camera_group_dict == camera_group_dict_0
    assert len(camera_group_0.cameras) == 3
    for i in range(n_cameras):
        assert camera_group_0.cameras[i].name == f"camera{i}"
        assert camera_group_0.cameras[i].size == tuple(size)
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].matrix, np.array(matrix)
        )
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].dist, np.array(distortions)
        )
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].rvec, np.array(rotation)
        )
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].tvec, np.array(translation)
        )


def test_camera_group_load(calibration_toml_path: str):
    """Test camera group load method."""

    camera_group = CameraGroup.load(calibration_toml_path)
    assert len(camera_group.cameras) == 8

    for camera, name in zip(
        camera_group.cameras,
        ["back", "backL", "mid", "midL", "side", "sideL", "top", "topL"],
    ):
        assert camera.name == name
        assert camera.size == (1280, 1024)


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
    assert instance_group._instance_by_camera == instance_by_camera
    assert instance_group._score == score
    assert instance_group._points.dtype == np.float64
    assert np.array_equal(instance_group._points, points.astype(np.float64))
    assert instance_group.metadata == metadata


def test_instance_group_to_dict_from_dict(
    instance_group_345: InstanceGroup, camera_group_345: CameraGroup
):
    """Test InstanceGroup to_dict and from_dict methods.

    Args:
        instance_group_345: Instance group with an `Instance` at each camera view.
        camera_group_345: Camera group with 3-4-5 triangle configuration.
    """
    instance_group = instance_group_345

    # Create necessary helper objects.

    def new_labeled_frame(inst: Instance | None = None):
        """Create a new labeled frame with or without the specified instance.

        Args:
            inst: Instance to include in the labeled frame. If None, a new instance is
                created instead.
        """
        video = Video(filename="test")
        if inst is None:
            inst = Instance([[8, 9], [10, 11]], skeleton=Skeleton(["A", "B"]))
        return LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[
                inst,
                Instance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
            ],
        )

    # Create labeled frames, with some irrelevant frames to make mapping more complex
    labeled_frames = []
    for inst in instance_group._instance_by_camera.values():
        labeled_frames.append(new_labeled_frame(inst))
        labeled_frames.append(new_labeled_frame())

    # Create our instance_to_lf_and_inst_idx dictionary.
    instance_to_lf_and_inst_idx = {
        inst: (inst_idx * 2, 0)  # inst_idx * 2 because we have irrelevant frames
        for inst_idx, inst in enumerate(instance_group._instance_by_camera.values())
    }

    # Test to_dict.

    instance_group_dict = instance_group.to_dict(
        instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx,
        camera_group=camera_group_345,
    )
    assert instance_group_dict["score"] == str(instance_group._score)
    assert np.array_equal(instance_group_dict["points"], instance_group._points)

    # Test from_dict.

    instance_group_0 = InstanceGroup.from_dict(
        instance_group_dict,
        labeled_frames=labeled_frames,
        camera_group=camera_group_345,
    )
    assert instance_group_0._score == instance_group._score
    assert np.array_equal(instance_group_0._points, instance_group._points)
    assert instance_group_0.metadata == instance_group.metadata
    assert len(instance_group_0._instance_by_camera) == len(
        instance_group._instance_by_camera
    )

    # Check the instances and cameras are the same.
    for (cam, inst), (cam_0, inst_0) in zip(
        instance_group._instance_by_camera.items(),
        instance_group_0._instance_by_camera.items(),
    ):
        assert inst == inst_0
        assert cam == cam_0

    # Check that the dictionary was not mutated.
    assert instance_group_dict.get("points", None) is not None
    assert instance_group_dict.get("score", None) is not None
    assert instance_group_dict.get("camcorder_to_lf_and_inst_idx_map", None) is not None


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
    assert frame_group._instance_groups == []
    assert frame_group._labeled_frame_by_camera == {}
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
    assert frame_group._instance_groups == instance_groups
    assert frame_group._labeled_frame_by_camera == labeled_frame_by_camera
    assert frame_group.metadata == metadata


def test_frame_group_to_dict_from_dict(
    frame_group_345: FrameGroup, camera_group_345: CameraGroup
):
    """Test FrameGroup to_dict and from_dict methods.

    Args:
        frame_group_345: Frame group with an `InstanceGroup` at each camera view.
        camera_group_345: Camera group with 3-4-5 triangle configuration
    """
    frame_group = frame_group_345
    camera_group = camera_group_345

    # Create necessary helper objects.

    # Create labeled frames, with some irrelevant frames to make mapping more complex.
    labeled_frames = []
    for lf in frame_group.labeled_frames:
        labeled_frames.append(lf)
        labeled_frames.append(
            LabeledFrame(video=Video(filename="test"), frame_idx=lf.frame_idx)
        )
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labeled_frames)}

    # Test to_dict.

    frame_group_dict = frame_group.to_dict(
        labeled_frame_to_idx=labeled_frame_to_idx, camera_group=camera_group
    )
    assert frame_group_dict["frame_idx"] == str(frame_group.frame_idx)

    # Test from_dict.

    frame_group_0 = FrameGroup.from_dict(
        frame_group_dict=frame_group_dict,
        labeled_frames=labeled_frames,
        camera_group=camera_group,
    )
    assert frame_group_0.frame_idx == frame_group.frame_idx
    assert len(frame_group_0._instance_groups) == len(frame_group._instance_groups)
    assert len(frame_group_0._labeled_frame_by_camera) == len(
        frame_group._labeled_frame_by_camera
    )
    assert frame_group_0.metadata == frame_group.metadata

    # Check the cameras and labeled frames are the same.
    for (cam, lf), (cam_0, lf_0) in zip(
        frame_group._labeled_frame_by_camera.items(),
        frame_group_0._labeled_frame_by_camera.items(),
    ):
        assert cam == cam_0
        assert lf == lf_0

    # Check the instance groups are the same.
    for instance_group, instance_group_0 in zip(
        frame_group._instance_groups, frame_group_0._instance_groups
    ):
        for (cam, inst), (cam_0, inst_0) in zip(
            instance_group._instance_by_camera.items(),
            instance_group_0._instance_by_camera.items(),
        ):
            assert inst == inst_0
            assert cam == cam_0


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
    assert session._frame_group_by_frame_idx == frame_group_by_frame_idx
    assert session.metadata == metadata


def test_recording_session_load(calibration_toml_path: str):
    """Test recording session load method.

    Args:
        calibration_toml_path: Path to calibration toml file.
    """

    session = RecordingSession.load(calibration_toml_path)
    assert len(session.camera_group.cameras) == 8
    assert len(session._video_by_camera) == 0
    assert len(session._camera_by_video) == 0
    assert len(session._frame_group_by_frame_idx) == 0
    assert session.metadata == {}

    cameras_dict: dict = toml.load(calibration_toml_path)
    cameras_dict_metadata = cameras_dict.pop("metadata")
    key_to_attr_map = {
        "size": "size",
        "matrix": "matrix",
        "distortions": "dist",
        "rotation": "rvec",
        "translation": "tvec",
    }
    assert len(session.camera_group.cameras) == len(cameras_dict)
    for camera, camera_dict in zip(session.camera_group.cameras, cameras_dict.values()):
        for key, value in camera_dict.items():
            attr = key_to_attr_map.get(key, key)
            attr_value = getattr(camera, attr)

            # Always serialized to a list, so convert to correct type before comparison.
            attr_type = type(attr_value)
            if attr_type == np.ndarray:
                np.testing.assert_array_almost_equal(attr_value, np.array(value))
            else:
                assert attr_value == attr_type(value)


def test_recording_session_from_calibration_dict(calibration_toml_path: str):
    """Test recording session from_calibration_dict method.

    Args:
        calibration_toml_path: Path to calibration toml file.
    """
    cameras_dict: dict = toml.load(calibration_toml_path)
    cameras_dict_metadata = cameras_dict.pop("metadata")
    session = RecordingSession.from_calibration_dict(cameras_dict)
    assert len(session.camera_group.cameras) == len(cameras_dict)
    assert len(session._video_by_camera) == 0
    assert len(session._camera_by_video) == 0
    assert len(session._frame_group_by_frame_idx) == 0
    assert session.metadata == {}

    key_to_attr_map = {
        "size": "size",
        "matrix": "matrix",
        "distortions": "dist",
        "rotation": "rvec",
        "translation": "tvec",
    }
    assert len(session.camera_group.cameras) == len(cameras_dict)
    for camera, camera_dict in zip(session.camera_group.cameras, cameras_dict.values()):
        for key, value in camera_dict.items():
            attr = key_to_attr_map.get(key, key)
            attr_value = getattr(camera, attr)

            # Always serialized to a list, so convert to correct type before comparison.
            attr_type = type(attr_value)
            if attr_type == np.ndarray:
                np.testing.assert_array_almost_equal(attr_value, np.array(value))
            else:
                assert attr_value == attr_type(value)


def assert_cameras_equal(camera: Camera, camera_0: Camera):
    """Compare two cameras.

    Args:
        camera: First camera.
        camera_0: Second camera.

    Raises:
        AssertionError: If the cameras are not equal.
    """
    camera_state = camera.__getstate__()
    camera_state_0 = camera_0.__getstate__()
    for key, value in camera_state.items():
        if value is None:
            assert camera_state_0[key] is None
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(camera_state_0[key], value)
        else:
            assert camera_state_0[key] == value


def test_recording_session_to_dict_from_dict(recording_session_345: RecordingSession):
    """Test recording session to_dict and from_dict methods.

    Args:
        frame_group_345: Frame group with an `InstanceGroup` at each camera view.
    """
    session = recording_session_345
    frame_group = session._frame_group_by_frame_idx[0]

    # Create necessary helper objects.

    # Create labeled frames, with some irrelevant frames to make mapping more complex.
    labeled_frames = []
    for lf in frame_group.labeled_frames:
        labeled_frames.append(lf)
        labeled_frames.append(
            LabeledFrame(video=Video(filename="test"), frame_idx=lf.frame_idx)
        )
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labeled_frames)}

    # Create videos list and index mapping.
    videos = []
    for video in session.videos:
        videos.append(video)
        videos.append(Video(filename="test"))
    video_to_idx = {video: idx for idx, video in enumerate(videos)}

    # Test to_dict.

    session_dict = session.to_dict(
        labeled_frame_to_idx=labeled_frame_to_idx, video_to_idx=video_to_idx
    )
    assert len(session_dict["frame_group_dicts"]) == len(
        session._frame_group_by_frame_idx
    )

    # Test from_dict.

    session_0: RecordingSession = RecordingSession.from_dict(
        session_dict=session_dict,
        labeled_frames=labeled_frames,
        videos=videos,
    )
    assert len(session_0.camera_group.cameras) == len(session.camera_group.cameras)
    assert len(session_0._video_by_camera) == len(session._video_by_camera)
    assert len(session_0._camera_by_video) == len(session._camera_by_video)
    assert len(session_0._frame_group_by_frame_idx) == len(
        session._frame_group_by_frame_idx
    )
    assert session_0.metadata == session.metadata

    # Check the cameras and videos are the same.
    for (video, camera), (video_0, camera_0) in zip(
        session._camera_by_video.items(), session_0._camera_by_video.items()
    ):
        assert video == video_0
        assert session._video_by_camera[camera] == video
        assert session_0._video_by_camera[camera_0] == video_0
        assert_cameras_equal(camera, camera_0)

    # Check the frame groups are the same.
    for (frame_idx, frame_group), (frame_idx_0, frame_group_0) in zip(
        session._frame_group_by_frame_idx.items(),
        session_0._frame_group_by_frame_idx.items(),
    ):
        assert frame_idx == frame_idx_0
        assert frame_group.frame_idx == frame_idx
        assert frame_group.frame_idx == frame_group_0.frame_idx
        assert len(frame_group._instance_groups) == len(frame_group_0._instance_groups)
        assert len(frame_group._labeled_frame_by_camera) == len(
            frame_group_0._labeled_frame_by_camera
        )
        assert frame_group.metadata == frame_group_0.metadata

        # Check the cameras and labeled frames are the same.
        for (camera, lf), (camera_0, lf_0) in zip(
            frame_group._labeled_frame_by_camera.items(),
            frame_group_0._labeled_frame_by_camera.items(),
        ):
            assert lf == lf_0
            assert_cameras_equal(camera, camera_0)

        # Check the instance groups are the same.
        for instance_group, instance_group_0 in zip(
            frame_group._instance_groups, frame_group_0._instance_groups
        ):
            for (cam, inst), (cam_0, inst_0) in zip(
                instance_group._instance_by_camera.items(),
                instance_group_0._instance_by_camera.items(),
            ):
                assert inst == inst_0
                assert_cameras_equal(cam, cam_0)
