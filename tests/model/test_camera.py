"""Tests for methods in the sleap_io.model.instance file."""

from __future__ import annotations

import numpy as np
import pytest

from sleap_io.model.camera import (
    Camera,
    CameraGroup,
    FrameGroup,
    InstanceGroup,
    RecordingSession,
    rodrigues_transformation,
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
    extrinsic_matrix[:3, :3] = rodrigues_transformation(np.array(rvec))[0]
    extrinsic_matrix[:3, 3] = tvec

    return extrinsic_matrix


def test_camera_rvec():
    """Test camera rotation vector converter and validator."""
    rvec = [1, 2, 3]

    # During initialization
    camera = Camera(rvec=rvec)
    np.testing.assert_array_equal(camera.rvec, rvec)
    extrinsic_matrix = construct_extrinsic_matrix(camera.rvec, camera.tvec)
    np.testing.assert_allclose(camera.extrinsic_matrix, extrinsic_matrix, atol=1e-6)
    np.testing.assert_allclose(
        camera.extrinsic_matrix[:3, :3],
        rodrigues_transformation(camera.rvec)[0],
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        camera = Camera(rvec=[1, 2])
    np.testing.assert_array_equal(camera.rvec, rvec)
    np.testing.assert_allclose(camera.extrinsic_matrix, extrinsic_matrix, atol=1e-6)

    # Test rvec validator
    camera = Camera()
    camera.rvec = rvec
    np.testing.assert_array_equal(camera.rvec, rvec)
    np.testing.assert_allclose(camera.extrinsic_matrix, extrinsic_matrix, atol=1e-6)
    np.testing.assert_allclose(
        camera.extrinsic_matrix[:3, :3],
        rodrigues_transformation(camera.rvec)[0],
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        camera.rvec = [1, 2]
    np.testing.assert_array_equal(camera.rvec, rvec)
    np.testing.assert_allclose(camera.extrinsic_matrix, extrinsic_matrix, atol=1e-6)


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
    np.testing.assert_allclose(
        extrinsic_matrix[:3, :3], rodrigues_transformation(camera.rvec)[0], atol=1e-6
    )
    np.testing.assert_array_equal(extrinsic_matrix[:3, 3], camera.tvec)

    # ... without rvec and tvec
    camera = Camera()
    extrinsic_matrix = camera.extrinsic_matrix
    np.testing.assert_array_equal(extrinsic_matrix, np.eye(4))

    # After initialization

    # Setting extrinsic matrix updates rvec and tvec
    # Use a valid rotation vector and translation vector
    test_rvec = np.array([0.5, 0.5, 0.5])
    test_tvec = np.array([1.0, 2.0, 3.0])

    # Create rotation matrix from test rotation vector
    valid_rotation = rodrigues_transformation(test_rvec)[0]

    # Create extrinsic matrix from rotation matrix and translation vector
    valid_extrinsic = np.eye(4)
    valid_extrinsic[:3, :3] = valid_rotation
    valid_extrinsic[:3, 3] = test_tvec

    camera.extrinsic_matrix = valid_extrinsic

    # Check that extrinsic matrix is preserved
    np.testing.assert_allclose(camera.extrinsic_matrix, valid_extrinsic, atol=1e-6)

    # Check that tvec is preserved exactly
    np.testing.assert_array_equal(camera.tvec, test_tvec)

    # Check that rotation matrix part is preserved (allows different rotation vectors)
    rotation_matrix_from_rvec = rodrigues_transformation(camera.rvec)[0]
    np.testing.assert_allclose(rotation_matrix_from_rvec, valid_rotation, atol=1e-6)

    # Invalid extrinsic matrix doesn't update rvec and tvec or extrinsic matrix
    with pytest.raises(ValueError):
        camera.extrinsic_matrix = np.eye(3)
    np.testing.assert_allclose(camera.extrinsic_matrix, valid_extrinsic, atol=1e-6)


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


def test_camera_repr(camera_group_345: CameraGroup):
    """Test camera repr method."""
    camera_group = camera_group_345
    camera = camera_group.cameras[0]
    repr_str = str(camera)


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


def test_recording_session_add_video_non_video():
    """Test adding a non-Video object to RecordingSession."""
    camera_group = CameraGroup()
    camera = Camera(name="test_cam")
    camera_group.cameras.append(camera)
    session = RecordingSession(camera_group=camera_group)

    # Test with a string which is not a Video object
    with pytest.raises(ValueError) as excinfo:
        session.add_video(video="not_a_video", camera=camera)

    assert "Expected `Video` object" in str(excinfo.value)
    assert "str" in str(excinfo.value)


def test_camera_group_cameras():
    """Test camera group cameras method."""
    camera1 = Camera(name="camera1")
    camera2 = Camera(name="camera2")
    camera_group = CameraGroup(cameras=[camera1, camera2])

    assert camera_group.cameras == [camera1, camera2]

    camera_group = CameraGroup()
    assert camera_group.cameras == []


def test_camera_group_repr(camera_group_345: CameraGroup):
    camera_group = camera_group_345
    repr_str = str(camera_group)

    for cam_idx, camera in enumerate(camera_group.cameras):
        camera.name = f"camera_{cam_idx}"
    repr_str = str(camera_group)


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


def test_instance_group_properties():
    """Test the properties of an InstanceGroup more thoroughly."""
    camera1 = Camera(name="cam1")
    camera2 = Camera(name="cam2")
    skeleton = Skeleton(["A", "B"])

    instance1 = Instance({"A": [1, 2], "B": [3, 4]}, skeleton=skeleton)
    instance2 = Instance({"A": [5, 6], "B": [7, 8]}, skeleton=skeleton)

    # Test instance_by_camera property
    instance_by_camera = {camera1: instance1, camera2: instance2}
    instance_group = InstanceGroup(instance_by_camera=instance_by_camera)

    # Check that instance_by_camera returns the right dictionary
    assert instance_group.instance_by_camera is instance_group._instance_by_camera
    assert instance_group.instance_by_camera == instance_by_camera

    # Test score property
    test_score = 0.95
    instance_group = InstanceGroup(
        instance_by_camera=instance_by_camera, score=test_score
    )
    assert instance_group.score == test_score

    # Test points property
    test_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    instance_group = InstanceGroup(
        instance_by_camera=instance_by_camera, points=test_points
    )
    assert instance_group.points is instance_group._points
    np.testing.assert_array_equal(instance_group.points, test_points)


def test_instance_group_repr(instance_group_345: InstanceGroup):
    instance_group = instance_group_345
    repr_str = str(instance_group)


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


def test_frame_group_repr(frame_group_345: FrameGroup):
    frame_group = frame_group_345
    repr_str = str(frame_group)


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


def test_recording_session_repr(recording_session_345: RecordingSession):
    """Test recording session repr method."""
    session = recording_session_345
    repr_str = str(session)


def test_rodrigues_transformation():
    """Test the Rodrigues transformation implementation."""
    # Test with rotation vectors
    test_vectors = [
        np.array([0, 0, 0], dtype=np.float64),  # Identity
        np.array([1, 0, 0], dtype=np.float64),  # X-axis rotation
        np.array([0, 1, 0], dtype=np.float64),  # Y-axis rotation
        np.array([0, 0, 1], dtype=np.float64),  # Z-axis rotation
        np.array([0.5, 0.5, 0.5], dtype=np.float64),  # Combined rotation
    ]

    # Expected rotation matrices for the test vectors (pre-computed)
    expected_matrices = [
        # Identity
        np.eye(3),
        # X-axis rotation (angle = 1)
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5403023, -0.84147098],
                [0.0, 0.84147098, 0.5403023],
            ]
        ),
        # Y-axis rotation (angle = 1)
        np.array(
            [
                [0.5403023, 0.0, 0.84147098],
                [0.0, 1.0, 0.0],
                [-0.84147098, 0.0, 0.5403023],
            ]
        ),
        # Z-axis rotation (angle = 1)
        np.array(
            [
                [0.5403023, -0.84147098, 0.0],
                [0.84147098, 0.5403023, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        # Combined rotation (angle = sqrt(0.75))
        np.array(
            [
                [0.76524, -0.322422, 0.557183],
                [0.557183, 0.76524, -0.322422],
                [-0.322422, 0.557183, 0.76524],
            ]
        ),
    ]

    for idx, rvec in enumerate(test_vectors):
        # Test vector to matrix conversion
        rotation_matrix, _ = rodrigues_transformation(rvec)
        np.testing.assert_allclose(rotation_matrix, expected_matrices[idx], atol=1e-6)

        # Test matrix to vector conversion
        recovered_rvec, _ = rodrigues_transformation(rotation_matrix)

        # For zero rotation, the vector should be zero
        if np.allclose(rvec, 0):
            np.testing.assert_allclose(recovered_rvec, np.zeros(3), atol=1e-6)
        else:
            # For non-zero rotations, the vectors may differ in magnitude but should represent
            # the same rotation when converted back to matrices
            recovered_matrix, _ = rodrigues_transformation(recovered_rvec)
            np.testing.assert_allclose(recovered_matrix, rotation_matrix, atol=1e-6)

    # Test with invalid input shapes
    with pytest.raises(ValueError):
        rodrigues_transformation(np.array([1, 2]))

    with pytest.raises(ValueError):
        rodrigues_transformation(np.array([[1, 2], [3, 4]]))

    # Test with 3x1 column vector input
    column_vector = np.array([[1], [0], [0]], dtype=np.float64)
    rotation_matrix, _ = rodrigues_transformation(column_vector)
    np.testing.assert_allclose(rotation_matrix, expected_matrices[1], atol=1e-6)

    # Test with 180-degree rotation case (sin_theta = 0)
    # Create a 180-degree rotation matrix around X-axis
    pi_rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    rvec, _ = rodrigues_transformation(pi_rotation_x)
    # The resulting vector should represent a pi rotation around X-axis
    # Convert it back to matrix to verify
    rotation_matrix, _ = rodrigues_transformation(rvec)
    np.testing.assert_allclose(rotation_matrix, pi_rotation_x, atol=1e-6)

    # Test another 180-degree rotation case with different largest diagonal
    pi_rotation_y = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    rvec, _ = rodrigues_transformation(pi_rotation_y)
    rotation_matrix, _ = rodrigues_transformation(rvec)
    np.testing.assert_allclose(rotation_matrix, pi_rotation_y, atol=1e-6)

    # Test a pathological case where a diagonal element equals -1.0
    # This is a specific case where we need to ensure our algorithm is robust
    pi_rotation_z = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    rvec, _ = rodrigues_transformation(pi_rotation_z)
    rotation_matrix, _ = rodrigues_transformation(rvec)
    np.testing.assert_allclose(rotation_matrix, pi_rotation_z, atol=1e-6)


def test_recording_session_cameras():
    """Test `RecordingSession.cameras` property."""
    camera_1 = Camera(name="camera_1")
    camera_2 = Camera(name="camera_2")
    camera_group = CameraGroup(cameras=[camera_1, camera_2])

    # Test with no videos
    session = RecordingSession(camera_group=camera_group)
    assert session.cameras == []

    # Test with a single video
    video_1 = Video(filename="test_video_1.mp4")
    session.add_video(video=video_1, camera=camera_1)
    assert session.cameras == [camera_1]

    # Test with multiple videos
    video_2 = Video(filename="test_video_2.mp4")
    session.add_video(video=video_2, camera=camera_2)
    assert session.cameras == [camera_1, camera_2]
