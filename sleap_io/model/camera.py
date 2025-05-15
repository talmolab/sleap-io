"""Data structure for a single camera view in a multi-camera setup."""

from __future__ import annotations

import attrs
import numpy as np
from attrs import define, field
from attrs.validators import instance_of

from sleap_io.model.instance import Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.video import Video


def rodrigues_transformation(input_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert between rotation vector and rotation matrix using Rodrigues' formula.

    This function implements the Rodrigues' rotation formula to convert between:
    1. A 3D rotation vector (axis-angle representation) to a 3x3 rotation matrix
    2. A 3x3 rotation matrix to a 3D rotation vector

    Args:
        input_matrix: A 3x3 rotation matrix or a 3x1 rotation vector.

    Returns:
        A tuple containing the converted matrix/vector and the Jacobian (None for now).

    Raises:
        ValueError: If the input is not a valid rotation matrix or vector.
    """
    # Matrix to vector conversion
    if input_matrix.shape == (3, 3):
        # Get the rotation angle (trace(R) = 1 + 2*cos(theta))
        cos_theta = (np.trace(input_matrix) - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure numerical stability
        theta = np.arccos(cos_theta)

        # Handle small angles or identity rotation
        if np.isclose(theta, 0.0, atol=1e-8):
            # For small angles or identity, return zero vector
            return np.zeros(3), None

        # Compute the rotation axis
        sin_theta = np.sin(theta)
        if np.isclose(sin_theta, 0.0, atol=1e-8):
            # Handle 180-degree rotation (sin_theta = 0)
            # Find the largest diagonal element
            diag = np.diag(input_matrix)
            k = np.argmax(diag)
            axis = np.zeros(3)
            if diag[k] > -1.0:
                # Extract the column with largest diagonal
                axis[k] = 1.0
                v = input_matrix[:, k] + axis
                axis = v / np.linalg.norm(v)
            rvec = theta * axis
        else:
            # Normal case: extract the skew-symmetric part
            axis = np.array(
                [
                    input_matrix[2, 1] - input_matrix[1, 2],
                    input_matrix[0, 2] - input_matrix[2, 0],
                    input_matrix[1, 0] - input_matrix[0, 1],
                ]
            ) / (2.0 * sin_theta)

            # Ensure the axis is a unit vector
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = axis / axis_norm

            rvec = theta * axis

        return rvec, None

    # Vector to matrix conversion
    elif input_matrix.shape == (3,) or input_matrix.shape == (3, 1):
        # Handle both flat and column vectors
        rvec = input_matrix.ravel()
        theta = np.linalg.norm(rvec)

        # Handle small angles
        if np.isclose(theta, 0.0, atol=1e-8):
            return np.eye(3), None

        # Normalize the rotation axis
        axis = rvec / theta

        # Create the cross-product matrix
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )

        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        K_squared = np.dot(K, K)

        rotation_matrix = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * K_squared

        return rotation_matrix, None

    else:
        raise ValueError(
            f"Input must be a 3x3 matrix or a 3-element vector, got shape {input_matrix.shape}"
        )


@define
class CameraGroup:
    """A group of cameras used to record a multi-view `RecordingSession`.

    Attributes:
        cameras: List of `Camera` objects in the group.
        metadata: Dictionary of metadata.
    """

    cameras: list[Camera] = field(factory=list, validator=instance_of(list))
    metadata: dict = field(factory=dict, validator=instance_of(dict))

    def __repr__(self):
        """Return a readable representation of the camera group."""
        camera_names = ", ".join([c.name or "None" for c in self.cameras])
        return f"CameraGroup(cameras={len(self.cameras)}:[{camera_names}])"


@define(eq=False)  # Set eq to false to make class hashable
class RecordingSession:
    """A recording session with multiple cameras.

    Attributes:
        camera_group: `CameraGroup` object containing cameras in the session.
        frame_groups: Dictionary mapping frame index to `FrameGroup`.
        videos: List of `Video` objects linked to `Camera`s in the session.
        cameras: List of `Camera` objects linked to `Video`s in the session.
        metadata: Dictionary of metadata.
    """

    camera_group: CameraGroup = field(
        factory=CameraGroup, validator=instance_of(CameraGroup)
    )
    _video_by_camera: dict[Camera, Video] = field(
        factory=dict, validator=instance_of(dict)
    )
    _camera_by_video: dict[Video, Camera] = field(
        factory=dict, validator=instance_of(dict)
    )
    _frame_group_by_frame_idx: dict[int, FrameGroup] = field(
        factory=dict, validator=instance_of(dict)
    )
    metadata: dict = field(factory=dict, validator=instance_of(dict))

    @property
    def frame_groups(self) -> dict[int, FrameGroup]:
        """Get dictionary of `FrameGroup` objects by frame index.

        Returns:
            Dictionary of `FrameGroup` objects by frame index.
        """
        return self._frame_group_by_frame_idx

    @property
    def videos(self) -> list[Video]:
        """Get list of `Video` objects in the `RecordingSession`.

        Returns:
            List of `Video` objects in `RecordingSession`.
        """
        return list(self._video_by_camera.values())

    @property
    def cameras(self) -> list[Camera]:
        """Get list of `Camera` objects linked to `Video`s in the `RecordingSession`.

        Returns:
            List of `Camera` objects in `RecordingSession`.
        """
        return list(self._video_by_camera.keys())

    def get_camera(self, video: Video) -> Camera | None:
        """Get `Camera` associated with `video`.

        Args:
            video: `Video` to get `Camera`

        Returns:
            `Camera` associated with `video` or None if not found
        """
        return self._camera_by_video.get(video, None)

    def get_video(self, camera: Camera) -> Video | None:
        """Get `Video` associated with `camera`.

        Args:
            camera: `Camera` to get `Video`

        Returns:
            `Video` associated with `camera` or None if not found
        """
        return self._video_by_camera.get(camera, None)

    def add_video(self, video: Video, camera: Camera):
        """Add `video` to `RecordingSession` and mapping to `camera`.

        Args:
            video: `Video` object to add to `RecordingSession`.
            camera: `Camera` object to associate with `video`.

        Raises:
            ValueError: If `camera` is not in associated `CameraGroup`.
            ValueError: If `video` is not a `Video` object.
        """
        # Raise ValueError if camera is not in associated camera group
        self.camera_group.cameras.index(camera)

        # Raise ValueError if `Video` is not a `Video` object
        if not isinstance(video, Video):
            raise ValueError(
                f"Expected `Video` object, but received {type(video)} object."
            )

        # Add camera to video mapping
        self._video_by_camera[camera] = video

        # Add video to camera mapping
        self._camera_by_video[video] = camera

    def remove_video(self, video: Video):
        """Remove `video` from `RecordingSession` and mapping to `Camera`.

        Args:
            video: `Video` object to remove from `RecordingSession`.

        Raises:
            ValueError: If `video` is not in associated `RecordingSession`.
        """
        # Remove video from camera mapping
        camera = self._camera_by_video.pop(video)

        # Remove camera from video mapping
        self._video_by_camera.pop(camera)

    def __repr__(self) -> str:
        """Return a readable representation of the session."""
        return (
            "RecordingSession("
            f"camera_group={len(self.camera_group.cameras)}cameras, "
            f"videos={len(self.videos)}, "
            f"frame_groups={len(self.frame_groups)}"
            ")"
        )


@define(eq=False)  # Set eq to false to make class hashable
class Camera:
    """A camera used to record in a multi-view `RecordingSession`.

    Attributes:
        matrix: Intrinsic camera matrix of size (3, 3) and type float64.
        dist: Radial-tangential distortion coefficients [k_1, k_2, p_1, p_2, k_3] of
            size (5,) and type float64.
        size: Image size (width, height) of camera in pixels of size (2,) and type int.
        rvec: Rotation vector in unnormalized axis-angle representation of size (3,) and
            type float64.
        tvec: Translation vector of size (3,) and type float64.
        extrinsic_matrix: Extrinsic matrix of camera of size (4, 4) and type float64.
        name: Camera name.
        metadata: Dictionary of metadata.
    """

    matrix: np.ndarray = field(
        default=np.eye(3),
        converter=lambda x: np.array(x, dtype="float64"),
    )
    dist: np.ndarray = field(
        default=np.zeros(5), converter=lambda x: np.array(x, dtype="float64").ravel()
    )
    size: tuple[int, int] = field(
        default=None, converter=attrs.converters.optional(tuple)
    )
    _rvec: np.ndarray = field(
        default=np.zeros(3), converter=lambda x: np.array(x, dtype="float64").ravel()
    )
    _tvec: np.ndarray = field(
        default=np.zeros(3), converter=lambda x: np.array(x, dtype="float64").ravel()
    )
    name: str = field(default=None, converter=attrs.converters.optional(str))
    _extrinsic_matrix: np.ndarray = field(init=False)
    metadata: dict = field(factory=dict, validator=instance_of(dict))

    @matrix.validator
    @dist.validator
    @size.validator
    @_rvec.validator
    @_tvec.validator
    @_extrinsic_matrix.validator
    def _validate_shape(self, attribute: attrs.Attribute, value):
        """Validate shape of attribute based on metadata.

        Args:
            attribute: Attribute to validate.
            value: Value of attribute to validate.

        Raises:
            ValueError: If attribute shape is not as expected.
        """
        # Define metadata for each attribute
        attr_metadata = {
            "matrix": {"shape": (3, 3), "type": np.ndarray},
            "dist": {"shape": (5,), "type": np.ndarray},
            "size": {"shape": (2,), "type": tuple},
            "_rvec": {"shape": (3,), "type": np.ndarray},
            "_tvec": {"shape": (3,), "type": np.ndarray},
            "_extrinsic_matrix": {"shape": (4, 4), "type": np.ndarray},
        }
        optional_attrs = ["size"]

        # Skip validation if optional attribute is None
        if attribute.name in optional_attrs and value is None:
            return

        # Validate shape of attribute
        expected_shape = attr_metadata[attribute.name]["shape"]
        expected_type = attr_metadata[attribute.name]["type"]
        if np.shape(value) != expected_shape:
            raise ValueError(
                f"{attribute.name} must be a {expected_type} of size {expected_shape}, "
                f"but received shape: {np.shape(value)} and type: {type(value)} for "
                f"value: {value}"
            )

    def __attrs_post_init__(self):
        """Initialize extrinsic matrix from rotation and translation vectors."""
        self._extrinsic_matrix = np.eye(4, dtype="float64")
        self._extrinsic_matrix[:3, :3] = rodrigues_transformation(self._rvec)[0]
        self._extrinsic_matrix[:3, 3] = self._tvec

    @property
    def rvec(self) -> np.ndarray:
        """Get rotation vector of camera.

        Returns:
            Rotation vector of camera of size 3.
        """
        return self._rvec

    @rvec.setter
    def rvec(self, value: np.ndarray):
        """Set rotation vector and update extrinsic matrix.

        Args:
            value: Rotation vector of size 3.
        """
        self._rvec = value
        self._extrinsic_matrix[:3, :3] = rodrigues_transformation(self._rvec)[0]

    @property
    def tvec(self) -> np.ndarray:
        """Get translation vector of camera.

        Returns:
            Translation vector of camera of size 3.
        """
        return self._tvec

    @tvec.setter
    def tvec(self, value: np.ndarray):
        """Set translation vector and update extrinsic matrix.

        Args:
            value: Translation vector of size 3.
        """
        self._tvec = value

        # Update extrinsic matrix
        self._extrinsic_matrix[:3, 3] = self._tvec

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """Get extrinsic matrix of camera.

        Returns:
            Extrinsic matrix of camera of size 4 x 4.
        """
        return self._extrinsic_matrix

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, value: np.ndarray):
        """Set extrinsic matrix and update rotation and translation vectors.

        Args:
            value: Extrinsic matrix of size 4 x 4.
        """
        self._extrinsic_matrix = value

        # Update rotation and translation vectors
        self._rvec = rodrigues_transformation(self._extrinsic_matrix[:3, :3])[0].ravel()
        self._tvec = self._extrinsic_matrix[:3, 3]

    def get_video(self, session: RecordingSession) -> Video | None:
        """Get video associated with recording session.

        Args:
            session: Recording session to get video for.

        Returns:
            Video associated with recording session or None if not found.
        """
        return session.get_video(camera=self)

    def __repr__(self) -> str:
        """Return a readable representation of the camera."""
        matrix_str = (
            "identity" if np.array_equal(self.matrix, np.eye(3)) else "non-identity"
        )
        dist_str = "zero" if np.array_equal(self.dist, np.zeros(5)) else "non-zero"
        size_str = "None" if self.size is None else self.size
        rvec_str = (
            "zero"
            if np.array_equal(self.rvec, np.zeros(3))
            else np.array2string(self.rvec, precision=2, suppress_small=True)
        )
        tvec_str = (
            "zero"
            if np.array_equal(self.tvec, np.zeros(3))
            else np.array2string(self.tvec, precision=2, suppress_small=True)
        )
        name_str = self.name if self.name is not None else "None"
        return (
            "Camera("
            f"matrix={matrix_str}, "
            f"dist={dist_str}, "
            f"size={size_str}, "
            f"rvec={rvec_str}, "
            f"tvec={tvec_str}, "
            f"name={name_str}"
            ")"
        )


@define(eq=False)  # Set eq to false to make class hashable
class InstanceGroup:
    """Defines a group of instances across the same frame index.

    Attributes:
        instances_by_camera: Dictionary of `Instance` objects by `Camera`.
        instances: List of `Instance` objects in the group.
        cameras: List of `Camera` objects that have an `Instance` associated.
        score: Optional score for the `InstanceGroup`. Setting the score will also
            update the score for all `instances` already in the `InstanceGroup`. The
            score for `instances` will not be updated upon initialization.
        points: Optional 3D points for the `InstanceGroup`.
        metadata: Dictionary of metadata.
    """

    _instance_by_camera: dict[Camera, Instance] = field(
        factory=dict, validator=instance_of(dict)
    )
    _score: float | None = field(
        default=None, converter=attrs.converters.optional(float)
    )
    _points: np.ndarray | None = field(
        default=None,
        converter=attrs.converters.optional(lambda x: np.array(x, dtype="float64")),
    )
    metadata: dict = field(factory=dict, validator=instance_of(dict))

    @property
    def instance_by_camera(self) -> dict[Camera, Instance]:
        """Get dictionary of `Instance` objects by `Camera`."""
        return self._instance_by_camera

    @property
    def instances(self) -> list[Instance]:
        """List of `Instance` objects."""
        return list(self._instance_by_camera.values())

    @property
    def cameras(self) -> list[Camera]:
        """List of `Camera` objects."""
        return list(self._instance_by_camera.keys())

    @property
    def score(self) -> float | None:
        """Get score for `InstanceGroup`."""
        return self._score

    @property
    def points(self) -> np.ndarray | None:
        """Get 3D points for `InstanceGroup`."""
        return self._points

    def get_instance(self, camera: Camera) -> Instance | None:
        """Get `Instance` associated with `camera`.

        Args:
            camera: `Camera` to get `Instance`.

        Returns:
            `Instance` associated with `camera` or None if not found.
        """
        return self._instance_by_camera.get(camera, None)

    def __repr__(self) -> str:
        """Return a readable representation of the instance group."""
        cameras_str = ", ".join([c.name or "None" for c in self.cameras])
        return f"InstanceGroup(cameras={len(self.cameras)}:[{cameras_str}])"


@define(eq=False)  # Set eq to false to make class hashable
class FrameGroup:
    """Defines a group of `InstanceGroups` across views at the same frame index.

    Attributes:
        frame_idx: Frame index for the `FrameGroup`.
        instance_groups: List of `InstanceGroup`s in the `FrameGroup`.
        cameras: List of `Camera` objects linked to `LabeledFrame`s in the `FrameGroup`.
        labeled_frames: List of `LabeledFrame`s in the `FrameGroup`.
        metadata: Metadata for the `FrameGroup` that is provided but not deserialized.
    """

    frame_idx: int = field(converter=int)
    _instance_groups: list[InstanceGroup] = field(
        factory=list, validator=instance_of(list)
    )
    _labeled_frame_by_camera: dict[Camera, LabeledFrame] = field(
        factory=dict, validator=instance_of(dict)
    )
    metadata: dict = field(factory=dict, validator=instance_of(dict))

    @property
    def instance_groups(self) -> list[InstanceGroup]:
        """List of `InstanceGroup`s."""
        return self._instance_groups

    @property
    def cameras(self) -> list[Camera]:
        """List of `Camera` objects."""
        return list(self._labeled_frame_by_camera.keys())

    @property
    def labeled_frames(self) -> list[LabeledFrame]:
        """List of `LabeledFrame`s."""
        return list(self._labeled_frame_by_camera.values())

    def get_frame(self, camera: Camera) -> LabeledFrame | None:
        """Get `LabeledFrame` associated with `camera`.

        Args:
            camera: `Camera` to get `LabeledFrame`.

        Returns:
            `LabeledFrame` associated with `camera` or None if not found.
        """
        return self._labeled_frame_by_camera.get(camera, None)

    def __repr__(self) -> str:
        """Return a readable representation of the frame group."""
        cameras_str = ", ".join([c.name or "None" for c in self.cameras])
        return (
            f"FrameGroup("
            f"frame_idx={self.frame_idx},"
            f"instance_groups={len(self.instance_groups)},"
            f"cameras={len(self.cameras)}:[{cameras_str}]"
            f")"
        )
