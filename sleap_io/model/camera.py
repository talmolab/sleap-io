"""Data structure for a single camera view in a multi-camera setup."""

from __future__ import annotations

import attrs
import cv2
import numpy as np
from attrs import define, field

from sleap_io.model.video import Video


@define
class CameraGroup:
    """A group of cameras used to record a multi-view `RecordingSession`.

    Attributes:
        cameras: List of `Camera` objects in the group.
    """

    cameras: list[Camera] = field(factory=list)


@define(eq=False)  # Set eq to false to make class hashable
class RecordingSession:
    """A recording session with multiple cameras.

    Attributes:
        camera_group: `CameraGroup` object containing cameras in the session.
        _video_by_camera: Dictionary mapping `Camera` to `Video`.
        _camera_by_video: Dictionary mapping `Video` to `Camera`.
    """

    camera_group: CameraGroup = field(factory=CameraGroup)
    _video_by_camera: dict[Camera, Video] = field(factory=dict)
    _camera_by_video: dict[Video, Camera] = field(factory=dict)

    @property
    def videos(self) -> list[Video]:
        """Get list of `Video` objects in `RecordingSession`.

        Returns:
            List of `Video` objects in `RecordingSession`.
        """
        return list(self._video_by_camera.values())

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
        _video_by_session: Dictionary mapping `RecordingSession` to `Video`.
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
        # Initialize extrinsic matrix
        self._extrinsic_matrix = np.eye(4, dtype="float64")
        self._extrinsic_matrix[:3, :3] = cv2.Rodrigues(self._rvec)[0]
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

        # Update extrinsic matrix
        rotation_matrix, _ = cv2.Rodrigues(self._rvec)
        self._extrinsic_matrix[:3, :3] = rotation_matrix

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
        self._rvec, _ = cv2.Rodrigues(self._extrinsic_matrix[:3, :3])
        self._tvec = self._extrinsic_matrix[:3, 3]

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort points using camera matrix and distortion coefficients.

        Args:
            points: Points to undistort of shape (N, 2).

        Returns:
            Undistorted points of shape (N, 2).
        """
        shape = points.shape
        points = points.reshape(-1, 1, 2)
        out = cv2.undistortPoints(points.astype("float64"), self.matrix, self.dist)
        return out.reshape(shape)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using camera matrix and distortion coefficients.

        Args:
            points: 3D points to project of any dtype and shape (..., 3) where "..." is
                any number of dimensions (including 0).

        Returns:
            Projected 2D points of same dtype as `points` and shape (..., 2) where "..."
            is the same as the "..." dimensions of `points`.
        """
        # Validate points in
        points_shape = points.shape
        try:
            if points_shape[-1] != 3:
                raise ValueError
            points = points.reshape(-1, 1, 3)
        except Exception as e:
            raise ValueError(
                "Expected points to be an array of 3D points of shape (..., 3) where "
                "'...' is any number of non-zero dimensions, but received shape "
                f"{points_shape}.\n\n{e}"
            )

        # Project points
        points_dtype = points.dtype
        points = points.astype("float64")  # Ensure points are float for cv2 project
        out, _ = cv2.projectPoints(
            points,
            self.rvec,
            self.tvec,
            self.matrix,
            self.dist,
        )
        return out.reshape(*points_shape[:-1], 2).astype(points_dtype)

    def get_video(self, session: RecordingSession) -> Video | None:
        """Get video associated with recording session.

        Args:
            session: Recording session to get video for.

        Returns:
            Video associated with recording session or None if not found.
        """
        return session.get_video(camera=self)

    def to_dict(self) -> dict:
        """Convert `Camera` to dictionary.

        Returns:
            Dictionary containing camera information with the following keys:
            - name: Camera name.
            - size: Image size (width, height) of camera in pixels of size (2,) and
                type int.
            - matrix: Intrinsic camera matrix of size (3, 3) and type float64.
            - distortions: Radial-tangential distortion coefficients
                [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
            - rotation: Rotation vector in unnormalized axis-angle representation of
                size (3,) and type float64.
            - translation: Translation vector of size (3,) and type float64.
        """
        camera_dict = {
            "name": self.name,
            "size": list(self.size),
            "matrix": self.matrix.tolist(),
            "distortions": self.dist.tolist(),
            "rotation": self.rvec.tolist(),
            "translation": self.tvec.tolist(),
        }

        return camera_dict

    @classmethod
    def from_dict(cls, camera_dict: dict) -> Camera:
        """Create `Camera` from dictionary.

        Args:
            camera_dict: Dictionary containing camera information with the following
                keys:
                - name: Camera name.
                - size: Image size (width, height) of camera in pixels of size (2,) and
                    type int.
                - matrix: Intrinsic camera matrix of size (3, 3) and type float64.
                - distortions: Radial-tangential distortion coefficients
                    [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
                - rotation: Rotation vector in unnormalized axis-angle representation of
                    size (3,) and type float64.
                - translation: Translation vector of size (3,) and type float64.

        Returns:
            `Camera` object created from dictionary.
        """
        camera = cls(
            name=camera_dict["name"],
            size=camera_dict["size"],
            matrix=camera_dict["matrix"],
            dist=camera_dict["distortions"],
            rvec=camera_dict["rotation"],
            tvec=camera_dict["translation"],
        )

        return camera

    # TODO: Remove this when we implement triangulation without aniposelib
    def __getattr__(self, name: str):
        """Get attribute by name.

        Args:
            name: Name of attribute to get.

        Returns:
            Value of attribute.

        Raises:
            AttributeError: If attribute does not exist.
        """
        if name in self.__attrs_attrs__:
            return getattr(self, name)

        # The aliases for methods called when triangulate with sleap_anipose
        method_aliases = {
            "get_name": self.name,
            "get_extrinsic_matrix": self.extrinsic_matrix,
        }

        def return_callable_method_alias():
            return method_aliases[name]

        if name in method_aliases:
            return return_callable_method_alias

        raise AttributeError(f"'Camera' object has no attribute or method '{name}'")
