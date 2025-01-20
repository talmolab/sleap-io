"""Data structure for a single camera view in a multi-camera setup."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import cv2
import numpy as np
import toml
from attrs import define, field

from sleap_io.model.video import Video


def triangulate_dlt_vectorized(
    points: np.ndarray, projection_matrices: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from multiple camera views using Direct Linear Transform.

    Args:
        points: Array of N 2D points from each camera view M of shape (M, N, 2).
        projection_matrices: Array of (3, 4) projection matrices for each camera M of
            shape (M, 3, 4).

    Returns:
        Triangulated 3D points of shape (N, 3).
    """
    n_cameras, n_points, _ = points.shape

    # Flatten points to shape needed for multiplication
    points_flattened = points.reshape(n_cameras, 2 * n_points, 1, order="C")

    # Create row selector matrix to select correct rows from projection matrix
    row_selector = np.zeros((n_cameras * n_points, 2, 2))
    row_selector[:, 0, 0] = -1  # Negate 1st row of projection matrix for x
    row_selector[:, 1, 1] = -1  # Negate 2nd row of projection matrix for y
    row_selector = row_selector.reshape(n_cameras, 2 * n_points, 2, order="C")

    # Concatenate row selector and points matrices to shape (M, 2N, 3)
    left_matrix = np.concatenate((row_selector, points_flattened), axis=2)

    # Get A (stacked in a weird way) of shape (M, 2N, 4)
    a_stacked = np.matmul(left_matrix, projection_matrices)

    # Reorganize A to shape (N, 2M, 4) s.t. each 3D point has A of shape 2M x 4
    a = (
        a_stacked.reshape(n_cameras, n_points, 2, 4)
        .transpose(1, 0, 2, 3)
        .reshape(n_points, 2 * n_cameras, 4)
    )

    # Remove rows with NaNs before SVD which may result in a ragged A (hence for loop)
    points_3d = []
    for a_slice in a:
        nan_mask = np.isnan(a_slice)
        if np.all(nan_mask):
            # If all rows are NaN, set point to NaN
            point_3d = np.full(3, np.nan)
            # TODO: Also filter out if ony 2 rows (a single point) is non-nan
        else:
            a_no_nan = a_slice[~nan_mask].reshape(-1, 4, order="C")

            _, _, vh = np.linalg.svd(a_no_nan)

            point_3d = vh[-1, :-1] / vh[-1, -1]
        points_3d.append(point_3d)

    points_3d = np.array(points_3d)

    return points_3d


@define
class CameraGroup:
    """A group of cameras used to record a multi-view `RecordingSession`.

    Attributes:
        cameras: List of `Camera` objects in the group.
    """

    cameras: list[Camera] = field(factory=list)

    def triangulate(
        self,
        points: np.ndarray,
        triangulation_func: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ] = triangulate_dlt_vectorized,
    ) -> np.ndarray:
        """Triangulate 2D points from multiple camera views M.

        This function reshapes the input `points` to shape (M, N, 2) where M is the
        number of camera views and N is the number of 2D points to triangulate. The
        points are then undistorted so that we can use the extrinsic matrices of the
        cameras as projection matrices to call `triangulation_func` and triangulate
        the 3D points.

        Args:
            points: Array of 2D points from each camera view of shape (M, ..., 2) where
                M is the number of camera views and "..." is any number of dimensions
                (including 0).
            triangulation_func: Function to use for triangulation. The
            triangulation_func should take the following arguments:
                - points: Array of undistorted 2D points from each camera view of shape
                    (M, N, 2) where M is the number of cameras and N is the number of
                    points.
                - projection_matrices: Array of (3, 4) projection matrices for each of
                    the M cameras of shape (M, 3, 4) - note that points are undistorted.
                and return the triangulated 3D points of shape (N, 3).
            Default is vectorized DLT.


        Raises:
            ValueError: If points are not of shape (M, ..., 2).
            ValueError: If number of cameras M do not match number of cameras in group.
            ValueError: If number of points returned by triangulation function does not
                match number of points in input.

        Returns:
            Triangulated 3D points of shape (..., 3) where "..." is any number of
            dimensions and matches the "..." dimensions of the input points array.
        """
        # Validate points in
        points_shape = points.shape
        try:
            n_cameras = points_shape[0]
            if n_cameras != len(self.cameras):
                raise ValueError
            if 2 != points.shape[-1]:
                raise ValueError
            if len(points_shape) != 3:
                points = points.reshape(n_cameras, -1, 2)
        except Exception as e:
            raise ValueError(
                "Expected points to be an array of 2D points from each camera view of "
                f"shape (M, ..., 2) where M = {len(self.cameras)} and '...' is any "
                f"number of dimensions, but received shape {points_shape}.\n\n{e}"
            )
        n_points = points.shape[1]
        # Undistort points
        points = points.astype("float64")  # Ensure float64 for opencv undistort
        for cam_idx, camera in enumerate(self.cameras):
            cam_points = camera.undistort_points(points[cam_idx])
            points[cam_idx] = cam_points

        # Since points are undistorted, use extrinsic matrices as projection matrices
        projection_matrices = np.array(
            [camera.extrinsic_matrix[:3] for camera in self.cameras]
        )

        # Triangulate points using provided function
        points_3d = triangulation_func(points, projection_matrices)
        n_points_returned = points_3d.shape[0]
        if n_points_returned != n_points:
            raise ValueError(
                f"Expected triangulation function to return {n_points} 3D points, but "
                f"received {n_points_returned} 3D points."
            )

        return points_3d.reshape(*points_shape[1:-1], 3)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using camera group.

        Args:
            points: 3D points to project of shape (..., 3). where "..." is any number
                of dimensions (including 0).

        Returns:
            Projected 2D points of shape (M, ..., 2) where M is the number of
            cameras and "..." matches the "..." dimensions of the input points array.
        """
        # Validate points in
        points = points.astype(np.float64)
        points_shape = points.shape
        try:
            # Check if points are 3D
            if points_shape[-1] != 3:
                raise ValueError
        except Exception as e:
            raise ValueError(
                "Expected points to be an array of 3D points of shape (..., 3) "
                "where '...' is any number of non-zero dimensions, but received shape "
                f"{points_shape}.\n\n{e}"
            )

        # Project 3D points to 2D for each camera
        n_cameras = len(self.cameras)
        n_points = np.prod(points_shape[:-1])
        projected_points = np.zeros((n_cameras, n_points, 2))
        for cam_idx, camera in enumerate(self.cameras):
            cam_points = camera.project(points)
            projected_points[cam_idx] = cam_points.reshape(n_points, 2)

        return projected_points.reshape(n_cameras, *points_shape[:-1], 2)

    @classmethod
    def from_dict(cls, calibration_dict: dict) -> CameraGroup:
        """Create `CameraGroup` from calibration dictionary.

        Args:
            calibration_dict: Dictionary containing calibration information for cameras.

        Returns:
            `CameraGroup` object created from calibration dictionary.
        """
        cameras = []
        for dict_name, camera_dict in calibration_dict.items():
            if dict_name == "metadata":
                continue
            camera = Camera.from_dict(camera_dict)
            cameras.append(camera)

        camera_group = cls(cameras=cameras)

        return camera_group

    def to_dict(self) -> dict:
        """Convert `CameraGroup` to dictionary.

        Returns:
            Dictionary containing camera group information with the following keys:
            - cam_n: Camera dictionary containing information for camera at index "n"
                with the following keys:
                - name: Camera name.
                - size: Image size (height, width) of camera in pixels of size (2,) and
                    type int.
                - matrix: Intrinsic camera matrix of size (3, 3) and type float64.
                - distortions: Radial-tangential distortion coefficients
                    [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
                - rotation: Rotation vector in unnormalized axis-angle representation of
                    size (3,) and type float64.
                - translation: Translation vector of size (3,) and type float64.
        """
        calibration_dict = {}
        for cam_idx, camera in enumerate(self.cameras):
            camera_dict = camera.to_dict()
            calibration_dict[f"cam_{cam_idx}"] = camera_dict

        return calibration_dict

    @classmethod
    def load(cls, filename: str) -> CameraGroup:
        """Load `CameraGroup` from JSON file.

        Args:
            filename: Path to JSON file to load `CameraGroup` from.

        Returns:
            `CameraGroup` object loaded from JSON file.
        """
        calibration_dict = toml.load(filename)
        camera_group = cls.from_dict(calibration_dict)

        return camera_group


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
            points: 3D points to project of shape (..., 3) where "..." is any number of
                dimensions (including 0).

        Returns:
            Projected 2D points of shape (..., 2) where "..." is the same as the input
            "..." dimensions.
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
        out, _ = cv2.projectPoints(
            points,
            self.rvec,
            self.tvec,
            self.matrix,
            self.dist,
        )
        return out.reshape(*points_shape[:-1], 2)

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
