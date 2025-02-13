"""Data structure for a single camera view in a multi-camera setup."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import cv2
import numpy as np
import toml
from attrs import define, field
from attrs.validators import instance_of

from sleap_io.model.instance import Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.video import Video


# TODO(LM): Move to algorithms in another repo.
def triangulate_dlt_vectorized(
    points: np.ndarray, projection_matrices: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from multiple camera views using Direct Linear Transform.

    Args:
        points: Array of N 2D points from each camera view M of dtype float64 and shape
            (M, N, 2) where N is the number of points.
        projection_matrices: Array of (3, 4) projection matrices for each camera M of
            shape (M, 3, 4).

    Returns:
        Triangulated 3D points of shape (N, 3) where N is the number of points.
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
        # Check that we have at least 2 views worth of non-nan points.
        nan_mask = np.isnan(a_slice)  # 2M x 4
        has_enough_matches = np.all(~nan_mask, axis=1).sum() >= 4  # Need 2 (x, y) pairs

        point_3d = np.full(3, np.nan)
        if has_enough_matches:
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
    metadata: dict = field(factory=dict)  # TODO(LM): Add metadata to CameraGroup

    # TODO: Remove this method (should be a util function in another repo).
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
            points: Array of 2D points from each camera view of any dtype and shape
                (M, ..., 2) where M is the number of camera views and "..." is any
                number of dimensions (including 0).
            triangulation_func: Function to use for triangulation. The
            triangulation_func should take the following arguments:
                points: Array of undistorted 2D points from each camera view of dtype
                    float64 and shape (M, N, 2) where M is the number of cameras and N
                    is the number of points.
                projection_matrices: Array of (3, 4) projection matrices for each of the
                    M cameras of shape (M, 3, 4) - note that points are undistorted.
                and return the triangulated 3D points of shape (N, 3) where N is the
                number of points.
            Default is vectorized DLT.


        Raises:
            ValueError: If points are not of shape (M, ..., 2).
            ValueError: If number of cameras M do not match number of cameras in group.
            ValueError: If number of points returned by triangulation function does not
                match number of points in input.

        Returns:
            Triangulated 3D points of same dtype as `points` and shape (..., 3) where
            "..." is any number of dimensions and matches the "..." dimensions of
            `points`.
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
        points_dtype = points.dtype
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

        # Reshape to (N, 3) and cast to the original dtype.
        points_3d = points_3d.reshape(*points_shape[1:-1], 3).astype(points_dtype)
        return points_3d

    # TODO: Remove this method (should be a util function in another repo).
    def project(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using camera group.

        Args:
            points: 3D points to project of any dtype and shape (..., 3). where "..." is
                any number of dimensions (including 0).

        Returns:
            Projected 2D points of same dtype as `points` and shape (M, ..., 2)
            where M is the number of cameras and "..." matches the "..." dimensions of
            `points`.
        """
        # Validate points in
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
        points_dtype = points.dtype
        points = points.astype(np.float64)  # Ensure float for opencv project
        n_cameras = len(self.cameras)
        n_points = np.prod(points_shape[:-1])
        projected_points = np.zeros((n_cameras, n_points, 2))
        for cam_idx, camera in enumerate(self.cameras):
            cam_points = camera.project(points)
            projected_points[cam_idx] = cam_points.reshape(n_points, 2)

        return projected_points.reshape(n_cameras, *points_shape[:-1], 2).astype(
            points_dtype
        )

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
                cam_n: Camera dictionary containing information for camera at index "n"
                    with the following keys:
                    name: Camera name.
                    size: Image size (height, width) of camera in pixels of size (2,)
                        and type int.
                    matrix: Intrinsic camera matrix of size (3, 3) and type float64.
                    distortions: Radial-tangential distortion coefficients
                        [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
                    rotation: Rotation vector in unnormalized axis-angle representation
                        of size (3,) and type float64.
                    translation: Translation vector of size (3,) and type float64.
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
    metadata: dict = field(factory=dict)
    _video_by_camera: dict[Camera, Video] = field(factory=dict)
    _camera_by_video: dict[Video, Camera] = field(factory=dict)
    _frame_group_by_frame_idx: dict[int, FrameGroup] = field(factory=dict)

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

    def to_dict(
        self,
        video_to_idx: dict[Video, int],
        labeled_frame_to_idx: dict[LabeledFrame, int],
    ) -> dict:
        """Unstructure `RecordingSession` to an invertible dictionary.

        Args:
            video_to_idx: Dictionary of `Video` to index in `Labels.videos`.
            labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
                `Labels.labeled_frames`.

        Returns:
            Dictionary of "calibration" and "camcorder_to_video_idx_map" needed to
            restructure a `RecordingSession`.
        """

        # Unstructure `CameraCluster` and `metadata`
        calibration_dict = self.camera_group.to_dict()

        # Store camcorder-to-video indices map where key is camcorder index
        # and value is video index from `Labels.videos`
        camera_to_video_idx_map = {}
        for cam_idx, camera in enumerate(self.camera_group.cameras):
            # Skip if Camcorder is not linked to any Video
            if camera not in self._video_by_camera:
                continue

            # Get video index from `Labels.videos`
            video = self._video_by_camera[camera]
            video_idx = video_to_idx.get(video, None)

            if video_idx is not None:
                camera_to_video_idx_map[str(cam_idx)] = str(video_idx)
            else:
                print(
                    f"Video {video} not found in `Labels.videos`. "
                    "Not saving to `RecordingSession` serialization."
                )

        # Store frame groups by frame index
        frame_group_dicts = []
        if len(labeled_frame_to_idx) > 0:  # Don't save if skipping labeled frames
            for frame_group in self._frame_group_by_frame_idx.values():
                # Only save `FrameGroup` if it has `InstanceGroup`s
                if len(frame_group.instance_groups) > 0:
                    frame_group_dict = frame_group.to_dict(
                        labeled_frame_to_idx=labeled_frame_to_idx
                    )
                    frame_group_dicts.append(frame_group_dict)

        return {
            "calibration": calibration_dict,
            "camcorder_to_video_idx_map": camera_to_video_idx_map,
            "frame_group_dicts": frame_group_dicts,
        }

    @classmethod
    def load(
        cls,
        filename,
        metadata: dict | None = None,
    ) -> "RecordingSession":
        """Loads cameras as `Camcorder`s from a calibration.toml file.

        Args:
            filename: Path to calibration.toml file.
            metadata: Dictionary of metadata.

        Returns:
            `RecordingSession` object.
        """

        camera_group: CameraGroup = CameraGroup.load(filename)
        return cls(
            camera_group=camera_group,
            metadata=(metadata or {}),
        )

    @classmethod
    def from_calibration_dict(cls, calibration_dict: dict) -> "RecordingSession":
        """Loads cameras as `Camcorder`s from a calibration dictionary.

        Args:
            calibration_dict: Dictionary of calibration data.

        Returns:
            `RecordingSession` object.
        """

        camera_group: CameraGroup = CameraGroup.from_dict(
            calibration_dict=calibration_dict,
        )
        return cls(camera_group=camera_group)

    @classmethod
    def from_dict(
        cls,
        session_dict: dict,
        videos: list[Video],
        labeled_frames: list[LabeledFrame],
    ) -> RecordingSession:
        """Restructure `RecordingSession` from an invertible dictionary.

        Args:
            session_dict: Dictionary of "calibration" and "camcorder_to_video_idx_map"
                needed to fully restructure a `RecordingSession`.
            videos_list: List containing `Video` objects (expected `Labels.videos`).
            labeled_frames_list: List containing `LabeledFrame` objects (expected
                `Labels.labeled_frames`).

        Returns:
            `RecordingSession` object.
        """

        # Restructure `RecordingSession` without `Video` to `Camcorder` mapping
        calibration_dict = session_dict["calibration"]
        session: RecordingSession = RecordingSession.from_calibration_dict(
            calibration_dict
        )

        # Retrieve all `Camcorder` and `Video` objects, then add to `RecordingSession`
        camcorder_to_video_idx_map = session_dict["camcorder_to_video_idx_map"]
        for cam_idx, video_idx in camcorder_to_video_idx_map.items():
            camcorder = session.camera_group.cameras[int(cam_idx)]
            video = videos[int(video_idx)]
            session.add_video(video, camcorder)

        # Reconstruct all `FrameGroup` objects and add to `RecordingSession`
        frame_group_dicts = session_dict.get("frame_group_dicts", [])
        for frame_group_dict in frame_group_dicts:

            try:
                # Add `FrameGroup` to `RecordingSession`
                FrameGroup.from_dict(
                    frame_group_dict=frame_group_dict,
                    session=session,
                    labeled_frames_list=labeled_frames,
                )
            except ValueError as e:
                print(
                    f"Error reconstructing FrameGroup: {frame_group_dict}. Skipping..."
                    f"\n{e}"
                )

        return session


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
    metadata: dict = field(factory=dict)

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
            name: Camera name.
            size: Image size (width, height) of camera in pixels of size (2,) and type
                int.
            matrix: Intrinsic camera matrix of size (3, 3) and type float64.
            distortions: Radial-tangential distortion coefficients
                [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
            rotation: Rotation vector in unnormalized axis-angle representation of size
                (3,) and type float64.
            translation: Translation vector of size (3,) and type float64.
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
                name: Camera name.
                size: Image size (width, height) of camera in pixels of size (2,) and
                    type int.
                matrix: Intrinsic camera matrix of size (3, 3) and type float64.
                distortions: Radial-tangential distortion coefficients
                    [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
                rotation: Rotation vector in unnormalized axis-angle representation of
                    size (3,) and type float64.
                translation: Translation vector of size (3,) and type float64.

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


@define(eq=False)  # Set eq to false to make class hashable
class InstanceGroup:
    """Defines a group of instances across the same frame index.

    Attributes:
        dummy_instance: Optional `PredictedInstance` object to fill in for missing
            instances.
        cameras: List of `Camcorder` objects that have an `Instance` associated.
        instances: List of `Instance` objects.
        instance_by_camera: Dictionary of `Instance` objects by `Camcorder`.
        score: Optional score for the `InstanceGroup`. Setting the score will also
            update the score for all `instances` already in the `InstanceGroup`. The
            score for `instances` will not be updated upon initialization.
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

    def to_dict(
        self,
        instance_to_lf_and_inst_idx: dict[Instance, tuple[str, str]],
        camera_group: CameraGroup,
    ) -> dict[str, str | dict[str, str]]:
        """Converts the `InstanceGroup` to a dictionary.

        Args:
            instance_to_lf_and_inst_idx: Dictionary mapping `Instance` objects to
                `LabeledFrame` indices (in `Labels.labeled_frames`) and `Instance`
                indices (in containing `LabeledFrame.instances`).
            camera_group: `CameraGroup` object that determines the order of the
                `Camcorder` objects when converting to a dictionary.

        Returns:
            Dictionary of the `InstanceGroup` with items:
                - camera_to_lf_and_inst_idx_map: Dictionary mapping `Camcorder` indices
                    (in `InstanceGroup.camera_cluster.cameras`) to both `LabeledFrame`
                    and `Instance` indices (from `instance_to_lf_and_inst_idx`).
        """
        camera_to_lf_and_inst_idx_map: dict[str, tuple[str, str]] = {
            str(camera_group.cameras.index(cam)): instance_to_lf_and_inst_idx[instance]
            for cam, instance in self._instance_by_camera.items()
        }

        # Only required key is camera_to_lf_and_inst_idx_map
        instance_group_dict = {
            "camera_to_lf_and_inst_idx_map": camera_to_lf_and_inst_idx_map,
        }

        # Optionally add score, points, and metadata if they are non-default values
        if self._score is not None:
            instance_group_dict["score"] = str(round(self._score, 4))
        if self._points is not None:
            instance_group_dict["points"] = self._points.tolist()
        if len(self.metadata) > 0:
            instance_group_dict.update(self.metadata)

        return instance_group_dict

    @classmethod
    def from_dict(
        cls,
        instance_group_dict: dict,
        labeled_frames: list[LabeledFrame],
        camera_group: CameraGroup,
    ):
        """Creates an `InstanceGroup` object from a dictionary.

        Args:
            instance_group_dict: Dictionary with the following necessary keys:
                camera_to_lf_and_inst_idx_map: Dictionary mapping `Camcorder` indices
                    to a tuple of `LabeledFrame` index (in `labeled_frames`) and
                    `Instance` index (in containing `LabeledFrame.instances`).
                and optional keys:
                score: Score for the `InstanceGroup`.
                points: 3D points for the `InstanceGroup`.
                etc. (metadata)
            labeled_frames: List of `LabeledFrame` objects (expecting
                `Labels.labeled_frames`) used to retrieve `Instance` objects.
            camera_group: `CameraGroup` object used to retrieve `Camera` objects.

        Returns:
            `InstanceGroup` object.
        """
        # Avoid mutating the dictionary
        instance_group_dict = instance_group_dict.copy()

        # Get the `Instance` objects
        camera_to_lf_and_inst_idx_map: dict[str, tuple[str, str]] = (
            instance_group_dict.pop("camera_to_lf_and_inst_idx_map")
        )

        instance_by_camera: dict[Camera, Instance] = {}
        for cam_idx, (lf_idx, inst_idx) in camera_to_lf_and_inst_idx_map.items():
            # Retrieve the `Camera`
            camera = camera_group.cameras[int(cam_idx)]

            # Retrieve the `Instance` from the `LabeledFrame
            labeled_frame = labeled_frames[int(lf_idx)]
            instance = labeled_frame.instances[int(inst_idx)]

            # Link the `Instance` to the `Camera`
            instance_by_camera[camera] = instance

        # Get all optional attributes
        score = None
        if "score" in instance_group_dict:
            score = instance_group_dict.pop("score")
        points = None
        if "points" in instance_group_dict:
            points = instance_group_dict.pop("points")

        # Metadata contains any information that the class doesn't deserialize.
        metadata = instance_group_dict  # Remaining keys are metadata.

        return cls(
            instance_by_camera=instance_by_camera,
            score=score,
            points=points,
            metadata=metadata,
        )


@define(eq=False)  # Set eq to false to make class hashable
class FrameGroup:
    """Defines a group of `InstanceGroups` across views at the same frame index.

    Attributes:
        frame_idx: Frame index for the `FrameGroup`.
        session: `RecordingSession` object that the `FrameGroup` is in.
        instance_groups: List of `InstanceGroup`s in the `FrameGroup`.
        labeled_frames: List of `LabeledFrame`s in the `FrameGroup`.
        cameras: List of `Camcorder`s that have `LabeledFrame`s.
    """

    # Instance attributes
    frame_idx: int = field(validator=instance_of(int))
    metadata: dict = field(factory=dict)
    _instance_groups: list[InstanceGroup] = field(
        factory=list, validator=instance_of(list)
    )  # Akin to `LabeledFrame.instances`

    # "Hidden" instance attributes

    # TODO(LM): This dict should be updated each time a LabeledFrame is added/removed
    # from the Labels object. Or if a video is added/removed from the RecordingSession.
    _labeled_frame_by_cam: dict[Camera, LabeledFrame] = field(factory=dict)
    _instances_by_cam: dict[Camera, set[Instance]] = field(factory=dict)

    @property
    def labeled_frames(self) -> list[LabeledFrame]:
        """List of `LabeledFrame`s."""
        # TODO(LM): Revisit whether we need to return a list instead of a view object
        return list(self._labeled_frame_by_cam.values())

    @property
    def instance_groups(self) -> list[InstanceGroup]:
        """List of `InstanceGroup`s."""
        return self._instance_groups

    @instance_groups.setter
    def instance_groups(self, instance_groups: list[InstanceGroup]):
        """Setter for `instance_groups` that updates `LabeledFrame`s and `Instance`s."""

        instance_groups_to_remove = set(self.instance_groups) - set(instance_groups)
        instance_groups_to_add = set(instance_groups) - set(self.instance_groups)

        # Update the `_labeled_frame_by_cam` and `_instances_by_cam` dictionary
        for instance_group in instance_groups_to_remove:
            self.remove_instance_group(instance_group=instance_group)

        for instance_group in instance_groups_to_add:
            self.add_instance_group(instance_group=instance_group)

    def add_instance_group(self, instance_group: InstanceGroup):
        """Add an `InstanceGroup` to the `FrameGroup`."""

        # Handle underlying dictionary updates
        ...

    def remove_instance_group(self, instance_group: InstanceGroup):
        """Remove an `InstanceGroup` from the `FrameGroup`."""

        # Handle underlying dictionary updates
        ...

    def to_dict(
        self,
        labeled_frame_to_idx: dict[LabeledFrame, int],
    ) -> dict[str, int | list[dict[str]]]:
        """Convert `FrameGroup` to a dictionary.

        Args:
            labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
                `Labels.labeled_frames`.
        """

        # Create dictionary of `Instance` to `LabeledFrame` index (in
        # `Labels.labeled_frames`) and `Instance` index in `LabeledFrame.instances``.
        instance_to_lf_and_inst_idx: dict[Instance, tuple[str, str]] = {
            inst: (str(labeled_frame_to_idx[labeled_frame]), str(inst_idx))
            for labeled_frame in self.labeled_frames
            for inst_idx, inst in enumerate(labeled_frame.instances)
        }

        frame_group_dict = {
            "instance_groups": [
                instance_group.to_dict(
                    instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx,
                )
                for instance_group in self._instance_groups
            ],
        }

        return frame_group_dict

    @classmethod
    def from_dict(
        cls,
        frame_group_dict: dict[str],
        labeled_frames_list: list[LabeledFrame],
    ):
        """Convert dictionary to `FrameGroup` object.

        Args:
            frame_group_dict: Dictionary of `FrameGroup` object.
            labeled_frames_list: List of `LabeledFrame` objects (expecting
                `Labels.labeled_frames`).

        Returns:
            `FrameGroup` object.
        """

        # Get `InstanceGroup` objects
        instance_groups = []
        for instance_group_dict in frame_group_dict["instance_groups"]:
            instance_group = InstanceGroup.from_dict(
                instance_group_dict=instance_group_dict,
                labeled_frames_list=labeled_frames_list,
            )
            instance_groups.append(instance_group)

        return cls(instance_groups=instance_groups)
