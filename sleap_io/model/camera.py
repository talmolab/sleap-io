"""Data structure for a single camera view in a multi-camera setup."""

from __future__ import annotations

import attrs
import cv2
import numpy as np
from attrs import define, field


@define
class Camera:
    """A camera used to record in a multi-view `RecordingSession`.

    Attributes:
        matrix: Intrinsic camera matrix of size (3, 3) and type float64.
        dist: Radial-tangential distortion coefficients [k_1, k_2, p_1, p_2, k_3] of
            size (5,) and type float64.
        size: Image size of camera in pixels of size (2,) and type int.
        rvec: Rotation vector in unnormalized axis-angle representation of size (3,) and
            type float64.
        tvec: Translation vector of size (3,) and type float64.
        extrinsic_matrix: Extrinsic matrix of camera of size (4, 4) and type float64.
        name: Camera name.
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
                f"but recieved shape: {np.shape(value)} and type: {type(value)} for "
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
        out = cv2.undistortPoints(points, self.matrix, self.dist)
        return out.reshape(shape)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using camera matrix and distortion coefficients.

        Args:
            points: 3D points to project of shape (N, 3) or (N, 1, 3).

        Returns:
            Projected 2D points of shape (N, 1, 2).
        """

        points = points.reshape(-1, 1, 3)
        out, _ = cv2.projectPoints(
            points,
            self.rvec,
            self.tvec,
            self.matrix,
            self.dist,
        )
        return out

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
