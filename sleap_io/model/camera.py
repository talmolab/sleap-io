"""Data structure for a single camera view in a multi-camera setup."""

from __future__ import annotations

import attrs
import numpy as np
from attrs import define, field


@define
class Camera:
    """A camera used to record in a multi-view `RecordingSession`.

    Attributes:
        matrix: Intrinsic camera matrix of size 3 x 3.
        dist: Radial-tangential distortion coefficients [k_1, k_2, p_1, p_2, k_3].
        size: Image size.
        rvec: Rotation vector in unnormalized axis-angle representation of size 3.
        tvec: Translation vector of size 3.
        name: Camera name.
    """

    matrix: np.ndarray = field(
        default=np.eye(3),
        converter=np.array,
    )
    dist: np.ndarray = field(
        default=np.zeros(5), converter=lambda x: np.array(x).ravel()
    )
    size: tuple[int, int] = field(
        default=None, converter=attrs.converters.optional(tuple)
    )
    rvec: np.ndarray = field(
        default=np.zeros(3), converter=lambda x: np.array(x).ravel()
    )
    tvec: np.ndarray = field(
        default=np.zeros(3), converter=lambda x: np.array(x).ravel()
    )
    name: str = field(default=None, converter=attrs.converters.optional(str))

    @matrix.validator
    @dist.validator
    @size.validator
    @rvec.validator
    @tvec.validator
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
            "rvec": {"shape": (3,), "type": np.ndarray},
            "tvec": {"shape": (3,), "type": np.ndarray},
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
                f"{attribute.name} must be a {expected_type} of size {expected_shape}"
            )
