"""Core Transform class for composable geometric transformations.

This module defines the Transform dataclass which represents a composable
geometric transformation that can be applied to both video frames and
landmark coordinates.
"""

from __future__ import annotations

import attrs
import numpy as np


@attrs.define
class Transform:
    """Composable geometric transformation for video frames and coordinates.

    Transforms are applied in a fixed pipeline order:
    crop -> scale -> rotate -> pad -> flip.
    This ensures consistent and predictable behavior when combining transforms.

    Attributes:
        crop: Crop region as (x1, y1, x2, y2) pixel coordinates. The region from
            (x1, y1) to (x2, y2) is extracted, where (x2, y2) is exclusive.
        scale: Scale factors as (scale_x, scale_y). Use (0.5, 0.5) for 50% size.
            Can also be specified as a single float for uniform scaling.
        rotate: Rotation angle in degrees. Positive values rotate clockwise.
        pad: Padding as (top, right, bottom, left) in pixels.
        quality: Interpolation quality for frame transforms. One of "nearest",
            "bilinear", or "bicubic".
        fill: Fill value for out-of-bounds regions. Can be a single int for
            grayscale or (R, G, B) tuple for color.
        clip_rotation: If True, rotation clips to original dimensions. If False
            (default), canvas expands to fit the entire rotated image.
        flip_h: If True, flip horizontally (mirror left-right).
        flip_v: If True, flip vertically (mirror top-bottom).
    """

    crop: tuple[int, int, int, int] | None = None
    scale: tuple[float, float] | None = None
    rotate: float | None = None
    pad: tuple[int, int, int, int] | None = None
    quality: str = "bilinear"
    fill: tuple[int, ...] | int = 0
    clip_rotation: bool = False
    flip_h: bool = False
    flip_v: bool = False

    def _rotation_output_size(
        self, width: int, height: int
    ) -> tuple[int, int, float, float]:
        """Compute output size and center offset for rotation.

        Args:
            width: Pre-rotation width.
            height: Pre-rotation height.

        Returns:
            Tuple of (new_width, new_height, offset_x, offset_y) where offsets
            are the translation needed to center the rotated content.
        """
        if self.rotate is None or self.rotate == 0 or self.clip_rotation:
            return (width, height, 0.0, 0.0)

        angle_rad = np.radians(abs(self.rotate))
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))

        # New bounding box dimensions
        new_width = int(np.ceil(width * cos_a + height * sin_a))
        new_height = int(np.ceil(width * sin_a + height * cos_a))

        # Offset to center the rotated image in the new canvas
        offset_x = (new_width - width) / 2
        offset_y = (new_height - height) / 2

        return (new_width, new_height, offset_x, offset_y)

    def output_size(self, input_size: tuple[int, int]) -> tuple[int, int]:
        """Compute output dimensions after applying the transformation.

        Args:
            input_size: Input (width, height) in pixels.

        Returns:
            Output (width, height) in pixels.
        """
        width, height = input_size

        # Apply crop
        if self.crop is not None:
            x1, y1, x2, y2 = self.crop
            width = x2 - x1
            height = y2 - y1

        # Apply scale
        if self.scale is not None:
            scale_x, scale_y = self.scale
            width = int(round(width * scale_x))
            height = int(round(height * scale_y))

        # Apply rotation (may expand canvas if not clipping)
        if self.rotate is not None and self.rotate != 0:
            width, height, _, _ = self._rotation_output_size(width, height)

        # Apply pad
        if self.pad is not None:
            top, right, bottom, left = self.pad
            width = width + left + right
            height = height + top + bottom

        return (width, height)

    def to_matrix(self, input_size: tuple[int, int]) -> np.ndarray:
        """Compute combined 3x3 affine transformation matrix.

        The transformation matrix can be used to transform homogeneous coordinates:
            [new_x]   [a  b  tx] [old_x]
            [new_y] = [c  d  ty] [old_y]
            [  1  ]   [0  0   1] [  1  ]

        Args:
            input_size: Input (width, height) in pixels.

        Returns:
            3x3 affine transformation matrix as numpy array.
        """
        # Start with identity matrix
        matrix = np.eye(3, dtype=np.float64)

        width, height = input_size

        # Apply crop (translate by negative crop origin)
        if self.crop is not None:
            x1, y1, x2, y2 = self.crop
            crop_matrix = np.array(
                [[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float64
            )
            matrix = crop_matrix @ matrix
            width = x2 - x1
            height = y2 - y1

        # Apply scale
        if self.scale is not None:
            scale_x, scale_y = self.scale
            scale_matrix = np.array(
                [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64
            )
            matrix = scale_matrix @ matrix
            width = int(round(width * scale_x))
            height = int(round(height * scale_y))

        # Apply rotation (about center of current frame)
        # Note: SLEAP uses center pixel indexing where (0, 0) is the center of the
        # top-left pixel. The geometric center of an image is at ((w-1)/2, (h-1)/2).
        if self.rotate is not None and self.rotate != 0:
            angle_rad = np.radians(self.rotate)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            cx, cy = (width - 1) / 2, (height - 1) / 2

            # Get rotation output size (may expand if not clipping)
            new_width, new_height, offset_x, offset_y = self._rotation_output_size(
                width, height
            )

            # Rotation about center, then translate to new center if expanded
            # Combined: T(new_cx, new_cy) @ R @ T(-cx, -cy)
            # Note: Image coordinates use y-down, so clockwise rotation matrix is:
            #   [cos, -sin]
            #   [sin,  cos]
            new_cx = (new_width - 1) / 2
            new_cy = (new_height - 1) / 2

            rotate_matrix = np.array(
                [
                    [cos_a, -sin_a, new_cx - cos_a * cx + sin_a * cy],
                    [sin_a, cos_a, new_cy - sin_a * cx - cos_a * cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            matrix = rotate_matrix @ matrix
            width, height = new_width, new_height

        # Apply pad (translate by padding offset)
        if self.pad is not None:
            top, right, bottom, left = self.pad
            pad_matrix = np.array(
                [[1, 0, left], [0, 1, top], [0, 0, 1]], dtype=np.float64
            )
            matrix = pad_matrix @ matrix
            width = width + left + right
            height = height + top + bottom

        # Apply horizontal flip (x -> (width - 1) - x)
        # With center pixel indexing, pixel centers range from 0 to width-1,
        # so we flip around (width-1)/2 by mapping x -> (width-1) - x.
        if self.flip_h:
            flip_h_matrix = np.array(
                [[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64
            )
            matrix = flip_h_matrix @ matrix

        # Apply vertical flip (y -> (height - 1) - y)
        # With center pixel indexing, pixel centers range from 0 to height-1,
        # so we flip around (height-1)/2 by mapping y -> (height-1) - y.
        if self.flip_v:
            flip_v_matrix = np.array(
                [[1, 0, 0], [0, -1, height - 1], [0, 0, 1]], dtype=np.float64
            )
            matrix = flip_v_matrix @ matrix

        return matrix

    def apply_to_points(
        self, points: np.ndarray, input_size: tuple[int, int]
    ) -> np.ndarray:
        """Transform landmark coordinates.

        Args:
            points: Coordinate array of shape (n_points, 2) or (n_points, D) where
                the first two columns are (x, y) coordinates. NaN values are preserved.
            input_size: Input (width, height) in pixels.

        Returns:
            Transformed coordinates with same shape as input.
        """
        if points.size == 0:
            return points.copy()

        # Handle both (n, 2) and (n, D) arrays
        xy = points[..., :2].copy()
        result = points.copy()

        # Get transformation matrix
        matrix = self.to_matrix(input_size)

        # Create mask for valid (non-NaN) points
        valid_mask = ~np.isnan(xy).any(axis=-1)

        if valid_mask.any():
            # Convert to homogeneous coordinates
            valid_xy = xy[valid_mask]
            ones = np.ones((valid_xy.shape[0], 1), dtype=np.float64)
            homogeneous = np.hstack([valid_xy, ones])

            # Apply transformation
            transformed = (matrix @ homogeneous.T).T

            # Extract x, y from homogeneous coordinates
            result[valid_mask, 0] = transformed[:, 0]
            result[valid_mask, 1] = transformed[:, 1]

        return result

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Transform a video frame.

        Args:
            frame: Input frame as numpy array with shape (H, W) or (H, W, C).

        Returns:
            Transformed frame as numpy array.
        """
        from sleap_io.transform.frame import transform_frame

        return transform_frame(
            frame,
            crop=self.crop,
            scale=self.scale,
            rotate=self.rotate,
            pad=self.pad,
            quality=self.quality,
            fill=self.fill,
            expand_rotation=not self.clip_rotation,
            flip_h=self.flip_h,
            flip_v=self.flip_v,
        )

    def __bool__(self) -> bool:
        """Return True if any transformation is defined."""
        return any(
            [
                self.crop is not None,
                self.scale is not None,
                self.rotate is not None and self.rotate != 0,
                self.pad is not None and any(p != 0 for p in self.pad),
                self.flip_h,
                self.flip_v,
            ]
        )


def parse_scale(value: str) -> tuple[float, float]:
    """Parse scale value from CLI string.

    Args:
        value: Scale string in one of these formats:
            - "0.5" -> uniform 50% scale
            - "640" -> width=640, height auto (aspect preserved)
            - "640,-1" -> width=640, height auto
            - "-1,480" -> height=480, width auto
            - "640,480" -> exact dimensions
            - "0.5,0.75" -> different ratios per axis

    Returns:
        Tuple of (scale_x, scale_y) factors. Returns None for auto-compute
        dimensions which must be resolved with input frame size.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    parts = value.split(",")

    if len(parts) == 1:
        # Single value: ratio or target width
        val = float(parts[0])
        if "." in parts[0] or val < 1:
            # Ratio
            return (val, val)
        else:
            # Target width (will need input size to compute ratio)
            # Return as negative to indicate pixel mode
            return (-val, -1.0)

    elif len(parts) == 2:
        val1 = float(parts[0])
        val2 = float(parts[1])

        # Check if both are ratios
        if ("." in parts[0] or val1 < 1) and ("." in parts[1] or val2 < 1 or val2 < 0):
            if val1 < 0:
                val1 = -1.0  # auto
            if val2 < 0:
                val2 = -1.0  # auto
            return (val1, val2)
        else:
            # Pixel dimensions (negative indicates pixel mode)
            return (-val1 if val1 > 0 else val1, -val2 if val2 > 0 else val2)
    else:
        raise ValueError(f"Invalid scale format: {value}")


def resolve_scale(
    scale: tuple[float, float], input_size: tuple[int, int]
) -> tuple[float, float]:
    """Resolve scale factors from parsed scale value.

    Args:
        scale: Scale tuple from parse_scale(). Negative values indicate pixel
            dimensions, -1 indicates auto-compute.
        input_size: Input (width, height) in pixels.

    Returns:
        Tuple of (scale_x, scale_y) as positive float ratios.
    """
    scale_x, scale_y = scale
    width, height = input_size

    # Both are positive ratios - return as-is
    if scale_x > 0 and scale_y > 0:
        return (scale_x, scale_y)

    # Convert negative pixel values to ratios
    target_w = -scale_x if scale_x < 0 and scale_x != -1 else None
    target_h = -scale_y if scale_y < 0 and scale_y != -1 else None

    if target_w is not None and target_h is not None:
        # Both dimensions specified
        return (target_w / width, target_h / height)
    elif target_w is not None:
        # Width specified, auto height
        ratio = target_w / width
        return (ratio, ratio)
    elif target_h is not None:
        # Height specified, auto width
        ratio = target_h / height
        return (ratio, ratio)
    else:
        # Both auto - this shouldn't happen in valid input
        return (1.0, 1.0)


def parse_crop(
    value: str, input_size: tuple[int, int] | None = None
) -> tuple[int, ...]:
    """Parse crop value from CLI string.

    Args:
        value: Crop string in format "x1,y1,x2,y2". Values can be:
            - Integers: pixel coordinates
            - Floats in [0.0, 1.0]: normalized coordinates
        input_size: Input (width, height) for resolving normalized coordinates.
            Required if using normalized values.

    Returns:
        Tuple of (x1, y1, x2, y2) as integer pixel coordinates.

    Raises:
        ValueError: If the value cannot be parsed or normalized coords used
            without input_size.
    """
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(f"Crop must have 4 values (x1,y1,x2,y2), got: {value}")

    values = [float(p) for p in parts]

    # Check if normalized (all floats in [0, 1])
    is_normalized = all("." in p for p in parts) and all(0 <= v <= 1 for v in values)

    if is_normalized:
        if input_size is None:
            raise ValueError("input_size required for normalized crop coordinates")
        width, height = input_size
        x1 = int(round(values[0] * width))
        y1 = int(round(values[1] * height))
        x2 = int(round(values[2] * width))
        y2 = int(round(values[3] * height))
    else:
        x1, y1, x2, y2 = [int(v) for v in values]

    return (x1, y1, x2, y2)


def parse_pad(value: str) -> tuple[int, int, int, int]:
    """Parse padding value from CLI string.

    Args:
        value: Padding string in format "top,right,bottom,left" or single value
            for uniform padding.

    Returns:
        Tuple of (top, right, bottom, left) as integers.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    parts = value.split(",")

    if len(parts) == 1:
        val = int(parts[0])
        return (val, val, val, val)
    elif len(parts) == 4:
        return tuple(int(p) for p in parts)  # type: ignore
    else:
        raise ValueError(f"Padding must have 1 or 4 values, got: {value}")
