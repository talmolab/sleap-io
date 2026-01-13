"""Point coordinate transformation functions.

This module provides functions for transforming landmark coordinates to match
transformed video frames. Each function corresponds to a geometric transformation
(crop, scale, rotate, pad) and adjusts coordinates accordingly.
"""

from __future__ import annotations

import numpy as np


def crop_points(
    points: np.ndarray,
    crop: tuple[int, int, int, int],
) -> np.ndarray:
    """Adjust point coordinates for a crop transformation.

    Args:
        points: Coordinate array of shape (..., 2) where the last dimension
            contains (x, y) coordinates. NaN values are preserved.
        crop: Crop region as (x1, y1, x2, y2) pixel coordinates.

    Returns:
        Adjusted coordinates with same shape as input.
    """
    x1, y1, x2, y2 = crop
    result = points.copy()
    result[..., 0] = points[..., 0] - x1
    result[..., 1] = points[..., 1] - y1
    return result


def scale_points(
    points: np.ndarray,
    scale: tuple[float, float],
) -> np.ndarray:
    """Adjust point coordinates for a scale transformation.

    Args:
        points: Coordinate array of shape (..., 2) where the last dimension
            contains (x, y) coordinates. NaN values are preserved.
        scale: Scale factors as (scale_x, scale_y).

    Returns:
        Adjusted coordinates with same shape as input.
    """
    scale_x, scale_y = scale
    result = points.copy()
    result[..., 0] = points[..., 0] * scale_x
    result[..., 1] = points[..., 1] * scale_y
    return result


def rotate_points(
    points: np.ndarray,
    angle: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Adjust point coordinates for a rotation transformation.

    Points are rotated clockwise about the specified center.

    Args:
        points: Coordinate array of shape (..., 2) where the last dimension
            contains (x, y) coordinates. NaN values are preserved.
        angle: Rotation angle in degrees. Positive is clockwise.
        center: Center of rotation as (cx, cy).

    Returns:
        Adjusted coordinates with same shape as input.
    """
    if angle == 0:
        return points.copy()

    cx, cy = center
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Translate to origin
    dx = points[..., 0] - cx
    dy = points[..., 1] - cy

    # Rotate (clockwise: positive angle)
    # x' = x*cos(a) + y*sin(a)
    # y' = -x*sin(a) + y*cos(a)
    result = points.copy()
    result[..., 0] = dx * cos_a + dy * sin_a + cx
    result[..., 1] = -dx * sin_a + dy * cos_a + cy

    return result


def pad_points(
    points: np.ndarray,
    padding: tuple[int, int, int, int],
) -> np.ndarray:
    """Adjust point coordinates for a pad transformation.

    Args:
        points: Coordinate array of shape (..., 2) where the last dimension
            contains (x, y) coordinates. NaN values are preserved.
        padding: Padding as (top, right, bottom, left) in pixels.

    Returns:
        Adjusted coordinates with same shape as input.
    """
    top, right, bottom, left = padding
    result = points.copy()
    result[..., 0] = points[..., 0] + left
    result[..., 1] = points[..., 1] + top
    return result


def transform_points(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Transform point coordinates using an affine matrix.

    Args:
        points: Coordinate array of shape (n_points, 2) where each row is (x, y).
            NaN values are preserved.
        matrix: 3x3 affine transformation matrix.

    Returns:
        Transformed coordinates with same shape as input.
    """
    if points.size == 0:
        return points.copy()

    result = points.copy()

    # Create mask for valid (non-NaN) points
    valid_mask = ~np.isnan(points).any(axis=-1)

    if valid_mask.any():
        # Convert to homogeneous coordinates
        valid_points = points[valid_mask]
        ones = np.ones((valid_points.shape[0], 1), dtype=np.float64)
        homogeneous = np.hstack([valid_points, ones])

        # Apply transformation
        transformed = (matrix @ homogeneous.T).T

        # Extract x, y from homogeneous coordinates
        result[valid_mask, 0] = transformed[:, 0]
        result[valid_mask, 1] = transformed[:, 1]

    return result


def count_out_of_bounds(
    points: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> int:
    """Count how many points fall outside the given bounds.

    Args:
        points: Coordinate array of shape (..., 2) where the last dimension
            contains (x, y) coordinates.
        bounds: Bounds as (x_min, y_min, x_max, y_max).

    Returns:
        Number of points outside bounds (excluding NaN points).
    """
    x_min, y_min, x_max, y_max = bounds

    # Flatten to (n, 2)
    flat_points = points.reshape(-1, 2)

    # Mask for valid (non-NaN) points
    valid_mask = ~np.isnan(flat_points).any(axis=-1)

    if not valid_mask.any():
        return 0

    valid_points = flat_points[valid_mask]

    out_of_bounds = (
        (valid_points[:, 0] < x_min)
        | (valid_points[:, 0] >= x_max)
        | (valid_points[:, 1] < y_min)
        | (valid_points[:, 1] >= y_max)
    )

    return int(out_of_bounds.sum())


def get_out_of_bounds_mask(
    points: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Get a boolean mask indicating which points are outside bounds.

    Args:
        points: Coordinate array of shape (n_points, 2) where each row is (x, y).
        bounds: Bounds as (x_min, y_min, x_max, y_max).

    Returns:
        Boolean array of shape (n_points,) where True indicates the point is
        out of bounds. NaN points are considered in bounds (not marked as OOB).
    """
    x_min, y_min, x_max, y_max = bounds

    # Mask for valid (non-NaN) points
    valid_mask = ~np.isnan(points).any(axis=-1)

    # Initialize result - NaN points are not considered OOB
    oob_mask = np.zeros(len(points), dtype=bool)

    if valid_mask.any():
        valid_points = points[valid_mask]
        oob = (
            (valid_points[:, 0] < x_min)
            | (valid_points[:, 0] >= x_max)
            | (valid_points[:, 1] < y_min)
            | (valid_points[:, 1] >= y_max)
        )
        oob_mask[valid_mask] = oob

    return oob_mask
