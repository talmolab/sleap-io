"""Frame transformation functions using PIL.

This module provides frame-level transformation operations for cropping,
scaling, rotating, and padding video frames.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Map quality string to PIL resampling filter
QUALITY_TO_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def _to_pil_compatible(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert frame to PIL-compatible format.

    PIL requires grayscale images to have shape (H, W), not (H, W, 1).

    Args:
        frame: Input frame array.

    Returns:
        Tuple of (pil_compatible_array, was_squeezed).
    """
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.squeeze(frame, axis=2), True
    return frame, False


def _from_pil_result(result: np.ndarray, was_squeezed: bool) -> np.ndarray:
    """Convert PIL result back to original format.

    Args:
        result: Array from PIL conversion.
        was_squeezed: Whether the original had a squeezed channel.

    Returns:
        Array with original channel format restored.
    """
    if was_squeezed and result.ndim == 2:
        return np.expand_dims(result, axis=2)
    return result


def crop_frame(
    frame: np.ndarray,
    crop: tuple[int, int, int, int],
    fill: tuple[int, ...] | int = 0,
) -> np.ndarray:
    """Crop a frame to the specified region.

    If the crop region extends beyond the frame bounds, the out-of-bounds area
    is filled with the fill value.

    Args:
        frame: Input frame as numpy array with shape (H, W) or (H, W, C).
        crop: Crop region as (x1, y1, x2, y2) pixel coordinates.
        fill: Fill value for out-of-bounds regions.

    Returns:
        Cropped frame as numpy array.
    """
    x1, y1, x2, y2 = crop
    h, w = frame.shape[:2]
    crop_w, crop_h = x2 - x1, y2 - y1

    # Compute valid source region
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    # Extract source region
    cropped = frame[src_y1:src_y2, src_x1:src_x2]

    # Check if padding is needed
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        # Create output array with fill value
        if frame.ndim == 3:
            output_shape = (crop_h, crop_w, frame.shape[2])
        else:
            output_shape = (crop_h, crop_w)

        output = np.full(output_shape, fill, dtype=frame.dtype)

        # Compute paste region
        paste_x1 = src_x1 - x1
        paste_y1 = src_y1 - y1
        paste_x2 = paste_x1 + (src_x2 - src_x1)
        paste_y2 = paste_y1 + (src_y2 - src_y1)

        output[paste_y1:paste_y2, paste_x1:paste_x2] = cropped
        return output

    return cropped


def scale_frame(
    frame: np.ndarray,
    scale: tuple[float, float],
    quality: str = "bilinear",
) -> np.ndarray:
    """Scale a frame by the given factors.

    Args:
        frame: Input frame as numpy array with shape (H, W) or (H, W, C).
        scale: Scale factors as (scale_x, scale_y).
        quality: Interpolation quality. One of "nearest", "bilinear", "bicubic".

    Returns:
        Scaled frame as numpy array.
    """
    scale_x, scale_y = scale

    if scale_x == 1.0 and scale_y == 1.0:
        return frame

    h, w = frame.shape[:2]
    new_w = int(round(w * scale_x))
    new_h = int(round(h * scale_y))

    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"Invalid output dimensions: {new_w}x{new_h}")

    resample = QUALITY_TO_RESAMPLE.get(quality, Image.Resampling.BILINEAR)

    # Handle grayscale images with shape (H, W, 1)
    pil_frame, was_squeezed = _to_pil_compatible(frame)
    pil_img = Image.fromarray(pil_frame)
    pil_img = pil_img.resize((new_w, new_h), resample)
    result = np.array(pil_img)
    return _from_pil_result(result, was_squeezed)


def rotate_frame(
    frame: np.ndarray,
    angle: float,
    quality: str = "bilinear",
    fill: tuple[int, ...] | int = 0,
    expand: bool = True,
) -> np.ndarray:
    """Rotate a frame by the given angle.

    The frame is rotated about its center. Positive angles rotate clockwise.

    Args:
        frame: Input frame as numpy array with shape (H, W) or (H, W, C).
        angle: Rotation angle in degrees. Positive is clockwise.
        quality: Interpolation quality. One of "nearest", "bilinear", "bicubic".
        fill: Fill value for areas outside the rotated image.
        expand: If True (default), expand canvas to fit entire rotated image.
            If False, keep original dimensions (clips corners).

    Returns:
        Rotated frame as numpy array. If expand=True, dimensions may change.
        If expand=False, dimensions match input.
    """
    if angle == 0:
        return frame

    resample = QUALITY_TO_RESAMPLE.get(quality, Image.Resampling.BILINEAR)

    # Convert fill to tuple if needed
    if isinstance(fill, int):
        if frame.ndim == 3:
            fill_color = (fill,) * frame.shape[2]
        else:
            fill_color = fill
    else:
        fill_color = fill

    # Handle grayscale images with shape (H, W, 1)
    pil_frame, was_squeezed = _to_pil_compatible(frame)
    pil_img = Image.fromarray(pil_frame)
    # PIL rotates counter-clockwise, so negate angle for clockwise
    pil_img = pil_img.rotate(
        -angle, resample=resample, expand=expand, fillcolor=fill_color
    )
    result = np.array(pil_img)
    return _from_pil_result(result, was_squeezed)


def flip_h_frame(frame: np.ndarray) -> np.ndarray:
    """Flip a frame horizontally (mirror left-right).

    Args:
        frame: Input frame as numpy array with shape (H, W) or (H, W, C).

    Returns:
        Horizontally flipped frame as numpy array with same dimensions.
    """
    return np.fliplr(frame).copy()


def flip_v_frame(frame: np.ndarray) -> np.ndarray:
    """Flip a frame vertically (mirror top-bottom).

    Args:
        frame: Input frame as numpy array with shape (H, W) or (H, W, C).

    Returns:
        Vertically flipped frame as numpy array with same dimensions.
    """
    return np.flipud(frame).copy()


def pad_frame(
    frame: np.ndarray,
    padding: tuple[int, int, int, int],
    fill: tuple[int, ...] | int = 0,
) -> np.ndarray:
    """Add padding around a frame.

    Args:
        frame: Input frame as numpy array with shape (H, W) or (H, W, C).
        padding: Padding as (top, right, bottom, left) in pixels.
        fill: Fill value for padded regions.

    Returns:
        Padded frame as numpy array.
    """
    top, right, bottom, left = padding

    if top == 0 and right == 0 and bottom == 0 and left == 0:
        return frame

    h, w = frame.shape[:2]
    new_h = h + top + bottom
    new_w = w + left + right

    if frame.ndim == 3:
        output_shape = (new_h, new_w, frame.shape[2])
    else:
        output_shape = (new_h, new_w)

    output = np.full(output_shape, fill, dtype=frame.dtype)
    output[top : top + h, left : left + w] = frame

    return output


def transform_frame(
    frame: np.ndarray,
    crop: tuple[int, int, int, int] | None = None,
    scale: tuple[float, float] | None = None,
    rotate: float | None = None,
    pad: tuple[int, int, int, int] | None = None,
    quality: str = "bilinear",
    fill: tuple[int, ...] | int = 0,
    expand_rotation: bool = True,
    flip_h: bool = False,
    flip_v: bool = False,
) -> np.ndarray:
    """Apply a sequence of transformations to a frame.

    Transforms are applied in order: crop -> scale -> rotate -> pad -> flip.

    Args:
        frame: Input frame as numpy array with shape (H, W) or (H, W, C).
        crop: Crop region as (x1, y1, x2, y2) pixel coordinates.
        scale: Scale factors as (scale_x, scale_y).
        rotate: Rotation angle in degrees. Positive is clockwise.
        pad: Padding as (top, right, bottom, left) in pixels.
        quality: Interpolation quality. One of "nearest", "bilinear", "bicubic".
        fill: Fill value for out-of-bounds and padded regions.
        expand_rotation: If True (default), expand canvas to fit rotated image.
            If False, keep original dimensions (clips corners).
        flip_h: If True, flip horizontally (mirror left-right).
        flip_v: If True, flip vertically (mirror top-bottom).

    Returns:
        Transformed frame as numpy array.
    """
    result = frame

    if crop is not None:
        result = crop_frame(result, crop, fill=fill)

    if scale is not None:
        result = scale_frame(result, scale, quality=quality)

    if rotate is not None and rotate != 0:
        result = rotate_frame(
            result, rotate, quality=quality, fill=fill, expand=expand_rotation
        )

    if pad is not None:
        result = pad_frame(result, pad, fill=fill)

    if flip_h:
        result = flip_h_frame(result)

    if flip_v:
        result = flip_v_frame(result)

    return result
