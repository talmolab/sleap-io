"""Data model for embedded preview frames in SLP files."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import attrs
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from sleap_io.model.video import Video

PREVIEW_FORMAT_VERSION = 1
DEFAULT_PREVIEW_MAX_SIZE = 768
DEFAULT_PREVIEW_QUALITY = 80


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image array to contiguous uint8 ``(H, W)`` or ``(H, W, 3)``."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[-1] != 3):
        raise ValueError(
            f"Expected 2D or 3D image with 1, 3, or 4 channels, got shape {arr.shape}."
        )

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.nanmax(arr)) if arr.size else 0.0
            if max_val <= 1.0:
                arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
            else:
                arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.integer):
            max_val = int(arr.max()) if arr.size else 0
            if max_val > 255 and arr.dtype == np.uint16:
                arr = (arr >> 8).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # Collapse identical RGB channels to grayscale for a smaller JPEG.
    if (
        arr.ndim == 3
        and arr.shape[-1] == 3
        and np.array_equal(arr[..., 0], arr[..., 1])
        and np.array_equal(arr[..., 0], arr[..., 2])
    ):
        arr = arr[..., 0]

    return np.ascontiguousarray(arr)


@attrs.define
class PreviewFrame:
    """A downscaled JPEG preview frame stored in an SLP file's ``/preview`` group.

    Preview frames allow external tools (such as macOS Finder Quick Look) to
    display a real video frame underneath pose overlays for linked ``.slp``
    files without requiring access to the external video file.

    Attributes:
        video: The `Video` this preview frame belongs to.
        frame_idx: 0-based frame index within `video`.
        orig_width: Full original video frame width in pixels (pose coordinate space).
        orig_height: Full original video frame height in pixels (pose coordinate space).
        width: Encoded preview image width in pixels (``<= orig_width``).
        height: Encoded preview image height in pixels (``<= orig_height``).
        data: Baseline JPEG-encoded image bytes (RGB or grayscale).
    """

    video: "Video"
    frame_idx: int
    orig_width: int
    orig_height: int
    width: int
    height: int
    data: bytes = attrs.field(repr=False)

    @classmethod
    def from_image(
        cls,
        video: "Video",
        frame_idx: int,
        image: np.ndarray,
        *,
        max_size: int = DEFAULT_PREVIEW_MAX_SIZE,
        quality: int = DEFAULT_PREVIEW_QUALITY,
    ) -> "PreviewFrame":
        """Create a `PreviewFrame` by downscaling and JPEG-encoding a raw frame.

        Args:
            video: The `Video` the frame came from.
            frame_idx: 0-based frame index in `video`.
            image: Raw frame array of shape ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)``,
                or ``(H, W, 4)``.
            max_size: Maximum long-edge dimension in pixels. Frames already within
                this bound are kept at native resolution (never upscaled).
            quality: Baseline JPEG quality (1-100, default 80).

        Returns:
            A `PreviewFrame` holding the encoded JPEG bytes and dimensions.
        """
        arr = _to_uint8_image(image)
        orig_height, orig_width = int(arr.shape[0]), int(arr.shape[1])
        mode = "L" if arr.ndim == 2 else "RGB"
        pil_img = Image.fromarray(arr, mode=mode)

        if max_size > 0 and max(orig_width, orig_height) > max_size:
            scale = max_size / float(max(orig_width, orig_height))
            new_w = max(1, int(round(orig_width * scale)))
            new_h = max(1, int(round(orig_height * scale)))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = BytesIO()
        pil_img.save(
            buf,
            format="JPEG",
            quality=int(np.clip(quality, 1, 100)),
            optimize=True,
            progressive=False,
        )
        return cls(
            video=video,
            frame_idx=int(frame_idx),
            orig_width=orig_width,
            orig_height=orig_height,
            width=int(pil_img.width),
            height=int(pil_img.height),
            data=buf.getvalue(),
        )

    def to_image(self) -> np.ndarray:
        """Decode the stored JPEG bytes back into a uint8 numpy image array."""
        with Image.open(BytesIO(self.data)) as img:
            return np.asarray(img)
