"""Rendering module for visualizing pose data using skia-python.

This module provides high-performance pose rendering with:
- Multiple color schemes (by track, instance, or node type)
- Various marker shapes (circle, square, diamond, triangle, cross)
- Configurable aesthetics (size, width, alpha)
- Custom rendering callbacks for overlays
- Video export with optional scaling

Example:
    >>> import sleap_io as sio
    >>> labels = sio.load_slp("predictions.slp")
    >>> sio.render_video(labels, "output.mp4")
    >>> sio.render_image(labels.labeled_frames[0], "frame.png")
"""

from sleap_io.rendering.callbacks import InstanceContext, RenderContext
from sleap_io.rendering.colors import (
    NAMED_COLORS,
    ColorSpec,
    get_palette,
    resolve_color,
)
from sleap_io.rendering.core import render_image, render_video

__all__ = [
    "render_video",
    "render_image",
    "get_palette",
    "resolve_color",
    "ColorSpec",
    "NAMED_COLORS",
    "RenderContext",
    "InstanceContext",
]
