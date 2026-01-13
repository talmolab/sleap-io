"""Transform module for coordinate-aware video transformations.

This module provides geometric transformations for videos and label coordinates:
- Cropping: Extract rectangular regions with coordinate offset
- Scaling: Resize by ratio or to pixel dimensions
- Rotation: Rotate around frame center with coordinate transformation
- Padding: Add borders with coordinate offset

All transformations automatically adjust landmark coordinates in SLEAP labels
to maintain alignment with the transformed video.

Example:
    >>> import sleap_io as sio
    >>> from pathlib import Path
    >>> labels = sio.load_slp("predictions.slp")
    >>> transform = sio.Transform(crop=(100, 100, 500, 500), scale=(0.5, 0.5))
    >>> result = sio.transform_labels(labels, transform, Path("output.slp"))
"""

from sleap_io.transform.core import Transform
from sleap_io.transform.video import transform_labels, transform_video

__all__ = ["Transform", "transform_labels", "transform_video"]
