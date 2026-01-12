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
    >>> from sleap_io.transform import Transform
    >>> labels = sio.load_slp("predictions.slp")
    >>> transform = Transform(crop=(100, 100, 500, 500), scale=(0.5, 0.5))
    >>> transformed = transform.apply_to_labels(labels, output_dir="output/")
"""

from sleap_io.transform.core import Transform

__all__ = ["Transform"]
