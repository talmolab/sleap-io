"""Data structures for data contained within a single video frame.

The `LabeledFrame` class is a data structure that contains `Instance`s and
`PredictedInstance`s that are associated with a single frame within a video.
"""

from __future__ import annotations
from sleap_io import Instance, PredictedInstance, Video
from attrs import define, field
from typing import Union
import numpy as np


@define(auto_attribs=True)
class LabeledFrame:
    """Labeled data for a single frame of a video.

    Args:
        video: The :class:`Video` associated with this `LabeledFrame`.
        frame_idx: The index of the `LabeledFrame` in the `Video`.
        instances: List of `Instance` objects associated with this `LabeledFrame`.
    """

    video: Video
    frame_idx: int
    instances: list[Union[Instance, PredictedInstance]] = field(factory=list)

    def __len__(self) -> int:
        """Return the number of instances in the frame."""
        return len(self.instances)

    def __getitem__(self, key: int) -> Union[Instance, PredictedInstance]:
        """Return the `Instance` at `key` index in the `instances` list."""
        return self.instances[key]

    def __iter__(self):
        """Iterate over `Instance`s in `instances` list."""
        return iter(self.instances)

    @property
    def user_instances(self) -> list[Instance]:
        """Frame instances that are user-labeled (`Instance` objects)."""
        return [inst for inst in self.instances if type(inst) == Instance]

    @property
    def predicted_instances(self) -> list[Instance]:
        """Frame instances that are user-labeled (`PredictedInstance` objects)."""
        return [inst for inst in self.instances if type(inst) == PredictedInstance]

    def numpy(self) -> np.ndarray:
        """Return all instances in the frame as a numpy array.

        Returns:
            Points as a numpy array of shape `(n_instances, n_nodes, 2)`.

            Note that the order of the instances is arbitrary.
        """
        n_instances = len(self.instances)
        n_nodes = len(self.instances[0]) if n_instances > 0 else 0
        pts = np.full((n_instances, n_nodes, 2), np.nan)
        for i, inst in enumerate(self.instances):
            pts[i] = inst.numpy()[:, 0:2]
        return pts

    @property
    def image(self) -> np.ndarray:
        """Return the image of the frame as a numpy array."""
        return self.video[self.frame_idx]
