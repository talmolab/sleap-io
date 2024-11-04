"""Data structures for data contained within a single video frame.

The `LabeledFrame` class is a data structure that contains `Instance`s and
`PredictedInstance`s that are associated with a single frame within a video.
"""

from __future__ import annotations
from sleap_io import Instance, PredictedInstance, Video
from attrs import define, field
from typing import Union
import numpy as np


@define(eq=False)
class LabeledFrame:
    """Labeled data for a single frame of a video.

    Attributes:
        video: The `Video` associated with this `LabeledFrame`.
        frame_idx: The index of the `LabeledFrame` in the `Video`.
        instances: List of `Instance` objects associated with this `LabeledFrame`.

    Notes:
        Instances of this class are hashed by identity, not by value. This means that
        two `LabeledFrame` instances with the same attributes will NOT be considered
        equal in a set or dict.
    """

    video: Video
    frame_idx: int = field(converter=int)
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
    def has_user_instances(self) -> bool:
        """Return True if the frame has any user-labeled instances."""
        for inst in self.instances:
            if type(inst) == Instance:
                return True
        return False

    @property
    def predicted_instances(self) -> list[Instance]:
        """Frame instances that are predicted by a model (`PredictedInstance` objects)."""
        return [inst for inst in self.instances if type(inst) == PredictedInstance]

    @property
    def has_predicted_instances(self) -> bool:
        """Return True if the frame has any predicted instances."""
        for inst in self.instances:
            if type(inst) == PredictedInstance:
                return True
        return False

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

    @property
    def unused_predictions(self) -> list[Instance]:
        """Return a list of "unused" `PredictedInstance` objects in frame.

        This is all of the `PredictedInstance` objects which do not have a corresponding
        `Instance` in the same track in the same frame.
        """
        unused_predictions = []
        any_tracks = [inst.track for inst in self.instances if inst.track is not None]
        if len(any_tracks):
            # Use tracks to determine which predicted instances have been used
            used_tracks = [
                inst.track
                for inst in self.instances
                if type(inst) == Instance and inst.track is not None
            ]
            unused_predictions = [
                inst
                for inst in self.instances
                if inst.track not in used_tracks and type(inst) == PredictedInstance
            ]

        else:
            # Use from_predicted to determine which predicted instances have been used
            # TODO: should we always do this instead of using tracks?
            used_instances = [
                inst.from_predicted
                for inst in self.instances
                if inst.from_predicted is not None
            ]
            unused_predictions = [
                inst
                for inst in self.instances
                if type(inst) == PredictedInstance and inst not in used_instances
            ]

        return unused_predictions

    def remove_predictions(self):
        """Remove all `PredictedInstance` objects from the frame."""
        self.instances = [inst for inst in self.instances if type(inst) == Instance]

    def remove_empty_instances(self):
        """Remove all instances with no visible points."""
        self.instances = [inst for inst in self.instances if not inst.is_empty]
