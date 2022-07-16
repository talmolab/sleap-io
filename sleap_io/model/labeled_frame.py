"""Data structured for data contained within a single video frame.

The `LabeledFrame` class is a data structure that contains `Instance`s and
`PredictedInstance`s that are associated with a single frame within a video.
"""

from __future__ import annotations
from sleap_io import Instance, PredictedInstance, Video
from attrs import define
import attr
from typing import Union


@define(auto_attribs=True)
class LabeledFrame:
    """Holds labeled data for a single frame of a video.

    Args:
        video: The :class:`Video` associated with this `LabeledFrame`.
        frame_idx: The index of the `LabeledFrame` in the `Video`.
        instances: List of `Instance` objects associated with this `LabeledFrame`.
    """

    def _set_instance_frame(
        self, attribute, new_instances: list[Instance]
    ) -> list[Instance]:
        """Set the list of `Instance` objects associated with this `LabeledFrame`.

        Updates the `frame` attribute on each `Instance` to the
        :class:`LabeledFrame` which will contain the `Instance`.
        The list of `Instance` objects replaces `Instance` objects that were previously
        associated with `LabeledFrame`.

        Args:
            instances: A list of `Instance` objects associated with this `LabeledFrame`.

        Returns:
            None
        """
        # Make sure to set the frame for each instance to this LabeledFrame
        for instance in new_instances:
            instance.frame = self

        return new_instances

    video: Video
    frame_idx: int
    instances: list[Union[Instance, PredictedInstance]] = attr.ib(
        factory=list, on_setattr=_set_instance_frame
    )
