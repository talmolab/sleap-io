from __future__ import annotations
from attrs import define, validators
import attr
from typing import List, Optional, Tuple, Union, Dict
from sleap_io.model.video import Video
from sleap_io.model.skeleton import Skeleton, Node
import numpy as np


@define(auto_attribs=True)
class Point:
    """A labeled point and any metadata associated with it.

    Args:
        x: The horizontal pixel location of point within image frame.
        y: The vertical pixel location of point within image frame.
        visible: Whether point is visible in the labelled image or not.
        complete: Has the point been verified by the user labeler.
    """

    x: float = np.nan
    y: float = np.nan
    visible: bool = attr.ib(default=True, kw_only=True)
    complete: bool = attr.ib(default=False, kw_only=True)


@define(auto_attribs=True)
class PredictedPoint(Point):
    """A predicted point is an output of the inference procedure.

    It has all the properties of a labeled point, plus a score.

    Args:
        score: The point-level prediction score.
    """

    score: float = 0.0


@define(auto_attribs=True, eq=True)
class Track:
    """A track object is associated with a set of animal/object instances
    across multiple frames of video.

    This allows tracking of unique entities in the video over time and space.

    Args:
        spawned_on: The video frame that this track was spawned on.
        name: A name given to this track for identifying purposes.
    """

    name: str = ""


@define(auto_attribs=True)
class Instance:
    """This class represents a labeled instance.

    Args:
        skeleton: The `Skeleton` that this `Instance` is associated with.
        points: A dictionary where keys are `Skeleton` node names and
            values are `Point` objects. Alternatively, a numpy array whose
            length and order matches `skeleton.nodes`.
        track: An optional multi-frame object track associated with
            this instance. This allows individual animals/objects to be
            tracked across frames.
        from_predicted: The `PredictedInstance` (if any) that this was
            copied from.
        frame: A back reference to the :class:`LabeledFrame` that this
            :class:`Instance` belongs to. This field is set when
            instances are added to :class:`LabeledFrame` objects.
    """

    skeleton: Skeleton = attr.ib(validator=validators.instance_of(Skeleton))
    points: Union[Dict[str, Point], Dict[str, PredictedPoint]] = attr.ib(default=None)
    track: Optional[Track] = None
    frame: Union[LabeledFrame, None] = None
    from_predicted: Optional[PredictedInstance] = attr.ib(default=None)

    @points.validator
    def _validate_all_points(self, attribute, points: Dict[str, Point]):
        """Validation method called by attrs.

        Checks that all the points defined for the `Skeleton` are found.

        Args:
            attribute: Attribute being validated; not used.
            points: Dict of `points`
                If dict, keys should be node names.

        Raises:
            ValueError: If a point is associated with a skeleton node
                name that doesn't exist.
            TypeError: With a human readable error message, the attribute (of type
                attrs.Attribute), the expected type, and the value it got.
        """
        if points is not None:
            try:
                for node_name in points.keys():
                    if Node(node_name) not in self.skeleton.nodes:
                        raise KeyError(
                            f"There is no node named {node_name} in {self.skeleton}"
                        )
            except AttributeError:
                raise TypeError(
                    "'{name}' must be {type!r} (got {value!r} that is a "
                    "{actual!r}).".format(
                        name=attribute.name,
                        type=dict,
                        actual=points.__class__,
                        value=points,
                    ),
                    attribute,
                    dict,
                    points,
                )

    @from_predicted.validator
    def _validate_type_is_PredictedInstance(self, attribute, value):
        if (value is not None) and (not isinstance(value, PredictedInstance)):
            raise TypeError(
                "'{name}' must be {type!r} (got {value!r} that is a "
                "{actual!r}).".format(
                    name=attribute.name,
                    type="PredictedInstance",
                    actual=value.__class__,
                    value=value,
                ),
                attribute,
                "PredictedInstance",
                value,
            )


@define(auto_attribs=True)
class PredictedInstance(Instance):
    """A `PredictedInstance` is an output of the inference procedure.

    Args:
        score: The instance-level grouping prediction score.
        tracking_score: The instance-level track matching score.
    """

    from_predicted: Optional[PredictedInstance] = attr.ib(
        default=None, validator=validators.instance_of(type(None))
    )
    score: float = attr.ib(default=0.0, converter=float)
    tracking_score: float = attr.ib(default=0.0, converter=float)


@define(auto_attribs=True)
class LabeledFrame:
    """Holds labeled data for a single frame of a video.

    Args:
        video: The :class:`Video` associated with this `LabeledFrame`.
        frame_idx: The index of the `LabeledFrame` in the video.
        instances: List of `Instance` objects associated with this `LabeledFrame`.
    """

    def _set_instance_frame(
        self, attribute, new_instances: List[Instance]
    ) -> List[Instance]:
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
    instances: Union[List[Instance], List[PredictedInstance]] = attr.ib(
        factory=List[Instance], on_setattr=_set_instance_frame
    )
