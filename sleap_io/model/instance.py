from __future__ import annotations
from attrs import define, validators
import attr
from typing import List, Optional, Tuple, Union, Dict
from sleap_io.model.video import Video
from sleap_io.model.skeleton import Skeleton, Node
import numpy as np
import math


@define(auto_atrribs=True)
class Point:
    """A labeled point and any metadata associated with it.

    Args:
        x: The horizontal pixel location of point within image frame.
        y: The vertical pixel location of point within image frame.
        visible: Whether point is visible in the labelled image or not.
        complete: Has the point been verified by the user labeler.
    """

    x: float = math.nan
    y: float = math.nan
    visible: bool = attr.ib(default=True, kw_only=True)
    complete: bool = attr.ib(default=False, kw_only=True)


@define(auto_atrribs=True)
class PredictedPoint(Point):
    """A predicted point is an output of the inference procedure.

    It has all the properties of a labeled point, plus a score.

    Args:
        score: The point-level prediction score.
    """

    score: float = 0.0

    @classmethod
    def from_point(cls, point: Point, score: float = 0.0) -> PredictedPoint:
        """Create a PredictedPoint from a Point

        Args:
            point: The point to copy all data from.
            score: The score for this predicted point.

        Returns:
            A scored point based on the point passed in.
        """
        return cls(
            x=point.x,
            y=point.y,
            visible=point.visible,
            complete=point.complete,
            score=score,
        )


# "By default, two instances of attrs classes are equal if all their fields are equal."
@define(auto_atrribs=True, eq=True)
class Track:
    """A track object is associated with a set of animal/object instances
    across multiple frames of video. This allows tracking of unique
    entities in the video over time and space.

    Args:
        spawned_on: The video frame that this track was spawned on.
        name: A name given to this track for identifying purposes.
    """

    name: str = ""


# NOTE:
# Instance cannot be a slotted class at the moment. This is because it creates
# attributes _frame and _point_array_cache after init. These are private variables
# that are created in post init so they are not serialized.


@define(auto_atrribs=True)
class Instance:
    """This class represents a labeled instance.

    Args:
        skeleton: The skeleton that this instance is associated with.
        points: A dictionary where keys are skeleton node names and
            values are Point objects. Alternatively, a point array whose
            length and order matches skeleton.nodes.
        track: An optional multi-frame object track associated with
            this instance. This allows individual animals/objects to be
            tracked across frames.
        from_predicted: The predicted instance (if any) that this was
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

        Checks that all the _points defined for the skeleton are found
        in the skeleton.

        Args:
            attribute: Attribute being validated; not used.
            points: Either dict of points or PointArray
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
                    type=self.type,
                    actual=value.__class__,
                    value=value,
                ),
                attribute,
                self.type,
                value,
            )

    @classmethod
    def from_pointsarray(
        cls, points: np.ndarray, skeleton: Skeleton, track: Optional[Track] = None
    ) -> Instance:
        """Create an instance from an array of points.

        Args:
            points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
                contains the points in (x, y) coordinates of each node. Missing nodes
                should be represented as `NaN`.
            skeleton: A `sleap.Skeleton` instance with `n_nodes` nodes to associate with
                the instance.
            track: Optional `sleap.Track` object to associate with the instance.

        Returns:
            A new `Instance` object.
        """
        predicted_points = dict()
        node_names: List[str] = [node.name for node in skeleton.nodes]
        # TODO(LM): Ensure ordering of nodes and points match up.
        for point, node_name in zip(points, node_names):
            if np.isnan(point).any():
                continue

            predicted_points[node_name] = Point(x=point[0], y=point[1])

        return cls(points=predicted_points, skeleton=skeleton, track=track)


@define(auto_atrribs=True)
class PredictedInstance(Instance):
    """A predicted instance is an output of the inference procedure.

    Args:
        score: The instance-level grouping prediction score.
        tracking_score: The instance-level track matching score.
    """

    from_predicted: Optional[PredictedInstance] = attr.ib(
        default=None, validator=validators.instance_of(type(None))
    )
    score: float = attr.ib(default=0.0, converter=float)
    tracking_score: float = attr.ib(default=0.0, converter=float)

    @classmethod
    def from_instance(cls, instance: Instance, score: float) -> PredictedInstance:
        """Create a `PredictedInstance` from an `Instance`.

        The fields are copied in a shallow manner with the exception of points. For each
        point in the instance a `PredictedPoint` is created with score set to default
        value.

        Args:
            instance: The `Instance` object to shallow copy data from.
            score: The score for this instance.

        Returns:
            A `PredictedInstance` for the given `Instance`.
        """
        kw_args = attr.asdict(
            instance,
            recurse=False,
        )
        kw_args["score"] = score
        return cls(**kw_args)

    @classmethod
    def from_arrays(
        cls,
        points: np.ndarray,
        point_confidences: np.ndarray,
        instance_score: float,
        skeleton: Skeleton,
        track: Optional[Track] = None,
    ) -> "PredictedInstance":
        """Create a predicted instance from data arrays.

        Args:
            points: A numpy array of shape `(n_nodes, 2)` and dtype `float32` that
                contains the points in `(x, y)` coordinates of each node. Missing nodes
                should be represented as `NaN`.
            point_confidences: A numpy array of shape `(n_nodes,)` and dtype `float32`
                that contains the confidence/score of the points.
            instance_score: Scalar float representing the overall instance score, e.g.,
                the PAF grouping score.
            skeleton: A sleap.Skeleton instance with n_nodes nodes to associate with the
                predicted instance.
            track: Optional `sleap.Track` to associate with the instance.

        Returns:
            A new `PredictedInstance`.
        """
        predicted_points = dict()
        node_names: List[str] = [node.name for node in skeleton.nodes]
        for point, confidence, node_name in zip(points, point_confidences, node_names):
            if np.isnan(point).any():
                continue

            predicted_points[node_name] = PredictedPoint(
                x=point[0], y=point[1], score=confidence
            )

        return cls(
            points=predicted_points,
            skeleton=skeleton,
            score=instance_score,
            track=track,
        )


@define(auto_atrribs=True)
class LabeledFrame:
    """Holds labeled data for a single frame of a video.

    Args:
        video: The :class:`Video` associated with this frame.
        frame_idx: The index of frame in video.
        instances: List of instances associated with the frame.
    """

    def _set_instance_frame(self, attribute, new_instances: List[Instance]):
        """Set the list of instances associated with this frame.

        Updates the `frame` attribute on each instance to the
        :class:`LabeledFrame` which will contain the instance.
        The list of instances replaces instances that were previously
        associated with frame.

        Args:
            instances: A list of instances associated with this frame.

        Returns:
            None
        """
        # Make sure to set the frame for each instance to this LabeledFrame
        for instance in new_instances:
            instance.frame = self

        print(f"{attribute}")

        # attribute.value = new_instances

    video: Video
    frame_idx: int
    instances: Union[List[Instance], List[PredictedInstance]] = attr.ib(
        factory=List[Instance], on_setattr=_set_instance_frame
    )
