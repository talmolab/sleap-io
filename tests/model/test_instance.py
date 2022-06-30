import numpy as np
import pytest
from sleap_io.model.instance import (
    Point,
    PredictedPoint,
    Track,
    Instance,
    LabeledFrame,
    PredictedInstance,
)
from sleap_io.model.skeleton import Skeleton, Node, Edge
from tests.fixture.video import test_video


def test_classes(test_video):

    point = Point(x=0, y=0)
    pred_point = PredictedPoint.from_point(point)
    track = Track()
    skeleton = Skeleton(
        nodes=[Node("head"), Node("thorax"), Node("abdomen")],
        edges=[
            Edge(source=Node("head"), destination=Node("thorax")),
            Edge(source=Node("thorax"), destination=Node("abdomen")),
        ],
    )
    instance1 = Instance(skeleton=skeleton, points={"head": point})

    pred_instance1 = PredictedInstance(skeleton)

    dummy = test_video
    labeled_frame = LabeledFrame(video=dummy, frame_idx=1, instances=[instance1])

    # Point
    assert point.x == 0
    assert point.y == 0
    assert point.visible == True
    assert point.complete == False

    # PredictedPoint
    assert pred_point.x == 0
    assert pred_point.y == 0
    assert pred_point.visible == True
    assert pred_point.complete == False
    assert pred_point.score == 0

    # Track
    assert track.name == ""

    # Instance
    assert instance1.skeleton == skeleton
    assert instance1.points == {"head": point}
    assert instance1.track == None
    assert instance1.frame == None
    assert instance1.from_predicted == None
    with pytest.raises(TypeError):
        Instance(skeleton=skeleton, from_predicted="foo")
    with pytest.raises(KeyError):
        Instance(skeleton=skeleton, points={"foo": "bar"})
    with pytest.raises(TypeError):
        Instance(skeleton=skeleton, points="foo")

    # PredictedInstance
    assert pred_instance1.from_predicted == None
    assert pred_instance1.score == 0.0
    assert pred_instance1.tracking_score == 0.0

    # LabeledFrame
    assert labeled_frame.video == dummy
    assert labeled_frame.instances == [instance1]
    assert labeled_frame.instances
    labeled_frame.instances = [instance1]
    assert labeled_frame.instances == [instance1]
