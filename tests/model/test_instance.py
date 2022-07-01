<<<<<<< HEAD
import numpy as np
import pytest
from sleap_io.model.instance import (
=======
import pytest
from sleap_io import (
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
    Point,
    PredictedPoint,
    Track,
    Instance,
    LabeledFrame,
    PredictedInstance,
<<<<<<< HEAD
)
from sleap_io.model.skeleton import Skeleton, Node
from tests.fixture.fixtures import test_video


def test_classes(test_video):

    point = Point(x=0, y=0)
    pred_point = PredictedPoint.from_point(point)
    track = Track()
    skeleton = Skeleton.from_names(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    instance1 = Instance(skeleton=skeleton, points={"head": point})
    pointsarray1 = np.array([[1, 1], [2, 2], [3, 3]], dtype="float32")
    pointsarray2 = np.array(
        [[1, 1, True, False], [2, 2, True, False], [3, 3, True, False]], dtype="float32"
    )
    pointsconfidence = np.array([1, 2, 3], dtype="float32")
    instance2 = Instance.from_pointsarray(points=pointsarray1, skeleton=skeleton)
    instance3 = Instance.from_pointsarray(points=pointsarray2, skeleton=skeleton)

    pred_instance1 = PredictedInstance(skeleton)
    pred_instance2 = PredictedInstance.from_instance(instance1, 0.0)
    pred_instance3 = PredictedInstance.from_arrays(
        points=pointsarray1,
        point_confidences=pointsconfidence,
        instance_score=0.0,
        skeleton=skeleton,
    )
    pred_instance4 = PredictedInstance.from_arrays(
        points=[np.nan],
        point_confidences=pointsconfidence,
        instance_score=0.0,
        skeleton=skeleton,
    )
    dummy = test_video
    labeledframe_ = LabeledFrame(video=dummy, frame_idx=1, instances=[instance1])

    # Point

=======
    Skeleton,
    Node,
    Edge,
)


def test_classes(dummy_video):

    point = Point(x=0, y=0)
    pred_point = PredictedPoint(x=0, y=0)
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

    dummy = dummy_video
    labeled_frame = LabeledFrame(video=dummy, frame_idx=1, instances=[instance1])

    # Point
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
    assert point.x == 0
    assert point.y == 0
    assert point.visible == True
    assert point.complete == False

    # PredictedPoint
<<<<<<< HEAD

=======
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
    assert pred_point.x == 0
    assert pred_point.y == 0
    assert pred_point.visible == True
    assert pred_point.complete == False
    assert pred_point.score == 0

    # Track
<<<<<<< HEAD

    assert track.name == ""

    # Instance

=======
    assert track.name == ""

    # Instance
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
    assert instance1.skeleton == skeleton
    assert instance1.points == {"head": point}
    assert instance1.track == None
    assert instance1.frame == None
    assert instance1.from_predicted == None
<<<<<<< HEAD
    assert len(instance2.points) == 3  # Instance2 from_pointsarray
    assert instance2.skeleton == skeleton
    assert (
        list(instance3.points.values())[0].visible == True
        and list(instance3.points.values())[0].complete == False
    )

=======
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
    with pytest.raises(TypeError):
        Instance(skeleton=skeleton, from_predicted="foo")
    with pytest.raises(KeyError):
        Instance(skeleton=skeleton, points={"foo": "bar"})
    with pytest.raises(TypeError):
        Instance(skeleton=skeleton, points="foo")

    # PredictedInstance
<<<<<<< HEAD

    assert pred_instance1.from_predicted == None
    assert pred_instance1.score == 0.0
    assert pred_instance1.tracking_score == 0.0
    assert pred_instance2.score == 0.0
    assert len(pred_instance3.points) == 3
    assert pred_instance3.score == 0.0
    assert pred_instance3.track == None
    assert pred_instance4.points == {}

    # LabeledFrame

    assert labeledframe_.video == dummy
    assert labeledframe_.instances == [instance1]
    assert labeledframe_.instances
    labeledframe_.instances = [instance1]
    assert labeledframe_.instances == [instance1]
=======
    assert pred_instance1.from_predicted == None
    assert pred_instance1.score == 0.0
    assert pred_instance1.tracking_score == 0.0

    # LabeledFrame
    assert labeled_frame.video == dummy
    assert labeled_frame.instances == [instance1]
    assert labeled_frame.instances
    labeled_frame.instances = [instance1]
    assert labeled_frame.instances == [instance1]
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
