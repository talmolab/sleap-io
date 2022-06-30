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
    numpy_array1 = np.array([[1, 1], [2, 2], [3, 3]], dtype="float32")
    numpy_array2 = np.array(
        [[1, 1, True, False], [2, 2, True, False], [3, 3, True, False]], dtype="float32"
    )
    points_confidence = np.array([1, 2, 3], dtype="float32")
    instance2 = Instance.from_numpy(points=numpy_array1, skeleton=skeleton)
    instance3 = Instance.from_numpy(points=numpy_array2, skeleton=skeleton)

    pred_instance1 = PredictedInstance(skeleton)
    pred_instance2 = PredictedInstance.from_instance(instance1, 0.0)
    pred_instance3 = PredictedInstance.from_numpyarray(
        points=numpy_array1,
        point_confidences=points_confidence,
        instance_score=0.0,
        skeleton=skeleton,
    )
    pred_instance4 = PredictedInstance.from_numpyarray(
        points=[np.nan],
        point_confidences=points_confidence,
        instance_score=0.0,
        skeleton=skeleton,
    )
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
    assert len(instance2.points) == 3  # Instance2 from_numpy
    assert instance2.skeleton == skeleton
    assert (
        list(instance3.points.values())[0].visible == True
        and list(instance3.points.values())[0].complete == False
    )
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
    assert pred_instance2.score == 0.0
    assert len(pred_instance3.points) == 3
    assert pred_instance3.score == 0.0
    assert pred_instance3.track == None
    assert pred_instance4.points == {}

    # LabeledFrame
    assert labeled_frame.video == dummy
    assert labeled_frame.instances == [instance1]
    assert labeled_frame.instances
    labeled_frame.instances = [instance1]
    assert labeled_frame.instances == [instance1]
