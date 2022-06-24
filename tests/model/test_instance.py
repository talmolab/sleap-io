from multiprocessing import dummy
from sleap_io.model.instance import (
    Point,
    PredictedPoint,
    Track,
    Instance,
    LabeledFrame,
    PredictedInstance,
)
from sleap_io.model.skeleton import Skeleton, Node
from sleap_io.model.video import DummyVideo
import numpy as np


def test_classes():

    point_ = Point(x=0, y=0)
    predpoint_ = PredictedPoint.from_point(point_)
    track_ = Track()
    skeleton_ = Skeleton.from_names(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    instance1 = Instance(skeleton=skeleton_, points={"head": point_})
    pointsarray = np.array([[1, 1], [2, 2], [3, 3]], dtype="float32")
    pointsconfidence = np.array([1, 2, 3], dtype="float32")
    instance2 = Instance.from_pointsarray(points=pointsarray, skeleton=skeleton_)
    predinstance1 = PredictedInstance(skeleton_)
    predinstance2 = PredictedInstance.from_instance(instance1, 0.0)
    predinstance3 = PredictedInstance.from_arrays(
        points=pointsarray,
        point_confidences=pointsconfidence,
        instance_score=0.0,
        skeleton=skeleton_,
    )
    dummy_ = DummyVideo()
    labeledframe_ = LabeledFrame(video=dummy_, frame_idx=1, instances=[instance1])

    # Point

    assert point_.x == 0
    assert point_.y == 0
    assert point_.visible == True
    assert point_.complete == False

    # PredictedPoint

    assert predpoint_.x == 0
    assert predpoint_.y == 0
    assert predpoint_.visible == True
    assert predpoint_.complete == False
    assert predpoint_.score == 0

    # Track

    assert track_.name == ""

    # Instance

    assert instance1.skeleton == skeleton_
    assert instance1.points == {"head": point_}
    assert instance1.track == None
    assert instance1.frame == None
    assert instance1.from_predicted == None
    assert len(instance2.points) == 3  # Instance from_pointsarray
    assert instance2.skeleton == skeleton_

    # PredictedInstance

    assert predinstance1.from_predicted == None
    assert predinstance1.score == 0.0
    assert predinstance1.tracking_score == 0.0
    assert predinstance2.score == 0.0
    assert len(predinstance3.points) == 3
    assert predinstance3.score == 0.0
    assert predinstance3.track == None

    # LabeledFrame

    assert labeledframe_.video == dummy_
    assert labeledframe_.instances == [instance1]
    print(labeledframe_.instances)
    labeledframe_.instances = [instance2]
    print(labeledframe_.instances)
    # assert labeledframe_.instances == [instance2]
