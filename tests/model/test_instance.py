from sleap_io.model.instance import Point, PredictedPoint, Track, Instance, LabeledFrame
from sleap_io.model.skeleton import Skeleton, Node


def test_defaults():
    point_ = Point(x=0, y=0)
    predpoint_ = PredictedPoint.from_point(point_)
    track_ = Track()
    skeleton_ = Skeleton.from_names(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    instance_ = Instance(skeleton=skeleton_)

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

    assert instance_.skeleton == skeleton_
    assert instance_.points == None
    assert instance_.track == None
    assert instance_.frame == None
    assert instance_.from_predicted == None
