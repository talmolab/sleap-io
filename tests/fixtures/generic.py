"""Fixtures for general use."""
from sleap_io import (
    Video,
    Point,
    PredictedPoint,
    Instance,
    PredictedInstance,
    Node,
    Skeleton,
    Edge,
    LabeledFrame,
    Track,
)
import pytest


class DummyVideo:
    """Fake `Video` backend, returns frames with all zeros.

    This can be useful when you want to look at labels for a dataset but don't
    have access to the real video.
    """

    filename: str = ""
    height: int = 2000
    width: int = 2000
    frames: int = 10000
    channels: int = 1
    dummy: bool = True


@pytest.fixture
def dummy_video():
    return DummyVideo()


@pytest.fixture
def dummy_labeled_frame(dummy_video):

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
    return labeled_frame
