"""Tests for methods in sleap_io.model.labeled_frame file."""
from numpy.testing import assert_equal
from sleap_io import Video, Skeleton, Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame


def test_labeled_frame():
    """Test initialization and methods of `LabeledFrame` class."""
    inst = Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"]))
    lf = LabeledFrame(
        video=Video(filename="test"),
        frame_idx=0,
        instances=[
            inst,
            PredictedInstance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
        ],
    )
    assert_equal(lf.numpy(), [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

    assert len(lf) == 2
    assert len(lf.user_instances) == 1
    assert type(lf.user_instances[0]) == Instance
    assert len(lf.predicted_instances) == 1
    assert type(lf.predicted_instances[0]) == PredictedInstance

    # Test LabeledFrame.__getitem__ method
    assert lf[0] == inst
