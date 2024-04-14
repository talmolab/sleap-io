"""Tests for methods in sleap_io.model.labeled_frame file."""

from numpy.testing import assert_equal
from sleap_io import Video, Skeleton, Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame
import numpy as np


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


def test_remove_predictions():
    """Test removing predictions from `LabeledFrame`."""
    inst = Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"]))
    lf = LabeledFrame(
        video=Video(filename="test"),
        frame_idx=0,
        instances=[
            inst,
            PredictedInstance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
        ],
    )

    assert len(lf) == 2
    assert len(lf.predicted_instances) == 1

    # Remove predictions
    lf.remove_predictions()

    assert len(lf) == 1
    assert len(lf.predicted_instances) == 0
    assert type(lf[0]) == Instance
    assert_equal(lf.numpy(), [[[0, 1], [2, 3]]])


def test_remove_empty_instances():
    """Test removing empty instances from `LabeledFrame`."""
    inst = Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"]))
    lf = LabeledFrame(
        video=Video(filename="test"),
        frame_idx=0,
        instances=[
            inst,
            Instance(
                [[np.nan, np.nan], [np.nan, np.nan]], skeleton=Skeleton(["A", "B"])
            ),
        ],
    )

    assert len(lf) == 2

    # Remove empty instances
    lf.remove_empty_instances()

    assert len(lf) == 1
    assert type(lf[0]) == Instance
    assert_equal(lf.numpy(), [[[0, 1], [2, 3]]])
