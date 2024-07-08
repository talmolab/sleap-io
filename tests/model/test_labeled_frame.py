"""Tests for methods in sleap_io.model.labeled_frame file."""

from numpy.testing import assert_equal
from sleap_io import Video, Skeleton, Instance, PredictedInstance, Track
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

    assert lf.has_predicted_instances
    assert lf.has_user_instances


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
    assert lf.has_predicted_instances
    assert lf.has_user_instances

    # Remove predictions
    lf.remove_predictions()

    assert len(lf) == 1
    assert len(lf.predicted_instances) == 0
    assert type(lf[0]) == Instance
    assert_equal(lf.numpy(), [[[0, 1], [2, 3]]])
    assert not lf.has_predicted_instances
    assert lf.has_user_instances


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


def test_labeled_frame_image(centered_pair_low_quality_path):
    video = Video.from_filename(centered_pair_low_quality_path)
    lf = LabeledFrame(video=video, frame_idx=0)
    assert_equal(lf.image, video[0])


def test_labeled_frame_unused_predictions():
    video = Video("test.mp4")
    skel = Skeleton(["A", "B"])
    track = Track("trk")

    lf1 = LabeledFrame(video=video, frame_idx=0)
    lf1.instances.append(
        Instance.from_numpy([[0, 0], [0, 0]], skeleton=skel, track=track)
    )
    lf1.instances.append(
        PredictedInstance.from_numpy(
            [[0, 0], [0, 0]], [1, 1], 1, skeleton=skel, track=track
        )
    )
    lf1.instances.append(
        PredictedInstance.from_numpy([[1, 1], [1, 1]], [1, 1], 1, skeleton=skel)
    )

    assert len(lf1.unused_predictions) == 1
    assert (lf1.unused_predictions[0].numpy() == 1).all()

    lf2 = LabeledFrame(video=video, frame_idx=1)
    lf2.instances.append(
        PredictedInstance.from_numpy([[0, 0], [0, 0]], [1, 1], 1, skeleton=skel)
    )
    lf2.instances.append(Instance.from_numpy([[0, 0], [0, 0]], skeleton=skel))
    lf2.instances[-1].from_predicted = lf2.instances[-2]
    lf2.instances.append(
        PredictedInstance.from_numpy([[1, 1], [1, 1]], [1, 1], 1, skeleton=skel)
    )

    assert len(lf2.unused_predictions) == 1
    assert (lf2.unused_predictions[0].numpy() == 1).all()
