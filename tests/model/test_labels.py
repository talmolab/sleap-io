"""Test methods and functions in the sleap_io.model.labels file."""
from numpy.testing import assert_equal
import pytest
from sleap_io import Video, Skeleton, Instance, PredictedInstance, LabeledFrame
from sleap_io.model.labels import Labels


def test_labels():
    """Test methods in the `Labels` data structure."""
    labels = Labels(
        [
            LabeledFrame(
                video=Video(filename="test", shape=(1, 1, 1, 1)),
                frame_idx=0,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"])),
                    PredictedInstance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
                ],
            )
        ]
    )

    assert len(labels) == 1
    assert type(labels[0]) == LabeledFrame
    assert labels[0].frame_idx == 0

    with pytest.raises(IndexError):
        labels[None]

    # Test Labels.__iter__ method
    for lf_idx, lf in enumerate(labels):
        assert lf == labels[lf_idx]

    assert str(labels) == "Labels(labeled_frames=1, videos=1, skeletons=1, tracks=0)"


def test_labels_numpy(labels_predictions: Labels):
    trx = labels_predictions.numpy(video=None, untracked=False)
    assert trx.shape == (1100, 27, 24, 2)

    trx = labels_predictions.numpy(video=None, untracked=False, return_confidence=True)
    assert trx.shape == (1100, 27, 24, 3)

    labels_single = Labels(
        labeled_frames=[
            LabeledFrame(
                video=lf.video, frame_idx=lf.frame_idx, instances=[lf.instances[0]]
            )
            for lf in labels_predictions
        ]
    )
    assert labels_single.numpy().shape == (1100, 1, 24, 2)

    assert labels_predictions.numpy(untracked=True).shape == (1100, 5, 24, 2)
    for lf in labels_predictions:
        for inst in lf:
            inst.track = None
    labels_predictions.tracks = []
    assert labels_predictions.numpy(untracked=False).shape == (1100, 0, 24, 2)
