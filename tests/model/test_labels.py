"""Test methods and functions in the sleap_io.model.labels file."""

from numpy.testing import assert_equal
import pytest
from sleap_io import (
    Video,
    Skeleton,
    Instance,
    PredictedInstance,
    LabeledFrame,
    load_slp,
    load_video,
)
from sleap_io.model.labels import Labels
import numpy as np


def test_labels():
    """Test methods in the `Labels` data structure."""
    labels = Labels(
        [
            LabeledFrame(
                video=Video(filename="test"),
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


def test_labels_find(slp_typical):
    labels = load_slp(slp_typical)

    results = labels.find(video=labels.video, frame_idx=0)
    assert len(results) == 1
    lf = results[0]
    assert lf.frame_idx == 0

    labels.labeled_frames.append(LabeledFrame(video=labels.video, frame_idx=1))

    results = labels.find(video=labels.video)
    assert len(results) == 2

    results = labels.find(video=labels.video, frame_idx=2)
    assert len(results) == 0

    results = labels.find(video=labels.video, frame_idx=2, return_new=True)
    assert len(results) == 1
    assert results[0].frame_idx == 2
    assert len(results[0]) == 0


def test_labels_video():
    labels = Labels()

    with pytest.raises(ValueError):
        labels.video

    vid = Video(filename="test")
    labels.videos.append(vid)
    assert labels.video == vid

    labels.videos.append(Video(filename="test2"))
    with pytest.raises(ValueError):
        labels.video


def test_labels_skeleton():
    labels = Labels()

    with pytest.raises(ValueError):
        labels.skeleton

    skel = Skeleton(["A"])
    labels.skeletons.append(skel)
    assert labels.skeleton == skel

    labels.skeletons.append(Skeleton(["B"]))
    with pytest.raises(ValueError):
        labels.skeleton


def test_labels_getitem(slp_typical):
    labels = load_slp(slp_typical)
    labels.labeled_frames.append(LabeledFrame(video=labels.video, frame_idx=1))
    assert len(labels) == 2
    assert labels[0].frame_idx == 0
    assert len(labels[:2]) == 2
    assert len(labels[[0, 1]]) == 2
    assert len(labels[np.array([0, 1])]) == 2
    assert labels[(labels.video, 0)].frame_idx == 0

    with pytest.raises(IndexError):
        labels[(labels.video, 2000)]

    assert len(labels[labels.video]) == 2

    with pytest.raises(IndexError):
        labels[Video(filename="test")]

    with pytest.raises(IndexError):
        labels[None]


def test_labels_save(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    labels.save(tmp_path / "test.slp")
    assert (tmp_path / "test.slp").exists()
