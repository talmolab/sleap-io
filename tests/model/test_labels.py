"""Test methods and functions in the sleap_io.model.labels file."""

from numpy.testing import assert_equal
import pytest
from sleap_io import (
    Video,
    Skeleton,
    Instance,
    PredictedInstance,
    LabeledFrame,
    Track,
    SuggestionFrame,
    load_slp,
    load_video,
)
from sleap_io.model.labels import Labels
import numpy as np
from pathlib import Path


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

    assert (
        str(labels)
        == "Labels(labeled_frames=1, videos=1, skeletons=1, tracks=0, suggestions=0)"
    )


def test_update(slp_real_data):
    base_labels = load_slp(slp_real_data)

    labels = Labels(base_labels.labeled_frames)
    assert len(labels.videos) == len(base_labels.videos) == 1
    assert len(labels.tracks) == len(base_labels.tracks) == 0
    assert len(labels.skeletons) == len(base_labels.skeletons) == 1

    new_video = Video.from_filename("fake.mp4")
    labels.suggestions.append(SuggestionFrame(video=new_video, frame_idx=0))

    new_track = Track("new_track")
    labels[0][0].track = new_track

    new_skel = Skeleton(["A", "B"])
    new_video2 = Video.from_filename("fake2.mp4")
    labels.append(
        LabeledFrame(
            video=new_video2,
            frame_idx=0,
            instances=[
                Instance.from_numpy(np.array([[0, 1], [2, 3]]), skeleton=new_skel)
            ],
        ),
        update=False,
    )

    labels.update()
    assert new_video in labels.videos
    assert new_video2 in labels.videos
    assert new_track in labels.tracks
    assert new_skel in labels.skeletons


def test_append_extend():
    labels = Labels()

    new_skel = Skeleton(["A", "B"])
    new_video = Video.from_filename("fake.mp4")
    new_track = Track("new_track")
    labels.append(
        LabeledFrame(
            video=new_video,
            frame_idx=0,
            instances=[
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=new_skel, track=new_track
                )
            ],
        ),
        update=True,
    )
    assert labels.videos == [new_video]
    assert labels.skeletons == [new_skel]
    assert labels.tracks == [new_track]

    new_video2 = Video.from_filename("fake.mp4")
    new_skel2 = Skeleton(["A", "B", "C"])
    new_track2 = Track("new_track2")
    labels.extend(
        [
            LabeledFrame(
                video=new_video,
                frame_idx=1,
                instances=[
                    Instance.from_numpy(
                        np.array([[0, 1], [2, 3]]), skeleton=new_skel, track=new_track2
                    )
                ],
            ),
            LabeledFrame(
                video=new_video2,
                frame_idx=0,
                instances=[
                    Instance.from_numpy(
                        np.array([[0, 1], [2, 3], [4, 5]]), skeleton=new_skel2
                    )
                ],
            ),
        ],
        update=True,
    )

    assert labels.videos == [new_video, new_video2]
    assert labels.skeletons == [new_skel, new_skel2]
    assert labels.tracks == [new_track, new_track2]


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


def test_labels_clean_unchanged(slp_real_data):
    labels = load_slp(slp_real_data)
    assert len(labels) == 10
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    assert labels[1].frame_idx == 990
    assert len(labels[1]) == 2
    assert len(labels.skeletons) == 1
    assert len(labels.videos) == 1
    assert len(labels.tracks) == 0
    labels.clean(
        frames=True, empty_instances=True, skeletons=True, tracks=True, videos=True
    )
    assert len(labels) == 10
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    assert labels[1].frame_idx == 990
    assert len(labels[1]) == 2
    assert len(labels.skeletons) == 1
    assert len(labels.videos) == 1
    assert len(labels.tracks) == 0


def test_labels_clean_frames(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    labels[0].instances = []
    labels.clean(
        frames=True, empty_instances=False, skeletons=False, tracks=False, videos=False
    )
    assert len(labels) == 9
    assert labels[0].frame_idx == 990
    assert len(labels[0]) == 2


def test_labels_clean_empty_instances(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    labels[0].instances = [
        Instance.from_numpy(
            np.full((len(labels.skeleton), 2), np.nan), skeleton=labels.skeleton
        )
    ]
    labels.clean(
        frames=False, empty_instances=True, skeletons=False, tracks=False, videos=False
    )
    assert len(labels) == 10
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 0

    labels.clean(
        frames=True, empty_instances=True, skeletons=False, tracks=False, videos=False
    )
    assert len(labels) == 9


def test_labels_clean_skeletons(slp_real_data):
    labels = load_slp(slp_real_data)
    labels.skeletons.append(Skeleton(["A", "B"]))
    assert len(labels.skeletons) == 2
    labels.clean(
        frames=False, empty_instances=False, skeletons=True, tracks=False, videos=False
    )
    assert len(labels) == 10
    assert len(labels.skeletons) == 1


def test_labels_clean_tracks(slp_real_data):
    labels = load_slp(slp_real_data)
    labels.tracks.append(Track(name="test1"))
    labels.tracks.append(Track(name="test2"))
    assert len(labels.tracks) == 2
    labels[0].instances[0].track = labels.tracks[1]
    labels.clean(
        frames=False, empty_instances=False, skeletons=False, tracks=True, videos=False
    )
    assert len(labels) == 10
    assert len(labels.tracks) == 1
    assert labels[0].instances[0].track == labels.tracks[0]
    assert labels.tracks[0].name == "test2"


def test_labels_clean_videos(slp_real_data):
    labels = load_slp(slp_real_data)
    labels.videos.append(Video(filename="test2"))
    assert len(labels.videos) == 2
    labels.clean(
        frames=False, empty_instances=False, skeletons=False, tracks=False, videos=True
    )
    assert len(labels) == 10
    assert len(labels.videos) == 1
    assert labels.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"


def test_labels_remove_predictions(slp_real_data):
    labels = load_slp(slp_real_data)
    assert len(labels) == 10
    assert sum([len(lf.predicted_instances) for lf in labels]) == 12
    labels.remove_predictions(clean=False)
    assert len(labels) == 10
    assert sum([len(lf.predicted_instances) for lf in labels]) == 0

    labels = load_slp(slp_real_data)
    labels.remove_predictions(clean=True)
    assert len(labels) == 5
    assert sum([len(lf.predicted_instances) for lf in labels]) == 0


def test_replace_videos(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"
    labels.replace_videos(
        old_videos=[labels.video], new_videos=[Video.from_filename("fake.mp4")]
    )

    for lf in labels:
        assert lf.video.filename == "fake.mp4"

    for sf in labels.suggestions:
        assert sf.video.filename == "fake.mp4"


def test_replace_filenames():
    labels = Labels(videos=[Video.from_filename("a.mp4"), Video.from_filename("b.mp4")])

    with pytest.raises(ValueError):
        labels.replace_filenames()

    with pytest.raises(ValueError):
        labels.replace_filenames(new_filenames=[], filename_map={})

    with pytest.raises(ValueError):
        labels.replace_filenames(new_filenames=[], prefix_map={})

    with pytest.raises(ValueError):
        labels.replace_filenames(filename_map={}, prefix_map={})

    with pytest.raises(ValueError):
        labels.replace_filenames(new_filenames=[], filename_map={}, prefix_map={})

    labels.replace_filenames(new_filenames=["c.mp4", "d.mp4"])
    assert [v.filename for v in labels.videos] == ["c.mp4", "d.mp4"]

    with pytest.raises(ValueError):
        labels.replace_filenames(["f.mp4"])

    labels.replace_filenames(
        filename_map={"c.mp4": "/a/b/c.mp4", "d.mp4": "/a/b/d.mp4"}
    )
    assert [Path(v.filename).as_posix() for v in labels.videos] == [
        "/a/b/c.mp4",
        "/a/b/d.mp4",
    ]

    labels.replace_filenames(prefix_map={"/a/b/": "/A/B"})
    assert [Path(v.filename).as_posix() for v in labels.videos] == [
        "/A/B/c.mp4",
        "/A/B/d.mp4",
    ]

    labels = Labels(videos=[Video.from_filename(["imgs/img0.png", "imgs/img1.png"])])
    labels.replace_filenames(
        filename_map={
            "imgs/img0.png": "train/imgs/img0.png",
            "imgs/img1.png": "train/imgs/img1.png",
        }
    )
    assert labels.video.filename == ["train/imgs/img0.png", "train/imgs/img1.png"]

    labels.replace_filenames(prefix_map={"train/": "test/"})
    assert labels.video.filename == ["test/imgs/img0.png", "test/imgs/img1.png"]
