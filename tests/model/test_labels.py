"""Test methods and functions in the sleap_io.model.labels file."""

from numpy.testing import assert_equal, assert_allclose
import pytest
from sleap_io import (
    Video,
    Skeleton,
    Instance,
    PredictedInstance,
    LabeledFrame,
    Track,
    SuggestionFrame,
    RecordingSession,
    load_slp,
    load_video,
)
from sleap_io.model.labels import Labels
import numpy as np
from pathlib import Path
import copy


def test_labels():
    """Test methods in the `Labels` data structure."""
    skel = Skeleton(["A", "B"])
    labels = Labels(
        [
            LabeledFrame(
                video=Video(filename="test"),
                frame_idx=0,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel),
                    PredictedInstance([[4, 5], [6, 7]], skeleton=skel),
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

    assert str(labels) == (
        "Labels(labeled_frames=1, videos=1, skeletons=1, tracks=0, suggestions=0, "
        "sessions=0)"
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
    labels.replace_videos(new_videos=[Video.from_filename("fake.mp4")])

    for lf in labels:
        assert lf.video.filename == "fake.mp4"

    for sf in labels.suggestions:
        assert sf.video.filename == "fake.mp4"

    assert labels.video.filename == "fake.mp4"


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


def test_split(slp_real_data, tmp_path):
    # n = 0
    labels = Labels()
    split1, split2 = labels.split(0.5)
    assert len(split1) == len(split2) == 0

    # n = 1
    labels.append(LabeledFrame(video=Video("test.mp4"), frame_idx=0))
    split1, split2 = labels.split(0.5)
    assert len(split1) == len(split2) == 1
    assert split1[0].frame_idx == 0
    assert split2[0].frame_idx == 0

    split1, split2 = labels.split(0.999)
    assert len(split1) == len(split2) == 1
    assert split1[0].frame_idx == 0
    assert split2[0].frame_idx == 0

    split1, split2 = labels.split(n=1)
    assert len(split1) == len(split2) == 1
    assert split1[0].frame_idx == 0
    assert split2[0].frame_idx == 0

    # Real data
    labels = load_slp(slp_real_data)
    assert len(labels) == 10

    split1, split2 = labels.split(n=0.6)
    assert len(split1) == 6
    assert len(split2) == 4

    # Rounding errors
    split1, split2 = labels.split(n=0.001)
    assert len(split1) == 1
    assert len(split2) == 9

    split1, split2 = labels.split(n=0.999)
    assert len(split1) == 9
    assert len(split2) == 1

    # Integer
    split1, split2 = labels.split(n=8)
    assert len(split1) == 8
    assert len(split2) == 2

    # Serialization round trip
    split1.save(tmp_path / "split1.slp")
    split1_ = load_slp(tmp_path / "split1.slp")
    assert len(split1) == len(split1_)
    assert split1.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"
    assert split1_.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"

    split2.save(tmp_path / "split2.slp")
    split2_ = load_slp(tmp_path / "split2.slp")
    assert len(split2) == len(split2_)
    assert split2.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"
    assert split2_.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"

    # Serialization round trip with embedded data
    labels = load_slp(slp_real_data)
    labels.save(tmp_path / "test.pkg.slp", embed=True)
    pkg = load_slp(tmp_path / "test.pkg.slp")

    split1, split2 = pkg.split(n=0.8)
    assert len(split1) == 8
    assert len(split2) == 2
    assert split1.video.filename == (tmp_path / "test.pkg.slp").as_posix()
    assert split2.video.filename == (tmp_path / "test.pkg.slp").as_posix()
    assert (
        split1.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert (
        split2.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )

    split1.save(tmp_path / "split1.pkg.slp", embed=True)
    split2.save(tmp_path / "split2.pkg.slp", embed=True)
    assert pkg.video.filename == (tmp_path / "test.pkg.slp").as_posix()
    assert (
        Path(split1.video.filename).as_posix()
        == (tmp_path / "split1.pkg.slp").as_posix()
    )
    assert (
        Path(split2.video.filename).as_posix()
        == (tmp_path / "split2.pkg.slp").as_posix()
    )
    assert (
        split1.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert (
        split2.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )

    split1_ = load_slp(tmp_path / "split1.pkg.slp")
    split2_ = load_slp(tmp_path / "split2.pkg.slp")
    assert len(split1_) == 8
    assert len(split2_) == 2
    assert (
        Path(split1_.video.filename).as_posix()
        == (tmp_path / "split1.pkg.slp").as_posix()
    )
    assert (
        Path(split2_.video.filename).as_posix()
        == (tmp_path / "split2.pkg.slp").as_posix()
    )
    assert (
        split1_.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert (
        split2_.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )


def test_make_training_splits(slp_real_data):
    labels = load_slp(slp_real_data)
    assert len(labels.user_labeled_frames) == 5

    train, val = labels.make_training_splits(0.8)
    assert len(train) == 4
    assert len(val) == 1

    train, val = labels.make_training_splits(3)
    assert len(train) == 3
    assert len(val) == 2

    train, val = labels.make_training_splits(0.8, 0.2)
    assert len(train) == 4
    assert len(val) == 1

    train, val, test = labels.make_training_splits(0.8, 0.1, 0.1)
    assert len(train) == 4
    assert len(val) == 1
    assert len(test) == 1

    train, val, test = labels.make_training_splits(n_train=0.6, n_test=1)
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1

    train, val, test = labels.make_training_splits(n_train=1, n_val=1, n_test=1)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1

    train, val, test = labels.make_training_splits(n_train=0.4, n_val=0.4, n_test=0.2)
    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 1


def test_make_training_splits_save(slp_real_data, tmp_path):
    labels = load_slp(slp_real_data)

    train, val, test = labels.make_training_splits(0.6, 0.2, 0.2, save_dir=tmp_path)

    train_, val_, test_ = (
        load_slp(tmp_path / "train.pkg.slp"),
        load_slp(tmp_path / "val.pkg.slp"),
        load_slp(tmp_path / "test.pkg.slp"),
    )

    assert len(train_) == len(train)
    assert len(val_) == len(val)
    assert len(test_) == len(test)

    assert train_.provenance["source_labels"] == slp_real_data
    assert val_.provenance["source_labels"] == slp_real_data
    assert test_.provenance["source_labels"] == slp_real_data


@pytest.mark.parametrize("embed", [True, False])
def test_make_training_splits_save(slp_real_data, tmp_path, embed):
    labels = load_slp(slp_real_data)

    train, val, test = labels.make_training_splits(
        0.6, 0.2, 0.2, save_dir=tmp_path, embed=embed
    )

    if embed:
        train_, val_, test_ = (
            load_slp(tmp_path / "train.pkg.slp"),
            load_slp(tmp_path / "val.pkg.slp"),
            load_slp(tmp_path / "test.pkg.slp"),
        )
    else:
        train_, val_, test_ = (
            load_slp(tmp_path / "train.slp"),
            load_slp(tmp_path / "val.slp"),
            load_slp(tmp_path / "test.slp"),
        )

    assert len(train_) == len(train)
    assert len(val_) == len(val)
    assert len(test_) == len(test)

    if embed:
        assert train_.provenance["source_labels"] == slp_real_data
        assert val_.provenance["source_labels"] == slp_real_data
        assert test_.provenance["source_labels"] == slp_real_data
    else:
        assert train_.video.filename == labels.video.filename
        assert val_.video.filename == labels.video.filename
        assert test_.video.filename == labels.video.filename

    if embed:
        for labels_ in [train_, val_, test_]:
            for lf in labels_:
                assert lf.image.shape == (384, 384, 1)


def test_labels_instances():
    labels = Labels()
    labels.append(
        LabeledFrame(
            video=Video("test.mp4"),
            frame_idx=0,
            instances=[
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=Skeleton(["A", "B"])
                )
            ],
        )
    )
    assert len(list(labels.instances)) == 1

    labels.append(
        LabeledFrame(
            video=labels.video,
            frame_idx=1,
            instances=[
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=labels.skeleton
                ),
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=labels.skeleton
                ),
            ],
        )
    )
    assert len(list(labels.instances)) == 3


def test_labels_rename_nodes(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]

    labels.rename_nodes({"head": "front", "abdomen": "back"})
    assert labels.skeleton.node_names == ["front", "back"]

    labels.skeletons.append(Skeleton(["A", "B"]))
    with pytest.raises(ValueError):
        labels.rename_nodes({"A": "a", "B": "b"})
    labels.rename_nodes({"A": "a", "B": "b"}, skeleton=labels.skeletons[1])
    assert labels.skeletons[1].node_names == ["a", "b"]


def test_labels_remove_nodes(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]
    assert_allclose(
        labels[0][0].numpy(), [[91.886988, 204.018843], [151.536969, 159.825034]]
    )

    labels.remove_nodes(["head"])
    assert labels.skeleton.node_names == ["abdomen"]
    assert_allclose(labels[0][0].numpy(), [[151.536969, 159.825034]])

    for inst in labels.instances:
        assert inst.numpy().shape == (1, 2)

    labels.skeletons.append(Skeleton())
    with pytest.raises(ValueError):
        labels.remove_nodes(["head"])


def test_labels_reorder_nodes(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]
    assert_allclose(
        labels[0][0].numpy(), [[91.886988, 204.018843], [151.536969, 159.825034]]
    )

    labels.reorder_nodes(["abdomen", "head"])
    assert labels.skeleton.node_names == ["abdomen", "head"]
    assert_allclose(
        labels[0][0].numpy(), [[151.536969, 159.825034], [91.886988, 204.018843]]
    )

    labels.skeletons.append(Skeleton())
    with pytest.raises(ValueError):
        labels.reorder_nodes(["head", "abdomen"])


def test_labels_replace_skeleton(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]
    inst = labels[0][0]
    assert_allclose(inst.numpy(), [[91.886988, 204.018843], [151.536969, 159.825034]])

    # Replace with full mapping
    new_skel = Skeleton(["ABDOMEN", "HEAD"])
    labels.replace_skeleton(new_skel, node_map={"abdomen": "ABDOMEN", "head": "HEAD"})
    assert labels.skeleton == new_skel
    inst = labels[0][0]
    assert inst.skeleton == new_skel
    assert_allclose(inst.numpy(), [[151.536969, 159.825034], [91.886988, 204.018843]])

    # Replace with partial (inferred) mapping
    new_skel = Skeleton(["x", "ABDOMEN"])
    labels.replace_skeleton(new_skel)
    assert labels.skeleton == new_skel
    inst = labels[0][0]
    assert inst.skeleton == new_skel
    assert_allclose(inst.numpy(), [[np.nan, np.nan], [151.536969, 159.825034]])

    # Replace with no mapping
    new_skel = Skeleton(["front", "back"])
    labels.replace_skeleton(new_skel)
    assert labels.skeleton == new_skel
    inst = labels[0][0]
    assert inst.skeleton == new_skel
    assert_allclose(inst.numpy(), [[np.nan, np.nan], [np.nan, np.nan]])


def test_labels_trim(centered_pair, tmpdir):
    labels = load_slp(centered_pair)

    new_path = tmpdir / "trimmed.slp"
    trimmed_labels = labels.trim(new_path, np.arange(100, 200))
    assert len(trimmed_labels) == 100
    assert trimmed_labels.video.filename == Path(new_path).with_suffix(".mp4")
    assert trimmed_labels.video.shape == (100, 384, 384, 1)
    assert trimmed_labels[0].frame_idx == 0
    assert_equal(trimmed_labels[0].numpy(), labels[(labels.video, 100)].numpy())

    labels.videos.append(Video.from_filename("fake.mp4"))
    with pytest.raises(ValueError):
        labels.trim(new_path, np.arange(100, 200))

    labels.trim(new_path, np.arange(100, 200), video=0)


def test_labels_sessions():
    labels = Labels()
    assert labels.sessions == []
    labels.__str__()

    session = RecordingSession()
    labels.sessions.append(session)
    assert labels.sessions == [session]
    labels.__str__()


def test_labels_numpy_user_instances():
    """Test the user_instances parameter in Labels.numpy method."""
    # Create a simple skeleton for testing
    skeleton = Skeleton(["A", "B"])
    video = Video(filename="test_video.mp4")

    # Create 3 tracks
    track1 = Track(name="track1")
    track2 = Track(name="track2")
    track3 = Track(name="track3")

    # Create 3 frames with different combinations of user and predicted instances
    frames = []

    # Frame 0: User and predicted instance with same track (track1)
    #          Predicted instance with track2
    lf0 = LabeledFrame(video=video, frame_idx=0)
    user_inst0 = Instance([[10, 10], [20, 20]], skeleton=skeleton, track=track1)
    # Create predicted instance with point scores
    pred_inst0 = PredictedInstance.from_numpy(
        [[11, 11], [21, 21]],
        skeleton=skeleton,
        point_scores=[0.8, 0.8],  # Adding point scores
        score=0.8,
        track=track1,
    )
    pred_inst1 = PredictedInstance.from_numpy(
        [[30, 30], [40, 40]],
        skeleton=skeleton,
        point_scores=[0.9, 0.9],  # Adding point scores
        score=0.9,
        track=track2,
    )
    lf0.instances = [user_inst0, pred_inst0, pred_inst1]
    frames.append(lf0)

    # Frame 1: User instance linked to predicted via from_predicted (no track)
    #          Another predicted instance with track3
    lf1 = LabeledFrame(video=video, frame_idx=1)
    pred_inst2 = PredictedInstance.from_numpy(
        [[12, 12], [22, 22]],
        skeleton=skeleton,
        point_scores=[0.7, 0.7],  # Adding point scores
        score=0.7,
    )
    user_inst1 = Instance([[15, 15], [25, 25]], skeleton=skeleton)
    user_inst1.from_predicted = pred_inst2
    pred_inst3 = PredictedInstance.from_numpy(
        [[35, 35], [45, 45]],
        skeleton=skeleton,
        point_scores=[0.85, 0.85],  # Adding point scores
        score=0.85,
        track=track3,
    )
    lf1.instances = [pred_inst2, user_inst1, pred_inst3]
    frames.append(lf1)

    # Frame 2: Single user instance and single predicted instance (trivial case)
    lf2 = LabeledFrame(video=video, frame_idx=2)
    user_inst2 = Instance([[50, 50], [60, 60]], skeleton=skeleton)
    pred_inst4 = PredictedInstance.from_numpy(
        [[55, 55], [65, 65]],
        skeleton=skeleton,
        point_scores=[0.95, 0.95],  # Adding point scores
        score=0.95,
    )
    lf2.instances = [user_inst2, pred_inst4]
    frames.append(lf2)

    # Create labels with all these frames
    labels = Labels(labeled_frames=frames)
    labels.tracks = [track1, track2, track3]

    # Test 1: With user_instances=True (default)
    # For tracked instances
    tracks = labels.numpy(untracked=False)
    # Shape should be (3 frames, 3 tracks, 2 nodes, 2 coordinates)
    assert tracks.shape == (3, 3, 2, 2)
    # Track1 in frame0 should be the user instance
    assert_equal(tracks[0, 0], [[10, 10], [20, 20]])
    # Track2 in frame0 should be the predicted instance
    assert_equal(tracks[0, 1], [[30, 30], [40, 40]])
    # Track3 in frame1 should be the predicted instance
    assert_equal(tracks[1, 2], [[35, 35], [45, 45]])

    # With confidence scores
    tracks_conf = labels.numpy(untracked=False, return_confidence=True)
    # Shape should be (3 frames, 3 tracks, 2 nodes, 3 values)
    assert tracks_conf.shape == (3, 3, 2, 3)
    # User instance should have confidence 1.0
    assert_equal(tracks_conf[0, 0, 0, 2], 1.0)
    assert_equal(tracks_conf[0, 0, 1, 2], 1.0)
    # Predicted instance should have its original confidence
    assert_allclose(tracks_conf[0, 1, 0, 2], 0.9)

    # Test 2: For untracked instances
    untracked = labels.numpy(untracked=True)
    # Shape should be (3 frames, max_instances_per_frame=2 [user and predicted], 2 nodes, 2 coordinates)
    assert untracked.shape == (3, 2, 2, 2)
    # Frame0 should have user instance first, then predicted instance track2
    assert_equal(untracked[0, 0], [[10, 10], [20, 20]])
    assert_equal(untracked[0, 1], [[30, 30], [40, 40]])
    # Frame1 should have user instance first, then predicted instance track3
    assert_equal(untracked[1, 0], [[15, 15], [25, 25]])
    assert_equal(untracked[1, 1], [[35, 35], [45, 45]])
    # Frame2 should have both instances
    assert_equal(untracked[2, 0], [[50, 50], [60, 60]])
    assert_equal(untracked[2, 1], [[55, 55], [65, 65]])

    # Test 3: with return_confidence=True
    untracked_conf = labels.numpy(untracked=True, return_confidence=True)
    # Shape should be (3 frames, max_instances_per_frame=2 [user and predicted], 2 nodes, 3 values)
    assert untracked_conf.shape == (3, 2, 2, 3)
    # Frame0 should have user instance first, then predicted instance track2
    assert_equal(untracked_conf[0, 0, 0, 2], 1.0)
    assert_equal(untracked_conf[0, 0, 1, 2], 1.0)
    # Predicted instance should have its original confidence
    assert_allclose(untracked_conf[0, 1, 0, 2], 0.9)
    # Frame1 should have user instance first, then predicted instance track3
    assert_equal(untracked_conf[1, 0, 0, 2], 1.0)
    assert_equal(untracked_conf[1, 0, 1, 2], 1.0)
    # Predicted instance should have its original confidence
    assert_allclose(untracked_conf[1, 1, 0, 2], 0.85)
    # Frame2 should have both instances
    assert_equal(untracked_conf[2, 0, 0, 2], 1.0)
    assert_equal(untracked_conf[2, 0, 1, 2], 1.0)
    assert_allclose(untracked_conf[2, 1, 0, 2], 0.95)

    # Test 4: With user_instances=False
    # For tracked instances
    pred_only_tracks = labels.numpy(untracked=False, user_instances=False)
    # Shape should be (3 frames, 3 tracks, 2 nodes, 2 coordinates)
    assert pred_only_tracks.shape == (3, 3, 2, 2)
    # Track1 in frame0 should be the predicted instance now
    assert_equal(pred_only_tracks[0, 0], [[11, 11], [21, 21]])

    # Test 5: For untracked instances with user_instances=False
    pred_only_untracked = labels.numpy(untracked=True, user_instances=False)
    # Shape should be (3 frames, max_predicted_instances_per_frame=2, 2 nodes, 2 coordinates)
    assert pred_only_untracked.shape == (3, 2, 2, 2)
    # Frame0 should have both predicted instances
    assert_equal(pred_only_untracked[0, 0], [[11, 11], [21, 21]])
    assert_equal(pred_only_untracked[0, 1], [[30, 30], [40, 40]])
    # Frame1 should have both predicted instances
    assert_equal(pred_only_untracked[1, 0], [[12, 12], [22, 22]])
    assert_equal(pred_only_untracked[1, 1], [[35, 35], [45, 45]])
    # Frame2 should have only one predicted instance
    assert_equal(pred_only_untracked[2, 0], [[55, 55], [65, 65]])
    assert np.isnan(pred_only_untracked[2, 1, 0, 0])  # Second slot should be empty

    # Test 6: Single instance project (the trivial case)
    # Create a single instance project
    single_frames = []
    lf_single = LabeledFrame(video=video, frame_idx=0)
    user_inst_single = Instance([[70, 70], [80, 80]], skeleton=skeleton)
    pred_inst_single = PredictedInstance.from_numpy(
        [[75, 75], [85, 85]],
        skeleton=skeleton,
        point_scores=[0.9, 0.9],  # Adding point scores
        score=0.9,
    )
    lf_single.instances = [user_inst_single, pred_inst_single]
    single_frames.append(lf_single)

    labels_single = Labels(labeled_frames=single_frames)

    # For single instance projects, user instances should be preferred
    single_tracks = labels_single.numpy()
    # Shape should be (1 frame, 1 instance, 2 nodes, 2 coordinates)
    assert single_tracks.shape == (1, 1, 2, 2)
    # Should be the user instance
    assert_equal(single_tracks[0, 0], [[70, 70], [80, 80]])

    # With user_instances=False
    single_tracks_pred = labels_single.numpy(user_instances=False)
    # Should be the predicted instance
    assert_equal(single_tracks_pred[0, 0], [[75, 75], [85, 85]])


def test_update_from_numpy(labels_predictions):
    """Test updating instances from numpy arrays."""
    import numpy as np
    from sleap_io import Track, PredictedInstance

    # Get original numpy representation
    original_arr = labels_predictions.numpy(return_confidence=True)

    # Modify the numpy array - shift all points by (10, 20)
    modified_arr = original_arr.copy()
    modified_arr[:, :, :, 0] += 10  # shift x by 10
    modified_arr[:, :, :, 1] += 20  # shift y by 20

    # Update the labels with modified array
    labels_predictions.update_from_numpy(modified_arr)

    # Get the updated numpy representation
    updated_arr = labels_predictions.numpy(return_confidence=True)

    # Verify points were updated correctly
    assert np.allclose(
        updated_arr[:, :, :, :2], modified_arr[:, :, :, :2], equal_nan=True
    )

    # Test creating new instances
    # Make a copy of the original labels
    import copy

    labels_copy = copy.deepcopy(labels_predictions)

    # Create new data for a non-existent track
    new_track = Track("new_track")
    labels_copy.tracks.append(new_track)

    # Add a new track to the array
    n_frames, n_tracks, n_nodes, n_dims = original_arr.shape
    new_arr = np.full(
        (n_frames, n_tracks + 1, n_nodes, n_dims), np.nan, dtype="float32"
    )
    new_arr[:, :-1] = modified_arr

    # Add data for the new track in the first frame
    new_arr[0, -1, :, 0] = 100  # x coordinates
    new_arr[0, -1, :, 1] = 200  # y coordinates
    new_arr[0, -1, :, 2] = 0.95  # confidence

    # Update with the new array
    tracks = labels_copy.tracks
    labels_copy.update_from_numpy(new_arr, tracks=tracks)

    # Verify the new instance was created
    first_frame = labels_copy.labeled_frames[0]

    # Check if a track with the name "new_track" exists in any of the first frame's instances
    has_new_track = any(
        inst.track and inst.track.name == "new_track" for inst in first_frame.instances
    )

    # This should now pass
    assert has_new_track, "New track instance not found in frame instances"

    # Also check in predicted_instances
    has_new_track_pred = any(
        inst.track and inst.track.name == "new_track"
        for inst in first_frame.predicted_instances
    )

    # This should also pass
    assert has_new_track_pred, "New track instance not found in predicted_instances"


def test_update_from_numpy_errors():
    """Test error handling in update_from_numpy."""
    import numpy as np
    import pytest
    from sleap_io import Labels, Video, Skeleton, Track

    # Create a basic labels object
    labels = Labels()
    labels.videos.append(Video("test1.mp4"))
    labels.videos.append(Video("test2.mp4"))
    labels.skeletons.append(Skeleton(["A", "B"]))

    # 1. Test array with incorrect dimensions
    with pytest.raises(ValueError, match="Array must have 4 dimensions"):
        # Create a 3D array instead of 4D
        invalid_arr = np.zeros((2, 3, 2))
        labels.update_from_numpy(invalid_arr)

    # 2. Test multiple videos but no video specified
    with pytest.raises(ValueError, match="Video must be specified"):
        # Valid 4D array but no video specified with multiple videos
        valid_arr = np.zeros((2, 1, 2, 3))
        labels.update_from_numpy(valid_arr)

    # 3. Test tracks mismatch
    with pytest.raises(ValueError, match="Number of tracks in array .* doesn't match"):
        # Valid array with more tracks than in labels
        labels.tracks = [Track("track1")]
        valid_arr = np.zeros((2, 2, 2, 3))  # 2 tracks in array, 1 in labels
        labels.update_from_numpy(valid_arr, video=labels.videos[0])

    # 4. Test no skeletons
    labels_no_skeleton = Labels()
    labels_no_skeleton.videos.append(Video("test.mp4"))
    # Add a track to match the array dimension to avoid the track mismatch error
    labels_no_skeleton.tracks.append(Track("track1"))
    with pytest.raises(ValueError, match="No skeletons available"):
        valid_arr = np.zeros((2, 1, 2, 3))
        labels_no_skeleton.update_from_numpy(valid_arr)


def test_update_from_numpy_no_create_missing(labels_predictions):
    """Test update_from_numpy with create_missing=False."""

    # Get original numpy representation and copy labels
    original_arr = labels_predictions.numpy(return_confidence=True)
    labels_copy = copy.deepcopy(labels_predictions)

    # Count initial frames
    initial_frame_count = len(labels_copy.labeled_frames)

    # Create modified array with new frame indices
    # This extends the array to have frames beyond what currently exists
    n_frames, n_tracks, n_nodes, n_dims = original_arr.shape
    extended_arr = np.full(
        (n_frames + 3, n_tracks, n_nodes, n_dims), np.nan, dtype="float32"
    )
    extended_arr[:n_frames] = original_arr

    # Add data for a new frame that doesn't exist yet
    extended_arr[n_frames, 0, :, 0] = 100  # x coordinates
    extended_arr[n_frames, 0, :, 1] = 200  # y coordinates
    extended_arr[n_frames, 0, :, 2] = 0.9  # confidence

    # Update with create_missing=False
    labels_copy.update_from_numpy(extended_arr, create_missing=False)

    # The frame count should not have changed
    assert (
        len(labels_copy.labeled_frames) == initial_frame_count
    ), "New frames should not be created with create_missing=False"


def test_update_from_numpy_update_user_instances(labels_predictions):
    """Test updating user instances with update_from_numpy."""

    # Get original data
    labels_copy = copy.deepcopy(labels_predictions)
    video = labels_copy.videos[0]
    skeleton = labels_copy.skeletons[0]
    tracks = labels_copy.tracks

    # Find an existing frame to modify
    existing_frame = labels_copy.labeled_frames[0]
    frame_idx = existing_frame.frame_idx

    # Clear and add our test instance
    existing_frame.instances = []

    # Create points data with the right shape for this skeleton
    points_data = np.full((len(skeleton.nodes), 2), np.nan)
    # Set only a few points with valid data
    points_data[0] = [50.0, 60.0]
    points_data[1] = [70.0, 80.0]

    # Add a user instance with the first track
    user_instance = Instance(points=points_data, skeleton=skeleton, track=tracks[0])
    existing_frame.instances = [user_instance]

    # Create array with modified data for this frame
    n_frames = 1
    arr = np.full(
        (n_frames, len(tracks), len(skeleton.nodes), 3), np.nan, dtype="float32"
    )

    # Set coordinates for first track (index 0)
    arr[0, 0, 0, 0] = 150.0  # New x for first point
    arr[0, 0, 0, 1] = 160.0  # New y for first point
    arr[0, 0, 1, 0] = 170.0  # New x for second point
    arr[0, 0, 1, 1] = 180.0  # New y for second point
    arr[0, 0, :, 2] = 1.0  # Confidence scores

    # Update with our array (which will update the first frame in the labels)
    labels_copy.update_from_numpy(arr, video=video)

    # Find the updated instance in the first frame
    updated_instance = None
    for inst in labels_copy.labeled_frames[0].instances:
        if inst.track == tracks[0]:
            updated_instance = inst
            break

    assert updated_instance is not None, "User instance not found in updated frame"

    # Verify the user instance was updated
    points = updated_instance.numpy()
    assert np.allclose(
        points[0], [150.0, 160.0]
    ), "User instance first point should be updated"
    assert np.allclose(
        points[1], [170.0, 180.0]
    ), "User instance second point should be updated"


def test_update_from_numpy_without_confidence():
    """Test update_from_numpy with array without confidence scores."""

    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array WITHOUT confidence scores (only x,y)
    arr = np.full((1, 1, 2, 2), np.nan, dtype="float32")
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 1, 0] = 30.0  # x for second point
    arr[0, 0, 1, 1] = 40.0  # y for second point

    # Update with the array
    labels.update_from_numpy(arr)

    # Verify a new instance was created with correct points
    assert len(labels[0].instances) == 1, "Should create one instance"
    points = labels[0].instances[0].numpy()
    assert np.allclose(points[0], [10.0, 20.0]), "First point should be set correctly"
    assert np.allclose(points[1], [30.0, 40.0]), "Second point should be set correctly"


def test_update_from_numpy_int_video_index():
    """Test update_from_numpy with integer video index."""
    import numpy as np
    from sleap_io import Labels, Video, Skeleton, Track

    # Create a labels object with multiple videos
    labels = Labels()
    video1 = Video("test1.mp4")
    video2 = Video("test2.mp4")
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")

    labels.videos.append(video1)
    labels.videos.append(video2)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create array with data
    arr = np.full((1, 1, 2, 3), np.nan, dtype="float32")
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 0, 2] = 1.0  # confidence

    # Update using the second video by index (1)
    labels.update_from_numpy(arr, video=1)

    # Verify a new frame was created for the second video
    assert len(labels.labeled_frames) == 1, "Should create one frame"
    assert (
        labels.labeled_frames[0].video == video2
    ), "Should create frame for second video"


def test_update_from_numpy_special_case():
    """Test the special case handling in update_from_numpy."""

    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("new_track")  # This will be the "new" track

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)  # Add both tracks

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than we'll specify in the tracks parameter
    arr = np.full((1, 2, 2, 3), np.nan, dtype="float32")
    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 0, 2] = 1.0  # confidence

    # Data for "additional" track in last column
    arr[0, 1, 0, 0] = 30.0  # x for first point
    arr[0, 1, 0, 1] = 40.0  # y for first point
    arr[0, 1, 0, 2] = 0.9  # confidence

    # Update with ONLY the first track in the tracks parameter
    # This will trigger the special case where n_tracks_arr > len(tracks)
    tracks_subset = [track1, track2]  # Only specifying one track for a two-track array
    labels.update_from_numpy(arr, tracks=tracks_subset)

    # Verify instances were created for both tracks
    assert len(labels[0].instances) == 2, "Should create instances for both tracks"

    # Check if we have instances for both tracks
    track_names = [inst.track.name for inst in labels[0].instances if inst.track]
    assert "track1" in track_names, "Should have an instance for track1"
    assert "new_track" in track_names, "Should have an instance for new_track"


def test_update_from_numpy_confidence_scores():
    """Test updating confidence scores in existing predicted instances with update_from_numpy."""

    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create a frame with a predicted instance
    frame = LabeledFrame(video=video, frame_idx=0)
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.8]),
        score=0.75,
        track=track,
    )
    frame.instances.append(pred_inst)
    labels.append(frame)

    # Create array with updated confidence scores but same positions
    arr = np.full((1, 1, 2, 3), np.nan, dtype="float32")
    arr[0, 0, 0, 0] = 10.0  # same x
    arr[0, 0, 0, 1] = 20.0  # same y
    arr[0, 0, 0, 2] = 0.95  # updated confidence
    arr[0, 0, 1, 0] = 30.0  # same x
    arr[0, 0, 1, 1] = 40.0  # same y
    arr[0, 0, 1, 2] = 0.98  # updated confidence

    # Update the labels with the array
    labels.update_from_numpy(arr)

    # Verify the instance's confidence scores were updated
    updated_inst = labels[0].instances[0]
    assert isinstance(
        updated_inst, PredictedInstance
    ), "Instance should remain a PredictedInstance"
    assert np.isclose(
        updated_inst["A"]["score"], 0.95
    ), "First node confidence score should be updated"
    assert np.isclose(
        updated_inst["B"]["score"], 0.98
    ), "Second node confidence score should be updated"

    # Check with numpy method that includes scores
    points_with_scores = updated_inst.numpy(scores=True)
    assert np.isclose(
        points_with_scores[0, 2], 0.95
    ), "First node confidence should be updated in numpy output"
    assert np.isclose(
        points_with_scores[1, 2], 0.98
    ), "Second node confidence should be updated in numpy output"


def test_update_from_numpy_inferred_tracks():
    """Test update_from_numpy using tracks inferred from labels."""
    import numpy as np
    from sleap_io import Labels, Video, Skeleton, Track, LabeledFrame, PredictedInstance

    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("track2")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)

    # Create a frame with an existing instance
    frame = LabeledFrame(video=video, frame_idx=0)
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.8]),
        score=0.75,
        track=track1,
    )
    frame.instances.append(pred_inst)
    labels.append(frame)

    # Create array matching the number of tracks in labels
    arr = np.full((1, 2, 2, 3), np.nan, dtype="float32")
    # Data for first track
    arr[0, 0, 0, 0] = 15.0  # updated x
    arr[0, 0, 0, 1] = 25.0  # updated y
    arr[0, 0, 0, 2] = 0.9  # confidence

    # Data for second track
    arr[0, 1, 0, 0] = 50.0  # x
    arr[0, 1, 0, 1] = 60.0  # y
    arr[0, 1, 0, 2] = 0.85  # confidence

    # Update WITHOUT specifying tracks explicitly - should use the tracks from labels
    labels.update_from_numpy(arr)

    # Verify both tracks have instances
    instances = labels[0].instances
    assert len(instances) == 2, "Should have two instances"

    # Find instance with track1
    track1_instance = next((inst for inst in instances if inst.track == track1), None)
    assert track1_instance is not None, "No instance found for track1"
    assert np.allclose(
        track1_instance.numpy()[0], [15.0, 25.0]
    ), "First track instance not updated correctly"

    # Find instance with track2
    track2_instance = next((inst for inst in instances if inst.track == track2), None)
    assert track2_instance is not None, "No instance created for track2"
    assert np.allclose(
        track2_instance.numpy()[0], [50.0, 60.0]
    ), "Second track instance not created correctly"


def test_update_from_numpy_special_case_new_track():
    """Test the special case for adding a new track in update_from_numpy."""
    import numpy as np
    from sleap_io import Labels, Video, Skeleton, Track, LabeledFrame

    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    new_track = Track("new_track")  # This will be added later

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than current labels.tracks
    arr = np.full((1, 2, 2, 3), np.nan, dtype="float32")
    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 0, 2] = 0.9  # confidence

    # Data for new track
    arr[0, 1, 0, 0] = 30.0  # x for first point
    arr[0, 1, 0, 1] = 40.0  # y for first point
    arr[0, 1, 0, 2] = 0.8  # confidence

    # Now add the new track to the tracks list
    labels.tracks.append(new_track)

    # Update with the array - this should trigger the special case
    tracks = labels.tracks
    labels.update_from_numpy(arr, tracks=tracks)

    # Verify the instance for the new track was created
    assert len(labels[0].instances) == 2, "Should create instances for both tracks"

    # Find the new track instance
    new_track_instance = next(
        (inst for inst in labels[0].instances if inst.track == new_track), None
    )

    # Verify it was created and has the right data
    assert new_track_instance is not None, "New track instance should be created"
    assert np.allclose(
        new_track_instance.numpy()[0], [30.0, 40.0]
    ), "New track data should be set correctly"
    assert np.isclose(
        new_track_instance.numpy(scores=True)[0, 2], 0.8
    ), "New track confidence should be set"


def test_update_from_numpy_nan_handling():
    """Test handling of NaN values in update_from_numpy."""
    import numpy as np
    from sleap_io import Labels, Video, Skeleton, Track, LabeledFrame, PredictedInstance

    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B", "C"])
    track = Track("track1")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create a frame with a predicted instance that has some points
    frame = LabeledFrame(video=video, frame_idx=0)
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.8, 0.9]),
        score=0.8,
        track=track,
    )
    frame.instances.append(pred_inst)
    labels.append(frame)

    # Create array with some NaN values
    arr = np.full((1, 1, 3, 3), np.nan, dtype="float32")
    # First point has values
    arr[0, 0, 0, 0] = 15.0  # updated x
    arr[0, 0, 0, 1] = 25.0  # updated y
    arr[0, 0, 0, 2] = 0.95  # updated confidence

    # Second point has NaN (should not update the existing point)
    arr[0, 0, 1, 0] = np.nan
    arr[0, 0, 1, 1] = np.nan
    arr[0, 0, 1, 2] = np.nan

    # Third point has values
    arr[0, 0, 2, 0] = 55.0  # updated x
    arr[0, 0, 2, 1] = 65.0  # updated y
    arr[0, 0, 2, 2] = 0.98  # updated confidence

    # Update the labels with the array
    labels.update_from_numpy(arr)

    # Verify only non-NaN values were updated
    updated_inst = labels[0].instances[0]
    points = updated_inst.numpy()

    # First point should be updated
    assert np.allclose(points[0], [15.0, 25.0]), "First point should be updated"

    # Second point should remain unchanged
    assert np.allclose(points[1], [30.0, 40.0]), "Second point should remain unchanged"

    # Third point should be updated
    assert np.allclose(points[2], [55.0, 65.0]), "Third point should be updated"

    # Check confidence scores
    if isinstance(updated_inst, PredictedInstance):
        scores = updated_inst.numpy(scores=True)[:, 2]
        assert np.isclose(scores[0], 0.95), "First point confidence should be updated"
        assert np.isclose(
            scores[1], 0.8
        ), "Second point confidence should remain unchanged"
        assert np.isclose(scores[2], 0.98), "Third point confidence should be updated"
