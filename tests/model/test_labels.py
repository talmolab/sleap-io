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
    load_slp,
    load_video,
)
from sleap_io.model.labels import Labels
import numpy as np
from pathlib import Path


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
