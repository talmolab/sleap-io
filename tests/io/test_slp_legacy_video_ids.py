"""Tests for the video ids of the legacy (2.x) `/frames` table.

Readers of the 2.x layout work out how the video ids are numbered from the ids
themselves, and an embedded video's `videoN` number can equal another video's
position in the video list. The writer picks ids that are read back correctly.
"""

from __future__ import annotations

import shutil
import warnings

import h5py
import numpy as np
import pytest
from numpy.testing import assert_equal

from sleap_io import (
    Instance,
    LabeledFrame,
    Labels,
    Skeleton,
    Video,
    load_slp,
    save_slp,
)
from sleap_io.io.slp import (
    _decode_video_ids,
    _video_ids_are_indices,
    _written_video_id_map,
    read_negative_frames,
    write_negative_frames,
)

# Keep the videos pointing at the package they are embedded in, so that they are
# written under that package's `videoN` numbers.
KEEP_PACKAGE = dict(embed=False, restore_original_videos=False)


def _frame(video: Video, frame_idx: int, skeleton: Skeleton, x: float = 1.0):
    """Return a labeled frame with one instance."""
    return LabeledFrame(
        video=video,
        frame_idx=frame_idx,
        instances=[Instance.from_numpy(np.array([[x, 2.0]]), skeleton=skeleton)],
    )


def _where(labels: Labels) -> list[tuple[int, int, bool]]:
    """Return (video position, frame index, is negative) of every frame."""
    return [
        (labels.videos.index(lf.video), lf.frame_idx, lf.is_negative) for lf in labels
    ]


@pytest.fixture
def videos(slp_legacy_negative_frames):
    """An external video and two embedded ones, stored as `video0` and `video4`."""
    package = load_slp(slp_legacy_negative_frames)
    video0, video4 = package.videos
    assert video0.backend.dataset == "video0/video"
    assert video4.backend.dataset == "video4/video"
    external = Video("external.mp4", open_backend=False)
    return external, video0, video4


def _save_legacy(labels: Labels, path, lazy: bool, tmp_path) -> None:
    """Save labels in the legacy layout, from eager or lazily loaded labels."""
    if lazy:
        # Go through a 3.0 file, which stores list indices, to get lazy labels.
        through = tmp_path / "through.slp"
        save_slp(labels, str(through), **KEEP_PACKAGE)
        labels = load_slp(str(through), lazy=True)
    save_slp(labels, str(path), legacy_format=True, **KEEP_PACKAGE)


@pytest.mark.parametrize("lazy", [False, True])
def test_colliding_ids_are_stored_as_list_indices(tmp_path, videos, lazy):
    """An embedded number equal to another video's position doesn't collide."""
    external, video0, _ = videos
    skeleton = Skeleton(["a"])
    labels = Labels(
        [_frame(external, 3, skeleton), _frame(video0, 0, skeleton)],
        videos=[external, video0],
    )
    path = tmp_path / "legacy.slp"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _save_legacy(labels, path, lazy, tmp_path)

    with h5py.File(path, "r") as f:
        # The external video is at position 0 and the embedded one is `video0`:
        # their own numbers would both be 0.
        assert_equal(_written_video_id_map(f, labels.videos), [0, 0])
        assert_equal(f["frames"]["video"], [0, 1])
        assert "video_ids" not in f["frames"].attrs
    for reload_lazy in (False, True):
        loaded = load_slp(str(path), lazy=reload_lazy)
        assert _where(loaded) == [(0, 3, False), (1, 0, False)]


@pytest.mark.parametrize("lazy", [False, True])
def test_ids_no_rule_can_read_are_marked(tmp_path, videos, lazy):
    """Ids that older readers can't work out are stored by position and marked."""
    external, video0, _ = videos
    skeleton = Skeleton(["a"])
    negative = LabeledFrame(video=external, frame_idx=5, is_negative=True)
    labels = Labels(
        [_frame(external, 3, skeleton), negative], videos=[external, video0]
    )
    path = tmp_path / "legacy.slp"
    with pytest.warns(UserWarning, match="can't be numbered in the legacy SLP layout"):
        _save_legacy(labels, path, lazy, tmp_path)

    with h5py.File(path, "r") as f:
        assert_equal(f["frames"]["video"], [0, 0])
        assert f["frames"].attrs["video_ids"] == "index"
        assert_equal(f["negative_frames"]["video_id"], [0])
        assert f["negative_frames"].attrs["video_ids"] == "index"
    for reload_lazy in (False, True):
        loaded = load_slp(str(path), lazy=reload_lazy)
        assert _where(loaded) == [(0, 3, False), (0, 5, True)]

    # Without the mark the ids are worked out from the numbers, which puts the
    # frames on the embedded video: the reason the mark is needed.
    with h5py.File(path, "r+") as f:
        del f["frames"].attrs["video_ids"]
        del f["negative_frames"].attrs["video_ids"]
    assert _where(load_slp(str(path))) == [(1, 3, False), (1, 5, True)]


@pytest.mark.parametrize("lazy", [False, True])
def test_embedded_numbers_are_kept_when_only_they_read_back(tmp_path, videos, lazy):
    """The embedded numbers are stored when list indices would be misread."""
    _, video0, video4 = videos
    skeleton = Skeleton(["a"])
    # `video4` is at position 0 and `video0` at position 1. A frame on position 0
    # alone would be read onto `video0` if it were stored as 0.
    labels = Labels([_frame(video4, 1, skeleton)], videos=[video4, video0])
    path = tmp_path / "legacy.slp"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _save_legacy(labels, path, lazy, tmp_path)

    with h5py.File(path, "r") as f:
        assert_equal(f["frames"]["video"], [4])
        assert "video_ids" not in f["frames"].attrs
    for reload_lazy in (False, True):
        loaded = load_slp(str(path), lazy=reload_lazy)
        assert _where(loaded) == [(0, 1, False)]
        assert loaded.videos[0].backend.dataset == "video4/video"


def test_marked_table_is_read_by_position(tmp_path, slp_legacy_negative_frames):
    """A marked 2.x table is read by position, whatever its ids look like."""
    path = tmp_path / "marked.slp"
    save_slp(
        load_slp(slp_legacy_negative_frames),
        str(path),
        legacy_format=True,
        **KEEP_PACKAGE,
    )
    # The package's videos are stored as `video0` and `video4`. Store every id
    # as 0 and swap the two videos in the list: unmarked, 0 is read as the
    # number of `video0`, now at position 1; marked, as position 0.
    with h5py.File(path, "r+") as f:
        frames = f["frames"][()]
        frames["video"] = 0
        f["frames"][...] = frames
        negative = f["negative_frames"][()]
        negative["video_id"] = 0
        f["negative_frames"][...] = negative
        rows = f["videos_json"][()]
        del f["videos_json"]
        f.create_dataset("videos_json", data=rows[::-1])

    # Unmarked: 0 is the number of `video0`, now at position 1.
    assert {where[0] for where in _where(load_slp(str(path)))} == {1}

    for mark in ("index", np.bytes_("index")):
        with h5py.File(path, "r+") as f:
            f["frames"].attrs["video_ids"] = mark
            f["negative_frames"].attrs["video_ids"] = mark
            assert _video_ids_are_indices(f["frames"])
        for lazy in (False, True):
            loaded = load_slp(str(path), lazy=lazy)
            assert {where[0] for where in _where(loaded)} == {0}
            assert [lf.is_negative for lf in loaded] == [False, True, False, True]

    with h5py.File(path, "r+") as f:
        f["frames"].attrs["video_ids"] = "sparse"
        f["negative_frames"].attrs["video_ids"] = "sparse"
        assert not _video_ids_are_indices(f["frames"])
        assert not _video_ids_are_indices(None)
    assert {where[0] for where in _where(load_slp(str(path)))} == {1}

    # Tables that disagree on the mark are reported.
    with h5py.File(path, "r+") as f:
        f["frames"].attrs["video_ids"] = "index"
    with pytest.warns(UserWarning, match="don't agree on how their video ids"):
        load_slp(str(path))


def test_decode_video_ids():
    """The rule for unmarked tables: exact list indices, else embedded numbers."""
    # Exactly 0..n-1: positions, whatever the embedded numbers are.
    assert_equal(_decode_video_ids([1, 0, 1], np.array([4, 0])), [1, 0, 1])
    # Otherwise embedded numbers map to positions, and other ids are kept.
    assert_equal(_decode_video_ids([15, 29, 15], np.array([0, 15, 29])), [1, 2, 1])
    assert_equal(_decode_video_ids([2], np.array([0, 15, 29])), [2])
    # Two videos under one id: the last one wins.
    assert_equal(_decode_video_ids([0], np.array([0, 0])), [1])
    assert_equal(_decode_video_ids([], np.array([0, 4])), [])


@pytest.mark.parametrize(
    "fixture, kwargs",
    [
        ("slp_typical", {}),
        ("slp_real_data", {}),
        ("slp_minimal_pkg", {}),
        ("slp_minimal_pkg", dict(embed=True)),
        ("slp_minimal_pkg", KEEP_PACKAGE),
        ("slp_legacy_negative_frames", {}),
        ("slp_legacy_negative_frames", dict(embed=True)),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_ordinary_saves_store_list_indices_unmarked(
    tmp_path, request, fixture, kwargs, lazy
):
    """Saves without a collision store the same ids as before, with no mark."""
    labels = load_slp(request.getfixturevalue(fixture), lazy=lazy)
    if lazy and kwargs.get("embed") is True:
        pytest.skip("Embedding frames materialises lazy labels.")
    expected = _where(labels)
    path = tmp_path / "legacy.slp"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        save_slp(labels, str(path), legacy_format=True, verbose=False, **kwargs)

    with h5py.File(path, "r") as f:
        positions = [where[0] for where in expected]
        assert_equal(f["frames"]["video"], positions)
        # The ids every earlier release wrote for these files.
        assert_equal(
            f["frames"]["video"],
            _written_video_id_map(f, labels.videos)[positions],
        )
        assert "video_ids" not in f["frames"].attrs
    assert _where(load_slp(str(path))) == expected


def test_write_negative_frames_follows_the_frames_table(tmp_path, videos):
    """`write_negative_frames` numbers and marks its videos as `/frames` does."""
    external, video0, _ = videos
    skeleton = Skeleton(["a"])
    path = tmp_path / "negative.slp"
    with h5py.File(path, "w"):
        pass

    # No negative frames: nothing is written.
    labels = Labels([_frame(external, 3, skeleton)], videos=[external, video0])
    write_negative_frames(str(path), labels)
    with h5py.File(path, "r") as f:
        assert "negative_frames" not in f

    # Frames on both videos: list indices, read correctly without a mark.
    labels = Labels(
        [
            _frame(external, 3, skeleton),
            LabeledFrame(video=video0, frame_idx=1, is_negative=True),
        ],
        videos=[external, video0],
    )
    write_negative_frames(str(path), labels)
    assert read_negative_frames(str(path)) == {(1, 1)}
    with h5py.File(path, "r") as f:
        assert "video_ids" not in f["negative_frames"].attrs

    # Frames on the external video only: marked, and the dataset is replaced.
    labels = Labels(
        [LabeledFrame(video=external, frame_idx=5, is_negative=True)],
        videos=[external, video0],
    )
    with pytest.warns(UserWarning, match="can't be numbered in the legacy SLP layout"):
        write_negative_frames(str(path), labels)
    assert read_negative_frames(str(path)) == {(0, 5)}
    with h5py.File(path, "r") as f:
        assert f["negative_frames"].attrs["video_ids"] == "index"


def test_write_negative_frames_follows_an_older_frames_table(
    tmp_path, slp_legacy_negative_frames
):
    """Negative markers take the ids of a `/frames` table another writer wrote.

    The fixture was written by an older writer, which stores the embedded numbers
    `video0` and `video4`. This writer would store list indices for the same
    labels; the markers must follow the file's table, or they match no frame.
    """
    path = tmp_path / "older.pkg.slp"
    shutil.copy(slp_legacy_negative_frames, path)
    with h5py.File(path, "r+") as f:
        assert_equal(f["frames"]["video"], [0, 0, 4, 4])
        del f["negative_frames"]

    write_negative_frames(str(path), load_slp(slp_legacy_negative_frames))
    with h5py.File(path, "r") as f:
        assert sorted(f["negative_frames"][()].tolist()) == [(0, 1), (4, 1)]
        assert "video_ids" not in f["negative_frames"].attrs
    assert _where(load_slp(str(path))) == [
        (0, 0, False),
        (0, 1, True),
        (1, 0, False),
        (1, 1, True),
    ]


@pytest.mark.parametrize("lazy", [False, True])
def test_mixed_numbering_is_used_before_marking(
    tmp_path, slp_legacy_negative_frames, lazy
):
    """Some videos by list index and some by number, when only that reads back.

    Videos `P/video4`, `P/video0` and `Q/video0`, with frames on the first two.
    List indices are misread (0 is `Q/video0`'s number), and so are the embedded
    numbers (0 is shared, and the later video wins). Storing `P/video4` by its
    number and `P/video0` by its list index is read correctly by every reader
    that works the ids out from opened videos, so nothing is marked.
    """
    other = tmp_path / "other.pkg.slp"
    shutil.copy(slp_legacy_negative_frames, other)
    p_video0, p_video4 = load_slp(slp_legacy_negative_frames).videos
    q_video0, _ = load_slp(str(other)).videos
    skeleton = Skeleton(["a"])
    labels = Labels(
        [_frame(p_video4, 0, skeleton), _frame(p_video0, 1, skeleton, x=3.0)],
        videos=[p_video4, p_video0, q_video0],
    )
    path = tmp_path / "legacy.slp"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _save_legacy(labels, path, lazy, tmp_path)

    with h5py.File(path, "r") as f:
        assert_equal(_written_video_id_map(f, labels.videos), [4, 0, 0])
        assert_equal(f["frames"]["video"], [4, 1])
        assert "video_ids" not in f["frames"].attrs
    for reload_lazy in (False, True):
        loaded = load_slp(str(path), lazy=reload_lazy)
        assert _where(loaded) == [(0, 0, False), (1, 1, False)]
        assert loaded.videos[0].backend.dataset == "video4/video"
        assert loaded.videos[1].filename == p_video0.filename
