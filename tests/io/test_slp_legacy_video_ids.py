"""Tests for the video ids of the legacy (2.x) `/frames` table.

Readers of the 2.x layout work out how the video ids are numbered from the ids
themselves, and an embedded video's `videoN` number can equal another video's
position in the video list. The writer picks ids that are read back correctly.
"""

from __future__ import annotations

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
    # Two videos stored as `video0` and `video4`, and every frame on position 1.
    # Unmarked, id 1 is no embedded number and is kept as a position; so make
    # the ids ambiguous instead: all 0, with the videos swapped in the list.
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
            assert _video_ids_are_indices(f["frames"])
        for lazy in (False, True):
            loaded = load_slp(str(path), lazy=lazy)
            assert {where[0] for where in _where(loaded)} == {0}
            assert [lf.is_negative for lf in loaded] == [False, True, False, True]

    with h5py.File(path, "r+") as f:
        f["frames"].attrs["video_ids"] = "sparse"
        assert not _video_ids_are_indices(f["frames"])
        assert not _video_ids_are_indices(None)
    assert {where[0] for where in _where(load_slp(str(path)))} == {1}


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
