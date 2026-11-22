"""Tests that saving an SLP file goes through a temporary file.

A save writes to a temporary file in the target's folder and renames it over the
target when it is complete, so the target holds either the previous file or the
complete new one.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import sleap_io.io.slp as slp
from sleap_io import (
    Instance,
    LabeledFrame,
    Labels,
    PredictedInstance,
    Skeleton,
    Video,
    load_slp,
    save_slp,
)
from sleap_io.io.slp import ExportCancelled

# Replacing a file that is open fails on Windows, as removing it always has, so
# saving a package over itself with its videos open only works elsewhere.
replaces_open_files = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows can't replace a file that is open."
)


def _labels(n_frames: int = 2) -> Labels:
    """Return a few labeled frames on a video that doesn't exist."""
    skeleton = Skeleton(["a", "b"])
    video = Video("missing.mp4", open_backend=False)
    return Labels(
        [
            LabeledFrame(
                video=video,
                frame_idx=i,
                instances=[
                    Instance.from_numpy(
                        np.array([[1.0 + i, 2.0], [3.0, 4.0]]), skeleton=skeleton
                    )
                ],
            )
            for i in range(n_frames)
        ]
    )


def _folder(path: Path) -> list[str]:
    """Return the names of the files in a folder, hidden ones included."""
    return sorted(p.name for p in path.iterdir())


def _fail(*args, **kwargs):
    """Stand in for a writer that fails partway through a save."""
    raise RuntimeError("disk full")


def _refuse(*args, **kwargs):
    """Stand in for a rename of a file that is in use."""
    raise PermissionError("in use")


@pytest.mark.parametrize("legacy_format", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_failed_save_keeps_previous_file(tmp_path, monkeypatch, legacy_format, lazy):
    """A save that fails partway leaves the previous file byte for byte."""
    path = tmp_path / "labels.slp"
    save_slp(_labels(2), str(path), legacy_format=legacy_format)
    before = path.read_bytes()

    source = tmp_path / "source.slp"
    save_slp(_labels(5), str(source))
    labels = load_slp(str(source), lazy=lazy)

    # The annotation tables are written after the pose tables, near the end.
    monkeypatch.setattr(slp, "write_label_images", _fail)
    with pytest.raises(RuntimeError, match="disk full"):
        save_slp(labels, str(path), legacy_format=legacy_format)

    assert path.read_bytes() == before
    assert _folder(tmp_path) == ["labels.slp", "source.slp"]
    assert len(load_slp(str(path))) == 2


@pytest.mark.parametrize("lazy", [False, True])
def test_failed_first_save_leaves_nothing(tmp_path, monkeypatch, lazy):
    """A failed save to a new path leaves no partial file behind."""
    source = tmp_path / "source.slp"
    save_slp(_labels(), str(source))
    labels = load_slp(str(source), lazy=lazy)

    monkeypatch.setattr(slp, "write_label_images", _fail)
    with pytest.raises(RuntimeError, match="disk full"):
        save_slp(labels, str(tmp_path / "new.slp"))

    assert _folder(tmp_path) == ["source.slp"]


def test_refused_save_keeps_previous_file(tmp_path):
    """A save refused by a check before writing doesn't touch the target."""
    path = tmp_path / "labels.slp"
    save_slp(_labels(2), str(path))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="compression must be one of"):
        save_slp(_labels(5), str(path), compression="zip")

    assert path.read_bytes() == before
    assert _folder(tmp_path) == ["labels.slp"]


@pytest.mark.parametrize("embed_inplace", [False, True])
def test_cancelled_embed_keeps_previous_file(tmp_path, slp_minimal_pkg, embed_inplace):
    """Cancelling an export leaves the previous file and the labels as they were."""
    path = tmp_path / "labels.pkg.slp"
    shutil.copy(slp_minimal_pkg, path)
    before = path.read_bytes()

    labels = load_slp(slp_minimal_pkg)
    videos = list(labels.videos)

    with pytest.raises(ExportCancelled):
        save_slp(
            labels,
            str(path),
            embed="user",
            embed_inplace=embed_inplace,
            progress_callback=lambda current, total, phase: False,
        )

    assert path.read_bytes() == before
    assert _folder(tmp_path) == ["labels.pkg.slp"]
    # The labels still refer to the videos they had, frames included.
    assert all(a is b for a, b in zip(labels.videos, videos))
    assert labels[0].video is videos[0]
    assert labels[0].image.shape == (384, 384, 1)


def test_failed_embed_inplace_restores_videos(tmp_path, monkeypatch, slp_minimal_pkg):
    """A save that fails after embedding in place gives the labels their videos."""
    labels = load_slp(slp_minimal_pkg)
    videos = list(labels.videos)

    monkeypatch.setattr(slp, "write_label_images", _fail)
    with pytest.raises(RuntimeError, match="disk full"):
        save_slp(
            labels,
            str(tmp_path / "new.pkg.slp"),
            embed="user",
            embed_inplace=True,
            verbose=False,
        )

    assert _folder(tmp_path) == []
    assert all(a is b for a, b in zip(labels.videos, videos))
    assert labels[0].video is videos[0]
    assert labels.videos[0].filename == slp_minimal_pkg


@pytest.mark.parametrize("as_path", [False, True])
@pytest.mark.parametrize("embed_inplace", [False, True])
def test_embedded_videos_refer_to_the_target(
    tmp_path, slp_minimal_pkg, as_path, embed_inplace
):
    """The temporary file's name is not left in the file or in the labels."""
    path = tmp_path / "out.pkg.slp"
    target = path if as_path else str(path)
    labels = load_slp(slp_minimal_pkg)
    expected = labels[0].image

    save_slp(labels, target, embed="user", embed_inplace=embed_inplace, verbose=False)

    assert _folder(tmp_path) == ["out.pkg.slp"]
    with h5py.File(path, "r") as f:
        videos_json = b"".join(bytes(row) for row in f["videos_json"][()])
        source_json = f["video0/source_video"].attrs["json"]
    assert b".tmp-" not in videos_json
    assert ".tmp-" not in source_json
    assert b'"filename":"."' in videos_json

    if embed_inplace:
        # The caller's labels now read their frames from the saved file.
        video = labels.videos[0]
        assert video.filename == target
        assert Path(video.backend.filename) == path
        assert video.backend.dataset == "video0/video"
        np.testing.assert_array_equal(labels[0].image, expected)
    else:
        assert labels.videos[0].filename == slp_minimal_pkg

    loaded = load_slp(str(path))
    assert Path(loaded.videos[0].filename) == path
    np.testing.assert_array_equal(loaded[0].image, expected)


@replaces_open_files
@pytest.mark.parametrize("embed", [None, True, "user"])
@pytest.mark.parametrize("lazy", [False, True])
def test_save_package_over_itself(tmp_path, slp_minimal_pkg, embed, lazy):
    """Saving a package over itself keeps its frames and its source video."""
    path = tmp_path / "labels.pkg.slp"
    shutil.copy(slp_minimal_pkg, path)
    original = load_slp(slp_minimal_pkg)
    expected = original[0].image
    source_filename = original.videos[0].source_video.filename

    labels = load_slp(str(path), lazy=lazy)
    save_slp(labels, str(path), embed=embed, verbose=False)

    assert _folder(tmp_path) == ["labels.pkg.slp"]
    loaded = load_slp(str(path))
    assert len(loaded) == len(original)
    video = loaded.videos[0]
    assert Path(video.filename) == path
    assert video.backend.dataset == "video0/video"
    np.testing.assert_array_equal(loaded[0].image, expected)
    # The package names the original video as its source, not itself.
    assert video.source_video.filename == source_filename
    assert video.source_video.source_video is None


def test_save_package_over_itself_with_closed_videos(tmp_path, slp_minimal_pkg):
    """A package loaded without opening its videos is saved over itself intact."""
    path = tmp_path / "labels.pkg.slp"
    shutil.copy(slp_minimal_pkg, path)
    expected = load_slp(slp_minimal_pkg)[0].image

    labels = load_slp(str(path), open_videos=False)
    save_slp(labels, str(path), embed=None, verbose=False)

    assert _folder(tmp_path) == ["labels.pkg.slp"]
    np.testing.assert_array_equal(load_slp(str(path))[0].image, expected)


@replaces_open_files
def test_save_package_without_source_over_itself(tmp_path, slp_minimal_pkg):
    """A package with no recorded source video doesn't name itself as its source."""
    path = tmp_path / "labels.pkg.slp"
    labels = load_slp(slp_minimal_pkg)
    expected = labels[0].image
    save_slp(labels, str(path), embed="user", verbose=False)
    with h5py.File(path, "r+") as f:
        del f["video0/source_video"]

    labels = load_slp(str(path))
    assert labels.videos[0].source_video is None
    save_slp(labels, str(path), embed=True, verbose=False)

    with h5py.File(path, "r") as f:
        assert "source_video" not in f["video0"]
    loaded = load_slp(str(path))
    assert loaded.videos[0].source_video is None
    np.testing.assert_array_equal(loaded[0].image, expected)


def test_temporary_file_is_hidden_beside_the_target(tmp_path, monkeypatch):
    """The temporary file is in the target's folder and keeps its extensions."""
    seen = []
    write_metadata = slp.write_metadata

    def spy(labels_path, *args, **kwargs):
        seen.append(Path(labels_path))
        return write_metadata(labels_path, *args, **kwargs)

    monkeypatch.setattr(slp, "write_metadata", spy)
    path = tmp_path / "labels.pkg.slp"
    save_slp(_labels(), str(path))

    (tmp,) = seen
    assert tmp.parent == tmp_path
    assert tmp.name.startswith(".tmp-")
    assert tmp.name.endswith(".pkg.slp")
    assert not tmp.exists()
    assert not slp._SAVE_TARGETS
    assert _folder(tmp_path) == ["labels.pkg.slp"]


@pytest.mark.parametrize(
    "name",
    ["labels.pkg.slp", "x.slp", "noext", "a" * 251 + ".slp", "b" * 240 + ".pkg.slp"],
)
def test_temporary_name_fits_where_the_target_fits(tmp_path, name):
    """The temporary name is recognisable, and no longer than a long target's."""
    final = tmp_path / name
    tmp = slp._temporary_save_path(final)
    assert tmp.parent == final.parent
    assert tmp != final
    assert tmp.suffixes[-2:] == final.suffixes[-2:]
    if len(name) < 200:
        assert tmp.name.startswith(".tmp-")
        assert len(tmp.name) == len(".tmp-12345678") + len("".join(final.suffixes))
    else:
        assert len(tmp.name) <= len(final.name)


def test_temporary_name_is_shortened_in_a_deep_folder(tmp_path):
    """A short name in a folder near the path limit isn't made longer."""
    folder = tmp_path
    while len(str(folder)) < 236:
        folder = folder / "deep_folder"
    final = folder / "x.slp"
    tmp = slp._temporary_save_path(final)
    # Four random characters is the least it keeps.
    assert tmp.name.startswith(".")
    assert len(tmp.name) == len(".1234.slp")
    assert tmp.suffix == ".slp"


def test_save_to_the_longest_file_name(tmp_path):
    """A target whose name is at the file system's limit still saves."""
    try:
        limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        pytest.skip("The file name limit of this file system isn't known.")
    path = tmp_path / ("a" * (limit - 4) + ".slp")
    try:
        path.touch()
    except OSError:
        pytest.skip("A name this long can't be created here at all.")
    path.unlink()
    save_slp(_labels(2), str(path))
    assert len(load_slp(str(path))) == 2
    save_slp(_labels(3), str(path))
    assert len(load_slp(str(path))) == 3
    assert _folder(tmp_path) == [path.name]


def test_save_to_missing_folder_names_the_target(tmp_path):
    """A target in a folder that doesn't exist is reported by its own name."""
    path = tmp_path / "nope" / "labels.slp"
    with pytest.raises(FileNotFoundError, match="labels.slp") as excinfo:
        save_slp(_labels(), str(path))
    assert ".tmp-" not in str(excinfo.value)
    assert _folder(tmp_path) == []


def test_rename_is_retried_on_windows(tmp_path, monkeypatch):
    """A rename refused for a moment is tried again on Windows, and only there."""
    tmp, final = tmp_path / "tmp.slp", tmp_path / "final.slp"
    calls = []
    replace = os.replace

    def refuse_twice(src, dst):
        calls.append(src)
        if len(calls) < 3:
            raise PermissionError("in use")
        replace(src, dst)

    monkeypatch.setattr(slp.os, "replace", refuse_twice)
    monkeypatch.setattr(slp.time, "sleep", lambda seconds: None)

    monkeypatch.setattr(slp.sys, "platform", "win32")
    tmp.write_bytes(b"new")
    slp._replace_file(tmp, final)
    assert final.read_bytes() == b"new"
    assert len(calls) == 3

    # A file that stays in use is reported.
    monkeypatch.setattr(slp.os, "replace", _refuse)
    tmp.write_bytes(b"newer")
    with pytest.raises(PermissionError):
        slp._replace_file(tmp, final)

    # Elsewhere a refusal is final at once.
    calls.clear()
    monkeypatch.setattr(slp.os, "replace", refuse_twice)
    monkeypatch.setattr(slp.sys, "platform", "linux")
    with pytest.raises(PermissionError):
        slp._replace_file(tmp, final)
    assert len(calls) == 1


def test_failed_restore_warns(tmp_path, monkeypatch, slp_minimal_pkg):
    """If the videos can't be put back after a failed save, a warning says so."""
    labels = load_slp(slp_minimal_pkg)
    replace_videos = Labels.replace_videos
    calls = []

    def fail_second(self, *args, **kwargs):
        calls.append(self)
        if len(calls) > 1:
            raise RuntimeError("no")
        return replace_videos(self, *args, **kwargs)

    monkeypatch.setattr(Labels, "replace_videos", fail_second)
    monkeypatch.setattr(slp, "write_label_images", _fail)
    with pytest.warns(UserWarning, match="could not be put back"):
        with pytest.raises(RuntimeError, match="disk full"):
            save_slp(
                labels,
                str(tmp_path / "new.pkg.slp"),
                embed="user",
                embed_inplace=True,
                verbose=False,
            )
    assert _folder(tmp_path) == []


def _three_frame_package(path: Path, video_path: str) -> list[np.ndarray]:
    """Write a package with frames 0, 1 and 2 embedded; only 1 and 2 user-labeled.

    Returns:
        The images of frames 0, 1 and 2 as stored in the package.
    """
    skeleton = Skeleton(["a"])
    video = Video.from_filename(video_path)
    points = np.array([[1.0, 2.0]])
    labels = Labels(
        [
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    PredictedInstance.from_numpy(points, skeleton=skeleton, score=1.0)
                ],
            ),
            LabeledFrame(
                video=video,
                frame_idx=1,
                instances=[Instance.from_numpy(points, skeleton=skeleton)],
            ),
            LabeledFrame(
                video=video,
                frame_idx=2,
                instances=[Instance.from_numpy(points, skeleton=skeleton)],
            ),
        ]
    )
    save_slp(labels, str(path), embed=True, verbose=False)
    return [lf.image for lf in load_slp(str(path))]


@replaces_open_files
@pytest.mark.parametrize("lazy", [False, True])
def test_labels_in_memory_follow_a_save_over_their_package(
    tmp_path, centered_pair_low_quality_path, lazy
):
    """After a package is saved over itself, the labels read the new file's frames.

    The new file embeds fewer frames, in other rows. The videos of the labels
    that were saved must not go on reading with the old file's row numbers.
    """
    path = tmp_path / "labels.pkg.slp"
    images = _three_frame_package(path, centered_pair_low_quality_path)
    assert not np.array_equal(images[1], images[2])

    labels = load_slp(str(path), lazy=lazy)
    video = labels.videos[0]
    np.testing.assert_array_equal(video[1], images[1])

    save_slp(labels, str(path), embed="user", verbose=False)

    # The same objects, reading the frames the new file holds.
    assert labels.videos[0] is video
    assert Path(video.filename) == path
    np.testing.assert_array_equal(video[1], images[1])
    np.testing.assert_array_equal(video[2], images[2])
    assert not video.backend.has_frame(0)
    assert [lf.frame_idx for lf in labels] == [0, 1, 2]
    np.testing.assert_array_equal(labels[1].image, images[1])
    np.testing.assert_array_equal(labels[2].image, images[2])

    # And a fresh load agrees.
    loaded = load_slp(str(path))
    np.testing.assert_array_equal(loaded[1].image, images[1])
    np.testing.assert_array_equal(loaded[2].image, images[2])


def test_closed_videos_follow_a_save_over_their_package(
    tmp_path, centered_pair_low_quality_path
):
    """Videos that were never opened read the new file once they are."""
    path = tmp_path / "labels.pkg.slp"
    images = _three_frame_package(path, centered_pair_low_quality_path)

    labels = load_slp(str(path), open_videos=False)
    video = labels.videos[0]
    save_slp(labels, str(path), embed=None, verbose=False)

    assert labels.videos[0] is video
    assert video.backend is None
    video.open()
    for frame_idx in range(3):
        np.testing.assert_array_equal(video[frame_idx], images[frame_idx])


def test_copied_video_does_not_name_the_replaced_file_as_source(
    tmp_path, centered_pair_low_quality_path
):
    """A video copied from another package doesn't name the replaced file."""
    first = tmp_path / "first.pkg.slp"
    second = tmp_path / "second.pkg.slp"
    images = _three_frame_package(first, centered_pair_low_quality_path)
    # The second package records the first as the source of its video.
    save_slp(load_slp(str(first)), str(second), embed=True, verbose=False)
    source = load_slp(str(second)).videos[0].source_video
    assert Path(source.filename) == first

    # Saved over the first, without opening the videos: the group is copied.
    labels = load_slp(str(second), open_videos=False)
    save_slp(labels, str(first), embed=None, verbose=False)

    loaded = load_slp(str(first))
    video = loaded.videos[0]
    assert Path(video.filename) == first
    assert video.source_video.filename == centered_pair_low_quality_path
    with h5py.File(first, "r") as f:
        assert "first.pkg.slp" not in f["video0/source_video"].attrs["json"]
    for frame_idx in range(3):
        np.testing.assert_array_equal(video[frame_idx], images[frame_idx])


def test_save_to_new_path_keeps_a_source_with_the_same_relative_name(
    tmp_path, monkeypatch, centered_pair_low_quality_path
):
    """A save to a new path records the source it is given, whatever its name."""
    video_path = str(Path(centered_pair_low_quality_path).resolve())
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    monkeypatch.chdir(one)
    _three_frame_package(Path("A.pkg.slp"), video_path)
    save_slp(load_slp("A.pkg.slp"), "B.pkg.slp", embed=True, verbose=False)
    shutil.copy("B.pkg.slp", two / "B.pkg.slp")

    # In another folder, B's source "A.pkg.slp" names a file that isn't there.
    # Saving to that name is a save to a new path.
    monkeypatch.chdir(two)
    save_slp(load_slp("B.pkg.slp"), "A.pkg.slp", embed=True, verbose=False)
    with h5py.File("A.pkg.slp", "r") as f:
        source = json.loads(f["video0/source_video"].attrs["json"])
    # The chain is the one given: B, then B's source A (by its recorded name),
    # then the original video.
    assert source["filename"] == "B.pkg.slp"
    assert source["source_video"]["filename"] == "A.pkg.slp"
    # Filenames are stored with forward slashes on every platform.
    assert Path(source["source_video"]["source_video"]["filename"]) == Path(video_path)


def test_flushing_to_disk_leaves_the_bytes_alone(tmp_path):
    """Flushing a file doesn't change it, even one that ends with Ctrl-Z.

    A handle opened for reading and writing in text mode on Windows drops a
    trailing Ctrl-Z byte (0x1A), which cuts an HDF5 file that ends with one.
    """
    path = tmp_path / "data.bin"
    for content in (b"abc\x1a", b"\r\n\x1a\x1a", b""):
        path.write_bytes(content)
        slp._sync_to_disk(path)
        assert path.read_bytes() == content
    # A missing file is not an error: the rename that follows reports it.
    slp._sync_to_disk(tmp_path / "missing.bin")
