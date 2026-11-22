"""Tests that saving an SLP file goes through a temporary file.

A save writes to a temporary file in the target's folder and renames it over the
target when it is complete, so the target holds either the previous file or the
complete new one.
"""

from __future__ import annotations

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
    """The temporary file is in the target's folder and keeps its extension."""
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
    assert tmp.name.endswith("-labels.pkg.slp")
    assert not tmp.exists()
    assert not slp._SAVE_TARGETS
    assert _folder(tmp_path) == ["labels.pkg.slp"]


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
