"""Tests for the optional `/preview` HDF5 group in SLP files."""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

import sleap_io as sio
from sleap_io.io.slp import (
    SlpLayoutWarning,
    _is_known_toplevel_name,
    read_preview,
    write_labels,
)
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.preview import (
    DEFAULT_PREVIEW_MAX_SIZE,
    PREVIEW_FORMAT_VERSION,
    PreviewFrame,
)
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


def test_preview_frame_from_image_rgb_and_downscale():
    """Large RGB image is aspect-preserving downscaled to max_size and JPEG-encoded."""
    video = Video(filename="fake.mp4", open_backend=False)
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(1000, 2000, 3), dtype=np.uint8)

    pf = PreviewFrame.from_image(video, frame_idx=12, image=img)
    assert pf.video is video
    assert pf.frame_idx == 12
    assert pf.orig_width == 2000
    assert pf.orig_height == 1000
    assert pf.width == DEFAULT_PREVIEW_MAX_SIZE
    assert pf.height == DEFAULT_PREVIEW_MAX_SIZE // 2
    assert pf.data[:2] == b"\xff\xd8"  # JPEG SOI

    decoded = pf.to_image()
    assert decoded.shape == (pf.height, pf.width, 3)


def test_preview_frame_no_upscale_and_grayscale_collapse():
    """Small frames keep native dimensions and identical RGB channels collapse to L."""
    video = Video(filename="fake.mp4", open_backend=False)
    gray = np.tile(np.arange(120, dtype=np.uint8)[None, :], (80, 1))
    rgb_gray = np.stack([gray, gray, gray], axis=-1)

    pf = PreviewFrame.from_image(video, frame_idx=0, image=rgb_gray)
    assert pf.orig_width == 120
    assert pf.orig_height == 80
    assert pf.width == 120
    assert pf.height == 80

    decoded = pf.to_image()
    assert decoded.ndim == 2
    assert decoded.shape == (80, 120)


@pytest.mark.parametrize("slp_version", ["v2", "v3"])
def test_preview_roundtrip_v2_and_v3(tmp_path, slp_version):
    """Explicit attached PreviewFrames round-trip cleanly in both v2 and v3 layouts."""
    skel = Skeleton(["head", "tail"])
    video = Video(filename="nonexistent.mp4", open_backend=False)
    inst = Instance.from_numpy(np.array([[10.0, 20.0], [30.0, 40.0]]), skeleton=skel)
    lf = LabeledFrame(video=video, frame_idx=5, instances=[inst])
    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    img[:, :200, 0] = 220
    pf = PreviewFrame.from_image(video, frame_idx=5, image=img)
    labels = Labels(labeled_frames=[lf], preview_frames=[pf])

    out_path = tmp_path / f"roundtrip_{slp_version}.slp"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels.save(out_path, slp_version=slp_version)

    # Preview alone must never trigger a format-3.0 layout warning.
    assert not any(issubclass(w.category, SlpLayoutWarning) for w in caught)

    # Verify exact HDF5 schema.
    with h5py.File(out_path, "r") as f:
        assert "preview" in f
        grp = f["preview"]
        assert int(grp.attrs["format_version"]) == PREVIEW_FORMAT_VERSION
        assert grp["video"].dtype == np.int64
        assert grp["frame_idx"].dtype == np.int64
        assert grp["orig_width"].dtype == np.int32
        assert grp["orig_height"].dtype == np.int32
        assert grp["width"].dtype == np.int32
        assert grp["height"].dtype == np.int32
        assert grp["data_start"].dtype == np.int64
        assert grp["data_end"].dtype == np.int64
        assert grp["data"].dtype == np.uint8
        assert grp["data"].compression is None
        assert grp["video"][0] == 0
        assert grp["frame_idx"][0] == 5
        assert grp["orig_width"][0] == 400
        assert grp["orig_height"][0] == 300
        assert grp["data_start"][0] == 0
        assert grp["data_end"][0] == len(pf.data)
        if slp_version == "v2":
            assert f["metadata"].attrs["format_id"] < 3.0
        else:
            assert f["metadata"].attrs["format_id"] >= 3.0

    # Eager reload
    loaded = sio.load_slp(out_path, open_videos=False)
    assert len(loaded.preview_frames) == 1
    rpf = loaded.preview_frames[0]
    assert rpf.video is loaded.videos[0]
    assert rpf.frame_idx == 5
    assert (rpf.orig_width, rpf.orig_height) == (400, 300)
    assert (rpf.width, rpf.height) == (400, 300)
    assert rpf.data == pf.data

    # Lazy reload + lazy save preserves /preview without materializing
    lazy = sio.load_slp(out_path, open_videos=False, lazy=True)
    assert lazy.is_lazy
    assert len(lazy.preview_frames) == 1
    assert lazy.preview_frames[0].video is lazy.videos[0]

    lazy_out = tmp_path / f"lazy_copy_{slp_version}.slp"
    lazy.save(lazy_out, slp_version=slp_version)
    assert lazy.is_lazy
    lazy_reloaded = sio.load_slp(lazy_out, open_videos=False)
    assert len(lazy_reloaded.preview_frames) == 1
    assert lazy_reloaded.preview_frames[0].data == pf.data


def test_preview_auto_does_not_force_v3_and_is_known_toplevel(tmp_path):
    """Attaching preview_frames keeps slp_version='auto' on v2 layout."""
    assert _is_known_toplevel_name("preview")

    video = Video(filename="fake.mp4", open_backend=False)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    labels = Labels(
        videos=[video],
        preview_frames=[PreviewFrame.from_image(video, 0, img)],
    )
    out = tmp_path / "auto_v2.slp"
    labels.save(out)
    with h5py.File(out, "r") as f:
        assert "preview" in f
        assert f["metadata"].attrs["format_id"] < 3.0


def test_preview_modes_explicit_false_true_and_int(
    tmp_path, centered_pair_low_quality_path
):
    """preview=False omits /preview; preview=True and preview=N capture frames."""
    skel = Skeleton(["head", "tail"])
    video = Video.from_filename(centered_pair_low_quality_path)
    lfs = [
        LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[
                PredictedInstance.from_numpy(
                    np.array([[100.0, 100.0], [120.0, 120.0]]), skeleton=skel
                )
            ],
        )
        for i in (0, 5, 10, 15, 20)
    ]
    # Add a user-labeled frame at index 10 to verify priority over predicted-only.
    lfs[2].instances.append(
        Instance.from_numpy(np.array([[50.0, 50.0], [60.0, 60.0]]), skeleton=skel)
    )
    labels = Labels(labeled_frames=lfs)

    # preview=False writes no /preview group.
    no_prev_path = tmp_path / "no_preview.slp"
    labels.save(no_prev_path, preview=False)
    with h5py.File(no_prev_path, "r") as f:
        assert "preview" not in f

    # Reload with closed / unopened decoder and verify preview="auto" does NOT decode.
    reloaded = sio.load_slp(no_prev_path)
    v2 = reloaded.videos[0]
    assert v2.backend._open_reader is None
    assert v2.backend._cached_shape is None
    auto_copy = tmp_path / "auto_no_decode.slp"
    reloaded.save(auto_copy, prefer_metadata=True)
    assert v2.backend._open_reader is None
    assert v2.backend._cached_shape is None
    with h5py.File(auto_copy, "r") as f:
        assert "preview" not in f

    # Explicit preview=True opens, captures 1 frame (preferring user-labeled frame 10),
    # and closes the reader it opened.
    explicit_one = tmp_path / "explicit_one.slp"
    reloaded.save(explicit_one, preview=True)
    assert v2.backend._open_reader is None
    pfs = read_preview(str(explicit_one), reloaded.videos)
    assert len(pfs) == 1
    assert pfs[0].frame_idx == 10
    assert (pfs[0].orig_width, pfs[0].orig_height) == (384, 384)
    assert pfs[0].data[:2] == b"\xff\xd8"

    # Explicit preview=3 captures 3 distinct frames.
    explicit_three = tmp_path / "explicit_three.slp"
    reloaded.save(explicit_three, preview=3)
    pfs3 = read_preview(str(explicit_three), reloaded.videos)
    assert len(pfs3) == 3
    assert len({pf.frame_idx for pf in pfs3}) == 3
    assert 10 in {pf.frame_idx for pf in pfs3}

    # preview=0 behaves like preview=False.
    zero_path = tmp_path / "preview_zero.slp"
    reloaded.save(zero_path, preview=0)
    with h5py.File(zero_path, "r") as f:
        assert "preview" not in f


def test_preview_auto_captures_when_reader_already_open(
    tmp_path, centered_pair_low_quality_path
):
    """preview='auto' automatically writes 1 preview when the video reader is active."""
    video = Video.from_filename(centered_pair_low_quality_path)
    _ = video[0]  # opens reader
    assert video.backend._open_reader is not None

    labels = Labels(labeled_frames=[LabeledFrame(video=video, frame_idx=0)])
    out = tmp_path / "auto_captured.slp"
    labels.save(out)

    loaded = sio.load_slp(out, open_videos=False)
    assert len(loaded.preview_frames) == 1
    assert loaded.preview_frames[0].frame_idx == 0
    assert loaded.preview_frames[0].data[:2] == b"\xff\xd8"


def test_preview_auto_skips_embedded_pkg(tmp_path, centered_pair_low_quality_path):
    """Embedded .pkg.slp saves omit /preview under preview='auto'."""
    video = Video.from_filename(centered_pair_low_quality_path)
    _ = video[0]
    labels = Labels(labeled_frames=[LabeledFrame(video=video, frame_idx=0)])

    pkg_path = tmp_path / "embedded.pkg.slp"
    labels.save(pkg_path, embed=True)
    with h5py.File(pkg_path, "r") as f:
        assert "preview" not in f

    # Re-saving an already-embedded pkg with embed=None also omits /preview.
    pkg_loaded = sio.load_slp(pkg_path)
    pkg_copy = tmp_path / "embedded_copy.pkg.slp"
    pkg_loaded.save(pkg_copy, embed=None)
    with h5py.File(pkg_copy, "r") as f:
        assert "preview" not in f


def test_preview_best_effort_missing_video_and_validation(tmp_path):
    """Missing videos never break preview=True saves; bad preview option raises."""
    video = Video(filename=str(tmp_path / "does_not_exist.mp4"), open_backend=False)
    labels = Labels(labeled_frames=[LabeledFrame(video=video, frame_idx=0)])

    out = tmp_path / "missing_vid.slp"
    labels.save(out, preview=True)
    with h5py.File(out, "r") as f:
        assert "preview" not in f

    with pytest.raises(ValueError, match="preview"):
        write_labels(str(tmp_path / "bad.slp"), labels, preview="invalid")
    with pytest.raises(ValueError, match="preview"):
        write_labels(str(tmp_path / "bad.slp"), labels, preview=-1)


def test_preview_preserved_across_copy_extract_replace_and_materialize(tmp_path):
    """copy/extract/replace_videos/materialize remap preview video references."""
    v1 = Video(filename="v1.mp4", open_backend=False)
    v2 = Video(filename="v2.mp4", open_backend=False)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    pf1 = PreviewFrame.from_image(v1, 0, img)
    pf2 = PreviewFrame.from_image(v2, 3, img)
    labels = Labels(
        labeled_frames=[
            LabeledFrame(video=v1, frame_idx=0),
            LabeledFrame(video=v2, frame_idx=3),
        ],
        preview_frames=[pf1, pf2],
    )

    # copy()
    copied = labels.copy(open_videos=False)
    assert len(copied.preview_frames) == 2
    assert copied.preview_frames[0].video is copied.videos[0]
    assert copied.preview_frames[1].video is copied.videos[1]
    assert copied.preview_frames[0].video is not v1

    # extract()
    extracted = labels.extract([0])
    assert len(extracted.preview_frames) == 1
    assert extracted.preview_frames[0].video is extracted.videos[0]

    # replace_videos()
    v1_new = Video(filename="v1_new.mp4", open_backend=False)
    copied.replace_videos(video_map={copied.videos[0]: v1_new})
    assert copied.preview_frames[0].video is v1_new

    # lazy copy() and materialize()
    slp_path = tmp_path / "two_vids.slp"
    labels.save(slp_path)
    lazy = sio.load_slp(str(slp_path), open_videos=False, lazy=True)
    lazy_copy = lazy.copy(open_videos=False)
    assert len(lazy_copy.preview_frames) == 2
    assert lazy_copy.preview_frames[0].video is lazy_copy.videos[0]
    assert lazy_copy.preview_frames[1].video is lazy_copy.videos[1]

    mat = lazy.materialize()
    assert len(mat.preview_frames) == 2
    assert mat.preview_frames[0].video is mat.videos[0]
    assert mat.preview_frames[1].video is mat.videos[1]


def test_preview_excluded_from_labels_equality():
    """preview_frames does not affect Labels equality comparisons."""
    v = Video(filename="v.mp4", open_backend=False)
    a = Labels(videos=[v])
    b = Labels(
        videos=[v],
        preview_frames=[
            PreviewFrame.from_image(v, 0, np.zeros((32, 32, 3), dtype=np.uint8))
        ],
    )
    assert a == b
