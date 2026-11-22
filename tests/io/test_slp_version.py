"""Tests for the SLP layout a save writes (`slp_version`) and the shared save options.

By default a save writes the 2.x layout that sleap-io < 0.10 and SLEAP can open,
unless the labels were loaded from a format 3.0 file or hold content only format
3.0 can store. The layout is decided once per save, and once for files written
together.
"""

from __future__ import annotations

import os
import re
import warnings

import click
import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import sleap_io as sio
from sleap_io import (
    Instance,
    LabeledFrame,
    Labels,
    LabelsSet,
    Skeleton,
    Video,
    load_slp,
    save_slp,
)
from sleap_io.io.cli import SLP_SAVE_OPTION_NAMES, cli
from sleap_io.io.format_defaults import V3_DEFAULTS_ENV
from sleap_io.io.slp import (
    LabelImageWriter,
    SlpLayoutWarning,
    merge_label_images,
    resolve_slp_version,
)
from sleap_io.model.label_image import UserLabelImage


def _plain(output: str) -> str:
    """Return command output without colours, box lines or line wrapping."""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    output = re.sub(r"[│╭╮╰╯─]", " ", output)
    return " ".join(output.split())


def _format_id(path) -> float:
    with h5py.File(path, "r") as f:
        return float(f["metadata"].attrs["format_id"])


def _is_v3(path) -> bool:
    """Whether a file is in format 3.0, checked by its tables, not only its number."""
    with h5py.File(path, "r") as f:
        is_group = isinstance(f["points"], h5py.Group) if "points" in f else None
    v3 = _format_id(path) >= 3.0
    if is_group is not None:
        # The format number and the layout of the tables always agree.
        assert is_group == v3
    return v3


def _labels(n_frames: int = 2) -> Labels:
    """Labels that the 2.x layout stores: hidden points keep their coordinates."""
    skeleton = Skeleton(["a", "b", "c"])
    video = Video("video.mp4", open_backend=False)
    frames = []
    for i in range(n_frames):
        inst = Instance.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) + i, skeleton=skeleton
        )
        inst.points["visible"][1] = False  # Hidden, with coordinates.
        frames.append(LabeledFrame(video=video, frame_idx=i, instances=[inst]))
    return Labels(frames)


def _with_unlabeled(labels: Labels) -> Labels:
    inst = labels[0].instances[0]
    inst.points["labeled"][2] = False
    inst.points["visible"][2] = False
    return labels


def _with_metadata(labels: Labels) -> Labels:
    labels[0].instances[0].metadata = {"note": "checked"}
    return labels


@pytest.fixture(autouse=True)
def _no_switch(monkeypatch):
    """Run each test with the format 3.0 defaults off, whatever the environment."""
    monkeypatch.delenv(V3_DEFAULTS_ENV, raising=False)
    sio.set_v3_defaults(None)
    yield
    sio.set_v3_defaults(None)


# --- The rule -----------------------------------------------------------------


def test_new_labels_are_saved_in_the_2x_layout(tmp_path):
    """New labels that fit the 2.x layout get it, with no warning."""
    path = tmp_path / "labels.slp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_slp(_labels(), str(path))
    assert not _is_v3(path)
    # The 2.x layout stores hidden points with their coordinates.
    with h5py.File(path, "r") as f:
        points = f["points"][()]
    assert not points["visible"][1]
    assert (points["x"][1], points["y"][1]) == (3.0, 4.0)


@pytest.mark.parametrize(
    "make, reason",
    [
        (_with_unlabeled, "1 point(s) marked as not labeled"),
        (_with_metadata, "metadata on instances, frames or bounding boxes"),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_content_only_3_0_stores_gives_3_0_with_one_warning(
    tmp_path, make, reason, lazy
):
    """Content the 2.x layout can't store makes the save write format 3.0."""
    labels = make(_labels())
    if lazy:
        # Lazy labels with this content come from a format 3.0 file; forget that,
        # so that the content check of the lazy store decides.
        through = tmp_path / "through.slp"
        save_slp(labels, str(through), slp_version="v3")
        labels = load_slp(str(through), lazy=True)
        labels._source_format_id = None
    path = tmp_path / "labels.slp"
    with pytest.warns(SlpLayoutWarning, match="can't store") as record:
        save_slp(labels, str(path))
    assert len([w for w in record if w.category is SlpLayoutWarning]) == 1
    assert reason in str(record[0].message)
    # The warning names the caller's line, not a line inside sleap-io.
    assert record[0].filename == __file__
    assert _is_v3(path)


def test_unknown_columns_need_3_0(tmp_path, slp_typical):
    """Unknown columns kept from a newer file make the save write format 3.0."""
    newer = tmp_path / "newer.slp"
    save_slp(load_slp(slp_typical), str(newer), slp_version="v3")
    with h5py.File(newer, "r+") as f:
        n = len(f["points/x"])
        f["points"].create_dataset("future_column", data=np.arange(n))
    for lazy in (False, True):
        labels = load_slp(str(newer), lazy=lazy)
        labels._source_format_id = None
        with pytest.warns(SlpLayoutWarning, match="unknown columns"):
            assert resolve_slp_version(labels) == "v3"


@pytest.mark.parametrize("lazy", [False, True])
def test_a_3_0_file_stays_3_0(tmp_path, lazy):
    """A file loaded as format 3.0 is saved as format 3.0, with no warning."""
    v3_file = tmp_path / "v3.slp"
    save_slp(_labels(), str(v3_file), slp_version="v3")
    labels = load_slp(str(v3_file), lazy=lazy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for derived in (
            labels,
            labels.copy(),
            labels.extract([0]),
            labels.materialize() if lazy else labels,
        ):
            path = tmp_path / "out.slp"
            save_slp(derived, str(path))
            assert _is_v3(path)
    # And a 2.x file stays 2.x.
    v2_file = tmp_path / "v2.slp"
    save_slp(load_slp(str(v3_file), lazy=lazy), str(v2_file), slp_version="v2")
    save_slp(load_slp(str(v2_file), lazy=lazy), str(tmp_path / "again.slp"))
    assert not _is_v3(tmp_path / "again.slp")


def test_explicit_versions(tmp_path):
    """`"v2"` and `"v3"` always write their layout; `legacy_format` is `"v2"`."""
    path = tmp_path / "labels.slp"
    save_slp(_labels(), str(path), slp_version="v3")
    assert _is_v3(path)
    save_slp(load_slp(str(path)), str(path), slp_version="v2")
    assert not _is_v3(path)
    save_slp(load_slp(str(path)), str(path), legacy_format=True)
    assert not _is_v3(path)
    # Strict 2.x: unlabeled points still raise unless they may be dropped.
    with pytest.raises(ValueError, match="unlabeled"):
        save_slp(_with_unlabeled(_labels()), str(path), slp_version="v2")
    with pytest.warns(UserWarning, match="hidden points"):
        save_slp(
            _with_unlabeled(_labels()),
            str(path),
            slp_version="v2",
            unlabeled_as="absent",
        )
    assert not _is_v3(path)
    with pytest.raises(ValueError, match="slp_version must be one of"):
        save_slp(_labels(), str(path), slp_version="3.0")
    with pytest.raises(ValueError, match="can't be combined"):
        save_slp(_labels(), str(path), slp_version="v3", legacy_format=True)


def test_layout_is_decided_before_the_target_is_touched(tmp_path):
    """A refused save leaves the existing file as it was."""
    path = tmp_path / "labels.slp"
    save_slp(_labels(), str(path))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_slp(_labels(), str(path), slp_version="nope")
    assert path.read_bytes() == before


# --- The switch -----------------------------------------------------------------


def test_the_switch_turns_on_3_0_and_explicit_options_win(tmp_path, monkeypatch):
    """The switch makes `"auto"` write format 3.0; explicit options still win."""
    path = tmp_path / "labels.slp"
    sio.set_v3_defaults(True)
    assert sio.get_v3_defaults()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_slp(_labels(), str(path))
    assert _is_v3(path)
    save_slp(_labels(), str(path), slp_version="v2")
    assert not _is_v3(path)
    sio.set_v3_defaults(None)

    for value, on in (("1", True), ("TRUE", True), ("on", True), ("0", False)):
        monkeypatch.setenv(V3_DEFAULTS_ENV, value)
        assert sio.get_v3_defaults() is on
        save_slp(_labels(), str(path))
        assert _is_v3(path) is on
    # A value set in code wins over the environment.
    monkeypatch.setenv(V3_DEFAULTS_ENV, "1")
    sio.set_v3_defaults(False)
    assert not sio.get_v3_defaults()
    sio.set_v3_defaults(None)
    monkeypatch.delenv(V3_DEFAULTS_ENV)

    with sio.v3_defaults():
        assert sio.get_v3_defaults()
        with sio.v3_defaults(False):
            assert not sio.get_v3_defaults()
        assert sio.get_v3_defaults()
    assert not sio.get_v3_defaults()
    with pytest.raises(TypeError):
        sio.set_v3_defaults("yes")


# --- Several files, one layout ----------------------------------------------------


def test_labels_set_save_shares_one_layout(tmp_path):
    """If one file of a set needs format 3.0, all are written in it, warned once."""
    labels_set = LabelsSet(
        {"train": _with_metadata(_labels()), "val": _labels(), "test": _labels()}
    )
    with pytest.warns(SlpLayoutWarning) as record:
        labels_set.save(tmp_path, embed=False)
    assert len([w for w in record if w.category is SlpLayoutWarning]) == 1
    assert all(_is_v3(tmp_path / f"{n}.slp") for n in ("train", "val", "test"))

    out = tmp_path / "plain"
    LabelsSet({"train": _labels(), "val": _labels()}).save(
        out, embed=False, compression="gzip"
    )
    assert not any(_is_v3(out / f"{n}.slp") for n in ("train", "val"))
    with h5py.File(out / "train.slp", "r") as f:
        assert f["points"].compression == "gzip"


def test_make_training_splits_passes_the_save_options(tmp_path):
    """`make_training_splits` takes the save options for its files."""
    labels = _labels(10)
    labels.make_training_splits(
        0.6, n_val=0.2, n_test=0.2, save_dir=tmp_path, embed=False, seed=0
    )
    assert not any(_is_v3(tmp_path / f"{n}.slp") for n in ("train", "val", "test"))
    labels.make_training_splits(
        0.6,
        n_val=0.2,
        save_dir=tmp_path / "v3",
        embed=False,
        seed=0,
        slp_version="v3",
    )
    assert all(_is_v3(tmp_path / "v3" / f"{n}.slp") for n in ("train", "val"))


def test_sio_split_shares_one_layout(tmp_path):
    """`sio split` writes its files in one layout."""
    # A format 3.0 source: every split is format 3.0.
    src = tmp_path / "labels.slp"
    save_slp(_labels(10), str(src), slp_version="v3")
    out = tmp_path / "split3"
    result = CliRunner().invoke(cli, ["split", str(src), "-o", str(out), "--seed", "0"])
    assert result.exit_code == 0, result.output
    assert all(_is_v3(out / f"{n}.slp") for n in ("train", "val"))

    plain = tmp_path / "plain.slp"
    save_slp(_labels(10), str(plain))
    out = tmp_path / "split2"
    result = CliRunner().invoke(
        cli, ["split", str(plain), "-o", str(out), "--seed", "0"]
    )
    assert result.exit_code == 0, result.output
    assert not any(_is_v3(out / f"{n}.slp") for n in ("train", "val"))


def test_resolve_slp_version_for_a_set():
    """One decision for several labels: format 3.0 if any needs it."""
    plain = [_labels(), _labels()]
    assert resolve_slp_version(plain) == "v2"
    with pytest.warns(SlpLayoutWarning):
        assert resolve_slp_version(plain + [_with_unlabeled(_labels())]) == "v3"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_slp_version([_with_metadata(_labels())], warn=False) == "v3"
        assert resolve_slp_version(plain, "v3") == "v3"
        assert resolve_slp_version(plain, legacy_format=True) == "v2"


# --- The label image writers --------------------------------------------------------


def test_label_image_writers_default_to_the_2x_layout(tmp_path):
    """`LabelImageWriter` and `merge_label_images` write 2.x unless told otherwise."""
    data = (np.arange(35, dtype=np.int32).reshape(5, 7) % 3).astype(np.int32)
    paths = []
    for name, version in (("a", "auto"), ("b", "auto"), ("c", "v3")):
        path = str(tmp_path / f"{name}.slp")
        with LabelImageWriter(
            path, video=Video("cam.mp4"), slp_version=version
        ) as writer:
            writer.add(UserLabelImage(data=data))
        paths.append(path)
    assert not _is_v3(paths[0]) and _is_v3(paths[2])
    np.testing.assert_array_equal(load_slp(paths[0]).label_images[0].data, data)

    merged = str(tmp_path / "merged.slp")
    merge_label_images(paths[:2], merged)
    assert not _is_v3(merged)
    assert len(load_slp(merged).label_images) == 2
    # A format 3.0 source keeps the merged file in format 3.0.
    merge_label_images(paths, merged)
    assert _is_v3(merged)
    merge_label_images(paths, merged, slp_version="v2")
    assert not _is_v3(merged)


# --- Commands -----------------------------------------------------------------------

# Commands that never write an SLP file. Every other command must take the shared
# save options, so that a new command that writes SLP files can't miss them.
COMMANDS_THAT_WRITE_NO_SLP = {"download", "export", "render", "show"}


def _commands(group: click.Group, prefix: str = ""):
    for name, command in sorted(group.commands.items()):
        if isinstance(command, click.Group):
            yield from _commands(command, f"{prefix}{name} ")
        else:
            yield f"{prefix}{name}", command


def test_every_command_that_writes_slp_takes_the_save_options():
    """Walk the command list: each SLP-writing command has the shared options."""
    missing = {}
    for name, command in _commands(cli):
        params = {p.name for p in command.params}
        if name in COMMANDS_THAT_WRITE_NO_SLP:
            continue
        lacking = [o for o in SLP_SAVE_OPTION_NAMES if o not in params]
        if lacking:
            missing[name] = lacking
    assert missing == {}
    names = {name for name, _ in _commands(cli)}
    # The exempt list names real commands, so it can't silently go stale.
    assert COMMANDS_THAT_WRITE_NO_SLP <= names


@pytest.mark.parametrize("version, v3", [("auto", False), ("v2", False), ("v3", True)])
def test_commands_write_the_chosen_layout(tmp_path, version, v3):
    """A sample of commands writes the layout `--slp-version` picks."""
    src = tmp_path / "labels.slp"
    save_slp(_labels(4), str(src))
    runs = {
        "convert": ["convert", str(src), "-o", str(tmp_path / "convert.slp")],
        "merge": ["merge", str(src), str(src), "-o", str(tmp_path / "merge.slp")],
        "unsplit": ["unsplit", str(src), str(src), "-o", str(tmp_path / "us.slp")],
        "fix": ["fix", str(src), "-o", str(tmp_path / "fix.slp")],
        "filenames": [
            "filenames",
            str(src),
            "-o",
            str(tmp_path / "filenames.slp"),
            "--filename",
            "other.mp4",
        ],
    }
    for name, args in runs.items():
        result = CliRunner().invoke(cli, [*args, "--slp-version", version])
        assert result.exit_code == 0, (name, result.output)
        out = args[args.index("-o") + 1]
        assert _is_v3(out) is v3, name


def test_command_option_errors(tmp_path):
    """Contradicting options are refused before anything is written."""
    src = tmp_path / "labels.slp"
    save_slp(_labels(), str(src))
    out = tmp_path / "out.slp"
    for args, message in (
        (["--legacy-format", "--slp-version", "v3"], "can't be combined"),
        (["--unlabeled-as", "absent"], "--unlabeled-as only applies"),
        (["--slp-version", "v3", "--unlabeled-as", "absent"], "only applies"),
    ):
        result = CliRunner().invoke(cli, ["convert", str(src), "-o", str(out), *args])
        assert result.exit_code != 0
        assert message in _plain(result.output)
        assert not os.path.exists(out)
    # `jpeg` is accepted, as in Python.
    result = CliRunner().invoke(
        cli, ["convert", str(src), "-o", str(out), "--image-format", "jpeg"]
    )
    assert result.exit_code == 0, result.output


# --- Review follow-ups ---------------------------------------------------------------


@pytest.mark.parametrize("version, v3", [("auto", False), ("v3", True)])
def test_options_reach_the_save_of_more_commands(
    tmp_path, slp_minimal_pkg, version, v3
):
    """`embed`, `unembed`, `split` and `transform` (embedded) save as asked."""
    plain = tmp_path / "plain.slp"
    save_slp(_labels(10), str(plain))
    outputs = {
        "embed": (
            ["embed", slp_minimal_pkg, "-o", str(tmp_path / "e.pkg.slp")],
            tmp_path / "e.pkg.slp",
        ),
        "unembed": (
            ["unembed", slp_minimal_pkg, "-o", str(tmp_path / "u.slp")],
            tmp_path / "u.slp",
        ),
        "split": (
            ["split", str(plain), "-o", str(tmp_path / "s"), "--seed", "0"],
            tmp_path / "s" / "train.slp",
        ),
        "transform": (
            [
                "transform",
                slp_minimal_pkg,
                "--scale",
                "0.5",
                "-o",
                str(tmp_path / "t.pkg.slp"),
            ],
            tmp_path / "t.pkg.slp",
        ),
    }
    for name, (args, out) in outputs.items():
        result = CliRunner().invoke(
            cli, [*args, "--slp-version", version, "--compression", "gzip"]
        )
        assert result.exit_code == 0, (name, result.output)
        assert _is_v3(out) is v3, name
        with h5py.File(out, "r") as f:
            table = f["points/x"] if v3 else f["points"]
            assert table.compression == "gzip", name


def test_a_save_sets_the_layout_the_next_save_keeps(tmp_path):
    """After a save, the next default save keeps the layout just written."""
    path = tmp_path / "labels.slp"
    labels = _labels()
    save_slp(labels, str(path), slp_version="v3")
    save_slp(labels, str(path))
    assert _is_v3(path)

    labels = load_slp(str(path))
    save_slp(labels, str(path), slp_version="v2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_slp(labels, str(path))
    assert not _is_v3(path)


def test_a_refused_2x_set_writes_nothing(tmp_path):
    """A strict 2.x set that one file can't fit is refused before any is written."""
    labels_set = LabelsSet({"train": _labels(), "val": _with_unlabeled(_labels())})
    with pytest.raises(ValueError, match="unlabeled"):
        labels_set.save(tmp_path / "out", embed=False, slp_version="v2")
    assert not (tmp_path / "out" / "train.slp").exists()

    src = tmp_path / "labels.slp"
    labels = _labels(10)
    for lf in labels:
        _with_unlabeled(Labels([lf]))
    save_slp(labels, str(src), slp_version="v3")
    result = CliRunner().invoke(
        cli, ["split", str(src), "-o", str(tmp_path / "s"), "--slp-version", "v2"]
    )
    assert result.exit_code != 0
    assert not (tmp_path / "s" / "train.slp").exists()


def test_a_mixed_set_warns(tmp_path):
    """A set where only some labels come from format 3.0 files says so."""
    v3_file = tmp_path / "v3.slp"
    save_slp(_labels(), str(v3_file), slp_version="v3")
    with pytest.warns(SlpLayoutWarning, match="come from format 3.0 files"):
        assert resolve_slp_version([load_slp(str(v3_file)), _labels()]) == "v3"


def test_bad_image_quality_is_refused_before_work(tmp_path):
    """`--image-quality` without a format is refused before the command runs."""
    src = tmp_path / "labels.slp"
    save_slp(_labels(), str(src))
    result = CliRunner().invoke(
        cli, ["fix", str(src), "-o", str(tmp_path / "o.slp"), "--image-quality", "5"]
    )
    assert result.exit_code != 0
    assert "--image-quality" in _plain(result.output)
    assert not (tmp_path / "o.slp").exists()
