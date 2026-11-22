"""Tests for the readings that are opt-in in sleap-io 0.10.

By default, operations keep what sleap-io 0.9 did wherever a newer reading would
create content only SLP format 3.0 stores, or change what is in a file: Ultralytics
visibility codes, nodes added by skeleton changes and merges, and the points of
files older than format 3.0. Each newer reading is an option, and the format 3.0
defaults (`sio.set_v3_defaults`) turn all of them on.
"""

from __future__ import annotations

import shutil
import warnings

import h5py
import numpy as np
import pytest

import sleap_io as sio
from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video, load_slp, save_slp
from sleap_io.io.format_defaults import V3_DEFAULTS_ENV
from sleap_io.io.ultralytics import denormalize_coordinates
from sleap_io.model.instance import PointsArray

STATES = sio.PointState


@pytest.fixture(autouse=True)
def _no_switch(monkeypatch):
    """Run each test with the format 3.0 defaults off, whatever the environment."""
    monkeypatch.delenv(V3_DEFAULTS_ENV, raising=False)
    sio.set_v3_defaults(None)
    yield
    sio.set_v3_defaults(None)


# --- Ultralytics ------------------------------------------------------------------


def test_ultralytics_default_reading_is_the_pre_0_10_one():
    """v=0 reads as hidden and v=1 as visible, unless asked otherwise."""
    rows = [(0.5, 0.25, 0), (0.25, 0.5, 1), (0.125, 0.125, 2)]
    points = denormalize_coordinates(rows, (400, 800))
    # Columns: x, y, visible, verified, labeled.
    assert np.isnan(points[0, :2]).all()
    np.testing.assert_array_equal(points[0, 2:], [0, 1, 1])
    np.testing.assert_array_equal(points[1], [200, 200, 1, 1, 1])

    ternary = denormalize_coordinates(rows, (400, 800), visibility_encoding="ternary")
    assert ternary[0, 4] == 0  # v=0 is unlabeled.
    assert ternary[1, 2] == 0  # v=1 is occluded.
    with sio.v3_defaults():
        np.testing.assert_array_equal(
            denormalize_coordinates(rows, (400, 800)), ternary
        )


def test_ultralytics_dataset_reads_without_unlabeled_points(ultralytics_dataset):
    """A dataset read with the defaults holds no point marked as not labeled."""
    labels = sio.load_ultralytics(ultralytics_dataset)
    assert all(inst.points["labeled"].all() for inst in labels.instances)
    assert sio.resolve_slp_version(labels) == "v2"


# --- Skeleton changes and merges ----------------------------------------------------


def _instance() -> Instance:
    skeleton = Skeleton(["a", "b"])
    return Instance.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton)


def _assert_hidden(points, index):
    assert points["labeled"][index]
    assert not points["visible"][index]
    assert np.isnan(points["xy"][index]).all()


@pytest.mark.parametrize("how", ["update", "replace_instance", "replace_labels"])
def test_new_nodes_are_hidden_by_default(how):
    """Nodes added by a skeleton change are hidden, as before 0.10."""
    inst = _instance()
    labels = Labels([LabeledFrame(video=Video("v.mp4"), frame_idx=0, instances=[inst])])
    if how == "update":
        inst.skeleton.add_node("c")
        inst.update_skeleton()
    elif how == "replace_instance":
        inst.replace_skeleton(Skeleton(["a", "b", "c"]))
    else:
        labels.replace_skeleton(Skeleton(["a", "b", "c"]))
        inst = labels[0].instances[0]
    _assert_hidden(inst.points, 2)
    assert inst.point_states[2] == STATES.ABSENT
    assert sio.resolve_slp_version(labels) == "v2"


def test_new_nodes_unlabeled_is_opt_in():
    """`unmapped="unlabeled"` or the switch marks new nodes as not labeled."""
    inst = _instance()
    inst.replace_skeleton(Skeleton(["a", "b", "c"]), unmapped="unlabeled")
    assert inst.point_states[2] == STATES.UNLABELED
    with sio.v3_defaults():
        inst = _instance()
        inst.replace_skeleton(Skeleton(["a", "b", "c"]))
        assert inst.point_states[2] == STATES.UNLABELED
        inst = _instance()
        inst.replace_skeleton(Skeleton(["a", "b", "c"]), unmapped="absent")
        assert inst.point_states[2] == STATES.ABSENT
    with pytest.raises(ValueError, match="unmapped must be one of"):
        _instance().replace_skeleton(Skeleton(["a"]), unmapped="nope")


@pytest.mark.parametrize("unmapped, state", [(None, 3), ("unlabeled", 0)])
def test_merge_fills_missing_nodes(unmapped, state):
    """A merge into a larger skeleton fills missing nodes as `unmapped` says."""
    video = Video("v.mp4")
    big = Skeleton(["a", "b", "c"])
    base = Labels(
        [
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[Instance.from_numpy(np.ones((3, 2)), skeleton=big)],
            )
        ]
    )
    small = Skeleton(["c", "b"])
    other = Labels(
        [
            LabeledFrame(
                video=video,
                frame_idx=1,
                instances=[Instance.from_numpy(np.ones((2, 2)), skeleton=small)],
            )
        ]
    )
    base.merge(other, skeleton="overlap", unmapped=unmapped)
    merged = base.find(video, 1)[0].instances[0]
    assert merged.skeleton is big
    assert merged.point_states[0] == state


# --- Loading files older than format 3.0 --------------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
def test_old_files_load_with_their_points_as_stored(
    tmp_path, slp_legacy_hidden_complete, lazy
):
    """Hidden points keep their coordinates, and states stay as stored."""
    with h5py.File(slp_legacy_hidden_complete, "r") as f:
        raw_points = f["points"][()]
        raw_pred = f["pred_points"][()]
    user, _, pred = load_slp(slp_legacy_hidden_complete, lazy=lazy)[0].instances
    # The hidden user point keeps its coordinates.
    assert not user.points["visible"][1]
    np.testing.assert_array_equal(
        user.points["xy"][1], [raw_points["x"][1], raw_points["y"][1]]
    )
    assert user.point_states[1] == STATES.OCCLUDED
    # The visible predicted point without a location stays visible.
    assert pred.points["visible"][1] and np.isnan(pred.points["xy"][1]).all()
    # numpy() masks hidden points, as sleap-io 0.9 did.
    assert np.isnan(user.numpy()[1]).all()

    # Saved back, the points are what the file stored.
    out = tmp_path / "resaved.slp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_slp(load_slp(slp_legacy_hidden_complete, lazy=lazy), str(out))
    with h5py.File(out, "r") as f:
        assert f["metadata"].attrs["format_id"] < 3.0
        for name, raw in (("points", raw_points), ("pred_points", raw_pred)):
            saved = f[name][()]
            for field in ("x", "y", "visible"):
                np.testing.assert_array_equal(saved[field], raw[field])
            np.testing.assert_array_equal(saved["complete"], raw["complete"])


@pytest.mark.parametrize("lazy", [False, True])
def test_old_file_clean_up_is_opt_in(tmp_path, slp_legacy_hidden_complete, lazy):
    """`clean_legacy_points=True` or the switch cleans up an old file's points."""
    for labels in (
        load_slp(slp_legacy_hidden_complete, lazy=lazy, clean_legacy_points=True),
        _load_with_switch(slp_legacy_hidden_complete, lazy),
    ):
        user, _, pred = labels[0].instances
        assert np.isnan(user.points["xy"][1]).all()
        assert user.point_states[1] == STATES.ABSENT
        assert not pred.points["visible"][1]
    # An explicit `False` wins over the switch.
    with sio.v3_defaults():
        user = load_slp(
            slp_legacy_hidden_complete, lazy=lazy, clean_legacy_points=False
        )[0].instances[0]
    assert user.point_states[1] == STATES.OCCLUDED
    with pytest.raises(TypeError):
        load_slp(slp_legacy_hidden_complete, clean_legacy_points="yes")


def _load_with_switch(path, lazy):
    with sio.v3_defaults():
        return load_slp(path, lazy=lazy)


def test_format_3_0_files_are_not_cleaned(tmp_path, slp_legacy_hidden_complete):
    """The clean-up only touches files older than format 3.0."""
    v3 = tmp_path / "v3.slp"
    save_slp(load_slp(slp_legacy_hidden_complete), str(v3), slp_version="v3")
    user = load_slp(str(v3), clean_legacy_points=True)[0].instances[0]
    assert user.point_states[1] == STATES.OCCLUDED


def test_remote_style_load_honours_the_option(tmp_path, slp_legacy_hidden_complete):
    """The option reaches the reader through `load_file` too."""
    path = tmp_path / "copy.slp"
    shutil.copy(slp_legacy_hidden_complete, path)
    user = sio.load_file(str(path), clean_legacy_points=True)[0].instances[0]
    assert user.point_states[1] == STATES.ABSENT


# --- The old field name ---------------------------------------------------------------


def test_complete_is_an_alias_of_verified():
    """`points["complete"]` reads and writes `verified`, with a warning."""
    points = PointsArray.empty(3)
    with pytest.warns(DeprecationWarning, match='now called "verified"'):
        points["complete"] = [True, False, True]
    np.testing.assert_array_equal(points["verified"], [True, False, True])
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(points["complete"], [True, False, True])
    with pytest.warns(DeprecationWarning):
        assert points[["xy", "complete"]].dtype.names == ("xy", "verified")
    inst = _instance()
    with pytest.warns(DeprecationWarning):
        inst.points["complete"][0] = True
    assert inst.points["verified"][0]
    # Other keys don't warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        points["visible"]
        points[0]
        points[["xy", "visible"]]
