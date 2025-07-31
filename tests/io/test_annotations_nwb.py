"""Tests for annotations → NWB export (annotations_nwb)."""

import datetime
import types
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from sleap_io import load_slp

import sys

_fake_cv2 = types.ModuleType("cv2")
_fake_cv2.CAP_PROP_FPS = 5


class _Cap:
    """Pretend every video opens and advertises 30 fps."""
    def isOpened(self):  # noqa: N802
        return True

    def get(self, _):  # noqa: D401
        return 30.0


_fake_cv2.VideoCapture = lambda *_a, **_k: _Cap()  # noqa: E731
_fake_cv2.imwrite = lambda *_a, **_k: True  # noqa: E731
sys.modules.setdefault("cv2", _fake_cv2)     # must exist *before* import


import sleap_io.io.annotations_nwb as ann  # now safe to import – cv2 is present

@pytest.fixture
def nwbfile():
    """Minimal NWBFile identical to the one used in core sleap-io tests."""
    return NWBFile(
        session_description="testing session",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.fixture(autouse=True)
def stub_external(monkeypatch, tmp_path):
    """Neutralise disk/ffmpeg side-effects for *every* test."""
    # ffmpeg calls
    monkeypatch.setattr(ann.subprocess, "run", lambda *_a, **_k: None)

    # Work in an isolated temp-dir so make_mjpeg’s files don’t pollute repo
    monkeypatch.chdir(tmp_path)

    # Lightweight stand-ins for the heavy frame + MJPEG pipeline; individual
    # tests can override if they want to exercise the real versions.
    monkeypatch.setattr(
        ann,
        "get_frames_from_slp",
        lambda _labels, *_a, **_k: (
            [("dummy.png", 0.04)],
            {0: [[0, "dummy_video"]]},
        ),
    )
    monkeypatch.setattr(
        ann,
        "make_mjpeg",
        lambda *_a, **_k: "annotated_frames.avi",
    )

def _expected_video_set(labels):
    return {Path(v.filename).as_posix() for v in labels.videos}


def test_create_skeletons_typical(slp_typical):
    labels = load_slp(slp_typical)

    skeletons, frame_indices, unique = ann.create_skeletons(labels)

    # one NWB skeleton per SLEAP skeleton
    assert len(skeletons.skeletons) == len(labels.skeletons)

    # every video encountered in the labels is represented
    assert set(frame_indices) == _expected_video_set(labels)

    # internal book-keeping matches
    assert set(unique) == {sk.name for sk in labels.skeletons}


def test_create_training_frames_identity_false(slp_typical):
    labels = load_slp(slp_typical)
    skeletons, _, unique = ann.create_skeletons(labels)

    # dummy MJPEG ImageSeries
    dummy_mjpeg = ann.ImageSeries(
        name="dummy",
        description="stand-in",
        format="external",
        external_file=["annotated_frames.avi"],
        starting_frame=[0],
        rate=30.0,
    )

    tf = ann.create_training_frames(
        labels=labels,
        unique_skeletons=unique,
        annotations_mjpeg=dummy_mjpeg,
        frame_map={0: [[0, "dummy"]]},
        identity=False,
    )

    # one training frame per labelled frame
    assert len(tf.training_frames) == len(labels.labeled_frames)


def test_create_source_videos_minimal():
    frame_indices = {"video_01.avi": [0, 1]}
    src_vids, annotated = ann.create_source_videos(frame_indices, "annotated_frames.avi")

    # original + annotated
    assert len(src_vids.image_series) == 2
    assert annotated in src_vids.image_series


def test_write_annotations_nwb_typical(slp_typical, tmp_path):
    labels = load_slp(slp_typical)
    out_path = tmp_path / "annotations_typical.nwb"

    ann.write_annotations_nwb(labels, str(out_path))

    with NWBHDF5IO(str(out_path), "r") as io:
        nwb = io.read()

        # containers created
        assert "behavior" in nwb.processing
        behavior_pm = nwb.processing["behavior"]

        assert "Skeletons" in behavior_pm.data_interfaces
        assert "PoseTraining" in behavior_pm.data_interfaces

        # subject auto-filled for DANDI compliance
        assert nwb.subject.subject_id == "subject1"
        assert nwb.subject.species == "Mus musculus"


def test_write_annotations_rejects_empty(nwbfile, slp_minimal):
    """Mirrors sleap-io’s ValueError path for no predicted instances."""
    labels = load_slp(slp_minimal)

    # emulate sleap-io assertion: must have predicted instances
    with pytest.raises(ValueError):
        ann.create_skeletons(labels)


def test_make_mjpeg_script(tmp_path, monkeypatch):
    """Exercise make_mjpeg with two fake images."""
    img1 = tmp_path / "f1.png"
    img2 = tmp_path / "f2.png"
    img1.write_bytes(b"\x89PNG\r\n\x1a\n")
    img2.write_bytes(b"\x89PNG\r\n\x1a\n")

    image_list = [(str(img1), 0.01), (str(img2), 0.01)]
    frame_map = {0: [[0, "v"]], 1: [[1, "v"]]}

    # capture ffmpeg invocation
    called = {}

    monkeypatch.setattr(ann.subprocess, "run", lambda cmd, *_a, **_k: called.setdefault("cmd", cmd))
    monkeypatch.chdir(tmp_path)

    out = ann.make_mjpeg(image_list, frame_map)

    assert (tmp_path / "input.txt").exists()
    assert (tmp_path / "frame_map.json").exists()
    assert out == "annotated_frames.avi"
    assert called["cmd"][0] == "ffmpeg"