"""Tests for nwb_ann I/O functionality."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

import sleap_io.io.nwb_ann as ann


def test_make_mjpeg_error(tmp_path, monkeypatch):
    image_list = [("img1.png", 0.5), ("img2.jpg", 1.0)]
    frame_map = {0: [[0, "video1"]], 1: [[1, "video1"]]}
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        ann.make_mjpeg(image_list, frame_map)

    txt_file = tmp_path / "input.txt"
    json_file = tmp_path / "frame_map.json"
    assert txt_file.exists(), "input.txt not created"
    assert json_file.exists(), "frame_map.json not created"

    loaded = json.loads(json_file.read_text())
    expected = {str(k): v for k, v in frame_map.items()}
    assert loaded == expected


def test_create_skeletons_basic():
    # Dummy nodes and skeleton
    Node = type("Node", (), {})
    node1 = Node()
    node1.name = "n1"

    node2 = Node()
    node2.name = "n2"

    DummySkel = type("DummySkel", (), {})
    skel = DummySkel()
    skel.name = "TestSkel"
    skel.nodes = [node1, node2]
    skel.edges = [(node1, node2)]

    DummyInst = type("DummyInst", (), {})
    inst = DummyInst()
    inst.skeleton = skel

    # Two labels referencing same video
    DummyVideo = type("Video", (), {})
    label1 = type("Label", (), {})()
    label1.video = DummyVideo()
    label1.video.filename = "vid1.mp4"
    label1.frame_idx = 0
    label1.instances = [inst]

    label2 = type("Label", (), {})()
    label2.video = DummyVideo()
    label2.video.filename = "vid1.mp4"
    label2.frame_idx = 1
    label2.instances = [inst]

    labels = [label1, label2]

    skeletons, frame_indices, unique_skeletons = ann.create_skeletons(labels)

    # Verify unique skeletons
    assert list(unique_skeletons.keys()) == ["TestSkel"]
    sk_obj = unique_skeletons["TestSkel"]
    assert sk_obj.name == "TestSkel"
    assert list(sk_obj.nodes) == ["n1", "n2"]

    # Verify frame_indices mapping
    assert set(frame_indices.keys()) == {"vid1.mp4"}
    assert frame_indices["vid1.mp4"] == [0, 1]

    # Verify Skeletons container
    assert hasattr(skeletons, "skeletons")
    assert len(skeletons.skeletons) == 1
    sk0 = skeletons.skeletons["TestSkel"]
    assert sk0.name == "TestSkel"
    assert list(sk0.nodes) == ["n1", "n2"]


def test_get_frames_from_slp_basic(tmp_path, monkeypatch):
    # Dummy LabeledFrame objects
    class DummyBackend:
        def get_frame(self, idx):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    class DummyVideo:
        def __init__(self, filename):
            self.filename = filename
            self.backend = DummyBackend()

    class DummyLF:
        def __init__(self, frame_idx, filename):
            self.frame_idx = frame_idx
            self.video = DummyVideo(filename)

    labels = type("Labels", (), {})()
    labels.labeled_frames = [DummyLF(0, "video1.mp4"), DummyLF(1, "video1.mp4")]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cv2, "imwrite", lambda fname, img: True)

    image_list, frame_map = ann.get_frames_from_slp(labels, mjpeg_frame_duration=5.0)

    # Should produce two entries
    assert len(image_list) == 2
    # Filenames should include the 'frames' directory
    assert all(Path(path).parent.name == "frames" for path, _ in image_list)
    # Durations match
    assert [dur for _, dur in image_list] == [5.0, 5.0]
    # frame_map keys as integers
    assert set(frame_map.keys()) == {0, 1}
    assert frame_map[0] == [0, "video1"]
    assert frame_map[1] == [1, "video1"]


def test_create_source_videos_basic(monkeypatch):
    frame_indices = {"video1.mp4": [0, 1], "video2.avi": [2]}
    output_mjpeg = "annot.avi"
    mjpeg_frame_rate = 24.0

    class FakeCap:
        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return True

        def get(self, prop):
            return 30.0

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)

    created_series = []

    class FakeImageSeries:
        def __init__(self, *args, **kwargs):
            created_series.append((args, kwargs))

    class FakeSourceVideos:
        def __init__(self, image_series):
            self.series = image_series

    monkeypatch.setattr(ann, "ImageSeries", FakeImageSeries)
    monkeypatch.setattr(ann, "SourceVideos", FakeSourceVideos)

    source_videos, annotations_series = ann.create_source_videos(
        frame_indices, output_mjpeg, mjpeg_frame_rate
    )

    assert isinstance(source_videos, FakeSourceVideos)
    assert isinstance(annotations_series, FakeImageSeries)
    assert len(source_videos.series) == len(frame_indices) + 1
    assert source_videos.series[-1] is annotations_series


def test_create_training_frames_basic(monkeypatch):
    class DummyVal:
        def __init__(self):
            self.points = [((1, 2), True), ((3, 4), False)]
            self.skeleton = type("Skel", (), {"name": "TestSkel"})()

    class DummyLF:
        def __init__(self, idx):
            self.frame_idx = idx
            self.instances = [DummyVal()]

    labels = type("Labels", (), {})()
    labels.labeled_frames = [DummyLF(5), DummyLF(7)]

    from sleap_io.io.nwb_ann import Skeleton

    fake_skel = Skeleton(
        name="TestSkel", nodes=[], edges=np.empty((0, 2), dtype="uint8")
    )
    fake_unique = {"TestSkel": fake_skel}
    fake_annotations = object()
    fake_frame_map = {5: [0, "video"], 7: [1, "video"]}

    created = []

    class FakeTrainingFrame:
        def __init__(
            self, name, skeleton_instances, source_video, source_video_frame_index
        ):
            self.name = name
            self.source_video = source_video
            self.source_video_frame_index = source_video_frame_index
            created.append(self)

    class FakeTrainingFrames:
        def __init__(self, training_frames):
            self.training_frames = training_frames

    monkeypatch.setattr(ann, "TrainingFrame", FakeTrainingFrame)
    monkeypatch.setattr(ann, "TrainingFrames", FakeTrainingFrames)

    result = ann.create_training_frames(
        labels, fake_unique, fake_annotations, fake_frame_map
    )

    assert isinstance(result, FakeTrainingFrames)
    assert len(result.training_frames) == 2
    assert result.training_frames[0].name == "frame_0"
    assert int(result.training_frames[0].source_video_frame_index) == 0
    assert int(result.training_frames[1].source_video_frame_index) == 1
    assert result.training_frames[0].source_video is fake_annotations


def test_write_annotations_nwb_success(tmp_path, monkeypatch):
    fake_skeletons = object()
    fake_frame_indices = {}
    fake_unique = {}
    fake_images = []
    fake_frame_map = {}
    fake_mjpeg = "out.avi"
    fake_source_videos = object()
    fake_annotations = object()
    fake_training = object()

    monkeypatch.setattr(
        ann,
        "create_skeletons",
        lambda labels: (fake_skeletons, fake_frame_indices, fake_unique),
    )
    monkeypatch.setattr(
        ann, "get_frames_from_slp", lambda labels: (fake_images, fake_frame_map)
    )
    monkeypatch.setattr(ann, "make_mjpeg", lambda imgs, fmap: fake_mjpeg)
    monkeypatch.setattr(
        ann,
        "create_source_videos",
        lambda fidx, mjp, rate=None: (fake_source_videos, fake_annotations),
    )
    monkeypatch.setattr(
        ann, "create_training_frames", lambda labels, uniq, ann_mjp, fmap: fake_training
    )

    class FakePose:
        def __init__(self, training_frames, source_videos):
            self.training_frames = training_frames
            self.source_videos = source_videos

    monkeypatch.setattr(ann, "PoseTraining", FakePose)

    class FakeSubject:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ann, "Subject", FakeSubject)

    class FakeNWBFile:
        def __init__(self, **kwargs):
            self.modules = []
            self.kwargs = kwargs

        def create_processing_module(self, name, description):
            mod = type("Mod", (), {"added": []})()

            def add(iface):
                mod.added.append(iface)

            mod.add = add
            self.modules.append((name, mod))
            return mod

    monkeypatch.setattr(ann, "NWBFile", FakeNWBFile)

    created_ios = []

    class FakeIO:
        def __init__(self, path, mode):
            self.path = path
            self.mode = mode
            created_ios.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, nwbfile):
            self.written = nwbfile

    monkeypatch.setattr(ann, "NWBHDF5IO", FakeIO)

    labels = object()
    out_path = tmp_path / "test.nwb"
    ann.write_annotations_nwb(
        labels,
        str(out_path),
        nwb_file_kwargs={"foo": True},
        nwb_subject_kwargs={"subject_id": "sub1"},
    )

    assert len(created_ios) == 1
    io_inst = created_ios[0]
    assert io_inst.path == str(out_path)
    assert io_inst.mode == "w"
    assert hasattr(io_inst, "written")
    nwbfile = io_inst.written
    assert isinstance(nwbfile, FakeNWBFile)

    assert len(nwbfile.modules) == 1
    name, proc = nwbfile.modules[0]
    assert name == "behavior"
    assert proc.added[0] is fake_skeletons
    assert isinstance(proc.added[1], FakePose)
