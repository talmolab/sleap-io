"""Tests for nwb_ann I/O functionality."""

import json

import cv2
import numpy as np

import sleap_io.io.nwb_ann as ann


def test_make_mjpeg_basic(tmp_path, monkeypatch):
    # Create dummy frames
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.ones((10, 10, 3), dtype=np.uint8) * 255,
    ]
    durations = [0.5, 1.0]
    frame_map = {0: [[0, "video1"]], 1: [[1, "video1"]]}

    monkeypatch.chdir(tmp_path)

    # Mock MJPEGFrameWriter to avoid actual video writing
    written_frames = []

    class MockMJPEGWriter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write_frame(self, frame):
            written_frames.append(frame)

    # Patch in the ann module since it imports MJPEGFrameWriter
    monkeypatch.setattr(ann, "MJPEGFrameWriter", MockMJPEGWriter)

    output_path = ann.make_mjpeg(frames, durations, frame_map)

    # Check that frame_map.json was created
    json_file = tmp_path / "frame_map.json"
    assert json_file.exists(), "frame_map.json not created"

    loaded = json.loads(json_file.read_text())
    expected = {str(k): v for k, v in frame_map.items()}
    assert loaded == expected

    # Check that frames were written
    assert len(written_frames) == 2
    assert output_path == str(tmp_path / "annotated_frames.avi")


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

    frames, durations, frame_map = ann.get_frames_from_slp(
        labels, mjpeg_frame_duration=5.0
    )

    # Should produce two frames
    assert len(frames) == 2
    # All frames should be numpy arrays
    assert all(isinstance(f, np.ndarray) for f in frames)
    # Durations match (converted from ms to seconds)
    assert durations == [0.005, 0.005]  # 5.0 ms = 0.005 s
    # frame_map keys as integers
    assert set(frame_map.keys()) == {0, 1}
    assert frame_map[0] == [(0, "video1")]
    assert frame_map[1] == [(1, "video1")]


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
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            elif prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            return 0

        def release(self):
            pass

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

    source_videos, annotations_series, video_dims = ann.create_source_videos(
        frame_indices, output_mjpeg, mjpeg_frame_rate
    )

    assert isinstance(source_videos, FakeSourceVideos)
    assert isinstance(annotations_series, FakeImageSeries)
    assert len(source_videos.series) == len(frame_indices) + 1
    assert source_videos.series[-1] is annotations_series
    assert "video1.mp4" in video_dims
    assert "video2.avi" in video_dims


def test_create_training_frames_basic(monkeypatch):
    class DummyVal:
        def __init__(self):
            self.points = [((1, 2), True), ((3, 4), False)]
            self.skeleton = type("Skel", (), {"name": "TestSkel"})()
            self.track = None

    class DummyLF:
        def __init__(self, idx):
            self.frame_idx = idx
            self.instances = [DummyVal()]
            self.video = type("Video", (), {"filename": "video.mp4"})()

    labels = type("Labels", (), {})()
    labels.labeled_frames = [DummyLF(5), DummyLF(7)]

    from sleap_io.io.nwb_ann import Skeleton

    fake_skel = Skeleton(
        name="TestSkel", nodes=[], edges=np.empty((0, 2), dtype="uint8")
    )
    fake_unique = {"TestSkel": fake_skel}
    fake_annotations = object()
    fake_frame_map = {5: [(0, "video")], 7: [(1, "video")]}

    created = []

    class FakeTrainingFrame:
        def __init__(
            self,
            name,
            annotator,
            skeleton_instances,
            source_video,
            source_video_frame_index,
        ):
            self.name = name
            self.annotator = annotator
            self.source_video = source_video
            self.source_video_frame_index = source_video_frame_index
            created.append(self)

    class FakeTrainingFrames:
        def __init__(self, training_frames):
            self.training_frames = training_frames

    monkeypatch.setattr(ann, "TrainingFrame", FakeTrainingFrame)
    monkeypatch.setattr(ann, "TrainingFrames", FakeTrainingFrames)

    result = ann.create_training_frames(
        labels, fake_unique, fake_annotations, fake_frame_map, annotator="test"
    )

    assert isinstance(result, FakeTrainingFrames)
    assert len(result.training_frames) == 2
    assert result.training_frames[0].name == "frame_0"
    assert created[0].annotator == "test"  # Check annotator was set
    assert int(result.training_frames[0].source_video_frame_index) == 0
    assert int(result.training_frames[1].source_video_frame_index) == 1
    assert result.training_frames[0].source_video is fake_annotations


def test_write_annotations_nwb_success(tmp_path, monkeypatch):
    fake_skeletons = object()
    fake_frame_indices = {}
    fake_unique = {}
    fake_frames = []
    fake_durations = []
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
        ann,
        "get_frames_from_slp",
        lambda labels: (fake_frames, fake_durations, fake_frame_map),
    )
    monkeypatch.setattr(
        ann, "make_mjpeg", lambda frames, durations, fmap, output_dir=None: fake_mjpeg
    )
    monkeypatch.setattr(
        ann,
        "create_source_videos",
        lambda fidx, mjp, rate=None, include_devices=False, nwbfile=None: (
            fake_source_videos,
            fake_annotations,
            {},
        ),
    )
    monkeypatch.setattr(
        ann,
        "create_training_frames",
        lambda labels, uniq, ann_mjp, fmap, annotator="SLEAP": fake_training,
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


def test_mjpeg_integration(tmp_path, monkeypatch):
    """Test that MJPEGFrameWriter actually creates a valid MJPEG video."""
    from sleap_io.io.video_writing import MJPEGFrameWriter

    # Create some test frames
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8),  # Black frame
        np.ones((100, 100, 3), dtype=np.uint8) * 128,  # Gray frame
        np.ones((100, 100, 3), dtype=np.uint8) * 255,  # White frame
    ]

    mjpeg_path = tmp_path / "test.avi"

    # Write frames to MJPEG
    with MJPEGFrameWriter(
        filename=mjpeg_path,
        fps=10.0,
        quality=2,
    ) as writer:
        for frame in frames:
            writer.write_frame(frame)

    # Verify the file was created
    assert mjpeg_path.exists()
    assert mjpeg_path.stat().st_size > 0

    # Verify we can read it back with cv2
    cap = cv2.VideoCapture(str(mjpeg_path))
    assert cap.isOpened()

    # Read frames back and verify count
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()
    assert frame_count == 3
