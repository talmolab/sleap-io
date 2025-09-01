"""Tests for nwb_ann I/O functionality."""

import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import sleap_io.io.nwb_ann as ann
from sleap_io.model.instance import Instance, Track
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


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
        lambda labels,
        uniq,
        ann_mjp,
        fmap,
        identity=False,
        annotator="SLEAP": fake_training,
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

    # Create a mock Labels object with empty iterator
    labels = type("Labels", (), {"__iter__": lambda self: iter([])})()
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


def test_extract_skeletons_from_nwb():
    """Test extracting SLEAP skeletons from NWB Skeletons container."""
    # Create mock NWB skeleton
    mock_nwb_skeleton = MagicMock()
    mock_nwb_skeleton.nodes = ["head", "thorax", "tail"]
    mock_nwb_skeleton.edges = np.array([[0, 1], [1, 2]], dtype="uint8")

    # Create mock Skeletons container
    mock_skeletons_container = MagicMock()
    mock_skeletons_container.skeletons = {"test_skeleton": mock_nwb_skeleton}

    # Extract skeletons
    sleap_skeletons = ann._extract_skeletons_from_nwb(mock_skeletons_container)

    assert "test_skeleton" in sleap_skeletons
    skeleton = sleap_skeletons["test_skeleton"]
    assert skeleton.name == "test_skeleton"
    assert skeleton.node_names == ["head", "thorax", "tail"]
    assert skeleton.edge_inds == [(0, 1), (1, 2)]


def test_load_frame_map(tmp_path):
    """Test loading frame map from JSON."""
    # Create test frame map
    frame_map_data = {"0": [[0, "video1"], [1, "video2"]], "5": [[2, "video1"]]}

    frame_map_path = tmp_path / "frame_map.json"
    with open(frame_map_path, "w") as f:
        json.dump(frame_map_data, f)

    # Load and verify
    loaded_map = ann._load_frame_map(frame_map_path)

    assert loaded_map[0] == [(0, "video1"), (1, "video2")]
    assert loaded_map[5] == [(2, "video1")]


def test_invert_frame_map():
    """Test inverting frame map for lookup."""
    frame_map = {0: [(0, "video1"), (1, "video2")], 5: [(2, "video1")]}

    inverted = ann._invert_frame_map(frame_map)

    assert inverted[("video1", 0)] == 0
    assert inverted[("video2", 1)] == 0
    assert inverted[("video1", 2)] == 5


def test_reconstruct_instances_from_training():
    """Test reconstructing SLEAP instances from NWB TrainingFrame."""
    # Create mock skeleton instance
    mock_skeleton_instance = MagicMock()
    mock_skeleton_instance.name = "test_skeleton.instance_0"
    mock_skeleton_instance.node_locations = np.array([[10, 20], [30, 40]])
    mock_skeleton_instance.node_visibility = [1.0, 0.5]
    mock_skeleton_instance.skeleton = MagicMock(name="test_skeleton")

    # Create mock skeleton instances container
    mock_skeleton_instances = MagicMock()
    mock_skeleton_instances.skeleton_instances = {"instance_0": mock_skeleton_instance}

    # Create mock training frame
    mock_training_frame = MagicMock()
    mock_training_frame.skeleton_instances = mock_skeleton_instances

    # Create SLEAP skeleton
    sleap_skeleton = Skeleton(name="test_skeleton", nodes=["node1", "node2"])
    sleap_skeletons = {"test_skeleton": sleap_skeleton}

    # Reconstruct instances
    instances = ann._reconstruct_instances_from_training(
        mock_training_frame, sleap_skeletons
    )

    assert len(instances) == 1
    instance = instances[0]
    assert instance.skeleton == sleap_skeleton
    assert np.allclose(instance.numpy()[:, :2], [[10, 20], [30, 40]])


def test_read_nwb_annotations_basic(tmp_path):
    """Test basic reading of NWB annotations."""
    # Create a simple NWB file with mocked PoseTraining data
    nwb_path = tmp_path / "test.nwb"
    frame_map_path = tmp_path / "frame_map.json"

    # Create frame map
    frame_map = {"0": [[0, "test_video"]], "1": [[1, "test_video"]]}
    with open(frame_map_path, "w") as f:
        json.dump(frame_map, f)

    # Mock NWB file reading
    with patch("sleap_io.io.nwb_ann.NWBHDF5IO") as mock_io:
        # Setup mock NWB file structure
        mock_nwbfile = MagicMock()
        mock_behavior_pm = MagicMock()

        # Mock Skeletons container
        mock_skeleton = MagicMock()
        mock_skeleton.nodes = ["head", "tail"]
        mock_skeleton.edges = np.array([[0, 1]], dtype="uint8")
        mock_skeletons = MagicMock()
        mock_skeletons.skeletons = {"test_skeleton": mock_skeleton}

        # Mock PoseTraining container
        mock_pose_training = MagicMock()
        mock_pose_training.__class__.__name__ = "PoseTraining"

        # Mock source videos
        mock_image_series = MagicMock()
        mock_image_series.external_file = ["test_video.mp4"]
        mock_source_videos = MagicMock()
        mock_source_videos.image_series = {"original_video_0": mock_image_series}
        mock_pose_training.source_videos = mock_source_videos

        # Mock training frames
        mock_training_frame = MagicMock()
        mock_training_frame.source_video_frame_index = 0

        # Mock skeleton instance
        mock_skel_instance = MagicMock()
        mock_skel_instance.name = "test_skeleton.instance_0"
        mock_skel_instance.node_locations = np.array([[10, 20], [30, 40]])
        mock_skel_instance.node_visibility = [1.0, 1.0]
        mock_skel_instance.skeleton = mock_skeleton

        mock_skel_instances = MagicMock()
        mock_skel_instances.skeleton_instances = {"instance_0": mock_skel_instance}
        mock_training_frame.skeleton_instances = mock_skel_instances

        mock_training_frames = MagicMock()
        mock_training_frames.training_frames = {"frame_0": mock_training_frame}
        mock_pose_training.training_frames = mock_training_frames

        # Setup behavior processing module
        mock_behavior_pm.data_interfaces = {
            "Skeletons": mock_skeletons,
            "PoseTraining": mock_pose_training,
        }

        # Make PoseTraining identifiable
        from ndx_pose import PoseTraining

        mock_pose_training.__class__ = PoseTraining

        mock_nwbfile.processing = {"behavior": mock_behavior_pm}

        # Setup IO mock
        mock_io_instance = mock_io.return_value.__enter__.return_value
        mock_io_instance.read.return_value = mock_nwbfile

        # Read the annotations
        labels = ann.read_nwb_annotations(
            nwb_path=str(nwb_path),
            frame_map_path=str(frame_map_path),
            load_source_videos=False,
        )

        # Verify results
        assert isinstance(labels, Labels)
        assert len(labels.skeletons) == 1
        assert labels.skeletons[0].name == "test_skeleton"
        assert len(labels.labeled_frames) == 1
        assert labels.labeled_frames[0].frame_idx == 0


def test_read_nwb_annotations_no_frame_map(tmp_path):
    """Test reading NWB annotations without frame map."""
    nwb_path = tmp_path / "test.nwb"

    with patch("sleap_io.io.nwb_ann.NWBHDF5IO") as mock_io:
        # Setup minimal mock structure
        mock_nwbfile = MagicMock()
        mock_behavior_pm = MagicMock()

        # Mock Skeletons
        mock_skeleton = MagicMock()
        mock_skeleton.nodes = ["node1"]
        mock_skeleton.edges = np.array([], dtype="uint8")
        mock_skeletons = MagicMock()
        mock_skeletons.skeletons = {"skel1": mock_skeleton}

        # Mock PoseTraining
        mock_pose_training = MagicMock()
        from ndx_pose import PoseTraining

        mock_pose_training.__class__ = PoseTraining

        # Empty source videos and training frames
        mock_pose_training.source_videos = None
        mock_training_frames = MagicMock()
        mock_training_frames.training_frames = {}
        mock_pose_training.training_frames = mock_training_frames

        mock_behavior_pm.data_interfaces = {
            "Skeletons": mock_skeletons,
            "PoseTraining": mock_pose_training,
        }

        mock_nwbfile.processing = {"behavior": mock_behavior_pm}
        mock_io_instance = mock_io.return_value.__enter__.return_value
        mock_io_instance.read.return_value = mock_nwbfile

        # Should work without frame map
        labels = ann.read_nwb_annotations(str(nwb_path))

        assert isinstance(labels, Labels)
        assert len(labels.skeletons) == 1
        assert len(labels.labeled_frames) == 0  # No training frames


def test_reconstruct_instances_with_tracks():
    """Test reconstructing instances with track identity."""
    # Create mock skeleton instances with tracks
    mock_instance1 = MagicMock()
    mock_instance1.name = "mouse_skeleton.track_1.instance_0"
    mock_instance1.node_locations = np.array([[10, 20], [30, 40]])
    mock_instance1.node_visibility = [1.0, 1.0]
    mock_instance1.skeleton = MagicMock(name="mouse_skeleton")

    mock_instance2 = MagicMock()
    mock_instance2.name = "mouse_skeleton.track_2.instance_1"
    mock_instance2.node_locations = np.array([[50, 60], [70, 80]])
    mock_instance2.node_visibility = [0.9, 0.8]
    mock_instance2.skeleton = MagicMock(name="mouse_skeleton")

    # Create mock skeleton instances container
    mock_skeleton_instances = MagicMock()
    mock_skeleton_instances.skeleton_instances = {
        "instance_0": mock_instance1,
        "instance_1": mock_instance2,
    }

    # Create mock training frame
    mock_training_frame = MagicMock()
    mock_training_frame.skeleton_instances = mock_skeleton_instances

    # Create SLEAP skeleton
    sleap_skeleton = Skeleton(name="mouse_skeleton", nodes=["head", "tail"])
    sleap_skeletons = {"mouse_skeleton": sleap_skeleton}

    # Reconstruct instances
    tracks = {}
    instances = ann._reconstruct_instances_from_training(
        mock_training_frame, sleap_skeletons, tracks
    )

    assert len(instances) == 2
    assert len(tracks) == 2

    # Check first instance
    assert instances[0].skeleton == sleap_skeleton
    assert instances[0].track.name == "track_1"
    assert np.allclose(instances[0].numpy()[:, :2], [[10, 20], [30, 40]])

    # Check second instance
    assert instances[1].skeleton == sleap_skeleton
    assert instances[1].track.name == "track_2"
    assert np.allclose(instances[1].numpy()[:, :2], [[50, 60], [70, 80]])

    # Verify tracks are properly created
    assert "track_1" in tracks
    assert "track_2" in tracks
    assert tracks["track_1"] == instances[0].track
    assert tracks["track_2"] == instances[1].track


def test_reconstruct_instances_malformed_names():
    """Test that reconstruction doesn't break with unexpected name formats."""
    # Test various malformed or unexpected instance names
    test_cases = [
        ("completely_random_name", "test_skeleton"),  # No recognizable format
        ("", "test_skeleton"),  # Empty name
        ("instance_0", "test_skeleton"),  # Missing skeleton name
        ("test_skeleton", "test_skeleton"),  # Missing instance marker
        (
            "test.skeleton.with.dots.instance_0",
            "test_skeleton",
        ),  # Skeleton name with dots
    ]

    for instance_name, skeleton_name in test_cases:
        # Create mock skeleton instance
        mock_skeleton_instance = MagicMock()
        mock_skeleton_instance.name = instance_name
        mock_skeleton_instance.node_locations = np.array([[10, 20], [30, 40]])
        mock_skeleton_instance.node_visibility = [1.0, 0.5]

        # Create a mock skeleton object with a name attribute
        mock_nwb_skeleton = MagicMock()
        mock_nwb_skeleton.name = skeleton_name
        mock_skeleton_instance.skeleton = mock_nwb_skeleton

        # Create mock skeleton instances container
        mock_skeleton_instances = MagicMock()
        mock_skeleton_instances.skeleton_instances = {
            "instance_0": mock_skeleton_instance
        }

        # Create mock training frame
        mock_training_frame = MagicMock()
        mock_training_frame.skeleton_instances = mock_skeleton_instances

        # Create SLEAP skeleton
        sleap_skeleton = Skeleton(name=skeleton_name, nodes=["node1", "node2"])
        sleap_skeletons = {skeleton_name: sleap_skeleton}

        # Reconstruct instances - should not raise an error
        instances = ann._reconstruct_instances_from_training(
            mock_training_frame, sleap_skeletons
        )

        # Should successfully create an instance using the NWB skeleton reference
        assert len(instances) == 1
        assert instances[0].skeleton == sleap_skeleton


def test_reconstruct_instances_single_skeleton_fallback():
    """Test fallback to single skeleton when name parsing fails."""
    # Create mock skeleton instance with unparsable name
    mock_skeleton_instance = MagicMock()
    mock_skeleton_instance.name = "some_random_format_12345"
    mock_skeleton_instance.node_locations = np.array([[10, 20]])
    mock_skeleton_instance.node_visibility = [1.0]
    mock_skeleton_instance.skeleton = None  # No NWB skeleton reference

    # Create mock skeleton instances container
    mock_skeleton_instances = MagicMock()
    mock_skeleton_instances.skeleton_instances = {"instance_0": mock_skeleton_instance}

    # Create mock training frame
    mock_training_frame = MagicMock()
    mock_training_frame.skeleton_instances = mock_skeleton_instances

    # Create single SLEAP skeleton
    sleap_skeleton = Skeleton(name="only_skeleton", nodes=["node1"])
    sleap_skeletons = {"only_skeleton": sleap_skeleton}

    # Reconstruct instances - should use the only available skeleton
    with pytest.warns(UserWarning, match="using the only available skeleton"):
        instances = ann._reconstruct_instances_from_training(
            mock_training_frame, sleap_skeletons
        )

    assert len(instances) == 1
    assert instances[0].skeleton == sleap_skeleton


def test_roundtrip_nwb_annotations(tmp_path):
    """Test writing and reading NWB annotations roundtrip."""
    import sleap_io as sio

    # Create test data with tracks
    skeleton = Skeleton(
        name="test_skeleton", nodes=["head", "thorax", "tail"], edges=[[0, 1], [1, 2]]
    )

    # Create video with mock backend
    video = Video(filename="test_video.mp4", backend=None)
    mock_backend = MagicMock()
    mock_backend.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    video.backend = mock_backend

    # Create instances with and without tracks
    track1 = Track(name="mouse1")
    track2 = Track(name="mouse2")

    instances_frame0 = [
        Instance.from_numpy(
            points_data=np.array([[10, 20, 1.0], [30, 40, 0.9], [50, 60, 0.8]]),
            skeleton=skeleton,
            track=track1,
        ),
        Instance.from_numpy(
            points_data=np.array([[70, 80, 1.0], [90, 100, 0.95], [110, 120, 0.85]]),
            skeleton=skeleton,
            track=track2,
        ),
    ]

    instances_frame5 = [
        Instance.from_numpy(
            points_data=np.array([[15, 25, 0.95], [35, 45, 0.88], [55, 65, 0.82]]),
            skeleton=skeleton,
            track=track1,
        )
    ]

    instances_frame10 = [
        Instance.from_numpy(
            points_data=np.array([[100, 110, 0.9], [120, 130, 0.85], [140, 150, 0.8]]),
            skeleton=skeleton,
            track=None,  # Instance without track
        )
    ]

    lf0 = LabeledFrame(video=video, frame_idx=0, instances=instances_frame0)
    lf5 = LabeledFrame(video=video, frame_idx=5, instances=instances_frame5)
    lf10 = LabeledFrame(video=video, frame_idx=10, instances=instances_frame10)

    original_labels = Labels(
        labeled_frames=[lf0, lf5, lf10],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track1, track2],
    )

    # Write to NWB
    nwb_path = tmp_path / "test_roundtrip.nwb"

    # Mock video capture for writing
    with patch("cv2.VideoCapture") as mock_cap:
        mock_video_cap = MagicMock()
        mock_video_cap.isOpened.return_value = True
        mock_video_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        mock_cap.return_value = mock_video_cap

        # Write the annotations (with identity=True to preserve tracks)
        # Note: We need to call the lower-level function directly to pass identity flag
        ann.write_annotations_nwb(
            labels=original_labels,
            nwbfile_path=str(nwb_path),
            output_dir=str(tmp_path),
            include_devices=False,
            annotator="test_annotator",
            nwb_file_kwargs=None,
            nwb_subject_kwargs=None,
        )

    # Verify files were created
    assert nwb_path.exists()
    assert (tmp_path / "frame_map.json").exists()
    assert (tmp_path / "annotated_frames.avi").exists()

    # Read back the annotations
    loaded_labels = sio.load_nwb_annotations(
        str(nwb_path),
        frame_map_path=str(tmp_path / "frame_map.json"),
        load_source_videos=False,
    )

    # Verify structure is preserved
    assert len(loaded_labels.skeletons) == 1
    assert loaded_labels.skeletons[0].name == "test_skeleton"
    assert loaded_labels.skeletons[0].node_names == ["head", "thorax", "tail"]
    assert loaded_labels.skeletons[0].edge_inds == [(0, 1), (1, 2)]

    # Verify frames
    assert len(loaded_labels.labeled_frames) == 3
    frame_indices = [lf.frame_idx for lf in loaded_labels.labeled_frames]
    assert sorted(frame_indices) == [0, 5, 10]

    # Verify instances and tracks
    for lf in loaded_labels.labeled_frames:
        if lf.frame_idx == 0:
            assert len(lf.instances) == 2
            # Check that tracks were preserved
            tracks_found = [
                inst.track.name if inst.track else None for inst in lf.instances
            ]
            assert set(tracks_found) == {"mouse1", "mouse2"}
        elif lf.frame_idx == 5:
            assert len(lf.instances) == 1
            assert lf.instances[0].track.name == "mouse1"
        elif lf.frame_idx == 10:
            assert len(lf.instances) == 1
            assert lf.instances[0].track is None  # No track

    # Verify point data (spot check)
    frame0 = next(lf for lf in loaded_labels.labeled_frames if lf.frame_idx == 0)
    # Find the instance with track "mouse1"
    mouse1_inst = next(
        inst for inst in frame0.instances if inst.track and inst.track.name == "mouse1"
    )
    points = mouse1_inst.numpy()
    assert np.allclose(points[:, :2], [[10, 20], [30, 40], [50, 60]], atol=0.1)


def test_read_nwb_annotations_missing_data(tmp_path):
    """Test error handling for missing PoseTraining data."""
    nwb_path = tmp_path / "test.nwb"

    with patch("sleap_io.io.nwb_ann.NWBHDF5IO") as mock_io:
        mock_nwbfile = MagicMock()
        mock_nwbfile.processing = {}  # No behavior module

        mock_io_instance = mock_io.return_value.__enter__.return_value
        mock_io_instance.read.return_value = mock_nwbfile

        with pytest.raises(ValueError, match="behavior"):
            ann.read_nwb_annotations(str(nwb_path))
