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


def test_read_nwb_annotations_skip_empty_instances_and_no_video(tmp_path):
    """Test skipping frames with no instances or no video."""
    import datetime

    from ndx_pose import (
        PoseTraining,
        SkeletonInstances,
        Skeletons,
        TrainingFrame,
        TrainingFrames,
    )
    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from pynwb import NWBHDF5IO, NWBFile

    # Create NWB file
    nwb_path = tmp_path / "test_skip.nwb"

    nwbfile = NWBFile(
        session_description="Test session",
        identifier="test",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="behavior processing module"
    )

    # Add skeleton
    skeleton = NWBSkeleton(
        name="test_skeleton", nodes=["A", "B"], edges=np.array([]).reshape(0, 2)
    )
    skeletons = Skeletons(skeletons=[skeleton])
    behavior_module.add(skeletons)

    # Create training frames with empty skeleton instances (tests line 889)
    empty_skeleton_instances = SkeletonInstances(skeleton_instances=[])
    training_frame1 = TrainingFrame(
        name="empty_frame",
        annotator="test",
        skeleton_instances=empty_skeleton_instances,
        source_video=None,
        source_video_frame_index=np.uint64(0),
    )

    # Create another frame that will have no video (tests line 896)
    # We'll mock _resolve_video_and_frame to return None for this
    training_frame2 = TrainingFrame(
        name="no_video_frame",
        annotator="test",
        skeleton_instances=empty_skeleton_instances,
        source_video=None,
        source_video_frame_index=np.uint64(1),
    )

    training_frames = TrainingFrames(training_frames=[training_frame1, training_frame2])
    pose_training = PoseTraining(training_frames=training_frames)
    behavior_module.add(pose_training)

    # Write NWB file
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    # Mock _resolve_video_and_frame to return None for second frame
    original_resolve = ann._resolve_video_and_frame

    def mock_resolve(mjpeg_frame_idx, inverted_map, video_map, mjpeg_video):
        if mjpeg_frame_idx == 1:
            return None, -1  # No video for frame 1
        return original_resolve(mjpeg_frame_idx, inverted_map, video_map, mjpeg_video)

    with patch.object(ann, "_resolve_video_and_frame", side_effect=mock_resolve):
        result = ann.read_nwb_annotations(str(nwb_path))

    # Both frames should be skipped
    assert len(result.labeled_frames) == 0


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


def test_create_source_videos_unopenable_video(tmp_path):
    """Test error handling for unopenable video."""
    frame_indices = {"nonexistent_video.mp4": [0, 1, 2]}

    with patch("sleap_io.io.nwb_ann.cv2.VideoCapture") as mock_cap:
        mock_cap_instance = mock_cap.return_value
        mock_cap_instance.isOpened.return_value = False

        with pytest.raises(RuntimeError, match="Cannot open video"):
            ann.create_source_videos(
                frame_indices, [], "test_annotations.avi", (640, 480)
            )


def test_create_source_videos_invalid_fps(tmp_path):
    """Test error handling for video with invalid FPS."""
    frame_indices = {"test_video.mp4": [0, 1, 2]}

    with patch("sleap_io.io.nwb_ann.cv2.VideoCapture") as mock_cap:
        mock_cap_instance = mock_cap.return_value
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.return_value = 0  # Invalid FPS

        with pytest.raises(RuntimeError, match="Cannot get fps"):
            ann.create_source_videos(
                frame_indices, [], "test_annotations.avi", (640, 480)
            )


def test_write_annotations_nwb_with_devices(tmp_path, monkeypatch):
    """Test write_annotations_nwb with include_devices=True."""
    import numpy as np

    from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
    from sleap_io.io.video_reading import ImageVideo

    # Create test images
    img_files = []
    for i in range(2):
        img_path = tmp_path / f"frame_{i}.png"
        img = np.zeros((32, 32, 1), dtype=np.uint8)  # Use 32x32 (divisible by 16)
        import imageio

        imageio.imwrite(img_path, img[:, :, 0])
        img_files.append(str(img_path))

    skeleton = Skeleton(nodes=["A", "B"], name="test_skeleton")
    video = Video(filename=img_files, backend=ImageVideo(img_files))

    instance = Instance.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
    )

    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, instances=[instance])],
    )

    # Test with include_devices=True
    ann.write_annotations_nwb(
        labels,
        str(tmp_path / "test_with_devices.nwb"),
        output_dir=str(tmp_path),
        include_devices=True,
        annotator="Test",
        nwb_subject_kwargs={"subject_id": "test"},
    )

    # Verify the NWB file was created
    assert (tmp_path / "test_with_devices.nwb").exists()


def test_get_frames_from_slp_duplicate_frame_indices_same_video(tmp_path, monkeypatch):
    """Test handling of duplicate frame indices within same video."""
    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create two labeled frames with same frame_idx in same video
    video = Video(filename="test_video.mp4", backend=None)
    skeleton = SleapSkeleton(nodes=["A"])
    instance = Instance.from_numpy(np.array([[10, 20, 1.0]]), skeleton=skeleton)

    lf1 = LabeledFrame(video=video, frame_idx=5, instances=[instance])
    lf2 = LabeledFrame(video=video, frame_idx=5, instances=[instance])

    labels = Labels(labeled_frames=[lf1, lf2], videos=[video], skeletons=[skeleton])

    # Mock the backend to return a frame
    mock_backend = MagicMock()
    mock_backend.get_frame.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(video, "backend", mock_backend)

    # This should only extract the frame once (cached)
    frames, durations, frame_map = ann.get_frames_from_slp(labels)

    assert len(frames) == 1  # Only one unique frame extracted
    assert frame_map[5] == [(0, "test_video")]  # Single entry
    mock_backend.get_frame.assert_called_once_with(5)


def test_get_frames_from_slp_duplicate_frame_indices_different_videos(
    tmp_path, monkeypatch
):
    """Test handling of same frame_idx appearing in different videos."""
    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create two videos with frames at the same index
    video1 = Video(filename="video1.mp4", backend=None)
    video2 = Video(filename="video2.mp4", backend=None)
    skeleton = SleapSkeleton(nodes=["A"])
    instance = Instance.from_numpy(np.array([[10, 20, 1.0]]), skeleton=skeleton)

    # Both have frame_idx=5
    lf1 = LabeledFrame(video=video1, frame_idx=5, instances=[instance])
    lf2 = LabeledFrame(video=video2, frame_idx=5, instances=[instance])

    labels = Labels(
        labeled_frames=[lf1, lf2], videos=[video1, video2], skeletons=[skeleton]
    )

    # Mock the backends
    mock_backend1 = MagicMock()
    mock_backend1.get_frame.return_value = np.ones((10, 10, 3), dtype=np.uint8)
    mock_backend2 = MagicMock()
    mock_backend2.get_frame.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(video1, "backend", mock_backend1)
    monkeypatch.setattr(video2, "backend", mock_backend2)

    # This should extract both frames
    frames, durations, frame_map = ann.get_frames_from_slp(labels)

    assert len(frames) == 2  # Two frames extracted
    # frame_map[5] should have two entries - this tests lines 181-182
    assert len(frame_map[5]) == 2
    assert (0, "video1") in frame_map[5]
    assert (1, "video2") in frame_map[5]


def test_get_frames_from_slp_duplicate_frames(tmp_path, monkeypatch):
    """Test get_frames_from_slp with duplicate frame indices."""
    import numpy as np

    from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video
    from sleap_io.io.video_reading import ImageVideo

    # Create test images
    img_files = []
    for i in range(2):
        img_path = tmp_path / f"frame_{i}.png"
        img = np.zeros((32, 32, 1), dtype=np.uint8)
        import imageio

        imageio.imwrite(img_path, img[:, :, 0])
        img_files.append(str(img_path))

    skeleton = Skeleton(nodes=["A", "B"], name="test_skeleton")
    video = Video(filename=img_files, backend=ImageVideo(img_files))

    instance1 = Instance.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
    )
    instance2 = Instance.from_numpy(
        np.array([[5.0, 6.0], [7.0, 8.0]]), skeleton=skeleton
    )

    # Create labels with duplicate frame indices
    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
        labeled_frames=[
            LabeledFrame(video=video, frame_idx=0, instances=[instance1]),
            LabeledFrame(video=video, frame_idx=0, instances=[instance2]),  # Same frame
        ],
    )

    frames, durations, frame_map = ann.get_frames_from_slp(labels)

    # Should only have 1 frame despite 2 labeled frames with same index
    assert len(frames) == 1
    assert len(durations) == 1
    assert len(frame_map) == 1
    assert 0 in frame_map  # Frame 0 should be in map


def test_create_skeletons_with_missing_edge_nodes():
    """Test create_skeletons with edges referencing non-existent nodes."""
    import warnings

    import numpy as np

    from sleap_io import Edge, Instance, LabeledFrame, Labels, Node, Skeleton, Video

    # Create nodes
    node_a = Node("A")
    node_b = Node("B")
    node_c = Node("C")  # This node won't be in the skeleton

    # Create skeleton with edges pointing to non-existent nodes
    skeleton = Skeleton(
        nodes=[node_a, node_b],  # Only 2 nodes
        edges=[Edge(node_a, node_b), Edge(node_b, node_c)],  # Edge to non-existent "C"
        name="test_skeleton",
    )

    # Mock video
    video = Video(filename=["test.png"], backend=None)

    instance = Instance.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),  # Only 2 points for A, B
        skeleton=skeleton,
    )

    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, instances=[instance])],
    )

    # Should warn about missing nodes and continue
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        skeletons, frame_indices, unique_skeletons = ann.create_skeletons(labels)

        # Should have warned about missing edge
        assert len(w) == 1
        assert "Skipped 1 edges" in str(w[0].message)
        assert "missing nodes" in str(w[0].message)


def test_load_frame_map_malformed_json(tmp_path):
    """Test read_nwb_annotations with malformed frame_map.json."""
    import datetime
    import warnings

    from ndx_pose import (
        PoseTraining,
        Skeletons,
        TrainingFrames,
    )
    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from pynwb import NWBHDF5IO, NWBFile

    # Create malformed JSON file
    malformed_json_path = tmp_path / "frame_map.json"
    malformed_json_path.write_text("{ invalid json content")

    # Create a minimal valid NWB file with real objects
    nwb_path = tmp_path / "test.nwb"

    # Create NWB file
    nwbfile = NWBFile(
        session_description="Test session",
        identifier="test",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    # Add behavior module
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="behavior processing module"
    )

    # Create and add skeletons
    skeleton = NWBSkeleton(
        name="test_skeleton", nodes=["A", "B"], edges=np.array([]).reshape(0, 2)
    )
    skeletons = Skeletons(skeletons=[skeleton])
    behavior_module.add(skeletons)

    # Create empty PoseTraining
    training_frames = TrainingFrames(training_frames=[])
    pose_training = PoseTraining(training_frames=training_frames)
    behavior_module.add(pose_training)

    # Write NWB file
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    # Should warn about malformed JSON and continue
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ann.read_nwb_annotations(
            str(nwb_path), frame_map_path=str(malformed_json_path)
        )

        # Should warn about JSON decode error
        assert any(
            "Could not load frame_map.json" in str(warning.message) for warning in w
        )
        assert isinstance(result, Labels)


def test_read_nwb_annotations_missing_skeletons(tmp_path):
    """Test error handling for missing Skeletons data."""
    nwb_path = tmp_path / "test.nwb"

    with patch("sleap_io.io.nwb_ann.NWBHDF5IO") as mock_io:
        mock_nwbfile = MagicMock()
        mock_behavior = MagicMock()
        mock_nwbfile.processing = {"behavior": mock_behavior}
        mock_behavior.data_interfaces = {}  # No Skeletons

        mock_io_instance = mock_io.return_value.__enter__.return_value
        mock_io_instance.read.return_value = mock_nwbfile

        with pytest.raises(ValueError, match="Skeletons"):
            ann.read_nwb_annotations(str(nwb_path))


def test_read_nwb_annotations_missing_pose_training(tmp_path):
    """Test error handling for missing PoseTraining data."""
    nwb_path = tmp_path / "test.nwb"

    with patch("sleap_io.io.nwb_ann.NWBHDF5IO") as mock_io:
        mock_nwbfile = MagicMock()
        mock_behavior = MagicMock()
        mock_skeletons = MagicMock()

        mock_nwbfile.processing = {"behavior": mock_behavior}
        mock_behavior.data_interfaces = {"Skeletons": mock_skeletons}
        # No PoseTraining in data_interfaces

        mock_io_instance = mock_io.return_value.__enter__.return_value
        mock_io_instance.read.return_value = mock_nwbfile

        with pytest.raises(ValueError, match="PoseTraining"):
            ann.read_nwb_annotations(str(nwb_path))


def test_read_nwb_annotations_with_load_source_videos(tmp_path):
    """Test read_nwb_annotations with load_source_videos=True."""
    import datetime

    from ndx_pose import (
        PoseTraining,
        Skeletons,
        SourceVideos,
        TrainingFrames,
    )
    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.image import ImageSeries

    # Create NWB file with videos
    nwb_path = tmp_path / "test_load_videos.nwb"

    nwbfile = NWBFile(
        session_description="Test session",
        identifier="test",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="behavior processing module"
    )

    # Add skeleton
    skeleton = NWBSkeleton(
        name="test_skeleton", nodes=["A", "B"], edges=np.array([]).reshape(0, 2)
    )
    skeletons = Skeletons(skeletons=[skeleton])
    behavior_module.add(skeletons)

    # Create source videos
    mjpeg_series = ImageSeries(
        name="annotated_frames",
        external_file=["test_mjpeg.avi"],
        starting_frame=[0],
        rate=30.0,
        format="external",
    )

    source_video_series = ImageSeries(
        name="original_video_0",
        external_file=["test_video.mp4"],
        starting_frame=[0],
        rate=30.0,
        format="external",
    )

    source_videos = SourceVideos(image_series=[source_video_series, mjpeg_series])

    # Create empty training frames
    training_frames = TrainingFrames(training_frames=[])
    pose_training = PoseTraining(
        training_frames=training_frames, source_videos=source_videos
    )
    behavior_module.add(pose_training)

    # Write NWB file
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    # Mock Video class to verify it's called without backend=None
    with patch("sleap_io.io.nwb_ann.Video") as mock_video_class:
        mock_video_class.return_value = MagicMock()

        # Read with load_source_videos=True (tests lines 859, 868)
        result = ann.read_nwb_annotations(str(nwb_path), load_source_videos=True)

        # Verify Video was called without backend=None for both videos
        calls = mock_video_class.call_args_list
        assert len(calls) == 2

        # Check that Video was called with just filename (no backend=None)
        for call in calls:
            assert "filename" in call.kwargs
            assert "backend" not in call.kwargs  # No backend parameter when loading

        assert isinstance(result, Labels)


def test_reconstruct_instances_skeleton_from_nwb_reference():
    """Test skeleton matching using NWB skeleton reference."""
    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from ndx_pose import (
        SkeletonInstance,
        SkeletonInstances,
        TrainingFrame,
    )

    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create SLEAP skeleton
    sleap_skeleton = SleapSkeleton(nodes=["A", "B"], name="test_skeleton")
    sleap_skeletons = {"test_skeleton": sleap_skeleton}

    # Create real NWB skeleton
    nwb_skeleton = NWBSkeleton(
        name="test_skeleton",
        nodes=["A", "B"],
        edges=np.array([]).reshape(0, 2),  # Empty edges with correct shape
    )

    # Create real NWB SkeletonInstance with skeleton reference
    skeleton_instance = SkeletonInstance(
        name="some_instance_name",
        id=np.uint64(1),
        node_locations=np.array([[10, 20], [30, 40]]),
        node_visibility=[1.0, 1.0],
        skeleton=nwb_skeleton,
    )

    # Create real SkeletonInstances container
    skeleton_instances = SkeletonInstances(skeleton_instances=[skeleton_instance])

    # Create real TrainingFrame
    training_frame = TrainingFrame(
        name="test_frame",
        annotator="test",
        skeleton_instances=skeleton_instances,
        source_video=None,
        source_video_frame_index=np.uint64(0),
    )

    instances = ann._reconstruct_instances_from_training(
        training_frame, sleap_skeletons, {}
    )

    # Should find skeleton from NWB reference
    assert len(instances) == 1
    assert instances[0].skeleton == sleap_skeleton


def test_reconstruct_instances_skeleton_from_name_parsing():
    """Test skeleton matching by parsing instance name."""
    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from ndx_pose import (
        SkeletonInstance,
        SkeletonInstances,
        TrainingFrame,
    )

    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create SLEAP skeleton
    sleap_skeleton = SleapSkeleton(nodes=["A", "B"], name="test_skeleton")
    sleap_skeletons = {"test_skeleton": sleap_skeleton}

    # Create a dummy NWB skeleton with a different name (to test name parsing)
    dummy_skeleton = NWBSkeleton(
        name="wrong_skeleton", nodes=["X", "Y"], edges=np.array([]).reshape(0, 2)
    )

    # Create real NWB SkeletonInstance with wrong skeleton reference
    skeleton_instance = SkeletonInstance(
        name="test_skeleton.track_1.instance_0",
        id=np.uint64(1),
        node_locations=np.array([[10, 20], [30, 40]]),
        node_visibility=[1.0, 1.0],
        skeleton=dummy_skeleton,  # Wrong NWB skeleton reference
    )
    # Create real SkeletonInstances container
    skeleton_instances = SkeletonInstances(skeleton_instances=[skeleton_instance])

    # Create real TrainingFrame
    training_frame = TrainingFrame(
        name="test_frame",
        annotator="test",
        skeleton_instances=skeleton_instances,
        source_video=None,
        source_video_frame_index=np.uint64(0),
    )

    instances = ann._reconstruct_instances_from_training(
        training_frame, sleap_skeletons, {}
    )

    # Should find skeleton by parsing name
    assert len(instances) == 1
    assert instances[0].skeleton == sleap_skeleton


def test_reconstruct_instances_skeleton_from_prefix_fallback():
    """Test skeleton matching using prefix fallback."""
    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from ndx_pose import (
        SkeletonInstance,
        SkeletonInstances,
        TrainingFrame,
    )

    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create SLEAP skeleton
    sleap_skeleton = SleapSkeleton(nodes=["A", "B"], name="test_skeleton")
    sleap_skeletons = {"test_skeleton": sleap_skeleton}

    # Create a dummy NWB skeleton with a different name
    dummy_skeleton = NWBSkeleton(
        name="wrong_skeleton", nodes=["X", "Y"], edges=np.array([]).reshape(0, 2)
    )

    # Create real NWB SkeletonInstance with name that starts with skeleton name
    skeleton_instance = SkeletonInstance(
        name="test_skeleton_some_other_format",
        id=np.uint64(1),
        node_locations=np.array([[10, 20], [30, 40]]),
        node_visibility=[1.0, 1.0],
        skeleton=dummy_skeleton,  # Wrong NWB skeleton reference
    )

    # Create real SkeletonInstances container
    skeleton_instances = SkeletonInstances(skeleton_instances=[skeleton_instance])

    # Create real TrainingFrame
    training_frame = TrainingFrame(
        name="test_frame",
        annotator="test",
        skeleton_instances=skeleton_instances,
        source_video=None,
        source_video_frame_index=np.uint64(0),
    )

    instances = ann._reconstruct_instances_from_training(
        training_frame, sleap_skeletons, {}
    )

    # Should find skeleton by prefix matching
    assert len(instances) == 1
    assert instances[0].skeleton == sleap_skeleton


def test_create_training_frames_video_name_mismatch_fallback():
    """Test frame mapping fallback when video name doesn't match."""
    from ndx_pose import Skeleton as NWBSkeleton
    from pynwb.image import ImageSeries

    from sleap_io.model.instance import Instance
    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create instance with skeleton
    sleap_skeleton = SleapSkeleton(nodes=["A"], name="test_skeleton")
    instance = Instance.from_numpy(np.array([[10, 20, 1.0]]), skeleton=sleap_skeleton)

    # Create labeled frame with specific video name
    mock_lf = LabeledFrame(
        video=Video(filename="my_video.mp4", backend=None),
        frame_idx=5,
        instances=[instance],
    )

    labels = Labels(
        labeled_frames=[mock_lf], videos=[mock_lf.video], skeletons=[sleap_skeleton]
    )

    # Create NWB skeleton
    nwb_skeleton = NWBSkeleton(
        name="test_skeleton", nodes=["A"], edges=np.array([]).reshape(0, 2)
    )
    unique_skeletons = {"test_skeleton": nwb_skeleton}

    # Create mock ImageSeries
    annotations_mjpeg = MagicMock(spec=ImageSeries)

    # Frame map has mapping but for different video name
    # This tests line 426 - fallback to mapped[0][0]
    frame_map = {
        5: [(10, "different_video")]  # Video name doesn't match "my_video"
    }

    training_frames = ann.create_training_frames(
        labels, unique_skeletons, annotations_mjpeg, frame_map
    )

    # Should use the fallback index 10 from mapped[0][0]
    assert len(training_frames.training_frames) == 1
    tf = training_frames.training_frames["frame_0"]
    assert tf.source_video_frame_index == 10


def test_create_training_frames_missing_frame_mapping():
    """Test create_training_frames with missing frame mapping."""
    import warnings

    from ndx_pose import Skeleton as NWBSkeleton
    from pynwb.image import ImageSeries

    from sleap_io.model.instance import Instance
    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create a proper Instance with skeleton
    sleap_skeleton = SleapSkeleton(nodes=["A"], name="test_skeleton")
    instance = Instance.from_numpy(np.array([[10, 20, 1.0]]), skeleton=sleap_skeleton)

    # Create mock labeled frame with frame_idx that's not in frame_map
    mock_lf = LabeledFrame(
        video=Video(filename="test_video.mp4", backend=None),
        frame_idx=99,  # Not in frame_map
        instances=[instance],
    )

    # Create Labels object
    labels = Labels(
        labeled_frames=[mock_lf], videos=[mock_lf.video], skeletons=[sleap_skeleton]
    )

    # Create NWB skeleton
    nwb_skeleton = NWBSkeleton(
        name="test_skeleton", nodes=["A"], edges=np.array([]).reshape(0, 2)
    )
    unique_skeletons = {"test_skeleton": nwb_skeleton}

    # Create mock ImageSeries for annotations_mjpeg
    annotations_mjpeg = MagicMock(spec=ImageSeries)

    frame_map = {}  # Empty frame map

    # Should warn about missing frame mapping
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ann.create_training_frames(
            labels, unique_skeletons, annotations_mjpeg, frame_map
        )

        # Should warn about missing frame mapping
        assert len(w) == 1
        assert "No frame mapping found" in str(w[0].message)


def test_reconstruct_instances_multiple_skeletons_no_match():
    """Test warning when multiple skeletons exist but none match."""
    import warnings

    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from ndx_pose import (
        SkeletonInstance,
        SkeletonInstances,
        TrainingFrame,
    )

    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create multiple SLEAP skeletons
    skeleton1 = SleapSkeleton(nodes=["A", "B"], name="skeleton1")
    skeleton2 = SleapSkeleton(nodes=["C", "D"], name="skeleton2")
    sleap_skeletons = {"skeleton1": skeleton1, "skeleton2": skeleton2}

    # Create a dummy NWB skeleton
    dummy_skeleton = NWBSkeleton(
        name="unmatched_skeleton", nodes=["X", "Y"], edges=np.array([]).reshape(0, 2)
    )

    # Create NWB SkeletonInstance with unmatched name
    skeleton_instance = SkeletonInstance(
        name="unmatched_skeleton_name",
        id=np.uint64(1),
        node_locations=np.array([[10, 20], [30, 40]]),
        node_visibility=[1.0, 1.0],
        skeleton=dummy_skeleton,  # Reference to unmatched skeleton
    )

    # Create SkeletonInstances container
    skeleton_instances = SkeletonInstances(skeleton_instances=[skeleton_instance])

    # Create TrainingFrame
    training_frame = TrainingFrame(
        name="test_frame",
        annotator="test",
        skeleton_instances=skeleton_instances,
        source_video=None,
        source_video_frame_index=np.uint64(0),
    )

    # Should warn and skip the instance (lines 697-702)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        instances = ann._reconstruct_instances_from_training(
            training_frame, sleap_skeletons, {}
        )

        # Should warn about not finding skeleton
        assert len(w) == 1
        assert "Could not find skeleton for instance" in str(w[0].message)
        assert "Available skeletons: ['skeleton1', 'skeleton2']" in str(w[0].message)

        # Should return empty list (instance was skipped)
        assert len(instances) == 0


def test_resolve_video_and_frame_with_inverted_map():
    """Test _resolve_video_and_frame with inverted frame map."""
    import warnings

    # Mock video
    video1 = Video(filename="video1.mp4", backend=None)
    video_map = {"video1": video1}

    # Test successful resolution
    inverted_map = {("video1", 5): 10}
    result_video, result_frame = ann._resolve_video_and_frame(
        5, inverted_map, video_map, None
    )
    assert result_video == video1
    assert result_frame == 10

    # Test fallback to MJPEG video when frame not found
    mjpeg_video = Video(filename="mjpeg.avi", backend=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_video, result_frame = ann._resolve_video_and_frame(
            99, inverted_map, video_map, mjpeg_video
        )
        # Should fallback to mjpeg_video (line 761)
        assert result_video == mjpeg_video
        assert result_frame == 99
        assert len(w) == 0  # No warning when mjpeg fallback succeeds

    # Test warning when mjpeg_video is None/falsy
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_video, result_frame = ann._resolve_video_and_frame(
            99, inverted_map, video_map, None
        )
        # Should warn and return None when frame can't be mapped
        assert result_video is None
        assert result_frame == -1
        assert len(w) == 1
        assert "Could not map MJPEG frame" in str(w[0].message)

    # Test failure when no MJPEG video available
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_video, result_frame = ann._resolve_video_and_frame(
            99, inverted_map, video_map, None
        )
        assert result_video is None
        assert result_frame == -1
        assert len(w) == 1
        assert "Could not map MJPEG frame" in str(w[0].message)


def test_reconstruct_instances_track_parsing_with_instance_in_name():
    """Test track parsing when 'instance' appears in track name."""
    from ndx_pose import (
        Skeleton as NWBSkeleton,
    )
    from ndx_pose import (
        SkeletonInstance,
        SkeletonInstances,
        TrainingFrame,
    )

    from sleap_io.model.skeleton import Skeleton as SleapSkeleton

    # Create SLEAP skeleton
    sleap_skeleton = SleapSkeleton(nodes=["A", "B"], name="test_skeleton")
    sleap_skeletons = {"test_skeleton": sleap_skeleton}

    # Create NWB skeleton
    nwb_skeleton = NWBSkeleton(
        name="test_skeleton", nodes=["A", "B"], edges=np.array([]).reshape(0, 2)
    )

    # Create SkeletonInstance with "instance" in the middle of the name
    # This should break at the first "instance" (line 717)
    skeleton_instance = SkeletonInstance(
        name="test_skeleton.instance_track.instance_0",
        id=np.uint64(1),
        node_locations=np.array([[10, 20], [30, 40]]),
        node_visibility=[1.0, 1.0],
        skeleton=nwb_skeleton,
    )

    # Create containers
    skeleton_instances = SkeletonInstances(skeleton_instances=[skeleton_instance])

    training_frame = TrainingFrame(
        name="test_frame",
        annotator="test",
        skeleton_instances=skeleton_instances,
        source_video=None,
        source_video_frame_index=np.uint64(0),
    )

    # Reconstruct with track dict
    tracks = {}
    instances = ann._reconstruct_instances_from_training(
        training_frame, sleap_skeletons, tracks
    )

    # Should not create a track because it breaks at first "instance"
    assert len(instances) == 1
    assert instances[0].track is None  # No track should be created
    assert len(tracks) == 0  # No tracks added


def test_resolve_video_and_frame_no_inverted_map():
    """Test _resolve_video_and_frame without inverted frame map."""
    # Test with MJPEG video
    mjpeg_video = Video(filename="mjpeg.avi", backend=None)
    result_video, result_frame = ann._resolve_video_and_frame(5, None, {}, mjpeg_video)
    assert result_video == mjpeg_video
    assert result_frame == 5

    # Test with video map fallback
    video1 = Video(filename="video1.mp4", backend=None)
    video_map = {"video1": video1}
    result_video, result_frame = ann._resolve_video_and_frame(5, None, video_map, None)
    assert result_video == video1
    assert result_frame == 5

    # Test with no videos - creates placeholder
    result_video, result_frame = ann._resolve_video_and_frame(5, None, {}, None)
    assert result_video.filename == "unknown_video.mp4"
    assert result_frame == 5


def test_sanitize_nwb_name():
    """Test that NWB names are properly sanitized."""
    # Test with path-like name containing slashes and colons
    path_name = (
        "M:/talmo/data/leap_datasets/BermanFlies/"
        "2018-05-03_cluster-sampled.k=10,n=150.labels.mat"
    )
    sanitized = ann._sanitize_nwb_name(path_name)
    assert "/" not in sanitized
    assert ":" not in sanitized
    expected = (
        "M__talmo_data_leap_datasets_BermanFlies_"
        "2018-05-03_cluster-sampled.k=10,n=150.labels.mat"
    )
    assert sanitized == expected

    # Test with normal name
    normal_name = "skeleton_name"
    assert ann._sanitize_nwb_name(normal_name) == normal_name

    # Test with mixed invalid characters
    mixed_name = "prefix:middle/suffix"
    sanitized = ann._sanitize_nwb_name(mixed_name)
    assert sanitized == "prefix_middle_suffix"


def test_create_skeletons_with_invalid_names():
    """Test that skeletons with invalid NWB names are properly sanitized."""
    # Create a skeleton with invalid characters in name (file path)
    skeleton_name = "C:/path/to/file:with:colons.mat"
    skeleton = Skeleton(nodes=["node1", "node2"], name=skeleton_name)

    # Create instances and labels
    instance = Instance.from_numpy(
        np.array([[10, 20, 1.0], [30, 40, 1.0]]), skeleton=skeleton
    )
    video = Video(filename="test_video.mp4", backend=None)
    labeled_frame = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    labels = Labels(
        labeled_frames=[labeled_frame], videos=[video], skeletons=[skeleton]
    )

    # Create NWB skeletons
    skeletons, frame_indices, unique_skeletons = ann.create_skeletons(labels)

    # Check that the skeleton name was sanitized
    sanitized_name = "C__path_to_file_with_colons.mat"
    assert sanitized_name in unique_skeletons
    assert unique_skeletons[sanitized_name].name == sanitized_name

    # Verify the Skeletons container has the sanitized skeleton
    assert len(skeletons.skeletons) == 1
    nwb_skeleton = skeletons.skeletons[sanitized_name]
    assert nwb_skeleton.name == sanitized_name
