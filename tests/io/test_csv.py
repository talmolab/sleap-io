"""Tests for CSV I/O operations."""

import json

import numpy as np
import pandas as pd
import pytest

import sleap_io as sio
from sleap_io.io import csv
from sleap_io.model.instance import Instance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_skeleton():
    """Create a simple skeleton with 2 nodes and 1 edge."""
    return Skeleton(
        nodes=["head", "tail"],
        edges=[("head", "tail")],
        name="simple",
    )


@pytest.fixture
def simple_labels(simple_skeleton, tmp_path):
    """Create simple Labels with 2 frames, 1 instance each."""
    video = Video(str(tmp_path / "video.mp4"))

    frames = []
    for i in range(2):
        inst = Instance(
            points={"head": [100 + i, 200 + i], "tail": [150 + i, 250 + i]},
            skeleton=simple_skeleton,
        )
        lf = LabeledFrame(video=video, frame_idx=i, instances=[inst])
        frames.append(lf)

    return Labels(labeled_frames=frames)


@pytest.fixture
def multi_animal_labels(simple_skeleton, tmp_path):
    """Create Labels with multiple tracked animals."""
    video = Video(str(tmp_path / "video.mp4"))
    track1 = sio.Track("animal1")
    track2 = sio.Track("animal2")

    frames = []
    for i in range(2):
        inst1 = Instance(
            points={"head": [100 + i, 200 + i], "tail": [150 + i, 250 + i]},
            skeleton=simple_skeleton,
            track=track1,
        )
        inst2 = Instance(
            points={"head": [300 + i, 400 + i], "tail": [350 + i, 450 + i]},
            skeleton=simple_skeleton,
            track=track2,
        )
        lf = LabeledFrame(video=video, frame_idx=i, instances=[inst1, inst2])
        frames.append(lf)

    return Labels(labeled_frames=frames)


# =============================================================================
# Format Detection Tests
# =============================================================================


class TestDetectFormat:
    """Tests for CSV format detection."""

    def test_detect_sleap_format(self, tmp_path):
        """Test detection of SLEAP Analysis CSV format."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "track,frame_idx,instance.score,head.x,head.y,head.score,"
            "tail.x,tail.y,tail.score\n"
            ",0,0.95,100,200,0.9,150,250,0.85\n"
        )
        assert csv.detect_csv_format(csv_path) == "sleap"

    def test_detect_dlc_format(self, tmp_path):
        """Test detection of DLC format."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "scorer,DLC,DLC,DLC,DLC\n"
            "bodyparts,head,head,tail,tail\n"
            "coords,x,y,x,y\n"
            "img0001.png,100,200,150,250\n"
        )
        assert csv.detect_csv_format(csv_path) == "dlc"

    def test_detect_instances_format(self, tmp_path):
        """Test detection of codec instances format."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "frame_idx,track,score,head.x,head.y,tail.x,tail.y\n"
            "0,animal1,0.95,100,200,150,250\n"
        )
        assert csv.detect_csv_format(csv_path) == "instances"

    def test_detect_points_format(self, tmp_path):
        """Test detection of codec points format."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "frame_idx,track,node,x,y,score\n0,animal1,head,100,200,0.9\n"
        )
        assert csv.detect_csv_format(csv_path) == "points"

    def test_detect_frames_format(self, tmp_path):
        """Test detection of codec frames format."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "frame_idx,inst0.head.x,inst0.head.y,inst0.tail.x,inst0.tail.y\n"
            "0,100,200,150,250\n"
        )
        assert csv.detect_csv_format(csv_path) == "frames"


class TestIsCSVFile:
    """Tests for is_csv_file function."""

    def test_csv_file_detected(self, tmp_path, simple_labels):
        """Test that valid CSV files are detected."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")
        assert csv.is_csv_file(csv_path)

    def test_non_csv_extension_not_detected(self, tmp_path):
        """Test that non-CSV extensions are not detected."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not a csv")
        assert not csv.is_csv_file(txt_path)


# =============================================================================
# SLEAP Format Tests
# =============================================================================


class TestSleapFormat:
    """Tests for SLEAP Analysis CSV format."""

    def test_write_sleap_format(self, tmp_path, simple_labels):
        """Test writing SLEAP format CSV."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")

        assert csv_path.exists()
        df = pd.read_csv(csv_path)

        assert "track" in df.columns
        assert "frame_idx" in df.columns
        assert "instance.score" in df.columns
        assert "head.x" in df.columns
        assert "tail.y" in df.columns
        assert len(df) == 2  # 2 frames, 1 instance each

    def test_read_sleap_format(self, tmp_path, simple_labels, simple_skeleton):
        """Test reading SLEAP format CSV."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")

        loaded = csv.read_labels(csv_path, format="sleap", skeleton=simple_skeleton)

        assert len(loaded) == 2
        assert len(loaded[0].instances) == 1

    def test_round_trip_sleap_format(self, tmp_path, simple_labels, simple_skeleton):
        """Test SLEAP format round-trip preserves data."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")
        loaded = csv.read_labels(csv_path, format="sleap", skeleton=simple_skeleton)

        # Check coordinates match
        orig_pts = simple_labels[0].instances[0].numpy()
        loaded_pts = loaded[0].instances[0].numpy()
        np.testing.assert_array_almost_equal(orig_pts[:, :2], loaded_pts[:, :2])


# =============================================================================
# DLC Format Tests
# =============================================================================


class TestDLCFormat:
    """Tests for DeepLabCut CSV format."""

    def test_write_dlc_single_animal(self, tmp_path, simple_labels):
        """Test writing single-animal DLC format."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="dlc", scorer="TestScorer")

        assert csv_path.exists()

        # Read with multi-header
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        assert df.columns.names == ["scorer", "bodyparts", "coords"]
        assert "TestScorer" in df.columns.get_level_values(0)

    def test_write_dlc_multi_animal(self, tmp_path, multi_animal_labels):
        """Test writing multi-animal DLC format."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(multi_animal_labels, csv_path, format="dlc")

        # Read with 4-level header
        df = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
        assert df.columns.names == ["scorer", "individuals", "bodyparts", "coords"]
        assert "animal1" in df.columns.get_level_values(1)
        assert "animal2" in df.columns.get_level_values(1)

    def test_read_dlc_format_existing_fixture(self, dlc_testdata):
        """Test reading existing DLC fixture."""
        labels = csv.read_labels(dlc_testdata, format="dlc")

        assert len(labels) > 0
        assert len(labels.skeletons) > 0


# =============================================================================
# Codec Format Passthrough Tests
# =============================================================================


class TestCodecFormats:
    """Tests for codec format passthrough (points, instances, frames)."""

    def test_write_read_points_format(self, tmp_path, simple_labels, simple_skeleton):
        """Test writing and reading points format."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="points")

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "node" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns

    def test_write_read_instances_format(
        self, tmp_path, simple_labels, simple_skeleton
    ):
        """Test writing and reading instances format."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="instances")

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "frame_idx" in df.columns
        assert "head.x" in df.columns

    def test_write_read_frames_format(self, tmp_path, simple_labels, simple_skeleton):
        """Test writing and reading frames format."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="frames")

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "frame_idx" in df.columns
        # Frames format has columns like inst0.head.x
        inst_cols = [c for c in df.columns if c.startswith("inst")]
        assert len(inst_cols) > 0


# =============================================================================
# Metadata Round-Trip Tests
# =============================================================================


class TestMetadataRoundTrip:
    """Tests for metadata JSON round-trip support."""

    def test_write_metadata(self, tmp_path, simple_labels):
        """Test metadata JSON is written when save_metadata=True."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap", save_metadata=True)

        json_path = tmp_path / "test.json"
        assert json_path.exists()

        with open(json_path) as f:
            metadata = json.load(f)

        assert "version" in metadata
        assert "videos" in metadata
        assert "skeletons" in metadata
        assert "tracks" in metadata
        assert "provenance" in metadata

    def test_metadata_contains_skeleton_edges(self, tmp_path, simple_labels):
        """Test metadata includes skeleton edges."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap", save_metadata=True)

        with open(tmp_path / "test.json") as f:
            metadata = json.load(f)

        assert len(metadata["skeletons"]) > 0
        skel = metadata["skeletons"][0]
        assert "edges" in skel
        assert ["head", "tail"] in skel["edges"]

    def test_round_trip_with_metadata(self, tmp_path, simple_labels, simple_skeleton):
        """Test full round-trip with metadata preserves skeleton edges."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap", save_metadata=True)

        # Load without providing skeleton - should come from metadata
        loaded = csv.read_labels(csv_path, format="sleap")

        assert len(loaded.skeletons) > 0
        assert len(loaded.skeletons[0].edges) > 0

    def test_metadata_not_written_by_default(self, tmp_path, simple_labels):
        """Test metadata JSON is NOT written when save_metadata=False (default)."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")

        json_path = tmp_path / "test.json"
        assert not json_path.exists()


# =============================================================================
# Package API Tests
# =============================================================================


class TestPackageAPI:
    """Tests for package-level API."""

    def test_sio_load_csv(self, tmp_path, simple_labels, simple_skeleton):
        """Test sio.load_csv() works."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")

        loaded = sio.load_csv(str(csv_path), skeleton=simple_skeleton)
        assert len(loaded) == 2

    def test_sio_save_csv(self, tmp_path, simple_labels):
        """Test sio.save_csv() works."""
        csv_path = tmp_path / "test.csv"
        sio.save_csv(simple_labels, str(csv_path))

        assert csv_path.exists()

    def test_labels_save_csv(self, tmp_path, simple_labels):
        """Test Labels.save() works with .csv extension."""
        csv_path = tmp_path / "test.csv"
        simple_labels.save(str(csv_path))

        assert csv_path.exists()

    def test_load_file_csv(self, tmp_path, simple_labels, simple_skeleton):
        """Test sio.load_file() auto-detects CSV."""
        csv_path = tmp_path / "test.csv"
        sio.save_csv(simple_labels, str(csv_path))

        loaded = sio.load_file(str(csv_path), skeleton=simple_skeleton)
        assert len(loaded) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_format_read(self, tmp_path, simple_labels):
        """Test that invalid format raises error on read."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")

        with pytest.raises(ValueError, match="Unknown CSV format"):
            csv.read_labels(csv_path, format="invalid_format")

    def test_invalid_format_write(self, tmp_path, simple_labels):
        """Test that invalid format raises error on write."""
        csv_path = tmp_path / "test.csv"

        with pytest.raises(ValueError, match="Unknown CSV format"):
            csv.write_labels(simple_labels, csv_path, format="invalid_format")

    def test_dlc_write_requires_skeleton(self, tmp_path):
        """Test that DLC write requires skeleton."""
        # Labels without skeleton
        video = Video(str(tmp_path / "video.mp4"))
        labels = Labels(labeled_frames=[], videos=[video])

        csv_path = tmp_path / "test.csv"

        with pytest.raises(ValueError, match="Cannot export DLC format without"):
            csv.write_labels(labels, csv_path, format="dlc")


# =============================================================================
# Auto-Detection Tests
# =============================================================================


class TestAutoDetection:
    """Tests for format auto-detection."""

    def test_auto_detect_sleap(self, tmp_path, simple_labels, simple_skeleton):
        """Test auto-detection of SLEAP format."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")

        # Load with auto-detection
        loaded = csv.read_labels(csv_path, format="auto", skeleton=simple_skeleton)
        assert len(loaded) == 2

    def test_auto_detect_instances(self, tmp_path, simple_labels, simple_skeleton):
        """Test auto-detection of instances format."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="instances")

        detected = csv.detect_csv_format(csv_path)
        assert detected == "instances"


# =============================================================================
# Node Inference Tests
# =============================================================================


class TestNodeInference:
    """Tests for node name inference from columns."""

    def test_infer_nodes_instances_format(self):
        """Test inferring node names from instances format columns."""
        columns = ["frame_idx", "track", "head.x", "head.y", "tail.x", "tail.y"]
        nodes = csv._infer_nodes_from_columns(columns, format="instances")
        assert set(nodes) == {"head", "tail"}

    def test_infer_nodes_frames_format(self):
        """Test inferring node names from frames format columns."""
        columns = [
            "frame_idx",
            "inst0.head.x",
            "inst0.head.y",
            "inst0.tail.x",
            "inst0.tail.y",
        ]
        nodes = csv._infer_nodes_from_columns(columns, format="frames")
        assert set(nodes) == {"head", "tail"}

    def test_infer_nodes_points_format(self):
        """Test that points format returns empty list (nodes in column)."""
        columns = ["frame_idx", "node", "x", "y"]
        nodes = csv._infer_nodes_from_columns(columns, format="points")
        assert nodes == []
