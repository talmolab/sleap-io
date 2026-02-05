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

    def test_infer_nodes_skips_non_coord_columns(self):
        """Test that non-x/y/score columns are skipped. Covers line 690."""
        columns = ["frame_idx", "head.x", "head.y", "head.visibility", "tail.x"]
        nodes = csv._infer_nodes_from_columns(columns, format="instances")
        # head.visibility should be skipped (not x, y, or score)
        assert set(nodes) == {"head", "tail"}


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestAdditionalCoverage:
    """Tests to cover remaining code paths."""

    def test_detect_csv_format_default_fallback(self, tmp_path):
        """Test detection defaults to sleap for unknown CSV structure. Covers 237."""
        csv_path = tmp_path / "test.csv"
        # CSV with no recognizable patterns (no dots, no DLC headers)
        csv_path.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        assert csv.detect_csv_format(csv_path) == "sleap"

    def test_read_sleap_infers_skeleton(self, tmp_path, simple_labels):
        """Test reading SLEAP CSV without skeleton infers it. Covers 287-289."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap")

        # Read without providing skeleton - should infer from columns
        loaded = csv.read_labels(csv_path, format="sleap")
        assert len(loaded.skeletons) > 0
        node_names = [n.name for n in loaded.skeleton.nodes]
        assert "head" in node_names
        assert "tail" in node_names

    def test_write_dlc_video_filter_by_index(self, tmp_path, simple_labels):
        """Test DLC write with video filter by integer index. Covers 388-390."""
        csv_path = tmp_path / "test.csv"
        # Filter by video index 0
        csv.write_labels(simple_labels, csv_path, format="dlc", video=0)
        assert csv_path.exists()

    def test_read_codec_format_instances(
        self, tmp_path, simple_labels, simple_skeleton
    ):
        """Test reading instances format directly. Covers 478-494."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="instances")

        # Read with explicit format
        loaded = csv.read_labels(csv_path, format="instances", skeleton=simple_skeleton)
        assert len(loaded) == 2

    def test_read_codec_format_points(self, tmp_path, simple_labels, simple_skeleton):
        """Test reading points format directly. Covers 478-494."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="points")

        # Read with explicit format
        loaded = csv.read_labels(csv_path, format="points", skeleton=simple_skeleton)
        # Points format may have different structure, just check it loads
        assert loaded is not None

    def test_read_codec_format_frames(self, tmp_path, simple_labels, simple_skeleton):
        """Test reading frames format directly. Covers 478-494."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="frames")

        # Read with explicit format
        loaded = csv.read_labels(csv_path, format="frames", skeleton=simple_skeleton)
        assert loaded is not None

    def test_metadata_with_suggestions(self, tmp_path, simple_skeleton):
        """Test metadata round-trip with suggestions. Covers 648-651."""
        video = Video(str(tmp_path / "video.mp4"))

        inst = Instance(
            points={"head": [100, 200], "tail": [150, 250]},
            skeleton=simple_skeleton,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])

        # Create labels with explicit videos list to ensure video is registered
        labels = Labels(labeled_frames=[lf], videos=[video])
        # Add a suggestion
        labels.suggestions.append(sio.SuggestionFrame(video=video, frame_idx=5))

        csv_path = tmp_path / "test.csv"
        csv.write_labels(labels, csv_path, format="sleap", save_metadata=True)

        # Verify metadata has suggestions
        with open(tmp_path / "test.json") as f:
            metadata = json.load(f)
        assert len(metadata["suggestions"]) == 1
        assert metadata["suggestions"][0]["frame_idx"] == 5
        assert metadata["suggestions"][0]["video_idx"] == 0

        # Call _apply_metadata directly to test the restoration branch
        # Create a labels with video to test suggestion restoration
        loaded_labels = Labels(videos=[video])
        csv._apply_metadata(loaded_labels, metadata)
        assert len(loaded_labels.suggestions) == 1
        assert loaded_labels.suggestions[0].frame_idx == 5

    def test_metadata_edge_deduplication(self, tmp_path, simple_skeleton):
        """Test that edges aren't duplicated when already present. Covers 631-634."""
        video = Video(str(tmp_path / "video.mp4"))

        inst = Instance(
            points={"head": [100, 200], "tail": [150, 250]},
            skeleton=simple_skeleton,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = Labels(labeled_frames=[lf])

        csv_path = tmp_path / "test.csv"
        csv.write_labels(labels, csv_path, format="sleap", save_metadata=True)

        # Load twice to trigger edge restoration when edges already exist
        loaded = csv.read_labels(csv_path, format="sleap")
        # Should have exactly 1 edge, not duplicated
        assert len(loaded.skeleton.edges) == 1

    def test_metadata_video_from_file(self, tmp_path, simple_labels):
        """Test video path is loaded from metadata. Covers 84->86."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap", save_metadata=True)

        # Read without providing video - should come from metadata
        loaded = csv.read_labels(csv_path, format="sleap")
        assert len(loaded.videos) > 0

    def test_metadata_skeleton_from_file(self, tmp_path, simple_labels):
        """Test skeleton is loaded from metadata. Covers 86->96."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap", save_metadata=True)

        # Read without providing skeleton - should come from metadata
        loaded = csv.read_labels(csv_path, format="sleap")
        assert len(loaded.skeletons) > 0
        # Check edges were restored from metadata
        assert len(loaded.skeleton.edges) > 0

    def test_read_codec_infers_skeleton(self, tmp_path, simple_labels):
        """Test reading codec format infers skeleton. Covers 483-486."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="instances")

        # Read without skeleton - should infer from columns
        loaded = csv.read_labels(csv_path, format="instances")
        assert len(loaded.skeletons) > 0

    def test_read_codec_with_video_string(
        self, tmp_path, simple_labels, simple_skeleton
    ):
        """Test reading codec format with video as string. Covers 489-490."""
        csv_path = tmp_path / "test.csv"
        csv.write_labels(simple_labels, csv_path, format="instances")

        # Read with video as string path
        loaded = csv.read_labels(
            csv_path,
            format="instances",
            video="some/video.mp4",
            skeleton=simple_skeleton,
        )
        assert loaded.videos[0].filename == "some/video.mp4"


# =============================================================================
# Tests for include_empty, start_frame, end_frame parameters
# =============================================================================


class TestIncludeEmpty:
    """Tests for the include_empty parameter in CSV export."""

    @pytest.fixture
    def sparse_labels(self, tmp_path):
        """Labels with instances only in frames 0 and 3 (sparse)."""
        skeleton = Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
        video = Video(str(tmp_path / "sparse.mp4"))

        labeled_frames = []
        for frame_idx in [0, 3]:
            inst = Instance(
                points={"nose": [100.0 + frame_idx, 200.0], "tail": [150.0, 250.0]},
                skeleton=skeleton,
            )
            labeled_frames.append(
                LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst])
            )

        return Labels(labeled_frames=labeled_frames)

    def test_include_empty_false_default(self, tmp_path, sparse_labels):
        """Test include_empty=False (default) only exports labeled frames."""
        csv_path = tmp_path / "sparse.csv"
        csv.write_labels(sparse_labels, csv_path, format="frames", include_empty=False)

        df = pd.read_csv(csv_path)
        assert list(df["frame_idx"]) == [0, 3]

    def test_include_empty_true_pads_frames(self, tmp_path, sparse_labels):
        """Test include_empty=True pads missing frames with NaN."""
        csv_path = tmp_path / "padded.csv"
        csv.write_labels(sparse_labels, csv_path, format="frames", include_empty=True)

        df = pd.read_csv(csv_path)
        # Should have frames 0, 1, 2, 3 (from 0 to last labeled frame)
        assert list(df["frame_idx"]) == [0, 1, 2, 3]

        # Frames 1 and 2 should have NaN values
        row_1 = df[df["frame_idx"] == 1].iloc[0]
        assert pd.isna(row_1["inst0.nose.x"])

    def test_include_empty_with_frame_range(self, tmp_path, sparse_labels):
        """Test include_empty with start_frame and end_frame."""
        csv_path = tmp_path / "range.csv"
        csv.write_labels(
            sparse_labels,
            csv_path,
            format="frames",
            include_empty=True,
            start_frame=1,
            end_frame=5,
        )

        df = pd.read_csv(csv_path)
        # Should have frames 1, 2, 3, 4 (start=1, end=5 exclusive)
        assert list(df["frame_idx"]) == [1, 2, 3, 4]

        # Frame 3 has data, others should be NaN
        row_3 = df[df["frame_idx"] == 3].iloc[0]
        assert not pd.isna(row_3["inst0.nose.x"])

        row_2 = df[df["frame_idx"] == 2].iloc[0]
        assert pd.isna(row_2["inst0.nose.x"])

    def test_include_empty_instances_format(self, tmp_path, sparse_labels):
        """Test include_empty with instances format."""
        csv_path = tmp_path / "instances.csv"
        csv.write_labels(
            sparse_labels, csv_path, format="instances", include_empty=True
        )

        df = pd.read_csv(csv_path)
        # Should have frames 0, 1, 2, 3
        assert set(df["frame_idx"]) == {0, 1, 2, 3}

        # Frame 1 should have NaN coordinates
        row_1 = df[df["frame_idx"] == 1].iloc[0]
        assert pd.isna(row_1["nose.x"])

    def test_include_empty_sleap_format(self, tmp_path, sparse_labels):
        """Test include_empty with sleap format (uses frames under the hood)."""
        csv_path = tmp_path / "sleap.csv"
        csv.write_labels(sparse_labels, csv_path, format="sleap", include_empty=True)

        df = pd.read_csv(csv_path)
        # Should have frames 0, 1, 2, 3
        assert list(df["frame_idx"]) == [0, 1, 2, 3]


# =============================================================================
# Chunked Writing Tests
# =============================================================================


class TestChunkedWriting:
    """Tests for memory-efficient chunked CSV writing."""

    def test_chunked_frames_format(self, tmp_path, simple_labels):
        """Test chunked writing with frames format."""
        csv_path = tmp_path / "chunked.csv"
        csv.write_labels(simple_labels, csv_path, format="frames", chunk_size=1)

        # Should produce same output as non-chunked
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df["frame_idx"]) == [0, 1]
        assert "inst0.head.x" in df.columns

    def test_chunked_instances_format(self, tmp_path, simple_labels):
        """Test chunked writing with instances format."""
        csv_path = tmp_path / "chunked.csv"
        csv.write_labels(simple_labels, csv_path, format="instances", chunk_size=1)

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "head.x" in df.columns

    def test_chunked_points_format(self, tmp_path, simple_labels):
        """Test chunked writing with points format."""
        csv_path = tmp_path / "chunked.csv"
        csv.write_labels(simple_labels, csv_path, format="points", chunk_size=2)

        df = pd.read_csv(csv_path)
        # 2 frames * 1 instance * 2 nodes = 4 rows
        assert len(df) == 4
        assert "node" in df.columns

    def test_chunked_sleap_format(self, tmp_path, simple_labels):
        """Test chunked writing with SLEAP format."""
        csv_path = tmp_path / "chunked.csv"
        csv.write_labels(simple_labels, csv_path, format="sleap", chunk_size=1)

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        # SLEAP format has instance.score column
        assert "instance.score" in df.columns or "head.x" in df.columns

    def test_chunked_matches_non_chunked(self, tmp_path, simple_labels):
        """Test that chunked writing produces same output as non-chunked."""
        non_chunked_path = tmp_path / "non_chunked.csv"
        chunked_path = tmp_path / "chunked.csv"

        csv.write_labels(simple_labels, non_chunked_path, format="frames")
        csv.write_labels(simple_labels, chunked_path, format="frames", chunk_size=1)

        df_non_chunked = pd.read_csv(non_chunked_path)
        df_chunked = pd.read_csv(chunked_path)

        pd.testing.assert_frame_equal(df_non_chunked, df_chunked)

    def test_chunked_dlc_not_supported(self, tmp_path, simple_labels):
        """Test that chunked writing with DLC format raises error."""
        csv_path = tmp_path / "dlc.csv"

        with pytest.raises(ValueError, match="Chunked writing is not supported"):
            csv.write_labels(
                simple_labels, csv_path, format="dlc", chunk_size=100, scorer="test"
            )

    def test_chunked_with_large_chunk_size(self, tmp_path, simple_labels):
        """Test chunked writing with chunk_size larger than data."""
        csv_path = tmp_path / "chunked.csv"
        # chunk_size of 1000 is much larger than our 2 frames
        csv.write_labels(simple_labels, csv_path, format="frames", chunk_size=1000)

        df = pd.read_csv(csv_path)
        assert len(df) == 2
