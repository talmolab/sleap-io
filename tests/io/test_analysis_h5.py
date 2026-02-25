"""Tests for SLEAP Analysis HDF5 I/O operations."""

import json

import h5py
import numpy as np
import pytest

import sleap_io as sio
from sleap_io.io import analysis_h5
from sleap_io.model.instance import Instance, PredictedInstance
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
        symmetries=[],
        name="simple",
    )


@pytest.fixture
def complex_skeleton():
    """Create a complex skeleton with symmetries."""
    return Skeleton(
        nodes=["nose", "left_ear", "right_ear", "tail"],
        edges=[("nose", "left_ear"), ("nose", "right_ear"), ("nose", "tail")],
        symmetries=[("left_ear", "right_ear")],
        name="mouse",
    )


@pytest.fixture
def simple_labels(simple_skeleton, tmp_path):
    """Create simple Labels with 3 frames, 1 tracked instance each."""
    video = Video(str(tmp_path / "video.mp4"))
    track = sio.Track("animal1")

    frames = []
    for i in range(3):
        inst = PredictedInstance.from_numpy(
            points_data=np.array([[100.0 + i, 200.0 + i], [150.0 + i, 250.0 + i]]),
            skeleton=simple_skeleton,
            point_scores=np.array([0.9, 0.85]),
            score=0.95,
            track=track,
            tracking_score=0.88,
        )
        lf = LabeledFrame(video=video, frame_idx=i, instances=[inst])
        frames.append(lf)

    return Labels(labeled_frames=frames, tracks=[track])


@pytest.fixture
def multi_animal_labels(simple_skeleton, tmp_path):
    """Create Labels with multiple tracked animals."""
    video = Video(str(tmp_path / "video.mp4"))
    track1 = sio.Track("animal1")
    track2 = sio.Track("animal2")

    frames = []
    for i in range(3):
        inst1 = PredictedInstance.from_numpy(
            points_data=np.array([[100.0 + i, 200.0 + i], [150.0 + i, 250.0 + i]]),
            skeleton=simple_skeleton,
            point_scores=np.array([0.9, 0.85]),
            score=0.95,
            track=track1,
        )
        inst2 = PredictedInstance.from_numpy(
            points_data=np.array([[300.0 + i, 400.0 + i], [350.0 + i, 450.0 + i]]),
            skeleton=simple_skeleton,
            point_scores=np.array([0.88, 0.82]),
            score=0.92,
            track=track2,
        )
        lf = LabeledFrame(video=video, frame_idx=i, instances=[inst1, inst2])
        frames.append(lf)

    return Labels(labeled_frames=frames, tracks=[track1, track2])


@pytest.fixture
def sparse_labels(simple_skeleton, tmp_path):
    """Create Labels with sparse occupancy (some tracks missing in some frames)."""
    video = Video(str(tmp_path / "video.mp4"))
    track1 = sio.Track("animal1")
    track2 = sio.Track("animal2")
    track3 = sio.Track("spurious")  # Low occupancy track

    frames = []

    # Frame 0: all three tracks
    insts = [
        PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]),
            simple_skeleton,
            np.array([0.9, 0.85]),
            0.95,
            track=track1,
        ),
        PredictedInstance.from_numpy(
            np.array([[300.0, 400.0], [350.0, 450.0]]),
            simple_skeleton,
            np.array([0.88, 0.82]),
            0.92,
            track=track2,
        ),
        PredictedInstance.from_numpy(
            np.array([[500.0, 600.0], [550.0, 650.0]]),
            simple_skeleton,
            np.array([0.5, 0.5]),
            0.5,
            track=track3,
        ),
    ]
    frames.append(LabeledFrame(video=video, frame_idx=0, instances=insts))

    # Frames 1-9: only track1 and track2
    for i in range(1, 10):
        insts = [
            PredictedInstance.from_numpy(
                np.array([[100.0 + i, 200.0 + i], [150.0 + i, 250.0 + i]]),
                simple_skeleton,
                np.array([0.9, 0.85]),
                0.95,
                track=track1,
            ),
            PredictedInstance.from_numpy(
                np.array([[300.0 + i, 400.0 + i], [350.0 + i, 450.0 + i]]),
                simple_skeleton,
                np.array([0.88, 0.82]),
                0.92,
                track=track2,
            ),
        ]
        frames.append(LabeledFrame(video=video, frame_idx=i, instances=insts))

    return Labels(labeled_frames=frames, tracks=[track1, track2, track3])


# =============================================================================
# Format Detection Tests
# =============================================================================


class TestFormatDetection:
    """Tests for Analysis HDF5 format detection."""

    def test_is_analysis_h5_file_valid(self, tmp_path):
        """Test detection of valid Analysis HDF5 file."""
        h5_path = tmp_path / "test.analysis.h5"

        # Create a minimal Analysis HDF5 file
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("track_occupancy", data=np.zeros((10, 2)))
            f.create_dataset("tracks", data=np.zeros((2, 2, 3, 10)))

        assert analysis_h5.is_analysis_h5_file(h5_path)

    def test_is_analysis_h5_file_jabs(self, tmp_path):
        """Test that JABS HDF5 files are not detected as Analysis."""
        h5_path = tmp_path / "test.h5"

        # Create a minimal JABS HDF5 file
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("poseest")
            grp.create_dataset("points", data=np.zeros((10, 12, 2)))

        assert not analysis_h5.is_analysis_h5_file(h5_path)

    def test_is_analysis_h5_file_invalid(self, tmp_path):
        """Test detection returns False for non-HDF5 files."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not an hdf5 file")

        assert not analysis_h5.is_analysis_h5_file(txt_path)

    def test_is_analysis_h5_file_nonexistent(self, tmp_path):
        """Test detection returns False for non-existent files."""
        assert not analysis_h5.is_analysis_h5_file(tmp_path / "nonexistent.h5")


# =============================================================================
# Write Tests
# =============================================================================


class TestWriteLabels:
    """Tests for writing Labels to Analysis HDF5."""

    def test_write_basic(self, simple_labels, tmp_path):
        """Test basic write with default options (matlab preset)."""
        h5_path = tmp_path / "output.analysis.h5"

        analysis_h5.write_labels(simple_labels, h5_path)

        assert h5_path.exists()

        # Verify file structure
        with h5py.File(h5_path, "r") as f:
            assert "tracks" in f
            assert "track_occupancy" in f
            assert "point_scores" in f
            assert "instance_scores" in f
            assert "tracking_scores" in f
            assert "track_names" in f
            assert "node_names" in f
            assert "video_path" in f
            assert f.attrs["preset"] == "matlab"

    def test_write_multi_animal(self, multi_animal_labels, tmp_path):
        """Test write with multiple tracked animals."""
        h5_path = tmp_path / "multi.analysis.h5"

        analysis_h5.write_labels(multi_animal_labels, h5_path)

        with h5py.File(h5_path, "r") as f:
            track_names = [n.decode() for n in f["track_names"][:]]
            assert len(track_names) == 2
            assert "animal1" in track_names
            assert "animal2" in track_names

            # Check stored shape (matlab preset: track, xy, node, frame)
            assert f["tracks"].shape == (2, 2, 2, 3)

    def test_write_with_min_occupancy(self, sparse_labels, tmp_path):
        """Test track filtering with min_occupancy parameter."""
        h5_path = tmp_path / "filtered.analysis.h5"

        # Filter tracks with <50% occupancy
        analysis_h5.write_labels(sparse_labels, h5_path, min_occupancy=0.5)

        with h5py.File(h5_path, "r") as f:
            track_names = [n.decode() for n in f["track_names"][:]]
            assert len(track_names) == 2  # Only animal1 and animal2
            assert "spurious" not in track_names

    def test_write_standard_preset(self, simple_labels, tmp_path):
        """Test write with standard preset (Python-native ordering)."""
        h5_path = tmp_path / "standard.analysis.h5"

        analysis_h5.write_labels(simple_labels, h5_path, preset="standard")

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["preset"] == "standard"
            # Standard: (frame, track, node, xy)
            assert f["tracks"].shape == (3, 1, 2, 2)
            dims = json.loads(f["tracks"].attrs["dims"])
            assert dims == ["frame", "track", "node", "xy"]

    def test_write_matlab_preset(self, simple_labels, tmp_path):
        """Test write with explicit matlab preset."""
        h5_path = tmp_path / "matlab.analysis.h5"

        analysis_h5.write_labels(simple_labels, h5_path, preset="matlab")

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["preset"] == "matlab"
            # MATLAB: (track, xy, node, frame)
            assert f["tracks"].shape == (1, 2, 2, 3)
            dims = json.loads(f["tracks"].attrs["dims"])
            assert dims == ["track", "xy", "node", "frame"]

    def test_write_custom_dims(self, simple_labels, tmp_path):
        """Test write with custom explicit dimension positions."""
        h5_path = tmp_path / "custom.analysis.h5"

        # Custom ordering: (node, frame, track, xy)
        analysis_h5.write_labels(
            simple_labels,
            h5_path,
            frame_dim=1,
            track_dim=2,
            node_dim=0,
            xy_dim=3,
        )

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["preset"] == "custom"
            # Custom: (node, frame, track, xy)
            assert f["tracks"].shape == (2, 3, 1, 2)
            dims = json.loads(f["tracks"].attrs["dims"])
            assert dims == ["node", "frame", "track", "xy"]

    def test_write_metadata(self, simple_labels, tmp_path):
        """Test that extended metadata is written."""
        h5_path = tmp_path / "metadata.analysis.h5"

        analysis_h5.write_labels(simple_labels, h5_path, save_metadata=True)

        with h5py.File(h5_path, "r") as f:
            assert f.attrs.get("format") == "analysis"
            assert f.attrs.get("sleap_io_version") == "1.0"
            assert "skeleton_name" in f.attrs
            assert "skeleton_symmetries" in f.attrs
            assert "video_backend_metadata" in f.attrs

    def test_write_no_metadata(self, simple_labels, tmp_path):
        """Test write without extended metadata."""
        h5_path = tmp_path / "no_metadata.analysis.h5"

        analysis_h5.write_labels(simple_labels, h5_path, save_metadata=False)

        with h5py.File(h5_path, "r") as f:
            # format and preset are always written
            assert f.attrs.get("format") == "analysis"
            assert f.attrs.get("preset") == "matlab"
            # But skeleton_name is only in extended metadata
            assert "skeleton_name" not in f.attrs

    def test_write_dimension_attributes_matlab(self, simple_labels, tmp_path):
        """Test that dimension names are stored correctly for matlab preset."""
        h5_path = tmp_path / "dims_matlab.analysis.h5"

        analysis_h5.write_labels(simple_labels, h5_path, preset="matlab")

        with h5py.File(h5_path, "r") as f:
            # Check dimension attributes (matlab order: track first)
            assert json.loads(f["tracks"].attrs["dims"]) == [
                "track",
                "xy",
                "node",
                "frame",
            ]
            # For 2D arrays, track comes first EXCEPT for track_occupancy.
            # track_occupancy is stored as (frame, track) to match SLEAP's original
            # behavior - this is a quirk in the original implementation that we
            # preserve for MATLAB compatibility.
            assert json.loads(f["track_occupancy"].attrs["dims"]) == ["frame", "track"]
            assert json.loads(f["point_scores"].attrs["dims"]) == [
                "track",
                "node",
                "frame",
            ]
            assert json.loads(f["instance_scores"].attrs["dims"]) == ["track", "frame"]

    def test_write_matlab_shapes_match_sleap(self, multi_animal_labels, tmp_path):
        """Test that matlab preset produces shapes exactly matching SLEAP reference.

        The SLEAP Analysis HDF5 format stores arrays in specific shapes for MATLAB
        column-major compatibility. This test verifies our output matches SLEAP's
        write_tracking_h5.py output exactly.

        Reference shapes (from SLEAP's write_tracking_h5.py after transpose):
        - tracks:           (tracks, 2, nodes, frames)
        - track_occupancy:  (frames, tracks)  <- note: different from other 2D!
        - point_scores:     (tracks, nodes, frames)
        - instance_scores:  (tracks, frames)
        - tracking_scores:  (tracks, frames)
        """
        h5_path = tmp_path / "sleap_compat.analysis.h5"
        analysis_h5.write_labels(multi_animal_labels, h5_path, preset="matlab")

        # Expected dimensions: 3 frames, 2 tracks, 2 nodes
        n_frames, n_tracks, n_nodes = 3, 2, 2

        with h5py.File(h5_path, "r") as f:
            # Verify shapes match SLEAP reference exactly
            assert f["tracks"].shape == (n_tracks, 2, n_nodes, n_frames)
            assert f["track_occupancy"].shape == (n_frames, n_tracks)  # SLEAP quirk!
            assert f["point_scores"].shape == (n_tracks, n_nodes, n_frames)
            assert f["instance_scores"].shape == (n_tracks, n_frames)
            assert f["tracking_scores"].shape == (n_tracks, n_frames)

    def test_write_dimension_attributes_standard(self, simple_labels, tmp_path):
        """Test dimension names for standard preset."""
        h5_path = tmp_path / "dims_standard.analysis.h5"

        analysis_h5.write_labels(simple_labels, h5_path, preset="standard")

        with h5py.File(h5_path, "r") as f:
            # Check dimension attributes (standard order)
            assert json.loads(f["tracks"].attrs["dims"]) == [
                "frame",
                "track",
                "node",
                "xy",
            ]
            assert json.loads(f["track_occupancy"].attrs["dims"]) == ["frame", "track"]
            assert json.loads(f["point_scores"].attrs["dims"]) == [
                "frame",
                "track",
                "node",
            ]
            assert json.loads(f["instance_scores"].attrs["dims"]) == ["frame", "track"]

    def test_write_labels_path(self, simple_labels, tmp_path):
        """Test that labels_path is stored correctly."""
        h5_path = tmp_path / "with_path.analysis.h5"
        source_path = "/path/to/source.slp"

        analysis_h5.write_labels(simple_labels, h5_path, labels_path=source_path)

        with h5py.File(h5_path, "r") as f:
            assert f["labels_path"][()].decode() == source_path

    def test_write_with_symmetries(self, complex_skeleton, tmp_path):
        """Test write with skeleton symmetries."""
        video = Video(str(tmp_path / "video.mp4"))
        track = sio.Track("animal1")

        inst = PredictedInstance.from_numpy(
            np.array([[10, 20], [30, 40], [50, 60], [70, 80]]),
            complex_skeleton,
            np.array([0.9, 0.85, 0.88, 0.92]),
            0.95,
            track=track,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = Labels(labeled_frames=[lf], tracks=[track])

        h5_path = tmp_path / "symmetries.analysis.h5"
        analysis_h5.write_labels(labels, h5_path)

        with h5py.File(h5_path, "r") as f:
            symmetries = json.loads(f.attrs["skeleton_symmetries"])
            assert len(symmetries) == 1
            assert set(symmetries[0]) == {"left_ear", "right_ear"}

    def test_write_preset_and_dims_mutually_exclusive(self, simple_labels, tmp_path):
        """Test that preset and explicit dims cannot be used together."""
        h5_path = tmp_path / "error.analysis.h5"

        with pytest.raises(ValueError, match="Cannot specify both"):
            analysis_h5.write_labels(
                simple_labels,
                h5_path,
                preset="matlab",
                frame_dim=0,
                track_dim=1,
                node_dim=2,
                xy_dim=3,
            )

    def test_write_incomplete_explicit_dims(self, simple_labels, tmp_path):
        """Test that incomplete explicit dims raises error."""
        h5_path = tmp_path / "error.analysis.h5"

        with pytest.raises(ValueError, match="all four must be specified"):
            analysis_h5.write_labels(
                simple_labels,
                h5_path,
                frame_dim=0,
                track_dim=1,
                # Missing node_dim and xy_dim
            )

    def test_write_invalid_explicit_dims(self, simple_labels, tmp_path):
        """Test that invalid explicit dims raises error."""
        h5_path = tmp_path / "error.analysis.h5"

        with pytest.raises(ValueError, match="permutation of"):
            analysis_h5.write_labels(
                simple_labels,
                h5_path,
                frame_dim=0,
                track_dim=0,  # Duplicate!
                node_dim=2,
                xy_dim=3,
            )


# =============================================================================
# Read Tests
# =============================================================================


class TestReadLabels:
    """Tests for reading Analysis HDF5 files."""

    def test_read_basic(self, simple_labels, tmp_path):
        """Test basic read from written file."""
        h5_path = tmp_path / "test.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        assert len(loaded) == 3
        assert len(loaded.tracks) == 1
        assert loaded.tracks[0].name == "animal1"

    def test_read_multi_animal(self, multi_animal_labels, tmp_path):
        """Test read with multiple animals."""
        h5_path = tmp_path / "multi.analysis.h5"
        analysis_h5.write_labels(multi_animal_labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        assert len(loaded) == 3
        assert len(loaded.tracks) == 2
        track_names = [t.name for t in loaded.tracks]
        assert "animal1" in track_names
        assert "animal2" in track_names

    def test_read_skeleton(self, simple_labels, tmp_path):
        """Test skeleton is reconstructed correctly."""
        h5_path = tmp_path / "skel.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        assert len(loaded.skeletons) == 1
        skel = loaded.skeleton
        assert len(skel.nodes) == 2
        assert skel.node_names == ["head", "tail"]
        assert len(skel.edges) == 1
        assert skel.name == "simple"

    def test_read_skeleton_symmetries(self, complex_skeleton, tmp_path):
        """Test skeleton symmetries are reconstructed."""
        video = Video(str(tmp_path / "video.mp4"))
        track = sio.Track("animal1")

        inst = PredictedInstance.from_numpy(
            np.array([[10, 20], [30, 40], [50, 60], [70, 80]]),
            complex_skeleton,
            np.array([0.9, 0.85, 0.88, 0.92]),
            0.95,
            track=track,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = Labels(labeled_frames=[lf], tracks=[track])

        h5_path = tmp_path / "symmetries.analysis.h5"
        analysis_h5.write_labels(labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        assert len(loaded.skeleton.symmetries) == 1

    def test_read_with_custom_video(self, simple_labels, tmp_path):
        """Test reading with custom video path."""
        h5_path = tmp_path / "test.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path)

        custom_video = str(tmp_path / "custom_video.mp4")
        loaded = analysis_h5.read_labels(h5_path, video=custom_video)

        assert loaded.video.filename == custom_video

    def test_read_with_video_object(self, simple_labels, tmp_path):
        """Test reading with Video object."""
        h5_path = tmp_path / "test.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path)

        custom_video = Video(str(tmp_path / "custom_video.mp4"))
        loaded = analysis_h5.read_labels(h5_path, video=custom_video)

        assert loaded.video.filename == custom_video.filename

    def test_read_point_scores(self, simple_labels, tmp_path):
        """Test point scores are loaded correctly."""
        h5_path = tmp_path / "scores.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        inst = loaded[0].instances[0]
        assert isinstance(inst, PredictedInstance)
        assert inst.score is not None
        assert inst.points["score"][0] is not None

    def test_read_tracking_scores(self, simple_labels, tmp_path):
        """Test tracking scores are loaded correctly."""
        h5_path = tmp_path / "tracking.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        inst = loaded[0].instances[0]
        assert isinstance(inst, PredictedInstance)
        assert inst.tracking_score is not None


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for round-trip consistency."""

    def test_round_trip_points_matlab(self, simple_labels, tmp_path):
        """Test that point coordinates survive round-trip with matlab preset."""
        h5_path = tmp_path / "roundtrip_matlab.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path, preset="matlab")

        loaded = analysis_h5.read_labels(h5_path)

        # Compare first instance points
        orig_pts = simple_labels[0].instances[0].numpy()
        loaded_pts = loaded[0].instances[0].numpy()

        np.testing.assert_allclose(orig_pts, loaded_pts)

    def test_round_trip_points_standard(self, simple_labels, tmp_path):
        """Test that point coordinates survive round-trip with standard preset."""
        h5_path = tmp_path / "roundtrip_standard.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path, preset="standard")

        loaded = analysis_h5.read_labels(h5_path)

        # Compare first instance points
        orig_pts = simple_labels[0].instances[0].numpy()
        loaded_pts = loaded[0].instances[0].numpy()

        np.testing.assert_allclose(orig_pts, loaded_pts)

        # Verify all frames are loaded correctly
        assert len(loaded) == len(simple_labels)

    def test_round_trip_custom_dims(self, simple_labels, tmp_path):
        """Test round-trip with custom dimension ordering."""
        h5_path = tmp_path / "roundtrip_custom.analysis.h5"
        analysis_h5.write_labels(
            simple_labels,
            h5_path,
            frame_dim=2,
            track_dim=0,
            node_dim=3,
            xy_dim=1,
        )

        loaded = analysis_h5.read_labels(h5_path)

        # Compare first instance points
        orig_pts = simple_labels[0].instances[0].numpy()
        loaded_pts = loaded[0].instances[0].numpy()

        np.testing.assert_allclose(orig_pts, loaded_pts)

    def test_round_trip_multi_animal(self, multi_animal_labels, tmp_path):
        """Test round-trip with multiple animals."""
        h5_path = tmp_path / "roundtrip_multi.analysis.h5"
        analysis_h5.write_labels(multi_animal_labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        assert len(loaded) == len(multi_animal_labels)
        assert len(loaded.tracks) == len(multi_animal_labels.tracks)

        # Compare points for all instances in first frame
        for i in range(len(multi_animal_labels[0].instances)):
            orig_pts = multi_animal_labels[0].instances[i].numpy()
            loaded_pts = loaded[0].instances[i].numpy()
            np.testing.assert_allclose(orig_pts, loaded_pts)

    def test_round_trip_provenance(self, simple_labels, tmp_path):
        """Test provenance survives round-trip."""
        simple_labels.provenance["source"] = "test"
        simple_labels.provenance["version"] = "1.0"

        h5_path = tmp_path / "provenance.analysis.h5"
        analysis_h5.write_labels(simple_labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)

        assert loaded.provenance.get("source") == "test"
        assert loaded.provenance.get("version") == "1.0"


# =============================================================================
# Integration Tests with Main API
# =============================================================================


class TestMainAPI:
    """Tests for main API integration."""

    def test_sio_save_load(self, simple_labels, tmp_path):
        """Test using sio.save_analysis_h5 and sio.load_analysis_h5."""
        h5_path = tmp_path / "api_test.analysis.h5"

        sio.save_analysis_h5(simple_labels, str(h5_path))
        loaded = sio.load_analysis_h5(str(h5_path))

        assert len(loaded) == len(simple_labels)

    def test_sio_save_with_preset(self, simple_labels, tmp_path):
        """Test sio.save_analysis_h5 with preset parameter."""
        h5_path = tmp_path / "api_preset.analysis.h5"

        sio.save_analysis_h5(simple_labels, str(h5_path), preset="standard")
        loaded = sio.load_analysis_h5(str(h5_path))

        assert len(loaded) == len(simple_labels)

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["preset"] == "standard"

    def test_load_file_detection(self, simple_labels, tmp_path):
        """Test that load_file correctly detects Analysis HDF5."""
        h5_path = tmp_path / "detect.h5"
        sio.save_analysis_h5(simple_labels, str(h5_path))

        # load_file should auto-detect Analysis HDF5
        loaded = sio.load_file(str(h5_path))

        assert len(loaded) == len(simple_labels)

    def test_save_file_h5_extension(self, simple_labels, tmp_path):
        """Test save_file with .h5 extension uses analysis_h5 by default."""
        h5_path = tmp_path / "savefile.h5"

        sio.save_file(simple_labels, str(h5_path))

        # Verify it was saved as Analysis HDF5
        assert analysis_h5.is_analysis_h5_file(h5_path)

    def test_save_file_explicit_format(self, simple_labels, tmp_path):
        """Test save_file with explicit format."""
        h5_path = tmp_path / "explicit.h5"

        sio.save_file(simple_labels, str(h5_path), format="analysis_h5")

        assert analysis_h5.is_analysis_h5_file(h5_path)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_labels(self, simple_skeleton, tmp_path):
        """Test handling of empty Labels."""
        video = Video(str(tmp_path / "video.mp4"))
        labels = Labels(videos=[video], skeletons=[simple_skeleton])

        h5_path = tmp_path / "empty.analysis.h5"

        with pytest.raises(ValueError, match="No labeled frames"):
            analysis_h5.write_labels(labels, h5_path)

    def test_single_frame(self, simple_skeleton, tmp_path):
        """Test with single frame."""
        video = Video(str(tmp_path / "video.mp4"))
        track = sio.Track("animal1")

        inst = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]),
            simple_skeleton,
            np.array([0.9, 0.85]),
            0.95,
            track=track,
        )
        lf = LabeledFrame(video=video, frame_idx=5, instances=[inst])
        labels = Labels(labeled_frames=[lf], tracks=[track])

        h5_path = tmp_path / "single.analysis.h5"
        analysis_h5.write_labels(labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)
        assert len(loaded) == 1

    def test_no_skeleton_edges(self, tmp_path):
        """Test with skeleton that has no edges."""
        skeleton = Skeleton(nodes=["point1", "point2"], edges=[], name="noedges")
        video = Video(str(tmp_path / "video.mp4"))
        track = sio.Track("animal1")

        inst = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]),
            skeleton,
            np.array([0.9, 0.85]),
            0.95,
            track=track,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = Labels(labeled_frames=[lf], tracks=[track])

        h5_path = tmp_path / "noedges.analysis.h5"
        analysis_h5.write_labels(labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)
        assert len(loaded.skeleton.edges) == 0

    def test_video_index_selection(self, simple_skeleton, tmp_path):
        """Test selecting video by index."""
        video1 = Video(str(tmp_path / "video1.mp4"))
        video2 = Video(str(tmp_path / "video2.mp4"))
        track = sio.Track("animal1")

        frames = []
        for i, video in enumerate([video1, video2]):
            inst = PredictedInstance.from_numpy(
                np.array([[100.0 + i * 100, 200.0], [150.0 + i * 100, 250.0]]),
                simple_skeleton,
                np.array([0.9, 0.85]),
                0.95,
                track=track,
            )
            lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
            frames.append(lf)

        labels = Labels(labeled_frames=frames, tracks=[track])

        h5_path = tmp_path / "video_idx.analysis.h5"
        analysis_h5.write_labels(labels, h5_path, video=1)

        with h5py.File(h5_path, "r") as f:
            assert "video2.mp4" in f["video_path"][()].decode()

    def test_invalid_preset(self, simple_labels, tmp_path):
        """Test that invalid preset raises ValueError."""
        h5_path = tmp_path / "invalid_preset.analysis.h5"

        with pytest.raises(ValueError, match="Unknown preset"):
            analysis_h5.write_labels(simple_labels, h5_path, preset="invalid")

    def test_legacy_file_without_dims(self, simple_skeleton, tmp_path):
        """Test reading legacy file without dims attributes (transpose=True)."""
        h5_path = tmp_path / "legacy.analysis.h5"

        # Create a legacy-style HDF5 file (without dims attributes)
        video = Video(str(tmp_path / "video.mp4"))
        track = sio.Track("animal1")
        inst = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]),
            simple_skeleton,
            np.array([0.9, 0.85]),
            0.95,
            track=track,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = Labels(labeled_frames=[lf], tracks=[track])

        # Write using current method
        analysis_h5.write_labels(labels, h5_path)

        # Remove dims attributes to simulate legacy file
        with h5py.File(h5_path, "r+") as f:
            for ds_name in [
                "tracks",
                "track_occupancy",
                "point_scores",
                "instance_scores",
                "tracking_scores",
            ]:
                if "dims" in f[ds_name].attrs:
                    del f[ds_name].attrs["dims"]
            # Set legacy transpose attribute instead of preset
            del f.attrs["preset"]
            f.attrs["transpose"] = True

        # Should still be able to read
        loaded = analysis_h5.read_labels(h5_path)
        assert len(loaded) == 1

    def test_legacy_file_transpose_false(self, simple_skeleton, tmp_path):
        """Test reading legacy file with transpose=False."""
        h5_path = tmp_path / "legacy_notranspose.analysis.h5"

        # Create legacy file with transpose=False format
        # Legacy shape was (frame, node, xy, track) for tracks
        with h5py.File(h5_path, "w") as f:
            # Store in legacy shape (frame, node, xy, track)
            tracks_data = np.array([[[[100.0], [150.0]], [[200.0], [250.0]]]])
            # Shape: (1, 2, 2, 1) = (frames, nodes, xy, tracks)
            f.create_dataset("tracks", data=tracks_data, compression="gzip")
            f.create_dataset("track_occupancy", data=np.array([[1]]))
            f.create_dataset("point_scores", data=np.array([[[0.9], [0.85]]]))
            f.create_dataset("instance_scores", data=np.array([[0.95]]))
            f.create_dataset("tracking_scores", data=np.array([[np.nan]]))
            f.create_dataset("track_names", data=[b"animal1"])
            f.create_dataset("node_names", data=[b"head", b"tail"])
            f.create_dataset("edge_names", data=[(b"head", b"tail")])
            f.create_dataset("edge_inds", data=[[0, 1]])
            f.create_dataset("video_path", data=str(tmp_path / "video.mp4"))
            f.create_dataset("provenance", data="{}")
            f.attrs["transpose"] = False
            f.attrs["format"] = "analysis"

        loaded = analysis_h5.read_labels(h5_path)
        assert len(loaded) == 1

    def test_user_instances(self, simple_skeleton, tmp_path):
        """Test handling of user (non-predicted) instances."""
        video = Video(str(tmp_path / "video.mp4"))
        track = sio.Track("animal1")

        # Create user instance (not PredictedInstance)
        user_inst = Instance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]),
            simple_skeleton,
            track=track,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst])
        labels = Labels(labeled_frames=[lf], tracks=[track])

        h5_path = tmp_path / "user_instances.analysis.h5"
        analysis_h5.write_labels(labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)
        assert len(loaded) == 1
        # Points should be preserved
        np.testing.assert_allclose(
            loaded[0].instances[0].numpy()[:, :2],  # x, y only
            user_inst.numpy()[:, :2],
        )

    def test_untracked_labels(self, simple_skeleton, tmp_path):
        """Test labels without tracks (single instance case)."""
        video = Video(str(tmp_path / "video.mp4"))

        # Create instance without track
        inst = PredictedInstance.from_numpy(
            np.array([[100.0, 200.0], [150.0, 250.0]]),
            simple_skeleton,
            np.array([0.9, 0.85]),
            0.95,
            track=None,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = Labels(labeled_frames=[lf], tracks=[])

        h5_path = tmp_path / "untracked.analysis.h5"
        analysis_h5.write_labels(labels, h5_path)

        loaded = analysis_h5.read_labels(h5_path)
        assert len(loaded) == 1
        # Points should be preserved even for untracked instances
        np.testing.assert_allclose(
            loaded[0].instances[0].numpy(),
            inst.numpy(),
        )


# =============================================================================
# Integration with Real Data
# =============================================================================


class TestRealData:
    """Tests using real test fixtures."""

    def test_centered_pair_roundtrip(self, centered_pair, tmp_path):
        """Test round-trip with centered_pair_predictions.slp fixture."""
        labels = sio.load_slp(centered_pair)

        h5_path = tmp_path / "centered_pair.analysis.h5"
        sio.save_analysis_h5(labels, str(h5_path))

        loaded = sio.load_analysis_h5(str(h5_path))

        # Verify structure
        assert len(loaded) == len(labels)
        assert len(loaded.skeleton.nodes) == len(labels.skeleton.nodes)

    def test_centered_pair_with_filtering(self, centered_pair, tmp_path):
        """Test track filtering with real data."""
        labels = sio.load_slp(centered_pair)

        h5_path_all = tmp_path / "all_tracks.analysis.h5"
        h5_path_filtered = tmp_path / "filtered.analysis.h5"

        sio.save_analysis_h5(labels, str(h5_path_all), min_occupancy=0.0)
        sio.save_analysis_h5(labels, str(h5_path_filtered), min_occupancy=0.5)

        # Filtered file should be smaller
        assert h5_path_filtered.stat().st_size < h5_path_all.stat().st_size

    def test_centered_pair_standard_preset(self, centered_pair, tmp_path):
        """Test standard preset with real data."""
        labels = sio.load_slp(centered_pair)

        h5_path = tmp_path / "centered_pair_standard.analysis.h5"
        sio.save_analysis_h5(labels, str(h5_path), preset="standard")

        loaded = sio.load_analysis_h5(str(h5_path))

        # Verify structure
        assert len(loaded) == len(labels)
