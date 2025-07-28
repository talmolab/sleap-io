"""Tests for DeepLabCut I/O operations."""

import numpy as np

import sleap_io as sio
from sleap_io.io import dlc


def test_is_dlc_file(dlc_maudlc_testdata, dlc_testdata):
    """Test DLC file detection."""
    # Should detect DLC files
    assert dlc.is_dlc_file(dlc_maudlc_testdata)
    assert dlc.is_dlc_file(dlc_testdata)

    # Should not detect non-DLC files
    assert not dlc.is_dlc_file("tests/data/slp/minimal_instance.slp")


def test_load_maudlc_testdata(dlc_maudlc_testdata):
    """Test loading multi-animal DLC data with individual tracking (MAUDLC)."""
    labels = sio.load_file(dlc_maudlc_testdata)

    assert isinstance(labels, sio.Labels)
    assert len(labels.skeletons) == 1
    assert len(labels.skeleton.nodes) == 5  # A, B, C, D, E
    assert len(labels.tracks) == 3  # Animal1, Animal2, single
    assert len(labels.labeled_frames) == 3  # 3 labeled frames

    # Check skeleton nodes
    node_names = [node.name for node in labels.skeleton.nodes]
    assert set(node_names) == {"A", "B", "C", "D", "E"}

    # Check track names
    track_names = [track.name for track in labels.tracks]
    assert set(track_names) == {"Animal1", "Animal2", "single"}

    # Check frame structure
    assert labels.labeled_frames[0].frame_idx == 0
    assert len(labels.labeled_frames[0].instances) == 2  # Frame 0: 2 instances
    assert labels.labeled_frames[1].frame_idx == 1
    assert len(labels.labeled_frames[1].instances) == 3  # Frame 1: 3 instances
    assert labels.labeled_frames[2].frame_idx == 3
    assert len(labels.labeled_frames[2].instances) == 2  # Frame 3: 2 instances


def test_load_madlc_testdata(dlc_madlc_testdata):
    """Test loading multi-animal DLC data (MADLC)."""
    labels = sio.load_file(dlc_madlc_testdata)

    assert isinstance(labels, sio.Labels)
    assert len(labels.skeletons) == 1
    assert len(labels.skeleton.nodes) == 3  # A, B, C
    assert len(labels.labeled_frames) == 3  # 3 labeled frames

    # Check skeleton nodes
    node_names = [node.name for node in labels.skeleton.nodes]
    assert set(node_names) == {"A", "B", "C"}

    # Check frame structure
    assert labels.labeled_frames[0].frame_idx == 0
    assert len(labels.labeled_frames[0].instances) == 2  # Frame 0: 2 instances
    assert labels.labeled_frames[1].frame_idx == 1
    assert len(labels.labeled_frames[1].instances) == 2  # Frame 1: 2 instances
    assert labels.labeled_frames[2].frame_idx == 3
    assert len(labels.labeled_frames[2].instances) == 1  # Frame 3: 1 instance


def test_load_sadlc_testdata(dlc_testdata):
    """Test loading single-animal DLC data (SADLC)."""
    labels = sio.load_file(dlc_testdata)

    assert isinstance(labels, sio.Labels)
    assert len(labels.skeletons) == 1
    assert len(labels.skeleton.nodes) == 3  # A, B, C
    assert len(labels.tracks) == 0  # No tracks for single animal
    assert len(labels.labeled_frames) == 3  # 3 labeled frames

    # Check skeleton nodes
    node_names = [node.name for node in labels.skeleton.nodes]
    assert set(node_names) == {"A", "B", "C"}

    # Check frame structure - 1 instance per frame
    for labeled_frame in labels.labeled_frames:
        assert len(labeled_frame.instances) == 1


def test_load_multiple_datasets(
    dlc_multiple_datasets_video1, dlc_multiple_datasets_video2
):
    """Test loading from multiple dataset structure."""
    labels1 = sio.load_file(dlc_multiple_datasets_video1)
    labels2 = sio.load_file(dlc_multiple_datasets_video2)

    # Both should have same structure
    for labels in [labels1, labels2]:
        assert isinstance(labels, sio.Labels)
        assert len(labels.skeletons) == 1
        assert len(labels.skeleton.nodes) == 3  # A, B, C
        assert len(labels.labeled_frames) >= 1  # At least one frame

        # Check skeleton nodes
        node_names = [node.name for node in labels.skeleton.nodes]
        assert set(node_names) == {"A", "B", "C"}


def test_coordinate_parsing(dlc_testdata):
    """Test that coordinates are correctly parsed."""
    labels = sio.load_file(dlc_testdata)

    # Check that we have valid coordinates
    has_valid_coords = False
    for labeled_frame in labels.labeled_frames:
        for instance in labeled_frame.instances:
            for point in instance.points["xy"]:
                if not np.isnan(point).all():
                    has_valid_coords = True
                    assert len(point) == 2  # x, y coordinates
                    assert isinstance(point[0], (np.integer, float, np.floating))
                    assert isinstance(point[1], (np.integer, float, np.floating))

    assert has_valid_coords, "Should have at least some valid coordinates"


def test_missing_coordinates(dlc_maudlc_testdata):
    """Test handling of missing coordinates (NaN values)."""
    labels = sio.load_file(dlc_maudlc_testdata)

    # Check that some coordinates are NaN (as expected from fixture description)
    has_nan_coords = False
    for labeled_frame in labels.labeled_frames:
        for instance in labeled_frame.instances:
            for point in instance.points["xy"]:
                if np.isnan(point).any():
                    has_nan_coords = True
                    break

    assert has_nan_coords, "Should have some NaN coordinates as per fixture description"


def test_video_creation(dlc_testdata):
    """Test that video objects are created correctly."""
    labels = sio.load_file(dlc_testdata)

    assert len(labels.videos) >= 1
    for video in labels.videos:
        # Video backends inherit from VideoBackend, not Video
        from sleap_io.io.video_reading import VideoBackend

        assert isinstance(video, VideoBackend)
        assert video.num_frames > 0


def test_track_assignment(dlc_maudlc_testdata):
    """Test that tracks are correctly assigned to instances in multi-animal data."""
    labels = sio.load_file(dlc_maudlc_testdata)

    # Check that some instances have tracks assigned
    has_tracks = False
    track_names_found = set()

    for labeled_frame in labels.labeled_frames:
        for instance in labeled_frame.instances:
            if instance.track is not None:
                has_tracks = True
                track_names_found.add(instance.track.name)

    assert has_tracks, "Multi-animal data should have track assignments"
    # Should have some of the expected track names
    expected_tracks = {"Animal1", "Animal2", "single"}
    assert len(track_names_found.intersection(expected_tracks)) > 0


def test_load_via_main_api(dlc_testdata):
    """Test loading DLC files through main load_file API."""
    # Test automatic format detection
    labels1 = sio.load_file(dlc_testdata)

    # Test explicit format specification
    labels2 = sio.load_file(dlc_testdata, format="dlc")

    # Both should produce same result
    assert len(labels1.labeled_frames) == len(labels2.labeled_frames)
    assert len(labels1.skeletons) == len(labels2.skeletons)
    assert len(labels1.tracks) == len(labels2.tracks)


def test_invalid_csv_not_detected(tmp_path):
    """Test that non-DLC CSV files are not detected as DLC files."""
    # Create a non-DLC CSV file
    invalid_csv = tmp_path / "not_dlc.csv"
    invalid_csv.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")

    assert not dlc.is_dlc_file(invalid_csv)


def test_empty_csv_not_detected(tmp_path):
    """Test that empty CSV files are not detected as DLC files."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    assert not dlc.is_dlc_file(empty_csv)


def test_nonexistent_file_not_detected():
    """Test that nonexistent files are not detected as DLC files."""
    assert not dlc.is_dlc_file("nonexistent_file.csv")


def test_malformed_csv_format_detection(tmp_path):
    """Test format detection with CSV that triggers exception handling."""
    # Create a CSV that looks like multi-animal but will fail the check
    malformed_csv = tmp_path / "malformed.csv"
    # This CSV has inconsistent structure that will trigger the exception path
    malformed_csv.write_text(
        "scorer,Scorer,Scorer,Scorer,Scorer,Scorer,Scorer\n"
        "not_individuals,Animal1,Animal1,Animal2,Animal2,single,single\n"
        "bodyparts,A,A,A,A,D,D\n"
        "coords,x,y,x,y,x,y\n"
        "frame1,1,2,3,4,5,6\n"
    )

    # This should fall back to single-animal format since "individuals" check fails
    labels = sio.load_file(malformed_csv)
    assert isinstance(labels, sio.Labels)
    assert len(labels.tracks) == 0  # Single-animal has no tracks


def test_extract_frame_index_no_numbers(dlc_testdata):
    """Test frame index extraction when filename has no numbers."""
    from sleap_io.io.dlc import _extract_frame_index

    # Test with filename without numbers
    assert _extract_frame_index("no_numbers.png") == 0
    assert _extract_frame_index("also-no-nums.jpg") == 0

    # Test normal case for comparison
    assert _extract_frame_index("img001.png") == 1
    assert _extract_frame_index("frame_042.jpg") == 42


def test_video_creation_with_existing_files(tmp_path, dlc_testdata):
    """Test video creation when image files actually exist."""
    from sleap_io.io.dlc import _get_or_create_video

    # Create actual image files
    img_dir = tmp_path / "labeled-data" / "video"
    img_dir.mkdir(parents=True)

    # Create a dummy image file
    img_file = img_dir / "img000.png"
    img_file.write_text("dummy")

    # Test with full path that exists
    video = _get_or_create_video("labeled-data/video/img000.png", tmp_path, None)
    assert video.filename == [str(tmp_path / "labeled-data/video/img000.png")]

    # Test with just the filename (should use simple path)
    video2 = _get_or_create_video("img000.png", img_dir, None)
    assert video2.filename == [str(img_dir / "img000.png")]
