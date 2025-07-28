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
        assert len(labels.videos) == 1  # Each should have one video

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
        assert isinstance(video, sio.Video)
        # Check that it's an image sequence video with multiple frames
        assert hasattr(video, "backend")
        assert video.backend is not None


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


def test_csv_parse_exception_handling(tmp_path):
    """Test that CSV parsing exceptions are handled gracefully."""
    # Create a CSV that looks like DLC but will cause pandas to fail
    # during the multi-animal header check
    bad_csv = tmp_path / "bad_structure.csv"
    bad_csv.write_text(
        "scorer,Scorer\nbodyparts,A\ncoords,x\n"
        # Only one header row when we try to read with header=[1,2,3]
    )

    # This should trigger the exception handler and fall back to single-animal format
    labels = sio.load_file(bad_csv)
    assert isinstance(labels, sio.Labels)
    assert len(labels.labeled_frames) == 0  # No valid data to parse


def test_extract_frame_index_no_numbers(dlc_testdata):
    """Test frame index extraction when filename has no numbers."""
    from sleap_io.io.dlc import _extract_frame_index

    # Test with filename without numbers
    assert _extract_frame_index("no_numbers.png") == 0
    assert _extract_frame_index("also-no-nums.jpg") == 0

    # Test normal case for comparison
    assert _extract_frame_index("img001.png") == 1
    assert _extract_frame_index("frame_042.jpg") == 42


def test_single_video_per_folder(dlc_testdata):
    """Test that a single Video object is created per video folder."""
    labels = sio.load_file(dlc_testdata)

    # Should have exactly one video for all frames
    assert len(labels.videos) == 1

    # All labeled frames should reference the same video
    video = labels.videos[0]
    for lf in labels.labeled_frames:
        assert lf.video is video

    # Frame indices should be correct (0, 1, 3 based on the test data)
    frame_indices = sorted([lf.frame_idx for lf in labels.labeled_frames])
    assert frame_indices == [0, 1, 3]


def test_dlc_with_nested_path_structure(tmp_path):
    """Test DLC loading when CSV references images with full path structure."""
    # Create nested directory structure
    data_dir = tmp_path / "project"
    labeled_dir = data_dir / "labeled-data" / "session1"
    labeled_dir.mkdir(parents=True)

    # Create images in the expected location
    for i in range(3):
        img_path = labeled_dir / f"img{i:03d}.png"
        img_path.write_text("dummy image")

    # Create CSV that references images with full path
    csv_path = labeled_dir / "test_data.csv"
    csv_content = (
        "scorer,Scorer,Scorer,Scorer,Scorer,Scorer,Scorer\n"
        "bodyparts,A,A,B,B,C,C\n"
        "coords,x,y,x,y,x,y\n"
        "labeled-data/session1/img000.png,0,1,2,3,4,5\n"
        "labeled-data/session1/img001.png,10,11,12,13,14,15\n"
        "labeled-data/session1/img002.png,20,21,22,23,24,25\n"
    )
    csv_path.write_text(csv_content)

    # Load from CSV - this tests the full path resolution (line 127)
    labels = sio.load_file(csv_path)
    assert len(labels.labeled_frames) == 3
    assert len(labels.videos) == 1

    # Now test from parent directory - this tests the parent path resolution (line 139)
    csv_parent_path = data_dir / "test_data.csv"
    csv_parent_path.write_text(csv_content)

    labels2 = sio.load_file(csv_parent_path)
    assert len(labels2.labeled_frames) == 3
    assert len(labels2.videos) == 1
