"""Tests for TrackMate CSV reader."""

import pytest

from sleap_io import Labels, Skeleton, load_file, load_trackmate, save_slp
from sleap_io.io.trackmate import is_trackmate_file, read_trackmate_csv
from sleap_io.model.centroid import PredictedCentroid

# -- Helpers to write synthetic TrackMate CSVs --

_SPOTS_HEADER = (
    "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,"
    "POSITION_T,FRAME,RADIUS,VISIBILITY\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius,Visibility\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R,Visibility\n"
    ",,,(quality),(pixel),(pixel),(pixel),(frame),,(pixel),\n"
)

_EDGES_HEADER = (
    "LABEL,TRACK_ID,SPOT_SOURCE_ID,SPOT_TARGET_ID,LINK_COST,"
    "SPEED,DISPLACEMENT\n"
    "Label,Track ID,Source spot ID,Target spot ID,Edge cost,"
    "Speed,Displacement\n"
    "Label,Track ID,Source ID,Target ID,Cost,Speed,Disp.\n"
    ",,,,(cost),(pixel/frame),(pixel)\n"
)


def _write_spots(path, rows):
    """Write a minimal TrackMate spots CSV.

    Each row is a tuple: (label, id, track_id, quality, x, y, z, t, frame, radius, vis)
    """
    with open(path, "w", newline="") as f:
        f.write(_SPOTS_HEADER)
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def _write_edges(path, rows):
    """Write a minimal TrackMate edges CSV.

    Each row is a tuple: (label, track_id, source_id, target_id, cost, speed, disp)
    """
    with open(path, "w", newline="") as f:
        f.write(_EDGES_HEADER)
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


# -- Tests --


def test_read_trackmate_basic(tmp_path):
    """Basic import of spots with two tracks."""
    spots = [
        ("ID100", 100, 0, 5.5, 10.0, 20.0, 0.0, 0.0, 0, 11.5, 1),
        ("ID101", 101, 0, 5.3, 12.0, 22.0, 0.0, 1.0, 1, 11.5, 1),
        ("ID200", 200, 1, 4.0, 50.0, 60.0, 0.0, 0.0, 0, 11.5, 1),
    ]
    spots_path = tmp_path / "test_spots.csv"
    _write_spots(spots_path, spots)

    labels = read_trackmate_csv(spots_path)

    assert len(labels.centroids) == 3
    assert len(labels.tracks) == 2

    c0 = labels.centroids[0]
    assert isinstance(c0, PredictedCentroid)
    assert c0.x == pytest.approx(10.0)
    assert c0.y == pytest.approx(20.0)
    assert c0.z is None  # 0.0 -> None
    assert c0.frame_idx == 0
    assert c0.score == pytest.approx(5.5)
    assert c0.name == "ID100"
    assert c0.source == "trackmate"
    assert c0.track is not None
    assert c0.track.name == "Track_0"

    c2 = labels.centroids[2]
    assert c2.track.name == "Track_1"


def test_read_trackmate_with_edges(tmp_path):
    """Edges CSV populates tracking_score on target centroids."""
    spots = [
        ("ID10", 10, 0, 5.0, 1.0, 2.0, 0.0, 0.0, 0, 11.5, 1),
        ("ID11", 11, 0, 5.0, 3.0, 4.0, 0.0, 1.0, 1, 11.5, 1),
        ("ID12", 12, 0, 5.0, 5.0, 6.0, 0.0, 2.0, 2, 11.5, 1),
    ]
    edges = [
        ("ID10 -> ID11", 0, 10, 11, 0.5, 2.0, 2.0),
        ("ID11 -> ID12", 0, 11, 12, 1.2, 1.5, 1.5),
    ]
    spots_path = tmp_path / "data_spots.csv"
    edges_path = tmp_path / "data_edges.csv"
    _write_spots(spots_path, spots)
    _write_edges(edges_path, edges)

    labels = read_trackmate_csv(spots_path, edges_path=edges_path)

    assert len(labels.centroids) == 3

    # First spot in track has no edge -> tracking_score is None.
    c0 = labels.centroids[0]
    assert c0.name == "ID10"
    assert c0.tracking_score is None

    # Target spots get tracking_score from edge cost.
    c1 = labels.centroids[1]
    assert c1.name == "ID11"
    assert c1.tracking_score == pytest.approx(0.5)

    c2 = labels.centroids[2]
    assert c2.name == "ID12"
    assert c2.tracking_score == pytest.approx(1.2)


def test_read_trackmate_auto_detect_edges(tmp_path):
    """Auto-detects sibling _edges.csv file."""
    spots = [
        ("ID1", 1, 0, 5.0, 1.0, 2.0, 0.0, 0.0, 0, 11.5, 1),
        ("ID2", 2, 0, 5.0, 3.0, 4.0, 0.0, 1.0, 1, 11.5, 1),
    ]
    edges = [("ID1 -> ID2", 0, 1, 2, 0.8, 1.0, 1.0)]

    _write_spots(tmp_path / "sample_spots.csv", spots)
    _write_edges(tmp_path / "sample_edges.csv", edges)

    # Don't pass edges_path — should be auto-detected.
    labels = read_trackmate_csv(tmp_path / "sample_spots.csv")

    c1 = labels.centroids[1]
    assert c1.tracking_score == pytest.approx(0.8)


def test_read_trackmate_with_video(tmp_path):
    """String video path creates a Video object."""
    spots = [("ID1", 1, 0, 5.0, 1.0, 2.0, 0.0, 0.0, 0, 11.5, 1)]
    _write_spots(tmp_path / "test_spots.csv", spots)

    labels = read_trackmate_csv(tmp_path / "test_spots.csv", video="my_video.tif")

    assert len(labels.videos) == 1
    assert labels.videos[0].filename == "my_video.tif"
    assert labels.centroids[0].video is labels.videos[0]


def test_read_trackmate_unassigned_spots(tmp_path):
    """Empty TRACK_ID -> track=None."""
    spots = [
        ("ID1", 1, "", 3.0, 10.0, 20.0, 0.0, 0.0, 0, 11.5, 1),
        ("ID2", 2, 0, 5.0, 30.0, 40.0, 0.0, 0.0, 0, 11.5, 1),
    ]
    _write_spots(tmp_path / "test_spots.csv", spots)

    labels = read_trackmate_csv(tmp_path / "test_spots.csv")

    assert labels.centroids[0].track is None
    assert labels.centroids[1].track is not None
    assert len(labels.tracks) == 1


def test_read_trackmate_z_coordinate(tmp_path):
    """Non-zero Z -> z field populated; zero Z -> z=None."""
    spots = [
        ("ID1", 1, 0, 5.0, 1.0, 2.0, 3.5, 0.0, 0, 11.5, 1),
        ("ID2", 2, 0, 5.0, 4.0, 5.0, 0.0, 1.0, 1, 11.5, 1),
    ]
    _write_spots(tmp_path / "test_spots.csv", spots)

    labels = read_trackmate_csv(tmp_path / "test_spots.csv")

    assert labels.centroids[0].z == pytest.approx(3.5)
    assert labels.centroids[1].z is None


def test_is_trackmate_file(tmp_path):
    """Positive and negative detection."""
    # Positive: real TrackMate header.
    spots_path = tmp_path / "spots.csv"
    _write_spots(spots_path, [])
    assert is_trackmate_file(spots_path)

    # Negative: generic CSV.
    generic = tmp_path / "generic.csv"
    generic.write_text("col_a,col_b,col_c\n1,2,3\n")
    assert not is_trackmate_file(generic)

    # Negative: non-existent file.
    assert not is_trackmate_file(tmp_path / "nope.csv")


def test_load_file_trackmate(tmp_path):
    """Auto-detection via load_file()."""
    spots = [("ID1", 1, 0, 5.0, 1.0, 2.0, 0.0, 0.0, 0, 11.5, 1)]
    spots_path = tmp_path / "test_spots.csv"
    _write_spots(spots_path, spots)

    labels = load_file(str(spots_path))

    assert isinstance(labels, Labels)
    assert len(labels.centroids) == 1
    assert labels.centroids[0].x == pytest.approx(1.0)


def test_load_trackmate_public_api(tmp_path):
    """load_trackmate() top-level API works."""
    spots = [("ID1", 1, 0, 5.0, 1.0, 2.0, 0.0, 0.0, 0, 11.5, 1)]
    spots_path = tmp_path / "test_spots.csv"
    _write_spots(spots_path, spots)

    labels = load_trackmate(str(spots_path))

    assert len(labels.centroids) == 1


def test_load_trackmate_roundtrip(tmp_path):
    """Load TrackMate -> save as SLP -> reload -> verify centroids preserved."""
    spots = [
        ("ID1", 1, 0, 5.0, 10.0, 20.0, 0.0, 0.0, 0, 11.5, 1),
        ("ID2", 2, 0, 4.5, 12.0, 22.0, 0.0, 1.0, 1, 11.5, 1),
        ("ID3", 3, 1, 3.0, 50.0, 60.0, 0.0, 0.0, 0, 11.5, 1),
    ]
    edges = [("ID1 -> ID2", 0, 1, 2, 0.75, 1.0, 1.0)]

    _write_spots(tmp_path / "data_spots.csv", spots)
    _write_edges(tmp_path / "data_edges.csv", edges)

    labels = read_trackmate_csv(tmp_path / "data_spots.csv")
    assert len(labels.centroids) == 3
    assert len(labels.tracks) == 2

    # Save as SLP and reload.
    slp_path = str(tmp_path / "roundtrip.slp")
    labels.skeletons = [Skeleton(["A"])]  # SLP needs at least one skeleton
    save_slp(labels, slp_path)

    loaded = load_file(slp_path)
    assert len(loaded.centroids) == 3
    assert len(loaded.tracks) == 2

    c0 = loaded.centroids[0]
    assert c0.x == pytest.approx(10.0)
    assert c0.score == pytest.approx(5.0)
    assert c0.tracking_score is None  # First in track

    c1 = loaded.centroids[1]
    assert c1.tracking_score == pytest.approx(0.75)

    c2 = loaded.centroids[2]
    assert c2.track is loaded.tracks[1]


def test_read_trackmate_not_found(tmp_path):
    """FileNotFoundError for missing spots CSV."""
    with pytest.raises(FileNotFoundError):
        read_trackmate_csv(tmp_path / "nonexistent_spots.csv")


def test_read_trackmate_bad_header(tmp_path):
    """ValueError for CSV without TrackMate signature."""
    bad = tmp_path / "bad.csv"
    bad.write_text("col_a,col_b\n1,2\n")
    with pytest.raises(ValueError, match="Not a TrackMate spots CSV"):
        read_trackmate_csv(bad)
