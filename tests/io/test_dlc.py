"""Tests for DeepLabCut I/O operations."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

import sleap_io as sio
from sleap_io.io import dlc
from sleap_io.model.labels_set import LabelsSet


def make_dlc_project(
    root,
    *,
    scorer="LM",
    task="proj",
    date="Jan1",
    iteration=0,
    bodyparts=("snout", "leftear", "rightear"),
    skeleton=(("snout", "leftear"), ("snout", "rightear")),
    folders=None,
    video_sets=None,
    make_images=True,
    train_indices=None,
    test_indices=None,
    train_fraction=0.8,
    shuffle=1,
    csv_scorer=None,
):
    """Build a minimal synthetic single-animal DLC project under ``root``.

    Args:
        root: Project root directory (created if needed).
        scorer: Project scorer name (used for config and CSV filenames).
        task: DLC ``Task`` name.
        date: DLC ``date`` string.
        iteration: DLC ``iteration`` index.
        bodyparts: Bodypart names.
        skeleton: Iterable of ``(src, dst)`` edge name pairs for the config.
        folders: Mapping of ``labeled-data`` folder name to a list of image
            basenames (without extension). Defaults to two videos.
        video_sets: Optional explicit ``video_sets`` mapping. If `None`, built
            from `folders` so each folder stem matches a ``<folder>.mp4`` entry.
        make_images: Whether to write dummy image files.
        train_indices: If given (with `test_indices`), write a Documentation
            pickle encoding these positional split indices.
        test_indices: Positional test indices for the Documentation pickle.
        train_fraction: Training fraction for the pickle name/contents.
        shuffle: Shuffle index for the pickle name.
        csv_scorer: Scorer name written into the CSV header (defaults to
            `scorer`); set differently to simulate a scorer mismatch.

    Returns:
        The path to the project's ``config.yaml``.
    """
    root = Path(root)
    if folders is None:
        folders = {"vid1": ["img000", "img001", "img002"], "vid2": ["img000", "img001"]}
    csv_scorer = scorer if csv_scorer is None else csv_scorer

    nbp = len(bodyparts)
    scorer_row = "scorer," + ",".join([csv_scorer] * (2 * nbp))
    bp_row = "bodyparts," + ",".join(bp for bp in bodyparts for _ in range(2))
    coord_row = "coords," + ",".join(["x", "y"] * nbp)

    for folder, imgs in folders.items():
        d = root / "labeled-data" / folder
        d.mkdir(parents=True, exist_ok=True)
        lines = [scorer_row, bp_row, coord_row]
        for i, img in enumerate(imgs):
            vals = ",".join(str(v) for v in range(i * 100, i * 100 + 2 * nbp))
            lines.append(f"labeled-data/{folder}/{img}.png,{vals}")
        (d / f"CollectedData_{scorer}.csv").write_text("\n".join(lines) + "\n")
        if make_images:
            for img in imgs:
                (d / f"{img}.png").write_text("dummy")

    if video_sets is None:
        video_sets = {
            str(root / "videos" / f"{folder}.mp4"): {"crop": "0, 100, 0, 100"}
            for folder in folders
        }

    cfg = {
        "Task": task,
        "scorer": scorer,
        "date": date,
        "iteration": iteration,
        "multianimalproject": False,
        "video_sets": video_sets,
        "bodyparts": list(bodyparts),
        "skeleton": [list(e) for e in skeleton],
        "TrainingFraction": [train_fraction],
    }
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))

    if train_indices is not None and test_indices is not None:
        tdir = (
            root
            / "training-datasets"
            / f"iteration-{iteration}"
            / f"UnaugmentedDataSet_{task}{date}"
        )
        tdir.mkdir(parents=True, exist_ok=True)
        first_folder = list(folders)[0]
        data = [
            {"image": ("labeled-data", first_folder, f"{folders[first_folder][0]}.png")}
        ]
        name = (
            f"Documentation_data-{task}_{int(round(train_fraction * 100))}"
            f"shuffle{shuffle}.pickle"
        )
        with open(tdir / name, "wb") as f:
            pickle.dump(
                [data, list(train_indices), list(test_indices), train_fraction],
                f,
                pickle.HIGHEST_PROTOCOL,
            )

    return root / "config.yaml"


def _frame_keys(labels):
    """Return sorted ``(folder, filename)`` keys for a Labels' frames."""
    keys = []
    for lf in labels.labeled_frames:
        fname = lf.video.filename
        if isinstance(fname, list):
            fname = fname[lf.frame_idx]
        p = Path(str(fname))
        keys.append((p.parent.name, p.name))
    return sorted(keys)


def test_is_dlc_file(dlc_maudlc_testdata, dlc_testdata):
    """Test DLC file detection."""
    # Should detect DLC files
    assert dlc.is_dlc_file(dlc_maudlc_testdata)
    assert dlc.is_dlc_file(dlc_testdata)

    # Should not detect non-DLC files
    assert not dlc.is_dlc_file("tests/data/slp/minimal_instance.slp")


@pytest.mark.parametrize(
    "dlc_fixture",
    [
        "dlc_maudlc_testdata",
        "dlc_maudlc_testdata_v2",
    ],
)
def test_load_maudlc_testdata(dlc_fixture, request):
    """Test loading multi-animal DLC data with individual tracking (MAUDLC)."""
    labels = sio.load_file(request.getfixturevalue(dlc_fixture))

    assert isinstance(labels, sio.Labels)
    assert len(labels.skeletons) == 1
    assert len(labels.skeleton.nodes) == 5  # A, B, C, D, E
    assert len(labels.tracks) == 3  # Animal1, Animal2, single
    assert (
        len(labels.labeled_frames) == 4
    )  # 4 labeled frames (including negative frame)

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
    assert labels.labeled_frames[2].frame_idx == 2
    assert len(labels.labeled_frames[2].instances) == 0  # Frame 2: 0 instances
    assert labels.labeled_frames[3].frame_idx == 3
    assert len(labels.labeled_frames[3].instances) == 2  # Frame 3: 2 instances


@pytest.mark.parametrize(
    "dlc_fixture",
    [
        "dlc_madlc_testdata",
        "dlc_madlc_testdata_v2",
    ],
)
def test_load_madlc_testdata(dlc_fixture, request):
    """Test loading multi-animal DLC data (MADLC)."""
    labels = sio.load_file(request.getfixturevalue(dlc_fixture))

    assert isinstance(labels, sio.Labels)
    assert len(labels.skeletons) == 1
    assert len(labels.skeleton.nodes) == 3  # A, B, C
    assert (
        len(labels.labeled_frames) == 4
    )  # 4 labeled frames (including negative frame)

    # Check skeleton nodes
    node_names = [node.name for node in labels.skeleton.nodes]
    assert set(node_names) == {"A", "B", "C"}

    # Check frame structure
    assert labels.labeled_frames[0].frame_idx == 0
    assert len(labels.labeled_frames[0].instances) == 2  # Frame 0: 2 instances
    assert labels.labeled_frames[1].frame_idx == 1
    assert len(labels.labeled_frames[1].instances) == 2  # Frame 1: 2 instances
    assert labels.labeled_frames[2].frame_idx == 2
    assert len(labels.labeled_frames[2].instances) == 0  # Frame 2: 0 instances
    assert labels.labeled_frames[3].frame_idx == 3
    assert len(labels.labeled_frames[3].instances) == 1  # Frame 3: 1 instance


@pytest.mark.parametrize(
    "dlc_fixture",
    [
        "dlc_testdata",
        "dlc_testdata_v2",
    ],
)
def test_load_sadlc_testdata(dlc_fixture, request):
    """Test loading single-animal DLC data (SADLC)."""
    labels = sio.load_file(request.getfixturevalue(dlc_fixture))

    assert isinstance(labels, sio.Labels)
    assert len(labels.skeletons) == 1
    assert len(labels.skeleton.nodes) == 3  # A, B, C
    assert len(labels.tracks) == 0  # No tracks for single animal
    assert (
        len(labels.labeled_frames) == 4
    )  # 4 labeled frames (including negative frame)

    # Check skeleton nodes
    node_names = [node.name for node in labels.skeleton.nodes]
    assert set(node_names) == {"A", "B", "C"}

    # Check frame structure - 1 instance in annotated frames; 0 in unannotated
    annotated = [lf for lf in labels.labeled_frames if lf.instances]
    assert all(len(lf.instances) == 1 for lf in annotated)
    unannotated = [lf for lf in labels.labeled_frames if not lf.instances]
    assert len(unannotated) == 1


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

    # Frame indices should be correct (0, 1, 2, 3 based on the test data)
    frame_indices = sorted([lf.frame_idx for lf in labels.labeled_frames])
    assert frame_indices == [0, 1, 2, 3]


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


# -----------------------------------------------------------------------------
# Config parsing and discovery
# -----------------------------------------------------------------------------


def test_read_dlc_config_valid(dlc_config):
    """Reading a valid DLC config returns the expected mapping."""
    cfg = dlc._read_dlc_config(dlc_config)
    assert isinstance(cfg, dict)
    assert cfg["Task"] == "maudlc_2.3.0"
    assert "video_sets" in cfg
    assert cfg["skeleton"] == [["A", "B"], ["B", "C"], ["A", "C"]]


def test_read_dlc_config_missing(tmp_path):
    """A missing config file warns and returns None."""
    with pytest.warns(UserWarning):
        assert dlc._read_dlc_config(tmp_path / "nope.yaml") is None


def test_read_dlc_config_non_mapping(tmp_path):
    """A YAML file that is not a mapping warns and returns None."""
    bad = tmp_path / "config.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.warns(UserWarning):
        assert dlc._read_dlc_config(bad) is None


def test_looks_like_dlc_config():
    """DLC config detection requires multiple characteristic keys."""
    assert dlc._looks_like_dlc_config({"video_sets": {}, "bodyparts": []})
    assert not dlc._looks_like_dlc_config({"bodyparts": []})
    assert not dlc._looks_like_dlc_config(["not", "a", "dict"])


def test_discover_config_walks_up(tmp_path):
    """Auto-discovery finds config.yaml two levels above the CSV."""
    config_path = make_dlc_project(tmp_path)
    csv = tmp_path / "labeled-data" / "vid1" / "CollectedData_LM.csv"
    found = dlc._discover_config(csv)
    assert found is not None
    assert Path(found) == Path(config_path)


def test_discover_config_none_when_absent(tmp_path):
    """Auto-discovery returns None when no config.yaml is in range."""
    csv = tmp_path / "labeled-data" / "vid1" / "data.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("scorer,A\n")
    assert dlc._discover_config(csv) is None


def test_discover_config_rejects_non_dlc_yaml(tmp_path):
    """A config.yaml that is not a DLC config is rejected by discovery."""
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"unrelated": 1}))
    csv = tmp_path / "labeled-data" / "vid1" / "data.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("scorer,A\n")
    assert dlc._discover_config(csv) is None


# -----------------------------------------------------------------------------
# Item 4: skeleton edges from config
# -----------------------------------------------------------------------------


def test_load_dlc_edges_from_explicit_config(dlc_maudlc_testdata, dlc_config):
    """Skeleton edges are imported from an explicit config; nodes unchanged."""
    labels = sio.load_dlc(dlc_maudlc_testdata, config=dlc_config)
    skel = labels.skeleton
    assert set(skel.node_names) == {"A", "B", "C", "D", "E"}  # nodes unchanged
    assert skel.edge_names == [("A", "B"), ("B", "C"), ("A", "C")]
    assert skel.name == "maudlc_2.3.0"


def test_load_dlc_no_config_has_no_edges(dlc_maudlc_testdata):
    """Without a config, no edges are added (backward compatible)."""
    labels = sio.load_dlc(dlc_maudlc_testdata, config=False)
    assert labels.skeleton.edges == []


def test_load_dlc_config_false_disables_discovery(tmp_path):
    """config=False reproduces legacy output even inside a project."""
    make_dlc_project(tmp_path)
    csv = tmp_path / "labeled-data" / "vid1" / "CollectedData_LM.csv"
    labels = sio.load_dlc(csv, config=False)
    assert labels.skeleton.edges == []


def test_load_dlc_autodiscovers_config(tmp_path):
    """config=None auto-discovers config.yaml and attaches edges."""
    make_dlc_project(tmp_path)
    csv = tmp_path / "labeled-data" / "vid1" / "CollectedData_LM.csv"
    labels = sio.load_dlc(csv)  # default config=None -> auto-discover
    assert labels.skeleton.edge_names == [("snout", "leftear"), ("snout", "rightear")]


def test_edges_referencing_absent_bodypart_dropped(tmp_path):
    """Edges naming a bodypart absent from the data are dropped with a warning."""
    config_path = make_dlc_project(
        tmp_path,
        skeleton=(("snout", "leftear"), ("snout", "ghost")),
    )
    with pytest.warns(UserWarning, match="ghost"):
        labels = sio.load_dlc_project(config_path)
    assert labels.skeleton.edge_names == [("snout", "leftear")]


def test_empty_skeleton_config_no_edges(tmp_path):
    """An empty config skeleton yields no edges and no error."""
    config_path = make_dlc_project(tmp_path, skeleton=())
    labels = sio.load_dlc_project(config_path)
    assert labels.skeleton.edges == []


def test_malformed_skeleton_entry_skipped(tmp_path):
    """Malformed skeleton entries (not 2-tuples) are skipped defensively."""
    skel = sio.Skeleton(nodes=["snout", "leftear"])
    cfg = {
        "Task": "t",
        "skeleton": [["snout", "leftear"], ["snout"], "bad", ["a", "b", "c"]],
    }
    with pytest.warns(UserWarning):
        dlc._attach_config_skeleton(skel, cfg)
    assert skel.edge_names == [("snout", "leftear")]


# -----------------------------------------------------------------------------
# Item 3: source video links from video_sets
# -----------------------------------------------------------------------------


def test_source_video_linked_by_stem(tmp_path):
    """Each image folder is linked to its source video matched by stem."""
    config_path = make_dlc_project(tmp_path)
    labels = sio.load_dlc_project(config_path)
    by_folder = {Path(v.filename[0]).parent.name: v for v in labels.videos}
    assert by_folder["vid1"].source_video is not None
    assert Path(by_folder["vid1"].source_video.filename).name == "vid1.mp4"
    assert by_folder["vid1"].original_video is not None


def test_source_video_none_on_stem_mismatch(tmp_path):
    """A folder whose name does not match any video stem gets no source link."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"video": ["img000", "img001"]},
        video_sets={str(tmp_path / "videos" / "different_name.mp4"): {}},
    )
    labels = sio.load_dlc_project(config_path)
    assert labels.videos[0].source_video is None


def test_source_video_placeholder_skipped(tmp_path):
    """Placeholder video_sets entries are skipped."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000"]},
        video_sets={
            "WILL BE AUTOMATICALLY UPDATED BY DEMO CODE": {"crop": "0, 1, 0, 1"}
        },
    )
    labels = sio.load_dlc_project(config_path)
    assert labels.videos[0].source_video is None


def test_source_video_windows_path_stem(tmp_path):
    """Windows backslash video paths have their stem extracted correctly."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000"]},
        video_sets={r"D:\data\videos\vid1.mp4": {"crop": "0, 1, 0, 1"}},
    )
    labels = sio.load_dlc_project(config_path)
    assert labels.videos[0].source_video is not None
    assert labels.videos[0].source_video.filename == r"D:\data\videos\vid1.mp4"


def test_source_video_search_path_repair(tmp_path):
    """video_search_paths repairs the original path by basename when present."""
    vids = tmp_path / "found_videos"
    vids.mkdir()
    (vids / "vid1.mp4").write_text("dummy")
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000"]},
        video_sets={r"X:\missing\vid1.mp4": {}},
    )
    labels = sio.load_dlc_project(config_path, video_search_paths=[vids])
    assert Path(labels.videos[0].source_video.filename) == vids / "vid1.mp4"


# -----------------------------------------------------------------------------
# load_dlc_project + load_file routing
# -----------------------------------------------------------------------------


def test_load_dlc_project_merges_videos(tmp_path):
    """A multi-video project merges into one Labels with a single skeleton."""
    config_path = make_dlc_project(tmp_path)
    labels = sio.load_dlc_project(config_path)
    assert len(labels.skeletons) == 1
    # Edges deduped to the config's 2 edges, not duplicated per video.
    assert len(labels.skeleton.edges) == 2
    assert len(labels.videos) == 2
    assert len(labels.labeled_frames) == 5  # 3 + 2
    assert labels.provenance["dlc_scorer"] == "LM"
    assert labels.provenance["dlc_task"] == "proj"
    assert "dlc_project" in labels.provenance


def test_load_dlc_project_from_directory(tmp_path):
    """A project directory can be passed directly."""
    make_dlc_project(tmp_path)
    labels = sio.load_dlc_project(tmp_path)
    assert len(labels.videos) == 2


def test_load_dlc_project_directory_no_config(tmp_path):
    """A directory without config.yaml raises a clear error."""
    with pytest.raises(FileNotFoundError):
        dlc._resolve_project_config_path(tmp_path)


def test_load_dlc_project_no_csvs(tmp_path):
    """A project with no annotation CSVs raises a clear error."""
    (tmp_path / "labeled-data").mkdir()
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"Task": "t", "scorer": "LM", "video_sets": {}, "bodyparts": []})
    )
    with pytest.raises(ValueError, match="No DLC annotation CSVs"):
        sio.load_dlc_project(tmp_path / "config.yaml")


def test_load_dlc_project_skips_non_dlc_csv(tmp_path):
    """A non-DLC CSV in a labeled-data folder is ignored during discovery."""
    config_path = make_dlc_project(tmp_path, folders={"vid1": ["img000"]})
    # Drop a non-DLC CSV into a new folder with a non-standard scorer name.
    extra = tmp_path / "labeled-data" / "vid2"
    extra.mkdir()
    (extra / "notes.csv").write_text("a,b,c\n1,2,3\n")
    labels = sio.load_dlc_project(config_path)
    # Only the real DLC folder is loaded.
    assert len(labels.videos) == 1


def test_load_file_routes_dlc_project_dir(tmp_path):
    """load_file routes a DLC project directory to load_dlc_project."""
    make_dlc_project(tmp_path)
    labels = sio.load_file(str(tmp_path))
    assert isinstance(labels, sio.Labels)
    assert len(labels.videos) == 2
    assert len(labels.skeleton.edges) == 2


def test_load_file_routes_config_yaml(tmp_path):
    """load_file routes a config.yaml path to load_dlc_project."""
    config_path = make_dlc_project(tmp_path)
    labels = sio.load_file(str(config_path))
    assert isinstance(labels, sio.Labels)
    assert len(labels.videos) == 2


def test_load_dlc_project_ignores_loader_kwargs(tmp_path):
    """load_dlc_project swallows benign loader kwargs forwarded by load_file."""
    config_path = make_dlc_project(tmp_path)
    # open_videos/lazy are always passed by load_file / `sio show`; they must not
    # raise a TypeError for DLC projects (their videos may not exist on disk).
    labels = sio.load_dlc_project(config_path, open_videos=False, lazy=True)
    assert isinstance(labels, sio.Labels)
    assert len(labels.videos) == 2


def test_load_file_routes_config_yaml_with_open_videos(tmp_path):
    """load_file forwards open_videos to a DLC project without crashing."""
    config_path = make_dlc_project(tmp_path)
    labels = sio.load_file(str(config_path), open_videos=False)
    assert isinstance(labels, sio.Labels)
    assert len(labels.videos) == 2


def test_load_file_routes_dlc_project_dir_with_open_videos(tmp_path):
    """load_file forwards open_videos to a DLC project directory without crashing."""
    make_dlc_project(tmp_path)
    labels = sio.load_file(str(tmp_path), open_videos=False)
    assert isinstance(labels, sio.Labels)
    assert len(labels.videos) == 2


def test_load_dlc_splits_ignores_loader_kwargs(tmp_path):
    """load_dlc_splits swallows benign loader kwargs (parity with load_dlc_project)."""
    config_path = make_dlc_project(
        tmp_path, train_indices=[0, 2, 4], test_indices=[1, 3]
    )
    splits = dlc.load_dlc_splits(config_path, open_videos=False, lazy=True)
    assert isinstance(splits, LabelsSet)
    assert set(splits.labels.keys()) == {"train", "test"}


def test_is_dlc_project_path(tmp_path):
    """Project path detection accepts dirs and config.yaml, rejects others."""
    config_path = make_dlc_project(tmp_path)
    assert dlc._is_dlc_project_path(tmp_path)
    assert dlc._is_dlc_project_path(config_path)
    assert not dlc._is_dlc_project_path(tmp_path / "labeled-data" / "vid1")
    # A non-config file and a nonexistent path are not DLC projects.
    other = tmp_path / "notes.txt"
    other.write_text("hi")
    assert not dlc._is_dlc_project_path(other)
    assert not dlc._is_dlc_project_path(tmp_path / "does_not_exist")


# -----------------------------------------------------------------------------
# Item 1: training-set splits
# -----------------------------------------------------------------------------


def test_load_dlc_splits_maps_frames(tmp_path):
    """Train/test indices map to the correct frames via the merged order."""
    # Merged lexicographic order of (folder, filename):
    #   0:(vid1,img000) 1:(vid1,img001) 2:(vid1,img002) 3:(vid2,img000) 4:(vid2,img001)
    config_path = make_dlc_project(
        tmp_path, train_indices=[0, 2, 4], test_indices=[1, 3]
    )
    splits = sio.load_dlc_splits(config_path)
    assert isinstance(splits, LabelsSet)
    assert set(splits.labels.keys()) == {"train", "test"}
    assert _frame_keys(splits["train"]) == [
        ("vid1", "img000.png"),
        ("vid1", "img002.png"),
        ("vid2", "img001.png"),
    ]
    assert _frame_keys(splits["test"]) == [
        ("vid1", "img001.png"),
        ("vid2", "img000.png"),
    ]


def test_load_dlc_splits_filters_negative_indices(tmp_path):
    """Sentinel -1 indices are filtered out of the splits."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        train_indices=[0, -1],
        test_indices=[1],
    )
    splits = sio.load_dlc_splits(config_path)
    assert _frame_keys(splits["train"]) == [("vid1", "img000.png")]
    assert _frame_keys(splits["test"]) == [("vid1", "img001.png")]


def test_load_dlc_splits_lexicographic_order(tmp_path):
    """Non-zero-padded filenames follow DLC's lexicographic order, with a warning."""
    # Lexicographically, "img10.png" < "img2.png", so position 0 is img10.
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img2", "img10"]},
        video_sets={str(tmp_path / "videos" / "vid1.mp4"): {}},
        train_indices=[0],
        test_indices=[1],
    )
    with pytest.warns(UserWarning, match="lexicographic"):
        splits = sio.load_dlc_splits(config_path)
    assert _frame_keys(splits["train"]) == [("vid1", "img10.png")]
    assert _frame_keys(splits["test"]) == [("vid1", "img2.png")]


def test_load_dlc_splits_missing_pickle(tmp_path):
    """A project without a Documentation pickle raises FileNotFoundError."""
    config_path = make_dlc_project(tmp_path)  # no train/test indices -> no pickle
    with pytest.raises(FileNotFoundError):
        sio.load_dlc_splits(config_path)


def test_load_dlc_splits_ambiguous_requires_selector(tmp_path):
    """Multiple shuffles require an explicit selector."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        train_indices=[0],
        test_indices=[1],
        shuffle=1,
    )
    # Add a second shuffle pickle.
    make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        train_indices=[1],
        test_indices=[0],
        shuffle=2,
    )
    with pytest.raises(ValueError, match="Multiple DLC splits"):
        sio.load_dlc_splits(config_path)
    # Selecting a shuffle disambiguates.
    splits = sio.load_dlc_splits(config_path, shuffle=2)
    assert _frame_keys(splits["train"]) == [("vid1", "img001.png")]


def test_load_dlc_splits_selector_no_match(tmp_path):
    """A selector that matches nothing raises FileNotFoundError."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        train_indices=[0],
        test_indices=[1],
        shuffle=1,
    )
    with pytest.raises(FileNotFoundError):
        sio.load_dlc_splits(config_path, shuffle=99)


def test_read_dlc_split_reads_indices(tmp_path):
    """_read_dlc_split returns train/test indices from meta[1]/meta[2]."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        train_indices=[0],
        test_indices=[1],
    )
    project_dir = Path(config_path).parent
    cfg = dlc._read_dlc_config(config_path)
    pkl = dlc._select_documentation_pickle(project_dir, cfg, None, None, None)
    train, test = dlc._read_dlc_split(pkl)
    assert train == [0]
    assert test == [1]


def test_dlc_merged_order_skips_scorer_mismatch(tmp_path):
    """Folders labeled by a different scorer are skipped in the merged order."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
    )
    # Add a second folder whose CSV scorer does not match the project scorer.
    make_dlc_project(
        tmp_path,
        folders={"vid2": ["img000"]},
        video_sets={
            str(tmp_path / "videos" / "vid1.mp4"): {},
            str(tmp_path / "videos" / "vid2.mp4"): {},
        },
        csv_scorer="OtherScorer",
    )
    cfg = dlc._read_dlc_config(config_path)
    with pytest.warns(UserWarning, match="different scorer|OtherScorer|scorer"):
        merged = dlc._dlc_merged_order(tmp_path, cfg)
    # Only vid1's two frames are included; vid2 (mismatched scorer) is skipped.
    assert ("vid2", "img000.png") not in merged
    assert ("vid1", "img000.png") in merged


def make_dlc_ma_project(
    root,
    *,
    scorer="LM",
    task="maproj",
    date="Jan1",
    folders=None,
    train_indices=None,
    test_indices=None,
):
    """Build a minimal synthetic multi-animal DLC project under ``root``."""
    root = Path(root)
    if folders is None:
        folders = {"m1": ["img000", "img001"], "m2": ["img000"]}
    individuals = ["Animal1", "Animal2"]
    bodyparts = ["A", "B"]

    scorer_cells, ind_cells, bp_cells, coord_cells = [], [], [], []
    for ind in individuals:
        for bp in bodyparts:
            for c in ["x", "y"]:
                scorer_cells.append(scorer)
                ind_cells.append(ind)
                bp_cells.append(bp)
                coord_cells.append(c)
    ncol = len(scorer_cells)
    rows = [
        "scorer," + ",".join(scorer_cells),
        "individuals," + ",".join(ind_cells),
        "bodyparts," + ",".join(bp_cells),
        "coords," + ",".join(coord_cells),
    ]

    for folder, imgs in folders.items():
        d = root / "labeled-data" / folder
        d.mkdir(parents=True, exist_ok=True)
        lines = list(rows)
        for i, img in enumerate(imgs):
            vals = ",".join(str(v) for v in range(i * 100, i * 100 + ncol))
            lines.append(f"labeled-data/{folder}/{img}.png,{vals}")
        (d / f"CollectedData_{scorer}.csv").write_text("\n".join(lines) + "\n")
        for img in imgs:
            (d / f"{img}.png").write_text("dummy")

    cfg = {
        "Task": task,
        "scorer": scorer,
        "date": date,
        "iteration": 0,
        "multianimalproject": True,
        "video_sets": {str(root / "videos" / f"{f}.mp4"): {} for f in folders},
        "individuals": individuals,
        "multianimalbodyparts": bodyparts,
        "bodyparts": "MULTI!",
        "skeleton": [["A", "B"]],
        "TrainingFraction": [0.8],
    }
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))

    if train_indices is not None and test_indices is not None:
        tdir = (
            root
            / "training-datasets"
            / "iteration-0"
            / f"UnaugmentedDataSet_{task}{date}"
        )
        tdir.mkdir(parents=True)
        with open(tdir / f"Documentation_data-{task}_80shuffle1.pickle", "wb") as f:
            pickle.dump(
                [[], list(train_indices), list(test_indices), 0.8],
                f,
                pickle.HIGHEST_PROTOCOL,
            )
    return root / "config.yaml"


def test_load_dlc_csv_shared_skeleton_default_tracks(tmp_path):
    """_load_dlc_csv with a shared skeleton but no tracks defaults tracks to []."""
    make_dlc_project(tmp_path, folders={"vid1": ["img000"]})
    skel = sio.Skeleton(nodes=["leftear", "rightear", "snout"])
    csv = tmp_path / "labeled-data" / "vid1" / "CollectedData_LM.csv"
    labels = dlc._load_dlc_csv(csv, skeleton=skel)
    assert labels.skeleton is skel
    assert labels.tracks == []


def test_load_dlc_project_multianimal(tmp_path):
    """A multi-animal project loads with shared skeleton, edges, and tracks."""
    config_path = make_dlc_ma_project(tmp_path)
    labels = sio.load_dlc_project(config_path)
    assert len(labels.skeletons) == 1
    assert set(labels.skeleton.node_names) == {"A", "B"}
    assert labels.skeleton.edge_names == [("A", "B")]
    assert {t.name for t in labels.tracks} == {"Animal1", "Animal2"}
    assert len(labels.videos) == 2


def test_load_dlc_splits_multianimal(tmp_path):
    """Splits work for a multi-animal project."""
    # Merged order: 0:(m1,img000) 1:(m1,img001) 2:(m2,img000)
    config_path = make_dlc_ma_project(tmp_path, train_indices=[0, 2], test_indices=[1])
    splits = sio.load_dlc_splits(config_path)
    assert _frame_keys(splits["train"]) == [("m1", "img000.png"), ("m2", "img000.png")]
    assert _frame_keys(splits["test"]) == [("m1", "img001.png")]


def test_load_dlc_splits_fallback_when_stems_mismatch(tmp_path):
    """When video_sets stems match no folder, merged order falls back to CSVs."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vidA": ["img000", "img001"]},
        video_sets={str(tmp_path / "videos" / "unrelated.mp4"): {}},
        train_indices=[0],
        test_indices=[1],
    )
    splits = sio.load_dlc_splits(config_path)
    assert _frame_keys(splits["train"]) == [("vidA", "img000.png")]
    assert _frame_keys(splits["test"]) == [("vidA", "img001.png")]


def test_load_dlc_splits_select_by_train_fraction(tmp_path):
    """A train_fraction selector filters the available pickles."""
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        train_indices=[0],
        test_indices=[1],
        train_fraction=0.8,
    )
    splits = sio.load_dlc_splits(config_path, train_fraction=0.8)
    assert _frame_keys(splits["train"]) == [("vid1", "img000.png")]


def test_select_documentation_pickle_unparsable_single(tmp_path):
    """A single pickle with an unparsable name is returned as-is."""
    config_path = make_dlc_project(tmp_path, folders={"vid1": ["img000"]})
    cfg = dlc._read_dlc_config(config_path)
    tdir = (
        tmp_path / "training-datasets" / "iteration-0" / "UnaugmentedDataSet_projJan1"
    )
    tdir.mkdir(parents=True)
    p = tdir / "Documentation_data-weirdname.pickle"
    with open(p, "wb") as f:
        pickle.dump([[], [0], [], 0.8], f)
    assert dlc._select_documentation_pickle(tmp_path, cfg, None, None, None) == p


def test_select_documentation_pickle_unparsable_ambiguous(tmp_path):
    """Multiple pickles with unparsable names raise a clear error."""
    config_path = make_dlc_project(tmp_path, folders={"vid1": ["img000"]})
    cfg = dlc._read_dlc_config(config_path)
    tdir = (
        tmp_path / "training-datasets" / "iteration-0" / "UnaugmentedDataSet_projJan1"
    )
    tdir.mkdir(parents=True)
    for name in ["Documentation_data-a.pickle", "Documentation_data-b.pickle"]:
        with open(tdir / name, "wb") as f:
            pickle.dump([[], [0], [], 0.8], f)
    with pytest.raises(ValueError, match="Could not parse"):
        dlc._select_documentation_pickle(tmp_path, cfg, None, None, None)


def test_find_project_csvs_fallback_nonstandard_name(tmp_path):
    """A DLC CSV not named CollectedData_<scorer> is found via glob fallback."""
    config_path = make_dlc_project(tmp_path, folders={"vid1": ["img000"]})
    d = tmp_path / "labeled-data" / "vid2"
    d.mkdir()
    (d / "img000.png").write_text("dummy")
    (d / "mydata.csv").write_text(
        "scorer,LM,LM,LM,LM,LM,LM\n"
        "bodyparts,snout,snout,leftear,leftear,rightear,rightear\n"
        "coords,x,y,x,y,x,y\n"
        "labeled-data/vid2/img000.png,0,1,2,3,4,5\n"
    )
    labels = sio.load_dlc_project(config_path)
    assert len(labels.videos) == 2


def test_find_project_csvs_skips_files_in_labeled_data(tmp_path):
    """Stray files directly under labeled-data/ are ignored."""
    config_path = make_dlc_project(tmp_path, folders={"vid1": ["img000"]})
    (tmp_path / "labeled-data" / "stray.txt").write_text("ignore me")
    labels = sio.load_dlc_project(config_path)
    assert len(labels.videos) == 1


def test_load_dlc_project_unreadable_config(tmp_path):
    """A config.yaml that is not a mapping raises a clear error."""
    (tmp_path / "labeled-data").mkdir()
    (tmp_path / "config.yaml").write_text("- a\n- b\n")
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="Could not read DLC config"):
            sio.load_dlc_project(tmp_path / "config.yaml")


def test_load_dlc_project_no_labeled_data_dir(tmp_path):
    """A project with no labeled-data directory raises a clear error."""
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"Task": "t", "scorer": "LM", "video_sets": {}, "bodyparts": []})
    )
    with pytest.raises(ValueError, match="No DLC annotation CSVs"):
        sio.load_dlc_project(tmp_path / "config.yaml")


def test_load_dlc_splits_unreadable_config(tmp_path):
    """load_dlc_splits raises a clear error when the config is not a mapping."""
    (tmp_path / "labeled-data").mkdir()
    (tmp_path / "config.yaml").write_text("- a\n- b\n")
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="Could not read DLC config"):
            sio.load_dlc_splits(tmp_path / "config.yaml")


def test_load_dlc_image_resolved_from_parent_dir(tmp_path):
    """Images one directory above the CSV are resolved (parent-path fallback)."""
    # CSV lives in a subdir; images live under <project>/labeled-data/session1/.
    (tmp_path / "labeled-data" / "session1").mkdir(parents=True)
    for i in range(2):
        (tmp_path / "labeled-data" / "session1" / f"img{i:03d}.png").write_text("x")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    csv = subdir / "data.csv"
    csv.write_text(
        "scorer,S,S,S,S,S,S\n"
        "bodyparts,A,A,B,B,C,C\n"
        "coords,x,y,x,y,x,y\n"
        "labeled-data/session1/img000.png,0,1,2,3,4,5\n"
        "labeled-data/session1/img001.png,6,7,8,9,10,11\n"
    )
    labels = sio.load_dlc(csv, config=False)
    assert len(labels.labeled_frames) == 2
    assert len(labels.videos) == 1


# -----------------------------------------------------------------------------
# Item 2: video_sets crop wiring (provenance + source_video views)
# -----------------------------------------------------------------------------

#: A real, openable video reused for the source-present from_crop path.
REAL_VIDEO = "tests/data/videos/small_robot_3_frame.mp4"


def test_parse_dlc_crop_string_reorders_to_sleap_rect():
    """A comma string 'x1, x2, y1, y2' reorders to sleap '(x1, y1, x2, y2)'."""
    # Non-square, non-zero-origin so a transpose bug is caught.
    assert dlc._parse_dlc_crop("10, 60, 20, 90") == (10, 20, 60, 90)


def test_parse_dlc_crop_list_form_reorders():
    """A list [x1, x2, y1, y2] reorders identically to the string form."""
    assert dlc._parse_dlc_crop([10, 60, 20, 90]) == (10, 20, 60, 90)


def test_parse_dlc_crop_float_strings_coerced_to_int():
    """Float-valued crop strings are coerced to int rect components."""
    assert dlc._parse_dlc_crop("10.0, 60.0, 20.0, 90.0") == (10, 20, 60, 90)


@pytest.mark.parametrize("crop", [None, "", "  ", "10, 60", "10, 60, 20", "a, b, c, d"])
def test_parse_dlc_crop_missing_or_bad_returns_none(crop):
    """Missing / empty / wrong-arity / unparsable crops return None."""
    assert dlc._parse_dlc_crop(crop) is None


def test_parse_dlc_crop_non_string_non_sequence_returns_none():
    """A crop value that is neither string nor sequence returns None."""
    assert dlc._parse_dlc_crop(42) is None


def test_parse_dlc_crop_inverted_warns_and_returns_none():
    """An inverted crop (x2 <= x1) warns and returns None."""
    with pytest.warns(UserWarning, match="inverted"):
        assert dlc._parse_dlc_crop("60, 10, 20, 90") is None


def test_parse_dlc_crop_identity_origin_returns_none():
    """An identity crop at origin (0, 0) is a no-op and returns None."""
    assert dlc._parse_dlc_crop("0, 384, 0, 384") is None


def test_video_sets_stem_map_carries_crop_rect():
    """The stem map returns (path, reordered_rect) for a non-square crop."""
    path = "/data/videos/vid1.mp4"
    cfg = {"video_sets": {path: {"crop": "10, 60, 20, 90"}}}
    stem_map = dlc._video_sets_stem_map(cfg)
    assert stem_map["vid1"] == (path, (10, 20, 60, 90))


def test_video_sets_stem_map_none_crop_when_absent():
    """The stem map returns (path, None) when no crop is configured."""
    path = "/data/videos/vid1.mp4"
    cfg = {"video_sets": {path: {}}}
    stem_map = dlc._video_sets_stem_map(cfg)
    assert stem_map["vid1"] == (path, None)


def test_dlc_crop_provenance_recorded_source_absent(tmp_path):
    """A non-square crop is recorded in provenance keyed by source path."""
    source = tmp_path / "videos" / "vid1.mp4"  # absent on disk
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        video_sets={str(source): {"crop": "10, 60, 20, 90"}},
    )
    labels = sio.load_dlc_project(config_path)
    assert labels.provenance["dlc_crops"] == {str(source): [10, 20, 60, 90]}


def test_dlc_crop_not_applied_to_labeled_video_or_points(tmp_path):
    """The labeled-data video stays uncropped and point coords are unchanged."""
    source = tmp_path / "videos" / "vid1.mp4"  # absent on disk
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        video_sets={str(source): {"crop": "10, 60, 20, 90"}},
    )
    labels = sio.load_dlc_project(config_path)
    video = labels.videos[0]
    # Crop is NOT on the labeled-data ImageVideo (would double-crop on reload).
    assert video.is_cropped is False
    # Points are verbatim cropped-space coords (no offset applied).
    # Row 0 writes range(0, 6) in declared bodypart order
    # (snout, leftear, rightear) -> snout=(0, 1), leftear=(2, 3), rightear=(4, 5).
    # Nodes are stored sorted (leftear, rightear, snout), so reorder accordingly.
    lf = next(lf for lf in labels.labeled_frames if lf.frame_idx == 0)
    pts = lf.instances[0].numpy()
    node_names = [n.name for n in labels.skeleton.nodes]
    declared = {"snout": [0, 1], "leftear": [2, 3], "rightear": [4, 5]}
    expected = np.array([declared[name] for name in node_names])
    np.testing.assert_array_equal(pts, expected)


def test_dlc_crop_source_absent_is_closed_video(tmp_path):
    """When the source mp4 is absent, source_video is a closed (uncropped) Video."""
    source = tmp_path / "videos" / "vid1.mp4"  # absent on disk
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000"]},
        video_sets={str(source): {"crop": "10, 60, 20, 90"}},
    )
    labels = sio.load_dlc_project(config_path)
    sv = labels.videos[0].source_video
    assert sv is not None
    # Closed source: no crop attached to backend_metadata (avoids save crash).
    assert sv.backend is None
    assert sv.is_cropped is False


def test_dlc_crop_source_present_but_unopenable_falls_back(tmp_path):
    """A source path that exists but cannot be opened falls back to a closed Video.

    Exercises the from_crop open-failure guard in _set_source_video: the rect is
    still recorded in provenance, but source_video is the plain closed Video.
    Uses an existing file with an unsupported extension so from_filename raises
    deterministically (avoids relying on a decoder rejecting a malformed video).
    """
    bad = tmp_path / "videos" / "vid1.bin"  # exists, but not a recognized video
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a real video at all")
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000"]},
        video_sets={str(bad): {"crop": "10, 60, 20, 90"}},
    )
    labels = sio.load_dlc_project(config_path)
    sv = labels.videos[0].source_video
    assert sv is not None
    assert sv.backend is None  # from_crop failed -> closed fallback
    assert sv.is_cropped is False
    # The persistent rect is still recorded despite the open failure.
    assert labels.provenance["dlc_crops"] == {str(bad): [10, 20, 60, 90]}


def test_dlc_crop_provenance_roundtrips_through_slp(tmp_path):
    """provenance['dlc_crops'] survives a save_slp + load_slp round-trip."""
    source = tmp_path / "videos" / "vid1.mp4"  # absent on disk
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        video_sets={str(source): {"crop": "10, 60, 20, 90"}},
    )
    labels = sio.load_dlc_project(config_path)
    slp_path = tmp_path / "out.slp"
    sio.save_slp(labels, slp_path)  # must not crash on a closed cropped source
    reloaded = sio.load_slp(slp_path)
    assert reloaded.provenance["dlc_crops"] == {str(source): [10, 20, 60, 90]}


def test_dlc_crop_source_present_from_crop_view(tmp_path):
    """With a real source mp4, source_video is a from_crop view (in-memory crop)."""
    # Folder stem must match the source video stem for the link to resolve.
    stem = Path(REAL_VIDEO).stem
    config_path = make_dlc_project(
        tmp_path,
        folders={stem: ["img000", "img001"]},
        video_sets={str(Path(REAL_VIDEO).resolve()): {"crop": "10, 60, 20, 90"}},
    )
    labels = sio.load_dlc_project(config_path)
    sv = labels.videos[0].source_video
    assert sv.is_cropped is True
    assert sv.crop_rect == (10, 20, 60, 90)
    # to_source_coords adds the crop origin (10, 20).
    pts = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(sv.to_source_coords(pts), [[11.0, 22.0]])
    # The labeled-data video itself stays uncropped.
    assert labels.videos[0].is_cropped is False
    # Provenance still records the persistent rect.
    src_key = str(Path(REAL_VIDEO).resolve())
    assert labels.provenance["dlc_crops"][src_key] == [10, 20, 60, 90]


def test_dlc_crop_source_present_save_slp_strips_inmemory_crop(tmp_path):
    """A from_crop source view does not crash save_slp; provenance persists."""
    stem = Path(REAL_VIDEO).stem
    config_path = make_dlc_project(
        tmp_path,
        folders={stem: ["img000", "img001"]},
        video_sets={str(Path(REAL_VIDEO).resolve()): {"crop": "10, 60, 20, 90"}},
    )
    labels = sio.load_dlc_project(config_path)
    slp_path = tmp_path / "out.slp"
    sio.save_slp(labels, slp_path)
    reloaded = sio.load_slp(slp_path)
    src_key = str(Path(REAL_VIDEO).resolve())
    assert reloaded.provenance["dlc_crops"][src_key] == [10, 20, 60, 90]


def test_dlc_crop_per_video_independent(tmp_path):
    """Distinct crops per video are recorded independently in provenance."""
    src1 = tmp_path / "videos" / "vid1.mp4"
    src2 = tmp_path / "videos" / "vid2.mp4"
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000"], "vid2": ["img000"]},
        video_sets={
            str(src1): {"crop": "10, 60, 20, 90"},
            str(src2): {"crop": "5, 25, 7, 47"},
        },
    )
    labels = sio.load_dlc_project(config_path)
    assert labels.provenance["dlc_crops"] == {
        str(src1): [10, 20, 60, 90],
        str(src2): [5, 7, 25, 47],
    }


def test_dlc_crop_single_csv_load_records_provenance(tmp_path):
    """load_dlc (single CSV + config) also records dlc_crops provenance."""
    source = tmp_path / "videos" / "vid1.mp4"
    config_path = make_dlc_project(
        tmp_path,
        folders={"vid1": ["img000", "img001"]},
        video_sets={str(source): {"crop": "10, 60, 20, 90"}},
    )
    csv = tmp_path / "labeled-data" / "vid1" / "CollectedData_LM.csv"
    labels = sio.load_dlc(csv, config=config_path)
    assert labels.provenance["dlc_crops"] == {str(source): [10, 20, 60, 90]}


def test_dlc_identity_crop_records_no_provenance(tmp_path):
    """The default identity crop (origin 0, 0) records no dlc_crops entry."""
    config_path = make_dlc_project(tmp_path)  # default crop "0, 100, 0, 100"
    labels = sio.load_dlc_project(config_path)
    assert "dlc_crops" not in labels.provenance


def test_dlc_madlc_230_config_no_crop_provenance(dlc_maudlc_testdata, dlc_config):
    """The madlc_230 fixture's no-op 0,384,0,384 crop yields no provenance."""
    labels = sio.load_dlc(dlc_maudlc_testdata, config=dlc_config)
    assert "dlc_crops" not in labels.provenance
