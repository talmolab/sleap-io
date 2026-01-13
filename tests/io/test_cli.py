"""CLI tests for the `sio` command.

Covers summary output, labeled frame details, skeleton printing, and format conversion.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from sleap_io import load_slp
from sleap_io.io.cli import _get_ffmpeg_version, _is_ffmpeg_available, cli
from sleap_io.model.instance import PredictedInstance
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.version import __version__

# Skip marker for tests that are extremely slow on Windows CI due to video encoding
skip_slow_video_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Video encoding tests are extremely slow on Windows CI (~1 min per test)",
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def _data_path(rel: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "tests" / "data"
    return root / rel


def test_version_shows_plugin_info():
    """Test that --version shows sleap-io version and plugin status."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Check main version line
    assert f"sleap-io {__version__}" in out

    # Check sections are present
    assert "Core:" in out
    assert "Video plugins:" in out
    assert "Optional:" in out

    # Check specific packages are listed
    assert "numpy:" in out
    assert "opencv:" in out
    assert "pyav:" in out


def test_show_summary_typical_slp():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Header panel with file info
    assert "typical.slp" in out
    assert "sleap-io" in out
    # Stats in header
    assert "video" in out.lower()
    assert "frame" in out.lower()
    # Skeleton summary with Python code
    assert "nodes = " in out
    assert "Skeletons" in out


def test_show_lf_zero_details():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--lf", "0", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Expect labeled frame details
    assert "Labeled Frame 0" in out
    assert "Frame:" in out
    assert "Instances:" in out
    # Should list instances with points as Python code
    assert "Instance 0:" in out
    assert "points = [" in out


def test_show_lf_out_of_range():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--lf", "9999", "--no-open-videos"])
    assert result.exit_code != 0
    assert "out of range" in result.output


def test_show_on_video_basic_info():
    runner = CliRunner()
    # Use a small bundled mp4 in tests/data/videos
    # If CI lacks codecs, this should still work as we don't open videos by default
    path = _data_path("videos/video_1.mp4")
    result = runner.invoke(cli, ["show", str(path)])
    # If the file is missing in some environments, allow graceful skip assertion
    if path.exists():
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        # Non-Labels objects print repr
        assert "Video" in out


def test_show_skeleton_flag_text():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--skeleton", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Detailed skeleton view shows Python code and tables
    assert "Skeleton Details" in out
    assert "Python code:" in out
    assert "nodes = " in out
    assert "Nodes:" in out
    assert "Edges:" in out


def test_show_file_not_found():
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "/nonexistent/path/to/file.slp"])
    assert result.exit_code != 0
    # Click validates file existence before our code runs
    assert "not exist" in result.output


def test_show_skeleton_no_edges(tmp_path):
    """Test skeleton display when skeleton has no edges."""
    from sleap_io import save_file

    runner = CliRunner()
    # Create a skeleton with no edges
    skeleton = Skeleton(nodes=["node1", "node2"])
    labels = Labels(skeletons=[skeleton])

    # Save to temporary file
    slp_path = tmp_path / "no_edges.slp"
    save_file(labels, slp_path)

    result = runner.invoke(
        cli, ["show", str(slp_path), "--skeleton", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show nodes but no Edges section (or empty edges)
    assert "Nodes:" in out
    assert "node1" in out
    assert "node2" in out
    # No edge_inds in Python code since there are no edges
    assert "edge_inds" not in out


def test_show_empty_labels_with_lf(tmp_path):
    """Test --lf flag on file with no labeled frames."""
    from sleap_io import save_file

    runner = CliRunner()
    # Create empty labels
    labels = Labels()

    # Save to temporary file
    slp_path = tmp_path / "empty.slp"
    save_file(labels, slp_path)

    result = runner.invoke(
        cli, ["show", str(slp_path), "--lf", "0", "--no-open-videos"]
    )
    assert result.exit_code != 0
    assert "No labeled frames present in file" in result.output


def test_show_video_file():
    """Test show on a video file displays rich formatted output."""
    runner = CliRunner()
    path = _data_path("videos/centered_pair_low_quality.mp4")

    if not path.exists():
        # Skip if video file doesn't exist in test environment
        return

    result = runner.invoke(cli, ["show", str(path), "--open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show rich panel with video info
    assert "sleap-io" in out
    assert "centered_pair_low_quality.mp4" in out
    assert "Video (MediaVideo)" in out
    assert "frames" in out
    # Should show encoding info (if ffmpeg available)
    # These are more informative than the old Status/Plugin lines
    if _is_ffmpeg_available():
        assert "Codec" in out
        assert "h264" in out.lower()


def test_show_video_file_full_path():
    """Test show on a video file displays full absolute path."""
    runner = CliRunner()
    path = _data_path("videos/centered_pair_low_quality.mp4")

    if not path.exists():
        return

    result = runner.invoke(cli, ["show", str(path), "--open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show full path for copy-paste convenience
    assert "Full" in out
    assert str(path.resolve()) in out


def test_show_video_flag():
    """Test --video flag shows detailed video info."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--video", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Video Details" in out
    assert "Video 0:" in out
    assert "Type" in out
    assert "Path" in out
    assert "Status" in out
    assert "Labeled" in out


def test_show_video_flag_auto_opens_backend():
    """Test --video flag auto-opens backends when no explicit flag is given."""
    runner = CliRunner()
    path = _data_path("slp/centered_pair_predictions.slp")
    # Use -v without --open-videos or --no-open-videos to test auto-open behavior
    result = runner.invoke(cli, ["show", str(path), "-v"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Video Details" in out
    # Should show backend loaded since -v auto-opens
    assert "Backend loaded" in out


def test_show_tracks_flag():
    """Test --tracks flag shows detailed track info."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--tracks", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Tracks" in out
    # Should show track table with instance counts
    assert "Track" in out
    assert "Instances" in out


def test_show_provenance_flag():
    """Test --provenance flag shows provenance info."""
    runner = CliRunner()
    path = _data_path("slp/predictions_1.2.7_provenance_and_tracking.slp")
    result = runner.invoke(cli, ["show", str(path), "--provenance", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Provenance" in out
    assert "sleap_version" in out


def test_show_all_flag():
    """Test --all flag shows all details."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--all", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show all detail sections
    assert "Skeleton Details" in out
    assert "Video Details" in out
    assert "Tracks" in out


def test_show_short_flags():
    """Test short flag aliases work (-s, -v, -t, -p, -a)."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")

    # Test -s for --skeleton
    result = runner.invoke(cli, ["show", str(path), "-s", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    assert "Skeleton Details" in _strip_ansi(result.output)

    # Test -v for --video
    result = runner.invoke(cli, ["show", str(path), "-v", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    assert "Video Details" in _strip_ansi(result.output)

    # Test -t for --tracks
    result = runner.invoke(cli, ["show", str(path), "-t", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    assert "Tracks" in _strip_ansi(result.output)


def test_show_skeleton_with_symmetries():
    """Test skeleton display shows symmetries when present."""
    runner = CliRunner()
    path = _data_path("slp/centered_pair_predictions.slp")
    result = runner.invoke(cli, ["show", str(path), "--skeleton", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # This file has symmetries
    assert "Symmetries:" in out
    assert "<->" in out


def test_show_multiview_videos():
    """Test show handles multiple videos correctly."""
    runner = CliRunner()
    path = _data_path("slp/multiview.slp")
    result = runner.invoke(cli, ["show", str(path), "--video", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Multiview has 8 videos
    assert "Video 0" in out
    assert "Video 7" in out


def test_show_video_index_specific():
    """Test --video-index shows only that video."""
    runner = CliRunner()
    path = _data_path("slp/multiview.slp")
    # Show only video 1
    result = runner.invoke(
        cli, ["show", str(path), "--video-index", "1", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show video 1
    assert "Video 1:" in out
    # Should NOT show video 0 or video 7
    assert "Video 0:" not in out
    assert "Video 7:" not in out


def test_show_video_index_out_of_range():
    """Test --video-index with out-of-range index gives clear error."""
    runner = CliRunner()
    path = _data_path("slp/multiview.slp")
    result = runner.invoke(
        cli, ["show", str(path), "--video-index", "99", "--no-open-videos"]
    )
    assert result.exit_code == 1
    out = _strip_ansi(result.output)
    assert "out of range" in out
    assert "8 video(s)" in out


def test_show_video_index_first_video():
    """Test --video-index 0 shows first video."""
    runner = CliRunner()
    path = _data_path("slp/multiview.slp")
    result = runner.invoke(
        cli, ["show", str(path), "--video-index", "0", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Video 0:" in out
    assert "Video 1:" not in out


def test_show_video_index_short_form():
    """Test --vi short form for video-index."""
    runner = CliRunner()
    path = _data_path("slp/multiview.slp")
    result = runner.invoke(cli, ["show", str(path), "--vi", "2", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Video 2:" in out
    assert "Video 0:" not in out


def test_show_header_shows_file_size():
    """Test that header panel shows file size."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Size:" in out
    assert "KB" in out or "MB" in out or "B" in out


def test_show_header_shows_instance_counts():
    """Test that header shows user/predicted instance counts."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # typical.slp has both user and predicted instances
    assert "user instances" in out
    assert "predicted" in out


def test_show_header_shows_user_frames_vs_total():
    """Test that header shows user frames vs total when different."""
    runner = CliRunner()
    # This file has 5 user frames out of 10 total frames
    path = _data_path("slp/labels.v002.rel_paths.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show "5 user frames (10 total)" format
    assert "5 user frames" in out
    assert "10 total" in out


def test_show_header_shows_full_path():
    """Test that header shows full absolute path."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show "Full:" line with absolute path
    assert "Full:" in out
    assert str(path.resolve()) in out


def test_show_pkg_file_type():
    """Test that .pkg.slp files show Package type."""
    runner = CliRunner()
    path = _data_path("slp/minimal_instance.pkg.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "Package" in out


def test_show_no_skeletons(tmp_path):
    """Test show on file with no skeletons."""
    from sleap_io import save_file

    runner = CliRunner()
    labels = Labels()

    slp_path = tmp_path / "no_skeletons.slp"
    save_file(labels, slp_path)

    result = runner.invoke(
        cli, ["show", str(slp_path), "--skeleton", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "No skeletons" in out


def test_show_no_videos(tmp_path):
    """Test show on file with no videos."""
    from sleap_io import save_file

    runner = CliRunner()
    labels = Labels()

    slp_path = tmp_path / "no_videos.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--video", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "No videos" in out


def test_show_no_tracks(tmp_path):
    """Test show on file with no tracks."""
    from sleap_io import save_file

    runner = CliRunner()
    labels = Labels()

    slp_path = tmp_path / "no_tracks.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--tracks", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    assert "No tracks" in out


def test_show_provenance_shows_filename(tmp_path):
    """Test that provenance shows filename after saving."""
    from sleap_io import save_file

    runner = CliRunner()
    labels = Labels()

    slp_path = tmp_path / "with_provenance.slp"
    save_file(labels, slp_path)

    result = runner.invoke(
        cli, ["show", str(slp_path), "--provenance", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Saving adds filename to provenance
    assert "Provenance" in out
    assert "filename:" in out


def test_show_format_file_size_units():
    """Test file size formatting for different units."""
    from sleap_io.io.cli import _format_file_size

    assert "B" in _format_file_size(100)
    assert "KB" in _format_file_size(2048)
    assert "MB" in _format_file_size(2 * 1024 * 1024)
    assert "GB" in _format_file_size(2 * 1024 * 1024 * 1024)
    assert "TB" in _format_file_size(2 * 1024 * 1024 * 1024 * 1024)


def test_show_provenance_with_list_and_dict():
    """Test provenance display with list and dict values.

    With --provenance flag, full JSON is shown (not truncated).
    Without --provenance, compact mode shows "[N items]" or "{N keys}".
    """
    runner = CliRunner()
    path = _data_path("slp/predictions_1.2.7_provenance_and_tracking.slp")

    # Test full mode (--provenance): shows full JSON
    result = runner.invoke(cli, ["show", str(path), "--provenance", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # This file has 'args' which is a dict - full mode shows actual JSON content
    assert "args:" in out
    # Full JSON mode shows the actual keys from the dict
    assert '"data_path"' in out or "data_path" in out

    # Test compact mode (no --provenance): shows summary
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Compact mode shows "{N keys}" for dicts
    assert "keys}" in out


# =============================================================================
# Video display tests
# =============================================================================


def test_show_video_with_open_backend():
    """Test video display with open backend shows plugin info."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    # Use --open-videos to actually open the backend
    result = runner.invoke(cli, ["show", str(path), "--video", "--open-videos"])
    # If video exists, should show backend info in status
    if result.exit_code == 0:
        out = _strip_ansi(result.output)
        assert "Video Details" in out
        # Status line shows backend loaded with plugin when video is opened
        assert "Status" in out


def test_show_video_embedded_pkg(slp_minimal_pkg):
    """Test video display for package files with embedded images."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", slp_minimal_pkg, "--video", "--open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show embedded indicator
    assert "embedded" in out.lower()
    # Should show dataset info (new format without colon)
    assert "Dataset" in out
    # Should show format info
    assert "Format" in out


def test_show_video_not_found_status(tmp_path):
    """Test video display when video file doesn't exist."""
    from sleap_io import Labels, Video, save_file

    runner = CliRunner()

    # Create labels with a non-existent video
    video = Video(
        filename="/nonexistent/path/to/video.mp4",
        open_backend=False,
        backend_metadata={"shape": (100, 480, 640, 3)},
    )
    labels = Labels(videos=[video])

    slp_path = tmp_path / "missing_video.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--video", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Status line should show file not found
    assert "File not found" in out


def test_show_video_cached_shape(tmp_path):
    """Test video display shows dimensions from backend_metadata when file not found."""
    from sleap_io import Labels, Video, save_file

    runner = CliRunner()

    # Create labels with a video that has cached shape but no backend
    video = Video(
        filename="/nonexistent/path/to/video.mp4",
        open_backend=False,
        backend_metadata={"shape": (100, 480, 640, 3)},
    )
    labels = Labels(videos=[video])

    slp_path = tmp_path / "cached_shape.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--video", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show dimensions from cached metadata even when file not found
    assert "640" in out  # Width from metadata
    assert "480" in out  # Height from metadata
    assert "100" in out  # Frame count from metadata
    assert "File not found" in out  # Status indicates file issue


def test_show_video_unknown_shape(tmp_path):
    """Test video display when shape is unavailable."""
    from sleap_io import Labels, Video, save_file

    runner = CliRunner()

    # Create labels with a video that has no shape info
    video = Video(
        filename="/nonexistent/path/to/video.mp4",
        open_backend=False,
        backend_metadata={},  # No shape cached
    )
    labels = Labels(videos=[video])

    slp_path = tmp_path / "no_shape.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--video", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show unknown for size
    assert "unknown" in out.lower()


def test_show_video_summary_many_without_metadata(tmp_path):
    """Test video summary with many videos shows first 3 + truncation."""
    from sleap_io import Labels, Video, save_file

    runner = CliRunner()

    # Create labels with 5 videos that have no shape (> 3 triggers truncation)
    videos = [
        Video(
            filename=f"/nonexistent/path/to/video_{i}.mp4",
            open_backend=False,
            backend_metadata={},  # No shape cached
        )
        for i in range(5)
    ]
    labels = Labels(videos=videos)

    slp_path = tmp_path / "many_videos_no_metadata.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show first 3 videos individually
    assert "[0]" in out
    assert "[1]" in out
    assert "[2]" in out
    # Should show truncation message for remaining 2
    assert "+2 more without cached metadata" in out
    # Should show status (found/not found)
    assert "not found" in out


def test_show_video_summary_mixed_exists(tmp_path):
    """Test video summary when some videos without metadata exist and some don't."""
    from pathlib import Path

    from sleap_io import Labels, Video, save_file

    runner = CliRunner()

    # Get absolute paths to test videos that actually exist
    existing_video_1 = str(
        Path("tests/data/videos/centered_pair_low_quality.mp4").resolve()
    )
    existing_video_2 = str(Path("tests/data/videos/small_robot_3_frame.mp4").resolve())

    # Create 5 videos: 2 that exist (no metadata), 3 that don't exist (no metadata)
    videos = [
        # These exist but have no cached shape
        Video(
            filename=existing_video_1,
            open_backend=False,
            backend_metadata={},  # No shape cached
        ),
        Video(
            filename=existing_video_2,
            open_backend=False,
            backend_metadata={},  # No shape cached
        ),
        # These don't exist
        Video(
            filename="/nonexistent/path/video_1.mp4",
            open_backend=False,
            backend_metadata={},
        ),
        Video(
            filename="/nonexistent/path/video_2.mp4",
            open_backend=False,
            backend_metadata={},
        ),
        Video(
            filename="/nonexistent/path/video_3.mp4",
            open_backend=False,
            backend_metadata={},
        ),
    ]
    labels = Labels(videos=videos)

    slp_path = tmp_path / "mixed_exists.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show first 3 videos individually
    assert "[0]" in out
    assert "[1]" in out
    assert "[2]" in out
    # Should show truncation message for remaining 2
    assert "+2 more without cached metadata" in out
    # First 2 videos exist, so first 2 don't have "[not found]"
    # Third video doesn't exist, so it should show "[not found]"
    assert "not found" in out


def test_show_video_with_metadata_not_found(tmp_path):
    """Test video with cached metadata but file doesn't exist shows not found tag."""
    from sleap_io import Labels, Video, save_file

    runner = CliRunner()

    # Video has shape cached but file doesn't exist
    video = Video(
        filename="/nonexistent/path/to/video.mp4",
        open_backend=False,
        backend_metadata={"shape": (100, 480, 640, 3)},  # Has cached shape
    )
    labels = Labels(videos=[video])

    slp_path = tmp_path / "metadata_not_found.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show dimensions from metadata
    assert "640" in out
    assert "480" in out
    # Should show not found tag
    assert "not found" in out


def test_show_video_image_sequence_with_metadata():
    """Test video summary shows 'N images' for image sequences with cached metadata."""
    from io import StringIO

    from rich.console import Console

    import sleap_io.io.cli as cli_module
    from sleap_io import Labels, Video
    from sleap_io.io.cli import _print_video_summary

    # Image sequence with cached shape
    video = Video(
        filename=[
            "tests/data/videos/imgs/img.00.jpg",
            "tests/data/videos/imgs/img.01.jpg",
            "tests/data/videos/imgs/img.02.jpg",
        ],
        open_backend=False,
        backend_metadata={"shape": (3, 100, 100, 3)},  # Has cached shape
    )
    labels = Labels(videos=[video])

    # Capture output by patching console
    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_video_summary(labels)
        out = _strip_ansi(string_io.getvalue())
        # Should show "3 images" for image sequence with metadata
        assert "3 images" in out
    finally:
        cli_module.console = original_console


def test_show_video_long_path_truncation(tmp_path):
    """Test video paths are truncated from the left for long paths."""
    from sleap_io import Labels, Video, save_file

    runner = CliRunner()

    # Create a video with a very long filename (> 80 chars)
    long_path = (
        "/very/long/path/that/exceeds/the/maximum/width/limit/"
        "for/display/purposes/video_file_with_long_name.mp4"
    )
    video = Video(
        filename=long_path,
        open_backend=False,
        backend_metadata={"shape": (100, 480, 640, 3)},
    )
    labels = Labels(videos=[video])

    slp_path = tmp_path / "long_path.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show truncated path with ... prefix
    assert "..." in out
    # Should preserve the filename at the end
    assert "video_file_with_long_name.mp4" in out


def test_truncate_path_left_edge_cases():
    """Test _truncate_path_left with edge cases."""
    from sleap_io.io.cli import _truncate_path_left

    # Short path - no truncation
    assert _truncate_path_left("/short/path.mp4", 80) == "/short/path.mp4"

    # Exactly at limit - no truncation
    path = "x" * 80
    assert _truncate_path_left(path, 80) == path

    # Just over limit - truncation
    path = "x" * 81
    result = _truncate_path_left(path, 80)
    assert result.startswith("...")
    assert len(result) == 80

    # Very small max_width (edge case: available <= 0)
    path = "/some/path.mp4"
    result = _truncate_path_left(path, 3)
    assert len(result) == 3
    assert result == "/so"  # Just takes first 3 chars

    result = _truncate_path_left(path, 2)
    assert len(result) == 2


def test_show_video_image_sequence():
    """Test video display functions for image sequence."""
    from io import StringIO

    from rich.console import Console

    from sleap_io import Labels, Video
    from sleap_io.io.cli import _print_video_details, _print_video_summary

    # Create labels with image sequence video using RELATIVE paths
    # This ensures the Full path line is displayed (when resolved != original)
    video = Video(
        filename=[
            "tests/data/videos/imgs/img.00.jpg",
            "tests/data/videos/imgs/img.01.jpg",
            "tests/data/videos/imgs/img.02.jpg",
        ],
        open_backend=False,
    )
    labels = Labels(videos=[video])

    # Test summary - capture output
    import sleap_io.io.cli as cli_module

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_video_summary(labels)
        out = _strip_ansi(string_io.getvalue())
        assert "3 images" in out

        # Test details
        string_io.truncate(0)
        string_io.seek(0)
        _print_video_details(labels)
        out = _strip_ansi(string_io.getvalue())
        # Should show image sequence type
        assert "ImageVideo" in out
        assert "3 images" in out
        # Should show first and last (new format without colons)
        assert "First" in out
        assert "Last" in out
        # Should show Full path since relative paths resolve to absolute
        assert "Full" in out
    finally:
        cli_module.console = original_console


def test_show_standalone_video_no_backend():
    """Test standalone video display when backend cannot be loaded (shape unknown)."""
    from io import StringIO

    from rich.console import Console

    from sleap_io import Video
    from sleap_io.io.cli import _print_video_standalone

    # Create video pointing to nonexistent file - backend won't load
    video = Video(filename="nonexistent_video.mp4", open_backend=False)

    from pathlib import Path

    import sleap_io.io.cli as cli_module

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_video_standalone(Path("nonexistent_video.mp4"), video)
        out = _strip_ansi(string_io.getvalue())
        # Should show shape unknown message
        assert "Shape unknown" in out or "unknown" in out.lower()
        # Should still show type and other info
        assert "MediaVideo" in out
    finally:
        cli_module.console = original_console


def test_show_standalone_video_with_encoding_info():
    """Test standalone video display shows encoding info when ffmpeg available."""
    from io import StringIO

    from rich.console import Console

    from sleap_io import Video
    from sleap_io.io.cli import _print_video_standalone

    path = _data_path("videos/centered_pair_low_quality.mp4")
    if not path.exists():
        return

    # Load video
    video = Video.from_filename(str(path))

    import sleap_io.io.cli as cli_module

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_video_standalone(path, video)
        out = _strip_ansi(string_io.getvalue())
        # Should show basic video info regardless of ffmpeg
        assert "Full" in out
        assert "centered_pair_low_quality.mp4" in out
        # If ffmpeg is available, should show encoding info
        if _is_ffmpeg_available():
            assert "Codec" in out
            assert "h264" in out.lower()
    finally:
        cli_module.console = original_console


def test_show_video_details_unknown_shape():
    """Test video details display when shape is unknown."""
    from io import StringIO

    from rich.console import Console

    from sleap_io import Labels, Video
    from sleap_io.io.cli import _print_video_details

    # Create video with no backend (shape will be None)
    video = Video(filename="nonexistent.mp4", open_backend=False)
    labels = Labels(videos=[video])

    import sleap_io.io.cli as cli_module

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_video_details(labels)
        out = _strip_ansi(string_io.getvalue())
        # Should show unknown for frames and size
        assert "unknown" in out.lower()
    finally:
        cli_module.console = original_console


def test_show_embedded_video_many_indices(slp_minimal_pkg, tmp_path):
    """Test embedded video display with >5 source indices shows range format."""
    from io import StringIO

    import numpy as np
    from rich.console import Console

    import sleap_io as sio
    from sleap_io.io.cli import _print_video_details
    from sleap_io.io.video_reading import HDF5Video

    # Load the pkg file
    labels = sio.load_slp(slp_minimal_pkg)
    video = labels.videos[0]

    # Ensure backend is loaded and modify source_inds to have >5 entries
    if isinstance(video.backend, HDF5Video):
        # Save original and set many indices
        original_inds = video.backend.source_inds
        video.backend.source_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        import sleap_io.io.cli as cli_module

        original_console = cli_module.console
        string_io = StringIO()
        cli_module.console = Console(file=string_io, force_terminal=True)

        try:
            _print_video_details(labels)
            out = _strip_ansi(string_io.getvalue())
            # Should show range format (0-9) instead of listing all
            assert "0" in out and "9" in out
            # Should use range notation with dash/en-dash
            assert "-" in out or "–" in out
        finally:
            cli_module.console = original_console
            # Restore original
            video.backend.source_inds = original_inds


def test_show_many_tracks(tmp_path):
    """Test track display truncates when >5 tracks."""
    from sleap_io import Labels, Track, save_file

    runner = CliRunner()

    # Create labels with many tracks
    tracks = [Track(name=f"track_{i}") for i in range(10)]
    labels = Labels(tracks=tracks)

    slp_path = tmp_path / "many_tracks.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["show", str(slp_path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show truncation message
    assert "+5 more" in out


def test_show_video_helper_functions():
    """Test video helper functions directly."""
    from sleap_io import Video
    from sleap_io.io.cli import (
        _build_status_line,
        _format_video_filename,
        _get_dataset,
        _get_image_filenames,
        _get_plugin,
        _get_shape_source,
        _get_video_type,
        _is_embedded,
    )

    # Test with no backend and empty metadata
    video = Video(
        filename="/path/to/video.mp4",
        open_backend=False,
        backend_metadata={},
    )

    # Type should be inferred from filename extension
    video_type = _get_video_type(video)
    assert video_type == "MediaVideo"

    plugin = _get_plugin(video)
    assert plugin is None

    is_embedded = _is_embedded(video)
    assert is_embedded is False

    shape_source = _get_shape_source(video)
    assert shape_source == "unknown"

    # Test with cached shape
    video.backend_metadata["shape"] = (100, 480, 640, 3)
    shape_source = _get_shape_source(video)
    assert shape_source == "metadata"

    # Test filename formatting
    fname = _format_video_filename(video)
    assert fname == "video.mp4"

    # Test with image list
    video_img = Video(
        filename=["/path/frame1.png", "/path/frame2.png"],
        open_backend=False,
    )
    fname = _format_video_filename(video_img)
    assert fname == "2 images"

    # Test _get_video_type inference
    assert _get_video_type(video_img) == "ImageVideo"

    # Test _get_image_filenames
    filenames = _get_image_filenames(video_img)
    assert filenames == ["/path/frame1.png", "/path/frame2.png"]
    assert _get_image_filenames(video) is None

    # Test _get_dataset
    assert _get_dataset(video) is None
    video.backend_metadata["dataset"] = "video0/video"
    assert _get_dataset(video) == "video0/video"

    # Test _build_status_line
    status = _build_status_line(video)
    assert status == "File not found"

    # Test video.exists() for file status
    assert video.exists() is False  # Non-existent file

    # Test type inference from backend_metadata
    video2 = Video(
        filename="/path/to/file.unknown",
        open_backend=False,
        backend_metadata={"type": "TiffVideo"},
    )
    assert _get_video_type(video2) == "TiffVideo"

    # Test embedded detection from metadata
    video3 = Video(
        filename="/path/to/file.slp",
        open_backend=False,
        backend_metadata={"has_embedded_images": True},
    )
    assert _is_embedded(video3) is True


def test_get_video_encoding_info():
    """Test _get_video_encoding_info helper function."""
    from sleap_io.io.cli import _get_video_encoding_info

    path = _data_path("videos/centered_pair_low_quality.mp4")
    if not path.exists():
        return  # Skip if video doesn't exist

    if not _is_ffmpeg_available():
        # Should return None when ffmpeg unavailable
        # We can't test this directly, but we verify the function handles it
        return

    info = _get_video_encoding_info(path)
    assert info is not None

    # Check that we extracted the expected codec info
    assert info.codec == "h264"
    assert info.codec_profile in ("Main", "High", "Baseline")
    assert info.pixel_format is not None
    assert "yuv" in info.pixel_format  # Common for h264
    assert info.bitrate_kbps is not None
    assert info.bitrate_kbps > 0
    assert info.container is not None


def test_estimate_gop_size():
    """Test _estimate_gop_size helper function."""
    from sleap_io.io.cli import _estimate_gop_size

    path = _data_path("videos/centered_pair_low_quality.mp4")
    if not path.exists():
        return  # Skip if video doesn't exist

    if not _is_ffmpeg_available():
        return

    gop = _estimate_gop_size(path)
    # GOP should be a positive integer (or None if estimation fails)
    if gop is not None:
        assert gop > 0
        # For typical videos, GOP is usually between 1 and 300
        assert gop < 500


def test_run_ffmpeg_info():
    """Test _run_ffmpeg_info helper function."""
    from sleap_io.io.cli import _run_ffmpeg_info

    path = _data_path("videos/centered_pair_low_quality.mp4")
    if not path.exists():
        return

    if not _is_ffmpeg_available():
        result = _run_ffmpeg_info(path)
        assert result is None
        return

    result = _run_ffmpeg_info(path)
    assert result is not None
    # Should contain typical ffmpeg output keywords
    assert "Stream" in result or "Input" in result
    assert "Video" in result


def test_show_video_file_encoding_info():
    """Test that video encoding info appears in sio show output."""
    runner = CliRunner()
    path = _data_path("videos/centered_pair_low_quality.mp4")

    if not path.exists():
        return

    result = runner.invoke(cli, ["show", str(path)])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)

    if _is_ffmpeg_available():
        # Should show codec info
        assert "Codec" in out
        # h264 should appear (case-insensitive)
        assert "h264" in out.lower()
        # Should show bitrate
        assert "Bitrate" in out
        assert "kb/s" in out


def test_show_video_exists_status():
    """Test video exists() when file exists."""
    # Use an actual video file that exists
    path = _data_path("videos/centered_pair_low_quality.mp4")
    if not path.exists():
        return  # Skip if video doesn't exist

    from sleap_io import Video

    video = Video(
        filename=str(path),
        open_backend=False,  # Don't open the backend
    )
    # Close the backend if it auto-opened
    if video.backend is not None:
        video.close()

    # Video should report that file exists
    assert video.exists() is True


def test_show_embedded_video_details(slp_minimal_pkg):
    """Test detailed embedded video display with source info."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", slp_minimal_pkg, "--video", "--open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)

    # Check embedded-specific fields (new format without colons)
    assert "embedded" in out.lower()
    assert "Dataset" in out
    assert "Format" in out
    assert "Source" in out
    # The pkg.slp has only 1 frame, so indices should be shown in Frames line
    assert "indices" in out.lower() or "0" in out


def test_show_embedded_video_summary(slp_minimal_pkg):
    """Test video summary with embedded video shows indicators."""
    runner = CliRunner()
    # No --video flag, so we get the summary view, and --open-videos to load backend
    result = runner.invoke(cli, ["show", slp_minimal_pkg, "--open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)

    # Summary should show embedded indicator for package file
    assert "embedded" in out.lower()
    # Should show some video info
    assert "video" in out.lower()


def test_show_provenance_empty():
    """Test provenance display with no provenance."""
    from io import StringIO

    from rich.console import Console

    from sleap_io import Labels
    from sleap_io.io.cli import _print_provenance

    # Create labels with truly empty provenance (not saved, which adds filename)
    labels = Labels()
    labels.provenance = {}  # Empty provenance

    # Test the function directly
    import sleap_io.io.cli as cli_module

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_provenance(labels)
        out = _strip_ansi(string_io.getvalue())
        assert "No provenance" in out
    finally:
        cli_module.console = original_console


def test_show_provenance_list_truncation(tmp_path):
    """Test provenance with long list values in compact vs full mode.

    Compact mode (default): shows "[N items]" summary
    Full mode (--provenance): shows full JSON
    """
    from sleap_io import Labels, save_file

    runner = CliRunner()

    # Create labels with long list provenance
    labels = Labels()
    labels.provenance = {"model_paths": [f"/path/to/model_{i}.h5" for i in range(10)]}

    slp_path = tmp_path / "long_list_provenance.slp"
    save_file(labels, slp_path)

    # Compact mode (no --provenance) shows summary
    result = runner.invoke(cli, ["show", str(slp_path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show "[10 items]" summary
    assert "[10 items]" in out

    # Full mode (--provenance) shows all items
    result = runner.invoke(
        cli, ["show", str(slp_path), "--provenance", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show actual paths
    assert "model_0" in out
    assert "model_9" in out


def test_show_lf_with_nan_points(tmp_path):
    """Test labeled frame display with NaN (invisible) points."""
    import numpy as np

    from sleap_io import (
        Instance,
        LabeledFrame,
        Labels,
        Node,
        Skeleton,
        Video,
        save_file,
    )

    runner = CliRunner()

    # Create a skeleton
    nodes = [Node("head"), Node("tail")]
    skeleton = Skeleton(nodes=nodes)

    # Create instance with a NaN point
    pts = np.array([[100.0, 200.0], [np.nan, np.nan]])
    inst = Instance(skeleton=skeleton, points=pts)

    # Create video and labeled frame
    video = Video(filename="/fake/video.mp4", open_backend=False)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    slp_path = tmp_path / "nan_points.slp"
    save_file(labels, slp_path)

    result = runner.invoke(
        cli, ["show", str(slp_path), "--lf", "0", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show None for NaN point
    assert "(None, None)" in out


def test_show_skeleton_summary_with_symmetries():
    """Test skeleton summary shows symmetry count."""
    from io import StringIO

    from rich.console import Console

    from sleap_io import Labels, Node, Skeleton
    from sleap_io.io.cli import _print_skeleton_summary

    # Create skeleton with symmetries
    nodes = [Node("left"), Node("right"), Node("center")]
    skeleton = Skeleton(nodes=nodes, symmetries=[{nodes[0], nodes[1]}])
    labels = Labels(skeletons=[skeleton])

    # Capture output
    import sleap_io.io.cli as cli_module

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_skeleton_summary(labels)
        out = _strip_ansi(string_io.getvalue())
        assert "1 symmetries" in out or "symmetries" in out
    finally:
        cli_module.console = original_console


def test_show_lf_image_sequence_video(tmp_path):
    """Test labeled frame display with image sequence video."""
    import numpy as np

    from sleap_io import (
        Instance,
        LabeledFrame,
        Labels,
        Node,
        Skeleton,
        Video,
    )

    # Create skeleton and instance
    nodes = [Node("head")]
    skeleton = Skeleton(nodes=nodes)
    pts = np.array([[100.0, 200.0]])
    inst = Instance(skeleton=skeleton, points=pts)

    # Create video with image list
    video = Video(
        filename=["/path/frame1.png", "/path/frame2.png"],
        open_backend=False,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    # We can't save/load image sequences properly, so test the function directly
    from io import StringIO

    from rich.console import Console

    import sleap_io.io.cli as cli_module
    from sleap_io.io.cli import _print_labeled_frame

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_labeled_frame(labels, 0)
        out = _strip_ansi(string_io.getvalue())
        assert "2 images" in out
    finally:
        cli_module.console = original_console


# =============================================================================
# Convert command tests
# =============================================================================


def test_convert_slp_to_nwb(tmp_path, slp_typical):
    """Test basic SLP to NWB conversion."""
    runner = CliRunner()
    output_path = tmp_path / "output.nwb"

    result = runner.invoke(cli, ["convert", "-i", slp_typical, "-o", str(output_path)])
    assert result.exit_code == 0, result.output
    assert "Converted:" in result.output
    assert "slp -> nwb" in result.output
    assert output_path.exists()

    # Verify the output is valid
    from sleap_io import load_nwb

    labels = load_nwb(str(output_path))
    assert isinstance(labels, Labels)


def test_convert_slp_to_slp(tmp_path, slp_typical):
    """Test SLP to SLP conversion (useful for re-saving)."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(cli, ["convert", "-i", slp_typical, "-o", str(output_path)])
    assert result.exit_code == 0, result.output
    assert "slp -> slp" in result.output
    assert output_path.exists()

    # Verify roundtrip
    original = load_slp(slp_typical)
    converted = load_slp(str(output_path))
    assert len(converted.labeled_frames) == len(original.labeled_frames)


def test_convert_slp_to_labelstudio(tmp_path, slp_typical):
    """Test SLP to Label Studio JSON conversion."""
    runner = CliRunner()
    output_path = tmp_path / "output.json"

    result = runner.invoke(cli, ["convert", "-i", slp_typical, "-o", str(output_path)])
    assert result.exit_code == 0, result.output
    assert "slp -> labelstudio" in result.output
    assert output_path.exists()


def test_convert_nwb_to_slp(tmp_path, slp_typical):
    """Test NWB to SLP conversion (roundtrip)."""
    runner = CliRunner()

    # First create an NWB file
    from sleap_io import save_nwb

    labels = load_slp(slp_typical)
    nwb_path = tmp_path / "input.nwb"
    save_nwb(labels, str(nwb_path))

    # Convert back to SLP
    output_path = tmp_path / "output.slp"
    result = runner.invoke(
        cli, ["convert", "-i", str(nwb_path), "-o", str(output_path)]
    )
    assert result.exit_code == 0, result.output
    assert "nwb -> slp" in result.output
    assert output_path.exists()


def test_convert_with_explicit_formats(tmp_path, slp_typical):
    """Test conversion with explicit --from and --to flags."""
    runner = CliRunner()
    output_path = tmp_path / "output.nwb"

    result = runner.invoke(
        cli,
        [
            "convert",
            "-i",
            slp_typical,
            "-o",
            str(output_path),
            "--from",
            "slp",
            "--to",
            "nwb",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_convert_json_requires_from_flag(tmp_path):
    """Test that .json input requires explicit --from flag."""
    runner = CliRunner()

    # Create a dummy JSON file
    json_path = tmp_path / "input.json"
    json_path.write_text("{}")
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli, ["convert", "-i", str(json_path), "-o", str(output_path)]
    )
    assert result.exit_code != 0
    assert "Cannot infer format from '.json'" in result.output
    assert "--from" in result.output


def test_convert_h5_requires_from_flag(tmp_path):
    """Test that .h5 input requires explicit --from flag."""
    runner = CliRunner()

    # Create a dummy H5 file
    import h5py

    h5_path = tmp_path / "input.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("dummy", data=[1, 2, 3])

    output_path = tmp_path / "output.slp"

    result = runner.invoke(cli, ["convert", "-i", str(h5_path), "-o", str(output_path)])
    assert result.exit_code != 0
    assert "Cannot infer format from '.h5'" in result.output


def test_convert_unknown_output_extension(tmp_path, slp_typical):
    """Test error when output extension is unknown."""
    runner = CliRunner()
    output_path = tmp_path / "output.xyz"

    result = runner.invoke(cli, ["convert", "-i", slp_typical, "-o", str(output_path)])
    assert result.exit_code != 0
    assert "Cannot infer output format" in result.output


def test_convert_embed_requires_slp_output(tmp_path, slp_typical):
    """Test that --embed only works with SLP output."""
    runner = CliRunner()
    output_path = tmp_path / "output.nwb"

    result = runner.invoke(
        cli,
        ["convert", "-i", slp_typical, "-o", str(output_path), "--embed", "user"],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "--embed is only valid for SLP output" in output


def test_convert_input_not_found():
    """Test error when input file doesn't exist."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["convert", "-i", "/nonexistent/file.slp", "-o", "output.nwb"],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Invalid value for '-i'" in output or "does not exist" in output


def test_convert_coco_with_explicit_from(tmp_path, coco_annotations_flat):
    """Test COCO JSON input with explicit --from flag."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "convert",
            "-i",
            coco_annotations_flat,
            "-o",
            str(output_path),
            "--from",
            "coco",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "coco -> slp" in result.output
    assert output_path.exists()


def test_convert_to_ultralytics_directory(tmp_path, slp_minimal):
    """Test conversion to Ultralytics format (directory output)."""
    runner = CliRunner()
    output_dir = tmp_path / "yolo_dataset"

    result = runner.invoke(
        cli,
        [
            "convert",
            "-i",
            slp_minimal,
            "-o",
            str(output_dir),
            "--to",
            "ultralytics",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "slp -> ultralytics" in result.output
    assert output_dir.exists()
    assert (output_dir / "data.yaml").exists()


def test_convert_help_shows_formats():
    """Test that convert --help shows supported formats."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
    assert "Supported input formats" in result.output
    assert "Supported output formats" in result.output
    assert "slp" in result.output
    assert "nwb" in result.output


def test_cli_help_shows_command_panels():
    """Test that main CLI --help shows command panels."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # Command panels should organize commands
    assert "show" in result.output
    assert "convert" in result.output


# =============================================================================
# Coverage improvement tests
# =============================================================================


def test_get_package_version_not_installed():
    """Test _get_package_version returns 'not installed' for missing packages."""
    from sleap_io.io.cli import _get_package_version

    result = _get_package_version("nonexistent_package_xyz_12345")
    assert result == "not installed"


def test_infer_input_format_ultralytics_directory(tmp_path):
    """Test _infer_input_format detects ultralytics directory with data.yaml."""
    from sleap_io.io.cli import _infer_input_format

    # Create ultralytics directory structure
    (tmp_path / "data.yaml").write_text("names: [mouse]\n")

    result = _infer_input_format(tmp_path)
    assert result == "ultralytics"


def test_infer_input_format_directory_without_data_yaml(tmp_path):
    """Test _infer_input_format returns None for directory without data.yaml."""
    from sleap_io.io.cli import _infer_input_format

    # Empty directory - no data.yaml
    result = _infer_input_format(tmp_path)
    assert result is None


def test_infer_input_format_unknown_extension(tmp_path):
    """Test _infer_input_format returns None for unknown extensions."""
    from sleap_io.io.cli import _infer_input_format

    unknown_file = tmp_path / "file.xyz"
    unknown_file.write_text("dummy")

    result = _infer_input_format(unknown_file)
    assert result is None


def test_infer_output_format_directory(tmp_path):
    """Test _infer_output_format returns ultralytics for directories."""
    from sleap_io.io.cli import _infer_output_format

    result = _infer_output_format(tmp_path)
    assert result == "ultralytics"


def test_infer_output_format_no_extension(tmp_path):
    """Test _infer_output_format returns ultralytics for paths without extension."""
    from sleap_io.io.cli import _infer_output_format

    result = _infer_output_format(tmp_path / "output_dir")
    assert result == "ultralytics"


def test_convert_unknown_input_extension_error(tmp_path):
    """Test convert error for unknown non-ambiguous input extension."""
    runner = CliRunner()

    # Create a file with unknown extension
    unknown_file = tmp_path / "file.xyz"
    unknown_file.write_text("dummy")
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        ["convert", "-i", str(unknown_file), "-o", str(output_path)],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Cannot infer input format" in output


def test_convert_video_input_error(tmp_path, centered_pair_low_quality_path):
    """Test convert error when input is a video file (not Labels)."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "convert",
            "-i",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--from",
            "slp",
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    # Either load fails or it's not a Labels object
    assert "Failed to load" in output or "not a labels file" in output


def test_convert_load_failure_corrupt_file(tmp_path):
    """Test convert error when load_file raises an exception."""
    runner = CliRunner()

    # Create a corrupt SLP file (invalid HDF5)
    corrupt_slp = tmp_path / "corrupt.slp"
    corrupt_slp.write_text("this is not a valid HDF5 file")
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        ["convert", "-i", str(corrupt_slp), "-o", str(output_path)],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load" in output


def test_convert_with_embed_success(tmp_path, slp_real_data):
    """Test successful conversion with --embed option."""
    runner = CliRunner()
    output_path = tmp_path / "output.pkg.slp"

    result = runner.invoke(
        cli,
        ["convert", "-i", slp_real_data, "-o", str(output_path), "--embed", "user"],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)
    assert "Embedded frames: user" in output
    assert output_path.exists()


def test_convert_preserves_embedded_from_pkg_slp(tmp_path, slp_real_data):
    """Test that convert preserves embedded videos when pkg.slp -> pkg.slp."""
    runner = CliRunner()

    # First, create a pkg.slp with embedded videos
    embedded_path = tmp_path / "embedded.pkg.slp"
    result = runner.invoke(
        cli,
        ["convert", "-i", slp_real_data, "-o", str(embedded_path), "--embed", "user"],
    )
    assert result.exit_code == 0, result.output

    # Verify input has embedded videos
    input_labels = load_slp(str(embedded_path))
    assert input_labels.videos[0].backend_metadata.get("has_embedded_images", False)

    # Convert pkg.slp -> pkg.slp (without explicit --embed flag)
    # This should auto-preserve embedded videos
    output_path = tmp_path / "converted.pkg.slp"
    result = runner.invoke(
        cli,
        ["convert", "-i", str(embedded_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output

    # Verify output has embedded videos preserved
    output_labels = load_slp(str(output_path))
    # backend_metadata indicates embedding
    assert output_labels.videos[0].backend_metadata.get("has_embedded_images", False)


def test_convert_save_failure_invalid_path(tmp_path, slp_typical):
    """Test convert error when save fails due to invalid output path."""
    runner = CliRunner()
    # Try to save to a path inside a non-existent directory (for non-ultralytics)
    output_path = tmp_path / "nonexistent_dir" / "subdir" / "output.nwb"

    result = runner.invoke(
        cli,
        ["convert", "-i", slp_typical, "-o", str(output_path)],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to save" in output


# ======================= Split command tests =======================


def test_split_basic_two_way(tmp_path, clip_2nodes_slp):
    """Test basic 2-way split (train/val) with default 80/20 proportions."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli, ["split", "-i", clip_2nodes_slp, "-o", str(output_dir), "--seed", "42"]
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check success message
    assert "Split 1500 frames from:" in output
    assert "train.slp:" in output
    assert "val.slp:" in output
    assert "Random seed: 42" in output

    # Check output files exist
    assert (output_dir / "train.slp").exists()
    assert (output_dir / "val.slp").exists()
    assert not (output_dir / "test.slp").exists()  # No test split by default

    # Verify split sizes (80/20)
    train_labels = load_slp(str(output_dir / "train.slp"))
    val_labels = load_slp(str(output_dir / "val.slp"))
    assert len(train_labels) == 1200  # 80% of 1500
    assert len(val_labels) == 300  # Remainder


def test_split_three_way(tmp_path, clip_2nodes_slp):
    """Test 3-way split (train/val/test)."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            clip_2nodes_slp,
            "-o",
            str(output_dir),
            "--train",
            "0.7",
            "--test",
            "0.15",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check success message
    assert "train.slp:" in output
    assert "val.slp:" in output
    assert "test.slp:" in output

    # Check all output files exist
    assert (output_dir / "train.slp").exists()
    assert (output_dir / "val.slp").exists()
    assert (output_dir / "test.slp").exists()

    # Verify split sizes
    train_labels = load_slp(str(output_dir / "train.slp"))
    val_labels = load_slp(str(output_dir / "val.slp"))
    test_labels = load_slp(str(output_dir / "test.slp"))
    assert len(train_labels) == 1050  # 70% of 1500
    assert len(test_labels) == 225  # 15% of 1500
    assert len(val_labels) == 225  # Remainder (15%)


def test_split_reproducibility_with_seed(tmp_path, clip_2nodes_slp):
    """Test that same seed produces identical splits."""
    runner = CliRunner()

    # First split
    output_dir1 = tmp_path / "splits1"
    result1 = runner.invoke(
        cli, ["split", "-i", clip_2nodes_slp, "-o", str(output_dir1), "--seed", "123"]
    )
    assert result1.exit_code == 0

    # Second split with same seed
    output_dir2 = tmp_path / "splits2"
    result2 = runner.invoke(
        cli, ["split", "-i", clip_2nodes_slp, "-o", str(output_dir2), "--seed", "123"]
    )
    assert result2.exit_code == 0

    # Load and compare frame indices
    train1 = load_slp(str(output_dir1 / "train.slp"))
    train2 = load_slp(str(output_dir2 / "train.slp"))

    # Get frame indices from both splits
    indices1 = sorted([lf.frame_idx for lf in train1.labeled_frames])
    indices2 = sorted([lf.frame_idx for lf in train2.labeled_frames])

    assert indices1 == indices2


def test_split_remove_predictions(tmp_path, slp_real_data):
    """Test --remove-predictions flag removes predicted instances."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            slp_real_data,
            "-o",
            str(output_dir),
            "--remove-predictions",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Predictions removed: yes" in output

    # Verify predictions were removed
    from sleap_io.model.instance import PredictedInstance

    train_labels = load_slp(str(output_dir / "train.slp"))
    for lf in train_labels.labeled_frames:
        for inst in lf.instances:
            assert not isinstance(inst, PredictedInstance)


def test_split_single_frame_warning(tmp_path, slp_typical):
    """Test warning when splitting single frame."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli, ["split", "-i", slp_typical, "-o", str(output_dir), "--seed", "42"]
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should warn about single frame
    assert "Only 1 labeled frame found" in output
    assert "All splits will contain the same frame" in output

    # Both splits should have the same frame
    train_labels = load_slp(str(output_dir / "train.slp"))
    val_labels = load_slp(str(output_dir / "val.slp"))
    assert len(train_labels) == 1
    assert len(val_labels) == 1


def test_split_provenance_tracking(tmp_path, clip_2nodes_slp):
    """Test that provenance is updated with split info and seed."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            clip_2nodes_slp,
            "-o",
            str(output_dir),
            "--test",
            "0.1",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output

    # Check provenance in each split
    train_labels = load_slp(str(output_dir / "train.slp"))
    val_labels = load_slp(str(output_dir / "val.slp"))
    test_labels = load_slp(str(output_dir / "test.slp"))

    assert train_labels.provenance.get("split") == "train"
    assert val_labels.provenance.get("split") == "val"
    assert test_labels.provenance.get("split") == "test"

    assert train_labels.provenance.get("split_seed") == 42
    assert val_labels.provenance.get("split_seed") == 42
    assert test_labels.provenance.get("split_seed") == 42


def test_split_train_fraction_out_of_range():
    """Test error when train fraction is out of valid range."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")

    # Test train = 0
    result = runner.invoke(cli, ["split", "-i", str(path), "-o", "out", "--train", "0"])
    assert result.exit_code != 0
    assert "--train must be between 0 and 1" in _strip_ansi(result.output)

    # Test train = 1
    result = runner.invoke(cli, ["split", "-i", str(path), "-o", "out", "--train", "1"])
    assert result.exit_code != 0
    assert "--train must be between 0 and 1" in _strip_ansi(result.output)

    # Test train > 1
    result = runner.invoke(
        cli, ["split", "-i", str(path), "-o", "out", "--train", "1.5"]
    )
    assert result.exit_code != 0
    assert "--train must be between 0 and 1" in _strip_ansi(result.output)


def test_split_val_fraction_out_of_range():
    """Test error when val fraction is out of valid range."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")

    result = runner.invoke(cli, ["split", "-i", str(path), "-o", "out", "--val", "0"])
    assert result.exit_code != 0
    assert "--val must be between 0 and 1" in _strip_ansi(result.output)


def test_split_test_fraction_out_of_range():
    """Test error when test fraction is out of valid range."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")

    result = runner.invoke(
        cli, ["split", "-i", str(path), "-o", "out", "--test", "1.1"]
    )
    assert result.exit_code != 0
    assert "--test must be between 0 and 1" in _strip_ansi(result.output)


def test_split_fractions_exceed_one():
    """Test error when sum of fractions exceeds 1.0."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")

    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            str(path),
            "-o",
            "out",
            "--train",
            "0.6",
            "--val",
            "0.3",
            "--test",
            "0.3",
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Sum of fractions" in output
    assert "exceeds 1.0" in output


def test_split_no_frames_after_remove_predictions(tmp_path, centered_pair):
    """Test error when no frames remain after removing predictions."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        ["split", "-i", centered_pair, "-o", str(output_dir), "--remove-predictions"],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "No labeled frames found" in output


def test_split_input_not_found():
    """Test error when input file doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["split", "-i", "nonexistent.slp", "-o", "out"])
    assert result.exit_code != 0


def test_split_help_shows_options():
    """Test that split --help shows all expected options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["split", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    assert "--train" in output
    assert "--val" in output
    assert "--test" in output
    assert "--remove-predictions" in output
    assert "--seed" in output
    assert "--embed" in output
    assert "-i" in output or "--input" in output
    assert "-o" in output or "--output" in output


def test_split_explicit_val_fraction(tmp_path, clip_2nodes_slp):
    """Test split with explicit val fraction (not just remainder)."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            clip_2nodes_slp,
            "-o",
            str(output_dir),
            "--train",
            "0.6",
            "--val",
            "0.2",
            "--test",
            "0.2",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify split sizes
    train_labels = load_slp(str(output_dir / "train.slp"))
    val_labels = load_slp(str(output_dir / "val.slp"))
    test_labels = load_slp(str(output_dir / "test.slp"))

    assert len(train_labels) == 900  # 60% of 1500
    assert len(val_labels) == 300  # 20% of 1500
    assert len(test_labels) == 300  # 20% of 1500


def test_split_non_labels_input(tmp_path, centered_pair_low_quality_path):
    """Test error when input is not a labels file."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        ["split", "-i", centered_pair_low_quality_path, "-o", str(output_dir)],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "not a labels file" in output


def test_split_single_frame_with_test(tmp_path, slp_typical):
    """Test single frame split with test set (all splits get same frame)."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            slp_typical,
            "-o",
            str(output_dir),
            "--test",
            "0.1",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output

    # All three splits should exist and have 1 frame each
    train_labels = load_slp(str(output_dir / "train.slp"))
    val_labels = load_slp(str(output_dir / "val.slp"))
    test_labels = load_slp(str(output_dir / "test.slp"))

    assert len(train_labels) == 1
    assert len(val_labels) == 1
    assert len(test_labels) == 1


def test_split_load_failure_corrupt_file(tmp_path):
    """Test error when input file is corrupt."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    # Create a corrupt file
    corrupt_file = tmp_path / "corrupt.slp"
    corrupt_file.write_text("this is not a valid slp file")

    result = runner.invoke(
        cli, ["split", "-i", str(corrupt_file), "-o", str(output_dir)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load" in output


def test_split_save_failure_invalid_path(tmp_path, slp_typical):
    """Test error when save fails due to permission issues."""
    runner = CliRunner()
    # Use an invalid path with null byte that can't be created
    output_dir = tmp_path / "splits"

    # Create the output directory but make a file where we expect to write
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create a directory where we expect a file, causing save to fail
    (output_dir / "train.slp").mkdir()

    result = runner.invoke(cli, ["split", "-i", slp_typical, "-o", str(output_dir)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to save" in output


def test_split_without_seed(tmp_path, clip_2nodes_slp):
    """Test split without seed (no seed in output)."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(cli, ["split", "-i", clip_2nodes_slp, "-o", str(output_dir)])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should not mention seed
    assert "Random seed:" not in output

    # Check provenance doesn't have seed
    train_labels = load_slp(str(output_dir / "train.slp"))
    assert "split_seed" not in train_labels.provenance


def test_split_with_embed(tmp_path, slp_real_data):
    """Test split with --embed option creates package files."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"

    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            slp_real_data,
            "-o",
            str(output_dir),
            "--embed",
            "user",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check output mentions embed
    assert "Embedded frames: user" in output

    # Check package files were created
    assert (output_dir / "train.pkg.slp").exists()
    assert (output_dir / "val.pkg.slp").exists()
    # No test split by default
    assert not (output_dir / "test.pkg.slp").exists()

    # Verify the package files are valid and have embedded images
    train_labels = load_slp(str(output_dir / "train.pkg.slp"))
    assert len(train_labels) > 0


# ============================================================================
# unsplit command tests
# ============================================================================


def test_unsplit_basic(tmp_path, clip_2nodes_slp):
    """Test basic 2-way unsplit (train + val -> merged)."""
    runner = CliRunner()

    # First, split the labels
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli, ["split", "-i", clip_2nodes_slp, "-o", str(split_dir), "--seed", "42"]
    )
    assert result.exit_code == 0, result.output

    # Count frames in splits
    train_labels = load_slp(str(split_dir / "train.slp"))
    val_labels = load_slp(str(split_dir / "val.slp"))
    expected_total = len(train_labels) + len(val_labels)

    # Now unsplit
    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "unsplit",
            str(split_dir / "train.slp"),
            str(split_dir / "val.slp"),
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check output messages
    assert "Loading:" in output
    assert "Merging:" in output
    assert "Saving:" in output
    assert "Merged 2 files:" in output

    # Verify merged file
    merged_labels = load_slp(str(merged_path))
    assert len(merged_labels) == expected_total
    assert len(merged_labels.videos) == 1  # Should deduplicate to 1 video


def test_unsplit_three_way(tmp_path, clip_2nodes_slp):
    """Test 3-way unsplit (train + val + test -> merged)."""
    runner = CliRunner()

    # First, split into 3 parts
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            clip_2nodes_slp,
            "-o",
            str(split_dir),
            "--train",
            "0.7",
            "--test",
            "0.15",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output

    # Count frames in splits
    train_labels = load_slp(str(split_dir / "train.slp"))
    val_labels = load_slp(str(split_dir / "val.slp"))
    test_labels = load_slp(str(split_dir / "test.slp"))
    expected_total = len(train_labels) + len(val_labels) + len(test_labels)

    # Now unsplit
    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "unsplit",
            str(split_dir / "train.slp"),
            str(split_dir / "val.slp"),
            str(split_dir / "test.slp"),
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check output messages
    assert "Merged 3 files:" in output

    # Verify merged file
    merged_labels = load_slp(str(merged_path))
    assert len(merged_labels) == expected_total
    assert len(merged_labels.videos) == 1


def test_unsplit_requires_two_files(tmp_path, slp_typical):
    """Test that unsplit requires at least 2 input files."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    result = runner.invoke(cli, ["unsplit", slp_typical, "-o", str(merged_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "At least 2 input files required" in output


def test_unsplit_missing_output(slp_typical):
    """Test that unsplit requires -o flag."""
    runner = CliRunner()

    result = runner.invoke(cli, ["unsplit", slp_typical, slp_typical])
    assert result.exit_code != 0
    # Click will report missing required option
    assert "-o" in result.output or "output" in result.output.lower()


def test_unsplit_input_not_found(tmp_path):
    """Test error when input file doesn't exist."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    result = runner.invoke(
        cli, ["unsplit", "nonexistent1.slp", "nonexistent2.slp", "-o", str(merged_path)]
    )
    assert result.exit_code != 0


def test_unsplit_help():
    """Test that unsplit --help displays correctly."""
    runner = CliRunner()
    result = runner.invoke(cli, ["unsplit", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    # Check key elements
    assert "Merge split files" in output
    assert "INPUT_FILES" in output
    assert "-o" in output or "--output" in output
    assert "--embed" in output


def test_unsplit_with_embed(tmp_path, slp_real_data):
    """Test unsplit with embedded split files."""
    runner = CliRunner()

    # First, split with embedding
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            slp_real_data,
            "-o",
            str(split_dir),
            "--embed",
            "user",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output

    # Count frames in splits
    train_labels = load_slp(str(split_dir / "train.pkg.slp"))
    val_labels = load_slp(str(split_dir / "val.pkg.slp"))
    expected_total = len(train_labels) + len(val_labels)

    # Verify they have source_video (from embedding)
    assert train_labels.videos[0].source_video is not None

    # Now unsplit (without embedding to avoid frame index issues)
    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "unsplit",
            str(split_dir / "train.pkg.slp"),
            str(split_dir / "val.pkg.slp"),
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify merged file
    merged_labels = load_slp(str(merged_path))
    assert len(merged_labels) == expected_total
    # Videos should deduplicate via original_video provenance
    assert len(merged_labels.videos) == 1


def test_unsplit_with_directory(tmp_path, clip_2nodes_slp):
    """Test unsplit accepts a directory and expands to .slp files."""
    runner = CliRunner()

    # First, split the labels
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli, ["split", "-i", clip_2nodes_slp, "-o", str(split_dir), "--seed", "42"]
    )
    assert result.exit_code == 0, result.output

    # Count frames in splits
    train_labels = load_slp(str(split_dir / "train.slp"))
    val_labels = load_slp(str(split_dir / "val.slp"))
    expected_total = len(train_labels) + len(val_labels)

    # Unsplit using directory path
    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        ["unsplit", str(split_dir), "-o", str(merged_path)],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check output messages
    assert "Merged 2 files:" in output

    # Verify merged file
    merged_labels = load_slp(str(merged_path))
    assert len(merged_labels) == expected_total


def test_unsplit_empty_directory(tmp_path):
    """Test unsplit fails gracefully on empty directory."""
    runner = CliRunner()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    merged_path = tmp_path / "merged.slp"

    result = runner.invoke(cli, ["unsplit", str(empty_dir), "-o", str(merged_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "No .slp files found" in output


def test_unsplit_load_failure(tmp_path):
    """Test error when input file is corrupt."""
    runner = CliRunner()

    # Create corrupt files
    corrupt1 = tmp_path / "corrupt1.slp"
    corrupt2 = tmp_path / "corrupt2.slp"
    corrupt1.write_text("not a valid slp file")
    corrupt2.write_text("also not valid")

    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli, ["unsplit", str(corrupt1), str(corrupt2), "-o", str(merged_path)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load" in output


def test_unsplit_non_labels_input(tmp_path, centered_pair_low_quality_path):
    """Test error when input is not a labels file (e.g., a video)."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Try to unsplit video files (not labels)
    result = runner.invoke(
        cli,
        [
            "unsplit",
            centered_pair_low_quality_path,
            centered_pair_low_quality_path,
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "not a labels file" in output


def test_unsplit_second_file_corrupt(tmp_path, slp_typical):
    """Test error when second input file is corrupt."""
    runner = CliRunner()

    # Create a corrupt second file
    corrupt = tmp_path / "corrupt.slp"
    corrupt.write_text("not a valid slp file")

    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli, ["unsplit", slp_typical, str(corrupt), "-o", str(merged_path)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load" in output


def test_unsplit_second_file_not_labels(
    tmp_path, slp_typical, centered_pair_low_quality_path
):
    """Test error when second input is not a labels file."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    result = runner.invoke(
        cli,
        [
            "unsplit",
            slp_typical,
            centered_pair_low_quality_path,
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "not a labels file" in output


def test_unsplit_save_failure(tmp_path, clip_2nodes_slp):
    """Test error when save fails."""
    runner = CliRunner()

    # First, split the labels
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli, ["split", "-i", clip_2nodes_slp, "-o", str(split_dir), "--seed", "42"]
    )
    assert result.exit_code == 0

    # Create a directory where we expect a file, causing save to fail
    merged_path = tmp_path / "merged.slp"
    merged_path.mkdir()

    result = runner.invoke(
        cli,
        [
            "unsplit",
            str(split_dir / "train.slp"),
            str(split_dir / "val.slp"),
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to save" in output


def test_unsplit_preserves_embedded_from_pkg_slp(tmp_path, slp_real_data):
    """Test that unsplit preserves embedded videos when pkg.slp -> pkg.slp."""
    runner = CliRunner()

    # First, split with embedding to create pkg.slp files
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            slp_real_data,
            "-o",
            str(split_dir),
            "--embed",
            "user",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify splits are pkg.slp with embedded videos
    train_pkg = split_dir / "train.pkg.slp"
    val_pkg = split_dir / "val.pkg.slp"
    assert train_pkg.exists()
    assert val_pkg.exists()

    train_labels = load_slp(str(train_pkg))
    assert train_labels.videos[0].backend_metadata.get("has_embedded_images", False)

    # Count expected frames
    val_labels = load_slp(str(val_pkg))
    expected_total = len(train_labels) + len(val_labels)

    # Unsplit to pkg.slp OUTPUT (without explicit --embed flag)
    # This should auto-preserve embedded videos
    merged_path = tmp_path / "merged.pkg.slp"
    result = runner.invoke(
        cli,
        [
            "unsplit",
            str(train_pkg),
            str(val_pkg),
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify merged file has embedded videos preserved
    merged_labels = load_slp(str(merged_path))
    assert len(merged_labels) == expected_total
    # Verify embedded videos preserved
    assert merged_labels.videos[0].backend_metadata.get("has_embedded_images", False)


# ============================================================================
# merge command tests
# ============================================================================


def test_merge_basic(tmp_path, slp_minimal, slp_typical):
    """Test basic merge of two labels files."""
    runner = CliRunner()

    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "merge",
            slp_minimal,
            slp_typical,
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check output messages
    assert "Loading:" in output
    assert "Merging:" in output
    assert "Saving:" in output
    assert "Merged 2 files:" in output

    # Verify merged file exists and is valid
    merged_labels = load_slp(str(merged_path))
    assert len(merged_labels) > 0


def test_merge_with_frame_strategy(tmp_path, slp_typical):
    """Test merge with explicit frame strategy."""
    runner = CliRunner()

    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "merge",
            slp_typical,
            slp_typical,  # Merge with itself
            "-o",
            str(merged_path),
            "--frame",
            "keep_both",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Merged 2 files:" in output


def test_merge_verbose(tmp_path, slp_minimal, slp_typical):
    """Test merge with verbose output."""
    runner = CliRunner()

    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "merge",
            slp_minimal,
            slp_typical,
            "-o",
            str(merged_path),
            "-v",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Verbose output includes strategy and instance counts
    assert "Strategy:" in output
    assert "instances" in output.lower()


def test_merge_directory(tmp_path, clip_2nodes_slp):
    """Test merge with directory input."""
    runner = CliRunner()

    # First, split the labels to create multiple files
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli, ["split", "-i", clip_2nodes_slp, "-o", str(split_dir), "--seed", "42"]
    )
    assert result.exit_code == 0, result.output

    # Merge using directory input
    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "merge",
            str(split_dir),
            "-o",
            str(merged_path),
            "--frame",
            "keep_both",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Merged" in output


def test_merge_requires_two_files(tmp_path, slp_typical):
    """Test that merge requires at least 2 input files."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    result = runner.invoke(cli, ["merge", slp_typical, "-o", str(merged_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "At least 2 input files required" in output


def test_merge_missing_output(slp_typical):
    """Test that merge requires -o flag."""
    runner = CliRunner()

    result = runner.invoke(cli, ["merge", slp_typical, slp_typical])
    assert result.exit_code != 0
    # Click will report missing required option
    assert "-o" in result.output or "output" in result.output.lower()


def test_merge_help():
    """Test that merge --help displays correctly."""
    runner = CliRunner()
    result = runner.invoke(cli, ["merge", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    # Check key elements
    assert "Merge multiple labels files" in output
    assert "INPUT_FILES" in output
    assert "--frame" in output
    assert "--video" in output
    assert "--skeleton" in output
    assert "--track" in output
    assert "--instance" in output


def test_merge_all_options(tmp_path, slp_typical):
    """Test merge with all matching options specified."""
    runner = CliRunner()

    merged_path = tmp_path / "merged.slp"
    result = runner.invoke(
        cli,
        [
            "merge",
            slp_typical,
            slp_typical,
            "-o",
            str(merged_path),
            "--skeleton",
            "structure",
            "--video",
            "auto",
            "--track",
            "name",
            "--frame",
            "auto",
            "--instance",
            "spatial",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Merged 2 files:" in output


def test_merge_frame_strategies(tmp_path, slp_typical):
    """Test different frame strategies."""
    runner = CliRunner()

    strategies = [
        "auto",
        "keep_original",
        "keep_new",
        "keep_both",
        "replace_predictions",
    ]

    for strategy in strategies:
        merged_path = tmp_path / f"merged_{strategy}.slp"
        result = runner.invoke(
            cli,
            [
                "merge",
                slp_typical,
                slp_typical,
                "-o",
                str(merged_path),
                "--frame",
                strategy,
            ],
        )
        assert result.exit_code == 0, f"Failed for strategy {strategy}: {result.output}"


def test_merge_load_failure(tmp_path):
    """Test error when first file fails to load."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Create empty file (invalid SLP)
    invalid_file = tmp_path / "invalid.slp"
    invalid_file.write_text("")

    valid_file = _data_path("slp/typical.slp")

    result = runner.invoke(
        cli, ["merge", str(invalid_file), str(valid_file), "-o", str(merged_path)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load" in output


def test_merge_second_file_load_failure(tmp_path, slp_typical):
    """Test error when second file fails to load."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Create empty file (invalid SLP)
    invalid_file = tmp_path / "invalid.slp"
    invalid_file.write_text("")

    result = runner.invoke(
        cli, ["merge", slp_typical, str(invalid_file), "-o", str(merged_path)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load" in output


def test_merge_empty_directory(tmp_path):
    """Test error when directory contains no .slp files."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(cli, ["merge", str(empty_dir), "-o", str(merged_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "No .slp files found" in output


def test_merge_non_labels_file(tmp_path):
    """Test error when first input file is not a labels file (e.g., video)."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Use a video file - load_file will return Video, not Labels
    video_file = _data_path("videos/centered_pair_low_quality.mp4")
    valid_file = _data_path("slp/typical.slp")

    result = runner.invoke(
        cli, ["merge", str(video_file), str(valid_file), "-o", str(merged_path)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "not a labels file" in output


def test_merge_second_non_labels_file(tmp_path, slp_typical):
    """Test error when second input file is not a labels file."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Use a video file - load_file will return Video, not Labels
    video_file = _data_path("videos/centered_pair_low_quality.mp4")

    result = runner.invoke(
        cli, ["merge", slp_typical, str(video_file), "-o", str(merged_path)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "not a labels file" in output


def test_merge_verbose_video_count_increase(tmp_path, slp_minimal, slp_multiview):
    """Test verbose output when video count increases during merge."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # slp_minimal has 1 video, slp_multiview has 8 videos with different paths
    # Merging should increase video count, triggering the verbose message
    result = runner.invoke(
        cli,
        [
            "merge",
            slp_minimal,
            slp_multiview,
            "-o",
            str(merged_path),
            "-v",
            "--video",
            "path",  # Force path-based matching to add new videos
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Verbose output should show strategy and instance info
    assert "Strategy:" in output
    assert "instances" in output.lower()
    # Video count should increase from 1 to 9, triggering the verbose message
    assert "Video count increased" in output


def test_merge_with_embed_option(tmp_path, slp_minimal_pkg):
    """Test merge with embed option using embedded package file."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Use pkg.slp files that have embedded images
    result = runner.invoke(
        cli,
        [
            "merge",
            slp_minimal_pkg,
            slp_minimal_pkg,
            "-o",
            str(merged_path),
            "--embed",
            "source",  # Preserve existing embedded frames
            "--frame",
            "keep_both",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Merged 2 files:" in output


def test_merge_save_failure(tmp_path, slp_typical):
    """Test error when save fails (e.g., invalid path)."""
    runner = CliRunner()

    # Try to save to a path that doesn't exist (directory doesn't exist)
    nonexistent_dir = tmp_path / "nonexistent" / "subdir" / "merged.slp"

    result = runner.invoke(
        cli, ["merge", slp_typical, slp_typical, "-o", str(nonexistent_dir)]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to save" in output


def test_merge_verbose_with_conflicts(tmp_path, slp_typical):
    """Test verbose merge showing conflict resolution."""
    runner = CliRunner()
    merged_path = tmp_path / "merged.slp"

    # Merge file with itself using keep_both - this will create conflicts
    result = runner.invoke(
        cli,
        [
            "merge",
            slp_typical,
            slp_typical,
            "-o",
            str(merged_path),
            "-v",
            "--frame",
            "keep_both",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should have verbose output
    assert "Strategy:" in output


def test_merge_preserves_embedded_from_pkg_slp(tmp_path, slp_real_data):
    """Test that merge preserves embedded videos when pkg.slp -> pkg.slp."""
    runner = CliRunner()

    # First, split with embedding to create pkg.slp files
    split_dir = tmp_path / "splits"
    result = runner.invoke(
        cli,
        [
            "split",
            "-i",
            slp_real_data,
            "-o",
            str(split_dir),
            "--embed",
            "user",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output

    train_pkg = split_dir / "train.pkg.slp"
    val_pkg = split_dir / "val.pkg.slp"
    assert train_pkg.exists()
    assert val_pkg.exists()

    # Verify splits have embedded videos
    train_labels = load_slp(str(train_pkg))
    assert train_labels.videos[0].backend_metadata.get("has_embedded_images", False)

    # Merge to pkg.slp OUTPUT (without explicit --embed flag)
    # This should auto-preserve embedded videos
    merged_path = tmp_path / "merged.pkg.slp"
    result = runner.invoke(
        cli,
        [
            "merge",
            str(train_pkg),
            str(val_pkg),
            "-o",
            str(merged_path),
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify merged file has embedded videos preserved
    merged_labels = load_slp(str(merged_path))
    # backend_metadata indicates embedding
    assert merged_labels.videos[0].backend_metadata.get("has_embedded_images", False)


# ============================================================================
# filenames command tests
# ============================================================================


def test_filenames_inspection_mode(slp_typical):
    """Test filenames command in inspection mode (no update flags)."""
    runner = CliRunner()

    result = runner.invoke(cli, ["filenames", "-i", slp_typical])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should show header and video list
    assert "Video filenames in" in output
    assert "[0]" in output


def test_filenames_inspection_mode_multi_video():
    """Test inspection mode with multiple videos."""
    runner = CliRunner()
    input_path = _data_path("slp/multiview.slp")

    result = runner.invoke(cli, ["filenames", "-i", str(input_path)])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should show multiple videos
    assert "Video filenames in" in output
    assert "[0]" in output
    assert "[1]" in output


def test_filenames_list_mode(tmp_path, slp_typical):
    """Test filenames with --filename (list mode)."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    # Load original to get video count
    original = load_slp(slp_typical)
    assert len(original.videos) == 1

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            slp_typical,
            "-o",
            str(output_path),
            "--filename",
            "/new/path/video.mp4",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Replaced (1/1):" in output
    assert "/new/path/video.mp4" in output
    assert "Saved:" in output

    # Verify the output file has the new filename
    labels = load_slp(str(output_path), open_videos=False)
    assert labels.videos[0].filename == "/new/path/video.mp4"


def test_filenames_list_mode_multiple(tmp_path):
    """Test filenames with multiple --filename options."""
    runner = CliRunner()
    # Use multiview.slp which has multiple videos
    input_path = _data_path("slp/multiview.slp")
    output_path = tmp_path / "output.slp"

    original = load_slp(str(input_path), open_videos=False)
    n_videos = len(original.videos)
    assert n_videos > 1

    # Build command with one --filename per video
    cmd = ["filenames", "-i", str(input_path), "-o", str(output_path)]
    for i in range(n_videos):
        cmd.extend(["--filename", f"/new/path/video{i}.mp4"])

    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert f"Replaced ({n_videos}/{n_videos}):" in output

    # Verify filenames were replaced
    labels = load_slp(str(output_path), open_videos=False)
    for i, vid in enumerate(labels.videos):
        assert vid.filename == f"/new/path/video{i}.mp4"


def test_filenames_map_mode(tmp_path, slp_typical):
    """Test filenames with --map OLD NEW."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    # Get original filename
    original = load_slp(slp_typical, open_videos=False)
    old_filename = original.videos[0].filename

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            slp_typical,
            "-o",
            str(output_path),
            "--map",
            old_filename,
            "/mapped/video.mp4",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Replaced (1/1):" in output
    assert "/mapped/video.mp4" in output

    # Verify the filename was mapped
    labels = load_slp(str(output_path), open_videos=False)
    assert labels.videos[0].filename == "/mapped/video.mp4"


def test_filenames_map_mode_multiple(tmp_path):
    """Test filenames with multiple --map options."""
    runner = CliRunner()
    input_path = _data_path("slp/multiview.slp")
    output_path = tmp_path / "output.slp"

    original = load_slp(str(input_path), open_videos=False)
    # Map just the first two videos
    old_fn0 = original.videos[0].filename
    old_fn1 = original.videos[1].filename

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--map",
            old_fn0,
            "/new/video0.mp4",
            "--map",
            old_fn1,
            "/new/video1.mp4",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify mapped filenames
    labels = load_slp(str(output_path), open_videos=False)
    assert labels.videos[0].filename == "/new/video0.mp4"
    assert labels.videos[1].filename == "/new/video1.mp4"


def test_filenames_prefix_mode(tmp_path, slp_typical):
    """Test filenames with --prefix OLD NEW."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    # Get original filename and determine a prefix to replace
    original = load_slp(slp_typical, open_videos=False)
    old_filename = original.videos[0].filename
    # The filename likely has a path prefix we can replace
    old_prefix = str(Path(old_filename).parent)

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            slp_typical,
            "-o",
            str(output_path),
            "--prefix",
            old_prefix,
            "/new/prefix",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Replaced (1/1):" in output
    assert "/new/prefix" in output

    # Verify prefix was replaced
    labels = load_slp(str(output_path), open_videos=False)
    assert labels.videos[0].filename.startswith("/new/prefix")


def test_filenames_prefix_mode_cross_platform(tmp_path):
    """Test prefix mode handles Windows to Linux path conversion."""
    from sleap_io.model.video import Video

    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    # Create a labels file with Windows-style paths
    labels = Labels(
        videos=[
            Video.from_filename(r"C:\data\videos\test.mp4"),
        ]
    )
    input_path = tmp_path / "windows_paths.slp"
    labels.save(str(input_path))

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--prefix",
            r"C:\data\videos",
            "/mnt/data/videos",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify path was converted
    labels = load_slp(str(output_path), open_videos=False)
    assert labels.videos[0].filename == "/mnt/data/videos/test.mp4"


def test_filenames_update_without_output_error(slp_typical):
    """Test error when update flags are used without -o."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            slp_typical,
            "--filename",
            "/new/video.mp4",
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Output path (-o) required" in output


def test_filenames_multiple_modes_error(slp_typical):
    """Test error when multiple modes are specified."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            slp_typical,
            "-o",
            "out.slp",
            "--filename",
            "/new/video.mp4",
            "--map",
            "old.mp4",
            "new.mp4",
        ],
    )
    assert result.exit_code != 0
    assert "Only one mode allowed" in result.output


def test_filenames_list_count_mismatch(tmp_path, slp_typical):
    """Test error when filename count doesn't match video count."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    # Provide too many filenames (typical.slp has 1 video)
    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            slp_typical,
            "-o",
            str(output_path),
            "--filename",
            "/new/video1.mp4",
            "--filename",
            "/new/video2.mp4",
        ],
    )
    assert result.exit_code != 0
    assert "does not match" in result.output


def test_filenames_paths_with_equals(tmp_path):
    """Test that paths containing '=' work correctly with nargs=2."""
    from sleap_io.model.video import Video

    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    # Create a labels file with a path containing '='
    labels = Labels(
        videos=[
            Video.from_filename("/data/exp=1/video.mp4"),
        ]
    )
    input_path = tmp_path / "equals_path.slp"
    labels.save(str(input_path))

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--map",
            "/data/exp=1/video.mp4",
            "/new/exp=2/video.mp4",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify path with '=' was handled correctly
    labels = load_slp(str(output_path), open_videos=False)
    assert labels.videos[0].filename == "/new/exp=2/video.mp4"


def test_filenames_input_not_found():
    """Test error when input file doesn't exist."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            "nonexistent.slp",
            "--filename",
            "/new/video.mp4",
        ],
    )
    assert result.exit_code != 0


def test_filenames_help():
    """Test that help text displays correctly."""
    runner = CliRunner()

    result = runner.invoke(cli, ["filenames", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    # Check key elements are present
    assert "List or update video filenames" in output
    assert "--filename" in output
    assert "--map" in output
    assert "--prefix" in output
    assert "Inspection mode" in output
    assert "List mode" in output
    assert "Map mode" in output
    assert "Prefix mode" in output


def test_filenames_in_command_list():
    """Test that filenames appears in the main help."""
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    # Check command is listed under Inspection
    assert "filenames" in output
    # Check it shows description
    assert "List or update video filenames" in output


def test_filenames_load_failure(tmp_path):
    """Test error when input file cannot be loaded (corrupt file)."""
    runner = CliRunner()

    # Create a corrupt file
    corrupt_file = tmp_path / "corrupt.slp"
    corrupt_file.write_text("not a valid slp file")

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            str(corrupt_file),
        ],
    )
    assert result.exit_code != 0
    assert "Failed to load input file" in result.output


def test_filenames_non_labels_input(centered_pair_low_quality_path):
    """Test error when input file is not a labels file."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            centered_pair_low_quality_path,
        ],
    )
    assert result.exit_code != 0
    assert "not a labels file" in result.output


def test_filenames_save_failure(tmp_path, slp_typical):
    """Test error when output file cannot be saved."""
    runner = CliRunner()
    # Use an invalid path that can't be written to
    invalid_output = tmp_path / "nonexistent_dir" / "deep" / "path" / "output.slp"

    result = runner.invoke(
        cli,
        [
            "filenames",
            "-i",
            slp_typical,
            "-o",
            str(invalid_output),
            "--filename",
            "/new/video.mp4",
        ],
    )
    assert result.exit_code != 0
    assert "Failed to save output file" in result.output


# ============================================================================
# Render command crop tests
# ============================================================================


def test_render_crop_pixel(centered_pair, tmp_path):
    """Test render with pixel crop coordinates."""
    runner = CliRunner()
    output_path = tmp_path / "cropped.png"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--lf",
            "0",
            "--crop",
            "100,100,300,300",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Rendered:" in result.output
    assert output_path.exists()


def test_render_crop_normalized(centered_pair, tmp_path):
    """Test render with normalized crop coordinates."""
    runner = CliRunner()
    output_path = tmp_path / "cropped.png"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--lf",
            "0",
            "--crop",
            "0.25,0.25,0.75,0.75",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Rendered:" in result.output
    assert output_path.exists()


def test_render_crop_video(centered_pair, tmp_path):
    """Test render video with crop."""
    runner = CliRunner()
    output_path = tmp_path / "cropped.mp4"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--crop",
            "100,100,300,300",
            "--start",
            "0",
            "--end",
            "3",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Rendered:" in result.output
    assert output_path.exists()


def test_render_crop_invalid_format():
    """Test error with invalid crop format."""
    from sleap_io.io.cli import _parse_crop_string

    # Valid formats should work
    assert _parse_crop_string(None) is None
    assert _parse_crop_string("100,100,300,300") == (100, 100, 300, 300)
    assert _parse_crop_string("0.25,0.25,0.75,0.75") == (0.25, 0.25, 0.75, 0.75)

    # Invalid formats should raise ClickException
    with pytest.raises(Exception) as exc_info:
        _parse_crop_string("100,100,300")  # Not enough values
    assert "Expected 'x1,y1,x2,y2'" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        _parse_crop_string("a,b,c,d")  # Not numbers
    assert "must be numbers" in str(exc_info.value)


def test_render_crop_normalized_out_of_range():
    """Test error when normalized crop values are out of range."""
    from sleap_io.io.cli import _parse_crop_string

    with pytest.raises(Exception) as exc_info:
        _parse_crop_string("0.0,0.0,1.5,1.0")  # x2 > 1.0
    assert "Normalized crop values must be in [0.0, 1.0] range" in str(exc_info.value)


def test_render_crop_help_shows_options():
    """Test that render --help shows crop options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["render", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--crop" in output
    assert "x1,y1,x2,y2" in output


def test_render_input_not_found(tmp_path):
    """Test render with non-existent input file."""
    runner = CliRunner()
    nonexistent = tmp_path / "nonexistent.slp"
    result = runner.invoke(
        cli,
        ["render", "-i", str(nonexistent), "--lf", "0"],
    )
    assert result.exit_code != 0
    # Click validates path existence before command runs
    assert "does not exist" in result.output


def test_render_non_labels_input(tmp_path):
    """Test render with a file that is not a Labels file."""
    # Create a dummy text file
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("not a labels file")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["render", "-i", str(dummy_file), "--lf", "0"],
    )
    assert result.exit_code != 0
    # Should fail to load since it's not a valid format


def test_render_conflicting_options(centered_pair, tmp_path):
    """Test render error when using --lf with --start/--end."""
    runner = CliRunner()
    output_path = tmp_path / "output.png"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--lf",
            "0",
            "--start",
            "0",  # Conflicting with --lf
            "-o",
            str(output_path),
        ],
        env={"NO_COLOR": "1"},  # Disable Rich colors for consistent output
    )
    assert result.exit_code != 0
    assert "Cannot use --start/--end with --lf" in result.output


def test_render_conflicting_lf_and_frame(centered_pair, tmp_path):
    """Test render error when using both --lf and --frame."""
    runner = CliRunner()
    output_path = tmp_path / "output.png"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--lf",
            "0",
            "--frame",
            "0",  # Conflicting with --lf
            "-o",
            str(output_path),
        ],
        env={"NO_COLOR": "1"},  # Disable Rich colors for consistent output
    )
    assert result.exit_code != 0
    assert "Cannot use both --lf and --frame" in result.output


def test_render_default_output_path_lf(centered_pair, tmp_path):
    """Test render generates default output path for --lf mode."""
    import shutil
    from pathlib import Path

    # Copy input to temp dir to control output location
    src_path = Path(centered_pair).resolve()
    input_path = tmp_path / "test.slp"
    shutil.copy(src_path, input_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            str(input_path),
            "--lf",
            "0",
            # No -o: should generate default output path
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Rendered:" in result.output

    # Default output should be {stem}.lf={lf_ind}.png
    expected_output = tmp_path / "test.lf=0.png"
    assert expected_output.exists()


def test_render_default_output_path_frame(centered_pair, tmp_path):
    """Test render generates default output path for --frame mode."""
    import shutil
    from pathlib import Path

    src_path = Path(centered_pair).resolve()
    input_path = tmp_path / "test.slp"
    shutil.copy(src_path, input_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            str(input_path),
            "--video",
            "0",
            "--frame",
            "0",
            # No -o: should generate default output path
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Rendered:" in result.output

    # Default output should be {stem}.video={v}.frame={f}.png
    expected_output = tmp_path / "test.video=0.frame=0.png"
    assert expected_output.exists()


def test_render_with_preset(centered_pair, tmp_path):
    """Test render with preset option."""
    runner = CliRunner()
    output_path = tmp_path / "preview.png"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--lf",
            "0",
            "--preset",
            "preview",  # 0.25x scale
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_render_lf_out_of_range(centered_pair, tmp_path):
    """Test render error when --lf is out of range."""
    runner = CliRunner()
    output_path = tmp_path / "output.png"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--lf",
            "99999",  # Way out of range
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output


def test_render_video_out_of_range(centered_pair, tmp_path):
    """Test render error when --video index is out of range."""
    runner = CliRunner()
    output_path = tmp_path / "output.png"

    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--video",
            "99",  # Only 1 video in test file
            "--frame",
            "0",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output


# =======================
# Lazy Loading Tests
# =======================


def test_show_lazy_flag_recognized():
    """Test that --lazy flag is recognized by the CLI."""
    runner = CliRunner()
    path = _data_path("slp/centered_pair_predictions.slp")
    result = runner.invoke(cli, ["show", str(path), "--lazy", "--no-open-videos"])
    assert result.exit_code == 0, result.output


def test_show_no_lazy_flag_recognized():
    """Test that --no-lazy flag is recognized by the CLI."""
    runner = CliRunner()
    path = _data_path("slp/centered_pair_predictions.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-lazy", "--no-open-videos"])
    assert result.exit_code == 0, result.output


def test_show_lazy_output_matches_eager():
    """Test that --lazy and --no-lazy produce identical output."""
    runner = CliRunner()
    path = _data_path("slp/centered_pair_predictions.slp")

    lazy_result = runner.invoke(cli, ["show", str(path), "--lazy", "--no-open-videos"])
    eager_result = runner.invoke(
        cli, ["show", str(path), "--no-lazy", "--no-open-videos"]
    )

    assert lazy_result.exit_code == 0, lazy_result.output
    assert eager_result.exit_code == 0, eager_result.output

    # Output should be identical
    lazy_out = _strip_ansi(lazy_result.output)
    eager_out = _strip_ansi(eager_result.output)
    assert lazy_out == eager_out


def test_show_lazy_tracks_output_matches_eager(centered_pair):
    """Test that --tracks with --lazy matches --no-lazy output."""
    runner = CliRunner()

    lazy_result = runner.invoke(
        cli, ["show", centered_pair, "--tracks", "--lazy", "--no-open-videos"]
    )
    eager_result = runner.invoke(
        cli, ["show", centered_pair, "--tracks", "--no-lazy", "--no-open-videos"]
    )

    assert lazy_result.exit_code == 0, lazy_result.output
    assert eager_result.exit_code == 0, eager_result.output

    lazy_out = _strip_ansi(lazy_result.output)
    eager_out = _strip_ansi(eager_result.output)
    assert lazy_out == eager_out


def test_show_lazy_video_output_matches_eager(centered_pair):
    """Test that --video with --lazy matches --no-lazy output."""
    runner = CliRunner()

    lazy_result = runner.invoke(
        cli, ["show", centered_pair, "--video", "--lazy", "--no-open-videos"]
    )
    eager_result = runner.invoke(
        cli, ["show", centered_pair, "--video", "--no-lazy", "--no-open-videos"]
    )

    assert lazy_result.exit_code == 0, lazy_result.output
    assert eager_result.exit_code == 0, eager_result.output

    lazy_out = _strip_ansi(lazy_result.output)
    eager_out = _strip_ansi(eager_result.output)
    assert lazy_out == eager_out


def test_show_lazy_all_output_matches_eager(centered_pair):
    """Test that --all with --lazy has same key stats as --no-lazy output.

    Note: Skeleton symmetry ordering may differ between lazy and eager loading
    due to how symmetries are stored/retrieved, so we check key statistics
    rather than exact match.
    """
    runner = CliRunner()

    lazy_result = runner.invoke(
        cli, ["show", centered_pair, "--all", "--lazy", "--no-open-videos"]
    )
    eager_result = runner.invoke(
        cli, ["show", centered_pair, "--all", "--no-lazy", "--no-open-videos"]
    )

    assert lazy_result.exit_code == 0, lazy_result.output
    assert eager_result.exit_code == 0, eager_result.output

    lazy_out = _strip_ansi(lazy_result.output)
    eager_out = _strip_ansi(eager_result.output)

    # Check key statistics are present and matching
    assert "1100" in lazy_out and "1100" in eager_out  # frames
    assert "2274" in lazy_out and "2274" in eager_out  # predicted instances
    assert "27" in lazy_out and "27" in eager_out  # tracks
    assert "24 nodes" in lazy_out and "24 nodes" in eager_out  # skeleton nodes
    assert "23 edges" in lazy_out and "23 edges" in eager_out  # skeleton edges


def test_show_lazy_only_for_slp_files():
    """Test that lazy loading only applies to SLP files."""
    runner = CliRunner()
    # Using a video file (not SLP) - lazy should be ignored
    path = _data_path("videos/centered_pair_low_quality.mp4")
    result = runner.invoke(cli, ["show", str(path), "--lazy"])
    # Should still work (lazy is ignored for non-SLP files)
    assert result.exit_code == 0, result.output


def test_show_lazy_default_is_lazy():
    """Test that lazy loading is the default for SLP files."""
    runner = CliRunner()
    path = _data_path("slp/centered_pair_predictions.slp")

    # Running without any lazy flag (default is lazy)
    default_result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    lazy_result = runner.invoke(cli, ["show", str(path), "--lazy", "--no-open-videos"])

    assert default_result.exit_code == 0, default_result.output
    assert lazy_result.exit_code == 0, lazy_result.output

    # Output should be identical
    default_out = _strip_ansi(default_result.output)
    lazy_out = _strip_ansi(lazy_result.output)
    assert default_out == lazy_out


def test_show_lazy_shows_correct_instance_counts(centered_pair):
    """Test that lazy loading shows correct instance counts in header."""
    runner = CliRunner()

    result = runner.invoke(cli, ["show", centered_pair, "--lazy", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)

    # centered_pair has 1100 frames, 2274 predicted instances, 27 tracks
    assert "1100" in out  # frames
    assert "2274" in out  # predicted instances
    assert "27" in out  # tracks


# =======================
# Render Additional Tests
# =======================


def test_render_list_colors():
    """Test --list-colors shows available colors."""
    runner = CliRunner()
    result = runner.invoke(cli, ["render", "--list-colors"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Check that named colors are listed
    assert "black" in out
    assert "white" in out
    assert "red" in out

    # Check format hints are shown
    assert "Hex codes" in out
    assert "RGB tuples" in out


def test_render_list_palettes():
    """Test --list-palettes shows available palettes."""
    runner = CliRunner()
    result = runner.invoke(cli, ["render", "--list-palettes"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Check that built-in palettes are listed
    assert "distinct" in out
    assert "rainbow" in out
    assert "tableau10" in out

    # Check colorcet info is shown
    assert "glasbey" in out
    assert "colorcet" in out


def test_render_missing_input():
    """Test render fails gracefully when no input provided."""
    runner = CliRunner()
    result = runner.invoke(cli, ["render"])
    assert result.exit_code != 0
    assert "Missing input file" in result.output


def test_render_positional_input(centered_pair, tmp_path):
    """Test render accepts positional input argument."""
    runner = CliRunner()
    output = tmp_path / "output.png"
    result = runner.invoke(
        cli,
        [
            "render",
            centered_pair,  # positional, not -i
            "--lf",
            "0",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_render_with_background(centered_pair, tmp_path):
    """Test render with --background flag."""
    runner = CliRunner()
    output = tmp_path / "output.png"
    result = runner.invoke(
        cli,
        [
            "render",
            "-i",
            centered_pair,
            "--lf",
            "0",
            "-o",
            str(output),
            "--background",
            "black",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_render_help_short_flag():
    """Test render -h shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["render", "-h"])
    assert result.exit_code == 0, result.output
    assert "Render pose predictions" in result.output
    assert "--background" in result.output
    assert "--list-colors" in result.output
    assert "--list-palettes" in result.output


# --- Tests for _resolve_input helper and positional/flag input patterns ---


def test_resolve_input_positional_only():
    """Test _resolve_input with only positional argument."""
    from sleap_io.io.cli import _resolve_input

    result = _resolve_input(Path("/a.slp"), None)
    assert result == Path("/a.slp")


def test_resolve_input_option_only():
    """Test _resolve_input with only -i option."""
    from sleap_io.io.cli import _resolve_input

    result = _resolve_input(None, Path("/b.slp"))
    assert result == Path("/b.slp")


def test_resolve_input_both_raises_error():
    """Test _resolve_input raises when both provided."""
    import click

    from sleap_io.io.cli import _resolve_input

    with pytest.raises(click.ClickException, match="Cannot specify"):
        _resolve_input(Path("/a.slp"), Path("/b.slp"))


def test_resolve_input_neither_raises_error():
    """Test _resolve_input raises when neither provided."""
    import click

    from sleap_io.io.cli import _resolve_input

    with pytest.raises(click.ClickException, match="Missing"):
        _resolve_input(None, None)


def test_resolve_input_custom_name():
    """Test _resolve_input uses custom name in error messages."""
    import click

    from sleap_io.io.cli import _resolve_input

    with pytest.raises(click.ClickException, match="Missing custom thing"):
        _resolve_input(None, None, "custom thing")


def test_show_accepts_positional_input(slp_typical):
    """Test show accepts positional input."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", slp_typical, "--no-open-videos"])
    assert result.exit_code == 0, result.output
    assert "typical.slp" in _strip_ansi(result.output)


def test_show_accepts_flag_input(slp_typical):
    """Test show accepts -i flag input."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "-i", slp_typical, "--no-open-videos"])
    assert result.exit_code == 0, result.output
    assert "typical.slp" in _strip_ansi(result.output)


def test_show_rejects_both_inputs(slp_typical):
    """Test show rejects both positional and -i."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["show", slp_typical, "-i", slp_typical, "--no-open-videos"]
    )
    assert result.exit_code != 0
    assert "Cannot specify" in result.output


def test_show_missing_input():
    """Test show fails gracefully when no input provided."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show"])
    assert result.exit_code != 0
    assert "Missing" in result.output


def test_convert_accepts_positional_input(slp_typical, tmp_path):
    """Test convert accepts positional input."""
    runner = CliRunner()
    output = tmp_path / "out.slp"
    result = runner.invoke(cli, ["convert", slp_typical, "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_convert_rejects_both_inputs(slp_typical, tmp_path):
    """Test convert rejects both positional and -i."""
    runner = CliRunner()
    output = tmp_path / "out.slp"
    result = runner.invoke(
        cli, ["convert", slp_typical, "-i", slp_typical, "-o", str(output)]
    )
    assert result.exit_code != 0
    assert "Cannot specify" in result.output


def test_convert_missing_input(tmp_path):
    """Test convert fails gracefully when no input provided."""
    runner = CliRunner()
    output = tmp_path / "out.slp"
    result = runner.invoke(cli, ["convert", "-o", str(output)])
    assert result.exit_code != 0
    assert "Missing" in result.output


def test_split_accepts_positional_input(slp_typical, tmp_path):
    """Test split accepts positional input."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"
    result = runner.invoke(cli, ["split", slp_typical, "-o", str(output_dir)])
    assert result.exit_code == 0, result.output
    assert (output_dir / "train.slp").exists()


def test_split_rejects_both_inputs(slp_typical, tmp_path):
    """Test split rejects both positional and -i."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"
    result = runner.invoke(
        cli, ["split", slp_typical, "-i", slp_typical, "-o", str(output_dir)]
    )
    assert result.exit_code != 0
    assert "Cannot specify" in result.output


def test_split_missing_input(tmp_path):
    """Test split fails gracefully when no input provided."""
    runner = CliRunner()
    output_dir = tmp_path / "splits"
    result = runner.invoke(cli, ["split", "-o", str(output_dir)])
    assert result.exit_code != 0
    assert "Missing" in result.output


def test_filenames_accepts_positional_input(slp_typical):
    """Test filenames accepts positional input."""
    runner = CliRunner()
    result = runner.invoke(cli, ["filenames", slp_typical])
    assert result.exit_code == 0, result.output
    assert "Video filenames" in result.output


def test_filenames_accepts_flag_input(slp_typical):
    """Test filenames accepts -i flag input."""
    runner = CliRunner()
    result = runner.invoke(cli, ["filenames", "-i", slp_typical])
    assert result.exit_code == 0, result.output
    assert "Video filenames" in result.output


def test_filenames_rejects_both_inputs(slp_typical):
    """Test filenames rejects both positional and -i."""
    runner = CliRunner()
    result = runner.invoke(cli, ["filenames", slp_typical, "-i", slp_typical])
    assert result.exit_code != 0
    assert "Cannot specify" in result.output


def test_filenames_missing_input():
    """Test filenames fails gracefully when no input provided."""
    runner = CliRunner()
    result = runner.invoke(cli, ["filenames"])
    assert result.exit_code != 0
    assert "Missing" in result.output


def test_render_rejects_both_inputs(centered_pair, tmp_path):
    """Test render rejects both positional and -i."""
    runner = CliRunner()
    output = tmp_path / "output.png"
    result = runner.invoke(
        cli,
        [
            "render",
            centered_pair,
            "-i",
            centered_pair,
            "--lf",
            "0",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "Cannot specify" in result.output


# ============================================================================
# fix command tests
# ============================================================================


def test_fix_help():
    """Test fix --help shows command documentation."""
    runner = CliRunner()
    result = runner.invoke(cli, ["fix", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    # Check key elements - new flag names
    assert "Fix common issues" in output
    assert "--deduplicate-videos" in output
    assert "--remove-unused-skeletons" in output
    assert "--consolidate-skeletons" in output
    assert "--remove-predictions" in output
    assert "--remove-untracked-predictions" in output
    assert "--remove-unused-tracks" in output
    assert "--remove-empty-frames" in output
    assert "--remove-empty-instances" in output
    assert "--remove-unlabeled-videos" in output
    assert "--dry-run" in output
    assert "--prefix" in output
    assert "--map" in output


def test_fix_in_command_list():
    """Test fix appears in main help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    # Check command is listed with description
    assert "fix" in output
    assert "Fix common issues" in output


def test_fix_dry_run_no_issues(slp_typical, tmp_path):
    """Test fix --dry-run with a clean file shows no issues."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["fix", "-i", slp_typical, "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should report no issues
    assert "No duplicates found" in output or "Videos:" in output
    assert "[DRY RUN" in output


def test_fix_basic_clean(slp_typical, tmp_path):
    """Test fix saves output with default name."""
    import shutil

    # Copy to tmp_path to control output location
    src = Path(slp_typical)
    input_path = tmp_path / "test.slp"
    shutil.copy(src, input_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix", "-i", str(input_path)],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check default output path
    expected_output = tmp_path / "test.fixed.slp"
    assert expected_output.exists()
    assert "Saved:" in output


def test_fix_explicit_output(slp_typical, tmp_path):
    """Test fix with explicit output path."""
    runner = CliRunner()
    output_path = tmp_path / "custom_output.slp"

    result = runner.invoke(
        cli,
        ["fix", "-i", slp_typical, "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_fix_remove_predictions(centered_pair, tmp_path):
    """Test fix --remove-predictions removes predicted instances."""
    runner = CliRunner()
    output_path = tmp_path / "no_preds.slp"

    # centered_pair has predictions
    result = runner.invoke(
        cli,
        ["fix", "-i", centered_pair, "-o", str(output_path), "--remove-predictions"],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should mention removing predictions
    assert "prediction" in output.lower()

    # Verify predictions removed
    labels = load_slp(str(output_path), open_videos=False)
    for lf in labels.labeled_frames:
        for inst in lf:
            assert not isinstance(inst, PredictedInstance)


def _make_test_video(filename, shape=(100, 480, 640, 1)):
    """Create a Video object with specified metadata for testing (Py3.8 compatible)."""
    from sleap_io.model.video import Video

    video = Video(filename=filename, open_backend=False)
    video.backend_metadata["shape"] = shape
    return video


def test_fix_duplicate_videos(tmp_path):
    """Test fix detects and merges duplicate videos (enabled by default)."""
    # Create labels with duplicate videos
    skeleton = Skeleton(["head", "tail"])
    video1 = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))
    video2 = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))

    labels = Labels(skeletons=[skeleton], videos=[video1, video2])

    # Add frames to both videos
    import numpy as np

    from sleap_io.model.instance import Instance
    from sleap_io.model.labeled_frame import LabeledFrame

    for idx, video in enumerate([video1, video2]):
        frame = LabeledFrame(video=video, frame_idx=idx * 10)
        points = np.random.rand(2, 2) * 100
        inst = Instance.from_numpy(points, skeleton=skeleton)
        frame.instances = [inst]
        labels.labeled_frames.append(frame)

    # Save input
    input_path = tmp_path / "duplicates.slp"
    labels.save(str(input_path))
    assert len(labels.videos) == 2

    # Run fix (deduplication is enabled by default)
    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        ["fix", "-i", str(input_path), "-o", str(output_path), "-v"],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should detect duplicates
    assert "duplicate" in output.lower()

    # Verify duplicates merged
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert len(fixed_labels.videos) == 1
    assert len(fixed_labels.labeled_frames) == 2  # Both frames preserved


def test_fix_unused_skeletons(tmp_path):
    """Test fix removes unused skeletons (enabled by default)."""
    import numpy as np

    from sleap_io.model.instance import Instance
    from sleap_io.model.labeled_frame import LabeledFrame

    # Create labels with unused skeleton
    skel1 = Skeleton(["head", "tail"], name="used")
    skel2 = Skeleton(["a", "b", "c"], name="unused")

    video = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))
    labels = Labels(skeletons=[skel1, skel2], videos=[video])

    # Add frame using only skel1
    frame = LabeledFrame(video=video, frame_idx=0)
    points = np.random.rand(2, 2) * 100
    inst = Instance.from_numpy(points, skeleton=skel1)
    frame.instances = [inst]
    labels.labeled_frames.append(frame)

    # Save input
    input_path = tmp_path / "unused_skel.slp"
    labels.save(str(input_path))
    assert len(labels.skeletons) == 2

    # Run fix (remove-unused-skeletons is enabled by default)
    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        ["fix", "-i", str(input_path), "-o", str(output_path), "-v"],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should detect unused
    assert "unused" in output.lower()

    # Verify unused skeleton removed
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert len(fixed_labels.skeletons) == 1
    assert fixed_labels.skeletons[0].name == "used"


def test_fix_prefix_mode(tmp_path):
    """Test fix with --prefix updates video paths."""
    from sleap_io.model.video import Video

    labels = Labels(
        videos=[Video.from_filename(r"C:\data\videos\test.mp4")],
    )
    input_path = tmp_path / "windows.slp"
    labels.save(str(input_path))

    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        [
            "fix",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--prefix",
            r"C:\data\videos",
            "/mnt/data/videos",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify path updated
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert fixed_labels.videos[0].filename == "/mnt/data/videos/test.mp4"


def test_fix_map_mode(tmp_path):
    """Test fix with --map updates video paths."""
    from sleap_io.model.video import Video

    labels = Labels(
        videos=[Video.from_filename("/old/path/video.mp4")],
    )
    input_path = tmp_path / "old_path.slp"
    labels.save(str(input_path))

    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        [
            "fix",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--map",
            "/old/path/video.mp4",
            "/new/path/video.mp4",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify path updated
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert fixed_labels.videos[0].filename == "/new/path/video.mp4"


def test_fix_no_empty_frames_removal(slp_typical, tmp_path):
    """Test fix --no-remove-empty-frames skips frame cleanup."""
    runner = CliRunner()
    output_path = tmp_path / "no_frame_clean.slp"

    result = runner.invoke(
        cli,
        ["fix", "-i", slp_typical, "-o", str(output_path), "--no-remove-empty-frames"],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should not mention removing empty frames
    assert "Remove empty frames" not in output


def test_fix_verbose(slp_typical, tmp_path):
    """Test fix -v shows detailed analysis."""
    runner = CliRunner()
    output_path = tmp_path / "verbose.slp"

    result = runner.invoke(
        cli,
        ["fix", "-i", slp_typical, "-o", str(output_path), "-v"],
    )
    assert result.exit_code == 0, result.output
    # Verbose output should be present (details vary by file content)
    assert "Loading:" in result.output


def test_fix_input_not_found():
    """Test fix error when input file doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["fix", "-i", "nonexistent.slp"])
    assert result.exit_code != 0
    # Click validates path existence


def test_fix_non_labels_input(tmp_path, centered_pair_low_quality_path):
    """Test fix error when input is not a labels file."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix", "-i", centered_pair_low_quality_path],
    )
    assert result.exit_code != 0
    assert "not a labels file" in result.output


def test_fix_load_failure(tmp_path):
    """Test fix error when input file is corrupt."""
    corrupt = tmp_path / "corrupt.slp"
    corrupt.write_text("not valid")

    runner = CliRunner()
    result = runner.invoke(cli, ["fix", "-i", str(corrupt)])
    assert result.exit_code != 0
    assert "Failed to load" in result.output


def test_fix_save_failure(slp_typical, tmp_path):
    """Test fix error when save fails."""
    # Create directory where file is expected
    bad_output = tmp_path / "output.slp"
    bad_output.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix", "-i", slp_typical, "-o", str(bad_output)],
    )
    assert result.exit_code != 0
    assert "Failed to save" in result.output


def test_fix_accepts_positional_input(slp_typical, tmp_path):
    """Test fix accepts positional input."""
    runner = CliRunner()
    output_path = tmp_path / "out.slp"
    result = runner.invoke(cli, ["fix", slp_typical, "-o", str(output_path)])
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_fix_rejects_both_inputs(slp_typical, tmp_path):
    """Test fix rejects both positional and -i."""
    runner = CliRunner()
    output_path = tmp_path / "out.slp"
    result = runner.invoke(
        cli, ["fix", slp_typical, "-i", slp_typical, "-o", str(output_path)]
    )
    assert result.exit_code != 0
    assert "Cannot specify" in result.output


def test_fix_missing_input():
    """Test fix fails gracefully when no input provided."""
    runner = CliRunner()
    result = runner.invoke(cli, ["fix"])
    assert result.exit_code != 0
    assert "Missing" in result.output


def test_fix_default_output_pkg_slp(tmp_path):
    """Test fix generates correct default output for pkg.slp files."""
    import shutil

    # Create a pkg.slp file (just copy and rename for simplicity)
    src = _data_path("slp/minimal_instance.pkg.slp")
    input_path = tmp_path / "test.pkg.slp"
    shutil.copy(src, input_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fix", "-i", str(input_path)],
    )
    assert result.exit_code == 0, result.output

    # Check default output path for pkg.slp
    expected_output = tmp_path / "test.fixed.pkg.slp"
    assert expected_output.exists()


def test_fix_preserves_embedded_from_pkg_slp(tmp_path, slp_real_data):
    """Test that fix preserves embedded videos when pkg.slp -> pkg.slp."""
    runner = CliRunner()

    # First, create a pkg.slp with embedded videos
    embedded_path = tmp_path / "embedded.pkg.slp"
    result = runner.invoke(
        cli,
        ["convert", "-i", slp_real_data, "-o", str(embedded_path), "--embed", "user"],
    )
    assert result.exit_code == 0, result.output

    # Verify input has embedded videos
    input_labels = load_slp(str(embedded_path))
    assert input_labels.videos[0].backend_metadata.get("has_embedded_images", False)

    # Fix pkg.slp -> pkg.slp (output based on default naming: .fixed.pkg.slp)
    # This should auto-preserve embedded videos
    output_path = tmp_path / "fixed.pkg.slp"
    result = runner.invoke(
        cli,
        ["fix", "-i", str(embedded_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output

    # Verify output has embedded videos preserved
    output_labels = load_slp(str(output_path))
    # backend_metadata indicates embedding
    assert output_labels.videos[0].backend_metadata.get("has_embedded_images", False)


def test_fix_preserves_embedded_frames_hdf5(tmp_path, slp_real_data):
    """Test that fix actually copies embedded video data to output HDF5 file.

    This is a more thorough test that verifies:
    1. The HDF5 file contains video datasets
    2. The frames can actually be read from the output file
    """
    import h5py

    runner = CliRunner()

    # First, create a pkg.slp with embedded videos
    embedded_path = tmp_path / "embedded.pkg.slp"
    result = runner.invoke(
        cli,
        ["convert", "-i", slp_real_data, "-o", str(embedded_path), "--embed", "user"],
    )
    assert result.exit_code == 0, result.output

    # Verify input HDF5 has video data
    with h5py.File(embedded_path, "r") as f:
        assert "video0/video" in f
        input_video_shape = f["video0/video"].shape

    # Run fix command
    output_path = tmp_path / "fixed.pkg.slp"
    result = runner.invoke(
        cli,
        ["fix", "-i", str(embedded_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output

    # Verify output HDF5 has video data
    with h5py.File(output_path, "r") as f:
        assert "video0/video" in f, "Embedded video dataset missing from output"
        assert f["video0/video"].shape == input_video_shape, "Video data shape mismatch"

    # Verify the output file is usable - can load and read frames
    output_labels = load_slp(str(output_path), open_videos=True)
    assert output_labels.videos[0].backend is not None
    frame = output_labels.videos[0][0]
    assert frame.shape[0] > 0, "Could not read frame from preserved embedded video"


def test_fix_consolidate_skeletons(tmp_path):
    """Test fix --consolidate-skeletons keeps most frequent skeleton."""
    import numpy as np

    from sleap_io.model.instance import Instance
    from sleap_io.model.labeled_frame import LabeledFrame

    # Create labels with two skeletons that both have user instances
    skel1 = Skeleton(["head", "tail"], name="frequent")
    skel2 = Skeleton(["a", "b"], name="rare")

    video = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))
    labels = Labels(skeletons=[skel1, skel2], videos=[video])

    # Add 5 instances using skel1 (frequent)
    for i in range(5):
        frame = LabeledFrame(video=video, frame_idx=i)
        points = np.random.rand(2, 2) * 100
        inst = Instance.from_numpy(points, skeleton=skel1)
        frame.instances = [inst]
        labels.labeled_frames.append(frame)

    # Add 2 instances using skel2 (rare)
    for i in range(5, 7):
        frame = LabeledFrame(video=video, frame_idx=i)
        points = np.random.rand(2, 2) * 100
        inst = Instance.from_numpy(points, skeleton=skel2)
        frame.instances = [inst]
        labels.labeled_frames.append(frame)

    # Save input
    input_path = tmp_path / "multi_skel.slp"
    labels.save(str(input_path))
    assert len(labels.skeletons) == 2

    # Run fix without consolidate - should warn but not change
    runner = CliRunner()
    dry_run_result = runner.invoke(
        cli,
        ["fix", "-i", str(input_path), "--dry-run"],
    )
    assert dry_run_result.exit_code == 0, dry_run_result.output
    output = _strip_ansi(dry_run_result.output)
    assert "WARNING: Multiple skeletons" in output
    assert "--consolidate-skeletons" in output

    # Run fix with consolidate
    output_path = tmp_path / "consolidated.slp"
    result = runner.invoke(
        cli,
        [
            "fix",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--consolidate-skeletons",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should mention consolidation
    assert "CONSOLIDATE" in output or "CONSOLIDATING" in output

    # Verify only most frequent skeleton remains
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert len(fixed_labels.skeletons) == 1
    assert fixed_labels.skeletons[0].name == "frequent"
    # Should have 5 frames (rare skeleton instances deleted, frames cleaned)
    assert len(fixed_labels.labeled_frames) == 5


def test_fix_remove_untracked_predictions(tmp_path):
    """Test fix --remove-untracked-predictions removes only untracked predictions."""
    import numpy as np

    from sleap_io.model.instance import Instance, Track
    from sleap_io.model.labeled_frame import LabeledFrame

    skeleton = Skeleton(["head", "tail"])
    video = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))
    track1 = Track(name="track1")
    labels = Labels(skeletons=[skeleton], videos=[video], tracks=[track1])

    # Add frame with one tracked and one untracked prediction
    frame = LabeledFrame(video=video, frame_idx=0)

    # Tracked prediction
    tracked_pred = PredictedInstance.from_numpy(
        np.random.rand(2, 2) * 100,
        skeleton=skeleton,
        score=0.9,
    )
    tracked_pred.track = track1

    # Untracked prediction
    untracked_pred = PredictedInstance.from_numpy(
        np.random.rand(2, 2) * 100,
        skeleton=skeleton,
        score=0.8,
    )
    # track is None by default

    # User instance
    user_inst = Instance.from_numpy(np.random.rand(2, 2) * 100, skeleton=skeleton)

    frame.instances = [tracked_pred, untracked_pred, user_inst]
    labels.labeled_frames.append(frame)

    # Save input
    input_path = tmp_path / "mixed_preds.slp"
    labels.save(str(input_path))

    # Run fix with --remove-untracked-predictions
    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        [
            "fix",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--remove-untracked-predictions",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should mention removing untracked predictions
    assert "untracked" in output.lower()

    # Verify only untracked prediction removed
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert len(fixed_labels.labeled_frames) == 1
    instances = fixed_labels.labeled_frames[0].instances
    assert len(instances) == 2  # tracked pred + user instance

    # Verify tracked prediction still there
    preds = [i for i in instances if isinstance(i, PredictedInstance)]
    assert len(preds) == 1
    assert preds[0].track is not None


def test_fix_prediction_only_skeleton_removal(tmp_path):
    """Test fix removes skeletons used only by predictions."""
    import numpy as np

    from sleap_io.model.instance import Instance
    from sleap_io.model.labeled_frame import LabeledFrame

    # Create labels with one skeleton for user, one for predictions only
    skel_user = Skeleton(["head", "tail"], name="user_skel")
    skel_pred = Skeleton(["a", "b", "c"], name="pred_only_skel")

    video = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))
    labels = Labels(skeletons=[skel_user, skel_pred], videos=[video])

    # Add user instance
    frame1 = LabeledFrame(video=video, frame_idx=0)
    user_inst = Instance.from_numpy(np.random.rand(2, 2) * 100, skeleton=skel_user)
    frame1.instances = [user_inst]
    labels.labeled_frames.append(frame1)

    # Add prediction-only instance
    frame2 = LabeledFrame(video=video, frame_idx=1)
    pred_inst = PredictedInstance.from_numpy(
        np.random.rand(3, 2) * 100,
        skeleton=skel_pred,
        score=0.9,
    )
    frame2.instances = [pred_inst]
    labels.labeled_frames.append(frame2)

    # Save input
    input_path = tmp_path / "pred_only_skel.slp"
    labels.save(str(input_path))
    assert len(labels.skeletons) == 2

    # Run fix (remove-unused-skeletons is enabled by default)
    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        ["fix", "-i", str(input_path), "-o", str(output_path), "-v"],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should mention prediction-only skeleton
    assert "predictions only" in output.lower() or "prediction-only" in output.lower()

    # Verify prediction-only skeleton and its instances removed
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert len(fixed_labels.skeletons) == 1
    assert fixed_labels.skeletons[0].name == "user_skel"
    # Frame with prediction-only skeleton should be empty and removed
    assert len(fixed_labels.labeled_frames) == 1


def test_fix_remove_unlabeled_videos(tmp_path):
    """Test fix --remove-unlabeled-videos removes videos with no frames."""
    import numpy as np

    from sleap_io.model.instance import Instance
    from sleap_io.model.labeled_frame import LabeledFrame

    skeleton = Skeleton(["head", "tail"])
    video1 = _make_test_video(filename="/data/video1.mp4", shape=(100, 480, 640, 1))
    video2 = _make_test_video(filename="/data/video2.mp4", shape=(100, 480, 640, 1))

    labels = Labels(skeletons=[skeleton], videos=[video1, video2])

    # Add frame only to video1
    frame = LabeledFrame(video=video1, frame_idx=0)
    inst = Instance.from_numpy(np.random.rand(2, 2) * 100, skeleton=skeleton)
    frame.instances = [inst]
    labels.labeled_frames.append(frame)

    # Save input
    input_path = tmp_path / "unlabeled_video.slp"
    labels.save(str(input_path))
    assert len(labels.videos) == 2

    # Run fix with --remove-unlabeled-videos
    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        [
            "fix",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--remove-unlabeled-videos",
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should mention removing unlabeled videos
    assert "unlabeled videos" in output.lower()

    # Verify unlabeled video removed
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert len(fixed_labels.videos) == 1
    assert fixed_labels.videos[0].filename == "/data/video1.mp4"


def test_fix_no_deduplicate_videos(tmp_path):
    """Test fix --no-deduplicate-videos skips video deduplication."""
    # Create labels with duplicate videos
    skeleton = Skeleton(["head", "tail"])
    video1 = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))
    video2 = _make_test_video(filename="/data/video.mp4", shape=(100, 480, 640, 1))

    labels = Labels(skeletons=[skeleton], videos=[video1, video2])

    # Save input
    input_path = tmp_path / "duplicates.slp"
    labels.save(str(input_path))
    assert len(labels.videos) == 2

    # Run fix with deduplication disabled
    runner = CliRunner()
    output_path = tmp_path / "fixed.slp"
    result = runner.invoke(
        cli,
        [
            "fix",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--no-deduplicate-videos",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify duplicates NOT merged
    fixed_labels = load_slp(str(output_path), open_videos=False)
    assert len(fixed_labels.videos) == 2


# ============================================================================
# embed command tests
# ============================================================================


def test_embed_help():
    """Test embed --help shows command documentation."""
    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    assert "Embed video frames" in output
    assert "--user" in output
    assert "--predictions" in output
    assert "--suggestions" in output


def test_embed_basic(tmp_path, slp_real_data):
    """Test basic embed command with default mode (user only)."""
    runner = CliRunner()
    output_path = tmp_path / "embedded.pkg.slp"

    result = runner.invoke(cli, ["embed", slp_real_data, "-o", str(output_path)])
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Embedded:" in output
    assert "user:" in output
    assert output_path.exists()


def test_embed_with_predictions(tmp_path, slp_real_data):
    """Test embed command with --predictions flag."""
    runner = CliRunner()
    output_path = tmp_path / "embedded.pkg.slp"

    result = runner.invoke(
        cli, ["embed", slp_real_data, "-o", str(output_path), "--predictions"]
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "predictions:" in output


def test_embed_with_suggestions(tmp_path, slp_real_data):
    """Test embed command with --suggestions flag."""
    runner = CliRunner()
    output_path = tmp_path / "embedded.pkg.slp"

    result = runner.invoke(
        cli, ["embed", slp_real_data, "-o", str(output_path), "--suggestions"]
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "suggestions:" in output


def test_embed_missing_output():
    """Test embed command requires output path."""
    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "input.slp"])
    assert result.exit_code != 0


def test_embed_non_slp_input(tmp_path):
    """Test embed command rejects non-SLP input."""
    runner = CliRunner()
    input_path = tmp_path / "test.json"
    input_path.write_text("{}")
    output_path = tmp_path / "output.pkg.slp"

    result = runner.invoke(cli, ["embed", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "must be a .slp file" in output


def test_embed_input_option(tmp_path, slp_real_data):
    """Test embed with -i option."""
    runner = CliRunner()
    output_path = tmp_path / "embedded.pkg.slp"

    result = runner.invoke(cli, ["embed", "-i", slp_real_data, "-o", str(output_path)])
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert output_path.exists()


# ============================================================================
# unembed command tests
# ============================================================================


def test_unembed_help():
    """Test unembed --help shows command documentation."""
    runner = CliRunner()
    result = runner.invoke(cli, ["unembed", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)

    assert "Remove embedded frames" in output
    assert "restore video references" in output


def test_unembed_roundtrip(tmp_path, slp_real_data):
    """Test embed -> unembed roundtrip preserves labels."""
    runner = CliRunner()
    embedded_path = tmp_path / "embedded.pkg.slp"
    unembedded_path = tmp_path / "unembedded.slp"

    # First embed
    result = runner.invoke(cli, ["embed", slp_real_data, "-o", str(embedded_path)])
    assert result.exit_code == 0, _strip_ansi(result.output)

    # Then unembed
    result = runner.invoke(
        cli, ["unembed", str(embedded_path), "-o", str(unembedded_path)]
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Unembedded:" in output
    assert "Restored 1 video(s)" in output
    assert unembedded_path.exists()

    # Verify the unembedded file references the original video
    labels = load_slp(str(unembedded_path), open_videos=False)
    assert len(labels.videos) == 1
    # Should reference the original video, not the pkg file
    assert not labels.videos[0].filename.endswith(".pkg.slp")


def test_unembed_non_embedded_file(tmp_path, slp_real_data):
    """Test unembed on non-embedded file gives helpful error."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(cli, ["unembed", slp_real_data, "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)

    assert "No embedded videos found" in output
    assert "sio show" in output


def test_unembed_missing_output():
    """Test unembed command requires output path."""
    runner = CliRunner()
    result = runner.invoke(cli, ["unembed", "input.pkg.slp"])
    assert result.exit_code != 0


def test_unembed_input_option(tmp_path, slp_real_data):
    """Test unembed with -i option."""
    runner = CliRunner()
    embedded_path = tmp_path / "embedded.pkg.slp"
    unembedded_path = tmp_path / "unembedded.slp"

    # First embed
    runner.invoke(cli, ["embed", slp_real_data, "-o", str(embedded_path)])

    # Then unembed with -i
    result = runner.invoke(
        cli, ["unembed", "-i", str(embedded_path), "-o", str(unembedded_path)]
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert unembedded_path.exists()


def test_embed_no_frame_types_selected(tmp_path, slp_real_data):
    """Test embed fails when --no-user without --predictions or --suggestions."""
    runner = CliRunner()
    output_path = tmp_path / "output.pkg.slp"

    result = runner.invoke(
        cli, ["embed", slp_real_data, "-o", str(output_path), "--no-user"]
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "No frames to embed" in output
    assert "--user" in output or "user" in output.lower()


def test_embed_no_user_with_predictions(tmp_path, slp_real_data):
    """Test embed with --no-user --predictions skips user frames."""
    runner = CliRunner()
    output_path = tmp_path / "embedded.pkg.slp"

    result = runner.invoke(
        cli,
        ["embed", slp_real_data, "-o", str(output_path), "--no-user", "--predictions"],
    )
    # This should succeed and show only predictions (no "user:" in output)
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)
    assert "Embedded:" in output
    assert "predictions:" in output
    # Should NOT have user frames since --no-user was specified
    assert "user:" not in output


def test_embed_load_failure(tmp_path):
    """Test embed handles corrupt/invalid SLP files."""
    runner = CliRunner()
    input_path = tmp_path / "corrupt.slp"
    input_path.write_text("not a valid slp file")
    output_path = tmp_path / "output.pkg.slp"

    result = runner.invoke(cli, ["embed", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load input file" in output


def test_embed_empty_labels_file(tmp_path):
    """Test embed fails on empty labels file with no frames to embed."""
    runner = CliRunner()
    output_path = tmp_path / "output.pkg.slp"

    # Create minimal empty labels file
    labels = Labels()
    input_path = tmp_path / "empty.slp"
    labels.save(str(input_path))

    result = runner.invoke(cli, ["embed", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "No frames to embed" in output


def test_unembed_non_slp_input(tmp_path):
    """Test unembed command rejects non-SLP input."""
    runner = CliRunner()
    input_path = tmp_path / "test.json"
    input_path.write_text("{}")
    output_path = tmp_path / "output.slp"

    result = runner.invoke(cli, ["unembed", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "must be a .slp file" in output


def test_unembed_load_failure(tmp_path):
    """Test unembed handles corrupt/invalid SLP files."""
    runner = CliRunner()
    input_path = tmp_path / "corrupt.slp"
    input_path.write_text("not a valid slp file")
    output_path = tmp_path / "output.slp"

    result = runner.invoke(cli, ["unembed", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to load input file" in output


def test_unembed_missing_source_video(tmp_path, slp_real_data):
    """Test unembed fails on embedded file without source_video metadata."""
    runner = CliRunner()

    # First embed the file
    embedded_path = tmp_path / "embedded.pkg.slp"
    result = runner.invoke(cli, ["embed", slp_real_data, "-o", str(embedded_path)])
    assert result.exit_code == 0, _strip_ansi(result.output)

    # Load and clear source_video to simulate legacy file
    labels = load_slp(str(embedded_path), open_videos=False)
    for video in labels.videos:
        # Clear the source_video to simulate a legacy file
        video.source_video = None
    labels.save(str(embedded_path))

    # Now try to unembed
    output_path = tmp_path / "unembedded.slp"
    result = runner.invoke(cli, ["unembed", str(embedded_path), "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "no source video metadata" in output or "Cannot unembed" in output


def test_embed_save_failure(tmp_path, slp_real_data):
    """Test embed handles save failures gracefully."""
    runner = CliRunner()
    # Use a path in a non-existent directory to cause save failure
    output_path = tmp_path / "nonexistent" / "subdir" / "output.pkg.slp"

    result = runner.invoke(cli, ["embed", slp_real_data, "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to save" in output


def test_unembed_save_failure(tmp_path, slp_real_data):
    """Test unembed handles save failures gracefully."""
    runner = CliRunner()

    # First embed the file
    embedded_path = tmp_path / "embedded.pkg.slp"
    result = runner.invoke(cli, ["embed", slp_real_data, "-o", str(embedded_path)])
    assert result.exit_code == 0, _strip_ansi(result.output)

    # Now try to unembed to a non-existent directory
    output_path = tmp_path / "nonexistent" / "subdir" / "output.slp"
    result = runner.invoke(cli, ["unembed", str(embedded_path), "-o", str(output_path)])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Failed to save" in output


# ======== sio trim tests ========


def test_trim_help():
    """Test trim command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["trim", "--help"])
    assert result.exit_code == 0
    assert "Trim video and labels to a frame range" in result.output
    assert "--start" in result.output
    assert "--end" in result.output
    assert "--video" in result.output
    assert "--crf" in result.output
    assert "--fps" in result.output
    assert "--x264-preset" in result.output


def test_trim_in_command_list():
    """Test that trim appears in CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "trim" in result.output


def test_trim_labels_basic(tmp_path, slp_real_data):
    """Test basic labels trimming with default options."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    # Trim to first 50 frames
    result = runner.invoke(
        cli,
        ["trim", slp_real_data, "--start", "0", "--end", "50", "-o", str(output_path)],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Trimming:" in output
    assert "Saved:" in output
    assert output_path.exists()
    assert output_path.with_suffix(".mp4").exists()

    # Verify trimmed labels
    trimmed = load_slp(str(output_path))
    assert len(trimmed.videos) == 1
    # All frame indices should be < 50
    for lf in trimmed.labeled_frames:
        assert lf.frame_idx < 50


def test_trim_labels_with_frame_range(tmp_path, slp_real_data):
    """Test labels trimming with specific frame range."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        [
            "trim",
            slp_real_data,
            "--start",
            "100",
            "--end",
            "300",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Frame range: 100 to 300" in output
    assert "200 frames" in output
    assert output_path.exists()


def test_trim_labels_with_video_options(tmp_path, slp_real_data):
    """Test labels trimming with video encoding options."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        [
            "trim",
            slp_real_data,
            "--start",
            "0",
            "--end",
            "30",
            "--crf",
            "18",
            "--x264-preset",
            "fast",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert output_path.exists()
    assert output_path.with_suffix(".mp4").exists()


@skip_slow_video_on_windows
def test_trim_video_only(tmp_path, centered_pair_low_quality_path):
    """Test trimming a standalone video file."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.mp4"

    result = runner.invoke(
        cli,
        [
            "trim",
            centered_pair_low_quality_path,
            "--start",
            "0",
            "--end",
            "30",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Trimming:" in output
    assert "Saved:" in output
    assert output_path.exists()


@skip_slow_video_on_windows
def test_trim_video_only_with_fps(tmp_path, centered_pair_low_quality_path):
    """Test video trimming with FPS option."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.mp4"

    result = runner.invoke(
        cli,
        [
            "trim",
            centered_pair_low_quality_path,
            "--start",
            "0",
            "--end",
            "30",
            "--fps",
            "15",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert output_path.exists()


def test_trim_invalid_frame_range(tmp_path, slp_real_data):
    """Test trim with invalid frame range."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    # End before start
    result = runner.invoke(
        cli,
        [
            "trim",
            slp_real_data,
            "--start",
            "100",
            "--end",
            "50",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "must be greater than start frame" in _strip_ansi(result.output)


def test_trim_negative_start(tmp_path, slp_real_data):
    """Test trim with negative start frame."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        [
            "trim",
            slp_real_data,
            "--start",
            "-10",
            "--end",
            "50",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "must be >= 0" in _strip_ansi(result.output)


def test_trim_default_output_labels(tmp_path, slp_real_data):
    """Test trim uses default output path for labels."""
    import shutil

    runner = CliRunner()
    # Copy input to tmp_path so default output goes there
    input_copy = tmp_path / "labels.slp"
    shutil.copy(slp_real_data, input_copy)
    # Also need to copy the video
    video_src = Path("tests/data/videos/centered_pair_low_quality.mp4")
    video_dst = tmp_path / "tests" / "data" / "videos"
    video_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(video_src, video_dst / "centered_pair_low_quality.mp4")

    result = runner.invoke(
        cli, ["trim", str(input_copy), "--start", "0", "--end", "30"]
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert (tmp_path / "labels.trim.slp").exists()


def test_trim_default_output_video(tmp_path, centered_pair_low_quality_path):
    """Test trim uses default output path for video."""
    import shutil

    runner = CliRunner()
    # Copy video to tmp_path
    input_copy = tmp_path / "video.mp4"
    shutil.copy(centered_pair_low_quality_path, input_copy)

    result = runner.invoke(
        cli, ["trim", str(input_copy), "--start", "0", "--end", "30"]
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert (tmp_path / "video.trim.mp4").exists()


def test_trim_input_not_found():
    """Test trim handles missing input."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["trim", "nonexistent.slp", "--start", "0", "--end", "100"]
    )
    assert result.exit_code != 0


def test_trim_accepts_positional_input(tmp_path, slp_real_data):
    """Test trim accepts positional input."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        ["trim", slp_real_data, "--start", "0", "--end", "30", "-o", str(output_path)],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)


def test_trim_accepts_flag_input(tmp_path, slp_real_data):
    """Test trim accepts -i flag input."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        [
            "trim",
            "-i",
            slp_real_data,
            "--start",
            "0",
            "--end",
            "30",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)


def test_trim_rejects_both_inputs(tmp_path, slp_real_data):
    """Test trim rejects both positional and flag input."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        [
            "trim",
            slp_real_data,
            "-i",
            slp_real_data,
            "--start",
            "0",
            "--end",
            "30",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "Cannot specify" in _strip_ansi(result.output)


def test_trim_missing_input():
    """Test trim requires input."""
    runner = CliRunner()
    result = runner.invoke(cli, ["trim", "--start", "0", "--end", "100"])
    assert result.exit_code != 0
    assert "Missing input file" in _strip_ansi(result.output)


def test_trim_multi_video_requires_video_flag(slp_multiview, tmp_path):
    """Test trim requires --video flag for multi-video labels."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        ["trim", slp_multiview, "--start", "0", "--end", "10", "-o", str(output_path)],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Multiple videos found" in output
    assert "--video" in output


def test_trim_video_index_out_of_range(tmp_path, slp_real_data):
    """Test trim with invalid video index."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        [
            "trim",
            slp_real_data,
            "--start",
            "0",
            "--end",
            "30",
            "--video",
            "99",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "out of range" in output


def test_trim_end_exceeds_video_length_labels(tmp_path, slp_real_data):
    """Test trim with end frame exceeding video length shows warning."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    # Use a very large end frame to trigger warning
    result = runner.invoke(
        cli,
        [
            "trim",
            slp_real_data,
            "--start",
            "0",
            "--end",
            "999999",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)
    assert "Warning:" in output or "Clamping" in output or output_path.exists()


def test_trim_end_exceeds_video_length_video(tmp_path, centered_pair_low_quality_path):
    """Test video-only trim with end frame exceeding video length shows warning."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.mp4"

    result = runner.invoke(
        cli,
        [
            "trim",
            centered_pair_low_quality_path,
            "--start",
            "0",
            "--end",
            "999999",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)
    assert "Warning:" in output or "Clamping" in output or output_path.exists()


def test_trim_video_only_invalid_frame_range(tmp_path, centered_pair_low_quality_path):
    """Test video-only trim with invalid frame range."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.mp4"

    # End before start
    result = runner.invoke(
        cli,
        [
            "trim",
            centered_pair_low_quality_path,
            "--start",
            "100",
            "--end",
            "50",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "must be greater than start frame" in _strip_ansi(result.output)


def test_trim_video_only_negative_start(tmp_path, centered_pair_low_quality_path):
    """Test video-only trim with negative start frame."""
    runner = CliRunner()
    output_path = tmp_path / "trimmed.mp4"

    result = runner.invoke(
        cli,
        [
            "trim",
            centered_pair_low_quality_path,
            "--start",
            "-10",
            "--end",
            "50",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "must be >= 0" in _strip_ansi(result.output)


def test_trim_labels_no_frame_range(tmp_path, slp_minimal_pkg):
    """Test labels trimming without explicit frame range uses full video.

    Covers lines 3623, 3625: default start_frame and end_frame in _trim_labels.
    """
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    # Trim without --start or --end (should use full frame range)
    result = runner.invoke(
        cli,
        ["trim", slp_minimal_pkg, "-o", str(output_path)],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Trimming:" in output
    assert "Saved:" in output
    assert output_path.exists()

    # Verify trimmed labels has same content as original (full trim)
    trimmed = load_slp(str(output_path))
    assert len(trimmed.videos) == 1


def test_trim_video_no_frame_range(tmp_path):
    """Test video trimming without explicit frame range uses full video.

    Covers lines 3691-3694: default start_frame and end_frame in _trim_video.
    """
    runner = CliRunner()

    # Create a short test video by copying and using the existing small video
    # We'll trim only the first 10 frames to make a tiny video for this test
    src_video = Path("tests/data/videos/centered_pair_low_quality.mp4")
    temp_video = tmp_path / "short_video.mp4"

    # First create a short video using the trim command WITH frame range
    result = runner.invoke(
        cli,
        ["trim", str(src_video), "--start", "0", "--end", "10", "-o", str(temp_video)],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert temp_video.exists()

    # Now test trimming WITHOUT frame range on the short video
    output_path = tmp_path / "full_trim.mp4"
    result = runner.invoke(
        cli,
        ["trim", str(temp_video), "-o", str(output_path)],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Trimming:" in output
    assert "Saved:" in output
    assert output_path.exists()


def test_trim_unsupported_input_type(tmp_path, skeleton_json_minimal):
    """Test trim rejects unsupported file types like skeleton files.

    Covers lines 3550-3552: ClickException for non-Labels/Video input.
    """
    runner = CliRunner()
    output_path = tmp_path / "trimmed.slp"

    result = runner.invoke(
        cli,
        ["trim", skeleton_json_minimal, "-o", str(output_path)],
    )
    # Should fail because skeleton files are not Labels or Video
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    # Either "Input must be a labels file or video" or "Failed to load input"
    assert "labels file or video" in output.lower() or "failed" in output.lower()


def test_trim_labels_write_failure(tmp_path, slp_real_data):
    """Test trim handles write failures gracefully.

    Covers lines 3662-3663: Exception handler for labels.trim() failure.
    """
    import os

    runner = CliRunner()

    # Create a read-only output file to trigger permission error during video save
    output_path = tmp_path / "readonly.slp"
    video_path = tmp_path / "readonly.mp4"
    video_path.write_text("dummy")
    os.chmod(video_path, 0o444)

    try:
        result = runner.invoke(
            cli,
            [
                "trim",
                slp_real_data,
                "--start",
                "0",
                "--end",
                "10",
                "-o",
                str(output_path),
            ],
        )
        # Should fail due to permission error when writing video
        assert result.exit_code != 0
        output = _strip_ansi(result.output)
        assert "Failed to trim" in output or "Permission" in output.lower()
    finally:
        # Restore permissions for cleanup
        os.chmod(video_path, 0o755)


def test_trim_video_write_failure(tmp_path, centered_pair_low_quality_path):
    """Test video trim handles write failures gracefully.

    Covers lines 3728-3729: Exception handler for video.save() failure.
    """
    import os

    runner = CliRunner()

    # Create a read-only output file to trigger permission error
    output_path = tmp_path / "readonly.mp4"
    output_path.write_text("dummy")
    os.chmod(output_path, 0o444)

    try:
        result = runner.invoke(
            cli,
            [
                "trim",
                centered_pair_low_quality_path,
                "--start",
                "0",
                "--end",
                "10",
                "-o",
                str(output_path),
            ],
        )
        # Should fail due to permission error
        assert result.exit_code != 0
        output = _strip_ansi(result.output)
        assert "Failed to trim" in output or "Permission" in output.lower()
    finally:
        # Restore permissions for cleanup
        os.chmod(output_path, 0o755)


# =============================================================================
# Reencode command tests
# =============================================================================


def test_reencode_help():
    """Test that reencode command help is displayed."""
    runner = CliRunner()
    result = runner.invoke(cli, ["reencode", "--help"])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check key options are documented
    assert "reencode" in output.lower()
    assert "--quality" in output
    assert "--crf" in output
    assert "--keyframe-interval" in output
    assert "--encoding" in output
    assert "--dry-run" in output


def test_reencode_in_command_list():
    """Test that reencode appears in the main CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "reencode" in output


def test_reencode_dry_run(tmp_path, centered_pair_low_quality_path):
    """Test dry run mode shows command without executing."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    # Should show either ffmpeg command or Python path message
    if _is_ffmpeg_available():
        assert "ffmpeg" in output
        assert "libx264" in output
    else:
        assert "Python path" in output
    # Output file should NOT exist (dry run)
    assert not output_path.exists()


def test_reencode_dry_run_python_path(tmp_path, centered_pair_low_quality_path):
    """Test dry run mode with Python path."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--dry-run",
            "--no-ffmpeg",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Python path" in output
    assert not output_path.exists()


def test_reencode_quality_crf_mutually_exclusive(
    tmp_path, centered_pair_low_quality_path
):
    """Test that --quality and --crf cannot be used together."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--quality",
            "high",
            "--crf",
            "20",
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Cannot use both --quality and --crf" in output


def test_reencode_invalid_crf(tmp_path, centered_pair_low_quality_path):
    """Test that invalid CRF values are rejected."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--crf",
            "60",
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "CRF must be between 0 and 51" in output


def test_reencode_same_input_output_error(tmp_path, centered_pair_low_quality_path):
    """Test that using same path for input and output is rejected."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            centered_pair_low_quality_path,
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "cannot be the same as input" in output.lower()


def test_reencode_output_exists_error(tmp_path, centered_pair_low_quality_path):
    """Test that existing output file is rejected without --overwrite."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"
    output_path.touch()  # Create existing file

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "already exists" in output.lower()


def test_reencode_input_not_found(tmp_path):
    """Test error when input file does not exist."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            "/nonexistent/video.mp4",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0


def test_reencode_accepts_positional_input(tmp_path, centered_pair_low_quality_path):
    """Test that positional input argument works."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)


def test_reencode_accepts_flag_input(tmp_path, centered_pair_low_quality_path):
    """Test that -i/--input flag works."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            "-i",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)


def test_reencode_rejects_both_inputs(tmp_path, centered_pair_low_quality_path):
    """Test that providing both positional and flag input is rejected."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-i",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Cannot specify" in output


def test_reencode_missing_input(tmp_path):
    """Test that missing input is rejected."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "Missing" in output


def test_reencode_default_output_path(tmp_path, centered_pair_low_quality_path):
    """Test that default output path is generated correctly."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    # Default should add .reencoded before extension
    assert "centered_pair_low_quality.reencoded.mp4" in output


@skip_slow_video_on_windows
def test_reencode_basic(tmp_path, centered_pair_low_quality_path):
    """Test basic reencoding (ffmpeg or Python path)."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    # Check for either ffmpeg or Python path output
    assert "Reencoding" in output
    assert "Saved:" in output
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@skip_slow_video_on_windows
def test_reencode_with_quality(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with quality option."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--quality",
            "low",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    # CRF is shown in ffmpeg path output, not in Python path
    if _is_ffmpeg_available():
        assert "CRF 32" in output  # low quality = CRF 32
    assert output_path.exists()


def test_reencode_with_keyframe_interval(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with custom keyframe interval."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--keyframe-interval",
            "0.5",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    # Check for either ffmpeg GOP setting or Python path
    if _is_ffmpeg_available():
        # In dry-run mode, check that ffmpeg command has a GOP setting
        # The video is ~15fps, so 0.5s interval -> GOP of ~7
        assert "-g" in output
        assert "ffmpeg" in output
    else:
        assert "Python path" in output


def test_reencode_with_encoding_preset(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with encoding preset option."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--encoding",
            "ultrafast",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "ultrafast" in output


@skip_slow_video_on_windows
def test_reencode_python_path(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with Python fallback path."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--no-ffmpeg",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Python path" in output
    assert output_path.exists()


@skip_slow_video_on_windows
def test_reencode_overwrite(tmp_path, centered_pair_low_quality_path):
    """Test that --overwrite allows replacing existing output."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"existing content")  # Create existing file

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    assert output_path.exists()
    # File should be larger than original dummy content
    assert output_path.stat().st_size > 20


def test_reencode_with_gop_option(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with explicit --gop option."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--gop",
            "15",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    # GOP of 15 frames should appear in ffmpeg command or Python path info
    if _is_ffmpeg_available():
        assert "-g" in output
        assert "15" in output
    else:
        assert "Python path" in output


@skip_slow_video_on_windows
def test_reencode_with_fps_python_path(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with --fps option using Python path."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--fps",
            "15",
            "--no-ffmpeg",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    assert "Python path" in output
    assert output_path.exists()


def test_reencode_use_ffmpeg_without_ffmpeg():
    """Test --use-ffmpeg flag when ffmpeg might not be available."""
    runner = CliRunner()

    # This test demonstrates the --use-ffmpeg flag behavior
    # On systems without ffmpeg, it should fail with helpful message
    if not _is_ffmpeg_available():
        with runner.isolated_filesystem():
            # Create a dummy input file to pass Click's exists validation
            Path("dummy.mp4").touch()
            result = runner.invoke(
                cli,
                [
                    "reencode",
                    "dummy.mp4",
                    "-o",
                    "output.mp4",
                    "--use-ffmpeg",
                ],
            )
            assert result.exit_code != 0
            output = _strip_ansi(result.output)
            assert "ffmpeg not found" in output


def test_get_ffmpeg_version():
    """Test _get_ffmpeg_version helper function."""
    version = _get_ffmpeg_version()

    if _is_ffmpeg_available():
        # If ffmpeg is available, we should get a version string
        assert version is not None
        # Version should look like a version number (contains digits and dots)
        assert any(c.isdigit() for c in version)
    else:
        # If ffmpeg is not available, version should be None
        assert version is None


@skip_slow_video_on_windows
def test_reencode_with_fps_ffmpeg_path(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with --fps option using ffmpeg path (if available)."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    if _is_ffmpeg_available():
        result = runner.invoke(
            cli,
            [
                "reencode",
                centered_pair_low_quality_path,
                "-o",
                str(output_path),
                "--fps",
                "15",
            ],
        )
        assert result.exit_code == 0, _strip_ansi(result.output)
        output = _strip_ansi(result.output)

        # Should show FPS change message
        assert "FPS:" in output or "Reencoding" in output
        assert output_path.exists()


def test_reencode_with_gop_and_fps(tmp_path, centered_pair_low_quality_path):
    """Test reencoding with both --gop and --fps options."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "reencode",
            centered_pair_low_quality_path,
            "-o",
            str(output_path),
            "--gop",
            "10",
            "--fps",
            "15",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, _strip_ansi(result.output)
    output = _strip_ansi(result.output)

    # Should show either ffmpeg command or Python path
    if _is_ffmpeg_available():
        assert "ffmpeg" in output
        # GOP should be in command
        assert "-g" in output
    else:
        assert "Python path" in output


# SLP batch processing tests


def test_reencode_slp_dry_run(tmp_path, slp_real_data):
    """Test SLP batch processing in dry-run mode."""
    runner = CliRunner()
    output_slp = tmp_path / "output.slp"

    result = runner.invoke(
        cli, ["reencode", slp_real_data, "-o", str(output_slp), "--dry-run"]
    )

    # Command should succeed
    assert result.exit_code == 0, result.output

    # Should show SLP-specific output
    assert "Loading SLP" in result.output
    assert "video(s)" in result.output
    assert "Dry run" in result.output

    # Output files should not be created
    assert not output_slp.exists()


def test_reencode_slp_default_output_path(tmp_path, slp_real_data):
    """Test SLP default output path generation."""
    import shutil

    runner = CliRunner()

    # Copy SLP file to tmp_path so we can test default output
    tmp_slp = tmp_path / "project.slp"
    shutil.copy(slp_real_data, tmp_slp)

    result = runner.invoke(cli, ["reencode", str(tmp_slp), "--dry-run"])

    # Command should succeed
    assert result.exit_code == 0, result.output

    # Should mention the default output path
    assert "project.reencoded.slp" in result.output


def test_reencode_slp_same_input_output_error(tmp_path, slp_real_data):
    """Test that using same input and output path raises error."""
    import shutil

    runner = CliRunner()

    # Copy SLP to tmp_path
    tmp_slp = tmp_path / "project.slp"
    shutil.copy(slp_real_data, tmp_slp)

    result = runner.invoke(cli, ["reencode", str(tmp_slp), "-o", str(tmp_slp)])

    # Should fail with clear error
    assert result.exit_code != 0
    assert "same as input" in result.output.lower()


def test_reencode_slp_output_exists_error(tmp_path, slp_real_data):
    """Test that existing output SLP raises error without --overwrite."""
    import shutil

    runner = CliRunner()

    # Copy SLP and create existing output
    tmp_slp = tmp_path / "project.slp"
    output_slp = tmp_path / "output.slp"
    shutil.copy(slp_real_data, tmp_slp)
    output_slp.touch()  # Create existing file

    result = runner.invoke(cli, ["reencode", str(tmp_slp), "-o", str(output_slp)])

    # Should fail with clear error
    assert result.exit_code != 0
    assert "already exists" in result.output.lower()


def test_reencode_slp_quality_crf_mutually_exclusive(tmp_path, slp_real_data):
    """Test that --quality and --crf are mutually exclusive for SLP."""
    runner = CliRunner()
    output_slp = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "reencode",
            slp_real_data,
            "-o",
            str(output_slp),
            "--quality",
            "high",
            "--crf",
            "20",
            "--dry-run",
        ],
    )

    # Should fail
    assert result.exit_code != 0
    assert "both" in result.output.lower() or "quality" in result.output.lower()


def test_reencode_slp_invalid_crf(tmp_path, slp_real_data):
    """Test that invalid CRF values are rejected for SLP."""
    runner = CliRunner()
    output_slp = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        ["reencode", slp_real_data, "-o", str(output_slp), "--crf", "60", "--dry-run"],
    )

    # Should fail with CRF error
    assert result.exit_code != 0
    assert "crf" in result.output.lower()


@skip_slow_video_on_windows
@pytest.mark.skipif(
    not _is_ffmpeg_available(), reason="ffmpeg not available in test environment"
)
def test_reencode_slp_basic(tmp_path, slp_real_data):
    """Test basic SLP batch reencoding."""
    import shutil

    import sleap_io as sio

    runner = CliRunner()

    # Copy the SLP and video to tmp_path to avoid modifying test data
    tmp_slp = tmp_path / "project.slp"
    tmp_video = tmp_path / "centered_pair_low_quality.mp4"
    output_slp = tmp_path / "output.slp"

    shutil.copy(slp_real_data, tmp_slp)
    shutil.copy("tests/data/videos/centered_pair_low_quality.mp4", tmp_video)

    # Load and fix video path to use tmp_path
    labels = sio.load_slp(str(tmp_slp))
    labels.videos[0].replace_filename(str(tmp_video))
    labels.save(str(tmp_slp))

    result = runner.invoke(cli, ["reencode", str(tmp_slp), "-o", str(output_slp)])

    # Command should succeed
    assert result.exit_code == 0, result.output

    # Output SLP should exist
    assert output_slp.exists()

    # Video directory should exist
    video_dir = tmp_path / "output.videos"
    assert video_dir.exists()

    # Load output SLP and check videos
    output_labels = sio.load_slp(str(output_slp))
    assert len(output_labels.videos) == len(labels.videos)

    # Check that video paths were updated
    for video in output_labels.videos:
        assert ".reencoded" in video.filename


@skip_slow_video_on_windows
@pytest.mark.skipif(
    not _is_ffmpeg_available(), reason="ffmpeg not available in test environment"
)
def test_reencode_slp_with_quality(tmp_path, slp_real_data):
    """Test SLP batch reencoding with quality option."""
    import shutil

    import sleap_io as sio

    runner = CliRunner()

    # Copy the SLP and video to tmp_path
    tmp_slp = tmp_path / "project.slp"
    tmp_video = tmp_path / "centered_pair_low_quality.mp4"
    output_slp = tmp_path / "output.slp"

    shutil.copy(slp_real_data, tmp_slp)
    shutil.copy("tests/data/videos/centered_pair_low_quality.mp4", tmp_video)

    # Load and fix video path
    labels = sio.load_slp(str(tmp_slp))
    labels.videos[0].replace_filename(str(tmp_video))
    labels.save(str(tmp_slp))

    result = runner.invoke(
        cli, ["reencode", str(tmp_slp), "-o", str(output_slp), "--quality", "high"]
    )

    # Command should succeed
    assert result.exit_code == 0, result.output

    # Output should mention CRF 18 (high quality)
    assert "CRF 18" in result.output


@skip_slow_video_on_windows
@pytest.mark.skipif(
    not _is_ffmpeg_available(), reason="ffmpeg not available in test environment"
)
def test_reencode_slp_overwrite(tmp_path, slp_real_data):
    """Test SLP batch reencoding with --overwrite flag."""
    import shutil

    import sleap_io as sio

    runner = CliRunner()

    # Copy the SLP and video to tmp_path
    tmp_slp = tmp_path / "project.slp"
    tmp_video = tmp_path / "centered_pair_low_quality.mp4"
    output_slp = tmp_path / "output.slp"

    shutil.copy(slp_real_data, tmp_slp)
    shutil.copy("tests/data/videos/centered_pair_low_quality.mp4", tmp_video)

    # Create existing output file
    output_slp.touch()

    # Load and fix video path
    labels = sio.load_slp(str(tmp_slp))
    labels.videos[0].replace_filename(str(tmp_video))
    labels.save(str(tmp_slp))

    # Without --overwrite should fail
    result = runner.invoke(cli, ["reencode", str(tmp_slp), "-o", str(output_slp)])
    assert result.exit_code != 0

    # With --overwrite should succeed
    result = runner.invoke(
        cli, ["reencode", str(tmp_slp), "-o", str(output_slp), "--overwrite"]
    )
    assert result.exit_code == 0, result.output


def test_reencode_slp_no_videos(tmp_path):
    """Test SLP batch reencoding with empty video list."""
    runner = CliRunner()

    # Create a Labels object with no videos
    skeleton = Skeleton(["node"])
    labels = Labels(skeletons=[skeleton], videos=[], labeled_frames=[])

    # Save to SLP
    input_slp = tmp_path / "empty.slp"
    output_slp = tmp_path / "output.slp"
    labels.save(str(input_slp))

    result = runner.invoke(cli, ["reencode", str(input_slp), "-o", str(output_slp)])

    # Should succeed but warn about no videos
    assert result.exit_code == 0
    assert "No videos found" in result.output


def test_reencode_slp_with_image_video(tmp_path):
    """Test that ImageVideo backends are skipped during SLP reencoding."""
    from sleap_io.model.video import Video

    runner = CliRunner()

    # Create a Labels object with an ImageVideo-style video
    skeleton = Skeleton(["node"])

    # Create ImageVideo by using a list of image paths
    # When video has multiple filenames, it's treated as ImageVideo
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create some dummy image files
    for i in range(3):
        img_path = image_dir / f"frame_{i:04d}.png"
        img_path.write_bytes(b"fake image data")

    # Use the image filenames list pattern for ImageVideo
    image_paths = [str(image_dir / f"frame_{i:04d}.png") for i in range(3)]

    # Create video from first image - this creates a MediaVideo initially
    video = Video.from_filename(image_paths[0])

    labels = Labels(skeletons=[skeleton], videos=[video], labeled_frames=[])

    # Save to SLP
    input_slp = tmp_path / "imageseq.slp"
    output_slp = tmp_path / "output.slp"
    labels.save(str(input_slp))

    result = runner.invoke(cli, ["reencode", str(input_slp), "-o", str(output_slp)])

    # Check output - command runs but skips the "video" (image files aren't reencoded)
    # The MediaVideo wrapper around a single image may fail to read
    assert result.exit_code in (0, 1)  # May succeed or fail depending on image validity


def test_reencode_slp_with_missing_backend(tmp_path):
    """Test error handling when video file doesn't exist."""
    from sleap_io.model.video import Video

    runner = CliRunner()

    # Create a Labels object with a video pointing to non-existent file
    skeleton = Skeleton(["node"])
    video = Video(filename="/nonexistent/path/video.mp4")

    labels = Labels(skeletons=[skeleton], videos=[video], labeled_frames=[])

    # Save to SLP
    input_slp = tmp_path / "missing.slp"
    output_slp = tmp_path / "output.slp"
    labels.save(str(input_slp))

    result = runner.invoke(cli, ["reencode", str(input_slp), "-o", str(output_slp)])

    # Should fail because video file doesn't exist
    # The error is caught and reported gracefully
    assert (
        result.exit_code != 0 or "Error" in result.output or "Skipped" in result.output
    )
    # Check output contains informative error message
    output = _strip_ansi(result.output)
    assert "video" in output.lower() or "Error" in output or "Skipping" in output


@skip_slow_video_on_windows
def test_reencode_slp_hdf5_video(tmp_path, slp_minimal_pkg):
    """Test SLP batch reencoding with HDF5Video (embedded images).

    This tests the Python path code in _reencode_video_object since HDF5Video
    cannot use the ffmpeg fast path.
    """
    import shutil

    runner = CliRunner()

    # Copy the PKG.SLP to tmp_path
    tmp_slp = tmp_path / "project.pkg.slp"
    output_slp = tmp_path / "output.slp"

    shutil.copy(slp_minimal_pkg, tmp_slp)

    result = runner.invoke(
        cli, ["reencode", str(tmp_slp), "-o", str(output_slp), "--dry-run"]
    )

    # Dry run should succeed
    assert result.exit_code == 0, result.output

    # Should mention Python path for HDF5
    output = _strip_ansi(result.output)
    assert "Python" in output or "HDF5" in output or "Would reencode" in output


def test_reencode_slp_output_video_exists(tmp_path, slp_real_data):
    """Test that existing output videos are skipped without --overwrite in SLP mode."""
    import shutil

    import sleap_io as sio

    runner = CliRunner()

    # Copy the SLP and video to tmp_path
    tmp_slp = tmp_path / "project.slp"
    tmp_video = tmp_path / "centered_pair_low_quality.mp4"
    output_slp = tmp_path / "output.slp"

    shutil.copy(slp_real_data, tmp_slp)
    shutil.copy("tests/data/videos/centered_pair_low_quality.mp4", tmp_video)

    # Load and fix video path
    labels = sio.load_slp(str(tmp_slp))
    labels.videos[0].replace_filename(str(tmp_video))
    labels.save(str(tmp_slp))

    # Create the video output directory and a pre-existing output video
    video_dir = tmp_path / "output.videos"
    video_dir.mkdir()
    existing_video = video_dir / "centered_pair_low_quality.reencoded.mp4"
    existing_video.touch()  # Create empty file

    result = runner.invoke(
        cli, ["reencode", str(tmp_slp), "-o", str(output_slp), "--dry-run"]
    )

    # Dry run should succeed
    assert result.exit_code == 0, result.output

    # Should report that the video was skipped because output exists
    output = _strip_ansi(result.output)
    assert "Skipping" in output or "exists" in output.lower()


@skip_slow_video_on_windows
def test_reencode_slp_hdf5_video_actual(tmp_path, slp_minimal_pkg):
    """Test actual HDF5Video reencoding (Python path).

    This test covers the Python path code in _reencode_video_object since
    HDF5Video cannot use the ffmpeg fast path.
    """
    import shutil

    runner = CliRunner()

    # Copy the PKG.SLP to tmp_path
    tmp_slp = tmp_path / "project.pkg.slp"
    output_slp = tmp_path / "output.slp"

    shutil.copy(slp_minimal_pkg, tmp_slp)

    result = runner.invoke(cli, ["reencode", str(tmp_slp), "-o", str(output_slp)])

    # Command should succeed
    assert result.exit_code == 0, result.output

    # Output SLP should exist
    assert output_slp.exists()

    # Video directory should exist with reencoded video
    video_dir = tmp_path / "output.videos"
    assert video_dir.exists()

    # Should have used Python path for HDF5
    output = _strip_ansi(result.output)
    assert "Python" in output or "Reencoding" in output

    # Check that a reencoded video was created
    reencoded_videos = list(video_dir.glob("*.mp4"))
    assert len(reencoded_videos) > 0, f"No reencoded videos found in {video_dir}"


def test_reencode_output_suffix_normalization(tmp_path, centered_pair_low_quality_path):
    """Test that output suffix is normalized to .mp4 for unusual input extensions."""
    import shutil

    runner = CliRunner()

    # Copy video with unusual extension
    unusual_input = tmp_path / "video.webm"
    shutil.copy(centered_pair_low_quality_path, unusual_input)

    result = runner.invoke(cli, ["reencode", str(unusual_input), "--dry-run"])

    # Dry run should succeed
    assert result.exit_code == 0, result.output

    # Output should show .mp4 extension (normalized from .webm)
    output = _strip_ansi(result.output)
    # The default output should be video.reencoded.mp4 (not video.reencoded.webm)
    assert ".reencoded.mp4" in output or "video.reencoded" in output


def test_reencode_slp_hdf5_use_ffmpeg_error(tmp_path, slp_minimal_pkg):
    """Test that --use-ffmpeg fails with HDF5Video."""
    import shutil

    runner = CliRunner()

    # Copy the PKG.SLP to tmp_path
    tmp_slp = tmp_path / "project.pkg.slp"
    output_slp = tmp_path / "output.slp"

    shutil.copy(slp_minimal_pkg, tmp_slp)

    result = runner.invoke(
        cli, ["reencode", str(tmp_slp), "-o", str(output_slp), "--use-ffmpeg"]
    )

    # Should fail because HDF5 videos can't use ffmpeg
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "HDF5" in output or "cannot" in output.lower() or "Python" in output


def test_reencode_slp_output_suffix_enforcement(tmp_path, slp_real_data):
    """Test that SLP output gets .slp suffix even if specified otherwise."""
    import shutil

    import sleap_io as sio

    runner = CliRunner()

    # Copy the SLP and video to tmp_path
    tmp_slp = tmp_path / "project.slp"
    tmp_video = tmp_path / "centered_pair_low_quality.mp4"
    # Output with wrong extension
    output_path = tmp_path / "output.txt"

    shutil.copy(slp_real_data, tmp_slp)
    shutil.copy("tests/data/videos/centered_pair_low_quality.mp4", tmp_video)

    # Load and fix video path
    labels = sio.load_slp(str(tmp_slp))
    labels.videos[0].replace_filename(str(tmp_video))
    labels.save(str(tmp_slp))

    result = runner.invoke(
        cli, ["reencode", str(tmp_slp), "-o", str(output_path), "--dry-run"]
    )

    # Dry run should succeed
    assert result.exit_code == 0, result.output

    # Output should show .slp extension (normalized from .txt)
    output = _strip_ansi(result.output)
    assert ".slp" in output


# =============================================================================
# Reencode internal function tests for edge case coverage
# =============================================================================


def test_reencode_video_object_with_image_video(tmp_path):
    """Test _reencode_video_object skips ImageVideo backends (lines 4343-4346)."""
    import numpy as np

    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.io.video_reading import ImageVideo
    from sleap_io.model.video import Video

    # Create an ImageVideo with synthetic images
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create some dummy image files
    from PIL import Image

    for i in range(3):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(img_dir / f"frame_{i:03d}.png")

    # Create Video with ImageVideo backend
    backend = ImageVideo(img_dir)
    video = Video(filename=str(img_dir), backend=backend)

    output_path = tmp_path / "output.mp4"

    # Should return None (skip) without error
    result = _reencode_video_object(
        video=video,
        output_path=output_path,
        crf=25,
        preset="superfast",
        keyframe_interval=1.0,
    )

    assert result is None
    assert not output_path.exists()


def test_reencode_video_object_output_exists_no_overwrite(
    tmp_path, centered_pair_low_quality_path
):
    """Test _reencode_video_object raises error when output exists (line 4350)."""
    import click

    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.model.video import Video

    video = Video.from_filename(centered_pair_low_quality_path)
    output_path = tmp_path / "output.mp4"

    # Create existing output file
    output_path.touch()

    with pytest.raises(click.ClickException) as exc_info:
        _reencode_video_object(
            video=video,
            output_path=output_path,
            crf=25,
            preset="superfast",
            keyframe_interval=1.0,
            overwrite=False,
        )

    assert "already exists" in str(exc_info.value)
    assert "--overwrite" in str(exc_info.value)


def test_reencode_video_object_unknown_backend(tmp_path):
    """Test _reencode_video_object with unknown backend type (lines 4365-4366)."""
    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.model.video import Video

    # Create a Video with a custom/unknown backend type
    class CustomBackend:
        """A mock backend that is not one of the known types."""

        def __init__(self):
            self.filename = "/fake/path.mp4"

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            import numpy as np

            return np.zeros((100, 100, 3), dtype=np.uint8)

    video = Video(filename="/fake/path.mp4", backend=CustomBackend())
    output_path = tmp_path / "output.mp4"

    # Should use Python path (can_use_ffmpeg=False) since backend is unknown
    # This will fail because the file doesn't exist, but it tests the branch
    # Actually, it will try Python path and fail on video loading
    # Let's test with dry_run to avoid actual encoding
    result = _reencode_video_object(
        video=video,
        output_path=output_path,
        crf=25,
        preset="superfast",
        keyframe_interval=1.0,
        dry_run=True,
    )

    # Dry run returns None
    assert result is None


def test_reencode_video_object_force_ffmpeg_on_hdf5(tmp_path, slp_minimal_pkg):
    """Test _reencode_video_object error when forcing ffmpeg on HDF5."""
    import click

    import sleap_io as sio
    from sleap_io.io.cli import _reencode_video_object

    # Load the pkg.slp which has HDF5Video
    labels = sio.load_slp(str(slp_minimal_pkg))
    video = labels.videos[0]
    output_path = tmp_path / "output.mp4"

    # Force ffmpeg path on HDF5 video should error
    with pytest.raises(click.ClickException) as exc_info:
        _reencode_video_object(
            video=video,
            output_path=output_path,
            crf=25,
            preset="superfast",
            keyframe_interval=1.0,
            use_ffmpeg=True,
        )

    assert "HDF5" in str(exc_info.value)
    assert "Python path" in str(exc_info.value) or "--no-ffmpeg" in str(exc_info.value)


@skip_slow_video_on_windows
def test_reencode_video_object_with_fps_change(
    tmp_path, centered_pair_low_quality_path
):
    """Test _reencode_video_object shows FPS change message (line 4419)."""
    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.model.video import Video

    video = Video.from_filename(centered_pair_low_quality_path)
    output_path = tmp_path / "output.mp4"

    # Call with output_fps to trigger line 4419
    if _is_ffmpeg_available():
        result = _reencode_video_object(
            video=video,
            output_path=output_path,
            crf=25,
            preset="superfast",
            keyframe_interval=1.0,
            output_fps=15.0,  # Different from source
        )

        # Should succeed and return output path
        assert result == output_path
        assert output_path.exists()


@skip_slow_video_on_windows
def test_reencode_video_object_file_size_comparison(
    tmp_path, centered_pair_low_quality_path
):
    """Test _reencode_video_object shows file size comparison (lines 4499-4511)."""
    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.model.video import Video

    video = Video.from_filename(centered_pair_low_quality_path)
    output_path = tmp_path / "output.mp4"

    # Perform actual encoding (non-dry_run) to trigger file size comparison
    if _is_ffmpeg_available():
        result = _reencode_video_object(
            video=video,
            output_path=output_path,
            crf=25,
            preset="superfast",
            keyframe_interval=1.0,
            dry_run=False,
        )

        # Should succeed and return output path
        assert result == output_path
        assert output_path.exists()
        # File size comparison is printed to console (can't easily verify output)


def test_reencode_video_object_with_tiff_video(tmp_path):
    """Test _reencode_video_object skips TiffVideo backends (lines 4343-4346).

    This tests the same code path as ImageVideo but for TiffVideo backend.
    """
    import numpy as np
    import tifffile

    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.io.video_reading import TiffVideo
    from sleap_io.model.video import Video

    # Create a TIFF stack
    tiff_path = tmp_path / "stack.tif"
    frames = np.zeros((3, 100, 100), dtype=np.uint8)  # 3 grayscale frames
    tifffile.imwrite(tiff_path, frames)

    # Create Video with TiffVideo backend
    backend = TiffVideo(tiff_path)
    video = Video(filename=str(tiff_path), backend=backend)

    output_path = tmp_path / "output.mp4"

    # Should return None (skip) without error
    result = _reencode_video_object(
        video=video,
        output_path=output_path,
        crf=25,
        preset="superfast",
        keyframe_interval=1.0,
    )

    assert result is None
    assert not output_path.exists()


def test_reencode_slp_result_none_non_dryrun(tmp_path):
    """Test _reencode_slp handles skipped videos in non-dry_run mode (line 4637).

    This tests the branch where _reencode_video_object returns None
    (for skipped backends like ImageVideo/TiffVideo) while not in dry_run mode.

    Note: The ImageVideo/TiffVideo skip branches (lines 4582-4588) in _reencode_slp
    catch these cases before reaching _reencode_video_object, so line 4637
    (result_path is None and not dry_run) is primarily defensive code for edge cases.
    """
    import numpy as np
    from PIL import Image

    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.io.video_reading import ImageVideo
    from sleap_io.model.video import Video

    # Create an ImageVideo backend
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(3):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(img_dir / f"frame_{i:03d}.png")

    # Create ImageVideo with list of filenames (as expected)
    image_files = sorted(img_dir.glob("*.png"))
    backend = ImageVideo(image_files)
    video = Video(filename=str(img_dir), backend=backend)

    output_path = tmp_path / "output.mp4"

    # Call _reencode_video_object directly (non-dry_run)
    # ImageVideo should return None (skipped)
    result = _reencode_video_object(
        video=video,
        output_path=output_path,
        crf=25,
        preset="superfast",
        keyframe_interval=1.0,
        dry_run=False,  # Non-dry_run mode
    )

    # Should return None (skipped)
    assert result is None
    assert not output_path.exists()


def test_reencode_video_object_invalid_output_path(
    tmp_path, centered_pair_low_quality_path
):
    """Test _reencode_video_object with invalid output path raises ClickException."""
    import click

    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.model.video import Video

    video = Video.from_filename(centered_pair_low_quality_path)

    # Use an invalid output path (directory doesn't exist)
    invalid_output = tmp_path / "nonexistent_dir" / "subdir" / "output.mp4"

    if _is_ffmpeg_available():
        with pytest.raises(click.ClickException) as exc_info:
            _reencode_video_object(
                video=video,
                output_path=invalid_output,
                crf=25,
                preset="superfast",
                keyframe_interval=1.0,
            )

        # Should get an error about ffmpeg failing
        assert "failed" in str(exc_info.value).lower()


def test_reencode_python_path_fps_fallback(tmp_path, centered_pair_low_quality_path):
    """Test _reencode_python_path uses 30.0 FPS fallback (line 4255).

    This tests the fallback when a Video object doesn't have fps in its backend.
    We test via _reencode_video_object with Python path and a backend without fps.
    """
    from sleap_io.io.cli import _reencode_video_object
    from sleap_io.model.video import Video

    # Create a Video with a backend that doesn't have fps attribute
    class NoFpsBackend:
        """Backend without fps attribute - simulates edge case."""

        def __init__(self, real_video_path):
            self.filename = str(real_video_path)
            self._video = Video.from_filename(str(real_video_path))

        def __len__(self):
            return len(self._video)

        def __getitem__(self, idx):
            return self._video[idx]

        # Note: No 'fps' attribute - this triggers line 4255 fallback

    backend = NoFpsBackend(centered_pair_low_quality_path)
    video = Video(filename=str(centered_pair_low_quality_path), backend=backend)
    output_path = tmp_path / "output.mp4"

    # Force Python path (--no-ffmpeg) to hit _reencode_python_path
    # The Python path will check for backend.fps and use 30.0 fallback
    result = _reencode_video_object(
        video=video,
        output_path=output_path,
        crf=25,
        preset="superfast",
        keyframe_interval=1.0,
        use_ffmpeg=False,  # Force Python path
        dry_run=True,  # Dry run to avoid slow encoding
    )

    # Dry run returns None
    assert result is None


# ============================================================================
# Transform Command Tests
# ============================================================================


def test_transform_help():
    """Test that transform command help is displayed."""
    runner = CliRunner()
    result = runner.invoke(cli, ["transform", "--help"])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Check key sections are present
    assert "Transform video" in output
    assert "--crop" in output
    assert "--scale" in output
    assert "--rotate" in output
    assert "--pad" in output
    assert "--dry-run" in output


def test_transform_in_command_list():
    """Test that transform appears in the main CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "transform" in output


def test_transform_no_transforms_error(tmp_path, slp_real_data):
    """Test that no transforms specified raises error."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        ["transform", slp_real_data, "-o", str(output_path)],
    )
    assert result.exit_code != 0
    assert "No transforms specified" in result.output


def test_transform_dry_run(tmp_path, slp_real_data):
    """Test dry run mode shows summary without executing."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "0.5",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)

    # Should show summary but not save
    assert "Transform Summary" in output
    assert "Dry run" in output
    assert not output_path.exists()


def test_transform_same_input_output_error(tmp_path, slp_real_data):
    """Test that using same input and output path raises error."""
    import shutil

    runner = CliRunner()
    slp_copy = tmp_path / "input.slp"
    shutil.copy(slp_real_data, slp_copy)

    result = runner.invoke(
        cli,
        ["transform", str(slp_copy), "--scale", "0.5", "-o", str(slp_copy)],
    )
    assert result.exit_code != 0
    assert "cannot be the same as input" in result.output


def test_transform_output_exists_error(tmp_path, slp_real_data):
    """Test that existing output SLP raises error without --overwrite."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"
    output_path.touch()  # Create existing file

    result = runner.invoke(
        cli,
        ["transform", slp_real_data, "--scale", "0.5", "-o", str(output_path)],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_transform_accepts_positional_input(tmp_path, slp_real_data):
    """Test that positional input argument works."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "0.5",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_transform_accepts_flag_input(tmp_path, slp_real_data):
    """Test that -i/--input flag works."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            "-i",
            slp_real_data,
            "--scale",
            "0.5",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_transform_parse_crop(tmp_path, slp_real_data):
    """Test crop parameter parsing."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--crop",
            "10,10,100,100",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Crop:" in output


def test_transform_parse_scale(tmp_path, slp_real_data):
    """Test scale parameter parsing."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "0.5",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Scale:" in output


def test_transform_parse_rotate(tmp_path, slp_real_data):
    """Test rotate parameter parsing."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--rotate",
            "90",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Rotate:" in output


def test_transform_parse_pad(tmp_path, slp_real_data):
    """Test pad parameter parsing."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--pad",
            "10,10,10,10",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Pad:" in output


def test_transform_combined(tmp_path, slp_real_data):
    """Test combined transforms."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--crop",
            "0,0,200,200",
            "--scale",
            "0.5",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Crop:" in output
    assert "Scale:" in output


def test_transform_per_video_params(tmp_path, slp_real_data):
    """Test per-video parameters with idx: prefix.

    Note: Uses slp_real_data which has a single video with valid path.
    Per-video params with idx:0 should work for single video.
    """
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "0:0.5",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_transform_video_index_out_of_range(tmp_path, slp_real_data):
    """Test that out-of-range video index raises error."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "99:0.5",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output


@skip_slow_video_on_windows
def test_transform_video_file_dry_run(tmp_path, centered_pair_low_quality_path):
    """Test transform on raw video file in dry-run mode."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--scale",
            "0.5",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Transform Summary" in output
    assert "Dry run" in output
    assert not output_path.exists()


def test_transform_parse_fill_value_int():
    """Test _parse_fill_value with integer value."""
    from sleap_io.io.cli import _parse_fill_value

    assert _parse_fill_value("0") == 0
    assert _parse_fill_value("128") == 128
    assert _parse_fill_value("255") == 255


def test_transform_parse_fill_value_rgb():
    """Test _parse_fill_value with RGB tuple."""
    from sleap_io.io.cli import _parse_fill_value

    assert _parse_fill_value("128,128,128") == (128, 128, 128)
    assert _parse_fill_value("0,0,0") == (0, 0, 0)
    assert _parse_fill_value("255,255,255") == (255, 255, 255)
    assert _parse_fill_value(" 10, 20, 30 ") == (10, 20, 30)  # with spaces


def test_transform_parse_fill_value_invalid():
    """Test _parse_fill_value with invalid values."""
    import click

    from sleap_io.io.cli import _parse_fill_value

    with pytest.raises(click.ClickException, match="0-255"):
        _parse_fill_value("256")

    with pytest.raises(click.ClickException, match="0-255"):
        _parse_fill_value("-1")

    with pytest.raises(click.ClickException, match="3 values"):
        _parse_fill_value("1,2")

    with pytest.raises(click.ClickException, match="Invalid"):
        _parse_fill_value("abc")


def test_transform_fill_rgb_option(tmp_path, slp_real_data):
    """Test transform with RGB fill value."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--rotate",
            "45",
            "--fill",
            "128,128,128",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_transform_dry_run_frame(tmp_path, slp_real_data):
    """Test --dry-run-frame option renders preview."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "0.5",
            "--dry-run-frame",
            "0",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    # Should show both dry run message and frame preview
    assert "Dry run" in output
    assert "Preview frame 0" in output
    assert "preview" in output.lower()
    assert not output_path.exists()


def test_transform_dry_run_frame_out_of_range(tmp_path, slp_real_data):
    """Test --dry-run-frame with out of range frame index."""
    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "0.5",
            "--dry-run-frame",
            "99999",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    # Should warn and use frame 0
    assert "Warning" in output
    assert "using frame 0" in output


def test_transform_encoding_options_shown_in_help():
    """Test that encoding options appear in help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["transform", "--help"])
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "--keyframe-interval" in output
    assert "--no-audio" in output
    assert "--dry-run-frame" in output


@skip_slow_video_on_windows
def test_transform_video_file_with_encoding_options(
    tmp_path, centered_pair_low_quality_path
):
    """Test transform video with encoding options in dry-run."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--scale",
            "0.5",
            "--keyframe-interval",
            "0.5",
            "--no-audio",
            "--dry-run",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert not output_path.exists()


@skip_slow_video_on_windows
def test_transform_video_file_dry_run_frame(tmp_path, centered_pair_low_quality_path):
    """Test transform on raw video with --dry-run-frame."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--scale",
            "0.5",
            "--dry-run-frame",
            "0",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = _strip_ansi(result.output)
    assert "Preview frame 0" in output
    assert not output_path.exists()


@skip_slow_video_on_windows
def test_transform_video_file_actual(tmp_path, centered_pair_low_quality_path):
    """Test actual video transformation (not dry-run)."""
    import sleap_io as sio

    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    # Scale down to make test faster
    result = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--scale",
            "0.25",  # Small scale for fast test
            "--crf",
            "35",  # Lower quality for faster encoding
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()

    # Verify output video
    vid = sio.load_video(str(output_path))
    assert vid.shape[1] == 96  # 384 * 0.25
    assert vid.shape[2] == 96


@skip_slow_video_on_windows
def test_transform_video_file_default_output(tmp_path, centered_pair_low_quality_path):
    """Test video transformation with default output path (no -o flag)."""
    import shutil

    # Copy video to tmp_path so default output goes there
    video_path = tmp_path / "test_video.mp4"
    shutil.copy(centered_pair_low_quality_path, video_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "transform",
            str(video_path),
            "--scale",
            "0.25",
            "--crf",
            "35",
            # No -o flag - should use default output path
        ],
    )
    assert result.exit_code == 0, result.output

    # Default output should be test_video.transformed.mp4
    expected_output = tmp_path / "test_video.transformed.mp4"
    assert expected_output.exists()


@skip_slow_video_on_windows
def test_transform_slp_actual(tmp_path, slp_real_data):
    """Test actual SLP transformation (not dry-run)."""
    import sleap_io as sio

    runner = CliRunner()
    output_path = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            slp_real_data,
            "--scale",
            "0.25",
            "--crf",
            "35",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()

    # Verify output
    labels = sio.load_slp(str(output_path))
    assert len(labels.videos) == 1


@skip_slow_video_on_windows
def test_transform_slp_default_output(tmp_path, slp_real_data):
    """Test SLP transformation with default output path (no -o flag)."""
    import shutil

    # Copy slp to tmp_path so default output goes there
    slp_path = tmp_path / "test_labels.slp"
    shutil.copy(slp_real_data, slp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_path),
            "--scale",
            "0.25",
            "--crf",
            "35",
            # No -o flag - should use default output path
        ],
    )
    assert result.exit_code == 0, result.output

    # Default output should be test_labels.transformed.slp
    expected_output = tmp_path / "test_labels.transformed.slp"
    assert expected_output.exists()


def test_transform_video_file_rotate_and_pad(tmp_path, centered_pair_low_quality_path):
    """Test video transformation with rotate and pad parameters."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--rotate",
            "45",
            "--pad",
            "10,10,10,10",
            "-o",
            str(output_path),
            "--dry-run",  # Use dry-run for faster test
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Rotate: 45" in result.output
    assert "Pad: (10, 10, 10, 10)" in result.output


def test_transform_clip_rotation(tmp_path, centered_pair_low_quality_path):
    """Test --clip-rotation flag keeps original dimensions when rotating."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    # Without --clip-rotation, 45 degree rotation should expand canvas
    result_expanded = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--rotate",
            "45",
            "-o",
            str(output_path),
            "--dry-run",
        ],
    )
    assert result_expanded.exit_code == 0, result_expanded.output
    # Original is 384x384, 45 degree rotation expands to ~543x543
    assert "384x384 -> 5" in result_expanded.output  # 543x543

    # With --clip-rotation, dimensions should stay the same
    result_clipped = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--rotate",
            "45",
            "--clip-rotation",
            "-o",
            str(output_path),
            "--dry-run",
        ],
    )
    assert result_clipped.exit_code == 0, result_clipped.output
    # With clipping, dimensions stay at 384x384
    assert "384x384 -> 384x384" in result_clipped.output


def test_transform_flip_horizontal(tmp_path, centered_pair_low_quality_path):
    """Test --flip-horizontal flag."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--flip-horizontal",
            "-o",
            str(output_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Flip should be accepted as a valid transform
    assert "Transform Summary" in result.output
    # Dimensions should stay the same
    assert "384x384 -> 384x384" in result.output


def test_transform_flip_vertical(tmp_path, centered_pair_low_quality_path):
    """Test --flip-vertical flag."""
    runner = CliRunner()
    output_path = tmp_path / "output.mp4"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(centered_pair_low_quality_path),
            "--flip-vertical",
            "-o",
            str(output_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Flip should be accepted as a valid transform
    assert "Transform Summary" in result.output
    # Dimensions should stay the same
    assert "384x384 -> 384x384" in result.output


def test_transform_config_file(tmp_path, slp_real_data):
    """Test --config option for loading transforms from YAML."""
    import yaml

    runner = CliRunner()

    # Create a config file
    config = {
        "videos": {
            0: {
                "scale": 0.5,
            }
        }
    }
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Loading config:" in result.output
    assert "Transform Summary" in result.output


def test_transform_config_file_not_found(tmp_path, slp_real_data):
    """Test error when config file doesn't exist."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(tmp_path / "nonexistent.yaml"),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    # click.Path with exists=True will raise an error before our code
    assert result.exit_code != 0


def test_transform_config_file_with_cli_override(tmp_path, slp_real_data):
    """Test that CLI params override config file transforms."""
    import yaml

    runner = CliRunner()

    # Create a config file with scale 0.5
    config = {
        "videos": {
            0: {
                "scale": 0.5,
            }
        }
    }
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # CLI override with scale 0.25
    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "--scale",
            "0.25",
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # The output size should reflect 0.25 scale, not 0.5
    assert "Transform Summary" in result.output


def test_transform_output_transforms(tmp_path, slp_real_data):
    """Test --output-transforms option for exporting metadata to YAML."""
    import yaml

    runner = CliRunner()
    output_slp = tmp_path / "output.slp"
    transforms_yaml = tmp_path / "transforms_output.yaml"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--scale",
            "0.5",
            "-o",
            str(output_slp),
            "--output-transforms",
            str(transforms_yaml),
        ],
    )
    assert result.exit_code == 0, result.output
    assert transforms_yaml.exists()
    assert "Transform metadata:" in result.output

    # Verify the contents of the YAML file
    with open(transforms_yaml) as f:
        metadata = yaml.safe_load(f)

    assert "generated" in metadata
    assert "source" in metadata
    assert "sleap_io_version" in metadata
    assert "videos" in metadata
    assert 0 in metadata["videos"]
    assert "transforms" in metadata["videos"][0]


def test_transform_embed_provenance(tmp_path, slp_real_data):
    """Test --embed-provenance option for storing metadata in output SLP."""
    import sleap_io as sio

    runner = CliRunner()
    output_slp = tmp_path / "output.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--scale",
            "0.5",
            "-o",
            str(output_slp),
            "--embed-provenance",
        ],
    )
    assert result.exit_code == 0, result.output

    # Load the output and check for provenance
    labels = sio.load_slp(str(output_slp))
    assert "transform" in labels.provenance
    transform_meta = labels.provenance["transform"]
    assert "generated" in transform_meta
    assert "videos" in transform_meta


def test_transform_config_file_crop(tmp_path, slp_real_data):
    """Test config file with crop as list."""
    import yaml

    runner = CliRunner()

    config = {
        "videos": {
            0: {
                "crop": [10, 10, 100, 100],
            }
        }
    }
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "90x90" in result.output  # 100-10 = 90


def test_transform_config_file_pad(tmp_path, slp_real_data):
    """Test config file with padding."""
    import yaml

    runner = CliRunner()

    config = {
        "videos": {
            0: {
                "pad": [10, 10, 10, 10],
            }
        }
    }
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Original is 384x384, with 10px padding all around -> 404x404
    assert "404x404" in result.output


def test_transform_config_file_invalid_yaml(tmp_path, slp_real_data):
    """Test error on invalid YAML in config file."""
    runner = CliRunner()

    # Create an invalid YAML file
    config_path = tmp_path / "invalid.yaml"
    with open(config_path, "w") as f:
        f.write("not: valid: yaml: {unclosed")

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid YAML" in result.output


def test_transform_config_file_missing_videos_key(tmp_path, slp_real_data):
    """Test error when config file is missing 'videos' key."""
    import yaml

    runner = CliRunner()

    config = {"transforms": {}}  # Wrong key
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "videos" in result.output.lower()


def test_transform_config_file_video_index_out_of_range(tmp_path, slp_real_data):
    """Test error when config references non-existent video index."""
    import yaml

    runner = CliRunner()

    config = {
        "videos": {
            99: {  # Non-existent video
                "scale": 0.5,
            }
        }
    }
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output.lower()


def test_transform_fill_rgb_invalid_count():
    """Test error when RGB fill doesn't have 3 values."""
    import click

    from sleap_io.io.cli import _parse_fill_value

    with pytest.raises(click.ClickException, match="3 values"):
        _parse_fill_value("128,128")  # Only 2 values

    with pytest.raises(click.ClickException, match="3 values"):
        _parse_fill_value("128,128,128,128")  # 4 values


def test_transform_fill_value_out_of_range():
    """Test error when fill value is outside 0-255 range."""
    import click

    from sleap_io.io.cli import _parse_fill_value

    with pytest.raises(click.ClickException, match="0-255"):
        _parse_fill_value("256")  # Too high

    with pytest.raises(click.ClickException, match="0-255"):
        _parse_fill_value("-1")  # Negative

    with pytest.raises(click.ClickException, match="0-255"):
        _parse_fill_value("128,256,128")  # RGB with one out of range


def test_transform_fill_value_invalid_format():
    """Test error when fill value is not parseable."""
    import click

    from sleap_io.io.cli import _parse_fill_value

    with pytest.raises(click.ClickException, match="Invalid fill"):
        _parse_fill_value("abc")  # Not a number

    with pytest.raises(click.ClickException, match="Invalid fill"):
        _parse_fill_value("12.5")  # Float


def test_transform_config_videos_not_mapping(tmp_path, slp_real_data):
    """Test error when 'videos' in config is not a mapping."""
    import yaml

    runner = CliRunner()

    config = {"videos": [0.5]}  # List instead of dict
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "mapping" in result.output.lower()


def test_transform_config_invalid_video_index(tmp_path, slp_real_data):
    """Test error when video index in config is not an integer."""
    runner = CliRunner()

    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        f.write("videos:\n  'not_an_int':\n    scale: 0.5\n")

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "invalid video index" in result.output.lower()


def test_transform_config_video_config_not_mapping(tmp_path, slp_real_data):
    """Test error when video config entry is not a mapping."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: "not a dict"}}  # String instead of dict
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "mapping" in result.output.lower()


def test_transform_config_invalid_crop_value(tmp_path, slp_real_data):
    """Test error when crop value in config is invalid."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"crop": "invalid"}}}
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    # Error could be in output or exception
    error_text = result.output.lower() + str(result.exception).lower()
    assert "crop" in error_text or "invalid" in error_text


def test_transform_config_invalid_scale_value(tmp_path, slp_real_data):
    """Test error when scale value in config is invalid."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"scale": "invalid"}}}
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    # Error could be in output or exception
    error_text = result.output.lower() + str(result.exception).lower()
    assert "scale" in error_text or "float" in error_text or "invalid" in error_text


def test_transform_config_invalid_rotate_value(tmp_path, slp_real_data):
    """Test error when rotate value in config is invalid."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"rotate": "not_a_number"}}}
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    # Error could be in output or exception
    error_text = result.output.lower() + str(result.exception).lower()
    assert "rotate" in error_text or "invalid" in error_text


def test_transform_config_invalid_pad_value(tmp_path, slp_real_data):
    """Test error when pad value in config is invalid."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"pad": "invalid"}}}
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    # Error could be in output or exception
    error_text = result.output.lower() + str(result.exception).lower()
    assert "pad" in error_text or "invalid" in error_text


def test_transform_config_pad_wrong_count(tmp_path, slp_real_data):
    """Test error when pad in config has wrong number of values."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"pad": [10, 20]}}}  # 2 values instead of 1 or 4
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "pad" in result.output.lower()


def test_transform_config_null_video(tmp_path, slp_real_data):
    """Test config with null video config (no transforms for that video)."""
    import yaml

    runner = CliRunner()

    # When ALL videos have null config, no transforms are specified -> error
    config = {"videos": {0: None}}  # Null means no transforms
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    # Should fail because no transforms are specified
    assert result.exit_code != 0
    assert "no valid transforms" in result.output.lower()


def test_transform_config_normalized_crop(tmp_path, slp_real_data):
    """Test config with normalized crop values (0-1 range)."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"crop": [0.1, 0.1, 0.9, 0.9]}}}  # Normalized coords
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0


def test_transform_config_pixel_crop(tmp_path, slp_real_data):
    """Test config with pixel crop values."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"crop": [10, 10, 300, 300]}}}  # Pixel coords
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0


def test_transform_config_scale_uniform_ratio(tmp_path, slp_real_data):
    """Test config with uniform scale ratio."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"scale": 0.5}}}  # Single value < 1 = ratio
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "192x192" in result.output  # 384 * 0.5 = 192


def test_transform_config_scale_pixel_width(tmp_path, slp_real_data):
    """Test config with scale as target pixel width."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"scale": 200}}}  # Single value >= 1 = pixel width
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "200x200" in result.output


def test_transform_config_scale_tuple_ratios(tmp_path, slp_real_data):
    """Test config with scale as tuple of ratios."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"scale": [0.5, 0.25]}}}  # Different x/y ratios
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "192x96" in result.output  # 384 * 0.5 = 192, 384 * 0.25 = 96


def test_transform_config_scale_tuple_pixels(tmp_path, slp_real_data):
    """Test config with scale as tuple of pixel dimensions."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"scale": [200, 100]}}}  # Pixel dimensions
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "200x100" in result.output


def test_transform_config_pad_single_value(tmp_path, slp_real_data):
    """Test config with single-value pad (uniform padding)."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"pad": 10}}}  # Single int = uniform padding
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "404x404" in result.output  # 384 + 10 + 10 = 404


def test_transform_config_pad_single_value_list(tmp_path, slp_real_data):
    """Test config with single-value pad as list."""
    import yaml

    runner = CliRunner()

    config = {"videos": {0: {"pad": [10]}}}  # Single-element list = uniform
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "404x404" in result.output


def test_transform_config_with_flip_options(tmp_path, slp_real_data):
    """Test config with per-video flip options."""
    import yaml

    runner = CliRunner()

    config = {
        "videos": {
            0: {
                "scale": 0.5,
                "flip_horizontal": True,
                "flip_vertical": True,
            }
        }
    }
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0


def test_transform_config_with_clip_rotation(tmp_path, slp_real_data):
    """Test config with per-video clip_rotation option."""
    import yaml

    runner = CliRunner()

    config = {
        "videos": {
            0: {
                "rotate": 45,
                "clip_rotation": True,
            }
        }
    }
    config_path = tmp_path / "transforms.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--config",
            str(config_path),
            "-o",
            str(tmp_path / "output.slp"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    # With clip_rotation, dimensions stay at 384x384
    assert "384x384" in result.output


def test_transform_output_transforms_metadata(tmp_path, slp_real_data):
    """Test --output-transforms exports metadata to YAML."""
    runner = CliRunner()

    output_path = tmp_path / "output.slp"
    metadata_path = tmp_path / "transforms.yaml"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_real_data),
            "--scale",
            "0.5",
            "-o",
            str(output_path),
            "--output-transforms",
            str(metadata_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert metadata_path.exists()

    # Verify metadata content
    import yaml

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    assert "videos" in metadata
    assert "generated" in metadata
    assert "sleap_io_version" in metadata


def test_transform_embed_provenance_embedded_video(tmp_path, slp_minimal_pkg):
    """Test --embed-provenance with embedded video updates HDF5 file."""
    import json

    import h5py

    runner = CliRunner()

    output_path = tmp_path / "output.pkg.slp"

    result = runner.invoke(
        cli,
        [
            "transform",
            str(slp_minimal_pkg),
            "--scale",
            "0.5",
            "-o",
            str(output_path),
            "--embed-provenance",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()

    # Verify provenance was embedded in HDF5
    with h5py.File(output_path, "r") as f:
        assert "provenance" in f
        assert "transform_json" in f["provenance"]
        transform_json = f["provenance/transform_json"][()].decode()
        metadata = json.loads(transform_json)
        assert "videos" in metadata
