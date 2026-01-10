"""CLI tests for the `sio` command.

Covers summary output, labeled frame details, skeleton printing, and format conversion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from sleap_io import load_slp
from sleap_io.io.cli import cli
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.version import __version__


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
    assert "Status" in out
    assert "Plugin" in out or "Backend" in out


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
    """Test that header shows labeled/predicted instance counts."""
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["show", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # typical.slp has both user and predicted instances
    assert "labeled" in out
    assert "predicted" in out


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
    """Test provenance display with list and dict values."""
    runner = CliRunner()
    path = _data_path("slp/predictions_1.2.7_provenance_and_tracking.slp")
    result = runner.invoke(cli, ["show", str(path), "--provenance", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # This file has 'args' which is a dict
    assert "args:" in out
    assert "keys" in out  # Shows "{...} (N keys)"


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


def test_show_standalone_video_with_backend_metadata():
    """Test standalone video display with backend_metadata grayscale field."""
    from io import StringIO

    from rich.console import Console

    from sleap_io import Video
    from sleap_io.io.cli import _print_video_standalone

    path = _data_path("videos/centered_pair_low_quality.mp4")
    if not path.exists():
        return

    # Load video and set backend_metadata
    video = Video.from_filename(str(path))
    video.backend_metadata = {"grayscale": True}

    import sleap_io.io.cli as cli_module

    original_console = cli_module.console
    string_io = StringIO()
    cli_module.console = Console(file=string_io, force_terminal=True)

    try:
        _print_video_standalone(path, video)
        out = _strip_ansi(string_io.getvalue())
        # Should show grayscale info from backend_metadata
        assert "Grayscale" in out
        assert "yes" in out
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
    """Test provenance with long list values gets truncated."""
    from sleap_io import Labels, save_file

    runner = CliRunner()

    # Create labels with long list provenance
    labels = Labels()
    labels.provenance = {"model_paths": [f"/path/to/model_{i}.h5" for i in range(10)]}

    slp_path = tmp_path / "long_list_provenance.slp"
    save_file(labels, slp_path)

    result = runner.invoke(
        cli, ["show", str(slp_path), "--provenance", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    out = _strip_ansi(result.output)
    # Should show truncation
    assert "10 total" in out


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
    assert "Replaced filenames in 1 video(s) using list mode" in output
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
    assert f"Replaced filenames in {n_videos} video(s) using list mode" in output

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
    assert "using map mode" in output

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
    assert "using prefix mode" in output

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
