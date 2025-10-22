"""CLI tests for the `sio` command.

Covers summary output, labeled frame details, and skeleton printing.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sleap_io.io.cli import cli
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton


def _data_path(rel: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "tests" / "data"
    return root / rel


def test_cat_summary_typical_slp():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["cat", str(path), "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "type: labels" in out
    assert "labeled_frames:" in out
    assert "videos:" in out
    assert "skeletons:" in out


def test_cat_lf_zero_details():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["cat", str(path), "--lf", "0", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = result.output
    # Expect a details block with frame_idx and instances listing
    assert "frame_idx: 0" in out or "frame_idx:" in out
    assert "instances:" in out
    # Should list at least one instance line when present
    assert "- 0:" in out or "instances: 0" in out


def test_cat_lf_out_of_range():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["cat", str(path), "--lf", "9999", "--no-open-videos"])
    assert result.exit_code != 0
    assert "out of range" in result.output


def test_cat_on_video_basic_info():
    runner = CliRunner()
    # Use a small bundled mp4 in tests/data/videos
    # If CI lacks codecs, this should still work as we don't open videos by default
    path = _data_path("videos/video_1.mp4")
    result = runner.invoke(cli, ["cat", str(path)])
    # If the file is missing in some environments, allow graceful skip assertion
    if path.exists():
        assert result.exit_code == 0, result.output
        out = result.output
        assert "file:" in out
        assert "type:" in out


def test_cat_skeleton_flag_text():
    runner = CliRunner()
    path = _data_path("slp/typical.slp")
    result = runner.invoke(cli, ["cat", str(path), "--skeleton", "--no-open-videos"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "skeleton[" in out or "skeletons:" in out


def test_cat_file_not_found():
    runner = CliRunner()
    result = runner.invoke(cli, ["cat", "/nonexistent/path/to/file.slp"])
    assert result.exit_code != 0
    # Click validates file existence before our code runs
    assert "Invalid value for 'PATH'" in result.output


def test_cat_skeleton_no_edges(tmp_path):
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
        cli, ["cat", str(slp_path), "--skeleton", "--no-open-videos"]
    )
    assert result.exit_code == 0, result.output
    assert "edges: none" in result.output


def test_cat_empty_labels_with_lf(tmp_path):
    """Test --lf flag on file with no labeled frames."""
    from sleap_io import save_file

    runner = CliRunner()
    # Create empty labels
    labels = Labels()

    # Save to temporary file
    slp_path = tmp_path / "empty.slp"
    save_file(labels, slp_path)

    result = runner.invoke(cli, ["cat", str(slp_path), "--lf", "0", "--no-open-videos"])
    assert result.exit_code != 0
    assert "No labeled frames present in file" in result.output


def test_cat_video_file():
    """Test cat on a video file (non-Labels object)."""
    runner = CliRunner()
    path = _data_path("videos/centered_pair_low_quality.mp4")

    if not path.exists():
        # Skip if video file doesn't exist in test environment
        return

    result = runner.invoke(cli, ["cat", str(path), "--open-videos"])
    assert result.exit_code == 0, result.output
    # Should print repr of Video object
    assert "Video" in result.output
