"""CLI tests for the `sio` command.

Covers summary output, labeled frame details, skeleton printing, and format conversion.
"""

from __future__ import annotations

import re
from pathlib import Path

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
    assert "cat" in result.output
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
