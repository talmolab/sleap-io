"""Command-line interface for sleap-io.

Provides a `sio` command with subcommands for inspecting and manipulating
SLEAP labels and related formats. This module intentionally keeps the
default behavior light-weight (no video backends by default) to work well
in minimal environments and CI.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Optional

import rich_click as click

from sleap_io.io import main as io_main
from sleap_io.io.utils import sanitize_filename
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.version import __version__

# Rich-click theme configuration
click.rich_click.THEME = "solarized-slim"
click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 100
click.rich_click.ERRORS_EPILOGUE = (
    "See [link=https://io.sleap.ai]io.sleap.ai[/] for documentation."
)

# Command panels for organized help display
click.rich_click.COMMAND_GROUPS = {
    "sio": [
        {"name": "Inspection", "commands": ["cat"]},
        {"name": "Transformation", "commands": ["convert"]},
    ]
}

# Supported formats for conversion
INPUT_FORMATS = [
    "slp",
    "nwb",
    "coco",
    "labelstudio",
    "alphatracker",
    "jabs",
    "dlc",
    "ultralytics",
    "leap",
]
OUTPUT_FORMATS = ["slp", "nwb", "coco", "labelstudio", "jabs", "ultralytics"]

# Extensions that require explicit --from format
AMBIGUOUS_EXTENSIONS = {".json", ".h5"}

# Extension to format mapping for unambiguous detection
EXTENSION_TO_FORMAT = {
    ".slp": "slp",
    ".nwb": "nwb",
    ".mat": "leap",
    ".csv": "dlc",
}

# Output extension to format mapping
OUTPUT_EXTENSION_TO_FORMAT = {
    ".slp": "slp",
    ".nwb": "nwb",
    ".json": "labelstudio",  # Default JSON output to labelstudio
}


def _get_package_version(package: str) -> str:
    """Get version of a package, or 'not installed' if not available."""
    try:
        return pkg_version(package)
    except Exception:
        return "not installed"


def _print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print version info with plugin status and exit."""
    if not value or ctx.resilient_parsing:
        return

    lines = [f"sleap-io {__version__}"]
    lines.append(f"python {sys.version.split()[0]}")
    lines.append("")

    # Core dependencies
    lines.append("Core:")
    lines.append(f"  numpy: {_get_package_version('numpy')}")
    lines.append(f"  h5py: {_get_package_version('h5py')}")
    lines.append(f"  imageio: {_get_package_version('imageio')}")
    lines.append("")

    # Video plugins
    lines.append("Video plugins:")
    lines.append(f"  opencv: {_get_package_version('opencv-python')}")
    lines.append(f"  pyav: {_get_package_version('av')}")
    lines.append(f"  imageio-ffmpeg: {_get_package_version('imageio-ffmpeg')}")
    lines.append("")

    # Optional dependencies
    lines.append("Optional:")
    lines.append(f"  pymatreader: {_get_package_version('pymatreader')}")

    click.echo("\n".join(lines))
    ctx.exit()


@click.group()
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show version and plugin info, then exit.",
)
def cli():
    """[bold cyan]sleap-io[/] - Standalone utilities for pose tracking data.

    Read, write, and manipulate pose data from [bold]SLEAP[/], [bold]NWB[/],
    [bold]COCO[/], [bold]DeepLabCut[/], and other formats.

    [dim]Examples:[/]

        $ sio cat labels.slp
        $ sio cat labels.slp --skeleton
    """
    pass


def _print_labels_summary(labels: Labels, src_path: str) -> str:
    num_videos = len(labels.videos)
    num_frames = len(labels.labeled_frames)
    num_instances = sum(len(lf) for lf in labels.labeled_frames)
    num_skeletons = len(labels.skeletons)

    lines = [
        f"file: {sanitize_filename(src_path)}",
        "type: labels",
        f"videos: {num_videos}",
        f"labeled_frames: {num_frames}",
        f"instances: {num_instances}",
        f"skeletons: {num_skeletons}",
    ]
    return "\n".join(lines)


def _format_point_preview(inst: Instance | PredictedInstance, limit: int = 3) -> str:
    names = inst.points["name"].tolist()
    xy = inst.numpy()
    # For PredictedInstance, numpy() may return NaNs for invisible as well; that's fine
    items = []
    for i, name in enumerate(names[:limit]):
        x, y = xy[i]
        vis = bool(inst.points[i]["visible"])  # structured dtype access
        items.append(f"{name}: {x:.1f},{y:.1f},{int(vis)}")
    return "; ".join(items)


def _print_lf_details(lf: LabeledFrame) -> str:
    video_name = (
        lf.video.filename
        if isinstance(lf.video.filename, str)
        else str(lf.video.filename)
    )
    lines = [
        f"frame_idx: {lf.frame_idx}",
        f"video: {video_name}",
        f"instances: {len(lf)}",
    ]
    for idx, inst in enumerate(lf):
        track = inst.track.name if inst.track is not None else None
        n_points = len(inst)
        n_visible = inst.n_visible
        preview = _format_point_preview(inst)
        if isinstance(inst, PredictedInstance):
            score = inst.score
            lines.append(
                f"- {idx}: predicted track={track} points={n_points} \
                    visible={n_visible} score={score:.2f} | {preview}"
            )
        else:
            lines.append(
                f"- {idx}: user track={track} points={n_points} \
                    visible={n_visible} | {preview}"
            )
    return "\n".join(lines)


def _print_skeleton_edges(sk: Skeleton) -> str:
    node_names = sk.node_names
    edges = sk.edge_names
    lines = [f"nodes ({len(node_names)}): {', '.join(node_names)}"]
    if edges:
        edge_lines = [f"{a} - {b}" for a, b in edges]
        lines.append("edges (source-destination):")
        lines.extend([f"  - {ln}" for ln in edge_lines])
    else:
        lines.append("edges: none")
    return "\n".join(lines)


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "open_videos",
    "--open-videos/--no-open-videos",
    default=False,
    help="Open video backends when reading files (defaults to no).",
)
@click.option(
    "lf_index",
    "--lf",
    type=int,
    default=None,
    help="Print details for labeled frame at index N (0-based).",
)
@click.option(
    "skeleton",
    "--skeleton",
    is_flag=True,
    help="Print skeleton details (node names).",
)
def cat(
    path: Path,
    open_videos: bool,
    lf_index: Optional[int],
    skeleton: bool,
):
    """Print a summary and optional details."""
    obj = io_main.load_file(str(path), open_videos=open_videos)

    if isinstance(obj, Labels):
        # Text output
        click.echo(_print_labels_summary(obj, str(path)))

        # Skeleton details (text)
        if skeleton and len(obj.skeletons):
            for i, sk in enumerate(obj.skeletons):
                click.echo("")
                click.echo(f"skeleton[{i}]")
                click.echo(_print_skeleton_edges(sk))

        # Labeled frame details
        if lf_index is not None:
            n = len(obj.labeled_frames)
            if n == 0:
                raise click.ClickException("No labeled frames present in file.")
            if lf_index < 0 or lf_index >= n:
                raise click.ClickException(f"--lf out of range (0..{n - 1})")
            lf = obj.labeled_frames[lf_index]
            click.echo("")
            click.echo(_print_lf_details(lf))
    else:
        # For non-Labels objects, print repr
        click.echo(repr(obj))


def _infer_input_format(path: Path) -> Optional[str]:
    """Infer input format from file path.

    Args:
        path: Path to the input file.

    Returns:
        Format string if unambiguous, None if ambiguous or unknown.
    """
    suffix = path.suffix.lower()

    # Check for ultralytics (directory with data.yaml)
    if path.is_dir():
        if (path / "data.yaml").exists():
            return "ultralytics"
        return None

    # Check unambiguous extensions
    if suffix in EXTENSION_TO_FORMAT:
        return EXTENSION_TO_FORMAT[suffix]

    # Ambiguous extensions require explicit --from
    if suffix in AMBIGUOUS_EXTENSIONS:
        return None

    return None


def _infer_output_format(path: Path) -> Optional[str]:
    """Infer output format from file path.

    Args:
        path: Path to the output file.

    Returns:
        Format string if determinable, None otherwise.
    """
    suffix = path.suffix.lower()

    # Check if it's a directory (likely ultralytics)
    if path.is_dir() or suffix == "":
        return "ultralytics"

    # Check known extensions
    if suffix in OUTPUT_EXTENSION_TO_FORMAT:
        return OUTPUT_EXTENSION_TO_FORMAT[suffix]

    return None


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input file path.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file path.",
)
@click.option(
    "--from",
    "input_format",
    type=click.Choice(INPUT_FORMATS),
    help="Input format (required for .json and .h5 files).",
)
@click.option(
    "--to",
    "output_format",
    type=click.Choice(OUTPUT_FORMATS),
    help="Output format (inferred from extension if not specified).",
)
@click.option(
    "--embed",
    type=click.Choice(["user", "all", "suggestions", "source"]),
    help="Embed frames in output (SLP format only).",
)
def convert(
    input_path: Path,
    output_path: Path,
    input_format: Optional[str],
    output_format: Optional[str],
    embed: Optional[str],
):
    """Convert between pose data formats.

    Reads a labels file in one format and writes it to another format.
    Input and output formats are inferred from file extensions when possible,
    but can be explicitly specified using --from and --to.

    [bold]Supported input formats:[/]
    slp, nwb, coco, labelstudio, alphatracker, jabs, dlc, ultralytics, leap

    [bold]Supported output formats:[/]
    slp, nwb, coco, labelstudio, jabs, ultralytics

    [dim]Examples:[/]

        $ sio convert -i labels.slp -o labels.nwb
        $ sio convert -i annotations.json -o labels.slp --from coco
        $ sio convert -i labels.slp -o labels.pkg.slp --embed user
        $ sio convert -i labels.slp -o dataset/ --to ultralytics
    """
    # Resolve input format
    resolved_input_format = input_format
    if resolved_input_format is None:
        resolved_input_format = _infer_input_format(input_path)
        if resolved_input_format is None:
            suffix = input_path.suffix.lower()
            if suffix in AMBIGUOUS_EXTENSIONS:
                raise click.ClickException(
                    f"Cannot infer format from '{suffix}' extension. "
                    f"Please specify --from with one of: {', '.join(INPUT_FORMATS)}"
                )
            else:
                raise click.ClickException(
                    f"Cannot infer input format from '{input_path.name}'. "
                    f"Please specify --from with one of: {', '.join(INPUT_FORMATS)}"
                )

    # Resolve output format
    resolved_output_format = output_format
    if resolved_output_format is None:
        resolved_output_format = _infer_output_format(output_path)
        if resolved_output_format is None:
            raise click.ClickException(
                f"Cannot infer output format from '{output_path.name}'. "
                f"Please specify --to with one of: {', '.join(OUTPUT_FORMATS)}"
            )

    # Validate embed option
    if embed is not None and resolved_output_format != "slp":
        raise click.ClickException("--embed is only valid for SLP output format.")

    # Determine if we need video access:
    # - Embedding frames requires video access
    # - Ultralytics output exports images, requires video access
    # - NWB output needs video metadata
    needs_video = (
        embed is not None
        or resolved_output_format == "ultralytics"
        or resolved_output_format == "nwb"
    )

    # Load the input file
    # Note: open_videos is only valid for SLP format
    try:
        load_kwargs = {"format": resolved_input_format}
        if resolved_input_format == "slp":
            load_kwargs["open_videos"] = needs_video
        labels = io_main.load_file(str(input_path), **load_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to load input file: {e}")

    if not isinstance(labels, Labels):
        raise click.ClickException(
            f"Input file is not a labels file (got {type(labels).__name__})."
        )

    # Prepare output directory for ultralytics
    if resolved_output_format == "ultralytics":
        output_path.mkdir(parents=True, exist_ok=True)

    # Save the output file
    try:
        save_kwargs = {"format": resolved_output_format}
        if embed is not None:
            save_kwargs["embed"] = embed
        io_main.save_file(labels, str(output_path), **save_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Success message
    click.echo(f"Converted: {input_path} -> {output_path}")
    click.echo(f"Format: {resolved_input_format} -> {resolved_output_format}")
    if embed:
        click.echo(f"Embedded frames: {embed}")
