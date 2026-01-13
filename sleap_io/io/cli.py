"""Command-line interface for sleap-io.

Provides a `sio` command with subcommands for inspecting and manipulating
SLEAP labels and related formats. This module intentionally keeps the
default behavior light-weight (video support via bundled imageio-ffmpeg) to work well
in minimal environments and CI.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import rich_click as click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sleap_io.io import main as io_main
from sleap_io.io.video_reading import HDF5Video
from sleap_io.model.instance import PredictedInstance
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video
from sleap_io.version import __version__


@dataclass
class VideoEncodingInfo:
    """Video encoding information extracted via ffmpeg.

    Attributes:
        codec: Video codec name (e.g., "h264", "hevc").
        codec_profile: Codec profile (e.g., "Main", "High").
        pixel_format: Pixel format (e.g., "yuv420p").
        bitrate_kbps: Bitrate in kilobits per second.
        fps: Frames per second.
        gop_size: Group of pictures size (keyframe interval).
        container: Container format (e.g., "mov", "avi").
    """

    codec: str | None = None
    codec_profile: str | None = None
    pixel_format: str | None = None
    bitrate_kbps: int | None = None
    fps: float | None = None
    gop_size: int | None = None
    container: str | None = None


# Rich console for formatted output
# Set minimum width to avoid truncation issues in CI/non-TTY environments
console = Console(width=120, soft_wrap=True)

# Rich-click theme configuration
click.rich_click.THEME = "solarized-slim"
click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 100
# Show documentation link only in --help, not in error messages
click.rich_click.ERRORS_EPILOGUE = ""

# Command panels for organized help display
click.rich_click.COMMAND_GROUPS = {
    "sio": [
        {"name": "Inspection", "commands": ["show", "filenames"]},
        {
            "name": "Transformation",
            "commands": [
                "convert",
                "split",
                "unsplit",
                "merge",
                "trim",
                "reencode",
                "transform",
            ],
        },
        {"name": "Embedding", "commands": ["embed", "unembed"]},
        {"name": "Maintenance", "commands": ["fix"]},
        {"name": "Rendering", "commands": ["render"]},
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
    "csv",
    "ultralytics",
    "leap",
]
OUTPUT_FORMATS = ["slp", "nwb", "coco", "labelstudio", "jabs", "ultralytics", "csv"]

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
    ".csv": "csv",
}


def _resolve_input(
    input_arg: Path | None,
    input_opt: Path | None,
    name: str = "input file",
) -> Path:
    """Resolve input from positional argument or -i option.

    Args:
        input_arg: Positional argument value (may be None).
        input_opt: -i/--input option value (may be None).
        name: Human-readable name for error messages.

    Returns:
        The resolved input path.

    Raises:
        click.ClickException: If both provided or neither provided.
    """
    if input_arg is not None and input_opt is not None:
        raise click.ClickException(
            f"Cannot specify {name} both as positional argument and with -i/--input."
        )

    input_path = input_arg or input_opt
    if input_path is None:
        raise click.ClickException(
            f"Missing {name}. Provide as positional argument or with -i/--input."
        )

    return input_path


def _get_package_version(package: str) -> str:
    """Get version of a package, or 'not installed' if not available."""
    try:
        return pkg_version(package)
    except Exception:
        return "not installed"


def _parse_crop_string(
    crop_str: str | None,
) -> tuple | None:
    """Parse crop string from CLI into crop specification.

    Args:
        crop_str: Crop string from CLI. Can be:
            - None: No cropping
            - "x1,y1,x2,y2": Coordinates (pixels if integers, normalized if floats)

    Returns:
        None or (x1, y1, x2, y2) tuple.

    Raises:
        click.ClickException: If crop string format is invalid.
    """
    if crop_str is None:
        return None

    crop_str = crop_str.strip()

    # Parse x1,y1,x2,y2 format
    parts = crop_str.split(",")
    if len(parts) != 4:
        raise click.ClickException(
            f"Invalid --crop format: '{crop_str}'. "
            "Expected 'x1,y1,x2,y2' (e.g., '100,100,300,300' or "
            "'0.25,0.25,0.75,0.75')."
        )

    try:
        # Check if any value contains a decimal point -> treat as normalized floats
        if any("." in p for p in parts):
            values = tuple(float(p.strip()) for p in parts)
            # Validate normalized range
            if not all(0.0 <= v <= 1.0 for v in values):
                raise click.ClickException(
                    f"Normalized crop values must be in [0.0, 1.0] range. Got: {values}"
                )
        else:
            values = tuple(int(p.strip()) for p in parts)
    except ValueError as e:
        raise click.ClickException(
            f"Invalid --crop values: '{crop_str}'. Values must be numbers. Error: {e}"
        )

    return values


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
    lines.append(f"  skia-python: {_get_package_version('skia-python')}")
    lines.append(f"  colorcet: {_get_package_version('colorcet')}")
    lines.append("")

    # Video plugins (optional)
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


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
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

        $ sio show labels.slp
        $ sio show labels.slp --skeleton

    [dim]Documentation:[/] [link=https://io.sleap.ai]io.sleap.ai[/]
    """
    pass


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_skeleton_python_code(sk: Skeleton) -> list[str]:
    """Generate copyable Python code for skeleton definition."""
    lines = []

    # Nodes as list of strings
    node_names = [f'"{n}"' for n in sk.node_names]
    lines.append(f"nodes = [{', '.join(node_names)}]")

    # Edge indices as list of tuples
    if sk.edges:
        edge_inds = []
        for e in sk.edges:
            src_idx = sk.node_names.index(e.source.name)
            dst_idx = sk.node_names.index(e.destination.name)
            edge_inds.append(f"({src_idx}, {dst_idx})")
        lines.append(f"edge_inds = [{', '.join(edge_inds)}]")

    return lines


def _print_header(path: Path, labels: Labels) -> None:
    """Print file header panel with basic stats."""
    # Calculate file size
    file_size = path.stat().st_size if path.exists() else 0

    # Determine file type
    is_pkg = ".pkg.slp" in path.name.lower()
    file_type = "Package (embedded)" if is_pkg else "Labels"

    # Count instances using fast stats (works for both eager and lazy)
    n_user_inst = labels.n_user_instances
    n_pred_inst = labels.n_pred_instances

    # Count frames using fast stats
    n_total_frames = len(labels.labeled_frames)
    n_user_frames = labels.n_user_frames

    # Build header content (without full path - shown below to avoid truncation)
    header_lines = [
        f"[bold cyan]{path.name}[/]",
        f"[dim]{path.parent}[/]",
        "",
        f"[dim]Type:[/]     {file_type}",
        f"[dim]Size:[/]     {_format_file_size(file_size)}",
    ]

    # Stats row - videos and frames
    stats_parts = [
        f"[bold]{len(labels.videos)}[/] video{'s' if len(labels.videos) != 1 else ''}",
    ]

    # Frame counts - show user frames vs total if different
    if n_user_frames > 0 and n_user_frames != n_total_frames:
        stats_parts.append(
            f"[bold]{n_user_frames}[/] user frames ([dim]{n_total_frames} total[/])"
        )
    else:
        stats_parts.append(
            f"[bold]{n_total_frames}[/] frame{'s' if n_total_frames != 1 else ''}"
        )

    # Instance counts - always clarify user vs predicted
    if n_user_inst > 0:
        stats_parts.append(f"[bold]{n_user_inst}[/] user instances")
    if n_pred_inst > 0:
        stats_parts.append(f"[bold]{n_pred_inst}[/] predicted")

    if labels.tracks:
        stats_parts.append(
            f"[bold]{len(labels.tracks)}[/] "
            f"track{'s' if len(labels.tracks) != 1 else ''}"
        )

    header_lines.append("")
    header_lines.append(" | ".join(stats_parts))

    console.print(
        Panel(
            "\n".join(header_lines),
            title="[bold]sleap-io[/]",
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Show full path below panel to avoid truncation
    full_path = path.resolve()
    console.print(f"  [dim]Full:[/] {full_path}")


def _print_video_standalone(path: Path, video: Video) -> None:
    """Print header panel and details for a standalone video file."""
    # Calculate file size
    file_size = path.stat().st_size if path.exists() else 0

    # Get video info using defensive helpers
    video_type = _get_video_type(video)

    # Build header content (without full path - shown below to avoid truncation)
    header_lines = [
        f"[bold cyan]{path.name}[/]",
        f"[dim]{path.parent}[/]",
        "",
        f"[dim]Type:[/]     Video ({video_type})",
        f"[dim]Size:[/]     {_format_file_size(file_size)}",
    ]

    # Shape info
    if video.shape:
        n_frames, h, w, c = video.shape
        channels = "grayscale" if c == 1 else "RGB" if c == 3 else "RGBA"
        stats_parts = [
            f"[bold]{n_frames}[/] frames",
            f"{w}×{h}",
            channels,
        ]
        header_lines.append("")
        header_lines.append(" | ".join(stats_parts))
    else:
        header_lines.append("")
        header_lines.append("[yellow]Shape unknown (backend not loaded)[/]")

    console.print(
        Panel(
            "\n".join(header_lines),
            title="[bold]sleap-io[/]",
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Additional details below panel
    console.print()
    full_path = path.resolve()
    console.print(f"  [dim]Full[/]      {full_path}")
    console.print(f"  [dim]Size[/]      {_format_file_size(file_size)}")

    # Show encoding info if ffmpeg is available
    enc_info = _get_video_encoding_info(path)
    if enc_info:
        # Build encoding line: codec (profile), pixel format
        enc_parts = []
        if enc_info.codec:
            codec_str = enc_info.codec
            if enc_info.codec_profile:
                codec_str += f" ({enc_info.codec_profile})"
            enc_parts.append(codec_str)
        if enc_info.pixel_format:
            enc_parts.append(enc_info.pixel_format)
        if enc_parts:
            console.print(f"  [dim]Codec[/]     {', '.join(enc_parts)}")

        # FPS - from ffmpeg or video backend
        fps = enc_info.fps or (video.fps if video.fps else None)
        if fps:
            console.print(f"  [dim]FPS[/]       {fps:.2f}")

        # Bitrate
        if enc_info.bitrate_kbps:
            console.print(f"  [dim]Bitrate[/]   {enc_info.bitrate_kbps} kb/s")

        # GOP size (keyframe interval) - estimate if not already known
        gop = enc_info.gop_size
        if gop is None:
            gop = _estimate_gop_size(path)
        if gop:
            # Show GOP with fps context for interpretability
            if fps and fps > 0:
                gop_secs = gop / fps
                console.print(f"  [dim]GOP[/]       {gop} frames ({gop_secs:.1f}s)")
            else:
                console.print(f"  [dim]GOP[/]       {gop} frames")

    console.print()


def _print_skeleton_summary(labels: Labels) -> None:
    """Print skeleton summary as copyable Python code."""
    if not labels.skeletons:
        return

    console.print("[bold]Skeletons[/]")
    for i, sk in enumerate(labels.skeletons):
        prefix = f"[dim]# skeleton[{i}][/] " if len(labels.skeletons) > 1 else ""
        name = sk.name if sk.name else "unnamed"

        # Stats line
        n_nodes = len(sk.nodes)
        n_edges = len(sk.edges)
        n_sym = len(sk.symmetries) if sk.symmetries else 0

        stats_parts = [f"{n_nodes} nodes", f"{n_edges} edges"]
        if n_sym > 0:
            stats_parts.append(f"{n_sym} symmetries")

        console.print(f"  {prefix}[cyan]{name}[/] ({', '.join(stats_parts)})")
        console.print()

        # Python code
        code_lines = _get_skeleton_python_code(sk)
        for line in code_lines:
            console.print(f"  [green]{line}[/]")


def _print_skeleton_details(labels: Labels) -> None:
    """Print detailed skeleton information."""
    if not labels.skeletons:
        console.print("[dim]No skeletons[/]")
        return

    for i, sk in enumerate(labels.skeletons):
        console.print()
        name = sk.name if sk.name else "[dim]unnamed[/]"

        # Stats
        n_nodes = len(sk.nodes)
        n_edges = len(sk.edges)
        n_sym = len(sk.symmetries) if sk.symmetries else 0
        stats_parts = [f"{n_nodes} nodes", f"{n_edges} edges"]
        if n_sym > 0:
            stats_parts.append(f"{n_sym} symmetries")

        console.print(f"[bold cyan]Skeleton {i}: {name}[/] ({', '.join(stats_parts)})")
        console.print()

        # Python code (copyable)
        console.print("[dim]Python code:[/]")
        code_lines = _get_skeleton_python_code(sk)
        for line in code_lines:
            console.print(f"  [green]{line}[/]")

        # Nodes table
        console.print()
        console.print("[dim]Nodes:[/]")
        table = Table(
            box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1)
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Name", style="cyan")

        for idx, node in enumerate(sk.nodes):
            table.add_row(str(idx), node.name)

        console.print(table)

        # Edges table
        if sk.edges:
            console.print()
            console.print("[dim]Edges:[/]")
            edge_table = Table(
                box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1)
            )
            edge_table.add_column("#", justify="right", style="dim", width=3)
            edge_table.add_column("Source", style="cyan")
            edge_table.add_column("", style="dim", width=2)
            edge_table.add_column("Destination", style="cyan")
            edge_table.add_column("Indices", style="dim")

            for idx, e in enumerate(sk.edges):
                src_idx = sk.node_names.index(e.source.name)
                dst_idx = sk.node_names.index(e.destination.name)
                edge_table.add_row(
                    str(idx),
                    e.source.name,
                    "->",
                    e.destination.name,
                    f"({src_idx}, {dst_idx})",
                )

            console.print(edge_table)

        # Symmetries table
        if sk.symmetries:
            console.print()
            console.print("[dim]Symmetries:[/]")
            sym_table = Table(
                box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1)
            )
            sym_table.add_column("#", justify="right", style="dim", width=3)
            sym_table.add_column("Node A", style="cyan")
            sym_table.add_column("", style="dim", width=3)
            sym_table.add_column("Node B", style="cyan")

            for idx, s in enumerate(sk.symmetries):
                # s.nodes is a set, convert to list
                nodes_list = list(s.nodes)
                sym_table.add_row(
                    str(idx),
                    nodes_list[0].name,
                    "<->",
                    nodes_list[1].name,
                )

            console.print(sym_table)


def _get_video_type(vid: Video) -> str:
    """Get video type from backend or backend_metadata.

    Uses defensive fallback: live backend -> metadata -> filename inference.

    Args:
        vid: Video object to inspect.

    Returns:
        Video type string (e.g., "MediaVideo", "HDF5Video", "ImageVideo").
    """
    # Priority 1: Live backend type
    if vid.backend is not None:
        return type(vid.backend).__name__

    # Priority 2: backend_metadata["type"] from SLP file
    if "type" in vid.backend_metadata:
        return vid.backend_metadata["type"]

    # Priority 3: Infer from filename
    if isinstance(vid.filename, list):
        return "ImageVideo"

    filename = vid.filename.lower()
    if filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".mj2")):
        return "MediaVideo"
    elif filename.endswith((".h5", ".hdf5", ".slp")):
        return "HDF5Video"
    elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return "ImageVideo"
    elif filename.endswith((".tif", ".tiff")):
        return "TiffVideo"

    return "Unknown"


def _is_embedded(vid: Video) -> bool:
    """Check if video has embedded images.

    Uses defensive fallback: live backend -> metadata -> False.

    Args:
        vid: Video object to inspect.

    Returns:
        True if video has embedded images.
    """
    # Priority 1: Live backend attribute
    if isinstance(vid.backend, HDF5Video):
        return vid.backend.has_embedded_images

    # Priority 2: backend_metadata from SLP file
    return vid.backend_metadata.get("has_embedded_images", False)


def _is_pkg_slp(path: Path) -> bool:
    """Check if a file path is a .pkg.slp file.

    Args:
        path: File path to check.

    Returns:
        True if the path ends with .pkg.slp (case-insensitive).
    """
    return path.name.lower().endswith(".pkg.slp")


def _should_preserve_embedded(
    input_paths: list[Path] | Path, output_path: Path, embed: str | None
) -> bool:
    """Check if embedded videos should be preserved by default.

    When all input files are .pkg.slp files and the output is also .pkg.slp,
    we should preserve the existing embedded videos unless the user explicitly
    specified an --embed option.

    Args:
        input_paths: Input file path(s). Can be a single Path or list of Paths.
        output_path: Output file path.
        embed: The user-specified embed option (None if not specified).

    Returns:
        True if embedded videos should be preserved.
    """
    # User explicitly specified embed option - respect their choice
    if embed is not None:
        return False

    # Output must be pkg.slp
    if not _is_pkg_slp(output_path):
        return False

    # Handle single path or list
    if isinstance(input_paths, Path):
        input_paths = [input_paths]

    # All inputs must be pkg.slp
    return all(_is_pkg_slp(p) for p in input_paths)


def _get_dataset(vid: Video) -> str | None:
    """Get HDF5 dataset path.

    Args:
        vid: Video object to inspect.

    Returns:
        Dataset path string or None.
    """
    # Priority 1: Live backend attribute
    if isinstance(vid.backend, HDF5Video):
        return vid.backend.dataset

    # Priority 2: backend_metadata from SLP file
    return vid.backend_metadata.get("dataset")


def _get_image_filenames(vid: Video) -> list[str] | None:
    """Get image sequence filenames.

    Args:
        vid: Video object to inspect.

    Returns:
        List of image paths or None.
    """
    # Priority 1: vid.filename if it's a list
    if isinstance(vid.filename, list):
        return vid.filename

    # Priority 2: backend_metadata from SLP file
    return vid.backend_metadata.get("filenames")


def _get_plugin(vid: Video) -> str | None:
    """Get video/image plugin. Only available when backend is loaded.

    Args:
        vid: Video object to inspect.

    Returns:
        Plugin name string or None.
    """
    if vid.backend is not None and hasattr(vid.backend, "plugin"):
        return vid.backend.plugin
    return None


def _get_shape_source(vid: Video) -> str:
    """Determine if shape comes from live backend or metadata.

    Args:
        vid: Video object to inspect.

    Returns:
        "live" if from backend, "metadata" if cached, "unknown" otherwise.
    """
    if vid.backend is not None:
        try:
            _ = vid.backend.shape
            return "live"
        except Exception:
            pass

    if "shape" in vid.backend_metadata:
        return "metadata"

    return "unknown"


def _format_video_filename(vid: Video) -> str:
    """Format video filename for display.

    Args:
        vid: Video object to inspect.

    Returns:
        Formatted filename string.
    """
    filenames = _get_image_filenames(vid)
    if filenames is not None:
        return f"{len(filenames)} images"
    return Path(vid.filename).name


def _truncate_path_left(path_str: str, max_width: int) -> str:
    """Truncate a path from the left to fit within max_width.

    Preserves the basename and as much of the path as possible,
    truncating from the left (root) side.

    Args:
        path_str: Full path string.
        max_width: Maximum width in characters.

    Returns:
        Truncated path with "..." prefix if needed.
    """
    if len(path_str) <= max_width:
        return path_str

    # Reserve space for ellipsis
    available = max_width - 3  # "..."
    if available <= 0:
        return path_str[:max_width]

    # Take from the right (end) of the path
    return "..." + path_str[-available:]


def _build_status_line(vid: Video) -> str:
    """Build a status line describing video accessibility.

    Args:
        vid: Video object to inspect.

    Returns:
        Status description string.
    """
    is_embedded = _is_embedded(vid)
    plugin = _get_plugin(vid)

    # Embedded videos are always accessible
    if is_embedded:
        if vid.backend is not None:
            return "Embedded, backend loaded"
        return "Embedded"

    # Check file existence
    file_exists = vid.exists()

    # Build status based on file existence and backend state
    if vid.backend is not None:
        status = "Backend loaded"
        if plugin:
            status += f" ({plugin})"
        return status
    elif file_exists:
        return "File exists, backend not loaded"
    else:
        return "File not found"


def _print_video_summary(labels: Labels) -> None:
    """Print video summary with clean table-like layout."""
    if not labels.videos:
        return

    console.print()
    n_videos = len(labels.videos)
    console.print(f"[bold]Videos[/] ({n_videos})")
    console.print()

    # Count videos with and without metadata
    videos_with_metadata = []
    videos_without_metadata = []

    for i, vid in enumerate(labels.videos):
        if vid.shape:
            videos_with_metadata.append((i, vid))
        else:
            videos_without_metadata.append((i, vid))

    # Print videos with metadata
    for i, vid in videos_with_metadata:
        # Get video info using defensive helpers
        is_embedded = _is_embedded(vid)

        # Shape info
        n_frames, h, w, c = vid.shape
        shape_str = f"{w}×{h}"
        frames_str = f"{n_frames} frames"

        # Build status tag (backslash escapes Rich markup brackets)
        tag = ""
        if is_embedded:
            tag = " [cyan]\\[embedded][/]"
        elif not vid.exists() and not isinstance(vid.filename, list):
            tag = " [yellow]\\[not found][/]"

        # Format path with smart left-truncation (80 chars max for path)
        filenames = _get_image_filenames(vid)
        if filenames is not None:
            path_display = f"{len(filenames)} images"
        else:
            path_display = _truncate_path_left(str(vid.filename), 80)

        idx_str = f"[dim][{i}][/]"
        console.print(
            f"  {idx_str} [cyan]{path_display}[/]  {shape_str}  {frames_str}{tag}"
        )

    # Summarize videos without metadata (don't spam)
    if videos_without_metadata:
        n_no_meta = len(videos_without_metadata)
        n_to_show = min(3, n_no_meta)

        # Show first 3 videos without metadata
        for i, vid in videos_without_metadata[:n_to_show]:
            filenames = _get_image_filenames(vid)
            if filenames is not None:
                path_display = f"{len(filenames)} images"
            else:
                path_display = _truncate_path_left(str(vid.filename), 80)

            tag = ""
            if not vid.exists() and not isinstance(vid.filename, list):
                tag = " [yellow]\\[not found][/]"

            idx_str = f"[dim][{i}][/]"
            console.print(
                f"  {idx_str} [cyan]{path_display}[/]  "
                f"[dim]?×?[/]  [dim]? frames[/]{tag}"
            )

        # Show truncation message if there are more
        if n_no_meta > 3:
            n_remaining = n_no_meta - 3
            console.print(f"  [dim]... +{n_remaining} more without cached metadata[/]")


def _print_video_details(labels: Labels, video_index: int | None = None) -> None:
    """Print detailed video information with consistent field ordering.

    Args:
        labels: Labels object containing videos.
        video_index: If None or -1, show all videos. Otherwise show specific video.
    """
    if not labels.videos:
        console.print("[dim]No videos[/]")
        return

    # Determine which videos to show
    if video_index is None or video_index == -1:
        videos_to_show = list(enumerate(labels.videos))
    else:
        if video_index < 0 or video_index >= len(labels.videos):
            n_vids = len(labels.videos)
            raise click.ClickException(
                f"Video index {video_index} out of range. "
                f"File has {n_vids} video(s) (indices 0-{n_vids - 1})."
            )
        videos_to_show = [(video_index, labels.videos[video_index])]

    # Pre-compute frame counts using fast path (works for both eager and lazy)
    frame_counts = labels.n_frames_per_video()

    for i, vid in videos_to_show:
        console.print()

        # Get video info using defensive helpers
        video_type = _get_video_type(vid)
        is_embedded = _is_embedded(vid)
        shape_source = _get_shape_source(vid)
        status = _build_status_line(vid)
        dataset = _get_dataset(vid)
        filenames = _get_image_filenames(vid)

        # Header with filename and embedded indicator
        fname = _format_video_filename(vid)
        header = f"[bold cyan]Video {i}:[/] {fname}"
        if is_embedded:
            header += " [cyan]\\[embedded][/]"
        console.print(header)
        console.print()

        # Type line - with embedded qualifier for HDF5
        type_str = video_type
        if is_embedded and video_type == "HDF5Video":
            type_str = "HDF5Video (embedded)"
        console.print(f"  [dim]Type[/]      {type_str}")

        # Path/Source info based on video type
        if filenames is not None:
            # Image sequence - show first/last
            console.print(f"  [dim]First[/]     {filenames[0]}")
            if len(filenames) > 1:
                console.print(f"  [dim]Last[/]      {filenames[-1]}")
            # Show full path of first image
            full_path = Path(filenames[0]).resolve()
            if str(full_path) != filenames[0]:
                console.print(f"  [dim]Full[/]      {full_path.parent}/")
        elif is_embedded:
            # Embedded video - show source if available
            source_filename = None
            if isinstance(vid.backend, HDF5Video) and vid.backend.source_filename:
                source_filename = vid.backend.source_filename
            if source_filename:
                console.print(f"  [dim]Source[/]    {source_filename}")
            if dataset:
                console.print(f"  [dim]Dataset[/]   {dataset}")
            # Show format info if backend is loaded
            if isinstance(vid.backend, HDF5Video):
                fmt = vid.backend.image_format.upper()
                order = vid.backend.channel_order
                console.print(f"  [dim]Format[/]    {fmt} ({order})")
        else:
            # Regular file - show path
            console.print(f"  [dim]Path[/]      {vid.filename}")
            # Show full path if different from stored path
            if isinstance(vid.filename, str):
                full_path = Path(vid.filename).resolve()
                if str(full_path) != vid.filename:
                    console.print(f"  [dim]Full[/]      {full_path}")

        # Status line
        console.print(f"  [dim]Status[/]    {status}")
        console.print()

        # Dimension info - indicate source if from metadata
        if vid.shape:
            n_frames, h, w, c = vid.shape
            channels = "grayscale" if c == 1 else "RGB" if c == 3 else "RGBA"
            meta_tag = " [dim][from metadata][/]" if shape_source == "metadata" else ""

            # Show frame indices for embedded videos
            if is_embedded and isinstance(vid.backend, HDF5Video):
                inds = vid.backend.source_inds
                if inds is not None and len(inds):
                    if len(inds) <= 5:
                        inds_str = ", ".join(str(x) for x in inds)
                    else:
                        inds_str = f"{inds[0]}–{inds[-1]}"
                    frames_str = f"{n_frames} (indices: {inds_str})"
                    console.print(f"  [dim]Frames[/]    {frames_str}")
                else:
                    console.print(f"  [dim]Frames[/]    {n_frames}")
            else:
                console.print(f"  [dim]Frames[/]    {n_frames}{meta_tag}")

            console.print(f"  [dim]Size[/]      {w} × {h} ({channels}){meta_tag}")
        else:
            console.print("  [dim]Frames[/]    [yellow]unknown[/]")
            console.print("  [dim]Size[/]      [yellow]unknown[/]")

        # Labeled frames in this video - use pre-computed count
        n_frames_labeled = frame_counts.get(vid, 0)
        console.print(f"  [dim]Labeled[/]   {n_frames_labeled} frames")

        # Show source video metadata for embedded videos
        if is_embedded and vid.source_video is not None:
            src = vid.source_video
            src_type = _get_video_type(src)
            console.print()
            console.print("  [dim]Source Video[/]")

            # Source video type and filename
            src_fname = Path(src.filename).name if isinstance(src.filename, str) else ""
            if src_fname:
                console.print(f"  [dim]  File[/]    {src_fname}")
            console.print(f"  [dim]  Type[/]    {src_type}")

            # Source video dimensions (if available in metadata)
            src_shape = src.backend_metadata.get("shape")
            if src_shape and len(src_shape) == 4:
                src_frames, src_h, src_w, src_c = src_shape
                src_ch = "grayscale" if src_c == 1 else "RGB" if src_c == 3 else "RGBA"
                console.print(f"  [dim]  Frames[/]  {src_frames}")
                console.print(f"  [dim]  Size[/]    {src_w} x {src_h} ({src_ch})")


def _print_tracks_summary(labels: Labels) -> None:
    """Print track summary (inline)."""
    if not labels.tracks:
        return

    console.print()
    console.print(f"[bold]Tracks[/] ({len(labels.tracks)})")
    track_names = ", ".join(t.name for t in labels.tracks[:5])
    if len(labels.tracks) > 5:
        track_names += f" ... (+{len(labels.tracks) - 5} more)"
    console.print(f"  {track_names}")


def _print_tracks_details(labels: Labels) -> None:
    """Print detailed track information."""
    if not labels.tracks:
        console.print("[dim]No tracks[/]")
        return

    # Use fast path for instance counts (works for both eager and lazy)
    track_counts = labels.n_instances_per_track()

    console.print()
    table = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Track", style="cyan")
    table.add_column("Instances", justify="right")

    for idx, track in enumerate(labels.tracks):
        table.add_row(
            str(idx), track.name or "[dim]unnamed[/]", str(track_counts.get(track, 0))
        )

    console.print(table)


def _print_labeled_frame(labels: Labels, frame_idx: int) -> None:
    """Print details for a specific labeled frame."""
    if frame_idx < 0 or frame_idx >= len(labels.labeled_frames):
        raise click.ClickException(
            f"--lf out of range (0..{len(labels.labeled_frames) - 1})"
        )

    lf = labels.labeled_frames[frame_idx]

    console.print()
    console.print(f"[bold cyan]Labeled Frame {frame_idx}[/]")
    console.print()

    # Video info
    if isinstance(lf.video.filename, str):
        video_name = Path(lf.video.filename).name
    else:
        video_name = f"{len(lf.video.filename)} images"
    console.print(f"  [dim]Video:[/]     {video_name}")
    console.print(f"  [dim]Frame:[/]     {lf.frame_idx}")
    console.print(f"  [dim]Instances:[/] {len(lf)}")

    # Instances as blocks
    for idx, inst in enumerate(lf.instances):
        console.print()

        inst_type = "predicted" if isinstance(inst, PredictedInstance) else "user"
        type_style = "yellow" if inst_type == "predicted" else "green"

        # Header line with metadata
        parts = [f"[{type_style}]{inst_type}[/]"]

        if inst.track:
            parts.append(f"track=[cyan]{inst.track.name}[/]")

        if isinstance(inst, PredictedInstance):
            parts.append(f"score={inst.score:.2f}")

        n_visible = inst.n_visible
        n_total = len(inst)
        parts.append(f"visible={n_visible}/{n_total}")

        console.print(f"  [bold]Instance {idx}:[/] {', '.join(parts)}")

        # Points as copyable Python list
        pts = inst.numpy()
        point_tuples = []
        for pt in pts:
            if not any(map(lambda x: x != x, pt)):  # Check for NaN (visible)
                point_tuples.append(f"({pt[0]:.2f}, {pt[1]:.2f})")
            else:
                point_tuples.append("(None, None)")  # NaN = not visible

        points_str = f"[{', '.join(point_tuples)}]"
        console.print(f"    [green]points = {points_str}[/]")


def _print_provenance(labels: Labels, compact: bool = False) -> None:
    """Print provenance information.

    Args:
        labels: Labels object to print provenance from.
        compact: If True, use compact formatting (for default view).
            If False, use expanded pretty printing (for --provenance flag).
    """
    import json

    from rich.syntax import Syntax

    if not labels.provenance:
        console.print("[dim]No provenance information[/]")
        return

    console.print()
    console.print("[bold cyan]Provenance[/]")
    console.print()

    for key, value in labels.provenance.items():
        if isinstance(value, (dict, list)):
            if compact:
                # Compact mode: show summary
                if isinstance(value, list):
                    value_str = f"[{len(value)} items]"
                else:
                    value_str = f"{{{len(value)} keys}}"
                console.print(f"  [dim]{key}:[/] {value_str}")
            else:
                # Full mode: pretty print JSON
                console.print(f"  [dim]{key}:[/]")
                try:
                    json_str = json.dumps(value, indent=2, default=str)
                    syntax = Syntax(
                        json_str,
                        "json",
                        theme="ansi_dark",
                        word_wrap=True,
                        background_color="default",
                    )
                    console.print(syntax)
                except (TypeError, ValueError):
                    # Fallback for non-JSON-serializable values
                    console.print(f"    {value}")
        else:
            # Simple scalar values - show in full
            console.print(f"  [dim]{key}:[/] {value}")


@cli.command()
@click.argument(
    "path_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "path_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input file (.slp, .nwb, video). Can also be passed as positional argument.",
)
@click.option(
    "lazy",
    "--lazy/--no-lazy",
    default=True,
    help="Use lazy loading for faster startup (SLP files only). Default: lazy.",
)
@click.option(
    "open_videos",
    "--open-videos/--no-open-videos",
    default=None,
    help="Open video backends. Default: only when -v/--all is used.",
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
    "-s",
    is_flag=True,
    help="Print detailed skeleton info (nodes, edges, symmetries).",
)
@click.option(
    "video",
    "--video",
    "-v",
    is_flag=True,
    help="Print detailed video info (opens backends by default).",
)
@click.option(
    "video_index",
    "--video-index",
    "--vi",
    type=int,
    default=None,
    help="Show only video at this index (0-based). Implies --video.",
)
@click.option(
    "tracks",
    "--tracks",
    "-t",
    is_flag=True,
    help="Print detailed track info with instance counts.",
)
@click.option(
    "provenance",
    "--provenance",
    "-p",
    is_flag=True,
    help="Print provenance/metadata info.",
)
@click.option(
    "show_all",
    "--all",
    "-a",
    is_flag=True,
    help="Print all available details.",
)
def show(
    path_arg: Path | None,
    path_opt: Path | None,
    lazy: bool,
    open_videos: bool | None,
    lf_index: int | None,
    skeleton: bool,
    video: bool,
    video_index: int | None,
    tracks: bool,
    provenance: bool,
    show_all: bool,
):
    """Print labels file summary with rich formatting.

    Shows a header panel with file info and key statistics, followed by
    skeleton definitions as copyable Python code.

    Use the detail flags to show additional information.

    [dim]Examples:[/]

        $ sio show labels.slp
        $ sio show -i labels.slp
        $ sio show labels.slp --skeleton
        $ sio show labels.slp --lf 0
        $ sio show labels.slp --all
    """
    # Resolve input from positional arg or -i option
    path = _resolve_input(path_arg, path_opt, "input file")

    # --video-index implies --video
    if video_index is not None:
        video = True

    # Expand --all flag
    if show_all:
        skeleton = True
        video = True
        tracks = True
        provenance = True

    # Determine whether to open videos:
    # - If explicitly set via --open-videos or --no-open-videos, use that
    # - Otherwise, open videos only when -v or --all is used
    if open_videos is None:
        open_videos = video or show_all

    # Use lazy loading only for SLP files (other formats don't support it)
    use_lazy = lazy and path.suffix.lower() == ".slp"
    try:
        obj = io_main.load_file(str(path), open_videos=open_videos, lazy=use_lazy)
    except (IndexError, KeyError):
        # Fall back to eager loading if lazy loading fails
        # (e.g., for files with sessions that require labeled_frames)
        if use_lazy:
            obj = io_main.load_file(str(path), open_videos=open_videos, lazy=False)
        else:
            raise

    if isinstance(obj, Labels):
        # Always show header
        _print_header(path, obj)
        console.print()

        # Determine if we're showing detailed views
        show_details = skeleton or video or tracks or lf_index is not None

        if not show_details:
            # Default: show compact summaries
            _print_skeleton_summary(obj)
            _print_video_summary(obj)
            _print_tracks_summary(obj)
        else:
            # Detailed views
            if skeleton:
                console.print("[bold]Skeleton Details[/]")
                _print_skeleton_details(obj)

            if video:
                console.print("[bold]Video Details[/]")
                _print_video_details(obj, video_index=video_index)

            if tracks:
                console.print("[bold]Tracks[/]")
                _print_tracks_details(obj)

            if lf_index is not None:
                if len(obj.labeled_frames) == 0:
                    raise click.ClickException("No labeled frames present in file.")
                _print_labeled_frame(obj, lf_index)

        # Always show provenance if present
        # Use compact mode by default, full mode when -p/--provenance is explicitly used
        if obj.provenance:
            _print_provenance(obj, compact=not provenance)

        console.print()
    elif isinstance(obj, Video):
        # Standalone video file
        _print_video_standalone(path, obj)
    else:
        # For other objects, print repr
        click.echo(repr(obj))


def _infer_input_format(path: Path) -> str | None:
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


def _infer_output_format(path: Path) -> str | None:
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
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input file path. Can also be passed as positional argument.",
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
@click.option(
    "--csv-format",
    "csv_format",
    type=click.Choice(["sleap", "dlc", "points", "instances", "frames"]),
    default="sleap",
    help="CSV output format. Default: sleap.",
)
@click.option(
    "--scorer",
    default="sleap-io",
    help="Scorer name for DLC CSV output.",
)
@click.option(
    "--save-metadata/--no-metadata",
    "save_metadata",
    default=False,
    help="Save JSON metadata file for round-trip support (CSV only).",
)
def convert(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path,
    input_format: str | None,
    output_format: str | None,
    embed: str | None,
    csv_format: str,
    scorer: str,
    save_metadata: bool,
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

        $ sio convert labels.slp -o labels.nwb
        $ sio convert -i labels.slp -o labels.nwb
        $ sio convert annotations.json -o labels.slp --from coco
        $ sio convert labels.slp -o labels.pkg.slp --embed user
        $ sio convert labels.slp -o dataset/ --to ultralytics
    """
    # Resolve input from positional arg or -i option
    input_path = _resolve_input(input_arg, input_opt, "input file")

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
        save_kwargs: dict = {"format": resolved_output_format}
        # Check for pkg.slp to pkg.slp preservation
        if resolved_output_format == "slp" and _should_preserve_embedded(
            input_path, output_path, embed
        ):
            # Preserve existing embedded videos from pkg.slp input
            save_kwargs["embed"] = None
        elif embed is not None:
            save_kwargs["embed"] = embed
        # Handle CSV-specific options
        if resolved_output_format == "csv":
            io_main.save_csv(
                labels,
                str(output_path),
                format=csv_format,
                scorer=scorer,
                save_metadata=save_metadata,
            )
        else:
            io_main.save_file(labels, str(output_path), **save_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Success message
    click.echo(f"Converted: {input_path} -> {output_path}")
    click.echo(f"Format: {resolved_input_format} -> {resolved_output_format}")
    if embed:
        click.echo(f"Embedded frames: {embed}")
    if resolved_output_format == "csv":
        click.echo(f"CSV format: {csv_format}")
        if save_metadata:
            click.echo("Metadata saved: yes")


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input labels file. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for split files.",
)
@click.option(
    "--train",
    "train_fraction",
    type=float,
    default=0.8,
    show_default=True,
    help="Training set fraction (0.0-1.0).",
)
@click.option(
    "--val",
    "val_fraction",
    type=float,
    default=None,
    help="Validation set fraction. Defaults to remainder after train and test.",
)
@click.option(
    "--test",
    "test_fraction",
    type=float,
    default=None,
    help="Test set fraction. If not specified, no test split is created.",
)
@click.option(
    "--remove-predictions",
    is_flag=True,
    default=False,
    help="Remove predicted instances before splitting (keep only user labels).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible splits.",
)
@click.option(
    "--embed",
    type=click.Choice(["user", "all", "suggestions", "source"]),
    help="Embed frames in output files.",
)
def split(
    input_arg: Path | None,
    input_opt: Path | None,
    output_dir: Path,
    train_fraction: float,
    val_fraction: float | None,
    test_fraction: float | None,
    remove_predictions: bool,
    seed: int | None,
    embed: str | None,
):
    """Split labels into train/val/test sets.

    Creates random train/validation/test splits from a labels file,
    saving each split to the output directory.

    [bold]Proportions:[/]
    By default, 80% of frames go to training and 20% to validation.
    Use --test to create a three-way split.

    [dim]Examples:[/]

        $ sio split labels.slp -o splits/
        $ sio split -i labels.slp -o splits/
        $ sio split labels.slp -o splits/ --train 0.7 --val 0.15 --test 0.15
        $ sio split labels.slp -o splits/ --remove-predictions --seed 42
        $ sio split labels.slp -o splits/ --embed user
    """
    # Resolve input from positional arg or -i option
    input_path = _resolve_input(input_arg, input_opt, "input labels file")

    from copy import deepcopy

    import numpy as np

    # Validate fractions
    if not 0 < train_fraction < 1:
        raise click.ClickException("--train must be between 0 and 1 (exclusive).")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise click.ClickException("--val must be between 0 and 1 (exclusive).")

    if test_fraction is not None and not 0 < test_fraction < 1:
        raise click.ClickException("--test must be between 0 and 1 (exclusive).")

    # Check that fractions don't exceed 1.0
    total = train_fraction
    if val_fraction is not None:
        total += val_fraction
    if test_fraction is not None:
        total += test_fraction

    if total > 1.0:
        raise click.ClickException(
            f"Sum of fractions ({total:.2f}) exceeds 1.0. "
            "Reduce --train, --val, or --test."
        )

    # Load the input file (don't need video access unless embedding)
    needs_video = embed is not None
    try:
        suffix = input_path.suffix.lower()
        load_kwargs: dict = {}
        if suffix == ".slp":
            load_kwargs["open_videos"] = needs_video
        labels = io_main.load_file(str(input_path), **load_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to load input file: {e}")

    if not isinstance(labels, Labels):
        raise click.ClickException(
            f"Input file is not a labels file (got {type(labels).__name__})."
        )

    # Optionally remove predictions
    if remove_predictions:
        labels = deepcopy(labels)
        labels.remove_predictions()
        labels.suggestions = []
        labels.clean()

    # Check we have enough frames
    n_frames = len(labels)
    if n_frames == 0:
        raise click.ClickException("No labeled frames found in input file.")

    if n_frames == 1:
        click.echo(
            "Warning: Only 1 labeled frame found. "
            "All splits will contain the same frame."
        )

    # Compute split sizes
    n_train = max(int(n_frames * train_fraction), 1)

    if test_fraction is not None:
        n_test = max(int(n_frames * test_fraction), 1)
    else:
        n_test = 0

    if val_fraction is not None:
        n_val = max(int(n_frames * val_fraction), 1)
    else:
        # Validation gets remainder
        n_val = max(n_frames - n_train - n_test, 1)

    # Ensure we don't exceed total frames
    if n_train + n_val + n_test > n_frames and n_frames > 1:
        # Adjust val to fit
        n_val = max(n_frames - n_train - n_test, 1)

    # Random sampling
    rng = np.random.default_rng(seed=seed)
    all_inds = np.arange(n_frames)
    rng.shuffle(all_inds)

    train_inds = all_inds[:n_train]
    remaining_inds = all_inds[n_train:]

    if n_test > 0:
        test_inds = remaining_inds[:n_test]
        val_inds = remaining_inds[n_test : n_test + n_val]
    else:
        test_inds = np.array([], dtype=int)
        val_inds = remaining_inds[:n_val]

    # Handle edge case: single frame goes to all splits
    if n_frames == 1:
        train_inds = np.array([0])
        val_inds = np.array([0])
        if n_test > 0:
            test_inds = np.array([0])

    # Extract splits
    labels_train = labels.extract(train_inds, copy=True)
    labels_val = labels.extract(val_inds, copy=True)
    if n_test > 0:
        labels_test = labels.extract(test_inds, copy=True)

    # Update provenance
    source_labels = labels.provenance.get("filename", str(input_path))
    labels_train.provenance["source_labels"] = source_labels
    labels_train.provenance["split"] = "train"
    labels_val.provenance["source_labels"] = source_labels
    labels_val.provenance["split"] = "val"
    if n_test > 0:
        labels_test.provenance["source_labels"] = source_labels
        labels_test.provenance["split"] = "test"

    # Add seed to provenance if specified
    if seed is not None:
        labels_train.provenance["split_seed"] = seed
        labels_val.provenance["split_seed"] = seed
        if n_test > 0:
            labels_test.provenance["split_seed"] = seed

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension
    if embed is not None:
        ext = ".pkg.slp"
    else:
        ext = ".slp"

    # Save splits
    try:
        save_kwargs: dict = {}
        if embed is not None:
            save_kwargs["embed"] = embed

        train_path = output_dir / f"train{ext}"
        io_main.save_file(labels_train, str(train_path), **save_kwargs)

        val_path = output_dir / f"val{ext}"
        io_main.save_file(labels_val, str(val_path), **save_kwargs)

        if n_test > 0:
            test_path = output_dir / f"test{ext}"
            io_main.save_file(labels_test, str(test_path), **save_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to save split files: {e}")

    # Success message
    click.echo(f"Split {n_frames} frames from: {input_path}")
    click.echo(f"Output directory: {output_dir}")
    click.echo("")
    click.echo(f"  train{ext}: {len(labels_train)} frames")
    click.echo(f"  val{ext}: {len(labels_val)} frames")
    if n_test > 0:
        click.echo(f"  test{ext}: {len(labels_test)} frames")
    if seed is not None:
        click.echo(f"\nRandom seed: {seed}")
    if remove_predictions:
        click.echo("Predictions removed: yes")
    if embed:
        click.echo(f"Embedded frames: {embed}")


@cli.command()
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output labels file.",
)
@click.option(
    "--embed",
    type=click.Choice(["user", "all", "suggestions", "source"]),
    help="Embed frames in output file.",
)
def unsplit(
    input_files: tuple[Path, ...],
    output_path: Path,
    embed: str | None,
):
    """Merge split files back into a single labels file.

    This is the inverse of the split command. Takes multiple split files
    (e.g., train.slp, val.slp, test.slp) and merges them back into one.

    [bold]Input:[/]
    Pass individual files or a directory containing .slp files.

    [bold]Video Deduplication:[/]
    Videos are automatically deduplicated if they have proper provenance
    metadata (from sio split --embed). For legacy split files without
    provenance, videos may not deduplicate - this is safe behavior.

    [dim]Examples:[/]

        $ sio unsplit train.slp val.slp -o merged.slp
        $ sio unsplit splits/ -o merged.slp
        $ sio unsplit train.pkg.slp val.pkg.slp test.pkg.slp -o merged.slp
    """
    # Expand directories to .slp files
    expanded_files: list[Path] = []
    for path in input_files:
        if path.is_dir():
            # Find all .slp files in directory (including .pkg.slp)
            slp_files = sorted(path.glob("*.slp"))
            if not slp_files:
                raise click.ClickException(f"No .slp files found in directory: {path}")
            expanded_files.extend(slp_files)
        else:
            expanded_files.append(path)

    # Require at least 2 input files
    if len(expanded_files) < 2:
        raise click.ClickException("At least 2 input files required.")

    # Load the first file
    first_file = expanded_files[0]
    click.echo(f"Loading: {first_file.name}")

    try:
        labels = io_main.load_file(str(first_file), open_videos=embed is not None)
    except Exception as e:
        raise click.ClickException(f"Failed to load {first_file}: {e}")

    if not isinstance(labels, Labels):
        raise click.ClickException(
            f"Input file is not a labels file (got {type(labels).__name__})."
        )

    click.echo(f"  {len(labels)} frames, {len(labels.videos)} videos")
    initial_video_count = len(labels.videos)

    # Merge remaining files
    for input_file in expanded_files[1:]:
        click.echo(f"Merging: {input_file.name}")

        try:
            other = io_main.load_file(str(input_file), open_videos=embed is not None)
        except Exception as e:
            raise click.ClickException(f"Failed to load {input_file}: {e}")

        if not isinstance(other, Labels):
            raise click.ClickException(
                f"Input file is not a labels file (got {type(other).__name__})."
            )

        frames_before = len(labels)

        # Merge with automatic video matching (uses original_video provenance)
        # and keep_both frame strategy (splits have no overlapping frames)
        labels.merge(other, video="auto", frame="keep_both")

        frames_added = len(labels) - frames_before
        click.echo(f"  +{frames_added} frames -> {len(labels)} total")

    # Check for video deduplication issues
    final_video_count = len(labels.videos)
    if final_video_count > initial_video_count:
        click.echo("")
        click.echo(
            f"Note: Video count increased from {initial_video_count} to "
            f"{final_video_count}. This may indicate legacy split files without "
            "provenance metadata. Videos were added as new (safe behavior)."
        )

    # Clean up split-specific provenance keys
    for key in ["split", "split_seed", "source_labels"]:
        labels.provenance.pop(key, None)

    # Save output
    click.echo("")
    click.echo(f"Saving: {output_path}")

    try:
        save_kwargs: dict = {}
        if _should_preserve_embedded(expanded_files, output_path, embed):
            # Preserve existing embedded videos from pkg.slp inputs
            save_kwargs["embed"] = None
        elif embed is not None:
            save_kwargs["embed"] = embed
        io_main.save_file(labels, str(output_path), **save_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    click.echo("")
    click.echo(f"Merged {len(expanded_files)} files:")
    click.echo(f"  {len(labels)} frames, {len(labels.videos)} videos")


@cli.command()
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output labels file.",
)
@click.option(
    "--skeleton",
    type=click.Choice(["structure", "subset", "overlap", "exact"]),
    default="structure",
    show_default=True,
    help="Skeleton matching method.",
)
@click.option(
    "--video",
    type=click.Choice(["auto", "path", "basename", "content", "shape", "image_dedup"]),
    default="auto",
    show_default=True,
    help="Video matching method.",
)
@click.option(
    "--track",
    type=click.Choice(["name", "identity"]),
    default="name",
    show_default=True,
    help="Track matching method.",
)
@click.option(
    "--frame",
    type=click.Choice(
        [
            "auto",
            "keep_original",
            "keep_new",
            "keep_both",
            "update_tracks",
            "replace_predictions",
        ]
    ),
    default="auto",
    show_default=True,
    help="Frame merge strategy for overlapping frames.",
)
@click.option(
    "--instance",
    type=click.Choice(["spatial", "identity", "iou"]),
    default="spatial",
    show_default=True,
    help="Instance matching method for overlapping frames.",
)
@click.option(
    "--embed",
    type=click.Choice(["user", "all", "suggestions", "source"]),
    help="Embed frames in output file.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed merge information.",
)
def merge(
    input_files: tuple[Path, ...],
    output_path: Path,
    skeleton: str,
    video: str,
    track: str,
    frame: str,
    instance: str,
    embed: str | None,
    verbose: bool,
):
    """Merge multiple labels files into one.

    A flexible merge command that combines annotations from multiple sources.
    Supports various matching strategies for skeletons, videos, tracks, and
    instance pairing.

    [bold]Common use cases:[/]

    - Merge predictions into a labeled project (default: frame=auto)
    - Combine annotation projects from different annotators
    - Update predictions with new model output (frame=replace_predictions)
    - Consolidate cross-platform/cross-machine projects

    [bold]Input:[/]
    Pass individual files or a directory containing .slp files.

    [bold]Matching strategies:[/]

    [dim]Skeleton:[/] How to match skeletons between files
      • structure: Same node names (any order) [default]
      • subset: Incoming nodes are subset of base
      • overlap: Sufficient overlap between node sets
      • exact: Nodes and edges must be identical

    [dim]Video:[/] How to identify same videos across files
      • auto: Safe cascade (path, basename, provenance) [default]
      • path: Exact path match only
      • basename: Match by filename (ignores directory)
      • content/shape/image_dedup: Special modes for image lists

    [dim]Frame:[/] How to handle overlapping frames
      • auto: Smart merge (preserve user labels, update predictions) [default]
      • keep_original: Ignore incoming for overlapping frames
      • keep_new: Replace with incoming for overlapping frames
      • keep_both: Keep all instances (may create duplicates)
      • replace_predictions: Replace predictions, keep user labels
      • update_tracks: Copy track assignments only

    [dim]Instance:[/] How to pair instances within frames
      • spatial: Match by centroid distance [default]
      • identity: Match by track identity
      • iou: Match by bounding box overlap

    [dim]Examples:[/]

        $ sio merge project.slp predictions.slp -o merged.slp
        $ sio merge base.slp new.slp -o merged.slp --frame replace_predictions
        $ sio merge file1.slp file2.slp file3.slp -o combined.slp --frame keep_both
        $ sio merge results/ -o combined.slp
        $ sio merge local.slp remote.slp -o merged.slp --video basename
    """
    # Expand directories to .slp files
    expanded_files: list[Path] = []
    for path in input_files:
        if path.is_dir():
            # Find all .slp files in directory (including .pkg.slp)
            slp_files = sorted(path.glob("*.slp"))
            if not slp_files:
                raise click.ClickException(f"No .slp files found in directory: {path}")
            expanded_files.extend(slp_files)
        else:
            expanded_files.append(path)

    # Require at least 2 input files
    if len(expanded_files) < 2:
        raise click.ClickException("At least 2 input files required.")

    # Load the first file
    first_file = expanded_files[0]
    click.echo(f"Loading: {first_file.name}")

    try:
        labels = io_main.load_file(str(first_file), open_videos=embed is not None)
    except Exception as e:
        raise click.ClickException(f"Failed to load {first_file}: {e}")

    if not isinstance(labels, Labels):
        raise click.ClickException(
            f"Input file is not a labels file (got {type(labels).__name__})."
        )

    initial_frames = len(labels)
    initial_videos = len(labels.videos)
    total_instances_added = 0
    total_conflicts = 0

    click.echo(f"  {initial_frames} frames, {initial_videos} videos")

    # Merge remaining files
    for input_file in expanded_files[1:]:
        click.echo(f"Merging: {input_file.name}")

        try:
            other = io_main.load_file(str(input_file), open_videos=embed is not None)
        except Exception as e:
            raise click.ClickException(f"Failed to load {input_file}: {e}")

        if not isinstance(other, Labels):
            raise click.ClickException(
                f"Input file is not a labels file (got {type(other).__name__})."
            )

        frames_before = len(labels)

        # Merge with user-specified strategies
        result = labels.merge(
            other,
            skeleton=skeleton,
            video=video,
            track=track,
            frame=frame,
            instance=instance,
        )

        frames_added = len(labels) - frames_before
        total_instances_added += result.instances_added
        total_conflicts += len(result.conflicts)

        if verbose:
            click.echo(f"  Strategy: {frame}")
            click.echo(f"  +{frames_added} frames, +{result.instances_added} instances")
            if result.conflicts:
                click.echo(f"  Conflicts: {len(result.conflicts)}")
        else:
            click.echo(f"  +{frames_added} frames -> {len(labels)} total")

    # Report video changes (merging can only add videos, never remove)
    final_videos = len(labels.videos)
    if verbose and final_videos > initial_videos:
        click.echo("")
        click.echo(
            f"Note: Video count increased from {initial_videos} to "
            f"{final_videos}. Videos from other files were added as new."
        )

    # Save output
    click.echo("")
    click.echo(f"Saving: {output_path}")

    try:
        save_kwargs: dict = {}
        if _should_preserve_embedded(expanded_files, output_path, embed):
            # Preserve existing embedded videos from pkg.slp inputs
            save_kwargs["embed"] = None
        elif embed is not None:
            save_kwargs["embed"] = embed
        io_main.save_file(labels, str(output_path), **save_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    click.echo("")
    click.echo(f"Merged {len(expanded_files)} files:")
    click.echo(f"  {len(labels)} frames, {final_videos} videos")
    if verbose:
        click.echo(f"  {total_instances_added} instances added")
        if total_conflicts:
            click.echo(f"  {total_conflicts} conflicts resolved")


@cli.command("filenames")
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input labels file. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output labels file (required for update mode).",
)
@click.option(
    "--filename",
    "new_filenames",
    multiple=True,
    help="New filename for each video (list mode, repeat for each video).",
)
@click.option(
    "--map",
    "filename_map",
    nargs=2,
    multiple=True,
    metavar="OLD NEW",
    help="Replace OLD filename with NEW (map mode).",
)
@click.option(
    "--prefix",
    "prefix_map",
    nargs=2,
    multiple=True,
    metavar="OLD NEW",
    help="Replace OLD path prefix with NEW (prefix mode).",
)
@click.option(
    "--source",
    "show_source",
    is_flag=True,
    default=False,
    help="Show source video filenames for embedded videos.",
)
@click.option(
    "--original",
    "show_original",
    is_flag=True,
    default=False,
    help="Show original video filenames (root of provenance chain).",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Show all details: full image lists, source videos, original videos.",
)
def filenames(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path | None,
    new_filenames: tuple[str, ...],
    filename_map: tuple[tuple[str, str], ...],
    prefix_map: tuple[tuple[str, str], ...],
    show_source: bool,
    show_original: bool,
    show_all: bool,
):
    r"""List or update video filenames in a labels file.

    By default, lists all video filenames for quick inspection.
    With -o and update flags, replaces video paths and saves to output.

    [bold]Inspection mode[/] (default):

        $ sio filenames labels.slp
        $ sio filenames -i labels.slp

    [bold]Update modes[/] (require -o):

    [bold]List mode[/] (--filename): Replace all video filenames in order.
    Provide one --filename for each video in the labels file.

    [bold]Map mode[/] (--map OLD NEW): Replace specific filenames.
    Use exact path matching to replace OLD with NEW.

    [bold]Prefix mode[/] (--prefix OLD NEW): Replace path prefixes.
    Cross-platform aware (handles Windows/Linux path differences).

    [dim]Examples:[/]

        $ sio filenames labels.slp
        $ sio filenames labels.slp -o out.slp --filename /new/video.mp4
        $ sio filenames labels.slp -o out.slp --prefix "C:\\data" /mnt/data
    """
    # Resolve input from positional arg or -i option
    input_path = _resolve_input(input_arg, input_opt, "input labels file")

    # Determine if any update flags are provided
    has_update_flags = (
        len(new_filenames) > 0 or len(filename_map) > 0 or len(prefix_map) > 0
    )

    # Load the input file (no video access needed)
    try:
        labels = io_main.load_file(str(input_path), open_videos=False)
    except Exception as e:
        raise click.ClickException(f"Failed to load input file: {e}")

    if not isinstance(labels, Labels):
        raise click.ClickException(
            f"Input file is not a labels file (got {type(labels).__name__})."
        )

    # Inspection mode: just list filenames
    if not has_update_flags:
        # Determine effective flags
        show_source_effective = show_source or show_all
        show_original_effective = show_original or show_all
        show_all_images = show_all

        click.echo(f"Video filenames in {input_path.name}:")
        for i, vid in enumerate(labels.videos):
            fn = vid.filename

            # Display filename(s)
            if isinstance(fn, list):
                if show_all_images:
                    click.echo(f"  [{i}] ({len(fn)} images)")
                    for img in fn:
                        click.echo(f"        {img}")
                else:
                    click.echo(f"  [{i}] {fn[0]} ... ({len(fn)} images)")
            else:
                click.echo(f"  [{i}] {fn}")

            # Display source video
            if show_source_effective and vid.source_video is not None:
                click.echo(f"      Source: {vid.source_video.filename}")

            # Display original video
            if show_original_effective and vid.original_video is not None:
                click.echo(f"      Original: {vid.original_video.filename}")

        return

    # Update mode: require -o
    if output_path is None:
        raise click.ClickException(
            "Output path (-o) required when using --filename, --map, or --prefix"
        )

    # Validate only one update mode
    modes = sum(
        [
            len(new_filenames) > 0,
            len(filename_map) > 0,
            len(prefix_map) > 0,
        ]
    )
    if modes > 1:
        raise click.ClickException(
            "Only one mode allowed: --filename, --map, or --prefix (not multiple)"
        )

    # Capture old filenames for comparison
    old_filenames = [v.filename for v in labels.videos]

    # Apply replacement (no video access needed)
    try:
        if new_filenames:
            labels.replace_filenames(
                new_filenames=list(new_filenames),
                open_videos=False,
            )
        elif filename_map:
            labels.replace_filenames(
                filename_map=dict(filename_map),
                open_videos=False,
            )
        elif prefix_map:
            labels.replace_filenames(
                prefix_map=dict(prefix_map),
                open_videos=False,
            )
    except ValueError as e:
        raise click.ClickException(str(e))

    # Capture new filenames and compare
    new_filenames_result = [v.filename for v in labels.videos]

    changed = []
    unchanged = []
    for i, (old, new) in enumerate(zip(old_filenames, new_filenames_result)):
        if old != new:
            changed.append((i, old, new))
        else:
            unchanged.append((i, old))

    # Save the output file
    try:
        io_main.save_file(labels, str(output_path))
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Report changes
    n_videos = len(labels.videos)
    if changed:
        click.echo(f"Replaced ({len(changed)}/{n_videos}):")
        for i, old, new in changed:
            # For image sequences, show first filename only
            old_display = old[0] if isinstance(old, list) else old
            new_display = new[0] if isinstance(new, list) else new
            click.echo(f"  [{i}] {old_display} -> {new_display}")

    if unchanged:
        click.echo(f"Unchanged ({len(unchanged)}/{n_videos}):")
        for i, old in unchanged:
            old_display = old[0] if isinstance(old, list) else old
            click.echo(f"  [{i}] {old_display}")

    click.echo(f"Saved: {output_path}")


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input labels file (.slp, .nwb, etc.). Can also be passed as positional arg.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output path. Default: {input}.viz.mp4 (video), {input}.lf={N}.png (image).",
)
# Frame selection options
@click.option(
    "--lf",
    "lf_ind",
    type=int,
    default=None,
    help="Render single labeled frame by index. Outputs PNG image.",
)
@click.option(
    "--frame",
    "frame_idx",
    type=int,
    default=None,
    help="Render frame by video frame index (with --video). Outputs PNG.",
)
@click.option(
    "--start",
    "start_frame_idx",
    type=int,
    default=None,
    help="Start frame index for video (0-based, inclusive).",
)
@click.option(
    "--end",
    "end_frame_idx",
    type=int,
    default=None,
    help="End frame index for video (0-based, exclusive). Default: last labeled frame.",
)
@click.option(
    "--video",
    "video_ind",
    type=int,
    default=0,
    show_default=True,
    help="Video index for multi-video labels.",
)
@click.option(
    "--all-frames/--labeled-only",
    "all_frames",
    default=None,
    help="Render all frames (--all-frames) or only labeled (--labeled-only). "
    "Default: --all-frames for single-video files.",
)
# Quality options
@click.option(
    "--preset",
    type=click.Choice(["preview", "draft", "final"]),
    default=None,
    help="Quality preset: preview=0.25x, draft=0.5x, final=1.0x scale. Default: 1.0x.",
)
@click.option(
    "--scale",
    type=float,
    default=None,
    help="Scale factor (overrides --preset). Default: 1.0.",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Output video FPS. Default: source video FPS.",
)
@click.option(
    "--crf",
    type=int,
    default=25,
    show_default=True,
    help="Video quality (2-32, lower=better quality, larger file).",
)
@click.option(
    "--x264-preset",
    "x264_preset",
    type=click.Choice(
        ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]
    ),
    default="superfast",
    show_default=True,
    help="H.264 encoding speed/compression trade-off.",
)
# Appearance options
@click.option(
    "--color-by",
    type=click.Choice(["auto", "track", "instance", "node"]),
    default="auto",
    show_default=True,
    help="Color scheme: auto (smart default), track, instance, or node.",
)
@click.option(
    "--palette",
    type=str,
    default="standard",
    show_default=True,
    help="Color palette (standard, tableau10, distinct, glasbey, etc.).",
)
@click.option(
    "--marker-shape",
    type=click.Choice(["circle", "square", "diamond", "triangle", "cross"]),
    default="circle",
    show_default=True,
    help="Node marker shape.",
)
@click.option(
    "--marker-size",
    type=float,
    default=4.0,
    show_default=True,
    help="Node marker radius in pixels.",
)
@click.option(
    "--line-width",
    type=float,
    default=2.0,
    show_default=True,
    help="Edge line width in pixels.",
)
@click.option(
    "--alpha",
    type=float,
    default=1.0,
    show_default=True,
    help="Pose overlay transparency (0.0-1.0).",
)
@click.option(
    "--no-nodes",
    is_flag=True,
    default=False,
    help="Hide node markers. Default: show nodes.",
)
@click.option(
    "--no-edges",
    is_flag=True,
    default=False,
    help="Hide skeleton edges. Default: show edges.",
)
# Crop options
@click.option(
    "--crop",
    "crop_str",
    type=str,
    default=None,
    help="Crop region: 'x1,y1,x2,y2' (pixels or normalized 0.0-1.0).",
)
# Background options
@click.option(
    "--background",
    type=str,
    default="video",
    show_default=True,
    help="Background: 'video' (use source video), or color name/hex/RGB "
    "(e.g., 'black', '#ff0000', '128,128,128'). Use a color if video unavailable.",
)
# Info options
@click.option(
    "--list-colors",
    is_flag=True,
    default=False,
    help="List available named colors and exit.",
)
@click.option(
    "--list-palettes",
    is_flag=True,
    default=False,
    help="List available color palettes and exit.",
)
def render(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path | None,
    lf_ind: int | None,
    frame_idx: int | None,
    start_frame_idx: int | None,
    end_frame_idx: int | None,
    video_ind: int,
    all_frames: bool | None,
    preset: str | None,
    scale: float | None,
    fps: float | None,
    crf: int,
    x264_preset: str,
    color_by: str,
    palette: str,
    marker_shape: str,
    marker_size: float,
    line_width: float,
    alpha: float,
    no_nodes: bool,
    no_edges: bool,
    crop_str: str | None,
    background: str,
    list_colors: bool,
    list_palettes: bool,
) -> None:
    """Render pose predictions as video or single image.

    [bold]Video mode[/] (default): Renders all labeled frames to a video file.

    [bold]Image mode[/]: Renders a single frame to PNG using --lf or --frame.

    [dim]Examples:[/]

        [bold]Video rendering:[/]

        $ sio render predictions.slp                         # -> predictions.viz.mp4

        $ sio render predictions.slp -o output.mp4           # Explicit output

        $ sio render predictions.slp --preset preview        # Fast 0.25x preview

        $ sio render predictions.slp --start 100 --end 200

        $ sio render predictions.slp --fps 15                # Slow motion

        $ sio render predictions.slp --all-frames            # Include unlabeled frames

        [bold]Single image rendering:[/]

        $ sio render predictions.slp --lf 0                  # -> predictions.lf=0.png

        $ sio render predictions.slp --frame 42              # -> *.frame=42.png

        $ sio render labels.slp --lf 5 -o frame.png          # Explicit output

        [bold]Cropping:[/]

        $ sio render predictions.slp --lf 0 --crop 100,100,300,300

        $ sio render predictions.slp --lf 0 --crop 0.25,0.25,0.75,0.75  # Normalized

        $ sio render predictions.slp -o cropped.mp4 --crop 100,100,300,300

        [bold]Background (when video unavailable):[/]

        $ sio render predictions.slp --background black      # Solid black background

        $ sio render predictions.slp --background "#333333"  # Custom hex color

        [bold]Color info:[/]

        $ sio render --list-colors                           # Show available colors

        $ sio render --list-palettes                         # Show available palettes
    """
    # Handle --list-colors flag
    if list_colors:
        from sleap_io.rendering.colors import NAMED_COLORS

        console.print("[bold]Available named colors:[/]")
        color_items = []
        for name, rgb in sorted(NAMED_COLORS.items()):
            hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            color_items.append(f"[{hex_code}]{name}[/] ({hex_code})")
        console.print("  " + ", ".join(color_items))
        console.print()
        console.print("[dim]You can also use:[/]")
        console.print("  - Hex codes: #ff0000, #333")
        console.print("  - RGB tuples: 255,128,0")
        console.print("  - Grayscale: 128 or 0.5")
        return

    # Handle --list-palettes flag
    if list_palettes:
        from sleap_io.rendering.colors import PALETTES

        console.print("[bold]Built-in palettes:[/]")
        for name in sorted(PALETTES.keys()):
            n_colors = len(PALETTES[name])
            console.print(f"  {name} ({n_colors} colors)")
        console.print()
        console.print("[bold]Colorcet palettes[/] (require [cyan]colorcet[/] package):")
        console.print("  glasbey, glasbey_hv, glasbey_cool, glasbey_warm")
        console.print("  fire, rainbow4, blues, and many more")
        console.print()
        console.print("[dim]Install colorcet: pip install colorcet[/]")
        return

    # Resolve input path from positional arg or -i option
    input_path = _resolve_input(input_arg, input_opt, "input file")

    # Load labels
    try:
        labels = io_main.load_file(str(input_path), open_videos=True)
    except Exception as e:
        raise click.ClickException(f"Failed to load input: {e}")

    if not isinstance(labels, Labels):
        raise click.ClickException(
            f"Input is not a labels file (got {type(labels).__name__})"
        )

    # Determine render mode: single image vs video
    single_image_mode = lf_ind is not None or frame_idx is not None

    # Detect single-video prediction file and auto-enable --all-frames
    # This is the common "pure prediction" case where every frame has predictions
    effective_all_frames = all_frames  # May be None, True, or False
    if not single_image_mode and all_frames is None:
        is_single_video = len(labels.videos) == 1
        if is_single_video:
            # Check if all labeled frames are from this video
            target_video = (
                labels.videos[video_ind] if video_ind < len(labels.videos) else None
            )
            all_from_same_video = all(
                lf.video == target_video for lf in labels.labeled_frames
            )
            if all_from_same_video:
                # Auto-enable all_frames for single-video prediction files
                effective_all_frames = True
    # Default to False if not auto-detected
    if effective_all_frames is None:
        effective_all_frames = False

    # Validate conflicting options
    if single_image_mode and (start_frame_idx is not None or end_frame_idx is not None):
        raise click.ClickException(
            "Cannot use --start/--end with --lf or --frame. "
            "Use --lf for single image or omit it for video."
        )

    if lf_ind is not None and frame_idx is not None:
        raise click.ClickException("Cannot use both --lf and --frame. Choose one.")

    # Determine output path
    if output_path is None:
        input_stem = input_path.stem
        if single_image_mode:
            if lf_ind is not None:
                output_path = input_path.with_name(f"{input_stem}.lf={lf_ind}.png")
            else:
                output_path = input_path.with_name(
                    f"{input_stem}.video={video_ind}.frame={frame_idx}.png"
                )
        else:
            output_path = input_path.with_suffix(".viz.mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle scale/preset
    effective_scale = 1.0
    if scale is not None:
        effective_scale = scale
    elif preset is not None:
        preset_scales = {"preview": 0.25, "draft": 0.5, "final": 1.0}
        effective_scale = preset_scales.get(preset, 1.0)

    # Parse crop specification
    crop = _parse_crop_string(crop_str)

    try:
        if single_image_mode:
            # Single image rendering
            from sleap_io.rendering import render_image

            if lf_ind is not None:
                # Render by labeled frame index
                if lf_ind < 0 or lf_ind >= len(labels.labeled_frames):
                    n_lf = len(labels.labeled_frames)
                    raise click.ClickException(
                        f"--lf {lf_ind} out of range. "
                        f"Labels has {n_lf} labeled frames (0-{n_lf - 1})."
                    )
                render_image(
                    labels,
                    save_path=output_path,
                    lf_ind=lf_ind,
                    crop=crop,
                    scale=effective_scale,
                    color_by=color_by,
                    palette=palette,
                    marker_shape=marker_shape,
                    marker_size=marker_size,
                    line_width=line_width,
                    alpha=alpha,
                    show_nodes=not no_nodes,
                    show_edges=not no_edges,
                    background=background,
                )
            else:
                # Render by video + frame_idx
                if video_ind >= len(labels.videos):
                    n_vid = len(labels.videos)
                    raise click.ClickException(
                        f"--video {video_ind} out of range. "
                        f"Labels has {n_vid} videos (0-{n_vid - 1})."
                    )
                video = labels.videos[video_ind]
                render_image(
                    labels,
                    save_path=output_path,
                    video=video,
                    frame_idx=frame_idx,
                    crop=crop,
                    scale=effective_scale,
                    color_by=color_by,
                    palette=palette,
                    marker_shape=marker_shape,
                    marker_size=marker_size,
                    line_width=line_width,
                    alpha=alpha,
                    show_nodes=not no_nodes,
                    show_edges=not no_edges,
                    background=background,
                )
        else:
            # Video rendering
            from sleap_io.rendering import render_video

            render_video(
                labels,
                output_path,
                video=video_ind,
                crop=crop,
                scale=effective_scale,
                fps=fps,
                crf=crf,
                x264_preset=x264_preset,
                color_by=color_by,
                palette=palette,
                marker_shape=marker_shape,
                marker_size=marker_size,
                line_width=line_width,
                alpha=alpha,
                show_nodes=not no_nodes,
                show_edges=not no_edges,
                start=start_frame_idx,
                end=end_frame_idx,
                include_unlabeled=effective_all_frames,
                show_progress=True,
                background=background,
            )
    except Exception as e:
        raise click.ClickException(f"Failed to render: {e}")

    click.echo(f"Rendered: {input_path} -> {output_path}")


# =============================================================================
# FIX COMMAND
# =============================================================================


def _get_default_output_path(input_path: Path) -> Path:
    """Generate default output path for fix command.

    Examples:
        labels.slp -> labels.fixed.slp
        labels.pkg.slp -> labels.fixed.pkg.slp
    """
    name = input_path.name
    if name.endswith(".pkg.slp"):
        new_name = name[:-8] + ".fixed.pkg.slp"
    elif name.endswith(".slp"):
        new_name = name[:-4] + ".fixed.slp"
    else:
        new_name = name + ".fixed"
    return input_path.parent / new_name


def _analyze_duplicate_videos(labels: Labels) -> list[tuple[Video, list[Video], int]]:
    """Find groups of duplicate videos using VideoMatcher.

    Returns:
        List of tuples: (canonical_video, [duplicates], frames_to_reassign)
    """
    from sleap_io.model.matching import VideoMatcher

    matcher = VideoMatcher(method="auto")
    video_groups: list[list[Video]] = []

    for video in labels.videos:
        matched_group = None
        for group in video_groups:
            if matcher.match(group[0], video):
                matched_group = group
                break
        if matched_group:
            matched_group.append(video)
        else:
            video_groups.append([video])

    # Convert to result format
    results = []
    for group in video_groups:
        if len(group) > 1:
            canonical = group[0]
            duplicates = group[1:]
            frames_to_reassign = sum(
                1 for lf in labels.labeled_frames if lf.video in duplicates
            )
            results.append((canonical, duplicates, frames_to_reassign))

    return results


def _analyze_skeletons(
    labels: Labels,
) -> tuple[
    list[
        tuple[Skeleton, int, int]
    ],  # All skeletons with (skel, user_count, pred_count)
    list[Skeleton],  # Unused
    list[Skeleton],  # Pred-only
    list[Skeleton],  # Has user labels
]:
    """Analyze skeleton usage.

    Returns:
        Tuple of (all_usage, unused, pred_only, user_skeletons)
    """
    from collections import defaultdict

    usage: dict[Skeleton, dict[str, int]] = defaultdict(lambda: {"user": 0, "pred": 0})

    # Initialize all skeletons
    for skel in labels.skeletons:
        _ = usage[skel]

    # Count usage
    for lf in labels.labeled_frames:
        for inst in lf:
            if isinstance(inst, PredictedInstance):
                usage[inst.skeleton]["pred"] += 1
            else:
                usage[inst.skeleton]["user"] += 1

    all_usage = [(s, u["user"], u["pred"]) for s, u in usage.items()]
    unused = [s for s, u in usage.items() if u["user"] == 0 and u["pred"] == 0]
    pred_only = [s for s, u in usage.items() if u["user"] == 0 and u["pred"] > 0]
    user_skeletons = [s for s, u in usage.items() if u["user"] > 0]

    return all_usage, unused, pred_only, user_skeletons


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input labels file. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output labels file. Default: {input}.fixed.slp",
)
# Fix operations
@click.option(
    "--deduplicate-videos/--no-deduplicate-videos",
    "deduplicate_videos",
    default=True,
    show_default=True,
    help="Merge duplicate video entries pointing to same file.",
)
# Skeleton handling
@click.option(
    "--remove-unused-skeletons/--no-remove-unused-skeletons",
    "remove_unused_skeletons",
    default=True,
    show_default=True,
    help="Remove skeletons with no instances or only predictions.",
)
@click.option(
    "--consolidate-skeletons/--no-consolidate-skeletons",
    "consolidate_skeletons",
    default=False,
    show_default=True,
    help="Keep most frequent skeleton, DELETE instances using other skeletons.",
)
# Remove predictions
@click.option(
    "--remove-predictions/--no-remove-predictions",
    "remove_predictions",
    default=False,
    show_default=True,
    help="Remove ALL predicted instances.",
)
@click.option(
    "--remove-untracked-predictions/--no-remove-untracked-predictions",
    "remove_untracked_predictions",
    default=False,
    show_default=True,
    help="Remove only predictions with no track assignment.",
)
# Remove unused metadata
@click.option(
    "--remove-unused-tracks/--no-remove-unused-tracks",
    "remove_unused_tracks",
    default=True,
    show_default=True,
    help="Remove tracks not used by any instance.",
)
# Remove empty data
@click.option(
    "--remove-empty-instances/--no-remove-empty-instances",
    "remove_empty_instances",
    default=False,
    show_default=True,
    help="Remove instances with no visible points.",
)
@click.option(
    "--remove-empty-frames/--no-remove-empty-frames",
    "remove_empty_frames",
    default=True,
    show_default=True,
    help="Remove frames with no instances.",
)
# Remove videos
@click.option(
    "--remove-unlabeled-videos/--no-remove-unlabeled-videos",
    "remove_unlabeled_videos",
    default=False,
    show_default=True,
    help="Remove videos with no labeled frames.",
)
# Filename options (passthrough to replace_filenames)
@click.option(
    "--prefix",
    "prefix_map",
    nargs=2,
    multiple=True,
    metavar="OLD NEW",
    help="Replace OLD path prefix with NEW.",
)
@click.option(
    "--map",
    "filename_map",
    nargs=2,
    multiple=True,
    metavar="OLD NEW",
    help="Replace OLD filename with NEW.",
)
# Mode options
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Analyze without making changes.",
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    default=False,
    help="Show detailed analysis.",
)
def fix(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path | None,
    deduplicate_videos: bool,
    remove_unused_skeletons: bool,
    consolidate_skeletons: bool,
    remove_predictions: bool,
    remove_untracked_predictions: bool,
    remove_unused_tracks: bool,
    remove_empty_instances: bool,
    remove_empty_frames: bool,
    remove_unlabeled_videos: bool,
    prefix_map: tuple[tuple[str, str], ...],
    filename_map: tuple[tuple[str, str], ...],
    dry_run: bool,
    verbose: bool,
):
    r"""Fix common issues in labels files.

    Automatically detects and fixes common problems:

    • [bold]Duplicate videos[/]: Merges videos that point to the same file
    • [bold]Multiple skeletons[/]: Removes unused or prediction-only skeletons
    • [bold]Predictions[/]: Optionally removes all or untracked predictions
    • [bold]Path fixes[/]: Updates video paths with --prefix or --map
    • [bold]Cleanup[/]: Removes empty frames, unused tracks, etc.

    [dim]Examples:[/]

        $ sio fix labels.slp                         # Auto-detect and fix
        $ sio fix labels.slp --dry-run               # Preview without saving
        $ sio fix labels.slp -o fixed.slp            # Explicit output
        $ sio fix labels.slp --remove-predictions    # Remove all predictions
        $ sio fix labels.slp --remove-untracked-predictions  # Surgical removal
        $ sio fix labels.slp --consolidate-skeletons # Force single skeleton
        $ sio fix labels.slp --prefix "C:\\data" /mnt/data
    """
    # Resolve input
    input_path = _resolve_input(input_arg, input_opt, "input labels file")

    # Determine default output path
    if output_path is None:
        output_path = _get_default_output_path(input_path)

    # Check for filename options
    has_filename_opts = len(prefix_map) > 0 or len(filename_map) > 0

    # Load the input file
    console.print(f"[bold]Loading:[/] {input_path}")
    try:
        labels = io_main.load_file(str(input_path), open_videos=False)
    except Exception as e:
        raise click.ClickException(f"Failed to load input file: {e}")

    if not isinstance(labels, Labels):
        raise click.ClickException(
            f"Input file is not a labels file (got {type(labels).__name__})."
        )

    # Print initial stats
    n_videos = len(labels.videos)
    n_frames = len(labels.labeled_frames)
    n_skeletons = len(labels.skeletons)
    n_tracks = len(labels.tracks)
    console.print(
        f"[dim]  {n_videos} videos, {n_frames} frames, "
        f"{n_skeletons} skeletons, {n_tracks} tracks[/]"
    )

    # ==========================================================================
    # ANALYSIS PHASE
    # ==========================================================================
    console.print()
    console.print("[bold]Analyzing...[/]")

    # Analyze duplicate videos
    duplicate_groups = _analyze_duplicate_videos(labels)
    has_duplicate_videos = len(duplicate_groups) > 0

    # Analyze skeletons
    all_skel_usage, unused_skels, pred_only_skels, user_skels = _analyze_skeletons(
        labels
    )
    has_skeleton_issues = len(unused_skels) > 0 or len(pred_only_skels) > 0
    has_multi_user_skeletons = len(user_skels) > 1

    # Find most frequent user skeleton for consolidation
    most_frequent_skeleton: Skeleton | None = None
    other_user_skeletons: list[Skeleton] = []
    if has_multi_user_skeletons:
        # Sort by user count descending
        user_skel_usage = [(s, u, p) for s, u, p in all_skel_usage if u > 0]
        user_skel_usage.sort(key=lambda x: x[1], reverse=True)
        most_frequent_skeleton = user_skel_usage[0][0]
        other_user_skeletons = [s for s, _, _ in user_skel_usage[1:]]

    # Count predictions and untracked predictions
    n_predictions = 0
    n_untracked_predictions = 0
    for lf in labels.labeled_frames:
        for inst in lf:
            if isinstance(inst, PredictedInstance):
                n_predictions += 1
                if inst.track is None:
                    n_untracked_predictions += 1

    # ==========================================================================
    # REPORT FINDINGS
    # ==========================================================================

    # Videos section
    console.print()
    if has_duplicate_videos:
        console.print(
            f"[yellow]⚠ Videos:[/] Found {len(duplicate_groups)} duplicate group(s)"
        )
        if verbose:
            for canonical, duplicates, n_reassign in duplicate_groups:
                console.print(f"  [dim]Canonical:[/] {canonical.filename}")
                for dup in duplicates:
                    console.print(f"  [dim]  Duplicate:[/] {dup.filename}")
                console.print(f"  [dim]  Frames to reassign:[/] {n_reassign}")
    else:
        console.print("[green]✓ Videos:[/] No duplicates found")

    # Skeletons section
    if has_skeleton_issues or has_multi_user_skeletons:
        status = "[yellow]⚠" if has_skeleton_issues else "[blue]ℹ"
        console.print(f"{status} Skeletons:[/]")
        if verbose or has_skeleton_issues or has_multi_user_skeletons:
            for skel, user_count, pred_count in all_skel_usage:
                status_str = ""
                if user_count == 0 and pred_count == 0:
                    status_str = " [yellow](unused)[/]"
                elif user_count == 0 and pred_count > 0:
                    status_str = " [yellow](predictions only)[/]"
                elif has_multi_user_skeletons and skel is most_frequent_skeleton:
                    status_str = " [green](most frequent)[/]"
                console.print(
                    f"  '{skel.name}': {user_count} user, {pred_count} pred{status_str}"
                )

        # Warning for multiple user skeletons
        if has_multi_user_skeletons and not consolidate_skeletons:
            console.print()
            console.print(
                "[yellow]⚠  WARNING: Multiple skeletons have user instances![/]"
            )
            if most_frequent_skeleton:
                other_counts = [
                    (s, u) for s, u, _ in all_skel_usage if s in other_user_skeletons
                ]
                total_other = sum(u for _, u in other_counts)
                console.print(
                    f"    Use --consolidate-skeletons to keep "
                    f"'{most_frequent_skeleton.name}' and remove {total_other} "
                    f"instances."
                )
                console.print(
                    "    This is irreversible - review carefully before proceeding."
                )
    else:
        console.print(f"[green]✓ Skeletons:[/] {n_skeletons} skeleton(s), all in use")

    # Predictions section
    if n_predictions > 0:
        untracked_info = ""
        if n_untracked_predictions > 0:
            untracked_info = f" ({n_untracked_predictions} untracked)"
        console.print(
            f"[blue]ℹ Predictions:[/] {n_predictions} predicted "
            f"instances{untracked_info}"
        )
    else:
        console.print("[green]✓ Predictions:[/] None")

    # ==========================================================================
    # DETERMINE ACTIONS
    # ==========================================================================

    actions = []

    # Video deduplication
    if deduplicate_videos and has_duplicate_videos:
        actions.append(f"Merge {len(duplicate_groups)} duplicate video group(s)")

    # Skeleton handling
    if remove_unused_skeletons and unused_skels:
        actions.append(f"Remove {len(unused_skels)} unused skeleton(s)")
    if remove_unused_skeletons and pred_only_skels:
        n_pred_instances = sum(p for s, u, p in all_skel_usage if s in pred_only_skels)
        actions.append(
            f"Remove {len(pred_only_skels)} prediction-only skeleton(s) "
            f"and {n_pred_instances} prediction(s)"
        )

    # Skeleton consolidation
    if consolidate_skeletons and has_multi_user_skeletons:
        other_instance_count = sum(
            u for s, u, _ in all_skel_usage if s in other_user_skeletons
        )
        actions.append(
            f"[red]CONSOLIDATE: Keep '{most_frequent_skeleton.name}', "
            f"DELETE {other_instance_count} instances from other skeletons[/]"
        )

    # Prediction removal
    if remove_predictions and n_predictions > 0:
        actions.append(f"Remove {n_predictions} prediction(s)")
    elif remove_untracked_predictions and n_untracked_predictions > 0:
        actions.append(f"Remove {n_untracked_predictions} untracked prediction(s)")

    # Filename changes
    if has_filename_opts:
        actions.append("Update video filenames")

    # Cleanup actions (these happen after other modifications)
    if remove_empty_instances:
        actions.append("Remove empty instances")
    if remove_unused_tracks:
        actions.append("Remove unused tracks")
    if remove_unlabeled_videos:
        actions.append("Remove unlabeled videos")
    if remove_empty_frames:
        actions.append("Remove empty frames")

    # Print planned actions
    console.print()
    if actions:
        console.print("[bold]Actions:[/]")
        for action in actions:
            prefix = "[dim]→[/]" if dry_run else "[green]→[/]"
            console.print(f"  {prefix} {action}")
    else:
        console.print("[green]No issues to fix.[/]")
        if not dry_run:
            console.print(f"[dim]No changes needed. Saving as: {output_path}[/]")

    if dry_run:
        console.print()
        console.print("[yellow][DRY RUN - no changes made][/]")
        return

    # ==========================================================================
    # APPLY FIXES (order matters!)
    # ==========================================================================

    # 1. Fix duplicate videos (early - affects frame references)
    if deduplicate_videos and has_duplicate_videos:
        for canonical, duplicates, _ in duplicate_groups:
            for dup in duplicates:
                # Reassign frames
                for lf in labels.labeled_frames:
                    if lf.video is dup:
                        lf.video = canonical
                # Remove duplicate video
                labels.videos.remove(dup)

    # 2. Skeleton consolidation (before other skeleton operations)
    if consolidate_skeletons and has_multi_user_skeletons:
        console.print()
        console.print("[red bold]CONSOLIDATING SKELETONS (destructive operation)[/]")
        console.print(f"    Keeping: '{most_frequent_skeleton.name}'")

        deleted_count = 0
        frames_affected = 0
        for lf in labels.labeled_frames:
            instances_to_remove = [
                inst for inst in lf.instances if inst.skeleton in other_user_skeletons
            ]
            if instances_to_remove:
                frames_affected += 1
                deleted_count += len(instances_to_remove)
                for inst in instances_to_remove:
                    lf.instances.remove(inst)

        # Remove the other skeletons
        for skel in other_user_skeletons:
            if skel in labels.skeletons:
                labels.skeletons.remove(skel)

        console.print(
            f"    Deleted {deleted_count} instances from {frames_affected} frames."
        )

    # 3. Remove unused skeletons (completely unused)
    if remove_unused_skeletons:
        for skel in unused_skels:
            if skel in labels.skeletons:
                labels.skeletons.remove(skel)

    # 4. Remove prediction-only skeletons and their predictions
    if remove_unused_skeletons and pred_only_skels:
        for lf in labels.labeled_frames:
            lf.instances = [
                inst for inst in lf.instances if inst.skeleton not in pred_only_skels
            ]
        for skel in pred_only_skels:
            if skel in labels.skeletons:
                labels.skeletons.remove(skel)

    # 5. Remove predictions
    if remove_predictions and n_predictions > 0:
        labels.remove_predictions(clean=False)
    elif remove_untracked_predictions and n_untracked_predictions > 0:
        for lf in labels.labeled_frames:
            lf.instances = [
                inst
                for inst in lf.instances
                if not (isinstance(inst, PredictedInstance) and inst.track is None)
            ]

    # 6. Apply filename changes
    if has_filename_opts:
        try:
            if prefix_map:
                labels.replace_filenames(
                    prefix_map=dict(prefix_map),
                    open_videos=False,
                )
            elif filename_map:
                labels.replace_filenames(
                    filename_map=dict(filename_map),
                    open_videos=False,
                )
        except ValueError as e:
            raise click.ClickException(f"Failed to update filenames: {e}")

    # 7. Cleanup (order: empty instances -> tracks -> unlabeled videos -> frames)
    # Empty instances first (may create empty frames)
    if remove_empty_instances:
        labels.clean(
            frames=False,
            empty_instances=True,
            skeletons=False,
            tracks=False,
            videos=False,
        )

    # Unused tracks
    if remove_unused_tracks:
        labels.clean(
            frames=False,
            empty_instances=False,
            skeletons=False,
            tracks=True,
            videos=False,
        )

    # Unlabeled videos
    if remove_unlabeled_videos:
        labels.clean(
            frames=False,
            empty_instances=False,
            skeletons=False,
            tracks=False,
            videos=True,
        )

    # Empty frames LAST (depends on all other removals)
    if remove_empty_frames:
        labels.clean(
            frames=True,
            empty_instances=False,
            skeletons=False,
            tracks=False,
            videos=False,
        )

    # ==========================================================================
    # SAVE OUTPUT
    # ==========================================================================

    try:
        save_kwargs: dict = {}
        if _should_preserve_embedded(input_path, output_path, embed=None):
            # Preserve existing embedded videos from pkg.slp input
            save_kwargs["embed"] = None
        io_main.save_file(labels, str(output_path), **save_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Print final stats
    console.print()
    console.print(f"[bold green]Saved:[/] {output_path}")
    console.print(
        f"[dim]  {len(labels.videos)} videos, {len(labels.labeled_frames)} frames, "
        f"{len(labels.skeletons)} skeletons, {len(labels.tracks)} tracks[/]"
    )


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input SLP file path. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output .pkg.slp file path.",
)
@click.option(
    "--user/--no-user",
    "include_user",
    default=True,
    show_default=True,
    help="Include user-labeled frames.",
)
@click.option(
    "--predictions/--no-predictions",
    "include_predictions",
    default=False,
    show_default=True,
    help="Include prediction-only frames (no user labels).",
)
@click.option(
    "--suggestions/--no-suggestions",
    "include_suggestions",
    default=False,
    show_default=True,
    help="Include suggested frames.",
)
def embed(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path,
    include_user: bool,
    include_predictions: bool,
    include_suggestions: bool,
):
    """Embed video frames into a labels file.

    Creates a portable .pkg.slp file with embedded images that can be shared
    without requiring the original video files.

    [bold]Frame selection:[/]
    - --user: Frames with user-labeled instances (default: on)
    - --predictions: Frames with only predicted instances (default: off)
    - --suggestions: Suggested frames for labeling (default: off)

    [dim]Examples:[/]

        $ sio embed labels.slp -o labels.pkg.slp
        $ sio embed labels.slp -o labels.pkg.slp --suggestions
        $ sio embed labels.slp -o labels.pkg.slp --predictions --suggestions
        $ sio embed labels.slp -o labels.pkg.slp --no-user --suggestions
    """
    # Resolve input from positional arg or -i option
    input_path = _resolve_input(input_arg, input_opt, "input file")

    # Validate input is SLP format
    if not str(input_path).lower().endswith(".slp"):
        raise click.ClickException("Input file must be a .slp file.")

    # Load the input file with video access for embedding
    try:
        labels = io_main.load_file(str(input_path), format="slp", open_videos=True)
    except Exception as e:
        raise click.ClickException(f"Failed to load input file: {e}")

    # Check if at least one frame type is selected
    if not include_user and not include_predictions and not include_suggestions:
        raise click.ClickException(
            "No frames to embed. Enable at least one of: --user, --predictions, "
            "--suggestions"
        )

    # Build list of frames to embed
    frames_to_embed: list[tuple[Video, int]] = []
    frame_counts: dict[str, int] = {}

    # Get user-labeled frames
    user_frame_set: set[tuple[Video, int]] = set()
    if include_user:
        user_frames = [(lf.video, lf.frame_idx) for lf in labels.user_labeled_frames]
        frames_to_embed.extend(user_frames)
        user_frame_set = set(user_frames)
        frame_counts["user"] = len(user_frames)

    # Get prediction-only frames (frames with predictions but no user instances)
    if include_predictions:
        pred_frames = [
            (lf.video, lf.frame_idx)
            for lf in labels.labeled_frames
            if not lf.has_user_instances
        ]
        frames_to_embed.extend(pred_frames)
        frame_counts["predictions"] = len(pred_frames)

    # Get suggested frames
    if include_suggestions:
        suggestion_frames = [(sf.video, sf.frame_idx) for sf in labels.suggestions]
        # Don't double-count frames already in user set
        new_suggestions = [f for f in suggestion_frames if f not in user_frame_set]
        frames_to_embed.extend(new_suggestions)
        frame_counts["suggestions"] = len(suggestion_frames)

    # Check if there are any frames to embed
    if not frames_to_embed:
        raise click.ClickException(
            "No frames to embed with the selected options. "
            "Try different flags or check if the file has the expected frame types."
        )

    # Save with embedding
    try:
        labels.save(str(output_path), embed=frames_to_embed)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Success message
    console.print(f"[bold green]Embedded:[/] {input_path} -> {output_path}")
    frame_summary = ", ".join(f"{k}: {v}" for k, v in frame_counts.items())
    console.print(f"[dim]Frames: {frame_summary}[/]")


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input .pkg.slp file path. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output .slp file path.",
)
def unembed(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path,
):
    """Remove embedded frames and restore video references.

    Converts a .pkg.slp file back to a regular .slp file that references the
    original video files. This requires that the original videos are accessible
    and that the labels file contains source video metadata.

    [bold]Notes:[/]
    - Original video files must be accessible at their stored paths
    - Legacy .pkg.slp files without source video metadata cannot be unembedded

    [dim]Examples:[/]

        $ sio unembed labels.pkg.slp -o labels.slp
        $ sio unembed -i labels.pkg.slp -o labels.slp
    """
    # Resolve input from positional arg or -i option
    input_path = _resolve_input(input_arg, input_opt, "input file")

    # Validate input is SLP format
    if not str(input_path).lower().endswith(".slp"):
        raise click.ClickException("Input file must be a .slp file.")

    # Load the input file
    try:
        labels = io_main.load_file(str(input_path), format="slp", open_videos=True)
    except Exception as e:
        raise click.ClickException(f"Failed to load input file: {e}")

    # Check for embedded videos
    embedded_videos = []
    for video in labels.videos:
        if _is_embedded(video):
            embedded_videos.append(video)

    if not embedded_videos:
        raise click.ClickException(
            "No embedded videos found. This file does not contain embedded frames.\n"
            "Use 'sio show --video' to inspect video details."
        )

    # Check for source video metadata (required for unembedding)
    videos_without_source = []
    for video in embedded_videos:
        source = object.__getattribute__(video, "source_video")
        if source is None:
            videos_without_source.append(video)

    if videos_without_source:
        # Build helpful error message
        n_missing = len(videos_without_source)
        n_total = len(embedded_videos)
        raise click.ClickException(
            f"Cannot unembed: {n_missing}/{n_total} embedded video(s) have no source "
            f"video metadata.\n\n"
            f"This typically happens with legacy .pkg.slp files created before "
            f"source video tracking was added.\n\n"
            f"To inspect video details, run:\n"
            f"  sio show '{input_path}' --video"
        )

    # Save with source video restoration
    try:
        labels.save(str(output_path), embed=False, restore_original_videos=True)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Success message with video info
    console.print(f"[bold green]Unembedded:[/] {input_path} -> {output_path}")
    console.print(f"[dim]Restored {len(embedded_videos)} video(s) to source paths[/]")


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input labels file or video. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output path. Default: {input}.trim.slp (labels) or {input}.trim.mp4 (video).",
)
# Frame range options
@click.option(
    "--start",
    "start_frame",
    type=int,
    default=None,
    help="Start frame index (0-based, inclusive). Default: 0.",
)
@click.option(
    "--end",
    "end_frame",
    type=int,
    default=None,
    help="End frame index (0-based, exclusive). Default: last frame + 1.",
)
# Video selection
@click.option(
    "--video",
    "video_ind",
    type=int,
    default=None,
    help="Video index for multi-video labels. Default: 0 if single video, required "
    "otherwise.",
)
# Video encoding options
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Output video FPS. Default: source video FPS.",
)
@click.option(
    "--crf",
    type=int,
    default=25,
    show_default=True,
    help="Video quality (2-32, lower=better quality, larger file).",
)
@click.option(
    "--x264-preset",
    "x264_preset",
    type=click.Choice(
        ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]
    ),
    default="superfast",
    show_default=True,
    help="H.264 encoding speed/compression trade-off.",
)
def trim(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path | None,
    start_frame: int | None,
    end_frame: int | None,
    video_ind: int | None,
    fps: float | None,
    crf: int,
    x264_preset: str,
) -> None:
    """Trim video and labels to a frame range.

    Creates a trimmed video clip and adjusts frame indices in the labels file
    accordingly. Can also trim standalone video files without labels.

    [bold]Labels mode[/]: Trims both the labels file and associated video.

    [bold]Video mode[/]: Trims a standalone video file (no labels).

    [dim]Examples:[/]

        [bold]Labels trimming:[/]

        $ sio trim labels.slp --start 100 --end 1000           # -> labels.trim.slp

        $ sio trim labels.slp --start 100 --end 1000 -o clip.slp

        $ sio trim labels.slp --start 100 --end 1000 --video 0

        $ sio trim labels.slp --start 100 --end 1000 --crf 18  # Higher quality

        [bold]Video-only trimming:[/]

        $ sio trim video.mp4 --start 100 --end 1000            # -> video.trim.mp4

        $ sio trim video.mp4 --start 0 --end 500 -o clip.mp4

        $ sio trim video.mp4 --start 100 --fps 15              # Change FPS
    """
    # Resolve input path from positional arg or -i option
    input_path = _resolve_input(input_arg, input_opt, "input file")

    # Try to load as labels first, then as video
    labels: Labels | None = None
    video: Video | None = None

    try:
        result = io_main.load_file(str(input_path), open_videos=True)
        if isinstance(result, Labels):
            labels = result
        else:
            # Must be Video per load_file's return type
            video = result
    except Exception as e:
        # Try loading as video directly
        try:
            video = io_main.load_video(str(input_path))
        except Exception:
            raise click.ClickException(f"Failed to load input: {e}")

    # Build video_kwargs from encoding options
    video_kwargs: dict[str, Any] = {
        "crf": crf,
        "preset": x264_preset,
    }
    if fps is not None:
        video_kwargs["fps"] = fps

    if labels is not None:
        # Labels mode: trim labels and video together
        _trim_labels(
            labels=labels,
            input_path=input_path,
            output_path=output_path,
            start_frame=start_frame,
            end_frame=end_frame,
            video_ind=video_ind,
            video_kwargs=video_kwargs,
        )
    else:
        # Video mode: trim video only (video is guaranteed non-None here)
        assert video is not None  # For type checker
        _trim_video(
            video=video,
            input_path=input_path,
            output_path=output_path,
            start_frame=start_frame,
            end_frame=end_frame,
            video_kwargs=video_kwargs,
        )


def _trim_labels(
    labels: Labels,
    input_path: Path,
    output_path: Path | None,
    start_frame: int | None,
    end_frame: int | None,
    video_ind: int | None,
    video_kwargs: dict[str, Any],
) -> None:
    """Trim labels and associated video to a frame range."""
    import numpy as np

    # Resolve video selection
    if video_ind is None:
        if len(labels.videos) == 1:
            video_ind = 0
        else:
            raise click.ClickException(
                f"Multiple videos found ({len(labels.videos)}). "
                "Use --video to specify which video to trim."
            )

    if video_ind >= len(labels.videos):
        raise click.ClickException(
            f"Video index {video_ind} out of range. "
            f"Labels has {len(labels.videos)} video(s) (0-{len(labels.videos) - 1})."
        )

    video = labels.videos[video_ind]

    # Determine frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(video)

    # Validate frame range
    if start_frame < 0:
        raise click.ClickException(f"Start frame must be >= 0 (got {start_frame}).")
    if end_frame <= start_frame:
        raise click.ClickException(
            f"End frame ({end_frame}) must be greater than start frame ({start_frame})."
        )
    if end_frame > len(video):
        console.print(
            f"[yellow]Warning: End frame ({end_frame}) exceeds video length "
            f"({len(video)}). Clamping to video length.[/]"
        )
        end_frame = len(video)

    # Determine output path
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}.trim.slp")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build frame indices array
    frame_inds = np.arange(start_frame, end_frame)

    console.print(f"[bold]Trimming:[/] {input_path}")
    console.print(f"[dim]  Video: {video.filename}[/]")
    console.print(
        f"[dim]  Frame range: {start_frame} to {end_frame} "
        f"({len(frame_inds)} frames)[/]"
    )

    # Use Labels.trim() method
    # Pass video index (int) instead of video object to avoid identity comparison issues
    try:
        trimmed_labels = labels.trim(
            save_path=output_path,
            frame_inds=frame_inds,
            video=video_ind,
            video_kwargs=video_kwargs,
        )
    except Exception as e:
        raise click.ClickException(f"Failed to trim: {e}")

    # Report success
    video_path = output_path.with_suffix(".mp4")
    console.print(f"[bold green]Saved:[/] {output_path}")
    console.print(f"[bold green]Video:[/] {video_path}")
    console.print(
        f"[dim]  {len(trimmed_labels.labeled_frames)} labeled frames "
        f"in trimmed output[/]"
    )


def _trim_video(
    video: Video,
    input_path: Path,
    output_path: Path | None,
    start_frame: int | None,
    end_frame: int | None,
    video_kwargs: dict[str, Any],
) -> None:
    """Trim a standalone video to a frame range."""
    import numpy as np

    # Determine frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(video)

    # Validate frame range
    if start_frame < 0:
        raise click.ClickException(f"Start frame must be >= 0 (got {start_frame}).")
    if end_frame <= start_frame:
        raise click.ClickException(
            f"End frame ({end_frame}) must be greater than start frame ({start_frame})."
        )
    if end_frame > len(video):
        console.print(
            f"[yellow]Warning: End frame ({end_frame}) exceeds video length "
            f"({len(video)}). Clamping to video length.[/]"
        )
        end_frame = len(video)

    # Determine output path
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}.trim.mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build frame indices array
    frame_inds = np.arange(start_frame, end_frame)

    console.print(f"[bold]Trimming:[/] {input_path}")
    console.print(
        f"[dim]  Frame range: {start_frame} to {end_frame} "
        f"({len(frame_inds)} frames)[/]"
    )

    # Use Video.save() method
    try:
        video.save(
            save_path=output_path,
            frame_inds=frame_inds,
            video_kwargs=video_kwargs,
        )
    except Exception as e:
        raise click.ClickException(f"Failed to trim video: {e}")

    console.print(f"[bold green]Saved:[/] {output_path}")


# =============================================================================
# Reencode command utilities
# =============================================================================


def _is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available via imageio-ffmpeg.

    Returns:
        True if ffmpeg is available and can be executed.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return exe is not None and len(exe) > 0
    except Exception:
        return False


def _get_ffmpeg_version() -> str | None:
    """Get ffmpeg version string via imageio-ffmpeg.

    Returns:
        Version string (e.g., "7.0.2-static") or None if not available.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_version()
    except Exception:
        return None


def _run_ffmpeg_info(video_path: str | Path, timeout: int = 10) -> str | None:
    """Run ffmpeg -i to get video info output.

    Args:
        video_path: Path to video file.
        timeout: Timeout in seconds.

    Returns:
        stderr output from ffmpeg (where metadata is printed), or None on failure.
    """
    if not _is_ffmpeg_available():
        return None

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg_exe, "-i", str(video_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # ffmpeg outputs info to stderr (returns non-zero since no output specified)
        return result.stderr
    except Exception:
        return None


def _get_video_encoding_info(video_path: str | Path) -> VideoEncodingInfo | None:
    """Get video encoding information using ffmpeg.

    Uses `ffmpeg -i` to extract metadata without requiring ffprobe.

    Args:
        video_path: Path to video file.

    Returns:
        VideoEncodingInfo with available metadata, or None if ffmpeg unavailable.
    """
    raw = _run_ffmpeg_info(video_path)
    if not raw:
        return None

    info = VideoEncodingInfo()

    # Parse container format
    # Pattern: Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'video.mp4':
    input_match = re.search(r"Input #\d+,\s*([^,]+)", raw)
    if input_match:
        info.container = input_match.group(1).strip()

    # Parse overall bitrate
    # Pattern: Duration: 00:01:13.33, start: 0.000000, bitrate: 104 kb/s
    bitrate_match = re.search(r"bitrate:\s*(\d+)\s*kb/s", raw)
    if bitrate_match:
        info.bitrate_kbps = int(bitrate_match.group(1))

    # Parse video stream info
    # Examples:
    # Stream #0:0: Video: h264 (High), yuv420p, 1920x1080, 25 fps
    # Stream #0:0[0x1](und): Video: h264 (Main), yuv420p, 384x384, 15 fps
    video_line_match = re.search(r"Stream.*Video:\s*(.+)", raw)
    if video_line_match:
        stream_info = video_line_match.group(1)

        # Codec and profile
        # Pattern: h264 (High) or h264 (Main) (avc1 / ...) or just h264
        codec_match = re.match(r"(\w+)(?:\s*\(([^)]+)\))?", stream_info)
        if codec_match:
            info.codec = codec_match.group(1)
            if codec_match.group(2) and "/" not in codec_match.group(2):
                info.codec_profile = codec_match.group(2)

        # Pixel format - common formats after codec section
        pix_fmt_match = re.search(
            r",\s*(yuv\w+|yuvj\w+|rgb\d+|bgr\d+|gray\w*)", stream_info
        )
        if pix_fmt_match:
            info.pixel_format = pix_fmt_match.group(1)

        # FPS - look for "XX fps" pattern
        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stream_info)
        if fps_match:
            info.fps = float(fps_match.group(1))

    return info


def _estimate_gop_size(video_path: str | Path, sample_frames: int = 300) -> int | None:
    """Estimate GOP size by analyzing frame types.

    Args:
        video_path: Path to video file.
        sample_frames: Number of frames to analyze.

    Returns:
        Estimated GOP size in frames, or None if cannot be determined.
    """
    if not _is_ffmpeg_available():
        return None

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [
                ffmpeg_exe,
                "-i",
                str(video_path),
                "-vf",
                f"select='lt(n,{sample_frames})',showinfo",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse showinfo output for frame types (type:I, type:P, type:B)
        frame_types = re.findall(r"type:([IPB])", result.stderr)

        if not frame_types:
            return None

        # Find I-frame (keyframe) positions
        i_frame_positions = [i for i, t in enumerate(frame_types) if t == "I"]

        if len(i_frame_positions) >= 2:
            # Calculate average distance between keyframes
            gaps = [
                i_frame_positions[i + 1] - i_frame_positions[i]
                for i in range(len(i_frame_positions) - 1)
            ]
            if gaps:
                return int(sum(gaps) / len(gaps))

        return None
    except Exception:
        return None


def _get_video_metadata(input_path: Path) -> dict:
    """Get video metadata using imageio.

    Args:
        input_path: Path to video file.

    Returns:
        Dictionary with keys: fps, width, height, num_frames, duration, codec.

    Raises:
        click.ClickException: If metadata extraction fails.
    """
    import imageio.v3 as iio
    import imageio_ffmpeg

    try:
        # Get metadata using imageio's FFMPEG plugin
        meta = iio.immeta(str(input_path), plugin="FFMPEG")

        fps = meta.get("fps", 30.0)
        duration = meta.get("duration", 0.0)
        size = meta.get("source_size", (0, 0))
        codec = meta.get("codec", "unknown")

        # Get frame count using imageio_ffmpeg (more reliable than inf from immeta)
        num_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(input_path))

        return {
            "fps": float(fps),
            "width": int(size[0]),
            "height": int(size[1]),
            "num_frames": int(num_frames),
            "duration": float(duration),
            "codec": codec,
        }
    except Exception as e:
        raise click.ClickException(f"Failed to read video metadata: {e}")


# Quality level to CRF mapping
_QUALITY_TO_CRF = {
    "lossless": 0,
    "high": 18,
    "medium": 25,
    "low": 32,
}


def _build_ffmpeg_reencode_command(
    input_path: Path,
    output_path: Path,
    fps: float,
    output_fps: float | None,
    crf: int,
    preset: str,
    keyframe_interval: float,
    overwrite: bool,
) -> list[str]:
    """Build ffmpeg command for reencoding.

    Args:
        input_path: Path to input video file.
        output_path: Path to output video file.
        fps: Input video frame rate.
        output_fps: Output frame rate (None to preserve input).
        crf: Constant rate factor (0-51).
        preset: x264 encoding preset.
        keyframe_interval: Keyframe interval in seconds.
        overwrite: Whether to overwrite existing output file.

    Returns:
        List of command arguments for ffmpeg.
    """
    import imageio_ffmpeg

    # Calculate GOP size from keyframe interval
    effective_fps = output_fps if output_fps is not None else fps
    gop_size = max(1, int(keyframe_interval * effective_fps))

    # Get ffmpeg executable from imageio-ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe]

    # Overwrite flag
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")

    # Input
    cmd.extend(["-i", str(input_path)])

    # Output frame rate (if changing)
    if output_fps is not None:
        cmd.extend(["-r", str(output_fps)])

    # Video codec and encoding settings
    # Note: We set sc_threshold=0 via x264-params for compatibility with older ffmpeg
    # versions (4.x) that don't support -x264opts. This disables scene detection
    # to ensure fixed keyframe intervals for scientific video workflows.
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-g",
            str(gop_size),  # Keyframe interval
            "-keyint_min",
            str(gop_size),  # Minimum keyframe interval
            "-sc_threshold",
            "0",  # Disable scene detection (compatible with ffmpeg 4.x+)
            "-bf",
            "0",  # No B-frames for better seekability
            "-pix_fmt",
            "yuv420p",  # Maximum compatibility
        ]
    )

    # Handle odd dimensions (pad to even)
    cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])

    # Copy audio if present
    cmd.extend(["-c:a", "copy"])

    # Output path
    cmd.append(str(output_path))

    return cmd


def _run_ffmpeg_with_progress(
    cmd: list[str],
    total_frames: int,
    description: str = "Reencoding",
) -> None:
    """Run ffmpeg command with progress bar.

    Args:
        cmd: FFmpeg command as list of arguments.
        total_frames: Total number of frames for progress tracking.
        description: Description for progress bar.

    Raises:
        click.ClickException: If ffmpeg fails.
    """
    import re
    import subprocess

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    # Add progress flag to ffmpeg
    cmd_with_progress = cmd.copy()
    # Insert progress flags after the ffmpeg executable path
    # Note: -stats_period not supported in ffmpeg 4.x, use -progress alone
    progress_flags = ["-progress", "pipe:1"]
    cmd_with_progress[1:1] = progress_flags

    process = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=total_frames)

        # Parse ffmpeg progress output
        frame_pattern = re.compile(r"frame=(\d+)")

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            match = frame_pattern.search(line)
            if match:
                current_frame = int(match.group(1))
                progress.update(task, completed=current_frame)

        progress.update(task, completed=total_frames)

    # Check for errors
    returncode = process.wait()
    if returncode != 0:
        stderr = process.stderr.read()
        raise click.ClickException(f"ffmpeg failed (exit code {returncode}):\n{stderr}")


def _reencode_python_path(
    input_path: Path,
    output_path: Path,
    output_fps: float | None,
    crf: int,
    preset: str,
    keyframe_interval: float,
) -> None:
    """Reencode video using Python frame-by-frame path.

    Args:
        input_path: Path to input video file.
        output_path: Path to output video file.
        output_fps: Output frame rate (None to preserve input).
        crf: Constant rate factor.
        preset: x264 encoding preset.
        keyframe_interval: Keyframe interval in seconds.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from sleap_io.io.video_writing import VideoWriter

    # Load video
    video = io_main.load_video(str(input_path))

    # For Video objects, get fps from backend if available
    if hasattr(video, "backend") and hasattr(video.backend, "fps"):
        input_fps = video.backend.fps
    else:
        input_fps = 30.0  # Default fallback

    effective_fps = output_fps if output_fps is not None else input_fps

    # Calculate GOP
    gop_size = max(1, int(keyframe_interval * effective_fps))

    # Build output params for keyframe control
    output_params = [
        "-g",
        str(gop_size),
        "-keyint_min",
        str(gop_size),
        "-bf",
        "0",
        "-x264opts",
        "scenecut=0",
    ]

    # Create writer
    writer = VideoWriter(
        filename=output_path,
        fps=effective_fps,
        crf=crf,
        preset=preset,
        output_params=output_params,
    )

    # Write frames with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reencoding (Python)", total=len(video))

        with writer:
            for i in range(len(video)):
                frame = video[i]
                writer.write_frame(frame)
                progress.update(task, advance=1)


def _reencode_video_object(
    video: "Video",
    output_path: Path,
    crf: int,
    preset: str,
    keyframe_interval: float,
    output_fps: float | None = None,
    use_ffmpeg: bool | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Reencode a Video object to a new file.

    This function handles both MediaVideo and HDF5Video backends, choosing the
    appropriate encoding path (ffmpeg or Python) based on the backend type and
    availability.

    Args:
        video: The Video object to reencode.
        output_path: Path to the output video file.
        crf: Constant rate factor (0-51, lower is better).
        preset: x264 encoding preset.
        keyframe_interval: Keyframe interval in seconds.
        output_fps: Output frame rate. If None, preserves source FPS.
        use_ffmpeg: Force ffmpeg path (True), Python path (False), or auto (None).
        overwrite: Whether to overwrite existing output file.
        dry_run: If True, show what would be done without executing.

    Returns:
        Path to the reencoded video file, or None if dry_run.

    Raises:
        click.ClickException: If reencoding fails or video type is unsupported.
    """
    from sleap_io.io.video_reading import HDF5Video, ImageVideo, MediaVideo, TiffVideo

    backend = video.backend
    backend_type = type(backend).__name__

    # Check backend type
    if isinstance(backend, (ImageVideo, TiffVideo)):
        console.print(
            f"[yellow]Skipping {video.filename}: {backend_type} cannot be reencoded[/]"
        )
        return None

    # Check if output already exists
    if output_path.exists() and not overwrite:
        raise click.ClickException(
            f"Output file already exists: {output_path}\nUse --overwrite to replace it."
        )

    # Determine which path to use
    ffmpeg_available = _is_ffmpeg_available()

    # HDF5Video with embedded images must use Python path
    if isinstance(backend, HDF5Video) and backend.has_embedded_images:
        can_use_ffmpeg = False
    elif isinstance(backend, MediaVideo):
        can_use_ffmpeg = ffmpeg_available
    elif isinstance(backend, HDF5Video):
        # HDF5 without embedded images (rank-4 array) - use Python path
        can_use_ffmpeg = False
    else:
        can_use_ffmpeg = False

    # Resolve final path choice
    if use_ffmpeg is True and not can_use_ffmpeg:
        if isinstance(backend, HDF5Video):
            raise click.ClickException(
                f"Cannot use ffmpeg for HDF5 video: {video.filename}\n"
                "HDF5 videos require the Python path (--no-ffmpeg)."
            )
        raise click.ClickException(
            f"ffmpeg not available for {backend_type}: {video.filename}"
        )

    use_ffmpeg_path = can_use_ffmpeg if use_ffmpeg is None else use_ffmpeg

    if use_ffmpeg_path:
        # FFmpeg fast path (for MediaVideo)
        input_path = Path(video.filename)

        try:
            metadata = _get_video_metadata(input_path)
            input_fps = metadata["fps"]
            total_frames = metadata["num_frames"]

            # Resolve keyframe interval
            effective_fps = output_fps if output_fps is not None else input_fps
            resolved_keyframe_interval = keyframe_interval

            # Build command
            cmd = _build_ffmpeg_reencode_command(
                input_path=input_path,
                output_path=output_path,
                fps=input_fps,
                output_fps=output_fps,
                crf=crf,
                preset=preset,
                keyframe_interval=resolved_keyframe_interval,
                overwrite=overwrite,
            )

            if dry_run:
                console.print(f"[dim]  Would run: {' '.join(cmd)}[/]")
                return None

            # Print info
            console.print(f"[bold]Reencoding:[/] {input_path}")
            console.print(f"[dim]  Output: {output_path}[/]")
            console.print(
                f"[dim]  Quality: CRF {crf}, "
                f"Preset: {preset}, "
                f"Keyframes: {resolved_keyframe_interval}s[/]"
            )
            if output_fps is not None:
                console.print(f"[dim]  FPS: {input_fps} -> {output_fps}[/]")

            # Run ffmpeg with progress
            _run_ffmpeg_with_progress(cmd, total_frames)

        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(f"ffmpeg reencoding failed: {e}")

    else:
        # Python path (for HDF5Video or when ffmpeg unavailable)
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        from sleap_io.io.video_writing import VideoWriter

        if dry_run:
            console.print(f"[dim]  Would reencode via Python: {video.filename}[/]")
            console.print(f"[dim]  Output: {output_path}[/]")
            return None

        console.print(f"[bold]Reencoding (Python):[/] {video.filename}")
        console.print(f"[dim]  Output: {output_path}[/]")

        # Get input FPS
        input_fps = video.fps if video.fps is not None else 30.0
        effective_fps = output_fps if output_fps is not None else input_fps

        # Calculate GOP
        gop_size = max(1, int(keyframe_interval * effective_fps))

        # Build output params for keyframe control
        output_params = [
            "-g",
            str(gop_size),
            "-keyint_min",
            str(gop_size),
            "-bf",
            "0",
            "-x264opts",
            "scenecut=0",
        ]

        # Create writer
        writer = VideoWriter(
            filename=output_path,
            fps=effective_fps,
            crf=crf,
            preset=preset,
            output_params=output_params,
        )

        # Write frames with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            description = f"Reencoding {Path(video.filename).name}"
            task = progress.add_task(description, total=len(video))

            with writer:
                for i in range(len(video)):
                    frame = video[i]
                    writer.write_frame(frame)
                    progress.update(task, advance=1)

    # Show file size comparison
    if not dry_run:
        input_size = Path(video.filename).stat().st_size
        output_size = output_path.stat().st_size
        size_change = (output_size - input_size) / input_size * 100

        console.print(f"[bold green]Saved:[/] {output_path}")
        console.print(
            f"[dim]  Size: {_format_file_size(input_size)} -> "
            f"{_format_file_size(output_size)} "
            f"({size_change:+.1f}%)[/]"
        )

    return output_path


def _reencode_slp(
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    keyframe_interval: float,
    output_fps: float | None = None,
    use_ffmpeg: bool | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Reencode all videos in an SLP file and create a new SLP with updated paths.

    This function loads an SLP file, reencodes each video to improve seekability,
    and saves a new SLP file with updated video paths.

    Args:
        input_path: Path to input SLP file.
        output_path: Path to output SLP file.
        crf: Constant rate factor (0-51, lower is better).
        preset: x264 encoding preset.
        keyframe_interval: Keyframe interval in seconds.
        output_fps: Output frame rate. If None, preserves source FPS.
        use_ffmpeg: Force ffmpeg path (True), Python path (False), or auto (None).
        overwrite: Whether to overwrite existing files.
        dry_run: If True, show what would be done without executing.

    Raises:
        click.ClickException: If processing fails.
    """
    from sleap_io.io.video_reading import ImageVideo, TiffVideo
    from sleap_io.model.video import Video

    # Check output SLP exists
    if output_path.exists() and not overwrite:
        raise click.ClickException(
            f"Output SLP already exists: {output_path}\nUse --overwrite to replace it."
        )

    # Load SLP file
    console.print(f"[bold]Loading SLP:[/] {input_path}")
    labels = io_main.load_slp(str(input_path))

    if not labels.videos:
        console.print("[yellow]No videos found in SLP file.[/]")
        return

    console.print(f"[dim]  Found {len(labels.videos)} video(s)[/]")

    # Create video output directory
    video_dir = output_path.with_name(output_path.stem + ".videos")
    if not dry_run:
        video_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]  Video output directory: {video_dir}[/]")

    # Track video replacements
    video_map: dict[Video, Video] = {}
    skipped_count = 0
    reencoded_count = 0

    # Process each video
    for i, video in enumerate(labels.videos):
        console.print(f"\n[bold]Video {i + 1}/{len(labels.videos)}:[/]")

        backend = video.backend
        backend_type = type(backend).__name__ if backend is not None else "Unknown"

        # Skip unsupported backends
        if isinstance(backend, (ImageVideo, TiffVideo)):
            console.print(
                f"[yellow]  Skipping: {video.filename}[/]\n"
                f"[dim]  Reason: {backend_type} cannot be reencoded[/]"
            )
            skipped_count += 1
            continue

        if backend is None:
            console.print(
                f"[yellow]  Skipping: {video.filename}[/]\n"
                "[dim]  Reason: Video backend not loaded[/]"
            )
            skipped_count += 1
            continue

        # Determine output video path
        original_name = Path(video.filename).stem
        output_video_path = video_dir / f"{original_name}.reencoded.mp4"

        # Check if output video exists
        if output_video_path.exists() and not overwrite:
            console.print(
                f"[yellow]  Skipping: {video.filename}[/]\n"
                f"[dim]  Reason: Output exists: {output_video_path}[/]"
            )
            skipped_count += 1
            continue

        try:
            # Reencode the video
            result_path = _reencode_video_object(
                video=video,
                output_path=output_video_path,
                crf=crf,
                preset=preset,
                keyframe_interval=keyframe_interval,
                output_fps=output_fps,
                use_ffmpeg=use_ffmpeg,
                overwrite=overwrite,
                dry_run=dry_run,
            )

            if result_path is not None:
                # Create new Video object with updated path
                new_video = Video.from_filename(
                    result_path.as_posix(), grayscale=video.grayscale
                )
                # Preserve source_video reference for provenance
                new_video.source_video = video
                video_map[video] = new_video
                reencoded_count += 1
            elif dry_run:
                reencoded_count += 1  # Count for dry run reporting
            else:
                skipped_count += 1

        except click.ClickException:
            raise
        except Exception as e:
            console.print(f"[red]  Error reencoding {video.filename}: {e}[/]")
            skipped_count += 1
            continue

    # Summary
    console.print("\n[bold]Summary:[/]")
    console.print(f"  Reencoded: {reencoded_count}")
    console.print(f"  Skipped: {skipped_count}")

    if dry_run:
        console.print(f"\n[bold]Dry run - would save SLP to:[/] {output_path}")
        return

    # Update video references in labels
    if video_map:
        labels.replace_videos(video_map=video_map)

    # Save the new SLP file
    console.print(f"\n[bold]Saving SLP:[/] {output_path}")
    labels.save(str(output_path))
    console.print(f"[bold green]Saved:[/] {output_path}")


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input video or SLP file. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output video path. Default: {input}.reencoded.mp4.",
)
# Quality options (mutually exclusive group)
@click.option(
    "--quality",
    type=click.Choice(["lossless", "high", "medium", "low"]),
    default=None,
    help="Quality level. Maps to CRF: lossless=0, high=18, medium=25, low=32.",
)
@click.option(
    "--crf",
    type=int,
    default=None,
    help="Constant rate factor (0-51, lower=better). Overrides --quality.",
)
# Keyframe options
@click.option(
    "--keyframe-interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Keyframe interval in seconds. Lower = better seekability, larger files.",
)
@click.option(
    "--gop",
    type=int,
    default=None,
    help="GOP size in frames. Overrides --keyframe-interval.",
)
# Frame rate
@click.option(
    "--fps",
    "output_fps",
    type=float,
    default=None,
    help="Output frame rate. Default: preserve source FPS.",
)
# Encoding speed
@click.option(
    "--encoding",
    "preset",
    type=click.Choice(
        [
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ]
    ),
    default="superfast",
    show_default=True,
    help="Encoding speed preset. Slower = better compression.",
)
# Execution control
@click.option(
    "--use-ffmpeg/--no-ffmpeg",
    "use_ffmpeg",
    default=None,
    help="Force ffmpeg fast path or Python path.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show ffmpeg command without executing.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output file.",
)
def reencode(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path | None,
    quality: str | None,
    crf: int | None,
    keyframe_interval: float,
    gop: int | None,
    output_fps: float | None,
    preset: str,
    use_ffmpeg: bool | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Reencode video for improved seekability.

    Creates a new video file with frequent keyframes for fast random access.
    This is useful for videos that seek slowly during annotation or playback.

    By default, uses ffmpeg directly for fast reencoding. Falls back to Python
    frame-by-frame path when ffmpeg is unavailable or --no-ffmpeg is specified.

    [bold]SLP batch mode:[/]
    When given an SLP file as input, reencodes all videos in the project:
    - Creates a video directory next to the output SLP
    - Reencodes each MediaVideo using ffmpeg (fast)
    - Reencodes HDF5 embedded videos using Python path
    - Skips image sequences and TIFF stacks (cannot be reencoded)
    - Saves a new SLP file with updated video paths

    [bold]Quality levels:[/]
    lossless: CRF 0 (mathematically identical, huge files)
    high: CRF 18 (visually lossless)
    medium: CRF 25 (good quality, reasonable size) [default]
    low: CRF 32 (smaller files, some quality loss)

    [dim]Examples:[/]

        $ sio reencode video.mp4 -o video.seekable.mp4

        $ sio reencode video.mp4 -o output.mp4 --quality high

        $ sio reencode video.mp4 -o output.mp4 --keyframe-interval 0.5

        $ sio reencode highspeed.mp4 -o preview.mp4 --fps 30 --quality low

        $ sio reencode video.mp4 -o output.mp4 --dry-run

        $ sio reencode video.mp4 -o output.mp4 --no-ffmpeg

    [dim]SLP batch examples:[/]

        $ sio reencode project.slp -o project.reencoded.slp

        $ sio reencode project.slp -o project.reencoded.slp --quality high

        $ sio reencode project.slp -o output.slp --dry-run
    """
    # Resolve input path
    input_path = _resolve_input(input_arg, input_opt, "input file")

    # Check if input is an SLP file - route to batch processing
    if input_path.suffix.lower() == ".slp":
        # Resolve output path for SLP
        if output_path is None:
            output_path = input_path.with_name(input_path.stem + ".reencoded.slp")

        # Ensure output has .slp extension
        if output_path.suffix.lower() != ".slp":
            output_path = output_path.with_suffix(".slp")

        # Check for same file
        if output_path.resolve() == input_path.resolve():
            raise click.ClickException(
                f"Output path cannot be the same as input: {input_path}\n"
                "Use a different output path or rename the output file."
            )

        # Resolve quality/CRF for SLP batch
        if crf is not None and quality is not None:
            raise click.ClickException(
                "Cannot use both --quality and --crf. Choose one."
            )

        if crf is not None:
            resolved_crf = crf
        elif quality is not None:
            resolved_crf = _QUALITY_TO_CRF[quality]
        else:
            resolved_crf = _QUALITY_TO_CRF["medium"]

        # Validate CRF
        if not 0 <= resolved_crf <= 51:
            raise click.ClickException(
                f"CRF must be between 0 and 51, got {resolved_crf}"
            )

        # Call SLP batch processing
        _reencode_slp(
            input_path=input_path,
            output_path=output_path,
            crf=resolved_crf,
            preset=preset,
            keyframe_interval=keyframe_interval,
            output_fps=output_fps,
            use_ffmpeg=use_ffmpeg,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        return

    # Resolve output path for video files
    if output_path is None:
        output_path = input_path.with_stem(f"{input_path.stem}.reencoded")
        if output_path.suffix.lower() not in {".mp4", ".mkv", ".avi", ".mov"}:
            output_path = output_path.with_suffix(".mp4")

    # Check for same file
    if output_path.resolve() == input_path.resolve():
        raise click.ClickException(
            f"Output path cannot be the same as input: {input_path}\n"
            "Use a different output path or rename the output file."
        )

    # Check output exists
    if output_path.exists() and not overwrite:
        raise click.ClickException(
            f"Output file already exists: {output_path}\nUse --overwrite to replace it."
        )

    # Resolve quality/CRF
    if crf is not None and quality is not None:
        raise click.ClickException("Cannot use both --quality and --crf. Choose one.")

    if crf is not None:
        resolved_crf = crf
    elif quality is not None:
        resolved_crf = _QUALITY_TO_CRF[quality]
    else:
        resolved_crf = _QUALITY_TO_CRF["medium"]  # Default

    # Validate CRF
    if not 0 <= resolved_crf <= 51:
        raise click.ClickException(f"CRF must be between 0 and 51, got {resolved_crf}")

    # Check ffmpeg availability
    ffmpeg_available = _is_ffmpeg_available()

    if use_ffmpeg is True and not ffmpeg_available:
        raise click.ClickException(
            "ffmpeg not found but --use-ffmpeg was specified. "
            "Please install ffmpeg or use --no-ffmpeg."
        )

    # Decide path to use
    use_ffmpeg_path = ffmpeg_available if use_ffmpeg is None else use_ffmpeg

    if use_ffmpeg_path:
        # FFmpeg fast path
        try:
            metadata = _get_video_metadata(input_path)
            input_fps = metadata["fps"]
            total_frames = metadata["num_frames"]

            # Resolve keyframe interval / GOP
            if gop is not None:
                # Convert GOP to interval for display
                effective_fps = output_fps if output_fps is not None else input_fps
                resolved_keyframe_interval = gop / effective_fps
            else:
                resolved_keyframe_interval = keyframe_interval

            # Build command
            cmd = _build_ffmpeg_reencode_command(
                input_path=input_path,
                output_path=output_path,
                fps=input_fps,
                output_fps=output_fps,
                crf=resolved_crf,
                preset=preset,
                keyframe_interval=resolved_keyframe_interval,
                overwrite=overwrite,
            )

            if dry_run:
                console.print("[bold]Dry run - ffmpeg command:[/]")
                console.print(" ".join(cmd))
                return

            # Print info
            console.print(f"[bold]Reencoding:[/] {input_path}")
            console.print(f"[dim]  Output: {output_path}[/]")
            console.print(
                f"[dim]  Quality: CRF {resolved_crf}, "
                f"Preset: {preset}, "
                f"Keyframes: {resolved_keyframe_interval}s[/]"
            )
            if output_fps is not None:
                console.print(f"[dim]  FPS: {input_fps} -> {output_fps}[/]")

            # Run ffmpeg with progress
            _run_ffmpeg_with_progress(cmd, total_frames)

        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(f"ffmpeg reencoding failed: {e}")

    else:
        # Python fallback path
        if dry_run:
            console.print("[bold]Dry run - would use Python path[/]")
            console.print(f"  Input: {input_path}")
            console.print(f"  Output: {output_path}")
            console.print(f"  CRF: {resolved_crf}, Preset: {preset}")
            return

        console.print(f"[bold]Reencoding (Python path):[/] {input_path}")
        console.print(f"[dim]  Output: {output_path}[/]")

        _reencode_python_path(
            input_path=input_path,
            output_path=output_path,
            output_fps=output_fps,
            crf=resolved_crf,
            preset=preset,
            keyframe_interval=keyframe_interval,
        )

    console.print(f"[bold green]Saved:[/] {output_path}")

    # Show file size comparison
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    size_change = (output_size - input_size) / input_size * 100

    console.print(
        f"[dim]  Size: {_format_file_size(input_size)} -> "
        f"{_format_file_size(output_size)} "
        f"({size_change:+.1f}%)[/]"
    )


def _parse_transform_param(
    value: str,
) -> tuple[int | None, str]:
    """Parse transform parameter with optional video index prefix.

    Args:
        value: Parameter value, optionally prefixed with "idx:" for per-video.

    Returns:
        Tuple of (video_index or None, parameter_value).
    """
    if ":" in value:
        idx_str, param = value.split(":", 1)
        return int(idx_str), param
    return None, value


def _parse_fill_value(value: str) -> tuple[int, ...] | int:
    """Parse fill value from string.

    Args:
        value: Fill value as string. Either a single integer (e.g., "0", "128")
            or comma-separated RGB values (e.g., "128,128,128").

    Returns:
        Single int for grayscale or tuple of ints for RGB.

    Raises:
        click.ClickException: If value cannot be parsed.
    """
    value = value.strip()
    if "," in value:
        try:
            parts = [int(p.strip()) for p in value.split(",")]
            if len(parts) != 3:
                raise click.ClickException(
                    f"RGB fill must have 3 values, got {len(parts)}: {value}"
                )
            for p in parts:
                if not 0 <= p <= 255:
                    raise click.ClickException(f"Fill values must be 0-255, got: {p}")
            return tuple(parts)
        except ValueError:
            raise click.ClickException(f"Invalid fill value: {value}")
    else:
        try:
            val = int(value)
            if not 0 <= val <= 255:
                raise click.ClickException(f"Fill value must be 0-255, got: {val}")
            return val
        except ValueError:
            raise click.ClickException(f"Invalid fill value: {value}")


def _build_transforms_from_params(
    n_videos: int,
    crop_params: tuple[str, ...],
    scale_params: tuple[str, ...],
    rotate_params: tuple[str, ...],
    pad_params: tuple[str, ...],
    input_sizes: list[tuple[int, int]],
    quality: str,
    fill: tuple[int, ...] | int,
    clip_rotation: bool = False,
    flip_h: bool = False,
    flip_v: bool = False,
) -> dict[int, Any]:
    """Build Transform objects from CLI parameters.

    Args:
        n_videos: Number of videos in the labels file.
        crop_params: Crop parameter strings (may have idx: prefix).
        scale_params: Scale parameter strings (may have idx: prefix).
        rotate_params: Rotation parameter strings (may have idx: prefix).
        pad_params: Padding parameter strings (may have idx: prefix).
        input_sizes: List of (width, height) for each video.
        quality: Interpolation quality.
        fill: Fill value for out-of-bounds regions (int or RGB tuple).
        clip_rotation: If True, rotation clips to original dimensions instead of
            expanding canvas to fit rotated image.
        flip_h: If True, flip horizontally.
        flip_v: If True, flip vertically.

    Returns:
        Dictionary mapping video index to Transform.
    """
    from sleap_io.transform import Transform
    from sleap_io.transform.core import (
        parse_crop,
        parse_pad,
        parse_scale,
        resolve_scale,
    )

    # Initialize per-video transform components
    video_crops: dict[int, tuple[int, int, int, int] | None] = {
        i: None for i in range(n_videos)
    }
    video_scales: dict[int, tuple[float, float] | None] = {
        i: None for i in range(n_videos)
    }
    video_rotates: dict[int, float | None] = {i: None for i in range(n_videos)}
    video_pads: dict[int, tuple[int, int, int, int] | None] = {
        i: None for i in range(n_videos)
    }

    # Parse crop parameters
    for crop_str in crop_params:
        idx, value = _parse_transform_param(crop_str)
        if idx is None:
            # Apply to all videos
            for i in range(n_videos):
                video_crops[i] = parse_crop(value, input_sizes[i])
        else:
            if idx >= n_videos:
                raise click.ClickException(
                    f"Video index {idx} out of range (have {n_videos} videos)"
                )
            video_crops[idx] = parse_crop(value, input_sizes[idx])

    # Parse scale parameters
    for scale_str in scale_params:
        idx, value = _parse_transform_param(scale_str)
        parsed = parse_scale(value)
        if idx is None:
            for i in range(n_videos):
                video_scales[i] = resolve_scale(parsed, input_sizes[i])
        else:
            if idx >= n_videos:
                raise click.ClickException(
                    f"Video index {idx} out of range (have {n_videos} videos)"
                )
            video_scales[idx] = resolve_scale(parsed, input_sizes[idx])

    # Parse rotation parameters
    for rotate_str in rotate_params:
        idx, value = _parse_transform_param(rotate_str)
        angle = float(value)
        if idx is None:
            for i in range(n_videos):
                video_rotates[i] = angle
        else:
            if idx >= n_videos:
                raise click.ClickException(
                    f"Video index {idx} out of range (have {n_videos} videos)"
                )
            video_rotates[idx] = angle

    # Parse padding parameters
    for pad_str in pad_params:
        idx, value = _parse_transform_param(pad_str)
        parsed = parse_pad(value)
        if idx is None:
            for i in range(n_videos):
                video_pads[i] = parsed
        else:
            if idx >= n_videos:
                raise click.ClickException(
                    f"Video index {idx} out of range (have {n_videos} videos)"
                )
            video_pads[idx] = parsed

    # Build Transform objects
    transforms = {}
    for i in range(n_videos):
        transform = Transform(
            crop=video_crops[i],
            scale=video_scales[i],
            rotate=video_rotates[i],
            pad=video_pads[i],
            quality=quality,
            fill=fill,
            clip_rotation=clip_rotation,
            flip_h=flip_h,
            flip_v=flip_v,
        )
        if transform:  # Only include non-empty transforms
            transforms[i] = transform

    return transforms


def _load_config_file(
    config_path: Path,
    input_sizes: list[tuple[int, int]],
    quality: str,
    fill: tuple[int, ...] | int,
    clip_rotation: bool = False,
    flip_h: bool = False,
    flip_v: bool = False,
) -> dict[int, Any]:
    """Load transforms from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.
        input_sizes: List of (width, height) for each video.
        quality: Interpolation quality.
        fill: Fill value for out-of-bounds regions.
        clip_rotation: If True, clip rotation to original dimensions.
        flip_h: If True, flip horizontally.
        flip_v: If True, flip vertically.

    Returns:
        Dictionary mapping video index to Transform.

    Raises:
        click.ClickException: If config file is invalid.

    Config file format:
        ```yaml
        videos:
          0:
            crop: [100, 100, 500, 500]
            scale: 0.5
            rotate: 0
            pad: [0, 0, 0, 0]
          1:
            crop: [200, 200, 600, 600]
            scale: [640, -1]
            rotate: 90
        ```
    """
    import yaml

    from sleap_io.transform import Transform
    from sleap_io.transform.core import (
        parse_crop,
        parse_pad,
        parse_scale,
        resolve_scale,
    )

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise click.ClickException(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise click.ClickException(f"Invalid YAML in config file: {e}")

    if not isinstance(config, dict) or "videos" not in config:
        raise click.ClickException(
            "Config file must have a 'videos' key with video index mappings."
        )

    videos_config = config["videos"]
    if not isinstance(videos_config, dict):
        raise click.ClickException("'videos' must be a mapping of video indices.")

    transforms = {}
    n_videos = len(input_sizes)

    for video_idx_raw, video_config in videos_config.items():
        # Parse video index
        try:
            video_idx = int(video_idx_raw)
        except (ValueError, TypeError):
            raise click.ClickException(
                f"Invalid video index in config: {video_idx_raw}"
            )

        if video_idx >= n_videos:
            raise click.ClickException(
                f"Video index {video_idx} out of range (have {n_videos} videos)"
            )

        if video_config is None:
            # No transforms for this video
            continue

        if not isinstance(video_config, dict):
            raise click.ClickException(
                f"Config for video {video_idx} must be a mapping, "
                f"got: {type(video_config).__name__}"
            )

        input_size = input_sizes[video_idx]

        # Parse crop
        crop = None
        if "crop" in video_config:
            crop_val = video_config["crop"]
            if isinstance(crop_val, (list, tuple)) and len(crop_val) == 4:
                # Check if normalized (all floats in 0-1)
                if all(isinstance(v, float) and 0 <= v <= 1 for v in crop_val):
                    w, h = input_size
                    crop = (
                        int(round(crop_val[0] * w)),
                        int(round(crop_val[1] * h)),
                        int(round(crop_val[2] * w)),
                        int(round(crop_val[3] * h)),
                    )
                else:
                    crop = tuple(int(v) for v in crop_val)  # type: ignore
            elif isinstance(crop_val, str):
                crop = parse_crop(crop_val, input_size)
            else:
                raise click.ClickException(
                    f"Invalid crop value for video {video_idx}: {crop_val}"
                )

        # Parse scale
        scale = None
        if "scale" in video_config:
            scale_val = video_config["scale"]
            if isinstance(scale_val, (int, float)):
                # Single value - uniform scale
                if scale_val < 1 and scale_val > 0:
                    scale = (scale_val, scale_val)
                else:
                    # Pixel width
                    parsed = (-float(scale_val), -1.0)
                    scale = resolve_scale(parsed, input_size)
            elif isinstance(scale_val, (list, tuple)) and len(scale_val) == 2:
                # Check if ratios or pixels
                val1, val2 = scale_val
                if all(isinstance(v, float) and 0 < v < 1 for v in scale_val if v > 0):
                    scale = (val1 if val1 > 0 else -1.0, val2 if val2 > 0 else -1.0)
                    if scale[0] == -1.0 or scale[1] == -1.0:
                        scale = resolve_scale(scale, input_size)
                else:
                    # Pixels
                    parsed = (
                        -float(val1) if val1 > 0 else val1,
                        -float(val2) if val2 > 0 else val2,
                    )
                    scale = resolve_scale(parsed, input_size)
            elif isinstance(scale_val, str):
                parsed = parse_scale(scale_val)
                scale = resolve_scale(parsed, input_size)
            else:
                raise click.ClickException(
                    f"Invalid scale value for video {video_idx}: {scale_val}"
                )

        # Parse rotate
        rotate = None
        if "rotate" in video_config:
            rotate_val = video_config["rotate"]
            try:
                rotate = float(rotate_val)
            except (ValueError, TypeError):
                raise click.ClickException(
                    f"Invalid rotate value for video {video_idx}: {rotate_val}"
                )

        # Parse pad
        pad = None
        if "pad" in video_config:
            pad_val = video_config["pad"]
            if isinstance(pad_val, (list, tuple)):
                if len(pad_val) == 4:
                    pad = tuple(int(v) for v in pad_val)  # type: ignore
                elif len(pad_val) == 1:
                    p = int(pad_val[0])
                    pad = (p, p, p, p)
                else:
                    raise click.ClickException(
                        f"Pad must have 1 or 4 values for video {video_idx}"
                    )
            elif isinstance(pad_val, int):
                pad = (pad_val, pad_val, pad_val, pad_val)
            elif isinstance(pad_val, str):
                pad = parse_pad(pad_val)
            else:
                raise click.ClickException(
                    f"Invalid pad value for video {video_idx}: {pad_val}"
                )

        # Parse flip options (per-video override)
        video_flip_h = video_config.get("flip_horizontal", flip_h)
        video_flip_v = video_config.get("flip_vertical", flip_v)

        # Parse clip_rotation (per-video override)
        video_clip_rotation = video_config.get("clip_rotation", clip_rotation)

        # Build transform
        transform = Transform(
            crop=crop,
            scale=scale,
            rotate=rotate,
            pad=pad,
            quality=quality,
            fill=fill,
            clip_rotation=video_clip_rotation,
            flip_h=video_flip_h,
            flip_v=video_flip_v,
        )

        if transform:
            transforms[video_idx] = transform

    return transforms


def _generate_transform_metadata(
    labels: "Labels",
    transforms: dict[int, Any],
    source_path: Path,
    output_path: Path,
) -> dict:
    """Generate transform metadata for output.

    Args:
        labels: Source Labels object.
        transforms: Dictionary mapping video index to Transform.
        source_path: Path to source file.
        output_path: Path to output file.

    Returns:
        Dictionary with transform metadata.
    """
    from datetime import datetime, timezone

    from sleap_io.version import __version__

    metadata = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "output": str(output_path),
        "sleap_io_version": __version__,
        "videos": {},
    }

    for video_idx, video in enumerate(labels.videos):
        if video_idx not in transforms:
            continue

        transform = transforms[video_idx]

        # Get video dimensions
        if hasattr(video, "shape") and video.shape is not None:
            h, w = video.shape[1:3]
            input_size = (w, h)
        else:
            input_size = None

        video_meta = {
            "input": video.filename,
            "input_size": list(input_size) if input_size else None,
        }

        if input_size:
            output_size = transform.output_size(input_size)
            video_meta["output_size"] = list(output_size)

            # Include affine transformation matrix
            matrix = transform.to_matrix(input_size)
            video_meta["coordinate_transform"] = {
                "matrix": matrix.tolist(),
            }

        # Include transform parameters
        video_meta["transforms"] = {
            "crop": list(transform.crop) if transform.crop else None,
            "scale": list(transform.scale) if transform.scale else None,
            "rotate": transform.rotate,
            "pad": list(transform.pad) if transform.pad else None,
            "flip_horizontal": transform.flip_h,
            "flip_vertical": transform.flip_v,
        }

        metadata["videos"][video_idx] = video_meta

    return metadata


def _export_transform_metadata(metadata: dict, output_path: Path) -> None:
    """Export transform metadata to YAML file.

    Args:
        metadata: Transform metadata dictionary.
        output_path: Path to output YAML file.
    """
    import yaml

    # Custom representer to handle numpy arrays and other types
    def represent_data(dumper: yaml.Dumper, data: Any) -> Any:
        if hasattr(data, "tolist"):
            return dumper.represent_list(data.tolist())
        return dumper.represent_data(data)

    with open(output_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


@cli.command()
@click.argument(
    "input_arg",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-i",
    "--input",
    "input_opt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input SLP file or video. Can also be passed as positional argument.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output path. Default: {input}.transformed.slp or .mp4.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file with per-video transforms.",
)
@click.option(
    "--output-transforms",
    "output_transforms_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Export transform metadata to YAML file.",
)
@click.option(
    "--embed-provenance/--no-embed-provenance",
    is_flag=True,
    default=True,
    help="Store transform metadata in output SLP file (default: enabled).",
)
# Transform options (can be repeated with idx: prefix for per-video)
@click.option(
    "--crop",
    "crop_params",
    multiple=True,
    help="Crop region: '[idx:]x1,y1,x2,y2'. Pixels or normalized (0.0-1.0).",
)
@click.option(
    "--scale",
    "scale_params",
    multiple=True,
    help="Scale: '[idx:]factor' or '[idx:]w,h'. E.g., 0.5, 640, 640,480.",
)
@click.option(
    "--rotate",
    "rotate_params",
    multiple=True,
    help="Rotation: '[idx:]degrees'. Clockwise positive.",
)
@click.option(
    "--clip-rotation",
    is_flag=True,
    default=False,
    help="Keep original dimensions when rotating (clips corners).",
)
@click.option(
    "--pad",
    "pad_params",
    multiple=True,
    help="Padding: '[idx:]top,right,bottom,left' or single value for uniform.",
)
@click.option(
    "--flip-horizontal",
    "flip_h",
    is_flag=True,
    default=False,
    help="Flip horizontally (mirror left-right).",
)
@click.option(
    "--flip-vertical",
    "flip_v",
    is_flag=True,
    default=False,
    help="Flip vertically (mirror top-bottom).",
)
# Quality options
@click.option(
    "--quality",
    type=click.Choice(["nearest", "bilinear", "bicubic"]),
    default="bilinear",
    show_default=True,
    help="Interpolation quality for transforms.",
)
@click.option(
    "--fill",
    type=str,
    default="0",
    show_default=True,
    help="Fill value for out-of-bounds regions. Int (0-255) or R,G,B.",
)
# Encoding options
@click.option(
    "--crf",
    type=int,
    default=25,
    show_default=True,
    help="Video quality (0-51, lower=better).",
)
@click.option(
    "--x264-preset",
    "x264_preset",
    type=click.Choice(
        ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]
    ),
    default="superfast",
    show_default=True,
    help="H.264 encoding speed/compression trade-off.",
)
@click.option(
    "--fps",
    "output_fps",
    type=float,
    default=None,
    help="Output frame rate. Default: preserve source FPS.",
)
@click.option(
    "--keyframe-interval",
    type=float,
    default=None,
    help="Keyframe interval in seconds. Lower = better seeking, larger files.",
)
@click.option(
    "--no-audio",
    is_flag=True,
    default=False,
    help="Strip audio from output videos.",
)
# Execution control
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview transforms without executing.",
)
@click.option(
    "--dry-run-frame",
    type=int,
    default=None,
    help="Frame index to render in dry-run preview. Implies --dry-run.",
)
@click.option(
    "--overwrite",
    "-y",
    is_flag=True,
    default=False,
    help="Overwrite existing output files.",
)
def transform(
    input_arg: Path | None,
    input_opt: Path | None,
    output_path: Path | None,
    config_path: Path | None,
    output_transforms_path: Path | None,
    embed_provenance: bool,
    crop_params: tuple[str, ...],
    scale_params: tuple[str, ...],
    rotate_params: tuple[str, ...],
    clip_rotation: bool,
    pad_params: tuple[str, ...],
    flip_h: bool,
    flip_v: bool,
    quality: str,
    fill: str,
    crf: int,
    x264_preset: str,
    output_fps: float | None,
    keyframe_interval: float | None,
    no_audio: bool,
    dry_run: bool,
    dry_run_frame: int | None,
    overwrite: bool,
) -> None:
    """Transform video and adjust label coordinates.

    Apply geometric transformations (crop, scale, rotate, pad, flip) to videos while
    automatically adjusting all landmark coordinates to maintain alignment.

    Transforms are applied in order: crop -> scale -> rotate -> pad -> flip.

    [bold]Per-video transforms:[/]
    Prefix any parameter with 'idx:' to apply to a specific video:
      --crop 0:100,100,500,500 --crop 1:200,200,600,600

    [dim]Examples:[/]

        [bold]Scale down 50%:[/]
        $ sio transform labels.slp --scale 0.5 -o scaled.slp

        [bold]Crop to region:[/]
        $ sio transform labels.slp --crop 100,100,500,500 -o cropped.slp

        [bold]Rotate 90 degrees:[/]
        $ sio transform labels.slp --rotate 90 -o rotated.slp

        [bold]Add padding:[/]
        $ sio transform labels.slp --pad 50,50,50,50 -o padded.slp

        [bold]Combined transforms:[/]
        $ sio transform labels.slp --crop 100,100,500,500 --scale 2.0 -o zoomed.slp

        [bold]Per-video crops:[/]
        $ sio transform multi_cam.slp --crop 0:100,100,500,500 -o cropped.slp

        [bold]Scale to target width:[/]
        $ sio transform labels.slp --scale 640 -o resized.slp

        [bold]Preview transforms:[/]
        $ sio transform labels.slp --crop 100,100,500,500 --dry-run

        [bold]Transform raw video:[/]
        $ sio transform video.mp4 --scale 0.5 -o video_scaled.mp4
    """
    from sleap_io.transform.video import compute_transform_summary, transform_labels

    # Resolve input path
    input_path = _resolve_input(input_arg, input_opt, "input file")

    # Parse fill value (int or R,G,B)
    fill_value = _parse_fill_value(fill)

    # Handle dry_run_frame (implies dry_run)
    if dry_run_frame is not None:
        dry_run = True

    # Check if input is video or SLP
    is_video = input_path.suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    }

    if is_video:
        # Handle raw video transformation
        _transform_video_file(
            input_path=input_path,
            output_path=output_path,
            crop_params=crop_params,
            scale_params=scale_params,
            rotate_params=rotate_params,
            pad_params=pad_params,
            quality=quality,
            fill=fill_value,
            crf=crf,
            preset=x264_preset,
            output_fps=output_fps,
            keyframe_interval=keyframe_interval,
            no_audio=no_audio,
            dry_run=dry_run,
            dry_run_frame=dry_run_frame,
            overwrite=overwrite,
            clip_rotation=clip_rotation,
            flip_h=flip_h,
            flip_v=flip_v,
        )
        return

    # Handle SLP file transformation
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".transformed.slp")

    # Ensure output has .slp extension
    if output_path.suffix.lower() != ".slp":
        output_path = output_path.with_suffix(".slp")

    # Check for same file
    if output_path.resolve() == input_path.resolve():
        raise click.ClickException(
            f"Output path cannot be the same as input: {input_path}\n"
            "Use a different output path."
        )

    # Check output exists
    if output_path.exists() and not overwrite:
        raise click.ClickException(
            f"Output file already exists: {output_path}\nUse --overwrite to replace it."
        )

    # Validate that at least one transform source is specified
    has_cli_transforms = any(
        [crop_params, scale_params, rotate_params, pad_params, flip_h, flip_v]
    )
    if not has_cli_transforms and not config_path:
        raise click.ClickException(
            "No transforms specified. Use --crop, --scale, --rotate, --pad, "
            "--flip-horizontal, --flip-vertical, or --config."
        )

    # Load SLP file
    console.print(f"[bold]Loading SLP:[/] {input_path}")
    labels = io_main.load_slp(str(input_path))

    if not labels.videos:
        raise click.ClickException("No videos found in SLP file.")

    console.print(f"[dim]  Found {len(labels.videos)} video(s)[/]")

    # Get input sizes for each video
    input_sizes = []
    for video in labels.videos:
        if hasattr(video, "shape") and video.shape is not None:
            h, w = video.shape[1:3]
            input_sizes.append((w, h))
        else:
            raise click.ClickException(
                f"Cannot determine dimensions for video: {video.filename}"
            )

    # Build transforms from config file first (lowest precedence)
    transforms: dict[int, Any] = {}
    if config_path:
        console.print(f"[dim]  Loading config: {config_path}[/]")
        transforms = _load_config_file(
            config_path=config_path,
            input_sizes=input_sizes,
            quality=quality,
            fill=fill_value,
            clip_rotation=clip_rotation,
            flip_h=flip_h,
            flip_v=flip_v,
        )

    # Build transforms from CLI parameters (higher precedence)
    if has_cli_transforms:
        try:
            cli_transforms = _build_transforms_from_params(
                n_videos=len(labels.videos),
                crop_params=crop_params,
                scale_params=scale_params,
                rotate_params=rotate_params,
                pad_params=pad_params,
                input_sizes=input_sizes,
                quality=quality,
                fill=fill_value,
                clip_rotation=clip_rotation,
                flip_h=flip_h,
                flip_v=flip_v,
            )
            # CLI params override config file
            transforms.update(cli_transforms)
        except ValueError as e:
            raise click.ClickException(str(e))

    if not transforms:
        raise click.ClickException("No valid transforms specified.")

    # Compute summary
    summary = compute_transform_summary(labels, transforms)

    # Print transform summary
    console.print("\n[bold]Transform Summary:[/]")
    for video_info in summary["videos"]:
        if video_info["has_transform"]:
            idx = video_info["index"]
            fname = video_info["filename"]
            console.print(f"\n  [bold]Video {idx}:[/] {fname}")
            if video_info.get("input_size") and video_info.get("output_size"):
                in_w, in_h = video_info["input_size"]
                out_w, out_h = video_info["output_size"]
                console.print(f"    Size: {in_w}x{in_h} -> {out_w}x{out_h}")
            t = video_info.get("transform", {})
            if t.get("crop"):
                console.print(f"    Crop: {t['crop']}")
            if t.get("scale"):
                console.print(f"    Scale: {t['scale']}")
            if t.get("rotate"):
                console.print(f"    Rotate: {t['rotate']}°")
            if t.get("pad"):
                console.print(f"    Pad: {t['pad']}")

    # Show warnings
    for warning in summary.get("warnings", []):
        console.print(f"[yellow]Warning: {warning}[/]")

    if dry_run:
        console.print(f"\n[bold]Dry run - would save SLP to:[/] {output_path}")

        # Render preview frame if requested
        if dry_run_frame is not None:
            # Find first video with a transform
            for video_idx, video in enumerate(labels.videos):
                if video_idx in transforms:
                    transform = transforms[video_idx]
                    frame_idx = dry_run_frame
                    n_frames = video.shape[0] if video.shape else 0

                    if frame_idx >= n_frames:
                        console.print(
                            f"[yellow]Warning: Frame {frame_idx} exceeds video "
                            f"length ({n_frames}), using frame 0[/]"
                        )
                        frame_idx = 0

                    console.print(
                        f"\n[bold]Preview frame {frame_idx} from video {video_idx}:[/]"
                    )

                    # Read and transform frame
                    frame = video[frame_idx]
                    if frame is not None:
                        transformed = transform.apply_to_frame(frame)

                        # Save preview to temp file
                        import tempfile

                        import numpy as np
                        from PIL import Image

                        preview_path = Path(tempfile.gettempdir()) / "sio_preview.png"
                        # Squeeze grayscale channel for PIL compatibility
                        preview_arr = np.squeeze(transformed)
                        # Ensure contiguous array
                        preview_arr = np.ascontiguousarray(preview_arr)
                        img = Image.fromarray(preview_arr)
                        img.save(preview_path)
                        console.print(f"  Saved preview to: {preview_path}")
                        console.print(
                            f"  Original: {frame.shape[1]}x{frame.shape[0]} -> "
                            f"Transformed: {img.size[0]}x{img.size[1]}"
                        )
                    break
        return

    # Check if all videos are embedded (no need for video output directory)
    from sleap_io.transform.video import _is_embedded_video

    all_embedded = all(_is_embedded_video(v) for v in labels.videos)

    # Create video output directory only for non-embedded videos
    video_output_dir = output_path.with_name(output_path.stem + ".videos")
    if not all_embedded:
        video_output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[dim]  Video output directory: {video_output_dir}[/]")

    # Progress callback
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        current_task = None

        def progress_callback(video_name: str, current: int, total: int) -> None:
            nonlocal current_task
            task_desc = f"Processing {video_name}"
            if current_task is None:
                current_task = progress.add_task(task_desc, total=total)
            progress.update(current_task, completed=current, description=task_desc)

        # Transform labels
        transformed_labels = transform_labels(
            labels=labels,
            transforms=transforms,
            output_path=output_path,
            video_output_dir=video_output_dir,
            fps=output_fps,
            crf=crf,
            preset=x264_preset,
            keyframe_interval=keyframe_interval,
            no_audio=no_audio,
            progress_callback=progress_callback,
            dry_run=False,
        )

    # Save the transformed SLP
    # Note: For embedded videos, transform_labels already saved the file
    # (to preserve the embedded video data). We only need to save for
    # non-embedded videos.
    has_embedded = any(_is_embedded_video(v) for v in labels.videos)

    # Generate transform metadata
    metadata = _generate_transform_metadata(
        labels=labels,
        transforms=transforms,
        source_path=input_path,
        output_path=output_path,
    )

    # Embed provenance in output SLP if requested
    if embed_provenance:
        transformed_labels.provenance["transform"] = metadata

    if not has_embedded:
        console.print(f"\n[bold]Saving SLP:[/] {output_path}")
        transformed_labels.save(str(output_path))
    elif embed_provenance:
        # For embedded videos, we need to update the provenance in the file
        # since it was already saved
        import json

        import h5py

        with h5py.File(output_path, "a") as f:
            if "provenance" not in f:
                f.create_group("provenance")
            # Store transform metadata as JSON
            if "transform_json" in f["provenance"]:
                del f["provenance/transform_json"]
            f["provenance"].create_dataset(
                "transform_json",
                data=json.dumps(metadata),
                dtype=h5py.special_dtype(vlen=str),
            )

    console.print(f"[bold green]Saved:[/] {output_path}")

    # Export transform metadata to YAML if requested
    if output_transforms_path:
        _export_transform_metadata(metadata, output_transforms_path)
        console.print(f"[bold green]Transform metadata:[/] {output_transforms_path}")


def _transform_video_file(
    input_path: Path,
    output_path: Path | None,
    crop_params: tuple[str, ...],
    scale_params: tuple[str, ...],
    rotate_params: tuple[str, ...],
    pad_params: tuple[str, ...],
    quality: str,
    fill: tuple[int, ...] | int,
    crf: int,
    preset: str,
    output_fps: float | None,
    keyframe_interval: float | None,
    no_audio: bool,
    dry_run: bool,
    dry_run_frame: int | None,
    overwrite: bool,
    clip_rotation: bool = False,
    flip_h: bool = False,
    flip_v: bool = False,
) -> None:
    """Transform a raw video file without labels.

    Args:
        input_path: Path to input video.
        output_path: Path for output video.
        crop_params: Crop parameter strings.
        scale_params: Scale parameter strings.
        rotate_params: Rotation parameter strings.
        pad_params: Padding parameter strings.
        quality: Interpolation quality.
        fill: Fill value (int or RGB tuple).
        crf: Video quality CRF.
        preset: Encoding preset.
        output_fps: Output FPS.
        keyframe_interval: Keyframe interval in seconds.
        no_audio: Strip audio from output.
        dry_run: Preview mode.
        dry_run_frame: Specific frame to render in preview.
        overwrite: Overwrite existing.
        clip_rotation: If True, rotation clips to original dimensions.
        flip_h: If True, flip horizontally.
        flip_v: If True, flip vertically.
    """
    from sleap_io.model.video import Video
    from sleap_io.transform import Transform
    from sleap_io.transform.core import (
        parse_crop,
        parse_pad,
        parse_scale,
        resolve_scale,
    )
    from sleap_io.transform.video import transform_video

    # Resolve output path
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + ".transformed")
        output_path = output_path.with_suffix(".mp4")

    # Check output exists
    if output_path.exists() and not overwrite:
        raise click.ClickException(
            f"Output file already exists: {output_path}\nUse --overwrite to replace it."
        )

    # Validate that at least one transform is specified
    if not any([crop_params, scale_params, rotate_params, pad_params, flip_h, flip_v]):
        raise click.ClickException(
            "No transforms specified. Use --crop, --scale, --rotate, --pad, "
            "--flip-horizontal, or --flip-vertical."
        )

    # Load video to get dimensions
    console.print(f"[bold]Loading video:[/] {input_path}")
    video = Video.from_filename(str(input_path))

    if video.shape is None:
        raise click.ClickException("Cannot determine video dimensions.")

    n_frames, h, w = video.shape[:3]
    input_size = (w, h)
    console.print(f"[dim]  {n_frames} frames, {w}x{h}[/]")

    # Parse transform parameters (only use non-indexed params for single video)
    crop = None
    scale = None
    rotate = None
    pad = None

    for crop_str in crop_params:
        idx, value = _parse_transform_param(crop_str)
        if idx is None or idx == 0:
            crop = parse_crop(value, input_size)

    for scale_str in scale_params:
        idx, value = _parse_transform_param(scale_str)
        if idx is None or idx == 0:
            parsed = parse_scale(value)
            scale = resolve_scale(parsed, input_size)

    for rotate_str in rotate_params:
        idx, value = _parse_transform_param(rotate_str)
        if idx is None or idx == 0:
            rotate = float(value)

    for pad_str in pad_params:
        idx, value = _parse_transform_param(pad_str)
        if idx is None or idx == 0:
            pad = parse_pad(value)

    # Create transform
    transform = Transform(
        crop=crop,
        scale=scale,
        rotate=rotate,
        pad=pad,
        quality=quality,
        fill=fill,
        clip_rotation=clip_rotation,
        flip_h=flip_h,
        flip_v=flip_v,
    )

    if not transform:
        raise click.ClickException("No valid transforms specified.")

    # Compute output size
    out_w, out_h = transform.output_size(input_size)

    # Print summary
    console.print("\n[bold]Transform Summary:[/]")
    console.print(f"  Size: {w}x{h} -> {out_w}x{out_h}")
    if crop:
        console.print(f"  Crop: {crop}")
    if scale:
        console.print(f"  Scale: {scale}")
    if rotate:
        console.print(f"  Rotate: {rotate}°")
    if pad:
        console.print(f"  Pad: {pad}")

    if dry_run:
        console.print(f"\n[bold]Dry run - would save video to:[/] {output_path}")

        # Render preview frame if requested
        if dry_run_frame is not None:
            frame_idx = dry_run_frame
            if frame_idx >= n_frames:
                console.print(
                    f"[yellow]Warning: Frame {frame_idx} exceeds video "
                    f"length ({n_frames}), using frame 0[/]"
                )
                frame_idx = 0

            console.print(f"\n[bold]Preview frame {frame_idx}:[/]")

            # Read and transform frame
            frame = video[frame_idx]
            if frame is not None:
                transformed = transform.apply_to_frame(frame)

                # Save preview to temp file
                import tempfile

                import numpy as np
                from PIL import Image

                preview_path = Path(tempfile.gettempdir()) / "sio_preview.png"
                # Squeeze grayscale channel for PIL compatibility
                preview_arr = np.squeeze(transformed)
                # Ensure contiguous array
                preview_arr = np.ascontiguousarray(preview_arr)
                img = Image.fromarray(preview_arr)
                img.save(preview_path)
                console.print(f"  Saved preview to: {preview_path}")
                console.print(
                    f"  Original: {frame.shape[1]}x{frame.shape[0]} -> "
                    f"Transformed: {img.size[0]}x{img.size[1]}"
                )
        return

    # Transform video with progress
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Transforming video", total=n_frames)

        def progress_callback(current: int, total: int) -> None:
            progress.update(task, completed=current)

        transform_video(
            video=video,
            output_path=output_path,
            transform=transform,
            fps=output_fps,
            crf=crf,
            preset=preset,
            keyframe_interval=keyframe_interval,
            no_audio=no_audio,
            progress_callback=progress_callback,
        )

    console.print(f"[bold green]Saved:[/] {output_path}")
