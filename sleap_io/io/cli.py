"""Command-line interface for sleap-io.

Provides a `sio` command with subcommands for inspecting and manipulating
SLEAP labels and related formats. This module intentionally keeps the
default behavior light-weight (video support via bundled imageio-ffmpeg) to work well
in minimal environments and CI.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Optional

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
        {"name": "Transformation", "commands": ["convert", "split", "unsplit"]},
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
    input_arg: Optional[Path],
    input_opt: Optional[Path],
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
    crop_str: Optional[str],
) -> Optional[tuple]:
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
    status = _build_status_line(video)
    plugin = _get_plugin(video)

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

    # Additional details below panel (full path here to avoid truncation)
    console.print()
    full_path = path.resolve()
    console.print(f"  [dim]Full[/]      {full_path}")
    console.print(f"  [dim]Status[/]    {status}")
    if plugin:
        console.print(f"  [dim]Plugin[/]    {plugin}")

    # Show backend metadata if available
    if video.backend_metadata:
        meta = video.backend_metadata
        if meta.get("grayscale") is not None:
            gs_str = "yes" if meta["grayscale"] else "no"
            console.print(f"  [dim]Grayscale[/] {gs_str}")

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


def _print_video_details(labels: Labels, video_index: Optional[int] = None) -> None:
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
    path_arg: Optional[Path],
    path_opt: Optional[Path],
    lazy: bool,
    open_videos: Optional[bool],
    lf_index: Optional[int],
    skeleton: bool,
    video: bool,
    video_index: Optional[int],
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
    input_arg: Optional[Path],
    input_opt: Optional[Path],
    output_path: Path,
    input_format: Optional[str],
    output_format: Optional[str],
    embed: Optional[str],
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
        if embed is not None:
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
    input_arg: Optional[Path],
    input_opt: Optional[Path],
    output_dir: Path,
    train_fraction: float,
    val_fraction: Optional[float],
    test_fraction: Optional[float],
    remove_predictions: bool,
    seed: Optional[int],
    embed: Optional[str],
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
    embed: Optional[str],
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
        if embed is not None:
            save_kwargs["embed"] = embed
        io_main.save_file(labels, str(output_path), **save_kwargs)
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    click.echo("")
    click.echo(f"Merged {len(expanded_files)} files:")
    click.echo(f"  {len(labels)} frames, {len(labels.videos)} videos")


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
    input_arg: Optional[Path],
    input_opt: Optional[Path],
    output_path: Optional[Path],
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
    input_arg: Optional[Path],
    input_opt: Optional[Path],
    output_path: Optional[Path],
    lf_ind: Optional[int],
    frame_idx: Optional[int],
    start_frame_idx: Optional[int],
    end_frame_idx: Optional[int],
    video_ind: int,
    all_frames: Optional[bool],
    preset: Optional[str],
    scale: Optional[float],
    fps: Optional[float],
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
    crop_str: Optional[str],
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
    input_arg: Optional[Path],
    input_opt: Optional[Path],
    output_path: Optional[Path],
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
    most_frequent_skeleton: Optional[Skeleton] = None
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
        io_main.save_file(labels, str(output_path))
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Print final stats
    console.print()
    console.print(f"[bold green]Saved:[/] {output_path}")
    console.print(
        f"[dim]  {len(labels.videos)} videos, {len(labels.labeled_frames)} frames, "
        f"{len(labels.skeletons)} skeletons, {len(labels.tracks)} tracks[/]"
    )
