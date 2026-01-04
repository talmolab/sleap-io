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
console = Console()

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
        {"name": "Inspection", "commands": ["show", "filenames"]},
        {"name": "Transformation", "commands": ["convert", "split"]},
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

        $ sio show labels.slp
        $ sio show labels.slp --skeleton
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

    # Count instances
    n_user = sum(len(lf.user_instances) for lf in labels.labeled_frames)
    n_pred = sum(len(lf.predicted_instances) for lf in labels.labeled_frames)

    # Build header content
    header_lines = [
        f"[bold cyan]{path.name}[/]",
        f"[dim]{path.parent}[/]",
        "",
        f"[dim]Type:[/]     {file_type}",
        f"[dim]Size:[/]     {_format_file_size(file_size)}",
    ]

    # Stats row
    stats_parts = [
        f"[bold]{len(labels.videos)}[/] video{'s' if len(labels.videos) != 1 else ''}",
        f"[bold]{len(labels.labeled_frames)}[/] "
        f"frame{'s' if len(labels.labeled_frames) != 1 else ''}",
    ]

    if n_user > 0:
        stats_parts.append(f"[bold]{n_user}[/] labeled")
    if n_pred > 0:
        stats_parts.append(f"[bold]{n_pred}[/] predicted")
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

    for i, vid in enumerate(labels.videos):
        # Get video info using defensive helpers
        fname = _format_video_filename(vid)
        is_embedded = _is_embedded(vid)

        # Shape info
        if vid.shape:
            n_frames, h, w, c = vid.shape
            shape_str = f"{w}×{h}"
            frames_str = f"{n_frames} frames"
        else:
            shape_str = "[dim]?×?[/]"
            frames_str = "[dim]? frames[/]"

        # Build status tag
        tag = ""
        if is_embedded:
            tag = " [cyan][embedded][/]"
        elif not vid.exists() and not isinstance(vid.filename, list):
            tag = " [yellow][not found][/]"

        # Format: [idx] filename          WxH    N frames  [tag]
        idx_str = f"[dim][{i}][/]"
        console.print(f"  {idx_str} [cyan]{fname}[/]  {shape_str}  {frames_str}{tag}")


def _print_video_details(labels: Labels) -> None:
    """Print detailed video information with consistent field ordering."""
    if not labels.videos:
        console.print("[dim]No videos[/]")
        return

    for i, vid in enumerate(labels.videos):
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
            header += " [cyan][embedded][/]"
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

        # Labeled frames in this video
        n_frames_labeled = sum(1 for lf in labels.labeled_frames if lf.video == vid)
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

    # Count instances per track
    track_counts = {t: 0 for t in labels.tracks}
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            if inst.track and inst.track in track_counts:
                track_counts[inst.track] += 1

    console.print()
    table = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Track", style="cyan")
    table.add_column("Instances", justify="right")

    for idx, track in enumerate(labels.tracks):
        table.add_row(
            str(idx), track.name or "[dim]unnamed[/]", str(track_counts[track])
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


def _print_provenance(labels: Labels) -> None:
    """Print provenance information."""
    if not labels.provenance:
        console.print("[dim]No provenance information[/]")
        return

    console.print()
    console.print("[bold cyan]Provenance[/]")
    console.print()

    for key, value in labels.provenance.items():
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value[:3])
            if len(value) > 3:
                value_str += f" ... ({len(value)} total)"
        elif isinstance(value, dict):
            value_str = f"{{...}} ({len(value)} keys)"
        else:
            value_str = str(value)[:60]
            if len(str(value)) > 60:
                value_str += "..."

        console.print(f"  [dim]{key}:[/] {value_str}")


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
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
    path: Path,
    open_videos: Optional[bool],
    lf_index: Optional[int],
    skeleton: bool,
    video: bool,
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
        $ sio show labels.slp --skeleton
        $ sio show labels.slp --lf 0
        $ sio show labels.slp --all
    """
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
    obj = io_main.load_file(str(path), open_videos=open_videos)

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
                _print_video_details(obj)

            if tracks:
                console.print("[bold]Tracks[/]")
                _print_tracks_details(obj)

            if lf_index is not None:
                if len(obj.labeled_frames) == 0:
                    raise click.ClickException("No labeled frames present in file.")
                _print_labeled_frame(obj, lf_index)

        if provenance:
            _print_provenance(obj)

        console.print()
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


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input labels file path.",
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
    input_path: Path,
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

        $ sio split -i labels.slp -o splits/
        $ sio split -i labels.slp -o splits/ --train 0.7 --val 0.15 --test 0.15
        $ sio split -i labels.slp -o splits/ --remove-predictions --seed 42
        $ sio split -i labels.slp -o splits/ --embed user
    """
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


@cli.command("filenames")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input labels file.",
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
def filenames(
    input_path: Path,
    output_path: Optional[Path],
    new_filenames: tuple[str, ...],
    filename_map: tuple[tuple[str, str], ...],
    prefix_map: tuple[tuple[str, str], ...],
):
    r"""List or update video filenames in a labels file.

    By default, lists all video filenames for quick inspection.
    With -o and update flags, replaces video paths and saves to output.

    [bold]Inspection mode[/] (default):

        $ sio filenames -i labels.slp

    [bold]Update modes[/] (require -o):

    [bold]List mode[/] (--filename): Replace all video filenames in order.
    Provide one --filename for each video in the labels file.

    [bold]Map mode[/] (--map OLD NEW): Replace specific filenames.
    Use exact path matching to replace OLD with NEW.

    [bold]Prefix mode[/] (--prefix OLD NEW): Replace path prefixes.
    Cross-platform aware (handles Windows/Linux path differences).

    [dim]Examples:[/]

        $ sio filenames -i labels.slp
        $ sio filenames -i labels.slp -o out.slp --filename /new/video.mp4
        $ sio filenames -i labels.slp -o out.slp --prefix "C:\\data" /mnt/data
    """
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
        click.echo(f"Video filenames in {input_path.name}:")
        for i, vid in enumerate(labels.videos):
            fn = vid.filename
            if isinstance(fn, list):
                click.echo(f"  [{i}] {fn[0]} ... ({len(fn)} images)")
            else:
                click.echo(f"  [{i}] {fn}")
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

    # Save the output file
    try:
        io_main.save_file(labels, str(output_path))
    except Exception as e:
        raise click.ClickException(f"Failed to save output file: {e}")

    # Success message
    n_videos = len(labels.videos)
    mode_name = "list" if new_filenames else "map" if filename_map else "prefix"
    click.echo(f"Replaced filenames in {n_videos} video(s) using {mode_name} mode")
    click.echo(f"Saved: {output_path}")
