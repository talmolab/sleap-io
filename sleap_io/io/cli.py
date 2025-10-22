"""Command-line interface for sleap-io.

Provides a `sio` command with subcommands for inspecting and manipulating
SLEAP labels and related formats. This module intentionally keeps the
default behavior light-weight (no video backends by default) to work well
in minimal environments and CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import rich_click as click

from sleap_io.io import main as io_main
from sleap_io.io.utils import sanitize_filename
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.skeleton import Skeleton

click.rich_click.USE_MARKDOWN = True


@click.group(help="sleap-io command line interface")
def cli():
    """Top-level command group for the sleap-io CLI."""
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
