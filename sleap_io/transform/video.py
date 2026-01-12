"""Video transformation pipeline.

This module provides functions for transforming videos and their associated
label coordinates. It handles loading videos, applying frame transformations,
updating landmark coordinates, and saving the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from sleap_io.transform.core import Transform

if TYPE_CHECKING:
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video


def transform_video(
    video: "Video",
    output_path: Path,
    transform: Transform,
    fps: float | None = None,
    crf: int = 25,
    preset: str = "superfast",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Transform a video file and save to a new path.

    Args:
        video: Source video object.
        output_path: Path to save transformed video.
        transform: Transform to apply to each frame.
        fps: Output frame rate. If None, uses source FPS.
        crf: Constant rate factor for video quality (0-51, lower is better).
        preset: x264 encoding preset.
        progress_callback: Optional callback called with (current_frame, total_frames).

    Returns:
        Path to the output video file.
    """
    from sleap_io.io.video_writing import VideoWriter

    # Get video properties
    n_frames = video.shape[0]
    if fps is None:
        if hasattr(video, "backend") and video.backend is not None:
            try:
                fps = video.backend.fps
            except Exception:
                fps = 30.0
        else:
            fps = 30.0

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write transformed video
    with VideoWriter(
        filename=output_path,
        fps=fps,
        crf=crf,
        preset=preset,
    ) as writer:
        for frame_idx in range(n_frames):
            # Read frame
            frame = video[frame_idx]
            if frame is None:
                continue

            # Apply transform
            transformed_frame = transform.apply_to_frame(frame)

            # Write frame
            writer(transformed_frame)

            # Progress callback
            if progress_callback is not None:
                progress_callback(frame_idx + 1, n_frames)

    return output_path


def transform_labels(
    labels: "Labels",
    transforms: dict[int, Transform] | Transform,
    output_path: Path,
    video_output_dir: Path | None = None,
    fps: float | None = None,
    crf: int = 25,
    preset: str = "superfast",
    progress_callback: Callable[[str, int, int], None] | None = None,
    dry_run: bool = False,
) -> "Labels":
    """Transform all videos in a Labels object and update coordinates.

    Args:
        labels: Source Labels object.
        transforms: Either a single Transform to apply to all videos, or a dict
            mapping video indices to their respective Transforms.
        output_path: Path to save transformed SLP file.
        video_output_dir: Directory for transformed videos. If None, uses
            "{output_path.stem}.videos/".
        fps: Output frame rate. If None, preserves source FPS.
        crf: Constant rate factor for video quality (0-51, lower is better).
        preset: x264 encoding preset.
        progress_callback: Optional callback called with (video_name, current, total).
        dry_run: If True, compute transforms but don't process videos.

    Returns:
        New Labels object with transformed videos and adjusted coordinates.
    """
    from sleap_io.model.video import Video

    # Normalize transforms to dict
    if isinstance(transforms, Transform):
        transforms_dict = {i: transforms for i in range(len(labels.videos))}
    else:
        transforms_dict = transforms

    # Setup video output directory
    if video_output_dir is None:
        video_output_dir = output_path.with_name(output_path.stem + ".videos")

    if not dry_run:
        video_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy of labels to modify
    new_labels = labels.copy()

    # Track video replacements
    video_map: dict[Video, Video] = {}

    # Process each video
    for video_idx, video in enumerate(labels.videos):
        transform = transforms_dict.get(video_idx, Transform())

        # Skip if no transform
        if not transform:
            continue

        # Get video dimensions for coordinate transformation
        if hasattr(video, "shape") and video.shape is not None:
            h, w = video.shape[1:3]
            input_size = (w, h)
        else:
            # Try to get from first frame
            try:
                frame = video[0]
                h, w = frame.shape[:2]
                input_size = (w, h)
            except Exception:
                raise ValueError(
                    f"Cannot determine dimensions for video {video_idx}: "
                    f"{video.filename}"
                )

        # Transform video file
        if not dry_run:
            video_name = Path(video.filename).stem
            output_video_path = video_output_dir / f"{video_name}.transformed.mp4"

            def _progress(current: int, total: int) -> None:
                if progress_callback is not None:
                    progress_callback(video_name, current, total)

            transform_video(
                video=video,
                output_path=output_video_path,
                transform=transform,
                fps=fps,
                crf=crf,
                preset=preset,
                progress_callback=_progress,
            )

            # Create new video object
            new_video = Video.from_filename(
                output_video_path.as_posix(), grayscale=video.grayscale
            )
            new_video.source_video = video
            video_map[video] = new_video

        # Update coordinates for this video's labeled frames
        for lf in new_labels.labeled_frames:
            if lf.video is not video:
                continue

            for instance in lf.instances:
                # Get current points
                points = instance.numpy(invisible_as_nan=False)

                # Transform coordinates
                transformed_points = transform.apply_to_points(points, input_size)

                # Update instance points
                instance.points["xy"] = transformed_points

    # Replace video references
    if video_map:
        new_labels.replace_videos(video_map=video_map)

    return new_labels


def compute_transform_summary(
    labels: "Labels",
    transforms: dict[int, Transform] | Transform,
) -> dict:
    """Compute a summary of transforms to be applied.

    This is useful for dry-run mode to preview what will happen.

    Args:
        labels: Source Labels object.
        transforms: Either a single Transform or dict mapping video indices.

    Returns:
        Dictionary with transform summary information.
    """
    # Normalize transforms to dict
    if isinstance(transforms, Transform):
        transforms_dict = {i: transforms for i in range(len(labels.videos))}
    else:
        transforms_dict = transforms

    summary = {
        "videos": [],
        "total_frames": 0,
        "total_instances": 0,
        "warnings": [],
    }

    for video_idx, video in enumerate(labels.videos):
        transform = transforms_dict.get(video_idx, Transform())

        # Get video info
        video_info = {
            "index": video_idx,
            "filename": video.filename,
            "has_transform": bool(transform),
        }

        # Get dimensions
        if hasattr(video, "shape") and video.shape is not None:
            n_frames, h, w = video.shape[:3]
            input_size = (w, h)
            video_info["input_size"] = input_size
            video_info["n_frames"] = n_frames
            summary["total_frames"] += n_frames
        else:
            video_info["input_size"] = None
            video_info["n_frames"] = None

        # Compute output size
        if transform and video_info["input_size"]:
            output_size = transform.output_size(input_size)
            video_info["output_size"] = output_size

            # Check for warnings
            out_w, out_h = output_size
            if out_w < 32 or out_h < 32:
                summary["warnings"].append(
                    f"Video {video_idx}: Output size very small ({out_w}x{out_h})"
                )

        # Count instances for this video
        n_instances = 0
        for lf in labels.labeled_frames:
            if lf.video is video:
                n_instances += len(lf.instances)
        video_info["n_instances"] = n_instances
        summary["total_instances"] += n_instances

        # Add transform details
        if transform:
            video_info["transform"] = {
                "crop": transform.crop,
                "scale": transform.scale,
                "rotate": transform.rotate,
                "pad": transform.pad,
            }

        summary["videos"].append(video_info)

    return summary
