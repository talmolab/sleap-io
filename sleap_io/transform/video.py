"""Video transformation pipeline.

This module provides functions for transforming videos and their associated
label coordinates. It handles loading videos, applying frame transformations,
updating landmark coordinates, and saving the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from sleap_io.transform.core import Transform
from sleap_io.transform.points import get_out_of_bounds_mask

if TYPE_CHECKING:
    from sleap_io.model.labels import Labels
    from sleap_io.model.video import Video


def _is_embedded_video(video: "Video") -> bool:
    """Check if a video has embedded images.

    Args:
        video: Video object to check.

    Returns:
        True if the video contains embedded images in HDF5 format.
    """
    from sleap_io.io.video_reading import HDF5Video

    if not hasattr(video, "backend") or video.backend is None:
        return False

    backend = video.backend
    if not isinstance(backend, HDF5Video):
        return False

    return backend.has_embedded_images


def _get_frame_indices(video: "Video") -> list[int]:
    """Get the frame indices to iterate over for a video.

    For regular videos, returns range(n_frames).
    For embedded videos, returns the embedded frame indices (source indices).

    Args:
        video: Video object.

    Returns:
        List of frame indices to iterate over.
    """
    from sleap_io.io.video_reading import HDF5Video

    if hasattr(video, "backend") and isinstance(video.backend, HDF5Video):
        if video.backend.frame_map:
            # Embedded video - return the source frame indices
            return video.backend.embedded_frame_inds

    # Regular video - sequential indices
    n_frames = video.shape[0] if video.shape else 0
    return list(range(n_frames))


def transform_video(
    video: "Video",
    output_path: Path,
    transform: Transform,
    fps: float | None = None,
    crf: int = 25,
    preset: str = "superfast",
    keyframe_interval: float | None = None,
    no_audio: bool = False,
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
        keyframe_interval: Interval between keyframes in seconds. If None, uses
            encoder default.
        no_audio: If True, strips audio from output.
        progress_callback: Optional callback called with (current_frame, total_frames).

    Returns:
        Path to the output video file.

    Notes:
        For embedded HDF5 videos, use `transform_embedded_video()` instead.
    """
    from sleap_io.io.video_writing import VideoWriter

    # Get frame indices to iterate over
    frame_inds = _get_frame_indices(video)
    n_frames = len(frame_inds)

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
        keyframe_interval=keyframe_interval,
        no_audio=no_audio,
    ) as writer:
        for i, frame_idx in enumerate(frame_inds):
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
                progress_callback(i + 1, n_frames)

    return output_path


def transform_embedded_video(
    video: "Video",
    output_path: Path,
    video_idx: int,
    transform: Transform,
    image_format: str = "png",
    plugin: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> "Video":
    """Transform an embedded video and write to an HDF5 output file.

    This function reads embedded frames, transforms them, and writes them to a new
    embedded video dataset in the output HDF5 file.

    Args:
        video: Source video object with embedded images.
        output_path: Path to the output SLP/HDF5 file (must already exist).
        video_idx: Index of this video in the labels (used for group naming).
        transform: Transform to apply to each frame.
        image_format: Image format for encoding ("png" or "jpg").
        plugin: Image plugin to use for encoding ("opencv" or "imageio").
        progress_callback: Optional callback called with (current_frame, total_frames).

    Returns:
        New Video object pointing to the embedded video in the output file.
    """
    import json
    import sys

    import h5py
    import numpy as np

    from sleap_io.io.video_reading import VideoBackend, get_default_image_plugin
    from sleap_io.model.video import Video

    # Determine plugin
    if plugin is None:
        plugin = get_default_image_plugin()
    if plugin is None:
        plugin = "opencv" if "cv2" in sys.modules else "imageio"

    # Get embedded frame indices
    frame_inds = _get_frame_indices(video)
    n_frames = len(frame_inds)

    # Handle empty videos (no embedded frames)
    if n_frames == 0:
        # Return a video object that references the same empty structure
        from sleap_io.io.video_reading import VideoBackend

        source_video = video.source_video if video.source_video is not None else video
        return Video(
            filename=str(output_path),
            backend=video.backend,  # Keep same backend reference
            source_video=source_video,
        )

    # Get input dimensions for computing output size
    if video.shape is not None:
        h, w = video.shape[1:3]
        input_size = (w, h)
    else:
        raise ValueError("Cannot determine dimensions for embedded video")

    # Compute output dimensions
    out_w, out_h = transform.output_size(input_size)

    # Determine number of channels from first frame
    first_frame = video[frame_inds[0]]
    channels = first_frame.shape[2] if first_frame.ndim == 3 else 1

    # Transform and encode all frames
    imgs_data = []
    for i, frame_idx in enumerate(frame_inds):
        frame = video[frame_idx]

        # Apply transform
        transformed = transform.apply_to_frame(frame)

        # Encode frame
        if plugin == "opencv":
            import cv2

            img_data = np.squeeze(
                cv2.imencode("." + image_format, transformed)[1]
            ).astype("int8")
            channel_order = "BGR"
        else:  # imageio
            import imageio.v3 as iio

            if transformed.shape[-1] == 1:
                transformed = transformed.squeeze(axis=-1)
            img_data = np.frombuffer(
                iio.imwrite("<bytes>", transformed, extension="." + image_format),
                dtype="int8",
            )
            channel_order = "RGB"

        imgs_data.append(img_data)

        if progress_callback is not None:
            progress_callback(i + 1, n_frames)

    # Write to HDF5 file
    group = f"video{video_idx}"

    with h5py.File(output_path, "a") as f:
        # Create dataset with fixed-length encoding
        img_bytes_len = max(len(img) for img in imgs_data)
        ds = f.create_dataset(
            f"{group}/video",
            shape=(len(imgs_data), img_bytes_len),
            dtype="int8",
            compression="gzip",
        )
        for i, img in enumerate(imgs_data):
            ds[i, : len(img)] = img

        # Store metadata with TRANSFORMED dimensions
        ds.attrs["format"] = image_format
        ds.attrs["channel_order"] = channel_order
        ds.attrs["frames"] = n_frames
        ds.attrs["height"] = out_h
        ds.attrs["width"] = out_w
        ds.attrs["channels"] = channels

        # Store FPS if available
        if video.fps is not None:
            ds.attrs["fps"] = video.fps

        # Store frame indices (same as source)
        f.create_dataset(f"{group}/frame_numbers", data=frame_inds)

        # Store source video reference
        source_video = video.source_video if video.source_video is not None else video
        grp = f.require_group(f"{group}/source_video")

        # Build source video dict
        source_dict = {
            "backend": {
                "filename": source_video.filename,
                "grayscale": source_video.grayscale,
            }
        }
        grp.attrs["json"] = json.dumps(source_dict, separators=(",", ":"))

    # Create and return the new embedded Video object
    embedded_video = Video(
        filename=str(output_path),
        backend=VideoBackend.from_filename(
            str(output_path),
            dataset=f"{group}/video",
            grayscale=video.grayscale,
            keep_open=False,
        ),
        source_video=source_video,
    )

    return embedded_video


def _update_videos_json(output_path: Path, videos: list["Video"]) -> None:
    """Update the videos_json dataset in an HDF5 file.

    This is used after writing embedded video data to update the video references
    without re-saving the entire file (which would overwrite the embedded data).

    Args:
        output_path: Path to the HDF5/SLP file.
        videos: List of Video objects with updated backend references.
    """
    import json

    import h5py

    # Build the videos JSON array
    videos_json_data = []
    for video in videos:
        # Build backend dict
        backend = video.backend
        backend_dict: dict = {}

        if hasattr(backend, "filename"):
            # For embedded videos, use "." as filename to indicate same file
            if str(video.filename) == str(output_path):
                backend_dict["filename"] = "."
            else:
                backend_dict["filename"] = backend.filename

        if hasattr(backend, "dataset") and backend.dataset:
            backend_dict["dataset"] = backend.dataset

        if hasattr(backend, "input_format"):
            backend_dict["input_format"] = backend.input_format

        if hasattr(backend, "convert_range"):
            backend_dict["convert_range"] = backend.convert_range

        video_dict = {"backend": backend_dict}
        videos_json_data.append(json.dumps(video_dict, separators=(",", ":")))

    # Update the videos_json dataset in place
    with h5py.File(output_path, "a") as f:
        if "videos_json" in f:
            del f["videos_json"]
        f.create_dataset(
            "videos_json",
            data=videos_json_data,
            dtype=h5py.special_dtype(vlen=str),
        )


def transform_labels(
    labels: "Labels",
    transforms: dict[int, Transform] | Transform,
    output_path: Path,
    video_output_dir: Path | None = None,
    fps: float | None = None,
    crf: int = 25,
    preset: str = "superfast",
    keyframe_interval: float | None = None,
    no_audio: bool = False,
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
            "{output_path.stem}.videos/". Ignored for embedded videos.
        fps: Output frame rate. If None, preserves source FPS. Ignored for embedded.
        crf: Constant rate factor for video quality (0-51, lower is better).
            Ignored for embedded videos.
        preset: x264 encoding preset. Ignored for embedded videos.
        keyframe_interval: Interval between keyframes in seconds. If None, uses
            encoder default. Ignored for embedded videos.
        no_audio: If True, strips audio from output. Ignored for embedded videos.
        progress_callback: Optional callback called with (video_name, current, total).
        dry_run: If True, compute transforms but don't process videos.

    Returns:
        New Labels object with transformed videos and adjusted coordinates.

    Notes:
        For embedded videos (`.pkg.slp`), the output will also be an embedded file
        with transformed frame images. The `video_output_dir` and video encoding
        parameters are ignored for embedded videos.
    """
    from sleap_io.model.video import Video

    # Normalize transforms to dict
    if isinstance(transforms, Transform):
        transforms_dict = {i: transforms for i in range(len(labels.videos))}
    else:
        transforms_dict = transforms

    # Check if any videos are embedded
    has_embedded = any(_is_embedded_video(v) for v in labels.videos)

    # Setup video output directory (for non-embedded videos)
    if video_output_dir is None:
        video_output_dir = output_path.with_name(output_path.stem + ".videos")

    if not dry_run and not has_embedded:
        video_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy of labels to modify
    new_labels = labels.copy()

    # Track video replacements
    video_map: dict[Video, Video] = {}

    # For embedded videos, we need to save labels first to create the HDF5 file
    # Then embed transformed frames into it
    embedded_videos_to_process: list[tuple[int, "Video", Transform]] = []

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
            frame_inds = _get_frame_indices(video)
            if frame_inds:
                try:
                    frame = video[frame_inds[0]]
                    h, w = frame.shape[:2]
                    input_size = (w, h)
                except Exception:
                    raise ValueError(
                        f"Cannot determine dimensions for video {video_idx}: "
                        f"{video.filename}"
                    )
            else:
                raise ValueError(
                    f"Cannot determine dimensions for video {video_idx}: "
                    f"{video.filename}"
                )

        # Handle embedded vs regular videos differently
        is_embedded = _is_embedded_video(video)

        if not dry_run:
            if is_embedded:
                # Queue embedded video for processing after labels are saved
                embedded_videos_to_process.append((video_idx, video, transform))
            else:
                # Regular video - transform to .mp4 file
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
                    keyframe_interval=keyframe_interval,
                    no_audio=no_audio,
                    progress_callback=_progress,
                )

                # Create new video object
                new_video = Video.from_filename(
                    output_video_path.as_posix(), grayscale=video.grayscale
                )
                new_video.source_video = video
                video_map[new_labels.videos[video_idx]] = new_video

        # Update coordinates for this video's labeled frames
        copied_video = new_labels.videos[video_idx]

        # Compute output bounds for OOB visibility check
        output_w, output_h = transform.output_size(input_size)
        output_bounds = (0, 0, output_w, output_h)

        for lf in new_labels.labeled_frames:
            if lf.video is not copied_video:
                continue

            for instance in lf.instances:
                points = instance.numpy(invisible_as_nan=False)
                transformed_points = transform.apply_to_points(points, input_size)
                instance.points["xy"] = transformed_points

                # Mark out-of-bounds points as not visible
                oob_mask = get_out_of_bounds_mask(transformed_points, output_bounds)
                instance.points["visible"][oob_mask] = False

    # Replace video references for regular videos
    if video_map:
        new_labels.replace_videos(video_map=video_map)

    # Process embedded videos after labels structure is ready
    # For embedded videos, we need to:
    # 1. Save the labels first (creates HDF5 structure)
    # 2. Transform and embed frames into the saved file
    # 3. Update video references in the returned labels
    if embedded_videos_to_process and not dry_run:
        # Save labels first to create the HDF5 file
        new_labels.save(str(output_path))

        # Now transform and embed each video
        embedded_video_map: dict[Video, Video] = {}
        for video_idx, video, transform in embedded_videos_to_process:
            video_name = Path(video.filename).stem

            def _progress(current: int, total: int) -> None:
                if progress_callback is not None:
                    progress_callback(video_name, current, total)

            new_embedded_video = transform_embedded_video(
                video=video,
                output_path=output_path,
                video_idx=video_idx,
                transform=transform,
                progress_callback=_progress,
            )

            embedded_video_map[new_labels.videos[video_idx]] = new_embedded_video

        # Replace video references for embedded videos
        if embedded_video_map:
            new_labels.replace_videos(video_map=embedded_video_map)

            # Update videos_json in the HDF5 file to point to the embedded data
            # We can't re-save because that would overwrite the embedded video data
            _update_videos_json(output_path, new_labels.videos)

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
        Dictionary with transform summary information including warnings for:
        - Rotation clips >20% of frame area
        - Landmarks go out of bounds
        - Output size <32px in any dimension
        - Non-integer dimensions (rounding applied)
        - Crop extends outside frame (padding applied)
    """
    import numpy as np

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
            input_size = None

        # Compute output size and generate warnings
        if transform and input_size:
            output_size = transform.output_size(input_size)
            video_info["output_size"] = output_size
            out_w, out_h = output_size
            in_w, in_h = input_size

            # Warning: Output size very small (<32px)
            if out_w < 32 or out_h < 32:
                summary["warnings"].append(
                    f"Video {video_idx}: Output size very small ({out_w}x{out_h})"
                )

            # Warning: Crop extends outside frame (padding applied)
            if transform.crop:
                x1, y1, x2, y2 = transform.crop
                if x1 < 0 or y1 < 0 or x2 > in_w or y2 > in_h:
                    summary["warnings"].append(
                        f"Video {video_idx}: Crop extends outside frame, "
                        f"padding will be applied"
                    )

            # Warning: Rotation clips >20% of frame
            if transform.rotate and transform.rotate != 0 and transform.clip_rotation:
                # Calculate how much area is lost when clipping rotation
                angle_rad = np.radians(abs(transform.rotate))
                cos_a = abs(np.cos(angle_rad))
                sin_a = abs(np.sin(angle_rad))

                # Get the size after crop/scale (before rotation)
                pre_rotate_w, pre_rotate_h = in_w, in_h
                if transform.crop:
                    x1, y1, x2, y2 = transform.crop
                    pre_rotate_w = x2 - x1
                    pre_rotate_h = y2 - y1
                if transform.scale:
                    scale_x, scale_y = transform.scale
                    pre_rotate_w = int(round(pre_rotate_w * scale_x))
                    pre_rotate_h = int(round(pre_rotate_h * scale_y))

                # Expanded dimensions without clipping
                expanded_w = int(np.ceil(pre_rotate_w * cos_a + pre_rotate_h * sin_a))
                expanded_h = int(np.ceil(pre_rotate_w * sin_a + pre_rotate_h * cos_a))

                # Calculate area loss percentage
                original_area = pre_rotate_w * pre_rotate_h
                expanded_area = expanded_w * expanded_h

                # The clipped content is the original minus what fits in the rotated box
                # For a simpler estimate, use the ratio of areas
                if expanded_area > original_area:
                    clip_fraction = 1 - (original_area / expanded_area)
                    clip_pct = clip_fraction * 100
                    if clip_pct > 20:
                        summary["warnings"].append(
                            f"Video {video_idx}: Rotation will clip ~{clip_pct:.0f}% "
                            f"of frame area"
                        )

            # Warning: Non-integer dimensions (rounding)
            if transform.scale:
                scale_x, scale_y = transform.scale
                # Get pre-scale dimensions
                pre_scale_w, pre_scale_h = in_w, in_h
                if transform.crop:
                    x1, y1, x2, y2 = transform.crop
                    pre_scale_w = x2 - x1
                    pre_scale_h = y2 - y1

                # Calculate exact dimensions
                exact_w = pre_scale_w * scale_x
                exact_h = pre_scale_h * scale_y
                rounded_w = int(round(exact_w))
                rounded_h = int(round(exact_h))

                # Check if rounding was needed
                if abs(exact_w - rounded_w) > 0.01 or abs(exact_h - rounded_h) > 0.01:
                    summary["warnings"].append(
                        f"Video {video_idx}: Dimensions rounded: "
                        f"{exact_w:.1f}x{exact_h:.1f} -> {rounded_w}x{rounded_h}"
                    )

        # Count instances and check for OOB landmarks
        n_instances = 0
        n_oob_landmarks = 0

        if transform and input_size:
            output_size = transform.output_size(input_size)
            out_w, out_h = output_size

            for lf in labels.labeled_frames:
                if lf.video is video:
                    n_instances += len(lf.instances)

                    for instance in lf.instances:
                        points = instance.numpy(invisible_as_nan=False)
                        if points.size == 0:
                            continue

                        # Transform points to output space
                        transformed = transform.apply_to_points(points, input_size)

                        # Check for OOB (excluding NaN points)
                        valid_mask = ~np.isnan(transformed).any(axis=-1)
                        valid_points = transformed[valid_mask]

                        if valid_points.size > 0:
                            oob = (
                                (valid_points[:, 0] < 0)
                                | (valid_points[:, 0] >= out_w)
                                | (valid_points[:, 1] < 0)
                                | (valid_points[:, 1] >= out_h)
                            )
                            n_oob_landmarks += np.sum(oob)
        else:
            for lf in labels.labeled_frames:
                if lf.video is video:
                    n_instances += len(lf.instances)

        video_info["n_instances"] = n_instances
        summary["total_instances"] += n_instances

        # Warning: Landmarks go out of bounds
        if n_oob_landmarks > 0:
            summary["warnings"].append(
                f"Video {video_idx}: {n_oob_landmarks} landmark(s) will be "
                f"outside frame bounds"
            )

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
