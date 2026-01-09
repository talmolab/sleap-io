"""Synthetic fixtures for merge/matching tests.

This module provides factory functions to create test data for:
- PKG.SLP / source_video testing
- Duplicate video prevention
- Cross-platform path matching
- ImageVideo merging
- Multi-resolution video disambiguation

These fixtures address critical coverage gaps identified in the merge safety
investigation.
All fixtures create synthetic in-memory objects that don't require external files.
"""

import numpy as np

from sleap_io import Labels, Skeleton, Video
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame


def make_video(
    filename: str = "/data/video.mp4",
    shape: tuple = (100, 480, 640, 1),
    source_video: Video | None = None,
) -> Video:
    """Create a Video object with specified metadata.

    Args:
        filename: Path string for the video.
        shape: Video shape as (frames, height, width, channels).
        source_video: Optional source video for embedded video scenarios.

    Returns:
        Video object with backend_metadata set.
    """
    video = Video(filename=filename, source_video=source_video, open_backend=False)
    video.backend_metadata["shape"] = shape
    return video


def make_labels(
    video: Video,
    skeleton: Skeleton | None = None,
    frame_indices: list[int] | None = None,
    add_instances: bool = True,
    predicted: bool = False,
) -> Labels:
    """Create Labels with optional frames and instances.

    Args:
        video: Video to use for labeled frames.
        skeleton: Skeleton to use. If None, creates 2-node skeleton.
        frame_indices: List of frame indices to create. Defaults to [0, 10, 20].
        add_instances: Whether to add instances to frames.
        predicted: If True, create PredictedInstances instead of Instances.

    Returns:
        Labels object with specified configuration.
    """
    if skeleton is None:
        skeleton = Skeleton(["head", "tail"])
    if frame_indices is None:
        frame_indices = [0, 10, 20]

    labels = Labels(skeletons=[skeleton])
    labels.videos = [video]

    for idx in frame_indices:
        frame = LabeledFrame(video=video, frame_idx=idx)
        if add_instances:
            points = np.random.rand(len(skeleton.nodes), 2) * 100
            if predicted:
                inst = PredictedInstance.from_numpy(
                    points, skeleton=skeleton, score=0.9
                )
            else:
                inst = Instance.from_numpy(points, skeleton=skeleton)
            frame.instances = [inst]
        labels.append(frame)

    return labels


# =============================================================================
# PKG.SLP / source_video fixtures (UC3)
# =============================================================================


def make_external_video_labels(
    video_path: str = "/data/recordings/video.mp4",
    shape: tuple = (100, 480, 640, 1),
) -> Labels:
    """Create Labels with an external video reference.

    This represents a typical SLP file with external video references,
    used as the base for merging predictions from PKG.SLP files.
    """
    skeleton = Skeleton(["head", "tail"])
    video = make_video(filename=video_path, shape=shape)
    return make_labels(video, skeleton=skeleton, frame_indices=[0, 10])


def make_pkg_slp_predictions(
    original_video_path: str = "/data/recordings/video.mp4",
    pkg_path: str = "predictions.pkg.slp",
    shape: tuple = (100, 480, 640, 1),
    skeleton: Skeleton | None = None,
) -> Labels:
    """Create Labels simulating predictions from a PKG.SLP file.

    PKG.SLP files embed video frames and store a source_video reference
    pointing to the original external video. This is critical for merging
    predictions back into the original Labels.

    Args:
        original_video_path: Path to the original external video.
        pkg_path: Path to the PKG.SLP file (typically used as video filename).
        shape: Video shape.
        skeleton: Skeleton to use. Must match base labels skeleton.

    Returns:
        Labels with embedded video that has source_video set.
    """
    if skeleton is None:
        skeleton = Skeleton(["head", "tail"])

    # Create source video reference (points to original file)
    source = make_video(filename=original_video_path, shape=shape)

    # Create embedded video (filename is the PKG.SLP path, has source_video)
    embedded = make_video(filename=pkg_path, shape=shape, source_video=source)

    return make_labels(
        embedded, skeleton=skeleton, frame_indices=[0, 10], predicted=True
    )


# =============================================================================
# Cross-platform path fixtures (UC2)
# =============================================================================


def make_cross_platform_labels_pair(
    windows_path: str = r"C:\data\video.mp4",
    linux_path: str = "/home/user/data/video.mp4",
    shape: tuple = (100, 480, 640, 1),
) -> tuple[Labels, Labels]:
    """Create two Labels with different path formats but same basename.

    This simulates the common scenario where labeling is done on Windows
    and training/inference is done on Linux (or vice versa).

    Returns:
        Tuple of (windows_labels, linux_labels).
    """
    skeleton = Skeleton(["head", "tail"])

    windows_video = make_video(filename=windows_path, shape=shape)
    linux_video = make_video(filename=linux_path, shape=shape)

    windows_labels = make_labels(
        windows_video, skeleton=skeleton, frame_indices=[0, 10]
    )
    linux_labels = make_labels(
        linux_video, skeleton=skeleton, frame_indices=[0, 10], predicted=True
    )

    return windows_labels, linux_labels


# =============================================================================
# Duplicate video fixtures (UC6)
# =============================================================================


def make_duplicate_video_scenario(
    video_path: str = "/data/video.mp4",
    shape: tuple = (100, 480, 640, 1),
) -> tuple[Video, Video]:
    """Create two Video objects pointing to the same file.

    This tests the scenario where a user accidentally adds the same video
    twice, creating duplicate Video objects with different identities
    but pointing to the same underlying file.

    Returns:
        Tuple of (video1, video2) - different objects, same path.
    """
    video1 = make_video(filename=video_path, shape=shape)
    video2 = make_video(filename=video_path, shape=shape)
    return video1, video2


def make_relative_vs_absolute_videos(
    base_path: str = "/data/recordings/video.mp4",
    relative_path: str = "recordings/video.mp4",
    shape: tuple = (100, 480, 640, 1),
) -> tuple[Video, Video]:
    """Create videos with relative and absolute paths to same file.

    This tests path resolution when one path is absolute and another
    is relative but both point to the same file.

    Returns:
        Tuple of (absolute_video, relative_video).
    """
    absolute_video = make_video(filename=base_path, shape=shape)
    relative_video = make_video(filename=relative_path, shape=shape)
    return absolute_video, relative_video


# =============================================================================
# ImageVideo fixtures (UC4)
# =============================================================================


def make_imagevideo(
    image_paths: list[str] | None = None,
    shape: tuple | None = None,
) -> Video:
    """Create a Video simulating an ImageVideo backend.

    Args:
        image_paths: List of image file paths. Defaults to 5 images.
        shape: Video shape. Defaults to (n_images, 480, 640, 3).

    Returns:
        Video with list of filenames (ImageVideo pattern).
    """
    if image_paths is None:
        image_paths = [f"/data/img_{i:03d}.jpg" for i in range(5)]
    if shape is None:
        shape = (len(image_paths), 480, 640, 3)

    video = Video(filename=image_paths, open_backend=False)
    video.backend_metadata["shape"] = shape
    return video


def make_overlapping_imagevideos(
    overlap_start: int = 2,
    overlap_end: int = 4,
    set1_size: int = 5,
    set2_size: int = 5,
) -> tuple[Video, Video]:
    """Create two ImageVideos with overlapping frame indices.

    This tests the IMAGE_DEDUP matcher which must build frame_idx_map
    for merging annotations on overlapping image sequences.

    Args:
        overlap_start: First overlapping image index in set1.
        overlap_end: Last overlapping image index in set1 (exclusive).
        set1_size: Number of images in first set.
        set2_size: Number of images in second set.

    Returns:
        Tuple of (video1, video2) with overlapping images.

    Example:
        overlap_start=2, overlap_end=4, set1_size=5, set2_size=5
        video1: [img_000, img_001, img_002, img_003, img_004]
        video2: [img_002, img_003, img_005, img_006, img_007]
        Overlap: img_002 (idx 2 in v1, idx 0 in v2)
                 img_003 (idx 3 in v1, idx 1 in v2)
    """
    # Set 1: sequential images
    set1_paths = [f"/data/img_{i:03d}.jpg" for i in range(set1_size)]

    # Set 2: overlapping images + new images
    overlap_paths = [
        f"/data/img_{i:03d}.jpg" for i in range(overlap_start, overlap_end)
    ]
    new_paths = [
        f"/data/img_{i:03d}.jpg"
        for i in range(set1_size, set1_size + set2_size - len(overlap_paths))
    ]
    set2_paths = overlap_paths + new_paths

    video1 = make_imagevideo(set1_paths)
    video2 = make_imagevideo(set2_paths)

    return video1, video2


# =============================================================================
# Multi-resolution disambiguation fixtures (UC5)
# =============================================================================


def make_same_resolution_videos(
    n_videos: int = 3,
    shape: tuple = (100, 480, 640, 1),
    basename_pattern: str = "camera_{}.mp4",
) -> list[Video]:
    """Create multiple videos with identical resolution.

    This tests disambiguation when multiple videos match by content
    (same shape and backend type). The AUTO matcher must use basename
    or other criteria to correctly identify the target video.

    Args:
        n_videos: Number of videos to create.
        shape: Video shape (same for all).
        basename_pattern: Pattern for generating filenames.

    Returns:
        List of Video objects with identical shapes.
    """
    videos = []
    for i in range(n_videos):
        path = f"/data/exp{i}/" + basename_pattern.format(i)
        video = make_video(filename=path, shape=shape)
        videos.append(video)
    return videos


def make_ambiguous_basename_scenario(
    shape: tuple = (100, 480, 640, 1),
) -> tuple[list[Video], Video]:
    """Create scenario where basename matching is ambiguous.

    Multiple videos have the same basename in different directories,
    all with the same shape. This tests whether the matcher can
    disambiguate using parent directory or other criteria.

    Returns:
        Tuple of (base_videos, prediction_video) where prediction_video
        has basename matching multiple base_videos.
    """
    base_videos = [
        make_video(filename="/data/exp1/fly.mp4", shape=shape),
        make_video(filename="/data/exp2/fly.mp4", shape=shape),
        make_video(filename="/data/exp3/fly.mp4", shape=shape),
    ]

    # Prediction for exp2's fly.mp4 - should match base_videos[1]
    prediction_video = make_video(filename="/predictions/exp2/fly.mp4", shape=shape)

    return base_videos, prediction_video
