"""This module handles direct I/O operations for working with .slp files."""

from __future__ import annotations
import numpy as np
import h5py
import simplejson as json
from typing import Union
from sleap_io import (
    Video,
    Skeleton,
    Edge,
    Symmetry,
    Node,
    Track,
    SuggestionFrame,
    Instance,
    PredictedInstance,
    LabeledFrame,
    Labels,
    Camera,
    CameraGroup,
    InstanceGroup,
    FrameGroup,
    RecordingSession,
)
from sleap_io.io.video_reading import VideoBackend, ImageVideo, MediaVideo, HDF5Video
from sleap_io.io.utils import read_hdf5_attrs, read_hdf5_dataset, is_file_accessible
from enum import IntEnum
from pathlib import Path
import imageio.v3 as iio
import sys

try:
    import cv2
except ImportError:
    pass


class InstanceType(IntEnum):
    """Enumeration of instance types to integers."""

    USER = 0
    PREDICTED = 1


def sanitize_filename(
    filename: str | Path | list[str] | list[Path],
) -> str | list[str]:
    """Sanitize a filename to a canonical posix-compatible format.

    Args:
        filename: A string or `Path` object or list of either to sanitize.

    Returns:
        A sanitized filename as a string (or list of strings if a list was provided)
        with forward slashes and posix-formatted.
    """
    if isinstance(filename, list):
        return [sanitize_filename(f) for f in filename]
    return Path(filename).as_posix().replace("\\", "/")


def make_video(
    labels_path: str,
    video_json: dict,
    open_backend: bool = True,
) -> Video:
    """Create a `Video` object from a JSON dictionary.

    Args:
        labels_path: A string path to the SLEAP labels file.
        video_json: A dictionary containing the video metadata.
        open_backend: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).
    """
    backend_metadata = video_json["backend"]
    video_path = backend_metadata["filename"]

    # Marker for embedded videos.
    source_video = None
    is_embedded = False
    if video_path == ".":
        video_path = labels_path
        is_embedded = True

    # Basic path resolution.
    video_path = Path(sanitize_filename(video_path))

    if is_embedded:
        # Try to recover the source video.
        with h5py.File(labels_path, "r") as f:
            dataset = backend_metadata["dataset"]
            if dataset.endswith("/video"):
                dataset = dataset[:-6]
            if dataset in f and "source_video" in f[dataset]:
                source_video_json = json.loads(
                    f[f"{dataset}/source_video"].attrs["json"]
                )
                source_video = make_video(
                    labels_path,
                    source_video_json,
                    open_backend=open_backend,
                )

    backend = None
    if open_backend:
        try:
            if not is_file_accessible(video_path):
                # Check for the same filename in the same directory as the labels file.
                candidate_video_path = Path(labels_path).parent / video_path.name
                if is_file_accessible(candidate_video_path):
                    video_path = candidate_video_path
                else:
                    # TODO (TP): Expand capabilities of path resolution to support more
                    # complex path finding strategies.
                    pass
        except (OSError, PermissionError, FileNotFoundError):
            pass

        # Convert video path to string.
        video_path = video_path.as_posix()

        if "filenames" in backend_metadata:
            # This is an ImageVideo.
            # TODO: Path resolution.
            video_path = backend_metadata["filenames"]
            video_path = [Path(sanitize_filename(p)) for p in video_path]

        try:
            grayscale = None
            if "grayscale" in backend_metadata:
                grayscale = backend_metadata["grayscale"]
            elif "shape" in backend_metadata:
                grayscale = backend_metadata["shape"][-1] == 1
            backend = VideoBackend.from_filename(
                video_path,
                dataset=backend_metadata.get("dataset", None),
                grayscale=grayscale,
                input_format=backend_metadata.get("input_format", None),
            )
        except Exception:
            backend = None

    return Video(
        filename=video_path,
        backend=backend,
        backend_metadata=backend_metadata,
        source_video=source_video,
        open_backend=open_backend,
    )


def read_videos(labels_path: str, open_backend: bool = True) -> list[Video]:
    """Read `Video` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        open_backend: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).

    Returns:
        A list of `Video` objects.
    """
    videos = []
    videos_metadata = read_hdf5_dataset(labels_path, "videos_json")
    for video_data in videos_metadata:
        video_json = json.loads(video_data)
        video = make_video(labels_path, video_json, open_backend=open_backend)
        videos.append(video)
    return videos


def video_to_dict(video: Video) -> dict:
    """Convert a `Video` object to a JSON-compatible dictionary.

    Args:
        video: A `Video` object to convert.

    Returns:
        A dictionary containing the video metadata.
    """
    video_filename = sanitize_filename(video.filename)
    if video.backend is None:
        return {"filename": video_filename, "backend": video.backend_metadata}

    if type(video.backend) == MediaVideo:
        return {
            "filename": video_filename,
            "backend": {
                "type": "MediaVideo",
                "shape": video.shape,
                "filename": video_filename,
                "grayscale": video.grayscale,
                "bgr": True,
                "dataset": "",
                "input_format": "",
            },
        }

    elif type(video.backend) == HDF5Video:
        return {
            "filename": video_filename,
            "backend": {
                "type": "HDF5Video",
                "shape": video.shape,
                "filename": (
                    "." if video.backend.has_embedded_images else video_filename
                ),
                "dataset": video.backend.dataset,
                "input_format": video.backend.input_format,
                "convert_range": False,
                "has_embedded_images": video.backend.has_embedded_images,
                "grayscale": video.grayscale,
            },
        }

    elif type(video.backend) == ImageVideo:
        return {
            "filename": video_filename,
            "backend": {
                "type": "ImageVideo",
                "shape": video.shape,
                "filename": sanitize_filename(video.backend.filename[0]),
                "filenames": sanitize_filename(video.backend.filename),
                "dataset": video.backend_metadata.get("dataset", None),
                "grayscale": video.grayscale,
                "input_format": video.backend_metadata.get("input_format", None),
            },
        }


def embed_video(
    labels_path: str,
    video: Video,
    group: str,
    frame_inds: list[int],
    image_format: str = "png",
    fixed_length: bool = True,
) -> Video:
    """Embed frames of a video in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        video: A `Video` object to embed in the labels file.
        group: The name of the group to store the embedded video in. Image data will be
            stored in a dataset named `{group}/video`. Frame indices will be stored
            in a data set named `{group}/frame_numbers`.
        frame_inds: A list of frame indices to embed.
        image_format: The image format to use for embedding. Valid formats are "png"
            (the default), "jpg" or "hdf5".
        fixed_length: If `True` (the default), the embedded images will be padded to the
            length of the largest image. If `False`, the images will be stored as
            variable length, which is smaller but may not be supported by all readers.

    Returns:
        An embedded `Video` object.

        If the video is already embedded, the original video will be returned. If not,
        a new `Video` object will be created with the embedded data.
    """
    # Load the image data and optionally encode it.
    imgs_data = []
    for frame_idx in frame_inds:
        frame = video[frame_idx]

        if image_format == "hdf5":
            img_data = frame
        else:
            if "cv2" in sys.modules:
                img_data = np.squeeze(
                    cv2.imencode("." + image_format, frame)[1]
                ).astype("int8")
            else:
                if frame.shape[-1] == 1:
                    frame = frame.squeeze(axis=-1)
                img_data = np.frombuffer(
                    iio.imwrite("<bytes>", frame, extension="." + image_format),
                    dtype="int8",
                )

        imgs_data.append(img_data)

    # Write the image data to the labels file.
    with h5py.File(labels_path, "a") as f:
        if image_format == "hdf5":
            f.create_dataset(
                f"{group}/video", data=imgs_data, compression="gzip", chunks=True
            )
        else:
            if fixed_length:
                img_bytes_len = 0
                for img in imgs_data:
                    img_bytes_len = max(img_bytes_len, len(img))
                ds = f.create_dataset(
                    f"{group}/video",
                    shape=(len(imgs_data), img_bytes_len),
                    dtype="int8",
                    compression="gzip",
                )
                for i, img in enumerate(imgs_data):
                    ds[i, : len(img)] = img
            else:
                ds = f.create_dataset(
                    f"{group}/video",
                    shape=(len(imgs_data),),
                    dtype=h5py.special_dtype(vlen=np.dtype("int8")),
                )
                for i, img in enumerate(imgs_data):
                    ds[i] = img

        # Store metadata.
        ds.attrs["format"] = image_format
        video_shape = video.shape
        (
            ds.attrs["frames"],
            ds.attrs["height"],
            ds.attrs["width"],
            ds.attrs["channels"],
        ) = video_shape

        # Store frame indices.
        f.create_dataset(f"{group}/frame_numbers", data=frame_inds)

        # Store source video.
        if video.source_video is not None:
            # If this is already an embedded dataset, retain the previous source video.
            source_video = video.source_video
        else:
            source_video = video

        # Create a new video object with the embedded data.
        embedded_video = Video(
            filename=labels_path,
            backend=VideoBackend.from_filename(
                labels_path,
                dataset=f"{group}/video",
                grayscale=video.grayscale,
                keep_open=False,
            ),
            source_video=source_video,
        )

        grp = f.require_group(f"{group}/source_video")
        grp.attrs["json"] = json.dumps(
            video_to_dict(source_video), separators=(",", ":")
        )

    return embedded_video


def embed_frames(
    labels_path: str,
    labels: Labels,
    embed: list[tuple[Video, int]],
    image_format: str = "png",
):
    """Embed frames in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to embed in the labels file.
        embed: A list of tuples of `(video, frame_idx)` specifying the frames to embed.
        image_format: The image format to use for embedding. Valid formats are "png"
            (the default), "jpg" or "hdf5".

    Notes:
        This function will embed the frames in the labels file and update the `Videos`
        and `Labels` objects in place.
    """
    to_embed_by_video = {}
    for video, frame_idx in embed:
        if video not in to_embed_by_video:
            to_embed_by_video[video] = []
        to_embed_by_video[video].append(frame_idx)

    for video in to_embed_by_video:
        to_embed_by_video[video] = np.unique(to_embed_by_video[video]).tolist()

    replaced_videos = {}
    for video, frame_inds in to_embed_by_video.items():
        video_ind = labels.videos.index(video)
        embedded_video = embed_video(
            labels_path,
            video,
            group=f"video{video_ind}",
            frame_inds=frame_inds,
            image_format=image_format,
        )

        labels.videos[video_ind] = embedded_video
        replaced_videos[video] = embedded_video

    if len(replaced_videos) > 0:
        labels.replace_videos(video_map=replaced_videos)


def embed_videos(
    labels_path: str, labels: Labels, embed: bool | str | list[tuple[Video, int]]
):
    """Embed videos in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file to save.
        labels: A `Labels` object to save.
        embed: Frames to embed in the saved labels file. One of `None`, `True`,
            `"all"`, `"user"`, `"suggestions"`, `"user+suggestions"`, `"source"` or list
            of tuples of `(video, frame_idx)`.

            If `None` is specified (the default) and the labels contains embedded
            frames, those embedded frames will be re-saved to the new file.

            If `True` or `"all"`, all labeled frames and suggested frames will be
            embedded.

            If `"source"` is specified, no images will be embedded and the source video
            will be restored if available.

            This argument is only valid for the SLP backend.
    """
    if embed is True:
        embed = "all"
    if embed == "user":
        embed = [(lf.video, lf.frame_idx) for lf in labels.user_labeled_frames]
    elif embed == "suggestions":
        embed = [(sf.video, sf.frame_idx) for sf in labels.suggestions]
    elif embed == "user+suggestions":
        embed = [(lf.video, lf.frame_idx) for lf in labels.user_labeled_frames]
        embed += [(sf.video, sf.frame_idx) for sf in labels.suggestions]
    elif embed == "all":
        embed = [(lf.video, lf.frame_idx) for lf in labels]
        embed += [(sf.video, sf.frame_idx) for sf in labels.suggestions]
    elif embed == "source":
        embed = []
    elif isinstance(embed, list):
        embed = embed
    else:
        raise ValueError(f"Invalid value for embed: {embed}")

    embed_frames(labels_path, labels, embed)


def write_videos(labels_path: str, videos: list[Video], restore_source: bool = False):
    """Write video metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects to store the metadata for.
        restore_source: If `True`, restore source videos if available and will not
            re-embed the embedded images. If `False` (the default), will re-embed images
            that were previously embedded.
    """
    video_jsons = []
    for video_ind, video in enumerate(videos):
        if type(video.backend) == HDF5Video and video.backend.has_embedded_images:
            if restore_source:
                video = video.source_video
            else:
                # If the video has embedded images, embed them images again if we haven't
                # already.
                already_embedded = False
                if Path(labels_path).exists():
                    with h5py.File(labels_path, "r") as f:
                        already_embedded = f"video{video_ind}/video" in f

                if not already_embedded:
                    video = embed_video(
                        labels_path,
                        video,
                        group=f"video{video_ind}",
                        frame_inds=video.backend.source_inds,
                        image_format=video.backend.image_format,
                    )

        video_json = video_to_dict(video)

        video_jsons.append(np.bytes_(json.dumps(video_json, separators=(",", ":"))))

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("videos_json", data=video_jsons, maxshape=(None,))


def read_tracks(labels_path: str) -> list[Track]:
    """Read `Track` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `Track` objects.
    """
    tracks = [json.loads(x) for x in read_hdf5_dataset(labels_path, "tracks_json")]
    track_objects = []
    for track in tracks:
        track_objects.append(Track(name=track[1]))
    return track_objects


def write_tracks(labels_path: str, tracks: list[Track]):
    """Write track metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        tracks: A list of `Track` objects to store the metadata for.
    """
    # TODO: Add support for track metadata like spawned on frame.
    SPAWNED_ON = 0
    tracks_json = [
        np.bytes_(json.dumps([SPAWNED_ON, track.name], separators=(",", ":")))
        for track in tracks
    ]
    with h5py.File(labels_path, "a") as f:
        f.create_dataset("tracks_json", data=tracks_json, maxshape=(None,))


def read_suggestions(labels_path: str, videos: list[Video]) -> list[SuggestionFrame]:
    """Read `SuggestionFrame` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects.

    Returns:
        A list of `SuggestionFrame` objects.
    """
    try:
        suggestions = read_hdf5_dataset(labels_path, "suggestions_json")
    except KeyError:
        return []
    suggestions = [json.loads(x) for x in suggestions]
    suggestions_objects = []
    for suggestion in suggestions:
        suggestions_objects.append(
            SuggestionFrame(
                video=videos[int(suggestion["video"])],
                frame_idx=suggestion["frame_idx"],
            )
        )
    return suggestions_objects


def write_suggestions(
    labels_path: str, suggestions: list[SuggestionFrame], videos: list[Video]
):
    """Write track metadata to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        suggestions: A list of `SuggestionFrame` objects to store the metadata for.
        videos: A list of `Video` objects.
    """
    GROUP = 0  # TODO: Handle storing extraneous metadata.
    suggestions_json = []
    for suggestion in suggestions:
        suggestion_dict = {
            "video": str(videos.index(suggestion.video)),
            "frame_idx": suggestion.frame_idx,
            "group": GROUP,
        }
        suggestion_json = np.bytes_(json.dumps(suggestion_dict, separators=(",", ":")))
        suggestions_json.append(suggestion_json)

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("suggestions_json", data=suggestions_json, maxshape=(None,))


def read_metadata(labels_path: str) -> dict:
    """Read metadata from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A dict containing the metadata from a SLEAP labels file.
    """
    md = read_hdf5_attrs(labels_path, "metadata", "json")
    return json.loads(md.decode())


def read_skeletons(labels_path: str) -> list[Skeleton]:
    """Read `Skeleton` dataset from a SLEAP labels file.

    Args:
        labels_path: A string that contains the path to the labels file.

    Returns:
        A list of `Skeleton` objects.
    """
    metadata = read_metadata(labels_path)

    # Get node names. This is a superset of all nodes across all skeletons. Note that
    # node ordering is specific to each skeleton, so we'll need to fix this afterwards.
    node_names = [x["name"] for x in metadata["nodes"]]

    skeleton_objects = []
    for skel in metadata["skeletons"]:
        # Parse out the cattr-based serialization stuff from the skeleton links.
        edge_inds, symmetry_inds = [], []
        for link in skel["links"]:
            if "py/reduce" in link["type"]:
                edge_type = link["type"]["py/reduce"][1]["py/tuple"][0]
            else:
                edge_type = link["type"]["py/id"]

            if edge_type == 1:  # 1 -> real edge, 2 -> symmetry edge
                edge_inds.append((link["source"], link["target"]))

            elif edge_type == 2:
                symmetry_inds.append((link["source"], link["target"]))

        # Re-index correctly.
        skeleton_node_inds = [node["id"] for node in skel["nodes"]]
        sorted_node_names = [node_names[i] for i in skeleton_node_inds]

        # Create nodes.
        nodes = []
        for name in sorted_node_names:
            nodes.append(Node(name=name))

        # Create edges.
        edge_inds = [
            (skeleton_node_inds.index(s), skeleton_node_inds.index(d))
            for s, d in edge_inds
        ]
        edges = []
        for edge in edge_inds:
            edges.append(Edge(source=nodes[edge[0]], destination=nodes[edge[1]]))

        # Create symmetries.
        symmetry_inds = [
            (skeleton_node_inds.index(s), skeleton_node_inds.index(d))
            for s, d in symmetry_inds
        ]
        symmetries = []
        for symmetry in symmetry_inds:
            symmetries.append(Symmetry([nodes[symmetry[0]], nodes[symmetry[1]]]))

        # Create the full skeleton.
        skel = Skeleton(
            nodes=nodes, edges=edges, symmetries=symmetries, name=skel["graph"]["name"]
        )
        skeleton_objects.append(skel)
    return skeleton_objects


def serialize_skeletons(skeletons: list[Skeleton]) -> tuple[list[dict], list[dict]]:
    """Serialize a list of `Skeleton` objects to JSON-compatible dicts.

    Args:
        skeletons: A list of `Skeleton` objects.

    Returns:
        A tuple of `nodes_dicts, skeletons_dicts`.

        `nodes_dicts` is a list of dicts containing the nodes in all the skeletons.

        `skeletons_dicts` is a list of dicts containing the skeletons.

    Notes:
        This function attempts to replicate the serialization of skeletons in legacy
        SLEAP which relies on a combination of networkx's graph serialization and our
        own metadata used to store nodes and edges independent of the graph structure.

        However, because sleap-io does not currently load in the legacy metadata, this
        function will not produce byte-level compatible serialization with legacy
        formats, even though the ordering and all attributes of nodes and edges should
        match up.
    """
    # Create global list of nodes with all nodes from all skeletons.
    nodes_dicts = []
    node_to_id = {}
    for skeleton in skeletons:
        for node in skeleton.nodes:
            if node not in node_to_id:
                # Note: This ID is not the same as the node index in the skeleton in
                # legacy SLEAP, but we do not retain this information in the labels, so
                # IDs will be different.
                #
                # The weight is also kept fixed here, but technically this is not
                # modified or used in legacy SLEAP either.
                #
                # TODO: Store legacy metadata in labels to get byte-level compatibility?
                node_to_id[node] = len(node_to_id)
                nodes_dicts.append({"name": node.name, "weight": 1.0})

    skeletons_dicts = []
    for skeleton in skeletons:
        # Build links dicts for normal edges.
        edges_dicts = []
        for edge_ind, edge in enumerate(skeleton.edges):
            if edge_ind == 0:
                edge_type = {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [1]},  # 1 = real edge, 2 = symmetry edge
                    ]
                }
            else:
                edge_type = {"py/id": 1}

            edges_dicts.append(
                {
                    # Note: Insert idx is not the same as the edge index in the skeleton
                    # in legacy SLEAP.
                    "edge_insert_idx": edge_ind,
                    "key": 0,  # Always 0.
                    "source": node_to_id[edge.source],
                    "target": node_to_id[edge.destination],
                    "type": edge_type,
                }
            )

        # Build links dicts for symmetry edges.
        for symmetry_ind, symmetry in enumerate(skeleton.symmetries):
            if symmetry_ind == 0:
                edge_type = {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [2]},  # 1 = real edge, 2 = symmetry edge
                    ]
                }
            else:
                edge_type = {"py/id": 2}

            src, dst = tuple(symmetry.nodes)
            edges_dicts.append(
                {
                    "key": 0,
                    "source": node_to_id[src],
                    "target": node_to_id[dst],
                    "type": edge_type,
                }
            )

        # Create skeleton dict.
        skeletons_dicts.append(
            {
                "directed": True,
                "graph": {
                    "name": skeleton.name,
                    "num_edges_inserted": len(skeleton.edges),
                },
                "links": edges_dicts,
                "multigraph": True,
                # In the order in Skeleton.nodes and must match up with nodes_dicts.
                "nodes": [{"id": node_to_id[node]} for node in skeleton.nodes],
            }
        )

    return skeletons_dicts, nodes_dicts


def write_metadata(labels_path: str, labels: Labels):
    """Write metadata to a SLEAP labels file.

    This function will write the skeletons and provenance for the labels.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.

    See also: serialize_skeletons
    """
    skeletons_dicts, nodes_dicts = serialize_skeletons(labels.skeletons)

    md = {
        "version": "2.0.0",
        "skeletons": skeletons_dicts,
        "nodes": nodes_dicts,
        "videos": [],
        "tracks": [],
        "suggestions": [],  # TODO: Handle suggestions metadata.
        "negative_anchors": {},
        "provenance": labels.provenance,
    }

    # Custom encoding.
    for k in md["provenance"]:
        if isinstance(md["provenance"][k], Path):
            # Path -> str
            md["provenance"][k] = md["provenance"][k].as_posix()

    with h5py.File(labels_path, "a") as f:
        grp = f.require_group("metadata")
        grp.attrs["format_id"] = 1.2
        grp.attrs["json"] = np.bytes_(json.dumps(md, separators=(",", ":")))


def read_points(labels_path: str) -> np.ndarray:
    """Read points dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A structured array of point data.
    """
    pts = read_hdf5_dataset(labels_path, "points")
    return pts


def read_pred_points(labels_path: str) -> np.ndarray:
    """Read predicted points dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A structured array of predicted point data.
    """
    pred_pts = read_hdf5_dataset(labels_path, "pred_points")
    return pred_pts


def read_instances(
    labels_path: str,
    skeletons: list[Skeleton],
    tracks: list[Track],
    points: np.ndarray,
    pred_points: np.ndarray,
    format_id: float,
) -> list[Union[Instance, PredictedInstance]]:
    """Read `Instance` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        skeletons: A list of `Skeleton` objects (see `read_skeletons`).
        tracks: A list of `Track` objects (see `read_tracks`).
        points: A structured array of point data (see `read_points`).
        pred_points: A structured array of predicted point data (see
            `read_pred_points`).
        format_id: The format version identifier used to specify the format of the input
            file.

    Returns:
        A list of `Instance` and/or `PredictedInstance` objects.
    """
    instances_data = read_hdf5_dataset(labels_path, "instances")

    instances = {}
    from_predicted_pairs = []
    for instance_data in instances_data:
        if format_id < 1.2:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
            ) = instance_data
            tracking_score = np.zeros_like(instance_score)
        elif format_id >= 1.2:
            (
                instance_id,
                instance_type,
                frame_id,
                skeleton_id,
                track_id,
                from_predicted,
                instance_score,
                point_id_start,
                point_id_end,
                tracking_score,
            ) = instance_data

        skeleton = skeletons[skeleton_id]
        track = tracks[track_id] if track_id >= 0 else None

        if instance_type == InstanceType.USER:
            pts_data = points[point_id_start:point_id_end]
            inst = Instance(
                np.column_stack([pts_data["x"], pts_data["y"]]),
                skeleton=skeleton,
                track=track,
                tracking_score=tracking_score,
            )
            inst.points["visible"] = pts_data["visible"]
            inst.points["complete"] = pts_data["complete"]
            instances[instance_id] = inst

        elif instance_type == InstanceType.PREDICTED:
            pts_data = pred_points[point_id_start:point_id_end]
            inst = PredictedInstance(
                np.column_stack([pts_data["x"], pts_data["y"]]),
                skeleton=skeleton,
                track=track,
                score=instance_score,
                tracking_score=tracking_score,
            )
            inst.points["score"] = pts_data["score"]
            inst.points["visible"] = pts_data["visible"]
            inst.points["complete"] = pts_data["complete"]
            instances[instance_id] = inst

        if from_predicted >= 0:
            from_predicted_pairs.append((instance_id, from_predicted))

    # Link instances based on from_predicted field.
    for instance_id, from_predicted in from_predicted_pairs:
        instances[instance_id].from_predicted = instances[from_predicted]

    # Convert instances back to list.
    instances = list(instances.values())

    return instances


def write_lfs(labels_path: str, labels: Labels):
    """Write labeled frames, instances and points to a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        labels: A `Labels` object to store the metadata for.
    """
    # We store the data in structured arrays for performance, so we first define the
    # dtype fields.
    instance_dtype = np.dtype(
        [
            ("instance_id", "i8"),
            ("instance_type", "u1"),
            ("frame_id", "u8"),
            ("skeleton", "u4"),
            ("track", "i4"),
            ("from_predicted", "i8"),
            ("score", "f4"),
            ("point_id_start", "u8"),
            ("point_id_end", "u8"),
            ("tracking_score", "f4"),  # FORMAT_ID >= 1.2
        ]
    )
    frame_dtype = np.dtype(
        [
            ("frame_id", "u8"),
            ("video", "u4"),
            ("frame_idx", "u8"),
            ("instance_id_start", "u8"),
            ("instance_id_end", "u8"),
        ]
    )
    point_dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")]
    )
    predicted_point_dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?"), ("score", "f8")]
    )

    # Next, we extract the data from the labels object into lists with the same fields.
    frames, instances, points, predicted_points, to_link = [], [], [], [], []
    inst_to_id = {}
    for lf in labels:
        frame_id = len(frames)
        instance_id_start = len(instances)
        for inst in lf:
            instance_id = len(instances)
            inst_to_id[id(inst)] = instance_id
            skeleton_id = labels.skeletons.index(inst.skeleton)
            track = labels.tracks.index(inst.track) if inst.track else -1
            from_predicted = -1
            if inst.from_predicted:
                to_link.append((instance_id, inst.from_predicted))
            score = 0.0

            if type(inst) == Instance:
                instance_type = InstanceType.USER
                tracking_score = inst.tracking_score
                point_id_start = len(points)

                for pt in inst.points:
                    points.append(
                        [pt["xy"][0], pt["xy"][1], pt["visible"], pt["complete"]]
                    )

                point_id_end = len(points)

            elif type(inst) == PredictedInstance:
                instance_type = InstanceType.PREDICTED
                score = inst.score
                tracking_score = inst.tracking_score
                point_id_start = len(predicted_points)

                for pt in inst.points:
                    predicted_points.append(
                        [
                            pt["xy"][0],
                            pt["xy"][1],
                            pt["visible"],
                            pt["complete"],
                            pt["score"],
                        ]
                    )

                point_id_end = len(predicted_points)

            else:
                raise ValueError(f"Unknown instance type: {type(inst)}")

            instances.append(
                [
                    instance_id,
                    int(instance_type),
                    frame_id,
                    skeleton_id,
                    track,
                    from_predicted,
                    score,
                    point_id_start,
                    point_id_end,
                    tracking_score,
                ]
            )

        instance_id_end = len(instances)

        frames.append(
            [
                frame_id,
                labels.videos.index(lf.video),
                lf.frame_idx,
                instance_id_start,
                instance_id_end,
            ]
        )

    # Link instances based on from_predicted field.
    for instance_id, from_predicted in to_link:
        # Source instance may be missing if predictions were removed from the labels, in
        # which case, remove the link.
        instances[instance_id][5] = inst_to_id.get(id(from_predicted), -1)

    # Create structured arrays.
    points = np.array([tuple(x) for x in points], dtype=point_dtype)
    predicted_points = np.array(
        [tuple(x) for x in predicted_points], dtype=predicted_point_dtype
    )
    instances = np.array([tuple(x) for x in instances], dtype=instance_dtype)
    frames = np.array([tuple(x) for x in frames], dtype=frame_dtype)

    # Write to file.
    with h5py.File(labels_path, "a") as f:
        f.create_dataset("points", data=points, dtype=points.dtype)
        f.create_dataset(
            "pred_points",
            data=predicted_points,
            dtype=predicted_points.dtype,
        )
        f.create_dataset(
            "instances",
            data=instances,
            dtype=instances.dtype,
        )
        f.create_dataset(
            "frames",
            data=frames,
            dtype=frames.dtype,
        )


def make_instance_group(
    instance_group_dict: dict,
    labeled_frames: list[LabeledFrame],
    camera_group: CameraGroup,
) -> InstanceGroup:
    """Creates an `InstanceGroup` object from a dictionary.

    Args:
        instance_group_dict: Dictionary with the following necessary key:
            - "camcorder_to_lf_and_inst_idx_map": Dictionary mapping `Camera` indices to
                a tuple of `LabeledFrame` index (in `labeled_frames`) and `Instance`
                index (in containing `LabeledFrame.instances`).
            and optional keys:
            - "score": A float representing the reprojection score for the
                `InstanceGroup`.
            - "points": 3D points for the `InstanceGroup`.
            - Any keys containing metadata.
        labeled_frames: List of `LabeledFrame` objects (expecting
            `Labels.labeled_frames`) used to retrieve `Instance` objects.
        camera_group: `CameraGroup` object used to retrieve `Camera` objects.

    Returns:
        `InstanceGroup` object.
    """
    # Avoid mutating the dictionary
    instance_group_dict = instance_group_dict.copy()

    # Get the `Instance` objects
    camera_to_lf_and_inst_idx_map: dict[str, tuple[str, str]] = instance_group_dict.pop(
        "camcorder_to_lf_and_inst_idx_map"
    )

    instance_by_camera: dict[Camera, Instance] = {}
    for cam_idx, (lf_idx, inst_idx) in camera_to_lf_and_inst_idx_map.items():
        # Retrieve the `Camera`
        camera = camera_group.cameras[int(cam_idx)]

        # Retrieve the `Instance` from the `LabeledFrame
        labeled_frame = labeled_frames[int(lf_idx)]
        instance = labeled_frame.instances[int(inst_idx)]

        # Link the `Instance` to the `Camera`
        instance_by_camera[camera] = instance

    # Get all optional attributes
    score = None
    if "score" in instance_group_dict:
        score = instance_group_dict.pop("score")
    points = None
    if "points" in instance_group_dict:
        points = instance_group_dict.pop("points")

    # Metadata contains any information that the class does not deserialize.
    metadata = instance_group_dict  # Remaining keys are metadata.

    return InstanceGroup(
        instance_by_camera=instance_by_camera,
        score=score,
        points=points,
        metadata=metadata,
    )


def make_frame_group(
    frame_group_dict: dict,
    labeled_frames: list[LabeledFrame],
    camera_group: CameraGroup,
) -> FrameGroup:
    """Create a `FrameGroup` object from a dictionary.

    Args:
        frame_group_dict: Dictionary representing a `FrameGroup` object with the
            following necessary key:
            - "instance_groups": List of dictionaries containing `InstanceGroup`
                information (see `make_instance_group` for what each dictionary
                contains).
            and optional keys:
            - "frame_idx": Frame index.
            - Any keys containing metadata.
        labeled_frames_list: List of `LabeledFrame` objects (expecting
            `Labels.labeled_frames`).
        camera_group: `CameraGroup` object used to retrieve `Camera` objects.

    Returns:
        `FrameGroup` object.
    """
    # Avoid mutating the dictionary
    frame_group_dict = frame_group_dict.copy()

    frame_idx = None

    # Get `InstanceGroup` objects
    instance_groups_info = frame_group_dict.pop("instance_groups")
    instance_groups = []
    labeled_frame_by_camera = {}
    for instance_group_dict in instance_groups_info:
        instance_group = make_instance_group(
            instance_group_dict=instance_group_dict,
            labeled_frames=labeled_frames,
            camera_group=camera_group,
        )
        instance_groups.append(instance_group)

        # Also retrieve the `LabeledFrame` by `Camera`. We do this for each
        # `InstanceGroup` to ensure that we have don't miss a `LabeledFrame`.
        camera_to_lf_and_inst_idx_map = instance_group_dict[
            "camcorder_to_lf_and_inst_idx_map"
        ]
        for cam_idx, (lf_idx, _) in camera_to_lf_and_inst_idx_map.items():
            # Retrieve the `Camera`
            camera = camera_group.cameras[int(cam_idx)]

            # Retrieve the `LabeledFrame`
            labeled_frame = labeled_frames[int(lf_idx)]
            labeled_frame_by_camera[camera] = labeled_frame

            # We can get the frame index from the `LabeledFrame` if any.
            frame_idx = labeled_frame.frame_idx

    # Get the frame index explicitly from the dictionary if it exists.
    if "frame_idx" in frame_group_dict:
        frame_idx = frame_group_dict.pop("frame_idx")

    # Metadata contains any information that the class doesn't deserialize.
    metadata = frame_group_dict  # Remaining keys are metadata.

    return FrameGroup(
        frame_idx=frame_idx,
        instance_groups=instance_groups,
        labeled_frame_by_camera=labeled_frame_by_camera,
        metadata=metadata,
    )


def make_camera(camera_dict: dict) -> Camera:
    """Create `Camera` from a dictionary.

    Args:
        camera_dict: Dictionary containing camera information with the following
            necessary keys:
            - "name": Camera name.
            - "size": Image size (width, height) of camera in pixels of size (2,) and
                type int.
            - "matrix": Intrinsic camera matrix of size (3, 3) and type float64.
            - "distortions": Radial-tangential distortion coefficients
                [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
            - "rotation": Rotation vector in unnormalized axis-angle representation of
                size (3,) and type float64.
            - "translation": Translation vector of size (3,) and type float64.
            and optional keys containing metadata.

    Returns:
        `Camera` object created from dictionary.
    """
    # Avoid mutating the dictionary.
    camera_dict = camera_dict.copy()

    # Get all attributes we deserialize.
    name = camera_dict.pop("name")
    size = camera_dict.pop("size")
    camera = Camera(
        name=name if len(name) > 0 else None,
        size=size if len(size) > 0 else None,
        matrix=camera_dict.pop("matrix"),
        dist=camera_dict.pop("distortions"),
        rvec=camera_dict.pop("rotation"),
        tvec=camera_dict.pop("translation"),
    )

    # Add remaining metadata to `Camera`
    camera.metadata = camera_dict

    return camera


def make_camera_group(calibration_dict: dict) -> CameraGroup:
    """Create a `CameraGroup` from a calibration dictionary.

    Args:
        calibration_dict: Dictionary containing calibration information for cameras
            with optional keys:
            - "metadata": Dictionary containing metadata for the `CameraGroup`.
            - Arbitrary (but unique) keys for every `Camera`, each containing a
                dictionary with camera information (see `make_camera` for what each
                dictionary contains).

    Returns:
        `CameraGroup` object created from calibration dictionary.
    """
    cameras = []
    metadata = {}
    for dict_name, camera_dict in calibration_dict.items():
        if dict_name == "metadata":
            metadata = camera_dict
            continue
        camera = make_camera(camera_dict)
        cameras.append(camera)

    return CameraGroup(cameras=cameras, metadata=metadata)


def make_session(
    session_dict: dict, videos: list[Video], labeled_frames: list[LabeledFrame]
) -> RecordingSession:
    """Create a `RecordingSession` from a dictionary.

    Args:
        session_dict: Dictionary with keys:
            - "calibration": Dictionary containing calibration information for cameras.
            - "camcorder_to_video_idx_map": Dictionary mapping camera index to video
                index.
            - "frame_group_dicts": List of dictionaries containing `FrameGroup`
                information. See `make_frame_group` for what each dictionary contains.
            - Any optional keys containing metadata.
        videos_list: List containing `Video` objects (expected `Labels.videos`).
        labeled_frames_list: List containing `LabeledFrame` objects (expected
            `Labels.labeled_frames`).

    Returns:
        `RecordingSession` object.
    """
    # Avoid modifying original dictionary
    session_dict = session_dict.copy()

    # Restructure `RecordingSession` without `Video` to `Camera` mapping
    calibration_dict = session_dict.pop("calibration")
    camera_group = make_camera_group(calibration_dict)

    # Retrieve all `Camera` and `Video` objects, then add to `RecordingSession`
    camcorder_to_video_idx_map = session_dict.pop("camcorder_to_video_idx_map")
    video_by_camera = {}
    camera_by_video = {}
    for cam_idx, video_idx in camcorder_to_video_idx_map.items():
        camera = camera_group.cameras[int(cam_idx)]
        video = videos[int(video_idx)]
        video_by_camera[camera] = video
        camera_by_video[video] = camera

    # Reconstruct all `FrameGroup` objects and add to `RecordingSession`
    frame_group_dicts = []
    if "frame_group_dicts" in session_dict:
        frame_group_dicts = session_dict.pop("frame_group_dicts")
    frame_group_by_frame_idx = {}
    for frame_group_dict in frame_group_dicts:
        try:
            # Add `FrameGroup` to `RecordingSession`
            frame_group = make_frame_group(
                frame_group_dict=frame_group_dict,
                labeled_frames=labeled_frames,
                camera_group=camera_group,
            )
            frame_group_by_frame_idx[frame_group.frame_idx] = frame_group
        except ValueError as e:
            print(
                f"Error reconstructing FrameGroup: {frame_group_dict}. Skipping..."
                f"\n{e}"
            )

    session = RecordingSession(
        camera_group=camera_group,
        video_by_camera=video_by_camera,
        camera_by_video=camera_by_video,
        frame_group_by_frame_idx=frame_group_by_frame_idx,
        metadata=session_dict,
    )

    return session


def read_sessions(
    labels_path: str, videos: list[Video], labeled_frames: list[LabeledFrame]
) -> list[RecordingSession]:
    """Read `RecordingSession` dataset from a SLEAP labels file.

    Expects a "sessions_json" dataset in the `labels_path` file, but will return an
    empty list if the dataset is not found.

    Args:
        labels_path: A string path to the SLEAP labels file.
        videos: A list of `Video` objects.
        labeled_frames: A list of `LabeledFrame` objects.

    Returns:
        A list of `RecordingSession` objects.
    """
    try:
        sessions = read_hdf5_dataset(labels_path, "sessions_json")
    except KeyError:
        return []
    sessions = [json.loads(x) for x in sessions]
    session_objects = []
    for session in sessions:
        session_objects.append(make_session(session, videos, labeled_frames))
    return session_objects


def instance_group_to_dict(
    instance_group: InstanceGroup,
    instance_to_lf_and_inst_idx: dict[Instance, tuple[int, int]],
    camera_group: CameraGroup,
) -> dict:
    """Convert `instance_group` to a dictionary.

    Args:
        instance_group: `InstanceGroup` object to convert to a dictionary.
        instance_to_lf_and_inst_idx: Dictionary mapping `Instance` objects to
            `LabeledFrame` indices (in `Labels.labeled_frames`) and `Instance` indices
            (in containing `LabeledFrame.instances`).
        camera_group: `CameraGroup` object that determines the order of the `Camera`
            objects when converting to a dictionary.

    Returns:
        Dictionary of the `InstanceGroup` with keys:
            - "camcorder_to_lf_and_inst_idx_map": Dictionary mapping `Camera` indices
                (in `InstanceGroup.camera_cluster.cameras`) to a tuple of `LabeledFrame`
                and `Instance` indices (from `instance_to_lf_and_inst_idx`)
            - Any optional keys containing metadata.
    """
    camera_to_lf_and_inst_idx_map: dict[int, tuple[int, int]] = {
        camera_group.cameras.index(cam): instance_to_lf_and_inst_idx[instance]
        for cam, instance in instance_group.instance_by_camera.items()
    }

    # Only required key is camcorder_to_lf_and_inst_idx_map
    instance_group_dict = {
        "camcorder_to_lf_and_inst_idx_map": camera_to_lf_and_inst_idx_map,
    }

    # Optionally add score, points, and metadata if they are non-default values
    if instance_group.score is not None:
        instance_group_dict["score"] = instance_group.score
    if instance_group.points is not None:
        instance_group_dict["points"] = instance_group.points.tolist()
    instance_group_dict.update(instance_group.metadata)

    return instance_group_dict


def frame_group_to_dict(
    frame_group: FrameGroup,
    labeled_frame_to_idx: dict[LabeledFrame, int],
    camera_group: CameraGroup,
) -> dict:
    """Convert `frame_group` to a dictionary.

    Args:
        frame_group: `FrameGroup` object to convert to a dictionary.
        labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
            `Labels.labeled_frames`.
        camera_group: `CameraGroup` object that determines the order of the `Camera`
            objects when converting to a dictionary.

    Returns:
        Dictionary of the `FrameGroup` with keys:
            - "instance_groups": List of dictionaries for each `InstanceGroup` in the
                `FrameGroup`. See `instance_group_to_dict` for what each dictionary
                contains.
            - "frame_idx": Frame index for the `FrameGroup`.
            - Any optional keys containing metadata.
    """
    # Create dictionary of `Instance` to `LabeledFrame` index (in
    # `Labels.labeled_frames`) and `Instance` index in `LabeledFrame.instances`.
    instance_to_lf_and_inst_idx: dict[Instance, tuple[int, int]] = {
        inst: (labeled_frame_to_idx[labeled_frame], inst_idx)
        for labeled_frame in frame_group.labeled_frames
        for inst_idx, inst in enumerate(labeled_frame.instances)
    }

    frame_group_dict = {
        "instance_groups": [
            instance_group_to_dict(
                instance_group,
                instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx,
                camera_group=camera_group,
            )
            for instance_group in frame_group.instance_groups
        ],
    }
    frame_group_dict["frame_idx"] = frame_group.frame_idx
    frame_group_dict.update(frame_group.metadata)

    return frame_group_dict


def camera_to_dict(camera: Camera) -> dict:
    """Convert `camera` to dictionary.

    Args:
        camera: `Camera` object to convert to a dictionary.

    Returns:
        Dictionary containing camera information with the following keys:
            - "name": Camera name.
            - "size": Image size (width, height) of camera in pixels of size (2,) and type
                int.
            - "matrix": Intrinsic camera matrix of size (3, 3) and type float64.
            - "distortions": Radial-tangential distortion coefficients
                [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
            - "rotation": Rotation vector in unnormalized axis-angle representation of
                size (3,) and type float64.
            - "translation": Translation vector of size (3,) and type float64.
            - Any optional keys containing metadata.

    """
    # Handle optional attributes
    name = "" if camera.name is None else camera.name
    size = "" if camera.size is None else list(camera.size)

    camera_dict = {
        "name": name,
        "size": size,
        "matrix": camera.matrix.tolist(),
        "distortions": camera.dist.tolist(),
        "rotation": camera.rvec.tolist(),
        "translation": camera.tvec.tolist(),
    }
    camera_dict.update(camera.metadata)

    return camera_dict


def camera_group_to_dict(camera_group: CameraGroup) -> dict:
    """Convert `camera_group` to dictionary.

    Args:
        camera_group: `CameraGroup` object to convert to a dictionary.

    Returns:
        Dictionary containing camera group information with the following keys:
            - cam_n: Camera dictionary containing information for camera at index "n"
                with the following keys:
                name: Camera name.
                size: Image size (height, width) of camera in pixels of size (2,)
                    and type int.
                matrix: Intrinsic camera matrix of size (3, 3) and type float64.
                distortions: Radial-tangential distortion coefficients
                    [k_1, k_2, p_1, p_2, k_3] of size (5,) and type float64.
                rotation: Rotation vector in unnormalized axis-angle representation
                    of size (3,) and type float64.
                translation: Translation vector of size (3,) and type float64.
            - "metadata": Dictionary of optional metadata.
    """
    calibration_dict = {}
    for cam_idx, camera in enumerate(camera_group.cameras):
        camera_dict = camera_to_dict(camera)
        calibration_dict[f"cam_{cam_idx}"] = camera_dict

    calibration_dict["metadata"] = camera_group.metadata.copy()

    return calibration_dict


def session_to_dict(
    session: RecordingSession,
    video_to_idx: dict[Video, int],
    labeled_frame_to_idx: dict[LabeledFrame, int],
) -> dict:
    """Convert `RecordingSession` to a dictionary.

    Args:
        session: `RecordingSession` object to convert to a dictionary.
        video_to_idx: Dictionary of `Video` to index in `Labels.videos`.
        labeled_frame_to_idx: Dictionary of `LabeledFrame` to index in
            `Labels.labeled_frames`.

    Returns:
        Dictionary of `RecordingSession` with the following keys:
            - "calibration": Dictionary containing calibration information for cameras.
            - "camcorder_to_video_idx_map": Dictionary mapping camera index to video
                index.
            - "frame_group_dicts": List of dictionaries containing `FrameGroup`
                information. See `frame_group_to_dict` for what each dictionary
                contains.
            - Any optional keys containing metadata.
    """
    # Unstructure `CameraCluster` and `metadata`
    calibration_dict = camera_group_to_dict(session.camera_group)

    # Store camera-to-video indices map where key is camera index
    # and value is video index from `Labels.videos`
    camera_to_video_idx_map = {}
    for cam_idx, camera in enumerate(session.camera_group.cameras):
        # Skip if Camera is not linked to any Video

        if camera not in session.cameras:
            continue

        # Get video index from `Labels.videos`
        video = session.get_video(camera)
        video_idx = video_to_idx.get(video, None)

        if video_idx is not None:
            camera_to_video_idx_map[cam_idx] = video_idx
        else:
            print(
                f"Video {video} not found in `Labels.videos`. "
                "Not saving to `RecordingSession` serialization."
            )

    # Store frame groups by frame index
    frame_group_dicts = []
    if len(labeled_frame_to_idx) > 0:  # Don't save if skipping labeled frames
        for frame_group in session.frame_groups.values():
            # Only save `FrameGroup` if it has `InstanceGroup`s
            if len(frame_group.instance_groups) > 0:
                frame_group_dict = frame_group_to_dict(
                    frame_group,
                    labeled_frame_to_idx=labeled_frame_to_idx,
                    camera_group=session.camera_group,
                )
                frame_group_dicts.append(frame_group_dict)

    session_dict = {
        "calibration": calibration_dict,
        "camcorder_to_video_idx_map": camera_to_video_idx_map,
        "frame_group_dicts": frame_group_dicts,
    }
    session_dict.update(session.metadata)

    return session_dict


def write_sessions(
    labels_path: str,
    sessions: list[RecordingSession],
    videos: list[Video],
    labeled_frames: list[LabeledFrame],
):
    """Write `RecordingSession` metadata to a SLEAP labels file.

    Creates a new dataset "sessions_json" in the `labels_path` file to store the
    sessions data.

    Args:
        labels_path: A string path to the SLEAP labels file.
        sessions: A list of `RecordingSession` objects to store in the `labels_path`
            file.
        videos: A list of `Video` objects referenced in the `RecordingSession`s
            (expecting `Labels.videos`).
        labeled_frames: A list of `LabeledFrame` objects referenced in the
            `RecordingSession`s (expecting `Labels.labeled_frames`).
    """
    sessions_json = []
    if len(sessions) > 0:
        labeled_frame_to_idx = {lf: i for i, lf in enumerate(labeled_frames)}
        video_to_idx = {video: i for i, video in enumerate(videos)}
    for session in sessions:
        session_json = session_to_dict(
            session=session,
            video_to_idx=video_to_idx,
            labeled_frame_to_idx=labeled_frame_to_idx,
        )
        sessions_json.append(np.bytes_(json.dumps(session_json, separators=(",", ":"))))

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("sessions_json", data=sessions_json, maxshape=(None,))


def read_labels(labels_path: str, open_videos: bool = True) -> Labels:
    """Read a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        open_videos: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).

    Returns:
        The processed `Labels` object.
    """
    tracks = read_tracks(labels_path)
    videos = read_videos(labels_path, open_backend=open_videos)
    skeletons = read_skeletons(labels_path)
    points = read_points(labels_path)
    pred_points = read_pred_points(labels_path)
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")
    instances = read_instances(
        labels_path, skeletons, tracks, points, pred_points, format_id
    )
    suggestions = read_suggestions(labels_path, videos)
    metadata = read_metadata(labels_path)
    provenance = metadata.get("provenance", dict())

    frames = read_hdf5_dataset(labels_path, "frames")
    labeled_frames = []
    for _, video_id, frame_idx, instance_id_start, instance_id_end in frames:
        labeled_frames.append(
            LabeledFrame(
                video=videos[video_id],
                frame_idx=int(frame_idx),
                instances=instances[instance_id_start:instance_id_end],
            )
        )

    sessions = read_sessions(labels_path, videos, labeled_frames)

    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        suggestions=suggestions,
        sessions=sessions,
        provenance=provenance,
    )
    labels.provenance["filename"] = labels_path

    return labels


def write_labels(
    labels_path: str,
    labels: Labels,
    embed: bool | str | list[tuple[Video, int]] | None = None,
):
    """Write a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file to save.
        labels: A `Labels` object to save.
        embed: Frames to embed in the saved labels file. One of `None`, `True`,
            `"all"`, `"user"`, `"suggestions"`, `"user+suggestions"`, `"source"` or list
            of tuples of `(video, frame_idx)`.

            If `None` is specified (the default) and the labels contains embedded
            frames, those embedded frames will be re-saved to the new file.

            If `True` or `"all"`, all labeled frames and suggested frames will be
            embedded.

            If `"source"` is specified, no images will be embedded and the source video
            will be restored if available.

            This argument is only valid for the SLP backend.
    """
    if Path(labels_path).exists():
        Path(labels_path).unlink()

    if embed:
        embed_videos(labels_path, labels, embed)
    write_videos(labels_path, labels.videos, restore_source=(embed == "source"))
    write_tracks(labels_path, labels.tracks)
    write_suggestions(labels_path, labels.suggestions, labels.videos)
    write_sessions(labels_path, labels.sessions, labels.videos, labels.labeled_frames)
    write_metadata(labels_path, labels)
    write_lfs(labels_path, labels)
