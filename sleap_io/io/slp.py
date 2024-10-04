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
    Point,
    PredictedPoint,
    Instance,
    PredictedInstance,
    LabeledFrame,
    Labels,
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
            if dataset in f:
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
                ds = f.create_dataset(
                    f"{group}/video",
                    shape=(len(imgs_data), max(len(img) for img in imgs_data)),
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


def read_points(labels_path: str) -> list[Point]:
    """Read `Point` dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `Point` objects.
    """
    pts = read_hdf5_dataset(labels_path, "points")
    return [
        Point(x=x, y=y, visible=visible, complete=complete)
        for x, y, visible, complete in pts
    ]


def read_pred_points(labels_path: str) -> list[PredictedPoint]:
    """Read `PredictedPoint` dataset from a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.

    Returns:
        A list of `PredictedPoint` objects.
    """
    pred_pts = read_hdf5_dataset(labels_path, "pred_points")
    return [
        PredictedPoint(x=x, y=y, visible=visible, complete=complete, score=score)
        for x, y, visible, complete, score in pred_pts
    ]


def read_instances(
    labels_path: str,
    skeletons: list[Skeleton],
    tracks: list[Track],
    points: list[Point],
    pred_points: list[PredictedPoint],
    format_id: float,
) -> list[Union[Instance, PredictedInstance]]:
    """Read `Instance` dataset in a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        skeletons: A list of `Skeleton` objects (see `read_skeletons`).
        tracks: A list of `Track` objects (see `read_tracks`).
        points: A list of `Point` objects (see `read_points`).
        pred_points: A list of `PredictedPoint` objects (see `read_pred_points`).
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
        else:
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

        if instance_type == InstanceType.USER:
            instances[instance_id] = Instance(
                points=points[point_id_start:point_id_end],  # type: ignore[arg-type]
                skeleton=skeletons[skeleton_id],
                track=tracks[track_id] if track_id >= 0 else None,
            )
            if from_predicted >= 0:
                from_predicted_pairs.append((instance_id, from_predicted))
        elif instance_type == InstanceType.PREDICTED:
            instances[instance_id] = PredictedInstance(
                points=pred_points[point_id_start:point_id_end],  # type: ignore[arg-type]
                skeleton=skeletons[skeleton_id],
                track=tracks[track_id] if track_id >= 0 else None,
                score=instance_score,
                tracking_score=tracking_score,
            )

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

            if type(inst) == Instance:
                instance_type = InstanceType.USER
                score = np.nan
                tracking_score = np.nan
                point_id_start = len(points)

                for node in inst.skeleton.nodes:
                    pt = inst.points[node]
                    points.append([pt.x, pt.y, pt.visible, pt.complete])

                point_id_end = len(points)

            elif type(inst) == PredictedInstance:
                instance_type = InstanceType.PREDICTED
                score = inst.score
                tracking_score = inst.tracking_score
                point_id_start = len(predicted_points)

                for node in inst.skeleton.nodes:
                    pt = inst.points[node]
                    predicted_points.append(
                        [pt.x, pt.y, pt.visible, pt.complete, pt.score]
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

    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        suggestions=suggestions,
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
    write_metadata(labels_path, labels)
    write_lfs(labels_path, labels)
