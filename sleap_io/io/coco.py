"""This module implements routines for reading and writing COCO-formatted datasets."""

from __future__ import annotations
import numpy as np
import simplejson as json
from pathlib import Path
from collections import defaultdict
from sleap_io import (
    Video,
    Skeleton,
    Track,
    Instance,
    LabeledFrame,
    Labels,
)

import sys
import imageio.v3 as iio

try:
    import cv2
except ImportError:
    pass


def read_ann(ann_json_path: str | Path):
    """Read annotations JSON file.

    Args:
        ann_json_path: Path to a JSON file with the annotations.

    Returns:
        A dictionary with the parsed data.
    """
    with open(ann_json_path, "r") as f:
        ann = json.load(f)
    return ann


def make_skeleton(ann: dict) -> Skeleton:
    """Parse skeleton metadata.

    Args:
        ann: Dictionary with decoded JSON data. Must contain a key named "categories".
            This key must contain sub-keys "keypoints" (node names), "skeleton" (edges),
            and optionally "name".

    Returns:
        The `Skeleton` object.

    Notes:
        This assumes that `skeleton` (edge indices) are 1-based.
    """
    return Skeleton(
        nodes=ann["categories"][0]["keypoints"],
        edges=(np.array(ann["categories"][0]["skeleton"]) - 1).tolist(),
        name=ann["categories"][0].get("name", None),
    )


def make_videos(
    ann: dict, imgs_prefix: str | Path | None = None
) -> tuple[list[Video], dict[int, tuple[int, int]]]:
    """Make videos and return mapping to indices.

    Args:
        ann: Dictionary with decoded JSON data. Must contain a key named "images".
        imgs_prefix: Optional path specifying a prefix to prepend to image filenames.

    Returns:
        A tuple of `videos, video_id_map`.

        `videos` is a list of `Video`s.

        `video_id_map` is a dictionary that maps an image ID to a tuple of
        `(video_ind, frame_ind)`, corresponding to the order in `videos`.

    Notes:
        This function will group images that have the same shape together into a single
        logical video.
    """
    if type(imgs_prefix) == str:
        imgs_prefix = Path(imgs_prefix)
    imgs_by_shape = defaultdict(list)
    video_id_map = {}
    for img in ann["images"]:
        shape = img["height"], img["width"]
        img_filename = img["file_name"]
        if imgs_prefix is not None:
            img_filename = (imgs_prefix / img_filename).as_posix()
        imgs_by_shape[shape].append(img_filename)
        video_id_map[ann["id"]] = (
            imgs_by_shape.keys().index(shape),
            len(imgs_by_shape[shape]) - 1,
        )

    videos = []
    for shape, imgs in imgs_by_shape.items():
        videos.append(
            Video.from_filename(imgs, backend_metadata={"shape": shape + (3,)})
        )

    return videos, video_id_map


def make_labels(
    ann: dict,
    videos: list[Video],
    video_id_map: dict[int, tuple[int, int]],
    skeleton: Skeleton,
) -> Labels:
    """Make a `Labels` object from annotations.

    Args:
        ann: Dictionary with decoded JSON data. Must contain a key named "annotations".
        videos: A list of `Video`s.
        video_id_map: A dictionary that maps an image ID to a tuple of
            `(video_ind, frame_ind)`, corresponding to the order in `videos`.
        skeleton: A `Skeleton`.

    Returns:
        A `Labels` file with parsed data.
    """
    tracks_by_id = {}

    lfs_by_ind = defaultdict(list)
    for an in ann["annotations"]:
        pts = np.array(an["keypoints"]).reshape(-1, 3)
        pts[pts[:, 3] != 2] = np.nan
        pts = pts[:, :2]

        video_ind, frame_ind = video_id_map[an["image_id"]]

        if "track_id" in an:
            track_id = an["track_id"]
            if track_id in tracks_by_id:
                tracks_by_id[track_id] = Track(name=f"{track_id}")
            track = tracks_by_id[track_id]
        else:
            track = None

        lfs_by_ind[(video_ind, frame_ind)].append(
            Instance.from_numpy(pts, skeleton=skeleton, track=track)
        )

    lfs = []
    for (video_ind, frame_ind), insts in lfs_by_ind.items():
        lfs.append(
            LabeledFrame(video=videos[video_ind], frame_idx=frame_ind, instances=insts)
        )
    labels = Labels(lfs)
    labels.provenance["info"] = ann.get("info", None)

    return labels


def read_labels(
    ann_json_path: str | Path, imgs_prefix: str | Path | None = None
) -> Labels:
    """Read and parse COCO annotations.

    Args:
        ann_json_path: Path to a JSON file with the annotations.
        imgs_prefix: Optional path specifying a prefix to prepend to image filenames.
            This is typically a path to the folder containing the images. If not
            provided, assumes that there exists an "images" folder in the parent
            directory of the folder containing the annotations.

    Returns:
        `Labels` with the parsed data.
    """
    ann = read_ann(ann_json_path)
    if imgs_prefix is None:
        imgs_prefix = Path(ann_json_path).parent / "images"
    videos, video_id_map = make_videos(ann, imgs_prefix=imgs_prefix)
    skeleton = make_skeleton(ann)
    labels = make_labels(ann, videos, video_id_map, skeleton)
    return labels


def write_labels(
    labels: Labels,
    dataset_folder: str | Path,
    split: str | None = None,
    img_format: str = "png",
):
    """Save a `Labels` to COCO format.

    Args:
        labels: A `Labels` object.
        dataset_folder: Path to a folder to save data to.
        split: Optional string specifying the split name.
        img_format: Format to save images to. Formats: "png" (default) or "jpg".

    Notes:
        If `split` was not provided, the annotations will be saved to
        `{dataset_folder}/annotations/ann.json` and images will be saved to
        `{dataset_folder}/images`.

        If `split` was provided, the annotations will be saved to
        `{dataset_folder}/annotations/ann_{split}.json` and images will be saved to
        `{dataset_folder}/images/{split}`.

        Calling this multiple times with the same dataset folder may overwrite previous
        data if `split` is not provided.
    """
    if split is None:
        ann_path = dataset_folder / "annotations" / "ann.json"
        imgs_folder = dataset_folder / "images"
    else:
        ann_path = dataset_folder / "annotations" / f"ann_{split}.json"
        imgs_folder = dataset_folder / "images" / split

    ann_path.parent.mkdir(parents=True, exist_ok=True)
    imgs_folder.mkdir(parents=True, exist_ok=True)

    lfs = labels.user_labeled_frames

    imgs = []
    img_filename_map = {}
    for img_id, lf in enumerate(lfs):
        img_filename = f"{img_id}.{img_format}"
        img_shape = video.shape[[1, 2]]
        imgs.append(
            {
                "id": img_id,
                "file_name": img_filename.as_posix(),
                "height": img_shape[0],
                "width": img_shape[1],
            }
        )
        img_filename_map[(lf.video, lf.frame_idx)] = img_filename

    for (video, frame_idx), img_filename in img_filename_map.items():
        img = video[frame_idx]
        img_path = (imgs_folder / img_filename).as_posix()
        if "cv2" in sys.modules:
            cv2.imwrite(img_path, img)
        else:
            iio.imwrite(img_path, img)

    inst_id = 0
    annotations = []
    for img_id, lf in enumerate(lfs):
        for inst in lf:
            ann = {}

            pts = inst.numpy()
            vis = np.isnan(pts).any(axis=1, keepdims=True).astype(int)
            vis[vis == 0] = 2  # labeled and visible
            # 1: labeled but not visible
            vis[vis == 1] = 0  # not labeled
            pts[np.isnan(pts)] = -1
            kps = np.concatenate([pts, vis], axis=1).reshape(-1).tolist()
            ann["keypoints"] = kps
            ann["id"] = inst_id
            ann["image_id"] = img_id
            ann["num_keypoints"] = len(pts)

            x, y = np.nanmin(pts, axis=0)
            w, h = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
            ann["bbox"] = [x, y, w, h]
            ann["iscrowd"] = 0
            ann["area"] = w * h
            ann["category_id"] = labels.skeletons.index(inst.skeleton)

            if inst.track is not None:
                ann["track_id"] = labels.tracks.index(inst.track)

            annotations.append(ann)
            inst_id += 1

    categories = []
    for skel_ind, skel in enumerate(labels.skeletons):
        category = {}
        category["supercategory"] = "animal"
        category["id"] = skel_ind
        category["name"] = skel.name
        category["keypoints"] = skel.node_names
        category["skeleton"] = (np.array(skel.edge_inds) + 1).tolist()
        categories.append(category)

    ann = {
        "info": labels.provenance.get("info", {}),
        "images": imgs,
        "annotations": annotations,
        "categories": categories,
    }

    with open(ann_path, "w") as f:
        json.dump(ann, f)
