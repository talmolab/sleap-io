"""This module handles direct I/O operations for working with Labelstudio files.

Some important nomenclature:
  - `tasks`: typically maps to a single frame of data to be annotated, closest
    correspondance is to `LabeledFrame`
  - `annotations`: collection of points, polygons, relations, etc. corresponds to
    `Instance`s and `Point`s, but a flattened hierarchy

"""

import datetime
import math
import uuid
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import simplejson as json

from sleap_io import Instance, LabeledFrame, Labels, Node, Point, Skeleton, Video


def read_labels(
    labels_path: str, skeleton: Optional[Union[Skeleton, List[str]]] = None
) -> Labels:
    """Read Label Studio style annotations from a file and return a `Labels` object.

    Args:
        labels_path: Path to the Label Studio annotation file, in json format.
        skeleton: An optional `Skeleton` object or list of node names. If not provided
            (the default), skeleton will be inferred from the data. It may be useful to
            provide this so the keypoint label types can be filtered to just the ones in
            the skeleton.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    with open(labels_path, "r") as task_file:
        tasks = json.load(task_file)

    if type(skeleton) == list:
        skeleton = Skeleton(nodes=skeleton)  # type: ignore[arg-type]
    elif skeleton is None:
        skeleton = infer_nodes(tasks)
    else:
        assert isinstance(skeleton, Skeleton)

    labels = parse_tasks(tasks, skeleton)
    labels.provenance["filename"] = labels_path
    return labels


def infer_nodes(tasks: List[Dict]) -> Skeleton:
    """Parse the loaded JSON tasks to create a minimal skeleton.

    Args:
        tasks: Collection of tasks loaded from Label Studio JSON.

    Returns:
        The inferred `Skeleton`.
    """
    node_names = set()
    for entry in tasks:
        if "annotations" in entry:
            key = "annotations"
        elif "completions" in entry:
            key = "completions"
        else:
            raise ValueError("Cannot find annotation data for entry!")

        for annotation in entry[key]:
            for datum in annotation["result"]:
                if datum["type"] == "keypointlabels":
                    for node_name in datum["value"]["keypointlabels"]:
                        node_names.add(node_name)

    skeleton = Skeleton(nodes=list(node_names))
    return skeleton


def parse_tasks(tasks: List[Dict], skeleton: Skeleton) -> Labels:
    """Read Label Studio style annotations from a file and return a `Labels` object.

    Args:
        tasks: Collection of tasks to be converted to `Labels`.
        skeleton: `Skeleton` with the nodes and edges to be used.

    Returns:
        Parsed labels as a `Labels` instance.
    """
    frames: List[LabeledFrame] = []
    for entry in tasks:
        # depending version, we have seen keys `annotations` and `completions`
        if "annotations" in entry:
            key = "annotations"
        elif "completions" in entry:
            key = "completions"
        else:
            raise ValueError("Cannot find annotation data for entry!")

        frames.append(task_to_labeled_frame(entry, skeleton, key=key))

    return Labels(frames)


def convert_labels(labels: Labels) -> List[dict]:
    """Convert a `Labels` object into Label Studio-formatted annotations.

    Args:
        labels: SLEAP `Labels` to be converted to Label Studio task format.

    Returns:
        Label Studio dictionaries of the `Labels` data.
    """
    out = []
    for frame in labels.labeled_frames:
        if frame.video.shape is not None:
            height = frame.video.shape[1]
            width = frame.video.shape[2]
        else:
            height = 100
            width = 100

        frame_annots = []

        for instance in frame.instances:
            inst_id = str(uuid.uuid4())
            frame_annots.append(
                {
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": 0,
                        "y": 0,
                        "width": width,
                        "height": height,
                        "rotation": 0,
                        "rectanglelabels": [
                            "instance_class"
                        ],  # TODO: need to handle instance classes / identity
                    },
                    "id": inst_id,
                    "from_name": "individuals",
                    "to_name": "image",
                    "type": "rectanglelabels",
                }
            )

            for node, point in instance.points.items():
                point_id = str(uuid.uuid4())

                # add this point
                frame_annots.append(
                    {
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        "value": {
                            "x": point.x / width * 100,
                            "y": point.y / height * 100,
                            "keypointlabels": [node.name],
                        },
                        "from_name": "keypoint-label",
                        "to_name": "image",
                        "type": "keypointlabels",
                        "id": point_id,
                    }
                )

                # add relationship of point to individual
                frame_annots.append(
                    {
                        "from_id": point_id,
                        "to_id": inst_id,
                        "type": "relation",
                        "direction": "right",
                    }
                )

        out.append(
            {
                "data": {
                    # 'image': f"/data/{up_deets['file']}"
                },
                "meta": {
                    "video": {
                        "filename": frame.video.filename,
                        "frame_idx": frame.frame_idx,
                        "shape": frame.video.shape,
                    }
                },
                "annotations": [
                    {
                        "result": frame_annots,
                        "was_cancelled": False,
                        "ground_truth": False,
                        "created_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "updated_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "lead_time": 0,
                        "result_count": 1,
                        # "completed_by": user['id']
                    }
                ],
            }
        )

    return out


def write_labels(labels: Labels, filename: str):
    """Convert and save a SLEAP `Labels` object to a Label Studio `.json` file.

    Args:
        labels: SLEAP `Labels` to be converted to Label Studio task format.
        filename: Path to save Label Studio annotations (`.json`).
    """

    def _encode(obj):
        if type(obj).__name__ == "uint64":
            return int(obj)

    ls_dicts = convert_labels(labels)
    with open(filename, "w") as f:
        json.dump(ls_dicts, f, indent=4, default=_encode)


def task_to_labeled_frame(
    task: dict, skeleton: Skeleton, key: str = "annotations"
) -> LabeledFrame:
    """Parse annotations from an entry.

    Args:
        task: Label Studio task to be parsed.
        skeleton: Skeleton to use for parsing.
        key: Key to use for parsing annotations. Defaults to "annotations".

    Returns:
        Parsed `LabeledFrame` instance.
    """
    if len(task[key]) > 1:
        warnings.warn(
            f"Task {task.get('id', '??')}: Multiple annotations found, "
            "only taking the first!"
        )

    # only parse the first entry result
    to_parse = task[key][0]["result"]

    individuals = filter_and_index(to_parse, "rectanglelabels")
    keypoints = filter_and_index(to_parse, "keypointlabels")
    relations = build_relation_map(to_parse)
    instances = []

    if len(individuals) > 0:
        # multi animal case:
        for indv_id, indv in individuals.items():
            points = {}
            for rel in relations[indv_id]:
                kpt = keypoints.pop(rel)
                node = Node(kpt["value"]["keypointlabels"][0])
                x_pos = (kpt["value"]["x"] * kpt["original_width"]) / 100
                y_pos = (kpt["value"]["y"] * kpt["original_height"]) / 100

                # If the value is a NAN, the user did not mark this keypoint
                if math.isnan(x_pos) or math.isnan(y_pos):
                    continue

                points[node] = Point(x_pos, y_pos)

            if len(points) > 0:
                instances.append(Instance(points, skeleton))

    # If this is multi-animal, any leftover keypoints should be unique bodyparts, and
    # will be collected here if single-animal, we only have 'unique bodyparts' [in a
    # way] and the process is identical
    points = {}
    for _, kpt in keypoints.items():
        node = Node(kpt["value"]["keypointlabels"][0])
        points[node] = Point(
            (kpt["value"]["x"] * kpt["original_width"]) / 100,
            (kpt["value"]["y"] * kpt["original_height"]) / 100,
            visible=True,
        )
    if len(points) > 0:
        instances.append(Instance(points, skeleton))

    video, frame_idx = video_from_task(task)

    return LabeledFrame(video, frame_idx, instances)


def filter_and_index(annotations: Iterable[dict], annot_type: str) -> Dict[str, dict]:
    """Filter annotations based on the type field and index them by ID.

    Args:
        annotations: annotations to filter and index
        annot_type: annotation type to filter e.x. 'keypointlabels' or 'rectanglelabels'

    Returns:
        Dict of ndexed and filtered annotations. Only annotations of type `annot_type`
        will survive, and annotations are indexed by ID.
    """
    filtered = list(filter(lambda d: d["type"] == annot_type, annotations))
    indexed = {item["id"]: item for item in filtered}
    return indexed


def build_relation_map(annotations: Iterable[dict]) -> Dict[str, List[str]]:
    """Build a two-way relationship map between annotations.

    Args:
        annotations: annotations, presumably, containing relation types

    Returns:
        A two way map of relations indexed by `from_id` and `to_id` fields.
    """
    relations = list(filter(lambda d: d["type"] == "relation", annotations))
    relmap: Dict[str, List[str]] = {}
    for rel in relations:
        if rel["from_id"] not in relmap:
            relmap[rel["from_id"]] = []
        relmap[rel["from_id"]].append(rel["to_id"])

        if rel["to_id"] not in relmap:
            relmap[rel["to_id"]] = []
        relmap[rel["to_id"]].append(rel["from_id"])
    return relmap


def video_from_task(task: dict) -> Tuple[Video, int]:
    """Given a Label Studio task, retrieve video information.

    Args:
        task: Label Studio task

    Returns:
        Video and frame index for this task
    """
    if "meta" in task and "video" in task["meta"]:
        video = Video(task["meta"]["video"]["filename"], task["meta"]["video"]["shape"])
        frame_idx = task["meta"]["video"]["frame_idx"]
        return video, frame_idx

    else:
        raise KeyError("Unable to locate video information for task!", task)
