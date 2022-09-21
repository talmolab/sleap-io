import datetime
import json
import uuid
from typing import Dict, Iterable, List, Tuple

import cv2
from sleap_io import Instance, LabeledFrame, Labels, Node, Point, Video


def read_labels(labels_path: str) -> Labels:
    """Read label-studio style annotations and return a `Labels` object

    Parameters:
    labels_path (str): path to the label-studio annotation file, in json format

    Returns:
    Labels - parsed labels
    """

    with open(labels_path, "r") as task_file:
        tasks = json.load(task_file)

    frames = []
    for entry in tasks:
        # depending version, we have seen keys `annotations` and `completions`
        if "annotations" in entry:
            key = "annotations"
        elif "completions" in entry:
            key = "completions"
        else:
            raise ValueError("Cannot find annotation data for entry!")

        frames.extend(entry_to_labeled_frame(entry, key=key))

    return Labels(frames)


def write_labels(labels: Labels) -> List[Dict]:
    """Convert a `Labels` object into label-studio annotations"""

    for frame in labels.labeled_frames:
        height = frame.video.shape[1]
        width = frame.video.shape[2]
        out = []

        for instance in frame.instances:
            inst_id = uuid.uuid4()
            out.append(
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
                point_id = uuid.uuid4()

                # add this point
                out.append(
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
                out.append(
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
                    # 'original_file': up_deets['original_file']
                },
                "annotations": {
                    "result": out,
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": datetime.datetime.utcnow().strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                    "updated_at": datetime.datetime.utcnow().strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                    "lead_time": 0,
                    "result_count": 1,
                    # "completed_by": user['id']
                },
            }
        )


def entry_to_labeled_frame(entry: dict, key: str = "annotations") -> LabeledFrame:
    """Parse annotations from an entry"""

    if len(entry[key]) > 1:
        print(
            "WARNING: Task {}: Multiple annotations found, only taking the first!".format(
                entry["id"]
            )
        )

    try:
        # only parse the first entry result
        to_parse = entry[key][0]["result"]

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
                    points[node] = Point(
                        (kpt["value"]["x"] * kpt["original_width"]) / 100,
                        (kpt["value"]["y"] * kpt["original_height"]) / 100,
                        visible=True,
                    )
                instances.append(Instance(points))

        # If this is multi-animal, any leftover keypoints should be unique bodyparts, and will be collected here
        # if single-animal, we only have 'unique bodyparts' [in a way] and the process is identical
        points = {}
        for _, kpt in keypoints.items():
            node = Node(kpt["value"]["keypointlabels"][0])
            points[node] = Point(
                (kpt["value"]["x"] * kpt["original_width"]) / 100,
                (kpt["value"]["y"] * kpt["original_height"]) / 100,
                visible=True,
            )
        instances.append(Instance(points))

        return LabeledFrame(video_from_entry(entry), 0, instances)
    except Exception as excpt:
        raise RuntimeError(
            "While working on Task #{}, encountered the following error:".format(
                entry["id"]
            )
        ) from excpt


def filter_and_index(annotations: Iterable[dict], annot_type: str) -> Dict[str, dict]:
    """Filter annotations based on the type field and index them by ID

    Parameters:
    annotation (Iterable[dict]): annotations to filter and index
    annot_type (str): annotation type to filter e.x. 'keypointlabels' or 'rectanglelabels'

    Returns:
    Dict[str, dict] - indexed and filtered annotations. Only annotations of type `annot_type`
    will survive, and annotations are indexed by ID
    """
    filtered = list(filter(lambda d: d["type"] == annot_type, annotations))
    indexed = {item["id"]: item for item in filtered}
    return indexed


def build_relation_map(annotations: Iterable[dict]) -> Dict[str, List[dict]]:
    """Build a two-way relationship map between annotations

    Parameters:
    annotations (Iterable[dict]): annotations, presumably, containing relation types

    Returns:
    Dict[str, List[Dict]]: a two way map of relations indexed by `from_id` and `to_id` fields
    """
    relations = list(filter(lambda d: d["type"] == "relation", annotations))
    relmap = {}
    for rel in relations:
        if rel["from_id"] not in relmap:
            relmap[rel["from_id"]] = []
        relmap[rel["from_id"]].append(rel["to_id"])

        if rel["to_id"] not in relmap:
            relmap[rel["to_id"]] = []
        relmap[rel["to_id"]].append(rel["from_id"])
    return relmap


def get_image_path(entry: dict) -> str:
    """Extract image file path from an annotation entry"""
    if "meta" in entry and "original_file" in entry["meta"]:
        return entry["meta"]["original_file"]
    elif "task_path" in entry:
        return entry["task_path"]
    elif "data" in entry and "image" in entry["data"]:
        return entry["data"]["image"]
    elif "data" in entry and "depth_image" in entry["data"]:
        return entry["data"]["depth_image"]


def get_image_shape(path: str) -> Tuple[int, int, int, int]:
    im = cv2.imread(path)
    return (1, *im.shape)


def video_from_entry(entry: dict) -> Video:
    path = get_image_path(entry)
    return Video(path, get_image_shape(path), backend="image")


# def pick_filenames_from_tasks(tasks: List[dict]) -> List[str]:
#     ''' Given Label Studio task list, pick and return a list of filenames

#     Parameters:
#     tasks (List[dict]): List of label studio task dicts

#     Returns:
#     List[str] - List of filenames from tasks
#     '''
#     annot = read_annotations(tasks)
#     return [a['file_name'] for a in annot]
