import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sleap_io import Instance, LabeledFrame, Labels, Node, Point, Skeleton, Video


def is_multianimal(dlc_config: dict) -> bool:
    """Returns True if `dlc_config` is configured for a multi-animal project, otherwise False

    Parameters:
    dlc_config (dict): DLC project configuration data

    Returns:
    bool: True if the config is setup for multi-animal projects, otherwise false
    """
    return "multianimalproject" in dlc_config and dlc_config["multianimalproject"]


def load_skeletons(dlc_config: dict) -> List[Skeleton]:
    """Load skeletons from a DLC configuration

    Parameters:
    dlc_config (dict): DLC project configuration data

    Returns:
    List of Sekeletons found in the DLC configuration
    """

    out = []
    if is_multianimal(dlc_config):
        ma_nodes = [Node(mabp) for mabp in dlc_config["multianimalbodyparts"]]
        ma_skeleton = Skeleton(ma_nodes, name="multi")
        out.append(ma_skeleton)

        uq_nodes = [Node(mabp) for mabp in dlc_config["uniquebodyparts"]]
        uq_skeleton = Skeleton(uq_nodes, name="unique")
        out.append(uq_skeleton)

    else:
        nodes = [Node(mabp) for mabp in dlc_config["bodyparts"]]
        out.append(Skeleton(nodes))

    return out


def load_dlc_annotations_for_image(dlc_config: dict, image_path: str) -> LabeledFrame:
    """Load existing DLC annotations for a given image. If no annotations could be found, None is returned

    Parameters:
    dlc_config (dict): DLC project configuration data
    image_path (str): path to an image file

    Returns:
    If annotation data can be found, a dictionary of annotation data is returned, otherwise None
    """
    is_ma = is_multianimal(dlc_config)
    skeletons = load_skeletons(dlc_config)
    try:
        img_rel_path = image_path.replace(dlc_config["project_path"] + os.path.sep, "")
        annot_path = os.path.join(
            os.path.dirname(image_path), f'CollectedData_{dlc_config["scorer"]}.h5'
        )
        if not os.path.exists(annot_path):
            raise FileExistsError(
                f'Unable to locate annotations, was expecting to find it here: "{annot_path}'
            )

        annots = pd.read_hdf(annot_path)

        instances: List[Instance] = []
        if is_ma:
            # Iterate over individuals
            for indv in annots.columns.levels[1].values:
                annots.index.get_loc_level(indv)[1].values

                points = {}
                # Iterate over this individual's body parts
                for node in annots.index.get_loc_level(indv)[1].values:
                    x_pos = annots.loc[
                        img_rel_path, (dlc_config["scorer"], indv, node, "x")
                    ]
                    y_pos = annots.loc[
                        img_rel_path, (dlc_config["scorer"], indv, node, "y")
                    ]

                    # If the value is a NAN, the user did not mark this keypoint
                    if math.isnan(x_pos) or math.isnan(y_pos):
                        continue

                    points[skeletons[0][node]] = Point(x_pos, y_pos)

                instances.append(Instance(points, skeletons[0]))

        else:
            points = {}
            # Iterate over this individual's body parts
            for node in annots.columns.levels[1].values:
                x_pos = annots.loc[
                    img_rel_path, (dlc_config["scorer"], indv, node, "x")
                ]
                y_pos = annots.loc[
                    img_rel_path, (dlc_config["scorer"], indv, node, "y")
                ]

                # If the value is a NAN, the user did not mark this keypoint
                if math.isnan(x_pos) or math.isnan(y_pos):
                    continue

                points[skeletons[0][node]] = Point(x_pos, y_pos)

            instances.append(Instance(points, skeletons[0]))

        video = Video(image_path)  # TODO: find a better way to represent the data
        frame_idx = 0  # TODO: Just putting zero for now. Maybe we can parse from the filename???

        return LabeledFrame(video, frame_idx, instances=instances)

    except:  # pylint: disable=bare-except
        raise


def make_index_from_dlc_config(dlc_config: dict) -> pd.MultiIndex:
    """Given a DLC configuration, prepare a pandas multi-index

    Parameters:
    dlc_config (dict): DLC project configuration data
    """
    if is_multianimal(dlc_config):
        cols = []
        for individual in dlc_config["individuals"]:
            for mabp in dlc_config["multianimalbodyparts"]:
                cols.append((dlc_config["scorer"], individual, mabp, "x"))
                cols.append((dlc_config["scorer"], individual, mabp, "y"))
        for unbp in dlc_config["uniquebodyparts"]:
            cols.append((dlc_config["scorer"], "single", unbp, "x"))
            cols.append((dlc_config["scorer"], "single", unbp, "y"))

        return pd.MultiIndex.from_tuples(
            cols, names=("scorer", "individuals", "bodyparts", "coords")
        )

    else:
        return pd.MultiIndex.from_product(
            [[dlc_config["scorer"]], dlc_config["bodyparts"], ["x", "y"]],
            names=["scorer", "bodyparts", "coords"],
        )


def split_labels_by_directory(labels: Labels) -> List[Labels]:
    """Split annotations into groups according to their file name

    Parameters:
    intermediate_annotations (List[dict]): "intermediate-style" annotations

    Returns:
    list of labels grouped by underlying video source
    """
    grouped: Dict[str, List[LabeledFrame]] = {}
    labels

    for labeled_frame in labels.labeled_frames:
        path, _ = os.path.split(labeled_frame.video.filename)
        _, group = os.path.split(path)
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(labeled_frame)

    return [Labels(frames) for group, frames in grouped.items()]


def labels_to_dlc(labels: Labels, dlc_config: dict) -> pd.DataFrame:
    """Convert a `Labels` instance to a DLC-format `pandas.DataFrame`"""

    is_ma = is_multianimal(dlc_config)
    col_idx = make_index_from_dlc_config(dlc_config)
    row_idx = []
    dlc_data: Dict[Tuple, List[float]] = {idx_val: [] for idx_val in col_idx.values}

    errors_found = 0
    for labeled_frame in labels.labeled_frames:
        row_idx.append(
            tuple(labeled_frame.video.filename.replace(r"\\", "/").split("/"))
        )
        # fill across the board with None
        for value in dlc_data.values():
            value.append(np.nan)

        for instance in labeled_frame.instances:

            for node, point in instance.points.items():
                key: Tuple
                if is_ma:
                    if instance["individual"] is None:
                        # unique bodypart
                        key = (
                            dlc_config["scorer"],
                            "single",
                            node.name,
                        )  # TODO - need to represent the unique bodyparts somehow!!
                    else:
                        # multi animal bodypart
                        key = (
                            dlc_config["scorer"],
                            instance["individual"],
                            node.name,
                        )  # TODO - need to represent the individual somehow!!
                else:
                    key = (dlc_config["scorer"], node.name)

                try:
                    dlc_data[(*key, "x")][-1] = point.x
                    dlc_data[(*key, "y")][-1] = point.y
                except KeyError:
                    errors_found += 1
                    # if annot['bodypart'] in dlc_config['multianimalbodyparts'] and annot['individual'] is None:
                    #     rationale = 'bodypart is a multianimal bodypart, but no relationship to an individual was found!'
                    # elif annot['bodypart'] in dlc_config['uniquebodyparts'] and annot['individual'] is not None:
                    #     rationale = 'bodypart is a unique bodypart and should not have a relationship with an individual, but one was found'
                    # else:
                    #     rationale = 'Unknown'

                    # message = 'ERROR! Data seems to violate the DLC annotation schema!\n' \
                    #         f' -> Task: {annot["task_id"]}\n' \
                    #         f' -> Image: "{annot["file_name"]}"\n' \
                    #         f' -> Bodypart: {annot["bodypart"]}\n'
                    # if is_ma:
                    #     message += f' -> Individual: {annot.get("individual", None)}\n'
                    # message += f' -> Rationale: {rationale}\n'
                    # print(message)
    row_index = pd.MultiIndex.from_tuples(row_idx)
    dlc_df = pd.DataFrame(dlc_data, index=row_index, columns=col_idx)

    return dlc_df
