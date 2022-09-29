import math
import os
from typing import Dict, List, Optional, Tuple, Union

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
        ma_nodes = [Node(bp) for bp in dlc_config["multianimalbodyparts"]]
        out.append(Skeleton(ma_nodes, name="multi"))

        uq_nodes = [Node(bp) for bp in dlc_config["uniquebodyparts"]]
        out.append(Skeleton(uq_nodes, name="unique"))

    else:
        nodes = [Node(bp) for bp in dlc_config["bodyparts"]]
        out.append(Skeleton(nodes, name="single"))

    return out


def load_dlc(dlc_config: dict) -> Labels:
    """Given a DLC configuration, load all labels from the training set.

    Parameters:
    dlc_config: DLC project configuration data

    Returns:
    Labels loaded from the DLC project
    """

    videos = dlc_config["video_sets"].keys()
    video_basenames = [os.path.splitext(os.path.basename(v))[0] for v in videos]

    all_annots = []
    for vb in video_basenames:
        annot_path = os.path.join(
            dlc_config["project_path"],
            "labeled-data",
            vb,
            f'CollectedData_{dlc_config["scorer"]}.h5',
        )
        if not os.path.exists(annot_path):
            raise FileExistsError(
                f'Unable to locate annotations, was expecting to find it here: "{annot_path}'
            )

        annots = pd.read_hdf(annot_path)
        all_annots.append(annots)

    return dlc_to_labels(pd.concat(all_annots), dlc_config)


def write_dlc(dlc_config: dict, labels: Labels):
    """Write Labels to a DLC project on disk
        writes both the csv format as well as the HDF5 format

    Parameters:
    dlc_config: DLC project configuration data
    labels: Labels to be written to the DLC project
    """

    split_labels = split_labels_by_directory(labels)

    grouped_dlc: Dict[str, pd.DataFrame] = {}
    for group, glabels in split_labels.items():
        grouped_dlc[group] = labels_to_dlc(glabels, dlc_config)

    for group, group_df in grouped_dlc.items():
        dest = os.path.join(
            dlc_config["project_path"],
            "labeled-data",
            group,
            f"CollectedData_{dlc_config['scorer']}",
        )

        group_df.to_csv(f"{dest}.csv")
        group_df.to_hdf(f"{dest}.h5", key="df_with_missing", mode="w")


def make_index_from_dlc_config(dlc_config: dict) -> pd.MultiIndex:
    """Given a DLC configuration, prepare a pandas multi-index

    Parameters:
    dlc_config: DLC project configuration data

    Returns:
    multiindex for dataframe columns
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


def split_labels_by_directory(labels: Labels) -> Dict[str, Labels]:
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

    return {group: Labels(frames) for group, frames in grouped.items()}


def dlc_to_labels(annots: pd.DataFrame, dlc_config: dict) -> Labels:
    """Convert a dlc-style dataframe to a Labels object"""
    skeletons = load_skeletons(dlc_config)
    frames: List[LabeledFrame] = []
    if is_multianimal(dlc_config):
        # Iterate over frames
        for frame in annots.index.values:
            ma_instances: List[Instance] = []

            # Iterate over individuals
            for indv in annots.columns.levels[1].values:

                # pick the correct skeleton to use
                skel: Skeleton
                if indv == "single":
                    # `single` is a sentinal for unique bodyparts
                    skel = next(filter(lambda s: s.name == "unique", skeletons))

                else:
                    # otherwise we are in a multi-animal context
                    skel = next(filter(lambda s: s.name == "multi", skeletons))

                points = {}
                # Iterate over this individual's body parts
                bodyparts = list(
                    set(
                        bp
                        for _, bp, _ in annots.columns.get_loc_level(
                            indv, level="individuals"
                        )[1].values
                    )
                )
                for node in bodyparts:
                    x_pos = (
                        annots.loc[frame, (dlc_config["scorer"], indv, node, "x")]
                        or np.nan
                    )
                    y_pos = (
                        annots.loc[frame, (dlc_config["scorer"], indv, node, "y")]
                        or np.nan
                    )

                    # If the value is a NAN, the user did not mark this keypoint
                    if math.isnan(x_pos) or math.isnan(y_pos):
                        continue

                    points[skel[node]] = Point(x_pos, y_pos)

                ma_instances.append(Instance(points, skel))

            ma_video, ma_frame_idx = video_from_index(frame, dlc_config)
            frames.append(LabeledFrame(ma_video, ma_frame_idx, ma_instances))

    else:
        # fetch the skeleton to use
        skel = next(filter(lambda s: s.name == "single", skeletons))
        # Iterate over frames
        for frame in annots.index.values:
            uq_instances: List[Instance] = []
            points = {}
            # Iterate over this individual's body parts
            for node in annots.columns.levels[1].values:
                x_pos = (
                    annots.loc[frame, (dlc_config["scorer"], indv, node, "x")] or np.nan
                )
                y_pos = (
                    annots.loc[frame, (dlc_config["scorer"], indv, node, "y")] or np.nan
                )

                # If the value is a NAN, the user did not mark this keypoint
                if math.isnan(x_pos) or math.isnan(y_pos):
                    continue

                points[skel[node]] = Point(x_pos, y_pos)

            uq_instances.append(Instance(points, skel))

            uq_video, uq_frame_idx = video_from_index(frame, dlc_config)
            frames.append(LabeledFrame(uq_video, uq_frame_idx, ma_instances))

    return Labels(frames)


def video_from_index(
    index: Union[str, Tuple[str]], dlc_config: dict
) -> Tuple[Video, int]:
    """Given an index from DLC-style dataframe, construct a video instance and the frame number"""
    vfname: str
    if isinstance(index, tuple):
        vfname = "/".join(index)
    else:
        vfname = str(index)
    video = Video(vfname)  # TODO: find a better way to represent the data
    frame_idx = (
        0  # TODO: Just putting zero for now. Maybe we can parse from the filename???
    )

    return video, frame_idx


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

        instance_names = list(dlc_config["individuals"]).copy()

        for instance in labeled_frame.instances:

            # determine individual type / identity
            instance_name: Optional[str] = None
            if is_ma:
                if instance.skeleton.name == "unique":
                    instance_name = "single"

                elif instance.skeleton.name == "multi":
                    instance_name = instance_names.pop(0)

                else:
                    raise ValueError(
                        "Type of instance is ambiguous. Skeleton should be named `unique` for unique body parts, or `multi` for multi-animal body parts!"
                    )

            for node, point in instance.points.items():
                key: Tuple
                if is_ma:
                    key = (
                        dlc_config["scorer"],
                        instance_name,
                        node.name,
                    )  # TODO - need to represent the unique bodyparts somehow!!

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
