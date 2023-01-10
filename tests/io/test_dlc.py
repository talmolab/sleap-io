"""Tests for functions in the sleap_io.io.labelstudio file."""
from typing import Optional

from sleap_io import Labels
from sleap_io.io.dlc import (
    get_skeleton_type,
    labels_to_dlc,
    dlc_to_labels,
    load_dlc,
    load_skeletons,
)
from sleap_io.io.slp import read_labels as slp_read_labels


def round_trip_labels(labels: Labels, dlc_config: Optional[dict] = None) -> Labels:
    if dlc_config is None:
        dlc_config = dlc_config_from_labels(labels)
    ls_labels = dlc_to_labels(labels_to_dlc(labels, dlc_config), dlc_config)
    return ls_labels


def dlc_config_from_labels(labels: Labels) -> dict:
    """Attempt to create a minimal functioning DLC config given a `Labels` instance.
    Intended only as a shim for testing purposes
    """
    dlc_config = {
        "project_path": "",
        "scorer": "johndoe",
        "multianimalproject": True,
        "individuals": [],
        "multianimalbodyparts": [],
        "uniquebodyparts": [],
        "bodyparts": "MULTI!",
        "video_sets": {},
    }

    max_instances = max(
        [
            max(len(lf.user_instances), len(lf.predicted_instances))
            for lf in labels.labeled_frames
        ]
    )
    dlc_config["individuals"] = [f"animal{i}" for i in range(max_instances)]

    dlc_config["multianimalbodyparts"] = [
        node for node in labels.skeletons[0].node_names
    ]

    return dlc_config


def test_labels_round_trip(slp_typical, slp_simple_skel, slp_minimal):
    """Test `read_labels` can read different types of .slp files."""

    # first on `slp_typical`
    labels = slp_read_labels(slp_typical)
    assert type(labels) == Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) == Labels

    # now on `slp_simple_skel`
    labels = slp_read_labels(slp_simple_skel)
    assert type(labels) == Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) == Labels

    # now on `slp_minimal`
    labels = slp_read_labels(slp_minimal)
    assert type(labels) == Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) == Labels


def test_read_labels(dlc_project_config):
    """Test round trip load and write from a DLC project"""
    labels = load_dlc(dlc_project_config)
    labels2 = round_trip_labels(labels, dlc_project_config)
    assert labels == labels2


def test_get_skeleton_type(dlc_project_config):
    skeletons = load_skeletons(dlc_project_config)
    for sk in skeletons:
        assert sk.name == get_skeleton_type(sk, dlc_project_config)
