"""Tests for functions in the sleap_io.io.labelstudio file."""
from sleap_io import Labels
from sleap_io.io.dlc import labels_to_dlc, dlc_to_labels, load_dlc
from sleap_io.io.slp import read_labels as slp_read_labels


def round_trip_labels(labels: Labels, dlc_config: dict) -> Labels:
    ls_labels = dlc_to_labels(labels_to_dlc(labels, dlc_config), dlc_config)
    return ls_labels


# def test_labels_round_trip(slp_typical, slp_simple_skel, slp_minimal):
#     """Test `read_labels` can read different types of .slp files."""

#     # first on `slp_typical`
#     labels = slp_read_labels(slp_typical)
#     assert type(labels) == Labels
#     ls_labels = round_trip_labels(labels)
#     assert type(ls_labels) == Labels

#     # now on `slp_simple_skel`
#     labels = slp_read_labels(slp_simple_skel)
#     assert type(labels) == Labels
#     ls_labels = round_trip_labels(labels)
#     assert type(ls_labels) == Labels

#     # now on `slp_minimal`
#     labels = slp_read_labels(slp_minimal)
#     assert type(labels) == Labels
#     ls_labels = round_trip_labels(labels)
#     assert type(ls_labels) == Labels


def test_read_labels(dlc_project_config):

    labels = load_dlc(dlc_project_config)
    labels2 = round_trip_labels(labels, dlc_project_config)
    assert labels == labels2
