"""Tests for functions in the sleap_io.io.labelstudio file."""

from sleap_io import Labels
from sleap_io.io.labelstudio import convert_labels, parse_tasks, read_labels
from sleap_io.io.slp import read_labels as slp_read_labels


def round_trip_labels(labels: Labels) -> Labels:
    ls_labels = parse_tasks(convert_labels(labels), labels.skeletons[0])
    return ls_labels


def test_labels_round_trip(slp_typical, slp_simple_skel, slp_minimal):
    """Test `read_labels` can read different types of .slp files."""
    # first on `slp_typical`
    labels = slp_read_labels(slp_typical)
    assert type(labels) is Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) is Labels

    # now on `slp_simple_skel`
    labels = slp_read_labels(slp_simple_skel)
    assert type(labels) is Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) is Labels

    # now on `slp_minimal`
    labels = slp_read_labels(slp_minimal)
    assert type(labels) is Labels
    ls_labels = round_trip_labels(labels)
    assert type(ls_labels) is Labels


def test_read_labels(ls_multianimal):
    file_path, skeleton = ls_multianimal

    ls_labels = read_labels(file_path, skeleton)
    _ = round_trip_labels(ls_labels)
    # assert ls_labels == rt_labels # TODO(TP): Fix equality check
