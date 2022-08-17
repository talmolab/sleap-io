from sleap_io import load_slp
from sleap_io import write_labels_to_nwb, append_labels_data_to_nwb

TEST_SLP_PREDICTIONS = "tests/data/hdf5_format_v1/centered_pair_predictions.slp"

labels = load_slp(TEST_SLP_PREDICTIONS)

def test_complex_file():
    labels = load_slp(TEST_SLP_PREDICTIONS)
    nwbfile_path = "temporal_test.nwb"
    write_labels_to_nwb(labels, nwbfile_path)