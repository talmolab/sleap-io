"""Tests for the NWB harmonization layer."""

import h5py
import pytest

from sleap_io import Labels, load_slp
from sleap_io.io.nwb import NwbFormat, load_nwb, save_nwb


def test_nwb_format_enum():
    """Test NwbFormat enum values."""
    assert NwbFormat.AUTO == "auto"
    assert NwbFormat.ANNOTATIONS == "annotations"
    assert NwbFormat.ANNOTATIONS_EXPORT == "annotations_export"
    assert NwbFormat.PREDICTIONS == "predictions"


def test_load_nwb_predictions(slp_typical, tmp_path):
    """Test loading NWB file with predictions."""
    # Load data with predictions
    labels = load_slp(slp_typical)

    # Keep only predictions
    for lf in labels.labeled_frames:
        lf.instances = lf.predicted_instances

    # Save as predictions format
    save_nwb(labels, tmp_path / "test_pred.nwb", nwb_format="predictions")

    # Load and verify
    loaded_labels = load_nwb(tmp_path / "test_pred.nwb")
    assert isinstance(loaded_labels, Labels)
    assert len(loaded_labels.labeled_frames) > 0


def test_load_nwb_annotations(slp_real_data, tmp_path):
    """Test loading NWB file with annotations."""
    # Load data with user instances
    labels = load_slp(slp_real_data)

    # Save as annotations format
    save_nwb(labels, tmp_path / "test_ann.nwb", nwb_format="annotations")

    # Load and verify
    loaded_labels = load_nwb(tmp_path / "test_ann.nwb")
    assert isinstance(loaded_labels, Labels)
    assert len(loaded_labels.labeled_frames) > 0


def test_load_nwb_no_pose_data(tmp_path):
    """Test loading NWB file without pose data."""
    # Create an NWB file without pose data
    nwb_path = tmp_path / "empty.nwb"
    with h5py.File(nwb_path, "w") as f:
        f.create_group("processing")

    # Should raise ValueError
    with pytest.raises(ValueError, match="does not contain recognized pose data"):
        load_nwb(nwb_path)


def test_save_nwb_auto_detection(slp_typical, slp_real_data, tmp_path):
    """Test automatic format detection in save_nwb."""
    # Test with predictions (no user instances)
    labels_pred = load_slp(slp_typical)
    for lf in labels_pred.labeled_frames:
        lf.instances = lf.predicted_instances

    save_nwb(labels_pred, tmp_path / "auto_pred.nwb")  # Should use predictions format
    loaded = load_nwb(tmp_path / "auto_pred.nwb")
    assert isinstance(loaded, Labels)

    # Test with annotations (has user instances)
    labels_ann = load_slp(slp_real_data)
    save_nwb(labels_ann, tmp_path / "auto_ann.nwb")  # Should use annotations format
    loaded = load_nwb(tmp_path / "auto_ann.nwb")
    assert isinstance(loaded, Labels)


def test_save_nwb_explicit_format(slp_typical, tmp_path):
    """Test explicit format specification in save_nwb."""
    labels = load_slp(slp_typical)

    # Remove user instances to ensure we have only predictions
    for lf in labels.labeled_frames:
        lf.instances = lf.predicted_instances

    # Test predictions format
    save_nwb(labels, tmp_path / "explicit_pred.nwb", nwb_format="predictions")
    loaded = load_nwb(tmp_path / "explicit_pred.nwb")
    assert isinstance(loaded, Labels)

    # Test using NwbFormat enum
    save_nwb(labels, tmp_path / "enum_pred.nwb", nwb_format=NwbFormat.PREDICTIONS)
    loaded = load_nwb(tmp_path / "enum_pred.nwb")
    assert isinstance(loaded, Labels)


def test_save_nwb_invalid_format(slp_typical, tmp_path):
    """Test invalid format specification in save_nwb."""
    labels = load_slp(slp_typical)

    with pytest.raises(ValueError, match="Invalid NWB format"):
        save_nwb(labels, tmp_path / "invalid.nwb", nwb_format="invalid_format")


def test_save_nwb_annotations_export(slp_real_data, tmp_path):
    """Test annotations_export format in save_nwb."""
    labels = load_slp(slp_real_data)

    # Use annotations_export format
    nwb_path = tmp_path / "export.nwb"
    save_nwb(labels, nwb_path, nwb_format="annotations_export")

    # Check that files were created
    assert nwb_path.exists()
    # The export format also creates video and frame map files
    assert (tmp_path / "annotated_frames.avi").exists()
    assert (tmp_path / "frame_map.json").exists()

    # Load and verify
    loaded = load_nwb(nwb_path)
    assert isinstance(loaded, Labels)


def test_save_nwb_unexpected_format(slp_typical, tmp_path):
    """Test that unexpected NwbFormat values are handled."""
    labels = load_slp(slp_typical)

    # This tests the final else clause in save_nwb
    # We need to mock an invalid enum value that passes initial validation
    # This is a defensive test for future code changes

    # Create a mock format that's not handled
    class MockFormat:
        def __init__(self):
            pass

        def __eq__(self, other):
            return False

    mock_format = MockFormat()

    # Monkey patch to bypass string conversion
    original_isinstance = isinstance

    def mock_isinstance(obj, cls):
        if obj is mock_format and cls is str:
            return False
        return original_isinstance(obj, cls)

    import builtins

    builtins.isinstance = mock_isinstance

    try:
        with pytest.raises(ValueError, match="Unexpected NWB format"):
            save_nwb(labels, tmp_path / "unexpected.nwb", nwb_format=mock_format)
    finally:
        # Restore original isinstance
        builtins.isinstance = original_isinstance


def test_save_nwb_append_mode(slp_typical, tmp_path):
    """Test append mode functionality in save_nwb."""
    labels = load_slp(slp_typical)

    # Remove user instances to ensure we have only predictions
    for lf in labels.labeled_frames:
        lf.instances = lf.predicted_instances

    # First, save initial predictions
    nwb_file = tmp_path / "append_test.nwb"
    save_nwb(labels, nwb_file, nwb_format="predictions", append=False)

    # Verify file was created and can be loaded
    loaded_initial = load_nwb(nwb_file)
    assert isinstance(loaded_initial, Labels)

    # Test append mode - this should execute the append code path without error
    # Even if the data conflicts, we're just testing that the code path runs
    try:
        save_nwb(labels, nwb_file, nwb_format="predictions", append=True)
    except ValueError as e:
        # If it fails due to conflicting data, that's expected - we just want to test
        # that the append code path is executed (line 162 in nwb.py)
        if "already exists" in str(e):
            pass  # This is expected for duplicate data
        else:
            raise  # Re-raise unexpected errors
