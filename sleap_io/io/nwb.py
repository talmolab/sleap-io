"""Harmonization layer for NWB I/O operations.

This module provides a unified interface for reading and writing SLEAP data to/from
NWB files, automatically detecting and routing to the appropriate backend based on
the data format (annotations vs predictions).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Union

import h5py

if TYPE_CHECKING:
    from sleap_io.model.labels import Labels


class NwbFormat(str, Enum):
    """NWB format types for SLEAP data."""

    AUTO = "auto"
    ANNOTATIONS = "annotations"
    ANNOTATIONS_EXPORT = "annotations_export"
    PREDICTIONS = "predictions"


def load_nwb(filename: Union[str, Path]) -> Labels:
    """Load an NWB dataset as a SLEAP Labels object.
    
    Automatically detects whether the file contains PoseTraining (annotations)
    or PoseEstimation (predictions) data and uses the appropriate backend.
    
    Args:
        filename: Path to an NWB file (.nwb).
        
    Returns:
        The dataset as a Labels object.
        
    Raises:
        ValueError: If the NWB file doesn't contain recognized pose data.
    """
    from sleap_io.io import nwb_annotations, nwb_predictions
    
    filename = Path(filename)
    
    # Check what type of data is in the file
    with h5py.File(filename, "r") as f:
        # Check for behavior processing module
        if "processing" in f and "behavior" in f["processing"]:
            behavior = f["processing"]["behavior"]
            
            # Check for PoseTraining (annotations)
            if "PoseTraining" in behavior:
                return nwb_annotations.load_labels(filename)
            
            # Check for PoseEstimation (predictions)  
            # Look for any key that's not one of the standard containers
            pose_estimation_found = False
            for key in behavior.keys():
                # PoseEstimation containers are named after tracks
                if key not in ["PoseTraining", "Skeletons"]:
                    # Verify it's actually a PoseEstimation container by checking attributes
                    if "neurodata_type" in behavior[key].attrs:
                        if behavior[key].attrs["neurodata_type"] == "PoseEstimation":
                            pose_estimation_found = True
                            break
            
            if pose_estimation_found:
                return nwb_predictions.read_nwb(filename)
    
    raise ValueError(
        f"NWB file '{filename}' does not contain recognized pose data "
        "(neither PoseTraining nor PoseEstimation found in behavior module)"
    )


def save_nwb(
    labels: Labels,
    filename: Union[str, Path],
    nwb_format: Union[NwbFormat, str] = NwbFormat.AUTO,
) -> None:
    """Save a SLEAP dataset to NWB format.
    
    Args:
        labels: A SLEAP Labels object to save.
        filename: Path to NWB file to save to. Must end in '.nwb'.
        nwb_format: Format to use for saving. Options are:
            - "auto" (default): Automatically detect based on data
            - "annotations": Save training annotations (PoseTraining)
            - "annotations_export": Export annotations with video frames
            - "predictions": Save predictions (PoseEstimation)
            
    Raises:
        ValueError: If an invalid format is specified.
    """
    from sleap_io.io import nwb_annotations, nwb_predictions
    
    filename = Path(filename)
    
    # Convert string to enum if needed
    if isinstance(nwb_format, str):
        try:
            nwb_format = NwbFormat(nwb_format)
        except ValueError:
            raise ValueError(
                f"Invalid NWB format: '{nwb_format}'. "
                f"Must be one of: {', '.join(f.value for f in NwbFormat)}"
            )
    
    # Auto-detect format if needed
    if nwb_format == NwbFormat.AUTO:
        # Check if there are any user instances
        has_user_instances = any(
            lf.has_user_instances for lf in labels.labeled_frames
        )
        
        if has_user_instances:
            nwb_format = NwbFormat.ANNOTATIONS
        else:
            nwb_format = NwbFormat.PREDICTIONS
    
    # Route to appropriate backend
    if nwb_format == NwbFormat.ANNOTATIONS:
        nwb_annotations.save_labels(labels, filename)
    elif nwb_format == NwbFormat.ANNOTATIONS_EXPORT:
        # Use export_labels for the export format
        output_dir = filename.parent
        nwb_filename = filename.name
        nwb_annotations.export_labels(
            labels,
            output_dir=output_dir,
            nwb_filename=nwb_filename,
            clean=True,  # Clean up intermediate files
        )
    elif nwb_format == NwbFormat.PREDICTIONS:
        # Always overwrite (no append)
        nwb_predictions.write_nwb(labels, filename)
    else:
        raise ValueError(f"Unexpected NWB format: {nwb_format}")