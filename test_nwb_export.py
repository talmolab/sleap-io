#!/usr/bin/env python
"""Temporary script to convert SLEAP labels to NWB annotation format."""

from pathlib import Path
import sleap_io as sio

# Input and output paths
input_path = Path("../Downloads/val.pkg.slp")
output_dir = Path("./tmp")
output_path = output_dir / "val.nwb"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading labels from: {input_path}")
labels = sio.load_file(str(input_path))

print(f"Loaded {len(labels)} labeled frames")
print(f"Number of videos: {len(labels.videos)}")
print(f"Number of skeletons: {len(labels.skeletons)}")
if labels.skeletons:
    for skeleton in labels.skeletons:
        print(f"  - Skeleton '{skeleton.name}' with {len(skeleton.nodes)} nodes")

print(f"\nSaving to NWB annotation format: {output_path}")
sio.save_nwb_annotations(
    labels,
    str(output_path),
    output_dir=str(output_dir),  # This is where frame_map.json and annotated_frames.avi will be saved
)
print(f"\nSuccessfully saved NWB file to: {output_path}")
print(f"Additional files created in {output_dir}:")
print(f"  - frame_map.json (frame mapping)")
print(f"  - annotated_frames.avi (MJPEG video of annotated frames)")
