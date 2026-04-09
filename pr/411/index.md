# sleap-io

[![CI](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-io/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-io)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-io?label=Latest)](https://github.com/talmolab/sleap-io/releases/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-io)
[![PyPI](https://img.shields.io/pypi/v/sleap-io?label=PyPI)](https://pypi.org/project/sleap-io)

A standalone Python library and CLI for working with animal pose tracking data.
Read, write, convert, and manipulate pose data across formats with minimal
dependencies.

Complements the core [SLEAP](https://github.com/talmolab/sleap) package but
does *not* include labeling, training, or inference.

## Features

- **Multi-format I/O** -- Read and write [SLEAP](formats/index.md#sleap-native-format-slp), [NWB](formats/index.md#nwb-format-nwb), [COCO](formats/index.md#coco-format-json), [DeepLabCut](formats/index.md#deeplabcut-format-h5-csv), [Ultralytics YOLO](formats/index.md#ultralytics-yolo-format), [JABS](formats/index.md#jabs-format-h5), [Label Studio](formats/index.md#label-studio-format-json), [CSV](formats/index.md#csv-format-csv), [Analysis HDF5](formats/index.md#sleap-analysis-hdf5-format-h5), [AlphaTracker](formats/index.md#alphatracker-format), and [LEAP](formats/index.md#leap-format-mat) formats
- **CLI tools** -- Inspect, convert, render, and transform data from the command line ([reference](cli.md))
- **Rendering** -- Produce publication-quality videos and images with pose overlays, customizable colors, markers, and presets ([guide](rendering.md))
- **Transforms** -- Crop, scale, rotate, pad, and flip videos with automatic coordinate adjustment ([guide](transforms.md))
- **Merging** -- Combine annotations from multiple sources with flexible matching strategies ([guide](merging.md))
- **Codecs** -- Convert to/from NumPy arrays, DataFrames (pandas/polars), and dictionaries ([guide](codecs.md))
- **Video I/O** -- Read any video format via pluggable backends (FFMPEG, OpenCV, PyAV) with a NumPy-like interface ([model](model/video.md))
- **Lazy loading** -- Load large SLP files up to 90x faster by deferring object creation ([details](formats/slp.md#lazy-loading))
- **Dataset splits** -- Create train/val/test splits and export to formats like Ultralytics YOLO ([example](examples.md#make-trainingvalidationtest-splits))

## Installation

```bash
pip install "sleap-io[all]"
```

Or use without installing:

```bash
uvx sleap-io show labels.slp
```

See [Installation](install.md) for all options including `uv`, `conda`, CLI tool install, and development setup.

## Quick start

### CLI

```bash
sio show labels.slp                                    # Inspect a file
sio convert -i labels.slp -o labels.nwb                # Convert formats
sio render -i predictions.slp -o output.mp4            # Render video
sio transform labels.slp --scale 0.5 -o scaled.slp    # Transform
```

### Python

```python
import sleap_io as sio

# Load and convert between formats
labels = sio.load_file("predictions.slp")
labels.save("predictions.nwb")

# Convert to NumPy arrays
trx = labels.numpy()  # (n_frames, n_tracks, n_nodes, 2)

# Merge annotations from multiple sources
base = sio.load_file("manual_annotations.slp")
base.merge(sio.load_file("predictions.slp"))
base.save("merged.slp")
```

See [Examples](examples.md) for more recipes including creating labels from scratch, NWB export, rendering, skeleton replacement, and YOLO/COCO export.

## Support

For technical inquiries, please [open an Issue](https://github.com/talmolab/sleap-io/issues).

For general SLEAP usage, see [sleap.ai](https://sleap.ai).

## License

BSD 3-Clause License. See [`LICENSE`](https://github.com/talmolab/sleap-io/blob/main/LICENSE) for details.