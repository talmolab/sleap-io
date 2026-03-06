# sleap-io

[![CI](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-io/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-io)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-io?label=Latest)](https://github.com/talmolab/sleap-io/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-io?label=PyPI)](https://pypi.org/project/sleap-io)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-io)

A standalone Python library and CLI for working with animal pose tracking data. Read, write, convert, and manipulate pose data across formats with minimal dependencies.

Complements the core [SLEAP](https://github.com/talmolab/sleap) package but does *not* include labeling, training, or inference.

**[Documentation](https://io.sleap.ai)** | **[Examples](https://io.sleap.ai/examples)** | **[CLI Reference](https://io.sleap.ai/cli)**

## Features

- **Multi-format I/O** -- Read and write [SLEAP](https://io.sleap.ai/formats/#sleap-native-format-slp), [NWB](https://io.sleap.ai/formats/#nwb-format-nwb), [COCO](https://io.sleap.ai/formats/#coco-format-json), [DeepLabCut](https://io.sleap.ai/formats/#deeplabcut-format-h5-csv), [Ultralytics YOLO](https://io.sleap.ai/formats/#ultralytics-yolo-format), [JABS](https://io.sleap.ai/formats/#jabs-format-h5), [Label Studio](https://io.sleap.ai/formats/#label-studio-format-json), [CSV](https://io.sleap.ai/formats/#csv-format-csv), [Analysis HDF5](https://io.sleap.ai/formats/#sleap-analysis-hdf5-format-h5), [AlphaTracker](https://io.sleap.ai/formats/#alphatracker-format), and [LEAP](https://io.sleap.ai/formats/#leap-format-mat) formats
- **CLI tools** -- Inspect, convert, render, and transform data from the command line (`sio show`, `sio convert`, `sio render`, `sio transform`)
- **Rendering** -- Produce publication-quality videos and images with pose overlays, customizable colors, markers, and presets
- **Transforms** -- Crop, scale, rotate, pad, and flip videos with automatic coordinate adjustment
- **Merging** -- Combine annotations from multiple sources with flexible matching strategies
- **Codecs** -- Convert to/from NumPy arrays, DataFrames (pandas/polars), and dictionaries
- **Video I/O** -- Read any video format via pluggable backends (FFMPEG, OpenCV, PyAV) with a NumPy-like interface
- **Lazy loading** -- Load large SLP files up to 90x faster by deferring object creation
- **Dataset splits** -- Create train/val/test splits and export to formats like Ultralytics YOLO

## Installation

### Quick start (no install needed)

Run CLI commands instantly with [`uvx`](https://docs.astral.sh/uv/):

```bash
uvx sleap-io show labels.slp
uvx sleap-io convert -i labels.slp -o labels.nwb
```

### Install as CLI tool

```bash
uv tool install "sleap-io[all]"
sio show labels.slp
```

### Install as Python library

```bash
pip install "sleap-io[all]"
# or: uv add "sleap-io[all]"
# or: conda install -c conda-forge sleap-io
```

### From source

```bash
pip install "sleap-io[all] @ git+https://github.com/talmolab/sleap-io.git@main"
```

### Optional extras

Video support works out of the box via `imageio-ffmpeg`. Optional extras provide faster backends and additional format support:

| Extra | Purpose |
|-------|---------|
| `opencv` | Faster video backend |
| `pyav` | Alternative video backend |
| `mat` | LEAP `.mat` file support |
| `polars` | Fast DataFrame operations |
| `all` | All of the above |

### Development

```bash
git clone https://github.com/talmolab/sleap-io.git && cd sleap-io
uv sync --all-extras
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more details.

## Usage

### CLI

```bash
# Inspect a labels file
sio show labels.slp

# Convert between formats
sio convert -i labels.slp -o labels.nwb

# Render video with pose overlays
sio render -i predictions.slp -o output.mp4

# Transform (scale, crop, rotate) with coordinate adjustment
sio transform labels.slp --scale 0.5 -o scaled.slp
```

### Python

#### Load and convert between formats

```python
import sleap_io as sio

labels = sio.load_file("predictions.slp")
labels.save("predictions.nwb")
```

Format is auto-detected from the extension. See [supported formats](https://io.sleap.ai/formats/).

#### Convert to NumPy arrays

```python
labels = sio.load_file("predictions.slp")

trx = labels.numpy()  # (n_frames, n_tracks, n_nodes, 2)
trx_with_scores = labels.numpy(return_confidence=True)  # (n_frames, n_tracks, n_nodes, 3)
```

#### Create labels from scratch

```python
import sleap_io as sio
import numpy as np

skeleton = sio.Skeleton(
    nodes=["head", "thorax", "abdomen"],
    edges=[("head", "thorax"), ("thorax", "abdomen")]
)

instance = sio.Instance.from_numpy(
    points=np.array([[10.2, 20.4], [5.8, 15.1], [0.3, 10.6]]),
    skeleton=skeleton
)

video = sio.load_video("test.mp4")
lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])
labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])
labels.save("labels.slp")
```

#### Render poses

```python
labels = sio.load_file("predictions.slp")
labels.render("output.mp4")                      # Full video
labels.render("preview.mp4", preset="preview")   # Fast 0.25x preview
sio.render_image(labels[0], "frame.png")         # Single frame
```

#### Merge annotations

```python
base = sio.load_file("manual_annotations.slp")
predictions = sio.load_file("predictions.slp")
base.merge(predictions)
base.save("merged.slp")
```

#### Create training splits

```python
labels = sio.load_file("labels.slp")
labels.make_training_splits(n_train=0.8, n_val=0.1, n_test=0.1, save_dir="splits/", seed=42)
```

See the **[Examples](https://io.sleap.ai/examples)** page for more recipes including NWB export, video re-encoding, skeleton replacement, path fixing, and YOLO/COCO export.

## Support

For technical inquiries, please [open an Issue](https://github.com/talmolab/sleap-io/issues).

For general SLEAP usage, see [sleap.ai](https://sleap.ai).

## License

BSD 3-Clause License. See [`LICENSE`](LICENSE) for details.
