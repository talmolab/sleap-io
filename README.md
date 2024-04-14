# sleap-io

[![CI](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-io/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-io)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-io?label=Latest)](https://github.com/talmolab/sleap-io/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-io?label=PyPI)](https://pypi.org/project/sleap-io)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-io)

Standalone utilities for working with animal pose tracking data.

This is intended to be a complement to the core [SLEAP](https://github.com/talmolab/sleap)
package that aims to provide functionality for interacting with pose tracking-related
data structures and file formats with minimal dependencies. This package *does not*
have any functionality related to labeling, training, or inference.

## Installation
```
pip install sleap-io
```

For development, use one of the following syntaxes:
```
conda env create -f environment.yml
```
```
pip install -e .[dev]
```
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information on development.

## Usage

### Load and save in different formats

```py
import sleap_io as sio

# Load from SLEAP file.
labels = sio.load_file("predictions.slp")

# Save to NWB file.
sio.save_file(labels, "predictions.nwb")
# Or:
# labels.save("predictions.nwb")
```

### Convert labels to raw arrays

```py
import sleap_io as sio

labels = sio.load_slp("tests/data/slp/centered_pair_predictions.slp")

# Convert predictions to point coordinates in a single array.
trx = labels.numpy()
n_frames, n_tracks, n_nodes, xy = trx.shape
assert xy == 2

# Convert to array with confidence scores appended.
trx_with_scores = labels.numpy(return_confidence=True)
n_frames, n_tracks, n_nodes, xy_score = trx.shape 
assert xy_score == 3
```

### Read video data

```py
import sleap_io as sio

video = sio.load_video("test.mp4")
n_frames, height, width, channels = video.shape

frame = video[0]
height, width, channels = frame.shape
```

### Create labels from raw data

```py
import sleap_io as sio
import numpy as np

# Create skeleton.
skeleton = sio.Skeleton(
    nodes=["head", "thorax", "abdomen"],
    edges=[("head", "thorax"), ("thorax", "abdomen")]
)

# Create video.
video = sio.load_video("test.mp4")

# Create instance.
instance = sio.Instance.from_numpy(
    points=np.array([
        [10.2, 20.4],
        [5.8, 15.1],
        [0.3, 10.6],
    ]),
    skeleton=skeleton
)

# Create labeled frame.
lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])

# Create labels.
labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

# Save.
labels.save("labels.slp")
```

## Support
For technical inquiries specific to this package, please [open an Issue](https://github.com/talmolab/sleap-io/issues)
with a description of your problem or request.

For general SLEAP usage, see the [main website](https://sleap.ai).

Other questions? Reach out to `talmo@salk.edu`.

## License
This package is distributed under a BSD 3-Clause License and can be used without
restrictions. See [`LICENSE`](LICENSE) for details.