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

## Usage

### Load and save in different formats

```py
import sleap_io as sio

# Load from SLEAP file.
labels = sio.load_file("predictions.slp")

# Save to NWB file.
labels.save("predictions.nwb")
```

**See also:** [`Labels.save`](model.md#sleap_io.Labels.save) and [Formats](formats.md)


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

**See also:** [`Labels.numpy`](model.md#sleap_io.Labels.numpy)


### Read video data

```py
import sleap_io as sio

video = sio.load_video("test.mp4")
n_frames, height, width, channels = video.shape

frame = video[0]
height, width, channels = frame.shape
```

**See also:** [`sio.load_video`](formats.md#sleap_io.load_video) and [`Video`](model.md#sleap_io.Video)



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

**See also:** [Model](model.md), [`Labels`](model.md#sleap_io.Labels),
[`LabeledFrame`](model.md#sleap_io.LabeledFrame),
[`Instance`](model.md#sleap_io.Instance),
[`PredictedInstance`](model.md#sleap_io.PredictedInstance),
[`Skeleton`](model.md#sleap_io.Skeleton), [`Video`](model.md#sleap_io.Video), [`Track`](model.md#sleap_io.Track), [`SuggestionFrame`](model.md#sleap_io.SuggestionFrame)


### Fix video paths

```py
import sleap_io as sio

labels = sio.load_file("labels.v001.slp")

# Fix paths using prefixes.
labels.replace_filenames(prefix_map={
    "D:/data/sleap_projects": "/home/user/sleap_projects",
    "C:/Users/sleaper/Desktop/test": "/home/user/sleap_projects",
})

labels.save("labels.v002.slp")
```

**See also:** [`Labels.replace_filenames`](model.md#sleap_io.Labels.replace_filenames)


### Save labels with embedded images

```py
import sleap_io as sio

# Load source labels.
labels = sio.load_file("labels.v001.slp")

# Save with embedded images for frames with user labeled data and suggested frames.
labels.save("labels.v001.pkg.slp", embed="user+suggestions")
```

**See also:** [`Labels.save`](model.md#sleap_io.Labels.save)


### Make training/validation/test splits

```py
import sleap_io as sio

# Load source labels.
labels = sio.load_file("labels.v001.slp")

# Make splits and export with embedded images.
labels.make_training_splits(n_train=0.8, n_val=0.1, n_test=0.1, save_dir="split1", seed=42)

# Splits will be saved as self-contained SLP package files with images and labels.
labels_train = sio.load_file("split1/train.pkg.slp")
labels_val = sio.load_file("split1/val.pkg.slp")
labels_test = sio.load_file("split1/test.pkg.slp")
```

**See also:** [`Labels.make_training_splits`](model.md#sleap_io.Labels.make_training_splits)


## Support
For technical inquiries specific to this package, please [open an Issue](https://github.com/talmolab/sleap-io/issues)
with a description of your problem or request.

For general SLEAP usage, see the [main website](https://sleap.ai).

Other questions? Reach out to `talmo@salk.edu`.

## License
This package is distributed under a BSD 3-Clause License and can be used without
restrictions. See [`LICENSE`](https://github.com/talmolab/sleap-io/blob/main/LICENSE) for details.