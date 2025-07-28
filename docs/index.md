# sleap-io

[![CI](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-io/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-io)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-io?label=Latest)](https://github.com/talmolab/sleap-io/releases/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-io)
[![PyPI](https://img.shields.io/pypi/v/sleap-io?label=PyPI)](https://pypi.org/project/sleap-io)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/sleap-io.svg)](https://anaconda.org/conda-forge/sleap-io/)

Standalone utilities for working with animal pose tracking data.

This is intended to be a complement to the core [SLEAP](https://github.com/talmolab/sleap)
package that aims to provide functionality for interacting with pose tracking-related
data structures and file formats with minimal dependencies. This package *does not*
have any functionality related to labeling, training, or inference.

## Features

The main purpose of this library is to provide utilities to load/save from different
[formats](formats.md) for pose data and standardize them into our common [Data Model](model.md).

- Read/write labels in [SLP](formats.md#sleap_io.load_slp), [NWB](formats.md#sleap_io.load_nwb), [DeepLabCut](formats.md#sleap_io.load_dlc), [JABS](formats.md#sleap_io.load_jabs), [LabelStudio](formats.md#sleap_io.load_labelstudio) and [Ultralytics YOLO](formats.md#sleap_io.load_ultralytics) formats.
- Support for [LabelsSet](model.md#sleap_io.LabelsSet) to manage multiple dataset splits (train/val/test) and export to different formats.
- [Read videos in any format](formats.md#sleap_io.load_video), work them in a [numpy-like interface](model.md#sleap_io.Video) whether the video files are accessible or not, and [easily save them out](formats.md#sleap_io.save_video).

This enables ease-of-use through format-agnostic operations that make it easy to work
with pose data, including utilities for common tasks. Some of these include:

- [Create labels from a custom format](examples.md#create-labels-from-raw-data)
- [Convert labels to numpy arrays for analysis](examples.md#convert-labels-to-raw-arrays)
- [Fix video paths in the labels](examples.md#fix-video-paths)
- [Make training/validation/test splits](examples.md#make-trainingvalidationtest-splits)
- [Work with dataset splits using LabelsSet](examples.md#working-with-dataset-splits-labelsset)
- [Replace a skeleton](examples.md#replace-skeleton)

See [Examples](examples.md) for more usage examples and recipes.


## Installation
```
pip install sleap-io
```

or

```
conda install -c conda-forge sleap-io
```

For development, use one of the following syntaxes:
```
conda env create -f environment.yml
```
```
pip install -e .[dev]
```


## Support
For technical inquiries specific to this package, please [open an Issue](https://github.com/talmolab/sleap-io/issues)
with a description of your problem or request.

For general SLEAP usage, see the [main website](https://sleap.ai).

Other questions? Reach out to `talmo@salk.edu`.

## License
This package is distributed under a BSD 3-Clause License and can be used without
restrictions. See [`LICENSE`](https://github.com/talmolab/sleap-io/blob/main/LICENSE) for details.