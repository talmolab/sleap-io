# sleap-io

[![CI](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-io/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-io)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-io?label=Latest)](https://github.com/talmolab/sleap-io/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-io?label=PyPI)](https://pypi.org/project/sleap-io)
<!-- TODO: ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-io) -->

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
```
import sleap_io as sio
import numpy as np

skeleton = sio.Skeleton(
    nodes=["head", "thorax", "abdomen"],
    edges=[("head", "thorax"), ("thorax", "abdomen")]
)

instance = sio.Instance.from_numpy(
    points=np.array([
        [10.2, 20.4],
        [5.8, 15.1],
        [0.3, 10.6],
    ]),
    skeleton=skeleton
)
```

## Support
For technical inquiries specific to this package, please [open an Issue](https://github.com/talmolab/sleap-io/issues)
with a description of your problem or request.

For general SLEAP usage, see the [main website](https://sleap.ai).

Other questions? Reach out to `talmo@salk.edu`.

## License
This package is distributed under a BSD 3-Clause License and can be used without
restrictions. See [`LICENSE`](LICENSE) for details.