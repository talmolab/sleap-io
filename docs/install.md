# Installation

sleap-io can be used as a CLI tool, a Python library, or a dependency of another package. Choose the installation method that best fits your use case.

---

## Quick Start

=== "One-off CLI usage"

    Run commands without installing anything:

    ```bash
    uvx sleap-io show labels.slp
    uvx sleap-io convert -i labels.slp -o labels.nwb
    ```

    Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). See [CLI Reference](cli.md) for available commands.

=== "Install CLI tool"

    Install as a system-wide command:

    ```bash
    uv tool install "sleap-io[all]"
    ```

    Then use:

    ```bash
    sio show labels.slp
    ```

=== "Python library"

    Add to your project:

    ```bash
    pip install "sleap-io[all]"
    # or: uv add "sleap-io[all]"
    ```

    Then import:

    ```python
    import sleap_io as sio
    labels = sio.load_file("labels.slp")
    ```

---

## Use Cases

### As a CLI utility for working with pose data

Use sleap-io to inspect, convert, and manipulate pose tracking files from the command line.

#### One-off usage (no installation)

The fastest way to use sleap-io is with [`uvx`](https://docs.astral.sh/uv/), which runs CLI tools in temporary isolated environments:

```bash
# Inspect a labels file
uvx sleap-io show labels.slp

# Convert between formats
uvx sleap-io convert -i labels.slp -o labels.nwb

# Use with all optional dependencies (requires --from for extras)
uvx --from "sleap-io[all]" sleap-io show labels.slp
```

This is ideal for:

- Quick one-off conversions
- CI/CD pipelines
- Trying sleap-io without installing

!!! tip "Installing uv"
    If you don't have `uv`, install it with:

    === "macOS/Linux"
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    === "Windows"
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

    === "pip"
        ```bash
        pip install uv
        ```

#### System-wide installation

For regular use, install sleap-io as a global tool:

=== "From PyPI"

    ```bash
    uv tool install "sleap-io[all]"
    ```

=== "From source (latest)"

    ```bash
    uv tool install "git+https://github.com/talmolab/sleap-io.git@main"
    ```

    !!! note
        Extras cannot be specified when installing from git. Install without extras or use the PyPI version.

After installation, use either `sio` or `sleap-io` as the command:

```bash
sio show labels.slp
sio convert -i labels.slp -o labels.nwb
sio --version
```

!!! info "The `[all]` extra"
    The `[all]` extra includes faster video backends (OpenCV, PyAV) and additional format support. You can omit it for a minimal installation:

    ```bash
    uv tool install sleap-io  # Basic installation
    ```

    Video support via imageio-ffmpeg is always included.

See the [CLI documentation](cli.md) for a complete command reference.

---

### As a Python utility library for manipulating pose data

Use sleap-io programmatically to load, manipulate, and save pose tracking data.

#### In a project environment

=== "pip"

    ```bash
    pip install "sleap-io[all]"
    ```

=== "uv pip"

    ```bash
    uv pip install "sleap-io[all]"
    ```

=== "uv add (pyproject.toml)"

    ```bash
    uv add "sleap-io[all]"
    ```

=== "conda"

    ```bash
    conda install -c conda-forge sleap-io
    ```

    !!! note "conda extras"
        The conda package includes all backends by default. Optional dependencies can be installed separately if needed.

#### From source (latest development version)

=== "pip"

    For environments not managed by uv:

    ```bash
    pip install "sleap-io[all]" @ git+https://github.com/talmolab/sleap-io.git@main
    ```

=== "uv pip"

    For quick installation into an existing virtual environment:

    ```bash
    uv pip install "sleap-io[all] @ git+https://github.com/talmolab/sleap-io.git@main"
    ```

=== "uv add (pyproject.toml)"

    For uv-managed projects (adds to pyproject.toml and syncs):

    ```bash
    uv add "sleap-io[all]" --git https://github.com/talmolab/sleap-io.git --branch main
    ```

#### Quick usage example

```python title="quick_demo.py"
# /// script
# dependencies = ["sleap-io[all]"]
# ///
import sleap_io as sio

# Load labels from any supported format
labels = sio.load_file("labels.slp")

# Inspect the data
print(f"Videos: {len(labels.videos)}")
print(f"Labeled frames: {len(labels.labeled_frames)}")
print(f"Skeleton: {labels.skeleton.node_names}")

# Convert to numpy for analysis
locations = labels.numpy()  # shape: (frames, tracks, nodes, 2)

# Save to a different format
sio.save_nwb(labels, "labels.nwb")
```

Run directly with uv (no installation needed):

```bash
uv run quick_demo.py
```

See [Examples](examples.md) for more usage patterns.

---

### As a dependency of another package or tool

Add sleap-io to your project's dependencies for pose data I/O.

#### In pyproject.toml

=== "Basic dependency"

    ```toml
    [project]
    dependencies = [
        "sleap-io[all]>=0.6.0",
    ]
    ```

=== "Optional dependency"

    ```toml
    [project.optional-dependencies]
    pose = ["sleap-io[all]>=0.6.0"]
    ```

#### From source (Git dependency)

For development versions or unreleased features:

=== "pyproject.toml (pip)"

    ```toml
    [project]
    dependencies = [
        "sleap-io[all] @ git+https://github.com/talmolab/sleap-io.git@main",
    ]
    ```

=== "pyproject.toml (uv)"

    ```toml
    [project]
    dependencies = [
        "sleap-io[all]",
    ]

    [tool.uv.sources]
    sleap-io = { git = "https://github.com/talmolab/sleap-io.git", branch = "main" }
    ```

=== "requirements.txt"

    ```
    sleap-io[all] @ git+https://github.com/talmolab/sleap-io.git@main
    ```

#### Pinning specific versions

For reproducible builds, pin to a specific version or commit:

```toml
# Pin to version
"sleap-io[all]==0.6.0"

# Pin to specific commit
"sleap-io[all] @ git+https://github.com/talmolab/sleap-io.git@abc1234"

# Pin to tag
"sleap-io[all] @ git+https://github.com/talmolab/sleap-io.git@v0.6.0"
```

---

### As a developer looking to extend capabilities

Set up a development environment to contribute to sleap-io or add new format support.

#### Clone and install

```bash
# Clone the repository
git clone https://github.com/talmolab/sleap-io.git
cd sleap-io

# Install with development dependencies (recommended)
uv sync --all-extras
```

This installs:

- All optional dependencies (`opencv`, `pyav`, `mat`, `polars`)
- Development tools (`pytest`, `ruff`, `mkdocs`)
- The package in editable mode

#### Alternative installation methods

=== "pip editable install"

    ```bash
    pip install -e .[dev,all]
    ```

=== "conda + pip"

    ```bash
    conda env create -f environment.yml
    conda activate sleap-io
    pip install -e .
    ```

#### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sleap_io

# Run specific test file
pytest tests/io/test_slp.py
```

#### Code style

```bash
# Format code
ruff format

# Check linting
ruff check

# Fix auto-fixable issues
ruff check --fix
```

#### Building documentation

```bash
# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

See [CLAUDE.md](https://github.com/talmolab/sleap-io/blob/main/CLAUDE.md) for detailed development guidelines.

---

## Optional Dependencies

sleap-io uses optional dependencies for specific features:

| Extra | Packages | Purpose |
|-------|----------|---------|
| `opencv` | `opencv-python` | Fastest video backend (2-3x faster) |
| `pyav` | `av` | Balanced speed/features video backend |
| `mat` | `pymatreader` | LEAP `.mat` file support |
| `polars` | `polars`, `pyarrow` | Fast dataframe operations |
| `all` | All of the above | Everything included |

Install specific extras:

```bash
pip install "sleap-io[opencv]"      # Just OpenCV
pip install "sleap-io[opencv,mat]"  # Multiple extras
pip install "sleap-io[all]"         # Everything
```

!!! info "Default video support"
    Video reading works out of the box via `imageio-ffmpeg`, which is always installed. The optional video backends provide faster performance or additional codec support.

!!! warning "OpenCV dependency conflicts"
    The `opencv-python` package can cause dependency conflicts in environments with other packages that also depend on OpenCV (e.g., some ML frameworks install `opencv-python-headless`). If you encounter conflicts, install without the `opencv` extra:

    ```bash
    pip install "sleap-io[pyav,mat]"  # Skip opencv
    ```

    The `pyav` backend provides good performance without the conflict risk.

---

## Upgrading

### CLI tool (uv tool)

```bash
uv tool upgrade sleap-io
```

### Python package

=== "pip"

    ```bash
    pip install --upgrade "sleap-io[all]"
    ```

=== "uv pip"

    ```bash
    uv pip install -U "sleap-io[all]"
    ```

    The `-U` flag allows upgrades and implies `--refresh` to update cached package metadata.

=== "uv add"

    ```bash
    uv add -U "sleap-io[all]"
    ```

    The `-U` flag upgrades the package and refreshes the lock file.

=== "conda"

    ```bash
    conda update -c conda-forge sleap-io
    ```

### From source

```bash
# In cloned repository
git pull origin main
uv sync --all-extras
```

Or reinstall:

```bash
pip install --upgrade "sleap-io[all] @ git+https://github.com/talmolab/sleap-io.git@main"
```

---

## Uninstalling

### CLI tool (uv tool)

```bash
uv tool uninstall sleap-io
```

### Python package

=== "pip"

    ```bash
    pip uninstall sleap-io
    ```

=== "uv pip"

    ```bash
    uv pip uninstall sleap-io
    ```

=== "uv remove"

    ```bash
    uv remove sleap-io
    ```

=== "conda"

    ```bash
    conda remove sleap-io
    ```

---

## Troubleshooting

### Video backends not found

Check installed backends:

```bash
sio --version
# Or in Python:
python -c "import sleap_io; print(sleap_io.Video.backend_plugins)"
```

Install missing backends:

```bash
pip install opencv-python av
```

### Import errors

Ensure you're in the correct environment:

```bash
which python
python -c "import sleap_io; print(sleap_io.__version__)"
```

### Permission issues with uv tool

Use `--python` to specify a Python version:

```bash
uv tool install "sleap-io[all]" --python 3.12
```

### Conflicting dependencies

Create a fresh environment:

```bash
uv venv
uv pip install "sleap-io[all]"
```

---

## See Also

- [CLI Reference](cli.md): Complete command documentation
- [Examples](examples.md): Python usage patterns
- [Formats](formats/): Supported file formats
- [Changelog](changelog.md): Version history
