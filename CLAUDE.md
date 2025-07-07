# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sleap-io is a standalone utility library for working with animal pose tracking data. It provides:
- Reading/writing pose tracking data in various formats (SLEAP, NWB, Label Studio, JABS)
- Data structure manipulation and conversion
- Video I/O operations
- Minimal dependencies (no labeling, training, or inference functionality)

## Key Architecture

### Core Data Model (`sleap_io/model/`)
- **Skeleton**: Defines skeletal structure with nodes, edges, and symmetries
- **Instance**: Represents pose instances (predicted or manual) with associated tracks
- **LabeledFrame**: Container for frames with labeled instances
- **Labels**: Top-level container for entire datasets
- **Video**: Video container and metadata management
- **Camera**: Multi-camera support for 3D reconstruction

### I/O System (`sleap_io/io/`)
- **Main API**: High-level functions in `main.py` (`load_file`, `save_file`, etc.)
- **Format-specific modules**: `slp.py`, `nwb.py`, `labelstudio.py`, `jabs.py`
- **Video backends**: Abstracted reading/writing in `video_reading.py` and `video_writing.py`

## Development Commands

### Environment Setup
```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate sleap-io

# Pip development install
pip install -e .[dev]
```

### Linting
```bash
# Format check (MUST pass before committing)
black --check sleap_io tests

# Docstring style check
pydocstyle --convention=google sleap_io/
```

### Testing
```bash
# Run full test suite with coverage
pytest --cov=sleap_io --cov-report=xml tests/

# Run specific test module
pytest tests/io/test_slp.py -v

# Check line-by-line coverage for a module
conda activate sleap-io && pytest tests/model/test_labels.py -v --cov=sleap_io --cov-report=json && coverage annotate --include="*/sleap_io/model/labels.py"

# Watch mode for development
ptw
```

### Building
```bash
# Build wheel
python -m build --wheel

# Build documentation locally
mike serve
```

## Code Style Requirements

1. **Formatting**: Use `black` with max line length of 88
2. **Docstrings**: Google style, document "Attributes" section in class-level docstring
3. **Type hints**: Always include for function arguments and return types
4. **Import order**: Standard library, third-party, local (enforced by black)

## Testing Guidelines

1. Use existing fixtures from `tests/fixtures/` when possible
2. Create minimal synthetic data for new tests rather than files
3. Use `tmp_path` for any I/O operations in tests
4. Write multiple focused tests rather than one complex test
5. Place tests in corresponding module under `tests/` (e.g., `sleap_io/io/slp.py` → `tests/io/test_slp.py`)
6. Never create new test modules unless a new package module was created

## Common Development Tasks

### Adding a New I/O Format
1. Create module in `sleap_io/io/` with reader/writer functions
2. Add format detection to `sleap_io/io/main.py`
3. Create comprehensive tests in `tests/io/`
4. Update documentation with format specifications

### Working with Video Backends
- Video reading is abstracted through `VideoBackend` classes
- Supported backends: MediaVideo (imageio-ffmpeg), HDF5Video, ImageVideo
- Backend selection is automatic based on file format

### Running a Single Test
```bash
pytest tests/path/to/test_module.py::TestClass::test_method -v
```

## Important Notes

- Version is defined in `sleap_io/version.py`
- CI runs on Ubuntu, Windows, macOS with Python 3.8-3.13
- All PRs require passing tests and linting
- Documentation is hosted at https://io.sleap.ai/