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
# UV (recommended - fast, reliable Python environment management)
uv sync --all-extras

# Conda (for backwards compatibility)
conda env create -f environment.yml
conda activate sleap-io

# Pip development install
pip install -e .[dev,all]
```

### Linting
```bash
# Auto-format and fix linting issues (from .claude/commands/lint.md)
uv run ruff format sleap_io tests && uv run ruff check --fix sleap_io tests

# Check formatting without making changes
uv run ruff format --check sleap_io tests
uv run ruff check sleap_io tests
```

### Testing
```bash
# Run full test suite with coverage
uv run pytest --cov=sleap_io --cov-report=xml tests/

# Run specific test module
uv run pytest tests/io/test_slp.py -v

# Quick coverage check with line-by-line annotations (from .claude/commands/coverage.md)
uv run pytest -q --maxfail=1 --cov --cov-branch && rm .coverage.* && uv run coverage annotate

# Check which files changed in PR for targeted coverage review
git diff --name-only $(git merge-base origin/main HEAD) | jq -R . | jq -s .

# Watch mode for development
uv run ptw
```

### Building
```bash
# Build wheel and source distribution
uv build

# Build documentation locally
uv run mike serve

# Deploy documentation version
uv run mike deploy --update-aliases 0.1.4 latest
```

## Code Style Requirements

1. **Formatting**: Use `ruff format` with max line length of 88
2. **Docstrings**: Google style, document "Attributes" section in class-level docstring
3. **Type hints**: Always include for function arguments and return types
4. **Import order**: Standard library, third-party, local (enforced by ruff)

## Testing Guidelines

1. Use existing fixtures from `tests/fixtures/` when possible
2. Create minimal synthetic data for new tests rather than files
3. Use `tmp_path` for any I/O operations in tests
4. Write multiple focused tests rather than one complex test
5. Place tests in corresponding module under `tests/` (e.g., `sleap_io/io/slp.py` → `tests/io/test_slp.py`)
6. Never create new test modules unless a new package module was created

## Using UV

UV is a fast, reliable Python package manager written in Rust. It replaces pip, pip-tools, pipx, poetry, pyenv, virtualenv, and more.

### Basic UV Commands
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies (including optional ones)
uv sync --all-extras

# Run any command in the virtual environment
uv run <command>

# Add a new dependency
uv add <package>

# Add a new dev dependency
uv add --dev <package>

# Update dependencies
uv sync --upgrade
```

## Claude Commands

The `.claude/commands` directory contains useful command shortcuts for Claude Code:

- **lint.md**: Auto-format and fix linting issues with ruff
- **coverage.md**: Run tests with coverage and generate line-by-line annotations  
- **pr-description.md**: Generate comprehensive PR descriptions using gh CLI

### PR Descriptions

When updating PR descriptions (from .claude/commands/pr-description.md):
1. Fetch current PR metadata and linked issues using `gh` CLI
2. Include: Summary, Key Changes, Example Usage, API Changes, Testing, and Design Decisions
3. Document reasoning behind implementation choices for future reference

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
uv run pytest tests/path/to/test_module.py::TestClass::test_method -v
```

## Important Notes

- Version is defined in `sleap_io/version.py`
- CI runs on Ubuntu, Windows, macOS with Python 3.8-3.13
- All PRs require passing tests and linting
- Documentation is hosted at https://io.sleap.ai/

## Testing Best Practices

- When adding tests, use global imports at the module-level rather than importing locally within a test function unless strictly needed (e.g., for import checking). Analyze current imports to find the best place to add the import statement and do not duplicate existing imports.

## Known Issues and Workarounds

- OpenCV and PyAV are now optional dependencies for video backends:
  - Install all: `uv sync --all-extras` or `pip install -e .[all]`
  - Install OpenCV only: `pip install -e .[opencv]`
  - Install PyAV only: `pip install -e .[av]`
- If you get an opencv/cv2 issue when running tests, try running the entire module, or entire test suite instead (this is an opencv issue with importing submodules of the tests).