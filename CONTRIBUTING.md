# Contributing / Developer Guide

This repository follows a set of standard development practices for modern Python packages.

## Development workflow

### Setting up
1. Clone the repository:
    ```
    git clone https://github.com/talmolab/sleap-io && cd sleap-io
    ```
2. Install the package in development mode:
   With [`uv`](https://docs.astral.sh/uv/) (*recommended*):
   ```
   uv sync --all-extras
   ```
   Or with [`conda`](https://docs.conda.io/en/latest/miniconda.html):
   ```
   conda env create -f environment.yml && conda activate sleap-io
   ```
   Or with pip:
   ```
   pip install -e .[dev,all]
   ```
3. Test that things are working:
   ```
   uv run pytest tests
   ```
   Or if using conda/pip:
   ```
   pytest tests
   ```

To reinstall the environment from scratch:

With `uv`:
```
uv sync --all-extras --refresh
```

Or with `conda`:
```
conda env remove -n sleap-io && conda env create -f environment.yml
```

We also recommend setting up `ruff` to run automatically in your IDE.

If using `uv`, ruff is already included in the dev dependencies and can be run with:
```
uv run ruff check
uv run ruff format
```

Alternatively, you can install it globally with [`pipx`](https://pypa.github.io/pipx/):
```
pip install pipx
pipx ensurepath
pipx install ruff
```
This will make `ruff` available everywhere (such as VSCode), but will not be dependent on your conda `base` environment.


### Contributing a change
Once you're set up, follow these steps to make a change:

1. If you don't have push access to the repository, start by [making a fork](https://github.com/talmolab/sleap-io/fork) of the repository.
2. Switch to the [`main`](https://github.com/talmolab/sleap-io/tree/main) branch and `git pull` to fetch the latest changes.
3. Create a new branch named `<username>/<feature_name>` with a descriptive title for the change. For example: `talmo/nwb_support` or `talmo/update_dependencies`.
4. Push as many commits as you want. Descriptive commit messages and titles are optional but recommended.
5. Open a [Pull Request](https://github.com/talmolab/sleap-io/compare) of your new branch against `main` with a description of your changes. Feel free to create a "Draft" pull request to work on it incrementally.
6. Once the [tests](#tests-and-standards) pass, request a review from a core developer and make any changes requested.
7. Once approved, perform a [squash merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-pull-request-commits) against `main` to incorporate your changes.


## Tests and standards
This repository employs [continuous integration](https://en.wikipedia.org/wiki/Continuous_integration) via [GitHub Actions](https://docs.github.com/en/actions) to enforce code quality.

See the [`.github/workflows`](.github/workflows) folder for how our checks are implemented.

### Packaging and dependency management
This package uses [`setuptools`](https://setuptools.pypa.io/en/latest/) as a [packaging and distribution system](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/).

Our package configuration is defined in [`pyproject.toml`](pyproject.toml) which contains:
- Build system configuration
- Package metadata and dependencies
- Optional dependencies (extras)

If new dependencies need to be introduced (or if versions need to be fenced), specify these in [`pyproject.toml`](pyproject.toml) in the `dependencies` section. For development-only dependencies (i.e., packages that are not needed for distribution), add them to the `[project.optional-dependencies]` → `dev` section. For optional features like OpenCV support, add them to a dedicated extras group (e.g., `opencv`).

These dependencies will only be installed when specifying the extras like: `pip install -e .[dev,all]` or `pip install sleap-io[dev,all]`. With `uv`, use `uv sync --all-extras` to install all optional dependencies.

Available extras:
- `dev`: Development tools (pytest, ruff, etc.)
- `opencv`: OpenCV support for video processing
- `av`: PyAV support for video processing
- `all`: All optional dependencies (opencv + av)

Best practices for adding dependencies include:
- Use permissive [version ranges](https://peps.python.org/pep-0440/#version-specifiers) so that the package remains future- and backward-compatible without requiring new releases.
- Don't pin to a single specific versions of dependencies unless absolutely necessary, and consider using [platform-specific specifiers](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#platform-specific-dependencies).

For more reference see:
- [Configuring setuptools using `setup.cfg` files](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html)
- [Setuptools Keywords](https://setuptools.pypa.io/en/latest/references/keywords.html)
- [PEP 508 - Dependency specification for Python Software Packages](https://peps.python.org/pep-0508/)

**Note:** We recommend using [`uv`](https://docs.astral.sh/uv/) for fast, reliable Python environment management. For backwards compatibility, a minimal conda environment is defined in [`environment.yml`](environment.yml) that simply installs the package via pip.


### Testing
Testing is done via [`pytest`](https://docs.pytest.org/).

Tests should be created in the [`tests/`](tests) subfolder following the convention `test_{MODULE_NAME}.py` which mimics the main module organization.

It is highly recommended checking out other existing tests for reference on how these are structured.

All tests must pass before a PR can be merged.

Tests will be run on every commit across multiple operating systems and Python versions (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).


### Coverage
We check for coverage by parsing the outputs from `pytest` and uploading to [Codecov](https://app.codecov.io/gh/talmolab/sleap-io).

All changes should aim to increase or maintain test coverage.

### Live coverage

*The following steps are based on [this guide](https://jasonstitt.com/perfect-python-live-test-coverage).*

1. If you already have an environment installed, ensure you have the latest dev tools (namely `pytest-watch`):
   - With `uv`: `uv sync --all-extras`
   - With pip: `pip install -e ."[dev]"`
2. Install the [Coverage Gutters extension](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) in VS Code.
3. Open a terminal and run the test watcher:
   - With `uv`: `uv run ptw`
   - With conda: `conda activate sleap-io && ptw`
   This will generate a new `lcov.info` file when it's done.
4. Enable the coverage gutters by using **Ctrl/Cmd**+**Shift**+**P**, then **Coverage Gutters: Display Coverage**.


### Code style
To standardize formatting conventions and linting, we use [`ruff`](https://docs.astral.sh/ruff/).

It's highly recommended to set this up in your local environment so your code is auto-formatted and linted before pushing commits.

Adherence to the `ruff` code style and linting rules is automatically checked on push (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).


### Documentation conventions
We require that all non-test code follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) conventions. This is checked via `ruff`.

For example, a method might be documented as:

```py
def load_tracks(filepath: str) -> np.ndarray:
    """Load the tracks from a SLEAP Analysis HDF5 file.

    Args:
        filepath: Path to a SLEAP Analysis HDF5 file.
    
    Returns:
        The loaded tracks as a `np.ndarray` of shape `(n_tracks, n_frames, n_nodes, 2)`
        and dtype `float32`.

    Raises:
        ValueError: If the file does not contain a `/tracks` dataset.

    See also: save_tracks
    """
    with h5py.File(filepath, "r") as f:
        if "tracks" not in f:
            raise ValueError(
                "The file does not contain a /tracks dataset. "
                "This may not have been generated by SLEAP."
            )
        tracks = f["tracks"][:]
    return tracks.astype("float32")
```

**Notes:**
- The first line should fit within the 88 character limit, be on the same line as the initial `"""`, and should use imperative tense (e.g., "Load X..." not "Loads X...").
- Use backticks (`) when possible to enable auto-linking for documentation.
- Always document shapes and data types when describing inputs/outputs that are arrays.

Adherence to the docstring conventions is automatically checked on push (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).


## Releases
### Versioning
This package follows standard [semver](https://semver.org/) version practices, i.e.:
```
{MAJOR}.{MINOR}.{PATCH}
```

For alpha/pre-releases, append `a{NUM}` to the version.

Valid examples:
```
0.0.1
0.1.10a2
```

### Build
The PyPI-compatible package settings are in [`pyproject.toml`](pyproject.toml).

The version number is set in [`sleap_io/version.py`](sleap_io/version.py) in the `__version__` variable. This is read automatically by setuptools during installation and build.

To manually build (e.g., locally):
```
uv build
```
Or with Python's build module:
```
python -m build --wheel
```

To trigger an automated build (via the [`.github/workflows/build.yml`](.github/workflows/build.yml) action), [publish a Release](https://github.com/talmolab/sleap-io/releases/new).


## Documentation website

1. Install `sleap-io` with the `dev` dependencies:
   - With `uv`: `uv sync --all-extras`
   - With pip: `pip install -e ".[dev]"`
   - With conda: `conda env create -f environment.yml`
2. Build and tag a new version of the docs: `uv run mike deploy --update-aliases 0.1.4 latest`
3. Preview live changes locally with: `uv run mike serve`
4. Manually push a specific version with: `uv run mike deploy --push --update-aliases --allow-empty 0.1.4 latest`
