# Contributing / Developer Guide

This repository follows a set of standard development practices for modern Python packages.

## Development workflow

### Setting up
1. Clone the repository:
    ```
    git clone https://github.com/talmolab/sleap-io && cd sleap-io
    ```
2. Install the package in development mode:
   With [`conda`](https://docs.conda.io/en/latest/miniconda.html) (*recommended*):
   ```
   conda env create -f environment.yml && conda activate sleap-io
   ```
   Or without conda:
   ```
   pip install -e .[dev]
   ```
3. Test that things are working:
   ```
   pytest tests
   ```

To reinstall the environment from scratch using `conda`:
```
conda env remove -n sleap-io && conda env create -f environment.yml
```

We also recommend setting up `black` to run automatically in your IDE.
A good way to do this without messing up your global environment is to use a tool like [`pipx`](https://pypa.github.io/pipx/):
```
pip install pipx
pipx ensurepath
pipx install black
```
This will make `black` available everywhere (such as VSCode), but will not be dependent on your conda `base` environment.


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

Our [build system](https://setuptools.pypa.io/en/latest/build_meta.html) is configured in two places:
- [`pyproject.toml`](pyproject.toml) which defines the basic build system configuration. This probably won't need to be changed.
- [`setup.cfg`](setup.cfg) which contains the [declarative configuration](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#declarative-config) of the package and its dependencies.

If new dependencies need to be introduced (or if versions need to be fenced), specify these in [`setup.cfg`](setup.cfg) in the `install_requires` section. For development-only dependencies (i.e., packages that are not needed for distribution), add them to the `[options.extras_require]` → `dev` section. These dependencies will only be installed when specifying the `dev` extras like: `pip install -e .[dev]` or `pip install sleap-io[dev]`.

Best practices for adding dependencies include:
- Use permissive [version ranges](https://peps.python.org/pep-0440/#version-specifiers) so that the package remains future- and backward-compatible without requiring new releases.
- Don't pin to a single specific versions of dependencies unless absolutely necessary, and consider using [platform-specific specifiers](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#platform-specific-dependencies).

For more reference see:
- [Configuring setuptools using `setup.cfg` files](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html)
- [Setuptools Keywords](https://setuptools.pypa.io/en/latest/references/keywords.html)
- [PEP 508 - Dependency specification for Python Software Packages](https://peps.python.org/pep-0508/)

**Note:** We use [`conda`](https://docs.conda.io/en/latest/miniconda.html) as a preferred method for defining and managing environments, but this is not required. A recommended development environment is defined in [`environment.yml`](environment.yml).


### Testing
Testing is done via [`pytest`](https://docs.pytest.org/).

Tests should be created in the [`tests/`](tests) subfolder following the convention `test_{MODULE_NAME}.py` which mimicks the main module organization.

It is highly recommended checking out other existing tests for reference on how these are structured.

All tests must pass before a PR can be merged.

Tests will be run on every commit across multiple operating systems and Python versions (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).


### Coverage
We check for coverage by parsing the outputs from `pytest` and uploading to [Codecov](https://app.codecov.io/gh/talmolab/sleap-io).

All changes should aim to increase or maintain test coverage.


### Code style
To standardize formatting conventions, we use [`black`](https://black.readthedocs.io/en/stable/).

It's highly recommended to set this up in your local environment so your code is auto-formatted before pushing commits.

Adherence to the `black` code style is automatically checked on push (see [`.github/workflows/lint.yml`](.github/workflows/lint.yml)).


### Documentation conventions
We require that all non-test code follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) conventions. This is checked via [`pydocstyle`](http://www.pydocstyle.org/).

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

Adherence to the docstring conventions is automatically checked on push (see [`.github/workflows/lint.yml`](.github/workflows/lint.yml)).


### Static type checking
All types must be statically defined and will be checked using [`mypy`](https://mypy.readthedocs.io/).

See [PEP 484](https://peps.python.org/pep-0484/) for reference.

Adherence to typing requirements is automatically checked on push (see [`.github/workflows/lint.yml`](.github/workflows/lint.yml)).


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
The PyPI-compatible package settings are in [setup.cfg].

The version number is set in [sleap_io/__init__.py] in the `__version__` variable. This is read automatically by setuptools during installation and build.

To manually build (e.g., locally):
```
python -m build --wheel
```

To trigger an automated build (via the [.github/workflows/build.yml] action), [publish a Release](https://github.com/talmolab/sleap-io/releases/new).


## Documentation website
**TODO:**
- [ ] Setup sphinx/autodoc.
- [ ] Describe documentation best practices.
