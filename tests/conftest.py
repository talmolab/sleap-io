"""Provide fixtures for the entire parent directory."""

import os

import pytest

from tests.fixtures.alphatracker import *  # noqa: F403
from tests.fixtures.camera import *  # noqa: F403
from tests.fixtures.coco import *  # noqa: F403
from tests.fixtures.cvat import *  # noqa: F403
from tests.fixtures.dlc import *  # noqa: F403
from tests.fixtures.jabs import *  # noqa: F403
from tests.fixtures.labels import *  # noqa: F403
from tests.fixtures.labelstudio import *  # noqa: F403
from tests.fixtures.leap import *  # noqa: F403
from tests.fixtures.slp import *  # noqa: F403
from tests.fixtures.ultralytics import *  # noqa: F403
from tests.fixtures.videos import *  # noqa: F403


@pytest.fixture(scope="session", autouse=True)
def force_eager_imports():
    """Force eager imports during testing to catch import issues.

    This fixture sets EAGER_IMPORT=1 for the entire test session, which forces
    the lazy_loader library to import all modules immediately rather than deferring
    them until first access. This ensures that:

    1. Tests don't mask missing imports that lazy loading might hide
    2. Import errors are caught early
    3. CI catches any lazy-loading issues

    The fixture is automatically applied to all tests (autouse=True) and runs once
    per test session (scope="session").
    """
    os.environ["EAGER_IMPORT"] = "1"
    yield
    os.environ.pop("EAGER_IMPORT", None)


@pytest.fixture(autouse=True)
def _no_v3_defaults_from_the_environment(monkeypatch):
    """Run every test with the defaults a release ships with.

    `SLEAP_IO_V3_DEFAULTS` in the environment would turn on the format 3.0
    defaults for the whole suite; tests that want them use `v3_defaults`.
    """
    monkeypatch.delenv("SLEAP_IO_V3_DEFAULTS", raising=False)


@pytest.fixture
def v3_defaults():
    """Turn on the format 3.0 defaults, for tests of the format 3.0 layout.

    Saves then write format 3.0 unless a test asks for another layout. The
    previous setting is restored afterwards.
    """
    from sleap_io.io import format_defaults

    previous = format_defaults._v3_defaults
    format_defaults.set_v3_defaults(True)
    yield
    format_defaults._v3_defaults = previous
