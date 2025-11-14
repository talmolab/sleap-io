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
