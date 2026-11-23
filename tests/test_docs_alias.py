"""Tests for the script that picks the docs alias of a release."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "docs_alias.py"
_spec = importlib.util.spec_from_file_location("docs_alias", _SCRIPT)
docs_alias = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docs_alias)

PUBLISHED = [
    {"version": "dev", "title": "dev", "aliases": []},
    {"version": "v0.10.0", "title": "v0.10.0", "aliases": ["latest"]},
    {"version": "v0.10.1a0", "title": "v0.10.1a0", "aliases": ["prerelease"]},
    {"version": "v0.9.5", "title": "v0.9.5", "aliases": []},
]


@pytest.mark.parametrize(
    "tag, alias",
    [
        ("v0.10.1", "latest"),  # Newer than every stable version.
        ("v0.10.0", "latest"),  # Re-publishing the newest one.
        ("v0.9.6", ""),  # An older line's patch doesn't take over.
        ("v0.11.0rc1", ""),  # Pre-releases never get "latest".
        ("v0.10.0a0", ""),  # Alphas, as this project names its pre-releases.
        ("0.10.0a0", ""),
        ("v0.11.0a1", ""),
        ("not-a-version", ""),
    ],
)
def test_docs_alias(tag, alias):
    """A stable release gets "latest" only if no newer stable one is published."""
    assert docs_alias.docs_alias(tag, PUBLISHED) == alias


def test_docs_alias_with_no_published_versions():
    """The first release gets "latest"."""
    assert docs_alias.docs_alias("v0.1.0", []) == "latest"
