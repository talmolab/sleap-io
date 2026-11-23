#!/usr/bin/env python
"""Print the docs alias a stable release gets: "latest", or nothing.

A stable release takes the "latest" alias only when no newer stable version is
already published, so a patch release of an older line (for example 0.9.6 after
0.10.0) adds its docs without taking over the default page.

Usage:
    python scripts/docs_alias.py TAG VERSIONS_JSON

`VERSIONS_JSON` is mike's `versions.json` from the docs branch. Prints `latest` or
an empty line.
"""

import json
import sys

from packaging.version import InvalidVersion, Version


def _parse(name: str) -> Version | None:
    """Return the version a docs entry or tag names, or `None` if it names none."""
    try:
        return Version(name[1:] if name.startswith("v") else name)
    except InvalidVersion:
        return None


def docs_alias(tag: str, versions: list[dict]) -> str:
    """Return "latest" if `tag` is at least the newest published stable version.

    Args:
        tag: The release tag, such as `v0.10.0`.
        versions: The entries of mike's `versions.json`, each with a `version` and
            a `title`.

    Returns:
        `"latest"`, or `""` when a newer stable version is already published, or
        the tag is a pre-release or not a version.
    """
    this = _parse(tag)
    if this is None or this.is_prerelease:
        return ""
    for entry in versions:
        for name in (entry.get("version", ""), entry.get("title", "")):
            other = _parse(str(name))
            if other is not None and not other.is_prerelease and other > this:
                return ""
    return "latest"


if __name__ == "__main__":
    with open(sys.argv[2]) as f:
        try:
            versions = json.load(f)
        except json.JSONDecodeError:
            versions = []
    print(docs_alias(sys.argv[1], versions))
