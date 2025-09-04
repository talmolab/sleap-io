#!/usr/bin/env python
"""Reorder versions.json for proper version dropdown sorting in mike docs."""

import json
import sys
from packaging import version

def reorder_versions(filepath="versions.json"):
    """Reorder versions.json to fix version dropdown ordering.
    
    Orders versions as: dev -> newest -> oldest
    
    Args:
        filepath: Path to versions.json file.
    """
    # Read current versions.json
    with open(filepath, 'r') as f:
        versions = json.load(f)
    
    # Separate dev version from others
    dev_versions = [v for v in versions if v['version'] == 'dev']
    regular_versions = [v for v in versions if v['version'] != 'dev']
    
    # Sort regular versions by semantic version of title (removing 'v' prefix)
    # reverse=True for newest first
    def get_version_key(v):
        title = v['title']
        # Remove 'v' prefix if present
        if title.startswith('v'):
            title = title[1:]
        try:
            return version.parse(title)
        except:
            return version.parse('0.0.0')
    
    regular_versions.sort(key=get_version_key, reverse=True)
    
    # Combine: dev first, then sorted regular versions (newest to oldest)
    sorted_versions = dev_versions + regular_versions
    
    # Write back
    with open(filepath, 'w') as f:
        json.dump(sorted_versions, f, indent=2)
    
    print(f"Reordered {len(sorted_versions)} versions in {filepath}")
    if sorted_versions:
        print(f"Order: {' -> '.join([v['title'] for v in sorted_versions[:5]])}{'...' if len(sorted_versions) > 5 else ''}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "versions.json"
    reorder_versions(filepath)