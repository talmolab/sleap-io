"""
Summarize uncovered lines from coverage annotate output, optionally restricted to PR diff lines.

This script scans for files named "*.py,cover" (created by `coverage annotate`)
and prints a concise per-file summary of missed line ranges like:

    src/main.py: 80-102,190-193

It supports:
- Restricting to files changed in your branch: --only-changed (via `git merge-base`)
- Restricting to *added/modified* lines in a PR diff: --only-pr-diff-lines (via `gh pr diff --patch`)
- JSON output and CI-friendly failure codes.

Examples:
    # Summarize all coverage annotate files
    uv run python scripts/cov_summary.py

    # Only show files changed vs origin/main
    uv run python scripts/cov_summary.py --only-changed

    # Limit to lines added/changed in the current PR using the GitHub CLI
    uv run python scripts/cov_summary.py --only-pr-diff-lines

    # Limit to a specific PR number
    uv run python scripts/cov_summary.py --only-pr-diff-lines --pr 1234

    # Print JSON instead of text
    uv run python scripts/cov_summary.py --format json

    # Non-zero exit if any misses are found (useful in CI)
    uv run python scripts/cov_summary.py --fail-on-miss
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: CLI arguments (usually sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Summarize missed lines from coverage annotate.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help='Root directory to search (default: ".").',
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Restrict to files changed vs --diff-base (maps .py -> .py,cover).",
    )
    parser.add_argument(
        "--diff-base",
        default="origin/main",
        help='Base ref for git diff when using --only-changed (default: "origin/main").',
    )
    parser.add_argument(
        "--only-pr-diff-lines",
        action="store_true",
        help="Restrict to *added/modified* line numbers from the unified diff of a PR (via `gh pr diff --patch`).",
    )
    parser.add_argument(
        "--pr",
        default=None,
        help="PR number or URL for --only-pr-diff-lines (default: current checked-out PR).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help='Output format: "text" or "json" (default: "text").',
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Exit with code 2 if any uncovered lines are found.",
    )
    return parser.parse_args(argv)


def git_changed_py_files(diff_base: str) -> List[Path]:
    """Return a list of changed Python files vs the merge-base with the given ref.

    Mirrors:
        git diff --name-only $(git merge-base origin/main HEAD)

    Args:
        diff_base: The base ref (e.g., "origin/main").

    Returns:
        List of Paths to changed *.py files (relative paths).
    """
    try:
        mb = subprocess.check_output(["git", "merge-base", diff_base, "HEAD"], text=True).strip()
        out = subprocess.check_output(["git", "diff", "--name-only", mb], text=True).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    return [Path(p) for p in out if p.endswith(".py")]


def iter_cover_files(root: Path) -> Iterable[Path]:
    """Yield all '*.py,cover' files under root.

    Args:
        root: Root directory to search.

    Yields:
        Paths to coverage annotation files.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py,cover"):
                yield Path(dirpath) / fn


def collapse_ranges(nums: Sequence[int]) -> List[Tuple[int, int]]:
    """Collapse sorted integers into inclusive (start, end) ranges.

    Args:
        nums: Sorted sequence of unique positive integers.

    Returns:
        List of (start, end) tuples.
    """
    if not nums:
        return []

    ranges: List[Tuple[int, int]] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append((start, prev))
        start = prev = n
    ranges.append((start, prev))
    return ranges


def parse_cover_file(path: Path) -> List[int]:
    """Parse a single '*.py,cover' file and return missed line numbers.

    Coverage annotate marks each source line with a leading symbol:
      - '!' for missed
      - '>' (and others) for executed/partial/etc.

    Args:
        path: Path to the coverage annotate file.

    Returns:
        List of 1-based line numbers that are missed.
    """
    missed: List[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh, start=1):
            if line.startswith("!"):
                missed.append(i)
    return missed


def summarize_all(cover_paths: Iterable[Path]) -> Dict[str, List[Tuple[int, int]]]:
    """Build a mapping of module file -> missed line ranges.

    Args:
        cover_paths: Iterable of '*.py,cover' paths.

    Returns:
        Dict mapping source file path (without the trailing ",cover")
        to a list of (start, end) missed ranges.
    """
    summary: Dict[str, List[Tuple[int, int]]] = {}
    for p in sorted(cover_paths):
        missed = parse_cover_file(p)
        if not missed:
            continue
        missed.sort()
        ranges = collapse_ranges(missed)
        key = str(p).removesuffix(",cover")
        summary[key] = ranges
    return summary


def filter_to_changed(
    summary: Dict[str, List[Tuple[int, int]]], changed_py: Iterable[Path]
) -> Dict[str, List[Tuple[int, int]]]:
    """Filter the summary to only files present in changed_py.

    Args:
        summary: Mapping of source file -> ranges.
        changed_py: Iterable of changed .py Paths.

    Returns:
        Filtered mapping.
    """
    changed_set = {str(p) for p in changed_py}
    return {k: v for k, v in summary.items() if k in changed_set}


def gh_available() -> bool:
    """Return True if the `gh` CLI is available on PATH."""
    try:
        subprocess.check_output(["gh", "--version"], text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def collect_pr_new_lines(pr: Optional[str] = None) -> Dict[str, List[int]]:
    """Collect new-file line numbers (added/modified) per file from a PR diff via `gh`.

    We parse a unified diff (patch) and record the **new** side line numbers
    for each '+' line within hunks. These correspond to added/modified lines.

    Args:
        pr: Optional PR number or URL. If None, uses the current checked-out PR.

    Returns:
        Mapping of path (as it appears on the 'b/' side) -> sorted list of new-side line numbers.
    """
    if not gh_available():
        return {}

    cmd = ["gh", "pr", "diff", "--patch"]
    if pr:
        cmd.insert(3, str(pr))

    try:
        patch = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return {}

    per_file: Dict[str, List[int]] = {}
    current_file: Optional[str] = None
    new_line: Optional[int] = None

    for raw in patch.splitlines():
        # File header
        m = _DIFF_HEADER_RE.match(raw)
        if m:
            # Use the 'b/' path (new file path)
            current_file = m.group(2)
            # Normalize paths to local filesystem style
            current_file = current_file.strip()
            continue

        # Hunk header
        m = _HUNK_RE.match(raw)
        if m:
            start = int(m.group(1))
            new_line = start
            continue

        if current_file is None or new_line is None:
            continue

        # Skip file header metadata lines inside diff
        if raw.startswith("+++ ") or raw.startswith("--- "):
            continue

        if raw.startswith("+"):
            # Added/modified line (on the new side)
            per_file.setdefault(current_file, []).append(new_line)
            new_line += 1
        elif raw.startswith("-"):
            # Removed line: advances old side only; new side line doesn't advance
            # so do not change new_line
            continue
        else:
            # Context line: advances both sides
            new_line += 1

    # Sort and unique
    for f, lines in per_file.items():
        uniq_sorted = sorted(set(lines))
        per_file[f] = uniq_sorted

    return per_file


def normalize_repo_relative(path: str) -> str:
    """Normalize a diff path to a repo-relative filesystem path string.

    Args:
        path: Path from diff (usually no leading slash, e.g., 'src/foo.py').

    Returns:
        Normalized repo-relative path string.
    """
    return str(Path(path))


def intersect_summary_with_pr_lines(
    summary: Dict[str, List[Tuple[int, int]]], pr_lines: Dict[str, List[int]]
) -> Dict[str, List[Tuple[int, int]]]:
    """Intersect coverage misses with PR-added/modified line numbers.

    Args:
        summary: Mapping of file -> missed (start, end) ranges.
        pr_lines: Mapping of file -> list of new-side line numbers from PR diff.

    Returns:
        Filtered mapping where each file's ranges are clipped to the PR lines.
    """
    if not pr_lines:
        return {}

    # Expand ranges to sets, intersect, then re-collapse
    out: Dict[str, List[Tuple[int, int]]] = {}

    # Build a fast lookup for PR lines per normalized repo path
    pr_map: Dict[str, set[int]] = {
        normalize_repo_relative(f): set(lines) for f, lines in pr_lines.items()
    }

    for file, ranges in summary.items():
        # The summary keys are repo-relative (from the .py,cover locations)
        if file not in pr_map:
            continue
        missed_set = set()
        for a, b in ranges:
            missed_set.update(range(a, b + 1))
        intersected = sorted(missed_set.intersection(pr_map[file]))
        if not intersected:
            continue
        clipped = collapse_ranges(intersected)
        out[file] = clipped

    return out


def print_text(summary: Dict[str, List[Tuple[int, int]]]) -> None:
    """Print the summary in human-readable text format.

    Args:
        summary: Mapping of source file -> list of (start, end) ranges.
    """
    if not summary:
        print("✅ No uncovered lines found.")
        return

    for file in sorted(summary.keys()):
        ranges = summary[file]
        parts = [f"{a}-{b}" if a != b else f"{a}" for a, b in ranges]
        print(f"{file}: {','.join(parts)}")


def print_json(summary: Dict[str, List[Tuple[int, int]]]) -> None:
    """Print the summary as JSON.

    Args:
        summary: Mapping of source file -> list of (start, end) ranges.
    """
    obj = {file: [{"start": a, "end": b} for a, b in ranges] for file, ranges in sorted(summary.items())}
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: Sequence[str]) -> int:
    """Entry point.

    Args:
        argv: CLI args (usually sys.argv[1:]).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    cover_paths = list(iter_cover_files(args.root))
    summary = summarize_all(cover_paths)

    if args.only_changed:
        changed = git_changed_py_files(args.diff_base)
        summary = filter_to_changed(summary, changed)

    if args.only_pr_diff_lines:
        pr_lines = collect_pr_new_lines(args.pr)
        # Note: The diff uses "b/<path>" for the new file; our keys are repo-relative paths.
        summary = intersect_summary_with_pr_lines(summary, pr_lines)

    if args.format == "json":
        print_json(summary)
    else:
        print_text(summary)

    if args.fail_on_miss and summary:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
