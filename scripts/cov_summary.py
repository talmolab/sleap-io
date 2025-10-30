#!/usr/bin/env python3
"""Summarize missed and partial coverage lines from JSON/XML.

Optionally restrict to changed files or PR diff lines.

Why:
    `coverage annotate` cannot encode "partial" lines; it only marks covered,
    missing, or excluded lines. Partial lines (executed lines with missing
    branches) are available in coverage JSON/XML and in the HTML view, which
    Codecov also uses. Switching to JSON/XML lets us match Codecov's partials.

Key features:
    - Parses coverage.json (preferred) or coverage.xml (fallback) to extract:
      * missed lines (hits == 0)
      * partial lines (branch line executed but not all exits taken)
    - Filters to changed files (vs. a base) or to PR-added/modified lines
      using `gh pr diff --patch`.
    - Output formats: text (default), json, md, gh-annotations.
    - Optional non-zero exit if any misses/partials exist.

Examples:
    # Summarize everything from coverage.xml
    uv run python scripts/cov_summary.py --source coverage.xml

    # Limit to files changed vs origin/main
    uv run python scripts/cov_summary.py --source coverage.xml --only-changed

    # Limit to PR diff lines (requires GitHub CLI)
    uv run python scripts/cov_summary.py --source coverage.xml --only-pr-diff-lines

    # Markdown table
    uv run python scripts/cov_summary.py --source coverage.xml --format md

    # Non-zero exit for CI if anything is missed/partial
    uv run python scripts/cov_summary.py --source coverage.xml --fail-on-miss
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ------------------------------- CLI -----------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: CLI args (usually sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(
        description="Summarize missed and partial lines from coverage JSON/XML."
    )
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Path to coverage.json or coverage.xml. If not provided, "
            "will try coverage.json then coverage.xml in CWD."
        ),
    )
    p.add_argument(
        "--only-changed",
        action="store_true",
        help="Restrict to files changed vs --diff-base (repo paths).",
    )
    p.add_argument(
        "--diff-base",
        default="origin/main",
        help='Base ref for git diff (default: "origin/main").',
    )
    p.add_argument(
        "--only-pr-diff-lines",
        action="store_true",
        help=(
            "Restrict to *added/modified* line numbers from a PR patch "
            "via `gh pr diff --patch`."
        ),
    )
    p.add_argument(
        "--pr",
        default=None,
        help=(
            "PR number or URL for --only-pr-diff-lines "
            "(default: current checked-out PR)."
        ),
    )
    p.add_argument(
        "--format",
        choices=("text", "json", "md", "gh-annotations"),
        default="text",
        help='Output format (default: "text").',
    )
    p.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Exit with code 2 if any uncovered (missed or partial) lines are found.",
    )
    return p.parse_args(argv)


# ----------------------------- Data types ------------------------------------


@dataclass
class FileCoverage:
    """Container for per-file coverage details."""

    missed: set[int] = field(default_factory=set)
    partial: set[int] = field(default_factory=set)
    # Optional: for JSON/XML -> explain which arcs are missing per line
    missing_branches: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True if there are no misses or partials."""
        return not self.missed and not self.partial


CoverageByFile = Dict[str, FileCoverage]


# ----------------------------- Utilities -------------------------------------


def collapse_ranges(nums: Sequence[int]) -> list[tuple[int, int]]:
    """Collapse sorted integers into inclusive (start, end) ranges.

    Args:
        nums: Sorted sequence of unique positive integers.

    Returns:
        List of (start, end) tuples.
    """
    if not nums:
        return []
    out: list[tuple[int, int]] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append((start, prev))
        start = prev = n
    out.append((start, prev))
    return out


def git_changed_py_files(diff_base: str) -> list[Path]:
    """Return changed Python files vs the merge-base with the given ref.

    Mirrors:
        git diff --name-only $(git merge-base origin/main HEAD)

    Args:
        diff_base: The base ref (e.g., "origin/main").

    Returns:
        List of Paths to changed *.py files (repo-relative).
    """
    try:
        mb = subprocess.check_output(
            ["git", "merge-base", diff_base, "HEAD"], text=True
        ).strip()
        out = subprocess.check_output(
            ["git", "diff", "--name-only", mb], text=True
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [Path(p) for p in out if p.endswith(".py")]


def gh_available() -> bool:
    """Return True if the `gh` CLI is available on PATH."""
    try:
        subprocess.check_output(["gh", "--version"], text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def collect_pr_new_lines(pr: Optional[str] = None) -> dict[str, list[int]]:
    """Collect new-side line numbers (added/modified) per file from a PR patch via `gh`.

    We parse a unified diff and record the **new** side line numbers for each '+'
    line within hunks. These correspond to added/modified lines.

    Args:
        pr: Optional PR number or URL. If None, uses the current checked-out PR.

    Returns:
        Mapping of 'b/<path>' -> sorted list of new-side line numbers.
    """
    if not gh_available():
        print(
            "WARN: GitHub CLI not found; cannot restrict to PR lines.",
            file=sys.stderr,
        )
        return {}
    cmd = ["gh", "pr", "diff", "--patch"]
    if pr:
        cmd.insert(3, str(pr))
    try:
        patch = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("WARN: Failed to read PR patch via `gh pr diff`.", file=sys.stderr)
        return {}

    per_file: dict[str, list[int]] = {}
    current_file: Optional[str] = None
    new_line: Optional[int] = None

    for raw in patch.splitlines():
        m = _DIFF_HEADER_RE.match(raw)
        if m:
            current_file = m.group(2).strip()  # 'b/<path>'
            new_line = None
            continue

        m = _HUNK_RE.match(raw)
        if m:
            new_line = int(m.group(1))
            continue

        if current_file is None or new_line is None:
            continue

        if raw.startswith("+++ ") or raw.startswith("--- "):
            continue

        if raw.startswith("+"):
            per_file.setdefault(current_file, []).append(new_line)
            new_line += 1
        elif raw.startswith("-"):
            # Removed lines advance only the old side.
            continue
        else:
            new_line += 1

    for f, lines in per_file.items():
        per_file[f] = sorted(set(lines))
    return per_file


def normalize_repo_relative(path: str) -> str:
    """Normalize a diff path to a repo-relative filesystem path string.

    Args:
        path: Path from diff (e.g., 'sleap_io/foo.py').

    Returns:
        Normalized repo-relative path string.
    """
    return str(Path(path))


# -------------------------- Coverage ingestion --------------------------------


def find_default_source() -> Optional[Path]:
    """Locate a default coverage artifact in the CWD.

    Returns:
        coverage.json if present, else coverage.xml if present, else None.
    """
    if Path("coverage.json").exists():
        return Path("coverage.json")
    if Path("coverage.xml").exists():
        return Path("coverage.xml")
    return None


def parse_condition_coverage(s: Optional[str]) -> tuple[int, int]:
    """Parse XML 'condition-coverage' like '50% (1/2)'.

    Args:
        s: Condition coverage attribute from XML line node.

    Returns:
        Tuple (covered, total). If missing/unparsable, returns (0, 0).
    """
    if not s:
        return (0, 0)
    m = re.search(r"\((\d+)/(\d+)\)", s)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def ingest_coverage_xml(xml_path: Path) -> CoverageByFile:
    """Ingest coverage XML and return per-file missed and partial lines.

    A line is:
        - missed: hits == 0.
        - partial: branch="true" AND hits > 0 AND condition-coverage < 100%.

    Args:
        xml_path: Path to coverage.xml.

    Returns:
        Mapping file -> FileCoverage.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    files: CoverageByFile = {}

    for cls in root.findall(".//class"):
        filename = cls.get("filename")
        if not filename:
            continue
        fc = files.setdefault(filename, FileCoverage())

        for ln in cls.findall(".//lines/line"):
            try:
                num = int(ln.get("number", "0"))
            except ValueError:
                continue
            hits = int(ln.get("hits", "0"))
            is_branch = ln.get("branch") == "true"
            cov_str = ln.get("condition-coverage")
            cov_covered, cov_total = parse_condition_coverage(cov_str)

            if hits == 0:
                fc.missed.add(num)

            # partial iff a branch line executed but not all exits were taken
            if is_branch and hits > 0 and cov_total > 0 and cov_covered < cov_total:
                fc.partial.add(num)
                mb = ln.get("missing-branches") or ""
                arcs: list[tuple[int, int]] = []
                for tok in re.findall(r"(\d+)->(\d+)", mb):
                    arcs.append((int(tok[0]), int(tok[1])))
                if arcs:
                    fc.missing_branches.setdefault(num, []).extend(arcs)

    return files


def ingest_coverage_json(json_path: Path) -> CoverageByFile:
    """Ingest coverage JSON and return per-file missed and partial lines.

    JSON fields (coverage.py ≥ 6.5):
        - missing_lines: list[int]
        - executed_lines: list[int]
        - missing_branches: list[[from_line, to_line]]

    We mark `from_line` of each missing branch as partial (if that line executed).

    Args:
        json_path: Path to coverage.json.

    Returns:
        Mapping file -> FileCoverage.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    files_data = data.get("files", {})

    files: CoverageByFile = {}

    for filename, fd in files_data.items():
        fc = files.setdefault(filename, FileCoverage())

        for n in fd.get("missing_lines", []):
            fc.missed.add(int(n))

        executed = {int(n) for n in fd.get("executed_lines", [])}

        for arc in fd.get("missing_branches", []):
            # arc is [from_line, to_line]
            try:
                src, dst = int(arc[0]), int(arc[1])
            except Exception:
                continue
            if src > 0 and (not executed or src in executed):
                fc.partial.add(src)
                fc.missing_branches.setdefault(src, []).append((src, dst))

    return files


def load_coverage(source: Optional[Path]) -> CoverageByFile:
    """Load coverage from JSON or XML, else fallback to annotate (misses only).

    Args:
        source: Coverage artifact path (JSON or XML). If None, autodetect.

    Returns:
        Mapping file -> FileCoverage.
    """
    src = source or find_default_source()
    if not src or not src.exists():
        return summarize_from_annotate(Path("."))  # no partials in this mode

    if src.suffix.lower() == ".json":
        return ingest_coverage_json(src)
    if src.suffix.lower() == ".xml":
        return ingest_coverage_xml(src)

    if Path("coverage.json").exists():
        return ingest_coverage_json(Path("coverage.json"))
    if Path("coverage.xml").exists():
        return ingest_coverage_xml(Path("coverage.xml"))

    return summarize_from_annotate(Path("."))


# -------------------------- Legacy annotate fallback --------------------------


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


def parse_cover_file(path: Path) -> list[int]:
    """Parse a single '*.py,cover' file and return missed line numbers.

    Coverage annotate marks:
        '!' -> missed
        '>' -> executed
        '-' -> excluded

    Args:
        path: Path to the coverage annotate file.

    Returns:
        List of 1-based line numbers that are missed.
    """
    missed: list[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh, start=1):
            if line.startswith("!"):
                missed.append(i)
    return missed


def summarize_from_annotate(root: Path) -> CoverageByFile:
    """Build coverage from annotate files (MISS only; no partials).

    Args:
        root: Root directory to search.

    Returns:
        Mapping file -> FileCoverage with only `missed` populated.
    """
    out: CoverageByFile = {}
    for p in sorted(iter_cover_files(root)):
        missed = parse_cover_file(p)
        if not missed:
            continue
        key = str(p).removesuffix(",cover")
        fc = out.setdefault(key, FileCoverage())
        fc.missed.update(missed)
    return out


# ------------------------------- Filtering ------------------------------------


def filter_to_changed(
    files: CoverageByFile, changed_py: Iterable[Path]
) -> CoverageByFile:
    """Filter coverage to only files present in `changed_py`.

    Args:
        files: Mapping of source file -> FileCoverage.
        changed_py: Iterable of changed .py Paths.

    Returns:
        Filtered mapping.
    """
    changed_set = {str(p) for p in changed_py}
    return {k: v for k, v in files.items() if k in changed_set}


def intersect_with_pr_lines(
    files: CoverageByFile, pr_lines: dict[str, list[int]]
) -> CoverageByFile:
    """Intersect misses/partials with PR-added/modified line numbers.

    Args:
        files: Mapping of file -> FileCoverage.
        pr_lines: Mapping of diff file -> list of new-side line numbers.

    Returns:
        Filtered mapping where each file's sets are clipped to PR lines.
    """
    if not pr_lines:
        return {}

    pr_map: dict[str, set[int]] = {
        normalize_repo_relative(f): set(lines) for f, lines in pr_lines.items()
    }

    out: CoverageByFile = {}
    for file, fc in files.items():
        if file not in pr_map:
            continue
        lines = pr_map[file]
        new_fc = FileCoverage()
        new_fc.missed = set(sorted(set(fc.missed).intersection(lines)))
        new_fc.partial = set(sorted(set(fc.partial).intersection(lines)))

        for ln, arcs in fc.missing_branches.items():
            if ln in new_fc.partial:
                new_fc.missing_branches[ln] = arcs

        if not new_fc.is_empty():
            out[file] = new_fc

    return out


# ------------------------------- Output ---------------------------------------


def _fmt_ranges(ranges: list[tuple[int, int]]) -> list[str]:
    return [f"{a}-{b}" if a != b else f"{a}" for a, b in ranges]


def _collapse(s: set[int]) -> list[tuple[int, int]]:
    return collapse_ranges(sorted(s))


def print_text(files: CoverageByFile) -> None:
    """Print human-readable summary grouped by file."""
    if not files:
        print("No uncovered lines found.")
        return

    for file in sorted(files.keys()):
        fc = files[file]
        parts: list[str] = []
        if fc.missed:
            parts.append("MISS=" + ",".join(_fmt_ranges(_collapse(fc.missed))))
        if fc.partial:
            parts.append("PARTIAL=" + ",".join(_fmt_ranges(_collapse(fc.partial))))
        print(f"{file}: " + ("; ".join(parts) if parts else "—"))


def print_md(files: CoverageByFile) -> None:
    """Print GitHub-flavored markdown table."""
    if not files:
        print("No uncovered lines found.")
        return

    print("| File | Missed | Partial |")
    print("| --- | --- | --- |")
    for file in sorted(files.keys()):
        fc = files[file]
        miss = ",".join(_fmt_ranges(_collapse(fc.missed))) or "—"
        part = ",".join(_fmt_ranges(_collapse(fc.partial))) or "—"
        print(f"| {file} | {miss} | {part} |")


def print_github_annotations(files: CoverageByFile) -> None:
    """Emit GitHub workflow commands to annotate lines in PR UI."""
    for file, fc in files.items():
        for ln in sorted(fc.missed):
            print(f"::warning file={file},line={ln}::Uncovered line")
        for ln in sorted(fc.partial):
            print(f"::notice file={file},line={ln}::Partially covered line")


def print_json(files: CoverageByFile) -> None:
    """Print machine-readable JSON with ranges and missing branch arcs."""
    obj = {}
    for file, fc in sorted(files.items()):
        obj[file] = {
            "missed": [{"start": a, "end": b} for a, b in _collapse(fc.missed)],
            "partial": [{"start": a, "end": b} for a, b in _collapse(fc.partial)],
            "missing_branches": {
                str(ln): [{"from": a, "to": b} for (a, b) in arcs]
                for ln, arcs in fc.missing_branches.items()
            },
        }
    print(json.dumps(obj, indent=2, sort_keys=True))


# --------------------------------- Main ---------------------------------------


def main(argv: Sequence[str]) -> int:
    """Entry point.

    Args:
        argv: CLI args (usually sys.argv[1:]).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    files = load_coverage(args.source)

    if args.only_changed:
        changed = git_changed_py_files(args.diff_base)
        files = filter_to_changed(files, changed)

    if args.only_pr_diff_lines:
        pr_lines = collect_pr_new_lines(args.pr)
        files = intersect_with_pr_lines(files, pr_lines)

    files = {k: v for k, v in files.items() if not v.is_empty()}

    if args.format == "json":
        print_json(files)
    elif args.format == "md":
        print_md(files)
    elif args.format == "gh-annotations":
        print_github_annotations(files)
    else:
        print_text(files)

    if args.fail_on_miss and files:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
