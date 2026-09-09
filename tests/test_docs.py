"""Guard tests for documentation code blocks.

The docs build executes every ```pycon``` block via ``markdown-exec`` and the
``hooks/markdown_exec_pycon.py`` hook, rendering the interleaved output -- including
any *traceback* -- directly into the published HTML. Each block runs with a fresh
set of globals, so a block that references a name defined in a previous block
silently ships a ``NameError`` traceback to readers (see the regression where two
``regions.md`` blocks rendered ``NameError`` into the docs).

Plain ```python``` blocks are *not* executed -- they are illustrative recipes that
read data the reader supplies (``cellpose_masks.npy``, ``microscopy.tif``, ...) --
so nothing catches API drift in them. They are checked statically instead: every
keyword argument passed to a ``sio.*`` call must exist in the real signature. That
guards against the class of bug where the docs kept advertising constructor kwargs
that had been removed (see #559, where ``sio.Labels(label_images=...)`` raised
``TypeError`` on copy-paste).

This module fails CI on a pull request, catching both regressions before they
reach the (push/release-only) docs build.
"""

import ast
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

import sleap_io

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
HOOK_PATH = REPO_ROOT / "hooks" / "markdown_exec_pycon.py"
_PYCON_BLOCK = re.compile(r"```pycon\n(.*?)```", re.DOTALL)
_TRACEBACK_MARKER = "Traceback (most recent call last)"


def _load_pycon_hook():
    """Import ``hooks/markdown_exec_pycon.py`` (not an installed package)."""
    spec = importlib.util.spec_from_file_location("markdown_exec_pycon", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_pycon_blocks() -> list[tuple[Path, int, str]]:
    """Return ``(md_file, block_index, code)`` for every docs ``pycon`` block."""
    blocks = []
    for md_file in sorted(DOCS_DIR.glob("**/*.md")):
        source = md_file.read_text(encoding="utf-8")
        for index, match in enumerate(_PYCON_BLOCK.finditer(source)):
            blocks.append((md_file, index, match.group(1)))
    return blocks


_BLOCKS = _collect_pycon_blocks()
_BLOCK_IDS = [
    f"{md.relative_to(REPO_ROOT).as_posix()}#block{idx}" for md, idx, _ in _BLOCKS
]


def test_docs_have_pycon_blocks():
    """Sanity check that block discovery is wired up (guards against a no-op suite)."""
    assert _BLOCKS, "No docs pycon blocks found -- discovery is broken."


@pytest.mark.parametrize("md_file, block_index, code", _BLOCKS, ids=_BLOCK_IDS)
def test_pycon_block_renders_without_traceback(md_file, block_index, code, monkeypatch):
    """Each docs ``pycon`` block must execute without rendering a traceback.

    Blocks load fixtures via paths relative to the repo root (e.g.
    ``tests/data/...``), so execution is pinned to the repo root, matching the
    docs-build working directory.
    """
    monkeypatch.chdir(REPO_ROOT)
    hook = _load_pycon_hook()
    rendered = hook._run_pycon_interleaved(code)
    assert _TRACEBACK_MARKER not in rendered, (
        f"{md_file.relative_to(REPO_ROOT).as_posix()} pycon block #{block_index} "
        f"renders a traceback into the published docs. Make the block self-contained "
        f"(each block runs with fresh globals).\n\nRendered output:\n{rendered}"
    )


# ``python`` blocks may sit inside an indented admonition, so capture the fence's
# own indent and strip it back off the body before parsing.
_PYTHON_BLOCK = re.compile(
    r"^([ \t]*)```+[ \t]*(?:python|py)\b[^\n]*\n(.*?)^[ \t]*```+[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

# Factories that forward ``**kwargs`` to ``cls(...)``, so an unrecognized kwarg
# lands on the annotation constructor rather than being silently accepted.
_KWARG_FORWARDING_FACTORIES = ("from_stack", "from_numpy", "from_binary_masks")

# Sentinel for a callable that takes ``**kwargs`` (any keyword is plausible).
_ANY_KWARGS = object()


def _dedent_block(indent: str, body: str) -> str:
    """Strip the fence's indentation off each line of a code block body."""
    if not indent:
        return body
    return "\n".join(
        line[len(indent) :] if line.startswith(indent) else line
        for line in body.split("\n")
    )


def _collect_python_blocks() -> list[tuple[Path, int, str]]:
    """Return ``(md_file, block_index, code)`` for every docs ```python``` block."""
    blocks = []
    for md_file in sorted(DOCS_DIR.glob("**/*.md")):
        source = md_file.read_text(encoding="utf-8")
        for index, match in enumerate(_PYTHON_BLOCK.finditer(source)):
            blocks.append((md_file, index, _dedent_block(*match.groups())))
    return blocks


_PYTHON_BLOCKS = _collect_python_blocks()
_PYTHON_BLOCK_IDS = [
    f"{md.relative_to(REPO_ROOT).as_posix()}#pyblock{idx}"
    for md, idx, _ in _PYTHON_BLOCKS
]


def _accepted_kwargs(obj):
    """Keyword names ``obj`` accepts, ``_ANY_KWARGS``, or ``None`` if unknown."""
    try:
        params = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return None
    names = set()
    for param in params:
        if param.kind is param.VAR_KEYWORD:
            return _ANY_KWARGS
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            names.add(param.name)
    return names


def _resolve_sio_call(node: ast.Call):
    """Resolve a ``sio.X(...)`` / ``sio.X.y(...)`` call to ``(path, object)``.

    Returns ``(None, None)`` for calls not rooted at the ``sio`` / ``sleap_io``
    module, or naming an attribute that does not exist on it.
    """
    path = []
    target = node.func
    while isinstance(target, ast.Attribute):
        path.append(target.attr)
        target = target.value
    if not isinstance(target, ast.Name) or target.id not in ("sio", "sleap_io"):
        return None, None
    path.reverse()
    obj = sleap_io
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None, None
    return path, obj


def _invalid_kwargs(code: str) -> list[str]:
    """Keyword arguments passed to ``sio.*`` calls that the real API rejects."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Pseudocode or a non-Python snippet mislabeled as ``python``.
        return []

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.keywords:
            continue
        path, obj = _resolve_sio_call(node)
        if obj is None:
            continue

        accepted = _accepted_kwargs(obj)
        if (
            accepted is _ANY_KWARGS
            and len(path) == 2
            and path[1] in _KWARG_FORWARDING_FACTORIES
        ):
            owner_kwargs = _accepted_kwargs(getattr(sleap_io, path[0], None))
            if isinstance(owner_kwargs, set):
                accepted = owner_kwargs | set(inspect.signature(obj).parameters)
        if accepted is None or accepted is _ANY_KWARGS:
            continue

        dotted = ".".join(path)
        problems.extend(
            f"line {node.lineno}: sio.{dotted}(..., {kw.arg}=...)"
            for kw in node.keywords
            if kw.arg is not None and kw.arg not in accepted
        )
    return problems


def test_docs_have_python_blocks():
    """Sanity check that block discovery is wired up (guards against a no-op suite)."""
    assert _PYTHON_BLOCKS, "No docs python blocks found -- discovery is broken."


@pytest.mark.parametrize(
    "md_file, block_index, code", _PYTHON_BLOCKS, ids=_PYTHON_BLOCK_IDS
)
def test_python_block_uses_real_kwargs(md_file, block_index, code):
    """Each docs ```python``` block must pass only keyword arguments that exist.

    These blocks are never executed by the docs build, so a removed or renamed
    parameter ships a copy-paste ``TypeError`` to readers instead of failing CI.
    """
    problems = _invalid_kwargs(code)
    assert not problems, (
        f"{md_file.relative_to(REPO_ROOT).as_posix()} python block #{block_index} "
        f"passes keyword arguments that sleap-io does not accept:\n  "
        + "\n  ".join(problems)
    )
