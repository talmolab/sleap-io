"""Guard tests for executable documentation code blocks.

The docs build executes every ```pycon``` block via ``markdown-exec`` and the
``hooks/markdown_exec_pycon.py`` hook, rendering the interleaved output -- including
any *traceback* -- directly into the published HTML. Each block runs with a fresh
set of globals, so a block that references a name defined in a previous block
silently ships a ``NameError`` traceback to readers (see the regression where two
``regions.md`` blocks rendered ``NameError`` into the docs).

This module fails CI on a pull request if any docs ``pycon`` block raises, catching
the regression before it reaches the (push/release-only) docs build.
"""

import importlib.util
import re
from pathlib import Path

import pytest

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
