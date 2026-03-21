"""MkDocs hook to execute ``pycon`` code blocks at build time.

Replaces the default ``markdown-exec`` pycon handler with one that interleaves
source lines and execution output, producing a rendered code block that looks
exactly like a real Python REPL session.

Requires the ``markdown-exec`` plugin to be enabled in ``mkdocs.yml`` (it
registers the ``pycon`` custom fence that this hook then patches).
"""

import logging
import sys
import traceback
from functools import partial
from io import StringIO
from types import ModuleType

log = logging.getLogger("mkdocs.hooks.markdown_exec_pycon")


def _run_pycon_interleaved(code: str) -> str:
    """Execute pycon code and interleave source lines with their output.

    Produces a single string with ``>>>``/``...`` prompts and output lines
    interleaved, exactly like a real Python REPL session.
    """
    lines = code.split("\n")
    result_lines: list[str] = []
    current_block: list[str] = []

    exec_globals: dict = {}
    exec_globals["__name__"] = "__pycon_exec__"
    sys.modules.setdefault("__pycon_exec__", ModuleType("__pycon_exec__"))

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith(">>> ") or (line.startswith(">>>") and len(line) == 3):
            # Start of a new statement
            result_lines.append(line)
            python_line = line[4:] if len(line) > 4 else ""
            current_block = [python_line]

            # Collect continuation lines
            j = i + 1
            while j < len(lines) and lines[j].startswith("... "):
                result_lines.append(lines[j])
                current_block.append(lines[j][4:])
                j += 1

            i = j

            # Skip any existing hardcoded output lines (non-prompt, non-empty)
            while (
                i < len(lines)
                and not lines[i].startswith(">>>")
                and lines[i] != ""
            ):
                i += 1

            # Execute the accumulated block
            block_code = "\n".join(current_block)
            if not block_code.strip():
                continue

            buffer = StringIO()
            exec_globals["print"] = partial(print, file=buffer)

            try:
                compiled = compile(block_code, "<pycon>", "exec")
                exec(compiled, exec_globals)  # noqa: S102
            except Exception:
                tb = traceback.format_exc()
                for tb_line in tb.strip().split("\n"):
                    result_lines.append(tb_line)

            output = buffer.getvalue()
            if output:
                for out_line in output.rstrip("\n").split("\n"):
                    result_lines.append(out_line)
        else:
            i += 1

    return "\n".join(result_lines)


def _highlight_pycon(code: str) -> str:
    """Syntax-highlight pycon code using Pygments and return HTML."""
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonConsoleLexer

    formatter = HtmlFormatter(nowrap=False, cssclass="highlight")
    return highlight(code, PythonConsoleLexer(), formatter)


def _custom_pycon_formatter(
    source, language, class_name, options, md, classes, id_value, attrs, **kwargs
):
    """Custom superfences formatter for pycon blocks.

    Executes the code and renders interleaved source + output with syntax
    highlighting, matching the ``pymdownx.superfences`` format signature.
    """
    try:
        interleaved = _run_pycon_interleaved(source)
    except Exception as error:
        log.warning("Error executing pycon block: %s", error)
        return f"<p><strong>Error executing pycon block:</strong> {error}</p>"

    if not interleaved.strip():
        return ""

    return _highlight_pycon(interleaved)


def _custom_pycon_validator(language, inputs, options, attrs, md):
    """Validator for pycon blocks: always accept for execution."""
    # Pop exec from inputs to avoid it being passed as an extra attribute.
    inputs.pop("exec", None)
    return True


def on_config(config):
    """Replace the markdown-exec pycon fence with our custom interleaved renderer.

    This hook runs AFTER the ``markdown-exec`` plugin's ``on_config``, which has
    already registered its custom fences. We find the ``pycon`` fence entry and
    replace its validator and formatter with our custom versions.
    """
    mdx_configs = config.get("mdx_configs", {})
    superfences = mdx_configs.get("pymdownx.superfences", {})
    custom_fences = superfences.get("custom_fences", [])

    for fence in custom_fences:
        if fence.get("name") == "pycon":
            fence["validator"] = _custom_pycon_validator
            fence["format"] = _custom_pycon_formatter
            log.debug("Patched pycon custom fence for interleaved REPL output")
            return

    log.warning("pycon custom fence not found — is markdown-exec enabled?")
