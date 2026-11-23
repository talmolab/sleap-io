"""The process-wide switch for the SLP format 3.0 defaults.

sleap-io reads both SLP layouts and, by default, writes the older 2.x layout that
sleap-io < 0.10 and SLEAP can open, unless a file's content needs format 3.0. It
also keeps the readings older releases used wherever a newer reading would create
content that only format 3.0 can store.

The switch turns on the format 3.0 defaults for a whole process: saves write format
3.0, and the newer readings apply. Explicit arguments always win over it.

It is off unless it is turned on with `set_v3_defaults(True)` or by setting the
environment variable `SLEAP_IO_V3_DEFAULTS` to `1`, `true`, `yes` or `on`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

#: The environment variable that turns on the format 3.0 defaults.
V3_DEFAULTS_ENV = "SLEAP_IO_V3_DEFAULTS"

_TRUE_VALUES = ("1", "true", "yes", "on")

# Set by `set_v3_defaults`; `None` defers to the environment variable.
_v3_defaults: bool | None = None


def set_v3_defaults(enabled: bool | None) -> None:
    """Turn the format 3.0 defaults on or off for this process.

    Args:
        enabled: `True` to turn them on, `False` to turn them off, or `None` to
            defer to the `SLEAP_IO_V3_DEFAULTS` environment variable again.

    See also: `get_v3_defaults`, `v3_defaults`.
    """
    global _v3_defaults
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError(f"enabled must be True, False or None, got {enabled!r}.")
    _v3_defaults = enabled


def get_v3_defaults() -> bool:
    """Return whether the format 3.0 defaults are on for this process.

    Returns:
        The value set with `set_v3_defaults`, or, if none is set, whether the
        `SLEAP_IO_V3_DEFAULTS` environment variable is `1`, `true`, `yes` or `on`
        (in any case).
    """
    if _v3_defaults is not None:
        return _v3_defaults
    return os.environ.get(V3_DEFAULTS_ENV, "").strip().lower() in _TRUE_VALUES


@contextmanager
def v3_defaults(enabled: bool = True) -> Iterator[None]:
    """Turn the format 3.0 defaults on (or off) within a `with` block.

    The setting is process-wide, not per thread: other threads see it while the
    block runs.

    Args:
        enabled: Whether the defaults are on inside the block.

    Examples:
        >>> import sleap_io as sio
        >>> with sio.v3_defaults():
        ...     pass  # saves here write SLP format 3.0
    """
    global _v3_defaults
    previous = _v3_defaults
    set_v3_defaults(enabled)
    try:
        yield
    finally:
        _v3_defaults = previous
