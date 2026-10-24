"""Per-record metadata for instances, labeled frames and bounding boxes.

`Instance`, `LabeledFrame` and `BoundingBox` each carry a `metadata` dictionary of
scalar values, for example COCO's `iscrowd`, `area` or annotation ids. The values are
restricted so that each key can be stored as one typed column per table on disk:

- Values are `bool`, `int`, `float` or `str`. Numpy scalars are converted to the
  matching Python type. `None` means "missing".
- Nested values (lists, dicts, arrays) are rejected. JSON-encode them into a `str`.
- Keys are non-empty strings without `/`.
- All records of one table share a kind per key (`bool`, `int`, `float` or `str`).
  `int` and `float` mix as `float`, except that integers beyond 2^53 can't be widened
  to `float` without losing precision, so mixing them with floats raises.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import attrs
import numpy as np

# Largest integer that a float64 represents exactly.
_MAX_EXACT_FLOAT_INT = 2**53


def _check_key(key: Any) -> str:
    """Validate a metadata key.

    Args:
        key: The key to check.

    Returns:
        The key, unchanged.

    Raises:
        TypeError: If the key is not a string.
        ValueError: If the key is empty or contains `/`.
    """
    if not isinstance(key, str):
        raise TypeError(f"Metadata keys must be strings, got {type(key).__name__}.")
    if not key:
        raise ValueError("Metadata keys must be non-empty.")
    if "/" in key:
        raise ValueError(f"Metadata keys can't contain '/': {key!r}.")
    return key


def normalize_metadata_value(value: Any) -> bool | int | float | str | None:
    """Convert a metadata value to a plain Python scalar.

    Args:
        value: A `bool`, `int`, `float`, `str` (or the numpy equivalents), or `None`.

    Returns:
        The value as a Python `bool`, `int`, `float`, `str`, or `None`.

    Raises:
        TypeError: If the value is not a supported scalar.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    raise TypeError(
        f"Metadata values must be bool, int, float, str or None, got "
        f"{type(value).__name__}. JSON-encode nested values into a str."
    )


def metadata_kind(value: bool | int | float | str) -> str:
    """Return the kind of a normalized metadata value.

    Args:
        value: A normalized (non-`None`) metadata value.

    Returns:
        One of `"bool"`, `"int"`, `"float"` or `"str"`.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


class RecordMetadata(dict):
    """A `dict` that only accepts valid metadata keys and scalar values.

    Values are validated (and numpy scalars converted) whenever they are set, so an
    invalid value raises at the point it's assigned rather than at save time.
    """

    __slots__ = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create from a mapping or iterable of pairs, validating every item."""
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        """Validate and set an item."""
        super().__setitem__(_check_key(key), normalize_metadata_value(value))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Validate and set several items, like `dict.update`."""
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Validate and set an item if missing, like `dict.setdefault`."""
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other: Any) -> "RecordMetadata":
        """Validate and merge items in place (`|=`)."""
        self.update(other)
        return self

    def copy(self) -> "RecordMetadata":
        """Return a shallow copy."""
        return RecordMetadata(self)

    def __reduce__(self) -> tuple:
        """Pickle as a plain mapping so unpickling revalidates nothing twice."""
        return (RecordMetadata, (dict(self),))


def to_metadata(value: Mapping | None) -> RecordMetadata:
    """Convert a mapping to a validated `RecordMetadata`.

    This is used as the attrs converter for the `metadata` field.

    Args:
        value: A mapping of metadata, or `None` for an empty one.

    Returns:
        A `RecordMetadata` holding the validated items. An existing `RecordMetadata` is
        returned as is.

    Raises:
        TypeError: If `value` is not a mapping, or holds an invalid key or value.
    """
    if isinstance(value, RecordMetadata):
        return value
    if value is None:
        return RecordMetadata()
    if not isinstance(value, Mapping):
        raise TypeError(f"metadata must be a dict, got {type(value).__name__}.")
    return RecordMetadata(value)


def merge_metadata(target: Mapping, source: Mapping) -> RecordMetadata:
    """Combine two records' metadata key by key, the target winning ties.

    Args:
        target: The metadata that takes precedence.
        source: The metadata whose keys are added where the target lacks them.

    Returns:
        A new `RecordMetadata` with the combined items.
    """
    merged = RecordMetadata(source)
    merged.update(target)
    return merged


def metadata_column_kinds(records: Iterable[Any], table: str) -> dict[str, str]:
    """Check that every key has one kind across a table's records.

    Args:
        records: Objects with a `metadata` attribute (for example all instances of a
            `Labels`).
        table: The table name used in error messages (for example `"instances"`).

    Returns:
        A dictionary mapping each key to its kind (`"bool"`, `"int"`, `"float"` or
        `"str"`). Keys whose values are all `None` are omitted.

    Raises:
        TypeError: If the records mix kinds under one key, other than `int` with
            `float`, or mix floats with integers beyond 2^53.
    """
    kinds: dict[str, str] = {}
    big_ints: set[str] = set()
    for record in records:
        for key, value in record.metadata.items():
            if value is None:
                continue
            kind = metadata_kind(value)
            if kind == "int" and abs(value) > _MAX_EXACT_FLOAT_INT:
                big_ints.add(key)
            current = kinds.get(key)
            if current is None or current == kind:
                kinds[key] = kind
            elif {current, kind} == {"int", "float"}:
                kinds[key] = "float"
            else:
                raise TypeError(
                    f"Metadata key {key!r} mixes {current} and {kind} values across "
                    f"{table}. Every record must use one kind per key."
                )
            if kinds[key] == "float" and key in big_ints:
                raise TypeError(
                    f"Metadata key {key!r} mixes floats with integers beyond 2^53 "
                    f"across {table}; they can't be widened to float exactly."
                )
    return kinds


def get_attrs_state(obj: Any) -> dict[str, Any]:
    """Return the pickle state of a slotted attrs object as a field dictionary.

    Args:
        obj: An instance of a slotted attrs class.

    Returns:
        A dictionary mapping field names to values, matching the format attrs uses.
    """
    return {a.name: getattr(obj, a.name) for a in attrs.fields(type(obj))}


def set_attrs_state(obj: Any, state: dict[str, Any] | tuple) -> None:
    """Restore a slotted attrs object from its pickle state.

    Unlike the method attrs generates, fields missing from the state (for example
    `metadata` in pickles made before sleap-io 0.10) are set to their defaults instead
    of being left unset.

    Args:
        obj: The object being unpickled or copied.
        state: A dictionary of field values, or a tuple in field order (pickles made
            with attrs < 22.2).
    """
    fields = attrs.fields(type(obj))
    if isinstance(state, tuple):
        state = dict(zip((a.name for a in fields), state))
    for a in fields:
        if a.name in state:
            value = state[a.name]
        elif isinstance(a.default, attrs.Factory):
            value = a.default.factory()
        else:
            value = a.default
        object.__setattr__(obj, a.name, value)
