"""Categorical-label storage for per-detection and per-identity attributes."""

from __future__ import annotations

from typing import Any


class CategoriesMixin:
    """Mixin adding per-detection categorical-label storage helpers.

    A *category* is a named categorical label describing a detection (or an
    `Identity`) along some dimension, e.g. ``{"sex": "M", "strain": "C57BL/6"}``.
    Categories are stored in a ``categories`` mapping keyed by dimension name.
    Values are typically plain strings but may be any JSON-serializable value
    (e.g. a list encoding a one-hot / class-probability distribution), so they
    round-trip through the SLP ``/instance_categories`` side-table.

    Classes using this mixin must declare a ``categories: dict`` attrs field. The
    mixin adds a convenience ``cat`` alias for that mapping plus ``set_category``
    / ``set_categories`` helpers without storing any state of its own
    (``__slots__`` is empty so slotted attrs subclasses keep their slots).

    Note:
        This plural ``categories`` mapping is distinct from the singular scalar
        ``category`` attribute on the geometry primitives (`Centroid`,
        `BoundingBox`, `SegmentationMask`, `ROI`), which names the detector class
        of a single shape. The ``cat`` alias deliberately avoids that name.
    """

    __slots__ = ()

    @property
    def cat(self) -> dict[str, Any]:
        """Alias for the ``categories`` mapping.

        Returns the live ``categories`` dict, so entries can be read or assigned
        in place, e.g. ``instance.cat["sex"] = "M"``.
        """
        return self.categories

    @cat.setter
    def cat(self, value: dict[str, Any]) -> None:
        self.categories = value

    def set_category(self, dim: str, value: Any) -> None:
        """Set a single category dimension.

        This is an in-place mutator (like ``dict.__setitem__``) and returns
        ``None``; equivalent to ``self.cat[dim] = value``.

        Args:
            dim: Category dimension name (e.g. ``"sex"``).
            value: Category value (typically a string, but any JSON-serializable
                value is allowed).
        """
        self.categories[dim] = value

    def set_categories(self, categories: dict[str, Any]) -> None:
        """Merge multiple category dimensions into the mapping.

        This is an in-place mutator (like ``dict.update``) and returns ``None``.

        Args:
            categories: A mapping of dimension names to values. Merged into the
                existing ``categories`` mapping; keys already present are
                overwritten.
        """
        self.categories.update(categories)
