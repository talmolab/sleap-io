"""Category data structure for ground-truth class membership of detections."""

from __future__ import annotations

from attrs import define, field
from attrs.validators import instance_of


@define(eq=False)
class Category:
    """Ground-truth class membership of a detection (e.g. species, sex, condition).

    Where `Track` is an ephemeral temporal trajectory within a single video and
    `Identity` names a specific individual across videos, `Category` names the
    *class* an individual belongs to -- a group of individuals that share some
    attribute, typically assigned by classification or retrieved via re-ID (e.g.
    ``"female_fly"``, ``"fur_shaved"``, ``"mouse"``). The per-detection binding is
    stored on ``Instance.category`` (and the analogous slot on the other detection
    modalities), alongside an optional ``category_score`` (assignment confidence)
    and ``category_embedding`` (the appearance vector it was classified from).

    Attributes:
        name: Human-readable name for this category (e.g., ``"female_fly"``). Not
            required to be unique, but ``name`` is how categories are matched
            across separately-loaded files and merges.
        metadata: Arbitrary string-keyed, string-valued metadata (e.g.
            ``{"color": "#e6194b", "supercategory": "insect"}``). Empty by default.

    Notes:
        `Category` objects use object-identity equality (``eq=False``), matching
        `Track` and `Identity`. Use `matches()` (default ``method="name"``) to
        compare categories across files, where Python object identity is not
        meaningful.
    """

    name: str = field(default="", validator=instance_of(str))
    metadata: dict[str, str] = field(factory=dict, validator=instance_of(dict))

    def matches(self, other: "Category", method: str = "name") -> bool:
        """Check if this category matches another category.

        Args:
            other: Another category to compare with.
            method: Matching method:

                - ``"name"`` (default): match by the `name` attribute, which
                  survives serialization and cross-file merges.
                - ``"identity"``: match by Python object identity (same object).

        Returns:
            True if the categories match according to the specified method.

        Raises:
            ValueError: If `method` is not one of the supported values.
        """
        if method == "name":
            return self.name == other.name
        elif method == "identity":
            return self is other
        else:
            raise ValueError(f"Unknown matching method: {method}")

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return f'Category(name="{self.name}")'


def to_category(value: "Category | str | None") -> "Category | None":
    """Coerce a category-like value to a `Category` (or ``None``).

    Promotes the legacy free-form ``category: str`` field (the object-detection
    class label) to a first-class `Category`, keeping existing ``category="mouse"``
    call sites working:

    - ``None`` or the empty string ``""`` (the old "unset" sentinel) -> ``None``.
    - a non-empty ``str`` -> ``Category(name=value)``.
    - an existing `Category` -> returned unchanged.

    Args:
        value: A `Category`, a class-label string, ``""``, or ``None``.

    Returns:
        A `Category`, or ``None`` if the input was ``None`` / ``""``.

    Raises:
        TypeError: If `value` is not a `Category`, `str`, or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, Category):
        return value
    if isinstance(value, str):
        if value == "":
            return None
        return Category(name=value)
    raise TypeError(
        f"category must be a Category, str, or None, got {type(value).__name__}."
    )
