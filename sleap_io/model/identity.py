"""Identity data structure for ground-truth animal identification."""

from __future__ import annotations

from attrs import define, field
from attrs.validators import instance_of


@define(eq=False)
class Identity:
    """Ground-truth animal identity, persistent across sessions and videos.

    Unlike `Track` (an ephemeral temporal trajectory within a single video),
    `Identity` represents a known animal that can be recognized across videos,
    sessions, and experiments. In multi-view setups, multiple per-camera `Track`s
    may map to a single `Identity`. The per-detection binding is stored on
    ``Instance.identity`` (and the analogous slot on the other detection
    modalities); the triangulated multi-view binding is stored on
    ``InstanceGroup.identity``.

    Attributes:
        name: Human-readable name for this identity (e.g., ``"mouse_A"``). Not
            required to be unique, but ``name`` is how identities are matched
            across separately-loaded files and merges.
        metadata: Arbitrary string-keyed, string-valued metadata (e.g.
            ``{"color": "#e6194b", "strain": "C57BL/6"}``). Empty by default.

    Notes:
        `Identity` objects use object-identity equality (``eq=False``), matching
        `Track`. Use `matches()` (default ``method="name"``) to compare identities
        across files, where Python object identity is not meaningful.
    """

    name: str = field(default="", validator=instance_of(str))
    metadata: dict[str, str] = field(factory=dict, validator=instance_of(dict))

    def matches(self, other: "Identity", method: str = "name") -> bool:
        """Check if this identity matches another identity.

        Args:
            other: Another identity to compare with.
            method: Matching method:

                - ``"name"`` (default): match by the `name` attribute, which
                  survives serialization and cross-file merges.
                - ``"identity"``: match by Python object identity (same object).

        Returns:
            True if the identities match according to the specified method.

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
        return f'Identity(name="{self.name}")'
