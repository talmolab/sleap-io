"""Identity data structure for ground-truth animal identification."""

import uuid as uuid_module

import attrs
from attrs import define, field
from attrs.validators import instance_of

from sleap_io.model.embedding import Embedding, EmbeddingMixin


def _generate_uuid() -> str:
    """Generate a new random UUID hex string for an `Identity`."""
    return uuid_module.uuid4().hex


@define(eq=False)
class Identity(EmbeddingMixin):
    """Ground-truth animal identity, persistent across sessions and videos.

    Unlike `Track` (an ephemeral temporal trajectory within a single video),
    `Identity` represents a known animal that can be recognized across videos,
    sessions, and experiments. It carries a stable `uuid` so it can be matched
    and deduplicated across separately-loaded files and merges, where Python
    object identity (used by `Track`) and positional indices both fail.

    In multi-view setups, multiple per-camera `Track`s may map to a single
    `Identity`. The per-instance binding is stored on `Instance.identity`; the
    triangulated multi-view binding is stored on `InstanceGroup.identity`.

    Attributes:
        name: Human-readable name for this identity (e.g., "mouse_A"). Not
            required to be unique.
        uuid: Stable 32-character hex string identifying this animal across files
            and merges. Auto-generated if not provided. This is the canonical
            cross-file matching key.
        color: Optional hex color string for visualization (e.g., "#e6194b").
        metadata: Arbitrary metadata dictionary.
        embeddings: A mapping from embedding-space name (e.g. ``"reid"``) to an
            `Embedding` prototype / gallery vector representing this identity in
            that space (e.g. the cluster centroid of its member instances). Empty
            by default.

    Notes:
        `Identity` objects use object-identity equality (`eq=False`), matching
        `Track`. Use `matches()` (default `method="uuid"`) to compare identities
        across files where object identity is not meaningful.
    """

    name: str = field(default="", validator=instance_of(str))
    uuid: str = field(factory=_generate_uuid, validator=instance_of(str))
    color: str | None = field(default=None, converter=attrs.converters.optional(str))
    metadata: dict = field(factory=dict, validator=instance_of(dict))
    embeddings: dict[str, Embedding] = field(factory=dict, repr=False)

    def matches(self, other: "Identity", method: str = "uuid") -> bool:
        """Check if this identity matches another identity.

        Args:
            other: Another identity to compare with.
            method: Matching method:

                - ``"uuid"`` (default): match by the stable `uuid` key, which
                  survives serialization and cross-file merges.
                - ``"name"``: match by the `name` attribute.
                - ``"identity"``: match by Python object identity (same object).

        Returns:
            True if the identities match according to the specified method.

        Raises:
            ValueError: If `method` is not one of the supported values.
        """
        if method == "uuid":
            return self.uuid == other.uuid
        elif method == "name":
            return self.name == other.name
        elif method == "identity":
            return self is other
        else:
            raise ValueError(f"Unknown matching method: {method}")

    def __repr__(self) -> str:
        """Return a readable string representation."""
        parts = [f'Identity(name="{self.name}"']
        if self.color is not None:
            parts.append(f', color="{self.color}"')
        parts.append(")")
        return "".join(parts)
