"""Identity data structure for ground-truth animal identification."""

import attrs
from attrs import define, field
from attrs.validators import instance_of


@define(eq=False)
class Identity:
    """Ground-truth animal identity, persistent across sessions and videos.

    Unlike Track (an ephemeral temporal trajectory within a single video),
    Identity represents a known animal that can be recognized across videos,
    sessions, and experiments.

    In multi-view setups, multiple per-camera Tracks may map to a single
    Identity. The mapping is stored on RecordingSession metadata.

    Attributes:
        name: Human-readable name for this identity (e.g., "mouse_A").
        color: Optional hex color string for visualization (e.g., "#e6194b").
        metadata: Arbitrary metadata dictionary.
    """

    name: str = field(default="", validator=instance_of(str))
    color: str | None = field(default=None, converter=attrs.converters.optional(str))
    metadata: dict = field(factory=dict, validator=instance_of(dict))

    def __repr__(self) -> str:
        """Return a readable string representation."""
        parts = [f'Identity(name="{self.name}"']
        if self.color is not None:
            parts.append(f', color="{self.color}"')
        parts.append(")")
        return "".join(parts)
