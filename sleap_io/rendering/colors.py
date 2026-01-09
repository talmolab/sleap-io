"""Color palette utilities for pose rendering.

This module provides built-in color palettes and utilities for converting colors
to Skia format for rendering pose data.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import skia

# Named colors (CSS-like subset)
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (139, 69, 19),
}

# Type alias for flexible color specification
# Note: Use Tuple (not tuple) for Python 3.8 compatibility in Union
ColorSpec = Union[
    Tuple[int, int, int],  # RGB int tuple: (255, 128, 0)
    Tuple[float, float, float],  # RGB float tuple: (1.0, 0.5, 0.0)
    int,  # Grayscale int: 128 → (128, 128, 128)
    float,  # Grayscale float: 0.5 → (127, 127, 127)
    str,  # Named, hex, or palette reference
]

# Built-in color palettes as RGB tuples
PALETTES: dict[str, list[tuple[int, int, int]]] = {
    # MATLAB default colors
    "standard": [
        (0, 114, 189),  # Blue
        (217, 83, 25),  # Orange
        (237, 177, 32),  # Yellow/Gold
        (126, 47, 142),  # Purple
        (119, 172, 48),  # Green
        (77, 190, 238),  # Light blue
        (162, 20, 47),  # Dark red
    ],
    # High-contrast distinct colors (good for instances/tracks)
    "distinct": [
        (255, 100, 100),  # Light red
        (100, 100, 255),  # Light blue
        (100, 255, 100),  # Light green
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
        (255, 180, 100),  # Orange
        (180, 100, 255),  # Purple
        (255, 150, 150),  # Pink
        (150, 255, 200),  # Mint
    ],
    # Rainbow spectrum (good for node types)
    "rainbow": [
        (255, 0, 0),  # Red
        (255, 127, 0),  # Orange
        (255, 255, 0),  # Yellow
        (127, 255, 0),  # Lime
        (0, 255, 0),  # Green
        (0, 255, 127),  # Spring
        (0, 255, 255),  # Cyan
        (0, 127, 255),  # Azure
        (0, 0, 255),  # Blue
        (127, 0, 255),  # Violet
        (255, 0, 255),  # Magenta
        (255, 0, 127),  # Rose
    ],
    # Warm colors
    "warm": [
        (255, 87, 51),  # Red-orange
        (255, 140, 0),  # Dark orange
        (255, 195, 0),  # Amber
        (255, 215, 0),  # Gold
        (255, 69, 0),  # Red-orange
        (220, 20, 60),  # Crimson
    ],
    # Cool colors
    "cool": [
        (0, 150, 255),  # Azure
        (0, 191, 255),  # Deep sky blue
        (30, 144, 255),  # Dodger blue
        (65, 105, 225),  # Royal blue
        (138, 43, 226),  # Blue violet
        (75, 0, 130),  # Indigo
    ],
    # Pastel colors (good for overlays)
    "pastel": [
        (255, 179, 186),  # Light pink
        (255, 223, 186),  # Peach
        (255, 255, 186),  # Light yellow
        (186, 255, 201),  # Light green
        (186, 225, 255),  # Light blue
        (218, 186, 255),  # Light purple
    ],
    # Seaborn-inspired palette
    "seaborn": [
        (76, 114, 176),  # Blue
        (221, 132, 82),  # Orange
        (85, 168, 104),  # Green
        (196, 78, 82),  # Red
        (129, 114, 179),  # Purple
        (147, 120, 96),  # Brown
        (218, 139, 195),  # Pink
        (140, 140, 140),  # Gray
        (204, 185, 116),  # Olive
        (100, 181, 205),  # Cyan
    ],
    # Tableau 10
    "tableau10": [
        (31, 119, 180),  # Blue
        (255, 127, 14),  # Orange
        (44, 160, 44),  # Green
        (214, 39, 40),  # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),  # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
        (188, 189, 34),  # Olive
        (23, 190, 207),  # Cyan
    ],
    # Viridis-inspired (perceptually uniform)
    "viridis": [
        (68, 1, 84),  # Dark purple
        (72, 40, 120),  # Purple
        (62, 73, 137),  # Blue-purple
        (49, 104, 142),  # Blue
        (38, 130, 142),  # Teal
        (31, 158, 137),  # Green-teal
        (53, 183, 121),  # Green
        (109, 205, 89),  # Yellow-green
        (180, 222, 44),  # Yellow
        (253, 231, 37),  # Bright yellow
    ],
}

# Type alias for palette names
PaletteName = Literal[
    "standard",
    "distinct",
    "rainbow",
    "warm",
    "cool",
    "pastel",
    "seaborn",
    "tableau10",
    "viridis",
    # Colorcet palettes (require optional dependency)
    "glasbey",
    "glasbey_hv",
    "glasbey_cool",
    "glasbey_warm",
]

# Type alias for color schemes
ColorScheme = Literal["track", "instance", "node", "auto"]


def get_palette(
    name: Union[PaletteName, str], n_colors: int
) -> list[tuple[int, int, int]]:
    """Get n colors from a named palette as RGB tuples.

    Args:
        name: Palette name. Built-in options: 'standard', 'distinct', 'rainbow',
            'warm', 'cool', 'pastel', 'seaborn', 'tableau10', 'viridis'.
            With colorcet installed: 'glasbey', 'glasbey_hv', 'glasbey_cool',
            'glasbey_warm'.
        n_colors: Number of colors needed.

    Returns:
        List of (R, G, B) tuples.

    Raises:
        ValueError: If palette name is not recognized.
    """
    # Try built-in palettes first
    if name in PALETTES:
        palette = PALETTES[name]
        return _extend_palette(palette, n_colors)

    # Try colorcet palettes
    import colorcet as cc

    if name in cc.palette:
        hex_colors = cc.palette[name]
        rgb_colors = [_hex_to_rgb(c) for c in hex_colors]
        return _extend_palette(rgb_colors, n_colors)

    # Unknown palette - raise error with available options
    raise ValueError(
        f"Unknown palette: {name}. "
        f"Available: {list(PALETTES.keys())} (built-in), "
        "or any colorcet palette (e.g., glasbey, glasbey_hv, fire, rainbow4)"
    )


def _extend_palette(
    palette: Sequence[tuple[int, int, int]], n_colors: int
) -> list[tuple[int, int, int]]:
    """Extend palette by cycling if more colors needed.

    Args:
        palette: Base palette colors.
        n_colors: Number of colors needed.

    Returns:
        List of n_colors RGB tuples.
    """
    if n_colors <= len(palette):
        return list(palette[:n_colors])

    extended = []
    for i in range(n_colors):
        extended.append(palette[i % len(palette)])
    return extended


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string like '#ff0000' or 'ff0000'.

    Returns:
        (R, G, B) tuple with values 0-255.
    """
    hex_color = hex_color.lstrip("#")
    # Handle 3-digit hex (#rgb -> #rrggbb)
    if len(hex_color) == 3:
        hex_color = hex_color[0] * 2 + hex_color[1] * 2 + hex_color[2] * 2
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def resolve_color(color: ColorSpec) -> tuple[int, int, int]:
    """Resolve a flexible color specification to an RGB tuple.

    Args:
        color: Color specification in various formats:
            - RGB int tuple: ``(255, 128, 0)``
            - RGB float tuple: ``(1.0, 0.5, 0.0)`` - values in 0.0-1.0 range
            - Grayscale int: ``128`` → ``(128, 128, 128)``
            - Grayscale float: ``0.5`` → ``(127, 127, 127)``
            - Named color: ``"black"``, ``"white"``, ``"red"``, etc.
            - Hex color: ``"#ff8000"`` or ``"#f80"``
            - Palette index: ``"tableau10[2]"``, ``"glasbey[5]"``

    Returns:
        RGB tuple of integers in 0-255 range.

    Raises:
        ValueError: If color specification is invalid.

    Examples:
        >>> resolve_color((255, 128, 0))
        (255, 128, 0)

        >>> resolve_color((1.0, 0.5, 0.0))
        (255, 127, 0)

        >>> resolve_color("red")
        (255, 0, 0)

        >>> resolve_color("#ff8000")
        (255, 128, 0)

        >>> resolve_color("#f80")
        (255, 136, 0)

        >>> resolve_color("tableau10[2]")
        (44, 160, 44)

        >>> resolve_color(128)
        (128, 128, 128)

        >>> resolve_color(0.5)
        (127, 127, 127)
    """
    # Grayscale int
    if isinstance(color, int):
        value = max(0, min(255, color))
        return (value, value, value)

    # Grayscale float (0.0-1.0 range)
    if isinstance(color, float):
        value = int(max(0.0, min(1.0, color)) * 255)
        return (value, value, value)

    # Tuple (RGB)
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError(f"RGB tuple must have 3 elements, got {len(color)}")

        r, g, b = color

        # Detect float vs int by Python type
        if isinstance(r, float) or isinstance(g, float) or isinstance(b, float):
            # Float tuple: 0.0-1.0 range
            r_int = int(max(0.0, min(1.0, float(r))) * 255)
            g_int = int(max(0.0, min(1.0, float(g))) * 255)
            b_int = int(max(0.0, min(1.0, float(b))) * 255)
            return (r_int, g_int, b_int)
        else:
            # Int tuple: 0-255 range
            r_int = max(0, min(255, int(r)))
            g_int = max(0, min(255, int(g)))
            b_int = max(0, min(255, int(b)))
            return (r_int, g_int, b_int)

    # String
    if isinstance(color, str):
        color_lower = color.lower().strip()

        # Named color
        if color_lower in NAMED_COLORS:
            return NAMED_COLORS[color_lower]

        # Hex color
        if color.startswith("#"):
            hex_part = color[1:]
            if len(hex_part) == 3 or len(hex_part) == 6:
                return _hex_to_rgb(color)
            raise ValueError(f"Invalid hex color: {color}")

        # Palette index: "palette_name[index]"
        palette_match = re.match(r"^(\w+)\[(\d+)\]$", color)
        if palette_match:
            palette_name = palette_match.group(1)
            index = int(palette_match.group(2))

            # Get palette colors
            try:
                palette_colors = get_palette(palette_name, index + 1)
                if index < len(palette_colors):
                    return palette_colors[index]
            except ValueError:
                pass

            raise ValueError(f"Invalid palette index: {color}")

        raise ValueError(
            f"Unknown color: {color!r}. Valid named colors: {list(NAMED_COLORS.keys())}"
        )

    raise TypeError(f"Invalid color type: {type(color).__name__}")


def rgb_to_skia_color(rgb: tuple[int, int, int], alpha: int = 255) -> "skia.Color4f":
    """Convert RGB tuple to Skia Color4f.

    Args:
        rgb: (R, G, B) tuple with values 0-255.
        alpha: Alpha value 0-255. Defaults to 255 (opaque).

    Returns:
        Skia Color4f object.
    """
    import skia

    return skia.Color(rgb[0], rgb[1], rgb[2], alpha)


def determine_color_scheme(
    has_tracks: bool,
    is_single_image: bool,
    scheme: ColorScheme = "auto",
) -> ColorScheme:
    """Determine the color scheme to use based on context.

    When scheme is "auto", uses smart defaults:
    - If tracks available: color by track (consistent across frames)
    - Else if single image: color by instance (distinguishes animals)
    - Else if multi-image: color by node type (prevents flicker)

    Args:
        has_tracks: Whether the labels have track assignments.
        is_single_image: Whether rendering a single image (vs video).
        scheme: Requested color scheme. "auto" uses smart defaults.

    Returns:
        Resolved color scheme to use.
    """
    if scheme != "auto":
        return scheme

    # Smart defaults
    if has_tracks:
        return "track"
    elif is_single_image:
        return "instance"
    else:
        return "node"


def build_color_map(
    scheme: ColorScheme,
    n_instances: int,
    n_nodes: int,
    n_tracks: int,
    track_indices: Optional[list[int]] = None,
    palette: Union[PaletteName, str] = "standard",
) -> dict[str, list[tuple[int, int, int]]]:
    """Build color mapping based on scheme.

    Args:
        scheme: Color scheme to use.
        n_instances: Number of instances in frame.
        n_nodes: Number of nodes in skeleton.
        n_tracks: Total number of tracks (for track coloring).
        track_indices: Track index for each instance (for track coloring).
        palette: Color palette name.

    Returns:
        Dictionary with 'instance_colors' and/or 'node_colors' lists.
    """
    colors = {}

    if scheme == "track":
        # Colors based on track identity
        n = max(n_tracks, n_instances) if n_tracks > 0 else n_instances
        palette_colors = get_palette(palette, n)

        if track_indices is not None:
            instance_colors = [
                palette_colors[idx % len(palette_colors)] for idx in track_indices
            ]
        else:
            instance_colors = palette_colors[:n_instances]

        colors["instance_colors"] = instance_colors

    elif scheme == "instance":
        # Colors based on instance index
        palette_colors = get_palette(palette, n_instances)
        colors["instance_colors"] = palette_colors

    elif scheme == "node":
        # Colors based on node type
        palette_colors = get_palette(palette, n_nodes)
        colors["node_colors"] = palette_colors

    return colors
