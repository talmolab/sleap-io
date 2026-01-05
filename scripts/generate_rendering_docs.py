#!/usr/bin/env python
"""Generate example images for rendering documentation.

This script generates all the example images used in docs/rendering.md.
Run from the repository root:

    uv run python scripts/generate_rendering_docs.py

Images are saved to docs/assets/rendering/ and should be committed to the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sleap_io as sio

# Paths (relative to repo root)
DATA_PATH = Path("tests/data/slp/centered_pair_predictions.slp")
OUTPUT_DIR = Path("docs/assets/rendering")


def get_font(size: int = 14, bold: bool = False):
    """Get a font for text overlays, with fallbacks."""
    # Try common system fonts
    font_names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "Helvetica Bold.ttf" if bold else "Helvetica.ttf",
    ]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    # Fallback to default
    return ImageFont.load_default()


def add_text_overlay(
    img: np.ndarray,
    text: str,
    position: str = "top-left",
    font_size: int = 16,
    padding: int = 6,
    bg_alpha: float = 0.7,
) -> np.ndarray:
    """Add text overlay with semi-transparent background to image.

    Args:
        img: Input image (H, W, 3) uint8
        text: Text to overlay
        position: One of "top-left", "top-center", "top-right",
                  "bottom-left", "bottom-center", "bottom-right"
        font_size: Font size in pixels
        padding: Padding around text
        bg_alpha: Background transparency (0-1)

    Returns:
        Image with text overlay
    """
    pil_img = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = get_font(font_size, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Calculate position
    img_w, img_h = pil_img.size
    if "left" in position:
        x = padding
    elif "right" in position:
        x = img_w - text_w - padding * 2
    else:  # center
        x = (img_w - text_w) // 2

    if "top" in position:
        y = padding
    else:  # bottom
        y = img_h - text_h - padding * 2

    # Draw background rectangle
    bg_color = (0, 0, 0, int(255 * bg_alpha))
    draw.rectangle(
        [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
        fill=bg_color,
    )

    # Draw text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # Composite
    result = Image.alpha_composite(pil_img, overlay)
    return np.array(result.convert("RGB"))


def save_image(img: np.ndarray, name: str, max_width: int = 400) -> Path:
    """Save image with optional downscaling for docs."""
    path = OUTPUT_DIR / name
    pil_img = Image.fromarray(img)

    # Downscale if too large for docs
    if pil_img.width > max_width:
        ratio = max_width / pil_img.width
        new_size = (max_width, int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    pil_img.save(path)
    print(f"  Saved: {path} ({pil_img.width}x{pil_img.height})")
    return path


def make_grid(
    images: list[np.ndarray], cols: int = 2, padding: int = 4, bg_color: int = 40
) -> np.ndarray:
    """Combine images into a grid with padding."""
    n = len(images)
    rows = (n + cols - 1) // cols

    h, w = images[0].shape[:2]
    grid = (
        np.ones(
            (rows * h + (rows - 1) * padding, cols * w + (cols - 1) * padding, 3),
            dtype=np.uint8,
        )
        * bg_color
    )

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y = r * (h + padding)
        x = c * (w + padding)
        grid[y : y + h, x : x + w] = img

    return grid


def main():
    """Generate all documentation images."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading predictions...")
    labels = sio.load_slp(DATA_PATH)
    lf = labels.labeled_frames[0]
    print(f"  {len(labels.labeled_frames)} frames, {len(labels.skeleton.nodes)} nodes")

    # Import skia for callbacks (late import since it's optional)
    import skia

    # =========================================================================
    # 1. Basic rendering - single frame
    # =========================================================================
    print("\n1. Basic rendering...")
    img = sio.render_image(lf)
    save_image(img, "render_basic.png")

    # =========================================================================
    # 2. Color schemes - individual images for each
    # =========================================================================
    print("\n2. Color schemes (individual images)...")

    # Color by track
    img = sio.render_image(lf, color_by="track")
    save_image(img, "render_color_track.png")

    # Color by instance
    img = sio.render_image(lf, color_by="instance")
    save_image(img, "render_color_instance.png")

    # Color by node
    img = sio.render_image(lf, color_by="node", palette="rainbow")
    save_image(img, "render_color_node.png")

    # =========================================================================
    # 3. Color palettes - with text overlays, color by node
    # =========================================================================
    print("\n3. Built-in color palettes (with overlays)...")
    palettes = ["distinct", "rainbow", "seaborn", "tableau10", "viridis", "pastel"]
    images = []
    for pal in palettes:
        img = sio.render_image(lf, color_by="node", palette=pal)
        img = add_text_overlay(img, pal, position="top-left", font_size=14)
        images.append(img)

    # 2x3 grid
    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_palettes.png", max_width=700)

    # =========================================================================
    # 4. Colorcet palettes gallery
    # =========================================================================
    print("\n4. Colorcet palettes (with overlays)...")
    try:
        import colorcet  # noqa: F401

        cc_palettes = ["glasbey", "glasbey_hv", "glasbey_cool", "glasbey_warm"]
        images = []
        for pal in cc_palettes:
            img = sio.render_image(lf, color_by="node", palette=pal)
            img = add_text_overlay(img, pal, position="top-left", font_size=14)
            images.append(img)

        grid = make_grid(images, cols=2, padding=4)
        save_image(grid, "render_palettes_colorcet.png", max_width=500)
    except ImportError:
        print("  Skipping colorcet (not installed)")

    # =========================================================================
    # 5. Marker shapes - with text overlays, color by node
    # =========================================================================
    print("\n5. Marker shapes (with overlays)...")
    shapes = ["circle", "square", "diamond", "triangle", "cross"]
    images = []
    for shape in shapes:
        img = sio.render_image(
            lf, marker_shape=shape, marker_size=6.0, color_by="node", palette="rainbow"
        )
        img = add_text_overlay(img, shape, position="top-left", font_size=14)
        images.append(img)

    # Add empty slot for 2x3 grid (or use 3+2 layout)
    # Use 3 cols, 2 rows with last cell empty
    images.append(np.ones_like(images[0]) * 40)  # dark gray placeholder
    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_marker_shapes.png", max_width=700)

    # =========================================================================
    # 6. Styling variations - with text overlays
    # =========================================================================
    print("\n6. Styling variations...")

    # Size variations
    print("  - Size variations...")
    size_configs = [
        ("small", {"marker_size": 3.0, "line_width": 1.5}),
        ("medium", {"marker_size": 6.0, "line_width": 3.0}),
        ("large", {"marker_size": 10.0, "line_width": 5.0}),
    ]
    images = []
    for label, cfg in size_configs:
        img = sio.render_image(lf, color_by="node", palette="rainbow", **cfg)
        img = add_text_overlay(img, label, position="top-left", font_size=14)
        images.append(img)

    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_sizes.png", max_width=700)

    # Alpha variations
    print("  - Alpha variations...")
    alpha_configs = [
        ("alpha=1.0", 1.0),
        ("alpha=0.5", 0.5),
        ("alpha=0.25", 0.25),
    ]
    images = []
    for label, alpha in alpha_configs:
        img = sio.render_image(lf, alpha=alpha, color_by="node", palette="rainbow")
        img = add_text_overlay(img, label, position="top-left", font_size=14)
        images.append(img)

    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_alpha.png", max_width=700)

    # Show nodes/edges toggle
    print("  - Toggle variations...")
    toggle_configs = [
        ("both", {"show_nodes": True, "show_edges": True}),
        ("edges only", {"show_nodes": False, "show_edges": True}),
        ("nodes only", {"show_nodes": True, "show_edges": False}),
    ]
    images = []
    for label, cfg in toggle_configs:
        img = sio.render_image(lf, color_by="node", palette="rainbow", **cfg)
        img = add_text_overlay(img, label, position="top-left", font_size=14)
        images.append(img)

    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_toggles.png", max_width=700)

    # =========================================================================
    # 7. Missing video fallback
    # =========================================================================
    print("\n7. Fallback rendering...")
    img = sio.render_image(
        lf,
        require_video=False,
        fallback_color=(40, 40, 40),
        color_by="node",
        palette="rainbow",
    )
    save_image(img, "render_fallback.png")

    # =========================================================================
    # 8. Callback examples
    # =========================================================================
    print("\n8. Callback examples...")

    # Instance labels callback
    def draw_instance_labels(ctx):
        """Draw track name above each instance."""
        centroid = ctx.get_centroid()
        if centroid is None:
            return

        cx, cy = ctx.world_to_canvas(centroid[0], centroid[1])

        # Create text
        font = skia.Font(skia.Typeface("Arial"), 14 * ctx.scale)
        label = ctx.track_name or f"Instance {ctx.instance_idx}"
        text_blob = skia.TextBlob(label, font)

        # Background for visibility
        bounds = font.measureText(label)
        bg_paint = skia.Paint(Color=skia.Color4f(0, 0, 0, 0.6))
        ctx.canvas.drawRect(
            skia.Rect(cx - 2, cy - 18 * ctx.scale, cx + bounds + 2, cy - 4 * ctx.scale),
            bg_paint,
        )

        # Text
        paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        ctx.canvas.drawTextBlob(text_blob, cx, cy - 6 * ctx.scale, paint)

    img = sio.render_image(lf, per_instance_callback=draw_instance_labels)
    save_image(img, "render_callback_labels.png")

    # Bounding box callback
    def draw_bounding_box(ctx):
        """Draw dashed bounding box around each instance."""
        bbox = ctx.get_bbox()
        if bbox is None:
            return

        x1, y1, x2, y2 = bbox
        x1, y1 = ctx.world_to_canvas(x1, y1)
        x2, y2 = ctx.world_to_canvas(x2, y2)

        pad = 8 * ctx.scale
        rect = skia.Rect(x1 - pad, y1 - pad, x2 + pad, y2 + pad)

        # Dashed stroke
        dash_effect = skia.DashPathEffect.Make([6, 3], 0)
        paint = skia.Paint(
            Color=skia.ColorWHITE,
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=2 * ctx.scale,
            PathEffect=dash_effect,
        )
        ctx.canvas.drawRect(rect, paint)

    img = sio.render_image(lf, per_instance_callback=draw_bounding_box)
    save_image(img, "render_callback_boxes.png")

    # Frame info callback
    def draw_frame_info(ctx):
        """Draw frame info in corner."""
        font = skia.Font(skia.Typeface("Arial"), 14)
        text = f"Frame: {ctx.frame_idx}  Instances: {len(ctx.instances)}"
        text_blob = skia.TextBlob(text, font)

        # Background rectangle
        bg_paint = skia.Paint(Color=skia.Color4f(0, 0, 0, 0.7))
        ctx.canvas.drawRect(skia.Rect(4, 4, 180, 24), bg_paint)

        # Text
        text_paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        ctx.canvas.drawTextBlob(text_blob, 8, 18, text_paint)

    img = sio.render_image(lf, post_render_callback=draw_frame_info)
    save_image(img, "render_callback_info.png")

    # Combined callbacks
    def combined_callback(ctx):
        draw_bounding_box(ctx)
        draw_instance_labels(ctx)

    img = sio.render_image(
        lf,
        per_instance_callback=combined_callback,
        post_render_callback=draw_frame_info,
    )
    save_image(img, "render_callback_combined.png")

    # =========================================================================
    # 9. Multi-frame montage (horizontal strip for Pattern 4)
    # =========================================================================
    print("\n9. Multi-frame montage (horizontal strip)...")
    # Get evenly spaced frames for a horizontal strip
    n_frames = 5
    step = max(1, len(labels.labeled_frames) // n_frames)
    frame_indices = list(range(0, len(labels.labeled_frames), step))[:n_frames]

    images = []
    for i in frame_indices:
        img = sio.render_image(labels, lf_ind=i, color_by="track")
        images.append(img)

    # Horizontal strip (1 row)
    grid = make_grid(images, cols=n_frames, padding=4)
    save_image(grid, "render_montage.png", max_width=900)

    # Summary
    n_images = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n✅ Generated {n_images} images in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
