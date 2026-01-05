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
from PIL import Image

import sleap_io as sio
from sleap_io.rendering import get_palette

# Paths (relative to repo root)
DATA_PATH = Path("tests/data/slp/centered_pair_predictions.slp")
OUTPUT_DIR = Path("docs/assets/rendering")


def save_image(img: np.ndarray, name: str, max_width: int = 600) -> Path:
    """Save image with optional downscaling for docs."""
    path = OUTPUT_DIR / name
    pil_img = Image.fromarray(img)

    # Downscale if too large for docs (keep reasonable size)
    if pil_img.width > max_width:
        ratio = max_width / pil_img.width
        new_size = (max_width, int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    pil_img.save(path)
    print(f"  Saved: {path} ({pil_img.width}x{pil_img.height})")
    return path


def make_grid(images: list[np.ndarray], cols: int = 2, padding: int = 4) -> np.ndarray:
    """Combine images into a grid with padding."""
    n = len(images)
    rows = (n + cols - 1) // cols

    h, w = images[0].shape[:2]
    grid = (
        np.ones(
            (rows * h + (rows - 1) * padding, cols * w + (cols - 1) * padding, 3),
            dtype=np.uint8,
        )
        * 255
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
    # 2. Color schemes comparison
    # =========================================================================
    print("\n2. Color schemes...")
    schemes = ["track", "instance", "node"]
    images = []
    for scheme in schemes:
        img = sio.render_image(lf, color_by=scheme)
        images.append(img)

    # Create 1x3 grid
    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_color_schemes.png", max_width=900)

    # Also save individuals for reference
    for scheme, img in zip(schemes, images):
        save_image(img, f"render_color_{scheme}.png")

    # =========================================================================
    # 3. Color palettes comparison
    # =========================================================================
    print("\n3. Color palettes...")
    palettes = ["distinct", "tableau10", "seaborn", "viridis"]
    images = []
    for pal in palettes:
        img = sio.render_image(lf, color_by="instance", palette=pal)
        images.append(img)

    grid = make_grid(images, cols=2, padding=4)
    save_image(grid, "render_palettes.png")

    # =========================================================================
    # 4. Marker shapes
    # =========================================================================
    print("\n4. Marker shapes...")
    shapes = ["circle", "square", "diamond", "triangle", "cross"]
    images = []
    for shape in shapes:
        img = sio.render_image(lf, marker_shape=shape, marker_size=6.0)
        images.append(img)

    # 2x3 grid (5 shapes + repeat first to fill)
    images.append(images[0])
    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_marker_shapes.png", max_width=900)

    # =========================================================================
    # 5. Styling variations
    # =========================================================================
    print("\n5. Styling variations...")

    # Size variations
    print("  - Size variations...")
    size_configs = [
        {"marker_size": 3.0, "line_width": 1.5},  # Small
        {"marker_size": 6.0, "line_width": 3.0},  # Medium
        {"marker_size": 10.0, "line_width": 5.0},  # Large
    ]
    images = []
    for cfg in size_configs:
        img = sio.render_image(lf, **cfg)
        images.append(img)

    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_sizes.png", max_width=900)

    # Alpha variations
    print("  - Alpha variations...")
    alpha_values = [1.0, 0.5, 0.25]
    images = []
    for alpha in alpha_values:
        img = sio.render_image(lf, alpha=alpha)
        images.append(img)

    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_alpha.png", max_width=900)

    # Show nodes/edges toggle
    print("  - Toggle variations...")
    toggle_configs = [
        {"show_nodes": True, "show_edges": True},  # Both
        {"show_nodes": False, "show_edges": True},  # Edges only
        {"show_nodes": True, "show_edges": False},  # Nodes only
    ]
    images = []
    for cfg in toggle_configs:
        img = sio.render_image(lf, **cfg)
        images.append(img)

    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_toggles.png", max_width=900)

    # =========================================================================
    # 6. Missing video fallback
    # =========================================================================
    print("\n6. Fallback rendering...")
    img = sio.render_image(
        lf,
        require_video=False,
        fallback_color=(40, 40, 40),  # Dark gray
    )
    save_image(img, "render_fallback.png")

    # =========================================================================
    # 7. Callback examples
    # =========================================================================
    print("\n7. Callback examples...")

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
            skia.Rect(
                cx - 2, cy - 18 * ctx.scale, cx + bounds + 2, cy - 4 * ctx.scale
            ),
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
        lf, per_instance_callback=combined_callback, post_render_callback=draw_frame_info
    )
    save_image(img, "render_callback_combined.png")

    # =========================================================================
    # 8. Multi-frame montage
    # =========================================================================
    print("\n8. Multi-frame montage...")
    # Get evenly spaced frames
    n_frames = 6
    step = max(1, len(labels.labeled_frames) // n_frames)
    frame_indices = list(range(0, len(labels.labeled_frames), step))[:n_frames]

    images = []
    for i in frame_indices:
        img = sio.render_image(labels, lf_ind=i, color_by="track")
        images.append(img)

    grid = make_grid(images, cols=3, padding=4)
    save_image(grid, "render_montage.png", max_width=900)

    # =========================================================================
    # 9. Color by node (rainbow skeleton)
    # =========================================================================
    print("\n9. Color by node detail...")
    img = sio.render_image(lf, color_by="node", palette="rainbow", marker_size=5.0)
    save_image(img, "render_color_node_detail.png")

    # Summary
    n_images = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n✅ All images saved to {OUTPUT_DIR}/")
    print(f"   Total: {n_images} images")


if __name__ == "__main__":
    main()
