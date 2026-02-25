#!/usr/bin/env python
"""Generate example images for rendering documentation.

This script generates all the example images used in docs/rendering.md.
Each image corresponds exactly to a code block in the documentation.

Run from the repository root:

    uv run python scripts/generate_rendering_docs.py

Images are saved to docs/assets/rendering/ and should be committed to the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import sleap_io as sio
from sleap_io.rendering import InstanceContext, RenderContext

# Paths
DATA_PATH = Path("tests/data/slp/centered_pair_predictions.slp")
OUTPUT_DIR = Path("docs/assets/rendering")


def save_image(img: np.ndarray, name: str, max_width: int = 400) -> Path:
    """Save image with optional downscaling for docs."""
    path = OUTPUT_DIR / name
    pil_img = Image.fromarray(img)

    # Downscale if too large
    if pil_img.width > max_width:
        ratio = max_width / pil_img.width
        new_size = (max_width, int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    pil_img.save(path)
    print(f"  {name} ({pil_img.width}x{pil_img.height})")
    return path


def main():
    """Generate all documentation images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading predictions...")
    labels = sio.load_slp(DATA_PATH)
    lf = labels.labeled_frames[0]
    print(f"  {len(labels.labeled_frames)} frames, {len(labels.skeleton.nodes)} nodes")

    # Import skia for callbacks
    import skia

    # =========================================================================
    # Basic rendering
    # =========================================================================
    print("\nBasic rendering...")
    img = sio.render_image(lf)
    save_image(img, "render_basic.png")

    # =========================================================================
    # Color schemes - each one individually
    # =========================================================================
    print("\nColor schemes...")

    img = sio.render_image(lf, color_by="track")
    save_image(img, "color_by_track.png")

    img = sio.render_image(lf, color_by="instance")
    save_image(img, "color_by_instance.png")

    img = sio.render_image(lf, color_by="node")
    save_image(img, "color_by_node.png")

    # =========================================================================
    # Built-in palettes - each one individually
    # =========================================================================
    print("\nBuilt-in palettes...")

    palettes = [
        "standard",
        "distinct",
        "rainbow",
        "warm",
        "cool",
        "pastel",
        "seaborn",
        "tableau10",
        "viridis",
    ]
    for palette in palettes:
        img = sio.render_image(lf, color_by="node", palette=palette)
        save_image(img, f"palette_{palette}.png")

    # =========================================================================
    # Colorcet palettes - each one individually
    # =========================================================================
    print("\nColorcet palettes...")
    try:
        import colorcet  # noqa: F401

        for palette in ["glasbey", "glasbey_hv", "glasbey_cool", "glasbey_warm"]:
            img = sio.render_image(lf, color_by="node", palette=palette)
            save_image(img, f"palette_{palette}.png")
    except ImportError:
        print("  Skipping colorcet (not installed)")

    # =========================================================================
    # Marker shapes - each one individually
    # =========================================================================
    print("\nMarker shapes...")

    for shape in ["circle", "square", "diamond", "triangle", "cross"]:
        img = sio.render_image(lf, marker_shape=shape, marker_size=6.0)
        save_image(img, f"shape_{shape}.png")

    # =========================================================================
    # Size variations - each one individually
    # =========================================================================
    print("\nSize variations...")

    img = sio.render_image(lf, marker_size=3.0, line_width=1.5)
    save_image(img, "size_small.png")

    img = sio.render_image(lf, marker_size=6.0, line_width=3.0)
    save_image(img, "size_medium.png")

    img = sio.render_image(lf, marker_size=10.0, line_width=5.0)
    save_image(img, "size_large.png")

    # =========================================================================
    # Alpha variations - each one individually
    # =========================================================================
    print("\nAlpha variations...")

    img = sio.render_image(lf, alpha=1.0)
    save_image(img, "alpha_100.png")

    img = sio.render_image(lf, alpha=0.5)
    save_image(img, "alpha_50.png")

    img = sio.render_image(lf, alpha=0.25)
    save_image(img, "alpha_25.png")

    # =========================================================================
    # Toggle variations - each one individually
    # =========================================================================
    print("\nToggle variations...")

    img = sio.render_image(lf, show_nodes=True, show_edges=True)
    save_image(img, "toggle_both.png")

    img = sio.render_image(lf, show_nodes=False, show_edges=True)
    save_image(img, "toggle_edges_only.png")

    img = sio.render_image(lf, show_nodes=True, show_edges=False)
    save_image(img, "toggle_nodes_only.png")

    # =========================================================================
    # Scaling and cropping
    # =========================================================================
    print("\nScaling and cropping...")

    # Scale comparison - show 3 scales side by side
    scales = [1.0, 0.5, 0.25]
    scale_imgs = []
    for s in scales:
        img = sio.render_image(lf, scale=s)
        # Resize all to same height for comparison
        pil_img = Image.fromarray(img)
        target_h = 192  # Common height
        ratio = target_h / pil_img.height
        new_w = int(pil_img.width * ratio)
        pil_img = pil_img.resize((new_w, target_h), Image.Resampling.NEAREST)
        scale_imgs.append(np.array(pil_img))
    # Pad to same width and concatenate
    max_w = max(img.shape[1] for img in scale_imgs)
    padded = []
    for img in scale_imgs:
        if img.shape[1] < max_w:
            pad = np.zeros((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)
    scale_comp = np.concatenate(padded, axis=0)
    save_image(scale_comp, "scale_comparison.png", max_width=600)

    # Crop region - crop a 200x200 region
    img = sio.render_image(lf, crop=(100, 100, 300, 300))
    save_image(img, "crop_region.png")

    # Zoomed crop - small region at 2x scale
    img = sio.render_image(lf, crop=(140, 120, 240, 220), scale=2.0)
    save_image(img, "crop_zoomed.png")

    # Auto-fit around all instances
    img = sio.render_image(lf, crop="auto")
    save_image(img, "crop_autofit.png")

    # =========================================================================
    # Background control
    # =========================================================================
    print("\nBackground control...")

    # Named color
    img = sio.render_image(lf, background="black")
    save_image(img, "background_black.png")

    # RGB tuple
    img = sio.render_image(lf, background=(40, 40, 40))
    save_image(img, "background_rgb.png")

    # Hex color (dark blue)
    img = sio.render_image(lf, background="#1a1a2e")
    save_image(img, "background_hex.png")

    # Palette color as background
    img = sio.render_image(lf, background="tableau10[0]")
    save_image(img, "background_palette.png")

    # =========================================================================
    # Callback examples
    # =========================================================================
    print("\nCallback examples...")

    # Instance labels
    def draw_labels(ctx: InstanceContext):
        centroid = ctx.get_centroid()
        if centroid is None:
            return
        cx, cy = ctx.world_to_canvas(*centroid)
        font = skia.Font(skia.Typeface("Arial"), 14)
        label = ctx.track_name or f"Instance {ctx.instance_idx}"
        blob = skia.TextBlob(label, font)
        # Background
        bounds = font.measureText(label)
        bg = skia.Paint(Color=skia.Color4f(0, 0, 0, 0.6))
        ctx.canvas.drawRect(skia.Rect(cx - 2, cy - 18, cx + bounds + 2, cy - 4), bg)
        # Text
        paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        ctx.canvas.drawTextBlob(blob, cx, cy - 6, paint)

    img = sio.render_image(lf, per_instance_callback=draw_labels)
    save_image(img, "callback_labels.png")

    # Bounding boxes
    def draw_bbox(ctx: InstanceContext):
        bbox = ctx.get_bbox()
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        x1, y1 = ctx.world_to_canvas(x1, y1)
        x2, y2 = ctx.world_to_canvas(x2, y2)
        pad = 8
        rect = skia.Rect(x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        dash = skia.DashPathEffect.Make([6, 3], 0)
        paint = skia.Paint(
            Color=skia.ColorWHITE,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=2,
            PathEffect=dash,
        )
        ctx.canvas.drawRect(rect, paint)

    img = sio.render_image(lf, per_instance_callback=draw_bbox)
    save_image(img, "callback_bbox.png")

    # Frame info
    def draw_frame_info(ctx: RenderContext):
        font = skia.Font(skia.Typeface("Arial"), 14)
        text = f"Frame: {ctx.frame_idx}  Instances: {len(ctx.instances)}"
        blob = skia.TextBlob(text, font)
        bg = skia.Paint(Color=skia.Color4f(0, 0, 0, 0.7))
        ctx.canvas.drawRect(skia.Rect(4, 4, 200, 24), bg)
        paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        ctx.canvas.drawTextBlob(blob, 8, 18, paint)

    img = sio.render_image(lf, post_render_callback=draw_frame_info)
    save_image(img, "callback_frame_info.png")

    # Combined callbacks
    def combined_per_instance(ctx: InstanceContext):
        draw_bbox(ctx)
        draw_labels(ctx)

    img = sio.render_image(
        lf,
        per_instance_callback=combined_per_instance,
        post_render_callback=draw_frame_info,
    )
    save_image(img, "callback_combined.png")

    # =========================================================================
    # Montage example (with full code shown in docs)
    # =========================================================================
    print("\nMontage example...")

    # Render multiple frames and concatenate horizontally
    frame_indices = [0, 100, 200, 300, 400]
    frames = []
    for i in frame_indices:
        if i < len(labels.labeled_frames):
            img = sio.render_image(labels.labeled_frames[i], color_by="track")
            frames.append(img)
    montage = np.concatenate(frames, axis=1)
    save_image(montage, "montage.png", max_width=900)

    # Summary
    n_images = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n✅ Generated {n_images} images in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
