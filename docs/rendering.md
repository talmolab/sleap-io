# Rendering

sleap-io provides high-performance pose visualization using [skia-python](https://kyamagu.github.io/skia-python/), a production-quality 2D graphics library. This guide covers all rendering capabilities from quick one-liners to advanced custom overlays.

!!! info "Required dependencies"
    Rendering requires optional dependencies:
    ```bash
    pip install sleap-io[all]        # All optional deps
    pip install sleap-io[rendering]  # Minimal rendering deps only
    ```

## Quick Start

### One-liner video rendering

```python
import sleap_io as sio

# Load and render in one line
sio.load_slp("predictions.slp").render("output.mp4")
```

### Render a single frame

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Get rendered array (H, W, 3) uint8
img = sio.render_image(labels.labeled_frames[0])

# Or save directly to file
sio.render_image(labels.labeled_frames[0], "frame.png")
```

### CLI rendering

```bash
# Render full video
sio render -i predictions.slp -o output.mp4

# Fast preview (0.25x resolution)
sio render -i predictions.slp --preset preview

# Single frame to PNG
sio render -i predictions.slp --lf 0
```

---

## API Overview

### Main Functions

| Function | Description |
|----------|-------------|
| [`sio.render_video()`](#render-video) | Render video with pose overlays |
| [`sio.render_image()`](#render-image) | Render single frame |
| [`Labels.render()`](#labelsrender) | Convenience method on Labels |

### Color Options

| Option | Description |
|--------|-------------|
| `color_by` | Scheme: `"auto"`, `"track"`, `"instance"`, `"node"` |
| `palette` | Color palette name |

### Appearance Options

| Option | Default | Description |
|--------|---------|-------------|
| `marker_shape` | `"circle"` | Node marker shape |
| `marker_size` | `4.0` | Node radius in pixels |
| `line_width` | `2.0` | Edge line width |
| `alpha` | `1.0` | Transparency (0.0-1.0) |
| `show_nodes` | `True` | Draw node markers |
| `show_edges` | `True` | Draw skeleton edges |

---

## Rendering Patterns

### Pattern 1: In-memory rendering

Render frames to numpy arrays for display, further processing, or custom pipelines.

```python
import sleap_io as sio
import matplotlib.pyplot as plt

labels = sio.load_slp("predictions.slp")

# Render single frame to array
img = sio.render_image(labels.labeled_frames[0])
print(img.shape)  # (H, W, 3) uint8

# Display with matplotlib
plt.imshow(img)
plt.axis("off")
plt.savefig("figure.png", dpi=300, bbox_inches="tight")
```

### Pattern 2: Render to disk

Save rendered output directly to image or video files.

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Save single frame to PNG
sio.render_image(labels, lf_ind=0, save_path="frame_0.png")

# Or use LabeledFrame directly
sio.render_image(labels.labeled_frames[42], "frame_42.png")

# Render video to MP4
sio.render_video(labels, "output.mp4")
```

### Pattern 3: Render specific video frame

Render by video frame index rather than labeled frame index.

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Render video frame 100 from video 0
sio.render_image(labels, video=0, frame_idx=100, save_path="frame_100.png")

# Or get as array
img = sio.render_image(labels, video=0, frame_idx=100)
```

### Pattern 4: Batch render multiple frames

Render many frames efficiently for creating figures or montages.

```python
import sleap_io as sio
import numpy as np

labels = sio.load_slp("predictions.slp")

# Render frames 0, 10, 20, ... 90
frames = []
for i in range(0, 100, 10):
    if i < len(labels.labeled_frames):
        img = sio.render_image(labels, lf_ind=i)
        frames.append(img)

# Stack into array for montage
montage = np.concatenate(frames, axis=1)  # Horizontal strip
```

### Pattern 5: Video clip rendering

Render a subset of frames to video.

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Render frames 100-200
sio.render_video(labels, "clip.mp4", start=100, end=200)

# Or render specific frame indices
sio.render_video(labels, "selected.mp4", frame_inds=[0, 50, 100, 150])
```

### Pattern 6: Include unlabeled frames

Render all frames in a range, not just those with predictions.

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")

# Render all video frames, showing blank poses where no predictions exist
sio.render_video(labels, "full_video.mp4", include_unlabeled=True)

# Combine with frame range
sio.render_video(labels, "clip.mp4", start=0, end=500, include_unlabeled=True)
```

---

## Color Schemes

Color scheme determines how poses are colored across instances and frames.

### Auto mode (default)

```python
# Auto-selects the best scheme based on data
sio.render_video(labels, "output.mp4", color_by="auto")
```

Auto mode uses these rules:

- If tracks available → color by track (consistent identity across frames)
- If single image → color by instance (distinguishes animals)
- If video without tracks → color by node (prevents flicker)

### Color by track

Each tracked animal gets a consistent color across all frames.

```python
sio.render_video(labels, "by_track.mp4", color_by="track")
```

**Best for:** tracked data where you want to follow specific animals.

### Color by instance

Each animal within a frame gets a unique color, but colors may change between frames.

```python
sio.render_video(labels, "by_instance.mp4", color_by="instance")
```

**Best for:** single-frame renders or untracked data.

### Color by node

Each body part gets a unique color (same for all animals).

```python
sio.render_video(labels, "by_node.mp4", color_by="node")
```

**Best for:** highlighting skeleton structure or comparing body parts.

---

## Color Palettes

### Built-in palettes

8 palettes are included with no additional dependencies:

| Palette | Description | Best for |
|---------|-------------|----------|
| `distinct` | High-contrast colors | Instances/tracks |
| `rainbow` | Spectrum colors | Node types |
| `warm` | Orange/red tones | Single-animal |
| `cool` | Blue/purple tones | Single-animal |
| `pastel` | Subtle colors | Overlays on busy backgrounds |
| `seaborn` | Professional look | Publications |
| `tableau10` | Data viz standard | Publications |
| `viridis` | Perceptually uniform | Scientific |

```python
# Use a built-in palette
sio.render_video(labels, "output.mp4", palette="tableau10")
sio.render_video(labels, "output.mp4", palette="rainbow")
```

### Colorcet palettes (included with `[all]`)

With the `[all]` extra, you get access to colorcet's expanded palettes:

| Palette | Colors | Description |
|---------|--------|-------------|
| `glasbey` | 256 | Maximally distinct colors |
| `glasbey_hv` | 256 | High visibility variant |
| `glasbey_cool` | 256 | Cool-toned variant |
| `glasbey_warm` | 256 | Warm-toned variant |

```python
# Great for many tracked instances
sio.render_video(labels, "output.mp4", palette="glasbey")

# High visibility for dark backgrounds
sio.render_video(labels, "output.mp4", palette="glasbey_hv")
```

### Getting palette colors programmatically

```python
from sleap_io.rendering import get_palette

# Get 10 colors from a palette
colors = get_palette("tableau10", 10)
# Returns: [(31, 119, 180), (255, 127, 14), ...]

# Palettes cycle if you request more colors than they have
colors = get_palette("distinct", 20)  # Repeats the 10 distinct colors
```

---

## Marker Shapes

Five marker shapes are available for node visualization:

| Shape | Description |
|-------|-------------|
| `circle` | Filled circle (default) |
| `square` | Filled square |
| `diamond` | Rotated square |
| `triangle` | Upward-pointing triangle |
| `cross` | Plus sign |

```python
# Different marker shapes
sio.render_video(labels, "circles.mp4", marker_shape="circle")
sio.render_video(labels, "squares.mp4", marker_shape="square")
sio.render_video(labels, "diamonds.mp4", marker_shape="diamond")
```

---

## Styling Options

### Adjust marker and line sizes

```python
sio.render_video(
    labels,
    "styled.mp4",
    marker_size=6.0,    # Larger nodes
    line_width=3.0,     # Thicker edges
)
```

### Transparency

```python
# Semi-transparent overlay
sio.render_video(labels, "translucent.mp4", alpha=0.7)

# Very subtle overlay
sio.render_video(labels, "subtle.mp4", alpha=0.3)
```

### Toggle elements

```python
# Edges only (no node markers)
sio.render_video(labels, "edges_only.mp4", show_nodes=False)

# Nodes only (no skeleton edges)
sio.render_video(labels, "nodes_only.mp4", show_edges=False)
```

---

## Quality and Performance

### Quality presets

Three presets balance speed vs quality:

| Preset | Scale | Use case |
|--------|-------|----------|
| `preview` | 0.25x | Quick checks, scrubbing |
| `draft` | 0.5x | Review, iteration |
| `final` | 1.0x | Publication, sharing |

```python
# Fast preview for checking predictions
sio.render_video(labels, "preview.mp4", preset="preview")

# Full quality for publication
sio.render_video(labels, "final.mp4", preset="final")

# Custom scale
sio.render_video(labels, "output.mp4", scale=0.75)
```

### Performance characteristics

Typical rendering speeds (measured on 1024x1024 frames with 13 nodes):

| Operation | Speed | Notes |
|-----------|-------|-------|
| Skia rendering | ~93 FPS | Vector graphics are fast |
| Video I/O | ~5 FPS | Bottleneck is video decoding |
| Preview (0.25x) | ~15 FPS | Reduced resolution helps |

**Tips for faster rendering:**

1. Use `preset="preview"` for quick checks
2. Use `start`/`end` to render clips instead of full videos
3. For maximum speed, pre-extract frames to image sequences

### Video encoding options

Control output quality and file size:

```python
sio.render_video(
    labels,
    "output.mp4",
    fps=30.0,           # Output frame rate
    crf=18,             # Quality (2-32, lower=better, default 25)
    x264_preset="slow", # Encoding speed (ultrafast to slow)
)
```

---

## Custom Rendering with Callbacks

Callbacks let you add custom graphics like labels, bounding boxes, or annotations. You get direct access to the Skia canvas for arbitrary drawing.

### Callback types

| Callback | When called | Use case |
|----------|-------------|----------|
| `pre_render_callback` | Before poses drawn | Background layers, grids, ROIs |
| `post_render_callback` | After all poses | Frame info, legends, watermarks |
| `per_instance_callback` | After each instance | Labels, bounding boxes, badges |

### Context objects

Callbacks receive context objects with useful information:

**`RenderContext`** (frame-level):
```python
ctx.canvas          # Skia canvas for drawing
ctx.frame_idx       # Current frame index
ctx.frame_size      # (width, height) tuple
ctx.instances       # List of point arrays
ctx.skeleton_edges  # Edge connectivity
ctx.node_names      # Node name list
ctx.scale           # Current scale factor
ctx.offset          # Current offset (x, y)
ctx.world_to_canvas(x, y)  # Coordinate transform
```

**`InstanceContext`** (instance-level):
```python
ctx.canvas          # Skia canvas for drawing
ctx.instance_idx    # Index in frame
ctx.points          # (n_nodes, 2) array
ctx.track_name      # Track name string
ctx.confidence      # Instance confidence score
ctx.get_centroid()  # Mean of valid points
ctx.get_bbox()      # (x1, y1, x2, y2) bounding box
ctx.world_to_canvas(x, y)  # Coordinate transform
```

### Example: Instance labels

Draw track names above each animal:

```python
import sleap_io as sio
from sleap_io.rendering import InstanceContext
import skia

def draw_instance_labels(ctx: InstanceContext):
    """Draw track name above each instance."""
    centroid = ctx.get_centroid()
    if centroid is None:
        return

    cx, cy = ctx.world_to_canvas(centroid[0], centroid[1])

    # Create text
    font = skia.Font(skia.Typeface("Arial"), 14 * ctx.scale)
    label = ctx.track_name or f"Instance {ctx.instance_idx}"
    text_blob = skia.TextBlob(label, font)

    # Draw with white fill
    paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
    ctx.canvas.drawTextBlob(text_blob, cx - 20, cy - 20 * ctx.scale, paint)

labels = sio.load_slp("predictions.slp")
sio.render_video(labels, "labeled.mp4", per_instance_callback=draw_instance_labels)
```

### Example: Bounding boxes

Draw dashed bounding boxes around instances:

```python
import sleap_io as sio
from sleap_io.rendering import InstanceContext
import skia

def draw_bounding_box(ctx: InstanceContext):
    """Draw dashed bounding box around each instance."""
    bbox = ctx.get_bbox()
    if bbox is None:
        return

    x1, y1, x2, y2 = bbox

    # Transform to canvas coordinates
    x1, y1 = ctx.world_to_canvas(x1, y1)
    x2, y2 = ctx.world_to_canvas(x2, y2)

    # Add padding
    pad = 10 * ctx.scale
    rect = skia.Rect(x1 - pad, y1 - pad, x2 + pad, y2 + pad)

    # Create dashed stroke
    dash_effect = skia.DashPathEffect.Make([8, 4], 0)
    paint = skia.Paint(
        Color=skia.ColorWHITE,
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=2 * ctx.scale,
        PathEffect=dash_effect,
    )

    ctx.canvas.drawRect(rect, paint)

labels = sio.load_slp("predictions.slp")
sio.render_video(labels, "boxes.mp4", per_instance_callback=draw_bounding_box)
```

### Example: Frame information overlay

Add frame number and instance count:

```python
import sleap_io as sio
from sleap_io.rendering import RenderContext
import skia

def draw_frame_info(ctx: RenderContext):
    """Draw frame info in corner."""
    font = skia.Font(skia.Typeface("Arial"), 16)
    text = f"Frame: {ctx.frame_idx} | Instances: {len(ctx.instances)}"
    text_blob = skia.TextBlob(text, font)

    # Background rectangle
    bg_paint = skia.Paint(Color=skia.Color4f(0, 0, 0, 0.5))
    ctx.canvas.drawRect(skia.Rect(5, 5, 250, 30), bg_paint)

    # Text
    text_paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
    ctx.canvas.drawTextBlob(text_blob, 10, 24, text_paint)

labels = sio.load_slp("predictions.slp")
sio.render_video(labels, "info.mp4", post_render_callback=draw_frame_info)
```

### Combining multiple callbacks

```python
def combined_instance_callback(ctx):
    """Draw both label and bounding box."""
    draw_bounding_box(ctx)
    draw_instance_labels(ctx)

sio.render_video(
    labels,
    "annotated.mp4",
    pre_render_callback=draw_grid,           # Background layer
    post_render_callback=draw_frame_info,    # Overlay layer
    per_instance_callback=combined_instance_callback,
)
```

### Skia drawing reference

Common Skia operations available in callbacks:

| Feature | Method |
|---------|--------|
| Text | `canvas.drawTextBlob(blob, x, y, paint)` |
| Circles | `canvas.drawCircle(x, y, radius, paint)` |
| Lines | `canvas.drawLine(x1, y1, x2, y2, paint)` |
| Rectangles | `canvas.drawRect(rect, paint)` |
| Rounded rects | `canvas.drawRoundRect(rect, rx, ry, paint)` |
| Paths/Polygons | `canvas.drawPath(path, paint)` |
| Images | `canvas.drawImage(image, x, y)` |

---

## Handling Missing Videos

When video files are unavailable, you can still render with fallback options:

### Fallback solid color background

```python
# Render with gray background if video is missing
sio.render_video(
    labels,
    "output.mp4",
    require_video=False,
    fallback_color=(128, 128, 128),  # Gray RGB
)

# Single frame with fallback
sio.render_image(
    labels.labeled_frames[0],
    "frame.png",
    require_video=False,
    fallback_color=(0, 0, 0),  # Black
)
```

### Custom background image

```python
import numpy as np
import sleap_io as sio

labels = sio.load_slp("predictions.slp")
lf = labels.labeled_frames[0]

# Create custom background (or load from file)
bg = np.zeros((1024, 1024, 3), dtype=np.uint8)
bg[:] = (30, 30, 30)  # Dark gray

# Render with custom background
img = sio.render_image(lf, image=bg)
```

---

## CLI Reference

Full CLI documentation is in the [CLI Guide](cli.md#sio-render---render-pose-videos-and-images). Quick reference:

```bash
# Basic rendering
sio render -i predictions.slp -o output.mp4

# Fast preview
sio render -i predictions.slp --preset preview

# Single frame
sio render -i predictions.slp --lf 0 -o frame.png

# Custom styling
sio render -i predictions.slp -o styled.mp4 \
    --color-by track \
    --palette tableau10 \
    --marker-shape diamond \
    --marker-size 6

# Render a clip
sio render -i predictions.slp -o clip.mp4 --start 100 --end 200
```

---

## API Reference

::: sleap_io.render_video
    options:
      show_root_heading: true
      heading_level: 3

::: sleap_io.render_image
    options:
      show_root_heading: true
      heading_level: 3

::: sleap_io.rendering.get_palette
    options:
      show_root_heading: true
      heading_level: 3

::: sleap_io.rendering.RenderContext
    options:
      show_root_heading: true
      heading_level: 3

::: sleap_io.rendering.InstanceContext
    options:
      show_root_heading: true
      heading_level: 3
