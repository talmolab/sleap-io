# Transforms

sleap-io provides coordinate-aware video transformations that automatically adjust landmark coordinates to maintain alignment. Apply geometric operations like crop, scale, rotate, pad, and flip to your pose tracking data.

---

## Quick Start

Transform a labels file from the command line:

```bash
sio transform labels.slp --scale 0.5 -o scaled.slp
```

Or use the Python API:

```python
import sleap_io as sio

labels = sio.load_slp("labels.slp")
transform = sio.Transform(scale=(0.5, 0.5))
transformed = sio.transform_labels(labels, transform, "scaled.slp")
```

Both approaches automatically:

- Transform all video frames
- Adjust all landmark coordinates using affine transformations
- Preserve alignment between poses and video

![Original frame](assets/transforms/original.png)

---

## How SLP Files Are Processed

When you transform a `.slp` file, the behavior depends on whether the videos are external references (MediaVideo) or embedded within the file.

### External videos (MediaVideo)

For standard `.slp` files that reference external video files:

- **Transformed videos** are saved as `.mp4` files in a new directory: `{output_name}.videos/`
- **Video naming**: Each video is saved as `{original_name}.transformed.mp4`
- **Overwrite mode**: If the output directory exists, videos with the same name are overwritten

```
input.slp                          # References video.mp4
    ↓
output.slp                         # References output.videos/video.transformed.mp4
output.videos/
    video.transformed.mp4          # Transformed video frames
```

Example:

```bash
sio transform predictions.slp --scale 0.5 -o scaled.slp
# Creates:
#   scaled.slp
#   scaled.videos/original_video.transformed.mp4
```

### Embedded videos (`.pkg.slp`)

For package files with embedded video frames:

- **Transformed frames** are embedded directly in the output `.slp` file
- **No separate video files** are created
- **File size**: Output may be larger due to re-encoded frames

```
input.pkg.slp                      # Contains embedded frames
    ↓
output.pkg.slp                     # Contains transformed embedded frames
```

Example:

```bash
sio transform predictions.pkg.slp --scale 0.5 -o scaled.pkg.slp
# Creates:
#   scaled.pkg.slp (with embedded transformed frames)
```

### Custom video output directory

You can specify where transformed videos are saved:

=== "Python"
    ```python
    import sleap_io as sio
    from pathlib import Path

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(scale=(0.5, 0.5))
    sio.transform_labels(
        labels,
        transform,
        Path("output.slp"),
        video_output_dir=Path("custom_videos/"),
    )
    ```

---

## Scale

Resize videos and coordinates uniformly or to specific dimensions.

### Uniform scale (ratio)

Scale by a ratio to shrink or enlarge:

=== "CLI"
    ```bash
    sio transform labels.slp --scale 0.5 -o scaled.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(scale=(0.5, 0.5))
    sio.transform_labels(labels, transform, "scaled.slp")
    ```

![Scale 50%](assets/transforms/scale_half.png)

### Target width

Specify a pixel width and auto-compute height to preserve aspect ratio:

=== "CLI"
    ```bash
    sio transform labels.slp --scale 640 -o scaled.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    # Compute scale factor from target width (preserving aspect ratio)
    input_width = labels.videos[0].shape[2]  # Get original width
    target_width = 640
    scale_factor = target_width / input_width
    transform = sio.Transform(scale=(scale_factor, scale_factor))
    sio.transform_labels(labels, transform, "scaled.slp")
    ```

![Scale to width 640](assets/transforms/scale_width.png)

### Exact dimensions

Specify exact output dimensions (may change aspect ratio):

=== "CLI"
    ```bash
    sio transform labels.slp --scale 800,600 -o scaled.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    # Compute scale factors from target dimensions
    h, w = labels.videos[0].shape[1:3]
    target_w, target_h = 800, 600
    transform = sio.Transform(scale=(target_w / w, target_h / h))
    sio.transform_labels(labels, transform, "scaled.slp")
    ```

![Scale to 800x600](assets/transforms/scale_dimensions.png)

### Scale format reference

| Format | Example | Result |
|--------|---------|--------|
| Ratio | `0.5` | 50% size |
| Width | `640` | Width=640, height auto |
| Height | `-1,480` | Width auto, height=480 |
| Exact | `640,480` | Exact 640x480 |
| Per-axis | `0.5,0.75` | 50% width, 75% height |

---

## Crop

Extract a rectangular region of interest.

### Pixel coordinates

Specify `(x1, y1, x2, y2)` in pixels:

- **0-based indexing**: Pixel (0, 0) is the top-left pixel of the image
- **Center pixel alignment**: Coordinates refer to the center of each pixel (SLEAP uses center pixel indexing)
- **Exclusive end**: The region includes pixels from (x1, y1) up to but not including (x2, y2), similar to Python slicing

For example, `--crop 128,256,640,768` extracts a 512×512 region starting at pixel (128, 256):

=== "CLI"
    ```bash
    sio transform labels.slp --crop 128,256,640,768 -o cropped.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    # Crop region: x from 128 to 640, y from 256 to 768
    transform = sio.Transform(crop=(128, 256, 640, 768))
    sio.transform_labels(labels, transform, "cropped.slp")
    ```

![Crop region](assets/transforms/crop_pixel.png)

### Normalized coordinates

Use values in `[0.0, 1.0]` for resolution-independent crops:

=== "CLI"
    ```bash
    sio transform labels.slp --crop 0.25,0.25,0.75,0.75 -o cropped.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    # Convert normalized to pixel coordinates
    h, w = labels.videos[0].shape[1:3]
    x1, y1, x2, y2 = int(0.25 * w), int(0.25 * h), int(0.75 * w), int(0.75 * h)
    transform = sio.Transform(crop=(x1, y1, x2, y2))
    sio.transform_labels(labels, transform, "cropped.slp")
    ```

![Crop with normalized coordinates](assets/transforms/crop_normalized.png)

### Crop and zoom

Combine crop with scale to zoom into a region:

=== "CLI"
    ```bash
    sio transform labels.slp --crop 256,256,768,768 --scale 2.0 -o zoomed.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(crop=(256, 256, 768, 768), scale=(2.0, 2.0))
    sio.transform_labels(labels, transform, "zoomed.slp")
    ```

![Crop and zoom](assets/transforms/crop_zoom.png)

---

## Rotate

Rotate frames around the center point.

### Cardinal rotations

Rotate by 90, 180, or 270 degrees:

=== "CLI"
    ```bash
    sio transform labels.slp --rotate 90 -o rotated.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(rotate=90)
    sio.transform_labels(labels, transform, "rotated.slp")
    ```

![Rotate 90 degrees](assets/transforms/rotate_90.png)

### Arbitrary angles with expansion

By default, the canvas expands to fit the rotated content:

=== "CLI"
    ```bash
    sio transform labels.slp --rotate 45 -o rotated.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(rotate=45, clip_rotation=False)  # Default
    sio.transform_labels(labels, transform, "rotated.slp")
    ```

![Rotate 45 degrees expanded](assets/transforms/rotate_45_expand.png)

### Clipped rotation

Keep original dimensions by clipping corners:

=== "CLI"
    ```bash
    sio transform labels.slp --rotate 45 --clip-rotation -o rotated.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(rotate=45, clip_rotation=True)
    sio.transform_labels(labels, transform, "rotated.slp")
    ```

![Rotate 45 degrees clipped](assets/transforms/rotate_45_clip.png)

---

## Pad

Add borders around the frame.

### Uniform padding

Add equal padding on all sides:

=== "CLI"
    ```bash
    sio transform labels.slp --pad 50 -o padded.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(pad=(50, 50, 50, 50))
    sio.transform_labels(labels, transform, "padded.slp")
    ```

![Uniform padding](assets/transforms/pad_uniform.png)

### Asymmetric padding

Specify `(top, right, bottom, left)` padding:

=== "CLI"
    ```bash
    sio transform labels.slp --pad 100,50,100,50 -o padded.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(pad=(100, 50, 100, 50))
    sio.transform_labels(labels, transform, "padded.slp")
    ```

![Asymmetric padding](assets/transforms/pad_asymmetric.png)

### Custom fill color

Use `--fill` to set the padding color:

=== "CLI"
    ```bash
    # Grayscale (0-255)
    sio transform labels.slp --pad 50 --fill 128 -o padded.slp

    # RGB for color videos
    sio transform labels.slp --pad 50 --fill 255,128,0 -o padded.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")

    # Grayscale fill
    transform = sio.Transform(pad=(50, 50, 50, 50), fill=128)
    sio.transform_labels(labels, transform, "padded_gray.slp")

    # RGB fill (for color videos)
    transform = sio.Transform(pad=(50, 50, 50, 50), fill=(255, 128, 0))
    sio.transform_labels(labels, transform, "padded_orange.slp")
    ```

![Gray fill padding](assets/transforms/pad_fill_gray.png)

![Light fill padding](assets/transforms/pad_fill_light.png)

---

## Flip

Mirror the frame horizontally or vertically.

### Horizontal flip

Mirror left-right:

=== "CLI"
    ```bash
    sio transform labels.slp --flip-horizontal -o flipped.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(flip_h=True)
    sio.transform_labels(labels, transform, "flipped.slp")
    ```

![Horizontal flip](assets/transforms/flip_horizontal.png)

### Vertical flip

Mirror top-bottom:

=== "CLI"
    ```bash
    sio transform labels.slp --flip-vertical -o flipped.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(flip_v=True)
    sio.transform_labels(labels, transform, "flipped.slp")
    ```

![Vertical flip](assets/transforms/flip_vertical.png)

### Both flips

Equivalent to 180° rotation:

=== "CLI"
    ```bash
    sio transform labels.slp --flip-horizontal --flip-vertical -o flipped.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(flip_h=True, flip_v=True)
    sio.transform_labels(labels, transform, "flipped.slp")
    ```

![Both flips](assets/transforms/flip_both.png)

---

## Transform Pipeline

Transforms are always applied in a fixed order:

**crop → scale → rotate → pad → flip**

This ensures predictable results when combining operations:

1. **Crop** extracts a region from the original frame
2. **Scale** resizes the cropped region
3. **Rotate** rotates around the frame center
4. **Pad** adds borders to the result
5. **Flip** mirrors the final image

### Combined example: Crop + Scale

=== "CLI"
    ```bash
    sio transform labels.slp \
        --crop 200,200,800,800 \
        --scale 0.5 \
        -o zoomed.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(
        crop=(200, 200, 800, 800),
        scale=(0.5, 0.5),
    )
    sio.transform_labels(labels, transform, "zoomed.slp")
    ```

![Crop and scale combined](assets/transforms/combined_crop_scale.png)

### Combined example: Scale + Pad + Flip

=== "CLI"
    ```bash
    sio transform labels.slp \
        --scale 0.5 \
        --pad 50 \
        --flip-horizontal \
        -o processed.slp
    ```

=== "Python"
    ```python
    import sleap_io as sio

    labels = sio.load_slp("labels.slp")
    transform = sio.Transform(
        scale=(0.5, 0.5),
        pad=(50, 50, 50, 50),
        flip_h=True,
    )
    sio.transform_labels(labels, transform, "processed.slp")
    ```

![Scale, pad, and flip combined](assets/transforms/combined_scale_pad_flip.png)

---

## Multi-Video Projects

For labels files with multiple videos, you can apply different transforms to each.

### Uniform parameters

Apply the same transform to all videos:

```bash
sio transform multi_cam.slp --scale 0.5 -o scaled.slp
```

### Per-video parameters

Use the `idx:` prefix to target specific videos:

```bash
sio transform multi_cam.slp \
    --crop 0:100,100,500,500 \
    --crop 1:200,200,600,600 \
    --scale 0.5 \
    -o processed.slp
```

### Config file

For complex multi-video scenarios, use a YAML config file:

```bash
sio transform multi_cam.slp --config transforms.yaml -o output.slp
```

---

## Config File Format

The config file specifies transforms per video index.

### Basic structure

```yaml
# transforms.yaml
videos:
  0:
    crop: [100, 100, 500, 500]
    scale: 0.5
    rotate: 0
    pad: [0, 0, 0, 0]
  1:
    crop: [200, 200, 600, 600]
    scale: [640, -1]  # Width=640, height auto
    rotate: 90
```

### Available options per video

| Key | Type | Description |
|-----|------|-------------|
| `crop` | `[x1, y1, x2, y2]` | Crop region (pixels or normalized) |
| `scale` | `float` or `[w, h]` | Scale factor or dimensions |
| `rotate` | `float` | Rotation angle in degrees |
| `pad` | `[top, right, bottom, left]` or `int` | Padding in pixels |
| `flip_horizontal` | `bool` | Mirror horizontally |
| `flip_vertical` | `bool` | Mirror vertically |
| `clip_rotation` | `bool` | Clip rotation to original dimensions |

### Example: Same transform for all videos

Currently, you must repeat the config for each video:

```yaml
videos:
  0:
    scale: 0.5
  1:
    scale: 0.5
  2:
    scale: 0.5
```

### Example: Different transforms per camera

```yaml
# Multi-camera alignment
videos:
  0:  # Top-down camera
    crop: [200, 200, 800, 800]
    scale: [640, 640]
  1:  # Side camera (needs rotation)
    rotate: 90
    crop: [100, 100, 500, 500]
    scale: [400, 400]
  2:  # Mirror camera
    flip_horizontal: true
    scale: 0.5
```

### Precedence

When combining CLI options with a config file:

**config file < uniform CLI options < indexed CLI options**

Indexed options (e.g., `--crop 0:...`) have the highest priority.

---

## Preview Mode

Preview transforms without processing using `--dry-run`:

```bash
sio transform labels.slp --scale 0.5 --dry-run
```

Output shows the transform summary:

```
Loading SLP: labels.slp
  Found 1 video(s)

Transform Summary:

  Video 0: video.mp4
    Size: 1024x1024 -> 512x512
    Scale: (0.5, 0.5)

Dry run - would save SLP to: labels.transformed.slp
```

### Preview a specific frame

Render a preview image with `--dry-run-frame`:

```bash
sio transform labels.slp --scale 0.5 --rotate 45 --dry-run-frame 0
```

This saves a preview PNG to `/tmp/sio_preview.png` showing the transformed frame.

---

## Metadata & Provenance

By default, transform metadata is automatically embedded in the output SLP file. This preserves a record of how the data was transformed for reproducibility and debugging.

### Embedded provenance format

The embedded metadata is stored in `labels.provenance["transform"]` and includes:

```yaml
generated: "2026-01-12T20:30:00+00:00"   # When the transform was applied
source: "/path/to/input.slp"              # Path to the source file
output: "/path/to/output.slp"             # Path to the output file
sleap_io_version: "0.3.0"                 # Version used
videos:
  0:                                      # Per-video transform details
    input: "/path/to/video.mp4"           # Source video path
    input_size: [1024, 1024]              # Original dimensions [width, height]
    output_size: [512, 512]               # Transformed dimensions
    coordinate_transform:
      matrix: [[...]]                     # 3x3 affine transformation matrix
    transforms:
      crop: [256, 256, 768, 768]          # Applied crop (or null)
      scale: [0.5, 0.5]                   # Applied scale (or null)
      rotate: null                        # Applied rotation (or null)
      pad: null                           # Applied padding (or null)
      flip_horizontal: false
      flip_vertical: false
      clip_rotation: false
```

Access the embedded metadata:

```python
import sleap_io as sio

labels = sio.load_slp("output.slp")
transform_info = labels.provenance.get("transform")

# Get source file path
source_path = transform_info["source"]

# Get transform matrix for coordinate conversion
matrix = transform_info["videos"]["0"]["coordinate_transform"]["matrix"]
```

To disable provenance embedding:

```bash
sio transform labels.slp --scale 0.5 --no-embed-provenance -o output.slp
```

### Export transform metadata

Save transform details to a standalone YAML file:

```bash
sio transform labels.slp --scale 0.5 \
    --output-transforms transforms_meta.yaml \
    -o output.slp
```

This creates a YAML file with the same format as the embedded provenance, useful for external tracking or when processing multiple files.

---

## Video Encoding Options

Control output video quality and format:

| Option | Default | Description |
|--------|---------|-------------|
| `--crf` | `25` | Quality (0-51, lower = better) |
| `--x264-preset` | `superfast` | Encoding speed vs compression |
| `--fps` | (source) | Output frame rate |
| `--keyframe-interval` | (none) | Seconds between keyframes |
| `--no-audio` | off | Strip audio from output |

Example for high-quality output with reliable seeking:

```bash
sio transform labels.slp --scale 0.5 \
    --crf 18 \
    --x264-preset slow \
    --keyframe-interval 0.5 \
    -o output.slp
```

---

## Coordinate Transformation

All landmark coordinates are automatically adjusted using affine transformation matrices.

| Transform | Coordinate Adjustment |
|-----------|----------------------|
| **Crop** | `new = old - offset` |
| **Scale** | `new = old * factor` |
| **Rotate** | Affine rotation around image center `((w-1)/2, (h-1)/2)` |
| **Pad** | `new = old + padding_offset` |
| **Flip H** | `new_x = (width - 1) - old_x` |
| **Flip V** | `new_y = (height - 1) - old_y` |

### Access the transformation matrix

```python
import sleap_io as sio

transform = sio.Transform(scale=(0.5, 0.5), rotate=45)
matrix = transform.to_matrix(input_size=(1024, 1024))
# Returns 3x3 affine transformation matrix
print(matrix)  # [[a, b, tx], [c, d, ty], [0, 0, 1]]
```

### Transform points manually

```python
import numpy as np
import sleap_io as sio

transform = sio.Transform(scale=(0.5, 0.5), rotate=45)
points = np.array([[100, 200], [300, 400]])
transformed_points = transform.apply_to_points(points, input_size=(1024, 1024))
```

---

## Raw Video Mode

Transform standalone video files (without labels):

```bash
sio transform video.mp4 --scale 0.5 -o video_scaled.mp4
```

When transforming raw video:

- Output is always MP4 format
- No coordinate transformations (no landmarks)
- Same transform options available

---

## CLI Reference

For the complete CLI option reference, see the [CLI Guide](cli.md#sio-transform).

---

## API Reference

::: sleap_io.transform.Transform
    options:
      show_root_heading: true
      heading_level: 3

::: sleap_io.transform.transform_labels
    options:
      show_root_heading: true
      heading_level: 3

::: sleap_io.transform.transform_video
    options:
      show_root_heading: true
      heading_level: 3
