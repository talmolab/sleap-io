# Command-Line Interface

sleap-io provides a command-line interface (CLI) for inspecting and converting pose tracking data without writing Python code. The CLI is designed for quick exploration, scripting, and integration into data pipelines.

## Installation

### Quick Usage with `uvx` (No Installation)

Run CLI commands instantly using [`uvx`](https://docs.astral.sh/uv/) without installing:

```bash
# Inspect a labels file
uvx sleap-io show labels.slp

# Convert between formats
uvx sleap-io convert labels.slp -o labels.nwb

# Embed frames in output (video reading works out of the box)
uvx sleap-io convert -i labels.slp -o labels.pkg.slp --embed user

# Export to Ultralytics format
uvx sleap-io convert -i labels.slp -o dataset/ --to ultralytics

# Check version
uvx sleap-io --version
```

### Permanent Installation with `uv tool`

For regular use, install sleap-io as a global tool:

```bash
# Basic install (includes video support via imageio-ffmpeg)
uv tool install sleap-io

# Or with faster video backends (OpenCV, PyAV)
uv tool install "sleap-io[all]"

# Now use the short command
sio show labels.slp
sio convert labels.slp -o labels.nwb
```

!!! tip "The `sio` command"
    After installation with `uv tool install`, you can use either `sio` or `sleap-io` as the command name. The short `sio` form is recommended for convenience.

!!! info "When do you need the `[all]` extra?"
    Video support works out of the box via the bundled imageio-ffmpeg backend.
    The `[all]` extra installs faster video backends (OpenCV, PyAV) for improved performance.
    
    Use `[all]` when:

    - Processing many videos or large files (OpenCV is ~2-3x faster)
    - You need specific codec support from PyAV

## Quick Reference

```bash
# Get help
sio --help
sio show --help
sio convert --help
sio split --help
sio unsplit --help
sio merge --help
sio embed --help
sio unembed --help
sio filenames --help
sio fix --help
sio render --help
sio trim --help
sio reencode --help
sio transform --help

# Check version and installed plugins
sio --version

# Inspect a labels file
sio show labels.slp                    # Basic summary
sio show labels.slp --skeleton         # Detailed skeleton info
sio show labels.slp --video            # Detailed video info (all videos)
sio show labels.slp --vi 0             # Show specific video by index
sio show labels.slp --tracks           # Track details
sio show labels.slp --provenance       # Metadata/provenance
sio show labels.slp --all              # Everything
sio show labels.slp --lf 0             # Labeled frame details

# Inspect a video file directly
sio show video.mp4                     # Video properties and metadata

# Convert between formats
sio convert labels.slp -o labels.nwb
sio convert labels.slp -o labels.pkg.slp --embed user
sio convert data.json -o labels.slp --from coco
sio convert labels.slp -o dataset/ --to ultralytics

# Split into train/val/test sets
sio split labels.slp -o splits/                          # 80/20 train/val
sio split labels.slp -o splits/ --train 0.7 --test 0.15  # 70/15/15 split
sio split labels.slp -o splits/ --remove-predictions     # User labels only
sio split labels.slp -o splits/ --seed 42                # Reproducible split

# Merge split files back into one
sio unsplit train.slp val.slp -o merged.slp              # Merge individual files
sio unsplit splits/ -o merged.slp                        # Merge all .slp in directory

# Merge labels files with flexible options
sio merge project.slp predictions.slp -o merged.slp      # Merge predictions into project
sio merge base.slp new.slp -o merged.slp --frame replace_predictions
sio merge results/ -o combined.slp --frame keep_both     # All .slp in directory
sio merge local.slp remote.slp -o merged.slp --video basename

# Embed video frames into a portable package
sio embed labels.slp -o labels.pkg.slp                   # User-labeled frames only
sio embed labels.slp -o labels.pkg.slp --predictions     # Include prediction frames
sio embed labels.slp -o labels.pkg.slp --suggestions     # Include suggested frames

# Remove embedded frames and restore video references
sio unembed labels.pkg.slp -o labels.slp                 # Restore original video paths

# Inspect and update video filenames
sio filenames labels.slp                                   # List video paths
sio filenames labels.slp -o out.slp --filename /new/video.mp4
sio filenames labels.slp -o out.slp --map old.mp4 /new/video.mp4
sio filenames labels.slp -o out.slp --prefix /old/path /new/path

# Fix common issues in labels files
sio fix labels.slp                                         # Auto-fix with defaults
sio fix labels.slp --dry-run                               # Preview without changes
sio fix labels.slp --remove-predictions                    # Also remove predictions
sio fix labels.slp --remove-untracked-predictions          # Surgical pred removal
sio fix labels.slp --consolidate-skeletons                 # Force single skeleton
sio fix labels.slp --prefix "C:\data" /mnt/data            # Fix cross-platform paths

# Render video with pose overlays
sio render predictions.slp                                 # -> predictions.viz.mp4
sio render predictions.slp --preset preview                # Fast 0.25x preview
sio render predictions.slp --start 100 --end 200
sio render predictions.slp --lf 0                          # Single frame -> PNG
sio render predictions.slp --lf 0 --crop auto              # Auto-fit to instances
sio render predictions.slp --color-by track --marker-shape diamond

# Trim video and labels to a frame range
sio trim labels.slp --start 100 --end 1000                 # -> labels.trim.slp
sio trim labels.slp --start 100 --end 1000 -o clip.slp
sio trim video.mp4 --start 100 --end 500                   # Video-only mode

# Reencode video for reliable seeking (critical for annotation)
sio reencode video.mp4 -o video.seekable.mp4               # Fix seeking issues
sio reencode video.mp4 -o output.mp4 --quality high        # Higher quality
sio reencode video.mp4 -o output.mp4 --keyframe-interval 0.5  # Max reliability
sio reencode highspeed.mp4 -o preview.mp4 --fps 30 --quality low  # Downsample
sio reencode video.mp4 -o output.mp4 --dry-run             # Show ffmpeg command
sio reencode project.slp -o project.reencoded.slp          # Batch reencode all videos in SLP

# Transform video and adjust landmark coordinates
sio transform labels.slp --scale 0.5 -o scaled.slp         # Scale down 50%
sio transform labels.slp --crop 100,100,500,500 -o crop.slp  # Crop to region
sio transform labels.slp --rotate 90 -o rotated.slp        # Rotate 90 degrees
sio transform labels.slp --flip-horizontal -o flipped.slp  # Mirror horizontally
sio transform labels.slp --crop 100,100,500,500 --scale 2.0 -o zoomed.slp
sio transform multi_cam.slp --crop 0:100,100,500,500 -o cropped.slp  # Per-video
sio transform labels.slp --scale 0.5 --dry-run             # Preview transforms
sio transform labels.slp --scale 0.5 --dry-run-frame 0     # Preview specific frame
sio transform video.mp4 --scale 0.5 -o video_scaled.mp4    # Transform raw video
```

---

## `sio show`

Display information about a SLEAP labels file or video file with rich formatted output.

```bash
sio show <path> [options]
sio show -i <path> [options]
```

!!! tip "Input as positional or flag"
    All commands accept the input file as a positional argument or with `-i`/`--input`.
    The positional form (e.g., `sio show labels.slp`) is simpler; the flag form
    (e.g., `sio show -i labels.slp`) is more explicit.

### Basic Usage

```bash
# View file summary with skeleton and video info
sio show labels.slp

# Inspect a video file directly
sio show video.mp4
```

**Example output:**

```
╭─ sleap-io ─────────────────────────────────────────────────────────────╮
│ labels.slp                                                             │
│ /home/user/projects/mouse-tracking                                     │
│                                                                        │
│ Type:     Labels                                                       │
│ Size:     2.4 MB                                                       │
│                                                                        │
│ 1 video | 100 frames | 200 labeled | 2 tracks                          │
╰────────────────────────────────────────────────────────────────────────╯
  Full: /home/user/projects/mouse-tracking/labels.slp

Skeletons
  mouse (7 nodes, 6 edges)

  nodes = ["nose", "head", "neck", "body", "tail_base", "tail_mid", "tail_tip"]
  edge_inds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

Videos (1)

  [0] video.mp4  1920×1080  5000 frames

Tracks (2)
  mouse_1, mouse_2
```

The default view shows:

- **Header panel**: File name, path, type, size, and key statistics
- **Skeleton**: Node and edge definitions as copyable Python code
- **Videos**: Quick summary of video files and dimensions
- **Tracks**: List of track names

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input file (can also pass as positional argument) |
| `--skeleton` | `-s` | Show detailed skeleton tables (nodes, edges, symmetries) |
| `--video` | `-v` | Show detailed video info (opens backends by default) |
| `--video-index N` | `--vi N` | Show only video at index N (0-based). Implies --video |
| `--tracks` | `-t` | Show track table with instance counts per track |
| `--provenance` | `-p` | Show provenance/metadata from the file |
| `--all` | `-a` | Show all details (combines all flags above) |
| `--lf N` | | Show details for labeled frame at index N |
| `--open-videos` | | Force open video backends |
| `--no-open-videos` | | Don't open video backends (overrides -v default) |

!!! tip "Viewing a specific video"
    Use `--video-index N` (or `--vi N`) to show only a specific video by its index. For example, `sio show labels.slp --vi 0` shows only the first video, while `sio show labels.slp -v` shows all videos.

### Detailed Skeleton View

```bash
sio show labels.slp --skeleton
```

Shows tables for nodes, edges, and symmetries:

```
Skeleton 0: fly (24 nodes, 23 edges, 10 symmetries)

Python code:
  nodes = ["head", "neck", "thorax", "abdomen", ...]
  edge_inds = [(1, 0), (2, 1), (2, 3), ...]

Nodes:
   # | Name
  ---+----------
   0 | head
   1 | neck
   2 | thorax
   ...

Edges:
   # | Source   |    | Destination | Indices
  ---+----------+----+-------------+----------
   0 | neck     | -> | head        | (1, 0)
   1 | thorax   | -> | neck        | (2, 1)
   ...

Symmetries:
   # | Node A   |     | Node B
  ---+----------+-----+-----------
   0 | wingR    | <-> | wingL
   1 | legL1    | <-> | legR1
   ...
```

### Detailed Video View

```bash
sio show labels.slp --video
# Or show a specific video by index:
sio show labels.slp --video-index 0
# Short form:
sio show labels.slp --vi 0
```

Shows comprehensive video information including backend status:

```
Video 0: video.mp4

  Type      MediaVideo
  Path      video.mp4
  Full      /data/videos/video.mp4
  Status    File exists, backend not loaded

  Frames    5000
  Size      1920 × 1080 (RGB)
  Labeled   100 frames
```

The "Full" path is shown when it differs from the stored path, making it easy to copy absolute paths.

For embedded package files (`.pkg.slp`):

```
Video 0: video.mp4 [embedded]

  Type      HDF5Video (embedded)
  Source    original_video.mp4
  Dataset   video0/video
  Format    PNG (RGB)
  Status    Embedded, backend loaded

  Frames    50 (indices: 0–49)
  Size      1920 × 1080 (RGB)
  Labeled   50 frames

  Source Video
    File    original_video.mp4
    Type    MediaVideo
    Frames  5000
    Size    1920 × 1080 (RGB)
```

### Track Details

```bash
sio show labels.slp --tracks
```

Shows a table with instance counts per track:

```
Tracks

   # | Track    | Instances
  ---+----------+-----------
   0 | mouse_1  | 1847
   1 | mouse_2  | 1823
```

### Labeled Frame Details

```bash
sio show labels.slp --lf 0
```

Shows detailed information for a specific labeled frame:

```
Labeled Frame 0

  Video:     video.mp4
  Frame:     42
  Instances: 2

  Instance 0: user, track=mouse_1, visible=7/7
    points = [(256.32, 189.45), (245.18, 195.23), (230.45, 210.67), ...]

  Instance 1: predicted, track=mouse_2, score=0.94, visible=7/7
    points = [(512.18, 302.56), (498.32, 315.89), (475.23, 340.12), ...]
```

The `points` list is copyable Python code matching the skeleton node order.

### Provenance Information

```bash
sio show labels.slp --provenance
```

Shows metadata stored in the file:

```
Provenance

  sleap_version: 1.3.3
  model_paths: centroid.210505_..., centered_instance.210505_... (2 total)
  created_at: 2024-01-15T14:30:00
```

### Show Everything

```bash
sio show labels.slp --all
```

Combines `--skeleton`, `--video`, `--tracks`, and `--provenance` for a complete view.

### Standalone Video Files

You can also use `sio show` to inspect video files directly:

```bash
sio show video.mp4
```

**Example output:**

```
╭─ sleap-io ────────────────────────────────────────────────────────────╮
│ video.mp4                                                             │
│ /data/videos                                                          │
│                                                                       │
│ Type:     Video (MediaVideo)                                          │
│ Size:     934.3 KB                                                    │
│                                                                       │
│ 1100 frames | 384×384 | grayscale                                     │
╰───────────────────────────────────────────────────────────────────────╯

  Full      /data/videos/video.mp4
  Status    Backend loaded (opencv)
  Plugin    opencv
```

This is useful for quickly checking video properties without needing a labels file.

---

## `sio convert`

Convert pose data between different file formats.

```bash
sio convert <input> -o <output> [options]
sio convert -i <input> -o <output> [options]
```

### Basic Usage

```bash
# Convert SLEAP to NWB
sio convert labels.slp -o labels.nwb

# Convert SLEAP to COCO
sio convert labels.slp -o annotations.json --to coco

# Convert COCO to SLEAP
sio convert annotations.json -o labels.slp --from coco
```

### Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file path (can also pass as positional argument) |
| `-o, --output` | Output file path (required) |
| `--from` | Input format (required for `.json` and `.h5` files) |
| `--to` | Output format (inferred from extension if not specified) |
| `--embed` | Embed video frames in output (`user`, `all`, `suggestions`, `source`) |
| `--csv-format` | CSV output format: `sleap`, `dlc`, `points`, `instances`, `frames` |
| `--scorer` | Scorer name for DLC CSV output (default: `sleap-io`) |
| `--save-metadata` | Save JSON metadata file for CSV round-trip support |

### Supported Formats

**Input formats:** `slp`, `nwb`, `coco`, `labelstudio`, `alphatracker`, `jabs`, `dlc`, `csv`, `ultralytics`, `leap`

**Output formats:** `slp`, `nwb`, `coco`, `labelstudio`, `jabs`, `ultralytics`, `csv`

### Format Detection

The CLI automatically detects formats from file extensions:

| Extension | Format |
|-----------|--------|
| `.slp` | SLEAP |
| `.nwb` | NWB |
| `.mat` | LEAP |
| `.csv` | DeepLabCut |
| Directory with `data.yaml` | Ultralytics |

**Ambiguous extensions** (`.json`, `.h5`) require explicit `--from`:

```bash
# .json could be COCO, Label Studio, or AlphaTracker
sio convert -i data.json -o labels.slp --from coco
sio convert -i data.json -o labels.slp --from labelstudio
sio convert -i data.json -o labels.slp --from alphatracker

# .h5 could be JABS or DeepLabCut
sio convert -i data.h5 -o labels.slp --from jabs
sio convert -i data.h5 -o labels.slp --from dlc
```

### Embedding Frames

Create self-contained package files with embedded video frames:

```bash
# Embed only user-labeled frames
sio convert labels.slp -o labels.pkg.slp --embed user

# Embed all frames (including predictions)
sio convert labels.slp -o labels.pkg.slp --embed all

# Embed labeled frames plus suggestions
sio convert labels.slp -o labels.pkg.slp --embed suggestions
```

!!! info "Embedding options"

    - **`user`**: Only frames with manual annotations (smallest file size)
    - **`suggestions`**: Manual annotations plus suggested frames
    - **`all`**: All frames with any labels including predictions
    - **`source`**: Re-embed from an already embedded source

### Export to Ultralytics YOLO

```bash
# Export to YOLO format directory
sio convert labels.slp -o yolo_dataset/ --to ultralytics
```

This creates a directory structure compatible with Ultralytics YOLO training:

```
yolo_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### Export to CSV

Export pose data to CSV format for use with spreadsheet tools or custom analysis pipelines:

```bash
# Export to SLEAP Analysis CSV (default)
sio convert labels.slp -o output.csv

# Export to DeepLabCut CSV format
sio convert labels.slp -o output.csv --csv-format dlc --scorer MyModel

# Export with metadata for round-trip support
sio convert labels.slp -o output.csv --save-metadata
# Creates: output.csv and output.json
```

#### CSV Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv-format` | `sleap` | CSV format: `sleap`, `dlc`, `points`, `instances`, `frames` |
| `--scorer` | `sleap-io` | Scorer name for DLC format |
| `--save-metadata` | off | Save JSON metadata file for round-trip support |

#### CSV Formats

- **`sleap`**: SLEAP Analysis CSV format (one row per instance)
- **`dlc`**: DeepLabCut format (multi-header, one row per frame)
- **`points`**: One row per point (most normalized)
- **`instances`**: One row per instance with node coordinates as columns
- **`frames`**: One row per frame with all instances multiplexed

---

## `sio split`

Split a labels file into train/validation/test sets for machine learning workflows.

```bash
sio split <input> -o <output_dir> [options]
sio split -i <input> -o <output_dir> [options]
```

### Basic Usage

```bash
# Default 80/20 train/val split
sio split labels.slp -o splits/

# Three-way 70/15/15 split
sio split labels.slp -o splits/ --train 0.7 --test 0.15

# Reproducible split with seed
sio split labels.slp -o splits/ --seed 42
```

**Example output:**

```
Split 1000 frames from: labels.slp
Output directory: splits/

  train.slp: 800 frames
  val.slp: 200 frames

Random seed: 42
```

### Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input labels file (can also pass as positional argument) |
| `-o, --output` | Output directory for split files (required) |
| `--train` | Training set fraction, 0.0-1.0 (default: 0.8) |
| `--val` | Validation set fraction (default: remainder after train and test) |
| `--test` | Test set fraction (if not specified, no test split is created) |
| `--remove-predictions` | Remove predicted instances, keep only user labels |
| `--seed` | Random seed for reproducible splits |
| `--embed` | Embed frames in output (`user`, `all`, `suggestions`, `source`) |

### Output Files

The command creates split files in the output directory:

```
splits/
├── train.slp      # Training set (or train.pkg.slp if --embed)
├── val.slp        # Validation set
└── test.slp       # Test set (only if --test specified)
```

Each output file includes provenance metadata:

- `source_labels`: Path to the original input file
- `split`: Split name (`train`, `val`, or `test`)
- `split_seed`: Random seed used (if specified)

### Removing Predictions

For training, you typically want only user-labeled (ground truth) data:

```bash
# Keep only user-labeled instances, remove predictions
sio split labels.slp -o splits/ --remove-predictions --seed 42
```

This:

1. Removes all `PredictedInstance` objects
2. Clears suggestions
3. Removes empty frames and unused tracks/skeletons

!!! warning "Predictions-only files"
    If your file contains only predictions (no user labels), `--remove-predictions` will result in an empty dataset and the command will fail with an error.

### Creating Embedded Packages

For portable training datasets with embedded frames:

```bash
# Create package files with embedded user-labeled frames
sio split labels.slp -o splits/ --embed user --seed 42
```

This creates `train.pkg.slp`, `val.pkg.slp`, and optionally `test.pkg.slp` with frames embedded directly in the files.

!!! info "Video access required"
    Embedding frames requires video file access. Video reading works out of the box via the bundled imageio-ffmpeg.

### Reproducibility

Always use `--seed` for reproducible experiments:

```bash
# Same seed = same split every time
sio split labels.slp -o run1/ --train 0.8 --test 0.1 --seed 42
sio split labels.slp -o run2/ --train 0.8 --test 0.1 --seed 42
# run1/ and run2/ will have identical splits
```

The seed is stored in each output file's provenance for traceability.

!!! warning "Seed sensitivity to preprocessing"
    The `--seed` guarantees reproducibility only when all other options are identical. In particular, `--remove-predictions` changes which frames are available for splitting (frames with only predictions are removed), which changes the frame count and indexing. This means:

    ```bash
    # These will produce DIFFERENT splits even with the same seed:
    sio split labels.slp -o run1/ --seed 42
    sio split labels.slp -o run2/ --seed 42 --remove-predictions
    ```

    To ensure reproducibility, always use the same combination of options (especially `--remove-predictions`) with your seed.

### Fraction Behavior

- **Default (no `--val` or `--test`)**: 80% train, 20% validation
- **With `--test` only**: Train gets `--train`, test gets `--test`, val gets remainder
- **With explicit `--val` and `--test`**: Each split gets its specified fraction

Fractions must be between 0 and 1, and their sum cannot exceed 1.0:

```bash
# Valid: 0.7 + 0.15 + 0.15 = 1.0
sio split labels.slp -o splits/ --train 0.7 --val 0.15 --test 0.15

# Error: 0.8 + 0.15 + 0.15 = 1.1 > 1.0
sio split labels.slp -o splits/ --train 0.8 --val 0.15 --test 0.15
```

---

## `sio unsplit`

Merge multiple split files back into a single labels file. This is the inverse of `sio split`.

```bash
sio unsplit <input_files...> -o <output> [options]
sio unsplit <directory> -o <output> [options]
```

### Basic Usage

```bash
# Merge individual split files
sio unsplit train.slp val.slp -o merged.slp
sio unsplit train.slp val.slp test.slp -o merged.slp

# Merge all .slp files in a directory
sio unsplit splits/ -o merged.slp

# Merge embedded package files
sio unsplit train.pkg.slp val.pkg.slp -o merged.slp
```

**Example output:**

```
Loading: train.slp
  800 frames, 1 videos
Merging: val.slp
  +200 frames -> 1000 total

Saving: merged.slp

Merged 2 files:
  1000 frames, 1 videos
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output labels file (required) |
| `--embed` | Embed frames in output (`user`, `all`, `suggestions`, `source`) |

### Video Deduplication

When merging split files created with `sio split --embed`, videos are automatically deduplicated using provenance metadata. The `original_video` chain ensures that embedded videos from the same source are merged back into a single video reference.

**Modern files (with provenance):**
```bash
# Videos deduplicate automatically
sio unsplit train.pkg.slp val.pkg.slp -o merged.slp
# Result: 1 video (deduplicated via original_video)
```

**Legacy files (without provenance):**
```bash
# Videos may not deduplicate
sio unsplit old_train.slp old_val.slp -o merged.slp
# Result: May have multiple videos (safe behavior)
```

!!! note "Safe behavior for legacy files"
    For split files created by older versions of SLEAP (without provenance metadata), videos may not deduplicate. This is intentional—the merge uses conservative matching to avoid data corruption. If needed, use `sio filenames` to fix video paths after merging.

### Directory Input

When a directory is provided, all `.slp` files in that directory are merged (sorted alphabetically):

```bash
# These are equivalent:
sio unsplit splits/ -o merged.slp
sio unsplit splits/train.slp splits/val.slp -o merged.slp  # If only train.slp and val.slp exist
```

This is convenient for merging all splits created by `sio split`.

---

## `sio merge`

A flexible merge command for combining annotations from multiple sources with full control over matching strategies. While `sio unsplit` is optimized for reunifying train/val/test splits, `sio merge` handles general-purpose merging scenarios.

```bash
sio merge <input_files...> -o <output> [options]
sio merge <directory> -o <output> [options]
```

### Basic Usage

```bash
# Merge predictions into a labeled project (most common)
sio merge project.slp predictions.slp -o merged.slp

# Merge with explicit frame strategy
sio merge base.slp new.slp -o merged.slp --frame replace_predictions

# Merge multiple files
sio merge file1.slp file2.slp file3.slp -o combined.slp

# Merge all .slp files in directory
sio merge results/ -o combined.slp

# Verbose output with detailed statistics
sio merge project.slp predictions.slp -o merged.slp -v
```

**Example output:**

```
Loading: project.slp
  500 frames, 1 videos
Merging: predictions.slp
  Strategy: auto
  +1000 frames, +2500 instances

Saving: merged.slp

Merged 2 files:
  1500 frames, 1 videos
  2500 instances added
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | (required) | Output labels file |
| `--skeleton` | `structure` | Skeleton matching method |
| `--video` | `auto` | Video matching method |
| `--track` | `name` | Track matching method |
| `--frame` | `auto` | Frame merge strategy |
| `--instance` | `spatial` | Instance matching method |
| `--embed` | (none) | Embed frames in output (`user`, `all`, `suggestions`, `source`) |
| `-v, --verbose` | off | Show detailed merge information |

### Matching Strategies

#### Skeleton Matching (`--skeleton`)

How to match skeletons between files:

| Method | Behavior |
|--------|----------|
| `structure` | Same node names, any order (default) |
| `subset` | Incoming nodes are subset of base |
| `overlap` | Sufficient overlap between node sets |
| `exact` | Nodes and edges must be identical |

```bash
# Allow partial skeleton matches
sio merge base.slp partial.slp -o merged.slp --skeleton subset
```

#### Video Matching (`--video`)

How to identify same videos across files:

| Method | Behavior |
|--------|----------|
| `auto` | Safe cascade: path, basename, provenance (default) |
| `path` | Exact path match only |
| `basename` | Match by filename (ignores directory) |
| `content` | Shape + backend type |
| `shape` | Match and merge by shape (image lists) |
| `image_dedup` | Deduplicate image lists |

```bash
# Cross-platform merging (different paths, same filename)
sio merge local.slp remote.slp -o merged.slp --video basename
```

#### Track Matching (`--track`)

How to match track identities:

| Method | Behavior |
|--------|----------|
| `name` | Match tracks with identical names (default) |
| `identity` | Match by track object identity |

#### Frame Strategy (`--frame`)

How to handle overlapping frames (same video and frame index in both files):

| Strategy | Behavior | Use case |
|----------|----------|----------|
| `auto` | Smart merge: preserve user labels, update predictions (default) | Human-in-the-loop workflows |
| `keep_original` | Ignore incoming for overlapping frames | Preserve base annotations |
| `keep_new` | Replace base with incoming for overlapping frames | Overwrite with new annotations |
| `keep_both` | Keep all instances (may create duplicates) | Manual deduplication later |
| `replace_predictions` | Replace predictions, keep user labels | Re-running inference |
| `update_tracks` | Copy track assignments only, don't modify poses | Update identity labels |

```bash
# Update predictions from new model, keep user labels
sio merge project.slp new_predictions.slp -o merged.slp --frame replace_predictions

# Keep everything (for manual deduplication)
sio merge annotator1.slp annotator2.slp -o combined.slp --frame keep_both
```

#### Instance Matching (`--instance`)

For frame strategies that pair instances (`auto`, `update_tracks`), how to match instances within a frame:

| Method | Behavior |
|--------|----------|
| `spatial` | Match by centroid distance (default) |
| `identity` | Match by track identity |
| `iou` | Match by bounding box overlap |

```bash
# Use IoU for instance matching
sio merge base.slp other.slp -o merged.slp --instance iou
```

### Common Scenarios

**Merging predictions into a labeled project:**

```bash
# Default auto strategy handles this well
sio merge project.slp predictions.slp -o merged.slp

# What happens:
# - User labels are preserved
# - Predictions are updated if newer
# - New frames from predictions are added
```

**Combining annotations from multiple annotators:**

```bash
# Keep all instances for later review
sio merge annotator1.slp annotator2.slp -o combined.slp --frame keep_both
```

**Updating predictions with a new model:**

```bash
# Replace all predictions, keep user labels
sio merge project.slp new_model_output.slp -o updated.slp --frame replace_predictions
```

**Cross-platform project consolidation:**

```bash
# Match videos by basename when paths differ
sio merge local_project.slp remote_project.slp -o consolidated.slp --video basename
```

### Differences from `sio unsplit`

| Aspect | `sio unsplit` | `sio merge` |
|--------|---------------|-------------|
| Purpose | Reunify train/val/test splits | General-purpose merging |
| Frame strategy | Fixed `keep_both` | Configurable (default `auto`) |
| Video matching | Fixed `auto` | Configurable (default `auto`) |
| Provenance | Removes split-specific keys | Preserves all provenance |
| Use case | After `sio split` | Any merge scenario |

Use `sio unsplit` when merging files created by `sio split`. Use `sio merge` for all other merging scenarios.

---

## `sio embed`

Create portable `.pkg.slp` files with video frames embedded directly in the file. This allows sharing labels without requiring access to the original video files.

```bash
sio embed <input> -o <output> [options]
sio embed -i <input> -o <output> [options]
```

### Basic Usage

```bash
# Embed user-labeled frames only (default)
sio embed labels.slp -o labels.pkg.slp

# Include prediction-only frames
sio embed labels.slp -o labels.pkg.slp --predictions

# Include suggested frames
sio embed labels.slp -o labels.pkg.slp --suggestions

# Combine flags for all frame types
sio embed labels.slp -o labels.pkg.slp --predictions --suggestions

# Embed only suggestions (skip user frames)
sio embed labels.slp -o labels.pkg.slp --no-user --suggestions
```

**Example output:**

```
Embedded: labels.slp -> labels.pkg.slp
Frames: user: 100, predictions: 50, suggestions: 25
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | (required) | Input SLP file (can also pass as positional argument) |
| `-o, --output` | (required) | Output .pkg.slp file path |
| `--user/--no-user` | on | Include frames with user-labeled instances |
| `--predictions/--no-predictions` | off | Include frames with only predicted instances |
| `--suggestions/--no-suggestions` | off | Include suggested frames for labeling |

### Frame Types

- **User frames**: Frames containing at least one user-labeled (non-predicted) instance. These are your ground truth annotations.
- **Prediction frames**: Frames containing only predicted instances (no user labels). Useful for sharing inference results.
- **Suggestions**: Frames marked for labeling in the suggestions list. May or may not have instances.

!!! tip "Choosing what to embed"
    - **For sharing training data**: Use default (`--user` only) for smallest file size
    - **For sharing predictions**: Add `--predictions` to include inference results
    - **For active learning**: Add `--suggestions` to include frames needing annotation

### When to Use `sio embed` vs `sio convert --embed`

Both commands can embed frames, but they serve different purposes:

- **`sio embed`**: Granular control over which frame types to include. Best for creating packages.
- **`sio convert --embed`**: Part of format conversion workflow. Uses simpler mode-based embedding.

```bash
# Equivalent commands:
sio embed labels.slp -o labels.pkg.slp
sio convert labels.slp -o labels.pkg.slp --embed user

# But sio embed allows more control:
sio embed labels.slp -o labels.pkg.slp --predictions --suggestions
```

---

## `sio unembed`

Convert an embedded `.pkg.slp` file back to a regular `.slp` file that references the original video files. This reverses the embedding process.

```bash
sio unembed <input> -o <output>
sio unembed -i <input> -o <output>
```

### Basic Usage

```bash
# Restore original video references
sio unembed labels.pkg.slp -o labels.slp
```

**Example output:**

```
Unembedded: labels.pkg.slp -> labels.slp
Restored 1 video(s) to source paths
```

### Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input .pkg.slp file (can also pass as positional argument) |
| `-o, --output` | Output .slp file path (required) |

### Requirements

The `unembed` command requires:

1. **Embedded videos**: The input file must contain embedded video frames
2. **Source video metadata**: The file must have provenance information about the original videos

If either requirement is not met, you'll get a helpful error message:

```
# No embedded videos
Error: No embedded videos found. This file does not contain embedded frames.
Use 'sio show --video' to inspect video details.

# Missing source metadata (legacy files)
Error: Cannot unembed: 1/1 embedded video(s) have no source video metadata.

This typically happens with legacy .pkg.slp files created before
source video tracking was added.

To inspect video details, run:
  sio show 'labels.pkg.slp' --video
```

!!! note "Original videos don't need to exist"
    The `unembed` command only updates the file references—it doesn't verify that the original video files exist. You can unembed even if the original videos are on a different machine, then use `sio filenames` to update the paths.

### Embed/Unembed Roundtrip

The embed and unembed operations are lossless inverses:

```bash
# Original file
sio show labels.slp
# -> 1 video, 100 frames

# Embed frames
sio embed labels.slp -o labels.pkg.slp

# Unembed back to original format
sio unembed labels.pkg.slp -o labels_restored.slp

# Verify: identical annotations
sio show labels_restored.slp
# -> 1 video, 100 frames (same as original)
```

All annotations, skeletons, tracks, and metadata are preserved exactly.

---

## `sio filenames`

List or update video file paths in a labels file. By default, lists all video filenames for quick inspection. With update options, replaces paths and saves to a new file.

Useful for:

- Quickly checking what video paths are in a labels file
- Moving labels to a new machine with different paths
- Fixing broken video references after reorganizing files
- Cross-platform path conversion (Windows ↔ Linux/macOS)

```bash
# Inspection mode (default)
sio filenames <input>
sio filenames -i <input>

# Update mode
sio filenames <input> -o <output> [update options]
```

### Inspection Mode

By default, without any update flags, the command lists all video filenames:

```bash
sio filenames labels.slp
```

**Example output:**

```
Video filenames in labels.slp:
  [0] /home/user/data/video.mp4
  [1] /home/user/data/video2.mp4
```

For image sequences:

```
Video filenames in labels.slp:
  [0] /data/frames/frame_0001.png ... (150 images)
```

This is a quick way to check video paths before deciding how to update them.

### Update Modes

When you provide `-o` and one of the update flags, the command updates paths and saves:

| Mode | Option | Description |
|------|--------|-------------|
| **List** | `--filename` | Replace all video filenames in order |
| **Map** | `--map OLD NEW` | Replace specific filenames by exact match |
| **Prefix** | `--prefix OLD NEW` | Replace path prefixes (cross-platform aware) |

You must specify exactly one update mode when updating.

### Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input labels file (can also pass as positional argument) |
| `-o, --output` | Output labels file path (required for update mode) |
| `--filename` | New filename (repeat for each video in list mode) |
| `--map OLD NEW` | Replace OLD filename with NEW (repeat for multiple mappings) |
| `--prefix OLD NEW` | Replace OLD prefix with NEW (repeat for multiple prefixes) |

### List Mode

Replace all video filenames in order. You must provide exactly one `--filename` for each video in the labels file:

```bash
# Single video file
sio filenames labels.slp -o fixed.slp \
    --filename /new/path/video.mp4

# Multiple videos (must match video count in file)
sio filenames multiview.slp -o fixed.slp \
    --filename /data/cam1.mp4 \
    --filename /data/cam2.mp4 \
    --filename /data/cam3.mp4
```

!!! warning "Video count must match"
    The number of `--filename` options must exactly match the number of videos in the labels file. Use `sio filenames labels.slp` to check the video count first.

### Map Mode

Replace specific filenames using exact matching. Only videos whose paths match will be updated:

```bash
# Replace a single video path
sio filenames labels.slp -o fixed.slp \
    --map video.mp4 /data/videos/video.mp4

# Replace multiple specific paths
sio filenames labels.slp -o fixed.slp \
    --map recording1.mp4 /nas/project/recording1.mp4 \
    --map recording2.mp4 /nas/project/recording2.mp4
```

Map mode is useful when you only need to update some videos or when you have the exact old and new paths.

### Prefix Mode

Replace path prefixes. This is the most flexible mode for relocating files:

```bash
# Move from absolute to relative paths
sio filenames labels.slp -o fixed.slp \
    --prefix /home/user/data ./data

# Cross-platform: Windows to Linux
sio filenames labels.slp -o fixed.slp \
    --prefix "C:\Users\lab\data" /mnt/data

# Cross-platform: Linux to Windows
sio filenames labels.slp -o fixed.slp \
    --prefix /mnt/data "D:\project\data"

# Multiple prefix replacements
sio filenames labels.slp -o fixed.slp \
    --prefix /old/videos /new/videos \
    --prefix /old/images /new/images
```

!!! tip "Cross-platform path handling"
    Prefix mode automatically normalizes path separators. You can match Windows paths (`C:\data`) with Linux-style prefixes (`C:/data`) and vice versa.

### Examples

**Scenario: Moving to a new machine**

Your labels file references `/home/alice/project/videos/mouse.mp4`, but on the new machine the path is `/data/experiments/videos/mouse.mp4`:

```bash
# First, check current paths
sio filenames labels.slp

# Update with new prefix
sio filenames labels.slp -o labels_new.slp \
    --prefix /home/alice/project /data/experiments
```

**Scenario: Sharing with a collaborator**

Make paths relative so the labels work from any base directory:

```bash
sio filenames labels.slp -o labels_portable.slp \
    --prefix /absolute/path/to/project .
```

**Scenario: Windows to Linux server**

Labels created on Windows need to work on a Linux cluster:

```bash
sio filenames labels.slp -o labels_linux.slp \
    --prefix "C:\Users\lab\experiment" /home/lab/experiment
```

**Scenario: Image sequences**

The command also works with image sequence videos (where `filename` is a list of image paths):

```bash
# Update prefix for all images in the sequence
sio filenames labels.slp -o fixed.slp \
    --prefix /old/frames /new/frames
```

---

## `sio fix`

Automatically detect and fix common problems in SLEAP labels files, including duplicate videos, unused skeletons, and predictions.

```bash
sio fix <input> [-o <output>] [options]
sio fix -i <input> [-o <output>] [options]
```

### Basic Usage

```bash
# Auto-detect and fix with safe defaults
sio fix labels.slp

# Preview changes without modifying
sio fix labels.slp --dry-run

# Explicit output path
sio fix labels.slp -o fixed.slp

# Verbose output showing details
sio fix labels.slp -v
```

**Example output:**

```
Loading: labels.slp
  26 videos, 715 frames, 2 skeletons, 27 tracks

Analyzing...

⚠ Videos: Found 2 duplicate group(s)
⚠ Skeletons:
  'fly_13pt': 1245 user, 500 pred (most frequent)
  'fly_copy': 0 user, 0 pred (unused)
✓ Video Color: All videos consistent
ℹ Predictions: 2274 predicted instances (150 untracked)

Actions:
  → Merge 2 duplicate video group(s)
  → Remove 1 unused skeleton(s)
  → Remove unused tracks
  → Remove empty frames

Saved: labels.fixed.slp
  25 videos, 715 frames, 1 skeleton, 27 tracks
```

### What Gets Fixed by Default

With default options, `sio fix` automatically:

1. **Merges duplicate videos**: Videos pointing to the same file are consolidated
2. **Removes unused skeletons**: Skeletons with no instances are removed
3. **Removes prediction-only skeletons**: Skeletons used only by predictions (not user labels) are removed along with their predictions
4. **Cleans up metadata**: Unused tracks are removed
5. **Removes empty frames**: Frames with no instances are removed

### Options Reference

#### Input/Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | (required) | Input labels file (can also pass as positional argument) |
| `-o, --output` | `{input}.fixed.slp` | Output file. For `.pkg.slp` files: `{input}.fixed.pkg.slp` |
| `--dry-run` | False | Analyze and show what would be done without making changes |
| `-v, --verbose` | False | Show detailed analysis with per-video and per-skeleton breakdowns |

#### Video Options

| Option | Default | Description |
|--------|---------|-------------|
| `--deduplicate-videos` | True | Merge duplicate video entries pointing to the same file |
| `--no-deduplicate-videos` | | Skip video deduplication |
| `--video-color` | | Set video color mode: `grayscale`, `rgb`, or `auto` |

#### Skeleton Options

| Option | Default | Description |
|--------|---------|-------------|
| `--remove-unused-skeletons` | True | Remove skeletons with no instances, or only predictions |
| `--no-remove-unused-skeletons` | | Keep all skeletons |
| `--consolidate-skeletons` | False | **DESTRUCTIVE**: Keep most frequent skeleton, delete instances from other skeletons |

!!! warning "Skeleton consolidation is destructive"
    The `--consolidate-skeletons` flag will permanently delete user-labeled instances that use non-primary skeletons. Use `--dry-run` first to review what will be deleted.

#### Prediction Options

| Option | Default | Description |
|--------|---------|-------------|
| `--remove-predictions` | False | Remove ALL predicted instances |
| `--remove-untracked-predictions` | False | Remove only predictions with no track assignment |

#### Cleanup Options

| Option | Default | Description |
|--------|---------|-------------|
| `--remove-unused-tracks` | True | Remove tracks not used by any instance |
| `--remove-empty-frames` | True | Remove frames with no instances |
| `--remove-empty-instances` | False | Remove instances with no visible points |
| `--remove-unlabeled-videos` | False | Remove videos with no labeled frames |

#### Path Fixing Options

| Option | Description |
|--------|-------------|
| `--prefix OLD NEW` | Replace OLD path prefix with NEW (repeatable) |
| `--map OLD NEW` | Replace exact filename OLD with NEW (repeatable) |

### Common Scenarios

**Scenario: Fixing duplicate video entries**

SLEAP can sometimes create duplicate video entries pointing to the same file. This causes issues when trying to delete labels:

```bash
# Fix automatically (enabled by default)
sio fix labels.slp

# Skip if you want to preserve duplicates
sio fix labels.slp --no-deduplicate-videos
```

**Scenario: Removing all predictions**

After manual review, remove all predictions to keep only user labels:

```bash
sio fix labels.slp --remove-predictions
```

**Scenario: Surgical prediction cleanup**

Remove only untracked predictions (keeping tracked predictions intact):

```bash
sio fix labels.slp --remove-untracked-predictions
```

**Scenario: Multiple skeletons with user labels**

When a file has multiple skeletons that both have user-labeled instances, `sio fix` will warn you:

```
⚠  WARNING: Multiple skeletons have user instances!
    Use --consolidate-skeletons to keep 'fly_13pt' and remove 23 instances.
    This is irreversible - review carefully before proceeding.
```

To force consolidation (keeping the most frequently used skeleton):

```bash
# Preview first!
sio fix labels.slp --consolidate-skeletons --dry-run

# Then apply
sio fix labels.slp --consolidate-skeletons
```

**Scenario: Cross-platform path fixing**

Fix Windows paths for use on Linux:

```bash
sio fix labels.slp --prefix "C:\data\videos" /mnt/data/videos
```

**Scenario: Aggressive cleanup**

Remove everything not essential (predictions, unlabeled videos, empty instances):

```bash
sio fix labels.slp \
    --remove-predictions \
    --remove-unlabeled-videos \
    --remove-empty-instances
```

**Scenario: Minimal cleanup (disable defaults)**

Only fix paths without any other cleanup:

```bash
sio fix labels.slp \
    --no-deduplicate-videos \
    --no-remove-unused-skeletons \
    --no-remove-unused-tracks \
    --no-remove-empty-frames \
    --prefix /old/path /new/path
```

**Scenario: Video color mode issues**

If videos are being incorrectly detected as grayscale or RGB (e.g., due to compression artifacts or very similar color channels), you can force the color mode:

```bash
# Force all videos to grayscale (single channel output)
sio fix labels.slp --video-color grayscale

# Force all videos to RGB (three channel output)
sio fix labels.slp --video-color rgb

# Reset to auto-detection
sio fix labels.slp --video-color auto
```

This is particularly useful when training models that expect a specific number of input channels.

---

## `sio render`

Create video files or single images with pose annotations overlaid on video frames.

```bash
# Video mode (default)
sio render <input> [-o <output>] [options]

# Image mode (single frame)
sio render <input> --lf <index> [-o <output>] [options]
```


!!! tip "Input argument"
    Like `sio show`, the input can be passed as a positional argument or with `-i`:
    ```bash
    sio render predictions.slp           # Positional (preferred)
    sio render -i predictions.slp        # Explicit flag (also works)
    ```

### Render Modes

**Video mode** (default): Renders all labeled frames to a video file.

**Image mode**: Renders a single frame to a PNG image. Use `--lf` or `--frame`.

### Basic Usage

```bash
# Render video with automatic output filename
sio render predictions.slp                      # -> predictions.viz.mp4

# Render with explicit output path
sio render predictions.slp -o output.mp4

# Fast preview (0.25x resolution)
sio render predictions.slp --preset preview

# Render a specific clip
sio render predictions.slp --start 100 --end 200

# Render a single frame to PNG
sio render predictions.slp --lf 0               # -> predictions.lf=0.png

# Render without source video (solid background)
sio render predictions.slp --background black
sio render predictions.slp --background "#333"
```

### Options Reference

#### Input/Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | (required) | Input labels file (can also pass as positional argument) |
| `-o, --output` | auto | Output path. Default: `{input}.viz.mp4` for video, `{input}.lf={N}.png` for image |
| `--background` | video | Background mode: `video` (load frames) or a color (e.g., `black`, `#333`) |

#### Frame Selection Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lf` | none | Render single labeled frame by index. Outputs PNG. |
| `--frame` | none | Render single frame by video frame index (use with `--video`). Outputs PNG. |
| `--start` | first labeled | Start frame index for video (0-based, inclusive) |
| `--end` | last labeled | End frame index for video (0-based, exclusive) |
| `--video` | 0 | Video index for multi-video labels |
| `--all-frames` / `--labeled-only` | auto | Render all frames or only labeled. Default: `--all-frames` for single-video files. |

#### Quality Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preset` | none (1.0x) | Quality preset: `preview` (0.25x), `draft` (0.5x), `final` (1.0x) |
| `--scale` | 1.0 | Scale factor (overrides `--preset`) |
| `--fps` | source FPS | Output video FPS. Change to slow down or speed up playback. |
| `--crf` | 25 | Video quality (2-32, lower=better quality, larger file) |
| `--x264-preset` | superfast | H.264 encoding speed trade-off (ultrafast to slow) |

#### Appearance Options

| Option | Default | Description |
|--------|---------|-------------|
| `--color-by` | auto | Color scheme: `auto`, `track`, `instance`, `node` |
| `--palette` | glasbey | Color palette (glasbey, tableau10, distinct, rainbow, etc.) |
| `--marker-shape` | circle | Node marker: `circle`, `square`, `diamond`, `triangle`, `cross` |
| `--marker-size` | 4.0 | Node marker radius in pixels |
| `--line-width` | 2.0 | Edge line width in pixels |
| `--alpha` | 1.0 | Pose overlay transparency (0.0-1.0) |
| `--no-nodes` | false | Hide node markers |
| `--no-edges` | false | Hide skeleton edges |

#### Crop Options (Single Image Only)

| Option | Default | Description |
|--------|---------|-------------|
| `--crop` | none | Crop region: `auto` or `x1,y1,x2,y2` (pixels or normalized 0.0-1.0) |
| `--crop-padding` | 0.2 | Padding for auto-crop as fraction of bounding box |

### Single Image Rendering

Render individual frames to PNG files for figures, thumbnails, or quick inspection:

```bash
# Render labeled frame by index (0-based)
sio render predictions.slp --lf 0               # -> predictions.lf=0.png
sio render predictions.slp --lf 42              # -> predictions.lf=42.png

# Render specific video frame by index
sio render predictions.slp --frame 100          # -> predictions.video=0.frame=100.png
sio render predictions.slp --frame 100 --video 1

# Explicit output path
sio render predictions.slp --lf 5 -o frame.png
```

!!! tip "Labeled frame vs frame index"
    - `--lf N` renders the Nth labeled frame in the file (regardless of video frame number)
    - `--frame N` renders video frame N (must have predictions at that frame)

### Cropping (Single Image Only)

Crop the output image to focus on specific regions or automatically fit around detected instances:

```bash
# Auto-fit: crop to bounding box of all instances with 20% padding (default)
sio render predictions.slp --lf 0 --crop auto

# Auto-fit with custom padding (30% of bounding box)
sio render predictions.slp --lf 0 --crop auto --crop-padding 0.3

# Pixel coordinates (x1, y1, x2, y2)
sio render predictions.slp --lf 0 --crop 100,100,300,300

# Normalized coordinates (center 50% of frame)
sio render predictions.slp --lf 0 --crop 0.25,0.25,0.75,0.75
```

The crop modes:

- **`auto`**: Automatically fit to the bounding box of all instances, with padding. Best for focusing on animals.
- **Pixel coordinates**: `x1,y1,x2,y2` as integers. Use for precise cropping when you know exact pixel locations.
- **Normalized coordinates**: `x1,y1,x2,y2` as floats between 0.0-1.0. Use for relative cropping that works across different video resolutions.

!!! note "Video mode limitation"
    Cropping is currently only supported for single image mode (`--lf` or `--frame`). Video cropping is not yet implemented.

### Video Frame Ranges

Render specific portions of the video:

```bash
# Frames 100 to 200 (0-based, end is exclusive)
sio render predictions.slp --start 100 --end 200

# From frame 500 to end
sio render predictions.slp --start 500

# First 100 frames
sio render predictions.slp --end 100
```

### Adjusting Playback Speed

Use `--fps` to control playback speed:

```bash
# Slow motion (half speed if source is 30fps)
sio render predictions.slp --fps 15

# Speed up (double speed if source is 30fps)
sio render predictions.slp --fps 60

# Fixed frame rate output
sio render predictions.slp --fps 24
```

### Quality Presets

Use presets for quick quality/speed trade-offs:

```bash
# Fast preview for checking results (0.25x resolution)
sio render predictions.slp --preset preview     # -> predictions.viz.mp4

# Draft quality for review (0.5x resolution)
sio render predictions.slp --preset draft

# Full quality for publication (1.0x resolution)
sio render predictions.slp --preset final

# Or specify exact scale
sio render predictions.slp --scale 0.75
```

### Color Schemes

Control how poses are colored:

```bash
# Auto-select based on data (default)
sio render predictions.slp --color-by auto

# Color by track identity (consistent across frames)
sio render predictions.slp --color-by track

# Color by instance (each animal in frame gets different color)
sio render predictions.slp --color-by instance

# Color by node type (each body part gets different color)
sio render predictions.slp --color-by node
```

The `auto` mode uses smart defaults:

- If tracks available → color by track
- If single frame → color by instance
- If video (multiple frames) → color by node

### Color Palettes

Choose from built-in or colorcet palettes:

```bash
# Built-in palettes
sio render predictions.slp --palette distinct
sio render predictions.slp --palette rainbow
sio render predictions.slp --palette tableau10

# Colorcet palettes
sio render predictions.slp --palette glasbey  # 256 distinct colors
sio render predictions.slp --palette glasbey_warm
```

Available built-in palettes: `distinct`, `rainbow`, `warm`, `cool`, `pastel`, `seaborn`, `tableau10`, `viridis`

!!! tip "Discover available options"
    Use the discovery flags to see all available colors and palettes:
    ```bash
    sio render --list-colors    # Show all named colors
    sio render --list-palettes  # Show all available palettes
    ```

### Marker Shapes and Styles

Customize the appearance of pose overlays:

```bash
# Different marker shapes
sio render predictions.slp --marker-shape circle
sio render predictions.slp --marker-shape square
sio render predictions.slp --marker-shape diamond
sio render predictions.slp --marker-shape triangle
sio render predictions.slp --marker-shape cross

# Adjust sizes
sio render predictions.slp --marker-size 6 --line-width 3

# Semi-transparent overlays
sio render predictions.slp --alpha 0.7

# Show only edges (no node markers)
sio render predictions.slp --no-nodes

# Show only nodes (no skeleton edges)
sio render predictions.slp --no-edges
```

### Multi-Video Labels

For labels with multiple videos, select which video to render:

```bash
# Render the second video (0-indexed)
sio render -i multiview.slp --video 1
```

### Example Workflow

```bash
# 1. Quick preview to check predictions
sio render predictions.slp --preset preview

# 2. Check a specific section
sio render predictions.slp --start 500 --end 600 --preset draft

# 3. Render a single interesting frame
sio render predictions.slp --lf 42 -o highlight.png

# 4. Final render with custom styling
sio render predictions.slp -o final.mp4 \
    --color-by track \
    --palette tableau10 \
    --marker-shape diamond \
    --marker-size 5 \
    --line-width 2.5
```

## `sio trim`

Trim a video and labels file to a specific frame range, adjusting frame indices accordingly.

```bash
# Labels mode: trim labels and video together
sio trim <labels> [--start <frame>] [--end <frame>] [-o <output>] [options]

# Video mode: trim standalone video
sio trim <video> [--start <frame>] [--end <frame>] [-o <output>] [options]
```

!!! tip "Two modes"
    **Labels mode**: Trims both the labels file and associated video, adjusting frame indices.

    **Video mode**: Trims a standalone video file (no labels).

### Basic Usage

```bash
# Trim labels and video to frames 100-1000
sio trim labels.slp --start 100 --end 1000              # -> labels.trim.slp + labels.trim.mp4

# Explicit output path
sio trim labels.slp --start 100 --end 1000 -o clip.slp

# Trim just a video file
sio trim video.mp4 --start 100 --end 500                # -> video.trim.mp4

# Trim from start to frame 1000
sio trim labels.slp --end 1000

# Trim from frame 100 to end of video
sio trim labels.slp --start 100
```

### Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | (required) | Input labels file or video (can also pass as positional arg) |
| `-o, --output` | auto | Output path. Default: `{input}.trim.slp` (labels) or `{input}.trim.mp4` (video) |
| `--start` | 0 | Start frame index (0-based, inclusive) |
| `--end` | last frame + 1 | End frame index (0-based, exclusive) |
| `--video` | auto | Video index for multi-video labels. Default: 0 if single video |
| `--fps` | source FPS | Output video FPS. Change to adjust playback speed. |
| `--crf` | 25 | Video quality (2-32, lower=better quality, larger file) |
| `--x264-preset` | superfast | H.264 encoding speed trade-off (ultrafast to slow) |

### Video Encoding Options

```bash
# Higher quality output (larger file)
sio trim labels.slp --start 100 --end 500 --crf 18

# Slow motion (half speed)
sio trim labels.slp --start 100 --end 500 --fps 15

# Fast, lower compression
sio trim labels.slp --start 100 --end 500 --x264-preset ultrafast
```

### Multi-Video Labels

For labels with multiple videos, select which video to trim:

```bash
# Trim the second video (0-indexed)
sio trim multiview.slp --start 100 --end 500 --video 1
```

!!! warning "Multi-video requirement"
    When trimming labels with multiple videos, you must specify `--video` to select which video to trim. The command will error if multiple videos are present and `--video` is not provided.

### Frame Index Adjustment

When trimming labels, frame indices are automatically adjusted to match the new video:

- Original frames 100-500 become frames 0-400 in the trimmed output
- Labeled frames outside the trim range are removed
- Suggestions are filtered and adjusted similarly

---

## `sio reencode`

Reencode videos with frequent keyframes for **reliable and fast** random access during annotation and playback. This command addresses two critical issues that can cause problems in annotation and computer vision workflows:

1. **Unreliable seeking**: Some video formats, codecs, or encoding settings cause frame-inaccurate seeking, where requesting frame N returns frame N-1 or N+1. This leads to misaligned annotations that don't match the actual image content.

2. **Slow seeking**: Videos with sparse keyframes require decoding many intermediate frames to reach a target frame, causing sluggish navigation during annotation.

!!! warning "Why seeking reliability matters"
    In pose estimation and annotation workflows, even single-frame seeking errors can corrupt your dataset. If you label "frame 100" but your video library actually returns frame 99, your ground truth annotations will be permanently misaligned with the images used during training. This is especially problematic because the errors are often inconsistent and hard to detect.

    **Common sources of unreliable seeking:**

    - Variable frame rate (VFR) videos from screen recordings or phones
    - Videos with B-frames and complex GOP structures
    - AVI files with missing or corrupt index
    - Some MP4s encoded with certain camera firmware
    - Seeking behavior differences between OpenCV, PyAV, and imageio-ffmpeg

```bash
sio reencode <input> [-o <output>] [options]
sio reencode -i <input> [-o <output>] [options]
```

### Basic Usage

```bash
# Reencode with default settings (medium quality, 1 keyframe/second)
sio reencode video.mp4 -o video.seekable.mp4

# Higher quality encoding
sio reencode video.mp4 -o output.mp4 --quality high

# More frequent keyframes for faster seeking
sio reencode video.mp4 -o output.mp4 --keyframe-interval 0.5

# Downsample high-speed video for preview
sio reencode highspeed.mp4 -o preview.mp4 --fps 30 --quality low

# Preview the ffmpeg command without executing
sio reencode video.mp4 -o output.mp4 --dry-run

# Force Python path (for HDF5-embedded sources or when ffmpeg unavailable)
sio reencode video.mp4 -o output.mp4 --no-ffmpeg
```

### SLP Batch Processing

When given an `.slp` file as input, the reencode command processes all videos in the SLEAP project:

```bash
# Reencode all videos in an SLP project
sio reencode project.slp -o project.reencoded.slp

# With quality option
sio reencode project.slp -o project.reencoded.slp --quality high

# Preview what would be done
sio reencode project.slp -o project.reencoded.slp --dry-run
```

This creates:

- A new SLP file with updated video paths
- A `{output_name}.videos/` directory containing the reencoded videos

**Behavior by video type:**

- **MediaVideo** (`.mp4`, `.avi`, `.mov`, etc.): Reencoded using ffmpeg (fast)
- **HDF5Video** (embedded videos in `.slp` files): Reencoded using Python path
- **ImageVideo** (image sequences): Skipped (cannot be reencoded to video)
- **TiffVideo** (TIFF stacks): Skipped (cannot be reencoded to video)

This is particularly useful for:

- Fixing seeking issues in all videos before starting annotation
- Standardizing video formats across a multi-video project
- Extracting embedded videos from `.pkg.slp` files to standalone MP4s

### Options Reference

#### Input/Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | (required) | Input video or SLP file (can also pass as positional argument) |
| `-o, --output` | `{input}.reencoded.mp4` or `.slp` | Output video/SLP path |
| `--overwrite` | False | Overwrite existing output file |
| `--dry-run` | False | Show ffmpeg command without executing |

#### Quality Options

| Option | Default | Description |
|--------|---------|-------------|
| `--quality` | medium | Quality level: `lossless`, `high`, `medium`, `low` |
| `--crf` | (from quality) | Direct CRF control (0-51, lower=better). Overrides `--quality` |

**Quality level mapping:**

| Level | CRF | Description |
|-------|-----|-------------|
| `lossless` | 0 | Mathematically identical (huge files) |
| `high` | 18 | Visually lossless |
| `medium` | 25 | Good quality, reasonable size (default) |
| `low` | 32 | Smaller files, some quality loss |

#### Keyframe Options

| Option | Default | Description |
|--------|---------|-------------|
| `--keyframe-interval` | 1.0 | Keyframe interval in seconds. Lower = better seekability, larger files |
| `--gop` | (from interval) | GOP size in frames. Overrides `--keyframe-interval` |

!!! tip "Choosing keyframe interval"
    More frequent keyframes improve both seeking speed and reliability:

    - **0.5 seconds**: 2 keyframes/second - most reliable, fastest seeking, ~50% larger files
    - **1.0 second**: 1 keyframe/second - good balance (default)
    - **2.0 seconds**: Smaller files, slightly slower seeking

    For maximum reliability in annotation workflows, prefer shorter intervals (0.5-1.0s).

#### Frame Rate Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fps` | source FPS | Output frame rate. Useful for downsampling high-speed video |

#### Encoding Options

| Option | Default | Description |
|--------|---------|-------------|
| `--encoding` | superfast | x264 encoding preset. Slower = better compression |
| `--use-ffmpeg/--no-ffmpeg` | auto | Force ffmpeg fast path or Python fallback |

Available encoding presets (fastest to slowest): `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow`

!!! note "FFmpeg vs Python path"
    - **FFmpeg path** (default): ~10x faster, uses direct ffmpeg subprocess
    - **Python path** (`--no-ffmpeg`): Frame-by-frame processing, works with any video source including HDF5-embedded videos

### How Reencoding Improves Seeking

The reencoded video has several properties that ensure reliable, frame-accurate seeking:

- **Regular keyframe intervals**: Keyframes (I-frames) can be decoded independently. With keyframes every 1 second at 30fps, the decoder never needs to process more than 30 frames to reach any target.
- **No B-frames**: The output uses only I-frames and P-frames, eliminating bidirectional dependencies that can confuse some decoders.
- **Constant frame rate**: Variable frame rate is converted to constant, ensuring frame indices map predictably to timestamps.
- **Clean GOP structure**: Each group of pictures has a consistent, predictable structure.
- **Standard H.264 encoding**: Widely supported codec with consistent behavior across video libraries.

### Common Scenarios

**Fixing unreliable seeking for annotation:**

```bash
# Videos from phones/screen recordings often have VFR issues
sio reencode phone_recording.mp4 -o reliable.mp4

# Camera videos with seeking problems
sio reencode problematic_camera.avi -o reliable.mp4

# Maximum reliability for critical annotation projects
sio reencode raw.mp4 -o reliable.mp4 --keyframe-interval 0.5 --quality high
```

**Fixing slow-seeking videos:**

```bash
# Basic reencoding with more frequent keyframes
sio reencode slow_video.mp4 -o fast_video.mp4

# Maximum seekability (keyframe every 0.5 seconds)
sio reencode slow_video.mp4 -o fast_video.mp4 --keyframe-interval 0.5
```

**Standardizing video format before annotation:**

```bash
# Convert and normalize any video before starting annotation
# This prevents seeking issues from corrupting your labels later
sio reencode experiment_video.mov -o experiment_video.mp4
```

**Creating preview videos from high-speed recordings:**

```bash
# Downsample 250fps to 30fps with lower quality
sio reencode highspeed_250fps.mp4 -o preview.mp4 --fps 30 --quality low
```

**High-quality archival encoding:**

```bash
# Visually lossless with slow compression for smaller files
sio reencode raw.mp4 -o archival.mp4 --quality high --encoding slow
```

**Processing HDF5-embedded videos:**

```bash
# Force Python path for embedded video sources
sio reencode embedded_video.h5 -o extracted.mp4 --no-ffmpeg
```

### Output

The command displays progress and file size comparison:

```
Reencoding: video.mp4
  Output: video.seekable.mp4
  Quality: CRF 25, Preset: superfast, Keyframes: 1.0s
[████████████████████████████████] 100% • 1000/1000 frames
Saved: video.seekable.mp4
  Size: 45.2 MB -> 52.1 MB (+15.3%)
```

!!! info "File size vs reliability trade-off"
    More frequent keyframes typically increase file size by 10-50% depending on video content. This trade-off is almost always worthwhile for annotation workflows—the cost of misaligned annotations due to seeking errors far exceeds the cost of additional storage.

---

## `sio transform`

Apply geometric transformations (crop, scale, rotate, pad, flip) to videos while automatically adjusting all landmark coordinates to maintain alignment.

!!! tip "Full Tutorial"
    For detailed examples with visual galleries, config file specifications, and Python API usage, see the [Transforms Guide](transforms.md).

```bash
sio transform <input> [OPTIONS] -o <output>
```

### Basic Usage

```bash
# Scale down 50%
sio transform labels.slp --scale 0.5 -o scaled.slp

# Crop to region (x1,y1,x2,y2)
sio transform labels.slp --crop 100,100,500,500 -o cropped.slp

# Rotate 90 degrees clockwise
sio transform labels.slp --rotate 90 -o rotated.slp

# Flip horizontally
sio transform labels.slp --flip-horizontal -o flipped.slp

# Combined transforms (applied in order: crop -> scale -> rotate -> pad -> flip)
sio transform labels.slp --crop 100,100,500,500 --scale 2.0 -o zoomed.slp

# Preview without processing
sio transform labels.slp --scale 0.5 --dry-run

# Transform raw video (no labels)
sio transform video.mp4 --scale 0.5 -o video_scaled.mp4
```

### Transform Pipeline

Transforms are always applied in a fixed order: **crop → scale → rotate → pad → flip**

All landmark coordinates are automatically transformed using affine matrices to maintain alignment.

### Options Reference

#### Input/Output

| Option | Description |
|--------|-------------|
| `-i, --input` | Input SLP file or video |
| `-o, --output` | Output path (default: `{input}.transformed.slp`) |
| `--config` | YAML config file with per-video transforms |
| `--output-transforms` | Export transform metadata to YAML |
| `--embed-provenance` | Store transform metadata in output SLP |
| `--overwrite, -y` | Overwrite existing output files |

#### Transforms

| Option | Format | Description |
|--------|--------|-------------|
| `--crop` | `[idx:]x1,y1,x2,y2` | Crop region (pixels or normalized 0.0-1.0) |
| `--scale` | `[idx:]value` or `[idx:]w,h` | Scale factor or target dimensions |
| `--rotate` | `[idx:]degrees` | Rotation angle (clockwise positive) |
| `--clip-rotation` | flag | Keep original dimensions when rotating |
| `--pad` | `[idx:]t,r,b,l` | Padding in pixels |
| `--flip-horizontal` | flag | Mirror left-right |
| `--flip-vertical` | flag | Mirror top-bottom |

#### Quality & Encoding

| Option | Default | Description |
|--------|---------|-------------|
| `--quality` | `bilinear` | Interpolation: `nearest`, `bilinear`, `bicubic` |
| `--fill` | `0` | Fill value (0-255 or `R,G,B`) |
| `--crf` | `25` | Video quality (0-51, lower = better) |
| `--x264-preset` | `superfast` | Encoding speed |
| `--fps` | (source) | Output frame rate |
| `--keyframe-interval` | (none) | Keyframe interval in seconds |
| `--no-audio` | off | Strip audio |

#### Execution

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview transforms without executing |
| `--dry-run-frame N` | Render specific frame in preview |

### Per-Video Parameters

For multi-video files, use the `idx:` prefix to target specific videos:

```bash
sio transform multi_cam.slp \
    --crop 0:100,100,500,500 \
    --crop 1:200,200,600,600 \
    --scale 0.5 \
    -o processed.slp
```

### Config File

For complex scenarios, use a YAML config file:

```bash
sio transform labels.slp --config transforms.yaml -o output.slp
```

See the [Transforms Guide](transforms.md#config-file-format) for config file format and examples.

---

## Use Cases

### Inspecting an Unknown Labels File

When you receive a `.slp` file and want to understand its contents:

```bash
# Quick overview
sio show mystery_file.slp

# Full details
sio show mystery_file.slp --all
```

### Checking Video Status Before Processing

Before running inference or embedding, verify videos are accessible:

```bash
sio show labels.slp --video
```

Look for `[not found]` tags indicating missing videos.

### Fixing Broken Video Paths

If videos show `[not found]`, update the paths:

```bash
# Check current video paths
sio filenames labels.slp

# Update with new location
sio filenames labels.slp -o labels_fixed.slp \
    --prefix /old/location /new/location

# Verify fix
sio filenames labels_fixed.slp
```

### Extracting Skeleton Definition

Get copyable Python code for the skeleton:

```bash
sio show labels.slp
```

Copy the `nodes = [...]` and `edge_inds = [...]` lines directly into your code.

### Converting for Sharing

Create a portable package with embedded frames:

```bash
# Embed user-labeled frames for sharing training data
sio embed project.slp -o project.pkg.slp

# Include predictions and suggestions for complete sharing
sio embed project.slp -o project.pkg.slp --predictions --suggestions

# Later, restore to reference original videos
sio unembed project.pkg.slp -o project_restored.slp
```

### Batch Conversion with Shell Scripts

Convert multiple files using shell loops:

=== "Bash"
    ```bash
    # Convert all SLEAP files to NWB
    for f in *.slp; do
        sio convert -i "$f" -o "${f%.slp}.nwb"
    done
    ```

=== "PowerShell"
    ```powershell
    # Convert all SLEAP files to NWB
    Get-ChildItem *.slp | ForEach-Object {
        sio convert -i $_.Name -o ($_.BaseName + ".nwb")
    }
    ```

### Importing from Other Tools

Import pose data from other annotation and analysis tools:

```bash
# From COCO format (e.g., mmpose, CVAT exports)
sio convert annotations.json -o labels.slp --from coco

# From Label Studio export
sio convert annotations.json -o labels.slp --from labelstudio

# From DeepLabCut analysis (CSV format)
sio convert video_DLC_results.csv -o labels.slp --from dlc
```

### Exporting for Training

Export to formats used by other frameworks:

```bash
# For Ultralytics YOLO
sio convert labels.slp -o yolo_data/ --to ultralytics

# For tools expecting COCO format
sio convert labels.slp -o annotations.json --to coco

# For NWB-based pipelines (auto-detects annotations vs predictions)
sio convert labels.slp -o data.nwb --to nwb
```

### Rendering Pose Videos

Create video visualizations of your pose predictions:

```bash
# Quick preview to check predictions
sio render predictions.slp -o preview.mp4 --preset preview

# Final render with custom styling for publication
sio render predictions.slp -o final.mp4 \
    --color-by track \
    --palette tableau10 \
    --marker-shape diamond
```

### Creating Training Splits

Prepare datasets for machine learning with reproducible splits:

```bash
# Standard 80/10/10 split for training
sio split labels.slp -o experiment1/ --train 0.8 --test 0.1 --seed 42

# Remove predictions and embed frames for portable training data
sio split labels.slp -o training_data/ --remove-predictions --embed user --seed 42
```

The `--seed` option ensures you can recreate the exact same split later, which is essential for reproducible experiments. Note that the seed is sensitive to `--remove-predictions` since it changes the frame count—use the same options consistently.

!!! tip "NWB training annotations"
    The CLI uses auto-detection for NWB format. For explicit control over NWB format (e.g., `annotations` vs `predictions` vs `annotations_export` with embedded video), use the Python API:

    ```python
    import sleap_io as sio
    labels = sio.load_slp("labels.slp")
    sio.save_nwb(labels, "training.nwb", nwb_format="annotations")
    ```

    See [NWB Format](formats/#nwb-format-nwb) for details.

### Merging Training Splits

After training experiments, you may want to recombine split files:

```bash
# Merge all splits back into a single file
sio unsplit experiment1/ -o combined.slp

# Or specify files explicitly
sio unsplit train.slp val.slp test.slp -o combined.slp
```

This is useful for:

- Combining results after separate processing of train/val/test sets
- Reconstructing the original dataset from archived splits
- Merging predictions made on different splits

Videos are automatically deduplicated if the splits were created with `sio split --embed`.

### CI/CD Integration

Use `uvx` in CI pipelines without installation overhead:

```yaml
# GitHub Actions example
- name: Validate labels file
  run: uvx sleap-io show labels.slp

- name: Convert to NWB
  run: uvx sleap-io convert -i labels.slp -o labels.nwb

- name: Create package with embedded frames
  run: uvx sleap-io convert -i labels.slp -o labels.pkg.slp --embed user
```

---

## Version and Plugin Info

Check installed version and available backends:

```bash
sio --version
```

**Example output:**

```
sleap-io 0.6.0
python 3.12.11

Core:
  numpy: 2.4.0
  h5py: 3.15.1
  imageio: 2.37.2
  skia-python: 138.0
  colorcet: 3.1.0

Video plugins:
  opencv: 4.8.1
  pyav: 12.0.0
  imageio-ffmpeg: 0.6.0

Optional:
  pymatreader: 0.0.32
```

!!! tip "Troubleshooting video issues"
    If video-related operations fail, check `sio --version` to verify video plugins are installed. Look for `not installed` next to any video plugin.

---

## See Also

- [Formats](formats/): Detailed format specifications and Python API
- [Examples](examples.md): Python code examples for common tasks
- [Model](model.md): Data model documentation
