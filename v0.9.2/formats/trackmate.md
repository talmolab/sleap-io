# TrackMate CSV Format

[TrackMate](https://imagej.net/plugins/trackmate/) is an ImageJ/Fiji plugin for
single-particle tracking in microscopy images. It exports tracking results as a
set of CSV files. sleap-io reads these exports and converts spot detections into
[`PredictedCentroid`][sleap_io.PredictedCentroid] objects with track assignments.

## File Structure

TrackMate exports three CSV files per video, sharing a common prefix:

| File | Description | Required |
|------|-------------|----------|
| `*_spots.csv` | Individual spot detections with coordinates, quality score, and track assignment | **Yes** |
| `*_edges.csv` | Frame-to-frame linkages with assignment cost | No (provides `tracking_score`) |
| `*_tracks.csv` | Track-level summary statistics | No (not used) |

All CSV files have **4 header rows** (field names, descriptions, abbreviations,
units) followed by data rows.

## Auto-Detection

sleap-io automatically detects TrackMate CSV files by checking for the column
signature `LABEL, ID, TRACK_ID, QUALITY, POSITION_X, POSITION_Y` in the first
row. When loading a spots file:

- The sibling `*_edges.csv` is auto-detected (by replacing `_spots` with
  `_edges` in the filename) and used to populate `tracking_score` on each
  centroid.
- A sibling `.tif` / `.tiff` video file is auto-detected (by stripping `_spots`
  from the stem) and associated with the loaded data.

## Data Mapping

| TrackMate field | sleap-io field |
|-----------------|----------------|
| `POSITION_X`, `POSITION_Y` | `PredictedCentroid.x`, `.y` |
| `POSITION_Z` | `PredictedCentroid.z` (`None` if 0.0) |
| `QUALITY` | `PredictedCentroid.score` |
| `LINK_COST` (edges) | `PredictedCentroid.tracking_score` |
| `TRACK_ID` | `Track` (named `Track_<id>`) |
| `LABEL` | `PredictedCentroid.name` |
| `FRAME` | `LabeledFrame.frame_idx` |

## Reading

```python
import sleap_io as sio

# Load from spots CSV (edges and video auto-detected)
labels = sio.load_trackmate("experiment_spots.csv")

# Auto-detection via load_file
labels = sio.load_file("experiment_spots.csv")

# With explicit video path
labels = sio.load_trackmate("experiment_spots.csv", video="experiment.tif")
```

## CLI

TrackMate CSV files can be converted to other formats via the `sio convert`
command. The format is auto-detected from CSV content:

```bash
# Auto-detected from CSV content
sio convert experiment_spots.csv -o experiment.slp

# Explicit format
sio convert experiment_spots.csv -o experiment.slp --from trackmate

# Convert to NWB
sio convert experiment_spots.csv -o experiment.nwb --from trackmate
```

## API

::: sleap_io.io.main.load_trackmate
    options:
      heading_level: 3
      show_root_toc_entry: false

::: sleap_io.io.trackmate.read_trackmate_csv
    options:
      heading_level: 3
      show_root_toc_entry: false

::: sleap_io.io.trackmate.is_trackmate_file
    options:
      heading_level: 3
      show_root_toc_entry: false
