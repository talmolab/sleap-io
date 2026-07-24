# Guides

Task-oriented how-to guides for working with sleap-io. Start with **Examples** for
copy-paste recipes, then reach for the focused guides below.

**[Examples](examples.md)**: Practical recipes — loading and saving, editing
skeletons and tracks, fixing video paths, exporting to other formats, rendering,
and more.

**[Remote loading](remote.md)**: Load `.slp` files and videos straight from
`http`/`https`, cloud storage (`s3://`, `gs://`, `az://`), and Google Drive URLs,
with lazy range-based streaming, optional persistent caching, and authentication.

**[Codecs](codecs.md)**: Convert `Labels` to and from NumPy arrays and pandas
DataFrames using the array and table codecs.

**[Merging](merging.md)**: Combine datasets from multiple sources with track,
skeleton, instance, and video matching.

**[Rendering](rendering.md)**: Render videos and images with pose overlays,
customizable colors, markers, presets, and motion trails.

**[Transforms](transforms.md)**: Crop, scale, rotate, and pad labels and their
videos while keeping coordinates aligned.

!!! tip "Reference material"

    For the data structures these guides operate on, see the
    **[Data model](model/index.md)**; for format-specific details, see
    **[Formats](formats/index.md)**; and for the `sio` command line, see the
    **[CLI reference](cli.md)**.
