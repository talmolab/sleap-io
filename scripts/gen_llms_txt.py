"""Generate ``llms.txt`` — a machine-readable index for LLM / agent consumers.

Wired into the mkdocs ``gen-files`` plugin (see ``mkdocs.yml``). Produces an
``llms.txt`` at the site root following the https://llmstxt.org convention: a
one-line description, curated links to the documentation pages (the site also
serves every page as raw markdown at the same path with a ``.md`` suffix, via
``hooks/copy_source_markdown.py``), and the public ``sleap_io`` API surface so an
agent can discover available functions and classes without crawling the site.

Links are relative so they resolve correctly under each ``mike``-deployed version
directory (e.g. ``/latest/examples.md``).
"""

import mkdocs_gen_files

import sleap_io as sio

DESCRIPTION = (
    "Standalone Python library for reading, writing, and manipulating animal "
    "pose-tracking data. Reads/writes SLEAP, NWB, DeepLabCut, COCO, Label Studio, "
    "JABS, Ultralytics, TrackMate, and more, with data-structure manipulation and "
    "video I/O. Minimal dependencies; no labeling, training, or inference."
)

GUIDES = [
    ("Examples", "examples.md", "Practical recipes for common tasks"),
    (
        "Remote loading",
        "remote.md",
        "Load .slp and videos from URLs, S3/GCS/Azure, and Google Drive",
    ),
    ("Codecs", "codecs.md", "Array conversion (NumPy / pandas) formats"),
    ("Merging", "merging.md", "Combine datasets; track / skeleton / video matching"),
    (
        "Rendering",
        "rendering.md",
        "Render videos and images with overlays, colors, and motion trails",
    ),
    ("Transforms", "transforms.md", "Crop, scale, rotate, and pad labels and videos"),
]
MODEL = [
    (
        "Data model overview",
        "model/index.md",
        "Containers and how annotations nest on frames",
    ),
    ("Labels", "model/labels.md", "Labels, LabeledFrame, LabelsSet containers"),
    ("Video", "model/video.md", "Lazy video backends"),
    ("Poses", "model/poses.md", "Skeleton, Node, Edge, Instance, Track"),
    ("3D", "model/3d.md", "Camera, RecordingSession, 3D instances"),
    ("Centroids", "model/centroids.md", "Point detections"),
    ("Boxes", "model/boxes.md", "Bounding boxes"),
    ("ROIs", "model/rois.md", "Vector polygon regions"),
    ("Segmentation", "model/segmentation.md", "Segmentation masks and label images"),
]
FORMATS = [
    (
        "Formats overview",
        "formats/index.md",
        "All supported I/O formats and their limitations",
    ),
    ("SLP", "formats/slp.md", "Native SLEAP format"),
    ("CSV", "formats/csv.md", "Flat tabular export/import"),
    ("Analysis HDF5", "formats/analysis_h5.md", "SLEAP analysis arrays"),
    ("TIFF", "formats/tiff.md", "Label images"),
    ("TrackMate", "formats/trackmate.md", "ImageJ/Fiji point tracking"),
]
REFERENCE = [
    ("Installation", "install.md", "Install options and extras"),
    ("CLI", "cli.md", "`sio` command-line reference"),
    ("Full API", "reference/", "Auto-generated API documentation"),
]

# Namespace re-exports that are modules, not callables/classes.
SUBMODULES = {"io", "model", "codecs"}


def _section(title: str, items: list[tuple[str, str, str]]) -> str:
    lines = [f"## {title}", ""]
    for name, url, desc in items:
        lines.append(f"- [{name}]({url}): {desc}" if desc else f"- [{name}]({url})")
    lines.append("")
    return "\n".join(lines)


def _api_section() -> str:
    names = set(sio.__all__) - {"__version__"} - SUBMODULES
    loaders = sorted(n for n in names if n.startswith("load_"))
    savers = sorted(n for n in names if n.startswith("save_"))
    remaining = names - set(loaders) - set(savers)
    classes = sorted(n for n in remaining if n[:1].isupper())
    functions = sorted(remaining - set(classes))

    def fmt(group: list[str]) -> str:
        return ", ".join(f"`sio.{n}`" for n in group)

    return "\n".join(
        [
            "## Public API",
            "",
            "`import sleap_io as sio` exposes:",
            "",
            f"- **Loaders**: {fmt(loaders)}",
            f"- **Savers**: {fmt(savers)}",
            f"- **Classes**: {fmt(classes)}",
            f"- **Functions**: {fmt(functions)}",
            "",
        ]
    )


content = "\n".join(
    [
        "# sleap-io",
        "",
        f"> {DESCRIPTION}",
        "",
        "Every documentation page is also served as raw markdown at the same path "
        "with a `.md` suffix (e.g. `examples.md`, `model/labels.md`).",
        "",
        _section("Guides", GUIDES),
        _section("Data model", MODEL),
        _section("Formats", FORMATS),
        _section("Reference", REFERENCE),
        _api_section(),
    ]
)

with mkdocs_gen_files.open("llms.txt", "w") as f:
    f.write(content)
