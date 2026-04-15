"""Tests for TIFF I/O of LabelImage data."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from sleap_io.io.tiff import read_label_images, write_label_images
from sleap_io.model.instance import Track
from sleap_io.model.label_image import LabelImage, UserLabelImage


def _make_label_array(h: int = 8, w: int = 8, n_objects: int = 2) -> np.ndarray:
    """Create a small synthetic label image array."""
    data = np.zeros((h, w), dtype=np.int32)
    for i in range(1, n_objects + 1):
        row = (i - 1) * (h // n_objects)
        data[row : row + h // n_objects, :] = i
    return data


def test_read_single_tiff(tmp_path):
    """Read a single TIFF, verify single LabelImage returned."""
    data = _make_label_array(8, 8, 2)
    tiff_path = tmp_path / "single.tif"
    tifffile.imwrite(str(tiff_path), data)

    result = read_label_images(tiff_path)

    assert len(result) == 1
    assert result[0].n_objects >= 0  # basic sanity check
    np.testing.assert_array_equal(result[0].data, data.astype(np.int32))
    assert result[0].n_objects == 2


def test_read_multipage_tiff(tmp_path):
    """Read a multi-page TIFF stack, verify frame count and data."""
    frames = [_make_label_array(8, 8, 2) for _ in range(3)]
    # Vary frame 1 to ensure distinct data
    frames[1] = np.zeros((8, 8), dtype=np.int32)
    frames[1][0:4, :] = 3

    tiff_path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(str(tiff_path)) as tw:
        for frame in frames:
            tw.write(frame)

    result = read_label_images(tiff_path, pages_as="time")

    assert len(result) == 3
    for i, li in enumerate(result):
        np.testing.assert_array_equal(li.data, frames[i].astype(np.int32))


def test_read_tiff_directory(tmp_path):
    """Read directory of per-frame TIFFs, verify sorted order."""
    frames = [_make_label_array(8, 8, 2) for _ in range(3)]
    # Write with zero-padded names
    for i, frame in enumerate(frames):
        tifffile.imwrite(str(tmp_path / f"{i:03d}.tif"), frame)

    result = read_label_images(tmp_path)

    assert len(result) == 3
    for i, li in enumerate(result):
        np.testing.assert_array_equal(li.data, frames[i].astype(np.int32))


def test_auto_track_creation(tmp_path):
    """Auto-create tracks when no tracks provided. Verify consistent Track objects."""
    # Frame 0 has objects 1, 2; Frame 1 has objects 1, 3
    frame0 = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    frame1 = np.array([[1, 1, 3, 3], [1, 1, 3, 3]], dtype=np.int32)

    tiff_path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(str(tiff_path)) as tw:
        tw.write(frame0)
        tw.write(frame1)

    result = read_label_images(tiff_path, pages_as="time")

    assert len(result) == 2

    # Track for label ID 1 should be the same object in both frames
    track1_frame0 = result[0].objects[1].track
    track1_frame1 = result[1].objects[1].track
    assert track1_frame0 is track1_frame1
    assert track1_frame0.name == "1"

    # Track for label ID 3 should only be in frame 1
    assert 3 not in result[0].objects
    track3_frame1 = result[1].objects[3].track
    assert track3_frame1.name == "3"

    # Track for label ID 2 should only be in frame 0
    assert 2 not in result[1].objects
    track2_frame0 = result[0].objects[2].track
    assert track2_frame0.name == "2"


def test_sidecar_roundtrip(tmp_path):
    """Write with sidecar, read back, verify track names and categories."""
    track_a = Track(name="cell_042")
    track_b = Track(name="cell_017")

    data = np.array([[1, 1, 3, 3], [1, 1, 3, 3]], dtype=np.int32)
    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(track=track_a, category="neuron"),
            3: LabelImage.Info(track=track_b, category="glia"),
        },
    )

    tiff_path = tmp_path / "labeled.tif"
    write_label_images(tiff_path, [li])

    # Verify sidecar was written
    sidecar_path = tmp_path / "labeled.tif.meta.json"
    assert sidecar_path.exists()
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    assert sidecar["format"] == "sleap-io-label-image-meta"
    assert sidecar["version"] == 3
    assert sidecar["axes"] == "YX"
    assert sidecar["objects"]["1"]["track"] == "cell_042"
    assert sidecar["objects"]["1"]["category"] == "neuron"
    assert sidecar["objects"]["3"]["track"] == "cell_017"
    assert sidecar["objects"]["3"]["category"] == "glia"

    # Read back and verify
    result = read_label_images(tiff_path)
    assert len(result) == 1
    li_read = result[0]
    assert li_read.objects[1].track.name == "cell_042"
    assert li_read.objects[1].category == "neuron"
    assert li_read.objects[3].track.name == "cell_017"
    assert li_read.objects[3].category == "glia"


def test_no_sidecar(tmp_path):
    """No sidecar: verify auto-created tracks with ID-based names."""
    data = np.array([[1, 2], [3, 0]], dtype=np.int32)
    tiff_path = tmp_path / "no_sidecar.tif"
    tifffile.imwrite(str(tiff_path), data)

    result = read_label_images(tiff_path)
    assert len(result) == 1
    li = result[0]

    # Tracks auto-created with names matching label IDs
    assert li.objects[1].track.name == "1"
    assert li.objects[2].track.name == "2"
    assert li.objects[3].track.name == "3"
    # No categories without sidecar
    assert li.objects[1].category == ""


def test_write_stack_roundtrip(tmp_path):
    """Write as stack, read back, verify round-trip."""
    frames_data = [
        np.array([[1, 2], [0, 3]], dtype=np.int32),
        np.array([[0, 1], [2, 0]], dtype=np.int32),
    ]
    track1 = Track(name="obj_a")
    track2 = Track(name="obj_b")
    track3 = Track(name="obj_c")

    label_images = []
    for i, data in enumerate(frames_data):
        objects = {}
        ids = np.unique(data)
        ids = ids[ids > 0]
        track_lookup = {1: track1, 2: track2, 3: track3}
        cat_lookup = {1: "cat_x", 2: "cat_y", 3: "cat_x"}
        for lid in ids:
            lid_int = int(lid)
            objects[lid_int] = LabelImage.Info(
                track=track_lookup[lid_int],
                category=cat_lookup[lid_int],
            )
        label_images.append(UserLabelImage(data=data, objects=objects))

    tiff_path = tmp_path / "roundtrip.tif"
    write_label_images(tiff_path, label_images, stack=True)
    result = read_label_images(tiff_path)

    assert len(result) == len(label_images)
    for i in range(len(result)):
        np.testing.assert_array_equal(result[i].data, frames_data[i])

    # Verify track names from sidecar
    assert result[0].objects[1].track.name == "obj_a"
    assert result[0].objects[2].track.name == "obj_b"
    assert result[0].objects[3].track.name == "obj_c"
    assert result[0].objects[1].category == "cat_x"
    assert result[0].objects[2].category == "cat_y"


def test_write_directory_roundtrip(tmp_path):
    """Write as directory, read back, verify round-trip."""
    frames_data = [
        np.array([[1, 0], [0, 2]], dtype=np.int32),
        np.array([[2, 2], [1, 1]], dtype=np.int32),
    ]
    track1 = Track(name="alpha")
    track2 = Track(name="beta")

    label_images = []
    for i, data in enumerate(frames_data):
        objects = {}
        ids = np.unique(data)
        ids = ids[ids > 0]
        track_lookup = {1: track1, 2: track2}
        for lid in ids:
            lid_int = int(lid)
            objects[lid_int] = LabelImage.Info(track=track_lookup[lid_int])
        label_images.append(UserLabelImage(data=data, objects=objects))

    dir_path = tmp_path / "frames_dir"
    write_label_images(dir_path, label_images, stack=False)

    # Verify directory structure
    assert dir_path.is_dir()
    tif_files = sorted(dir_path.glob("*.tif"))
    assert len(tif_files) == 2

    # Read back
    result = read_label_images(dir_path)

    assert len(result) == 2
    for i in range(len(result)):
        np.testing.assert_array_equal(result[i].data, frames_data[i])

    # Verify sidecar was written for the directory
    sidecar_path = tmp_path / "frames_dir.meta.json"
    assert sidecar_path.exists()

    # Verify track names roundtrip
    assert result[0].objects[1].track.name == "alpha"
    assert result[0].objects[2].track.name == "beta"


def test_empty_label_image_roundtrip(tmp_path):
    """Write and read back a LabelImage with all-zero data (no objects)."""
    data = np.zeros((4, 4), dtype=np.int32)
    li = UserLabelImage(data=data)

    tiff_path = tmp_path / "empty.tif"
    write_label_images(tiff_path, [li])
    result = read_label_images(tiff_path)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0].data, data)
    assert result[0].n_objects == 0
    assert len(result[0].objects) == 0


def test_tiff_spatial_metadata_roundtrip(tmp_path):
    """Write and read back a LabelImage with scale/offset via sidecar."""
    data = _make_label_array(8, 8, 2)
    li = UserLabelImage(data=data, scale=(0.5, 0.5), offset=(10.0, 20.0))

    tiff_path = tmp_path / "spatial.tif"
    write_label_images(tiff_path, [li])
    result = read_label_images(tiff_path)

    assert len(result) == 1
    assert result[0].scale == (0.5, 0.5)
    assert result[0].offset == (10.0, 20.0)
    assert result[0].has_spatial_transform is True


def test_tiff_default_spatial_roundtrip(tmp_path):
    """LabelImage with default scale/offset reads back with defaults."""
    data = _make_label_array(8, 8, 2)
    li = UserLabelImage(data=data)

    tiff_path = tmp_path / "default.tif"
    write_label_images(tiff_path, [li])
    result = read_label_images(tiff_path)

    assert len(result) == 1
    assert result[0].scale == (1.0, 1.0)
    assert result[0].offset == (0.0, 0.0)
    assert result[0].has_spatial_transform is False


def test_tiff_sidecar_v1_compat(tmp_path):
    """Old sidecar without scale/offset loads with defaults."""
    data = _make_label_array(8, 8, 2)
    li = UserLabelImage(data=data)

    tiff_path = tmp_path / "old.tif"
    write_label_images(tiff_path, [li])

    # Manually write a v1 sidecar (no scale/offset)
    sidecar_path = tmp_path / "old.tif.meta.json"
    sidecar = {"format": "sleap-io-label-image-meta", "version": 1, "objects": {}}
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)

    result = read_label_images(tiff_path)
    assert len(result) == 1
    assert result[0].scale == (1.0, 1.0)
    assert result[0].offset == (0.0, 0.0)


def _write_plain_multipage(path, pages):
    """Write a list of 2D arrays as a plain multi-page TIFF (no metadata)."""
    with tifffile.TiffWriter(str(path)) as tw:
        for p in pages:
            tw.write(p)


def _make_class_pages(h: int = 8, w: int = 8):
    """Three disjoint binary masks representing three classes."""
    p0 = np.zeros((h, w), dtype=np.uint8)
    p1 = np.zeros((h, w), dtype=np.uint8)
    p2 = np.zeros((h, w), dtype=np.uint8)
    p0[0:2, :] = 1
    p1[3:5, :] = 1
    p2[6:8, :] = 1
    return [p0, p1, p2]


def test_pages_as_classes_explicit(tmp_path):
    """pages_as='classes' composites pages into one LabelImage with label IDs 1..N."""
    pages = _make_class_pages()
    tiff_path = tmp_path / "classes.tif"
    _write_plain_multipage(tiff_path, pages)

    result = read_label_images(
        tiff_path,
        categories=["nuclei", "glia", "debris"],
        pages_as="classes",
    )

    assert len(result) == 1
    li = result[0]
    assert set(np.unique(li.data).tolist()) == {0, 1, 2, 3}
    assert li.objects[1].category == "nuclei"
    assert li.objects[2].category == "glia"
    assert li.objects[3].category == "debris"


def test_pages_as_time_explicit_overrides_metadata(tmp_path):
    """pages_as='time' forces time even if metadata says otherwise."""
    pages = _make_class_pages()
    tiff_path = tmp_path / "classes.tif"
    _write_plain_multipage(tiff_path, pages)
    # Sidecar tries to flag as classes...
    sidecar = {
        "format": "sleap-io-label-image-meta",
        "version": 3,
        "axes": "CYX",
        "objects": {},
    }
    with open(str(tiff_path) + ".meta.json", "w") as f:
        json.dump(sidecar, f)

    # ...but user forces time mode.
    result = read_label_images(tiff_path, pages_as="time")
    assert len(result) == 3


def test_sidecar_axes_routes_to_classes(tmp_path):
    """Sidecar 'axes': 'CYX' triggers class-stack loading."""
    pages = _make_class_pages()
    tiff_path = tmp_path / "classes.tif"
    _write_plain_multipage(tiff_path, pages)
    sidecar = {
        "format": "sleap-io-label-image-meta",
        "version": 3,
        "axes": "CYX",
        "objects": {
            "1": {"category": "nuclei"},
            "2": {"category": "glia"},
            "3": {"category": "debris"},
        },
    }
    with open(str(tiff_path) + ".meta.json", "w") as f:
        json.dump(sidecar, f)

    result = read_label_images(tiff_path)

    assert len(result) == 1
    assert result[0].objects[1].category == "nuclei"
    assert result[0].objects[2].category == "glia"
    assert result[0].objects[3].category == "debris"


def test_imagej_hyperstack_cyx_auto(tmp_path):
    """ImageJ hyperstack with channels metadata auto-routes to classes."""
    # ImageJ hyperstack layout: shape (channels, H, W) with imagej=True.
    pages = _make_class_pages()
    stack = np.stack(pages, axis=0)  # (C, H, W)
    tiff_path = tmp_path / "ij_cyx.tif"
    tifffile.imwrite(str(tiff_path), stack, imagej=True, metadata={"axes": "CYX"})

    result = read_label_images(tiff_path, categories=["nuclei", "glia", "debris"])
    assert len(result) == 1
    assert set(np.unique(result[0].data).tolist()) == {0, 1, 2, 3}
    assert result[0].objects[1].category == "nuclei"


def test_imagej_hyperstack_tyx_auto(tmp_path):
    """ImageJ hyperstack with time metadata routes to time (multi-frame)."""
    frames = [_make_label_array(8, 8, 2) for _ in range(4)]
    stack = np.stack(frames, axis=0).astype(np.uint16)
    tiff_path = tmp_path / "ij_tyx.tif"
    tifffile.imwrite(str(tiff_path), stack, imagej=True, metadata={"axes": "TYX"})

    result = read_label_images(tiff_path)
    assert len(result) == 4


def test_ambiguous_multipage_warns(tmp_path):
    """Plain multi-page TIFF with no metadata emits a UserWarning."""
    pages = _make_class_pages()
    tiff_path = tmp_path / "ambiguous.tif"
    _write_plain_multipage(tiff_path, pages)

    with pytest.warns(UserWarning, match="pages are time"):
        result = read_label_images(tiff_path)
    # Behavior is unchanged: still returns N frames.
    assert len(result) == 3


def test_ambiguous_singlepage_no_warning(tmp_path):
    """Single-page TIFF is unambiguous; no warning even without metadata."""
    data = _make_label_array(8, 8, 2)
    tiff_path = tmp_path / "single.tif"
    tifffile.imwrite(str(tiff_path), data)

    with warnings_as_errors():
        read_label_images(tiff_path)


def test_pages_as_invalid_raises(tmp_path):
    """Unknown pages_as value raises ValueError."""
    pages = _make_class_pages()
    tiff_path = tmp_path / "x.tif"
    _write_plain_multipage(tiff_path, pages)

    with pytest.raises(ValueError, match="pages_as"):
        read_label_images(tiff_path, pages_as="frames")


def test_categories_dict_class_mode(tmp_path):
    """Categories as dict[int,str] also works in class mode."""
    pages = _make_class_pages()
    tiff_path = tmp_path / "classes.tif"
    _write_plain_multipage(tiff_path, pages)

    result = read_label_images(
        tiff_path,
        categories={1: "nuclei", 2: "glia", 3: "debris"},
        pages_as="classes",
    )
    assert result[0].objects[1].category == "nuclei"
    assert result[0].objects[3].category == "debris"


def test_directory_pages_as_classes(tmp_path):
    """Directory of per-class TIFFs treated as one frame with classes."""
    pages = _make_class_pages()
    class_dir = tmp_path / "classes"
    class_dir.mkdir()
    for i, p in enumerate(pages):
        tifffile.imwrite(str(class_dir / f"{i:03d}.tif"), p)

    result = read_label_images(
        class_dir,
        categories=["a", "b", "c"],
        pages_as="classes",
    )
    assert len(result) == 1
    assert result[0].objects[1].category == "a"


def test_tcyx_splits_into_per_frame_class_stacks(tmp_path):
    """ImageJ TCYX hyperstack produces one LabelImage per time point."""
    T, C, H, W = 2, 3, 8, 8
    data = np.zeros((T, C, H, W), dtype=np.uint8)
    for t in range(T):
        data[t, 0, 0:2, :] = 1
        data[t, 1, 3:5, :] = 1
        data[t, 2, 6:8, :] = 1
    tiff_path = tmp_path / "tcyx.tif"
    tifffile.imwrite(str(tiff_path), data, imagej=True, metadata={"axes": "TCYX"})

    result = read_label_images(tiff_path, categories=["a", "b", "c"])
    assert len(result) == T
    for li in result:
        assert set(np.unique(li.data).tolist()) == {0, 1, 2, 3}
        assert li.objects[1].category == "a"
        assert li.objects[3].category == "c"


def test_class_pages_preserve_distinct_ids(tmp_path):
    """Pages stamped with distinct integer IDs (e.g. COCO 5/17/99) round-trip.

    When each per-class page has a single unique non-zero value and the
    values differ across pages, those values are preserved as label IDs
    rather than being renumbered positionally.
    """
    h, w = 8, 8
    p0 = np.zeros((h, w), dtype=np.uint8)
    p1 = np.zeros((h, w), dtype=np.uint8)
    p2 = np.zeros((h, w), dtype=np.uint8)
    p0[0:2, :] = 5
    p1[3:5, :] = 17
    p2[6:8, :] = 99

    tiff_path = tmp_path / "coco_ids.tif"
    _write_plain_multipage(tiff_path, [p0, p1, p2])

    result = read_label_images(
        tiff_path,
        categories=["car", "cat", "backpack"],
        pages_as="classes",
    )

    li = result[0]
    assert set(np.unique(li.data).tolist()) == {0, 5, 17, 99}
    assert li.objects[5].category == "car"
    assert li.objects[17].category == "cat"
    assert li.objects[99].category == "backpack"


def test_class_pages_binary_fall_back_to_positional(tmp_path):
    """Purely binary pages (all values in {0,1}) fall back to positional IDs."""
    pages = _make_class_pages()  # each page has only {0, 1}
    tiff_path = tmp_path / "binary.tif"
    _write_plain_multipage(tiff_path, pages)

    result = read_label_images(
        tiff_path, pages_as="classes", categories=["a", "b", "c"]
    )
    li = result[0]
    assert set(np.unique(li.data).tolist()) == {0, 1, 2, 3}


def test_class_pages_mixed_values_fall_back_to_positional(tmp_path):
    """A page with multiple distinct nonzero values disables ID preservation."""
    h, w = 8, 8
    p0 = np.zeros((h, w), dtype=np.uint8)
    p1 = np.zeros((h, w), dtype=np.uint8)
    p0[0:2, :] = 5
    p0[0, 0] = 7  # page 0 has two nonzero values -> ambiguous
    p1[3:5, :] = 17

    tiff_path = tmp_path / "mixed.tif"
    _write_plain_multipage(tiff_path, [p0, p1])

    result = read_label_images(tiff_path, pages_as="classes")
    # Fall back: positional IDs 1..N; original values are cast to bool.
    assert set(np.unique(result[0].data).tolist()) == {0, 1, 2}


def test_empty_directory_returns_empty_list(tmp_path):
    """Directory with no TIFFs returns an empty list."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert read_label_images(empty_dir) == []


def test_directory_rejects_3d_file(tmp_path):
    """A 3D TIFF inside a directory raises ValueError."""
    arr = np.zeros((3, 8, 8), dtype=np.uint8)
    d = tmp_path / "d"
    d.mkdir()
    tifffile.imwrite(str(d / "bad.tif"), arr)
    with pytest.raises(ValueError, match="2D"):
        read_label_images(d)


def test_ome_cyx_metadata(tmp_path):
    """OME-TIFF with CYX axes auto-routes to class mode."""
    pages = _make_class_pages()
    stack = np.stack(pages, axis=0)
    tiff_path = tmp_path / "ome_cyx.ome.tif"
    tifffile.imwrite(str(tiff_path), stack, ome=True, metadata={"axes": "CYX"})

    result = read_label_images(tiff_path, categories=["a", "b", "c"])
    assert len(result) == 1
    assert result[0].objects[1].category == "a"


def test_write_empty_label_images_is_noop(tmp_path):
    """write_label_images on an empty list does nothing and doesn't error."""
    out = tmp_path / "noop.tif"
    write_label_images(out, [])
    assert not out.exists()


def test_3d_single_page_still_rejected(tmp_path):
    """A single 3D page (not multi-page) still raises ValueError."""
    arr = np.zeros((3, 8, 8), dtype=np.uint8)
    tiff_path = tmp_path / "3d.tif"
    tifffile.imwrite(str(tiff_path), arr)  # writes as 1 page of shape (3,8,8)

    with pytest.raises(ValueError, match="2D"):
        read_label_images(tiff_path, pages_as="time")


class warnings_as_errors:
    """Context manager that turns warnings into errors for scoped assertions."""

    def __enter__(self):
        """Enter scope; escalate warnings to errors."""
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *args):
        """Exit scope; restore prior warning filters."""
        return self._ctx.__exit__(*args)
