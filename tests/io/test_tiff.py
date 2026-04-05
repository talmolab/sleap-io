"""Tests for TIFF I/O of LabelImage data."""

from __future__ import annotations

import json

import numpy as np
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
    assert result[0].frame_idx == 0
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

    result = read_label_images(tiff_path)

    assert len(result) == 3
    for i, li in enumerate(result):
        assert li.frame_idx == i
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
        assert li.frame_idx == i
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

    result = read_label_images(tiff_path)

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
        frame_idx=0,
    )

    tiff_path = tmp_path / "labeled.tif"
    write_label_images(tiff_path, [li])

    # Verify sidecar was written
    sidecar_path = tmp_path / "labeled.tif.meta.json"
    assert sidecar_path.exists()
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    assert sidecar["format"] == "sleap-io-label-image-meta"
    assert sidecar["version"] == 1
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
        label_images.append(UserLabelImage(data=data, objects=objects, frame_idx=i))

    tiff_path = tmp_path / "roundtrip.tif"
    write_label_images(tiff_path, label_images, stack=True)
    result = read_label_images(tiff_path)

    assert len(result) == len(label_images)
    for i in range(len(result)):
        np.testing.assert_array_equal(result[i].data, frames_data[i])
        assert result[i].frame_idx == i

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
        label_images.append(UserLabelImage(data=data, objects=objects, frame_idx=i))

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
        assert result[i].frame_idx == i

    # Verify sidecar was written for the directory
    sidecar_path = tmp_path / "frames_dir.meta.json"
    assert sidecar_path.exists()

    # Verify track names roundtrip
    assert result[0].objects[1].track.name == "alpha"
    assert result[0].objects[2].track.name == "beta"


def test_empty_label_image_roundtrip(tmp_path):
    """Write and read back a LabelImage with all-zero data (no objects)."""
    data = np.zeros((4, 4), dtype=np.int32)
    li = UserLabelImage(data=data, frame_idx=0)

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
    li = UserLabelImage(data=data, frame_idx=0, scale=(0.5, 0.5), offset=(10.0, 20.0))

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
    li = UserLabelImage(data=data, frame_idx=0)

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
    li = UserLabelImage(data=data, frame_idx=0)

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
