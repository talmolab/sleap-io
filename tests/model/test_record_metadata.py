"""Tests for the sleap_io.model.record_metadata module."""

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from sleap_io import Instance, LabeledFrame, Skeleton, Video
from sleap_io.model.bbox import UserBoundingBox
from sleap_io.model.record_metadata import (
    RecordMetadata,
    merge_metadata,
    metadata_column_kinds,
    metadata_kind,
    normalize_metadata_value,
    to_metadata,
)


def test_normalize_metadata_value_scalars():
    """Python scalars pass through and numpy scalars become Python scalars."""
    assert normalize_metadata_value(None) is None
    assert normalize_metadata_value(True) is True
    assert normalize_metadata_value("a") == "a"
    assert normalize_metadata_value(3) == 3
    assert normalize_metadata_value(2.5) == 2.5

    for value, expected_type in [
        (np.bool_(True), bool),
        (np.int64(3), int),
        (np.uint8(3), int),
        (np.float32(2.5), float),
        (np.str_("a"), str),
    ]:
        normalized = normalize_metadata_value(value)
        assert type(normalized) is expected_type
        assert normalized == value


@pytest.mark.parametrize("value", [[1, 2], {"a": 1}, (1,), np.array([1]), b"x"])
def test_normalize_metadata_value_rejects_nested(value):
    """Nested and non-scalar values raise."""
    with pytest.raises(TypeError, match="JSON-encode"):
        normalize_metadata_value(value)


def test_metadata_kind():
    """Each normalized value maps to one kind, with bool distinct from int."""
    assert metadata_kind(True) == "bool"
    assert metadata_kind(1) == "int"
    assert metadata_kind(1.0) == "float"
    assert metadata_kind("1") == "str"


def test_record_metadata_validates_on_set():
    """Invalid keys and values raise when they are set."""
    md = RecordMetadata({"area": np.float64(1.5)}, iscrowd=np.bool_(False))
    assert md == {"area": 1.5, "iscrowd": False}
    assert type(md["iscrowd"]) is bool

    with pytest.raises(TypeError):
        md["bbox"] = [1, 2, 3, 4]
    with pytest.raises(TypeError):
        md.update({"bbox": [1, 2]})
    with pytest.raises(TypeError):
        md.setdefault("bbox", {"x": 1})
    with pytest.raises(TypeError):
        md |= {"bbox": [1]}
    with pytest.raises(TypeError, match="strings"):
        md[1] = 1
    with pytest.raises(ValueError, match="non-empty"):
        md[""] = 1
    with pytest.raises(ValueError, match="'/'"):
        md["a/b"] = 1
    assert "bbox" not in md


@pytest.mark.parametrize("key", [".", ".."])
def test_record_metadata_rejects_dot_keys(key):
    """`.` and `..` can't be stored as HDF5 group names, so they are rejected."""
    md = RecordMetadata()
    with pytest.raises(ValueError, match="'.' or '..'"):
        md[key] = 1
    with pytest.raises(ValueError, match="'.' or '..'"):
        RecordMetadata({key: 1})
    assert md == {}
    # Keys that only contain dots are fine.
    md["..."] = 1
    md[".a"] = 2
    assert md == {"...": 1, ".a": 2}


def test_record_metadata_rejects_nul_keys():
    """HDF5 truncates names at a NUL character, so such keys are rejected."""
    md = RecordMetadata()
    with pytest.raises(ValueError, match="NUL"):
        md["a\x00b"] = 1
    assert md == {}


@pytest.mark.parametrize("key", [".", ".."])
def test_metadata_on_records_rejects_dot_keys(key):
    """Instance, frame and box metadata reject `.` and `..` keys when set."""
    inst = Instance.empty(Skeleton(["A"]))
    lf = LabeledFrame(video=Video("test.mp4"), frame_idx=0)
    box = UserBoundingBox(x1=0, y1=0, x2=1, y2=1)
    for record in (inst, lf, box):
        with pytest.raises(ValueError, match="'.' or '..'"):
            record.metadata[key] = 1
        with pytest.raises(ValueError, match="'.' or '..'"):
            record.metadata = {key: 1}
        assert record.metadata == {}


def test_record_metadata_dict_methods():
    """setdefault, |= and copy keep the validated type."""
    md = RecordMetadata()
    assert md.setdefault("id", np.int32(7)) == 7
    assert md.setdefault("id", 8) == 7
    md |= {"name": "a"}
    assert md == {"id": 7, "name": "a"}

    md_copy = md.copy()
    assert isinstance(md_copy, RecordMetadata)
    assert md_copy == md and md_copy is not md


def test_record_metadata_none_means_missing():
    """None is stored as a missing value."""
    md = RecordMetadata(score=None)
    assert md["score"] is None


def test_record_metadata_pickle_and_copy():
    """Pickling and copying keep the type and items."""
    md = RecordMetadata(a=1, b="x")
    for restored in (pickle.loads(pickle.dumps(md)), copy.deepcopy(md)):
        assert isinstance(restored, RecordMetadata)
        assert restored == md


def test_to_metadata():
    """The converter accepts mappings and None and rejects other types."""
    assert to_metadata(None) == {}
    md = RecordMetadata(a=1)
    assert to_metadata(md) is md
    assert isinstance(to_metadata({"a": 1}), RecordMetadata)
    with pytest.raises(TypeError, match="dict"):
        to_metadata([("a", 1)])
    with pytest.raises(TypeError):
        to_metadata({"a": [1]})


def test_merge_metadata_target_wins():
    """Merging combines keys, the target winning ties."""
    merged = merge_metadata({"a": 1, "b": 2}, {"b": 20, "c": 30})
    assert merged == {"a": 1, "b": 2, "c": 30}
    assert isinstance(merged, RecordMetadata)


def _records(*values):
    """Make records whose metadata key "k" holds each value."""
    return [SimpleNamespace(metadata=RecordMetadata(k=value)) for value in values]


@pytest.mark.parametrize(
    "values,kind",
    [
        ((True, False), "bool"),
        ((1, 2), "int"),
        ((1.5, 2.5), "float"),
        (("a", "b"), "str"),
        ((1, 2.5), "float"),
        ((2.5, 1), "float"),
        ((None, 1), "int"),
        ((2**60, 3), "int"),
    ],
)
def test_metadata_column_kinds(values, kind):
    """Records sharing a kind (or mixing int and float) give one kind per key."""
    assert metadata_column_kinds(_records(*values), "instances") == {"k": kind}


def test_metadata_column_kinds_all_none():
    """Keys whose values are all None have no kind."""
    assert metadata_column_kinds(_records(None, None), "instances") == {}


@pytest.mark.parametrize(
    "values", [(True, 1), (1, "1"), (1.0, "a"), (False, 0.0), ("a", True)]
)
def test_metadata_column_kinds_rejects_mixes(values):
    """Mixing kinds other than int with float raises."""
    with pytest.raises(TypeError, match="mixes"):
        metadata_column_kinds(_records(*values), "instances")


@pytest.mark.parametrize("values", [(2**60, 1.5), (1.5, 2**60), (-(2**60), 0.5)])
def test_metadata_column_kinds_rejects_big_int_float_mix(values):
    """Integers beyond 2^53 can't be widened to float."""
    with pytest.raises(TypeError, match="2\\^53"):
        metadata_column_kinds(_records(*values), "instances")


def test_metadata_on_records():
    """Instances, frames and boxes validate metadata on construction and set."""
    skel = Skeleton(["A"])
    inst = Instance.empty(skel)
    assert inst.metadata == {}
    inst.metadata["ann_id"] = np.int64(5)
    assert type(inst.metadata["ann_id"]) is int

    inst.metadata = {"iscrowd": False}
    assert isinstance(inst.metadata, RecordMetadata)
    with pytest.raises(TypeError):
        inst.metadata = {"bbox": [0, 0, 1, 1]}

    lf = LabeledFrame(video=Video("test.mp4"), frame_idx=0, metadata={"image_id": 1})
    assert isinstance(lf.metadata, RecordMetadata)
    with pytest.raises(TypeError):
        lf.metadata["file"] = {"name": "a"}

    box = UserBoundingBox(x1=0, y1=0, x2=1, y2=1, metadata={"area": 1.0})
    assert box.metadata == {"area": 1.0}
    with pytest.raises(TypeError):
        UserBoundingBox(x1=0, y1=0, x2=1, y2=1, metadata={"bbox": [1]})


def test_record_metadata_shallow_copy():
    """`copy.copy` returns an independent `RecordMetadata`."""
    md = RecordMetadata(a=1)
    md_copy = copy.copy(md)
    md_copy["b"] = 2
    assert isinstance(md_copy, RecordMetadata)
    assert md == {"a": 1}


def test_set_attrs_state_defaults_plain_fields():
    """Fields missing from a state get their plain (non-factory) defaults too."""
    inst = Instance.empty(Skeleton(["A"]), tracking_score=0.5)
    state = inst.__getstate__()
    del state["tracking_score"]
    restored = Instance.__new__(Instance)
    restored.__setstate__(state)
    assert restored.tracking_score is None
