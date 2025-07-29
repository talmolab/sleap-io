"""Tests for sleap_io.model.labels_set module."""

import pytest

from sleap_io import load_slp
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.labels import Labels
from sleap_io.model.labels_set import LabelsSet
from sleap_io.model.video import Video


def test_labels_set_creation(slp_minimal):
    """Test creating a LabelsSet with basic functionality."""
    # Load Labels from fixture
    labels = load_slp(slp_minimal)

    # Create from dict
    labels_set = LabelsSet({"split1": labels, "split2": labels})

    assert len(labels_set) == 2
    assert "split1" in labels_set
    assert "split2" in labels_set
    assert "split3" not in labels_set


def test_labels_set_dict_access(slp_minimal):
    """Test dictionary-style access to LabelsSet."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet()

    # Test setitem
    labels_set["train"] = labels
    labels_set["val"] = labels

    # Test getitem
    assert labels_set["train"] == labels
    assert labels_set["val"] == labels

    # Test KeyError
    with pytest.raises(KeyError):
        _ = labels_set["test"]

    # Test delitem
    del labels_set["val"]
    assert len(labels_set) == 1
    assert "val" not in labels_set


def test_labels_set_tuple_unpacking(slp_minimal):
    """Test tuple-style unpacking of LabelsSet."""
    labels = load_slp(slp_minimal)

    # Create a two-item set
    labels_set = LabelsSet({"train": labels, "val": labels})

    # Test unpacking
    train, val = labels_set
    assert train == labels
    assert val == labels

    # Test three-item unpacking
    labels_set["test"] = labels
    train, val, test = labels_set
    assert test == labels


def test_labels_set_index_access(slp_minimal):
    """Test integer index access for tuple-like behavior."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet({"first": labels, "second": labels})

    # Test index access
    assert labels_set[0] == labels
    assert labels_set[1] == labels

    # Test IndexError
    with pytest.raises(IndexError):
        _ = labels_set[2]


def test_labels_set_iteration(slp_minimal):
    """Test iteration over LabelsSet."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet({"train": labels, "val": labels, "test": labels})

    # Test that iteration yields Labels objects, not keys
    labels_list = list(labels_set)
    assert len(labels_list) == 3
    assert all(isinstance(label, Labels) for label in labels_list)

    # Test items iteration
    for name, label in labels_set.items():
        assert isinstance(name, str)
        assert isinstance(label, Labels)


def test_labels_set_dict_methods(slp_minimal):
    """Test dictionary-like methods of LabelsSet."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet({"train": labels, "val": labels})

    # Test keys()
    keys = list(labels_set.keys())
    assert keys == ["train", "val"]

    # Test values()
    values = list(labels_set.values())
    assert len(values) == 2
    assert all(v == labels for v in values)

    # Test items()
    items = list(labels_set.items())
    assert len(items) == 2
    assert items[0][0] == "train"
    assert items[1][0] == "val"

    # Test get()
    assert labels_set.get("train") == labels
    assert labels_set.get("missing") is None
    assert labels_set.get("missing", "default") == "default"


def test_labels_set_type_validation(slp_minimal):
    """Test type validation in LabelsSet."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet()

    # Test invalid key type
    with pytest.raises(TypeError, match="Key must be a string"):
        labels_set[123] = labels

    # Test invalid value type
    with pytest.raises(TypeError, match="Value must be a Labels object"):
        labels_set["train"] = "not a labels object"


def test_labels_set_repr(slp_minimal):
    """Test string representation of LabelsSet."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet({"train": labels, "val": labels})

    repr_str = repr(labels_set)
    assert "LabelsSet(" in repr_str
    assert "train:" in repr_str
    assert "val:" in repr_str
    assert "labeled frames" in repr_str


def test_labels_set_save(tmp_path, slp_minimal):
    """Test saving LabelsSet to directory."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet({"train": labels, "val": labels, "test": labels})

    # Test save without embedding (default would try to embed which fails for
    # missing videos)
    save_dir_no_embed = tmp_path / "splits_no_embed"
    labels_set.save(save_dir_no_embed, embed=False)

    assert save_dir_no_embed.exists()
    assert (save_dir_no_embed / "train.slp").exists()
    assert (save_dir_no_embed / "val.slp").exists()
    assert (save_dir_no_embed / "test.slp").exists()

    # Test can load back the saved files
    train_labels = load_slp(save_dir_no_embed / "train.slp")
    assert isinstance(train_labels, Labels)
    assert len(train_labels) == len(labels)


def test_labels_set_from_labels_lists(slp_minimal):
    """Test creating LabelsSet from list of Labels."""
    labels = load_slp(slp_minimal)

    # Create from list with auto-generated names
    labels_list = [labels, labels, labels]
    labels_set = LabelsSet.from_labels_lists(labels_list)

    assert len(labels_set) == 3
    assert "split1" in labels_set
    assert "split2" in labels_set
    assert "split3" in labels_set

    # Create from list with custom names
    names = ["train", "val", "test"]
    labels_set = LabelsSet.from_labels_lists(labels_list, names=names)

    assert len(labels_set) == 3
    assert "train" in labels_set
    assert "val" in labels_set
    assert "test" in labels_set

    # Test mismatched lengths
    with pytest.raises(ValueError, match="Number of names"):
        LabelsSet.from_labels_lists(labels_list, names=["train", "val"])


def test_labels_set_empty():
    """Test empty LabelsSet behavior."""
    labels_set = LabelsSet()

    assert len(labels_set) == 0
    assert list(labels_set) == []
    assert list(labels_set.keys()) == []
    assert list(labels_set.values()) == []
    assert list(labels_set.items()) == []
    assert repr(labels_set) == "LabelsSet()"


def test_labels_set_single_item_unpacking(slp_minimal):
    """Test unpacking a single-item LabelsSet."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet({"only": labels})

    # Single item unpacking
    (only,) = labels_set
    assert only == labels

    # Also test with brackets
    [only] = labels_set
    assert only == labels


def test_labels_set_order_preservation(slp_minimal):
    """Test that LabelsSet preserves insertion order."""
    labels = load_slp(slp_minimal)

    # Create labels set with specific order
    labels_set = LabelsSet()
    labels_set["third"] = labels
    labels_set["first"] = labels
    labels_set["second"] = labels

    # Check that keys maintain insertion order
    keys = list(labels_set.keys())
    assert keys == ["third", "first", "second"]

    # Check that unpacking maintains insertion order
    a, b, c = labels_set
    assert a == labels_set["third"]
    assert b == labels_set["first"]
    assert c == labels_set["second"]


def test_labels_set_with_different_labels():
    """Test LabelsSet with different Labels objects."""
    # Create different Labels objects
    labels1 = Labels()
    labels2 = Labels()
    labels3 = Labels()

    # Add different numbers of frames to each
    video = Video.from_filename("fake_video.mp4")

    # Add 1 frame to labels1
    labels1.append(LabeledFrame(video=video, frame_idx=0))

    # Add 2 frames to labels2
    labels2.append(LabeledFrame(video=video, frame_idx=0))
    labels2.append(LabeledFrame(video=video, frame_idx=1))

    # Add 3 frames to labels3
    labels3.append(LabeledFrame(video=video, frame_idx=0))
    labels3.append(LabeledFrame(video=video, frame_idx=1))
    labels3.append(LabeledFrame(video=video, frame_idx=2))

    # Create LabelsSet
    labels_set = LabelsSet({"small": labels1, "medium": labels2, "large": labels3})

    # Test that each has correct size
    assert len(labels_set["small"]) == 1
    assert len(labels_set["medium"]) == 2
    assert len(labels_set["large"]) == 3

    # Test repr shows different sizes
    repr_str = repr(labels_set)
    assert "small: 1 labeled frames" in repr_str
    assert "medium: 2 labeled frames" in repr_str
    assert "large: 3 labeled frames" in repr_str


def test_labels_set_save_ultralytics(tmp_path):
    """Test saving LabelsSet in ultralytics format."""
    # Create simple labels without video dependencies
    from sleap_io import Instance, LabeledFrame, Labels, Node, Skeleton, Video

    # Create a simple skeleton
    skeleton = Skeleton([Node("node1"), Node("node2")])

    # Create labels with a fake video
    labels = Labels(skeletons=[skeleton])
    video = Video.from_filename("fake_video.mp4")

    # Add a labeled frame
    lf = LabeledFrame(video=video, frame_idx=0)
    instance = Instance.from_numpy([[10, 20], [30, 40]], skeleton=skeleton)
    lf.instances.append(instance)
    labels.append(lf)

    labels_set = LabelsSet({"train": labels, "val": labels, "test": labels})

    # Save as ultralytics format
    save_dir = tmp_path / "yolo_dataset"
    labels_set.save(save_dir, format="ultralytics", verbose=False)

    # Check that the directory structure was created
    assert save_dir.exists()
    assert (save_dir / "data.yaml").exists()

    # Test split name mapping
    labels_set_mapped = LabelsSet(
        {"training": labels, "validation": labels, "testing": labels}
    )
    save_dir_mapped = tmp_path / "yolo_dataset_mapped"
    labels_set_mapped.save(save_dir_mapped, format="ultralytics", verbose=False)

    # Check data.yaml was created
    assert (save_dir_mapped / "data.yaml").exists()


def test_labels_set_save_invalid_format(tmp_path, slp_minimal):
    """Test error handling for invalid save format."""
    labels = load_slp(slp_minimal)
    labels_set = LabelsSet({"train": labels})

    # Test invalid format
    with pytest.raises(ValueError, match="Unknown format: invalid_format"):
        labels_set.save(tmp_path, format="invalid_format")
