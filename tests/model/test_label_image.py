"""Tests for LabelImage data model."""

import numpy as np
import pytest

from sleap_io.model.instance import Track
from sleap_io.model.label_image import (
    LabelImage,
    PredictedLabelImage,
    UserLabelImage,
)
from sleap_io.model.mask import UserSegmentationMask
from sleap_io.model.video import Video


def test_info_defaults():
    """Info should have sensible defaults."""
    info = LabelImage.Info()
    assert info.track is None
    assert info.category == ""
    assert info.name == ""
    assert info.instance is None


def test_info_with_fields():
    """Info should accept all fields."""
    track = Track(name="t1")
    info = LabelImage.Info(track=track, category="neuron", name="cell_1")
    assert info.track is track
    assert info.category == "neuron"
    assert info.name == "cell_1"


def test_construction_basic():
    """UserLabelImage should be constructable with a 2D array."""
    data = np.zeros((10, 20), dtype=np.int32)
    data[2:5, 3:7] = 1
    li = UserLabelImage(data=data)
    assert li.height == 10
    assert li.width == 20
    assert li.n_objects == 1


def test_construction_dtype_cast():
    """UserLabelImage should cast data to int32."""
    data = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    li = UserLabelImage(data=data)
    assert li.data.dtype == np.int32


def test_construction_non_2d_raises():
    """UserLabelImage should raise ValueError for non-2D data."""
    with pytest.raises(ValueError, match="must be 2D"):
        UserLabelImage(data=np.zeros((3, 4, 5), dtype=np.int32))


def test_construction_negative_raises():
    """UserLabelImage should raise ValueError for negative values."""
    with pytest.raises(ValueError, match="negative"):
        UserLabelImage(data=np.array([[0, -1]], dtype=np.int32))


def test_construction_abstract_raises():
    """LabelImage base class should raise TypeError."""
    with pytest.raises(TypeError, match="abstract"):
        LabelImage(data=np.zeros((2, 2), dtype=np.int32))


def test_properties():
    """Test height, width, n_objects, label_ids."""
    data = np.zeros((8, 12), dtype=np.int32)
    data[0:2, 0:2] = 3
    data[4:6, 4:6] = 7
    li = UserLabelImage(data=data)

    assert li.height == 8
    assert li.width == 12
    assert li.n_objects == 2
    np.testing.assert_array_equal(li.label_ids, [3, 7])


def test_tracks_property():
    """Tracks property should return tracks from objects metadata."""
    t1, t2 = Track(name="a"), Track(name="b")
    li = UserLabelImage(
        data=np.array([[1, 2]], dtype=np.int32),
        objects={
            1: LabelImage.Info(track=t1),
            2: LabelImage.Info(track=t2),
        },
    )
    assert li.tracks == [t1, t2]


def test_tracks_property_skips_none():
    """Tracks should skip objects without tracks."""
    t1 = Track(name="a")
    li = UserLabelImage(
        data=np.array([[1, 2]], dtype=np.int32),
        objects={
            1: LabelImage.Info(track=t1),
            2: LabelImage.Info(track=None),
        },
    )
    assert li.tracks == [t1]


def test_categories_property():
    """Categories should return unique non-empty categories."""
    li = UserLabelImage(
        data=np.array([[1, 2, 3]], dtype=np.int32),
        objects={
            1: LabelImage.Info(category="neuron"),
            2: LabelImage.Info(category="glia"),
            3: LabelImage.Info(category="neuron"),
        },
    )
    assert li.categories == {"neuron", "glia"}


def test_getitem_by_track():
    """__getitem__ should return binary mask for a tracked object."""
    t1 = Track(name="a")
    data = np.array([[0, 1, 0], [1, 0, 2]], dtype=np.int32)
    li = UserLabelImage(
        data=data,
        objects={1: LabelImage.Info(track=t1)},
    )
    mask = li[t1]
    expected = np.array([[False, True, False], [True, False, False]])
    np.testing.assert_array_equal(mask, expected)


def test_getitem_missing_track_raises():
    """__getitem__ should raise KeyError for missing track."""
    t1, t2 = Track(name="a"), Track(name="b")
    li = UserLabelImage(
        data=np.array([[1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=t1)},
    )
    with pytest.raises(KeyError):
        li[t2]


def test_contains():
    """__contains__ should check track presence in objects."""
    t1, t2 = Track(name="a"), Track(name="b")
    li = UserLabelImage(
        data=np.array([[1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=t1)},
    )
    assert t1 in li
    assert t2 not in li


def test_get_track_mask():
    """get_track_mask should be equivalent to __getitem__."""
    t1 = Track(name="a")
    data = np.array([[0, 1], [1, 0]], dtype=np.int32)
    li = UserLabelImage(
        data=data,
        objects={1: LabelImage.Info(track=t1)},
    )
    np.testing.assert_array_equal(li.get_track_mask(t1), li[t1])


def test_get_category_mask():
    """get_category_mask should return union of matching objects."""
    data = np.array([[1, 0, 2], [3, 0, 0]], dtype=np.int32)
    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(category="neuron"),
            2: LabelImage.Info(category="glia"),
            3: LabelImage.Info(category="neuron"),
        },
    )
    mask = li.get_category_mask("neuron")
    expected = np.array([[True, False, False], [True, False, False]])
    np.testing.assert_array_equal(mask, expected)


def test_get_category_mask_no_match():
    """get_category_mask should return all-False for no matches."""
    li = UserLabelImage(
        data=np.array([[1]], dtype=np.int32),
        objects={1: LabelImage.Info(category="neuron")},
    )
    mask = li.get_category_mask("glia")
    assert not mask.any()
    assert mask.shape == (1, 1)


def test_items():
    """Items should iterate over (track, category, mask) tuples in label order."""
    t1, t2 = Track(name="a"), Track(name="b")
    data = np.array([[1, 2], [0, 1]], dtype=np.int32)
    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(track=t1, category="neuron"),
            2: LabelImage.Info(track=t2, category="glia"),
        },
    )
    result = list(li.items())
    assert len(result) == 2
    assert result[0][0] is t1
    assert result[0][1] == "neuron"
    np.testing.assert_array_equal(result[0][2], data == 1)
    assert result[1][0] is t2
    assert result[1][1] == "glia"
    np.testing.assert_array_equal(result[1][2], data == 2)


def test_items_no_metadata():
    """Items should yield default Info for labels not in objects dict."""
    data = np.array([[1, 2]], dtype=np.int32)
    li = UserLabelImage(data=data)  # No objects metadata
    result = list(li.items())
    assert len(result) == 2
    assert result[0][0] is None  # Default track
    assert result[0][1] == ""  # Default category


def test_from_numpy_auto_tracks():
    """from_numpy with tracks=None should auto-create tracks."""
    data = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data)

    assert li.n_objects == 3
    assert len(li.objects) == 3
    assert li.objects[1].track.name == "1"
    assert li.objects[2].track.name == "2"
    assert li.objects[3].track.name == "3"


def test_from_numpy_tracks_list():
    """from_numpy with tracks=list should map positionally."""
    t1, t2 = Track(name="a"), Track(name="b")
    data = np.array([[1, 2]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data, tracks=[t1, t2])

    assert li.objects[1].track is t1
    assert li.objects[2].track is t2


def test_from_numpy_tracks_dict():
    """from_numpy with tracks=dict should map explicitly."""
    t5 = Track(name="five")
    data = np.array([[0, 5]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data, tracks={5: t5})

    assert li.objects[5].track is t5


def test_from_numpy_categories_list():
    """from_numpy with categories=list should map positionally."""
    data = np.array([[1, 2]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data, categories=["neuron", "glia"])

    assert li.objects[1].category == "neuron"
    assert li.objects[2].category == "glia"


def test_from_numpy_categories_dict():
    """from_numpy with categories=dict should map explicitly."""
    data = np.array([[0, 5]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data, categories={5: "neuron"})

    assert li.objects[5].category == "neuron"


def test_from_numpy_with_kwargs():
    """from_numpy should pass kwargs to constructor."""
    vid = Video(filename="test.mp4")
    data = np.array([[1]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data, video=vid, frame_idx=42, source="test")

    assert li.video is vid
    assert li.frame_idx == 42
    assert li.source == "test"


def test_from_masks():
    """from_masks should compose masks into a label image."""
    mask1 = UserSegmentationMask.from_numpy(
        np.array([[True, False], [False, False]]),
        track=Track(name="t1"),
        category="neuron",
        name="cell_1",
    )
    mask2 = UserSegmentationMask.from_numpy(
        np.array([[False, True], [True, False]]),
        track=Track(name="t2"),
        category="glia",
        name="cell_2",
    )
    li = UserLabelImage.from_masks([mask1, mask2])

    assert li.height == 2
    assert li.width == 2
    assert li.n_objects == 2
    assert li.data[0, 0] == 1
    assert li.data[0, 1] == 2
    assert li.data[1, 0] == 2
    assert li.data[1, 1] == 0
    assert li.objects[1].track.name == "t1"
    assert li.objects[1].category == "neuron"
    assert li.objects[2].track.name == "t2"
    assert li.objects[2].category == "glia"


def test_from_masks_overlap():
    """from_masks should assign overlapping pixels to last mask."""
    mask1 = UserSegmentationMask.from_numpy(np.ones((3, 3), dtype=bool))
    mask2 = UserSegmentationMask.from_numpy(
        np.array([[True, True, False], [True, True, False], [False, False, False]])
    )
    li = UserLabelImage.from_masks([mask1, mask2])

    # Overlap region should be label 2 (last mask wins)
    assert li.data[0, 0] == 2
    assert li.data[0, 1] == 2
    # Non-overlap region of mask1
    assert li.data[0, 2] == 1
    assert li.data[2, 2] == 1


def test_from_masks_empty_raises():
    """from_masks with empty list should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        UserLabelImage.from_masks([])


def test_from_masks_inconsistent_shape_raises():
    """from_masks with different shapes should raise ValueError."""
    m1 = UserSegmentationMask.from_numpy(np.zeros((3, 3), dtype=bool))
    m2 = UserSegmentationMask.from_numpy(np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="same shape"):
        UserLabelImage.from_masks([m1, m2])


def test_to_masks():
    """to_masks should decompose into per-object binary masks."""
    t1, t2 = Track(name="a"), Track(name="b")
    data = np.array([[1, 0], [0, 2]], dtype=np.int32)
    vid = Video(filename="test.mp4")
    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(track=t1, category="neuron", name="cell_1"),
            2: LabelImage.Info(track=t2, category="glia", name="cell_2"),
        },
        video=vid,
        frame_idx=5,
        source="test_source",
    )
    masks = li.to_masks()

    assert len(masks) == 2
    # First mask (label 1)
    np.testing.assert_array_equal(masks[0].data, [[True, False], [False, False]])
    assert masks[0].track is t1
    assert masks[0].category == "neuron"
    assert masks[0].name == "cell_1"
    assert masks[0].video is vid
    assert masks[0].frame_idx == 5
    assert masks[0].source == "test_source"
    # Second mask (label 2)
    np.testing.assert_array_equal(masks[1].data, [[False, False], [False, True]])
    assert masks[1].track is t2
    assert masks[1].category == "glia"


def test_to_masks_from_masks_roundtrip():
    """from_masks -> to_masks should preserve mask data."""
    mask_data_1 = np.zeros((5, 5), dtype=bool)
    mask_data_1[0:2, 0:2] = True
    mask_data_2 = np.zeros((5, 5), dtype=bool)
    mask_data_2[3:5, 3:5] = True

    t1, t2 = Track(name="a"), Track(name="b")
    m1 = UserSegmentationMask.from_numpy(mask_data_1, track=t1, category="c1")
    m2 = UserSegmentationMask.from_numpy(mask_data_2, track=t2, category="c2")

    li = UserLabelImage.from_masks([m1, m2])
    recovered = li.to_masks()

    assert len(recovered) == 2
    np.testing.assert_array_equal(recovered[0].data, mask_data_1)
    np.testing.assert_array_equal(recovered[1].data, mask_data_2)
    assert recovered[0].track.name == "a"
    assert recovered[1].track.name == "b"


def test_eq_false():
    """Two LabelImages with same data should not be equal (identity semantics)."""
    data = np.array([[1, 2]], dtype=np.int32)
    li1 = UserLabelImage(data=data.copy())
    li2 = UserLabelImage(data=data.copy())
    assert li1 is not li2
    assert li1 != li2


def test_labels_integration():
    """LabelImage should work with Labels.get_label_images()."""
    from sleap_io.model.labels import Labels

    vid1 = Video(filename="a.mp4")
    vid2 = Video(filename="b.mp4")
    t1 = Track(name="t1")

    li1 = UserLabelImage(
        data=np.array([[1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=t1, category="neuron")},
        video=vid1,
        frame_idx=0,
    )
    li2 = UserLabelImage(
        data=np.array([[2]], dtype=np.int32),
        objects={2: LabelImage.Info(category="glia")},
        video=vid1,
        frame_idx=1,
    )
    li3 = UserLabelImage(
        data=np.array([[1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=t1)},
        video=vid2,
        frame_idx=0,
    )

    labels = Labels(label_images=[li1, li2, li3])

    # Filter by video
    assert labels.get_label_images(video=vid1) == [li1, li2]
    assert labels.get_label_images(video=vid2) == [li3]

    # Filter by frame_idx
    assert labels.get_label_images(frame_idx=0) == [li1, li3]
    assert labels.get_label_images(frame_idx=1) == [li2]

    # Filter by track (checks objects metadata)
    assert labels.get_label_images(track=t1) == [li1, li3]

    # Filter by category
    assert labels.get_label_images(category="neuron") == [li1]
    assert labels.get_label_images(category="glia") == [li2]

    # Combined filters
    assert labels.get_label_images(video=vid1, track=t1) == [li1]


def test_info_score():
    info = LabelImage.Info(category="neuron", score=0.95)
    assert info.score == 0.95

    info_no_score = LabelImage.Info(category="neuron")
    assert info_no_score.score is None


def test_label_image_is_predicted():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data)
    assert li.is_predicted is False

    user_li = UserLabelImage(data=data)
    assert user_li.is_predicted is False
    assert isinstance(user_li, LabelImage)

    pred_li = PredictedLabelImage(data=data, score=0.9)
    assert pred_li.is_predicted is True
    assert isinstance(pred_li, LabelImage)
    assert pred_li.score == 0.9


def test_user_label_image():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    t1 = Track(name="1")
    li = UserLabelImage(
        data=data,
        objects={1: LabelImage.Info(track=t1, category="neuron")},
    )
    assert not li.is_predicted
    assert li.n_objects == 2
    assert t1 in li


def test_predicted_label_image():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    t1 = Track(name="1")
    li = PredictedLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(track=t1, category="neuron", score=0.9),
            2: LabelImage.Info(category="glia", score=0.7),
        },
        score=0.85,
    )
    assert li.is_predicted
    assert li.score == 0.85
    assert li.objects[1].score == 0.9
    assert li.objects[2].score == 0.7


def test_predicted_label_image_with_score_map():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    score_map = np.random.rand(2, 2).astype(np.float32)
    li = PredictedLabelImage(data=data, score=0.85, score_map=score_map)
    assert li.score_map is not None
    assert li.score_map.shape == (2, 2)
