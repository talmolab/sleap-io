"""Tests for LabelImage data model."""

import copy

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
    """from_numpy with create_tracks=True should auto-create tracks."""
    data = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data, create_tracks=True)

    assert li.n_objects == 3
    assert len(li.objects) == 3
    assert li.objects[1].track.name == "1"
    assert li.objects[2].track.name == "2"
    assert li.objects[3].track.name == "3"


def test_from_numpy_no_tracks_by_default():
    """from_numpy with default tracks=None should not create tracks."""
    data = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data)

    assert li.n_objects == 3
    assert li.objects[1].track is None
    assert li.objects[2].track is None
    assert li.objects[3].track is None


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


def test_label_image_scale_offset_defaults():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data)
    assert li.scale == (1.0, 1.0)
    assert li.offset == (0.0, 0.0)
    assert li.has_spatial_transform is False


def test_label_image_scale_offset_custom():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data, scale=(0.5, 0.5), offset=(10.0, 20.0))
    assert li.scale == (0.5, 0.5)
    assert li.offset == (10.0, 20.0)
    assert li.has_spatial_transform is True


def test_label_image_scale_validation():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match="Scale values must be positive"):
        UserLabelImage(data=data, scale=(0.0, 1.0))


def test_label_image_image_extent():
    data = np.zeros((10, 20), dtype=np.int32)
    li = UserLabelImage(data=data, scale=(0.5, 0.5))
    assert li.image_extent == (20, 40)


def test_label_image_resampled():
    data = np.array(
        [[0, 1, 1, 0], [2, 2, 0, 0], [0, 0, 3, 3], [0, 0, 0, 0]], dtype=np.int32
    )
    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(category="a"),
            2: LabelImage.Info(category="b"),
            3: LabelImage.Info(category="c"),
        },
        scale=(0.5, 0.5),
        offset=(10.0, 20.0),
    )

    resampled = li.resampled(8, 8)
    assert resampled.height == 8
    assert resampled.width == 8
    assert resampled.scale == (1.0, 1.0)
    assert resampled.offset == (0.0, 0.0)
    assert resampled.has_spatial_transform is False
    assert isinstance(resampled, UserLabelImage)
    assert set(resampled.objects.keys()) == {1, 2, 3}
    assert resampled.objects[1].category == "a"


def test_predicted_label_image_resampled():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    score_map = np.random.rand(2, 2).astype(np.float32)
    li = PredictedLabelImage(
        data=data,
        score=0.9,
        score_map=score_map,
        scale=(0.5, 0.5),
        score_map_scale=(0.25, 0.25),
    )

    resampled = li.resampled(4, 4)
    assert isinstance(resampled, PredictedLabelImage)
    assert resampled.score == 0.9
    assert resampled.score_map is not None
    assert resampled.score_map.shape == (4, 4)
    assert resampled.score_map_scale == (1.0, 1.0)
    assert resampled.score_map_offset == (0.0, 0.0)


def test_predicted_label_image_score_map_spatial():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = PredictedLabelImage(
        data=data,
        score_map_scale=(0.5, 0.5),
        score_map_offset=(5.0, 10.0),
    )
    assert li.score_map_scale == (0.5, 0.5)
    assert li.score_map_offset == (5.0, 10.0)


def test_from_masks_propagates_scale_offset():
    m1 = UserSegmentationMask.from_numpy(
        np.array([[True, False], [False, False]]),
        scale=(0.5, 0.5),
        offset=(10.0, 20.0),
    )
    m2 = UserSegmentationMask.from_numpy(
        np.array([[False, False], [False, True]]),
        scale=(0.5, 0.5),
        offset=(10.0, 20.0),
    )
    li = UserLabelImage.from_masks([m1, m2])
    assert li.scale == (0.5, 0.5)
    assert li.offset == (10.0, 20.0)


def test_from_masks_inconsistent_scale_raises():
    m1 = UserSegmentationMask.from_numpy(
        np.array([[True, False], [False, False]]),
        scale=(0.5, 0.5),
    )
    m2 = UserSegmentationMask.from_numpy(
        np.array([[False, False], [False, True]]),
        scale=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="same scale and offset"):
        UserLabelImage.from_masks([m1, m2])


def test_from_masks_inconsistent_offset_raises():
    m1 = UserSegmentationMask.from_numpy(
        np.array([[True, False], [False, False]]),
        offset=(0.0, 0.0),
    )
    m2 = UserSegmentationMask.from_numpy(
        np.array([[False, False], [False, True]]),
        offset=(10.0, 20.0),
    )
    with pytest.raises(ValueError, match="same scale and offset"):
        UserLabelImage.from_masks([m1, m2])


def test_to_masks_propagates_scale_offset():
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data, scale=(0.5, 0.5), offset=(10.0, 20.0))
    masks = li.to_masks()
    assert len(masks) == 2
    for m in masks:
        assert m.scale == (0.5, 0.5)
        assert m.offset == (10.0, 20.0)


# -- from_binary_masks tests --


def test_from_binary_masks_list_input():
    """from_binary_masks should composite a list of 2D bool arrays."""
    m1 = np.array([[True, False], [False, False]])
    m2 = np.array([[False, True], [False, False]])
    li = UserLabelImage.from_binary_masks([m1, m2])
    assert li.height == 2
    assert li.width == 2
    assert li.n_objects == 2
    assert li.data[0, 0] == 1
    assert li.data[0, 1] == 2
    assert li.data[1, 0] == 0
    assert 1 in li.objects
    assert 2 in li.objects


def test_from_binary_masks_ndarray_input():
    """from_binary_masks should accept a stacked (N, H, W) array."""
    masks = np.zeros((2, 3, 3), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 2, 2] = True
    li = UserLabelImage.from_binary_masks(masks)
    assert li.n_objects == 2
    assert li.data[0, 0] == 1
    assert li.data[2, 2] == 2


def test_from_binary_masks_single_mask():
    """from_binary_masks should accept a single 2D array."""
    mask = np.array([[True, False], [False, True]])
    li = UserLabelImage.from_binary_masks(mask)
    assert li.n_objects == 1
    assert li.data[0, 0] == 1
    assert li.data[1, 1] == 1


def test_from_binary_masks_overlap():
    """Overlapping pixels should be assigned to the last mask."""
    m1 = np.array([[True, True], [False, False]])
    m2 = np.array([[True, False], [True, False]])
    li = UserLabelImage.from_binary_masks([m1, m2])
    # Pixel (0, 0) overlaps: last mask (m2, label 2) wins.
    assert li.data[0, 0] == 2
    # Pixel (0, 1) only in m1.
    assert li.data[0, 1] == 1
    # Pixel (1, 0) only in m2.
    assert li.data[1, 0] == 2


def test_from_binary_masks_with_tracks():
    """Tracks list should map positionally to objects."""
    m1 = np.array([[True, False], [False, False]])
    m2 = np.array([[False, True], [False, False]])
    t1 = Track(name="cell_a")
    t2 = Track(name="cell_b")
    li = UserLabelImage.from_binary_masks([m1, m2], tracks=[t1, t2])
    assert li.objects[1].track is t1
    assert li.objects[2].track is t2


def test_from_binary_masks_create_tracks():
    """create_tracks=True should auto-create one Track per mask."""
    masks = np.zeros((3, 2, 2), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 0, 1] = True
    masks[2, 1, 0] = True
    li = UserLabelImage.from_binary_masks(masks, create_tracks=True)
    assert li.objects[1].track is not None
    assert li.objects[1].track.name == "1"
    assert li.objects[2].track.name == "2"
    assert li.objects[3].track.name == "3"


def test_from_binary_masks_no_tracks_default():
    """Default should have no tracks."""
    m1 = np.array([[True, False], [False, False]])
    li = UserLabelImage.from_binary_masks([m1])
    assert li.objects[1].track is None


def test_from_binary_masks_with_categories():
    """Categories list should map positionally to objects."""
    m1 = np.array([[True, False], [False, False]])
    m2 = np.array([[False, True], [False, False]])
    li = UserLabelImage.from_binary_masks([m1, m2], categories=["neuron", "glia"])
    assert li.objects[1].category == "neuron"
    assert li.objects[2].category == "glia"


def test_from_binary_masks_with_names():
    """Names list should map positionally to objects."""
    m1 = np.array([[True, False], [False, False]])
    m2 = np.array([[False, True], [False, False]])
    li = UserLabelImage.from_binary_masks([m1, m2], names=["obj_1", "obj_2"])
    assert li.objects[1].name == "obj_1"
    assert li.objects[2].name == "obj_2"


def test_from_binary_masks_with_scores():
    """Scores list should map to Info.score."""
    m1 = np.array([[True, False], [False, False]])
    m2 = np.array([[False, True], [False, False]])
    li = UserLabelImage.from_binary_masks([m1, m2], scores=[0.95, 0.87])
    assert li.objects[1].score == 0.95
    assert li.objects[2].score == 0.87


def test_from_binary_masks_predicted():
    """PredictedLabelImage.from_binary_masks should set image-level score."""
    m1 = np.array([[True, False], [False, False]])
    li = PredictedLabelImage.from_binary_masks(
        [m1], scores=[0.95], score=0.9, source="sam"
    )
    assert isinstance(li, PredictedLabelImage)
    assert li.score == 0.9
    assert li.source == "sam"
    assert li.objects[1].score == 0.95


def test_from_binary_masks_kwargs():
    """Extra kwargs should pass through to the constructor."""
    m1 = np.array([[True, False], [False, False]])
    video = Video(filename="test.mp4")
    li = UserLabelImage.from_binary_masks(
        [m1], video=video, frame_idx=5, source="cellpose"
    )
    assert li.video is video
    assert li.frame_idx == 5
    assert li.source == "cellpose"


def test_from_binary_masks_uint8_input():
    """uint8 arrays with 0/1 values should be cast to bool correctly."""
    m1 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    m2 = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    li = UserLabelImage.from_binary_masks([m1, m2])
    assert li.data[0, 0] == 1
    assert li.data[0, 1] == 2


def test_from_binary_masks_empty_raises():
    """Empty mask list should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        UserLabelImage.from_binary_masks([])

    with pytest.raises(ValueError, match="empty"):
        UserLabelImage.from_binary_masks(np.zeros((0, 3, 3), dtype=bool))


def test_from_binary_masks_inconsistent_shape_raises():
    """Mismatched shapes should raise ValueError."""
    m1 = np.zeros((2, 2), dtype=bool)
    m2 = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="same shape"):
        UserLabelImage.from_binary_masks([m1, m2])


def test_from_binary_masks_tracks_length_mismatch_raises():
    """Wrong tracks length should raise ValueError."""
    m1 = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="tracks length"):
        UserLabelImage.from_binary_masks(
            [m1], tracks=[Track(name="a"), Track(name="b")]
        )


def test_from_binary_masks_scores_length_mismatch_raises():
    """Wrong scores length should raise ValueError."""
    m1 = np.zeros((2, 2), dtype=bool)
    m2 = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="scores length"):
        UserLabelImage.from_binary_masks([m1, m2], scores=[0.5])


# -- from_stack tests --


def test_from_stack_basic():
    """from_stack should create one LabelImage per frame."""
    stack = np.zeros((3, 4, 5), dtype=np.int32)
    stack[0, 0, 0] = 1
    stack[1, 1, 1] = 2
    stack[2, 2, 2] = 3

    result = UserLabelImage.from_stack(stack)

    assert len(result) == 3
    for i, li in enumerate(result):
        assert isinstance(li, UserLabelImage)
        assert li.height == 4
        assert li.width == 5
        assert li.frame_idx == i


def test_from_stack_list_input():
    """from_stack should accept a list of 2D arrays."""
    frames = [
        np.array([[1, 0], [0, 0]], dtype=np.int32),
        np.array([[0, 2], [0, 0]], dtype=np.int32),
    ]
    result = UserLabelImage.from_stack(frames)

    assert len(result) == 2
    assert result[0].data[0, 0] == 1
    assert result[1].data[0, 1] == 2


def test_from_stack_auto_tracks():
    """create_tracks=True should share Track objects across frames."""
    stack = np.zeros((3, 4, 4), dtype=np.int32)
    stack[0, 0, 0] = 1
    stack[0, 1, 1] = 2
    stack[1, 0, 0] = 1  # same ID as frame 0
    stack[2, 2, 2] = 2  # same ID as frame 0

    result = UserLabelImage.from_stack(stack, create_tracks=True)

    # Track for ID 1 in frame 0 and frame 1 should be the same object
    assert result[0].objects[1].track is result[1].objects[1].track
    # Track for ID 2 in frame 0 and frame 2 should be the same object
    assert result[0].objects[2].track is result[2].objects[2].track
    # Track names should be string of the label ID
    assert result[0].objects[1].track.name == "1"
    assert result[0].objects[2].track.name == "2"


def test_from_stack_tracks_list():
    """Tracks provided as a list should be shared across frames."""
    t1, t2 = Track(name="a"), Track(name="b")
    stack = np.zeros((2, 3, 3), dtype=np.int32)
    stack[0, 0, 0] = 1
    stack[0, 1, 1] = 2
    stack[1, 0, 0] = 1

    result = UserLabelImage.from_stack(stack, tracks=[t1, t2])

    assert result[0].objects[1].track is t1
    assert result[0].objects[2].track is t2
    assert result[1].objects[1].track is t1


def test_from_stack_tracks_dict():
    """Tracks provided as a dict should map correctly."""
    t5 = Track(name="five")
    stack = np.zeros((2, 3, 3), dtype=np.int32)
    stack[0, 0, 0] = 5
    stack[1, 1, 1] = 5

    result = UserLabelImage.from_stack(stack, tracks={5: t5})

    assert result[0].objects[5].track is t5
    assert result[1].objects[5].track is t5


def test_from_stack_categories():
    """Categories should be applied consistently across frames."""
    stack = np.zeros((2, 3, 3), dtype=np.int32)
    stack[0, 0, 0] = 1
    stack[0, 1, 1] = 2
    stack[1, 0, 0] = 1

    # List form
    result = UserLabelImage.from_stack(stack, categories=["neuron", "glia"])
    assert result[0].objects[1].category == "neuron"
    assert result[0].objects[2].category == "glia"
    assert result[1].objects[1].category == "neuron"

    # Dict form
    result = UserLabelImage.from_stack(stack, categories={1: "neuron", 2: "glia"})
    assert result[0].objects[1].category == "neuron"
    assert result[0].objects[2].category == "glia"


def test_from_stack_frame_idx_custom():
    """Custom frame_idx should be applied per frame."""
    stack = np.zeros((3, 2, 2), dtype=np.int32)
    result = UserLabelImage.from_stack(stack, frame_idx=[10, 20, 30])

    assert result[0].frame_idx == 10
    assert result[1].frame_idx == 20
    assert result[2].frame_idx == 30


def test_from_stack_predicted():
    """PredictedLabelImage.from_stack with scalar score."""
    stack = np.zeros((2, 3, 3), dtype=np.int32)
    stack[0, 0, 0] = 1

    result = PredictedLabelImage.from_stack(stack, score=0.9)

    assert len(result) == 2
    for li in result:
        assert isinstance(li, PredictedLabelImage)
        assert li.score == 0.9


def test_from_stack_predicted_per_frame_score():
    """PredictedLabelImage.from_stack with per-frame scores."""
    stack = np.zeros((3, 3, 3), dtype=np.int32)

    result = PredictedLabelImage.from_stack(stack, score=[0.9, 0.8, 0.7])

    assert result[0].score == 0.9
    assert result[1].score == 0.8
    assert result[2].score == 0.7


def test_from_stack_predicted_score_map():
    """PredictedLabelImage.from_stack with score_map slicing."""
    stack = np.zeros((2, 4, 4), dtype=np.int32)
    sm = np.random.rand(2, 4, 4).astype(np.float32)

    result = PredictedLabelImage.from_stack(stack, score_map=sm)

    assert result[0].score_map is not None
    np.testing.assert_array_equal(result[0].score_map, sm[0])
    assert result[1].score_map is not None
    np.testing.assert_array_equal(result[1].score_map, sm[1])


def test_from_stack_track_consistency():
    """Track for an ID appearing in non-consecutive frames is shared."""
    stack = np.zeros((3, 3, 3), dtype=np.int32)
    stack[0, 0, 0] = 1  # ID 1 in frame 0
    # frame 1: no ID 1
    stack[2, 1, 1] = 1  # ID 1 in frame 2

    result = UserLabelImage.from_stack(stack, create_tracks=True)

    assert 1 in result[0].objects
    assert 1 not in result[1].objects
    assert 1 in result[2].objects
    assert result[0].objects[1].track is result[2].objects[1].track


def test_from_stack_kwargs():
    """Shared kwargs should propagate to all frames."""
    vid = Video(filename="test.mp4")
    stack = np.zeros((2, 3, 3), dtype=np.int32)

    result = UserLabelImage.from_stack(
        stack,
        video=vid,
        source="cellpose",
        scale=(0.5, 0.5),
        offset=(10.0, 20.0),
    )

    for li in result:
        assert li.video is vid
        assert li.source == "cellpose"
        assert li.scale == (0.5, 0.5)
        assert li.offset == (10.0, 20.0)


def test_from_stack_non_3d_raises():
    """from_stack with a 2D array should raise ValueError."""
    with pytest.raises(ValueError, match="\\(T, H, W\\)"):
        UserLabelImage.from_stack(np.zeros((3, 3), dtype=np.int32))


def test_from_stack_frame_idx_mismatch_raises():
    """from_stack with wrong frame_idx length should raise."""
    stack = np.zeros((3, 3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="frame_idx length"):
        UserLabelImage.from_stack(stack, frame_idx=[0, 1])


def test_from_stack_empty():
    """from_stack with 0-frame input should return empty list."""
    stack = np.zeros((0, 3, 3), dtype=np.int32)
    assert UserLabelImage.from_stack(stack) == []


def test_from_stack_score_mismatch_raises():
    """from_stack with wrong score list length should raise."""
    stack = np.zeros((3, 3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="score list length"):
        PredictedLabelImage.from_stack(stack, score=[0.9, 0.8])


def test_from_stack_score_map_shape_raises():
    """from_stack with wrong score_map shape should raise."""
    stack = np.zeros((3, 3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="score_map must be"):
        PredictedLabelImage.from_stack(stack, score_map=np.zeros((2, 3, 3)))


def test_from_stack_invalid_data_type_raises():
    """from_stack with non-array data should raise."""
    with pytest.raises(ValueError, match="numpy array or list"):
        UserLabelImage.from_stack("not an array")


def test_label_image_no_data_no_loader_raises():
    """Accessing .data with no data and no lazy loader raises ValueError."""
    li = UserLabelImage(data=np.zeros((2, 2), dtype=np.int32))
    li._data = None
    li._lazy_loader = None
    with pytest.raises(ValueError, match="no data and no lazy loader"):
        _ = li.data


def test_label_image_data_setter():
    """Setting .data updates cached dimensions and clears lazy loader."""
    li = UserLabelImage(data=np.zeros((4, 6), dtype=np.int32))
    assert li.height == 4
    assert li.width == 6

    # Set new data
    li.data = np.zeros((8, 10), dtype=np.int32)
    assert li.height == 8
    assert li.width == 10
    assert li._lazy_loader is None

    # Set to None
    li.data = None
    assert li._data is None


def test_label_image_deepcopy_materialized():
    """Deepcopy with already-materialized data copies without lazy load."""
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data)
    li_copy = copy.deepcopy(li)
    np.testing.assert_array_equal(li_copy.data, data)
    assert li_copy.data is not li.data


def test_label_image_deepcopy_lazy():
    """Deepcopy with lazy loader materializes data before copying."""
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data)
    # Simulate lazy state
    li._data = None
    li._lazy_loader = lambda: data.copy()
    li_copy = copy.deepcopy(li)
    np.testing.assert_array_equal(li_copy.data, data)


def test_label_image_deepcopy_no_data():
    """Deepcopy with no data and no loader produces data=None."""
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data)
    li._data = None
    li._lazy_loader = None
    li_copy = copy.deepcopy(li)
    assert li_copy._data is None


def test_predicted_label_image_deepcopy_with_score_map():
    """Deepcopy preserves score_map on PredictedLabelImage."""
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    sm = np.random.rand(2, 2).astype(np.float32)
    li = PredictedLabelImage(data=data, score=0.9, score_map=sm)
    li_copy = copy.deepcopy(li)
    np.testing.assert_array_equal(li_copy.score_map, sm)
    assert li_copy.score_map is not sm
    assert li_copy.score == pytest.approx(0.9)


def test_predicted_label_image_score_map_setter():
    """Setting score_map clears lazy loader."""
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = PredictedLabelImage(data=data, score=0.5)
    sm = np.random.rand(2, 2).astype(np.float32)
    li.score_map = sm
    np.testing.assert_array_equal(li.score_map, sm)
    assert li._score_map_lazy_loader is None


def test_predicted_label_image_score_map_lazy():
    """Lazy score_map loader is triggered on first access."""
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    sm = np.random.rand(2, 2).astype(np.float32)
    li = PredictedLabelImage(data=data, score=0.5)
    li._score_map = None
    li._score_map_lazy_loader = lambda: sm.copy()
    result = li.score_map
    np.testing.assert_array_equal(result, sm)
    assert li._score_map_lazy_loader is None


def test_from_numpy_accumulate_tracks():
    """from_numpy with dict tracks + create_tracks accumulates new IDs."""
    shared = {}
    data0 = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li0 = UserLabelImage.from_numpy(data0, tracks=shared, create_tracks=True)
    assert set(shared.keys()) == {1, 2}
    assert li0.objects[1].track is shared[1]

    # Frame with overlapping + new ID
    data1 = np.array([[1, 0], [0, 3]], dtype=np.int32)
    li1 = UserLabelImage.from_numpy(data1, tracks=shared, create_tracks=True)
    assert set(shared.keys()) == {1, 2, 3}
    # Track 1 is the same object across both frames
    assert li0.objects[1].track is li1.objects[1].track


def test_from_numpy_dict_tracks_without_create():
    """from_numpy with dict tracks but create_tracks=False uses only provided."""
    tracks = {1: Track(name="a")}
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage.from_numpy(data, tracks=tracks)
    assert li.objects[1].track is tracks[1]
    assert li.objects[2].track is None
    # Dict not mutated
    assert 2 not in tracks


def test_height_width_fallback_to_data_shape():
    """height/width fall back to data.shape when _height/_width are 0."""
    data = np.zeros((5, 7), dtype=np.int32)
    li = UserLabelImage(data=data)
    # Reset cached dims to 0 to trigger fallback
    li._height = 0
    li._width = 0
    assert li.height == 5
    assert li.width == 7
