"""Tests for SegmentationMask data model."""

import numpy as np
import pytest

from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.instance import Instance, Track
from sleap_io.model.mask import (
    PredictedSegmentationMask,
    SegmentationMask,
    UserSegmentationMask,
    _decode_rle,
    _encode_rle,
    _resize_nearest,
)
from sleap_io.model.skeleton import Skeleton


def test_encode_rle_all_zeros():
    mask = np.zeros((5, 5), dtype=bool)
    rle = _encode_rle(mask)
    assert len(rle) == 1
    assert rle[0] == 25  # All zeros


def test_encode_rle_all_ones():
    mask = np.ones((5, 5), dtype=bool)
    rle = _encode_rle(mask)
    # Should start with 0-length zero-run, then 25 ones
    assert rle[0] == 0
    assert rle[1] == 25


def test_encode_rle_empty():
    mask = np.zeros((0, 0), dtype=bool)
    rle = _encode_rle(mask)
    assert len(rle) == 0


def test_encode_decode_roundtrip():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    mask[7, 1] = True

    rle = _encode_rle(mask)
    decoded = _decode_rle(rle, 10, 10)
    np.testing.assert_array_equal(mask, decoded)


def test_encode_decode_roundtrip_random():
    rng = np.random.RandomState(42)
    mask = rng.rand(20, 30) > 0.5
    rle = _encode_rle(mask)
    decoded = _decode_rle(rle, 20, 30)
    np.testing.assert_array_equal(mask, decoded)


def test_decode_rle_empty():
    rle = np.array([], dtype=np.uint32)
    mask = _decode_rle(rle, 5, 5)
    assert mask.shape == (5, 5)
    assert not mask.any()


def test_segmentation_mask_abstract():
    """SegmentationMask cannot be instantiated directly."""
    with pytest.raises(TypeError, match="SegmentationMask is abstract"):
        SegmentationMask(rle_counts=np.array([25], dtype=np.uint32), height=5, width=5)


def test_segmentation_mask_identity_equality():
    mask1 = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    mask2 = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    assert mask1 is not mask2
    assert mask1 != mask2


def test_segmentation_mask_from_numpy():
    data = np.zeros((10, 15), dtype=bool)
    data[2:5, 3:8] = True

    mask = UserSegmentationMask.from_numpy(data, name="test")
    assert mask.height == 10
    assert mask.width == 15
    assert mask.name == "test"


def test_from_numpy_rejects_multi_class_array():
    """Multi-class int input raises rather than silently binarizing."""
    arr = np.array([[5, 5, 0], [0, 17, 99]], dtype=np.int32)
    with pytest.raises(ValueError, match="binary"):
        UserSegmentationMask.from_numpy(arr)


def test_from_numpy_rejects_multi_instance_array():
    """Multi-instance int input (e.g., Cellpose output) is rejected."""
    arr = np.array([[0, 1, 2], [3, 0, 2], [1, 0, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="LabelImage"):
        UserSegmentationMask.from_numpy(arr)


def test_from_numpy_accepts_binary_uint8():
    """Integer arrays with only values {0, 1} are still valid binary masks."""
    arr = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    mask = UserSegmentationMask.from_numpy(arr)
    assert mask.area == 3


def test_from_numpy_accepts_binary_int32():
    """int32 with only {0, 1} still works (nonzero is the one foreground class)."""
    arr = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
    mask = UserSegmentationMask.from_numpy(arr)
    assert mask.area == 3


def test_from_numpy_explicit_binarization():
    """User opts into binarization by casting to bool first."""
    arr = np.array([[5, 17, 0], [99, 0, 5]], dtype=np.int32)
    # Opt-in: explicit cast to bool bypasses the guard.
    mask = UserSegmentationMask.from_numpy(arr.astype(bool))
    assert mask.area == 4


def test_segmentation_mask_data():
    original = np.zeros((10, 10), dtype=bool)
    original[3:7, 2:8] = True

    mask = UserSegmentationMask.from_numpy(original)
    decoded = mask.data
    np.testing.assert_array_equal(original, decoded)


def test_segmentation_mask_area():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True  # 3 rows * 4 cols = 12 pixels

    mask = UserSegmentationMask.from_numpy(data)
    assert mask.area == 12


def test_segmentation_mask_bbox():
    data = np.zeros((20, 20), dtype=bool)
    data[5:10, 3:8] = True

    mask = UserSegmentationMask.from_numpy(data)
    x, y, w, h = mask.bbox
    assert x == 3.0
    assert y == 5.0
    assert w == 5.0
    assert h == 5.0


def test_segmentation_mask_bbox_empty():
    data = np.zeros((10, 10), dtype=bool)
    mask = UserSegmentationMask.from_numpy(data)
    assert mask.bbox == (0.0, 0.0, 0.0, 0.0)


def test_segmentation_mask_to_polygon():
    data = np.zeros((20, 20), dtype=bool)
    data[5:15, 5:15] = True

    mask = UserSegmentationMask.from_numpy(data, name="test_mask", category="cat")
    roi = mask.to_polygon()

    assert roi.name == "test_mask"
    assert roi.category == "cat"
    assert roi.geometry.area > 0


def test_segmentation_mask_to_polygon_empty():
    data = np.zeros((10, 10), dtype=bool)
    mask = UserSegmentationMask.from_numpy(data)
    roi = mask.to_polygon()
    assert roi.geometry.is_empty


def test_segmentation_mask_to_polygon_nonconvex():
    """to_polygon should handle non-convex (C/L/U) shapes correctly."""
    # C-shaped mask (concavity on the right)
    data = np.zeros((10, 10), dtype=bool)
    data[0:3, 0:8] = True  # Top bar
    data[3:7, 0:3] = True  # Left column
    data[7:10, 0:8] = True  # Bottom bar
    actual_area = data.sum()

    mask = UserSegmentationMask.from_numpy(data)
    roi = mask.to_polygon()

    # The polygon area should be close to the actual mask area (not inflated)
    assert roi.geometry.area == pytest.approx(actual_area, abs=2)


def test_segmentation_mask_is_predicted():
    data = np.zeros((5, 5), dtype=bool)
    mask = UserSegmentationMask.from_numpy(data)
    assert mask.is_predicted is False

    user_mask = UserSegmentationMask.from_numpy(data)
    assert user_mask.is_predicted is False
    assert isinstance(user_mask, SegmentationMask)

    pred_mask = PredictedSegmentationMask.from_numpy(data, score=0.95)
    assert pred_mask.is_predicted is True
    assert isinstance(pred_mask, SegmentationMask)
    assert pred_mask.score == 0.95


def test_user_segmentation_mask():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    mask = UserSegmentationMask.from_numpy(data, name="cell", category="neuron")
    assert mask.name == "cell"
    assert mask.category == "neuron"
    assert mask.area == 12
    assert not mask.is_predicted


def test_predicted_segmentation_mask():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    mask = PredictedSegmentationMask.from_numpy(data, score=0.85, category="neuron")
    assert mask.score == 0.85
    assert mask.is_predicted
    assert mask.area == 12
    assert mask.score_map is None


def test_predicted_segmentation_mask_with_score_map():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    score_map = np.random.rand(10, 10).astype(np.float32)
    mask = PredictedSegmentationMask.from_numpy(data, score=0.85, score_map=score_map)
    assert mask.score == 0.85
    assert mask.score_map is not None
    assert mask.score_map.shape == (10, 10)


def test_segmentation_mask_instance_idx():
    mask = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    assert mask._instance_idx == -1
    mask._instance_idx = 3
    assert mask._instance_idx == 3


def test_resize_nearest():
    arr = np.array([[1, 2], [3, 4]])
    result = _resize_nearest(arr, 4, 4)
    assert result.shape == (4, 4)
    assert result[0, 0] == 1
    assert result[0, 2] == 2
    assert result[2, 0] == 3
    assert result[2, 2] == 4


def test_resize_nearest_bool():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True
    result = _resize_nearest(mask, 8, 8)
    assert result.shape == (8, 8)
    assert result[:4, :4].all()
    assert not result[4:, :].any()


def test_segmentation_mask_scale_offset_defaults():
    mask = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    assert mask.scale == (1.0, 1.0)
    assert mask.offset == (0.0, 0.0)
    assert mask.has_spatial_transform is False


def test_segmentation_mask_scale_offset_custom():
    mask = UserSegmentationMask.from_numpy(
        np.zeros((5, 5), dtype=bool), scale=(0.5, 0.5), offset=(10.0, 20.0)
    )
    assert mask.scale == (0.5, 0.5)
    assert mask.offset == (10.0, 20.0)
    assert mask.has_spatial_transform is True


def test_segmentation_mask_scale_validation():
    with pytest.raises(ValueError, match="Scale values must be positive"):
        UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool), scale=(0.0, 1.0))
    with pytest.raises(ValueError, match="Scale values must be positive"):
        UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool), scale=(1.0, -1.0))


def test_segmentation_mask_image_extent():
    mask = UserSegmentationMask.from_numpy(
        np.zeros((10, 20), dtype=bool), scale=(0.5, 0.5)
    )
    assert mask.image_extent == (20, 40)

    mask2 = UserSegmentationMask.from_numpy(
        np.zeros((10, 10), dtype=bool), scale=(0.25, 0.25)
    )
    assert mask2.image_extent == (40, 40)

    mask3 = UserSegmentationMask.from_numpy(np.zeros((10, 10), dtype=bool))
    assert mask3.image_extent == (10, 10)


def test_segmentation_mask_from_numpy_stride():
    mask = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool), stride=2)
    assert mask.scale == (0.5, 0.5)
    assert mask.has_spatial_transform is True

    mask4 = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool), stride=4)
    assert mask4.scale == (0.25, 0.25)


def test_segmentation_mask_resampled():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    mask = UserSegmentationMask.from_numpy(
        data, scale=(0.5, 0.5), offset=(10.0, 20.0), name="cell", category="neuron"
    )

    resampled = mask.resampled(20, 20)
    assert resampled.height == 20
    assert resampled.width == 20
    assert resampled.scale == (1.0, 1.0)
    assert resampled.offset == (0.0, 0.0)
    assert resampled.name == "cell"
    assert resampled.category == "neuron"
    assert resampled.has_spatial_transform is False
    assert resampled.data.any()
    assert isinstance(resampled, UserSegmentationMask)


def test_predicted_segmentation_mask_resampled():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    score_map = np.random.rand(10, 10).astype(np.float32)
    mask = PredictedSegmentationMask.from_numpy(
        data,
        score=0.9,
        score_map=score_map,
        scale=(0.5, 0.5),
        score_map_scale=(0.25, 0.25),
    )

    resampled = mask.resampled(20, 20)
    assert isinstance(resampled, PredictedSegmentationMask)
    assert resampled.score == 0.9
    assert resampled.score_map is not None
    assert resampled.score_map.shape == (20, 20)
    assert resampled.score_map_scale == (1.0, 1.0)
    assert resampled.score_map_offset == (0.0, 0.0)


def test_predicted_segmentation_mask_score_map_spatial():
    mask = PredictedSegmentationMask.from_numpy(
        np.zeros((5, 5), dtype=bool),
        score_map_scale=(0.5, 0.5),
        score_map_offset=(5.0, 10.0),
    )
    assert mask.score_map_scale == (0.5, 0.5)
    assert mask.score_map_offset == (5.0, 10.0)


def test_segmentation_mask_bbox_with_scale():
    data = np.zeros((20, 20), dtype=bool)
    data[5:10, 3:8] = True
    mask = UserSegmentationMask.from_numpy(data, scale=(0.5, 0.5))
    x, y, w, h = mask.bbox
    assert x == pytest.approx(6.0)  # 3 / 0.5
    assert y == pytest.approx(10.0)  # 5 / 0.5
    assert w == pytest.approx(10.0)  # 5 / 0.5
    assert h == pytest.approx(10.0)  # 5 / 0.5


def test_segmentation_mask_bbox_with_offset():
    data = np.zeros((20, 20), dtype=bool)
    data[5:10, 3:8] = True
    mask = UserSegmentationMask.from_numpy(data, offset=(10.0, 20.0))
    x, y, w, h = mask.bbox
    assert x == pytest.approx(13.0)  # 3 + 10
    assert y == pytest.approx(25.0)  # 5 + 20
    assert w == pytest.approx(5.0)
    assert h == pytest.approx(5.0)


def test_segmentation_mask_to_polygon_with_scale():
    data = np.zeros((10, 10), dtype=bool)
    data[2:4, 3:6] = True
    mask = UserSegmentationMask.from_numpy(data, scale=(0.5, 0.5))
    roi = mask.to_polygon()
    bounds = roi.geometry.bounds  # (minx, miny, maxx, maxy)
    assert bounds[0] == pytest.approx(6.0)  # 3 / 0.5
    assert bounds[1] == pytest.approx(4.0)  # 2 / 0.5
    assert bounds[2] == pytest.approx(12.0)  # 6 / 0.5
    assert bounds[3] == pytest.approx(8.0)  # 4 / 0.5


def test_segmentation_mask_to_polygon_with_offset():
    data = np.zeros((10, 10), dtype=bool)
    data[2:4, 3:6] = True
    mask = UserSegmentationMask.from_numpy(data, offset=(10.0, 20.0))
    roi = mask.to_polygon()
    bounds = roi.geometry.bounds
    assert bounds[0] == pytest.approx(13.0)  # 3 + 10
    assert bounds[1] == pytest.approx(22.0)  # 2 + 20
    assert bounds[2] == pytest.approx(16.0)  # 6 + 10
    assert bounds[3] == pytest.approx(24.0)  # 4 + 20


def test_to_bbox_basic():
    data = np.zeros((10, 10), dtype=bool)
    data[5:10, 3:8] = True
    mask = UserSegmentationMask.from_numpy(data)
    bb = mask.to_bbox()
    assert isinstance(bb, UserBoundingBox)
    x, y, w, h = mask.bbox
    assert bb.xywh == pytest.approx((x, y, w, h))


def test_to_bbox_metadata():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 1:4] = True
    track = Track(name="t1")
    mask = UserSegmentationMask.from_numpy(
        data,
        track=track,
        category="cell",
        name="obj1",
        source="manual",
    )
    bb = mask.to_bbox()
    assert bb.track is track
    assert bb.category == "cell"
    assert bb.name == "obj1"
    assert bb.source == "manual"


def test_to_bbox_predicted():
    data = np.zeros((10, 10), dtype=bool)
    data[0:3, 0:3] = True
    mask = PredictedSegmentationMask.from_numpy(data, score=0.95)
    bb = mask.to_bbox()
    assert isinstance(bb, PredictedBoundingBox)
    assert bb.score == pytest.approx(0.95)


def test_to_bbox_with_scale():
    data = np.zeros((10, 10), dtype=bool)
    data[2:4, 3:6] = True
    mask = UserSegmentationMask.from_numpy(data, scale=(0.5, 0.5))
    bb = mask.to_bbox()
    # mask coords: x=3, y=2, w=3, h=2 -> image: x=6, y=4, w=6, h=4
    assert bb.xywh == pytest.approx((6.0, 4.0, 6.0, 4.0))


def test_to_bbox_with_offset():
    data = np.zeros((10, 10), dtype=bool)
    data[2:4, 3:6] = True
    mask = UserSegmentationMask.from_numpy(data, offset=(10.0, 20.0))
    bb = mask.to_bbox()
    assert bb.xywh == pytest.approx((13.0, 22.0, 3.0, 2.0))


def test_to_bbox_empty_mask():
    data = np.zeros((10, 10), dtype=bool)
    mask = UserSegmentationMask.from_numpy(data)
    bb = mask.to_bbox()
    assert bb.xywh == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_to_user_basic():
    data = np.zeros((10, 10), dtype=bool)
    data[5:10, 3:8] = True
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9)
    user = pred.to_user()
    assert isinstance(user, UserSegmentationMask)
    assert user.is_predicted is False
    assert user.height == pred.height
    assert user.width == pred.width
    np.testing.assert_array_equal(user.data, pred.data)


def test_to_user_metadata():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 1:4] = True
    track = Track(name="t1")
    pred = PredictedSegmentationMask.from_numpy(
        data,
        score=0.8,
        track=track,
        tracking_score=0.7,
        category="cell",
        name="obj1",
        source="model",
    )
    user = pred.to_user()
    assert user.track is track
    assert user.tracking_score == pytest.approx(0.7)
    assert user.category == "cell"
    assert user.name == "obj1"
    assert user.source == "model"


def test_to_user_drops_score_and_score_map():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    score_map = np.random.rand(10, 10).astype(np.float32)
    pred = PredictedSegmentationMask.from_numpy(data, score=0.85, score_map=score_map)
    user = pred.to_user()
    assert not hasattr(user, "score")
    assert not hasattr(user, "score_map")
    assert user.is_predicted is False


def test_to_user_sets_from_predicted():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9)
    user = pred.to_user()
    assert user.from_predicted is pred


def test_to_user_link_false():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9)
    user = pred.to_user(link=False)
    assert user.from_predicted is None


def test_to_user_preserves_scale_offset():
    data = np.zeros((10, 10), dtype=bool)
    data[2:4, 3:6] = True
    pred = PredictedSegmentationMask.from_numpy(
        data, score=0.9, scale=(0.5, 0.5), offset=(10.0, 20.0)
    )
    user = pred.to_user()
    # Transform metadata is carried over verbatim, not applied to the raster.
    assert user.scale == (0.5, 0.5)
    assert user.offset == (10.0, 20.0)


def test_to_user_preserves_instance():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    skeleton = Skeleton(["a", "b"])
    instance = Instance.from_numpy(np.array([[0.0, 0.0], [1.0, 1.0]]), skeleton)
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9, instance=instance)
    user = pred.to_user()
    assert user.instance is instance


def test_to_user_preserves_instance_idx():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9)
    pred._instance_idx = 3
    user = pred.to_user()
    assert user._instance_idx == 3


def test_to_user_copies_rle_counts():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9)
    user = pred.to_user()
    # Independent buffer with equal values.
    assert user.rle_counts is not pred.rle_counts
    np.testing.assert_array_equal(user.rle_counts, pred.rle_counts)
    # Mutating the user copy does not affect the prediction.
    user.rle_counts[0] += 1
    assert not np.array_equal(user.rle_counts, pred.rle_counts)


def test_to_user_empty_mask():
    data = np.zeros((10, 10), dtype=bool)
    pred = PredictedSegmentationMask.from_numpy(data, score=0.5)
    user = pred.to_user()
    assert isinstance(user, UserSegmentationMask)
    assert user.area == 0
    assert user.from_predicted is pred


def test_to_user_not_defined_on_user_mask():
    """to_user() is a predicted-only conversion (no self-referential link)."""
    user = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    assert not hasattr(user, "to_user")


def test_user_mask_from_predicted_defaults_none():
    user = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    assert user.from_predicted is None


def test_from_predicted_excluded_from_repr():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9)
    user = pred.to_user()
    assert "from_predicted" not in repr(user)


def test_from_predicted_does_not_affect_identity_equality():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True
    pred = PredictedSegmentationMask.from_numpy(data, score=0.9)
    user1 = pred.to_user()
    user2 = pred.to_user()
    # Identity equality holds despite a shared from_predicted link.
    assert user1.from_predicted is user2.from_predicted
    assert user1 != user2
    assert user1 == user1
