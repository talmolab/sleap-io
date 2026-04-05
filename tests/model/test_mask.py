"""Tests for SegmentationMask data model."""

import numpy as np
import pytest

from sleap_io.model.mask import (
    PredictedSegmentationMask,
    SegmentationMask,
    UserSegmentationMask,
    _decode_rle,
    _encode_rle,
    _resize_nearest,
)


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
