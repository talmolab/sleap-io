"""Tests for SegmentationMask data model."""

import numpy as np
import pytest

from sleap_io.model.mask import SegmentationMask, _decode_rle, _encode_rle
from sleap_io.model.roi import AnnotationType


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


def test_segmentation_mask_identity_equality():
    mask1 = SegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    mask2 = SegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    assert mask1 is not mask2
    assert mask1 != mask2


def test_segmentation_mask_from_numpy():
    data = np.zeros((10, 15), dtype=bool)
    data[2:5, 3:8] = True

    mask = SegmentationMask.from_numpy(data, name="test")
    assert mask.height == 10
    assert mask.width == 15
    assert mask.name == "test"
    assert mask.annotation_type == AnnotationType.SEGMENTATION


def test_segmentation_mask_data():
    original = np.zeros((10, 10), dtype=bool)
    original[3:7, 2:8] = True

    mask = SegmentationMask.from_numpy(original)
    decoded = mask.data
    np.testing.assert_array_equal(original, decoded)


def test_segmentation_mask_area():
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 3:7] = True  # 3 rows * 4 cols = 12 pixels

    mask = SegmentationMask.from_numpy(data)
    assert mask.area == 12


def test_segmentation_mask_bbox():
    data = np.zeros((20, 20), dtype=bool)
    data[5:10, 3:8] = True

    mask = SegmentationMask.from_numpy(data)
    x, y, w, h = mask.bbox
    assert x == 3.0
    assert y == 5.0
    assert w == 5.0
    assert h == 5.0


def test_segmentation_mask_bbox_empty():
    data = np.zeros((10, 10), dtype=bool)
    mask = SegmentationMask.from_numpy(data)
    assert mask.bbox == (0.0, 0.0, 0.0, 0.0)


def test_segmentation_mask_to_polygon():
    data = np.zeros((20, 20), dtype=bool)
    data[5:15, 5:15] = True

    mask = SegmentationMask.from_numpy(data, name="test_mask", category="cat")
    roi = mask.to_polygon()

    assert roi.name == "test_mask"
    assert roi.category == "cat"
    assert roi.geometry.area > 0


def test_segmentation_mask_to_polygon_empty():
    data = np.zeros((10, 10), dtype=bool)
    mask = SegmentationMask.from_numpy(data)
    roi = mask.to_polygon()
    assert roi.geometry.is_empty


def test_segmentation_mask_annotation_type_default():
    mask = SegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))
    assert mask.annotation_type == AnnotationType.SEGMENTATION


def test_segmentation_mask_with_score():
    mask = SegmentationMask.from_numpy(np.ones((5, 5), dtype=bool), score=0.95)
    assert mask.score == 0.95
