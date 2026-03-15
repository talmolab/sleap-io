"""Tests for BoundingBox data model."""

import math

import numpy as np
import pytest

from sleap_io.model.bbox import BoundingBox, PredictedBoundingBox, UserBoundingBox
from sleap_io.model.video import Video


def test_bbox_basic():
    bbox = BoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert bbox.x_center == 50
    assert bbox.y_center == 50
    assert bbox.width == 100
    assert bbox.height == 80
    assert bbox.angle == 0.0
    assert bbox.video is None
    assert bbox.frame_idx is None
    assert bbox.track is None
    assert bbox.instance is None
    assert bbox.category == ""
    assert bbox.name == ""
    assert bbox.source == ""


def test_bbox_identity_equality():
    bbox1 = BoundingBox(x_center=50, y_center=50, width=100, height=80)
    bbox2 = BoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert bbox1 is not bbox2
    assert bbox1 != bbox2


def test_bbox_from_xyxy():
    bbox = BoundingBox.from_xyxy(10, 20, 110, 100)
    assert bbox.x_center == 60
    assert bbox.y_center == 60
    assert bbox.width == 100
    assert bbox.height == 80


def test_bbox_from_xyxy_swapped_raises():
    """from_xyxy raises ValueError for swapped coordinates."""
    with pytest.raises(ValueError, match="x2 >= x1"):
        BoundingBox.from_xyxy(110, 100, 10, 20)
    with pytest.raises(ValueError, match="y2 >= y1"):
        BoundingBox.from_xyxy(10, 100, 110, 20)


def test_bbox_from_xyxy_with_kwargs():
    video = Video(filename="test.mp4")
    bbox = BoundingBox.from_xyxy(
        10, 20, 110, 100, video=video, frame_idx=5, category="mouse"
    )
    assert bbox.video is video
    assert bbox.frame_idx == 5
    assert bbox.category == "mouse"


def test_bbox_from_xywh():
    bbox = BoundingBox.from_xywh(10, 20, 100, 80)
    assert bbox.x_center == 60
    assert bbox.y_center == 60
    assert bbox.width == 100
    assert bbox.height == 80


def test_bbox_from_xywh_with_kwargs():
    bbox = BoundingBox.from_xywh(10, 20, 100, 80, name="box1", source="manual")
    assert bbox.name == "box1"
    assert bbox.source == "manual"


def test_bbox_xyxy():
    bbox = BoundingBox(x_center=60, y_center=60, width=100, height=80)
    x1, y1, x2, y2 = bbox.xyxy
    assert x1 == 10
    assert y1 == 20
    assert x2 == 110
    assert y2 == 100


def test_bbox_xyxy_rotated_raises():
    bbox = BoundingBox(x_center=50, y_center=50, width=100, height=80, angle=0.5)
    with pytest.raises(ValueError, match="axis-aligned"):
        _ = bbox.xyxy


def test_bbox_xywh():
    bbox = BoundingBox(x_center=60, y_center=60, width=100, height=80)
    x, y, w, h = bbox.xywh
    assert x == 10
    assert y == 20
    assert w == 100
    assert h == 80


def test_bbox_xywh_rotated_raises():
    bbox = BoundingBox(x_center=50, y_center=50, width=100, height=80, angle=0.5)
    with pytest.raises(ValueError, match="axis-aligned"):
        _ = bbox.xywh


def test_bbox_area():
    bbox = BoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert bbox.area == 8000.0


def test_bbox_bounds_axis_aligned():
    bbox = BoundingBox(x_center=60, y_center=60, width=100, height=80)
    assert bbox.bounds == (10.0, 20.0, 110.0, 100.0)


def test_bbox_bounds_rotated():
    bbox = BoundingBox(x_center=50, y_center=50, width=100, height=0, angle=math.pi / 4)
    minx, miny, maxx, maxy = bbox.bounds
    # A 100-wide, 0-tall box rotated 45 degrees should have extent ~70.7 in each axis
    assert minx == pytest.approx(50 - 50 * math.sqrt(2) / 2, abs=0.01)
    assert maxx == pytest.approx(50 + 50 * math.sqrt(2) / 2, abs=0.01)


def test_bbox_corners_axis_aligned():
    bbox = BoundingBox(x_center=50, y_center=50, width=20, height=10)
    corners = bbox.corners
    assert corners.shape == (4, 2)
    np.testing.assert_array_almost_equal(
        corners,
        [
            [40, 45],  # TL
            [60, 45],  # TR
            [60, 55],  # BR
            [40, 55],  # BL
        ],
    )


def test_bbox_corners_rotated():
    bbox = BoundingBox(x_center=50, y_center=50, width=20, height=0, angle=math.pi / 2)
    corners = bbox.corners
    assert corners.shape == (4, 2)
    # 90 degree rotation: width becomes vertical
    np.testing.assert_array_almost_equal(corners[0], [50, 40], decimal=5)
    np.testing.assert_array_almost_equal(corners[1], [50, 60], decimal=5)


def test_bbox_is_predicted():
    bbox = BoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert not bbox.is_predicted

    user_bbox = UserBoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert not user_bbox.is_predicted

    pred_bbox = PredictedBoundingBox(
        x_center=50, y_center=50, width=100, height=80, score=0.9
    )
    assert pred_bbox.is_predicted


def test_bbox_is_rotated():
    bbox1 = BoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert not bbox1.is_rotated

    bbox2 = BoundingBox(x_center=50, y_center=50, width=100, height=80, angle=0.5)
    assert bbox2.is_rotated


def test_bbox_is_rotated_tolerance():
    """Angles below tolerance threshold are treated as not rotated."""
    bbox = BoundingBox(x_center=50, y_center=50, width=100, height=80, angle=1e-15)
    assert not bbox.is_rotated
    # xyxy and xywh should work without raising since this is effectively unrotated
    _ = bbox.xyxy
    _ = bbox.xywh


def test_user_bbox():
    bbox = UserBoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert isinstance(bbox, BoundingBox)
    assert isinstance(bbox, UserBoundingBox)
    assert not bbox.is_predicted


def test_user_bbox_from_xyxy():
    bbox = UserBoundingBox.from_xyxy(10, 20, 110, 100)
    assert isinstance(bbox, UserBoundingBox)
    assert bbox.x_center == 60


def test_user_bbox_from_xywh():
    bbox = UserBoundingBox.from_xywh(10, 20, 100, 80)
    assert isinstance(bbox, UserBoundingBox)
    assert bbox.x_center == 60


def test_predicted_bbox():
    bbox = PredictedBoundingBox(
        x_center=50, y_center=50, width=100, height=80, score=0.95
    )
    assert isinstance(bbox, BoundingBox)
    assert isinstance(bbox, PredictedBoundingBox)
    assert bbox.is_predicted
    assert bbox.score == 0.95


def test_predicted_bbox_default_score():
    bbox = PredictedBoundingBox(x_center=50, y_center=50, width=100, height=80)
    assert bbox.score == 0.0


def test_predicted_bbox_from_xyxy():
    bbox = PredictedBoundingBox.from_xyxy(10, 20, 110, 100, score=0.8)
    assert isinstance(bbox, PredictedBoundingBox)
    assert bbox.score == 0.8
    assert bbox.x_center == 60


def test_predicted_bbox_from_xywh():
    bbox = PredictedBoundingBox.from_xywh(10, 20, 100, 80, score=0.7)
    assert isinstance(bbox, PredictedBoundingBox)
    assert bbox.score == 0.7


def test_bbox_to_roi():
    bbox = UserBoundingBox(
        x_center=50,
        y_center=50,
        width=20,
        height=10,
        name="box1",
        category="mouse",
        source="manual",
    )
    roi = bbox.to_roi()
    assert roi.name == "box1"
    assert roi.category == "mouse"
    assert roi.source == "manual"
    assert roi.area == pytest.approx(200.0)
    assert roi.bounds == pytest.approx((40.0, 45.0, 60.0, 55.0))


def test_bbox_to_roi_rotated():
    bbox = UserBoundingBox(
        x_center=50, y_center=50, width=20, height=10, angle=math.pi / 2
    )
    roi = bbox.to_roi()
    # Rotated 90 degrees: width/height swap in bounds
    assert roi.area == pytest.approx(200.0)


def test_bbox_to_roi_preserves_metadata():
    video = Video(filename="test.mp4")
    bbox = UserBoundingBox(
        x_center=50,
        y_center=50,
        width=20,
        height=10,
        video=video,
        frame_idx=3,
    )
    roi = bbox.to_roi()
    assert roi.video is video
    assert roi.frame_idx == 3


def test_bbox_to_mask():
    bbox = UserBoundingBox(x_center=10, y_center=10, width=6, height=4)
    mask = bbox.to_mask(height=20, width=20)
    assert mask.height == 20
    assert mask.width == 20
    assert mask.area > 0
    data = mask.data
    assert data[10, 10]  # Center should be filled
    assert not data[0, 0]  # Corner should be empty


def test_bbox_roundtrip_xyxy():
    bbox = BoundingBox.from_xyxy(10, 20, 50, 60)
    assert bbox.xyxy == pytest.approx((10, 20, 50, 60))


def test_bbox_roundtrip_xywh():
    bbox = BoundingBox.from_xywh(10, 20, 40, 40)
    assert bbox.xywh == pytest.approx((10, 20, 40, 40))
