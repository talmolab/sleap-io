"""Tests for BoundingBox data model."""

import math

import numpy as np
import pytest

from sleap_io.model.bbox import BoundingBox, PredictedBoundingBox, UserBoundingBox
from sleap_io.model.centroid import PredictedCentroid, UserCentroid
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Track


def test_bbox_abstract():
    """BoundingBox cannot be instantiated directly."""
    with pytest.raises(TypeError, match="BoundingBox is abstract"):
        BoundingBox(x1=0, y1=0, x2=10, y2=10)


def test_bbox_identity_field_and_to_roi():
    """BoundingBox carries identity; from_xywh + to_roi() preserve it."""
    ident = Identity(name="fly_A")
    b = UserBoundingBox(
        x1=0.0, y1=0.0, x2=10.0, y2=10.0, identity=ident, identity_score=0.7
    )
    assert b.identity is ident
    assert b.identity_score == pytest.approx(0.7)
    # from_xywh threads identity via **kwargs.
    assert UserBoundingBox.from_xywh(0, 0, 5, 5, identity=ident).identity is ident
    # to_roi carries identity onto the ROI.
    roi = b.to_roi()
    assert roi.identity is ident
    assert roi.identity_score == pytest.approx(0.7)
    # Defaults to None.
    assert UserBoundingBox(x1=0, y1=0, x2=1, y2=1).identity is None


def test_bbox_basic():
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    assert bbox.x_center == 50
    assert bbox.y_center == 50
    assert bbox.width == 100
    assert bbox.height == 80
    assert bbox.angle == 0.0
    assert bbox.track is None
    assert bbox.instance is None
    assert bbox.category == ""
    assert bbox.name == ""
    assert bbox.source == ""


def test_bbox_identity_equality():
    bbox1 = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    bbox2 = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    assert bbox1 is not bbox2
    assert bbox1 != bbox2


def test_bbox_from_xyxy():
    bbox = UserBoundingBox.from_xyxy(10, 20, 110, 100)
    assert bbox.x_center == 60
    assert bbox.y_center == 60
    assert bbox.width == 100
    assert bbox.height == 80


def test_bbox_from_xyxy_swapped_raises():
    """from_xyxy raises ValueError for swapped coordinates."""
    with pytest.raises(ValueError, match="x2 >= x1"):
        UserBoundingBox.from_xyxy(110, 100, 10, 20)
    with pytest.raises(ValueError, match="y2 >= y1"):
        UserBoundingBox.from_xyxy(10, 100, 110, 20)


def test_bbox_from_xyxy_with_kwargs():
    bbox = UserBoundingBox.from_xyxy(10, 20, 110, 100, category="mouse")
    assert bbox.category == "mouse"


def test_bbox_from_xywh():
    bbox = UserBoundingBox.from_xywh(10, 20, 100, 80)
    assert bbox.x_center == 60
    assert bbox.y_center == 60
    assert bbox.width == 100
    assert bbox.height == 80


def test_bbox_from_xywh_with_kwargs():
    bbox = UserBoundingBox.from_xywh(10, 20, 100, 80, name="box1", source="manual")
    assert bbox.name == "box1"
    assert bbox.source == "manual"


def test_bbox_xyxy():
    bbox = UserBoundingBox(x1=10, y1=20, x2=110, y2=100)
    x1, y1, x2, y2 = bbox.xyxy
    assert x1 == 10
    assert y1 == 20
    assert x2 == 110
    assert y2 == 100


def test_bbox_xyxy_rotated_raises():
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90, angle=0.5)
    with pytest.raises(ValueError, match="axis-aligned"):
        _ = bbox.xyxy


def test_bbox_xywh():
    bbox = UserBoundingBox(x1=10, y1=20, x2=110, y2=100)
    x, y, w, h = bbox.xywh
    assert x == 10
    assert y == 20
    assert w == 100
    assert h == 80


def test_bbox_xywh_rotated_raises():
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90, angle=0.5)
    with pytest.raises(ValueError, match="axis-aligned"):
        _ = bbox.xywh


def test_bbox_area():
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    assert bbox.area == 8000.0


def test_bbox_bounds_axis_aligned():
    bbox = UserBoundingBox(x1=10, y1=20, x2=110, y2=100)
    assert bbox.bounds == (10.0, 20.0, 110.0, 100.0)


def test_bbox_bounds_rotated():
    bbox = UserBoundingBox(x1=0, y1=50, x2=100, y2=50, angle=math.pi / 4)
    minx, miny, maxx, maxy = bbox.bounds
    # A 100-wide, 0-tall box rotated 45 degrees should have extent ~70.7 in each axis
    assert minx == pytest.approx(50 - 50 * math.sqrt(2) / 2, abs=0.01)
    assert maxx == pytest.approx(50 + 50 * math.sqrt(2) / 2, abs=0.01)


def test_bbox_corners_axis_aligned():
    bbox = UserBoundingBox(x1=40, y1=45, x2=60, y2=55)
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
    bbox = UserBoundingBox(x1=40, y1=50, x2=60, y2=50, angle=math.pi / 2)
    corners = bbox.corners
    assert corners.shape == (4, 2)
    # 90 degree rotation: width becomes vertical
    np.testing.assert_array_almost_equal(corners[0], [50, 40], decimal=5)
    np.testing.assert_array_almost_equal(corners[1], [50, 60], decimal=5)


def test_bbox_is_predicted():
    user_bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    assert not user_bbox.is_predicted

    pred_bbox = PredictedBoundingBox(x1=0, y1=10, x2=100, y2=90, score=0.9)
    assert pred_bbox.is_predicted


def test_bbox_is_rotated():
    bbox1 = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    assert not bbox1.is_rotated

    bbox2 = UserBoundingBox(x1=0, y1=10, x2=100, y2=90, angle=0.5)
    assert bbox2.is_rotated


def test_bbox_is_rotated_tolerance():
    """Angles below tolerance threshold are treated as not rotated."""
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90, angle=1e-15)
    assert not bbox.is_rotated
    # xyxy and xywh should work without raising since this is effectively unrotated
    _ = bbox.xyxy
    _ = bbox.xywh


def test_user_bbox():
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
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
    bbox = PredictedBoundingBox(x1=0, y1=10, x2=100, y2=90, score=0.95)
    assert isinstance(bbox, BoundingBox)
    assert isinstance(bbox, PredictedBoundingBox)
    assert bbox.is_predicted
    assert bbox.score == 0.95


def test_predicted_bbox_default_score():
    bbox = PredictedBoundingBox(x1=0, y1=10, x2=100, y2=90)
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
        x1=40,
        y1=45,
        x2=60,
        y2=55,
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
    bbox = UserBoundingBox(x1=40, y1=45, x2=60, y2=55, angle=math.pi / 2)
    roi = bbox.to_roi()
    # Rotated 90 degrees: width/height swap in bounds
    assert roi.area == pytest.approx(200.0)


def test_bbox_to_roi_preserves_metadata():
    bbox = UserBoundingBox(
        x1=40,
        y1=45,
        x2=60,
        y2=55,
        name="b1",
        category="mouse",
    )
    roi = bbox.to_roi()
    assert roi.name == "b1"
    assert roi.category == "mouse"


def test_bbox_to_roi_preserves_tracking_score():
    """to_roi() carries tracking_score through (regression for L4)."""
    bbox = PredictedBoundingBox(
        x1=40, y1=45, x2=60, y2=55, score=0.9, tracking_score=0.7
    )
    roi = bbox.to_roi()
    assert roi.tracking_score == pytest.approx(0.7)


def test_bbox_to_mask():
    bbox = UserBoundingBox(x1=7, y1=8, x2=13, y2=12)
    mask = bbox.to_mask(height=20, width=20)
    assert mask.height == 20
    assert mask.width == 20
    assert mask.area > 0
    data = mask.data
    assert data[10, 10]  # Center should be filled
    assert not data[0, 0]  # Corner should be empty


def test_bbox_centroid():
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    assert bbox.centroid_xy == (50.0, 50.0)


def test_bbox_centroid_nonsquare():
    bbox = UserBoundingBox(x1=10, y1=20, x2=50, y2=30)
    assert bbox.centroid_xy == (30.0, 25.0)


def test_bbox_roundtrip_xyxy():
    bbox = UserBoundingBox.from_xyxy(10, 20, 50, 60)
    assert bbox.xyxy == pytest.approx((10, 20, 50, 60))


def test_bbox_roundtrip_xywh():
    bbox = UserBoundingBox.from_xywh(10, 20, 40, 40)
    assert bbox.xywh == pytest.approx((10, 20, 40, 40))


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


def test_bbox_is_empty_false():
    """A box with finite corners is not empty."""
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    assert bbox.is_empty is False


@pytest.mark.parametrize("corner", ["x1", "y1", "x2", "y2"])
def test_bbox_is_empty_each_corner_nan(corner):
    """Any single NaN corner makes the box empty."""
    coords = dict(x1=0.0, y1=10.0, x2=100.0, y2=90.0)
    coords[corner] = float("nan")
    bbox = UserBoundingBox(**coords)
    assert bbox.is_empty is True


def test_bbox_is_empty_returns_plain_bool():
    """is_empty returns a builtin bool, not a numpy bool."""
    bbox = UserBoundingBox(x1=0, y1=0, x2=1, y2=1)
    assert type(bbox.is_empty) is bool


# ---------------------------------------------------------------------------
# to_centroid
# ---------------------------------------------------------------------------


def test_bbox_to_centroid_user():
    """User box yields a UserCentroid at its center."""
    bbox = UserBoundingBox(x1=0, y1=10, x2=100, y2=90)
    centroid = bbox.to_centroid()
    assert isinstance(centroid, UserCentroid)
    assert not isinstance(centroid, PredictedCentroid)
    assert centroid.x == pytest.approx(50.0)
    assert centroid.y == pytest.approx(50.0)


def test_bbox_to_centroid_propagates_metadata():
    """Metadata is inherited onto the centroid."""
    track = Track(name="t1")
    ident = Identity(name="fly_A")
    inst = object()
    bbox = UserBoundingBox(
        x1=0,
        y1=10,
        x2=100,
        y2=90,
        track=track,
        tracking_score=0.6,
        identity=ident,
        identity_score=0.8,
        instance=inst,
        category="mouse",
        name="b1",
        source="manual",
    )
    centroid = bbox.to_centroid()
    assert centroid.track is track
    assert centroid.tracking_score == pytest.approx(0.6)
    assert centroid.identity is ident
    assert centroid.identity_score == pytest.approx(0.8)
    assert centroid.instance is inst
    assert centroid.category == "mouse"
    assert centroid.name == "b1"
    assert centroid.source == "manual"


def test_bbox_to_centroid_predicted_carries_score():
    """Predicted box yields a PredictedCentroid carrying its score."""
    bbox = PredictedBoundingBox(
        x1=10, y1=20, x2=50, y2=30, score=0.93, tracking_score=0.4
    )
    centroid = bbox.to_centroid()
    assert isinstance(centroid, PredictedCentroid)
    assert centroid.score == pytest.approx(0.93)
    assert centroid.tracking_score == pytest.approx(0.4)
    assert centroid.x == pytest.approx(30.0)
    assert centroid.y == pytest.approx(25.0)


def test_bbox_to_centroid_degenerate_returns_nan():
    """Degenerate box returns a NaN (empty) centroid by default."""
    bbox = UserBoundingBox(x1=float("nan"), y1=0, x2=10, y2=10)
    centroid = bbox.to_centroid()
    assert isinstance(centroid, UserCentroid)
    assert centroid.is_empty
    assert np.isnan(centroid.x)
    assert np.isnan(centroid.y)


def test_bbox_to_centroid_degenerate_predicted_returns_nan():
    """Degenerate predicted box returns a NaN PredictedCentroid with score."""
    bbox = PredictedBoundingBox(x1=0, y1=0, x2=10, y2=float("nan"), score=0.5)
    centroid = bbox.to_centroid()
    assert isinstance(centroid, PredictedCentroid)
    assert centroid.score == pytest.approx(0.5)
    assert centroid.is_empty


def test_bbox_to_centroid_degenerate_error_on_empty():
    """error_on_empty raises for degenerate boxes."""
    bbox = UserBoundingBox(x1=float("nan"), y1=0, x2=10, y2=10)
    with pytest.raises(ValueError, match="degenerate"):
        bbox.to_centroid(error_on_empty=True)


def test_bbox_to_centroid_error_on_empty_ok_when_finite():
    """error_on_empty does not raise for finite boxes."""
    bbox = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)
    centroid = bbox.to_centroid(error_on_empty=True)
    assert centroid.x == pytest.approx(5.0)
    assert centroid.y == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# pad
# ---------------------------------------------------------------------------


def test_bbox_pad_scalar():
    """Scalar padding inflates both axes equally."""
    bbox = UserBoundingBox(x1=10, y1=20, x2=30, y2=40)
    padded = bbox.pad(5)
    assert (padded.x1, padded.y1, padded.x2, padded.y2) == pytest.approx(
        (5, 15, 35, 45)
    )


def test_bbox_pad_per_axis():
    """Tuple padding inflates each axis independently."""
    bbox = UserBoundingBox(x1=10, y1=20, x2=30, y2=40)
    padded = bbox.pad((2, 7))
    assert (padded.x1, padded.y1, padded.x2, padded.y2) == pytest.approx(
        (8, 13, 32, 47)
    )


def test_bbox_pad_negative_shrinks():
    """Negative padding shrinks the box and is not clamped."""
    bbox = UserBoundingBox(x1=10, y1=20, x2=30, y2=40)
    padded = bbox.pad(-3)
    assert (padded.x1, padded.y1, padded.x2, padded.y2) == pytest.approx(
        (13, 23, 27, 37)
    )


def test_bbox_pad_zero_is_noop():
    """Zero padding leaves coordinates unchanged."""
    bbox = UserBoundingBox(x1=10, y1=20, x2=30, y2=40)
    padded = bbox.pad(0)
    assert (padded.x1, padded.y1, padded.x2, padded.y2) == pytest.approx(
        (10, 20, 30, 40)
    )


def test_bbox_pad_returns_new_object():
    """Pad returns a new object, leaving the original unchanged."""
    bbox = UserBoundingBox(x1=10, y1=20, x2=30, y2=40)
    padded = bbox.pad(5)
    assert padded is not bbox
    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (10, 20, 30, 40)


def test_bbox_pad_preserves_angle_and_metadata():
    """Pad preserves angle and all metadata on a user box."""
    track = Track(name="t1")
    ident = Identity(name="fly_A")
    inst = object()
    bbox = UserBoundingBox(
        x1=10,
        y1=20,
        x2=30,
        y2=40,
        angle=0.5,
        track=track,
        tracking_score=0.6,
        identity=ident,
        identity_score=0.8,
        instance=inst,
        category="mouse",
        name="b1",
        source="manual",
    )
    padded = bbox.pad(5)
    assert isinstance(padded, UserBoundingBox)
    assert padded.angle == pytest.approx(0.5)
    assert padded.track is track
    assert padded.tracking_score == pytest.approx(0.6)
    assert padded.identity is ident
    assert padded.identity_score == pytest.approx(0.8)
    assert padded.instance is inst
    assert padded.category == "mouse"
    assert padded.name == "b1"
    assert padded.source == "manual"


def test_bbox_pad_predicted_preserves_type_and_score():
    """Padding a predicted box keeps the type and score."""
    bbox = PredictedBoundingBox(x1=10, y1=20, x2=30, y2=40, score=0.9, angle=0.3)
    padded = bbox.pad(5)
    assert isinstance(padded, PredictedBoundingBox)
    assert padded.score == pytest.approx(0.9)
    assert padded.angle == pytest.approx(0.3)
    assert (padded.x1, padded.y1, padded.x2, padded.y2) == pytest.approx(
        (5, 15, 35, 45)
    )
