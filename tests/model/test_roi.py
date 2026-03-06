"""Tests for ROI data model."""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from sleap_io.model.roi import ROI, AnnotationType, _rasterize_geometry
from sleap_io.model.video import Video


def test_annotation_type_enum():
    assert AnnotationType.DEFAULT == 0
    assert AnnotationType.BOUNDING_BOX == 1
    assert AnnotationType.SEGMENTATION == 2
    assert AnnotationType.ARENA == 3
    assert AnnotationType.ANCHOR == 4


def test_roi_identity_equality():
    roi1 = ROI(geometry=box(0, 0, 10, 10))
    roi2 = ROI(geometry=box(0, 0, 10, 10))
    assert roi1 is not roi2
    assert roi1 != roi2  # Identity equality via eq=False


def test_roi_from_bbox():
    roi = ROI.from_bbox(10, 20, 30, 40)
    assert roi.annotation_type == AnnotationType.BOUNDING_BOX
    assert roi.bounds == (10.0, 20.0, 40.0, 60.0)
    assert roi.area == pytest.approx(30.0 * 40.0)


def test_roi_from_xyxy():
    roi = ROI.from_xyxy(10, 20, 40, 60)
    assert roi.annotation_type == AnnotationType.BOUNDING_BOX
    assert roi.bounds == (10.0, 20.0, 40.0, 60.0)
    assert roi.area == pytest.approx(30.0 * 40.0)


def test_roi_from_polygon():
    coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    roi = ROI.from_polygon(coords)
    assert roi.annotation_type == AnnotationType.SEGMENTATION
    assert roi.area == pytest.approx(100.0)


def test_roi_from_polygon_with_kwargs():
    coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    roi = ROI.from_polygon(coords, name="test", category="cat1")
    assert roi.name == "test"
    assert roi.category == "cat1"


def test_roi_is_predicted():
    roi1 = ROI(geometry=box(0, 0, 10, 10))
    assert not roi1.is_predicted

    roi2 = ROI(geometry=box(0, 0, 10, 10), score=0.9)
    assert roi2.is_predicted


def test_roi_is_static():
    roi1 = ROI(geometry=box(0, 0, 10, 10))
    assert roi1.is_static

    roi2 = ROI(geometry=box(0, 0, 10, 10), frame_idx=5)
    assert not roi2.is_static


def test_roi_is_bbox():
    # Rectangle aligned to axes
    roi = ROI.from_bbox(0, 0, 10, 20)
    assert roi.is_bbox

    # Non-rectangular polygon
    roi2 = ROI.from_polygon([(0, 0), (10, 0), (5, 10)])
    assert not roi2.is_bbox

    # Non-polygon geometry
    roi3 = ROI(geometry=Point(0, 0))
    assert not roi3.is_bbox


def test_roi_bounds():
    roi = ROI.from_bbox(5, 10, 20, 30)
    assert roi.bounds == (5.0, 10.0, 25.0, 40.0)


def test_roi_centroid():
    roi = ROI.from_bbox(0, 0, 10, 10)
    cx, cy = roi.centroid
    assert cx == pytest.approx(5.0)
    assert cy == pytest.approx(5.0)


def test_roi_to_mask():
    roi = ROI.from_bbox(2, 3, 4, 5, name="test_roi", category="cat")
    mask = roi.to_mask(height=20, width=20)

    assert mask.height == 20
    assert mask.width == 20
    assert mask.name == "test_roi"
    assert mask.category == "cat"
    assert mask.area > 0

    # Check that the mask data has foreground pixels in the right region
    data = mask.data
    assert data[5, 4]  # Inside the bbox
    assert not data[0, 0]  # Outside the bbox


def test_roi_annotation_type_converter():
    roi = ROI(geometry=box(0, 0, 10, 10), annotation_type=1)
    assert roi.annotation_type == AnnotationType.BOUNDING_BOX
    assert isinstance(roi.annotation_type, AnnotationType)


def test_roi_with_video():
    video = Video(filename="test.mp4")
    roi = ROI.from_bbox(0, 0, 10, 10, video=video, frame_idx=5)
    assert roi.video is video
    assert roi.frame_idx == 5


def test_rasterize_geometry_empty():
    point = Point(5, 5)
    mask = _rasterize_geometry(point, 10, 10)
    assert mask.shape == (10, 10)
    assert not mask.any()  # Points can't be rasterized


def test_rasterize_geometry_polygon():
    poly = box(2, 2, 8, 8)
    mask = _rasterize_geometry(poly, 10, 10)
    assert mask.shape == (10, 10)
    assert mask.any()
    # Center should be filled
    assert mask[5, 5]
    # Corners should not be filled
    assert not mask[0, 0]
