"""Tests for ROI data model."""

import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

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


def test_rasterize_geometry_unsupported_type():
    """Unsupported geometry types should raise TypeError."""
    point = Point(5, 5)
    with pytest.raises(TypeError, match="Unsupported geometry type"):
        _rasterize_geometry(point, 10, 10)


def test_roi_is_bbox_rotated_rectangle():
    """A rotated (non-axis-aligned) rectangle should not be considered a bbox."""
    # Diamond shape: 4 edges, all diagonal (both dx and dy are non-zero)
    coords = [(5, 0), (10, 5), (5, 10), (0, 5)]
    roi = ROI.from_polygon(coords)
    # It has 5 coords (closed ring) but edges are diagonal
    assert not roi.is_bbox


def test_rasterize_geometry_polygon_with_hole():
    """Rasterizing a polygon with a hole should leave the interior unfilled."""
    outer = [(1, 1), (9, 1), (9, 9), (1, 9)]
    inner = [(3, 3), (7, 3), (7, 7), (3, 7)]
    poly = Polygon(outer, [inner])
    mask = _rasterize_geometry(poly, 10, 10)

    # Outer region should be filled
    assert mask[2, 5]
    # Inner hole should be empty
    assert not mask[5, 5]
    # Total filled area should be less than the outer polygon alone
    outer_only = _rasterize_geometry(Polygon(outer), 10, 10)
    assert mask.sum() < outer_only.sum()


def test_rasterize_geometry_polygon():
    poly = box(2, 2, 8, 8)
    mask = _rasterize_geometry(poly, 10, 10)
    assert mask.shape == (10, 10)
    assert mask.any()
    # Center should be filled
    assert mask[5, 5]
    # Corners should not be filled
    assert not mask[0, 0]


def test_roi_geometry_validation():
    """Creating ROI with invalid geometry should raise TypeError."""
    with pytest.raises(TypeError, match="geometry must be a Shapely BaseGeometry"):
        ROI(geometry="not a geometry")
    with pytest.raises(TypeError, match="geometry must be a Shapely BaseGeometry"):
        ROI(geometry=42)
    with pytest.raises(TypeError, match="geometry must be a Shapely BaseGeometry"):
        ROI(geometry=None)


def test_roi_geometry_validation_accepts_valid():
    """Creating ROI with valid Shapely geometry types should succeed."""
    roi_point = ROI(geometry=Point(5, 5))
    assert roi_point.geometry is not None

    roi_line = ROI(geometry=LineString([(0, 0), (10, 10)]))
    assert roi_line.geometry is not None

    roi_poly = ROI(geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
    assert roi_poly.geometry is not None


def test_rasterize_multipolygon():
    """MultiPolygon with disjoint polygons should rasterize both regions."""
    poly1 = box(1, 1, 4, 4)
    poly2 = box(6, 6, 9, 9)
    multi = MultiPolygon([poly1, poly2])

    mask = _rasterize_geometry(multi, 10, 10)
    assert mask.shape == (10, 10)

    # Both regions should be filled
    assert mask[2, 2]  # Inside poly1
    assert mask[7, 7]  # Inside poly2

    # Area between the two polygons should be empty
    assert not mask[5, 5]
    # Corner should be empty
    assert not mask[0, 0]

    # Total filled area should be roughly the sum of both polygon areas
    single1 = _rasterize_geometry(poly1, 10, 10)
    single2 = _rasterize_geometry(poly2, 10, 10)
    assert mask.sum() == single1.sum() + single2.sum()


def test_rasterize_linestring_raises():
    """LineString should raise TypeError when rasterizing."""
    line = LineString([(0, 0), (10, 10)])
    with pytest.raises(TypeError, match="Unsupported geometry type"):
        _rasterize_geometry(line, 10, 10)


def test_roi_from_multi_polygon():
    """Create ROI from multiple polygon coordinate sequences."""
    polygons = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(20, 20), (30, 20), (30, 30), (20, 30)],
    ]
    roi = ROI.from_multi_polygon(polygons)
    assert roi.annotation_type == AnnotationType.SEGMENTATION
    assert isinstance(roi.geometry, MultiPolygon)
    assert len(list(roi.geometry.geoms)) == 2
    assert roi.area == pytest.approx(200.0)


def test_roi_from_multi_polygon_with_kwargs():
    """from_multi_polygon should pass through kwargs."""
    polygons = [[(0, 0), (5, 0), (5, 5), (0, 5)]]
    roi = ROI.from_multi_polygon(polygons, name="multi", category="test")
    assert roi.name == "multi"
    assert roi.category == "test"


def test_roi_explode_multi_polygon():
    """Exploding a MultiPolygon ROI should produce individual ROIs."""
    polygons = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(20, 20), (30, 20), (30, 30), (20, 30)],
    ]
    roi = ROI.from_multi_polygon(polygons, name="test", category="cat", score=0.9)
    parts = roi.explode()
    assert len(parts) == 2
    for part in parts:
        assert isinstance(part.geometry, Polygon)
        assert part.name == "test"
        assert part.category == "cat"
        assert part.score == 0.9
    assert parts[0].area == pytest.approx(100.0)
    assert parts[1].area == pytest.approx(100.0)


def test_roi_explode_single_polygon():
    """Exploding a single Polygon ROI should return [self]."""
    roi = ROI.from_bbox(0, 0, 10, 10)
    parts = roi.explode()
    assert len(parts) == 1
    assert parts[0] is roi
