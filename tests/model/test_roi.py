"""Tests for ROI data model."""

import numpy as np
import pytest
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.centroid import PredictedCentroid, UserCentroid
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Instance, Track
from sleap_io.model.roi import (
    ROI,
    AnnotationType,
    PredictedROI,
    UserROI,
    _apply_padding,
    _geometry_to_bbox_coords,
    _pose_to_geometry,
    _rasterize_geometry,
)
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


def test_roi_identity_field_and_to_mask():
    """ROI carries identity; from_xyxy + to_mask() preserve it."""
    ident = Identity(name="arena_A")
    r = UserROI(geometry=box(0, 0, 10, 10), identity=ident, identity_score=0.6)
    assert r.identity is ident
    assert r.identity_score == pytest.approx(0.6)
    # from_xyxy threads identity via **kwargs.
    assert UserROI.from_xyxy(0, 0, 5, 5, identity=ident).identity is ident
    # to_mask carries identity onto the SegmentationMask.
    mask = r.to_mask(height=12, width=12)
    assert mask.identity is ident
    assert mask.identity_score == pytest.approx(0.6)
    # Defaults to None.
    assert UserROI(geometry=box(0, 0, 1, 1)).identity is None


def test_roi_explode_preserves_identity():
    """explode() preserves identity on each split ROI."""
    ident = Identity(name="arena_A")
    multi = MultiPolygon([box(0, 0, 2, 2), box(5, 5, 7, 7)])
    r = UserROI(geometry=multi, identity=ident, identity_score=0.4)
    parts = r.explode()
    assert len(parts) == 2
    assert all(p.identity is ident for p in parts)
    assert all(p.identity_score == pytest.approx(0.4) for p in parts)


def test_annotation_type_enum():
    """AnnotationType enum is kept for backward compatibility."""
    assert AnnotationType.DEFAULT == 0
    assert AnnotationType.BOUNDING_BOX == 1
    assert AnnotationType.SEGMENTATION == 2
    assert AnnotationType.ARENA == 3
    assert AnnotationType.ANCHOR == 4


def test_roi_abstract():
    """ROI cannot be instantiated directly; use UserROI or PredictedROI."""
    with pytest.raises(TypeError, match="ROI is abstract"):
        ROI(geometry=box(0, 0, 5, 5))


def test_roi_identity_equality():
    roi1 = UserROI(geometry=box(0, 0, 10, 10))
    roi2 = UserROI(geometry=box(0, 0, 10, 10))
    assert roi1 is not roi2
    assert roi1 != roi2  # Identity equality via eq=False


def test_roi_from_bbox():
    roi = UserROI.from_bbox(10, 20, 30, 40)
    assert roi.bounds == (10.0, 20.0, 40.0, 60.0)
    assert roi.area == pytest.approx(30.0 * 40.0)


def test_roi_from_xyxy():
    roi = UserROI.from_xyxy(10, 20, 40, 60)
    assert roi.bounds == (10.0, 20.0, 40.0, 60.0)
    assert roi.area == pytest.approx(30.0 * 40.0)


def test_roi_from_polygon():
    coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    roi = UserROI.from_polygon(coords)
    assert roi.area == pytest.approx(100.0)


def test_roi_from_polygon_with_kwargs():
    coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    roi = UserROI.from_polygon(coords, name="test", category="cat1")
    assert roi.name == "test"
    assert roi.category == "cat1"


def test_roi_video():
    """ROI can have a video reference."""
    video = Video(filename="test.mp4")
    roi = UserROI(geometry=box(0, 0, 10, 10), video=video)
    assert roi.video is video

    roi_no_video = UserROI(geometry=box(0, 0, 10, 10))
    assert roi_no_video.video is None


def test_roi_is_bbox():
    # Rectangle aligned to axes
    roi = UserROI.from_bbox(0, 0, 10, 20)
    assert roi.is_bbox

    # Non-rectangular polygon
    roi2 = UserROI.from_polygon([(0, 0), (10, 0), (5, 10)])
    assert not roi2.is_bbox

    # Non-polygon geometry
    roi3 = UserROI(geometry=Point(0, 0))
    assert not roi3.is_bbox


def test_roi_bounds():
    roi = UserROI.from_bbox(5, 10, 20, 30)
    assert roi.bounds == (5.0, 10.0, 25.0, 40.0)


def test_roi_centroid_xy():
    roi = UserROI.from_bbox(0, 0, 10, 10)
    cx, cy = roi.centroid_xy
    assert cx == pytest.approx(5.0)
    assert cy == pytest.approx(5.0)


def test_roi_to_mask():
    roi = UserROI.from_bbox(2, 3, 4, 5, name="test_roi", category="cat")
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


def test_predicted_roi_to_mask_is_predicted():
    """A PredictedROI rasterizes to a PredictedSegmentationMask with its score."""
    from sleap_io.model.mask import PredictedSegmentationMask, UserSegmentationMask

    roi = PredictedROI.from_bbox(2, 3, 4, 5, category="cat", score=0.8)
    mask = roi.to_mask(height=20, width=20)

    assert isinstance(mask, PredictedSegmentationMask)
    assert mask.score == pytest.approx(0.8)
    assert mask.category == "cat"
    assert mask.area > 0

    # A user ROI still produces a user mask (no score field).
    user_mask = UserROI.from_bbox(2, 3, 4, 5).to_mask(height=20, width=20)
    assert type(user_mask) is UserSegmentationMask


def test_roi_to_mask_preserves_tracking_score():
    """to_mask() carries tracking_score through (regression for L4)."""
    roi = UserROI.from_bbox(2, 3, 4, 5, tracking_score=0.7)
    mask = roi.to_mask(height=20, width=20)
    assert mask.tracking_score == pytest.approx(0.7)


def test_roi_with_video():
    video = Video(filename="test.mp4")
    roi = UserROI.from_bbox(0, 0, 10, 10, video=video)
    assert roi.video is video


def test_rasterize_geometry_unsupported_type():
    """Unsupported geometry types should raise TypeError."""
    point = Point(5, 5)
    with pytest.raises(TypeError, match="Unsupported geometry type"):
        _rasterize_geometry(point, 10, 10)


def test_roi_is_bbox_rotated_rectangle():
    """A rotated (non-axis-aligned) rectangle should not be considered a bbox."""
    # Diamond shape: 4 edges, all diagonal (both dx and dy are non-zero)
    coords = [(5, 0), (10, 5), (5, 10), (0, 5)]
    roi = UserROI.from_polygon(coords)
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
        UserROI(geometry="not a geometry")
    with pytest.raises(TypeError, match="geometry must be a Shapely BaseGeometry"):
        UserROI(geometry=42)
    with pytest.raises(TypeError, match="geometry must be a Shapely BaseGeometry"):
        UserROI(geometry=None)


def test_roi_geometry_validation_accepts_valid():
    """Creating ROI with valid Shapely geometry types should succeed."""
    roi_point = UserROI(geometry=Point(5, 5))
    assert roi_point.geometry is not None

    roi_line = UserROI(geometry=LineString([(0, 0), (10, 10)]))
    assert roi_line.geometry is not None

    roi_poly = UserROI(geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
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
    roi = UserROI.from_multi_polygon(polygons)
    assert isinstance(roi.geometry, MultiPolygon)
    assert len(list(roi.geometry.geoms)) == 2
    assert roi.area == pytest.approx(200.0)


def test_roi_from_multi_polygon_with_kwargs():
    """from_multi_polygon should pass through kwargs."""
    polygons = [[(0, 0), (5, 0), (5, 5), (0, 5)]]
    roi = UserROI.from_multi_polygon(polygons, name="multi", category="test")
    assert roi.name == "multi"
    assert roi.category == "test"


def test_roi_explode_multi_polygon():
    """Exploding a MultiPolygon ROI should produce individual ROIs."""
    polygons = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(20, 20), (30, 20), (30, 30), (20, 30)],
    ]
    roi = UserROI.from_multi_polygon(polygons, name="test", category="cat")
    parts = roi.explode()
    assert len(parts) == 2
    for part in parts:
        assert isinstance(part.geometry, Polygon)
        assert part.name == "test"
        assert part.category == "cat"
    assert parts[0].area == pytest.approx(100.0)
    assert parts[1].area == pytest.approx(100.0)


def test_roi_explode_preserves_tracking_score():
    """explode() carries tracking_score onto each part (regression for L4)."""
    polygons = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(20, 20), (30, 20), (30, 30), (20, 30)],
    ]
    roi = UserROI.from_multi_polygon(polygons, tracking_score=0.7)
    parts = roi.explode()
    assert len(parts) == 2
    for part in parts:
        assert part.tracking_score == pytest.approx(0.7)


def test_roi_explode_single_polygon():
    """Exploding a single Polygon ROI should return [self]."""
    roi = UserROI.from_bbox(0, 0, 10, 10)
    parts = roi.explode()
    assert len(parts) == 1
    assert parts[0] is roi


def test_roi_geo_interface():
    """__geo_interface__ returns a valid GeoJSON Feature dict."""
    roi = UserROI.from_bbox(0, 0, 10, 10)
    gi = roi.__geo_interface__
    assert gi["type"] == "Feature"
    assert gi["geometry"]["type"] == "Polygon"
    assert "coordinates" in gi["geometry"]
    assert "properties" in gi


def test_roi_geo_interface_metadata():
    """__geo_interface__ properties contain correct metadata values."""
    roi = UserROI.from_bbox(
        0,
        0,
        10,
        10,
        name="test_roi",
        category="arena",
        source="manual",
    )
    props = roi.__geo_interface__["properties"]
    assert props["name"] == "test_roi"
    assert props["category"] == "arena"
    assert props["source"] == "manual"


def test_roi_geo_interface_defaults():
    """__geo_interface__ with default ROI has expected default property values."""
    roi = UserROI(geometry=box(0, 0, 5, 5))
    props = roi.__geo_interface__["properties"]
    assert props["name"] == ""
    assert props["category"] == ""
    assert props["source"] == ""


def test_roi_is_predicted():
    user_roi = UserROI(geometry=box(0, 0, 5, 5))
    assert user_roi.is_predicted is False
    assert isinstance(user_roi, ROI)

    pred_roi = PredictedROI(geometry=box(0, 0, 5, 5), score=0.9)
    assert pred_roi.is_predicted is True
    assert isinstance(pred_roi, ROI)
    assert pred_roi.score == 0.9


def test_user_roi():
    roi = UserROI.from_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], name="arena")
    assert roi.name == "arena"
    assert not roi.is_predicted
    assert roi.area > 0


def test_predicted_roi():
    roi = PredictedROI(geometry=box(0, 0, 5, 5), score=0.75, category="arena")
    assert roi.score == 0.75
    assert roi.is_predicted
    assert roi.category == "arena"


def test_roi_from_bbox_deprecation():
    with pytest.warns(DeprecationWarning, match="ROI.from_bbox"):
        roi = UserROI.from_bbox(0, 0, 10, 10)
    assert roi.is_bbox


def test_roi_from_xyxy_deprecation():
    with pytest.warns(DeprecationWarning, match="ROI.from_xyxy"):
        roi = UserROI.from_xyxy(0, 0, 10, 10)
    assert roi.is_bbox


# ---------------------------------------------------------------------------
# ROI.is_empty
# ---------------------------------------------------------------------------


def test_roi_is_empty_true_for_empty_polygon():
    """An empty geometry reports is_empty True."""
    roi = UserROI(geometry=Polygon())
    assert roi.is_empty is True


def test_roi_is_empty_false_for_real_geometry():
    """A geometry with spatial extent reports is_empty False."""
    roi = UserROI.from_bbox(0, 0, 10, 10)
    assert roi.is_empty is False


# ---------------------------------------------------------------------------
# ROI.to_centroid
# ---------------------------------------------------------------------------


def test_roi_to_centroid_geometric_centroid():
    """to_centroid() returns the geometric centroid by default."""
    roi = UserROI.from_bbox(0, 0, 10, 10)
    c = roi.to_centroid()
    assert isinstance(c, UserCentroid)
    assert c.x == pytest.approx(5.0)
    assert c.y == pytest.approx(5.0)
    assert not c.is_empty


def test_roi_to_centroid_representative_point():
    """representative=True uses representative_point (inside concave shapes)."""
    # A U-shape whose geometric centroid falls *outside* the polygon.
    u = Polygon([(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)])
    roi = UserROI(geometry=u)

    centroid = roi.to_centroid(representative=False)
    assert not u.contains(Point(centroid.x, centroid.y))

    rep = roi.to_centroid(representative=True)
    expected = u.representative_point()
    assert rep.x == pytest.approx(expected.x)
    assert rep.y == pytest.approx(expected.y)
    assert u.contains(Point(rep.x, rep.y))


def test_roi_to_centroid_empty_returns_nan():
    """Empty geometry yields a degenerate (NaN) centroid by default."""
    roi = UserROI(geometry=Polygon())
    c = roi.to_centroid()
    assert isinstance(c, UserCentroid)
    assert np.isnan(c.x)
    assert np.isnan(c.y)
    assert c.is_empty


def test_roi_to_centroid_empty_error_on_empty_raises():
    """error_on_empty raises instead of returning a NaN centroid."""
    roi = UserROI(geometry=Polygon())
    with pytest.raises(ValueError, match="empty ROI geometry"):
        roi.to_centroid(error_on_empty=True)


def test_roi_to_centroid_predicted_carries_score_and_metadata():
    """Predicted ROI -> PredictedCentroid with score + propagated metadata."""
    track = Track(name="t1")
    ident = Identity(name="animal_7")
    sk = Skeleton(["a", "b", "c"])
    inst = Instance.from_numpy(
        np.array([[0, 0], [10, 0], [5, 10]], dtype=float), skeleton=sk
    )
    roi = PredictedROI(
        geometry=box(0, 0, 10, 10),
        score=0.85,
        track=track,
        tracking_score=0.7,
        identity=ident,
        identity_score=0.4,
        instance=inst,
        category="mouse",
        name="m7",
        source="manual",
    )
    c = roi.to_centroid()
    assert isinstance(c, PredictedCentroid)
    assert c.score == pytest.approx(0.85)
    assert c.track is track
    assert c.tracking_score == pytest.approx(0.7)
    assert c.identity is ident
    assert c.identity_score == pytest.approx(0.4)
    assert c.instance is inst
    assert c.category == "mouse"
    assert c.name == "m7"
    assert c.source == "manual"


# ---------------------------------------------------------------------------
# ROI.to_bbox
# ---------------------------------------------------------------------------


def test_roi_to_bbox_axis_aligned():
    """Axis-aligned to_bbox() returns the geometry bounds with angle 0."""
    roi = UserROI.from_bbox(2, 3, 6, 4)  # xywh -> bounds (2,3,8,7)
    b = roi.to_bbox()
    assert isinstance(b, UserBoundingBox)
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((2.0, 3.0, 8.0, 7.0))
    assert b.angle == pytest.approx(0.0)


def test_roi_to_bbox_scalar_padding():
    """Scalar padding inflates the box outward on both axes."""
    roi = UserROI.from_bbox(0, 0, 10, 10)  # bounds (0,0,10,10)
    b = roi.to_bbox(padding=2.0)
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((-2.0, -2.0, 12.0, 12.0))


def test_roi_to_bbox_tuple_padding():
    """Per-axis padding inflates width/height independently."""
    roi = UserROI.from_bbox(0, 0, 10, 10)
    b = roi.to_bbox(padding=(1.0, 3.0))
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((-1.0, -3.0, 11.0, 13.0))


def test_roi_to_bbox_rotated():
    """rotated=True fits a minimum-area oriented box recovering w/h/angle."""
    # Rectangle 6x2 centered at origin, rotated 30 degrees.
    rect = rotate(box(-3, -1, 3, 1), 30, origin=(0, 0), use_radians=False)
    roi = UserROI(geometry=rect)
    b = roi.to_bbox(rotated=True)
    assert b.width == pytest.approx(6.0, abs=1e-6)
    assert b.height == pytest.approx(2.0, abs=1e-6)
    assert b.angle == pytest.approx(np.radians(30.0), abs=1e-6)
    assert b.x_center == pytest.approx(0.0, abs=1e-6)
    assert b.y_center == pytest.approx(0.0, abs=1e-6)
    # The reconstructed corners must reproduce the source rectangle (the
    # consistency-with-BoundingBox.corners contract). Compare both corner sets
    # lexicographically since ordering/winding may differ.
    src = np.array(rect.exterior.coords[:4], dtype=float)
    got = np.asarray(b.corners, dtype=float)
    src = src[np.lexsort((src[:, 1], src[:, 0]))]
    got = got[np.lexsort((got[:, 1], got[:, 0]))]
    assert np.allclose(src, got, atol=1e-6)


def test_roi_to_bbox_empty_returns_nan():
    """Empty geometry yields a degenerate (NaN) box by default."""
    roi = UserROI(geometry=Polygon())
    b = roi.to_bbox()
    assert b.is_empty
    assert np.isnan(b.x1) and np.isnan(b.x2)
    assert b.angle == pytest.approx(0.0)


def test_roi_to_bbox_empty_error_on_empty_raises():
    """error_on_empty raises instead of returning a NaN box."""
    roi = UserROI(geometry=Polygon())
    with pytest.raises(ValueError, match="empty ROI geometry"):
        roi.to_bbox(error_on_empty=True)


def test_roi_to_bbox_predicted_carries_score_and_metadata():
    """Predicted ROI -> PredictedBoundingBox with score + metadata."""
    track = Track(name="t1")
    roi = PredictedROI(
        geometry=box(0, 0, 10, 10),
        score=0.6,
        track=track,
        tracking_score=0.9,
        category="fly",
        name="f1",
        source="net",
    )
    b = roi.to_bbox()
    assert isinstance(b, PredictedBoundingBox)
    assert b.score == pytest.approx(0.6)
    assert b.track is track
    assert b.tracking_score == pytest.approx(0.9)
    assert b.category == "fly"
    assert b.name == "f1"
    assert b.source == "net"


# ---------------------------------------------------------------------------
# _apply_padding
# ---------------------------------------------------------------------------


def test_apply_padding_scalar():
    """Scalar padding expands all four edges equally."""
    assert _apply_padding(0, 0, 10, 10, 2) == (-2, -2, 12, 12)


def test_apply_padding_tuple():
    """Tuple padding applies px to x edges and py to y edges."""
    assert _apply_padding(0, 0, 10, 10, (1, 3)) == (-1, -3, 11, 13)


def test_apply_padding_negative_shrinks():
    """Negative padding shrinks the box and is not clamped."""
    assert _apply_padding(0, 0, 10, 10, -1) == (1, 1, 9, 9)


# ---------------------------------------------------------------------------
# _pose_to_geometry
# ---------------------------------------------------------------------------


def test_pose_to_geometry_shapes_nodes_only():
    """node_radius>0 buffers each visible node into circles."""
    pts = np.array([[0, 0], [10, 0], [5, 10]], dtype=float)
    geom = _pose_to_geometry(
        pts, [(0, 1), (1, 2)], method="shapes", node_radius=1.0, edge_radius=0.0
    )
    assert not geom.is_empty
    # Three disjoint disks -> MultiPolygon ~ 3 * pi * r^2.
    assert geom.area == pytest.approx(3 * np.pi, abs=0.1)


def test_pose_to_geometry_shapes_edges_only():
    """edge_radius>0 buffers each fully-visible edge into a capsule."""
    pts = np.array([[0, 0], [10, 0], [5, 10]], dtype=float)
    geom = _pose_to_geometry(
        pts, [(0, 1), (1, 2)], method="shapes", node_radius=0.0, edge_radius=1.0
    )
    assert not geom.is_empty
    assert geom.geom_type in ("Polygon", "MultiPolygon")


def test_pose_to_geometry_shapes_edge_skipped_when_endpoint_invisible():
    """Edges with an occluded endpoint are not drawn (empty if no shapes)."""
    pts = np.array([[0, 0], [np.nan, np.nan], [5, 10]], dtype=float)
    # Both edges touch the occluded middle node -> no edge shapes produced.
    geom = _pose_to_geometry(
        pts, [(0, 1), (1, 2)], method="shapes", node_radius=0.0, edge_radius=1.0
    )
    assert geom.is_empty


def test_pose_to_geometry_shapes_misconfig_raises():
    """Both radii 0 is a misconfiguration: always raises."""
    pts = np.array([[0, 0], [10, 0]], dtype=float)
    with pytest.raises(ValueError, match="at least one of node_radius"):
        _pose_to_geometry(
            pts, [(0, 1)], method="shapes", node_radius=0.0, edge_radius=0.0
        )


def test_pose_to_geometry_shapes_misconfig_raises_even_with_no_points():
    """Misconfig guard fires before the empty-points check."""
    pts = np.array([[np.nan, np.nan]], dtype=float)
    with pytest.raises(ValueError, match="at least one of node_radius"):
        _pose_to_geometry(pts, [], method="shapes", node_radius=0.0, edge_radius=0.0)


def test_pose_to_geometry_shapes_no_visible_points_empty():
    """All-occluded points yield an empty Polygon for shapes."""
    pts = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float)
    geom = _pose_to_geometry(pts, [(0, 1)], method="shapes", node_radius=1.0)
    assert geom.is_empty
    assert isinstance(geom, Polygon)


def test_pose_to_geometry_convex_hull_three_points():
    """convex_hull of >=3 visible points is a filled polygon."""
    pts = np.array([[0, 0], [10, 0], [5, 10]], dtype=float)
    geom = _pose_to_geometry(pts, [], method="convex_hull")
    assert isinstance(geom, Polygon)
    assert geom.area == pytest.approx(50.0)


def test_pose_to_geometry_convex_hull_one_point_is_point():
    """convex_hull of a single visible point is a Point (degenerate)."""
    pts = np.array([[5, 5], [np.nan, np.nan]], dtype=float)
    geom = _pose_to_geometry(pts, [], method="convex_hull")
    assert isinstance(geom, Point)


def test_pose_to_geometry_convex_hull_with_radius_buffers():
    """A radius buffers a degenerate hull into a Polygon."""
    pts = np.array([[5, 5], [np.nan, np.nan]], dtype=float)
    geom = _pose_to_geometry(pts, [], method="convex_hull", radius=2.0)
    assert isinstance(geom, Polygon)
    assert geom.area == pytest.approx(np.pi * 4, abs=0.2)


def test_pose_to_geometry_convex_hull_no_visible_points_empty():
    """All-occluded points yield an empty Polygon for convex_hull."""
    pts = np.array([[np.nan, np.nan]], dtype=float)
    geom = _pose_to_geometry(pts, [], method="convex_hull")
    assert geom.is_empty
    assert isinstance(geom, Polygon)


def test_pose_to_geometry_unknown_method_raises():
    """An unrecognized method raises ValueError."""
    pts = np.array([[0, 0], [10, 0]], dtype=float)
    with pytest.raises(ValueError, match="Unknown method"):
        _pose_to_geometry(pts, [(0, 1)], method="banana")


# ---------------------------------------------------------------------------
# _geometry_to_bbox_coords
# ---------------------------------------------------------------------------


def test_geometry_to_bbox_coords_axis_aligned():
    """Axis-aligned bounds with angle 0."""
    coords = _geometry_to_bbox_coords(box(0, 0, 4, 2), rotated=False)
    assert coords == pytest.approx((0.0, 0.0, 4.0, 2.0, 0.0))


def test_geometry_to_bbox_coords_empty():
    """Empty geometry yields all-NaN coords with angle 0."""
    x1, y1, x2, y2, angle = _geometry_to_bbox_coords(Polygon(), rotated=True)
    assert np.isnan(x1) and np.isnan(y1) and np.isnan(x2) and np.isnan(y2)
    assert angle == pytest.approx(0.0)


def test_geometry_to_bbox_coords_rotated_recovers_box():
    """Rotated MRR recovers width, height, center, and angle."""
    rect = rotate(box(-3, -1, 3, 1), 30, origin=(0, 0), use_radians=False)
    x1, y1, x2, y2, angle = _geometry_to_bbox_coords(rect, rotated=True)
    assert (x2 - x1) == pytest.approx(6.0, abs=1e-6)
    assert (y2 - y1) == pytest.approx(2.0, abs=1e-6)
    assert (x1 + x2) / 2 == pytest.approx(0.0, abs=1e-6)
    assert (y1 + y2) / 2 == pytest.approx(0.0, abs=1e-6)
    assert angle == pytest.approx(np.radians(30.0), abs=1e-6)


def test_geometry_to_bbox_coords_rotated_axis_aligned_preserves_area():
    """Rotated fit of an axis-aligned box preserves area and center."""
    x1, y1, x2, y2, _ = _geometry_to_bbox_coords(box(0, 0, 4, 2), rotated=True)
    assert (x2 - x1) * (y2 - y1) == pytest.approx(8.0, abs=1e-6)
    assert (x1 + x2) / 2 == pytest.approx(2.0, abs=1e-6)
    assert (y1 + y2) / 2 == pytest.approx(1.0, abs=1e-6)


def test_geometry_to_bbox_coords_rotated_degenerate_falls_back():
    """A collinear geometry has a degenerate MRR; fall back to bounds, angle 0."""
    line = LineString([(0, 0), (10, 0)])
    coords = _geometry_to_bbox_coords(line, rotated=True)
    assert coords == pytest.approx((0.0, 0.0, 10.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Round-trip: to_roi geometry rasterization consistency (via Centroid path)
# ---------------------------------------------------------------------------


def test_pose_to_geometry_then_rasterize_consistent():
    """A shapes geometry rasterizes to a non-empty mask matching its extent."""
    pts = np.array([[5, 5], [15, 5]], dtype=float)
    geom = _pose_to_geometry(
        pts, [(0, 1)], method="shapes", node_radius=0.0, edge_radius=3.0
    )
    roi = UserROI(geometry=geom)
    mask = roi.to_mask(height=20, width=25)
    assert mask.area > 0
    # The mask centroid roughly tracks the segment midpoint (10, 5).
    cy, cx = np.argwhere(mask.data).mean(axis=0)
    assert cx == pytest.approx(10.0, abs=1.5)
    assert cy == pytest.approx(5.0, abs=1.5)
