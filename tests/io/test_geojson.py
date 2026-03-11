"""Tests for GeoJSON I/O."""

import json

import pytest
from shapely.geometry import LineString, Point

import sleap_io as sio
from sleap_io.io.geojson import read_rois, write_rois
from sleap_io.model.labels import Labels
from sleap_io.model.roi import ROI, AnnotationType


def test_write_rois_basic(tmp_path):
    """Written file has FeatureCollection structure with correct feature count."""
    rois = [ROI.from_bbox(0, 0, 10, 10), ROI.from_bbox(20, 20, 5, 5)]
    path = tmp_path / "rois.geojson"
    write_rois(rois, path)

    with open(path) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 2
    for feat in data["features"]:
        assert feat["type"] == "Feature"
        assert "geometry" in feat
        assert "properties" in feat


def test_read_rois_basic(tmp_path):
    """Write then read back preserves count and geometry type."""
    rois = [ROI.from_bbox(0, 0, 10, 10), ROI.from_polygon([(0, 0), (5, 0), (5, 5)])]
    path = tmp_path / "rois.geojson"
    write_rois(rois, path)

    loaded = read_rois(path)
    assert len(loaded) == 2
    assert loaded[0].geometry.geom_type == "Polygon"
    assert loaded[1].geometry.geom_type == "Polygon"


def test_roundtrip_bbox(tmp_path):
    """Bounding box ROI preserves geometry and metadata on roundtrip."""
    roi = ROI.from_bbox(10, 20, 30, 40, name="my_box", category="det")
    path = tmp_path / "bbox.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    assert len(loaded) == 1
    result = loaded[0]
    assert result.bounds == pytest.approx(roi.bounds)
    assert result.name == "my_box"
    assert result.category == "det"
    assert result.annotation_type == AnnotationType.BOUNDING_BOX


def test_roundtrip_polygon(tmp_path):
    """Polygon coordinates preserved on roundtrip."""
    coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    roi = ROI.from_polygon(coords)
    path = tmp_path / "poly.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    assert len(loaded) == 1
    assert loaded[0].geometry.geom_type == "Polygon"
    assert loaded[0].area == pytest.approx(roi.area)


def test_roundtrip_multi_polygon(tmp_path):
    """MultiPolygon geometry roundtrips correctly."""
    polygons = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(20, 20), (30, 20), (30, 30), (20, 30)],
    ]
    roi = ROI.from_multi_polygon(polygons)
    path = tmp_path / "multi.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    assert len(loaded) == 1
    assert loaded[0].geometry.geom_type == "MultiPolygon"
    assert loaded[0].area == pytest.approx(roi.area)


def test_roundtrip_point(tmp_path):
    """Point geometry roundtrips correctly."""
    roi = ROI(geometry=Point(5, 10), annotation_type=AnnotationType.ANCHOR)
    path = tmp_path / "point.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    assert len(loaded) == 1
    assert loaded[0].geometry.geom_type == "Point"
    assert loaded[0].geometry.x == pytest.approx(5)
    assert loaded[0].geometry.y == pytest.approx(10)
    assert loaded[0].annotation_type == AnnotationType.ANCHOR


def test_roundtrip_linestring(tmp_path):
    """LineString geometry roundtrips correctly."""
    roi = ROI(geometry=LineString([(0, 0), (10, 10), (20, 0)]))
    path = tmp_path / "line.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    assert len(loaded) == 1
    assert loaded[0].geometry.geom_type == "LineString"
    assert loaded[0].geometry.length == pytest.approx(roi.geometry.length)


def test_roundtrip_all_metadata(tmp_path):
    """All metadata properties preserved on roundtrip."""
    roi = ROI.from_bbox(
        0,
        0,
        10,
        10,
        name="full_meta",
        category="cat1",
        score=0.85,
        source="model_v2",
        frame_idx=7,
        annotation_type=AnnotationType.SEGMENTATION,
    )
    path = tmp_path / "meta.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    result = loaded[0]
    assert result.name == "full_meta"
    assert result.category == "cat1"
    assert result.score == pytest.approx(0.85)
    assert result.source == "model_v2"
    assert result.frame_idx == 7
    assert result.annotation_type == AnnotationType.SEGMENTATION


def test_score_none_roundtrip(tmp_path):
    """None score serializes as JSON null and roundtrips to None."""
    roi = ROI.from_bbox(0, 0, 10, 10)
    assert roi.score is None
    path = tmp_path / "null_score.geojson"
    write_rois([roi], path)

    # Verify null in JSON
    with open(path) as f:
        data = json.load(f)
    assert data["features"][0]["properties"]["score"] is None

    loaded = read_rois(path)
    assert loaded[0].score is None


def test_empty_collection(tmp_path):
    """Empty list roundtrips to empty list."""
    path = tmp_path / "empty.geojson"
    write_rois([], path)
    loaded = read_rois(path)
    assert loaded == []


def test_annotation_type_roundtrip(tmp_path):
    """Each AnnotationType enum value preserved on roundtrip."""
    for at in AnnotationType:
        roi = ROI.from_bbox(0, 0, 5, 5, annotation_type=at)
        path = tmp_path / f"at_{at.name}.geojson"
        write_rois([roi], path)
        loaded = read_rois(path)
        assert loaded[0].annotation_type == at


def test_movement_compatibility_output(tmp_path):
    """Output contains roi_type property for movement compatibility."""
    roi = ROI.from_bbox(0, 0, 10, 10, annotation_type=AnnotationType.ARENA)
    path = tmp_path / "movement.geojson"
    write_rois([roi], path)

    with open(path) as f:
        data = json.load(f)
    props = data["features"][0]["properties"]
    assert "roi_type" in props
    assert props["roi_type"] == "ARENA"


def test_movement_compatible_input(tmp_path):
    """Input with only roi_type (no annotation_type) parses correctly."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
                "properties": {"roi_type": "ARENA"},
            }
        ],
    }
    path = tmp_path / "movement_in.geojson"
    with open(path, "w") as f:
        json.dump(geojson, f)

    loaded = read_rois(path)
    assert len(loaded) == 1
    assert loaded[0].annotation_type == AnnotationType.ARENA


def test_unknown_properties_ignored(tmp_path):
    """Extra properties in the Feature don't cause errors."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [5, 5]},
                "properties": {
                    "name": "test",
                    "unknown_field": 42,
                    "extra": "data",
                },
            }
        ],
    }
    path = tmp_path / "extra.geojson"
    with open(path, "w") as f:
        json.dump(geojson, f)

    loaded = read_rois(path)
    assert len(loaded) == 1
    assert loaded[0].name == "test"


def test_missing_properties_defaults(tmp_path):
    """Feature without properties results in ROI with defaults."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1, 2]},
                "properties": None,
            }
        ],
    }
    path = tmp_path / "noprops.geojson"
    with open(path, "w") as f:
        json.dump(geojson, f)

    loaded = read_rois(path)
    assert len(loaded) == 1
    roi = loaded[0]
    assert roi.name == ""
    assert roi.category == ""
    assert roi.score is None
    assert roi.source == ""
    assert roi.frame_idx is None
    assert roi.annotation_type == AnnotationType.DEFAULT


def test_single_feature_input(tmp_path):
    """A single Feature (not FeatureCollection) is accepted."""
    geojson = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [3, 4]},
        "properties": {"name": "single"},
    }
    path = tmp_path / "single.geojson"
    with open(path, "w") as f:
        json.dump(geojson, f)

    loaded = read_rois(path)
    assert len(loaded) == 1
    assert loaded[0].name == "single"


def test_load_file_geojson(tmp_path):
    """sio.load_file with .geojson returns Labels with ROIs."""
    rois = [ROI.from_bbox(0, 0, 10, 10)]
    path = tmp_path / "test.geojson"
    write_rois(rois, path)

    result = sio.load_file(str(path))
    assert isinstance(result, Labels)
    assert len(result.rois) == 1


def test_save_file_geojson(tmp_path):
    """sio.save_file with .geojson writes valid GeoJSON."""
    labels = Labels(rois=[ROI.from_bbox(0, 0, 10, 10)])
    path = tmp_path / "test.geojson"
    sio.save_file(labels, str(path))

    with open(path) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1


def test_top_level_load_save(tmp_path):
    """sio.load_geojson and sio.save_geojson work at the top level."""
    rois = [ROI.from_bbox(5, 5, 15, 15, name="top_level")]
    path = str(tmp_path / "top.geojson")
    sio.save_geojson(rois, path)

    loaded = sio.load_geojson(path)
    assert len(loaded) == 1
    assert loaded[0].name == "top_level"
