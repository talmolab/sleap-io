"""Tests for GeoJSON I/O."""

import json

import pytest
from shapely.geometry import LineString, Point

import sleap_io as sio
from sleap_io.io.geojson import read_rois, write_rois
from sleap_io.model.labels import Labels
from sleap_io.model.roi import ROI


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
    roi = ROI(geometry=Point(5, 10), category="anchor")
    path = tmp_path / "point.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    assert len(loaded) == 1
    assert loaded[0].geometry.geom_type == "Point"
    assert loaded[0].geometry.x == pytest.approx(5)
    assert loaded[0].geometry.y == pytest.approx(10)
    assert loaded[0].category == "anchor"


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
        source="model_v2",
    )
    path = tmp_path / "meta.geojson"
    write_rois([roi], path)
    loaded = read_rois(path)

    result = loaded[0]
    assert result.name == "full_meta"
    assert result.category == "cat1"
    assert result.source == "model_v2"


def test_empty_collection(tmp_path):
    """Empty list roundtrips to empty list."""
    path = tmp_path / "empty.geojson"
    write_rois([], path)
    loaded = read_rois(path)
    assert loaded == []


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
    assert roi.source == ""
    assert roi.frame_idx is None


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


def test_backward_compat_old_geojson_properties(tmp_path):
    """Old GeoJSON files with annotation_type/score properties are read without error."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
                "properties": {
                    "name": "old_roi",
                    "annotation_type": 3,
                    "annotation_type_name": "ARENA",
                    "roi_type": "ARENA",
                    "score": 0.85,
                },
            }
        ],
    }
    path = tmp_path / "old_format.geojson"
    with open(path, "w") as f:
        json.dump(geojson, f)

    loaded = read_rois(path)
    assert len(loaded) == 1
    assert loaded[0].name == "old_roi"
    # Old properties are ignored, not an error
