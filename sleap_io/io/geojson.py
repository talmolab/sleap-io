"""GeoJSON I/O for ROIs.

Provides reading and writing of ROIs in GeoJSON format (RFC 7946). The output is
a GeoJSON FeatureCollection where each Feature corresponds to one ROI. This format
is human-readable, compatible with the `movement` library (v0.15.0+), and supported
by the broader geospatial Python ecosystem (Shapely, GeoPandas, QGIS, QuPath).
"""

from __future__ import annotations

import json
from pathlib import Path

from sleap_io.model.roi import ROI, AnnotationType


def write_rois(rois: list[ROI], filename: str | Path) -> None:
    """Write ROIs to a GeoJSON FeatureCollection file.

    Args:
        rois: List of ROIs to write.
        filename: Path to the output ``.geojson`` file.
    """
    feature_collection = {
        "type": "FeatureCollection",
        "features": [_roi_to_feature(roi) for roi in rois],
    }
    with open(filename, "w") as f:
        json.dump(feature_collection, f, indent=2)


def read_rois(filename: str | Path) -> list[ROI]:
    """Read ROIs from a GeoJSON file.

    Accepts both a FeatureCollection and a single Feature as input. Features with
    null geometries are skipped.

    Args:
        filename: Path to a ``.geojson`` file.

    Returns:
        A list of ROIs parsed from the file.
    """
    with open(filename) as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = []

    rois = []
    for feature in features:
        geom = feature.get("geometry")
        if geom is None:
            continue
        rois.append(_feature_to_roi(feature))
    return rois


def _roi_to_feature(roi: ROI) -> dict:
    """Convert an ROI to a GeoJSON Feature dict.

    Args:
        roi: The ROI to convert.

    Returns:
        A GeoJSON Feature dictionary with geometry and properties.
    """
    return roi.__geo_interface__


def _feature_to_roi(feature: dict) -> ROI:
    """Convert a GeoJSON Feature dict to an ROI.

    Args:
        feature: A GeoJSON Feature dictionary.

    Returns:
        An ROI with geometry and metadata from the Feature.
    """
    from shapely.geometry import shape

    geometry = shape(feature["geometry"])
    props = feature.get("properties") or {}

    # Resolve annotation type: prefer int value, fall back to name string
    annotation_type = AnnotationType.DEFAULT
    if "annotation_type" in props:
        annotation_type = AnnotationType(props["annotation_type"])
    elif "annotation_type_name" in props:
        annotation_type = AnnotationType[props["annotation_type_name"]]
    elif "roi_type" in props:
        annotation_type = AnnotationType[props["roi_type"]]

    return ROI(
        geometry=geometry,
        annotation_type=annotation_type,
        name=props.get("name", ""),
        category=props.get("category", ""),
        score=props.get("score"),
        source=props.get("source", ""),
        frame_idx=props.get("frame_idx"),
    )
