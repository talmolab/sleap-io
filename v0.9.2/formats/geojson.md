# GeoJSON Format (.geojson)

[GeoJSON](https://geojson.org/) (RFC 7946) is a JSON-based format for encoding geographic data structures. sleap-io uses it to store ROIs (regions of interest) as a human-readable, standalone format. The output is compatible with the [movement](https://github.com/neuroinformatics-unit/movement) library (v0.15.0+) and the broader geospatial Python ecosystem (Shapely, GeoPandas, QGIS, QuPath).

Each ROI is serialized as a GeoJSON Feature with geometry and metadata properties. The `ROI` class also implements the Python `__geo_interface__` protocol for direct interoperability with Shapely and other geo-aware libraries.

## Examples

```python
import sleap_io as sio
from sleap_io.model.roi import UserROI
from shapely.geometry import box

# Create some ROIs
rois = [
    UserROI(geometry=box(100, 200, 150, 280), name="box1", category="animal"),
    UserROI.from_polygon([(0, 0), (50, 0), (50, 50)], name="region"),
]

# Save to GeoJSON
sio.save_geojson(rois, "rois.geojson")

# Load back
loaded_rois = sio.load_geojson("rois.geojson")

# Also works with load_file/save_file (wraps in Labels)
labels = sio.load_file("rois.geojson")  # Returns Labels(rois=...)
sio.save_file(labels, "rois.geojson")
```

::: sleap_io.io.main.load_geojson

::: sleap_io.io.main.save_geojson
