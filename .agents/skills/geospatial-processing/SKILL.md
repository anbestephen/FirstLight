---
name: geospatial-processing
description: Handles raster alignment, GeoTIFF imagery loading, and SAM 3.1 mask-to-meter conversion using Rasterio and Shapely.
---

# Geospatial Processing Guidelines
- **Coordinate Alignment:** Always reproject OpenStreetMap vector graphs (`EPSG:4326`) into the native coordinate reference system (CRS) or local UTM zone of the input GeoTIFF raster before running spatial intersections.
- **Cross-Sectional Sampling:** To compute road passability, cast orthogonal ray segments at regular intervals (e.g., every 5 meters) along the OSM road centerline, intersecting against the SAM 3.1 debris, water, and road surface masks to find the minimum clear span.

