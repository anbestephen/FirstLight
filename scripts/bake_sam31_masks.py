"""
Bake real SAM 3.1 measurements for every Mexico Beach road (`scripts/bake_sam31_masks.py`).

1. Tile the NOAA Hurricane Michael GeoTIFF along the OSM road network (504 px tiles,
   32 px overlap) and run SAM 3.1 once per tile with the text prompts "road", "debris"
   and "standing water" (fp16 on Apple Silicon MPS).
2. Mosaic the per-pixel probabilities into `data/sam31_probabilities.tif` (3 x uint8 bands,
   same grid as the imagery) so measurements can be re-run without SAM.
3. Measure visible-pavement clear width along every OSM edge with `src/sam_measure.py` and
   write `data/sam31_edge_measurements.json` (per-edge stations, widths, causes, confidence).
4. Vectorise the masks near roads into `data/sam31_mexico_beach_masks.geojson` for display.

Every polygon and width in the outputs comes from SAM 3.1 pixels; nothing is placed by rule.

Usage:
  .venv/bin/python scripts/bake_sam31_masks.py               # full bake (runs SAM)
  .venv/bin/python scripts/bake_sam31_masks.py --remeasure   # reuse sam31_probabilities.tif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import osmnx as ox
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.features import rasterize, shapes as raster_shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import LineString, mapping, shape
from shapely.ops import transform as shapely_transform, unary_union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import (  # noqa: E402
    SAM31GeometricInstrument,
    TARGET_METRIC_CRS,
    WGS84_CRS,
    compute_raster_gsd_and_crs,
)
from src.sam_measure import SAM_PROMPTS, describe_edge_measurement, measure_edge  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RASTER_PATH = DATA_DIR / "noaa_michael_mexico_beach.tif"
GRAPHML_PATH = DATA_DIR / "osm_road_graph.graphml"
PROB_TIF_PATH = DATA_DIR / "sam31_probabilities.tif"
MEASUREMENTS_PATH = DATA_DIR / "sam31_edge_measurements.json"
MASKS_GEOJSON_PATH = DATA_DIR / "sam31_mexico_beach_masks.geojson"

TILE_PX = 504
OVERLAP_PX = 32
ROAD_BUFFER_M = 20.0
CLASS_KEYS = ("road", "debris", "water")
VECTOR_CLASS_LABEL = {"road": "road_surface", "debris": "debris", "water": "water"}
MIN_POLYGON_AREA_M2 = {"road": 10.0, "debris": 2.0, "water": 4.0}


def edge_lines_utm(G: Any) -> dict[tuple[Any, Any, int], LineString]:
    """UTM centerline for every directed edge (straight node-to-node line when OSM has no geometry)."""
    to_utm = Transformer.from_crs(WGS84_CRS, TARGET_METRIC_CRS, always_xy=True)
    lines: dict[tuple[Any, Any, int], LineString] = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        geom = data.get("geometry")
        if not isinstance(geom, LineString) or geom.is_empty:
            geom = LineString(
                [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
            )
        lines[(u, v, int(k))] = shapely_transform(to_utm.transform, geom)
    return lines


def tile_origins(size: int) -> list[int]:
    stride = TILE_PX - OVERLAP_PX
    origins = list(range(0, max(1, size - TILE_PX) + 1, stride))
    if origins[-1] != size - TILE_PX:
        origins.append(size - TILE_PX)
    return origins


def run_sam_mosaic(src: Any, near_roads: np.ndarray, limit_tiles: int | None, t_start: float) -> np.ndarray:
    """Run SAM 3.1 over every road tile and mosaic uint8 probabilities (0-255) per class."""
    height, width = src.height, src.width
    windows = [
        (r0, c0)
        for r0 in tile_origins(height)
        for c0 in tile_origins(width)
        if near_roads[r0 : r0 + TILE_PX, c0 : c0 + TILE_PX].any()
    ]
    if limit_tiles:
        windows = windows[:limit_tiles]
    print(f"{len(windows)} road tiles of {TILE_PX}px", flush=True)

    probs = np.zeros((len(CLASS_KEYS), height, width), dtype=np.uint8)
    instrument = SAM31GeometricInstrument(checkpoint_dir=PROJECT_ROOT / "models" / "sam3.1")
    half = OVERLAP_PX // 2
    for i, (r0, c0) in enumerate(windows):
        rgb = np.transpose(
            src.read([1, 2, 3], window=Window(c0, r0, TILE_PX, TILE_PX)), (1, 2, 0)
        ).astype(np.uint8)
        maps = instrument.segment_tile_probabilities(Image.fromarray(rgb), SAM_PROMPTS)
        # Write each tile's interior (half the overlap trimmed) so seams use the tile that
        # saw more context; tiles on the raster border keep their outer edge.
        top = 0 if r0 == 0 else half
        left = 0 if c0 == 0 else half
        bottom = TILE_PX if r0 + TILE_PX >= height else TILE_PX - half
        right = TILE_PX if c0 + TILE_PX >= width else TILE_PX - half
        for b, key in enumerate(CLASS_KEYS):
            tile_u8 = np.clip(maps[key][top:bottom, left:right] * 255.0, 0, 255).astype(np.uint8)
            region = probs[b, r0 + top : r0 + bottom, c0 + left : c0 + right]
            np.maximum(region, tile_u8, out=region)
        if (i + 1) % 10 == 0 or i + 1 == len(windows):
            print(f"  SAM 3.1 tile {i + 1}/{len(windows)} ({time.perf_counter() - t_start:.0f}s)", flush=True)
    instrument.unload_weights()
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake SAM 3.1 road measurements")
    parser.add_argument("--limit-tiles", type=int, default=None, help="process only N tiles (debug)")
    parser.add_argument(
        "--remeasure", action="store_true", help="reuse data/sam31_probabilities.tif instead of running SAM"
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    gsd = float(compute_raster_gsd_and_crs(RASTER_PATH, target_crs=TARGET_METRIC_CRS)["gsd_m_per_px"])
    G = ox.load_graphml(GRAPHML_PATH)
    lines = edge_lines_utm(G)

    with rasterio.open(RASTER_PATH) as src:
        height, width = src.height, src.width
        raster_crs = src.crs.to_string()
        transform = src.transform
        profile = src.profile
        utm_to_raster = Transformer.from_crs(TARGET_METRIC_CRS, raster_crs, always_xy=True)
        road_buffer_raster = shapely_transform(
            utm_to_raster.transform, unary_union([ln.buffer(ROAD_BUFFER_M) for ln in lines.values()])
        )
        near_roads = rasterize(
            [(road_buffer_raster, 1)], out_shape=(height, width), transform=transform, dtype="uint8"
        ).astype(bool)

        if args.remeasure:
            with rasterio.open(PROB_TIF_PATH) as prob_src:
                probs = prob_src.read()
            print(f"Reusing SAM 3.1 probabilities from {PROB_TIF_PATH}", flush=True)
        else:
            probs = run_sam_mosaic(src, near_roads, args.limit_tiles, t_start)

    if not args.remeasure:
        profile.update(
            count=len(CLASS_KEYS), dtype="uint8", compress="deflate", nodata=None, photometric="minisblack"
        )
        profile.pop("alpha", None)
        with rasterio.open(PROB_TIF_PATH, "w", **profile) as dst:
            dst.write(probs)
            dst.descriptions = tuple(f"sam-3.1:{SAM_PROMPTS[k]}" for k in CLASS_KEYS)
        print(f"Saved probability mosaic -> {PROB_TIF_PATH}", flush=True)

    # Measure every edge; reverse directions share one measurement of the same pavement.
    inv = ~transform

    def sampler(pts_utm: np.ndarray) -> dict[str, np.ndarray]:
        xs, ys = utm_to_raster.transform(pts_utm[:, 0], pts_utm[:, 1])
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        coords = np.vstack([np.asarray(rows) - 0.5, np.asarray(cols) - 0.5])
        return {
            key: ndimage.map_coordinates(probs[b], coords, order=1, mode="constant", cval=0.0) / 255.0
            for b, key in enumerate(CLASS_KEYS)
        }

    to_wgs84 = Transformer.from_crs(TARGET_METRIC_CRS, WGS84_CRS, always_xy=True)
    measured: dict[frozenset, dict[str, Any]] = {}
    edges_out: dict[str, Any] = {}
    for (u, v, k), line in lines.items():
        canon = frozenset({(u, v, k), (v, u, k)})
        if canon not in measured:
            m = measure_edge(line, sampler)
            record = m.to_dict()
            record["reason"] = describe_edge_measurement(m, gsd)
            pinch = next((st for st in m.stations if abs(st.station_m - m.pinch_station_m) < 1e-6), None)
            record["bottleneck_lonlat"] = list(to_wgs84.transform(*pinch.center_utm)) if pinch else None
            measured[canon] = record
        edges_out[f"{u}-{v}-{k}"] = measured[canon]

    widths = [r["min_clear_width_m"] for r in measured.values()]
    summary = {
        "roads_measured": len(measured),
        "water_roads": sum(r["has_water_obstruction"] for r in measured.values()),
        "drone_recon_roads": sum(r["needs_drone_recon"] for r in measured.values()),
        "median_min_clear_width_m": round(float(np.median(widths)), 2),
        "fire_engine_passable": sum(w > 3.5 for w in widths),
        "ambulance_passable": sum(w > 2.2 for w in widths),
        "atv_passable": sum(w > 1.2 for w in widths),
    }
    MEASUREMENTS_PATH.write_text(
        json.dumps(
            {
                "model_id": "sam-3.1",
                "mask_source": "sam-3.1",
                "prompts": SAM_PROMPTS,
                "metric_crs": TARGET_METRIC_CRS,
                "gsd_m_per_px": round(gsd, 4),
                "tile_px": TILE_PX,
                "summary": summary,
                "edges": edges_out,
            }
        ),
        encoding="utf-8",
    )
    print(f"Saved measurements -> {MEASUREMENTS_PATH}\n{summary}", flush=True)

    # Vectorise masks near roads for display. Polygons come from connected components so
    # each one carries its own mean SAM probability.
    raster_to_utm = Transformer.from_crs(raster_crs, TARGET_METRIC_CRS, always_xy=True)
    features: list[dict[str, Any]] = []
    for b, key in enumerate(CLASS_KEYS):
        mask = (probs[b] >= 128) & near_roads
        if not mask.any():
            continue
        labels, n_labels = ndimage.label(mask)
        mean_by_label = ndimage.mean(probs[b], labels=labels, index=np.arange(1, n_labels + 1)) / 255.0
        for geom_dict, label_val in raster_shapes(
            labels.astype(np.int32), mask=mask, transform=transform, connectivity=4
        ):
            poly_utm = shapely_transform(raster_to_utm.transform, shape(geom_dict)).simplify(0.4)
            if poly_utm.is_empty or poly_utm.area < MIN_POLYGON_AREA_M2[key]:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(shapely_transform(to_wgs84.transform, poly_utm)),
                    "properties": {
                        "class_label": VECTOR_CLASS_LABEL[key],
                        "prompt": SAM_PROMPTS[key],
                        "confidence_score": round(float(mean_by_label[int(label_val) - 1]), 4),
                        "area_m2": round(float(poly_utm.area), 2),
                        "gsd_m_per_px": round(gsd, 4),
                        "mask_source": "sam-3.1",
                    },
                }
            )
    MASKS_GEOJSON_PATH.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "properties": {
                    "model_id": "sam-3.1",
                    "mask_source": "sam-3.1",
                    "prompts": SAM_PROMPTS,
                    "crs": WGS84_CRS,
                    "gsd_m_per_px": round(gsd, 4),
                    "feature_count": len(features),
                },
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    print(
        f"Saved {len(features)} SAM 3.1 mask polygons -> {MASKS_GEOJSON_PATH} "
        f"(total {time.perf_counter() - t_start:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
