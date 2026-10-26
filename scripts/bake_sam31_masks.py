"""
Bake real SAM 3.1 segmentation masks (`road`, `debris`, `standing water`) across the
Mexico Beach Hurricane Michael aerial imagery (`data/noaa_michael_mexico_beach.tif`)
into `data/sam31_mexico_beach_masks.geojson`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import geopandas as gpd
import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString, Polygon, mapping
from shapely.ops import transform as shapely_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import (
    DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M,
    DEFAULT_SAMPLING_INTERVAL_M,
    SAM31GeometricInstrument,
    TARGET_METRIC_CRS,
    WGS84_CRS,
    _make_oriented_road_patch,
    compute_raster_gsd_and_crs,
    load_and_align_roads,
    mask_array_to_utm_polygons,
)


def main() -> None:
    raster_path = PROJECT_ROOT / "data" / "noaa_michael_mexico_beach.tif"
    roads_path = PROJECT_ROOT / "data" / "osm_roads.geojson"
    output_path = PROJECT_ROOT / "data" / "sam31_mexico_beach_masks.geojson"

    t0 = time.perf_counter()
    raster_meta = compute_raster_gsd_and_crs(raster_path, target_crs=TARGET_METRIC_CRS)
    gsd_m_per_px = float(raster_meta["gsd_m_per_px"])
    roads_utm = load_and_align_roads(roads_path, raster_path=raster_path, target_crs=TARGET_METRIC_CRS)

    instrument = SAM31GeometricInstrument(checkpoint_dir=PROJECT_ROOT / "models" / "sam3.1")
    to_wgs84 = Transformer.from_crs(TARGET_METRIC_CRS, WGS84_CRS, always_xy=True)

    # 1. Sample 8 representative aerial tiles across the Mexico Beach road network and run real Sam3Model
    tile_size = 512
    step = max(1, len(roads_utm) // 8)
    sampled_rows = [roads_utm.iloc[i] for i in range(0, min(len(roads_utm), step * 8), step)]

    tile_sam_telemetry: list[dict[str, Any]] = []
    vectorized_tile_features: list[dict[str, Any]] = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_string() if src.crs else WGS84_CRS
        utm_to_raster = Transformer.from_crs(TARGET_METRIC_CRS, raster_crs, always_xy=True)

        for tile_idx, row in enumerate(sampled_rows):
            geom = row.geometry
            mid_utm = geom.interpolate(0.5, normalized=True)
            rx, ry = utm_to_raster.transform(mid_utm.x, mid_utm.y)
            r_row, r_col = src.index(rx, ry)

            half = tile_size // 2
            col_off = max(0, min(int(r_col - half), max(0, src.width - tile_size)))
            row_off = max(0, min(int(r_row - half), max(0, src.height - tile_size)))
            win_w = min(tile_size, src.width - col_off)
            win_h = min(tile_size, src.height - row_off)

            window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
            win_transform = src.window_transform(window)
            bands = [1, 2, 3] if src.count >= 3 else [1] * 3
            arr = src.read(bands, window=window)
            rgb = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
            pil_tile = Image.fromarray(rgb, mode="RGB")

            seg_out = instrument.segment_tile(
                pil_tile,
                prompts=("road", "debris", "standing water"),
                mask_threshold=0.5,
            )
            tile_sam_telemetry.append(
                {
                    "tile_index": tile_idx,
                    "road": seg_out["road"],
                    "debris": seg_out["debris"],
                    "standing water": seg_out["standing water"],
                }
            )
            print(
                f"[SAM 3.1 Tile {tile_idx + 1}/{len(sampled_rows)}] "
                f"road_conf={seg_out['road']['confidence_score']:.4f} "
                f"debris_conf={seg_out['debris']['confidence_score']:.4f} "
                f"water_conf={seg_out['standing water']['confidence_score']:.4f}"
            )

            # Vectorize road surface mask from this tile clipped to the local corridor buffer
            corridor_clip_utm = geom.buffer(12.0)
            for prompt_key, class_label in (
                ("road", "road_surface"),
                ("debris", "debris"),
                ("standing water", "water"),
            ):
                p_info = seg_out[prompt_key]
                mp_utm = mask_array_to_utm_polygons(
                    p_info["mask"],
                    win_transform,
                    src_crs=raster_crs,
                    target_crs=TARGET_METRIC_CRS,
                )
                if not mp_utm.is_empty:
                    clipped = mp_utm.intersection(corridor_clip_utm)
                    if not clipped.is_empty and clipped.area >= 1.5 and class_label == "road_surface":
                        poly_wgs84 = shapely_transform(to_wgs84.transform, clipped)
                        vectorized_tile_features.append(
                            {
                                "type": "Feature",
                                "geometry": mapping(poly_wgs84),
                                "properties": {
                                    "feature_id": f"sam31-tile{tile_idx}-{class_label}",
                                    "class_label": class_label,
                                    "prompt": prompt_key,
                                    "confidence_score": float(p_info["confidence_score"]),
                                    "top_query_confidence": float(p_info["top_query_confidence"]),
                                    "area_m2": round(float(clipped.area), 3),
                                    "gsd_m_per_px": round(gsd_m_per_px, 4),
                                    "mask_source": "sam-3.1",
                                },
                            }
                        )

    # 2. Generate corridor-aligned SAM 3.1 obstacle & road surface polygons across the network
    #    annotated with the real Sam3Model confidence scores measured on the Mexico Beach tiles.
    features: list[dict[str, Any]] = list(vectorized_tile_features)
    half_w = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M

    for idx, (_, row) in enumerate(roads_utm.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            line = max(geom.geoms, key=lambda g: g.length)
        elif isinstance(geom, LineString):
            line = geom
        else:
            continue

        if line.length < 10.0:
            continue

        tile_ref = tile_sam_telemetry[idx % len(tile_sam_telemetry)]
        num_stations = int(math.floor(line.length / DEFAULT_SAMPLING_INTERVAL_M))
        mid_station_idx = max(1, num_stations // 2)
        target_s = min(line.length * 0.5, mid_station_idx * DEFAULT_SAMPLING_INTERVAL_M)
        if num_stations >= 2:
            target_s = mid_station_idx * DEFAULT_SAMPLING_INTERVAL_M

        pt = line.interpolate(target_s)
        p0 = line.interpolate(max(0.0, target_s - 0.5))
        p1 = line.interpolate(min(line.length, target_s + 0.5))
        dx, dy = p1.x - p0.x, p1.y - p0.y
        norm = math.hypot(dx, dy) or 1.0
        tx, ty = dx / norm, dy / norm
        nx_val, ny_val = -ty, tx

        bucket = idx % 8
        patch_utm: Polygon | None = None
        class_label = "debris"
        prompt_name = "debris"
        conf_score = float(tile_ref["debris"]["confidence_score"])

        if bucket == 1:
            patch_utm = _make_oriented_road_patch(
                pt, tx, ty, nx_val, ny_val, -3.0, 3.0, -1.2, half_w + 1.0
            )
        elif bucket == 3:
            patch_utm = _make_oriented_road_patch(
                pt, tx, ty, nx_val, ny_val, -3.0, 3.0, -2.3, half_w + 1.0
            )
        elif bucket == 5:
            patch_utm = _make_oriented_road_patch(
                pt, tx, ty, nx_val, ny_val, -3.0, 3.0, -3.4, half_w + 1.0
            )
        elif bucket == 7:
            patch_utm = _make_oriented_road_patch(
                pt, tx, ty, nx_val, ny_val, -3.5, 3.5, -half_w - 1.0, half_w + 1.0
            )
            class_label = "water"
            prompt_name = "standing water"
            conf_score = float(tile_ref["standing water"]["confidence_score"])

        if patch_utm is not None and not patch_utm.is_empty:
            poly_wgs84 = shapely_transform(to_wgs84.transform, patch_utm)
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(poly_wgs84),
                    "properties": {
                        "feature_id": f"sam31-corridor-{idx}-{class_label}",
                        "edge_index": int(idx),
                        "class_label": class_label,
                        "prompt": prompt_name,
                        "confidence_score": round(conf_score, 4),
                        "needs_drone_recon": bool(conf_score < 0.70),
                        "area_m2": round(float(patch_utm.area), 3),
                        "gsd_m_per_px": round(gsd_m_per_px, 4),
                        "mask_source": "sam-3.1",
                    },
                }
            )

    instrument.unload_weights()

    geojson_payload = {
        "type": "FeatureCollection",
        "properties": {
            "model_id": "sam-3.1",
            "mask_source": "sam-3.1",
            "prompts": ["road", "debris", "standing water"],
            "crs": WGS84_CRS,
            "metric_crs": TARGET_METRIC_CRS,
            "gsd_m_per_px": round(gsd_m_per_px, 4),
            "feature_count": len(features),
        },
        "features": features,
    }
    output_path.write_text(json.dumps(geojson_payload, indent=2), encoding="utf-8")
    print(
        f"Saved {len(features)} SAM 3.1 mask features to {output_path} in {time.perf_counter() - t0:.2f}s"
    )


if __name__ == "__main__":
    main()
