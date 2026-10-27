"""
Build offline web-map tiles from the AOI imagery (`scripts/build_map_tiles.py`).

The dashboards previously stretched one 512 px preview over the whole 2.5 km AOI, so the
image turned to mush when zoomed in. This cuts standard XYZ (Web Mercator) 256 px tiles from
the full-resolution GeoTIFF so Leaflet loads only what is on screen, at native detail:
zoom 19 is ~0.26 m/px at Mexico Beach's latitude, matching the 0.28 m NOAA imagery.

Tiles are written to `data/tiles/<layer>/<z>/<x>/<y>.jpg` (tiles with no imagery are
skipped) and served by the app at `/tiles/...`; no network is needed at demo time.

Usage: .venv/bin/python scripts/build_map_tiles.py [--layer post|pre] [--min-zoom 13] [--max-zoom 19]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "post": PROJECT_ROOT / "data" / "noaa_michael_mexico_beach.tif",
    "pre": PROJECT_ROOT / "data" / "naip_2017_mexico_beach.tif",
}
TILES_DIR = PROJECT_ROOT / "data" / "tiles"
TILE_PX = 256
WEB_MERCATOR_HALF = 20037508.342789244
BACKGROUND_RGB = (11, 17, 32)  # dashboard background, used where a tile edge has no imagery


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_bounds_mercator(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    size = 2 * WEB_MERCATOR_HALF / 2**z
    left = -WEB_MERCATOR_HALF + x * size
    top = WEB_MERCATOR_HALF - y * size
    return left, top - size, left + size, top


def build(layer: str, min_zoom: int, max_zoom: int) -> None:
    src_path = SOURCES[layer]
    out_root = TILES_DIR / layer
    t0 = time.perf_counter()
    written = skipped = 0
    with rasterio.open(src_path) as src:
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        bands = [1, 2, 3]
        for z in range(min_zoom, max_zoom + 1):
            x0, y0 = lonlat_to_tile(lon0, lat1, z)
            x1, y1 = lonlat_to_tile(lon1, lat0, z)
            # Downsampled zooms average many source pixels; native zooms interpolate.
            resampling = Resampling.average if z < max_zoom - 1 else Resampling.bilinear
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    merc = tile_bounds_mercator(x, y, z)
                    src_bounds = transform_bounds("EPSG:3857", src.crs, *merc)
                    window = from_bounds(*src_bounds, transform=src.transform)
                    window = window.round_offsets().round_lengths()
                    if window.width < 1 or window.height < 1:
                        skipped += 1
                        continue
                    pad = max(2, int(0.02 * max(window.width, window.height)))
                    window = rasterio.windows.Window(
                        window.col_off - pad, window.row_off - pad, window.width + 2 * pad, window.height + 2 * pad
                    )
                    data = src.read(bands, window=window, boundless=True, fill_value=0)
                    if not data.any():
                        skipped += 1
                        continue
                    tile = np.zeros((3, TILE_PX, TILE_PX), dtype=np.uint8)
                    reproject(
                        data,
                        tile,
                        src_transform=src.window_transform(window),
                        src_crs=src.crs,
                        dst_transform=transform_from_bounds(*merc, TILE_PX, TILE_PX),
                        dst_crs="EPSG:3857",
                        resampling=resampling,
                    )
                    empty = tile.max(axis=0) == 0
                    if empty.all():
                        skipped += 1
                        continue
                    rgb = np.transpose(tile, (1, 2, 0)).copy()
                    rgb[empty] = BACKGROUND_RGB
                    path = out_root / str(z) / str(x) / f"{y}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(path, format="JPEG", quality=85, optimize=True)
                    written += 1
            print(f"  {layer} z{z}: tiles x {x0}-{x1}, y {y0}-{y1}", flush=True)
    size_mb = sum(p.stat().st_size for p in out_root.rglob("*.jpg")) / 1e6
    print(f"{layer}: {written} tiles ({skipped} empty skipped), {size_mb:.0f} MB, {time.perf_counter() - t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline XYZ map tiles")
    parser.add_argument("--layer", choices=sorted(SOURCES), default="post")
    parser.add_argument("--min-zoom", type=int, default=13)
    parser.add_argument("--max-zoom", type=int, default=19)
    args = parser.parse_args()
    build(args.layer, args.min_zoom, args.max_zoom)


if __name__ == "__main__":
    main()
