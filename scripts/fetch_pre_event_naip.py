"""
Fetch pre-Hurricane-Michael imagery for the Mexico Beach AOI (`scripts/fetch_pre_event_naip.py`).

Source: USDA NAIP 2017 (1.0 m, RGBN, public domain), the last survey before Michael's
landfall on Oct 10, 2018. Two quarter-quads cover the AOI (flown Oct 24 and Nov 10, 2017).
The cloud-optimised GeoTIFFs are read in place from Microsoft Planetary Computer (anonymous
SAS token) and only the AOI window is downloaded, mosaicked, and written to
`data/naip_2017_mexico_beach.tif` (EPSG:4326, matching the NOAA post-event GeoTIFF).

Run once while online; the app and grading scripts then work fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POST_EVENT_PATH = PROJECT_ROOT / "data" / "noaa_michael_mexico_beach.tif"
OUT_PATH = PROJECT_ROOT / "data" / "naip_2017_mexico_beach.tif"

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
NAIP_YEAR = "2017"


def _json(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main() -> None:
    with rasterio.open(POST_EVENT_PATH) as post:
        bounds_wgs84 = transform_bounds(post.crs, "EPSG:4326", *post.bounds)

    items = _json(
        STAC_SEARCH,
        {
            "collections": ["naip"],
            "bbox": list(bounds_wgs84),
            "datetime": f"{NAIP_YEAR}-01-01T00:00:00Z/{NAIP_YEAR}-12-31T23:59:59Z",
            "limit": 10,
        },
    )["features"]
    if not items:
        raise SystemExit(f"No NAIP {NAIP_YEAR} items cover {bounds_wgs84}")
    token = _json(SAS_TOKEN)["token"]

    sources = []
    for item in items:
        href = f"{item['assets']['image']['href']}?{token}"
        print(f"  {item['id']}  flown {item['properties']['datetime'][:10]}  gsd {item['properties']['gsd']} m")
        sources.append(rasterio.open(f"/vsicurl/{href}"))

    # Mosaic only the AOI (in the NAIP UTM grid), keeping native 1 m pixels and RGB bands.
    naip_crs = sources[0].crs
    aoi_native = transform_bounds("EPSG:4326", naip_crs, *bounds_wgs84)
    mosaic, transform = merge(sources, bounds=aoi_native, indexes=[1, 2, 3])
    for src in sources:
        src.close()

    profile = {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": 3,
        "dtype": mosaic.dtype,
        "crs": naip_crs,
        "transform": transform,
        "compress": "deflate",
        "photometric": "RGB",
    }
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(mosaic)
        dst.update_tags(
            source="USDA NAIP via Microsoft Planetary Computer (public domain)",
            items=",".join(i["id"] for i in items),
            acquired=",".join(i["properties"]["datetime"][:10] for i in items),
        )
    print(f"Saved {mosaic.shape[2]}x{mosaic.shape[1]} px pre-event mosaic -> {OUT_PATH}")
    print(f"non-zero coverage: {float(np.mean(mosaic.max(axis=0) > 0)):.1%}")


if __name__ == "__main__":
    main()
