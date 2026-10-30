"""
Re-measure roads from drone imagery (`src/drone.py`).

Roads flagged "drone check advised" are measured again on CRASAR's drone orthomosaics of
Mexico Beach (DJI Mavic Pro, Oct 13 and Oct 14, 2018, ~2 cm/px; CRASAR-U-DROIDs, CC BY 4.0).
The orthomosaics are 11-15 GB, but they carry internal overviews, so only a patch around the
road is read over HTTP at 1/8 resolution (~0.16 m/px, about 1.8x sharper than the NOAA image).
SAM 3.1 then segments that patch with the same prompts, and `measure_edge` measures it with the
same 5 m cross-sections, so the two measurements differ only in the imagery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely.geometry import LineString

from src.sam_measure import (
    SAM_PROMPTS,
    StationMeasurement,
    describe_edge_measurement,
    max_width_for_highway,
    measure_edge,
    summarise_stations,
)

HF_BASE = "https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs/resolve/main/test/imagery/UAS"


@dataclass(frozen=True)
class DroneScene:
    name: str
    flown: str  # ISO date of the flight

    @property
    def url(self) -> str:
        return f"/vsicurl/{HF_BASE}/{self.name}"


SCENES = (
    DroneScene("10132018-MexicoBeach.geo.tif", "2018-10-13"),
    DroneScene("10142018-MexicoBeach.geo.tif", "2018-10-14"),
)
SOURCE_LABEL = "CRASAR-U-DROIDs sUAS (DJI Mavic Pro), CC BY 4.0"
OVERVIEW_LEVEL = 2  # 1/8 of ~1.96 cm/px = ~0.16 m/px
PATCH_MARGIN_M = 20.0
MIN_COVERAGE = 0.97  # share of the road's stations that must have drone pixels
TILE_PX = 504
OVERLAP_PX = 32
# At ~0.16 m/px one confident cross-section is reliable, so a single near-total gap closes the road
# (the NOAA bake requires two stations to suppress 0.28 m noise). Chosen with the CRASAR road
# labels in view: it keeps every road CRASAR marks blocked as closed as the NOAA measurement did.
DRONE_SUSTAINED = False
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}


@dataclass
class DronePatch:
    """RGB pixels around one road, the mask of pixels the drone actually imaged, and georeferencing."""

    scene: DroneScene
    rgb: np.ndarray  # (H, W, 3) uint8
    valid: np.ndarray  # (H, W) bool
    transform: Any  # rasterio Affine, EPSG:32616
    gsd_m_per_px: float


def read_patch(scene: DroneScene, line_utm: LineString, *, opener: Callable[..., Any] = rasterio.open) -> DronePatch:
    """Read the patch of `scene` around a UTM 16N road centerline at the overview resolution."""
    x0, y0, x1, y1 = line_utm.buffer(PATCH_MARGIN_M).bounds
    with rasterio.Env(**GDAL_ENV), opener(scene.url, overview_level=OVERVIEW_LEVEL) as src:
        window = from_bounds(x0, y0, x1, y1, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        data = src.read(window=window, boundless=False)
        transform = src.window_transform(window)
        gsd = abs(float(transform.a))
    rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
    valid = data[3] > 0 if data.shape[0] >= 4 else rgb.any(axis=2)
    return DronePatch(scene, rgb, valid, transform, gsd)


def coverage(patch: DronePatch, line_utm: LineString, samples: int = 25) -> float:
    """Share of points along the centerline that fall on imaged (non-empty) drone pixels."""
    inv = ~patch.transform
    hits = 0
    for t in np.linspace(0.0, 1.0, samples):
        p = line_utm.interpolate(float(t), normalized=True)
        col, row = inv * (p.x, p.y)
        r, c = int(row), int(col)
        hits += 0 <= r < patch.valid.shape[0] and 0 <= c < patch.valid.shape[1] and bool(patch.valid[r, c])
    return hits / samples


def _tile_origins(size: int) -> list[int]:
    if size <= TILE_PX:
        return [0]
    stride = TILE_PX - OVERLAP_PX
    origins = list(range(0, size - TILE_PX + 1, stride))
    if origins[-1] != size - TILE_PX:
        origins.append(size - TILE_PX)
    return origins


def segment_patch(patch: DronePatch, instrument: Any) -> dict[str, np.ndarray]:
    """SAM 3.1 probabilities over the patch in overlapping 504 px tiles; empty pixels stay 0."""
    from PIL import Image

    h, w = patch.valid.shape
    probs = {key: np.zeros((h, w), dtype=np.float32) for key in SAM_PROMPTS}
    for r0 in _tile_origins(h):
        for c0 in _tile_origins(w):
            tile = patch.rgb[r0 : r0 + TILE_PX, c0 : c0 + TILE_PX]
            if not patch.valid[r0 : r0 + TILE_PX, c0 : c0 + TILE_PX].any():
                continue
            maps = instrument.segment_tile_probabilities(Image.fromarray(tile), SAM_PROMPTS)
            th, tw = tile.shape[:2]
            for key, prob in maps.items():
                region = probs[key][r0 : r0 + th, c0 : c0 + tw]
                np.maximum(region, prob[:th, :tw], out=region)
    for key in probs:
        probs[key][~patch.valid] = 0.0
    return probs


def remeasure(
    line_utm: LineString,
    highway: Any,
    patch: DronePatch,
    probs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Measure the road on drone-image probabilities; same record shape as the NOAA bake."""
    inv = ~patch.transform

    def sampler(pts: np.ndarray) -> dict[str, np.ndarray]:
        cols, rows = inv * (pts[:, 0], pts[:, 1])
        coords = np.vstack([np.asarray(rows) - 0.5, np.asarray(cols) - 0.5])
        return {k: ndimage.map_coordinates(p, coords, order=1, mode="constant", cval=0.0) for k, p in probs.items()}

    m = measure_edge(line_utm, sampler, max_width_m=max_width_for_highway(highway), sustained=DRONE_SUSTAINED)
    record = m.to_dict()
    record["reason"] = _drone_reason(m, patch.gsd_m_per_px, patch.scene.flown)
    record["measurement_source"] = "drone"
    record["imagery"] = {
        "scene": patch.scene.name,
        "flown": patch.scene.flown,
        "gsd_m_per_px": round(patch.gsd_m_per_px, 4),
        "source": SOURCE_LABEL,
    }
    return record


def _drone_reason(m: Any, gsd: float, flown: str) -> str:
    return describe_edge_measurement(m, gsd).replace("SAM 3.1", f"SAM 3.1 on drone imagery ({flown})", 1)


def rederive(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute a stored drone record's verdict from its saved cross-sections (no SAM rerun)."""
    stations = [StationMeasurement.from_dict(st) for st in record["stations"]]
    m = summarise_stations(
        stations, float(record["length_m"]), float(record["debris_area_m2"]), float(record["water_area_m2"]),
        sustained=DRONE_SUSTAINED,
    )
    out = {**record, **m.to_dict()}
    out["reason"] = _drone_reason(m, float(record["imagery"]["gsd_m_per_px"]), record["imagery"]["flown"])
    return out
