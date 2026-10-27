"""
Blind manual labelling for the held-out benchmark (`src/labeling.py`).

Serves the samples chosen by `scripts/make_benchmark_samples.py` and records human labels:
- road cross-sections: the clear drivable width, marked by clicking both edges on a
  0.25 m/px metric (EPSG:32616) crop of the NOAA post-event image;
- buildings: a 4-class damage grade from 2017 NAIP (pre) vs 2018 NOAA (post) crops.

Model outputs (SAM 3.1 widths, Muse Spark grades) are never sent to the labelling UI, so
the labels stay independent ground truth for `benchmarks/run_benchmarks.py`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import threading
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import numpy as np
from PIL import Image
from pydantic import BaseModel
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GT_DIR = DATA_DIR / "ground_truth"
POST_EVENT_PATH = DATA_DIR / "noaa_michael_mexico_beach.tif"
PRE_EVENT_PATH = DATA_DIR / "naip_2017_mexico_beach.tif"
ROAD_SAMPLES_PATH = GT_DIR / "road_width_samples.json"
BUILDING_SAMPLES_PATH = GT_DIR / "building_samples.json"
ROAD_LABELS_PATH = GT_DIR / "road_widths_manual.json"
BUILDING_LABELS_PATH = GT_DIR / "building_damage_manual.json"

METRIC_CRS = "EPSG:32616"
CROP_RES_M = 0.25
ROAD_HALF_EXTENT_M = 40.0
BUILDING_HALF_EXTENT_M = 30.0
DAMAGE_GRADES = ("no_damage", "minor_damage", "major_damage", "destroyed", "cant_tell")

router = APIRouter()
_lock = threading.Lock()
_png_cache: dict[str, bytes] = {}


class RoadLabel(BaseModel):
    """Two clicked edge points in crop pixels, or a status when no width can be marked."""

    p1: list[float] | None = None
    p2: list[float] | None = None
    status: Literal["measured", "blocked", "cant_tell"] = "measured"


class BuildingLabel(BaseModel):
    grade: Literal["no_damage", "minor_damage", "major_damage", "destroyed", "cant_tell"]


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_labels(path: Path, sample_id: str, record: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        payload = _read_json(path, {"labels": {}})
        payload["labels"][sample_id] = {**record, "labeled_at": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def _samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(404, f"{path.name} missing; run scripts/make_benchmark_samples.py")
    return json.loads(path.read_text(encoding="utf-8"))["samples"]


def _sample(path: Path, sample_id: str) -> dict[str, Any]:
    for s in _samples(path):
        if s["sample_id"] == sample_id:
            return s
    raise HTTPException(404, f"Unknown sample {sample_id}")


def crop_origin(center_utm: list[float], half_m: float) -> tuple[float, float, int]:
    """Top-left UTM corner and pixel size of a square crop centred on `center_utm`."""
    size_px = int(round(2 * half_m / CROP_RES_M))
    return center_utm[0] - half_m, center_utm[1] + half_m, size_px


def pixel_to_utm(px: float, py: float, center_utm: list[float], half_m: float) -> tuple[float, float]:
    x0, y0, _ = crop_origin(center_utm, half_m)
    return x0 + px * CROP_RES_M, y0 - py * CROP_RES_M


def utm_to_pixel(x: float, y: float, center_utm: list[float], half_m: float) -> tuple[float, float]:
    x0, y0, _ = crop_origin(center_utm, half_m)
    return (x - x0) / CROP_RES_M, (y0 - y) / CROP_RES_M


def render_metric_crop(raster_path: Path, center_utm: list[float], half_m: float) -> np.ndarray:
    """Reproject a square window around `center_utm` into a 0.25 m/px EPSG:32616 RGB crop."""
    x0, y0, size = crop_origin(center_utm, half_m)
    dst_transform = from_origin(x0, y0, CROP_RES_M, CROP_RES_M)
    dst = np.zeros((3, size, size), dtype=np.uint8)
    with rasterio.open(raster_path) as src:
        bounds = transform_bounds(METRIC_CRS, src.crs, x0 - 5, y0 - 2 * half_m - 5, x0 + 2 * half_m + 5, y0 + 5)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        data = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        reproject(
            data,
            dst,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=METRIC_CRS,
            resampling=Resampling.bilinear,
        )
    return np.transpose(dst, (1, 2, 0))


def _png(key: str, render) -> Response:
    if key not in _png_cache:
        buf = io.BytesIO()
        Image.fromarray(render()).save(buf, format="PNG")
        _png_cache[key] = buf.getvalue()
    return Response(content=_png_cache[key], media_type="image/png")


@router.get("/api/label/roads")
def list_road_samples() -> dict[str, Any]:
    """Road cross-sections to label (ray in crop pixels) plus labels recorded so far."""
    out = []
    for s in _samples(ROAD_SAMPLES_PATH):
        ray_px = [utm_to_pixel(x, y, s["center_utm"], ROAD_HALF_EXTENT_M) for x, y in s["ray_utm"]]
        out.append({"sample_id": s["sample_id"], "ray_px": ray_px, "crop_res_m": CROP_RES_M})
    return {"samples": out, "labels": _read_json(ROAD_LABELS_PATH, {"labels": {}})["labels"]}


@router.get("/api/label/roads/{sample_id}/post.png")
def road_crop(sample_id: str) -> Response:
    s = _sample(ROAD_SAMPLES_PATH, sample_id)
    return _png(f"road:{sample_id}", lambda: render_metric_crop(POST_EVENT_PATH, s["center_utm"], ROAD_HALF_EXTENT_M))


@router.post("/api/label/roads/{sample_id}")
def save_road_label(sample_id: str, label: RoadLabel) -> dict[str, Any]:
    """Store the clear drivable width (metres along the cross-section) for one sample."""
    s = _sample(ROAD_SAMPLES_PATH, sample_id)
    record: dict[str, Any] = {"status": label.status, "edge_id": s["edge_id"], "station_index": s["station_index"]}
    if label.status == "measured":
        if not label.p1 or not label.p2:
            raise HTTPException(400, "Click both edges of the clear road (p1 and p2).")
        a = pixel_to_utm(*label.p1, s["center_utm"], ROAD_HALF_EXTENT_M)
        b = pixel_to_utm(*label.p2, s["center_utm"], ROAD_HALF_EXTENT_M)
        (rx0, ry0), (rx1, ry1) = s["ray_utm"]
        norm = math.hypot(rx1 - rx0, ry1 - ry0) or 1.0
        # Width is measured along the cross-section, matching how SAM 3.1 widths are measured.
        width = abs((b[0] - a[0]) * (rx1 - rx0) / norm + (b[1] - a[1]) * (ry1 - ry0) / norm)
        record.update({"width_m": round(width, 2), "p1_utm": [round(c, 2) for c in a], "p2_utm": [round(c, 2) for c in b]})
    elif label.status == "blocked":
        record["width_m"] = 0.0
    _write_labels(ROAD_LABELS_PATH, sample_id, record)
    return record


@router.get("/api/label/buildings")
def list_building_samples() -> dict[str, Any]:
    out = []
    for s in _samples(BUILDING_SAMPLES_PATH):
        outline = [utm_to_pixel(x, y, s["centroid_utm"], BUILDING_HALF_EXTENT_M) for x, y in s["footprint_utm"]]
        out.append({"sample_id": s["sample_id"], "outline_px": outline})
    return {
        "samples": out,
        "grades": DAMAGE_GRADES,
        "pre_event_available": PRE_EVENT_PATH.exists(),
        "labels": _read_json(BUILDING_LABELS_PATH, {"labels": {}})["labels"],
    }


@router.get("/api/label/buildings/{sample_id}/{phase}.png")
def building_crop(sample_id: str, phase: Literal["pre", "post"]) -> Response:
    s = _sample(BUILDING_SAMPLES_PATH, sample_id)
    path = PRE_EVENT_PATH if phase == "pre" else POST_EVENT_PATH
    if not path.exists():
        raise HTTPException(404, f"{path.name} missing")
    return _png(f"bldg:{sample_id}:{phase}", lambda: render_metric_crop(path, s["centroid_utm"], BUILDING_HALF_EXTENT_M))


@router.post("/api/label/buildings/{sample_id}")
def save_building_label(sample_id: str, label: BuildingLabel) -> dict[str, Any]:
    s = _sample(BUILDING_SAMPLES_PATH, sample_id)
    record = {"grade": label.grade, "building_id": s["building_id"]}
    _write_labels(BUILDING_LABELS_PATH, sample_id, record)
    return record
