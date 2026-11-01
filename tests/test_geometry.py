"""Tests for the shared geospatial helpers (`src/geometry.py`)."""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.geometry import (
    AMBULANCE_MIN_WIDTH_M,
    ATV_MIN_WIDTH_M,
    FIRE_ENGINE_MIN_WIDTH_M,
    TARGET_METRIC_CRS,
    WGS84_CRS,
    compute_passability_flags,
    compute_raster_gsd_and_crs,
    extract_corridor_aerial_crop_png,
    load_sam31_masks_utm,
    render_cross_section_overlay_png,
)


def _write_raster(path: Path, crs: str, transform, size: int = 64) -> Path:
    data = np.random.default_rng(0).integers(0, 255, size=(3, size, size), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=crs, transform=transform
    ) as dst:
        dst.write(data)
    return path


@pytest.mark.parametrize(
    ("width_m", "water", "expected"),
    [
        (0.0, False, (False, False, False)),
        (ATV_MIN_WIDTH_M, False, (False, False, False)),  # a vehicle needs more than its minimum
        (1.3, False, (True, False, False)),
        (AMBULANCE_MIN_WIDTH_M + 0.01, False, (True, True, False)),
        (FIRE_ENGINE_MIN_WIDTH_M + 0.01, False, (True, True, True)),
        (12.0, True, (False, False, False)),  # standing water closes the road for everyone
    ],
)
def test_passability_thresholds_and_water(width_m: float, water: bool, expected: tuple[bool, bool, bool]) -> None:
    flags = compute_passability_flags(width_m, has_water_obstruction=water)
    assert (flags.atv, flags.ambulance, flags.fire_engine) == expected
    assert flags.to_dict() == dict(zip(("atv", "ambulance", "fire_engine"), expected))


def test_gsd_of_a_metric_raster_is_its_pixel_size(tmp_path: Path) -> None:
    path = _write_raster(tmp_path / "utm.tif", TARGET_METRIC_CRS, from_origin(652000, 3315000, 0.28, 0.28))
    meta = compute_raster_gsd_and_crs(path)
    assert meta["native_crs"] == TARGET_METRIC_CRS
    assert meta["gsd_m_per_px"] == pytest.approx(0.28)
    assert meta["width_px"] == meta["height_px"] == 64
    assert meta["bands"] == 3


def test_gsd_of_a_degree_raster_is_measured_in_metres(tmp_path: Path) -> None:
    # 0.0000025 degrees per pixel at Mexico Beach's latitude is roughly a quarter of a metre.
    path = _write_raster(tmp_path / "wgs84.tif", WGS84_CRS, from_origin(-85.42, 29.95, 2.5e-6, 2.5e-6))
    meta = compute_raster_gsd_and_crs(path)
    assert meta["native_crs"] == WGS84_CRS
    assert 0.2 < meta["gsd_m_per_px"] < 0.3
    x0, y0, x1, y1 = meta["metric_bounds"]
    assert 600_000 < x0 < x1 < 700_000  # UTM zone 16N eastings


def test_mask_loader_merges_polygons_by_class_in_metres(tmp_path: Path) -> None:
    path = tmp_path / "masks.geojson"
    gpd.GeoDataFrame(
        {
            "class_label": ["road", "debris", "debris", "standing water"],
            "confidence_score": [0.9, 0.8, 0.7, 0.6],
        },
        geometry=[
            box(-85.4150, 29.9410, -85.4149, 29.9411),
            box(-85.4140, 29.9410, -85.4139, 29.9411),
            box(-85.4130, 29.9410, -85.4129, 29.9411),
            box(-85.4120, 29.9410, -85.4119, 29.9411),
        ],
        crs=WGS84_CRS,
    ).to_file(path, driver="GeoJSON")

    masks = load_sam31_masks_utm(path)
    assert len(masks["road_surface"].geoms) == 1
    assert len(masks["debris"].geoms) == 2
    assert len(masks["water"].geoms) == 1
    # About 10 m x 11 m once in metres, not a tiny area in degrees.
    assert 50 < masks["debris"].geoms[0].area < 200
    assert masks["mean_confidence"] == pytest.approx(0.75)
    assert {f["class_label"] for f in masks["features_utm"]} == {"road_surface", "debris", "water"}


def test_missing_mask_file_gives_empty_masks(tmp_path: Path) -> None:
    masks = load_sam31_masks_utm(tmp_path / "absent.geojson")
    assert masks["debris"].is_empty and masks["water"].is_empty and masks["road_surface"].is_empty
    assert masks["features_utm"] == []


def test_aerial_crop_is_a_square_png_and_has_a_placeholder(tmp_path: Path) -> None:
    path = _write_raster(tmp_path / "img.tif", WGS84_CRS, from_origin(-85.42, 29.95, 2.5e-6, 2.5e-6), size=400)
    png = extract_corridor_aerial_crop_png(path, lon=-85.4195, lat=29.9495, size_px=128)
    assert Image.open(io.BytesIO(png)).size == (128, 128)

    placeholder = extract_corridor_aerial_crop_png(tmp_path / "missing.tif", lon=0.0, lat=0.0, size_px=64)
    assert Image.open(io.BytesIO(placeholder)).size == (64, 64)


def test_real_sam31_checkpoint_loads_in_fp16_when_present() -> None:
    torch = pytest.importorskip("torch")
    from src.geometry import SAM31GeometricInstrument

    if not Path("models/sam3.1/model.safetensors").exists():
        pytest.skip("SAM 3.1 weights not downloaded")
    instrument = SAM31GeometricInstrument(checkpoint_dir="models/sam3.1")
    assert instrument.dtype == torch.float16
    assert instrument.device in ("mps", "cpu")
    instrument.unload_weights()
    assert instrument.model is None


def test_cross_section_overlay_draws_each_measured_span_where_it_was_measured(tmp_path: Path) -> None:
    path = _write_raster(tmp_path / "utm.tif", TARGET_METRIC_CRS, from_origin(652000, 3315050, 0.25, 0.25), size=200)
    y = 3315025.0
    stations = []
    for i, x in enumerate(range(652010, 652041, 5)):
        width = 0.0 if x == 652030 else 4.0
        stations.append({
            "station_m": 5.0 * i, "center_utm": [x, y], "ray_utm": [[x, y - 15], [x, y + 15]],
            "clear_segment_utm": [[x, y - 2], [x, y + 2]] if width else None,
            "clear_width_m": width, "cause": "debris" if width == 0 else "clear",
        })
    png = render_cross_section_overlay_png(path, {"stations": stations, "pinch_station_m": 20.0}, size_px=384)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    assert img.size == (384, 384)
    # Frame: 30 m of rays + 2 x 4 m margin = 38 m square centred on (652025, y).
    to_px = lambda ux, uy: (round((ux - 652006) / 38 * 384), round((y + 19 - uy) / 38 * 384))  # noqa: E731
    assert img.getpixel(to_px(652020, y + 1)) == (16, 185, 129)  # 4 m span: green, fire engines pass
    dot_x, dot_y = to_px(652030, y)
    assert img.getpixel((dot_x - 2, dot_y)) == (249, 115, 22)  # no clear span: orange debris dot
