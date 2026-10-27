"""
Unit and integration tests for Milestone 1: Geospatial alignment and SAM 3.1
cross-sectional width measurement engine (`src/geometry.py`).
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from src.geometry import (
    AMBULANCE_MIN_WIDTH_M,
    ATV_MIN_WIDTH_M,
    FIRE_ENGINE_MIN_WIDTH_M,
    TARGET_METRIC_CRS,
    WGS84_CRS,
    SAM31GeometricInstrument,
    compute_passability_flags,
    compute_raster_gsd_and_crs,
    evaluate_road_network_clearance,
    generate_orthogonal_cross_sections,
    load_and_align_roads,
    mask_array_to_utm_polygons,
    measure_road_edge_clearance,
)


@pytest.fixture()
def synthetic_utm_raster_fixture(tmp_path: Path) -> dict:
    """
    Create a synthetic GeoTIFF in EPSG:32616 (UTM Zone 16N) with exact 0.25 m/px GSD
    (400 x 400 px = 100m x 100m extent) and corresponding binary road, debris, and water masks.
    """
    origin_x = 653000.0
    origin_y = 3314000.0
    gsd = 0.25  # 0.25 meters per pixel
    width_px = 400
    height_px = 400
    transform = from_origin(origin_x, origin_y, gsd, gsd)

    tif_path = tmp_path / "synthetic_utm16n.tif"
    rgb_data = np.full((3, height_px, width_px), 128, dtype=np.uint8)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=height_px,
        width=width_px,
        count=3,
        dtype=np.uint8,
        crs=TARGET_METRIC_CRS,
        transform=transform,
    ) as dst:
        dst.write(rgb_data)

    # Road centerline runs horizontally from x = 653010 to 653090 (80m long) at y = 3313950
    # In pixel coordinates:
    # row = (3314000 - y) / 0.25 -> y=3313950 is row 200.
    # Nominal road surface: 6.0m total width -> y in [3313947, 3313953] -> rows [188, 212)
    road_mask = np.zeros((height_px, width_px), dtype=np.uint8)
    road_mask[188:212, 40:360] = 1  # 24 rows * 0.25m = 6.0m wide road corridor

    # Debris mask blocking 3.2m of the 6.0m road width near x in [653040, 653050] (cols 160:200)
    # Rows 188:201 -> 13 rows * 0.25m = 3.25m blocked -> leaves 6.0 - 3.25 = 2.75m clear width!
    debris_mask = np.zeros((height_px, width_px), dtype=np.uint8)
    debris_mask[188:201, 160:200] = 1

    # Water mask (initially empty; populated in water tests)
    water_mask = np.zeros((height_px, width_px), dtype=np.uint8)
    water_mask[185:215, 240:280] = 1  # Full washout across road at cols 240:280

    centerline_utm = LineString([(653015.0, 3313950.0), (653085.0, 3313950.0)])

    return {
        "tif_path": tif_path,
        "transform": transform,
        "gsd": gsd,
        "road_mask": road_mask,
        "debris_mask": debris_mask,
        "water_mask": water_mask,
        "centerline_utm": centerline_utm,
    }


def test_compute_raster_gsd_and_crs_on_synthetic_fixture(
    synthetic_utm_raster_fixture: dict,
) -> None:
    """Verify exact GSD (0.25 m/px) and CRS extraction on synthetic GeoTIFF."""
    meta = compute_raster_gsd_and_crs(synthetic_utm_raster_fixture["tif_path"])
    assert meta["native_crs"] == TARGET_METRIC_CRS
    assert meta["target_crs"] == TARGET_METRIC_CRS
    assert math.isclose(meta["gsd_m_per_px"], 0.25, rel_tol=1e-5)
    assert meta["width_px"] == 400
    assert meta["height_px"] == 400


def test_orthogonal_cross_section_sampling_5m_intervals() -> None:
    """
    Verify orthogonal cross-sections are cast every 5.0m along a road centerline
    and are strictly perpendicular to the road tangent vector.
    """
    # 25m diagonal road in UTM coordinates (3-4-5 triangle scaled by 5: dx=15m, dy=20m -> L=25m)
    line = LineString([(653000.0, 3313000.0), (653015.0, 3313020.0)])
    assert math.isclose(line.length, 25.0, rel_tol=1e-6)

    samples = generate_orthogonal_cross_sections(line, interval_m=5.0, ray_half_width_m=10.0)
    stations = [s[0] for s in samples]
    assert stations == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]

    # Tangent vector is (15, 20); check dot product with every ray vector is 0
    tx, ty = 15.0, 20.0
    for _, center_pt, ray in samples:
        assert math.isclose(ray.length, 20.0, rel_tol=1e-6)
        (x0, y0), (x1, y1) = list(ray.coords)
        rx, ry = x1 - x0, y1 - y0
        dot_product = tx * rx + ty * ry
        assert math.isclose(dot_product, 0.0, abs_tol=1e-6)
        assert ray.distance(center_pt) < 1e-6


def test_raster_mask_to_utm_polygons_and_cross_section_width(
    synthetic_utm_raster_fixture: dict,
) -> None:
    """
    Verify raster masks converted to UTM polygons produce exact metric widths when
    sampled by 5m orthogonal cross-sections.
    """
    transform = synthetic_utm_raster_fixture["transform"]
    road_poly = mask_array_to_utm_polygons(
        synthetic_utm_raster_fixture["road_mask"], transform
    )
    debris_poly = mask_array_to_utm_polygons(
        synthetic_utm_raster_fixture["debris_mask"], transform
    )
    centerline = synthetic_utm_raster_fixture["centerline_utm"]

    # 1. Without debris: clear width along the 6.0m road mask must be 6.0m
    unobstructed = measure_road_edge_clearance(
        centerline_utm=centerline,
        debris_mask_utm=None,
        water_mask_utm=None,
        road_surface_mask_utm=road_poly,
        gsd_m_per_px=0.25,
        sampling_interval_m=5.0,
    )
    assert math.isclose(unobstructed.min_clear_width_m, 6.0, abs_tol=0.05)
    assert unobstructed.passability.atv is True
    assert unobstructed.passability.ambulance is True
    assert unobstructed.passability.fire_engine is True

    # 2. With 3.25m debris intrusion on the 6.0m road: leaves 2.75m clear span
    partially_blocked = measure_road_edge_clearance(
        centerline_utm=centerline,
        debris_mask_utm=debris_poly,
        water_mask_utm=None,
        road_surface_mask_utm=road_poly,
        gsd_m_per_px=0.25,
        sampling_interval_m=5.0,
    )
    assert math.isclose(partially_blocked.min_clear_width_m, 2.75, abs_tol=0.05)
    # 2.75m > 1.2m (ATV True), 2.75m > 2.2m (Ambulance True), 2.75m <= 3.5m (Fire Engine False)
    assert partially_blocked.passability.atv is True
    assert partially_blocked.passability.ambulance is True
    assert partially_blocked.passability.fire_engine is False
    assert math.isfinite(partially_blocked.edge_weight_atv)
    assert math.isfinite(partially_blocked.edge_weight_ambulance)
    assert math.isinf(partially_blocked.edge_weight_fire_engine)


def test_split_debris_uses_max_contiguous_lane_not_sum() -> None:
    """
    If debris sits in the middle of an 8.0m wide road (y in [-4, +4]) blocking y in [-1.5, +1.5]
    (3.0m wide central rubble pile), two 2.5m lanes remain on either side.
    A vehicle cannot straddle central rubble, so min_clear_width_m must be 2.5m (not 5.0m).
    """
    centerline = LineString([(653000.0, 3313500.0), (653050.0, 3313500.0)])
    road_poly = box(652995.0, 3313496.0, 653055.0, 3313504.0)  # 8.0m wide road
    central_debris = box(653020.0, 3313498.5, 653030.0, 3313501.5)  # 3.0m wide in center

    res = measure_road_edge_clearance(
        centerline_utm=centerline,
        debris_mask_utm=central_debris,
        water_mask_utm=None,
        road_surface_mask_utm=road_poly,
        gsd_m_per_px=0.25,
        sampling_interval_m=5.0,
    )
    assert math.isclose(res.min_clear_width_m, 2.5, abs_tol=0.02)
    assert res.passability.atv is True
    assert res.passability.ambulance is True
    assert res.passability.fire_engine is False


def test_water_mask_forces_impassable_and_infinite_weight(
    synthetic_utm_raster_fixture: dict,
) -> None:
    """
    Any road segment overlapping a water mask must be marked impassable for all
    vehicle classes with weight = infinity, regardless of road width.
    """
    transform = synthetic_utm_raster_fixture["transform"]
    road_poly = mask_array_to_utm_polygons(
        synthetic_utm_raster_fixture["road_mask"], transform
    )
    water_poly = mask_array_to_utm_polygons(
        synthetic_utm_raster_fixture["water_mask"], transform
    )
    centerline = synthetic_utm_raster_fixture["centerline_utm"]

    res = measure_road_edge_clearance(
        centerline_utm=centerline,
        debris_mask_utm=None,
        water_mask_utm=water_poly,
        road_surface_mask_utm=road_poly,
        gsd_m_per_px=0.25,
        sampling_interval_m=5.0,
    )
    assert res.has_water_obstruction is True
    assert res.min_clear_width_m == 0.0
    assert res.passability.atv is False
    assert res.passability.ambulance is False
    assert res.passability.fire_engine is False
    assert math.isinf(res.edge_weight_atv)
    assert math.isinf(res.edge_weight_ambulance)
    assert math.isinf(res.edge_weight_fire_engine)
    assert "WATER" in res.reason


@pytest.mark.parametrize(
    ("width_m", "has_water", "expected"),
    [
        (4.2, False, (True, True, True)),
        (3.5, False, (True, True, False)),  # Strictly > 3.5m required for Fire Engine
        (2.8, False, (True, True, False)),
        (2.2, False, (True, False, False)),  # Strictly > 2.2m required for Ambulance
        (1.6, False, (True, False, False)),
        (1.2, False, (False, False, False)),  # Strictly > 1.2m required for ATV
        (0.5, False, (False, False, False)),
        (6.0, True, (False, False, False)),  # Water overrides width
    ],
)
def test_vehicle_clearance_thresholds_boundary_conditions(
    width_m: float,
    has_water: bool,
    expected: tuple[bool, bool, bool],
) -> None:
    """Verify strict inequality boundary conditions for ATV (>1.2m), Ambulance (>2.2m), Fire (>3.5m)."""
    flags = compute_passability_flags(width_m, has_water_obstruction=has_water)
    assert (flags.atv, flags.ambulance, flags.fire_engine) == expected


def test_real_mexico_beach_osm_reprojection_and_network_evaluation() -> None:
    """
    Verify `data/osm_roads.geojson` reprojects cleanly into `EPSG:32616` against
    `data/noaa_michael_mexico_beach.tif` and computes cross-sectional widths and passability flags.
    """
    roads_path = Path("data/osm_roads.geojson")
    raster_path = Path("data/noaa_michael_mexico_beach.tif")
    assert roads_path.exists()
    assert raster_path.exists()

    aligned_gdf = load_and_align_roads(roads_path, raster_path=raster_path)
    assert aligned_gdf.crs.to_string() == TARGET_METRIC_CRS
    assert len(aligned_gdf) > 50

    # Evaluate a subset of 25 edges for fast test execution with simulated disaster masks
    subset_gdf = aligned_gdf.iloc[:25].copy()
    enriched_gdf, records = evaluate_road_network_clearance(
        roads_source=subset_gdf,
        raster_path=raster_path,
        sampling_interval_m=5.0,
        masks_geojson_path=None,
        simulate_if_no_masks=True,
    )

    assert len(records) == len(subset_gdf)
    assert "min_clear_width_m" in enriched_gdf.columns
    assert "passable_atv" in enriched_gdf.columns
    assert "passable_ambulance" in enriched_gdf.columns
    assert "passable_fire_engine" in enriched_gdf.columns

    # Verify GSD matches the Mexico Beach NOAA GeoTIFF (~0.2804 m/px)
    assert math.isclose(records[0].gsd_m_per_px, 0.2804, rel_tol=1e-2)

    # Verify simulated disaster masks produce a mix of passable, restricted, and water-blocked edges
    assert any(r.passability.fire_engine for r in records)
    assert any(r.passability.ambulance and not r.passability.fire_engine for r in records)
    assert any(r.has_water_obstruction for r in records)


def test_sam31_geometric_instrument_mps_fp16_lifecycle() -> None:
    """Verify SAM 3.1 checkpoint loads in fp16 on Apple Silicon MPS and unloads cleanly."""
    checkpoint_path = Path("models/sam3.1/model.safetensors")
    if not checkpoint_path.exists():
        pytest.skip("SAM 3.1 checkpoint not cached locally")

    instrument = SAM31GeometricInstrument(checkpoint_dir="models/sam3.1")
    info = instrument.load_weights()
    assert info["tensor_count"] == 1468
    assert "float16" in info["dtype"]
    instrument.unload_weights()
