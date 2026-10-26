"""
End-to-end integration tests for Milestone 4: FastAPI backend (`src/app.py`)
and zero-CDN offline field dashboard (`ui/index.html`, `ui/app.js`, `ui/style.css`).
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.app import app

client = TestClient(app)


def test_health_endpoint_reports_tiers_and_mps_mlx_health() -> None:
    """Verify GET /api/health returns VLM tier readiness, MPS/MLX health, and raster GSD."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert data["offline_ready"] is True
    assert "vlm_tiers" in data
    assert data["vlm_tiers"]["tier_3_deterministic_ready"] is True
    assert data["vlm_tiers"]["tier_1_timeout_s"] == 2.5

    assert "device_health" in data
    assert data["device_health"]["enforced_precision"] == "float16 / 4-bit"
    assert data["device_health"]["unified_memory"]["cleared"] is True

    assert "geospatial" in data
    assert data["geospatial"]["metric_crs"] == "EPSG:32616"
    assert abs(data["geospatial"]["gsd_m_per_px"] - 0.2804) < 0.01
    assert data["geospatial"]["graph_nodes"] == 125
    assert data["geospatial"]["graph_edges"] == 312


def test_layers_endpoint_returns_cached_geojson_and_envelope() -> None:
    """Verify GET /api/layers returns Mexico Beach roads, buildings, and raster envelope GeoJSON."""
    res = client.get("/api/layers")
    assert res.status_code == 200
    data = res.json()

    assert data["roads"]["type"] == "FeatureCollection"
    assert len(data["roads"]["features"]) == 312
    sample_road_props = data["roads"]["features"][0]["properties"]
    assert "min_clear_width_m" in sample_road_props
    assert "passable_atv" in sample_road_props
    assert "passable_ambulance" in sample_road_props
    assert "passable_fire_engine" in sample_road_props

    assert data["buildings"]["type"] == "FeatureCollection"
    assert len(data["buildings"]["features"]) == 334

    assert data["raster_envelope"]["type"] == "Feature"
    assert data["raster_envelope"]["geometry"]["type"] == "Polygon"
    assert len(data["bounds_wgs84"]) == 4


def test_assess_endpoint_runs_vlm_failover_and_releases_memory() -> None:
    """Verify POST /api/assess grades damage through VLMFailoverRouter and clears memory."""
    payload = {
        "target_id": "hwy98-test-seg",
        "road_name": "US Highway 98 E",
        "min_clear_width_m": 2.45,
        "debris_area_m2": 31.2,
        "water_area_m2": 0.0,
        "has_water_obstruction": False,
        "building_debris_overlap_ratio": 0.42,
        "gsd_m_per_px": 0.2804,
        "mode": "force_deterministic",
    }
    res = client.post("/api/assess", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["target_id"] == "hwy98-test-seg"
    assert data["building_damage_severity"] == "major_damage"
    assert data["road_damage_severity"] == "minor_damage"
    assert data["is_high_severity"] is True
    assert data["passability"] == {
        "atv": True,
        "ambulance": True,
        "fire_engine": False,
    }
    assert data["memory_cleared"] is True
    assert "2.45m" in data["passability_rationale"]
    assert "0.2804 m/px GSD" in data["passability_rationale"]


def test_route_endpoint_across_all_vehicle_classes() -> None:
    """Verify POST /api/route returns GeoJSON route geometry, steps, and class comparisons."""
    layers = client.get("/api/layers").json()
    origin = [layers["default_origin"]["lon"], layers["default_origin"]["lat"]]
    dest = [layers["default_destination"]["lon"], layers["default_destination"]["lat"]]

    for vc in ("atv", "ambulance", "fire_engine"):
        res = client.post(
            "/api/route",
            json={
                "origin": origin,
                "destination": dest,
                "vehicle_class": vc,
                "algorithm": "dijkstra",
            },
        )
        assert res.status_code == 200
        rdata = res.json()
        assert rdata["vehicle_class"] == vc
        assert rdata["geojson_geometry"]["type"] == "LineString"
        assert "edge_steps" in rdata
        assert "blocked_edges_avoided" in rdata
        assert "all_classes_summary" in rdata
        assert set(rdata["all_classes_summary"].keys()) == {
            "atv",
            "ambulance",
            "fire_engine",
        }


def test_clearance_roi_and_simulate_clear_endpoints() -> None:
    """
    Verify GET /api/clearance-roi returns bottlenecks sorted descending by ROI and
    POST /api/simulate-clear updates clearances in real time.
    """
    res = client.get("/api/clearance-roi?vehicle_class=fire_engine")
    assert res.status_code == 200
    data = res.json()

    assert data["vehicle_class"] == "fire_engine"
    recs = data["recommendations"]
    assert len(recs) > 0

    for i in range(len(recs) - 1):
        assert recs[i]["clearance_roi"] >= recs[i + 1]["clearance_roi"]
        assert recs[i]["water_overlap_ratio"] == 0.0

    # Simulate clearing the top-ranked bottleneck
    top = recs[0]
    sim_res = client.post(
        "/api/simulate-clear",
        json={
            "edge_u": top["edge_u"],
            "edge_v": top["edge_v"],
            "edge_key": top["edge_key"],
            "vehicle_class": "fire_engine",
        },
    )
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    assert len(sim_data["cleared_edges"]) == 1

    # Reset simulated clearances
    reset_res = client.post(
        "/api/simulate-clear",
        json={"reset": True, "vehicle_class": "fire_engine"},
    )
    assert reset_res.status_code == 200
    assert reset_res.json()["cleared_edges"] == []


def test_zero_cdn_static_asset_integrity() -> None:
    """
    Verify ui/index.html, ui/style.css, and ui/app.js contain zero external CDN
    dependencies and all referenced script/style/image assets resolve locally with 200 OK.
    """
    index_res = client.get("/")
    assert index_res.status_code == 200
    html = index_res.text

    # Confirm zero external http:// or https:// references in HTML, CSS, and JS
    for ui_file in ("ui/index.html", "ui/style.css", "ui/app.js"):
        content = Path(ui_file).read_text(encoding="utf-8")
        external_urls = re.findall(r"https?://[^\s\"')>]+", content)
        assert external_urls == [], f"Found external CDN URL(s) in {ui_file}: {external_urls}"

    # Extract all <script src="..."> and <link href="..."> from index.html and verify 200 OK
    script_srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html)
    link_hrefs = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', html)
    all_assets = script_srcs + link_hrefs + ["/api/raster-preview.png"]

    assert "/vendor/leaflet.js" in script_srcs
    assert "/app.js" in script_srcs
    assert "/vendor/leaflet.css" in link_hrefs
    assert "/style.css" in link_hrefs

    for asset_path in all_assets:
        assert asset_path.startswith("/"), f"Asset {asset_path} must be a local path"
        asset_res = client.get(asset_path)
        assert asset_res.status_code == 200, f"Failed to load local asset {asset_path}"
        assert len(asset_res.content) > 100
