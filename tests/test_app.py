"""
End-to-end integration tests for Milestone 4 & Milestone 5: FastAPI backend (`src/app.py`),
5m orthogonal caliper ray & SAM 3.1 mask endpoints, 256x256 NOAA GeoTIFF aerial crop
extraction, Comms Blackout switch, and zero-CDN offline field dashboard (`ui/`).
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import re

from fastapi.testclient import TestClient
from PIL import Image
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
    assert "edge_id" in sample_road_props
    assert "min_clear_width_m" in sample_road_props
    assert "passable_atv" in sample_road_props
    assert "passable_ambulance" in sample_road_props
    assert "passable_fire_engine" in sample_road_props

    assert data["buildings"]["type"] == "FeatureCollection"
    assert len(data["buildings"]["features"]) == 334

    assert data["raster_envelope"]["type"] == "Feature"
    assert data["raster_envelope"]["geometry"]["type"] == "Polygon"
    assert len(data["bounds_wgs84"]) == 4


def test_choke_point_caliper_endpoint_returns_5m_rays_and_sam31_masks() -> None:
    """
    Verify GET /api/choke-point/{edge_id}/caliper returns an EPSG:4326 GeoJSON
    FeatureCollection with SAM 3.1 obstacle masks and 5.0m orthogonal caliper rays
    annotated with station_m, clear_width_m, is_pinch_point, and vehicle passability.
    """
    roi_res = client.get("/api/clearance-roi?vehicle_class=fire_engine")
    assert roi_res.status_code == 200
    recs = roi_res.json()["recommendations"]
    assert len(recs) > 0
    top_edge_id = recs[0]["edge_id"]

    cal_res = client.get(f"/api/choke-point/{top_edge_id}/caliper")
    assert cal_res.status_code == 200
    cal_data = cal_res.json()

    assert cal_data["type"] == "FeatureCollection"
    props = cal_data["properties"]
    assert props["crs"] == "EPSG:4326"
    assert props["metric_crs"] == "EPSG:32616"
    assert math.isclose(props["gsd_m_per_px"], 0.2804, rel_tol=1e-2)
    assert props["ray_count"] > 0

    features = cal_data["features"]
    debris_features = [
        f for f in features if f["properties"].get("feature_type") == "debris_mask"
    ]
    ray_features = [
        f for f in features if f["properties"].get("feature_type") == "caliper_ray"
    ]

    assert len(debris_features) > 0
    assert len(ray_features) == props["ray_count"]

    pinch_rays = [r for r in ray_features if r["properties"]["is_pinch_point"] is True]
    assert len(pinch_rays) == 1

    for r in ray_features:
        rp = r["properties"]
        assert "station_m" in rp
        assert "clear_width_m" in rp
        assert "is_pinch_point" in rp
        assert "passability" in rp
        assert set(rp["passability"].keys()) == {"atv", "ambulance", "fire_engine"}
        assert isinstance(rp["atv"], bool)
        assert isinstance(rp["ambulance"], bool)
        assert isinstance(rp["fire_engine"], bool)


def test_choke_point_aerial_crop_png_endpoint() -> None:
    """
    Verify GET /api/choke-point/{edge_id}/crop.png extracts a valid 256x256 RGB PNG
    patch directly from data/noaa_michael_mexico_beach.tif.
    """
    roi_res = client.get("/api/clearance-roi?vehicle_class=fire_engine")
    top_edge_id = roi_res.json()["recommendations"][0]["edge_id"]

    crop_res = client.get(f"/api/choke-point/{top_edge_id}/crop.png")
    assert crop_res.status_code == 200
    assert crop_res.headers["content-type"] == "image/png"

    # Confirm valid PNG magic header bytes (\x89PNG\r\n\x1a\n)
    assert crop_res.content[:8] == b"\x89PNG\r\n\x1a\n"

    img = Image.open(io.BytesIO(crop_res.content))
    assert img.format == "PNG"
    assert img.size == (256, 256)
    assert img.mode == "RGB"


def test_assess_endpoint_with_force_offline_comms_blackout_switch() -> None:
    """
    Verify POST /api/assess with force_offline=False vs force_offline=True
    ("Simulate Comms Blackout").
    """
    roi_res = client.get("/api/clearance-roi?vehicle_class=fire_engine")
    top_edge_id = roi_res.json()["recommendations"][0]["edge_id"]

    # 1. force_offline = False: attempts Tier 1 Cloud first, then fails over to local edge
    online_res = client.post(
        "/api/assess",
        json={
            "edge_id": top_edge_id,
            "force_offline": False,
        },
    )
    assert online_res.status_code == 200
    online_data = online_res.json()
    assert online_data["force_offline"] is False
    assert online_data["fallback_chain"][0].startswith("tier_1_cloud:")
    assert online_data["memory_cleared"] is True
    assert online_data["latency_ms"] >= 0.0
    assert online_data["crop_url"] == f"/api/choke-point/{top_edge_id}/crop.png"

    # 2. force_offline = True ("Simulate Comms Blackout"): intentionally bypasses Tier 1
    blackout_res = client.post(
        "/api/assess",
        json={
            "edge_id": top_edge_id,
            "force_offline": True,
        },
    )
    assert blackout_res.status_code == 200
    blackout_data = blackout_res.json()
    assert blackout_data["force_offline"] is True
    assert blackout_data["fallback_chain"][0] == "tier_1_cloud:bypassed:force_offline"
    assert blackout_data["tier_used"] in (
        "tier_2_edge_mlx_vlm",
        "tier_3_deterministic_geometry",
    )
    assert blackout_data["memory_cleared"] is True
    assert blackout_data["latency_ms"] >= 0.0
    assert "0.2804 m/px GSD" in blackout_data["passability_rationale"]


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


def test_zero_cdn_static_asset_integrity_and_milestone5_controls() -> None:
    """
    Verify ui/index.html, ui/style.css, and ui/app.js contain zero external CDN
    dependencies, expose all Milestone 5 controls, and all referenced local assets resolve.
    """
    index_res = client.get("/")
    assert index_res.status_code == 200
    html = index_res.text

    # Confirm Milestone 5 UI elements exist
    assert "toggle-comms-blackout" in html
    assert "toggle-caliper-rays" in html
    assert "btn-export-manifest" in html
    assert "corridor-inspector-dock" in html

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


def test_milestone6_meta_models_briefing_voice_and_schematic_endpoints() -> None:
    """
    Verify Milestone 6 endpoints (`GET /api/meta-models`, `POST /api/incident-briefing`,
    `POST /api/voice-dispatch`, and `GET /api/choke-point/{edge_id}/recon-schematic.png`)
    in both normal (`force_offline=False`) and Comms Blackout (`force_offline=True`) modes.
    """
    # 1. GET /api/meta-models
    for offline_flag in (False, True):
        m_res = client.get(f"/api/meta-models?force_offline={str(offline_flag).lower()}")
        assert m_res.status_code == 200
        m_data = m_res.json()
        assert m_data["force_offline"] is offline_flag
        assert m_data["total_models"] == 6
        returned_ids = [m["model_id"] for m in m_data["models"]]
        assert returned_ids == [
            "sam-3.1",
            "muse-spark",
            "muse-wonder",
            "muse-voice-transcribe",
            "muse-image",
            "open-weight-mlx-4bit",
        ]

    # 2. POST /api/incident-briefing (muse-wonder)
    for offline_flag in (False, True):
        b_res = client.post(
            "/api/incident-briefing",
            json={
                "vehicle_class": "fire_engine",
                "force_offline": offline_flag,
                "max_corridors": 4,
            },
        )
        assert b_res.status_code == 200
        b_data = b_res.json()
        assert b_data["model_id"] == "muse-wonder"
        assert b_data["force_offline"] is offline_flag
        assert len(b_data["sector_priorities"]) > 0
        assert len(b_data["executive_summary"]) > 20
        assert b_data["memory_cleared"] is True

    # 3. POST /api/voice-dispatch (muse-voice-transcribe)
    for offline_flag in (False, True):
        v_res = client.post(
            "/api/voice-dispatch",
            json={
                "voice_command_text": "Dispatch Fire Engine and simulate clearing top choke point on 15th Street",
                "force_offline": offline_flag,
                "auto_apply_clearance": False,
            },
        )
        assert v_res.status_code == 200
        v_data = v_res.json()
        assert v_data["model_id"] == "muse-voice-transcribe"
        assert v_data["force_offline"] is offline_flag
        assert v_data["vehicle_class"] == "fire_engine"
        assert v_data["simulate_clear_edge_id"] is not None
        assert v_data["trigger_route"] is True

    # 4. GET /api/choke-point/{edge_id}/recon-schematic.png (muse-image)
    roi_res = client.get("/api/clearance-roi?vehicle_class=fire_engine")
    top_rec = roi_res.json()["recommendations"][0]
    edge_id = f"{top_rec['edge_u']}-{top_rec['edge_v']}-{top_rec['edge_key']}"

    for offline_flag in (False, True):
        s_res = client.get(
            f"/api/choke-point/{edge_id}/recon-schematic.png?force_offline={str(offline_flag).lower()}"
        )
        assert s_res.status_code == 200
        assert s_res.headers["content-type"] == "image/png"
        assert s_res.content[:8] == b"\x89PNG\r\n\x1a\n"
        img = Image.open(io.BytesIO(s_res.content))
        assert img.size == (256, 256)

