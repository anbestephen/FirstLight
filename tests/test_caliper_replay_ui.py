"""
Static checks for the SAM 3.1 caliper measurement replay on the commander dashboard
(`ui/app.js`, `ui/index.html`, `ui/style.css`).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_caliper_replay_markup_and_reduced_motion_support() -> None:
    html = client.get("/").text
    assert 'id="caliper-scan-readout"' in html
    assert 'aria-live="polite"' in html

    js = client.get("/app.js").text
    assert "function animateCaliperSweep" in js
    assert "prefers-reduced-motion: reduce" in js
    # The sweep must not wait on the multi-second VLM call before drawing.
    assert "const vlmDone = runVlmAssessmentForEdge(inspectedEdgeId);" in js
    # Mask provenance is shown from the API's `mask_source`, never hard-coded.
    assert "maskSourceLabel(props.mask_source)" in js

    css = client.get("/style.css").text
    assert "path.pinch-pulse" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_caliper_payload_has_fields_the_replay_reads() -> None:
    roi = client.get("/api/clearance-roi?vehicle_class=fire_engine&limit=1").json()
    edge_id = roi["recommendations"][0]["edge_id"] if roi["recommendations"] else "0"
    caliper = client.get(f"/api/choke-point/{edge_id}/caliper").json()

    props = caliper["properties"]
    for key in ("min_clear_width_m", "pinch_point_station_m", "passability", "mask_source", "road_name"):
        assert key in props
    rays = [f for f in caliper["features"] if f["properties"]["feature_type"] == "caliper_ray"]
    assert rays
    for ray in rays:
        p = ray["properties"]
        assert {"station_m", "clear_width_m", "is_pinch_point", "full_ray_coords_wgs84"} <= p.keys()
    assert sum(1 for r in rays if r["properties"]["is_pinch_point"]) == 1
