"""
Tests for live commander-to-field dispatch sync (`src/dispatch.py`, `/api/dispatch*`) and
the phone-first field view (`/field`).
"""

from __future__ import annotations

import asyncio
import json
import re

from fastapi.testclient import TestClient
import pytest

from src.app import _merge_field_steps, app, state
from src.dispatch import DispatchHub

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_network_uplink_probe() -> None:
    """Keep tests offline: never probe the real Meta API host."""
    state.dispatch.uplink_probe = None
    yield
    client.post("/api/simulate-clear", json={"reset": True})


def _read_first_sse_event(path: str) -> dict:
    with client.stream("GET", path) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        event_name = None
        for line in res.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name == "dispatch":
                return json.loads(line[len("data: "):])
    raise AssertionError("stream ended without a dispatch event")


def test_dispatch_snapshot_contains_route_passability_and_connectivity() -> None:
    res = client.get("/api/dispatch")
    assert res.status_code == 200
    snap = res.json()

    assert snap["vehicle_class"] in ("atv", "ambulance", "fire_engine")
    assert len(snap["origin"]) == 2 and len(snap["destination"]) == 2
    route = snap["route"]
    for key in ("reachable", "total_distance_m", "estimated_transit_time_s", "coordinates", "steps"):
        assert key in route
    assert isinstance(snap["impassable_edge_ids"], list)
    assert isinstance(snap["water_edge_ids"], list)
    assert snap["connectivity"]["ai_tier"] in ("tier_1_cloud", "tier_2_edge")


def test_post_dispatch_bumps_version_only_on_real_changes() -> None:
    client.post("/api/dispatch", json={"vehicle_class": "fire_engine", "force_offline": False})
    v0 = client.get("/api/dispatch").json()["version"]

    snap = client.post("/api/dispatch", json={"vehicle_class": "ambulance"}).json()
    assert snap["version"] == v0 + 1
    assert snap["reason"] == "vehicle_changed"
    assert snap["vehicle_class"] == "ambulance"

    # Re-posting the identical assignment must not re-broadcast to phones.
    again = client.post("/api/dispatch", json={"vehicle_class": "ambulance"}).json()
    assert again["version"] == snap["version"]

    blackout = client.post("/api/dispatch", json={"force_offline": True}).json()
    assert blackout["reason"] == "comms_blackout_on"
    assert blackout["connectivity"]["ai_tier"] == "tier_2_edge"
    client.post("/api/dispatch", json={"force_offline": False, "vehicle_class": "fire_engine"})


def test_post_dispatch_rejects_bad_vehicle_and_coordinates() -> None:
    assert client.post("/api/dispatch", json={"vehicle_class": "hovercraft"}).status_code == 400
    assert client.post("/api/dispatch", json={"origin": [-85.41]}).status_code == 400


def test_simulated_clearance_is_broadcast_with_road_name() -> None:
    roi = client.get("/api/clearance-roi?vehicle_class=fire_engine&limit=1").json()
    if not roi["recommendations"]:
        pytest.skip("no debris choke points in the current mask set")
    top = roi["recommendations"][0]
    v0 = client.get("/api/dispatch").json()["version"]

    client.post(
        "/api/simulate-clear",
        json={"edge_u": top["edge_u"], "edge_v": top["edge_v"], "edge_key": top["edge_key"]},
    )
    snap = client.get("/api/dispatch").json()
    assert snap["version"] == v0 + 1
    assert snap["reason"] == "debris_cleared"
    assert snap["reason_detail"]
    assert {"u": top["edge_u"], "v": top["edge_v"], "key": top["edge_key"]} in snap["cleared_edges"]

    client.post("/api/simulate-clear", json={"reset": True})
    assert client.get("/api/dispatch").json()["reason"] == "clearances_reset"


def test_sse_stream_pushes_snapshot_and_tracks_field_devices() -> None:
    snap = _read_first_sse_event("/api/dispatch/stream?role=field&limit=1")
    assert "route" in snap
    assert snap["connectivity"]["field_clients"] == 1
    # The stream closed, so the device count must drop back.
    assert client.get("/api/dispatch").json()["connectivity"]["field_clients"] == 0


async def test_uplink_loss_is_detected_and_broadcast() -> None:
    probe_results = iter([True, False, True])
    hub = DispatchHub(uplink_probe=lambda: next(probe_results))

    await hub._run_uplink_probe()
    assert hub.uplink_online is True
    assert hub.version == 0  # first observation is a baseline, not a change

    await hub._run_uplink_probe()
    assert hub.uplink_online is False
    assert hub.last_change_reason == "uplink_lost"
    assert hub.ai_tier() == "tier_2_edge"

    await hub._run_uplink_probe()
    assert hub.last_change_reason == "uplink_restored"
    assert hub.ai_tier() == "tier_1_cloud"
    assert hub.version == 2


async def test_hub_stream_respects_limit_and_releases_client_count() -> None:
    hub = DispatchHub(uplink_probe=None)
    frames = []
    async for frame in hub.stream(lambda a: {"route": {}}, role="field", limit=1):
        frames.append(frame)
    assert frames[0].startswith("retry:")
    assert frames[1].startswith("event: dispatch")
    assert hub.field_clients == 0
    await asyncio.sleep(0)


def test_merge_field_steps_collapses_same_road_and_names_unnamed_segments() -> None:
    steps = [
        {"name": "15th Street", "length_m": 40.0, "travel_time_s": 4.0, "min_clear_width_m": 6.0},
        {"name": "15th Street", "length_m": 60.0, "travel_time_s": 6.0, "min_clear_width_m": 2.5},
        {"name": "Segment 1->2", "length_m": 20.0, "travel_time_s": 2.0, "min_clear_width_m": 8.0},
    ]
    merged = _merge_field_steps(steps)
    assert len(merged) == 2
    assert merged[0]["length_m"] == 100.0
    assert merged[0]["min_clear_width_m"] == 2.5
    assert merged[0]["instruction"] == "Continue 100 m on 15th Street"
    assert merged[1]["road"] == "unnamed road"


def test_field_view_assets_are_served_with_zero_cdn() -> None:
    page = client.get("/field")
    assert page.status_code == 200
    assert "/api/dispatch" not in page.text  # stream URL lives in field.js
    assert "field.js" in page.text and "/vendor/leaflet.js" in page.text
    assert not re.search(r'(src|href)="https?://', page.text)

    js = client.get("/field.js")
    assert js.status_code == 200
    assert "EventSource" in js.text and "/api/dispatch/stream?role=field" in js.text
    assert "https://" not in js.text

    assert client.get("/field.css").status_code == 200
