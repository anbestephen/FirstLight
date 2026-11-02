"""Drone re-measurement (`src/drone.py`) and the dashboard's drone check (`/api/.../drone-check`)."""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString

from src.drone import SCENES, DronePatch, coverage, remeasure, segment_patch


class FakeSam:
    """Marks a 6 m-wide pavement band along the patch's middle row as road."""

    def __init__(self, band_rows: slice) -> None:
        self.band_rows = band_rows

    def segment_tile_probabilities(self, image, prompts):
        w, h = image.size
        road = np.zeros((h, w), dtype=np.float32)
        road[self.band_rows] = 0.95
        return {"road": road, "debris": np.zeros_like(road), "water": np.zeros_like(road)}


def _patch(valid_cols: int | None = None) -> tuple[DronePatch, LineString]:
    gsd = 0.16
    h, w = 400, 600  # 64 m x 96 m
    transform = from_origin(652000.0, 3315064.0, gsd, gsd)
    valid = np.ones((h, w), dtype=bool)
    if valid_cols is not None:
        valid[:, valid_cols:] = False
    patch = DronePatch(SCENES[0], np.full((h, w, 3), 120, dtype=np.uint8), valid, transform, gsd)
    y = 3315064.0 - 200 * gsd  # centerline along the middle row
    return patch, LineString([(652010.0, y), (652086.0, y)])


def test_remeasure_reads_the_width_sam_finds_on_the_drone_patch() -> None:
    patch, line = _patch()
    assert coverage(patch, line) == 1.0
    probs = segment_patch(patch, FakeSam(slice(181, 219)))  # 38 rows x 0.16 m ~ 6 m
    record = remeasure(line, "residential", patch, probs)
    assert 5.5 <= record["min_clear_width_m"] <= 6.3
    assert record["measurement_source"] == "drone"
    assert record["imagery"]["flown"] == "2018-10-13" and record["imagery"]["gsd_m_per_px"] == 0.16
    assert "drone imagery (2018-10-13)" in record["reason"]


def test_partial_drone_coverage_is_detected_and_empty_pixels_never_count_as_road() -> None:
    patch, line = _patch(valid_cols=300)  # drone imaged only the west half
    assert 0.4 < coverage(patch, line) < 0.6
    probs = segment_patch(patch, FakeSam(slice(181, 219)))
    assert probs["road"][:, 300:].max() == 0.0


def test_drone_check_updates_the_live_graph_and_can_be_undone() -> None:
    from src.app import app, state

    client = TestClient(app)
    state.ensure_loaded()
    eid = "95800288-95800290-0"
    u, v, k = eid.split("-")
    noaa = copy.deepcopy(state.sam_measurements["edges"][eid])
    drone = copy.deepcopy(noaa)
    for st in drone["stations"]:
        st.update(clear_width_m=4.25, cause="clear")
    drone.update(
        min_clear_width_m=4.25, needs_drone_recon=False, measurement_source="drone",
        imagery={"scene": "x", "flown": "2018-10-13", "gsd_m_per_px": 0.157, "source": "test"},
        noaa_min_clear_width_m=noaa["min_clear_width_m"], reason="SAM 3.1 on drone imagery (2018-10-13): 4.25 m",
    )
    saved = state.drone_measurements
    state.drone_measurements = {"edges": {eid: drone, f"{v}-{u}-{k}": drone}, "skipped": {}}
    try:
        before = client.get(f"/api/choke-point/{eid}/caliper").json()["properties"]
        assert before["drone_available"] is True and before["min_clear_width_m"] == 0.0

        result = client.post(f"/api/choke-point/{eid}/drone-check").json()
        assert result["noaa_min_clear_width_m"] == 0.0 and result["drone_min_clear_width_m"] == 4.25
        assert result["passability"] == {"atv": True, "ambulance": True, "fire_engine": True}
        edge = state.active_graph[int(u)][int(v)][int(k)]
        assert edge["min_clear_width_m"] == 4.25 and edge["passable_fire_engine"] is True
        after = client.get(f"/api/choke-point/{eid}/caliper").json()["properties"]
        assert after["measurement_source"] == "drone" and after["drone_available"] is False
        road = next(f for f in client.get("/api/layers").json()["roads"]["features"] if f["properties"]["edge_id"] == eid)
        assert road["properties"]["measurement_source"] == "drone"
        assert state.dispatch.last_change_reason == "road_remeasured"

        assert client.post("/api/drone/reset").json() == {"reverted_roads": 1}
        assert state.active_graph[int(u)][int(v)][int(k)]["min_clear_width_m"] == 0.0
        assert state.sam_measurements["edges"][eid]["min_clear_width_m"] == noaa["min_clear_width_m"]
    finally:
        state.revert_drone_measurements()
        state.drone_measurements = saved


def test_drone_check_explains_roads_without_drone_imagery() -> None:
    from src.app import app, state

    state.ensure_loaded()
    saved = state.drone_measurements
    state.drone_measurements = {"edges": {}, "skipped": {"95800288-95800290-0": "outside drone coverage"}}
    try:
        res = TestClient(app).post("/api/choke-point/95800288-95800290-0/drone-check")
        assert res.status_code == 404 and "outside drone coverage" in res.json()["detail"]
    finally:
        state.drone_measurements = saved


def test_drone_rule_closes_a_road_on_one_narrow_cross_section() -> None:
    from src.sam_measure import StationMeasurement, summarise_stations

    def station(i: int, width: float) -> StationMeasurement:
        return StationMeasurement(
            station_m=5.0 * i, center_utm=(5.0 * i, 0.0), ray_utm=((5.0 * i, -15.0), (5.0 * i, 15.0)),
            clear_segment_utm=None, clear_width_m=width, center_offset_m=0.0,
            cause="debris" if width < 1 else "clear", confidence=0.9,
        )

    stations = [station(i, w) for i, w in enumerate([5.0, 5.0, 0.4, 5.0, 5.0])]
    assert summarise_stations(stations, 20.0, 0.0, 0.0).min_clear_width_m == 5.0  # NOAA: needs two stations
    strict = summarise_stations(stations, 20.0, 0.0, 0.0, sustained=False)
    assert strict.min_clear_width_m == 0.4 and strict.pinch_station_m == 10.0


def test_drone_checked_road_still_renders_without_the_downloaded_patch(tmp_path, monkeypatch) -> None:
    import src.app as app_module
    from src.app import app, state

    client = TestClient(app)
    state.ensure_loaded()
    eid = next(iter(state.drone_measurements["edges"]))
    monkeypatch.setattr(app_module, "DRONE_CACHE_DIR", tmp_path)  # a fresh clone: no data/drone_cache
    try:
        assert client.post(f"/api/choke-point/{eid}/drone-check").status_code == 200
        props = client.get(f"/api/choke-point/{eid}/caliper").json()["properties"]
        assert props["measurement_source"] == "drone" and props["drone_image_cached"] is False
        schematic = client.get(f"/api/choke-point/{eid}/recon-schematic.png")
        assert schematic.status_code == 200 and schematic.headers["content-type"] == "image/png"
        assert client.get(f"/api/choke-point/{eid}/crop.png").status_code == 200
    finally:
        state.revert_drone_measurements()
