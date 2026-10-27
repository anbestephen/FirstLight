"""
Unit tests for SAM 3.1 visible-pavement measurement (`src/sam_measure.py`) using synthetic
probability maps, so the geometry is verified without loading SAM.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from src.sam_measure import (
    describe_edge_measurement,
    measure_edge,
    measure_station,
    station_positions,
)

LINE = LineString([(0.0, 0.0), (100.0, 0.0)])


def make_sampler(road, debris=lambda x, y: False, water=lambda x, y: False):
    def sampler(pts: np.ndarray) -> dict[str, np.ndarray]:
        xs, ys = pts[:, 0], pts[:, 1]
        return {
            "road": np.array([1.0 if road(x, y) else 0.0 for x, y in zip(xs, ys)]),
            "debris": np.array([1.0 if debris(x, y) else 0.0 for x, y in zip(xs, ys)]),
            "water": np.array([1.0 if water(x, y) else 0.0 for x, y in zip(xs, ys)]),
        }

    return sampler


def offset_pavement(x: float, y: float) -> bool:
    # 7 m wide pavement whose center sits 3 m off the OSM centerline (OSM misalignment).
    return -0.5 <= y <= 6.5


def test_station_positions_inset_from_intersections() -> None:
    stations = station_positions(100.0)
    assert stations[0] == pytest.approx(2.5)
    assert stations[-1] == pytest.approx(97.5)
    assert all(b - a <= 5.0 + 1e-9 for a, b in zip(stations, stations[1:]))
    assert station_positions(4.0) == [2.0]


def test_clear_road_width_measured_despite_osm_offset() -> None:
    m = measure_edge(LINE, make_sampler(offset_pavement))
    assert m.min_clear_width_m == pytest.approx(7.0, abs=0.3)
    assert not m.has_water_obstruction
    assert all(st.cause == "clear" for st in m.stations)
    assert m.sam_confidence > 0.9


def test_debris_pile_sets_sustained_minimum_width() -> None:
    def pavement(x, y):
        if 38.0 <= x <= 58.0:  # debris leaves only a 1.5 m strip on the north side
            return 5.0 <= y <= 6.5
        return offset_pavement(x, y)

    def debris(x, y):
        return 38.0 <= x <= 58.0 and -0.5 <= y < 5.0

    m = measure_edge(LINE, make_sampler(pavement, debris))
    # One 0.25 m sample of quantisation, the same order as the 0.28 m/px imagery.
    assert m.min_clear_width_m == pytest.approx(1.5, abs=0.3)
    assert 38.0 <= m.pinch_station_m <= 58.0
    assert m.debris_area_m2 > 0
    pinch = next(s for s in m.stations if s.station_m == m.pinch_station_m)
    assert pinch.cause == "narrowed_debris"
    text = describe_edge_measurement(m, 0.2804)
    assert "debris" in text and "Ambulance" not in text and "ATV" in text


def test_single_station_obstruction_does_not_close_the_road() -> None:
    # A car-sized gap that only one 5 m station sees.
    def pavement(x, y):
        if 51.0 <= x <= 54.0:
            return False
        return offset_pavement(x, y)

    m = measure_edge(LINE, make_sampler(pavement))
    assert m.min_clear_width_m == pytest.approx(7.0, abs=0.3)


def test_standing_water_over_two_stations_makes_edge_impassable() -> None:
    def pavement(x, y):
        return offset_pavement(x, y) and not (30.0 <= x <= 45.0)

    def water(x, y):
        return 30.0 <= x <= 45.0 and -8.0 <= y <= 8.0

    m = measure_edge(LINE, make_sampler(pavement, water=water))
    assert m.has_water_obstruction
    assert m.min_clear_width_m == 0.0
    assert m.water_area_m2 > 0
    assert "standing water" in describe_edge_measurement(m, 0.2804)


def test_bridge_over_water_is_not_a_water_obstruction() -> None:
    def water(x, y):
        return 30.0 <= x <= 45.0 and abs(y) <= 15.0

    m = measure_edge(LINE, make_sampler(offset_pavement, water=water))
    assert not m.has_water_obstruction
    assert m.min_clear_width_m == pytest.approx(7.0, abs=0.3)


def test_unseen_pavement_is_flagged_for_drone_recon() -> None:
    def pavement(x, y):
        return offset_pavement(x, y) and not (20.0 <= x <= 35.0)  # tree canopy

    m = measure_edge(LINE, make_sampler(pavement))
    assert m.min_clear_width_m == 0.0
    assert m.not_visible_stations >= 2
    assert m.needs_drone_recon
    assert "verify by drone" in describe_edge_measurement(m, 0.2804)


def test_nearest_pavement_run_wins_over_parallel_parking_lot() -> None:
    def pavement(x, y):
        return -2.0 <= y <= 2.0 or 9.0 <= y <= 14.0  # road + parallel lot 9 m away

    m, _, _ = measure_station(LINE, 50.0, make_sampler(pavement))
    assert m.clear_width_m == pytest.approx(4.0, abs=0.3)
    assert m.center_offset_m == pytest.approx(0.0)


def test_runs_wider_than_the_road_type_cap_are_trimmed_and_flagged() -> None:
    def parking_lot(x, y):
        return -12.0 <= y <= 12.0  # 24 m of "pavement": road plus a lot

    m, _, _ = measure_station(LINE, 50.0, make_sampler(parking_lot), max_width_m=10.0)
    assert m.capped
    assert m.clear_width_m == pytest.approx(10.0)
    (x0, y0), (x1, y1) = m.clear_segment_utm
    assert abs(y1 - y0) == pytest.approx(10.0)

    uncapped, _, _ = measure_station(LINE, 50.0, make_sampler(offset_pavement), max_width_m=10.0)
    assert not uncapped.capped


def test_max_width_for_highway_types() -> None:
    from src.sam_measure import max_width_for_highway

    assert max_width_for_highway("residential") == 10.0
    assert max_width_for_highway("primary") == 12.0
    assert max_width_for_highway(["track", "residential"]) == 10.0
    assert max_width_for_highway(None) == 10.0


def _graded_sampler(road_prob):
    def sampler(pts: np.ndarray) -> dict[str, np.ndarray]:
        z = np.zeros(len(pts))
        return {"road": np.array([road_prob(y) for y in pts[:, 1]]), "debris": z, "water": z}

    return sampler


def test_width_is_measured_over_the_confident_core_not_the_blurred_edge() -> None:
    # 6 m of confident pavement with 1 m of blurred (p=0.6) edge on each side.
    def road(y):
        return 0.95 if abs(y) <= 3.0 else (0.6 if abs(y) <= 4.0 else 0.0)

    m, _, _ = measure_station(LINE, 50.0, _graded_sampler(road))
    assert m.clear_width_m == pytest.approx(6.0, abs=0.3)


def test_faint_pavement_without_a_confident_core_keeps_the_road_open() -> None:
    m, _, _ = measure_station(LINE, 50.0, _graded_sampler(lambda y: 0.6 if abs(y) <= 3.0 else 0.0))
    assert m.cause == "clear"
    assert m.clear_width_m == pytest.approx(6.0, abs=0.3)
