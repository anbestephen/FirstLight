"""
Tests for the held-out benchmark math (`benchmarks/run_benchmarks.py`) on synthetic labels.
"""

from __future__ import annotations

import pytest

from benchmarks.run_benchmarks import building_metrics, road_metrics


def _measurements(*stations: tuple[float, str, float]) -> dict:
    return {
        "e-1-0": {
            "stations": [
                {"clear_width_m": w, "cause": cause, "confidence": conf} for w, cause, conf in stations
            ]
        }
    }


def test_road_metrics_errors_and_dangerous_false_passable() -> None:
    meas = _measurements((6.0, "clear", 0.9), (3.0, "narrowed_debris", 0.7), (0.0, "not_visible", 0.3))
    labels = {
        "road-01": {"edge_id": "e-1-0", "station_index": 0, "status": "measured", "width_m": 5.5},
        # SAM says 3.0 m but a human sees only 2.0 m -> SAM wrongly passes an ambulance.
        "road-02": {"edge_id": "e-1-0", "station_index": 1, "status": "measured", "width_m": 2.0},
        "road-03": {"edge_id": "e-1-0", "station_index": 2, "status": "blocked", "width_m": 0.0},
    }
    m = road_metrics(labels, meas)
    assert m["labeled"] == 3
    assert m["mae_m"] == pytest.approx((0.5 + 1.0 + 0.0) / 3, abs=1e-3)
    assert m["bias_m"] == pytest.approx((0.5 + 1.0 + 0.0) / 3, abs=1e-3)
    assert m["passability"]["ambulance"]["false_passable"] == 1
    assert m["passability"]["ambulance"]["false_blocked"] == 0
    assert m["passability"]["fire_engine"]["agreement"] == 1.0
    assert m["sam_not_visible_stations"] == {"n": 1, "human_blocked": 1, "human_cant_tell": 0}
    low_bin = m["confidence_calibration"][0]
    assert low_bin["n"] == 1 and low_bin["mae_m"] == 0.0


def test_road_metrics_excludes_cant_tell_and_handles_no_labels() -> None:
    meas = _measurements((4.0, "not_visible", 0.2))
    labels = {"road-01": {"edge_id": "e-1-0", "station_index": 0, "status": "cant_tell"}}
    m = road_metrics(labels, meas)
    assert m == {"labeled": 0, "cant_tell": 1}
    assert road_metrics({}, meas) == {"labeled": 0, "cant_tell": 0}


def test_building_metrics_high_severity_scores() -> None:
    labels = {
        "b1": {"building_id": "bldg-1", "grade": "destroyed"},
        "b2": {"building_id": "bldg-2", "grade": "major_damage"},
        "b3": {"building_id": "bldg-3", "grade": "no_damage"},
        "b4": {"building_id": "bldg-4", "grade": "cant_tell"},
    }
    preds = {"bldg-1": "destroyed", "bldg-2": "minor_damage", "bldg-3": "no_damage", "bldg-4": "destroyed"}
    m = building_metrics(labels, preds, "muse-spark-1.3 per-building pre/post")
    assert m["labeled"] == 3
    assert m["four_class_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["within_one_grade"] == 1.0
    assert m["high_severity_precision"] == 1.0
    assert m["high_severity_recall"] == 0.5
    assert m["confusion"]["major_damage"]["minor_damage"] == 1
