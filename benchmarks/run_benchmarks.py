"""
FirstLight held-out benchmark (`benchmarks/run_benchmarks.py`).

Compares model outputs against labels a human recorded blind in the `/label` tool:
1. SAM 3.1 clear width at each sampled 5 m cross-section vs the hand-measured width
   (`data/ground_truth/road_widths_manual.json`): MAE, bias, per-vehicle passability
   agreement (false-passable reported separately: it is the dangerous error), and how
   SAM's confidence relates to error (used to set the drone-recon threshold).
2. Building damage grades (`data/osm_buildings_graded.geojson`) vs hand labels from
   pre/post imagery (`data/ground_truth/building_damage_manual.json`).

This script only reads labels; it never creates or edits them.
Writes `benchmarks/benchmark_results.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
ROAD_LABELS_PATH = GT_DIR / "road_widths_manual.json"
BUILDING_LABELS_PATH = GT_DIR / "building_damage_manual.json"
BUILDING_SAMPLES_PATH = GT_DIR / "building_samples.json"
MEASUREMENTS_PATH = PROJECT_ROOT / "data" / "sam31_edge_measurements.json"
GRADED_BUILDINGS_PATH = PROJECT_ROOT / "data" / "osm_buildings_graded.geojson"
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "benchmark_results.json"

VEHICLE_THRESHOLDS_M = {"atv": 1.2, "ambulance": 2.2, "fire_engine": 3.5}
CONFIDENCE_BINS = ((0.0, 0.6), (0.6, 0.8), (0.8, 1.01))
HIGH_SEVERITY = {"major_damage", "destroyed"}
GRADES = ("no_damage", "minor_damage", "major_damage", "destroyed")


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def road_metrics(labels: dict[str, dict[str, Any]], measurements: dict[str, Any]) -> dict[str, Any]:
    """Station-level SAM 3.1 width vs hand-measured width (blocked counts as 0 m)."""
    pairs = []
    cant_tell = []
    for sample_id, lab in sorted(labels.items()):
        st = measurements[lab["edge_id"]]["stations"][lab["station_index"]]
        if lab["status"] == "cant_tell":
            cant_tell.append({"sample_id": sample_id, "sam_cause": st["cause"]})
            continue
        pairs.append(
            {
                "sample_id": sample_id,
                "human_width_m": float(lab["width_m"]),
                "sam_width_m": float(st["clear_width_m"]),
                "sam_cause": st["cause"],
                "sam_confidence": float(st["confidence"]),
            }
        )
    if not pairs:
        return {"labeled": 0, "cant_tell": len(cant_tell)}

    errors = [p["sam_width_m"] - p["human_width_m"] for p in pairs]
    abs_err = [abs(e) for e in errors]
    passability: dict[str, Any] = {}
    for vc, thr in VEHICLE_THRESHOLDS_M.items():
        agree = false_pass = false_block = 0
        for p in pairs:
            human_ok, sam_ok = p["human_width_m"] > thr, p["sam_width_m"] > thr
            agree += human_ok == sam_ok
            false_pass += sam_ok and not human_ok
            false_block += human_ok and not sam_ok
        passability[vc] = {
            "agreement": _rate(agree, len(pairs)),
            "false_passable": false_pass,
            "false_blocked": false_block,
        }

    calibration = []
    for lo, hi in CONFIDENCE_BINS:
        in_bin = [p for p, e in zip(pairs, abs_err) if lo <= p["sam_confidence"] < hi]
        bin_err = [abs(p["sam_width_m"] - p["human_width_m"]) for p in in_bin]
        calibration.append(
            {
                "confidence": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "n": len(in_bin),
                "mae_m": round(statistics.mean(bin_err), 3) if bin_err else None,
                "ambulance_agreement": _rate(
                    sum((p["human_width_m"] > 2.2) == (p["sam_width_m"] > 2.2) for p in in_bin), len(in_bin)
                ),
            }
        )

    not_visible = [p for p in pairs if p["sam_cause"] == "not_visible"]
    return {
        "labeled": len(pairs),
        "cant_tell": len(cant_tell),
        "mae_m": round(statistics.mean(abs_err), 3),
        "median_abs_error_m": round(statistics.median(abs_err), 3),
        "rmse_m": round(math.sqrt(statistics.mean(e * e for e in errors)), 3),
        "bias_m": round(statistics.mean(errors), 3),
        "passability": passability,
        "confidence_calibration": calibration,
        "sam_not_visible_stations": {
            "n": len(not_visible),
            "human_blocked": sum(p["human_width_m"] == 0.0 for p in not_visible),
            "human_cant_tell": sum(1 for c in cant_tell if c["sam_cause"] == "not_visible"),
        },
        "pairs": pairs,
    }


def building_metrics(
    labels: dict[str, dict[str, Any]], predictions: dict[str, str], grading_source: str
) -> dict[str, Any]:
    """Model damage grade vs hand label (cant_tell excluded)."""
    rows = [
        (lab["grade"], predictions.get(lab["building_id"]))
        for lab in labels.values()
        if lab["grade"] != "cant_tell" and predictions.get(lab["building_id"])
    ]
    if not rows:
        return {"labeled": 0, "grading_source": grading_source}
    tp = sum(h in HIGH_SEVERITY and p in HIGH_SEVERITY for h, p in rows)
    fp = sum(h not in HIGH_SEVERITY and p in HIGH_SEVERITY for h, p in rows)
    fn = sum(h in HIGH_SEVERITY and p not in HIGH_SEVERITY for h, p in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "labeled": len(rows),
        "grading_source": grading_source,
        "four_class_accuracy": _rate(sum(h == p for h, p in rows), len(rows)),
        "within_one_grade": _rate(sum(abs(GRADES.index(h) - GRADES.index(p)) <= 1 for h, p in rows), len(rows)),
        "high_severity_accuracy": _rate(sum((h in HIGH_SEVERITY) == (p in HIGH_SEVERITY) for h, p in rows), len(rows)),
        "high_severity_precision": round(precision, 4),
        "high_severity_recall": round(recall, 4),
        "high_severity_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "confusion": {h: {p: sum(1 for a, b in rows if a == h and b == p) for p in GRADES} for h in GRADES},
    }


def _labels(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["labels"] if path.exists() else {}


def run_benchmark() -> dict[str, Any]:
    road_labels = _labels(ROAD_LABELS_PATH)
    building_labels = _labels(BUILDING_LABELS_PATH)
    measurements = json.loads(MEASUREMENTS_PATH.read_text(encoding="utf-8"))["edges"]

    graded = json.loads(GRADED_BUILDINGS_PATH.read_text(encoding="utf-8")) if GRADED_BUILDINGS_PATH.exists() else {}
    predictions = {
        f["properties"]["building_id"]: f["properties"]["damage_severity"] for f in graded.get("features", [])
    }
    source = str(graded.get("properties", {}).get("grading_method", graded.get("properties", {}).get("grading_source", "unknown")))

    results = {
        "benchmark": "FirstLight held-out manual labels (Mexico Beach, Hurricane Michael)",
        "label_files": {
            "roads": str(ROAD_LABELS_PATH.relative_to(PROJECT_ROOT)),
            "buildings": str(BUILDING_LABELS_PATH.relative_to(PROJECT_ROOT)),
        },
        "road_widths_sam31": road_metrics(road_labels, measurements),
        "building_damage": building_metrics(building_labels, predictions, source),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = run_benchmark()
    roads, bldg = out["road_widths_sam31"], out["building_damage"]
    if not roads.get("labeled") and not bldg.get("labeled"):
        print("No manual labels yet: open http://localhost:8000/label, then re-run this script.")
    print(json.dumps({k: v for k, v in roads.items() if k != "pairs"}, indent=2))
    print(json.dumps(bldg, indent=2))
