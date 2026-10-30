"""
Independent validation against CRASAR-U-DROIDs expert labels (Mexico Beach, Hurricane Michael).

CRASAR (Center for Robot-Assisted Search and Rescue) flew drones over Mexico Beach on Oct 13
and Oct 14, 2018 (~2 cm/px orthomosaics) and its annotators labelled building damage on the
xBD scale and road conditions along OpenStreetMap road lines. None of that was seen by
FirstLight: our grades and widths come from the Oct 11 NOAA image (0.28 m/px).

1. Buildings: each of our 334 OSM buildings is matched to the CRASAR footprint it overlaps most
   (IoU >= 0.3). Muse Spark's grade is scored against CRASAR's drone label. For context we also
   score our own hand labels, and Muse Spark against CRASAR's labels drawn on the NOAA image.
2. Roads: for each measured road that CRASAR's labelled road lines cover, the worst CRASAR
   condition within 3 m of the centerline, ignoring the 8 m at each end (the intersection,
   where labels belong to the crossing street), is compared with the SAM 3.1 clear width.

Caveats: the drone flights are 2-3 days after the NOAA image, so some roads may have been
partly cleared, and CRASAR's road classes ("Partial Obstruction") are not widths.

Data: CRASAR-U-DROIDs, CC-BY-4.0, https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs
Usage: ./.venv/bin/python benchmarks/crasar_validation.py   (downloads ~4 MB of labels once)
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import urllib.request

import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import substring

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_benchmarks import building_metrics  # noqa: E402
from src.geometry import AMBULANCE_MIN_WIDTH_M, ATV_MIN_WIDTH_M, FIRE_ENGINE_MIN_WIDTH_M  # noqa: E402

CRASAR_DIR = PROJECT_ROOT / "data" / "external" / "crasar"
BASE = "https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs/resolve/main/test/annotations"
SCENES = ("10132018-MexicoBeach", "10142018-MexicoBeach")
FILES = {
    "uas_buildings": "UAS/building_damage_assessment/{scene}.geo.tif.json",
    "uas_roads": "UAS/road_damage_assessment/{scene}.geo.tif.json",
    "noaa_buildings": "CREWED/building_damage_assessment/{scene}.geo.tif_20181011a_RGB.geo.tif.json",
}
UTM = "EPSG:32616"
GRADE = {"no damage": "no_damage", "minor damage": "minor_damage", "major damage": "major_damage", "destroyed": "destroyed"}
MATCH_IOU = 0.3
ROAD_BUFFER_M = 3.0
ROAD_COVERAGE_MIN = 0.8
# Labels within this distance of a road's ends belong to the crossing street at the intersection.
INTERSECTION_TRIM_M = 8.0
# Worst road condition along a road, mapped to what it means for vehicles.
BLOCKING = {"Total Obstruction", "Total Destruction", "Total Flooding"}
NARROWING = {"Partial Obstruction", "Partial Road Condition", "Partial Flooding"}
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "crasar_validation.json"


def fetch() -> dict[str, list[Path]]:
    CRASAR_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[Path]] = {k: [] for k in FILES}
    for key, pattern in FILES.items():
        for scene in SCENES:
            dest = CRASAR_DIR / f"{key}_{scene}.json"
            if not dest.exists():
                urllib.request.urlretrieve(f"{BASE}/{pattern.format(scene=scene)}", dest)
            paths[key].append(dest)
    return paths


def _polygon(points: list[dict[str, float]]) -> Polygon:
    return Polygon([(p["lon"], p["lat"]) for p in points])


def crasar_buildings(paths: list[Path]) -> gpd.GeoDataFrame:
    rows = [
        {"label": GRADE.get(b["label"]), "crasar_id": b["building_id"], "geometry": _polygon(b["EPSG:4326"])}
        for path in paths
        for b in json.loads(path.read_text())
    ]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(UTM)
    return gdf[gdf["label"].notna()]  # drop "un-classified"


def match_buildings(ours: gpd.GeoDataFrame, theirs: gpd.GeoDataFrame) -> dict[str, str]:
    """OSM building_id -> CRASAR label, using the CRASAR footprint with the highest IoU."""
    matches: dict[str, str] = {}
    index = theirs.sindex
    for bid, geom in zip(ours["building_id"], ours.geometry):
        best_iou, best_label = 0.0, None
        for i in index.query(geom, predicate="intersects"):
            other = theirs.geometry.iloc[i]
            iou = geom.intersection(other).area / geom.union(other).area
            if iou > best_iou:
                best_iou, best_label = iou, theirs["label"].iloc[i]
        if best_iou >= MATCH_IOU:
            matches[bid] = best_label
    return matches


def as_labels(matches: dict[str, str]) -> dict[str, dict[str, str]]:
    return {f"crasar-{bid}": {"building_id": bid, "grade": grade} for bid, grade in matches.items()}


def road_validation(paths: list[Path]) -> dict:
    measurements = json.loads((PROJECT_ROOT / "data" / "sam31_edge_measurements.json").read_text())["edges"]
    roads = gpd.read_file(PROJECT_ROOT / "data" / "osm_roads.geojson").to_crs(UTM)
    roads["edge_id"] = roads["u"].astype(str) + "-" + roads["v"].astype(str) + "-" + roads["key"].astype(str)

    lines, damage = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        lines += [LineString([(p["lon"], p["lat"]) for p in r["EPSG:4326"]]) for r in payload["road_lines"]]
        damage += [{"label": p["label"], "geometry": _polygon(p["EPSG:4326"])} for p in payload["polygons"]]
    covered_area = gpd.GeoSeries(lines, crs="EPSG:4326").to_crs(UTM).buffer(5.0).union_all()
    damage_gdf = gpd.GeoDataFrame(damage, crs="EPSG:4326").to_crs(UTM)

    seen: set[tuple] = set()
    rows = []
    for edge_id, geom in zip(roads["edge_id"], roads.geometry):
        u, v, k = edge_id.split("-")
        undirected = (min(u, v), max(u, v), k)
        m = measurements.get(edge_id)
        if m is None or undirected in seen or geom.length < 3 * INTERSECTION_TRIM_M:
            continue  # unmeasured, the reverse direction, or an intersection stub
        seen.add(undirected)
        if geom.intersection(covered_area).length / geom.length < ROAD_COVERAGE_MIN:
            continue  # CRASAR did not label this road
        middle = substring(geom, INTERSECTION_TRIM_M, geom.length - INTERSECTION_TRIM_M)
        near = damage_gdf[damage_gdf.intersects(middle.buffer(ROAD_BUFFER_M))]["label"]
        labels = set(near)
        crasar = "blocked" if labels & BLOCKING else "narrowed" if labels & NARROWING else "clear"
        width = float(m["min_clear_width_m"])
        ours = (
            "blocked" if m.get("has_water_obstruction") or width <= ATV_MIN_WIDTH_M
            else "narrowed" if width <= FIRE_ENGINE_MIN_WIDTH_M
            else "open"
        )
        rows.append({"edge_id": edge_id, "crasar": crasar, "sam": ours, "width_m": width,
                     "crasar_labels": sorted(labels), "drone_recon": bool(m.get("needs_drone_recon"))})

    table = Counter((r["sam"], r["crasar"]) for r in rows)
    blocked = [r for r in rows if r["crasar"] == "blocked"]
    we_open = [r for r in rows if r["sam"] == "open"]
    amb_blocked_by_us = [r for r in blocked if r["width_m"] <= AMBULANCE_MIN_WIDTH_M]
    over_cautious = [r for r in rows if r["sam"] == "blocked" and r["crasar"] == "clear"]
    return {
        "roads_compared": len(rows),
        "crosstab_sam_vs_crasar": {f"{s}|{c}": n for (s, c), n in sorted(table.items())},
        "crasar_total_blockages": len(blocked),
        "of_which_sam_closed_to_ambulance": len(amb_blocked_by_us),
        "sam_open_to_fire_engine": len(we_open),
        "of_which_crasar_total_blockage": sum(r["crasar"] == "blocked" for r in we_open),
        "sam_blocked_crasar_clear": len(over_cautious),
        "of_which_flagged_for_drone_recon": sum(r["drone_recon"] for r in over_cautious),
        "rows": rows,
    }


def main() -> None:
    paths = fetch()
    graded = gpd.read_file(PROJECT_ROOT / "data" / "osm_buildings_graded.geojson").to_crs(UTM)
    spark = dict(zip(graded["building_id"], graded["damage_severity"]))
    drone = match_buildings(graded, crasar_buildings(paths["uas_buildings"]))
    noaa = match_buildings(graded, crasar_buildings(paths["noaa_buildings"]))

    hand = {}
    for name in ("building_damage_manual.json", "building_damage_manual_test.json"):
        p = PROJECT_ROOT / "data" / "ground_truth" / name
        if p.exists():
            hand.update({lab["building_id"]: lab["grade"] for lab in json.loads(p.read_text())["labels"].values()
                         if lab["grade"] != "cant_tell"})

    results = {
        "source": "CRASAR-U-DROIDs (CC-BY-4.0), Mexico Beach sUAS scenes 10132018 and 10142018",
        "buildings": {
            "osm_buildings": len(graded),
            "matched_to_crasar_drone_labels": len(drone),
            "muse_spark_vs_crasar_drone": building_metrics(as_labels(drone), spark, "muse-spark-1.3 (NOAA+NAIP)"),
            "muse_spark_vs_crasar_noaa_image": building_metrics(as_labels(noaa), spark, "muse-spark-1.3 (NOAA+NAIP)"),
            "hand_labels_vs_crasar_drone": building_metrics(
                as_labels({b: g for b, g in drone.items() if b in hand}), hand, "FirstLight hand labels (NOAA)"
            ),
        },
        "roads": road_validation(paths["uas_roads"]),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in results["roads"].items() if k != "rows"}
    b = results["buildings"]
    print(json.dumps({"matched": b["matched_to_crasar_drone_labels"],
                      "spark_vs_drone": {k: b["muse_spark_vs_crasar_drone"].get(k) for k in
                                         ("labeled", "four_class_accuracy", "within_one_grade",
                                          "high_severity_precision", "high_severity_recall")},
                      "spark_vs_noaa_labels": {k: b["muse_spark_vs_crasar_noaa_image"].get(k) for k in
                                               ("labeled", "four_class_accuracy", "within_one_grade",
                                                "high_severity_precision", "high_severity_recall")},
                      "hand_vs_drone": {k: b["hand_labels_vs_crasar_drone"].get(k) for k in
                                        ("labeled", "four_class_accuracy", "within_one_grade")},
                      "roads": summary}, indent=1))


if __name__ == "__main__":
    main()
