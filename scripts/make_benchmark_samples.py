"""
Pick the held-out benchmark samples for manual labelling (`scripts/make_benchmark_samples.py`).

Writes geometry only - never a model output - so labels collected in the `/label` tool are
independent of SAM 3.1 and Muse Spark:

- `data/ground_truth/road_width_samples.json`: 40 cross-sections (road, station, ray), drawn
  with a fixed seed and stratified over the SAM station causes so blocked/occluded cases are
  represented (stratification uses the cause only to choose *where* to look).
- `data/ground_truth/building_samples.json`: 40 building footprints drawn uniformly at random.

Re-running with the same seed reproduces the same samples; existing labels are not touched.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GT_DIR = DATA_DIR / "ground_truth"
SEED = 20261027
ROAD_SAMPLES_PER_CAUSE = {"clear": 20, "narrowed_debris": 7, "debris": 7, "not_visible": 6}
BUILDING_SAMPLES = 40


def main() -> None:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    measurements = json.loads((DATA_DIR / "sam31_edge_measurements.json").read_text())["edges"]
    by_cause: dict[str, list[dict]] = {cause: [] for cause in ROAD_SAMPLES_PER_CAUSE}
    seen_roads: set[tuple] = set()
    for edge_id in sorted(measurements):
        u, v, k = edge_id.split("-")
        canon = (min(u, v), max(u, v), k)
        if canon in seen_roads:
            continue  # both directions share one measurement
        seen_roads.add(canon)
        for idx, st in enumerate(measurements[edge_id]["stations"]):
            if st["cause"] in by_cause:
                by_cause[st["cause"]].append(
                    {
                        "edge_id": edge_id,
                        "station_index": idx,
                        "station_m": st["station_m"],
                        "center_utm": st["center_utm"],
                        "ray_utm": st["ray_utm"],
                    }
                )

    road_samples = []
    for cause, n in ROAD_SAMPLES_PER_CAUSE.items():
        road_samples.extend(rng.sample(by_cause[cause], min(n, len(by_cause[cause]))))
    rng.shuffle(road_samples)
    for i, s in enumerate(road_samples):
        s["sample_id"] = f"road-{i + 1:02d}"

    buildings = gpd.read_file(DATA_DIR / "osm_buildings.geojson").to_crs("EPSG:32616")
    picks = sorted(rng.sample(range(len(buildings)), BUILDING_SAMPLES))
    building_samples = []
    for i, idx in enumerate(picks):
        geom = buildings.geometry.iloc[idx]
        building_samples.append(
            {
                "sample_id": f"bldg-{i + 1:02d}",
                "building_id": f"bldg-{idx}",
                "centroid_utm": [round(geom.centroid.x, 2), round(geom.centroid.y, 2)],
                "footprint_utm": [[round(x, 2), round(y, 2)] for x, y in geom.exterior.coords],
            }
        )

    meta = {"seed": SEED, "crs": "EPSG:32616", "note": "geometry only; no model outputs"}
    (GT_DIR / "road_width_samples.json").write_text(json.dumps({**meta, "samples": road_samples}, indent=1))
    (GT_DIR / "building_samples.json").write_text(json.dumps({**meta, "samples": building_samples}, indent=1))
    print(f"{len(road_samples)} road cross-sections, {len(building_samples)} buildings -> {GT_DIR}")


if __name__ == "__main__":
    main()
