"""
Re-measure every drone-flagged road on CRASAR's drone imagery (`scripts/drone_remeasure.py`).

For each road the NOAA measurement flagged "drone check advised", pick the drone scene that
images the whole road, read that patch at ~0.16 m/px, run SAM 3.1 on it and measure it again.
Writes `data/sam31_drone_measurements.json` (both edge directions, same record format as the NOAA
bake plus `measurement_source` and `imagery`) and caches each patch in `data/drone_cache/` for the
inspector's aerial view. The dashboard applies these on demand ("Drone check").

Usage: ./.venv/bin/python scripts/drone_remeasure.py [--limit N] [--edge U-V-K ...] [--rederive]
Needs the internet the first time (patches are read from Hugging Face over HTTP).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import osmnx as ox
import rasterio
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from bake_sam31_masks import edge_lines_utm  # noqa: E402
from src.drone import GDAL_ENV, MIN_COVERAGE, SCENES, coverage, read_patch, rederive, remeasure, segment_patch  # noqa: E402
from src.geometry import TARGET_METRIC_CRS, SAM31GeometricInstrument  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT_PATH = DATA / "sam31_drone_measurements.json"
CACHE_DIR = DATA / "drone_cache"


def scene_footprints() -> dict[str, Any]:
    footprints = {}
    with rasterio.Env(**GDAL_ENV):
        for scene in SCENES:
            with rasterio.open(scene.url) as src:
                footprints[scene.name] = box(*src.bounds)
    return footprints


def save_patch(path: Path, patch) -> None:
    rgb = patch.rgb.transpose(2, 0, 1)
    alpha = (patch.valid * 255).astype("uint8")[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=rgb.shape[1], width=rgb.shape[2], count=4, dtype="uint8",
        crs=TARGET_METRIC_CRS, transform=patch.transform, compress="deflate",
    ) as dst:
        dst.write(rgb, [1, 2, 3])
        dst.write(alpha, [4])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--edge", action="append", default=[], help="only these edge ids (u-v-k)")
    parser.add_argument("--rederive", action="store_true", help="recompute verdicts from stored cross-sections only")
    args = parser.parse_args()
    if args.rederive:
        out = json.loads(OUT_PATH.read_text())
        out["edges"] = {eid: rederive(rec) for eid, rec in out["edges"].items()}
        OUT_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"re-derived {len(out['edges'])} records", flush=True)
        return

    noaa = json.loads((DATA / "sam31_edge_measurements.json").read_text())["edges"]
    G = ox.load_graphml(DATA / "osm_road_graph.graphml")
    lines = edge_lines_utm(G)
    footprints = scene_footprints()
    out = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {"edges": {}, "skipped": {}}
    CACHE_DIR.mkdir(exist_ok=True)
    instrument = SAM31GeometricInstrument(checkpoint_dir=PROJECT_ROOT / "models" / "sam3.1")

    todo, seen = [], set()
    for (u, v, k), line in lines.items():
        eid = f"{u}-{v}-{k}"
        undirected = frozenset({(u, v, k), (v, u, k)})
        if undirected in seen or not noaa.get(eid, {}).get("needs_drone_recon"):
            continue
        if args.edge and eid not in args.edge and f"{v}-{u}-{k}" not in args.edge:
            continue
        seen.add(undirected)
        todo.append((u, v, k, line))
    todo = todo[: args.limit] if args.limit else todo
    print(f"{len(todo)} drone-flagged roads to check", flush=True)

    t0 = time.perf_counter()
    for i, (u, v, k, line) in enumerate(todo, 1):
        eid, rev = f"{u}-{v}-{k}", f"{v}-{u}-{k}"
        best = None
        for scene in SCENES:
            if not footprints[scene.name].intersects(line):
                continue
            patch = read_patch(scene, line)
            cov = coverage(patch, line)
            if best is None or cov > best[1]:
                best = (patch, cov)
        if best is None or best[1] < MIN_COVERAGE:
            reason = "outside drone coverage" if best is None else f"drone imagery covers only {best[1]:.0%} of the road"
            out["skipped"][eid] = reason
            print(f"  [{i}/{len(todo)}] {eid}: skipped ({reason})", flush=True)
            continue
        patch = best[0]
        record = remeasure(line, G[u][v][k].get("highway"), patch, segment_patch(patch, instrument))
        record["noaa_min_clear_width_m"] = noaa[eid]["min_clear_width_m"]
        save_patch(CACHE_DIR / f"{eid}.tif", patch)
        out["edges"][eid] = out["edges"][rev] = record
        out["skipped"].pop(eid, None)
        print(
            f"  [{i}/{len(todo)}] {eid}: NOAA {noaa[eid]['min_clear_width_m']:.2f} m -> drone "
            f"{record['min_clear_width_m']:.2f} m ({patch.scene.flown}, {time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        OUT_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")  # resumable

    out["source"] = "SAM 3.1 on CRASAR-U-DROIDs drone orthomosaics of Mexico Beach (CC BY 4.0), 1/8 overview"
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    instrument.unload_weights()


if __name__ == "__main__":
    main()
