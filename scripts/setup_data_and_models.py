"""Check and fetch what FirstLight needs to run (`scripts/setup_data_and_models.py`).

Core: the committed data the dashboard, routing and rules-only agent run on. A fresh clone has
all of it, so `--check` passes without any download.
Optional, each enabling one feature:
  - offline map tiles      (built locally from the committed NOAA image, no internet)
  - drone image patches    (inspector photos after a drone check; ~124 MB from Hugging Face)
  - Muse Glimmer 30B 4-bit (the on-laptop AI tier; ~17 GB)
  - SAM 3.1 weights        (only to re-run the SAM bake; the measurements are committed)
  - Meta Model API key     (the cloud tier: copy .env.example to .env)

Usage:
    ./.venv/bin/python scripts/setup_data_and_models.py --check
    ./.venv/bin/python scripts/setup_data_and_models.py --download --no-models   # judge quick path
    ./.venv/bin/python scripts/setup_data_and_models.py --download               # everything
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# The Mexico Beach crop is committed at data/noaa_michael_mexico_beach.tif. Its source is the
# Oct 11, 2018 flight, which NOAA now serves only as a whole-flight archive (14.9 GB).
NOAA_SOURCE_ARCHIVE = "https://stormscdn.ngs.noaa.gov/downloads/20181011a_RGB.tar"
SAM31_HF_REPO = "facebook/sam3.1"
MLX_VLM_HF_REPO = "mlx-community/Muse-Glimmer-30B-4bit"


CORE_FILES = [
    ("NOAA post-storm image, Oct 11 2018 (0.28 m/px)", DATA_DIR / "noaa_michael_mexico_beach.tif", 1_000_000),
    ("OpenStreetMap road graph", DATA_DIR / "osm_road_graph.graphml", 10_000),
    ("OpenStreetMap roads", DATA_DIR / "osm_roads.geojson", 10_000),
    ("Muse Spark building grades (334 buildings)", DATA_DIR / "osm_buildings_graded.geojson", 20_000),
    ("SAM 3.1 road measurements (158 roads)", DATA_DIR / "sam31_edge_measurements.json", 100_000),
    ("SAM 3.1 masks", DATA_DIR / "sam31_mexico_beach_masks.geojson", 10_000),
    ("SAM 3.1 drone re-measurements (74 roads)", DATA_DIR / "sam31_drone_measurements.json", 10_000),
    ("Benchmark samples and hand labels", DATA_DIR / "ground_truth" / "road_widths_manual_test.json", 1_000),
]
GLIMMER_DIR = MODELS_DIR / "Muse-Glimmer-30B-4bit"
SAM_DIR = MODELS_DIR / "sam3.1"
TILES_DIR = DATA_DIR / "tiles" / "post"
DRONE_CACHE_DIR = DATA_DIR / "drone_cache"


def _glimmer_complete() -> bool:
    """config.json plus every weight shard the index lists."""
    index = GLIMMER_DIR / "model.safetensors.index.json"
    if not (GLIMMER_DIR / "config.json").exists() or not index.exists():
        return False
    shards = set(json.loads(index.read_text())["weight_map"].values())
    return all((GLIMMER_DIR / name).exists() and (GLIMMER_DIR / name).stat().st_size > 0 for name in shards)


def _drone_patches() -> tuple[int, int]:
    """(patches on disk, roads with a stored drone re-measurement)."""
    path = DATA_DIR / "sam31_drone_measurements.json"
    if not path.exists():
        return 0, 0
    roads = set()
    for eid in json.loads(path.read_text())["edges"]:
        u, v, k = eid.split("-")
        roads.add((min(u, v), max(u, v), k))
    have = len(list(DRONE_CACHE_DIR.glob("*.tif"))) if DRONE_CACHE_DIR.exists() else 0
    return have, len(roads)


def _api_key_set() -> bool:
    """MODEL_API_KEY in the environment or in .env (its value is never printed)."""
    env = PROJECT_ROOT / ".env"
    if os.environ.get("MODEL_API_KEY"):
        return True
    return env.exists() and any(
        line.split("=", 1)[0].strip().removeprefix("export ").strip() == "MODEL_API_KEY"
        and line.split("=", 1)[1].strip()
        for line in env.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )


def verify_environment() -> tuple[bool, list[dict]]:
    """Core files (all required) and optional features (each reported with what it enables)."""
    checks: list[dict] = []
    for label, path, min_bytes in CORE_FILES:
        size = path.stat().st_size if path.exists() else 0
        checks.append({"group": "core", "label": label, "path": str(path.relative_to(PROJECT_ROOT)), "passed": size >= min_bytes})

    have, want = _drone_patches()
    optional = [
        ("Offline map tiles (full-resolution imagery)", "data/tiles/post", TILES_DIR.exists() and any(TILES_DIR.iterdir())),
        (f"Drone image patches ({have}/{want})", "data/drone_cache", want > 0 and have >= want),
        ("Muse Glimmer 30B 4-bit, all weights (on-laptop AI)", "models/Muse-Glimmer-30B-4bit", _glimmer_complete()),
        ("SAM 3.1 weights (only to re-run the SAM bake)", "models/sam3.1", (SAM_DIR / "model.safetensors").exists()),
        ("Meta Model API key (cloud AI tier)", ".env", _api_key_set()),
    ]
    for label, path, ok in optional:
        checks.append({"group": "optional", "label": label, "path": path, "passed": bool(ok)})
    return all(c["passed"] for c in checks if c["group"] == "core"), checks


def download_missing_assets(*, models: bool = True) -> None:
    """Build or fetch the optional assets that are missing (models only when `models`)."""
    if not (DATA_DIR / "noaa_michael_mexico_beach.tif").exists():
        raise SystemExit(
            "data/noaa_michael_mexico_beach.tif is missing. It is part of the repository: restore it with "
            f"`git checkout -- data/noaa_michael_mexico_beach.tif` (source archive: {NOAA_SOURCE_ARCHIVE})."
        )
    if not (TILES_DIR.exists() and any(TILES_DIR.iterdir())):
        print("[build] Offline map tiles from the NOAA image (no internet) ...", flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_map_tiles.py")], check=True)
    have, want = _drone_patches()
    if have < want:
        print(f"[download] Drone image patches from Hugging Face ({want - have} roads) ...", flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "drone_remeasure.py"), "--patches-only"], check=True)
    if not models:
        return
    from huggingface_hub import snapshot_download

    if not _glimmer_complete():
        print(f"[download] {MLX_VLM_HF_REPO} -> {GLIMMER_DIR} (~17 GB) ...", flush=True)
        snapshot_download(repo_id=MLX_VLM_HF_REPO, local_dir=str(GLIMMER_DIR))
    if not (SAM_DIR / "model.safetensors").exists():
        print(f"[download] {SAM31_HF_REPO} -> {SAM_DIR} ...", flush=True)
        snapshot_download(repo_id=SAM31_HF_REPO, local_dir=str(SAM_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or fetch FirstLight's data and models.")
    parser.add_argument("--check", action="store_true", help="Report what is present (default).")
    parser.add_argument("--download", action="store_true", help="Build or fetch missing optional assets.")
    parser.add_argument("--no-models", action="store_true", help="With --download: skip the model weights.")
    args = parser.parse_args()

    if args.download:
        download_missing_assets(models=not args.no_models)

    core_ok, checks = verify_environment()
    print("=" * 78)
    for group, title in (("core", "Core (committed; required)"), ("optional", "Optional features")):
        print(title)
        for item in (c for c in checks if c["group"] == group):
            status = "OK" if item["passed"] else ("MISSING" if group == "core" else "not set")
            print(f"  [{status:7s}] {item['label']:52s} ({item['path']})")
    print("=" * 78)
    if not core_ok:
        print("Core data is missing: re-clone the repository or `git checkout -- data/`.")
        return 1
    print("Core data present: the dashboard, routing and rules-only agent run.")
    if not all(c["passed"] for c in checks):
        print("Optional assets: ./.venv/bin/python scripts/setup_data_and_models.py --download [--no-models]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
