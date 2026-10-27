"""Reproducibility verification and automated fetch script for FirstLight judges.

Verifies the presence and structural integrity of all pre-staged geospatial datasets,
baked SAM 3.1 segmentation polygons, Muse Spark 1.3 graded building footprints, and
local model weights (`models/sam3.1` and `models/Llama-3.2-11B-Vision-Instruct-4bit`).

Usage:
    ./.venv/bin/python scripts/setup_data_and_models.py --check
    ./.venv/bin/python scripts/setup_data_and_models.py --download
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

NOAA_GEOTIFF_URL = (
    "https://storms.ngs.noaa.gov/storms/michael/data/20181011a_RGB/"
    "20181011a_299230_0854230_rgb.tif"
)
SAM31_HF_REPO = "facebook/sam3.1"
MLX_VLM_HF_REPO = "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"


def verify_environment() -> tuple[bool, list[dict]]:
    """Check all required data files, baked GeoJSONs, benchmarks, and model weights."""
    checks: list[dict] = []

    required_files = [
        (
            "NOAA Hurricane Michael GeoTIFF (EPSG:32616, 0.2804 m/px)",
            DATA_DIR / "noaa_michael_mexico_beach.tif",
            1_000_000,
        ),
        (
            "OSM Mexico Beach Road Graph (GraphML)",
            DATA_DIR / "osm_road_graph.graphml",
            10_000,
        ),
        (
            "OSM Mexico Beach Roads (GeoJSON)",
            DATA_DIR / "osm_roads.geojson",
            10_000,
        ),
        (
            "OSM Mexico Beach Building Footprints (GeoJSON)",
            DATA_DIR / "osm_buildings.geojson",
            10_000,
        ),
        (
            "Baked SAM 3.1 Pavement/Debris/Water Masks (sam-3.1)",
            DATA_DIR / "sam31_mexico_beach_masks.geojson",
            5_000,
        ),
        (
            "Muse Spark 1.3 Graded Buildings (334 structures)",
            DATA_DIR / "osm_buildings_graded.geojson",
            20_000,
        ),
        (
            "Baked SAM 3.1 Road Measurements (158 roads)",
            DATA_DIR / "sam31_edge_measurements.json",
            100_000,
        ),
        (
            "Held-out Benchmark Samples (40 roads, 40 buildings)",
            DATA_DIR / "ground_truth" / "road_width_samples.json",
            1_000,
        ),
        (
            "Meta SAM 3.1 Local Weights (model.safetensors)",
            MODELS_DIR / "sam3.1" / "model.safetensors",
            100_000_000,
        ),
        (
            "Apple Silicon MLX 4-bit Llama 3.2 11B Vision Config",
            MODELS_DIR / "Llama-3.2-11B-Vision-Instruct-4bit" / "config.json",
            100,
        ),
    ]

    all_ok = True
    for label, path, min_bytes in required_files:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        passed = exists and size >= min_bytes
        if not passed:
            all_ok = False
        checks.append(
            {
                "label": label,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": exists,
                "size_bytes": size,
                "passed": passed,
            }
        )

    # Validate feature counts on baked artifacts if present
    sam_path = DATA_DIR / "sam31_mexico_beach_masks.geojson"
    if sam_path.exists():
        sam_data = json.loads(sam_path.read_text(encoding="utf-8"))
        n_features = len(sam_data.get("features", []))
        checks.append(
            {
                "label": f"SAM 3.1 Mask Feature Count ({n_features} polygons)",
                "path": "data/sam31_mexico_beach_masks.geojson",
                "exists": True,
                "size_bytes": n_features,
                "passed": n_features >= 50,
            }
        )
        if n_features < 50:
            all_ok = False

    bldg_path = DATA_DIR / "osm_buildings_graded.geojson"
    if bldg_path.exists():
        bldg_data = json.loads(bldg_path.read_text(encoding="utf-8"))
        n_bldgs = len(bldg_data.get("features", []))
        checks.append(
            {
                "label": f"Muse Spark 1.3 Graded Building Count ({n_bldgs} buildings)",
                "path": "data/osm_buildings_graded.geojson",
                "exists": True,
                "size_bytes": n_bldgs,
                "passed": n_bldgs >= 300,
            }
        )
        if n_bldgs < 300:
            all_ok = False

    return all_ok, checks


def download_missing_assets() -> None:
    """Download or regenerate any missing datasets or HuggingFace model weights."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    geotiff_path = DATA_DIR / "noaa_michael_mexico_beach.tif"
    if not geotiff_path.exists():
        print(f"[download] Fetching NOAA Hurricane Michael GeoTIFF -> {geotiff_path} ...")
        urllib.request.urlretrieve(NOAA_GEOTIFF_URL, geotiff_path)

    sam_dir = MODELS_DIR / "sam3.1"
    if not (sam_dir / "model.safetensors").exists():
        print(f"[download] Downloading {SAM31_HF_REPO} -> {sam_dir} ...")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=SAM31_HF_REPO, local_dir=str(sam_dir))

    mlx_dir = MODELS_DIR / "Llama-3.2-11B-Vision-Instruct-4bit"
    if not (mlx_dir / "config.json").exists():
        print(f"[download] Downloading {MLX_VLM_HF_REPO} -> {mlx_dir} ...")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MLX_VLM_HF_REPO, local_dir=str(mlx_dir))

    sam_geojson = DATA_DIR / "sam31_mexico_beach_masks.geojson"
    if not sam_geojson.exists():
        print("[rebuild] Running scripts/bake_sam31_masks.py ...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "bake_sam31_masks.py")],
            check=True,
        )

    graded_geojson = DATA_DIR / "osm_buildings_graded.geojson"
    if not graded_geojson.exists():
        print("[rebuild] Running scripts/grade_buildings_muse_spark.py ...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "grade_buildings_muse_spark.py")],
            check=True,
        )

    bench_json = BENCHMARKS_DIR / "benchmark_results.json"
    if not bench_json.exists():
        print("[rebuild] Running benchmarks/run_benchmarks.py ...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "benchmarks" / "run_benchmarks.py")],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify or download FirstLight datasets, SAM 3.1 masks, and local MLX/PyTorch models."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all local datasets, baked model artifacts, and weights without downloading.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing NOAA/HuggingFace assets and regenerate baked GeoJSON artifacts.",
    )
    args = parser.parse_args()

    if args.download:
        download_missing_assets()

    all_ok, checks = verify_environment()
    print("=" * 78)
    print("FirstLight Reproducibility & Asset Verification Report")
    print("=" * 78)
    for item in checks:
        status = "PASS" if item["passed"] else "MISSING"
        print(f"[{status:7s}] {item['label']:55s} ({item['path']})")
    print("=" * 78)

    if not all_ok:
        print(
            "One or more required assets are missing. Run:\n"
            "  ./.venv/bin/python scripts/setup_data_and_models.py --download"
        )
        return 1

    print("All required geospatial datasets, SAM 3.1 masks, Muse Spark grades, and model weights verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
