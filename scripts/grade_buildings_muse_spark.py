"""
Grade every Mexico Beach building with Muse Spark from pre/post imagery
(`scripts/grade_buildings_muse_spark.py`).

For each OSM building footprint, one composite image is sent to `muse-spark-1.3`:
BEFORE = USDA NAIP 2017 (1 m) and AFTER = NOAA Oct 11, 2018 (0.28 m), both resampled to the
same 0.25 m/px metric crop with the footprint outlined. The model returns an xBD damage grade
with its own confidence and one sentence of visual evidence. Every answer is stored per
building - nothing is assigned by rule or quota.

Answers are appended to `data/muse_spark_building_grades.jsonl` as they arrive, so a run can
be interrupted and resumed without paying for a building twice. `--write-geojson` builds
`data/osm_buildings_graded.geojson` once every building has a grade.

Usage:
  .venv/bin/python scripts/grade_buildings_muse_spark.py --limit 20   # trial (reports tokens)
  .venv/bin/python scripts/grade_buildings_muse_spark.py              # grade the rest
  .venv/bin/python scripts/grade_buildings_muse_spark.py --write-geojson
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv
import geopandas as gpd
import httpx
from PIL import Image, ImageDraw
from shapely.geometry import mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.failover import META_MODEL_API_BASE_URL, normalize_damage_severity  # noqa: E402
from src.labeling import (  # noqa: E402
    CROP_RES_M,
    POST_EVENT_PATH,
    PRE_EVENT_PATH,
    render_metric_crop,
    utm_to_pixel,
)

DATA_DIR = PROJECT_ROOT / "data"
BUILDINGS_PATH = DATA_DIR / "osm_buildings.geojson"
GRADES_JSONL = DATA_DIR / "muse_spark_building_grades.jsonl"
GRADED_GEOJSON = DATA_DIR / "osm_buildings_graded.geojson"

MODEL = "muse-spark-1.3"
MIN_HALF_EXTENT_M = 25.0
CONTEXT_M = 8.0
MAX_PANEL_PX = 320
GRADING_METHOD = "muse-spark-1.3 per-building pre/post (NAIP 2017 vs NOAA 2018), prompt xbd-v2"
# v2 uses the xBD definitions verbatim-in-spirit and reserves "destroyed" for buildings with no
# standing structure (v1 over-graded major damage as destroyed on the dev labels).
PROMPT_VERSION = "xbd-v2"
PROMPT = (
    "You are grading hurricane damage to ONE building for emergency responders. The image has two "
    "panels of the same area: LEFT = BEFORE (Oct/Nov 2017, 1 m aerial, blurrier), RIGHT = AFTER "
    "Hurricane Michael (Oct 11, 2018, 0.28 m aerial). The building is inside the yellow dashed outline "
    "(from OpenStreetMap; it may be offset by a few metres). Grade only that building on the xBD scale:\n"
    "- no_damage: undisturbed; no visible roof or structural damage.\n"
    "- minor_damage: roof elements or cover missing, tarps, visible cracks, debris on or around it, "
    "but the roof outline is intact.\n"
    "- major_damage: partial wall or roof collapse; a large part of the roof is gone but walls stand.\n"
    "- destroyed: completely collapsed, pancaked, swept off its foundation, or no longer present "
    "(bare slab or rubble only).\n"
    "Choose destroyed ONLY when no standing structure remains. Debris around a building is not damage "
    "to that building. Reply with ONLY a JSON object: {\"damage_severity\": one of no_damage, "
    "minor_damage, major_damage, destroyed, \"confidence\": number 0-1, \"visual_evidence\": one short "
    "sentence describing what changed between BEFORE and AFTER}."
)


def half_extent_m(centroid_utm: list[float], footprint_utm: list[list[float]]) -> float:
    """Crop half-size that fits the whole footprint plus some context (at least 50 m across)."""
    reach = max(max(abs(x - centroid_utm[0]), abs(y - centroid_utm[1])) for x, y in footprint_utm)
    return max(MIN_HALF_EXTENT_M, reach + CONTEXT_M)


def composite_png(centroid_utm: list[float], footprint_utm: list[list[float]]) -> bytes:
    """BEFORE | AFTER side by side (same metric grid) with the footprint outlined in both."""
    half = half_extent_m(centroid_utm, footprint_utm)
    panels = []
    outline = [utm_to_pixel(x, y, centroid_utm, half) for x, y in footprint_utm]
    for path, label in ((PRE_EVENT_PATH, "BEFORE 2017"), (POST_EVENT_PATH, "AFTER 2018")):
        img = Image.fromarray(render_metric_crop(path, centroid_utm, half))
        draw = ImageDraw.Draw(img)
        draw.line(outline + [outline[0]], fill=(250, 204, 21), width=2)
        if img.size[0] > MAX_PANEL_PX:
            img = img.resize((MAX_PANEL_PX, MAX_PANEL_PX), Image.LANCZOS)
            draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 92, 16], fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 255))
        panels.append(img)
    size = panels[0].size[0]
    sheet = Image.new("RGB", (size * 2 + 6, size), (255, 255, 255))
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (size + 6, 0))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def load_buildings() -> list[dict[str, Any]]:
    gdf = gpd.read_file(BUILDINGS_PATH)
    utm = gdf.to_crs("EPSG:32616")
    out = []
    for idx in range(len(gdf)):
        geom = utm.geometry.iloc[idx]
        out.append(
            {
                "building_id": f"bldg-{idx}",
                "building_type": str(gdf.iloc[idx].get("building") or "yes"),
                "geometry_wgs84": gdf.geometry.iloc[idx],
                "centroid_utm": [geom.centroid.x, geom.centroid.y],
                "footprint_utm": [list(c) for c in geom.exterior.coords],
            }
        )
    return out


def load_done() -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if GRADES_JSONL.exists():
        for line in GRADES_JSONL.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("damage_severity") and rec.get("prompt_version") == PROMPT_VERSION:
                done[rec["building_id"]] = rec
    return done


def parse_answer(content: str) -> dict[str, Any] | None:
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    raw = str(data.get("damage_severity", "")).strip().lower()
    if raw not in ("no_damage", "minor_damage", "major_damage", "destroyed"):
        raw = normalize_damage_severity(raw, default="")  # type: ignore[arg-type]
        if not raw:
            return None
    try:
        conf = float(data.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    return {"damage_severity": raw, "model_confidence": conf, "visual_evidence": str(data.get("visual_evidence", ""))[:300]}


async def grade_one(client: httpx.AsyncClient, api_key: str, bldg: dict[str, Any], effort: str | None) -> dict[str, Any]:
    image_b64 = base64.b64encode(
        await asyncio.to_thread(composite_png, bldg["centroid_utm"], bldg["footprint_utm"])
    ).decode("ascii")
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    if effort:
        payload["reasoning_effort"] = effort
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            resp = await client.post(
                f"{META_MODEL_API_BASE_URL}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = parse_answer(data["choices"][0]["message"].get("content") or "")
            if answer:
                return {
                    "building_id": bldg["building_id"],
                    **answer,
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "reasoning_effort": effort,
                    "usage": data.get("usage", {}),
                    "latency_s": round(time.perf_counter() - t0, 2),
                }
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            if attempt == 2:
                return {"building_id": bldg["building_id"], "error": f"{type(exc).__name__}: {exc}"[:200]}
        await asyncio.sleep(1.5 * (attempt + 1))
    return {"building_id": bldg["building_id"], "error": "unparseable answer"}


async def run_grading(limit: int | None, concurrency: int, effort: str | None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("MODEL_API_KEY is required in .env")
    if not PRE_EVENT_PATH.exists():
        raise SystemExit("Pre-event imagery missing: run scripts/fetch_pre_event_naip.py first")

    buildings = load_buildings()
    done = load_done()
    todo = [b for b in buildings if b["building_id"] not in done][: limit or None]
    print(f"{len(done)} already graded, grading {len(todo)} now with {MODEL} (effort={effort})", flush=True)

    sem = asyncio.Semaphore(concurrency)
    totals = {"ok": 0, "err": 0, "prompt": 0, "completion": 0, "reasoning": 0}
    async with httpx.AsyncClient() as client:

        async def worker(b: dict[str, Any]) -> None:
            async with sem:
                rec = await grade_one(client, api_key, b, effort)
            with GRADES_JSONL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if "error" in rec:
                totals["err"] += 1
                print(f"  {b['building_id']}: ERROR {rec['error']}", flush=True)
                return
            totals["ok"] += 1
            usage = rec.get("usage") or {}
            totals["prompt"] += int(usage.get("prompt_tokens", 0))
            totals["completion"] += int(usage.get("completion_tokens", 0))
            totals["reasoning"] += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
            print(
                f"  {b['building_id']}: {rec['damage_severity']:<12} conf={rec['model_confidence']} "
                f"{rec['latency_s']}s  {rec['visual_evidence'][:70]}",
                flush=True,
            )

        await asyncio.gather(*(worker(b) for b in todo))

    n = max(1, totals["ok"])
    print(
        f"\nDone: {totals['ok']} graded, {totals['err']} errors. Tokens per building: "
        f"prompt {totals['prompt'] / n:.0f}, completion {totals['completion'] / n:.0f} "
        f"(reasoning {totals['reasoning'] / n:.0f}). Remaining: {len(buildings) - len(load_done())}",
        flush=True,
    )


def write_geojson() -> None:
    buildings = load_buildings()
    done = load_done()
    missing = [b["building_id"] for b in buildings if b["building_id"] not in done]
    if missing:
        raise SystemExit(f"{len(missing)} buildings not graded yet (e.g. {missing[:3]}); run grading first")
    counts: dict[str, int] = {}
    features = []
    for b in buildings:
        rec = done[b["building_id"]]
        sev = rec["damage_severity"]
        counts[sev] = counts.get(sev, 0) + 1
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(b["geometry_wgs84"]),
                "properties": {
                    "building_id": b["building_id"],
                    "building_type": b["building_type"],
                    "damage_severity": sev,
                    "building_damage_severity": sev,
                    "is_high_severity": sev in ("major_damage", "destroyed"),
                    "model_confidence": rec.get("model_confidence"),
                    "grading_source": MODEL,
                    "grading_method": GRADING_METHOD,
                    "visual_cue_summary": f"[MUSE-SPARK-1.3 PRE/POST] {rec.get('visual_evidence', '')}",
                },
            }
        )
    GRADED_GEOJSON.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "properties": {
                    "grading_source": MODEL,
                    "grading_method": GRADING_METHOD,
                    "pre_event": "USDA NAIP 2017",
                    "post_event": "NOAA 2018-10-11",
                    "total_buildings": len(features),
                    "severity_distribution": counts,
                },
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(features)} per-building grades -> {GRADED_GEOJSON} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade buildings with Muse Spark from pre/post imagery")
    parser.add_argument("--limit", type=int, default=None, help="grade at most N new buildings")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--effort", default="low", help="muse-spark reasoning_effort ('' for model default)")
    parser.add_argument("--write-geojson", action="store_true", help="build the graded GeoJSON from the cache")
    parser.add_argument("--preview", metavar="BUILDING_ID", help="save the composite image for one building")
    args = parser.parse_args()

    if args.preview:
        b = next(b for b in load_buildings() if b["building_id"] == args.preview)
        out = PROJECT_ROOT / f"{args.preview}_prepost.png"
        out.write_bytes(composite_png(b["centroid_utm"], b["footprint_utm"]))
        print(f"Saved {out}")
    elif args.write_geojson:
        write_geojson()
    else:
        asyncio.run(run_grading(args.limit, args.concurrency, args.effort or None))


if __name__ == "__main__":
    main()
