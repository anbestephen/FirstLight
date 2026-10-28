"""
Offline-tier model comparison on the held-out TEST building labels.

For each local MLX model: grade the 38 test buildings with the same xBD prompt and before/after
composite Muse Spark uses (scored with `building_metrics`), then check that the commander agent's
JSON action protocol yields a tool call for three demo questions. Muse Glimmer is run with low
reasoning ("low") and with reasoning skipped ("direct").

Usage: ./.venv/bin/python benchmarks/edge_model_comparison.py [model ...]
Writes `benchmarks/edge_model_comparison.json` (~45 min for all models on an M1 Max).
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grade_buildings_muse_spark as g  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_vlm import generate  # noqa: E402

from benchmarks.run_benchmarks import building_metrics  # noqa: E402
from src.agent import CommanderAgent  # noqa: E402
from src.agent_tools import AgentContext, AgentTools  # noqa: E402
from src.failover import ResidentMLXRuntime, format_local_prompt, local_final_answer  # noqa: E402

MODELS = {
    "llama-3.2-11b-vision": (PROJECT_ROOT / "models/Llama-3.2-11B-Vision-Instruct-4bit", {"base": True}),
    "muse-glimmer-30b": (PROJECT_ROOT / "models/Muse-Glimmer-30B-4bit", {"low": True, "direct": False}),
}
QUESTIONS = [
    "Get an ambulance to the worst-hit block",
    "Which debris pile should we clear first for fire engines?",
    "Is 15th Street passable for an ambulance?",
]
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "edge_model_comparison.json"


def main(names: list[str]) -> None:
    labels = json.loads((PROJECT_ROOT / "data/ground_truth/building_damage_manual_test.json").read_text())["labels"]
    wanted = {l["building_id"] for l in labels.values() if l["grade"] != "cant_tell"}
    buildings = [b for b in g.load_buildings() if b["building_id"] in wanted]
    tmp = Path(tempfile.mkdtemp())
    images = {}
    for b in buildings:
        path = tmp / f"{b['building_id']}.png"
        path.write_bytes(g.composite_png(b["centroid_utm"], b["footprint_utm"]))
        images[b["building_id"]] = str(path)

    agent = CommanderAgent(
        AgentTools(AgentContext(graph=lambda: None, buildings=lambda: [], command_post_lonlat=lambda: (0, 0), buildings_source=PROJECT_ROOT)),
        api_key=None,
        api_base_url="",
    )
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    for name in names:
        model_path, variants = MODELS[name]
        runtime = ResidentMLXRuntime(model_path)
        t0 = time.time()
        model, processor, config = runtime.ensure_loaded()
        load_s = time.time() - t0
        for variant, reasoning in variants.items():
            preds, latency, tps = {}, [], []
            for b in buildings:
                prompt = format_local_prompt(processor, config, g.PROMPT, num_images=1, reasoning=reasoning)
                t = time.time()
                out = generate(model, processor, prompt, image=[images[b["building_id"]]],
                               max_tokens=1024, temp=0.0, verbose=False)
                latency.append(time.time() - t)
                tps.append(out.generation_tps)
                answer = g.parse_answer(local_final_answer(out.text))
                if answer:
                    preds[b["building_id"]] = answer["damage_severity"]
            metrics = building_metrics(labels, preds, name)
            tool_calls = 0
            for q in QUESTIONS:
                prompt = format_local_prompt(processor, config, agent._edge_prompt(q, []), num_images=0, reasoning=reasoning)
                text = local_final_answer(generate(model, processor, prompt, max_tokens=1024, temp=0.0, verbose=False).text)
                try:
                    tool_calls += int("tool" in json.loads(text[text.find("{"): text.rfind("}") + 1]))
                except ValueError:
                    pass
            key = f"{name}:{variant}"
            results[key] = {
                "load_s": round(load_s, 1),
                "parsed": f"{len(preds)}/{len(buildings)}",
                "median_s_per_grade": round(sorted(latency)[len(latency) // 2], 1),
                "median_tok_s": round(sorted(tps)[len(tps) // 2], 1),
                "exact": metrics.get("four_class_accuracy"),
                "within_one": metrics.get("within_one_grade"),
                "high_sev_recall": metrics.get("high_severity_recall"),
                "high_sev_precision": metrics.get("high_severity_precision"),
                "agent_tool_calls": f"{tool_calls}/{len(QUESTIONS)}",
            }
            print(key, json.dumps(results[key]), flush=True)
        runtime.unload()
        del model, processor
        gc.collect()
        mx.clear_cache()
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
