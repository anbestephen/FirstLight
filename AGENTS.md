# AGENTS.md — FirstLight Project Rules

The submitted hackathon proposal is the product spec; this file and the README describe how
the code meets it and may change.

## System
- **Stack:** FastAPI, Python 3.11+, PyTorch (MPS), MLX (`mlx-vlm`), Rasterio, GeoPandas, OSMnx, NetworkX, Shapely, vendored Leaflet.
- **Hardware:** Apple Silicon M1 Max, 64 GB unified memory. Use `float16` or 4-bit; never assume `bf16` acceleration.
- **SAM 3.1 is a measuring instrument:** road clear width in metres from 5 m orthogonal cross-sections over SAM probability maps (`src/sam_measure.py`), not just segmentation.
- **Meta models:** SAM 3.1 (local), Muse Spark 1.3 via the Meta Model API (building grades from pre/post imagery, commander agent, corridor grading), Muse Glimmer 30B 4-bit on MLX (offline fallback; low reasoning for grading, none for agent steps). Use versioned API model ids (e.g. `muse-spark-1.3`).

## Rules
- **Offline core:** measurement, routing, dashboards, field sync and the no-model agent tier must run with no network.
- **No invented numbers:** every number shown to a user comes from measurement, routing or a stored model output. Agent answers are checked by `verify_numbers`; tools in `src/agent_tools.py` never call a model.
- **Honest provenance:** label which tier or model produced every output; never present a template or fallback as model output.
- **Benchmark integrity:** labels in `data/ground_truth/` are written only by the `/label` tool. Tune on the dev set; report the test set. Scripts never generate ground truth.
- **Safety over convenience:** never dispatch to a destination without a passable route; unseen pavement counts as blocked and is flagged for drone reconnaissance.

## Workflow
- Trunk-based on `master`; one commit per feature with a single short line, e.g. `fix(routing): keep ambulances off footways`. No AI co-author trailers.
- Run `./.venv/bin/pytest` before committing. Do not change commit dates.
