# AGENTS.md — FirstLight Project Rules

## Core System Architecture
- **Tech Stack:** FastAPI, Python 3.11+, PyTorch (MPS backend), MLX (`mlx-vlm`), Rasterio, GeoPandas, OSMnx, NetworkX, Shapely.
- **Hardware Profile:** Apple Silicon MacBook Pro M1 Max, 64 GB Unified Memory. Always enforce `float16` (`fp16`) or 4-bit quantization; never assume `bf16` hardware acceleration.
- **Key Invariant:** SAM 3.1 is used as a *geometric measurement instrument* (calculating clear road widths in meters via orthogonal cross-sections), not just a basic semantic image segmenter.

## Execution Constraints
- **Offline Resilience:** Every core geometry and routing function must execute 100% locally without network dependencies.
- **No Hallucinated Counts/Metrics:** All UI reason strings and priority explanations must be mathematically derived from computed pixel-to-meter ratios and graph weights.

