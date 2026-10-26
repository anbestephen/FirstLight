---
name: mlx-failover
description: Implements the 3-tier connectivity router and local Apple Silicon MLX vision model wrapper for offline VLM inference.
---

# MLX & Circuit Breaker Guidelines
- **Tier Structure:** 
  1. Primary: Cloud Muse Spark via Meta Model API.
  2. Edge Fallback: Quantized local Meta vision model via `mlx-vlm` on Apple Silicon.
  3. Deterministic Fallback: Pure geometric rules and graph routing.
- **Memory Safety:** Explicitly manage model loading and unloading in unified memory to prevent swapping during local offline execution.

