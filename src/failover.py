"""
3-Tier VLM damage grading and connectivity circuit-breaker router (`src/failover.py`).

Key Invariants (AGENTS.md & mlx-failover skill):
1. Tier Structure:
   - Tier 1 (Cloud Primary): Meta Model API (`https://api.meta.ai/v1/chat/completions`,
     Muse Spark / Llama 3.2 Vision) using `MODEL_API_KEY` with an explicit 2.5s timeout.
   - Tier 2 (Edge Fallback): Local quantized Apple Silicon inference via `mlx-vlm` using
     `models/Llama-3.2-11B-Vision-Instruct-4bit` with explicit unified memory release
     between runs to prevent memory swapping.
   - Tier 3 (Deterministic Fallback): Pure geometric damage and obstacle heuristics derived
     from SAM 3.1 pixel-to-meter ratios, clear widths, and debris/water overlap ratios.
2. No Hallucinated Counts/Metrics: Every rationale string embeds the exact measured
   geometric metrics (`min_clear_width_m`, `debris_area_m2`, `gsd_m_per_px`, passability).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import gc
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Literal

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from src.geometry import (
    AMBULANCE_MIN_WIDTH_M,
    ATV_MIN_WIDTH_M,
    FIRE_ENGINE_MIN_WIDTH_M,
    PassabilityFlags,
    compute_passability_flags,
)

DamageSeverity = Literal["no_damage", "minor_damage", "major_damage", "destroyed"]

VALID_DAMAGE_SEVERITIES: tuple[DamageSeverity, ...] = (
    "no_damage",
    "minor_damage",
    "major_damage",
    "destroyed",
)

SEVERITY_TO_SCORE: dict[DamageSeverity, int] = {
    "no_damage": 0,
    "minor_damage": 1,
    "major_damage": 2,
    "destroyed": 3,
}

META_MODEL_API_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_CLOUD_MODEL = "muse-spark"
DEFAULT_LOCAL_MLX_MODEL_PATH = "models/Llama-3.2-11B-Vision-Instruct-4bit"
DEFAULT_CLOUD_TIMEOUT_S = 2.5


class ConnectivityTier(str, Enum):
    """Execution tiers in the FirstLight 3-tier connectivity circuit breaker."""

    TIER_1_CLOUD = "tier_1_cloud_meta_api"
    TIER_2_MLX = "tier_2_edge_mlx_vlm"
    TIER_3_DETERMINISTIC = "tier_3_deterministic_geometry"


class FailoverMode(str, Enum):
    """Operating mode for the connectivity router (supports live demo overrides)."""

    AUTO = "auto"
    FORCE_CLOUD = "force_cloud"
    FORCE_MLX = "force_mlx"
    FORCE_DETERMINISTIC = "force_deterministic"


@dataclass
class GeometricTelemetry:
    """
    Measured SAM 3.1 & raster geometric telemetry passed into the 3-tier router
    so all damage grades and passability explanations remain grounded in real metrics.
    """

    target_id: str = "segment-0"
    min_clear_width_m: float = 4.0
    debris_area_m2: float = 0.0
    water_area_m2: float = 0.0
    has_water_obstruction: bool = False
    building_debris_overlap_ratio: float = 0.0
    building_water_overlap_ratio: float = 0.0
    gsd_m_per_px: float = 0.2804
    road_name: str = "Highway 98"


class DamageAssessmentResult(BaseModel):
    """Structured VLM / deterministic damage and road passability assessment."""

    target_id: str
    building_damage_severity: DamageSeverity
    road_damage_severity: DamageSeverity
    severity_score: int = Field(ge=0, le=3)
    is_high_severity: bool
    passability: dict[str, bool]
    min_clear_width_m: float
    debris_area_m2: float
    water_area_m2: float
    has_water_obstruction: bool
    gsd_m_per_px: float
    passability_rationale: str
    visual_cue_summary: str
    tier_used: ConnectivityTier
    fallback_chain: list[str] = Field(default_factory=list)
    latency_ms: float
    memory_cleared: bool = False


def normalize_damage_severity(raw_value: Any, default: DamageSeverity = "minor_damage") -> DamageSeverity:
    """Normalize a string or integer severity label into the strict 4-class schema."""
    if raw_value is None:
        return default
    if isinstance(raw_value, int):
        mapping: dict[int, DamageSeverity] = {
            0: "no_damage",
            1: "minor_damage",
            2: "major_damage",
            3: "destroyed",
        }
        return mapping.get(raw_value, default)

    text = str(raw_value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in VALID_DAMAGE_SEVERITIES:
        return text  # type: ignore[return-value]
    if "destroy" in text or "collapse" in text or "total" in text or "catastrophic" in text:
        return "destroyed"
    if "major" in text or "severe" in text or "heavy" in text:
        return "major_damage"
    if "minor" in text or "light" in text or "partial" in text or "moderate" in text:
        return "minor_damage"
    if "no_damage" in text or "none" in text or "intact" in text or "undamaged" in text or "clear" in text:
        return "no_damage"
    return default


def release_unified_memory() -> dict[str, Any]:
    """
    Explicitly release Apple Silicon MLX and PyTorch MPS unified memory caches
    between local model runs to prevent memory pressure or swapping on 64 GB Unified Memory.
    """
    gc.collect()
    mlx_active_after = 0
    mlx_cache_after = 0

    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

        if hasattr(mx, "get_active_memory"):
            mlx_active_after = int(mx.get_active_memory())
            mlx_cache_after = int(mx.get_cache_memory())
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            mlx_active_after = int(mx.metal.get_active_memory())
            mlx_cache_after = int(mx.metal.get_cache_memory())
    except Exception:
        pass

    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    gc.collect()
    return {
        "cleared": True,
        "mlx_active_memory_bytes": mlx_active_after,
        "mlx_cache_memory_bytes": mlx_cache_after,
    }


def _encode_image_to_base64(image_source: str | Path | Image.Image | bytes | None) -> str | None:
    """Encode an image path, PIL Image, or raw bytes to a base64 PNG string."""
    if image_source is None:
        return None
    if isinstance(image_source, bytes):
        return base64.b64encode(image_source).decode("ascii")
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.exists():
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    if isinstance(image_source, Image.Image):
        buf = io.BytesIO()
        image_source.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    return None


def _build_grounding_rationale(
    telemetry: GeometricTelemetry,
    passability: PassabilityFlags,
    building_severity: DamageSeverity,
    road_severity: DamageSeverity,
    visual_cue_summary: str,
) -> str:
    """
    Construct a clear passability rationale string that mathematically combines
    the measured SAM 3.1 geometric metrics (`min_clear_width_m`, `debris_area_m2`,
    `gsd_m_per_px`) with the visual/structural damage grading.
    """
    if telemetry.has_water_obstruction or telemetry.water_area_m2 > 0.0:
        pass_desc = (
            f"IMPASSABLE for all vehicle classes (weight=inf) due to {telemetry.water_area_m2:.2f} m² "
            f"SAM 3.1 water mask overlap on {telemetry.road_name}"
        )
    else:
        allowed: list[str] = []
        blocked: list[str] = []
        if passability.fire_engine:
            allowed.append(f"Fire Engine (>{FIRE_ENGINE_MIN_WIDTH_M:.1f}m)")
        else:
            blocked.append(f"Fire Engine (>{FIRE_ENGINE_MIN_WIDTH_M:.1f}m)")
        if passability.ambulance:
            allowed.append(f"Ambulance (>{AMBULANCE_MIN_WIDTH_M:.1f}m)")
        else:
            blocked.append(f"Ambulance (>{AMBULANCE_MIN_WIDTH_M:.1f}m)")
        if passability.atv:
            allowed.append(f"ATV/Foot (>{ATV_MIN_WIDTH_M:.1f}m)")
        else:
            blocked.append(f"ATV/Foot (>{ATV_MIN_WIDTH_M:.1f}m)")

        if allowed and not blocked:
            pass_desc = f"Passable for all classes ({', '.join(allowed)})"
        elif allowed:
            pass_desc = f"Passable for {', '.join(allowed)}; blocked for {', '.join(blocked)}"
        else:
            pass_desc = f"Blocked for all vehicle classes ({', '.join(blocked)})"

    return (
        f"[{telemetry.road_name}] Measured min clear width {telemetry.min_clear_width_m:.2f}m "
        f"at {telemetry.gsd_m_per_px:.4f} m/px GSD ({telemetry.debris_area_m2:.2f} m² debris). "
        f"{pass_desc}. Structural grade: building={building_severity}, road={road_severity}. "
        f"Cue: {visual_cue_summary}"
    )


def evaluate_deterministic_tier3(
    telemetry: GeometricTelemetry,
) -> tuple[DamageSeverity, DamageSeverity, str]:
    """
    Tier 3 Deterministic Fallback: Compute building and road damage severity and visual
    cue explanation strictly from geometric overlap ratios and clear width measurements.
    """
    # 1. Building damage severity from debris/water footprint overlap ratio
    b_ratio = max(
        float(telemetry.building_debris_overlap_ratio),
        float(telemetry.building_water_overlap_ratio),
    )
    if b_ratio >= 0.50 or (telemetry.has_water_obstruction and telemetry.debris_area_m2 >= 35.0):
        building_severity: DamageSeverity = "destroyed"
    elif b_ratio >= 0.20 or telemetry.debris_area_m2 >= 20.0:
        building_severity = "major_damage"
    elif b_ratio >= 0.05 or telemetry.debris_area_m2 > 0.0:
        building_severity = "minor_damage"
    else:
        building_severity = "no_damage"

    # 2. Road damage severity from water washout and measured clear width
    if telemetry.has_water_obstruction or telemetry.min_clear_width_m <= ATV_MIN_WIDTH_M:
        road_severity: DamageSeverity = "destroyed"
    elif telemetry.min_clear_width_m <= AMBULANCE_MIN_WIDTH_M:
        road_severity = "major_damage"
    elif telemetry.min_clear_width_m <= FIRE_ENGINE_MIN_WIDTH_M or telemetry.debris_area_m2 > 0.0:
        road_severity = "minor_damage"
    else:
        road_severity = "no_damage"

    if telemetry.has_water_obstruction:
        cue = (
            f"Deterministic geometry detected {telemetry.water_area_m2:.2f} m² standing water/washout "
            f"and {telemetry.debris_area_m2:.2f} m² debris."
        )
    elif telemetry.debris_area_m2 > 0.0:
        cue = (
            f"Deterministic geometry measured {telemetry.debris_area_m2:.2f} m² debris reducing "
            f"corridor span to {telemetry.min_clear_width_m:.2f}m."
        )
    else:
        cue = (
            f"Deterministic geometry confirmed unobstructed {telemetry.min_clear_width_m:.2f}m corridor."
        )

    return building_severity, road_severity, cue


def parse_vlm_json_response(
    raw_text: str,
    telemetry: GeometricTelemetry,
) -> tuple[DamageSeverity, DamageSeverity, str]:
    """
    Parse structured JSON or key-value text returned by Tier 1 (Meta API) or Tier 2 (MLX VLM)
    into `(building_damage_severity, road_damage_severity, visual_cue_summary)`.
    """
    det_b, det_r, det_cue = evaluate_deterministic_tier3(telemetry)
    if not raw_text or not raw_text.strip():
        return det_b, det_r, det_cue

    cleaned = raw_text.strip()
    # Extract JSON object if wrapped in markdown fences
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = cleaned[start_idx : end_idx + 1]
        try:
            parsed = json.loads(json_candidate)
            b_sev = normalize_damage_severity(
                parsed.get("building_damage_severity") or parsed.get("building_severity"),
                default=det_b,
            )
            r_sev = normalize_damage_severity(
                parsed.get("road_damage_severity") or parsed.get("road_severity"),
                default=det_r,
            )
            cue = str(
                parsed.get("visual_cue_summary")
                or parsed.get("passability_rationale")
                or parsed.get("rationale")
                or det_cue
            ).strip()
            return b_sev, r_sev, cue
        except json.JSONDecodeError:
            pass

    # Fallback text extraction if VLM responded in prose
    b_sev = normalize_damage_severity(cleaned, default=det_b)
    r_sev = normalize_damage_severity(cleaned, default=det_r)
    cue = cleaned.splitlines()[0][:240]
    return b_sev, r_sev, cue


class VLMFailoverRouter:
    """
    3-Tier Connectivity & VLM Failover Circuit Breaker:
      - Tier 1: Cloud Meta Model API (`https://api.meta.ai/v1/chat/completions`) with 2.5s timeout
      - Tier 2: Local Apple Silicon `mlx-vlm` (`models/Llama-3.2-11B-Vision-Instruct-4bit`)
      - Tier 3: Deterministic Geometric Heuristics
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base_url: str = META_MODEL_API_BASE_URL,
        cloud_model: str = DEFAULT_CLOUD_MODEL,
        cloud_timeout_s: float = DEFAULT_CLOUD_TIMEOUT_S,
        local_mlx_model_path: str | Path = DEFAULT_LOCAL_MLX_MODEL_PATH,
        mode: FailoverMode = FailoverMode.AUTO,
        mlx_runner: Callable[[str, str | Path | Image.Image | None, Path], str] | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("MODEL_API_KEY", "")
        self.api_base_url = api_base_url.rstrip("/")
        self.cloud_model = cloud_model
        self.cloud_timeout_s = float(cloud_timeout_s)
        self.local_mlx_model_path = Path(local_mlx_model_path)
        self.mode = mode
        self._custom_mlx_runner = mlx_runner
        self.last_tier_used: ConnectivityTier = ConnectivityTier.TIER_3_DETERMINISTIC
        self.last_fallback_chain: list[str] = []

    def _build_vlm_prompt(self, telemetry: GeometricTelemetry) -> str:
        """Create a compact, telemetry-grounded prompt requesting strict JSON output."""
        return (
            "You are FirstLight, an emergency disaster response damage assessment system. "
            "Analyze the post-hurricane road and adjacent building condition using these exact "
            f"SAM 3.1 geometric measurements: road='{telemetry.road_name}', "
            f"min_clear_width_m={telemetry.min_clear_width_m:.2f}, "
            f"debris_area_m2={telemetry.debris_area_m2:.2f}, "
            f"water_area_m2={telemetry.water_area_m2:.2f}, "
            f"has_water_obstruction={telemetry.has_water_obstruction}, "
            f"gsd_m_per_px={telemetry.gsd_m_per_px:.4f}. "
            "Respond ONLY with a JSON object containing keys: "
            "'building_damage_severity' (one of: 'no_damage', 'minor_damage', 'major_damage', 'destroyed'), "
            "'road_damage_severity' (one of: 'no_damage', 'minor_damage', 'major_damage', 'destroyed'), "
            "and 'visual_cue_summary' (concise visual and geometric justification)."
        )

    def _call_tier1_cloud_sync(
        self,
        telemetry: GeometricTelemetry,
        image_source: str | Path | Image.Image | bytes | None = None,
        http_client: httpx.Client | None = None,
    ) -> tuple[DamageSeverity, DamageSeverity, str]:
        """
        Tier 1 (Cloud Primary): Call Meta Model API (`https://api.meta.ai/v1/chat/completions`)
        with an explicit 2.5-second timeout.
        """
        if not self.api_key:
            raise RuntimeError("MODEL_API_KEY is not set for Tier 1 Meta Model API.")

        prompt = self._build_vlm_prompt(telemetry)
        b64_img = _encode_image_to_base64(image_source)

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if b64_img:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                }
            )

        payload = {
            "model": self.cloud_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.api_base_url}/chat/completions"

        timeout = httpx.Timeout(self.cloud_timeout_s)
        if http_client is not None:
            response = http_client.post(endpoint, json=payload, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(endpoint, json=payload, headers=headers)

        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        return parse_vlm_json_response(raw_content, telemetry)

    async def _call_tier1_cloud_async(
        self,
        telemetry: GeometricTelemetry,
        image_source: str | Path | Image.Image | bytes | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> tuple[DamageSeverity, DamageSeverity, str]:
        """Async Tier 1 Meta Model API call with strict 2.5s timeout."""
        if not self.api_key:
            raise RuntimeError("MODEL_API_KEY is not set for Tier 1 Meta Model API.")

        prompt = self._build_vlm_prompt(telemetry)
        b64_img = _encode_image_to_base64(image_source)

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if b64_img:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                }
            )

        payload = {
            "model": self.cloud_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.api_base_url}/chat/completions"

        timeout = httpx.Timeout(self.cloud_timeout_s)
        if async_client is not None:
            response = await async_client.post(endpoint, json=payload, headers=headers, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        return parse_vlm_json_response(raw_content, telemetry)

    def _call_tier2_mlx(
        self,
        telemetry: GeometricTelemetry,
        image_source: str | Path | Image.Image | None = None,
    ) -> tuple[DamageSeverity, DamageSeverity, str, bool]:
        """
        Tier 2 (Edge Fallback): Execute local 4-bit quantized Apple Silicon inference via
        `mlx-vlm` (`models/Llama-3.2-11B-Vision-Instruct-4bit`), enforcing strict unified
        memory cleanup (`del model, processor`, `gc.collect()`, `mx.clear_cache()`) in a
        `finally` block.
        """
        if not self.local_mlx_model_path.exists():
            raise FileNotFoundError(
                f"Local MLX VLM model directory not found: {self.local_mlx_model_path}"
            )

        prompt = self._build_vlm_prompt(telemetry)
        model = None
        processor = None
        memory_cleared = False

        try:
            if self._custom_mlx_runner is not None:
                raw_output = self._custom_mlx_runner(prompt, image_source, self.local_mlx_model_path)
            else:
                from mlx_vlm import generate, load
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm.utils import load_config

                config = load_config(str(self.local_mlx_model_path))
                model, processor = load(str(self.local_mlx_model_path))
                formatted_prompt = apply_chat_template(
                    processor,
                    config,
                    prompt,
                    num_images=1 if image_source is not None else 0,
                )
                images_arg = [str(image_source)] if isinstance(image_source, (str, Path)) else None
                gen_res = generate(
                    model,
                    processor,
                    formatted_prompt,
                    image=images_arg,
                    max_tokens=160,
                    temp=0.0,
                    verbose=False,
                )
                raw_output = gen_res.text if hasattr(gen_res, "text") else str(gen_res)

            b_sev, r_sev, cue = parse_vlm_json_response(raw_output, telemetry)
        finally:
            del model
            del processor
            mem_info = release_unified_memory()
            memory_cleared = bool(mem_info.get("cleared", False))

        return b_sev, r_sev, cue, memory_cleared

    def assess_damage(
        self,
        telemetry: GeometricTelemetry,
        image_source: str | Path | Image.Image | bytes | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> DamageAssessmentResult:
        """
        Execute the 3-tier connectivity circuit breaker synchronously:
        Tier 1 (Cloud Meta API, 2.5s timeout) -> Tier 2 (Local MLX VLM) -> Tier 3 (Deterministic).
        """
        t0 = time.perf_counter()
        fallback_chain: list[str] = []
        tier_used = ConnectivityTier.TIER_3_DETERMINISTIC
        memory_cleared = False
        b_sev: DamageSeverity | None = None
        r_sev: DamageSeverity | None = None
        cue: str = ""

        # Step 1: Attempt Tier 1 (Cloud Primary) unless forced to MLX or Deterministic
        if self.mode in (FailoverMode.AUTO, FailoverMode.FORCE_CLOUD):
            try:
                b_sev, r_sev, cue = self._call_tier1_cloud_sync(
                    telemetry, image_source=image_source, http_client=http_client
                )
                tier_used = ConnectivityTier.TIER_1_CLOUD
                fallback_chain.append("tier_1_cloud:success")
            except Exception as exc:
                fallback_chain.append(f"tier_1_cloud:error:{type(exc).__name__}")

        # Step 2: Attempt Tier 2 (Local Apple Silicon MLX VLM) if Tier 1 did not succeed
        if b_sev is None and self.mode in (
            FailoverMode.AUTO,
            FailoverMode.FORCE_CLOUD,
            FailoverMode.FORCE_MLX,
        ):
            try:
                img_for_mlx = image_source if not isinstance(image_source, bytes) else None
                b_sev, r_sev, cue, memory_cleared = self._call_tier2_mlx(
                    telemetry, image_source=img_for_mlx
                )
                tier_used = ConnectivityTier.TIER_2_MLX
                fallback_chain.append("tier_2_mlx:success")
            except Exception as exc:
                fallback_chain.append(f"tier_2_mlx:error:{type(exc).__name__}")
                mem_info = release_unified_memory()
                memory_cleared = bool(mem_info.get("cleared", False))

        # Step 3: Tier 3 Deterministic Geometric Fallback
        if b_sev is None or r_sev is None:
            b_sev, r_sev, cue = evaluate_deterministic_tier3(telemetry)
            tier_used = ConnectivityTier.TIER_3_DETERMINISTIC
            fallback_chain.append("tier_3_deterministic:success")

        return self._finalize_result(
            telemetry=telemetry,
            building_severity=b_sev,
            road_severity=r_sev,
            visual_cue_summary=cue,
            tier_used=tier_used,
            fallback_chain=fallback_chain,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            memory_cleared=memory_cleared,
        )

    async def assess_damage_async(
        self,
        telemetry: GeometricTelemetry,
        image_source: str | Path | Image.Image | bytes | None = None,
        *,
        async_client: httpx.AsyncClient | None = None,
    ) -> DamageAssessmentResult:
        """
        Execute the 3-tier connectivity circuit breaker asynchronously:
        Tier 1 (Cloud Meta API, 2.5s timeout) -> Tier 2 (Local MLX VLM) -> Tier 3 (Deterministic).
        """
        t0 = time.perf_counter()
        fallback_chain: list[str] = []
        tier_used = ConnectivityTier.TIER_3_DETERMINISTIC
        memory_cleared = False
        b_sev: DamageSeverity | None = None
        r_sev: DamageSeverity | None = None
        cue: str = ""

        if self.mode in (FailoverMode.AUTO, FailoverMode.FORCE_CLOUD):
            try:
                b_sev, r_sev, cue = await self._call_tier1_cloud_async(
                    telemetry, image_source=image_source, async_client=async_client
                )
                tier_used = ConnectivityTier.TIER_1_CLOUD
                fallback_chain.append("tier_1_cloud:success")
            except Exception as exc:
                fallback_chain.append(f"tier_1_cloud:error:{type(exc).__name__}")

        if b_sev is None and self.mode in (
            FailoverMode.AUTO,
            FailoverMode.FORCE_CLOUD,
            FailoverMode.FORCE_MLX,
        ):
            try:
                img_for_mlx = image_source if not isinstance(image_source, bytes) else None
                b_sev, r_sev, cue, memory_cleared = self._call_tier2_mlx(
                    telemetry, image_source=img_for_mlx
                )
                tier_used = ConnectivityTier.TIER_2_MLX
                fallback_chain.append("tier_2_mlx:success")
            except Exception as exc:
                fallback_chain.append(f"tier_2_mlx:error:{type(exc).__name__}")
                mem_info = release_unified_memory()
                memory_cleared = bool(mem_info.get("cleared", False))

        if b_sev is None or r_sev is None:
            b_sev, r_sev, cue = evaluate_deterministic_tier3(telemetry)
            tier_used = ConnectivityTier.TIER_3_DETERMINISTIC
            fallback_chain.append("tier_3_deterministic:success")

        return self._finalize_result(
            telemetry=telemetry,
            building_severity=b_sev,
            road_severity=r_sev,
            visual_cue_summary=cue,
            tier_used=tier_used,
            fallback_chain=fallback_chain,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            memory_cleared=memory_cleared,
        )

    def _finalize_result(
        self,
        *,
        telemetry: GeometricTelemetry,
        building_severity: DamageSeverity,
        road_severity: DamageSeverity,
        visual_cue_summary: str,
        tier_used: ConnectivityTier,
        fallback_chain: list[str],
        latency_ms: float,
        memory_cleared: bool,
    ) -> DamageAssessmentResult:
        """Assemble the validated `DamageAssessmentResult` schema."""
        self.last_tier_used = tier_used
        self.last_fallback_chain = list(fallback_chain)

        passability_flags = compute_passability_flags(
            min_clear_width_m=telemetry.min_clear_width_m,
            has_water_obstruction=telemetry.has_water_obstruction,
        )
        rationale = _build_grounding_rationale(
            telemetry=telemetry,
            passability=passability_flags,
            building_severity=building_severity,
            road_severity=road_severity,
            visual_cue_summary=visual_cue_summary,
        )
        sev_score = SEVERITY_TO_SCORE[building_severity]
        return DamageAssessmentResult(
            target_id=telemetry.target_id,
            building_damage_severity=building_severity,
            road_damage_severity=road_severity,
            severity_score=sev_score,
            is_high_severity=sev_score >= 2,
            passability=passability_flags.to_dict(),
            min_clear_width_m=round(telemetry.min_clear_width_m, 3),
            debris_area_m2=round(telemetry.debris_area_m2, 3),
            water_area_m2=round(telemetry.water_area_m2, 3),
            has_water_obstruction=telemetry.has_water_obstruction,
            gsd_m_per_px=round(telemetry.gsd_m_per_px, 4),
            passability_rationale=rationale,
            visual_cue_summary=visual_cue_summary,
            tier_used=tier_used,
            fallback_chain=fallback_chain,
            latency_ms=round(latency_ms, 3),
            memory_cleared=memory_cleared,
        )
