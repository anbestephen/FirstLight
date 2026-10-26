"""
Unit and integration tests for Milestone 2: 3-Tier VLM damage grading and
connectivity circuit-breaker router (`src/failover.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
import pytest

from src.failover import (
    DEFAULT_CLOUD_TIMEOUT_S,
    META_MODEL_API_BASE_URL,
    VALID_DAMAGE_SEVERITIES,
    ConnectivityTier,
    FailoverMode,
    GeometricTelemetry,
    VLMFailoverRouter,
    evaluate_deterministic_tier3,
    normalize_damage_severity,
    release_unified_memory,
)


@pytest.fixture()
def sample_telemetry_blocked_fire_engine() -> GeometricTelemetry:
    """Road segment with 2.65m clear span (allows ATV & Ambulance, blocks Fire Engine >3.5m)."""
    return GeometricTelemetry(
        target_id="hwy98-edge-42",
        min_clear_width_m=2.65,
        debris_area_m2=28.4,
        water_area_m2=0.0,
        has_water_obstruction=False,
        building_debris_overlap_ratio=0.35,
        building_water_overlap_ratio=0.0,
        gsd_m_per_px=0.2804,
        road_name="US Highway 98 E",
    )


@pytest.fixture()
def sample_rgb_patch() -> Image.Image:
    """Synthetic 64x64 RGB image patch for VLM encoding tests."""
    return Image.new("RGB", (64, 64), color=(180, 140, 110))


def test_tier1_cloud_meta_api_success_and_timeout_config(
    sample_telemetry_blocked_fire_engine: GeometricTelemetry,
    sample_rgb_patch: Image.Image,
) -> None:
    """
    Verify Tier 1 calls https://api.meta.ai/v1/chat/completions with MODEL_API_KEY
    and a 2.5s timeout, parsing structured severity output accurately.
    """
    captured_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        vlm_payload = {
            "building_damage_severity": "major_damage",
            "road_damage_severity": "minor_damage",
            "visual_cue_summary": "Roof decking peeled back with 28.4 m² timber debris encroaching northern lane.",
        }
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(vlm_payload),
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as client:
        router = VLMFailoverRouter(
            api_key="test-meta-spark-key-123",
            cloud_timeout_s=DEFAULT_CLOUD_TIMEOUT_S,
        )
        assert router.cloud_timeout_s == 2.5

        result = router.assess_damage(
            sample_telemetry_blocked_fire_engine,
            image_source=sample_rgb_patch,
            http_client=client,
        )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert str(req.url) == f"{META_MODEL_API_BASE_URL}/chat/completions"
    assert req.headers["Authorization"] == "Bearer test-meta-spark-key-123"

    assert result.tier_used == ConnectivityTier.TIER_1_CLOUD
    assert result.fallback_chain == ["tier_1_cloud:success"]
    assert result.building_damage_severity == "major_damage"
    assert result.road_damage_severity == "minor_damage"
    assert result.is_high_severity is True
    assert result.passability == {
        "atv": True,
        "ambulance": True,
        "fire_engine": False,
    }
    assert "2.65m" in result.passability_rationale
    assert "0.2804 m/px GSD" in result.passability_rationale
    assert "28.40 m²" in result.passability_rationale


def test_tier1_timeout_automatically_falls_back_to_tier2_mlx(
    sample_telemetry_blocked_fire_engine: GeometricTelemetry,
    sample_rgb_patch: Image.Image,
    tmp_path: Path,
) -> None:
    """
    Mock Tier 1 timeout (httpx.ReadTimeout) and verify automatic fallback to Tier 2 (MLX)
    with explicit unified memory release (`memory_cleared=True`).
    """
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Meta Model API timed out after 2.5s", request=request)

    mlx_called_with: dict[str, Any] = {}

    def mock_mlx_runner(
        prompt: str,
        image_source: str | Path | Image.Image | None,
        model_path: Path,
    ) -> str:
        mlx_called_with["prompt"] = prompt
        mlx_called_with["model_path"] = model_path
        return json.dumps(
            {
                "building_damage_severity": "destroyed",
                "road_damage_severity": "minor_damage",
                "visual_cue_summary": "Complete structural collapse adjacent to US Highway 98 E.",
            }
        )

    # Use either real cached models/Llama-3.2-11B-Vision-Instruct-4bit or tmp_path if running in CI
    mlx_dir = Path("models/Llama-3.2-11B-Vision-Instruct-4bit")
    if not mlx_dir.exists():
        mlx_dir = tmp_path / "Llama-3.2-11B-Vision-Instruct-4bit"
        mlx_dir.mkdir(parents=True)

    transport = httpx.MockTransport(timeout_handler)
    with httpx.Client(transport=transport) as client:
        router = VLMFailoverRouter(
            api_key="test-meta-key",
            local_mlx_model_path=mlx_dir,
            mlx_runner=mock_mlx_runner,
        )
        result = router.assess_damage(
            sample_telemetry_blocked_fire_engine,
            image_source=sample_rgb_patch,
            http_client=client,
        )

    assert result.tier_used == ConnectivityTier.TIER_2_MLX
    assert result.fallback_chain == [
        "tier_1_cloud:error:ReadTimeout",
        "tier_2_mlx:success",
    ]
    assert result.memory_cleared is True
    assert result.building_damage_severity == "destroyed"
    assert result.road_damage_severity == "minor_damage"
    assert result.severity_score == 3
    assert result.is_high_severity is True
    assert "US Highway 98 E" in mlx_called_with["prompt"]
    assert "2.65" in mlx_called_with["prompt"]


def test_tier1_and_tier2_failure_falls_back_to_tier3_deterministic(
    sample_telemetry_blocked_fire_engine: GeometricTelemetry,
) -> None:
    """
    Mock both Tier 1 network failure (ConnectError) and Tier 2 MLX failure (RuntimeError)
    to verify graceful fallback to Tier 3 deterministic geometric heuristics.
    """
    def network_down_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("WAN router unplugged", request=request)

    def failing_mlx_runner(
        prompt: str,
        image_source: str | Path | Image.Image | None,
        model_path: Path,
    ) -> str:
        raise RuntimeError("Simulated MLX kernel failure")

    transport = httpx.MockTransport(network_down_handler)
    with httpx.Client(transport=transport) as client:
        router = VLMFailoverRouter(
            api_key="test-meta-key",
            local_mlx_model_path="models/Llama-3.2-11B-Vision-Instruct-4bit",
            mlx_runner=failing_mlx_runner,
        )
        result = router.assess_damage(
            sample_telemetry_blocked_fire_engine,
            http_client=client,
        )

    assert result.tier_used == ConnectivityTier.TIER_3_DETERMINISTIC
    assert result.fallback_chain == [
        "tier_1_cloud:error:ConnectError",
        "tier_2_mlx:error:RuntimeError",
        "tier_3_deterministic:success",
    ]
    assert result.memory_cleared is True
    assert result.building_damage_severity in VALID_DAMAGE_SEVERITIES
    assert result.road_damage_severity in VALID_DAMAGE_SEVERITIES
    # With building_debris_overlap_ratio=0.35 (>=0.20) and min_clear_width_m=2.65 (<=3.5m):
    assert result.building_damage_severity == "major_damage"
    assert result.road_damage_severity == "minor_damage"
    assert "2.65m" in result.passability_rationale
    assert "28.40 m²" in result.passability_rationale


async def test_async_failover_tier1_to_tier2_and_tier3(
    sample_telemetry_blocked_fire_engine: GeometricTelemetry,
) -> None:
    """Verify async `assess_damage_async` follows the exact 3-tier circuit breaker sequence."""
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connect timeout after 2.5s", request=request)

    def mock_mlx_runner(
        prompt: str,
        image_source: str | Path | Image.Image | None,
        model_path: Path,
    ) -> str:
        return '{"building_damage_severity": "minor_damage", "road_damage_severity": "minor_damage", "visual_cue_summary": "Light shingle debris on shoulder."}'

    transport = httpx.MockTransport(timeout_handler)
    async with httpx.AsyncClient(transport=transport) as async_client:
        router = VLMFailoverRouter(
            api_key="test-meta-key",
            local_mlx_model_path="models/Llama-3.2-11B-Vision-Instruct-4bit",
            mlx_runner=mock_mlx_runner,
        )
        res = await router.assess_damage_async(
            sample_telemetry_blocked_fire_engine,
            async_client=async_client,
        )

    assert res.tier_used == ConnectivityTier.TIER_2_MLX
    assert res.fallback_chain == [
        "tier_1_cloud:error:ConnectTimeout",
        "tier_2_mlx:success",
    ]
    assert res.building_damage_severity == "minor_damage"
    assert res.is_high_severity is False


@pytest.mark.parametrize(
    ("b_overlap", "width_m", "water", "debris_m2", "expected_b", "expected_r"),
    [
        (0.0, 6.0, False, 0.0, "no_damage", "no_damage"),
        (0.08, 3.0, False, 8.5, "minor_damage", "minor_damage"),
        (0.30, 1.8, False, 22.0, "major_damage", "major_damage"),
        (0.65, 0.7, False, 48.0, "destroyed", "destroyed"),
        (0.10, 5.5, True, 40.0, "destroyed", "destroyed"),
    ],
)
def test_deterministic_tier3_severity_rules(
    b_overlap: float,
    width_m: float,
    water: bool,
    debris_m2: float,
    expected_b: str,
    expected_r: str,
) -> None:
    """Verify Tier 3 deterministic geometric damage severity grading across all 4 severity classes."""
    telemetry = GeometricTelemetry(
        target_id="test-seg",
        min_clear_width_m=width_m,
        debris_area_m2=debris_m2,
        water_area_m2=15.0 if water else 0.0,
        has_water_obstruction=water,
        building_debris_overlap_ratio=b_overlap,
    )
    b_sev, r_sev, cue = evaluate_deterministic_tier3(telemetry)
    assert b_sev == expected_b
    assert r_sev == expected_r
    assert len(cue) > 10


def test_water_obstruction_forces_impassable_across_all_tiers() -> None:
    """Verify water obstruction overrides passability in the final assessment result."""
    water_telemetry = GeometricTelemetry(
        target_id="canal-pkwy-flooded",
        min_clear_width_m=0.0,
        debris_area_m2=5.0,
        water_area_m2=64.0,
        has_water_obstruction=True,
        road_name="Canal Parkway",
    )
    router = VLMFailoverRouter(mode=FailoverMode.FORCE_DETERMINISTIC)
    result = router.assess_damage(water_telemetry)

    assert result.tier_used == ConnectivityTier.TIER_3_DETERMINISTIC
    assert result.passability == {"atv": False, "ambulance": False, "fire_engine": False}
    assert "IMPASSABLE" in result.passability_rationale
    assert "64.00 m²" in result.passability_rationale


def test_normalize_damage_severity_and_memory_hygiene() -> None:
    """Verify severity label normalization and Apple Silicon unified memory release helper."""
    assert normalize_damage_severity("catastrophic collapse") == "destroyed"
    assert normalize_damage_severity("Severe Structural Damage") == "major_damage"
    assert normalize_damage_severity("partial roof loss") == "minor_damage"
    assert normalize_damage_severity("intact") == "no_damage"
    assert normalize_damage_severity(3) == "destroyed"
    assert normalize_damage_severity(0) == "no_damage"

    mem_stats = release_unified_memory()
    assert mem_stats["cleared"] is True
    assert mem_stats["mlx_active_memory_bytes"] >= 0
