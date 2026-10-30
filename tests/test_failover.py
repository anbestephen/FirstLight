"""Tests for corridor damage grading through the three tiers (`src/failover.py`)."""

from __future__ import annotations

import io
import json
from pathlib import Path

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
    MetaModelSuite,
    VLMFailoverRouter,
    evaluate_deterministic_tier3,
    normalize_damage_severity,
    parse_vlm_json_response,
    release_unified_memory,
)


def _road(**overrides) -> GeometricTelemetry:
    """2.65 m clear: an ambulance fits, a fire engine (3.5 m) does not."""
    base = dict(
        target_id="edge-42",
        min_clear_width_m=2.65,
        debris_area_m2=28.4,
        building_debris_overlap_ratio=0.35,
        gsd_m_per_px=0.2804,
        road_name="US Highway 98 E",
    )
    return GeometricTelemetry(**{**base, **overrides})


def _grade_json(building: str, road: str, cue: str = "Roof decking gone; debris on the north lane.") -> str:
    return json.dumps({"building_damage_severity": building, "road_damage_severity": road, "visual_cue_summary": cue})


def _cloud(reply: str, seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return httpx.MockTransport(handle)


def _raising(exc_type: type[Exception]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exc_type("simulated", request=request)

    return httpx.MockTransport(handle)


@pytest.fixture()
def local_model_dir(tmp_path: Path) -> Path:
    path = tmp_path / "Muse-Glimmer-30B-4bit"
    path.mkdir()
    return path


def test_cloud_tier_sends_versioned_model_and_grounds_the_rationale() -> None:
    seen: list[httpx.Request] = []
    with httpx.Client(transport=_cloud(_grade_json("major_damage", "minor_damage"), seen)) as client:
        router = VLMFailoverRouter(api_key="key-123", cloud_timeout_s=DEFAULT_CLOUD_TIMEOUT_S)
        result = router.assess_damage(_road(), image_source=Image.new("RGB", (32, 32)), http_client=client)

    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == f"{META_MODEL_API_BASE_URL}/chat/completions"
    assert request.headers["Authorization"] == "Bearer key-123"
    assert body["model"] == "muse-spark-1.3" and body["max_tokens"] == 2048
    assert body["messages"][0]["content"][1]["type"] == "image_url"
    assert router.cloud_timeout_s == 15.0

    assert result.tier_used == ConnectivityTier.TIER_1_CLOUD
    assert result.fallback_chain == ["tier_1_cloud:success"]
    assert (result.building_damage_severity, result.road_damage_severity) == ("major_damage", "minor_damage")
    assert result.is_high_severity and result.severity_score == 2
    assert result.visual_cue_summary.startswith("[MUSE-SPARK-1.3 CLOUD VLM]")
    # Passability is computed from the measurement, not taken from the model.
    assert result.passability == {"atv": True, "ambulance": True, "fire_engine": False}
    for measured in ("2.65m", "0.2804 m/px GSD", "28.40 m²"):
        assert measured in result.passability_rationale


def test_cloud_timeout_falls_back_to_the_laptop_model(local_model_dir: Path) -> None:
    prompts: list[str] = []

    def laptop(prompt: str, image, model_path: Path) -> str:
        prompts.append(prompt)
        return _grade_json("destroyed", "minor_damage", "Complete collapse beside the road.")

    with httpx.Client(transport=_raising(httpx.ReadTimeout)) as client:
        router = VLMFailoverRouter(api_key="k", local_mlx_model_path=local_model_dir, mlx_runner=laptop)
        result = router.assess_damage(_road(), image_source=Image.new("RGB", (32, 32)), http_client=client)

    assert result.tier_used == ConnectivityTier.TIER_2_MLX
    assert result.fallback_chain == ["tier_1_cloud:error:ReadTimeout", "tier_2_mlx:success"]
    assert result.memory_cleared is True
    assert result.building_damage_severity == "destroyed" and result.severity_score == 3
    assert "US Highway 98 E" in prompts[0] and "2.65" in prompts[0]


def test_both_models_failing_leaves_the_rules_tier(local_model_dir: Path) -> None:
    def broken(prompt: str, image, model_path: Path) -> str:
        raise RuntimeError("MLX kernel failure")

    with httpx.Client(transport=_raising(httpx.ConnectError)) as client:
        router = VLMFailoverRouter(api_key="k", local_mlx_model_path=local_model_dir, mlx_runner=broken)
        result = router.assess_damage(_road(), http_client=client)

    assert result.tier_used == ConnectivityTier.TIER_3_DETERMINISTIC
    assert result.fallback_chain == [
        "tier_1_cloud:error:ConnectError",
        "tier_2_mlx:error:RuntimeError",
        "tier_3_deterministic:success",
    ]
    assert result.memory_cleared is True
    # 35% of the building under debris -> major; 2.65 m clear -> minor road damage.
    assert (result.building_damage_severity, result.road_damage_severity) == ("major_damage", "minor_damage")
    assert result.visual_cue_summary.startswith("[DETERMINISTIC SAM/GEOMETRY FALLBACK]")


async def test_async_grading_follows_the_same_tiers(local_model_dir: Path) -> None:
    def laptop(prompt: str, image, model_path: Path) -> str:
        return _grade_json("minor_damage", "minor_damage", "Light shingle debris on the shoulder.")

    async with httpx.AsyncClient(transport=_raising(httpx.ConnectTimeout)) as client:
        router = VLMFailoverRouter(api_key="k", local_mlx_model_path=local_model_dir, mlx_runner=laptop)
        result = await router.assess_damage_async(_road(), async_client=client)

    assert result.tier_used == ConnectivityTier.TIER_2_MLX
    assert result.fallback_chain == ["tier_1_cloud:error:ConnectTimeout", "tier_2_mlx:success"]
    assert result.is_high_severity is False


async def test_missing_model_is_reported_and_rules_never_claim_a_model_ran() -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "model_not_found"}}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(not_found)) as client:
        router = VLMFailoverRouter(api_key="k", local_mlx_model_path="models/no-such-weights")
        result = await router.assess_damage_async(_road(debris_area_m2=24.0), async_client=client)

    assert result.fallback_chain[0] == "tier_1_cloud:http_404:model_not_found"
    assert result.fallback_chain[1] == "tier_2_mlx:error:FileNotFoundError"
    assert result.tier_used == ConnectivityTier.TIER_3_DETERMINISTIC
    assert "[DETERMINISTIC SAM/GEOMETRY FALLBACK]" in result.passability_rationale
    assert "MLX" not in result.passability_rationale


def test_rules_only_mode_never_runs_a_model_even_in_a_blackout(local_model_dir: Path) -> None:
    calls: list[str] = []

    def laptop(prompt: str, image, model_path: Path) -> str:
        calls.append(prompt)
        return _grade_json("destroyed", "destroyed")

    router = VLMFailoverRouter(
        api_key="k", local_mlx_model_path=local_model_dir, mlx_runner=laptop, mode=FailoverMode.FORCE_DETERMINISTIC
    )
    result = router.assess_damage(_road(), force_offline=True)
    assert calls == []
    assert result.tier_used == ConnectivityTier.TIER_3_DETERMINISTIC
    assert result.fallback_chain == ["tier_1_cloud:bypassed:force_offline", "tier_3_deterministic:success"]


def test_standing_water_closes_the_road_for_every_vehicle() -> None:
    flooded = GeometricTelemetry(
        target_id="canal-pkwy", min_clear_width_m=0.0, debris_area_m2=5.0, water_area_m2=64.0,
        has_water_obstruction=True, road_name="Canal Parkway",
    )
    result = VLMFailoverRouter(mode=FailoverMode.FORCE_DETERMINISTIC).assess_damage(flooded)
    assert result.passability == {"atv": False, "ambulance": False, "fire_engine": False}
    assert "IMPASSABLE" in result.passability_rationale and "64.00 m²" in result.passability_rationale


@pytest.mark.parametrize(
    ("cover", "width_m", "water", "debris_m2", "building", "road"),
    [
        (0.0, 6.0, False, 0.0, "no_damage", "no_damage"),
        (0.08, 3.0, False, 8.5, "minor_damage", "minor_damage"),
        (0.30, 1.8, False, 22.0, "major_damage", "major_damage"),
        (0.65, 0.7, False, 48.0, "destroyed", "destroyed"),
        (0.10, 5.5, True, 40.0, "destroyed", "destroyed"),
    ],
)
def test_rules_tier_grades(cover: float, width_m: float, water: bool, debris_m2: float, building: str, road: str) -> None:
    t = GeometricTelemetry(
        min_clear_width_m=width_m, debris_area_m2=debris_m2, water_area_m2=15.0 if water else 0.0,
        has_water_obstruction=water, building_debris_overlap_ratio=cover,
    )
    b, r, cue = evaluate_deterministic_tier3(t)
    assert (b, r) == (building, road)
    assert cue.startswith("[DETERMINISTIC SAM/GEOMETRY FALLBACK]")


def test_model_replies_are_parsed_from_json_or_prose() -> None:
    t = _road()
    wrapped = "Here you go:\n```json\n" + _grade_json("destroyed", "major_damage", "Slab only.") + "\n```"
    assert parse_vlm_json_response(wrapped, t, tier_prefix="[X]") == ("destroyed", "major_damage", "[X] Slab only.")
    # A missing field takes the geometry-only value instead of a guess.
    b, r, _ = parse_vlm_json_response('{"building_damage_severity": "no damage"}', t)
    assert (b, r) == ("no_damage", evaluate_deterministic_tier3(t)[1])
    b, r, cue = parse_vlm_json_response("The house is completely destroyed.", t)
    assert b == r == "destroyed" and cue == "The house is completely destroyed."
    assert parse_vlm_json_response("", t) == evaluate_deterministic_tier3(t)


def test_severity_synonyms_scores_and_memory_release() -> None:
    cases = {
        "catastrophic collapse": "destroyed",
        "Severe Structural Damage": "major_damage",
        "significant": "major_damage",
        "moderate": "minor_damage",
        "partial roof loss": "minor_damage",
        "intact": "no_damage",
        "undamaged": "no_damage",
        3: "destroyed",
        0: "no_damage",
        "unrecognised": "minor_damage",
    }
    for raw, expected in cases.items():
        assert normalize_damage_severity(raw) == expected
    assert set(VALID_DAMAGE_SEVERITIES) == {"no_damage", "minor_damage", "major_damage", "destroyed"}
    stats = release_unified_memory()
    assert stats["cleared"] is True and stats["mlx_active_memory_bytes"] >= 0


async def test_meta_model_suite_offline_briefing_and_schematic() -> None:
    suite = MetaModelSuite(VLMFailoverRouter(api_key="k", local_mlx_model_path="models/Muse-Glimmer-30B-4bit"))
    status = suite.get_models_status(force_offline=False)
    assert status["total_models"] == 6
    assert [m["model_id"] for m in status["models"]] == [
        "sam-3.1", "muse-spark-1.3", "muse-wonder-1.0", "muse-voice-transcribe-1.0", "muse-image-1.0",
        "open-weight-mlx-4bit",
    ]

    recs = [{
        "edge_id": "10-20-0", "road_name": "US Highway 98", "debris_area_sqm": 28.4,
        "high_severity_structures_unlocked": 6, "clearance_roi": 0.2113,
        "current_clear_width_m": 2.1, "restored_clear_width_m": 8.0,
    }]
    briefing = await suite.generate_incident_briefing(recs, vehicle_class="fire_engine", force_offline=True)
    assert briefing.tier_used == ConnectivityTier.TIER_3_DETERMINISTIC
    assert "tier_3_deterministic_briefing" in briefing.fallback_chain
    assert briefing.total_structures_unlockable == 6
    assert "[DETERMINISTIC GEOMETRIC TRIAGE — NO LLM EXECUTED]" in briefing.executive_summary

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(40, 60, 80)).save(buf, format="PNG")
    png = suite.generate_recon_schematic_png(
        buf.getvalue(), edge_id="10-20-0", road_name="US Highway 98", min_clear_width_m=2.1,
        nominal_width_m=8.0, debris_area_sqm=28.4, force_offline=True,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and Image.open(io.BytesIO(png)).size == (256, 256)
    assert "tier_3_deterministic_caliper_overlay" in suite.last_schematic_meta["fallback_chain"]


def test_local_prompt_uses_low_reasoning_and_keeps_only_the_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_utils = pytest.importorskip("mlx_vlm.prompt_utils")
    from src.failover import format_local_prompt, local_final_answer

    monkeypatch.setattr(
        prompt_utils,
        "apply_chat_template",
        lambda *_, **__: "<|start|>system<|message|>Reasoning strength: high.<|eot|><|start|>assistant",
    )
    formatted = format_local_prompt(object(), {}, "grade it", num_images=0)
    assert "Reasoning strength: low." in formatted and "Reasoning strength: high." not in formatted
    assert format_local_prompt(object(), {}, "x", num_images=0, reasoning=False).endswith(" to=user<|message|>")
    raw = 'to=self<|message|>the roof is gone<|eom|><|start|>assistant to=user<|message|>{"answer": "x"}'
    assert local_final_answer(raw) == '{"answer": "x"}'
    assert local_final_answer('{"answer": "y"}') == '{"answer": "y"}'
