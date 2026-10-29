"""
Tests for live-demo latency hardening in `src/failover.py`: resident Tier 2 MLX weights,
Muse Spark `reasoning_effort`, the Tier 1 response cache, and uplink-down fast failover.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import threading
from pathlib import Path
import time

import httpx
from PIL import Image
import pytest

from src.failover import (
    ConnectivityTier,
    GeometricTelemetry,
    LocalModelPreempted,
    ResidentMLXRuntime,
    VLMFailoverRouter,
)


def _telemetry() -> GeometricTelemetry:
    return GeometricTelemetry(
        target_id="edge-7",
        min_clear_width_m=2.8,
        debris_area_m2=33.4,
        gsd_m_per_px=0.2804,
        road_name="15th Street",
    )


def _cloud_ok(captured: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = {
            "building_damage_severity": "minor_damage",
            "road_damage_severity": "major_damage",
            "visual_cue_summary": "Sand overwash narrows the paved lane.",
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    return httpx.MockTransport(handler)


@pytest.fixture()
def fake_mlx_dir(tmp_path: Path) -> Path:
    (tmp_path / "model.safetensors").write_bytes(b"\0")
    return tmp_path


def test_cloud_payload_sends_reasoning_effort() -> None:
    captured: list[httpx.Request] = []
    router = VLMFailoverRouter(api_key="k", cloud_reasoning_effort="low")
    with httpx.Client(transport=_cloud_ok(captured)) as client:
        router.assess_damage(_telemetry(), http_client=client)
    body = json.loads(captured[0].content)
    assert body["reasoning_effort"] == "low"
    assert body["model"] == "muse-spark-1.3"

    captured.clear()
    router_default = VLMFailoverRouter(api_key="k", cloud_reasoning_effort=None)
    with httpx.Client(transport=_cloud_ok(captured)) as client:
        router_default.assess_damage(_telemetry(), http_client=client)
    assert "reasoning_effort" not in json.loads(captured[0].content)


def test_repeat_assessment_is_served_from_real_cloud_cache() -> None:
    captured: list[httpx.Request] = []
    router = VLMFailoverRouter(api_key="k")
    crop = Image.new("RGB", (32, 32), color=(120, 110, 90))
    with httpx.Client(transport=_cloud_ok(captured)) as client:
        first = router.assess_damage(_telemetry(), image_source=crop, http_client=client)
        second = router.assess_damage(_telemetry(), image_source=crop, http_client=client)

    assert len(captured) == 1
    assert first.cloud_cache_hit is False
    assert second.cloud_cache_hit is True
    assert second.tier_used == ConnectivityTier.TIER_1_CLOUD
    assert second.fallback_chain == ["tier_1_cloud:cache_hit"]
    assert second.road_damage_severity == first.road_damage_severity == "major_damage"

    # A different crop or measurement is a different question: no stale reuse.
    other = GeometricTelemetry(**{**vars(_telemetry()), "min_clear_width_m": 1.1})
    with httpx.Client(transport=_cloud_ok(captured)) as client:
        third = router.assess_damage(other, image_source=crop, http_client=client)
    assert third.cloud_cache_hit is False
    assert len(captured) == 2


def test_prewarm_populates_cache_without_edge_fallback() -> None:
    captured: list[httpx.Request] = []
    router = VLMFailoverRouter(api_key="k")
    transport = _cloud_ok(captured)
    original = router._call_tier1_cloud_sync

    def call_with_mock(telemetry, image_source=None, http_client=None):
        with httpx.Client(transport=transport) as client:
            return original(telemetry, image_source=image_source, http_client=client)

    router._call_tier1_cloud_sync = call_with_mock  # type: ignore[method-assign]
    assert router.prewarm_cloud_assessment(_telemetry()) is True
    assert router.prewarm_cloud_assessment(_telemetry()) is True
    assert len(captured) == 1

    no_key = VLMFailoverRouter(api_key="")
    assert no_key.prewarm_cloud_assessment(_telemetry()) is False


def test_known_uplink_loss_skips_cloud_without_waiting_for_timeout(fake_mlx_dir: Path) -> None:
    def never_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Tier 1 must not be attempted while the uplink is down")

    router = VLMFailoverRouter(
        api_key="k",
        local_mlx_model_path=fake_mlx_dir,
        mlx_runner=lambda prompt, img, path: json.dumps(
            {"building_damage_severity": "major_damage", "road_damage_severity": "major_damage"}
        ),
        uplink_online_hint=lambda: False,
    )
    t0 = time.perf_counter()
    with httpx.Client(transport=httpx.MockTransport(never_called)) as client:
        result = router.assess_damage(_telemetry(), http_client=client)
    # Far below the 15 s cloud timeout (the slack covers first-time torch/mlx imports).
    assert time.perf_counter() - t0 < 5.0
    assert result.tier_used == ConnectivityTier.TIER_2_MLX
    assert result.fallback_chain[0] == "tier_1_cloud:skipped:uplink_down"


def test_resident_runtime_loads_weights_once_across_requests(fake_mlx_dir: Path) -> None:
    loads: list[str] = []

    def fake_loader(path: str):
        loads.append(path)
        return object(), object(), {"model_type": "mllama"}

    runtime = ResidentMLXRuntime(fake_mlx_dir, loader=fake_loader)
    outputs = iter(
        [
            json.dumps({"building_damage_severity": "destroyed", "road_damage_severity": "major_damage"}),
            json.dumps({"building_damage_severity": "minor_damage", "road_damage_severity": "no_damage"}),
        ]
    )
    runtime.run = lambda fn, **_: (runtime.ensure_loaded(), next(outputs))[1]  # type: ignore[method-assign]
    router = VLMFailoverRouter(
        api_key="",
        local_mlx_model_path=fake_mlx_dir,
        mlx_runtime=runtime,
    )

    first = router.assess_damage(_telemetry(), force_offline=True)
    repeat = router.assess_damage(_telemetry(), force_offline=True)
    other = router.assess_damage(replace(_telemetry(), target_id="edge-8", road_name="16th Street"), force_offline=True)

    assert first.tier_used == repeat.tier_used == other.tier_used == ConnectivityTier.TIER_2_MLX
    assert first.building_damage_severity == "destroyed"
    # Same corridor and image: served from the edge cache, no second generation.
    assert repeat.building_damage_severity == "destroyed"
    assert other.building_damage_severity == "minor_damage"
    assert len(loads) == 1
    assert runtime.status() == {"resident": True, "load_seconds": runtime.load_seconds, "load_count": 1}

    runtime.unload()
    assert runtime.is_loaded is False


async def test_async_tier2_runs_off_the_event_loop(fake_mlx_dir: Path) -> None:
    generating = False

    def slow_runner(prompt, img, path):
        nonlocal generating
        generating = True
        time.sleep(0.4)
        generating = False
        return json.dumps({"building_damage_severity": "major_damage", "road_damage_severity": "minor_damage"})

    router = VLMFailoverRouter(api_key="", local_mlx_model_path=fake_mlx_dir, mlx_runner=slow_runner)
    ticks_during_generation = 0

    async def heartbeat() -> None:
        nonlocal ticks_during_generation
        for _ in range(12):
            await asyncio.sleep(0.05)
            if generating:
                ticks_during_generation += 1

    result, _ = await asyncio.gather(
        router.assess_damage_async(_telemetry(), force_offline=True),
        heartbeat(),
    )
    assert result.tier_used == ConnectivityTier.TIER_2_MLX
    # If generation blocked the loop, no heartbeat could run while it was in progress.
    assert ticks_during_generation >= 3


def test_mllama_cross_attention_kv_memo_matches_original_and_skips_reprojection() -> None:
    mx = pytest.importorskip("mlx.core")
    language = pytest.importorskip("mlx_vlm.models.mllama.language")
    from mlx_vlm.models.mllama.config import TextConfig

    from src.mlx_patches import apply_mllama_cross_attention_kv_memo

    config = TextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
    layer = language.MllamaTextCrossAttention(config, layer_idx=3)
    mx.random.seed(0)
    states = mx.random.normal((1, 12, 64))
    tokens = [mx.random.normal((1, 1, 64)) for _ in range(3)]

    original = language.MllamaTextCrossAttention.__call__
    reference = [original(layer, t, cross_attention_states=states) for t in tokens]

    assert apply_mllama_cross_attention_kv_memo() is True
    assert apply_mllama_cross_attention_kv_memo() is True  # idempotent

    projections = 0
    real_k_proj = layer.k_proj

    def counting_k_proj(x):
        nonlocal projections
        projections += 1
        return real_k_proj(x)

    layer.k_proj = counting_k_proj
    patched = [layer(t, cross_attention_states=states) for t in tokens]

    assert projections == 1  # image K/V projected once, reused for every decode step
    for ref, out in zip(reference, patched):
        assert mx.allclose(ref, out, atol=1e-5).item()


def test_commander_question_preempts_a_running_corridor_grade(fake_mlx_dir: Path) -> None:
    runtime = ResidentMLXRuntime(fake_mlx_dir, loader=lambda path: (object(), object(), {}))
    grading = threading.Event()
    outcome: dict[str, object] = {}

    def slow_grade(*_):
        grading.set()
        for _ in range(300):  # ~30 s "generation", polled per token
            if runtime.preempt_requested:
                raise LocalModelPreempted("yielded")
            time.sleep(0.1)
        return "graded"

    def run_grade() -> None:
        try:
            outcome["grade"] = runtime.run(slow_grade, preemptible=True)
        except LocalModelPreempted:
            outcome["grade"] = "preempted"

    worker = threading.Thread(target=run_grade)
    worker.start()
    assert grading.wait(5)
    t0 = time.perf_counter()
    assert runtime.run(lambda *_: "agent step") == "agent step"
    assert time.perf_counter() - t0 < 1.0
    worker.join(5)
    assert outcome["grade"] == "preempted"
    # A later grade runs normally once no question is waiting.
    assert runtime.run(lambda *_: "graded", preemptible=True) == "graded"
