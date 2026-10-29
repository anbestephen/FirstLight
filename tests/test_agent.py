"""
Tests for the commander agent (`src/agent.py`, `src/agent_tools.py`): number verification,
the Muse Spark tool loop (mocked HTTP), content-filter fallback, and the no-model tier.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.agent import CommanderAgent, verify_numbers
from src.app import BUILDINGS_GRADED_GEOJSON_PATH, state
from src.agent_tools import AgentContext, AgentTools


@pytest.fixture(scope="module")
def tools() -> AgentTools:
    state.ensure_loaded()
    return AgentTools(
        AgentContext(
            graph=lambda: state.active_graph,
            buildings=lambda: state.get_buildings_geojson()["features"],
            command_post_lonlat=lambda: state.default_route_endpoints()[0],
            buildings_source=BUILDINGS_GRADED_GEOJSON_PATH,
            dispatch=None,
        )
    )


async def _collect(agent: CommanderAgent, question: str, **kw) -> list[dict]:
    return [ev async for ev in agent.run(question, **kw)]


def test_verify_numbers_traces_rounded_values_and_flags_invented_ones() -> None:
    results = [{"distance_km": 1.2, "eta_min": 3.4, "road": "15th Street", "width": 2.75}]
    ok = verify_numbers("1.2 km on 15th Street, about 3.4 min, 2.8 m clear", results)
    assert ok["unverified"] == [] and ok["total"] == 4
    bad = verify_numbers("about 7 minutes", results)
    assert bad["unverified"] == ["7"]


def test_road_matching_handles_abbreviations(tools: AgentTools) -> None:
    assert tools.match_road("15th st") == "15th Street"
    assert tools.measure_road("no such road")["error"].startswith("No road")


def _cloud(responses: list[dict]) -> tuple[httpx.MockTransport, list[dict]]:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": responses[len(seen) - 1]}]})

    return httpx.MockTransport(handler), seen


def _tool_call(name: str, args: dict) -> dict:
    return {
        "content": "",
        "tool_calls": [{"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }


async def test_cloud_tier_calls_tools_and_verifies_answer(tools: AgentTools) -> None:
    width = tools.measure_road("15th Street")["narrowest_clear_width_m"]
    transport, seen = _cloud(
        [
            _tool_call("measure_road", {"road_name": "15th Street"}),
            {"content": f"15th Street's narrowest clear width is {width} m."},
        ]
    )
    agent = CommanderAgent(tools, api_key="k", api_base_url="https://api.meta.ai/v1", transport=transport)
    events = await _collect(agent, "Is 15th Street passable?")

    assert [e["type"] for e in events] == ["tier", "tool_call", "tool_result", "answer"]
    answer = events[-1]
    assert answer["tier"] == "tier_1_cloud"
    assert answer["verification"]["unverified"] == []
    # The tool result was sent back to Muse Spark as a `tool` message.
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert "tools" in seen[0] and seen[0]["model"] == "muse-spark-1.3"


async def test_cloud_tier_gets_one_chance_to_fix_untraceable_numbers(tools: AgentTools) -> None:
    transport, seen = _cloud(
        [
            _tool_call("reachability_summary", {"vehicle_class": "ambulance"}),
            {"content": "An ambulance reaches 999 buildings."},
            {"content": "An ambulance cannot be summarised without the tool numbers."},
        ]
    )
    agent = CommanderAgent(tools, api_key="k", api_base_url="https://api.meta.ai/v1", transport=transport)
    events = await _collect(agent, "How many damaged buildings can an ambulance reach?")
    assert any(e["type"] == "revision" and e["unverified"] == ["999"] for e in events)
    assert "999" in seen[2]["messages"][-1]["content"]
    assert events[-1]["verification"]["unverified"] == []


async def test_content_filter_decline_falls_back_without_a_model(tools: AgentTools) -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(400, json={"error": {"code": "content_policy_violation"}})
    )
    agent = CommanderAgent(tools, api_key="k", api_base_url="https://api.meta.ai/v1", transport=transport)
    events = await _collect(agent, "Which pile should we clear first for fire engines?")
    fallback = next(e for e in events if e["type"] == "fallback")
    assert "content filter" in fallback["reason"]
    assert events[-1]["tier"] == "tier_3_deterministic"
    assert events[-1]["verification"]["unverified"] == []


async def test_offline_deterministic_tier_routes_and_never_invents_numbers(tools: AgentTools) -> None:
    agent = CommanderAgent(tools, api_key="k", api_base_url="https://api.meta.ai/v1")
    for q in ("Get an ambulance to the worst-hit block", "Is 15th Street passable?", "What should we clear first?"):
        events = await _collect(agent, q, force_offline=True)
        assert events[0] == {"type": "tier", "tier": "tier_3_deterministic"}
        assert events[-1]["type"] == "answer"
        assert events[-1]["verification"]["unverified"] == []


def test_dispatch_refuses_destinations_with_no_passable_route(tools: AgentTools) -> None:
    hotspots = tools.damage_hotspots(10)["hotspots"]
    blocked = next(h for h in hotspots if not h["reachable_from_command_post"]["fire_engine"])
    result = tools.dispatch_field_unit("fire_engine", blocked["id"])
    assert result["dispatched"] is False
    assert "nothing was sent" in result["reason"]


def test_unreachable_route_says_whether_the_best_clearance_opens_it(tools: AgentTools) -> None:
    hotspots = tools.damage_hotspots(10)["hotspots"]
    blocked = next(h for h in hotspots if not h["reachable_from_command_post"]["ambulance"])
    result = tools.plan_route("ambulance", blocked["id"])
    assert result["reachable"] is False
    best = result["highest_impact_clearance_for_this_vehicle"]
    assert isinstance(best["opens_this_route"], bool)
