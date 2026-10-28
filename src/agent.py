"""
FirstLight commander agent (`src/agent.py`).

The commander asks in plain language ("get an ambulance to the worst-hit block"). A model
plans which deterministic tools to call (`src/agent_tools.py`), the tools compute routes and
priorities from SAM 3.1 measurements and Muse Spark grades, and the model writes a short
answer. Then every number in the answer is checked against the tool results; numbers that
cannot be traced are reported (and the model gets one chance to fix them).

Three tiers, mirroring the grading failover:
1. Muse Spark 1.3 via the Meta Model API with native tool calling.
2. Muse Glimmer 30B 4-bit running locally on MLX, same tools via a JSON action protocol.
3. No model: a keyword intent parser calls the same tools and fills an answer template.

`CommanderAgent.run()` yields events (tier, tool_call, tool_result, answer) so the UI can
show each step as it happens.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Callable

import httpx

from src.agent_tools import TOOL_SCHEMAS, VEHICLES, AgentTools, numbers_in
from src.failover import LOCAL_MAX_TOKENS, LOCAL_MODEL_LABEL, format_local_prompt, local_final_answer

MAX_STEPS = 6
SYSTEM_PROMPT = (
    "You are FirstLight, the routing assistant for an emergency incident commander in Mexico Beach, "
    "Florida, the day after Hurricane Michael. Road widths come from SAM 3.1 measurements of aerial "
    "imagery; building damage comes from Muse Spark grades. Rules:\n"
    "1. Always call tools to get facts. Never guess widths, distances, times or counts.\n"
    "2. Every number you write must appear in a tool result. Use the provided units "
    "(distance_km, eta_min, metres) instead of converting.\n"
    "3. Vehicles need a clear width over 1.2 m (ATV), 2.2 m (ambulance) or 3.5 m (fire engine).\n"
    "4. If a route is blocked, say which vehicle can get through and which debris pile to clear.\n"
    "5. highest_impact_clearance_for_this_vehicle is the best clearance anywhere for that vehicle; "
    "never claim it opens the route that was asked about.\n"
    "6. Refer to roads by name and to hotspots by their nearby road (\"the worst-hit block near 16th "
    "Street\"); never mention edge ids or node ids. Write areas as m².\n"
    "7. Only dispatch a field unit when the commander asks to send, dispatch or get a vehicle somewhere.\n"
    "8. Answer in at most 4 short sentences a responder can act on, including the key numbers "
    "(clear width, km, minutes, building counts)."
)


class CloudDeclined(RuntimeError):
    """Meta Model API refused the request (e.g. a content-filter false positive)."""
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


def verify_numbers(answer: str, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that every number in `answer` appears in some tool result (allowing rounding)."""
    known = numbers_in(tool_results)
    found = []
    for raw in _NUMBER_RE.findall(answer):
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        ok = any(
            abs(value - k) <= 1e-9
            or round(k, decimals) == value
            or (decimals == 0 and abs(value - k) < 0.5)
            for k in known
        )
        found.append({"value": raw, "verified": ok})
    unverified = [f["value"] for f in found if not f["verified"]]
    return {"total": len(found), "verified": len(found) - len(unverified), "unverified": unverified}


class CommanderAgent:
    def __init__(
        self,
        tools: AgentTools,
        *,
        api_key: str | None,
        api_base_url: str,
        model: str = "muse-spark-1.3",
        reasoning_effort: str | None = "low",
        uplink_online: Callable[[], bool | None] | None = None,
        local_runtime: Any = None,  # src.failover.ResidentMLXRuntime
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.tools = tools
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.uplink_online = uplink_online
        self.local_runtime = local_runtime
        self.timeout_s = timeout_s
        self._transport = transport

    def _tiers(self, force_offline: bool) -> list[str]:
        tiers = []
        if self.api_key and not force_offline and not (self.uplink_online and self.uplink_online() is False):
            tiers.append("tier_1_cloud")
        if self.local_runtime is not None:
            tiers.append("tier_2_edge")
        tiers.append("tier_3_deterministic")
        return tiers

    async def run(self, question: str, *, force_offline: bool = False) -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        for tier in self._tiers(force_offline):
            yield {"type": "tier", "tier": tier}
            runner = {
                "tier_1_cloud": self._run_cloud,
                "tier_2_edge": self._run_edge,
                "tier_3_deterministic": self._run_deterministic,
            }[tier]
            try:
                async for event in runner(question):
                    if event["type"] == "answer":
                        event["tier"] = tier
                        event["latency_ms"] = round((time.perf_counter() - t0) * 1000.0)
                    yield event
                return
            except CloudDeclined as exc:
                yield {"type": "fallback", "tier": tier, "reason": str(exc)}
            except Exception as exc:  # noqa: BLE001 - any tier failure falls through to the next
                yield {"type": "fallback", "tier": tier, "reason": f"{type(exc).__name__}: {exc}"[:200]}

    async def _call_tool(self, name: str, args: dict[str, Any], results: list[dict[str, Any]]):
        yield {"type": "tool_call", "name": name, "args": args}
        result = await asyncio.to_thread(self.tools.call, name, args)
        results.append(result)
        yield {"type": "tool_result", "name": name, "args": args, "result": result}

    # ---- Tier 1: Muse Spark native tool calling --------------------------------------------
    async def _chat(self, client: httpx.AsyncClient, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        resp = await client.post(
            f"{self.api_base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
        )
        if resp.status_code == 400 and "content_policy_violation" in resp.text:
            raise CloudDeclined("Meta Cloud's content filter declined this wording")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    async def _run_cloud(self, question: str) -> AsyncIterator[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        results: list[dict[str, Any]] = []
        retried = False
        async with httpx.AsyncClient(transport=self._transport) as client:
            for _ in range(MAX_STEPS):
                msg = await self._chat(client, messages)
                calls = msg.get("tool_calls") or []
                if calls:
                    messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                    for call in calls:
                        name = call["function"]["name"]
                        try:
                            args = json.loads(call["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        async for ev in self._call_tool(name, args, results):
                            yield ev
                        messages.append(
                            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(results[-1])}
                        )
                    continue
                answer = (msg.get("content") or "").strip()
                check = verify_numbers(answer, results)
                if check["unverified"] and not retried:
                    retried = True
                    messages.append({"role": "assistant", "content": answer})
                    messages.append(
                        {
                            "role": "user",
                            "content": "These numbers are not in any tool result: "
                            + ", ".join(check["unverified"])
                            + ". Rewrite the answer using only numbers from the tool results.",
                        }
                    )
                    yield {"type": "revision", "unverified": check["unverified"]}
                    continue
                yield {"type": "answer", "text": answer, "verification": check, "model": self.model}
                return
        raise RuntimeError("Muse Spark did not finish within the step limit")

    # ---- Tier 2: local Muse Glimmer (MLX), JSON action protocol ------------------------
    def _edge_prompt(self, question: str, transcript: list[str]) -> str:
        tools = "\n".join(
            f"- {t['function']['name']}({', '.join(t['function']['parameters'].get('properties', {}))}): "
            f"{t['function']['description']}"
            for t in TOOL_SCHEMAS
        )
        return (
            f"{SYSTEM_PROMPT}\n\nTools:\n{tools}\n\n"
            "Reply with exactly ONE JSON object and nothing else: either "
            '{"tool": "<name>", "args": {...}} to call a tool, or {"answer": "<text>"} when done.\n\n'
            f"Commander: {question}\n" + "\n".join(transcript)
        )

    def _generate_local(self, prompt: str) -> str:
        def fn(model: Any, processor: Any, config: dict[str, Any]) -> str:
            from mlx_vlm import generate

            formatted = format_local_prompt(processor, config, prompt, num_images=0, reasoning=False)
            out = generate(model, processor, formatted, max_tokens=LOCAL_MAX_TOKENS, temp=0.0, verbose=False)
            return local_final_answer(out.text if hasattr(out, "text") else str(out))

        return self.local_runtime.run(fn)

    async def _run_edge(self, question: str) -> AsyncIterator[dict[str, Any]]:
        transcript: list[str] = []
        results: list[dict[str, Any]] = []
        for _ in range(MAX_STEPS):
            raw = await asyncio.to_thread(self._generate_local, self._edge_prompt(question, transcript))
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("local model did not return JSON")
            action = json.loads(raw[start : end + 1])
            if "answer" in action:
                answer = str(action["answer"]).strip()
                yield {
                    "type": "answer",
                    "text": answer,
                    "verification": verify_numbers(answer, results),
                    "model": LOCAL_MODEL_LABEL,
                }
                return
            name, args = str(action.get("tool", "")), dict(action.get("args") or {})
            async for ev in self._call_tool(name, args, results):
                yield ev
            transcript.append(f"Tool {name}({json.dumps(args)}) returned: {json.dumps(results[-1])}")
        raise RuntimeError("local model did not finish within the step limit")

    # ---- Tier 3: no model -----------------------------------------------------------------
    @staticmethod
    def _vehicle_in(question: str) -> str | None:
        q = question.lower()
        if "fire" in q or "engine" in q or "truck" in q:
            return "fire_engine"
        if "ambulance" in q or "medic" in q:
            return "ambulance"
        if "atv" in q or "foot" in q:
            return "atv"
        return None

    async def _run_deterministic(self, question: str) -> AsyncIterator[dict[str, Any]]:
        q = question.lower()
        vc = self._vehicle_in(question) or "ambulance"
        results: list[dict[str, Any]] = []
        road = None
        for name in self.tools.road_names():
            if name.lower() in q:
                road = name
                break

        if any(w in q for w in ("clear", "bulldoz", "pile", "priority", "first")):
            async for ev in self._call_tool("rank_clearance", {"vehicle_class": vc, "limit": 3}, results):
                yield ev
            top = results[-1]["choke_points"]
            text = (
                f"Clear {top[0]['road']} first: {top[0]['debris_m2']} m² of debris; it unlocks "
                f"{top[0]['high_severity_buildings_unlocked']} severely damaged buildings for {results[-1]['vehicle']}."
                if top
                else f"No debris clearance unlocks more buildings for {results[-1]['vehicle']}."
            )
        elif road and not any(w in q for w in ("send", "dispatch", "get", "route")):
            async for ev in self._call_tool("measure_road", {"road_name": road}, results):
                yield ev
            r = results[-1]
            ok = [v for v in VEHICLES if r["passable_end_to_end"][v]]
            text = (
                f"{r['road']}: narrowest clear width {r['narrowest_clear_width_m']} m. "
                + (f"Passable end to end for {', '.join(ok)}." if ok else "Not passable end to end for any vehicle.")
            )
        else:
            destination = road
            if destination is None:
                async for ev in self._call_tool("damage_hotspots", {"limit": 1}, results):
                    yield ev
                destination = "hotspot-1"
            async for ev in self._call_tool("plan_route", {"vehicle_class": vc, "destination": destination}, results):
                yield ev
            r = results[-1]
            if r["reachable"]:
                text = f"{r['vehicle']} can reach {r['destination']}: {r['distance_km']} km, about {r['eta_min']} min."
                if any(w in q for w in ("send", "dispatch", "get")):
                    async for ev in self._call_tool(
                        "dispatch_field_unit", {"vehicle_class": vc, "destination": destination}, results
                    ):
                        yield ev
                    text += f" Dispatched to {results[-1]['field_devices_notified']} field device(s)."
            else:
                alt = [v for v, ok in r.get("other_vehicles_reachable", {}).items() if ok]
                best = r.get("highest_impact_clearance_for_this_vehicle")
                text = f"{r['vehicle']} cannot reach {r['destination']}."
                if alt:
                    text += f" {', '.join(alt)} can get through."
                if best:
                    text += (
                        f" Highest-impact clearance for this vehicle: {best['debris_m2']} m² on {best['road']}, "
                        f"unlocking {best['high_severity_buildings_unlocked']} severely damaged buildings."
                    )
        yield {
            "type": "answer",
            "text": text,
            "verification": verify_numbers(text, results),
            "model": "none (deterministic tools + template)",
        }
