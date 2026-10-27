"""
Deterministic tools for the FirstLight commander agent (`src/agent_tools.py`).

Every number an agent tier may say comes from one of these functions, which only read the
SAM 3.1-measured road graph, the Muse Spark building grades, and graph routing. They never
call a model. Results include pre-rounded human units (km, minutes) so answers can quote
them exactly and the verifier in `src/agent.py` can trace every number back to a tool.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import math
from pathlib import Path
import re
from typing import Any, Callable

import networkx as nx

from src.routing import (
    VEHICLE_LABEL,
    VEHICLE_MIN_WIDTH_M,
    _format_edge_name,
    compute_shortest_route,
    find_nearest_node,
    normalize_vehicle_class,
    rank_choke_point_clearance_roi,
)

VEHICLES = ("atv", "ambulance", "fire_engine")
HIGH_SEVERITY = {"major_damage", "destroyed"}


@dataclass
class AgentContext:
    """Live app state the tools read (injected so tools stay testable without the server)."""

    graph: Callable[[], nx.MultiDiGraph]
    buildings: Callable[[], list[dict[str, Any]]]
    command_post_lonlat: Callable[[], tuple[float, float]]
    buildings_source: Path
    dispatch: Any = None  # src.dispatch.DispatchHub


def _road_label(data: dict[str, Any], u: Any, v: Any) -> str:
    name = _format_edge_name(data.get("name"), u, v)
    return "unnamed road" if name.startswith("Segment ") else name


def _unique_edges(G: nx.MultiDiGraph):
    """Each physical road segment once (OSM stores both directions)."""
    seen: set[tuple] = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        key = (frozenset((u, v)), k)
        if key in seen:
            continue
        seen.add(key)
        yield u, v, k, data


def _reachable_nodes(G: nx.MultiDiGraph, origin: Any, vehicle_class: str) -> set[Any]:
    attr = f"passable_{vehicle_class}"
    H = nx.DiGraph()
    H.add_node(origin)
    H.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get(attr))
    return nx.descendants(H, origin) | {origin}


class AgentTools:
    """The tool belt shared by every agent tier."""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ---- helpers -------------------------------------------------------------------
    def _origin(self, G: nx.MultiDiGraph) -> Any:
        lon, lat = self.ctx.command_post_lonlat()
        return find_nearest_node(G, lon, lat)

    def road_names(self) -> list[str]:
        G = self.ctx.graph()
        return sorted({_road_label(d, u, v) for u, v, _, d in _unique_edges(G)} - {"unnamed road"})

    def match_road(self, query: str) -> str | None:
        """Resolve a free-text road reference ("15th st", "Highway 98") to an OSM road name."""
        names = self.road_names()
        q = query.lower().strip()
        q = re.sub(r"\b(st|st\.)\b", "street", q)
        q = re.sub(r"\b(hwy|us)\b", "highway", q)
        lowered = {n.lower(): n for n in names}
        if q in lowered:
            return lowered[q]
        for low, name in lowered.items():
            if q in low or low in q:
                return name
        close = difflib.get_close_matches(q, list(lowered), n=1, cutoff=0.6)
        return lowered[close[0]] if close else None

    def _hotspot_nodes(self) -> list[tuple[Any, int, int]]:
        """(node, high-severity count, destroyed count) ranked by damage, deterministic order."""
        counts: dict[str, list[int]] = {}
        for f in self.ctx.buildings():
            p = f["properties"]
            if p.get("damage_severity") in HIGH_SEVERITY:
                c = counts.setdefault(str(p["nearest_node"]), [0, 0])
                c[0] += 1
                c[1] += p["damage_severity"] == "destroyed"
        G = self.ctx.graph()
        lookup = {str(n): n for n in G.nodes}
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
        return [(lookup[n], c[0], c[1]) for n, c in ranked if n in lookup]

    def _resolve_destination(self, destination: str) -> tuple[Any, str]:
        G = self.ctx.graph()
        dest = str(destination).strip()
        m = re.fullmatch(r"hotspot-(\d+)", dest.lower())
        if m:
            hotspots = self._hotspot_nodes()
            idx = int(m.group(1)) - 1
            if not 0 <= idx < len(hotspots):
                raise ValueError(f"No {dest}; call damage_hotspots first.")
            node = hotspots[idx][0]
            return node, f"{dest} near {self._node_road(G, node)}"
        road = self.match_road(dest)
        if road is None:
            raise ValueError(f"Unknown destination '{destination}'. Use a road name or hotspot-N.")
        coords = []
        for u, v, _, d in _unique_edges(G):
            if _road_label(d, u, v) == road:
                coords += [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        return find_nearest_node(G, lon, lat), road

    @staticmethod
    def _node_road(G: nx.MultiDiGraph, node: Any) -> str:
        names = [_road_label(d, u, v) for u, v, d in G.edges(node, data=True)]
        named = [n for n in names if n != "unnamed road"]
        return named[0] if named else (names[0] if names else "unnamed road")

    # ---- tools ---------------------------------------------------------------------
    def measure_road(self, road_name: str) -> dict[str, Any]:
        """SAM 3.1 measured clearance for a named road (all of its segments)."""
        road = self.match_road(road_name)
        if road is None:
            return {"error": f"No road matching '{road_name}'.", "known_roads": self.road_names()[:25]}
        G = self.ctx.graph()
        segs = [(u, v, d) for u, v, _, d in _unique_edges(G) if _road_label(d, u, v) == road]
        narrowest = min(segs, key=lambda s: float(s[2].get("min_clear_width_m", 0.0)))
        return {
            "road": road,
            "segments": len(segs),
            "length_m": round(sum(float(d.get("length_m", 0.0)) for _, _, d in segs)),
            "narrowest_clear_width_m": round(float(narrowest[2].get("min_clear_width_m", 0.0)), 1),
            "narrowest_reason": narrowest[2].get("clearance_reason", ""),
            "standing_water": any(bool(d.get("has_water_obstruction")) for _, _, d in segs),
            "segments_flagged_for_drone_recon": sum(bool(d.get("needs_drone_recon")) for _, _, d in segs),
            "passable_end_to_end": {
                vc: all(bool(d.get(f"passable_{vc}")) for _, _, d in segs) for vc in VEHICLES
            },
            "vehicle_min_width_m": VEHICLE_MIN_WIDTH_M,
            "measured_by": "SAM 3.1 visible-pavement cross-sections every 5 m",
        }

    def damage_hotspots(self, limit: int = 3) -> dict[str, Any]:
        """Blocks with the most Muse Spark high-severity buildings, and who can reach them."""
        G = self.ctx.graph()
        origin = self._origin(G)
        reach = {vc: _reachable_nodes(G, origin, vc) for vc in VEHICLES}
        out = []
        for i, (node, n_high, n_destroyed) in enumerate(self._hotspot_nodes()[: max(1, min(int(limit), 10))]):
            out.append(
                {
                    "id": f"hotspot-{i + 1}",
                    "near_road": self._node_road(G, node),
                    "high_severity_buildings": n_high,
                    "destroyed_buildings": n_destroyed,
                    "reachable_from_command_post": {vc: node in reach[vc] for vc in VEHICLES},
                }
            )
        return {"hotspots": out, "graded_by": "Muse Spark 1.3 (pre/post imagery)"}

    def plan_route(self, vehicle_class: str, destination: str) -> dict[str, Any]:
        """Fastest passable route from the command post for one vehicle class."""
        vc = normalize_vehicle_class(vehicle_class)
        G = self.ctx.graph()
        dest_node, label = self._resolve_destination(destination)
        route = compute_shortest_route(G, origin=self._origin(G), destination=dest_node, vehicle_class=vc)
        result: dict[str, Any] = {
            "vehicle": VEHICLE_LABEL[vc],
            "required_clear_width_m": VEHICLE_MIN_WIDTH_M[vc],
            "destination": label,
            "reachable": route.reachable,
            "blocked_segments_avoided": len(route.blocked_edges_avoided),
        }
        if route.reachable:
            steps: list[dict[str, Any]] = []
            for st in route.edge_steps:
                name = "unnamed road" if st.name.startswith("Segment ") else st.name
                if steps and steps[-1]["road"] == name:
                    steps[-1]["length_m"] += st.length_m
                    steps[-1]["narrowest_m"] = min(steps[-1]["narrowest_m"], st.min_clear_width_m)
                else:
                    steps.append({"road": name, "length_m": st.length_m, "narrowest_m": st.min_clear_width_m})
            for st in steps:
                st["length_m"] = round(st["length_m"])
                st["narrowest_m"] = round(st["narrowest_m"], 1)
            result.update(
                {
                    "distance_m": round(route.total_distance_m),
                    "distance_km": round(route.total_distance_m / 1000.0, 1),
                    "eta_min": round(route.estimated_transit_time_s / 60.0, 1),
                    "steps": steps[:8],
                }
            )
        else:
            alt = {}
            for other in VEHICLES:
                if other != vc:
                    r = compute_shortest_route(G, origin=self._origin(G), destination=dest_node, vehicle_class=other)
                    alt[other] = r.reachable
            result["other_vehicles_reachable"] = alt
            top = self.rank_clearance(vc, limit=1)["choke_points"]
            # The network-wide best clearance for this vehicle; it does not necessarily open this route.
            result["highest_impact_clearance_for_this_vehicle"] = top[0] if top else None
        return result

    def rank_clearance(self, vehicle_class: str, limit: int = 3) -> dict[str, Any]:
        """Debris piles whose clearance reaches the most damaged buildings per m² cleared."""
        vc = normalize_vehicle_class(vehicle_class)
        G = self.ctx.graph()
        recs = rank_choke_point_clearance_roi(
            G,
            origin_node=self._origin(G),
            vehicle_class=vc,
            buildings_source=self.ctx.buildings_source,
            max_recommendations=max(1, min(int(limit), 10)),
        )
        return {
            "vehicle": VEHICLE_LABEL[vc],
            "choke_points": [
                {
                    "rank": r.rank,
                    "road": "unnamed road" if r.road_name.startswith("Segment ") else r.road_name,
                    "edge_id": f"{r.edge_u}-{r.edge_v}-{r.edge_key}",
                    "clear_width_now_m": round(r.current_clear_width_m, 1),
                    "clear_width_after_m": round(r.restored_clear_width_m, 1),
                    "debris_m2": round(r.debris_area_sqm),
                    "high_severity_buildings_unlocked": r.high_severity_structures_unlocked,
                    "buildings_per_100_m2_cleared": round(100.0 * r.clearance_roi, 1),
                }
                for r in recs
            ],
        }

    def reachability_summary(self, vehicle_class: str) -> dict[str, Any]:
        """How many Muse Spark high-severity buildings a vehicle can reach from the command post."""
        vc = normalize_vehicle_class(vehicle_class)
        G = self.ctx.graph()
        reach = {str(n) for n in _reachable_nodes(G, self._origin(G), vc)}
        high = [f["properties"] for f in self.ctx.buildings() if f["properties"].get("damage_severity") in HIGH_SEVERITY]
        reachable = sum(str(p["nearest_node"]) in reach for p in high)
        return {
            "vehicle": VEHICLE_LABEL[vc],
            "high_severity_buildings_total": len(high),
            "reachable": reachable,
            "unreachable": len(high) - reachable,
            "reachable_percent": round(100.0 * reachable / len(high)) if high else 0,
        }

    def dispatch_field_unit(self, vehicle_class: str, destination: str) -> dict[str, Any]:
        """Send a vehicle assignment to every field phone, only if a passable route exists."""
        vc = normalize_vehicle_class(vehicle_class)
        G = self.ctx.graph()
        node, label = self._resolve_destination(destination)
        origin = self._origin(G)
        if not compute_shortest_route(G, origin=origin, destination=node, vehicle_class=vc).reachable:
            others = {
                other: compute_shortest_route(G, origin=origin, destination=node, vehicle_class=other).reachable
                for other in VEHICLES
                if other != vc
            }
            return {
                "dispatched": False,
                "vehicle": VEHICLE_LABEL[vc],
                "destination": label,
                "reason": "no passable route from the command post; nothing was sent to field units",
                "other_vehicles_reachable": others,
            }
        dest = (float(G.nodes[node]["x"]), float(G.nodes[node]["y"]))
        devices = 0
        if self.ctx.dispatch is not None:
            self.ctx.dispatch.update_assignment(
                vehicle_class=vc, origin_lonlat=self.ctx.command_post_lonlat(), destination_lonlat=dest
            )
            devices = int(self.ctx.dispatch.field_clients)
        return {
            "dispatched": True,
            "vehicle": VEHICLE_LABEL[vc],
            "vehicle_class": vc,
            "destination": label,
            "destination_lonlat": [round(dest[0], 6), round(dest[1], 6)],
            "field_devices_notified": devices,
        }

    # ---- registry ------------------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = TOOL_FUNCS.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return fn(self, **args)
        except (TypeError, ValueError, KeyError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


TOOL_FUNCS: dict[str, Callable[..., dict[str, Any]]] = {
    "measure_road": AgentTools.measure_road,
    "damage_hotspots": AgentTools.damage_hotspots,
    "plan_route": AgentTools.plan_route,
    "rank_clearance": AgentTools.rank_clearance,
    "reachability_summary": AgentTools.reachability_summary,
    "dispatch_field_unit": AgentTools.dispatch_field_unit,
}

_VEHICLE_ARG = {"type": "string", "enum": list(VEHICLES), "description": "atv, ambulance or fire_engine"}
_DEST_ARG = {"type": "string", "description": "A road name (e.g. '15th Street') or a hotspot id like 'hotspot-1'"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "measure_road",
            "description": "SAM 3.1 measured clear width and per-vehicle passability for a named road.",
            "parameters": {"type": "object", "properties": {"road_name": {"type": "string"}}, "required": ["road_name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "damage_hotspots",
            "description": "Blocks with the most severely damaged buildings (Muse Spark grades) and which vehicles can reach them.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 10}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_route",
            "description": "Fastest passable route from the command post for a vehicle class; if blocked, says which other vehicles can get through and the highest-impact debris clearance for this vehicle (which may not be on this route).",
            "parameters": {
                "type": "object",
                "properties": {"vehicle_class": _VEHICLE_ARG, "destination": _DEST_ARG},
                "required": ["vehicle_class", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_clearance",
            "description": "Debris piles ranked by severely damaged buildings unlocked per m² cleared for a vehicle class.",
            "parameters": {
                "type": "object",
                "properties": {"vehicle_class": _VEHICLE_ARG, "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
                "required": ["vehicle_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reachability_summary",
            "description": "How many severely damaged buildings a vehicle class can currently reach from the command post.",
            "parameters": {"type": "object", "properties": {"vehicle_class": _VEHICLE_ARG}, "required": ["vehicle_class"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_field_unit",
            "description": "Send the route assignment to field phones. Only call when the commander asks to send, dispatch or get a vehicle somewhere.",
            "parameters": {
                "type": "object",
                "properties": {"vehicle_class": _VEHICLE_ARG, "destination": _DEST_ARG},
                "required": ["vehicle_class", "destination"],
            },
        },
    },
]


def numbers_in(obj: Any) -> list[float]:
    """Every number in a tool result, including digits inside strings (e.g. '15th Street')."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and math.isnan(obj)):
            out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(m.replace(",", "")) for m in re.findall(r"\d+(?:\.\d+)?", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(numbers_in(k) if isinstance(k, str) else [])
            out.extend(numbers_in(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(numbers_in(v))
    return out
