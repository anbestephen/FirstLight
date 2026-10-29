"""
Multi-class NetworkX emergency vehicle routing and Choke-Point Clearance ROI engine (`src/routing.py`).

Key Invariants (AGENTS.md & routing-engine skill):
1. Vehicle Clearance Thresholds:
   - ATV / On Foot (`atv`): requires `min_clear_width_m >= 1.2m`
   - Ambulance / Pickup (`ambulance`): requires `min_clear_width_m >= 2.2m`
   - Fire Engine (`fire_engine`): requires `min_clear_width_m >= 3.5m`
2. Impassable Water Rule:
   - Any edge with `water_overlap_ratio > 0` (or `has_water_obstruction == True`) receives
     `edge_weight = float('inf')` (strictly impassable for all vehicle classes, regardless of width).
3. Choke-Point Clearance ROI:
   - `clearance_roi = structures_unlocked / debris_area_sqm`
   - Ranked strictly by `clearance_roi` descending with zero hallucinated metrics; all
     explanation strings are mathematically derived from pixel-to-meter geometry and graph weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point

from src.geometry import (
    AMBULANCE_MIN_WIDTH_M,
    ATV_MIN_WIDTH_M,
    DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M,
    FIRE_ENGINE_MIN_WIDTH_M,
    TARGET_METRIC_CRS,
    WGS84_CRS,
    evaluate_road_network_clearance,
)


class VehicleClass(str, Enum):
    """Supported emergency response vehicle classes."""

    ATV = "atv"
    AMBULANCE = "ambulance"
    FIRE_ENGINE = "fire_engine"


VEHICLE_MIN_WIDTH_M: dict[str, float] = {
    VehicleClass.ATV.value: ATV_MIN_WIDTH_M,  # 1.2m
    VehicleClass.AMBULANCE.value: AMBULANCE_MIN_WIDTH_M,  # 2.2m
    VehicleClass.FIRE_ENGINE.value: FIRE_ENGINE_MIN_WIDTH_M,  # 3.5m
}

# OSM ways that only ATVs / people on foot may use, whatever their measured width.
PEDESTRIAN_HIGHWAYS = frozenset({"footway", "path", "steps", "pedestrian", "cycleway"})
FOOT_ONLY_EXEMPT_VEHICLES = frozenset({VehicleClass.ATV.value})


def is_pedestrian_only(highway: Any) -> bool:
    types = highway if isinstance(highway, list) else [highway]
    return all(str(t) in PEDESTRIAN_HIGHWAYS for t in types)


VEHICLE_LABEL: dict[str, str] = {
    VehicleClass.ATV.value: "ATV / Foot (>=1.2m)",
    VehicleClass.AMBULANCE.value: "Ambulance / Pickup (>=2.2m)",
    VehicleClass.FIRE_ENGINE.value: "Fire Engine (>=3.5m)",
}

# Nominal emergency response speeds in meters per second
VEHICLE_SPEED_MPS: dict[str, float] = {
    VehicleClass.ATV.value: 8.0,  # ~28.8 km/h
    VehicleClass.AMBULANCE.value: 13.5,  # ~48.6 km/h
    VehicleClass.FIRE_ENGINE.value: 11.2,  # ~40.3 km/h
}


def normalize_vehicle_class(vehicle_class: str | VehicleClass) -> str:
    """Normalize vehicle class identifier to `'atv'`, `'ambulance'`, or `'fire_engine'`."""
    if isinstance(vehicle_class, VehicleClass):
        return vehicle_class.value
    raw = str(vehicle_class).strip().lower().replace("-", "_").replace(" ", "_")
    if raw in ("atv", "atv_foot", "foot", "on_foot"):
        return VehicleClass.ATV.value
    if raw in ("ambulance", "ambulance_pickup", "pickup", "ems"):
        return VehicleClass.AMBULANCE.value
    if raw in ("fire_engine", "fire", "engine", "firetruck"):
        return VehicleClass.FIRE_ENGINE.value
    raise ValueError(f"Unknown vehicle_class '{vehicle_class}'. Expected atv, ambulance, or fire_engine.")


@dataclass
class RouteEdgeStep:
    """Turn-by-turn edge annotation along a computed vehicle route."""

    u: int | str
    v: int | str
    key: int
    name: str
    length_m: float
    travel_time_s: float
    min_clear_width_m: float
    debris_area_sqm: float
    water_overlap_ratio: float
    passable_for: dict[str, bool]
    instruction: str
    sam_confidence: float = 0.78
    needs_drone_recon: bool = False
    coordinates_lonlat: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "u": self.u,
            "v": self.v,
            "key": self.key,
            "name": self.name,
            "length_m": round(self.length_m, 2),
            "travel_time_s": round(self.travel_time_s, 2),
            "min_clear_width_m": round(self.min_clear_width_m, 2),
            "debris_area_sqm": round(self.debris_area_sqm, 2),
            "water_overlap_ratio": round(self.water_overlap_ratio, 4),
            "passable_for": self.passable_for,
            "sam_confidence": round(self.sam_confidence, 4),
            "needs_drone_recon": self.needs_drone_recon,
            "instruction": self.instruction,
            "coordinates_lonlat": [
                [round(lon, 6), round(lat, 6)] for lon, lat in self.coordinates_lonlat
            ],
        }


@dataclass
class RouteResult:
    """Complete routing result for a single vehicle class."""

    vehicle_class: str
    required_width_m: float
    reachable: bool
    origin_node: int | str
    destination_node: int | str
    node_path: list[int | str]
    geojson_geometry: dict[str, Any]
    total_distance_m: float
    estimated_transit_time_s: float
    edge_steps: list[RouteEdgeStep]
    blocked_edges_avoided: list[dict[str, Any]]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_class": self.vehicle_class,
            "required_width_m": self.required_width_m,
            "reachable": self.reachable,
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "node_path": self.node_path,
            "geojson_geometry": self.geojson_geometry,
            "total_distance_m": round(self.total_distance_m, 2),
            "estimated_transit_time_s": round(self.estimated_transit_time_s, 2),
            "edge_steps": [step.to_dict() for step in self.edge_steps],
            "blocked_edges_avoided": self.blocked_edges_avoided,
            "rationale": self.rationale,
        }


@dataclass
class ChokePointROIRecommendation:
    """Ranked Choke-Point Clearance ROI recommendation."""

    rank: int
    edge_u: int | str
    edge_v: int | str
    edge_key: int
    road_name: str
    vehicle_class: str
    current_clear_width_m: float
    restored_clear_width_m: float
    debris_area_sqm: float
    water_overlap_ratio: float
    gsd_m_per_px: float
    high_severity_structures_unlocked: int
    newly_reachable_nodes: list[int | str]
    clearance_roi: float
    bottleneck_lonlat: tuple[float, float]
    explanation: str
    sam_confidence: float = 0.78
    needs_drone_recon: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "edge_id": f"{self.edge_u}-{self.edge_v}-{self.edge_key}",
            "edge_u": self.edge_u,
            "edge_v": self.edge_v,
            "edge_key": self.edge_key,
            "road_name": self.road_name,
            "vehicle_class": self.vehicle_class,
            "current_clear_width_m": round(self.current_clear_width_m, 2),
            "restored_clear_width_m": round(self.restored_clear_width_m, 2),
            "debris_area_sqm": round(self.debris_area_sqm, 2),
            "water_overlap_ratio": round(self.water_overlap_ratio, 4),
            "gsd_m_per_px": round(self.gsd_m_per_px, 4),
            "sam_confidence": round(self.sam_confidence, 4),
            "needs_drone_recon": self.needs_drone_recon,
            "high_severity_structures_unlocked": self.high_severity_structures_unlocked,
            "newly_reachable_nodes": self.newly_reachable_nodes,
            "clearance_roi": round(self.clearance_roi, 4),
            "bottleneck_lonlat": [
                round(self.bottleneck_lonlat[0], 6),
                round(self.bottleneck_lonlat[1], 6),
            ],
            "explanation": self.explanation,
        }


def is_edge_passable_for_vehicle(
    min_clear_width_m: float,
    water_overlap_ratio: float,
    vehicle_class: str | VehicleClass,
) -> bool:
    """
    Evaluate whether an edge is passable for `vehicle_class`.

    Rules:
    - Water Rule: any `water_overlap_ratio > 0` is strictly impassable (`False`) for all classes.
    - Width Thresholds:
      * ATV / Foot: `min_clear_width_m >= 1.2m`
      * Ambulance / Pickup: `min_clear_width_m >= 2.2m`
      * Fire Engine: `min_clear_width_m >= 3.5m`
    """
    if float(water_overlap_ratio) > 0.0:
        return False
    vc = normalize_vehicle_class(vehicle_class)
    threshold_m = VEHICLE_MIN_WIDTH_M[vc]
    return float(min_clear_width_m) >= threshold_m


def compute_edge_routing_weight(
    length_m: float,
    min_clear_width_m: float,
    water_overlap_ratio: float,
    vehicle_class: str | VehicleClass,
) -> float:
    """
    Compute the NetworkX edge weight for `vehicle_class`.

    Returns `float('inf')` if `water_overlap_ratio > 0` or `min_clear_width_m < threshold`.
    Otherwise returns positive travel cost in meters (`length_m`).
    """
    if not is_edge_passable_for_vehicle(min_clear_width_m, water_overlap_ratio, vehicle_class):
        return float("inf")
    return max(float(length_m), 0.001)


def annotate_graph_with_geometry_metrics(
    graph: nx.MultiDiGraph | nx.DiGraph | nx.Graph,
    *,
    gsd_m_per_px: float = 0.2804,
    default_nominal_width_m: float = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M * 2.0,
) -> nx.MultiDiGraph:
    """
    Ensure every edge in `graph` has standardized geometric routing attributes:
      - `length_m`
      - `min_clear_width_m`
      - `nominal_width_m`
      - `water_overlap_ratio`
      - `debris_area_sqm`
      - `gsd_m_per_px`
      - `weight_atv`, `weight_ambulance`, `weight_fire_engine`
      - `passable_atv`, `passable_ambulance`, `passable_fire_engine`
    """
    if not isinstance(graph, nx.MultiDiGraph):
        G = nx.MultiDiGraph(graph)
    else:
        G = graph.copy()

    for u, v, k, data in G.edges(keys=True, data=True):
        raw_len = data.get("length_m", data.get("length", 50.0))
        try:
            length_m = float(raw_len)
        except (TypeError, ValueError):
            length_m = 50.0

        nominal_w = float(data.get("nominal_width_m", default_nominal_width_m))
        min_w = float(data.get("min_clear_width_m", nominal_w))

        # Support either explicit water_overlap_ratio or water_area_m2 / has_water_obstruction
        if "water_overlap_ratio" in data and data["water_overlap_ratio"] is not None:
            water_ratio = float(data["water_overlap_ratio"])
        elif data.get("has_water_obstruction"):
            corridor_area = max(length_m * nominal_w, 1.0)
            water_area = float(data.get("water_area_m2", corridor_area * 0.25))
            water_ratio = max(water_area / corridor_area, 0.01)
        elif float(data.get("water_area_m2", 0.0)) > 0.0:
            corridor_area = max(length_m * nominal_w, 1.0)
            water_ratio = float(data["water_area_m2"]) / corridor_area
        else:
            water_ratio = 0.0

        if water_ratio > 0.0:
            min_w = 0.0

        debris_sqm = float(
            data.get("debris_area_sqm", data.get("debris_area_m2", 0.0))
        )
        edge_gsd = float(data.get("gsd_m_per_px", gsd_m_per_px))

        data["length_m"] = length_m
        data["nominal_width_m"] = nominal_w
        data["min_clear_width_m"] = min_w
        data["water_overlap_ratio"] = water_ratio
        data["has_water_obstruction"] = water_ratio > 0.0
        data["debris_area_sqm"] = debris_sqm
        data["debris_area_m2"] = debris_sqm
        data["gsd_m_per_px"] = edge_gsd

        foot_only = is_pedestrian_only(data.get("highway"))
        for vc in (
            VehicleClass.ATV.value,
            VehicleClass.AMBULANCE.value,
            VehicleClass.FIRE_ENGINE.value,
        ):
            passable = is_edge_passable_for_vehicle(min_w, water_ratio, vc)
            weight = compute_edge_routing_weight(length_m, min_w, water_ratio, vc)
            if foot_only and vc not in FOOT_ONLY_EXEMPT_VEHICLES:
                passable, weight = False, math.inf
            data[f"passable_{vc}"] = passable
            data[f"weight_{vc}"] = weight

    return G


def load_sam31_measurements(path: str | Path) -> dict[str, Any]:
    """Load `data/sam31_edge_measurements.json` (baked SAM 3.1 per-edge measurements)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_sam31_measurements(G: nx.MultiDiGraph, payload: dict[str, Any]) -> None:
    """
    Annotate graph edges in place from baked SAM 3.1 measurements. `nominal_width_m` (the
    width restored by a simulated debris clearance) is the median visible pavement width
    SAM measured on the unobstructed stations of that same road.
    """
    gsd = float(payload.get("gsd_m_per_px", 0.2804))
    edges = payload.get("edges", {})
    default_nominal = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M * 2.0
    for u, v, k, data in G.edges(keys=True, data=True):
        rec = edges.get(f"{u}-{v}-{int(k)}")
        if rec is None:
            data["mask_source"] = "unmeasured"
            data["needs_drone_recon"] = True
            continue
        clear_widths = [st["clear_width_m"] for st in rec["stations"] if st["cause"] == "clear"]
        nominal_w = float(np.median(clear_widths)) if clear_widths else default_nominal
        length_m = float(rec["length_m"])
        has_water = bool(rec["has_water_obstruction"])
        water_ratio = (
            max(float(rec["water_area_m2"]) / max(length_m * nominal_w, 1.0), 0.01) if has_water else 0.0
        )
        data["length_m"] = length_m
        data["nominal_width_m"] = round(nominal_w, 2)
        data["min_clear_width_m"] = float(rec["min_clear_width_m"])
        data["water_overlap_ratio"] = water_ratio
        data["has_water_obstruction"] = has_water
        data["debris_area_sqm"] = float(rec["debris_area_m2"])
        data["water_area_m2"] = float(rec["water_area_m2"])
        data["gsd_m_per_px"] = gsd
        data["bottleneck_lonlat"] = rec.get("bottleneck_lonlat")
        data["sam_confidence"] = float(rec["sam_confidence"])
        data["needs_drone_recon"] = bool(rec["needs_drone_recon"])
        data["mask_source"] = str(payload.get("mask_source", "sam-3.1"))
        data["clearance_reason"] = str(rec.get("reason", ""))


def load_annotated_routing_graph(
    graphml_path: str | Path = "data/osm_road_graph.graphml",
    roads_geojson_path: str | Path | None = "data/osm_roads.geojson",
    raster_path: str | Path | None = "data/noaa_michael_mexico_beach.tif",
    *,
    masks_geojson_path: str | Path | None = "data/sam31_mexico_beach_masks.geojson",
    simulate_if_no_masks: bool = False,
    measurements_path: str | Path | None = "data/sam31_edge_measurements.json",
) -> nx.MultiDiGraph:
    """
    Load the cached OpenStreetMap road graph (`data/osm_road_graph.graphml`) and annotate
    every edge with `min_clear_width_m`, `water_overlap_ratio`, `debris_area_sqm`,
    `sam_confidence`, `needs_drone_recon`, and vehicle-specific edge weights.

    When `measurements_path` exists (baked by `scripts/bake_sam31_masks.py`), the real
    SAM 3.1 visible-pavement measurements are used; otherwise widths are computed from
    `masks_geojson_path` polygons via `src/geometry.py`.
    """
    G = ox.load_graphml(Path(graphml_path))

    # Ensure node x/y coordinates are float
    for _, node_data in G.nodes(data=True):
        if "x" in node_data:
            node_data["x"] = float(node_data["x"])
        if "y" in node_data:
            node_data["y"] = float(node_data["y"])

    if measurements_path is not None and Path(measurements_path).exists():
        apply_sam31_measurements(G, load_sam31_measurements(measurements_path))
        assign_display_names(G)
        return annotate_graph_with_geometry_metrics(G)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    edges_reset = edges_gdf.reset_index()

    _, clearances = evaluate_road_network_clearance(
        roads_source=edges_reset,
        raster_path=raster_path if (raster_path and Path(raster_path).exists()) else None,
        masks_geojson_path=masks_geojson_path,
        simulate_if_no_masks=simulate_if_no_masks,
    )

    for idx, row in edges_reset.iterrows():
        u = row["u"]
        v = row["v"]
        k = row["key"] if "key" in row else 0
        if idx < len(clearances) and G.has_edge(u, v, k):
            rec = clearances[idx]
            edge_data = G[u][v][k]
            nominal_w = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M * 2.0
            corridor_area = max(rec.length_m * nominal_w, 1.0)
            water_ratio = (
                rec.water_area_m2 / corridor_area
                if rec.has_water_obstruction or rec.water_area_m2 > 0.0
                else 0.0
            )
            if rec.has_water_obstruction and water_ratio <= 0.0:
                water_ratio = 0.1

            edge_data["length_m"] = rec.length_m
            edge_data["nominal_width_m"] = nominal_w
            edge_data["min_clear_width_m"] = rec.min_clear_width_m
            edge_data["water_overlap_ratio"] = water_ratio
            edge_data["has_water_obstruction"] = rec.has_water_obstruction
            edge_data["debris_area_sqm"] = rec.debris_area_m2
            edge_data["water_area_m2"] = rec.water_area_m2
            edge_data["gsd_m_per_px"] = rec.gsd_m_per_px
            edge_data["bottleneck_lonlat"] = rec.bottleneck_point_lonlat
            edge_data["sam_confidence"] = rec.sam_confidence
            edge_data["needs_drone_recon"] = rec.needs_drone_recon
            edge_data["mask_source"] = rec.mask_source
            edge_data["clearance_reason"] = rec.reason

    assign_display_names(G)
    return annotate_graph_with_geometry_metrics(G)


def build_vehicle_routing_subgraph(
    graph: nx.MultiDiGraph,
    vehicle_class: str | VehicleClass,
) -> nx.MultiDiGraph:
    """
    Return a vehicle-specific routing graph where every edge has attribute `weight`
    set to `compute_edge_routing_weight(...)` (`float('inf')` if blocked or flooded).
    """
    vc = normalize_vehicle_class(vehicle_class)
    G = annotate_graph_with_geometry_metrics(graph)
    weight_attr = f"weight_{vc}"

    for _, _, _, data in G.edges(keys=True, data=True):
        data["weight"] = data.get(weight_attr, float("inf"))

    return G


def find_nearest_node(
    graph: nx.MultiDiGraph,
    lon: float,
    lat: float,
) -> int | str:
    """Find the graph node closest to `(lon, lat)` in WGS84 coordinates."""
    best_node: int | str | None = None
    best_dist_sq = float("inf")
    for node_id, data in graph.nodes(data=True):
        nx_lon = float(data.get("x", data.get("lon", 0.0)))
        ny_lat = float(data.get("y", data.get("lat", 0.0)))
        # Approximate metric scaling at ~30 deg latitude
        dx = (nx_lon - lon) * 96486.0
        dy = (ny_lat - lat) * 110852.0
        d2 = dx * dx + dy * dy
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_node = node_id

    if best_node is None:
        raise ValueError("Graph contains no nodes.")
    return best_node


def _extract_edge_lonlat_coords(
    graph: nx.MultiDiGraph,
    u: int | str,
    v: int | str,
    edge_data: dict[str, Any],
) -> list[tuple[float, float]]:
    """Extract WGS84 `(lon, lat)` coordinates for a directed edge `(u, v)`."""
    geom = edge_data.get("geometry")
    if isinstance(geom, LineString) and not geom.is_empty:
        coords = [(float(x), float(y)) for x, y in geom.coords]
        # Verify orientation matches u -> v
        u_x = float(graph.nodes[u].get("x", coords[0][0]))
        u_y = float(graph.nodes[u].get("y", coords[0][1]))
        d_start = math.hypot(coords[0][0] - u_x, coords[0][1] - u_y)
        d_end = math.hypot(coords[-1][0] - u_x, coords[-1][1] - u_y)
        if d_end < d_start:
            coords.reverse()
        return coords

    u_x = float(graph.nodes[u].get("x", 0.0))
    u_y = float(graph.nodes[u].get("y", 0.0))
    v_x = float(graph.nodes[v].get("x", 0.0))
    v_y = float(graph.nodes[v].get("y", 0.0))
    return [(u_x, u_y), (v_x, v_y)]


def _select_best_parallel_edge(
    graph: nx.MultiDiGraph,
    u: int | str,
    v: int | str,
    weight_attr: str,
) -> tuple[int, dict[str, Any]]:
    """Select the minimum-weight parallel edge between `u` and `v` in a MultiDiGraph."""
    edge_dict = graph[u][v]
    best_k = min(
        edge_dict.keys(),
        key=lambda k: float(edge_dict[k].get(weight_attr, float("inf"))),
    )
    return best_k, edge_dict[best_k]


def _raw_road_name(raw_name: Any) -> str | None:
    if isinstance(raw_name, list):
        valid = [str(x) for x in raw_name if x]
        return ", ".join(valid) if valid else None
    if raw_name is None or (isinstance(raw_name, float) and math.isnan(raw_name)):
        return None
    text = str(raw_name).strip()
    return text or None


def _format_edge_name(
    raw_name: Any, u: int | str, v: int | str, display_name: str | None = None
) -> str:
    """OSM road name, else the edge's `display_name` ("unnamed road off 26th Street")."""
    return _raw_road_name(raw_name) or display_name or "unnamed road"


def assign_display_names(G: nx.MultiDiGraph) -> None:
    """
    Name each unnamed OSM segment after the named streets it connects to, so responders see
    "unnamed road off 26th Street" (or "between A and B") instead of node ids.
    """
    def named_at(node: Any) -> list[str]:
        names: list[str] = []
        for a, b, d in list(G.out_edges(node, data=True)) + list(G.in_edges(node, data=True)):
            n = _raw_road_name(d.get("name"))
            if n and n not in names:
                names.append(n)
        return names

    for u, v, data in G.edges(data=True):
        if _raw_road_name(data.get("name")):
            continue
        neighbours: list[str] = []
        for n in named_at(u) + named_at(v):
            if n not in neighbours:
                neighbours.append(n)
        if len(neighbours) >= 2:
            data["display_name"] = f"unnamed road between {neighbours[0]} and {neighbours[1]}"
        elif neighbours:
            data["display_name"] = f"unnamed road off {neighbours[0]}"
        else:
            data["display_name"] = "unnamed road"


def compute_shortest_route(
    graph: nx.MultiDiGraph,
    origin: int | str | tuple[float, float],
    destination: int | str | tuple[float, float],
    vehicle_class: str | VehicleClass = VehicleClass.FIRE_ENGINE,
    *,
    algorithm: Literal["dijkstra", "astar"] = "dijkstra",
) -> RouteResult:
    """
    Compute the shortest passable route for `vehicle_class` using Dijkstra or A*,
    strictly respecting vehicle width thresholds and infinite-weight water rules.

    Returns GeoJSON LineString coordinates, total distance in meters, transit time
    estimates, turn-by-turn edge passability annotations, and avoided bottlenecks.
    """
    vc = normalize_vehicle_class(vehicle_class)
    required_w = VEHICLE_MIN_WIDTH_M[vc]
    speed_mps = VEHICLE_SPEED_MPS[vc]
    G = annotate_graph_with_geometry_metrics(graph)

    origin_node = (
        find_nearest_node(G, origin[0], origin[1])
        if isinstance(origin, tuple)
        else origin
    )
    dest_node = (
        find_nearest_node(G, destination[0], destination[1])
        if isinstance(destination, tuple)
        else destination
    )

    weight_attr = f"weight_{vc}"

    # Build a passable subgraph containing only finite-weight edges for this vehicle class
    passable_edges = [
        (u, v, k)
        for u, v, k, d in G.edges(keys=True, data=True)
        if math.isfinite(float(d.get(weight_attr, float("inf"))))
    ]
    G_passable = G.edge_subgraph(passable_edges).copy()

    # Also compute baseline unobstructed shortest path on the full road graph (ignoring debris/water)
    # so we can report which specific bottlenecks blocked or detoured this vehicle class.
    blocked_edges_avoided: list[dict[str, Any]] = []
    try:
        baseline_nodes = nx.shortest_path(G, origin_node, dest_node, weight="length_m")
        for i in range(len(baseline_nodes) - 1):
            bu, bv = baseline_nodes[i], baseline_nodes[i + 1]
            bk, bdata = _select_best_parallel_edge(G, bu, bv, "length_m")
            if not math.isfinite(float(bdata.get(weight_attr, float("inf")))):
                blocked_edges_avoided.append(
                    {
                        "u": bu,
                        "v": bv,
                        "key": bk,
                        "name": _format_edge_name(bdata.get("name"), bu, bv, bdata.get("display_name")),
                        "min_clear_width_m": round(float(bdata.get("min_clear_width_m", 0.0)), 2),
                        "debris_area_sqm": round(float(bdata.get("debris_area_sqm", 0.0)), 2),
                        "water_overlap_ratio": round(float(bdata.get("water_overlap_ratio", 0.0)), 4),
                        "reason": (
                            "Standing water washout (weight=inf)"
                            if float(bdata.get("water_overlap_ratio", 0.0)) > 0.0
                            else f"Clear width {float(bdata.get('min_clear_width_m', 0.0)):.2f}m < {required_w:.1f}m required for {vc}"
                        ),
                    }
                )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    # Check if origin and destination are in G_passable and connected
    if (
        origin_node not in G_passable
        or dest_node not in G_passable
    ):
        reason = (
            f"NO PASSABLE ROUTE for {VEHICLE_LABEL[vc]}: destination isolated by "
            f"{len(blocked_edges_avoided)} obstructed segment(s)."
        )
        return RouteResult(
            vehicle_class=vc,
            required_width_m=required_w,
            reachable=False,
            origin_node=origin_node,
            destination_node=dest_node,
            node_path=[],
            geojson_geometry={"type": "LineString", "coordinates": []},
            total_distance_m=0.0,
            estimated_transit_time_s=0.0,
            edge_steps=[],
            blocked_edges_avoided=blocked_edges_avoided,
            rationale=reason,
        )

    try:
        if algorithm == "astar":
            def _heuristic(n1: int | str, n2: int | str) -> float:
                x1 = float(G.nodes[n1].get("x", 0.0))
                y1 = float(G.nodes[n1].get("y", 0.0))
                x2 = float(G.nodes[n2].get("x", 0.0))
                y2 = float(G.nodes[n2].get("y", 0.0))
                return math.hypot((x2 - x1) * 96486.0, (y2 - y1) * 110852.0)

            node_path = nx.astar_path(
                G_passable, origin_node, dest_node, heuristic=_heuristic, weight=weight_attr
            )
        else:
            node_path = nx.dijkstra_path(
                G_passable, origin_node, dest_node, weight=weight_attr
            )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        reason = (
            f"NO PASSABLE ROUTE for {VEHICLE_LABEL[vc]}: all connecting corridors blocked "
            f"(<{required_w:.1f}m clear width or standing water)."
        )
        return RouteResult(
            vehicle_class=vc,
            required_width_m=required_w,
            reachable=False,
            origin_node=origin_node,
            destination_node=dest_node,
            node_path=[],
            geojson_geometry={"type": "LineString", "coordinates": []},
            total_distance_m=0.0,
            estimated_transit_time_s=0.0,
            edge_steps=[],
            blocked_edges_avoided=blocked_edges_avoided,
            rationale=reason,
        )

    edge_steps: list[RouteEdgeStep] = []
    full_coords: list[list[float]] = []
    total_distance_m = 0.0
    total_time_s = 0.0

    for idx in range(len(node_path) - 1):
        u = node_path[idx]
        v = node_path[idx + 1]
        k, edata = _select_best_parallel_edge(G_passable, u, v, weight_attr)

        seg_len_m = float(edata.get("length_m", 0.0))
        seg_width_m = float(edata.get("min_clear_width_m", 8.0))
        seg_debris = float(edata.get("debris_area_sqm", 0.0))
        seg_water = float(edata.get("water_overlap_ratio", 0.0))
        seg_name = _format_edge_name(edata.get("name"), u, v, edata.get("display_name"))

        # Reduce effective speed by 30% when navigating partial debris corridors
        effective_speed = speed_mps * (0.7 if seg_debris > 0.0 else 1.0)
        seg_time_s = seg_len_m / max(effective_speed, 0.5)

        total_distance_m += seg_len_m
        total_time_s += seg_time_s

        coords_lonlat = _extract_edge_lonlat_coords(G, u, v, edata)
        for lon, lat in coords_lonlat:
            pt_pair = [round(lon, 6), round(lat, 6)]
            if not full_coords or full_coords[-1] != pt_pair:
                full_coords.append(pt_pair)

        passable_for = {
            "atv": bool(edata.get("passable_atv", True)),
            "ambulance": bool(edata.get("passable_ambulance", True)),
            "fire_engine": bool(edata.get("passable_fire_engine", True)),
        }
        instruction = (
            f"Proceed {seg_len_m:.1f}m on {seg_name} "
            f"(min clear width {seg_width_m:.2f}m, {seg_debris:.1f} m² debris)."
        )
        seg_conf = float(edata.get("sam_confidence", 0.78))
        seg_drone = bool(edata.get("needs_drone_recon", seg_conf < 0.70))
        edge_steps.append(
            RouteEdgeStep(
                u=u,
                v=v,
                key=k,
                name=seg_name,
                length_m=seg_len_m,
                travel_time_s=seg_time_s,
                min_clear_width_m=seg_width_m,
                debris_area_sqm=seg_debris,
                water_overlap_ratio=seg_water,
                passable_for=passable_for,
                instruction=instruction,
                sam_confidence=seg_conf,
                needs_drone_recon=seg_drone,
                coordinates_lonlat=coords_lonlat,
            )
        )

    if len(node_path) == 1 and not full_coords:
        lon0 = float(G.nodes[origin_node].get("x", 0.0))
        lat0 = float(G.nodes[origin_node].get("y", 0.0))
        full_coords = [[round(lon0, 6), round(lat0, 6)]]

    detour_note = (
        f" Detoured around {len(blocked_edges_avoided)} obstructed segment(s)."
        if blocked_edges_avoided
        else " Direct primary corridor clear."
    )
    rationale = (
        f"Route verified for {VEHICLE_LABEL[vc]}: {total_distance_m:.1f}m total distance, "
        f"ETA {total_time_s:.1f}s across {len(edge_steps)} road segment(s).{detour_note}"
    )

    return RouteResult(
        vehicle_class=vc,
        required_width_m=required_w,
        reachable=True,
        origin_node=origin_node,
        destination_node=dest_node,
        node_path=list(node_path),
        geojson_geometry={
            "type": "LineString",
            "coordinates": full_coords,
        },
        total_distance_m=total_distance_m,
        estimated_transit_time_s=total_time_s,
        edge_steps=edge_steps,
        blocked_edges_avoided=blocked_edges_avoided,
        rationale=rationale,
    )


def compute_multi_class_routes(
    graph: nx.MultiDiGraph,
    origin: int | str | tuple[float, float],
    destination: int | str | tuple[float, float],
) -> dict[str, RouteResult]:
    """Compute routes simultaneously for ATV, Ambulance, and Fire Engine."""
    G = annotate_graph_with_geometry_metrics(graph)
    return {
        vc: compute_shortest_route(G, origin, destination, vehicle_class=vc)
        for vc in (
            VehicleClass.ATV.value,
            VehicleClass.AMBULANCE.value,
            VehicleClass.FIRE_ENGINE.value,
        )
    }


def assign_buildings_to_graph_nodes(
    graph: nx.MultiDiGraph,
    buildings_source: str | Path | gpd.GeoDataFrame | None = "data/osm_buildings_graded.geojson",
) -> nx.MultiDiGraph:
    """
    Map high-severity damaged building structures onto their nearest road network nodes.

    If graph nodes already have `high_severity_buildings` populated, preserves them;
    otherwise loads `data/osm_buildings_graded.geojson` (real `muse-spark-1.3` graded footprints)
    and assigns high-severity structure counts strictly from verified `damage_severity` grades.
    """
    G = graph.copy()
    existing_counts = sum(
        int(d.get("high_severity_buildings", 0)) for _, d in G.nodes(data=True)
    )
    if existing_counts > 0:
        return G

    for _, d in G.nodes(data=True):
        d.setdefault("high_severity_buildings", 0)
        d.setdefault("total_buildings", 0)

    if buildings_source is None:
        return G

    if isinstance(buildings_source, (str, Path)):
        b_path = Path(buildings_source)
        graded_sibling = b_path.parent / "osm_buildings_graded.geojson"
        if b_path.name == "osm_buildings.geojson" and graded_sibling.exists():
            b_path = graded_sibling
        if not b_path.exists():
            return G
        buildings_gdf = gpd.read_file(b_path)
    else:
        buildings_gdf = buildings_source.copy()

    if buildings_gdf.empty:
        return G

    if buildings_gdf.crs is None:
        buildings_gdf = buildings_gdf.set_crs(WGS84_CRS)
    buildings_wgs84 = buildings_gdf.to_crs(WGS84_CRS)

    for _, row in buildings_wgs84.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        centroid = geom.centroid
        nearest_n = find_nearest_node(G, float(centroid.x), float(centroid.y))
        G.nodes[nearest_n]["total_buildings"] = int(G.nodes[nearest_n].get("total_buildings", 0)) + 1

        sev = row.get("damage_severity", row.get("building_damage_severity"))
        is_high = bool(
            sev in ("major_damage", "destroyed")
            or row.get("is_high_severity") is True
        )

        if is_high:
            G.nodes[nearest_n]["high_severity_buildings"] = (
                int(G.nodes[nearest_n].get("high_severity_buildings", 0)) + 1
            )

    return G


def rank_choke_point_clearance_roi(
    graph: nx.MultiDiGraph,
    *,
    origin_node: int | str | None = None,
    vehicle_class: str | VehicleClass = VehicleClass.FIRE_ENGINE,
    buildings_source: str | Path | gpd.GeoDataFrame | None = "data/osm_buildings_graded.geojson",
    max_recommendations: int | None = None,
) -> list[ChokePointROIRecommendation]:
    """
    Identify critical debris bottleneck edges where clearing debris unlocks isolated
    civilian building clusters, and rank them strictly by Clearance ROI descending:

        Clearance ROI = (Count of high-severity building structures unlocked)
                        / (Square meters of debris requiring clearance)

    Standing-water edges (`water_overlap_ratio > 0`) are strictly excluded from debris
    clearance ROI because water washouts cannot be resolved by rubble clearance.
    """
    vc = normalize_vehicle_class(vehicle_class)
    required_w = VEHICLE_MIN_WIDTH_M[vc]
    weight_attr = f"weight_{vc}"

    G = annotate_graph_with_geometry_metrics(graph)
    G = assign_buildings_to_graph_nodes(G, buildings_source=buildings_source)

    if origin_node is None:
        # Select origin node as the node in the largest passable component (or Westernmost node)
        origin_node = min(
            G.nodes(),
            key=lambda n: float(G.nodes[n].get("x", 0.0)),
        )

    # 1. Build baseline passable graph for `vc` and find currently reachable nodes from `origin_node`
    passable_edges = [
        (u, v, k)
        for u, v, k, d in G.edges(keys=True, data=True)
        if math.isfinite(float(d.get(weight_attr, float("inf"))))
    ]
    G_base = nx.MultiDiGraph()
    G_base.add_nodes_from(G.nodes(data=True))
    for u, v, k in passable_edges:
        G_base.add_edge(u, v, key=k, **G[u][v][k])

    baseline_reachable: set[int | str] = (
        nx.descendants(G_base, origin_node) | {origin_node}
        if origin_node in G_base
        else set()
    )

    # 2. Identify candidate debris choke-point edges (blocked by debris, NOT by standing water)
    # Group bidirectional edges (u, v, k) and (v, u, k) that represent the same physical road segment
    seen_undirected_keys: set[tuple[Any, Any, int]] = set()
    recommendations: list[ChokePointROIRecommendation] = []

    for u, v, k, data in G.edges(keys=True, data=True):
        water_ratio = float(data.get("water_overlap_ratio", 0.0))
        if water_ratio > 0.0 or data.get("has_water_obstruction"):
            # Water washouts are strictly impassable and cannot be cleared as rubble
            continue

        debris_sqm = float(data.get("debris_area_sqm", 0.0))
        current_w = float(data.get("min_clear_width_m", 8.0))
        restored_w = float(data.get("nominal_width_m", DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M * 2.0))

        # Only consider edges currently blocked for `vc` that have positive debris area
        if current_w >= required_w or debris_sqm <= 0.0 or restored_w < required_w:
            continue

        pair_id = (min(str(u), str(v)), max(str(u), str(v)), int(k))
        if pair_id in seen_undirected_keys:
            continue
        seen_undirected_keys.add(pair_id)

        # Simulate clearing debris on this physical corridor (both u->v and v->u if present)
        G_sim = G_base.copy()
        G_sim.add_edge(u, v, key=k, **data)
        if G.has_edge(v, u, k):
            rev_data = G[v][u][k]
            if float(rev_data.get("water_overlap_ratio", 0.0)) == 0.0:
                G_sim.add_edge(v, u, key=k, **rev_data)

        sim_reachable = (
            nx.descendants(G_sim, origin_node) | {origin_node}
            if origin_node in G_sim
            else set()
        )
        newly_unlocked_nodes = sorted(
            sim_reachable - baseline_reachable,
            key=lambda x: str(x),
        )

        # If the choke point is in a downstream cluster not yet connected to origin_node,
        # also evaluate local cluster unlock from the upstream side `u` of the choke point.
        if not newly_unlocked_nodes and u in G_base:
            base_from_u = nx.descendants(G_base, u) | {u}
            sim_from_u = nx.descendants(G_sim, u) | {u}
            newly_unlocked_nodes = sorted(
                sim_from_u - base_from_u,
                key=lambda x: str(x),
            )

        unlocked_structures = sum(
            int(G.nodes[n].get("high_severity_buildings", 0))
            for n in newly_unlocked_nodes
        )
        if unlocked_structures <= 0:
            continue

        roi = float(unlocked_structures) / float(debris_sqm)
        road_name = _format_edge_name(data.get("name"), u, v, data.get("display_name"))
        gsd = float(data.get("gsd_m_per_px", 0.2804))

        bottleneck_lonlat = data.get("bottleneck_lonlat")
        if not bottleneck_lonlat:
            coords = _extract_edge_lonlat_coords(G, u, v, data)
            mid_idx = len(coords) // 2
            bottleneck_lonlat = coords[mid_idx]

        explanation = (
            f"Clearing {debris_sqm:.2f} m² of debris on {road_name} "
            f"(widening clear span from {current_w:.2f}m to {restored_w:.2f}m at {gsd:.4f} m/px GSD) "
            f"unlocks {VEHICLE_LABEL[vc]} access to {unlocked_structures} high-severity structure(s) "
            f"(ROI: {roi:.4f} structures/m²)."
        )

        sam_conf = float(data.get("sam_confidence", 0.78))
        needs_drone = bool(data.get("needs_drone_recon", sam_conf < 0.70))
        recommendations.append(
            ChokePointROIRecommendation(
                rank=0,
                edge_u=u,
                edge_v=v,
                edge_key=int(k),
                road_name=road_name,
                vehicle_class=vc,
                current_clear_width_m=current_w,
                restored_clear_width_m=restored_w,
                debris_area_sqm=debris_sqm,
                water_overlap_ratio=water_ratio,
                gsd_m_per_px=gsd,
                high_severity_structures_unlocked=unlocked_structures,
                newly_reachable_nodes=newly_unlocked_nodes,
                clearance_roi=roi,
                bottleneck_lonlat=(float(bottleneck_lonlat[0]), float(bottleneck_lonlat[1])),
                explanation=explanation,
                sam_confidence=sam_conf,
                needs_drone_recon=needs_drone,
            )
        )

    # Sort strictly by clearance_roi descending, then structures unlocked descending, then debris ascending
    recommendations.sort(
        key=lambda r: (
            -r.clearance_roi,
            -r.high_severity_structures_unlocked,
            r.debris_area_sqm,
            str(r.edge_u),
            str(r.edge_v),
        )
    )

    if max_recommendations is not None:
        recommendations = recommendations[:max_recommendations]

    for idx, rec in enumerate(recommendations, start=1):
        rec.rank = idx

    return recommendations


def simulate_choke_point_clearance(
    graph: nx.MultiDiGraph,
    edge_u: int | str,
    edge_v: int | str,
    edge_key: int = 0,
) -> nx.MultiDiGraph:
    """
    Return a copy of `graph` where debris on `(edge_u, edge_v, edge_key)` (and its
    reverse direction if present and non-flooded) has been cleared, restoring full
    nominal road width. Standing water edges cannot be cleared.
    """
    G = graph.copy()
    for a, b in ((edge_u, edge_v), (edge_v, edge_u)):
        if G.has_edge(a, b, edge_key):
            edata = G[a][b][edge_key]
            if float(edata.get("water_overlap_ratio", 0.0)) == 0.0:
                nominal_w = float(
                    edata.get("nominal_width_m", DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M * 2.0)
                )
                edata["min_clear_width_m"] = nominal_w
                edata["debris_area_sqm"] = 0.0
                edata["debris_area_m2"] = 0.0

    return annotate_graph_with_geometry_metrics(G)


# A blocked segment where SAM 3.1 saw neither pavement nor debris (sand, canopy, shadow) has no
# measured clearing cost. The planner charges it as this much debris so measured piles are
# preferred, and reports such segments separately rather than inventing a debris figure.
UNMEASURED_SEGMENT_COST_M2 = 50.0


def plan_least_debris_clearance(
    graph: nx.MultiDiGraph,
    origin: int | str,
    destination: int | str,
    vehicle_class: str | VehicleClass,
) -> dict[str, Any] | None:
    """
    Cheapest set of blocked segments to clear so `vehicle_class` can reach `destination`.

    Least-cost path where passable segments cost almost nothing (length only breaks ties) and a
    clearable blocked segment costs its SAM-measured debris area in m². Segments under standing
    water, footways closed to the vehicle, and roads too narrow even when clear are excluded.
    The plan is then checked by clearing those segments and re-running the normal router.
    Returns None when no clearable path exists.
    """
    vc = normalize_vehicle_class(vehicle_class)
    G = annotate_graph_with_geometry_metrics(graph)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for u, v, k, d in G.edges(keys=True, data=True):
        tie_break = float(d["length_m"]) / 1000.0
        if d[f"passable_{vc}"]:
            cost = tie_break
        elif (
            d["has_water_obstruction"]
            or (is_pedestrian_only(d.get("highway")) and vc not in FOOT_ONLY_EXEMPT_VEHICLES)
            or float(d["nominal_width_m"]) < VEHICLE_MIN_WIDTH_M[vc]
        ):
            continue
        else:
            debris = float(d["debris_area_m2"])
            cost = (debris if debris > 0 else UNMEASURED_SEGMENT_COST_M2) + tie_break
        if not H.has_edge(u, v) or H[u][v]["cost"] > cost:
            H.add_edge(u, v, cost=cost, key=k)

    try:
        path = nx.shortest_path(H, origin, destination, weight="cost")
    except nx.NetworkXNoPath:
        return None

    segments: list[dict[str, Any]] = []
    cleared = graph
    for u, v in zip(path, path[1:]):
        k = H[u][v]["key"]
        d = G[u][v][k]
        if d[f"passable_{vc}"]:
            continue
        segments.append(
            {
                "edge": (u, v, k),
                "name": _format_edge_name(d.get("name"), u, v, d.get("display_name")),
                "debris_m2": float(d["debris_area_m2"]),
                "debris_measured": float(d["debris_area_m2"]) > 0,
                "clear_width_now_m": float(d["min_clear_width_m"]),
                "clear_width_after_m": float(d["nominal_width_m"]),
                "needs_drone_recon": bool(d.get("needs_drone_recon")),
            }
        )
        cleared = simulate_choke_point_clearance(cleared, u, v, k)

    route = compute_shortest_route(cleared, origin=origin, destination=destination, vehicle_class=vc)
    return {"segments": segments, "route_after": route}
