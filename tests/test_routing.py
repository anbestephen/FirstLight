"""
Unit and integration tests for Milestone 3: Multi-class NetworkX vehicle routing
and Choke-Point Clearance ROI engine (`src/routing.py`).
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pytest
from shapely.geometry import LineString

from src.routing import (
    VehicleClass,
    annotate_graph_with_geometry_metrics,
    compute_edge_routing_weight,
    compute_multi_class_routes,
    compute_shortest_route,
    is_edge_passable_for_vehicle,
    load_annotated_routing_graph,
    plan_least_debris_clearance,
    rank_choke_point_clearance_roi,
    simulate_choke_point_clearance,
)


@pytest.fixture()
def differential_routing_graph() -> nx.MultiDiGraph:
    """
    Synthetic 5-node road network with three parallel corridors from Node 1 (Staging)
    to Node 4 (Target):
      - Corridor A (1 -> 2 -> 4): 100m total, min_clear_width_m = 1.6m (ATV only)
      - Corridor B (1 -> 3 -> 4): 220m total, min_clear_width_m = 2.6m (ATV & Ambulance)
      - Corridor C (1 -> 5 -> 4): 400m total, min_clear_width_m = 5.2m (All classes incl. Fire Engine)
    """
    G = nx.MultiDiGraph()
    coords = {
        1: (-85.4200, 29.9400),
        2: (-85.4150, 29.9410),
        3: (-85.4150, 29.9430),
        4: (-85.4100, 29.9400),
        5: (-85.4150, 29.9460),
    }
    for nid, (lon, lat) in coords.items():
        G.add_node(nid, x=lon, y=lat, high_severity_buildings=0)

    # Corridor A: 1 -> 2 -> 4 (50m + 50m = 100m, width = 1.6m)
    G.add_edge(
        1,
        2,
        key=0,
        name="Alley A1",
        length_m=50.0,
        min_clear_width_m=1.6,
        debris_area_sqm=18.0,
        water_overlap_ratio=0.0,
        geometry=LineString([coords[1], coords[2]]),
    )
    G.add_edge(
        2,
        4,
        key=0,
        name="Alley A2",
        length_m=50.0,
        min_clear_width_m=4.5,
        debris_area_sqm=0.0,
        water_overlap_ratio=0.0,
        geometry=LineString([coords[2], coords[4]]),
    )

    # Corridor B: 1 -> 3 -> 4 (110m + 110m = 220m, width = 2.6m)
    G.add_edge(
        1,
        3,
        key=0,
        name="Canal Side Rd",
        length_m=110.0,
        min_clear_width_m=2.6,
        debris_area_sqm=12.5,
        water_overlap_ratio=0.0,
        geometry=LineString([coords[1], coords[3]]),
    )
    G.add_edge(
        3,
        4,
        key=0,
        name="Canal Connector",
        length_m=110.0,
        min_clear_width_m=4.8,
        debris_area_sqm=0.0,
        water_overlap_ratio=0.0,
        geometry=LineString([coords[3], coords[4]]),
    )

    # Corridor C: 1 -> 5 -> 4 (200m + 200m = 400m, width = 5.2m)
    G.add_edge(
        1,
        5,
        key=0,
        name="Highway 98 Bypass W",
        length_m=200.0,
        min_clear_width_m=5.2,
        debris_area_sqm=0.0,
        water_overlap_ratio=0.0,
        geometry=LineString([coords[1], coords[5]]),
    )
    G.add_edge(
        5,
        4,
        key=0,
        name="Highway 98 Bypass E",
        length_m=200.0,
        min_clear_width_m=5.2,
        debris_area_sqm=0.0,
        water_overlap_ratio=0.0,
        geometry=LineString([coords[5], coords[4]]),
    )

    return annotate_graph_with_geometry_metrics(G)


def test_differential_multi_class_routing(
    differential_routing_graph: nx.MultiDiGraph,
) -> None:
    """
    Verify differential routing across vehicle classes:
    - ATV (>=1.2m) selects 100m path [1, 2, 4]
    - Ambulance (>=2.2m) selects 220m path [1, 3, 4] (blocked on [1, 2, 4])
    - Fire Engine (>=3.5m) selects 400m path [1, 5, 4] (blocked on [1, 2, 4] and [1, 3, 4])
    """
    routes = compute_multi_class_routes(differential_routing_graph, origin=1, destination=4)

    atv_route = routes["atv"]
    amb_route = routes["ambulance"]
    fire_route = routes["fire_engine"]

    assert atv_route.reachable is True
    assert atv_route.node_path == [1, 2, 4]
    assert math.isclose(atv_route.total_distance_m, 100.0, rel_tol=1e-5)
    assert atv_route.geojson_geometry["type"] == "LineString"
    assert len(atv_route.geojson_geometry["coordinates"]) == 3
    assert len(atv_route.edge_steps) == 2

    assert amb_route.reachable is True
    assert amb_route.node_path == [1, 3, 4]
    assert math.isclose(amb_route.total_distance_m, 220.0, rel_tol=1e-5)
    assert len(amb_route.blocked_edges_avoided) == 1
    assert amb_route.blocked_edges_avoided[0]["name"] == "Alley A1"

    assert fire_route.reachable is True
    assert fire_route.node_path == [1, 5, 4]
    assert math.isclose(fire_route.total_distance_m, 400.0, rel_tol=1e-5)
    assert len(fire_route.blocked_edges_avoided) == 1

    # Also verify A* produces the identical node path and distance for Fire Engine
    fire_astar = compute_shortest_route(
        differential_routing_graph, 1, 4, vehicle_class=VehicleClass.FIRE_ENGINE, algorithm="astar"
    )
    assert fire_astar.node_path == [1, 5, 4]
    assert math.isclose(fire_astar.total_distance_m, 400.0, rel_tol=1e-5)


def test_standing_water_strict_infinite_weight_enforcement() -> None:
    """
    Verify any edge with water_overlap_ratio > 0 receives weight = float('inf')
    and is strictly impassable for all vehicle classes, regardless of width.
    """
    G = nx.MultiDiGraph()
    G.add_node("A", x=-85.42, y=29.94)
    G.add_node("B", x=-85.41, y=29.94)
    # 10.0m wide highway, 0 debris, but 4% standing water overlap
    G.add_edge(
        "A",
        "B",
        key=0,
        name="Flooded Coastal Hwy",
        length_m=150.0,
        min_clear_width_m=10.0,
        debris_area_sqm=0.0,
        water_overlap_ratio=0.04,
    )
    G = annotate_graph_with_geometry_metrics(G)
    edata = G["A"]["B"][0]

    for vc in ("atv", "ambulance", "fire_engine"):
        assert is_edge_passable_for_vehicle(10.0, 0.04, vc) is False
        assert math.isinf(compute_edge_routing_weight(150.0, 10.0, 0.04, vc))
        assert edata[f"passable_{vc}"] is False
        assert math.isinf(edata[f"weight_{vc}"])

        route = compute_shortest_route(G, "A", "B", vehicle_class=vc)
        assert route.reachable is False
        assert route.node_path == []

    # Verify simulating debris clearance does NOT make a flooded edge passable
    G_cleared = simulate_choke_point_clearance(G, "A", "B", 0)
    assert math.isinf(G_cleared["A"]["B"][0]["weight_fire_engine"])


def test_choke_point_clearance_roi_deterministic_ranking() -> None:
    """
    Verify Choke-Point Clearance ROI engine ranks debris bottlenecks strictly by
    (high_severity_structures_unlocked / debris_area_sqm) descending and excludes
    standing-water edges.
    """
    G = nx.MultiDiGraph()
    G.add_node(0, x=-85.425, y=29.940, high_severity_buildings=0)  # Staging base

    # Cluster 10: 10 high-severity buildings behind 20.0 m² debris -> ROI = 0.5000
    G.add_node(10, x=-85.420, y=29.941, high_severity_buildings=10)
    G.add_edge(
        0,
        10,
        key=0,
        name="Oak Ave Bridge",
        length_m=80.0,
        nominal_width_m=8.0,
        min_clear_width_m=2.0,  # Blocked for Fire Engine (>=3.5m)
        debris_area_sqm=20.0,
        water_overlap_ratio=0.0,
        gsd_m_per_px=0.2804,
    )

    # Cluster 20: 15 high-severity buildings behind 10.0 m² debris -> ROI = 1.5000 (Rank #1)
    G.add_node(20, x=-85.420, y=29.944, high_severity_buildings=15)
    G.add_edge(
        0,
        20,
        key=0,
        name="Pine St Causeway",
        length_m=95.0,
        nominal_width_m=8.0,
        min_clear_width_m=1.8,  # Blocked for Fire Engine (>=3.5m)
        debris_area_sqm=10.0,
        water_overlap_ratio=0.0,
        gsd_m_per_px=0.2804,
    )

    # Cluster 30: 5 high-severity buildings behind 25.0 m² debris -> ROI = 0.2000 (Rank #3)
    G.add_node(30, x=-85.420, y=29.947, high_severity_buildings=5)
    G.add_edge(
        0,
        30,
        key=0,
        name="Cedar Ln Access",
        length_m=110.0,
        nominal_width_m=8.0,
        min_clear_width_m=2.4,  # Blocked for Fire Engine (>=3.5m)
        debris_area_sqm=25.0,
        water_overlap_ratio=0.0,
        gsd_m_per_px=0.2804,
    )

    # Cluster 40: 50 high-severity buildings behind flooded road -> Must be excluded!
    G.add_node(40, x=-85.420, y=29.949, high_severity_buildings=50)
    G.add_edge(
        0,
        40,
        key=0,
        name="Flooded Beach Blvd",
        length_m=70.0,
        nominal_width_m=8.0,
        min_clear_width_m=0.0,
        debris_area_sqm=5.0,
        water_overlap_ratio=0.35,
        gsd_m_per_px=0.2804,
    )

    ranked = rank_choke_point_clearance_roi(
        G,
        origin_node=0,
        vehicle_class=VehicleClass.FIRE_ENGINE,
        buildings_source=None,
    )

    assert len(ranked) == 3
    assert [r.road_name for r in ranked] == [
        "Pine St Causeway",
        "Oak Ave Bridge",
        "Cedar Ln Access",
    ]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert math.isclose(ranked[0].clearance_roi, 1.50, rel_tol=1e-5)
    assert math.isclose(ranked[1].clearance_roi, 0.50, rel_tol=1e-5)
    assert math.isclose(ranked[2].clearance_roi, 0.20, rel_tol=1e-5)

    # Verify explanation string is mathematically grounded in exact metrics
    top = ranked[0]
    assert "10.00 m²" in top.explanation
    assert "1.80m to 8.00m" in top.explanation
    assert "0.2804 m/px GSD" in top.explanation
    assert "15 high-severity structure(s)" in top.explanation
    assert "1.5000 structures/m²" in top.explanation

    # Verify simulating clearance on Rank #1 unlocks Fire Engine route to Node 20
    before_route = compute_shortest_route(G, 0, 20, vehicle_class=VehicleClass.FIRE_ENGINE)
    assert before_route.reachable is False

    G_after = simulate_choke_point_clearance(G, top.edge_u, top.edge_v, top.edge_key)
    after_route = compute_shortest_route(G_after, 0, 20, vehicle_class=VehicleClass.FIRE_ENGINE)
    assert after_route.reachable is True
    assert after_route.node_path == [0, 20]


def test_real_mexico_beach_osm_graph_routing_and_roi() -> None:
    """
    End-to-end integration test on `data/osm_road_graph.graphml` and `data/osm_buildings.geojson`.
    """
    graphml_path = Path("data/osm_road_graph.graphml")
    assert graphml_path.exists()

    G = load_annotated_routing_graph(graphml_path=graphml_path)
    assert len(G.nodes) == 125
    assert len(G.edges) == 312

    # Pick westernmost and easternmost nodes in the largest ATV-passable component
    nodes_by_lon = sorted(G.nodes(), key=lambda n: float(G.nodes[n].get("x", 0.0)))
    origin = nodes_by_lon[0]
    destination = nodes_by_lon[-1]

    multi_routes = compute_multi_class_routes(G, origin=origin, destination=destination)
    assert set(multi_routes.keys()) == {"atv", "ambulance", "fire_engine"}
    for vc, res in multi_routes.items():
        assert res.vehicle_class == vc
        assert isinstance(res.geojson_geometry, dict)
        assert res.geojson_geometry["type"] == "LineString"

    # Verify Choke-Point Clearance ROI ranking executes cleanly and sorts descending
    roi_list = rank_choke_point_clearance_roi(
        G,
        origin_node=origin,
        vehicle_class=VehicleClass.FIRE_ENGINE,
        buildings_source="data/osm_buildings.geojson",
    )
    assert len(roi_list) > 0
    for i in range(len(roi_list) - 1):
        assert roi_list[i].clearance_roi >= roi_list[i + 1].clearance_roi
        assert roi_list[i].water_overlap_ratio == 0.0


def test_ambulances_and_fire_engines_never_use_footways() -> None:
    import networkx as nx

    from src.routing import annotate_graph_with_geometry_metrics

    G = nx.MultiDiGraph()
    G.add_edge(1, 2, key=0, length_m=40.0, min_clear_width_m=6.0, highway="footway")
    G.add_edge(2, 3, key=0, length_m=40.0, min_clear_width_m=6.0, highway="residential")
    G.add_edge(3, 4, key=0, length_m=40.0, min_clear_width_m=6.0, highway=["track", "residential"])
    A = annotate_graph_with_geometry_metrics(G)
    foot = A[1][2][0]
    assert foot["passable_atv"] is True
    assert foot["passable_ambulance"] is False and foot["passable_fire_engine"] is False
    assert A[2][3][0]["passable_fire_engine"] is True
    assert A[3][4][0]["passable_ambulance"] is True  # mixed types are drivable


def test_unnamed_segments_get_names_from_connected_streets() -> None:
    import networkx as nx

    from src.routing import _format_edge_name, assign_display_names

    G = nx.MultiDiGraph()
    G.add_edge(1, 2, key=0, name="26th Street")
    G.add_edge(2, 3, key=0)  # unnamed, touches 26th Street at node 2
    G.add_edge(3, 4, key=0, name="US Highway 98")
    G.add_edge(5, 6, key=0)  # isolated, no named neighbours
    G.add_edge(2, 7, key=0)  # unnamed spur off 26th Street only
    assign_display_names(G)
    assert G[2][3][0]["display_name"] == "unnamed road between 26th Street and US Highway 98"
    assert G[2][7][0]["display_name"] == "unnamed road off 26th Street"
    assert G[5][6][0]["display_name"] == "unnamed road"
    assert _format_edge_name(None, 2, 3, G[2][3][0]["display_name"]).startswith("unnamed road between")
    assert _format_edge_name("26th Street", 1, 2) == "26th Street"


def test_clearance_plan_picks_least_debris_among_clearable_ways_in() -> None:
    """Origin 1 -> target 4; every way in is blocked for an ambulance (needs 2.2 m)."""
    G = nx.MultiDiGraph()
    coords = {1: (-85.42, 29.94), 2: (-85.415, 29.941), 3: (-85.415, 29.943), 4: (-85.41, 29.94),
              5: (-85.415, 29.946), 6: (-85.415, 29.948)}
    for nid, (lon, lat) in coords.items():
        G.add_node(nid, x=lon, y=lat)

    def road(u, v, name, width, debris, **extra):
        attrs = {"water_overlap_ratio": 0.0, "nominal_width_m": 7.0, **extra}
        G.add_edge(u, v, key=0, name=name, length_m=100.0, min_clear_width_m=width, debris_area_sqm=debris,
                   geometry=LineString([coords[u], coords[v]]), **attrs)

    road(1, 2, "Debris Rd", 0.0, 40.0)          # clearable, 40 m²
    road(2, 4, "Debris Rd", 7.0, 0.0)
    road(1, 3, "Flooded Rd", 0.0, 10.0, water_overlap_ratio=0.2)   # water cannot be cleared
    road(3, 4, "Flooded Rd", 7.0, 0.0)
    road(1, 5, "Beach Path", 0.0, 5.0, highway="footway")          # ambulances stay off footways
    road(5, 4, "Beach Path", 7.0, 0.0)
    road(1, 6, "Narrow Lane", 0.0, 15.0)                           # too narrow even when clear
    G[1][6][0]["nominal_width_m"] = 1.8
    road(6, 4, "Narrow Lane", 7.0, 0.0)

    plan = plan_least_debris_clearance(G, 1, 4, "ambulance")
    assert [s["name"] for s in plan["segments"]] == ["Debris Rd"]
    assert plan["segments"][0]["debris_m2"] == 40.0
    assert plan["route_after"].reachable

    # An ATV may use the footway, where only 5 m² blocks it.
    atv = plan_least_debris_clearance(G, 1, 4, "atv")
    assert [s["name"] for s in atv["segments"]] == ["Beach Path"]

    # With the only clearable road gone, no plan exists.
    G.remove_edge(1, 2)
    assert plan_least_debris_clearance(G, 1, 4, "ambulance") is None
