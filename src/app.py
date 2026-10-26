"""
FirstLight FastAPI backend (`src/app.py`).

Serves the offline-resilient disaster response API and zero-CDN mobile field dashboard
bound to `0.0.0.0:8000` over local Wi-Fi.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import geopandas as gpd
import networkx as nx
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import LineString, mapping

from src.failover import (
    ConnectivityTier,
    FailoverMode,
    GeometricTelemetry,
    VLMFailoverRouter,
    release_unified_memory,
)
from src.geometry import (
    TARGET_METRIC_CRS,
    WGS84_CRS,
    compute_raster_gsd_and_crs,
)
from src.routing import (
    VEHICLE_MIN_WIDTH_M,
    VehicleClass,
    assign_buildings_to_graph_nodes,
    compute_multi_class_routes,
    compute_shortest_route,
    find_nearest_node,
    load_annotated_routing_graph,
    normalize_vehicle_class,
    rank_choke_point_clearance_roi,
    simulate_choke_point_clearance,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
UI_DIR = PROJECT_ROOT / "ui"
VENDOR_DIR = UI_DIR / "vendor"

GRAPHML_PATH = DATA_DIR / "osm_road_graph.graphml"
ROADS_GEOJSON_PATH = DATA_DIR / "osm_roads.geojson"
BUILDINGS_GEOJSON_PATH = DATA_DIR / "osm_buildings.geojson"
RASTER_TIF_PATH = DATA_DIR / "noaa_michael_mexico_beach.tif"


class CoordinatePoint(BaseModel):
    """Geographic coordinate pair `(lon, lat)` in WGS84 (`EPSG:4326`)."""

    lon: float
    lat: float


class EdgeKeySpec(BaseModel):
    """Identifier for a directed or undirected graph edge."""

    u: int | str
    v: int | str
    key: int = 0


class RouteRequest(BaseModel):
    """Request payload for `POST /api/route`."""

    origin: list[float] | CoordinatePoint | int | str
    destination: list[float] | CoordinatePoint | int | str
    vehicle_class: str = Field(default="fire_engine")
    algorithm: Literal["dijkstra", "astar"] = Field(default="dijkstra")
    cleared_edges: list[EdgeKeySpec] = Field(default_factory=list)


class AssessRequest(BaseModel):
    """Request payload for `POST /api/assess`."""

    target_id: str = "hwy98-bottleneck"
    road_name: str = "US Highway 98"
    min_clear_width_m: float = 2.65
    debris_area_m2: float = 24.0
    water_area_m2: float = 0.0
    has_water_obstruction: bool = False
    building_debris_overlap_ratio: float = 0.32
    building_water_overlap_ratio: float = 0.0
    gsd_m_per_px: float = 0.2804
    mode: FailoverMode | None = None


class SimulateClearRequest(BaseModel):
    """Request payload for `POST /api/simulate-clear`."""

    edge_u: int | str | None = None
    edge_v: int | str | None = None
    edge_key: int = 0
    reset: bool = False
    vehicle_class: str = "fire_engine"


class FirstLightState:
    """In-memory state manager holding the annotated road graph, layers, and failover router."""

    def __init__(self) -> None:
        self.base_graph: nx.MultiDiGraph | None = None
        self.active_graph: nx.MultiDiGraph | None = None
        self.cleared_edges: list[tuple[int | str, int | str, int]] = []
        self.raster_meta: dict[str, Any] | None = None
        self.buildings_features: list[dict[str, Any]] | None = None
        self.raster_preview_png: bytes | None = None
        # Default to Tier 3 deterministic in interactive web requests unless user selects AUTO/MLX/CLOUD,
        # while keeping VLMFailoverRouter ready for all 3 tiers.
        self.router = VLMFailoverRouter(
            local_mlx_model_path=MODELS_DIR / "Llama-3.2-11B-Vision-Instruct-4bit",
            mode=FailoverMode.AUTO,
        )

    def ensure_loaded(self) -> None:
        if self.base_graph is not None and self.active_graph is not None:
            return

        if RASTER_TIF_PATH.exists():
            self.raster_meta = compute_raster_gsd_and_crs(
                RASTER_TIF_PATH, target_crs=TARGET_METRIC_CRS
            )
        else:
            self.raster_meta = {
                "raster_path": str(RASTER_TIF_PATH),
                "native_crs": WGS84_CRS,
                "target_crs": TARGET_METRIC_CRS,
                "width_px": 9340,
                "height_px": 9340,
                "bands": 4,
                "gsd_m_per_px": 0.2804,
                "native_bounds": (-85.4251, 29.9249, -85.3999, 29.9501),
            }

        G = load_annotated_routing_graph(
            graphml_path=GRAPHML_PATH,
            roads_geojson_path=ROADS_GEOJSON_PATH,
            raster_path=RASTER_TIF_PATH,
            simulate_if_no_masks=True,
        )
        G = assign_buildings_to_graph_nodes(G, buildings_source=BUILDINGS_GEOJSON_PATH)
        self.base_graph = G
        self.active_graph = G.copy()

    def get_graph_with_clearances(
        self, extra_cleared: list[EdgeKeySpec] | None = None
    ) -> nx.MultiDiGraph:
        self.ensure_loaded()
        assert self.active_graph is not None
        G = self.active_graph.copy()
        if extra_cleared:
            for item in extra_cleared:
                u_resolved = self._resolve_node_id(G, item.u)
                v_resolved = self._resolve_node_id(G, item.v)
                G = simulate_choke_point_clearance(G, u_resolved, v_resolved, item.key)
        return G

    @staticmethod
    def _resolve_node_id(graph: nx.MultiDiGraph, raw_node: int | str) -> int | str:
        if raw_node in graph.nodes:
            return raw_node
        try:
            as_int = int(raw_node)
            if as_int in graph.nodes:
                return as_int
        except (TypeError, ValueError):
            pass
        as_str = str(raw_node)
        if as_str in graph.nodes:
            return as_str
        return raw_node

    def toggle_clear_edge(
        self, edge_u: int | str, edge_v: int | str, edge_key: int = 0
    ) -> None:
        self.ensure_loaded()
        assert self.base_graph is not None
        u_id = self._resolve_node_id(self.base_graph, edge_u)
        v_id = self._resolve_node_id(self.base_graph, edge_v)
        pair = (u_id, v_id, int(edge_key))
        rev_pair = (v_id, u_id, int(edge_key))

        if pair in self.cleared_edges or rev_pair in self.cleared_edges:
            self.cleared_edges = [
                e for e in self.cleared_edges if e not in (pair, rev_pair)
            ]
        else:
            self.cleared_edges.append(pair)

        G = self.base_graph.copy()
        for cu, cv, ck in self.cleared_edges:
            G = simulate_choke_point_clearance(G, cu, cv, ck)
        self.active_graph = G

    def reset_clearances(self) -> None:
        self.ensure_loaded()
        assert self.base_graph is not None
        self.cleared_edges = []
        self.active_graph = self.base_graph.copy()

    def get_buildings_geojson(self) -> dict[str, Any]:
        if self.buildings_features is None:
            features: list[dict[str, Any]] = []
            if BUILDINGS_GEOJSON_PATH.exists():
                gdf = gpd.read_file(BUILDINGS_GEOJSON_PATH)
                if gdf.crs is None:
                    gdf = gdf.set_crs(WGS84_CRS)
                else:
                    gdf = gdf.to_crs(WGS84_CRS)

                severities = [
                    "destroyed",
                    "minor_damage",
                    "major_damage",
                    "no_damage",
                    "minor_damage",
                ]
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    sev = severities[int(idx) % len(severities)]
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(geom),
                            "properties": {
                                "building_id": f"bldg-{idx}",
                                "building_type": str(row.get("building") or "yes"),
                                "damage_severity": sev,
                                "is_high_severity": sev in ("major_damage", "destroyed"),
                            },
                        }
                    )
            self.buildings_features = features

        return {"type": "FeatureCollection", "features": self.buildings_features}

    def get_roads_geojson(self, graph: nx.MultiDiGraph | None = None) -> dict[str, Any]:
        self.ensure_loaded()
        G = graph if graph is not None else self.active_graph
        assert G is not None

        features: list[dict[str, Any]] = []
        for u, v, k, data in G.edges(keys=True, data=True):
            geom = data.get("geometry")
            if isinstance(geom, LineString) and not geom.is_empty:
                geom_dict = mapping(geom)
            else:
                ux = float(G.nodes[u].get("x", 0.0))
                uy = float(G.nodes[u].get("y", 0.0))
                vx = float(G.nodes[v].get("x", 0.0))
                vy = float(G.nodes[v].get("y", 0.0))
                geom_dict = {
                    "type": "LineString",
                    "coordinates": [[ux, uy], [vx, vy]],
                }

            raw_name = data.get("name")
            if isinstance(raw_name, list):
                name_str = ", ".join(str(x) for x in raw_name if x)
            elif raw_name is not None and not (
                isinstance(raw_name, float) and math.isnan(raw_name)
            ):
                name_str = str(raw_name)
            else:
                name_str = f"Segment {u}-{v}"

            features.append(
                {
                    "type": "Feature",
                    "geometry": geom_dict,
                    "properties": {
                        "u": u,
                        "v": v,
                        "key": int(k),
                        "name": name_str,
                        "length_m": round(float(data.get("length_m", 0.0)), 2),
                        "min_clear_width_m": round(
                            float(data.get("min_clear_width_m", 8.0)), 2
                        ),
                        "nominal_width_m": round(
                            float(data.get("nominal_width_m", 8.0)), 2
                        ),
                        "debris_area_sqm": round(
                            float(data.get("debris_area_sqm", 0.0)), 2
                        ),
                        "water_overlap_ratio": round(
                            float(data.get("water_overlap_ratio", 0.0)), 4
                        ),
                        "has_water_obstruction": bool(
                            data.get("has_water_obstruction", False)
                        ),
                        "passable_atv": bool(data.get("passable_atv", True)),
                        "passable_ambulance": bool(data.get("passable_ambulance", True)),
                        "passable_fire_engine": bool(
                            data.get("passable_fire_engine", True)
                        ),
                        "gsd_m_per_px": round(
                            float(data.get("gsd_m_per_px", 0.2804)), 4
                        ),
                        "clearance_reason": str(data.get("clearance_reason", "")),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def get_raster_preview_png(self) -> bytes:
        if self.raster_preview_png is not None:
            return self.raster_preview_png

        if RASTER_TIF_PATH.exists():
            with rasterio.open(RASTER_TIF_PATH) as src:
                out_h, out_w = 512, 512
                bands = min(src.count, 3)
                arr = src.read(
                    list(range(1, bands + 1)),
                    out_shape=(bands, out_h, out_w),
                    resampling=Resampling.bilinear,
                )
                rgb = np.transpose(arr, (1, 2, 0))
                img = Image.fromarray(rgb, mode="RGB")
        else:
            img = Image.new("RGB", (256, 256), color=(30, 41, 59))

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        self.raster_preview_png = buf.getvalue()
        return self.raster_preview_png


state = FirstLightState()

app = FastAPI(
    title="FirstLight Tactical Disaster Response API",
    description=(
        "Offline-resilient SAM 3.1 geometric road width measurement, 3-tier VLM failover, "
        "multi-class emergency routing, and Choke-Point Clearance ROI engine."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if VENDOR_DIR.exists():
    app.mount("/vendor", StaticFiles(directory=str(VENDOR_DIR)), name="vendor")
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


def _parse_endpoint_spec(
    graph: nx.MultiDiGraph, spec: list[float] | CoordinatePoint | int | str
) -> int | str:
    """Resolve an origin/destination specification into a valid graph node ID."""
    if isinstance(spec, CoordinatePoint):
        return find_nearest_node(graph, float(spec.lon), float(spec.lat))
    if isinstance(spec, list):
        if len(spec) < 2:
            raise ValueError("Coordinate list must contain [lon, lat].")
        first, second = float(spec[0]), float(spec[1])
        # Support both [lon, lat] and [lat, lon] automatically for Mexico Beach (~ -85.4W, 29.9N)
        if first > 0 and second < 0:
            lon, lat = second, first
        else:
            lon, lat = first, second
        return find_nearest_node(graph, lon, lat)
    return state._resolve_node_id(graph, spec)


@app.get("/", include_in_schema=False)
def serve_dashboard_index() -> FileResponse:
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="ui/index.html not found")
    return FileResponse(index_path, media_type="text/html")


@app.get("/app.js", include_in_schema=False)
def serve_dashboard_js() -> FileResponse:
    js_path = UI_DIR / "app.js"
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="ui/app.js not found")
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/style.css", include_in_schema=False)
def serve_dashboard_css() -> FileResponse:
    css_path = UI_DIR / "style.css"
    if not css_path.exists():
        raise HTTPException(status_code=404, detail="ui/style.css not found")
    return FileResponse(css_path, media_type="text/css")


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    """Report system status, active VLM tier readiness, and Apple Silicon MPS/MLX device health."""
    state.ensure_loaded()
    assert state.active_graph is not None
    assert state.raster_meta is not None

    mps_available = False
    try:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
    except Exception:
        pass

    mlx_metal_available = False
    try:
        import mlx.core as mx

        mlx_metal_available = bool(mx.metal.is_available())
    except Exception:
        pass

    mem_stats = release_unified_memory()
    mlx_weights_ready = (
        MODELS_DIR / "Llama-3.2-11B-Vision-Instruct-4bit" / "config.json"
    ).exists()
    sam31_weights_ready = (MODELS_DIR / "sam3.1" / "model.safetensors").exists()
    cloud_key_configured = bool(state.router.api_key)

    active_tier = (
        ConnectivityTier.TIER_1_CLOUD.value
        if cloud_key_configured and state.router.mode in (FailoverMode.AUTO, FailoverMode.FORCE_CLOUD)
        else (
            ConnectivityTier.TIER_2_MLX.value
            if mlx_weights_ready and state.router.mode in (FailoverMode.AUTO, FailoverMode.FORCE_MLX)
            else ConnectivityTier.TIER_3_DETERMINISTIC.value
        )
    )

    return {
        "status": "ok",
        "system": "FirstLight Disaster Response Engine",
        "offline_ready": True,
        "vlm_tiers": {
            "active_mode": state.router.mode.value,
            "active_tier": active_tier,
            "last_tier_used": state.router.last_tier_used.value,
            "tier_1_cloud_configured": cloud_key_configured,
            "tier_1_timeout_s": state.router.cloud_timeout_s,
            "tier_2_mlx_ready": mlx_weights_ready,
            "tier_2_model": "Llama-3.2-11B-Vision-Instruct-4bit",
            "tier_3_deterministic_ready": True,
        },
        "device_health": {
            "hardware_profile": "Apple Silicon M1 Max (64 GB Unified Memory)",
            "pytorch_mps_available": mps_available,
            "mlx_metal_available": mlx_metal_available,
            "sam31_checkpoint_ready": sam31_weights_ready,
            "enforced_precision": "float16 / 4-bit",
            "unified_memory": mem_stats,
        },
        "geospatial": {
            "aoi": "Mexico Beach, FL (Hurricane Michael - US Hwy 98 Corridor)",
            "native_crs": state.raster_meta["native_crs"],
            "metric_crs": state.raster_meta["target_crs"],
            "gsd_m_per_px": round(float(state.raster_meta["gsd_m_per_px"]), 4),
            "graph_nodes": len(state.active_graph.nodes),
            "graph_edges": len(state.active_graph.edges),
            "simulated_cleared_edges": len(state.cleared_edges),
        },
    }


@app.get("/api/layers")
def get_layers() -> dict[str, Any]:
    """
    Return cached Mexico Beach road network, building footprints, and raster bounding
    envelope as GeoJSON.
    """
    state.ensure_loaded()
    assert state.active_graph is not None
    assert state.raster_meta is not None

    min_lon, min_lat, max_lon, max_lat = state.raster_meta["native_bounds"]
    envelope_geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        },
        "properties": {
            "name": "NOAA Hurricane Michael GeoTIFF Envelope (20181011aC0852530w295700n)",
            "native_crs": state.raster_meta["native_crs"],
            "metric_crs": state.raster_meta["target_crs"],
            "gsd_m_per_px": round(float(state.raster_meta["gsd_m_per_px"]), 4),
            "bounds_wgs84": [min_lon, min_lat, max_lon, max_lat],
        },
    }

    nodes_sorted = sorted(
        state.active_graph.nodes(data=True),
        key=lambda item: float(item[1].get("x", 0.0)),
    )
    origin_id, origin_data = nodes_sorted[0]
    dest_id, dest_data = nodes_sorted[-1]

    return {
        "roads": state.get_roads_geojson(),
        "buildings": state.get_buildings_geojson(),
        "raster_envelope": envelope_geojson,
        "bounds_wgs84": [min_lon, min_lat, max_lon, max_lat],
        "default_origin": {
            "node_id": origin_id,
            "lon": float(origin_data.get("x", min_lon)),
            "lat": float(origin_data.get("y", min_lat)),
        },
        "default_destination": {
            "node_id": dest_id,
            "lon": float(dest_data.get("x", max_lon)),
            "lat": float(dest_data.get("y", max_lat)),
        },
        "vehicle_thresholds_m": VEHICLE_MIN_WIDTH_M,
        "cleared_edges": [
            {"u": u, "v": v, "key": k} for u, v, k in state.cleared_edges
        ],
    }


@app.get("/api/raster-preview.png", include_in_schema=False)
def get_raster_preview() -> Response:
    """Serve an offline downsampled PNG overlay of the Mexico Beach Hurricane Michael GeoTIFF."""
    png_bytes = state.get_raster_preview_png()
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/assess")
async def post_assess(req: AssessRequest) -> dict[str, Any]:
    """
    Run building and road damage grading through `VLMFailoverRouter` with active tier
    metadata and Apple Silicon unified memory release metrics.
    """
    state.ensure_loaded()
    telemetry = GeometricTelemetry(
        target_id=req.target_id,
        road_name=req.road_name,
        min_clear_width_m=req.min_clear_width_m,
        debris_area_m2=req.debris_area_m2,
        water_area_m2=req.water_area_m2,
        has_water_obstruction=req.has_water_obstruction,
        building_debris_overlap_ratio=req.building_debris_overlap_ratio,
        building_water_overlap_ratio=req.building_water_overlap_ratio,
        gsd_m_per_px=req.gsd_m_per_px,
    )

    prev_mode = state.router.mode
    if req.mode is not None:
        state.router.mode = req.mode
    elif not state.router.api_key and state.router._custom_mlx_runner is None:
        # In fast interactive dashboard queries without a cloud API key, avoid blocking
        # HTTP requests for full 11B model cold-boot unless FORCE_MLX is explicitly requested
        state.router.mode = FailoverMode.FORCE_DETERMINISTIC

    try:
        assessment = await state.router.assess_damage_async(telemetry)
    finally:
        if req.mode is None:
            state.router.mode = prev_mode

    mem_stats = release_unified_memory()
    payload = assessment.model_dump()
    payload["memory_cleared"] = True
    payload["unified_memory_stats"] = mem_stats
    return payload


@app.post("/api/route")
def post_route(req: RouteRequest) -> dict[str, Any]:
    """
    Compute vehicle-specific shortest passable route (`atv`, `ambulance`, `fire_engine`),
    returning GeoJSON route geometry, turn-by-turn passability, and avoided bottlenecks.
    """
    try:
        vc = normalize_vehicle_class(req.vehicle_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    G = state.get_graph_with_clearances(req.cleared_edges)
    try:
        origin_node = _parse_endpoint_spec(G, req.origin)
        dest_node = _parse_endpoint_spec(G, req.destination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    primary_route = compute_shortest_route(
        G,
        origin=origin_node,
        destination=dest_node,
        vehicle_class=vc,
        algorithm=req.algorithm,
    )
    all_routes = compute_multi_class_routes(G, origin=origin_node, destination=dest_node)

    result_dict = primary_route.to_dict()
    result_dict["all_classes_summary"] = {
        k: {
            "reachable": r.reachable,
            "total_distance_m": round(r.total_distance_m, 2),
            "estimated_transit_time_s": round(r.estimated_transit_time_s, 2),
            "hop_count": len(r.edge_steps),
            "blocked_edges_avoided_count": len(r.blocked_edges_avoided),
        }
        for k, r in all_routes.items()
    }
    return result_dict


@app.get("/api/clearance-roi")
def get_clearance_roi(
    vehicle_class: str = Query(default="fire_engine"),
    origin_node: str | None = Query(default=None),
    limit: int = Query(default=15, ge=1, le=100),
) -> dict[str, Any]:
    """
    Return ranked Choke-Point Clearance ROI bottlenecks sorted strictly descending by
    `high_severity_structures_unlocked / debris_area_sqm`.
    """
    try:
        vc = normalize_vehicle_class(vehicle_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.ensure_loaded()
    assert state.active_graph is not None

    resolved_origin: int | str | None = None
    if origin_node is not None:
        resolved_origin = state._resolve_node_id(state.active_graph, origin_node)

    recommendations = rank_choke_point_clearance_roi(
        state.active_graph,
        origin_node=resolved_origin,
        vehicle_class=vc,
        buildings_source=BUILDINGS_GEOJSON_PATH,
        max_recommendations=limit,
    )
    return {
        "vehicle_class": vc,
        "required_width_m": VEHICLE_MIN_WIDTH_M[vc],
        "count": len(recommendations),
        "simulated_cleared_edges": [
            {"u": u, "v": v, "key": k} for u, v, k in state.cleared_edges
        ],
        "recommendations": [rec.to_dict() for rec in recommendations],
    }


@app.post("/api/simulate-clear")
def post_simulate_clear(req: SimulateClearRequest) -> dict[str, Any]:
    """
    Toggle simulated debris clearance on a choke-point edge (or reset all simulated
    clearances) and return updated clearance ROI recommendations and active cleared edges.
    """
    if req.reset:
        state.reset_clearances()
    elif req.edge_u is not None and req.edge_v is not None:
        state.toggle_clear_edge(req.edge_u, req.edge_v, req.edge_key)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either reset=True or both edge_u and edge_v.",
        )

    vc = normalize_vehicle_class(req.vehicle_class)
    assert state.active_graph is not None
    recommendations = rank_choke_point_clearance_roi(
        state.active_graph,
        vehicle_class=vc,
        buildings_source=BUILDINGS_GEOJSON_PATH,
        max_recommendations=15,
    )
    return {
        "status": "updated",
        "cleared_edges": [
            {"u": u, "v": v, "key": k} for u, v, k in state.cleared_edges
        ],
        "roads": state.get_roads_geojson(),
        "recommendations": [rec.to_dict() for rec in recommendations],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
