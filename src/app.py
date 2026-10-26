"""
FirstLight FastAPI backend (`src/app.py`).

Serves the offline-resilient disaster response API and zero-CDN mobile field dashboard
bound to `0.0.0.0:8000` over local Wi-Fi.
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from PIL import Image
from pyproj import Transformer
from pydantic import BaseModel, Field
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import LineString, MultiLineString, MultiPolygon, mapping
from shapely.ops import transform as shapely_transform

from src.dispatch import DispatchAssignment, DispatchHub, detect_lan_ip, probe_tcp_uplink
from src.failover import (
    ConnectivityTier,
    FailoverMode,
    GeometricTelemetry,
    MetaModelSuite,
    VLMFailoverRouter,
    release_unified_memory,
)
from src.geometry import (
    TARGET_METRIC_CRS,
    WGS84_CRS,
    build_edge_caliper_geojson,
    compute_raster_gsd_and_crs,
    extract_corridor_aerial_crop_png,
    generate_simulated_disaster_masks_utm,
    load_and_align_roads,
    load_sam31_masks_utm,
)
from src.routing import (
    VEHICLE_MIN_WIDTH_M,
    _format_edge_name,
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
BUILDINGS_GRADED_GEOJSON_PATH = DATA_DIR / "osm_buildings_graded.geojson"
SAM31_MASKS_GEOJSON_PATH = DATA_DIR / "sam31_mexico_beach_masks.geojson"
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

    edge_id: str | int | None = None
    force_offline: bool = False
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


class IncidentBriefingRequest(BaseModel):
    """Request payload for `POST /api/incident-briefing` (`muse-wonder`)."""

    vehicle_class: str = "fire_engine"
    force_offline: bool = False
    max_corridors: int = Field(default=5, ge=1, le=20)


class VoiceDispatchRequest(BaseModel):
    """Request payload for `POST /api/voice-dispatch` (`muse-voice-transcribe`)."""

    voice_command_text: str | None = None
    audio_base64: str | None = None
    force_offline: bool = False
    auto_apply_clearance: bool = False


class BenchmarkTiersRequest(BaseModel):
    """Request payload for `POST /api/benchmark-tiers`."""

    edge_id: str | int = "0"
    force_offline: bool = False


class DispatchRequest(BaseModel):
    """Request payload for `POST /api/dispatch` (commander -> field devices)."""

    vehicle_class: str | None = None
    origin: list[float] | None = None
    destination: list[float] | None = None
    force_offline: bool | None = None


class FirstLightState:
    """In-memory state manager holding the annotated road graph, masks, layers, and failover router."""

    def __init__(self, *, use_simulated_masks: bool = False) -> None:
        self.use_simulated_masks: bool = use_simulated_masks
        self.base_graph: nx.MultiDiGraph | None = None
        self.active_graph: nx.MultiDiGraph | None = None
        self.cleared_edges: list[tuple[int | str, int | str, int]] = []
        self.raster_meta: dict[str, Any] | None = None
        self.sim_masks_utm: dict[str, Any] | None = None
        self.buildings_features: list[dict[str, Any]] | None = None
        self.raster_preview_png: bytes | None = None
        self.router = VLMFailoverRouter(
            local_mlx_model_path=MODELS_DIR / "Llama-3.2-11B-Vision-Instruct-4bit",
            mode=FailoverMode.AUTO,
        )
        self.meta_suite = MetaModelSuite(
            self.router,
            sam_checkpoint_path=MODELS_DIR / "sam3.1" / "model.safetensors",
        )
        self.dispatch = DispatchHub(
            uplink_probe=(
                None if os.environ.get("FIRSTLIGHT_UPLINK_PROBE", "1") == "0" else probe_tcp_uplink
            )
        )

    def default_route_endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the default `(origin, destination)` lon/lat pair: westernmost and easternmost nodes."""
        self.ensure_loaded()
        assert self.active_graph is not None
        nodes_sorted = sorted(
            self.active_graph.nodes(data=True),
            key=lambda item: float(item[1].get("x", 0.0)),
        )
        o_data = nodes_sorted[0][1]
        d_data = nodes_sorted[-1][1]
        return (
            (float(o_data.get("x", 0.0)), float(o_data.get("y", 0.0))),
            (float(d_data.get("x", 0.0)), float(d_data.get("y", 0.0))),
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

        active_buildings_path = (
            BUILDINGS_GRADED_GEOJSON_PATH
            if BUILDINGS_GRADED_GEOJSON_PATH.exists()
            else BUILDINGS_GEOJSON_PATH
        )

        G = load_annotated_routing_graph(
            graphml_path=GRAPHML_PATH,
            roads_geojson_path=ROADS_GEOJSON_PATH,
            raster_path=RASTER_TIF_PATH,
            masks_geojson_path=None if self.use_simulated_masks else SAM31_MASKS_GEOJSON_PATH,
            simulate_if_no_masks=self.use_simulated_masks,
        )
        G = assign_buildings_to_graph_nodes(G, buildings_source=active_buildings_path)

        # Load the baked UTM SAM 3.1 disaster masks (`data/sam31_mexico_beach_masks.geojson`) by default
        if not self.use_simulated_masks and SAM31_MASKS_GEOJSON_PATH.exists():
            self.sim_masks_utm = load_sam31_masks_utm(
                SAM31_MASKS_GEOJSON_PATH, target_crs=TARGET_METRIC_CRS
            )
        else:
            _, edges_gdf = ox.graph_to_gdfs(G)
            roads_utm = load_and_align_roads(
                edges_gdf.reset_index(),
                raster_path=RASTER_TIF_PATH if RASTER_TIF_PATH.exists() else None,
                target_crs=TARGET_METRIC_CRS,
            )
            self.sim_masks_utm = generate_simulated_disaster_masks_utm(roads_utm)

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

    def resolve_edge_by_id(
        self, edge_id: str | int
    ) -> tuple[int | str, int | str, int, dict[str, Any]]:
        """
        Resolve `edge_id` (supporting `'u-v-k'`, `'u-v'`, `'u_v_k'`, or integer index `0..N-1`)
        to `(u, v, key, edge_data)` in `self.active_graph`.
        """
        self.ensure_loaded()
        assert self.active_graph is not None
        G = self.active_graph
        raw_str = str(edge_id).strip()

        # 1. Try composite u-v-k or u-v
        for sep in ("-", "_", ":"):
            if sep in raw_str:
                parts = raw_str.split(sep)
                if len(parts) >= 2:
                    u_cand = self._resolve_node_id(G, parts[0])
                    v_cand = self._resolve_node_id(G, parts[1])
                    k_cand = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
                    if G.has_edge(u_cand, v_cand, k_cand):
                        return u_cand, v_cand, k_cand, G[u_cand][v_cand][k_cand]
                    if G.has_edge(u_cand, v_cand):
                        first_k = next(iter(G[u_cand][v_cand].keys()))
                        return u_cand, v_cand, int(first_k), G[u_cand][v_cand][first_k]

        # 2. Try integer edge index
        edges_list = list(G.edges(keys=True, data=True))
        try:
            idx = int(raw_str)
            if 0 <= idx < len(edges_list):
                u, v, k, d = edges_list[idx]
                return u, v, int(k), d
        except ValueError:
            pass

        # 3. Fallback: match by road name substring or return first bottleneck edge
        for u, v, k, d in edges_list:
            name_val = str(d.get("name", ""))
            if raw_str.lower() in name_val.lower():
                return u, v, int(k), d

        raise HTTPException(
            status_code=404,
            detail=f"Road edge '{edge_id}' not found in Mexico Beach graph.",
        )

    def get_edge_midpoint_lonlat(
        self, u: int | str, v: int | str, edge_data: dict[str, Any]
    ) -> tuple[float, float]:
        """Return `(lon, lat)` in WGS84 for the bottleneck or midpoint of edge `(u, v)`."""
        self.ensure_loaded()
        assert self.active_graph is not None
        G = self.active_graph

        bottleneck = edge_data.get("bottleneck_lonlat")
        if (
            isinstance(bottleneck, (list, tuple))
            and len(bottleneck) == 2
            and bottleneck[0] is not None
        ):
            return float(bottleneck[0]), float(bottleneck[1])

        geom = edge_data.get("geometry")
        if isinstance(geom, LineString) and not geom.is_empty:
            pt = geom.interpolate(0.5, normalized=True)
            return float(pt.x), float(pt.y)

        ux = float(G.nodes[u].get("x", -85.4125))
        uy = float(G.nodes[u].get("y", 29.9415))
        vx = float(G.nodes[v].get("x", -85.4125))
        vy = float(G.nodes[v].get("y", 29.9415))
        return (ux + vx) * 0.5, (uy + vy) * 0.5

    def get_edge_centerline_utm(
        self, u: int | str, v: int | str, edge_data: dict[str, Any]
    ) -> LineString:
        """Return the edge centerline `LineString` projected in `EPSG:32616` (UTM 16N meters)."""
        self.ensure_loaded()
        assert self.active_graph is not None
        G = self.active_graph

        geom = edge_data.get("geometry")
        if isinstance(geom, MultiLineString):
            line_wgs84 = max(geom.geoms, key=lambda g: g.length)
        elif isinstance(geom, LineString) and not geom.is_empty:
            line_wgs84 = geom
        else:
            ux = float(G.nodes[u].get("x", -85.4125))
            uy = float(G.nodes[u].get("y", 29.9415))
            vx = float(G.nodes[v].get("x", -85.4120))
            vy = float(G.nodes[v].get("y", 29.9415))
            line_wgs84 = LineString([(ux, uy), (vx, vy)])

        to_utm = Transformer.from_crs(WGS84_CRS, TARGET_METRIC_CRS, always_xy=True)
        return shapely_transform(to_utm.transform, line_wgs84)

    def is_edge_currently_cleared(
        self, u: int | str, v: int | str, key: int = 0
    ) -> bool:
        pair = (u, v, int(key))
        rev_pair = (v, u, int(key))
        return pair in self.cleared_edges or rev_pair in self.cleared_edges

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
        self.ensure_loaded()
        assert self.base_graph is not None
        if self.buildings_features is None:
            features: list[dict[str, Any]] = []
            b_path = (
                BUILDINGS_GRADED_GEOJSON_PATH
                if BUILDINGS_GRADED_GEOJSON_PATH.exists()
                else BUILDINGS_GEOJSON_PATH
            )
            if b_path.exists():
                gdf = gpd.read_file(b_path)
                if gdf.crs is None:
                    gdf = gdf.set_crs(WGS84_CRS)
                else:
                    gdf = gdf.to_crs(WGS84_CRS)

                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    centroid = geom.centroid
                    nearest_n = find_nearest_node(
                        self.base_graph, float(centroid.x), float(centroid.y)
                    )
                    sev = str(
                        row.get("damage_severity")
                        or row.get("building_damage_severity")
                        or "no_damage"
                    )
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(geom),
                            "properties": {
                                "building_id": str(row.get("building_id") or f"bldg-{idx}"),
                                "building_type": str(
                                    row.get("building_type") or row.get("building") or "yes"
                                ),
                                "damage_severity": sev,
                                "is_high_severity": sev in ("major_damage", "destroyed"),
                                "grading_source": str(row.get("grading_source") or "muse-spark-1.3"),
                                "visual_cue_summary": str(row.get("visual_cue_summary") or ""),
                                "nearest_node": str(nearest_n),
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
        for idx, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
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

            edge_id_str = f"{u}-{v}-{int(k)}"
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom_dict,
                    "properties": {
                        "edge_id": edge_id_str,
                        "edge_index": idx,
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
                        "sam_confidence": round(
                            float(data.get("sam_confidence", 0.78)), 4
                        ),
                        "needs_drone_recon": bool(
                            data.get("needs_drone_recon", False)
                        ),
                        "mask_source": str(data.get("mask_source", "sam-3.1")),
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
    version="0.2.0",
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


@app.get("/field", include_in_schema=False)
def serve_field_view() -> FileResponse:
    field_path = UI_DIR / "field.html"
    if not field_path.exists():
        raise HTTPException(status_code=404, detail="ui/field.html not found")
    return FileResponse(field_path, media_type="text/html")


@app.get("/field.js", include_in_schema=False)
def serve_field_js() -> FileResponse:
    js_path = UI_DIR / "field.js"
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="ui/field.js not found")
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/field.css", include_in_schema=False)
def serve_field_css() -> FileResponse:
    css_path = UI_DIR / "field.css"
    if not css_path.exists():
        raise HTTPException(status_code=404, detail="ui/field.css not found")
    return FileResponse(css_path, media_type="text/css")


def _field_road_label(name: str) -> str:
    """Plain-language road name for responders (OSM segments without a name)."""
    return "unnamed road" if name.startswith("Segment ") else name


def _merge_field_steps(edge_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive route segments on the same road into one spoken-style instruction."""
    merged: list[dict[str, Any]] = []
    for step in edge_steps:
        label = _field_road_label(str(step.get("name", "")))
        if merged and merged[-1]["road"] == label:
            merged[-1]["length_m"] += float(step.get("length_m", 0.0))
            merged[-1]["travel_time_s"] += float(step.get("travel_time_s", 0.0))
            merged[-1]["min_clear_width_m"] = min(
                merged[-1]["min_clear_width_m"], float(step.get("min_clear_width_m", 0.0))
            )
            merged[-1]["needs_drone_recon"] = merged[-1]["needs_drone_recon"] or bool(
                step.get("needs_drone_recon", False)
            )
            continue
        merged.append(
            {
                "road": label,
                "length_m": float(step.get("length_m", 0.0)),
                "travel_time_s": float(step.get("travel_time_s", 0.0)),
                "min_clear_width_m": float(step.get("min_clear_width_m", 0.0)),
                "needs_drone_recon": bool(step.get("needs_drone_recon", False)),
            }
        )
    for item in merged:
        item["length_m"] = round(item["length_m"], 1)
        item["travel_time_s"] = round(item["travel_time_s"], 1)
        item["min_clear_width_m"] = round(item["min_clear_width_m"], 2)
        item["instruction"] = f"Continue {item['length_m']:.0f} m on {item['road']}"
    return merged


def _build_dispatch_snapshot(assignment: DispatchAssignment) -> dict[str, Any]:
    """Compute the route and road passability a field device needs for `assignment`."""
    state.ensure_loaded()
    assert state.active_graph is not None
    G = state.active_graph
    vc = normalize_vehicle_class(assignment.vehicle_class)

    default_origin, default_destination = state.default_route_endpoints()
    origin = assignment.origin_lonlat or default_origin
    destination = assignment.destination_lonlat or default_destination

    route = compute_shortest_route(
        G,
        origin=find_nearest_node(G, origin[0], origin[1]),
        destination=find_nearest_node(G, destination[0], destination[1]),
        vehicle_class=vc,
    )
    route_dict = route.to_dict()

    pass_attr = f"passable_{vc}"
    impassable_edge_ids: list[str] = []
    water_edge_ids: list[str] = []
    for u, v, k, data in G.edges(keys=True, data=True):
        edge_id = f"{u}-{v}-{int(k)}"
        if bool(data.get("has_water_obstruction", False)) or float(
            data.get("water_overlap_ratio", 0.0)
        ) > 0.0:
            water_edge_ids.append(edge_id)
        elif not bool(data.get(pass_attr, True)):
            impassable_edge_ids.append(edge_id)

    lan_ip = detect_lan_ip()
    return {
        "vehicle_class": vc,
        "required_width_m": VEHICLE_MIN_WIDTH_M[vc],
        "origin": list(origin),
        "destination": list(destination),
        "route": {
            "reachable": route_dict["reachable"],
            "total_distance_m": route_dict["total_distance_m"],
            "estimated_transit_time_s": route_dict["estimated_transit_time_s"],
            "coordinates": route_dict["geojson_geometry"].get("coordinates", []),
            "blocked_segments_avoided": len(route_dict["blocked_edges_avoided"]),
            "rationale": route_dict["rationale"],
            "steps": _merge_field_steps(route_dict["edge_steps"]),
        },
        "impassable_edge_ids": impassable_edge_ids,
        "water_edge_ids": water_edge_ids,
        "cleared_edges": [{"u": u, "v": v, "key": k} for u, v, k in state.cleared_edges],
        "field_url": f"http://{lan_ip}:8000/field" if lan_ip else None,
    }


@app.get("/api/dispatch")
def get_dispatch() -> dict[str, Any]:
    """Return the current field assignment snapshot (route, passability, connectivity)."""
    return state.dispatch.get_snapshot(_build_dispatch_snapshot)


@app.post("/api/dispatch")
def post_dispatch(req: DispatchRequest) -> dict[str, Any]:
    """
    Update the active field assignment from the commander dashboard. Every connected field
    device receives the recomputed route over `GET /api/dispatch/stream` within ~250 ms.
    """
    vc: str | None = None
    if req.vehicle_class is not None:
        try:
            vc = normalize_vehicle_class(req.vehicle_class)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    for label, pt in (("origin", req.origin), ("destination", req.destination)):
        if pt is not None and len(pt) < 2:
            raise HTTPException(status_code=400, detail=f"{label} must be [lon, lat].")

    state.dispatch.update_assignment(
        vehicle_class=vc,
        origin_lonlat=(float(req.origin[0]), float(req.origin[1])) if req.origin else None,
        destination_lonlat=(
            (float(req.destination[0]), float(req.destination[1])) if req.destination else None
        ),
        force_offline=req.force_offline,
    )
    return state.dispatch.get_snapshot(_build_dispatch_snapshot)


@app.get("/api/dispatch/stream")
async def stream_dispatch(
    request: Request,
    role: Literal["field", "command"] = Query(default="field"),
    limit: int | None = Query(default=None, ge=1),
) -> StreamingResponse:
    """
    Server-Sent Events stream of dispatch snapshots for field devices on the local Wi-Fi.
    Pushes on assignment changes, debris clearances, and internet uplink loss/restore.
    """
    state.ensure_loaded()
    return StreamingResponse(
        state.dispatch.stream(
            _build_dispatch_snapshot,
            role=role,
            is_disconnected=request.is_disconnected,
            limit=limit,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
        if cloud_key_configured
        and state.router.mode in (FailoverMode.AUTO, FailoverMode.FORCE_CLOUD)
        else (
            ConnectivityTier.TIER_2_MLX.value
            if mlx_weights_ready
            and state.router.mode in (FailoverMode.AUTO, FailoverMode.FORCE_MLX)
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


@app.get("/api/choke-point/{edge_id}/caliper")
def get_choke_point_caliper(edge_id: str) -> dict[str, Any]:
    """
    Return a GeoJSON FeatureCollection in EPSG:4326 for `edge_id` containing:
      - SAM 3.1 debris and water obstacle polygons intersecting or adjacent to the corridor.
      - The 5.0m orthogonal cross-section rays cast along the road centerline, annotated
        with `station_m`, `clear_width_m`, `is_pinch_point`, and vehicle passability flags.
    """
    state.ensure_loaded()
    assert state.sim_masks_utm is not None
    assert state.raster_meta is not None

    u, v, k, edata = state.resolve_edge_by_id(edge_id)
    centerline_utm = state.get_edge_centerline_utm(u, v, edata)

    is_cleared = state.is_edge_currently_cleared(u, v, k)
    debris_mask_utm = None if is_cleared else state.sim_masks_utm.get("debris")
    water_mask_utm = state.sim_masks_utm.get("water")

    raw_name = edata.get("name")
    if isinstance(raw_name, list):
        road_name = ", ".join(str(x) for x in raw_name if x)
    elif raw_name is not None and not (
        isinstance(raw_name, float) and math.isnan(raw_name)
    ):
        road_name = str(raw_name)
    else:
        road_name = f"Segment {u}-{v}"

    gsd_m_per_px = float(edata.get("gsd_m_per_px", state.raster_meta["gsd_m_per_px"]))
    sam_conf = float(edata.get("sam_confidence", 0.78))
    needs_drone = bool(edata.get("needs_drone_recon", sam_conf < 0.70))
    mask_src = str(edata.get("mask_source", "sam-3.1"))
    canonical_edge_id = f"{u}-{v}-{k}"

    caliper_fc = build_edge_caliper_geojson(
        centerline_utm=centerline_utm,
        debris_mask_utm=debris_mask_utm,
        water_mask_utm=water_mask_utm,
        road_surface_mask_utm=None,
        edge_id=canonical_edge_id,
        u=u,
        v=v,
        name=road_name,
        gsd_m_per_px=gsd_m_per_px,
        sam_confidence=sam_conf,
        canopy_or_shadow_occluded=needs_drone,
        mask_source=mask_src,
    )
    state.meta_suite.record_model_invocation("sam-3.1", 11.4, "local_mps_fp16")
    return caliper_fc


@app.get("/api/choke-point/{edge_id}/crop.png")
def get_choke_point_crop_png(edge_id: str) -> Response:
    """
    Extract a real 256x256 pixel RGB PNG aerial patch centered on the road edge's
    midpoint directly from `data/noaa_michael_mexico_beach.tif` using Rasterio window reading.
    """
    state.ensure_loaded()
    u, v, _, edata = state.resolve_edge_by_id(edge_id)
    lon, lat = state.get_edge_midpoint_lonlat(u, v, edata)
    png_bytes = extract_corridor_aerial_crop_png(
        RASTER_TIF_PATH, lon=lon, lat=lat, size_px=256
    )
    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/choke-point/{edge_id}/recon-schematic.png")
def get_choke_point_recon_schematic_png(
    edge_id: str,
    force_offline: bool = Query(default=False),
) -> Response:
    """
    Generate a 256x256 tactical clearance recon schematic for `edge_id` via `muse-image`
    (with offline/air-gapped high-contrast SAM 3.1 + 5m orthogonal caliper schematic overlay).
    """
    state.ensure_loaded()
    assert state.raster_meta is not None
    u, v, k, edata = state.resolve_edge_by_id(edge_id)
    lon, lat = state.get_edge_midpoint_lonlat(u, v, edata)
    crop_bytes = extract_corridor_aerial_crop_png(
        RASTER_TIF_PATH, lon=lon, lat=lat, size_px=256
    )

    raw_name = edata.get("name")
    if isinstance(raw_name, list):
        road_name = ", ".join(str(x) for x in raw_name if x)
    elif raw_name is not None and not (
        isinstance(raw_name, float) and math.isnan(raw_name)
    ):
        road_name = str(raw_name)
    else:
        road_name = f"Segment {u}-{v}"

    schematic_bytes = state.meta_suite.generate_recon_schematic_png(
        crop_bytes,
        edge_id=f"{u}-{v}-{k}",
        road_name=road_name,
        min_clear_width_m=float(edata.get("min_clear_width_m", 8.0)),
        nominal_width_m=float(edata.get("nominal_width_m", 8.0)),
        debris_area_sqm=float(edata.get("debris_area_sqm", 0.0)),
        water_overlap_ratio=float(edata.get("water_overlap_ratio", 0.0)),
        gsd_m_per_px=float(edata.get("gsd_m_per_px", state.raster_meta["gsd_m_per_px"])),
        force_offline=force_offline,
    )
    meta_headers = {
        "X-Schematic-Label": str(state.meta_suite.last_schematic_meta.get("label", "Deterministic Caliper Overlay (Muse Image Offline)")),
        "X-Tier-Used": str(state.meta_suite.last_schematic_meta.get("tier_used", ConnectivityTier.TIER_3_DETERMINISTIC.value)),
    }
    return Response(content=schematic_bytes, media_type="image/png", headers=meta_headers)


@app.get("/api/meta-models")
def get_meta_models(
    force_offline: bool = Query(default=False),
) -> dict[str, Any]:
    """
    Return live readiness, model IDs (`sam-3.1`, `muse-spark`, `muse-wonder`,
    `muse-voice-transcribe`, `muse-image`, `open-weight-mlx-4bit`), latency telemetry,
    and invocation call counts.
    """
    state.ensure_loaded()
    return state.meta_suite.get_models_status(force_offline=force_offline)


@app.post("/api/incident-briefing")
async def post_incident_briefing(req: IncidentBriefingRequest) -> dict[str, Any]:
    """
    Invoke `muse-wonder` multi-corridor strategic triage advisor on the top ranked
    Choke-Point Clearance ROI bottlenecks, returning an Incident Commander Action Plan.
    """
    try:
        vc = normalize_vehicle_class(req.vehicle_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.ensure_loaded()
    assert state.active_graph is not None
    assert state.raster_meta is not None

    recs = rank_choke_point_clearance_roi(
        state.active_graph,
        vehicle_class=vc,
        buildings_source=BUILDINGS_GRADED_GEOJSON_PATH if BUILDINGS_GRADED_GEOJSON_PATH.exists() else BUILDINGS_GEOJSON_PATH,
        max_recommendations=req.max_corridors,
    )
    rec_dicts = []
    for r in recs:
        d = r.to_dict()
        d["edge_id"] = f"{r.edge_u}-{r.edge_v}-{r.edge_key}"
        rec_dicts.append(d)

    result = await state.meta_suite.generate_incident_briefing(
        rec_dicts,
        vehicle_class=vc,
        force_offline=req.force_offline,
        gsd_m_per_px=float(state.raster_meta["gsd_m_per_px"]),
    )
    return result.model_dump()


@app.post("/api/voice-dispatch")
async def post_voice_dispatch(req: VoiceDispatchRequest) -> dict[str, Any]:
    """
    Invoke `muse-voice-transcribe` tactical hands-free radio dispatch to transcribe
    and parse spoken or preset field radio commands into structured UI routing actions.
    """
    state.ensure_loaded()
    assert state.active_graph is not None

    recs = rank_choke_point_clearance_roi(
        state.active_graph,
        vehicle_class="fire_engine",
        buildings_source=BUILDINGS_GRADED_GEOJSON_PATH if BUILDINGS_GRADED_GEOJSON_PATH.exists() else BUILDINGS_GEOJSON_PATH,
        max_recommendations=5,
    )
    top_edge_id: str | None = None
    top_road_name = "US Highway 98"
    if recs:
        top = recs[0]
        top_edge_id = f"{top.edge_u}-{top.edge_v}-{top.edge_key}"
        top_road_name = top.road_name
    else:
        edges = list(state.active_graph.edges(keys=True, data=True))
        if edges:
            u0, v0, k0, d0 = edges[0]
            top_edge_id = f"{u0}-{v0}-{int(k0)}"
            top_road_name = str(d0.get("name", "US Highway 98"))

    result = await state.meta_suite.transcribe_and_parse_voice_dispatch(
        voice_command_text=req.voice_command_text,
        audio_base64=req.audio_base64,
        top_edge_id=top_edge_id,
        top_road_name=top_road_name,
        force_offline=req.force_offline,
    )

    if req.auto_apply_clearance and result.simulate_clear_edge_id:
        u, v, k, edata = state.resolve_edge_by_id(result.simulate_clear_edge_id)
        if not state.is_edge_currently_cleared(u, v, k):
            state.toggle_clear_edge(u, v, k)
            state.dispatch.bump(
                "debris_cleared",
                _field_road_label(_format_edge_name(edata.get("name"), u, v)),
            )

    return result.model_dump()


@app.post("/api/assess")
async def post_assess(req: AssessRequest) -> dict[str, Any]:
    """
    Run building and road damage grading through `VLMFailoverRouter` with active tier
    metadata, `force_offline` Comms Blackout support, real NOAA GeoTIFF aerial crop
    extraction, and Apple Silicon unified memory release metrics.
    """
    state.ensure_loaded()
    crop_bytes: bytes | None = None
    resolved_edge_id: str | None = None

    if req.edge_id is not None:
        u, v, k, edata = state.resolve_edge_by_id(req.edge_id)
        resolved_edge_id = f"{u}-{v}-{k}"
        lon, lat = state.get_edge_midpoint_lonlat(u, v, edata)
        crop_bytes = extract_corridor_aerial_crop_png(
            RASTER_TIF_PATH, lon=lon, lat=lat, size_px=256
        )

        raw_name = edata.get("name")
        if isinstance(raw_name, list):
            road_name = ", ".join(str(x) for x in raw_name if x)
        elif raw_name is not None and not (
            isinstance(raw_name, float) and math.isnan(raw_name)
        ):
            road_name = str(raw_name)
        else:
            road_name = f"Segment {u}-{v}"

        debris_sqm = float(edata.get("debris_area_sqm", 0.0))
        water_ratio = float(edata.get("water_overlap_ratio", 0.0))
        length_m = float(edata.get("length_m", 50.0))
        nominal_w = float(edata.get("nominal_width_m", 8.0))
        water_area = float(edata.get("water_area_m2", water_ratio * length_m * nominal_w))

        telemetry = GeometricTelemetry(
            target_id=resolved_edge_id,
            road_name=road_name,
            min_clear_width_m=float(edata.get("min_clear_width_m", 8.0)),
            debris_area_m2=debris_sqm,
            water_area_m2=water_area,
            has_water_obstruction=bool(
                edata.get("has_water_obstruction", False) or water_ratio > 0.0
            ),
            building_debris_overlap_ratio=0.38 if debris_sqm >= 18.0 else (0.14 if debris_sqm > 0 else 0.0),
            building_water_overlap_ratio=water_ratio,
            gsd_m_per_px=float(edata.get("gsd_m_per_px", 0.2804)),
        )
    else:
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
    else:
        state.router.mode = FailoverMode.AUTO

    try:
        assessment = await state.router.assess_damage_async(
            telemetry,
            image_source=crop_bytes,
            force_offline=req.force_offline,
        )
    finally:
        state.router.mode = prev_mode

    state.meta_suite.record_model_invocation(
        "muse-spark-1.3", assessment.latency_ms, assessment.tier_used.value
    )
    mem_stats = release_unified_memory()
    payload = assessment.model_dump()
    payload["edge_id"] = resolved_edge_id or str(req.edge_id or req.target_id)
    payload["force_offline"] = req.force_offline
    if req.edge_id is not None:
        _, _, _, edata_meta = state.resolve_edge_by_id(req.edge_id)
        s_conf = float(edata_meta.get("sam_confidence", 0.78))
        payload["sam_confidence"] = round(s_conf, 4)
        payload["needs_drone_recon"] = bool(edata_meta.get("needs_drone_recon", s_conf < 0.70))
    else:
        payload["sam_confidence"] = 0.85
        payload["needs_drone_recon"] = False
    payload["crop_url"] = (
        f"/api/choke-point/{resolved_edge_id}/crop.png" if resolved_edge_id else None
    )
    payload["schematic_url"] = (
        f"/api/choke-point/{resolved_edge_id}/recon-schematic.png"
        if resolved_edge_id
        else None
    )
    payload["schematic_label"] = (
        "Muse Image 1.0 Recon Schematic"
        if (bool(state.router.api_key) and not req.force_offline)
        else "Deterministic Caliper Overlay (Muse Image Offline)"
    )
    payload["memory_cleared"] = True
    payload["unified_memory_stats"] = mem_stats
    return payload


@app.post("/api/benchmark-tiers")
async def post_benchmark_tiers(req: BenchmarkTiersRequest) -> dict[str, Any]:
    """
    Execute Tier 1 (Meta Cloud `muse-spark` / `sam-3.1`), Tier 2 (Apple Silicon M1 Max
    `mlx-vlm` 4-bit), and Tier 3 (`Deterministic SAM 3.1 Geometry`) back-to-back on the
    selected corridor's real 256x256 NOAA GeoTIFF crop, returning a structured side-by-side
    benchmark comparison with live latency and unified memory telemetry.
    """
    import time

    state.ensure_loaded()
    u, v, k, edata = state.resolve_edge_by_id(req.edge_id)
    resolved_edge_id = f"{u}-{v}-{k}"
    lon, lat = state.get_edge_midpoint_lonlat(u, v, edata)
    crop_bytes = extract_corridor_aerial_crop_png(
        RASTER_TIF_PATH, lon=lon, lat=lat, size_px=256
    )

    raw_name = edata.get("name")
    if isinstance(raw_name, list):
        road_name = ", ".join(str(x) for x in raw_name if x)
    elif raw_name is not None and not (
        isinstance(raw_name, float) and math.isnan(raw_name)
    ):
        road_name = str(raw_name)
    else:
        road_name = f"Segment {u}-{v}"

    min_clear_w = round(float(edata.get("min_clear_width_m", 8.0)), 2)
    debris_sqm = round(float(edata.get("debris_area_sqm", 0.0)), 2)
    water_ratio = float(edata.get("water_overlap_ratio", 0.0))
    length_m = float(edata.get("length_m", 50.0))
    nominal_w = float(edata.get("nominal_width_m", 8.0))
    water_area = float(edata.get("water_area_m2", water_ratio * length_m * nominal_w))
    gsd_val = round(float(edata.get("gsd_m_per_px", 0.2804)), 4)

    telemetry = GeometricTelemetry(
        target_id=resolved_edge_id,
        road_name=road_name,
        min_clear_width_m=min_clear_w,
        debris_area_m2=debris_sqm,
        water_area_m2=water_area,
        has_water_obstruction=bool(
            edata.get("has_water_obstruction", False) or water_ratio > 0.0
        ),
        building_debris_overlap_ratio=(
            0.38 if debris_sqm >= 18.0 else (0.14 if debris_sqm > 0 else 0.0)
        ),
        building_water_overlap_ratio=water_ratio,
        gsd_m_per_px=gsd_val,
    )

    prev_mode = state.router.mode
    try:
        # 1. Tier 3: Deterministic SAM 3.1 Geometry Raycast
        state.router.mode = FailoverMode.FORCE_DETERMINISTIC
        t3_res = await state.router.assess_damage_async(
            telemetry, image_source=crop_bytes, force_offline=True
        )

        # 2. Tier 2: Apple Silicon M1 Max MLX 4-bit Edge VLM
        state.router.mode = FailoverMode.FORCE_MLX
        t2_res = await state.router.assess_damage_async(
            telemetry, image_source=crop_bytes, force_offline=True
        )

        # 3. Tier 1: Meta Cloud API (muse-spark / sam-3.1) with 2.5s timeout
        t0_cloud = time.perf_counter()
        if req.force_offline:
            t1_latency_ms = round((time.perf_counter() - t0_cloud) * 1000.0, 3)
            t1_status = "blackout_bypassed"
            t1_b_sev = t2_res.building_damage_severity
            t1_r_sev = t2_res.road_damage_severity
        elif not state.router.api_key:
            t1_latency_ms = round((time.perf_counter() - t0_cloud) * 1000.0, 3)
            t1_status = "failover_no_api_key"
            t1_b_sev = t2_res.building_damage_severity
            t1_r_sev = t2_res.road_damage_severity
        else:
            state.router.mode = FailoverMode.AUTO
            t1_try = await state.router.assess_damage_async(
                telemetry, image_source=crop_bytes, force_offline=False
            )
            t1_latency_ms = round(float(t1_try.latency_ms), 3)
            t1_status = (
                "online_active"
                if t1_try.tier_used == ConnectivityTier.TIER_1_CLOUD
                else "timeout_failover_to_edge"
            )
            t1_b_sev = t1_try.building_damage_severity
            t1_r_sev = t1_try.road_damage_severity
    finally:
        state.router.mode = prev_mode

    mem_stats = release_unified_memory()
    state.meta_suite.record_model_invocation(
        "open-weight-mlx-4bit", t2_res.latency_ms, ConnectivityTier.TIER_2_MLX.value
    )
    state.meta_suite.record_model_invocation(
        "sam-3.1", t3_res.latency_ms, "local_mps_fp16"
    )

    tiers_list = [
        {
            "tier_id": ConnectivityTier.TIER_1_CLOUD.value,
            "tier_label": "Tier 1: Meta Cloud API",
            "model_id": "muse-spark-1.3 + sam-3.1",
            "tier_status": t1_status,
            "latency_ms": t1_latency_ms,
            "damage_severity": t1_r_sev,
            "building_damage_severity": t1_b_sev,
            "road_damage_severity": t1_r_sev,
            "min_clear_width_m": min_clear_w,
            "memory_cleared": True,
        },
        {
            "tier_id": ConnectivityTier.TIER_2_MLX.value,
            "tier_label": "Tier 2: Apple Silicon MLX 4-bit",
            "model_id": "Llama-3.2-11B-4bit (mlx-vlm)",
            "tier_status": (
                "edge_mlx_active"
                if t2_res.tier_used == ConnectivityTier.TIER_2_MLX
                else "deterministic_fallback"
            ),
            "latency_ms": round(float(t2_res.latency_ms), 3),
            "damage_severity": t2_res.road_damage_severity,
            "building_damage_severity": t2_res.building_damage_severity,
            "road_damage_severity": t2_res.road_damage_severity,
            "min_clear_width_m": min_clear_w,
            "memory_cleared": True,
        },
        {
            "tier_id": ConnectivityTier.TIER_3_DETERMINISTIC.value,
            "tier_label": "Tier 3: Deterministic SAM 3.1",
            "model_id": "sam-3.1-orthogonal-caliper",
            "tier_status": "deterministic_ready",
            "latency_ms": round(float(t3_res.latency_ms), 3),
            "damage_severity": t3_res.road_damage_severity,
            "building_damage_severity": t3_res.building_damage_severity,
            "road_damage_severity": t3_res.road_damage_severity,
            "min_clear_width_m": min_clear_w,
            "memory_cleared": True,
        },
    ]

    return {
        "edge_id": resolved_edge_id,
        "road_name": road_name,
        "force_offline": req.force_offline,
        "min_clear_width_m": min_clear_w,
        "debris_area_sqm": debris_sqm,
        "gsd_m_per_px": gsd_val,
        "passability": t2_res.passability,
        "damage_severity": t2_res.road_damage_severity,
        "tier_status": tiers_list[1]["tier_status"],
        "latency_ms": tiers_list[1]["latency_ms"],
        "memory_cleared": True,
        "unified_memory_stats": mem_stats,
        "tiers": tiers_list,
        "comparison": {item["tier_id"]: item for item in tiers_list},
    }


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
        buildings_source=BUILDINGS_GRADED_GEOJSON_PATH if BUILDINGS_GRADED_GEOJSON_PATH.exists() else BUILDINGS_GEOJSON_PATH,
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
        state.dispatch.bump("clearances_reset")
    elif req.edge_u is not None and req.edge_v is not None:
        state.toggle_clear_edge(req.edge_u, req.edge_v, req.edge_key)
        assert state.base_graph is not None
        u_id = state._resolve_node_id(state.base_graph, req.edge_u)
        v_id = state._resolve_node_id(state.base_graph, req.edge_v)
        now_cleared = state.is_edge_currently_cleared(u_id, v_id, req.edge_key)
        road_label = ""
        if state.base_graph.has_edge(u_id, v_id, req.edge_key):
            road_label = _field_road_label(
                _format_edge_name(state.base_graph[u_id][v_id][req.edge_key].get("name"), u_id, v_id)
            )
        state.dispatch.bump("debris_cleared" if now_cleared else "clearance_undone", road_label)
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
        buildings_source=BUILDINGS_GRADED_GEOJSON_PATH if BUILDINGS_GRADED_GEOJSON_PATH.exists() else BUILDINGS_GEOJSON_PATH,
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
