"""
Geospatial alignment and SAM 3.1 orthogonal cross-sectional road width measurement engine.

Key Invariants (AGENTS.md & geospatial-processing / routing-engine skills):
1. Coordinate Alignment: Always reproject OpenStreetMap vector graphs (EPSG:4326) into the
   native metric CRS or local UTM zone (EPSG:32616 for Mexico Beach, FL) before running
   spatial intersections.
2. Orthogonal Cross-Sectional Sampling: Cast perpendicular ray segments every 5.0 meters
   along each road centerline, intersecting against road surface, debris, and water masks
   to determine the exact minimum contiguous clear width in meters (`min_clear_width_m`).
3. Vehicle Clearance Thresholds:
   - ATV / On Foot (`atv`): `min_clear_width_m > 1.2m`
   - Ambulance / Pickup (`ambulance`): `min_clear_width_m > 2.2m`
   - Fire Engine (`fire_engine`): `min_clear_width_m > 3.5m`
4. Impassable Water Rule: Any road segment overlapping a water mask has its weight set to
   `math.inf` (impassable for all vehicle classes), regardless of width.
5. Hardware Profile: Apple Silicon MPS strictly in `torch.float16` (`fp16`), never `bfloat16`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import shapes as raster_shapes
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.ops import transform as shapely_transform, unary_union

# Constants & Vehicle Thresholds
TARGET_METRIC_CRS = "EPSG:32616"  # WGS 84 / UTM zone 16N (meters)
WGS84_CRS = "EPSG:4326"

DEFAULT_SAMPLING_INTERVAL_M = 5.0
DEFAULT_RAY_HALF_WIDTH_M = 10.0
DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M = 4.0  # 8.0m total nominal paved width when no road mask supplied

ATV_MIN_WIDTH_M = 1.2
AMBULANCE_MIN_WIDTH_M = 2.2
FIRE_ENGINE_MIN_WIDTH_M = 3.5

VEHICLE_CLEARANCE_THRESHOLDS_M: dict[str, float] = {
    "atv": ATV_MIN_WIDTH_M,
    "ambulance": AMBULANCE_MIN_WIDTH_M,
    "fire_engine": FIRE_ENGINE_MIN_WIDTH_M,
}


@dataclass(frozen=True)
class PassabilityFlags:
    """Boolean passability flags for the three FirstLight emergency vehicle classes."""

    atv: bool
    ambulance: bool
    fire_engine: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "atv": self.atv,
            "ambulance": self.ambulance,
            "fire_engine": self.fire_engine,
        }


@dataclass
class CrossSectionSample:
    """Single orthogonal cross-sectional ray measurement at a station along a road centerline."""

    station_m: float
    center_point_utm: tuple[float, float]
    ray_line_utm: LineString
    clear_segment_utm: LineString
    clear_width_m: float
    water_intersected: bool
    debris_intersected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_m": round(self.station_m, 3),
            "center_point_utm": (
                round(self.center_point_utm[0], 3),
                round(self.center_point_utm[1], 3),
            ),
            "ray_coords_utm": list(self.ray_line_utm.coords),
            "clear_coords_utm": list(self.clear_segment_utm.coords),
            "clear_width_m": round(self.clear_width_m, 4),
            "water_intersected": self.water_intersected,
            "debris_intersected": self.debris_intersected,
        }


@dataclass
class RoadEdgeClearance:
    """Complete metric clearance and passability evaluation for a single road segment."""

    edge_id: str | int
    u: int | None
    v: int | None
    name: str
    length_m: float
    gsd_m_per_px: float
    min_clear_width_m: float
    mean_clear_width_m: float
    max_clear_width_m: float
    has_water_obstruction: bool
    debris_area_m2: float
    water_area_m2: float
    bottleneck_point_utm: tuple[float, float]
    bottleneck_point_lonlat: tuple[float, float]
    passability: PassabilityFlags
    edge_weight_atv: float
    edge_weight_ambulance: float
    edge_weight_fire_engine: float
    reason: str
    sam_confidence: float = 0.85
    needs_drone_recon: bool = False
    mask_source: str = "sam-3.1"
    cross_sections: list[CrossSectionSample] = field(default_factory=list)

    def to_dict(self, include_cross_sections: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "edge_id": self.edge_id,
            "u": self.u,
            "v": self.v,
            "name": self.name,
            "length_m": round(self.length_m, 3),
            "gsd_m_per_px": round(self.gsd_m_per_px, 4),
            "min_clear_width_m": round(self.min_clear_width_m, 3),
            "mean_clear_width_m": round(self.mean_clear_width_m, 3),
            "max_clear_width_m": round(self.max_clear_width_m, 3),
            "has_water_obstruction": self.has_water_obstruction,
            "debris_area_m2": round(self.debris_area_m2, 3),
            "water_area_m2": round(self.water_area_m2, 3),
            "bottleneck_point_utm": (
                round(self.bottleneck_point_utm[0], 3),
                round(self.bottleneck_point_utm[1], 3),
            ),
            "bottleneck_point_lonlat": (
                round(self.bottleneck_point_lonlat[0], 6),
                round(self.bottleneck_point_lonlat[1], 6),
            ),
            "passability": self.passability.to_dict(),
            "edge_weight_atv": self.edge_weight_atv,
            "edge_weight_ambulance": self.edge_weight_ambulance,
            "edge_weight_fire_engine": self.edge_weight_fire_engine,
            "sam_confidence": round(self.sam_confidence, 4),
            "needs_drone_recon": self.needs_drone_recon,
            "mask_source": self.mask_source,
            "reason": self.reason,
        }
        if include_cross_sections:
            payload["cross_sections"] = [cs.to_dict() for cs in self.cross_sections]
        return payload


def compute_passability_flags(
    min_clear_width_m: float,
    has_water_obstruction: bool = False,
) -> PassabilityFlags:
    """
    Evaluate vehicle class passability strictly from measured metric width and water overlap.

    Rules:
    - Any water overlap renders the road impassable (False) for all vehicle classes.
    - ATV / On Foot: min_clear_width_m > 1.2m
    - Ambulance / Pickup: min_clear_width_m > 2.2m
    - Fire Engine: min_clear_width_m > 3.5m
    """
    if has_water_obstruction:
        return PassabilityFlags(atv=False, ambulance=False, fire_engine=False)
    return PassabilityFlags(
        atv=bool(min_clear_width_m > ATV_MIN_WIDTH_M),
        ambulance=bool(min_clear_width_m > AMBULANCE_MIN_WIDTH_M),
        fire_engine=bool(min_clear_width_m > FIRE_ENGINE_MIN_WIDTH_M),
    )


def compute_vehicle_edge_weights(
    length_m: float,
    min_clear_width_m: float,
    has_water_obstruction: bool = False,
) -> dict[str, float]:
    """
    Compute NetworkX graph edge weights per vehicle class.

    Any road overlapping a water mask or failing the minimum width threshold
    receives weight = math.inf.
    """
    flags = compute_passability_flags(min_clear_width_m, has_water_obstruction)
    base_cost = max(float(length_m), 0.001)
    return {
        "atv": base_cost if flags.atv else math.inf,
        "ambulance": base_cost if flags.ambulance else math.inf,
        "fire_engine": base_cost if flags.fire_engine else math.inf,
    }


def compute_raster_gsd_and_crs(
    raster_path: str | Path,
    target_crs: str = TARGET_METRIC_CRS,
) -> dict[str, Any]:
    """
    Inspect a GeoTIFF raster and compute its metric Ground Sample Distance (GSD in m/px)
    in the target UTM coordinate system (`EPSG:32616`).
    """
    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        native_crs = src.crs.to_string() if src.crs else WGS84_CRS
        bounds = src.bounds
        width, height = src.width, src.height

        crs_obj = CRS.from_user_input(native_crs)
        if crs_obj.is_projected and crs_obj.axis_info[0].unit_name in ("metre", "meter", "m"):
            gsd_x_m = abs(float(src.transform.a))
            gsd_y_m = abs(float(src.transform.e))
            metric_transform = src.transform
            metric_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        else:
            metric_transform, _, _ = calculate_default_transform(
                native_crs,
                target_crs,
                width,
                height,
                *bounds,
            )
            gsd_x_m = abs(float(metric_transform.a))
            gsd_y_m = abs(float(metric_transform.e))
            transformer = Transformer.from_crs(native_crs, target_crs, always_xy=True)
            min_x, min_y = transformer.transform(bounds.left, bounds.bottom)
            max_x, max_y = transformer.transform(bounds.right, bounds.top)
            metric_bounds = (min_x, min_y, max_x, max_y)

        mean_gsd_m = float(math.sqrt(gsd_x_m * gsd_y_m))
        return {
            "raster_path": str(raster_path),
            "native_crs": native_crs,
            "target_crs": target_crs,
            "width_px": width,
            "height_px": height,
            "bands": src.count,
            "gsd_x_m": gsd_x_m,
            "gsd_y_m": gsd_y_m,
            "gsd_m_per_px": mean_gsd_m,
            "native_bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
            "metric_bounds": metric_bounds,
            "metric_transform": metric_transform,
        }


def load_and_align_roads(
    roads_source: str | Path | gpd.GeoDataFrame,
    raster_path: str | Path | None = None,
    target_crs: str = TARGET_METRIC_CRS,
    clip_to_raster: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load OpenStreetMap roads from `osm_roads.geojson` (or an existing GeoDataFrame)
    and reproject into the target metric CRS (`EPSG:32616`).
    """
    if isinstance(roads_source, gpd.GeoDataFrame):
        gdf = roads_source.copy()
    else:
        gdf = gpd.read_file(Path(roads_source))

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84_CRS)

    if raster_path is not None and clip_to_raster:
        with rasterio.open(Path(raster_path)) as src:
            raster_crs = src.crs.to_string() if src.crs else WGS84_CRS
            b = src.bounds
            raster_box_gdf = gpd.GeoDataFrame(
                geometry=[box(b.left, b.bottom, b.right, b.top)],
                crs=raster_crs,
            ).to_crs(gdf.crs)
            clip_poly = raster_box_gdf.geometry.iloc[0]
            gdf = gdf[gdf.intersects(clip_poly)].copy()

    gdf_utm = gdf.to_crs(target_crs)
    # Explode any MultiLineStrings into individual LineStrings or keep primary LineString
    gdf_utm = gdf_utm[gdf_utm.geometry.notnull() & ~gdf_utm.geometry.is_empty].copy()
    return gdf_utm


def _normalize_to_multipolygon(geom: Any) -> MultiPolygon:
    """Convert any Shapely polygon/collection/iterable into a clean MultiPolygon."""
    if geom is None:
        return MultiPolygon()
    if isinstance(geom, (list, tuple)):
        polys: list[Polygon] = []
        for item in geom:
            mp = _normalize_to_multipolygon(item)
            polys.extend(mp.geoms)
        if not polys:
            return MultiPolygon()
        merged = unary_union(polys)
        return _normalize_to_multipolygon(merged)

    if geom.is_empty:
        return MultiPolygon()
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return _normalize_to_multipolygon(polys)
    return MultiPolygon()


def mask_array_to_utm_polygons(
    mask: np.ndarray,
    transform: Affine,
    src_crs: str | CRS = TARGET_METRIC_CRS,
    target_crs: str = TARGET_METRIC_CRS,
) -> MultiPolygon:
    """
    Convert a 2D binary segmentation mask array (`bool` or `uint8 > 0`) into a vector
    `MultiPolygon` projected in `target_crs` (`EPSG:32616`).
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(binary):
        return MultiPolygon()

    extracted_polys: list[Polygon] = []
    for geom_dict, val in raster_shapes(binary, mask=binary > 0, transform=transform):
        if int(val) > 0:
            poly = shape(geom_dict)
            if not poly.is_empty and poly.is_valid:
                extracted_polys.append(poly)
            elif not poly.is_empty:
                extracted_polys.append(poly.buffer(0))

    if not extracted_polys:
        return MultiPolygon()

    merged = unary_union(extracted_polys)
    src_crs_obj = CRS.from_user_input(src_crs)
    dst_crs_obj = CRS.from_user_input(target_crs)
    if src_crs_obj != dst_crs_obj:
        transformer = Transformer.from_crs(src_crs_obj, dst_crs_obj, always_xy=True)
        merged = shapely_transform(transformer.transform, merged)

    return _normalize_to_multipolygon(merged)


def generate_orthogonal_cross_sections(
    centerline: LineString,
    interval_m: float = DEFAULT_SAMPLING_INTERVAL_M,
    ray_half_width_m: float = DEFAULT_RAY_HALF_WIDTH_M,
) -> list[tuple[float, Point, LineString]]:
    """
    Cast orthogonal cross-section ray segments at regular `interval_m` (default 5.0m)
    intervals along a metric UTM road centerline.

    Returns a list of `(station_m, center_point, orthogonal_ray_linestring)`.
    """
    if centerline.is_empty or centerline.length <= 0.0:
        return []

    total_length = float(centerline.length)
    if total_length < interval_m:
        stations = [total_length * 0.5]
    else:
        num_steps = int(math.floor(total_length / interval_m))
        stations = [i * interval_m for i in range(num_steps + 1)]
        if total_length - stations[-1] > 1e-3:
            stations.append(total_length)

    delta = min(0.5, max(total_length * 0.05, 0.05))
    samples: list[tuple[float, Point, LineString]] = []

    for s in stations:
        pt = centerline.interpolate(s)
        s_prev = max(0.0, s - delta)
        s_next = min(total_length, s + delta)
        if s_next - s_prev < 1e-6:
            p0 = Point(centerline.coords[0])
            p1 = Point(centerline.coords[-1])
        else:
            p0 = centerline.interpolate(s_prev)
            p1 = centerline.interpolate(s_next)

        dx = p1.x - p0.x
        dy = p1.y - p0.y
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            nx_val, ny_val = 0.0, 1.0
        else:
            # Unit normal vector (-dy, dx) orthogonal to tangent (dx, dy)
            nx_val = -dy / norm
            ny_val = dx / norm

        left_pt = (pt.x - ray_half_width_m * nx_val, pt.y - ray_half_width_m * ny_val)
        right_pt = (pt.x + ray_half_width_m * nx_val, pt.y + ray_half_width_m * ny_val)
        ray = LineString([left_pt, right_pt])
        samples.append((float(s), pt, ray))

    return samples


def _extract_max_contiguous_line(geom: Any, fallback_point: Point) -> LineString:
    """
    Extract the longest contiguous `LineString` from an intersection geometry.

    When debris sits in the middle of a road cross-section, the remaining clear span may
    be a `MultiLineString` of two narrow side strips. Because a vehicle cannot straddle
    solid debris, we take the maximum contiguous clear sub-segment length.
    """
    if geom is None or geom.is_empty:
        return LineString([fallback_point, fallback_point])
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        lines = [
            g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty and g.length > 0
        ]
        if not lines:
            return LineString([fallback_point, fallback_point])
        return max(lines, key=lambda ln: ln.length)
    return LineString([fallback_point, fallback_point])


def _format_clearance_reason(
    min_clear_width_m: float,
    gsd_m_per_px: float,
    debris_area_m2: float,
    water_area_m2: float,
    has_water_obstruction: bool,
    passability: PassabilityFlags,
) -> str:
    """
    Build a strictly mathematical, non-hallucinated diagnostic string derived from
    pixel-to-meter ratios, measured clear widths, and obstruction areas.
    """
    if has_water_obstruction:
        return (
            f"IMPASSABLE (WATER): Overlaps {water_area_m2:.2f} m² water mask at "
            f"{gsd_m_per_px:.4f} m/px GSD; weight=inf for all vehicle classes."
        )

    allowed: list[str] = []
    blocked: list[str] = []
    if passability.fire_engine:
        allowed.append("Fire Engine (>3.5m)")
    else:
        blocked.append("Fire Engine (>3.5m)")
    if passability.ambulance:
        allowed.append("Ambulance (>2.2m)")
    else:
        blocked.append("Ambulance (>2.2m)")
    if passability.atv:
        allowed.append("ATV (>1.2m)")
    else:
        blocked.append("ATV (>1.2m)")

    status_part = (
        f"Passable for {', '.join(allowed)}"
        if allowed
        else "BLOCKED for all vehicle classes (<=1.2m)"
    )
    if blocked and allowed:
        status_part += f"; blocked for {', '.join(blocked)}"

    return (
        f"Min clear width {min_clear_width_m:.2f}m (measured at {gsd_m_per_px:.4f} m/px GSD, "
        f"{debris_area_m2:.2f} m² debris). {status_part}."
    )


def measure_road_edge_clearance(
    centerline_utm: LineString,
    debris_mask_utm: Polygon | MultiPolygon | None = None,
    water_mask_utm: Polygon | MultiPolygon | None = None,
    road_surface_mask_utm: Polygon | MultiPolygon | None = None,
    *,
    edge_id: str | int = 0,
    u: int | None = None,
    v: int | None = None,
    name: str = "Unnamed Road",
    gsd_m_per_px: float = 0.2804,
    sampling_interval_m: float = DEFAULT_SAMPLING_INTERVAL_M,
    ray_half_width_m: float = DEFAULT_RAY_HALF_WIDTH_M,
    nominal_road_half_width_m: float = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M,
    utm_crs: str = TARGET_METRIC_CRS,
    sam_confidence: float = 0.85,
    canopy_or_shadow_occluded: bool = False,
    mask_source: str = "sam-3.1",
) -> RoadEdgeClearance:
    """
    Measure orthogonal cross-sectional road widths every `sampling_interval_m` (5.0m)
    along a UTM road centerline against SAM 3.1 road surface, debris, and water masks.
    """
    debris_mp = _normalize_to_multipolygon(debris_mask_utm)
    water_mp = _normalize_to_multipolygon(water_mask_utm)

    if road_surface_mask_utm is not None and not road_surface_mask_utm.is_empty:
        road_corridor = _normalize_to_multipolygon(road_surface_mask_utm)
    else:
        # Round-capped corridor around the road centerline representing the nominal road width
        # at all stations including endpoints (s=0 and s=L) and curves.
        corridor_poly = centerline_utm.buffer(
            nominal_road_half_width_m,
            cap_style="round",
            join_style="round",
        )
        road_corridor = _normalize_to_multipolygon(corridor_poly)

    debris_intersection = road_corridor.intersection(debris_mp) if not debris_mp.is_empty else MultiPolygon()
    water_intersection = road_corridor.intersection(water_mp) if not water_mp.is_empty else MultiPolygon()

    debris_area_m2 = float(debris_intersection.area) if not debris_intersection.is_empty else 0.0
    water_area_m2 = float(water_intersection.area) if not water_intersection.is_empty else 0.0

    clear_road_corridor = (
        road_corridor.difference(debris_mp) if not debris_mp.is_empty else road_corridor
    )

    raw_rays = generate_orthogonal_cross_sections(
        centerline=centerline_utm,
        interval_m=sampling_interval_m,
        ray_half_width_m=ray_half_width_m,
    )

    cross_sections: list[CrossSectionSample] = []
    any_water_hit = water_area_m2 > 1e-6 or (
        not water_mp.is_empty and centerline_utm.intersects(water_mp)
    )

    for station_m, center_pt, ray_line in raw_rays:
        ray_on_road = ray_line.intersection(road_corridor)
        water_hit = False
        if not water_mp.is_empty:
            water_hit = bool(
                ray_on_road.intersects(water_mp) or center_pt.intersects(water_mp)
            )
        if water_hit:
            any_water_hit = True

        debris_hit = False
        if not debris_mp.is_empty and not ray_on_road.is_empty:
            debris_hit = bool(ray_on_road.intersects(debris_mp))

        if water_hit:
            clear_seg = LineString([center_pt, center_pt])
            clear_width_m = 0.0
        else:
            clear_geom = ray_line.intersection(clear_road_corridor)
            clear_seg = _extract_max_contiguous_line(clear_geom, center_pt)
            clear_width_m = float(clear_seg.length)

        cross_sections.append(
            CrossSectionSample(
                station_m=station_m,
                center_point_utm=(center_pt.x, center_pt.y),
                ray_line_utm=ray_line,
                clear_segment_utm=clear_seg,
                clear_width_m=clear_width_m,
                water_intersected=water_hit,
                debris_intersected=debris_hit,
            )
        )

    if not cross_sections:
        mid_pt = centerline_utm.interpolate(0.5, normalized=True) if not centerline_utm.is_empty else Point(0.0, 0.0)
        bottleneck_utm = (mid_pt.x, mid_pt.y)
        min_width = 0.0
        mean_width = 0.0
        max_width = 0.0
    else:
        bottleneck_cs = min(cross_sections, key=lambda cs: cs.clear_width_m)
        bottleneck_utm = bottleneck_cs.center_point_utm
        widths = [cs.clear_width_m for cs in cross_sections]
        min_width = 0.0 if any_water_hit else float(min(widths))
        mean_width = 0.0 if any_water_hit else float(sum(widths) / len(widths))
        max_width = 0.0 if any_water_hit else float(max(widths))

    to_wgs84 = Transformer.from_crs(utm_crs, WGS84_CRS, always_xy=True)
    lon, lat = to_wgs84.transform(bottleneck_utm[0], bottleneck_utm[1])

    passability = compute_passability_flags(
        min_clear_width_m=min_width,
        has_water_obstruction=any_water_hit,
    )
    weights = compute_vehicle_edge_weights(
        length_m=float(centerline_utm.length),
        min_clear_width_m=min_width,
        has_water_obstruction=any_water_hit,
    )
    reason = _format_clearance_reason(
        min_clear_width_m=min_width,
        gsd_m_per_px=gsd_m_per_px,
        debris_area_m2=debris_area_m2,
        water_area_m2=water_area_m2,
        has_water_obstruction=any_water_hit,
        passability=passability,
    )

    needs_drone_recon = bool(float(sam_confidence) < 0.70 or canopy_or_shadow_occluded)

    return RoadEdgeClearance(
        edge_id=edge_id,
        u=u,
        v=v,
        name=name,
        length_m=float(centerline_utm.length),
        gsd_m_per_px=float(gsd_m_per_px),
        min_clear_width_m=min_width,
        mean_clear_width_m=mean_width,
        max_clear_width_m=max_width,
        has_water_obstruction=any_water_hit,
        debris_area_m2=debris_area_m2,
        water_area_m2=water_area_m2,
        bottleneck_point_utm=bottleneck_utm,
        bottleneck_point_lonlat=(float(lon), float(lat)),
        passability=passability,
        edge_weight_atv=weights["atv"],
        edge_weight_ambulance=weights["ambulance"],
        edge_weight_fire_engine=weights["fire_engine"],
        reason=reason,
        sam_confidence=float(sam_confidence),
        needs_drone_recon=needs_drone_recon,
        mask_source=mask_source,
        cross_sections=cross_sections,
    )


evaluate_road_edge_geometry = measure_road_edge_clearance


def _make_oriented_road_patch(
    center_pt: Point,
    tx: float,
    ty: float,
    nx_val: float,
    ny_val: float,
    t_min: float,
    t_max: float,
    n_min: float,
    n_max: float,
) -> Polygon:
    """Create a polygon in UTM coordinates oriented to a road's local (tangent, normal) frame."""
    corners = [
        (
            center_pt.x + t_min * tx + n_min * nx_val,
            center_pt.y + t_min * ty + n_min * ny_val,
        ),
        (
            center_pt.x + t_max * tx + n_min * nx_val,
            center_pt.y + t_max * ty + n_min * ny_val,
        ),
        (
            center_pt.x + t_max * tx + n_max * nx_val,
            center_pt.y + t_max * ty + n_max * ny_val,
        ),
        (
            center_pt.x + t_min * tx + n_max * nx_val,
            center_pt.y + t_min * ty + n_max * ny_val,
        ),
    ]
    return Polygon(corners)


def generate_simulated_disaster_masks_utm(
    roads_utm_gdf: gpd.GeoDataFrame,
    seed: int = 42,
) -> dict[str, MultiPolygon]:
    """
    Generate deterministic, spatially realistic post-hurricane debris and coastal surge water
    masks in `EPSG:32616` along the road network for offline simulation and testing.

    Creates representative bottleneck categories across the network:
    - Clear roads (> 3.5m clear width: passable for all classes)
    - Fire-Engine-blocked roads (~2.8m clear width: Ambulance & ATV only)
    - Ambulance-blocked roads (~1.7m clear width: ATV / On Foot only)
    - Completely debris-blocked roads (~0.6m clear width: blocked for all vehicles)
    - Coastal storm-surge water washouts (weight = infinity)
    """
    if roads_utm_gdf.empty:
        return {
            "road_surface": MultiPolygon(),
            "debris": MultiPolygon(),
            "water": MultiPolygon(),
        }

    _ = seed  # deterministic placement by edge index
    road_buffers: list[Polygon] = []
    debris_patches: list[Polygon] = []
    water_patches: list[Polygon] = []
    half_w = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M  # 4.0m (total nominal width = 8.0m)

    for idx, (_, row) in enumerate(roads_utm_gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            line = max(geom.geoms, key=lambda g: g.length)
        elif isinstance(geom, LineString):
            line = geom
        else:
            continue

        road_buffers.append(line.buffer(half_w, cap_style="round", join_style="round"))
        if line.length < 10.0:
            continue

        # Snap obstruction center to an exact 5.0m sampling station near the middle of the edge
        num_stations = int(math.floor(line.length / DEFAULT_SAMPLING_INTERVAL_M))
        mid_station_idx = max(1, num_stations // 2)
        target_s = min(line.length * 0.5, mid_station_idx * DEFAULT_SAMPLING_INTERVAL_M)
        if num_stations >= 2:
            target_s = mid_station_idx * DEFAULT_SAMPLING_INTERVAL_M

        pt = line.interpolate(target_s)
        p0 = line.interpolate(max(0.0, target_s - 0.5))
        p1 = line.interpolate(min(line.length, target_s + 0.5))
        dx, dy = p1.x - p0.x, p1.y - p0.y
        norm = math.hypot(dx, dy) or 1.0
        tx, ty = dx / norm, dy / norm
        nx_val, ny_val = -ty, tx

        bucket = idx % 8
        if bucket == 1:
            # Leave ~2.8m clear span on [-4.0m, -1.2m] (blocks Fire Engine >3.5m, allows Ambulance & ATV)
            debris_patches.append(
                _make_oriented_road_patch(pt, tx, ty, nx_val, ny_val, -3.0, 3.0, -1.2, half_w + 1.0)
            )
        elif bucket == 3:
            # Leave ~1.7m clear span on [-4.0m, -2.3m] (blocks Fire Engine & Ambulance, allows ATV >1.2m)
            debris_patches.append(
                _make_oriented_road_patch(pt, tx, ty, nx_val, ny_val, -3.0, 3.0, -2.3, half_w + 1.0)
            )
        elif bucket == 5:
            # Leave ~0.6m clear span on [-4.0m, -3.4m] (blocks all vehicles)
            debris_patches.append(
                _make_oriented_road_patch(pt, tx, ty, nx_val, ny_val, -3.0, 3.0, -3.4, half_w + 1.0)
            )
        elif bucket == 7:
            # Coastal storm-surge water washout across road (weight = infinity)
            water_patches.append(
                _make_oriented_road_patch(pt, tx, ty, nx_val, ny_val, -3.5, 3.5, -half_w - 1.0, half_w + 1.0)
            )

    return {
        "road_surface": _normalize_to_multipolygon(road_buffers),
        "debris": _normalize_to_multipolygon(debris_patches),
        "water": _normalize_to_multipolygon(water_patches),
    }


def load_sam31_masks_utm(
    masks_geojson_path: str | Path = "data/sam31_mexico_beach_masks.geojson",
    target_crs: str = TARGET_METRIC_CRS,
) -> dict[str, Any]:
    """
    Load baked SAM 3.1 vector masks (`data/sam31_mexico_beach_masks.geojson`) and reproject
    into `EPSG:32616` (`target_crs`), preserving per-feature `confidence_score` and `mask_source`.
    """
    path = Path(masks_geojson_path)
    if not path.exists():
        return {
            "road_surface": MultiPolygon(),
            "debris": MultiPolygon(),
            "water": MultiPolygon(),
            "features_utm": [],
            "mean_confidence": 0.85,
            "mask_source": "sam-3.1",
        }

    gdf = gpd.read_file(path)
    if gdf.empty:
        return {
            "road_surface": MultiPolygon(),
            "debris": MultiPolygon(),
            "water": MultiPolygon(),
            "features_utm": [],
            "mean_confidence": 0.85,
            "mask_source": "sam-3.1",
        }

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84_CRS)
    gdf_utm = gdf.to_crs(target_crs)

    road_polys: list[Polygon] = []
    debris_polys: list[Polygon] = []
    water_polys: list[Polygon] = []
    features_utm: list[dict[str, Any]] = []
    conf_vals: list[float] = []

    for _, row in gdf_utm.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        mp = _normalize_to_multipolygon(geom)
        label = str(
            row.get("class_label")
            or row.get("prompt")
            or row.get("feature_type")
            or "debris"
        ).lower()
        conf = float(row.get("confidence_score", 0.78))
        src_tag = str(row.get("mask_source", "sam-3.1"))
        conf_vals.append(conf)

        if "water" in label:
            water_polys.extend(mp.geoms)
            norm_label = "water"
        elif "road" in label:
            road_polys.extend(mp.geoms)
            norm_label = "road_surface"
        else:
            debris_polys.extend(mp.geoms)
            norm_label = "debris"

        for poly in mp.geoms:
            features_utm.append(
                {
                    "geometry": poly,
                    "class_label": norm_label,
                    "confidence_score": conf,
                    "mask_source": src_tag,
                }
            )

    mean_conf = float(sum(conf_vals) / len(conf_vals)) if conf_vals else 0.80
    return {
        "road_surface": _normalize_to_multipolygon(road_polys),
        "debris": _normalize_to_multipolygon(debris_polys),
        "water": _normalize_to_multipolygon(water_polys),
        "features_utm": features_utm,
        "mean_confidence": mean_conf,
        "mask_source": "sam-3.1",
    }


def evaluate_road_network_clearance(
    roads_source: str | Path | gpd.GeoDataFrame,
    raster_path: str | Path | None = None,
    *,
    debris_mask_utm: Polygon | MultiPolygon | np.ndarray | None = None,
    water_mask_utm: Polygon | MultiPolygon | np.ndarray | None = None,
    road_surface_mask_utm: Polygon | MultiPolygon | np.ndarray | None = None,
    masks_geojson_path: str | Path | None = "data/sam31_mexico_beach_masks.geojson",
    mask_transform: Affine | None = None,
    mask_crs: str = TARGET_METRIC_CRS,
    sampling_interval_m: float = DEFAULT_SAMPLING_INTERVAL_M,
    ray_half_width_m: float = DEFAULT_RAY_HALF_WIDTH_M,
    nominal_road_half_width_m: float = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M,
    simulate_if_no_masks: bool = True,
) -> tuple[gpd.GeoDataFrame, list[RoadEdgeClearance]]:
    """
    Reproject `osm_roads.geojson` into `EPSG:32616`, intersect 5m orthogonal cross-sections
    against debris and water masks, and return both an enriched `GeoDataFrame` (in `EPSG:32616`)
    and a list of `RoadEdgeClearance` telemetry records.
    """
    gsd_m_per_px = 0.2804
    if raster_path is not None and Path(raster_path).exists():
        raster_meta = compute_raster_gsd_and_crs(raster_path, target_crs=TARGET_METRIC_CRS)
        gsd_m_per_px = float(raster_meta["gsd_m_per_px"])
        if mask_transform is None:
            mask_transform = raster_meta["metric_transform"]

    roads_utm = load_and_align_roads(
        roads_source=roads_source,
        raster_path=raster_path,
        target_crs=TARGET_METRIC_CRS,
    )

    if isinstance(debris_mask_utm, np.ndarray):
        if mask_transform is None:
            raise ValueError("mask_transform is required when debris_mask_utm is a numpy array.")
        debris_poly = mask_array_to_utm_polygons(
            debris_mask_utm, mask_transform, src_crs=mask_crs, target_crs=TARGET_METRIC_CRS
        )
    else:
        debris_poly = _normalize_to_multipolygon(debris_mask_utm)

    if isinstance(water_mask_utm, np.ndarray):
        if mask_transform is None:
            raise ValueError("mask_transform is required when water_mask_utm is a numpy array.")
        water_poly = mask_array_to_utm_polygons(
            water_mask_utm, mask_transform, src_crs=mask_crs, target_crs=TARGET_METRIC_CRS
        )
    else:
        water_poly = _normalize_to_multipolygon(water_mask_utm)

    if isinstance(road_surface_mask_utm, np.ndarray):
        if mask_transform is None:
            raise ValueError("mask_transform is required when road_surface_mask_utm is a numpy array.")
        road_poly = mask_array_to_utm_polygons(
            road_surface_mask_utm, mask_transform, src_crs=mask_crs, target_crs=TARGET_METRIC_CRS
        )
    else:
        road_poly = _normalize_to_multipolygon(road_surface_mask_utm)

    sam_features_utm: list[dict[str, Any]] = []
    default_sam_conf = 0.82
    active_mask_source = "sam-3.1"

    if debris_mask_utm is None and water_mask_utm is None:
        if masks_geojson_path is not None and Path(masks_geojson_path).exists():
            baked = load_sam31_masks_utm(masks_geojson_path, target_crs=TARGET_METRIC_CRS)
            debris_poly = baked["debris"]
            water_poly = baked["water"]
            sam_features_utm = baked.get("features_utm", [])
            default_sam_conf = float(baked.get("mean_confidence", 0.78))
            active_mask_source = str(baked.get("mask_source", "sam-3.1"))
        elif simulate_if_no_masks:
            sim_masks = generate_simulated_disaster_masks_utm(roads_utm)
            debris_poly = sim_masks["debris"]
            water_poly = sim_masks["water"]
            active_mask_source = "simulated"

    results: list[RoadEdgeClearance] = []
    for idx, row in roads_utm.iterrows():
        geom = row.geometry
        if isinstance(geom, MultiLineString):
            line_utm = max(geom.geoms, key=lambda g: g.length)
        elif isinstance(geom, LineString):
            line_utm = geom
        else:
            continue

        raw_name = row.get("name") if hasattr(row, "get") else None
        if isinstance(raw_name, list):
            name_str = ", ".join(str(x) for x in raw_name if x)
        elif raw_name is not None and not (isinstance(raw_name, float) and math.isnan(raw_name)):
            name_str = str(raw_name)
        else:
            name_str = f"Edge-{idx}"

        u_val = int(row["u"]) if "u" in row and row["u"] is not None and not pd_isna(row["u"]) else None
        v_val = int(row["v"]) if "v" in row and row["v"] is not None and not pd_isna(row["v"]) else None

        # Compute per-corridor SAM 3.1 confidence from intersecting mask features
        corridor_buf = line_utm.buffer(nominal_road_half_width_m + 2.0)
        hit_confs = [
            float(f["confidence_score"])
            for f in sam_features_utm
            if f["geometry"].intersects(corridor_buf)
        ]
        edge_sam_conf = float(min(hit_confs)) if hit_confs else default_sam_conf
        occluded = bool(row.get("canopy_occluded", False) if hasattr(row, "get") else False)

        edge_clearance = measure_road_edge_clearance(
            centerline_utm=line_utm,
            debris_mask_utm=debris_poly,
            water_mask_utm=water_poly,
            road_surface_mask_utm=road_poly if not road_poly.is_empty else None,
            edge_id=idx,
            u=u_val,
            v=v_val,
            name=name_str,
            gsd_m_per_px=gsd_m_per_px,
            sampling_interval_m=sampling_interval_m,
            ray_half_width_m=ray_half_width_m,
            nominal_road_half_width_m=nominal_road_half_width_m,
            utm_crs=TARGET_METRIC_CRS,
            sam_confidence=edge_sam_conf,
            canopy_or_shadow_occluded=occluded,
            mask_source=active_mask_source,
        )
        results.append(edge_clearance)

    enriched = roads_utm.copy()
    if results:
        enriched["length_m"] = [r.length_m for r in results]
        enriched["gsd_m_per_px"] = [r.gsd_m_per_px for r in results]
        enriched["min_clear_width_m"] = [r.min_clear_width_m for r in results]
        enriched["mean_clear_width_m"] = [r.mean_clear_width_m for r in results]
        enriched["has_water_obstruction"] = [r.has_water_obstruction for r in results]
        enriched["debris_area_m2"] = [r.debris_area_m2 for r in results]
        enriched["water_area_m2"] = [r.water_area_m2 for r in results]
        enriched["passable_atv"] = [r.passability.atv for r in results]
        enriched["passable_ambulance"] = [r.passability.ambulance for r in results]
        enriched["passable_fire_engine"] = [r.passability.fire_engine for r in results]
        enriched["weight_atv"] = [r.edge_weight_atv for r in results]
        enriched["weight_ambulance"] = [r.edge_weight_ambulance for r in results]
        enriched["weight_fire_engine"] = [r.edge_weight_fire_engine for r in results]
        enriched["sam_confidence"] = [r.sam_confidence for r in results]
        enriched["needs_drone_recon"] = [r.needs_drone_recon for r in results]
        enriched["mask_source"] = [r.mask_source for r in results]
        enriched["clearance_reason"] = [r.reason for r in results]

    return enriched, results


def pd_isna(val: Any) -> bool:
    """Lightweight NaN/None check without requiring pandas import overhead."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


class SAM31GeometricInstrument:
    """
    SAM 3.1 Geometric Measurement Instrument wrapper for Apple Silicon M1 Max.

    Enforces `torch.float16` (`fp16`) on the PyTorch `mps` device and provides:
      - `load_weights()` / `unload_weights()`: unified memory lifecycle management
      - `load_model()`: loads local HuggingFace `Sam3Model` & `Sam3Processor` (`models/sam3.1`)
        onto Apple Silicon MPS in `torch.float16`
      - `segment_tile()`: performs real text-prompted segmentation (`["road", "debris", "standing water"]`)
        returning binary masks and real SAM 3.1 `confidence_score` (0.0–1.0)
    """

    DEFAULT_PROMPTS: tuple[str, ...] = ("road", "debris", "standing water")

    def __init__(
        self,
        checkpoint_dir: str | Path = "models/sam3.1",
        device: str | None = None,
    ) -> None:
        import torch

        self.checkpoint_dir = Path(checkpoint_dir)
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        self.dtype = torch.float16
        self._loaded_tensors: int = 0
        self._probe_tensor: Any = None
        self.model: Any = None
        self.processor: Any = None

    def load_weights(self) -> dict[str, Any]:
        """Load SAM 3.1 checkpoint state offline in `torch.float16` onto Apple Silicon MPS."""
        import torch
        from safetensors.torch import load_file

        safetensors_path = self.checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"SAM 3.1 checkpoint not found at {safetensors_path}")

        state_dict = load_file(str(safetensors_path), device="cpu")
        self._loaded_tensors = len(state_dict)
        first_key = next(iter(state_dict.keys()))
        self._probe_tensor = state_dict[first_key].to(device=self.device, dtype=torch.float16)
        del state_dict
        gc.collect()
        return {
            "checkpoint": str(safetensors_path),
            "device": str(self._probe_tensor.device),
            "dtype": str(self._probe_tensor.dtype),
            "tensor_count": self._loaded_tensors,
        }

    def load_model(self) -> tuple[Any, Any]:
        """Load full `Sam3Processor` and `Sam3Model` in `torch.float16` on Apple Silicon MPS."""
        import torch
        from transformers import Sam3Model, Sam3Processor

        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        self.processor = Sam3Processor.from_pretrained(str(self.checkpoint_dir))
        self.model = Sam3Model.from_pretrained(
            str(self.checkpoint_dir),
            dtype=torch.float16,
        ).to(self.device, dtype=torch.float16)
        self.model.eval()
        return self.model, self.processor

    def segment_tile(
        self,
        image: Any,
        prompts: list[str] | tuple[str, ...] = DEFAULT_PROMPTS,
        *,
        mask_threshold: float = 0.5,
    ) -> dict[str, dict[str, Any]]:
        """
        Execute real SAM 3.1 segmentation on an aerial tile using `Sam3Model` (`float16` on MPS)
        for text prompts `["road", "debris", "standing water"]`.

        Returns a dictionary mapping each prompt to:
          - `mask`: 2D `uint8` binary mask (`(H, W)`)
          - `confidence_score`: float in `[0.0, 1.0]` from `Sam3Model` query/semantic logits
          - `mask_source`: `"sam-3.1"`
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if isinstance(image, np.ndarray):
            arr = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type for SAM31GeometricInstrument: {type(image)}")

        width_px, height_px = pil_img.size
        model, processor = self.load_model()

        img_inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self.device, dtype=torch.float16)

        results: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            vision_embeds = model.get_vision_features(pixel_values)
            for prompt in prompts:
                txt_inputs = processor(text=prompt, return_tensors="pt")
                out = model(
                    vision_embeds=vision_embeds,
                    input_ids=txt_inputs["input_ids"].to(self.device),
                    attention_mask=txt_inputs["attention_mask"].to(self.device),
                )
                query_probs = out.pred_logits.float().sigmoid()[0]
                top_vals, top_idx = torch.topk(query_probs, k=min(5, query_probs.numel()))
                top_conf = float(top_vals[0].item())

                # Combine semantic segmentation map with top-3 query masks and upsample to (H, W)
                sem_prob = out.semantic_seg.float().sigmoid()  # [1, 1, 288, 288]
                q_prob = out.pred_masks[:, top_idx[:3]].float().sigmoid().max(dim=1, keepdim=True).values
                combined_prob = torch.maximum(sem_prob, q_prob)
                upsampled = F.interpolate(
                    combined_prob,
                    size=(height_px, width_px),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                binary_mask = (upsampled >= mask_threshold).cpu().numpy().astype(np.uint8)
                pos_pixels = upsampled[upsampled >= mask_threshold]
                mean_pos_prob = float(pos_pixels.mean().item()) if pos_pixels.numel() > 0 else top_conf
                confidence_score = round(float(0.6 * top_conf + 0.4 * mean_pos_prob), 4)

                results[prompt] = {
                    "mask": binary_mask,
                    "confidence_score": max(0.0, min(1.0, confidence_score)),
                    "top_query_confidence": round(top_conf, 4),
                    "positive_pixels": int(binary_mask.sum()),
                    "mask_source": "sam-3.1",
                }

        return results

    def unload_weights(self) -> None:
        """Explicitly release MPS unified memory cache."""
        import torch

        self._probe_tensor = None
        self._loaded_tensors = 0
        self.model = None
        self.processor = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def build_edge_caliper_geojson(
    centerline_utm: LineString,
    debris_mask_utm: Polygon | MultiPolygon | None = None,
    water_mask_utm: Polygon | MultiPolygon | None = None,
    road_surface_mask_utm: Polygon | MultiPolygon | None = None,
    *,
    edge_id: str | int = 0,
    u: int | str | None = None,
    v: int | str | None = None,
    name: str = "Unnamed Road",
    gsd_m_per_px: float = 0.2804,
    sampling_interval_m: float = DEFAULT_SAMPLING_INTERVAL_M,
    ray_half_width_m: float = DEFAULT_RAY_HALF_WIDTH_M,
    nominal_road_half_width_m: float = DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M,
    utm_crs: str = TARGET_METRIC_CRS,
    sam_confidence: float = 0.85,
    canopy_or_shadow_occluded: bool = False,
    mask_source: str = "sam-3.1",
) -> dict[str, Any]:
    """
    Build a GeoJSON FeatureCollection in EPSG:4326 (`WGS84`) for a road edge containing:
      1. SAM 3.1 debris and water obstacle polygons intersecting or adjacent to the corridor.
      2. The 5.0m orthogonal cross-section caliper rays cast along the road centerline,
         annotated with `station_m`, `clear_width_m`, `is_pinch_point`, and vehicle
         passability flags (`atv`, `ambulance`, `fire_engine`).
    """
    clearance = measure_road_edge_clearance(
        centerline_utm=centerline_utm,
        debris_mask_utm=debris_mask_utm,
        water_mask_utm=water_mask_utm,
        road_surface_mask_utm=road_surface_mask_utm,
        edge_id=edge_id,
        u=int(u) if isinstance(u, int) else None,
        v=int(v) if isinstance(v, int) else None,
        name=name,
        gsd_m_per_px=gsd_m_per_px,
        sampling_interval_m=sampling_interval_m,
        ray_half_width_m=ray_half_width_m,
        nominal_road_half_width_m=nominal_road_half_width_m,
        utm_crs=utm_crs,
        sam_confidence=sam_confidence,
        canopy_or_shadow_occluded=canopy_or_shadow_occluded,
        mask_source=mask_source,
    )

    to_wgs84 = Transformer.from_crs(utm_crs, WGS84_CRS, always_xy=True)
    corridor_search_utm = centerline_utm.buffer(
        ray_half_width_m + 4.0, cap_style="round", join_style="round"
    )
    road_corridor_utm = (
        _normalize_to_multipolygon(road_surface_mask_utm)
        if (road_surface_mask_utm is not None and not road_surface_mask_utm.is_empty)
        else _normalize_to_multipolygon(
            centerline_utm.buffer(nominal_road_half_width_m, cap_style="round", join_style="round")
        )
    )

    debris_mp = _normalize_to_multipolygon(debris_mask_utm)
    water_mp = _normalize_to_multipolygon(water_mask_utm)

    local_debris_utm = (
        debris_mp.intersection(corridor_search_utm)
        if not debris_mp.is_empty
        else MultiPolygon()
    )
    local_water_utm = (
        water_mp.intersection(corridor_search_utm)
        if not water_mp.is_empty
        else MultiPolygon()
    )

    features: list[dict[str, Any]] = []

    # 1. Road Corridor Polygon Feature
    if not road_corridor_utm.is_empty:
        corridor_wgs84 = shapely_transform(to_wgs84.transform, road_corridor_utm)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(corridor_wgs84),
                "properties": {
                    "feature_type": "road_corridor",
                    "edge_id": str(edge_id),
                    "road_name": name,
                    "nominal_width_m": round(nominal_road_half_width_m * 2.0, 2),
                    "gsd_m_per_px": round(gsd_m_per_px, 4),
                },
            }
        )

    # 2. SAM 3.1 Debris Obstacle Polygons
    local_debris_mp = _normalize_to_multipolygon(local_debris_utm)
    for idx, poly in enumerate(local_debris_mp.geoms):
        poly_wgs84 = shapely_transform(to_wgs84.transform, poly)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(poly_wgs84),
                "properties": {
                    "feature_type": "debris_mask",
                    "obstacle_id": f"debris-{edge_id}-{idx}",
                    "edge_id": str(edge_id),
                    "area_m2": round(float(poly.area), 3),
                    "gsd_m_per_px": round(gsd_m_per_px, 4),
                },
            }
        )

    # 3. SAM 3.1 Water Obstacle Polygons
    local_water_mp = _normalize_to_multipolygon(local_water_utm)
    for idx, poly in enumerate(local_water_mp.geoms):
        poly_wgs84 = shapely_transform(to_wgs84.transform, poly)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(poly_wgs84),
                "properties": {
                    "feature_type": "water_mask",
                    "obstacle_id": f"water-{edge_id}-{idx}",
                    "edge_id": str(edge_id),
                    "area_m2": round(float(poly.area), 3),
                    "gsd_m_per_px": round(gsd_m_per_px, 4),
                },
            }
        )

    # 4. Orthogonal 5.0m Cross-Section Caliper Rays
    pinch_idx = -1
    if clearance.cross_sections:
        pinch_idx = min(
            range(len(clearance.cross_sections)),
            key=lambda i: clearance.cross_sections[i].clear_width_m,
        )

    for idx, cs in enumerate(clearance.cross_sections):
        is_pinch = idx == pinch_idx
        cs_flags = compute_passability_flags(
            min_clear_width_m=cs.clear_width_m,
            has_water_obstruction=cs.water_intersected,
        )
        # Use clear_segment_utm if non-degenerate (>0.05m), else full ray_line_utm so ray is visible
        active_line_utm = (
            cs.clear_segment_utm
            if cs.clear_segment_utm.length > 0.05
            else cs.ray_line_utm
        )
        ray_wgs84 = shapely_transform(to_wgs84.transform, active_line_utm)
        full_ray_wgs84 = shapely_transform(to_wgs84.transform, cs.ray_line_utm)

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(ray_wgs84),
                "properties": {
                    "feature_type": "caliper_ray",
                    "edge_id": str(edge_id),
                    "ray_index": idx,
                    "station_m": round(cs.station_m, 3),
                    "clear_width_m": round(cs.clear_width_m, 3),
                    "is_pinch_point": is_pinch,
                    "water_intersected": cs.water_intersected,
                    "debris_intersected": cs.debris_intersected,
                    "passability": cs_flags.to_dict(),
                    "atv": cs_flags.atv,
                    "ambulance": cs_flags.ambulance,
                    "fire_engine": cs_flags.fire_engine,
                    "full_ray_coords_wgs84": list(full_ray_wgs84.coords),
                    "gsd_m_per_px": round(gsd_m_per_px, 4),
                },
            }
        )

    pinch_station_m = (
        round(clearance.cross_sections[pinch_idx].station_m, 3)
        if pinch_idx >= 0
        else 0.0
    )

    return {
        "type": "FeatureCollection",
        "properties": {
            "edge_id": str(edge_id),
            "u": u,
            "v": v,
            "road_name": name,
            "crs": WGS84_CRS,
            "metric_crs": utm_crs,
            "gsd_m_per_px": round(gsd_m_per_px, 4),
            "length_m": round(clearance.length_m, 3),
            "min_clear_width_m": round(clearance.min_clear_width_m, 3),
            "mean_clear_width_m": round(clearance.mean_clear_width_m, 3),
            "max_clear_width_m": round(clearance.max_clear_width_m, 3),
            "pinch_point_station_m": pinch_station_m,
            "bottleneck_lonlat": list(clearance.bottleneck_point_lonlat),
            "debris_area_m2": round(clearance.debris_area_m2, 3),
            "water_area_m2": round(clearance.water_area_m2, 3),
            "has_water_obstruction": clearance.has_water_obstruction,
            "passability": clearance.passability.to_dict(),
            "sam_confidence": round(clearance.sam_confidence, 4),
            "needs_drone_recon": clearance.needs_drone_recon,
            "mask_source": clearance.mask_source,
            "reason": clearance.reason,
            "ray_count": len(clearance.cross_sections),
        },
        "features": features,
    }


def extract_corridor_aerial_crop_png(
    raster_path: str | Path,
    lon: float,
    lat: float,
    size_px: int = 256,
) -> bytes:
    """
    Extract a real `size_px x size_px` (256x256) RGB PNG aerial patch centered on
    `(lon, lat)` directly from `data/noaa_michael_mexico_beach.tif` using Rasterio
    window reading.
    """
    import io
    from PIL import Image

    raster_path = Path(raster_path)
    if not raster_path.exists():
        img = Image.new("RGB", (size_px, size_px), color=(30, 41, 59))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_string() != WGS84_CRS:
            transformer = Transformer.from_crs(WGS84_CRS, src.crs, always_xy=True)
            x_coord, y_coord = transformer.transform(lon, lat)
        else:
            x_coord, y_coord = lon, lat

        row, col = src.index(x_coord, y_coord)
        half = size_px // 2

        col_off = max(0, min(int(col - half), max(0, src.width - size_px)))
        row_off = max(0, min(int(row - half), max(0, src.height - size_px)))
        win_w = min(size_px, src.width - col_off)
        win_h = min(size_px, src.height - row_off)

        window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
        band_indices = [1, 2, 3] if src.count >= 3 else [1] * 3
        arr = src.read(band_indices, window=window)

        rgb = np.transpose(arr, (1, 2, 0))
        img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        if img.size != (size_px, size_px):
            img = img.resize((size_px, size_px))

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

