"""
SAM 3.1 visible-pavement width measurement (`src/sam_measure.py`).

SAM 3.1 is run once per aerial tile with the text prompts "road", "debris" and
"standing water" (see `scripts/bake_sam31_masks.py`), producing per-pixel probability
maps. This module turns those maps into metric clear widths along every OSM road:

1. Every 5 m along the UTM centerline, cast a 30 m orthogonal ray (±15 m) and sample the
   SAM probabilities every 0.25 m.
2. The clear width at that station is the contiguous run of SAM "road" pixels
   (p >= 0.5) nearest the OSM centerline. Picking the nearest run absorbs the few-metre
   misalignment between OSM centerlines and the NOAA imagery.
3. When no pavement is visible near the centerline, the gap is attributed to standing
   water, debris, or "not visible" (tree canopy, shadow, misalignment) from the other SAM
   maps. Not-visible gaps are treated as impassable but flagged for drone reconnaissance.
4. An obstruction must persist over two consecutive stations (>= 5 m of road) before it
   sets the edge's minimum clear width, so a single parked car or shadow cannot close a road.

All functions are pure: probability lookups are injected via `ProbabilitySampler`, so the
measurement math is unit-testable without loading SAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np
from shapely.geometry import LineString

from src.geometry import (
    AMBULANCE_MIN_WIDTH_M,
    ATV_MIN_WIDTH_M,
    FIRE_ENGINE_MIN_WIDTH_M,
    compute_passability_flags,
)

SAM_PROMPTS: dict[str, str] = {
    "road": "road",
    "debris": "debris",
    "water": "standing water",
}
STATION_INTERVAL_M = 5.0
RAY_HALF_WIDTH_M = 15.0
RAY_SAMPLE_STEP_M = 0.25
PAVEMENT_THRESHOLD = 0.5
MAX_CENTER_OFFSET_M = 10.0
ENDPOINT_MARGIN_M = 2.5
CAUSE_WINDOW_M = 5.0
DEBRIS_CAUSE_THRESHOLD = 0.4
WATER_CAUSE_THRESHOLD = 0.5
DRONE_RECON_CONFIDENCE = 0.6
THRESHOLD_BAND_M = 0.35

# (n_points, 2) UTM xy -> {"road": p[], "debris": p[], "water": p[]} in [0, 1]
ProbabilitySampler = Callable[[np.ndarray], dict[str, np.ndarray]]


@dataclass
class StationMeasurement:
    """One orthogonal cross-section measured against the SAM 3.1 probability maps."""

    station_m: float
    center_utm: tuple[float, float]
    ray_utm: tuple[tuple[float, float], tuple[float, float]]
    clear_segment_utm: tuple[tuple[float, float], tuple[float, float]] | None
    clear_width_m: float
    center_offset_m: float | None
    cause: str  # "clear" | "narrowed_debris" | "debris" | "water" | "not_visible"
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_m": round(self.station_m, 2),
            "center_utm": [round(c, 2) for c in self.center_utm],
            "ray_utm": [[round(c, 2) for c in p] for p in self.ray_utm],
            "clear_segment_utm": (
                [[round(c, 2) for c in p] for p in self.clear_segment_utm]
                if self.clear_segment_utm
                else None
            ),
            "clear_width_m": round(self.clear_width_m, 2),
            "center_offset_m": None if self.center_offset_m is None else round(self.center_offset_m, 2),
            "cause": self.cause,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class EdgeMeasurement:
    """Edge-level clearance derived from SAM 3.1 station measurements."""

    length_m: float
    min_clear_width_m: float
    pinch_station_m: float
    has_water_obstruction: bool
    debris_area_m2: float
    water_area_m2: float
    sam_confidence: float
    needs_drone_recon: bool
    not_visible_stations: int
    stations: list[StationMeasurement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_m": round(self.length_m, 2),
            "min_clear_width_m": round(self.min_clear_width_m, 2),
            "pinch_station_m": round(self.pinch_station_m, 2),
            "has_water_obstruction": self.has_water_obstruction,
            "debris_area_m2": round(self.debris_area_m2, 2),
            "water_area_m2": round(self.water_area_m2, 2),
            "sam_confidence": round(self.sam_confidence, 3),
            "needs_drone_recon": self.needs_drone_recon,
            "not_visible_stations": self.not_visible_stations,
            "stations": [s.to_dict() for s in self.stations],
        }


def station_positions(length_m: float, interval_m: float = STATION_INTERVAL_M) -> list[float]:
    """Stations every `interval_m`, inset from intersections where rays run along cross streets."""
    if length_m <= 2.0 * ENDPOINT_MARGIN_M:
        return [length_m / 2.0]
    usable = length_m - 2.0 * ENDPOINT_MARGIN_M
    count = max(1, int(math.floor(usable / interval_m)) + 1)
    if count == 1:
        return [length_m / 2.0]
    spacing = usable / (count - 1)
    return [ENDPOINT_MARGIN_M + i * spacing for i in range(count)]


def _unit_normal(line: LineString, s: float) -> tuple[float, float]:
    delta = min(1.0, max(0.05, line.length * 0.05))
    p0 = line.interpolate(max(0.0, s - delta))
    p1 = line.interpolate(min(line.length, s + delta))
    dx, dy = p1.x - p0.x, p1.y - p0.y
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 0.0, 1.0
    return -dy / norm, dx / norm


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of consecutive True values."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def measure_station(
    line: LineString,
    station_m: float,
    sampler: ProbabilitySampler,
    *,
    half_width_m: float = RAY_HALF_WIDTH_M,
    step_m: float = RAY_SAMPLE_STEP_M,
) -> tuple[StationMeasurement, dict[str, np.ndarray], np.ndarray]:
    """Measure one cross-section; also returns the raw samples for area integration."""
    center = line.interpolate(station_m)
    nx_, ny_ = _unit_normal(line, station_m)
    offsets = np.arange(-half_width_m, half_width_m + 1e-9, step_m)
    pts = np.column_stack([center.x + offsets * nx_, center.y + offsets * ny_])
    probs = sampler(pts)
    road = probs["road"]

    pavement = road >= PAVEMENT_THRESHOLD
    candidates = []
    for a, b in _runs(pavement):
        lo, hi = offsets[a] - step_m / 2.0, offsets[b] + step_m / 2.0
        dist = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
        if dist <= MAX_CENTER_OFFSET_M:
            candidates.append((dist, -(hi - lo), a, b, lo, hi))

    near = np.abs(offsets) <= CAUSE_WINDOW_M
    debris_near = float(probs["debris"][near].mean()) if near.any() else 0.0
    water_near = float(probs["water"][near].mean()) if near.any() else 0.0
    # Certainty of the pavement decision across the central ±10 m of the ray.
    central = np.abs(offsets) <= MAX_CENTER_OFFSET_M
    confidence = float(np.mean(np.abs(2.0 * road[central] - 1.0))) if central.any() else 0.0

    ray = ((float(pts[0, 0]), float(pts[0, 1])), (float(pts[-1, 0]), float(pts[-1, 1])))
    if candidates:
        dist, _, a, b, lo, hi = min(candidates)
        width = float(hi - lo)
        seg = (
            (center.x + lo * nx_, center.y + lo * ny_),
            (center.x + hi * nx_, center.y + hi * ny_),
        )
        edge_flank = np.zeros_like(pavement)
        flank = int(round(1.5 / step_m))
        edge_flank[max(0, a - flank) : a] = True
        edge_flank[b + 1 : b + 1 + flank] = True
        narrowed = bool(edge_flank.any() and probs["debris"][edge_flank].mean() >= DEBRIS_CAUSE_THRESHOLD)
        measurement = StationMeasurement(
            station_m=station_m,
            center_utm=(center.x, center.y),
            ray_utm=ray,
            clear_segment_utm=seg,
            clear_width_m=width,
            center_offset_m=float(dist if lo > 0 else -dist),
            cause="narrowed_debris" if narrowed else "clear",
            confidence=confidence,
        )
    else:
        if water_near >= WATER_CAUSE_THRESHOLD:
            cause = "water"
        elif debris_near >= DEBRIS_CAUSE_THRESHOLD:
            cause = "debris"
        else:
            cause = "not_visible"
        measurement = StationMeasurement(
            station_m=station_m,
            center_utm=(center.x, center.y),
            ray_utm=ray,
            clear_segment_utm=None,
            clear_width_m=0.0,
            center_offset_m=None,
            cause=cause,
            confidence=confidence,
        )
    return measurement, probs, offsets


def _sustained_min(stations: list[StationMeasurement]) -> tuple[float, int]:
    """
    Minimum width that persists over two consecutive stations (an obstruction >= one
    station interval long). Returns (width, index of the narrower station of that pair).
    """
    if len(stations) == 1:
        return stations[0].clear_width_m, 0
    best_w, best_i = math.inf, 0
    for i in range(len(stations) - 1):
        pair_w = max(stations[i].clear_width_m, stations[i + 1].clear_width_m)
        if pair_w < best_w:
            best_w = pair_w
            best_i = i if stations[i].clear_width_m <= stations[i + 1].clear_width_m else i + 1
    return float(best_w), best_i


def measure_edge(
    line: LineString,
    sampler: ProbabilitySampler,
    *,
    interval_m: float = STATION_INTERVAL_M,
    half_width_m: float = RAY_HALF_WIDTH_M,
    step_m: float = RAY_SAMPLE_STEP_M,
) -> EdgeMeasurement:
    """Measure a UTM road centerline against SAM 3.1 probability maps."""
    stations: list[StationMeasurement] = []
    debris_area = 0.0
    water_area = 0.0
    for s in station_positions(line.length, interval_m):
        m, probs, offsets = measure_station(line, s, sampler, half_width_m=half_width_m, step_m=step_m)
        stations.append(m)
        # Integrate SAM debris/water pixels within ±5 m of the road over this station's
        # 5 m slice: area ~= (#samples above threshold) * step * interval.
        near = np.abs(offsets) <= CAUSE_WINDOW_M
        debris_area += float((probs["debris"][near] >= 0.5).sum()) * step_m * interval_m
        water_area += float((probs["water"][near] >= 0.5).sum()) * step_m * interval_m

    min_w, pinch_i = _sustained_min(stations)
    causes = [st.cause for st in stations]
    water_run = len(stations) == 1 and causes[0] == "water"
    for i in range(len(causes) - 1):
        if causes[i] == "water" and causes[i + 1] == "water":
            water_run = True
            break

    confidence = float(np.mean([st.confidence for st in stations])) if stations else 0.0
    not_visible = sum(1 for c in causes if c == "not_visible")
    near_threshold = any(
        abs(min_w - t) <= THRESHOLD_BAND_M
        for t in (ATV_MIN_WIDTH_M, AMBULANCE_MIN_WIDTH_M, FIRE_ENGINE_MIN_WIDTH_M)
    )
    pinch_is_not_visible = stations[pinch_i].cause == "not_visible" if stations else False

    return EdgeMeasurement(
        length_m=float(line.length),
        min_clear_width_m=0.0 if water_run else min_w,
        pinch_station_m=stations[pinch_i].station_m if stations else 0.0,
        has_water_obstruction=water_run,
        debris_area_m2=debris_area,
        water_area_m2=water_area,
        sam_confidence=confidence,
        needs_drone_recon=bool(
            confidence < DRONE_RECON_CONFIDENCE or pinch_is_not_visible or near_threshold
        ),
        not_visible_stations=not_visible,
        stations=stations,
    )


def describe_edge_measurement(m: EdgeMeasurement, gsd_m_per_px: float) -> str:
    """Plain, number-backed reason string for one measured edge."""
    flags = compute_passability_flags(m.min_clear_width_m, m.has_water_obstruction)
    if m.has_water_obstruction:
        return (
            f"SAM 3.1: standing water over the road at {m.pinch_station_m:.0f} m "
            f"({m.water_area_m2:.0f} m² water within 5 m, {gsd_m_per_px:.4f} m/px); impassable."
        )
    passing = [
        label
        for ok, label in (
            (flags.fire_engine, "Fire Engine"),
            (flags.ambulance, "Ambulance"),
            (flags.atv, "ATV"),
        )
        if ok
    ]
    verdict = f"passable for {', '.join(passing)}" if passing else "blocked for all vehicle classes"
    pinch_cause = next(
        (st.cause for st in m.stations if abs(st.station_m - m.pinch_station_m) < 1e-6), "clear"
    )
    cause_note = {
        "debris": "pavement buried by SAM-detected debris",
        "narrowed_debris": "debris narrowing the pavement",
        "not_visible": "no pavement visible (canopy/shadow) - verify by drone",
        "water": "standing water",
        "clear": "visible pavement",
    }[pinch_cause]
    return (
        f"SAM 3.1 visible pavement {m.min_clear_width_m:.2f} m at {m.pinch_station_m:.0f} m "
        f"({cause_note}; {m.debris_area_m2:.0f} m² debris within 5 m; "
        f"{gsd_m_per_px:.4f} m/px); {verdict}."
    )


def build_caliper_geojson_from_measurement(
    record: dict[str, Any],
    line_utm: LineString,
    masks_utm: dict[str, Any],
    *,
    edge_id: str,
    u: Any,
    v: Any,
    name: str,
    gsd_m_per_px: float,
    simulated_clearance: bool = False,
    utm_crs: str = "EPSG:32616",
) -> dict[str, Any]:
    """
    Render one baked SAM 3.1 edge measurement as the caliper GeoJSON (EPSG:4326) consumed by
    the dashboard: SAM pavement/debris/water masks near the road plus one feature per 5 m
    cross-section (the measured clear span, or the full ray where no pavement was visible).
    """
    from pyproj import Transformer
    from shapely.geometry import mapping
    from shapely.ops import transform as shapely_transform

    to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    corridor = line_utm.buffer(RAY_HALF_WIDTH_M + 3.0)
    features: list[dict[str, Any]] = []
    for mask_key, feature_type in (
        ("road_surface", "road_surface_mask"),
        ("debris", "debris_mask"),
        ("water", "water_mask"),
    ):
        geom = masks_utm.get(mask_key)
        if geom is None or geom.is_empty:
            continue
        clipped = geom.intersection(corridor)
        if clipped.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(shapely_transform(to_wgs84.transform, clipped)),
                "properties": {
                    "feature_type": feature_type,
                    "area_m2": round(float(clipped.area), 2),
                    "gsd_m_per_px": round(gsd_m_per_px, 4),
                    "mask_source": "sam-3.1",
                },
            }
        )

    pinch_station = float(record["pinch_station_m"])
    for idx, st in enumerate(record["stations"]):
        ray = [tuple(p) for p in st["ray_utm"]]
        seg = [tuple(p) for p in st["clear_segment_utm"]] if st["clear_segment_utm"] else ray
        flags = compute_passability_flags(st["clear_width_m"], st["cause"] == "water")
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(shapely_transform(to_wgs84.transform, LineString(seg))),
                "properties": {
                    "feature_type": "caliper_ray",
                    "edge_id": edge_id,
                    "ray_index": idx,
                    "station_m": st["station_m"],
                    "clear_width_m": st["clear_width_m"],
                    "cause": st["cause"],
                    "confidence": st["confidence"],
                    "is_pinch_point": abs(float(st["station_m"]) - pinch_station) < 1e-6,
                    "water_intersected": st["cause"] == "water",
                    "debris_intersected": st["cause"] in ("debris", "narrowed_debris"),
                    "passability": flags.to_dict(),
                    "atv": flags.atv,
                    "ambulance": flags.ambulance,
                    "fire_engine": flags.fire_engine,
                    "full_ray_coords_wgs84": [list(to_wgs84.transform(*p)) for p in ray],
                    "gsd_m_per_px": round(gsd_m_per_px, 4),
                },
            }
        )

    widths = [float(st["clear_width_m"]) for st in record["stations"]] or [0.0]
    min_w = float(record["min_clear_width_m"])
    has_water = bool(record["has_water_obstruction"])
    return {
        "type": "FeatureCollection",
        "properties": {
            "edge_id": edge_id,
            "u": u,
            "v": v,
            "road_name": name,
            "crs": "EPSG:4326",
            "metric_crs": utm_crs,
            "gsd_m_per_px": round(gsd_m_per_px, 4),
            "length_m": record["length_m"],
            "min_clear_width_m": min_w,
            "mean_clear_width_m": round(float(np.mean(widths)), 3),
            "max_clear_width_m": round(float(np.max(widths)), 3),
            "pinch_point_station_m": pinch_station,
            "bottleneck_lonlat": record.get("bottleneck_lonlat"),
            "debris_area_m2": record["debris_area_m2"],
            "water_area_m2": record["water_area_m2"],
            "has_water_obstruction": has_water,
            "passability": compute_passability_flags(min_w, has_water).to_dict(),
            "sam_confidence": record["sam_confidence"],
            "needs_drone_recon": record["needs_drone_recon"],
            "mask_source": "sam-3.1",
            "reason": record.get("reason", ""),
            "ray_count": len(record["stations"]),
            "simulated_clearance": simulated_clearance,
        },
        "features": features,
    }
