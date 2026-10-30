"""
Shared geospatial constants and helpers (`src/geometry.py`).

- Vehicle clear-width thresholds and the per-vehicle passability check used everywhere.
- Raster inspection (ground sample distance in metres, CRS) for the NOAA GeoTIFF.
- Loading the baked SAM 3.1 mask polygons for the measurement replay.
- `SAM31GeometricInstrument`: runs SAM 3.1 (fp16 on Apple Silicon MPS) for the bake script.
- Aerial crops of the NOAA image for the inspector and damage grading.

Widths themselves are measured in `src/sam_measure.py`; routing lives in `src/routing.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import io
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

WGS84_CRS = "EPSG:4326"
TARGET_METRIC_CRS = "EPSG:32616"  # WGS 84 / UTM zone 16N: metres at Mexico Beach, FL

# Used when a road has no measured width to restore to (half of an 8 m two-lane street).
DEFAULT_NOMINAL_ROAD_HALF_WIDTH_M = 4.0

# Minimum clear width each vehicle class needs to get through.
ATV_MIN_WIDTH_M = 1.2
AMBULANCE_MIN_WIDTH_M = 2.2
FIRE_ENGINE_MIN_WIDTH_M = 3.5


@dataclass(frozen=True)
class PassabilityFlags:
    """Whether each vehicle class fits through a measured clear width."""

    atv: bool
    ambulance: bool
    fire_engine: bool

    def to_dict(self) -> dict[str, bool]:
        return {"atv": self.atv, "ambulance": self.ambulance, "fire_engine": self.fire_engine}


def compute_passability_flags(min_clear_width_m: float, has_water_obstruction: bool = False) -> PassabilityFlags:
    """A vehicle passes when the clear width is wider than its minimum; standing water stops all."""
    if has_water_obstruction:
        return PassabilityFlags(atv=False, ambulance=False, fire_engine=False)
    width = float(min_clear_width_m)
    return PassabilityFlags(
        atv=width > ATV_MIN_WIDTH_M,
        ambulance=width > AMBULANCE_MIN_WIDTH_M,
        fire_engine=width > FIRE_ENGINE_MIN_WIDTH_M,
    )


def compute_raster_gsd_and_crs(raster_path: str | Path, target_crs: str = TARGET_METRIC_CRS) -> dict[str, Any]:
    """
    Describe a GeoTIFF: its native CRS, size, and ground sample distance in metres per pixel.
    A raster stored in degrees is measured as it would be after reprojection to `target_crs`.
    """
    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        native_crs = src.crs.to_string() if src.crs else WGS84_CRS
        b = src.bounds
        crs = CRS.from_user_input(native_crs)
        in_metres = crs.is_projected and crs.axis_info[0].unit_name in ("metre", "meter", "m")
        if in_metres:
            transform = src.transform
            metric_bounds = (b.left, b.bottom, b.right, b.top)
        else:
            transform, _, _ = calculate_default_transform(native_crs, target_crs, src.width, src.height, *b)
            to_metric = Transformer.from_crs(native_crs, target_crs, always_xy=True)
            x0, y0 = to_metric.transform(b.left, b.bottom)
            x1, y1 = to_metric.transform(b.right, b.top)
            metric_bounds = (x0, y0, x1, y1)
        gsd_x, gsd_y = abs(float(transform.a)), abs(float(transform.e))
        return {
            "raster_path": str(raster_path),
            "native_crs": native_crs,
            "target_crs": target_crs,
            "width_px": src.width,
            "height_px": src.height,
            "bands": src.count,
            "gsd_x_m": gsd_x,
            "gsd_y_m": gsd_y,
            "gsd_m_per_px": math.sqrt(gsd_x * gsd_y),
            "native_bounds": (b.left, b.bottom, b.right, b.top),
            "metric_bounds": metric_bounds,
            "metric_transform": transform,
        }


def _polygons(geom: Any) -> list[Polygon]:
    """Flatten a polygon, multipolygon, collection or list of them into plain polygons."""
    if geom is None:
        return []
    if isinstance(geom, (list, tuple)):
        return [p for g in geom for p in _polygons(g)]
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [p for g in geom.geoms for p in _polygons(g)]
    return []


def _union(polys: list[Polygon]) -> MultiPolygon:
    merged = unary_union(polys) if polys else MultiPolygon()
    return MultiPolygon(_polygons(merged))


def load_sam31_masks_utm(
    masks_geojson_path: str | Path = "data/sam31_mexico_beach_masks.geojson",
    target_crs: str = TARGET_METRIC_CRS,
) -> dict[str, Any]:
    """
    Load the baked SAM 3.1 mask polygons and reproject them to metres. Returns one merged
    MultiPolygon per class ("road_surface", "debris", "water") plus the individual features.
    A missing or empty file yields empty masks, so the replay simply draws no polygons.
    """
    by_class: dict[str, list[Polygon]] = {"road_surface": [], "debris": [], "water": []}
    features: list[dict[str, Any]] = []
    scores: list[float] = []

    path = Path(masks_geojson_path)
    gdf = gpd.read_file(path) if path.exists() else gpd.GeoDataFrame()
    if not gdf.empty:
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84_CRS)
        for _, row in gdf.to_crs(target_crs).iterrows():
            polys = _polygons(row.geometry)
            if not polys:
                continue
            label = str(row.get("class_label") or row.get("prompt") or row.get("feature_type") or "debris").lower()
            cls = "water" if "water" in label else "road_surface" if "road" in label else "debris"
            score = float(row.get("confidence_score", 0.78))
            scores.append(score)
            by_class[cls].extend(polys)
            source = str(row.get("mask_source", "sam-3.1"))
            features.extend(
                {"geometry": p, "class_label": cls, "confidence_score": score, "mask_source": source} for p in polys
            )

    return {
        **{cls: _union(polys) for cls, polys in by_class.items()},
        "features_utm": features,
        "mean_confidence": sum(scores) / len(scores) if scores else 0.85,
        "mask_source": "sam-3.1",
    }


class SAM31GeometricInstrument:
    """
    SAM 3.1 (`transformers.Sam3Model`) loaded from a local checkpoint in float16 on the Apple
    Silicon GPU (MPS), or the CPU when MPS is unavailable. Used once, by the bake script.
    """

    def __init__(self, checkpoint_dir: str | Path = "models/sam3.1", device: str | None = None) -> None:
        import torch

        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float16
        self.model: Any = None
        self.processor: Any = None

    def load_model(self) -> tuple[Any, Any]:
        """Load the processor and model once; later calls reuse them."""
        import torch
        from transformers import Sam3Model, Sam3Processor

        if self.model is None:
            self.processor = Sam3Processor.from_pretrained(str(self.checkpoint_dir))
            self.model = Sam3Model.from_pretrained(str(self.checkpoint_dir), dtype=torch.float16)
            self.model = self.model.to(self.device, dtype=torch.float16).eval()
        return self.model, self.processor

    def segment_tile_probabilities(
        self,
        image: Any,
        prompts: dict[str, str],
        *,
        instance_score_threshold: float = 0.4,
    ) -> dict[str, np.ndarray]:
        """
        Run SAM 3.1 once per tile (shared vision features) and return a float32 probability
        map `(H, W)` per prompt key: the max of the semantic map and every instance mask
        whose query score exceeds `instance_score_threshold`, gated by SAM 3's presence
        score for that concept in the tile, upsampled to the tile size. Without the presence
        gate the dense semantic map can saturate across tiles that contain no such object.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        pil_img = (
            image.convert("RGB")
            if isinstance(image, Image.Image)
            else Image.fromarray(image).convert("RGB")
        )
        width_px, height_px = pil_img.size
        model, processor = self.load_model()
        pixel_values = processor(images=pil_img, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=torch.float16
        )

        maps: dict[str, np.ndarray] = {}
        with torch.no_grad():
            vision_embeds = model.get_vision_features(pixel_values)
            for key, prompt in prompts.items():
                txt = processor(text=prompt, return_tensors="pt")
                out = model(
                    vision_embeds=vision_embeds,
                    input_ids=txt["input_ids"].to(self.device),
                    attention_mask=txt["attention_mask"].to(self.device),
                )
                prob = out.semantic_seg.float().sigmoid()  # [1, 1, h, w]
                scores = out.pred_logits.float().sigmoid()[0]
                keep = scores > instance_score_threshold
                if bool(keep.any()):
                    inst = out.pred_masks[:, keep].float().sigmoid().max(dim=1, keepdim=True).values
                    prob = torch.maximum(prob, inst)
                presence_logits = getattr(out, "presence_logits", None)
                if presence_logits is not None:
                    prob = prob * presence_logits.float().sigmoid().max()
                maps[key] = (
                    F.interpolate(
                        prob, size=(height_px, width_px), mode="bilinear", align_corners=False
                    )[0, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        return maps

    def unload_weights(self) -> None:
        """Drop the model and release GPU memory."""
        import torch

        self.model = None
        self.processor = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


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
