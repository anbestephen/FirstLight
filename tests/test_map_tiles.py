"""Tests for offline map tiles (`scripts/build_map_tiles.py`, `/tiles`)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from scripts.build_map_tiles import lonlat_to_tile, tile_bounds_mercator
from src.app import app

client = TestClient(app)


def test_tile_math_matches_web_mercator() -> None:
    assert lonlat_to_tile(0.0, 0.0, 1) == (1, 1)
    # Mexico Beach at z19 (cross-checked with the standard slippy-map formula).
    x, y = lonlat_to_tile(-85.4125, 29.9415, 19)
    left, bottom, right, top = tile_bounds_mercator(x, y, 19)
    assert right - left == pytest.approx(76.437, abs=0.01)  # metres per tile edge at z19
    assert top > bottom


def test_tiles_are_served_when_built() -> None:
    root = Path(__file__).resolve().parent.parent / "data" / "tiles" / "post"
    tile = next(root.rglob("*.jpg"), None) if root.exists() else None
    if tile is None:
        pytest.skip("tiles not built; run scripts/build_map_tiles.py")
    z, x, name = tile.parts[-3:]
    res = client.get(f"/tiles/post/{z}/{x}/{name}")
    assert res.status_code == 200
    assert res.content[:3] == b"\xff\xd8\xff"  # JPEG
    assert client.get("/tiles/post/1/0/0.jpg").status_code == 404
