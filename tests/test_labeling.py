"""
Tests for the blind ground-truth labelling tool (`src/labeling.py`, `/label`).
Labels are written to a temporary directory so real ground truth is never touched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import src.labeling as labeling
from src.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(labeling, "ROAD_LABELS_PATH", tmp_path / "road_widths_manual.json")
    monkeypatch.setattr(labeling, "BUILDING_LABELS_PATH", tmp_path / "building_damage_manual.json")


def test_pixel_utm_round_trip() -> None:
    center = [612345.0, 3312345.0]
    px = labeling.utm_to_pixel(center[0] + 3.0, center[1] - 7.5, center, 40.0)
    assert px == pytest.approx((160 + 12, 160 + 30))
    assert labeling.pixel_to_utm(*px, center, 40.0) == pytest.approx((center[0] + 3.0, center[1] - 7.5))


def test_label_page_and_samples_never_expose_model_outputs() -> None:
    assert client.get("/label").status_code == 200
    roads = client.get("/api/label/roads").json()
    assert len(roads["samples"]) == 40
    # Only geometry reaches the labeller - no widths, causes, confidences or grades.
    assert all(set(s) == {"sample_id", "ray_px", "crop_res_m"} for s in roads["samples"])
    bldgs = client.get("/api/label/buildings").json()
    assert len(bldgs["samples"]) == 40
    assert all(set(s) == {"sample_id", "outline_px"} for s in bldgs["samples"])


def test_road_width_is_measured_along_the_cross_section() -> None:
    sample = client.get("/api/label/roads").json()["samples"][0]
    (ax, ay), (bx, by) = sample["ray_px"]
    length_px = math.hypot(bx - ax, by - ay)
    # Two clicks 24 px apart along the ray (0.25 m/px) -> 6.0 m.
    ux, uy = (bx - ax) / length_px, (by - ay) / length_px
    p1 = [ax + 40 * ux, ay + 40 * uy]
    p2 = [ax + 64 * ux, ay + 64 * uy]
    rec = client.post(f"/api/label/roads/{sample['sample_id']}", json={"p1": p1, "p2": p2}).json()
    assert rec["status"] == "measured"
    assert rec["width_m"] == pytest.approx(6.0, abs=0.01)

    stored = json.loads(labeling.ROAD_LABELS_PATH.read_text())["labels"][sample["sample_id"]]
    assert stored["width_m"] == pytest.approx(6.0, abs=0.01)
    assert "labeled_at" in stored


def test_blocked_and_cant_tell_statuses() -> None:
    samples = client.get("/api/label/roads").json()["samples"]
    assert client.post(f"/api/label/roads/{samples[1]['sample_id']}", json={"status": "blocked"}).json()["width_m"] == 0.0
    cant = client.post(f"/api/label/roads/{samples[2]['sample_id']}", json={"status": "cant_tell"}).json()
    assert "width_m" not in cant
    assert client.post(f"/api/label/roads/{samples[3]['sample_id']}", json={"status": "measured"}).status_code == 400


def test_building_grade_and_crops() -> None:
    sample = client.get("/api/label/buildings").json()["samples"][0]
    rec = client.post(f"/api/label/buildings/{sample['sample_id']}", json={"grade": "destroyed"}).json()
    assert rec["grade"] == "destroyed"
    assert client.post(f"/api/label/buildings/{sample['sample_id']}", json={"grade": "awful"}).status_code == 422
    post = client.get(f"/api/label/buildings/{sample['sample_id']}/post.png")
    assert post.status_code == 200 and post.headers["content-type"] == "image/png"
