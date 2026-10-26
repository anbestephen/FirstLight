/**
 * FirstLight Offline Field Dashboard Application (`ui/app.js`).
 * Zero external CDN dependencies — operates 100% over local Wi-Fi when WAN is unplugged.
 */

(function () {
  "use strict";

  const state = {
    vehicleClass: "fire_engine",
    origin: null,
    destination: null,
    pickMode: null, // "origin" | "destination" | null
    clearedEdges: [],
    roadsGeojson: null,
    buildingsGeojson: null,
    boundsWgs84: [-85.4251, 29.9249, -85.3999, 29.9501],
    selectedSegment: null,
    map: null,
    rasterOverlay: null,
    buildingsLayer: null,
    roadsLayer: null,
    routeLayer: null,
    markersLayer: null,
  };

  const VEHICLE_LABELS = {
    atv: "ATV / Foot (>=1.2m)",
    ambulance: "Ambulance / Pickup (>=2.2m)",
    fire_engine: "Fire Engine (>=3.5m)",
  };

  function initMap() {
    const centerLat = 29.9415;
    const centerLon = -85.4125;

    const map = L.map("map", {
      center: [centerLat, centerLon],
      zoom: 15,
      zoomControl: true,
      attributionControl: false,
    });

    state.map = map;
    state.buildingsLayer = L.layerGroup().addTo(map);
    state.roadsLayer = L.layerGroup().addTo(map);
    state.routeLayer = L.layerGroup().addTo(map);
    state.markersLayer = L.layerGroup().addTo(map);

    map.on("click", function (ev) {
      if (!state.pickMode) {
        return;
      }
      const lat = ev.latlng.lat;
      const lon = ev.latlng.lng;
      if (state.pickMode === "origin") {
        state.origin = { lon: lon, lat: lat };
      } else if (state.pickMode === "destination") {
        state.destination = { lon: lon, lat: lat };
      }
      setPickMode(null);
      refreshRoute();
    });
  }

  function setPickMode(mode) {
    state.pickMode = mode;
    const btnOrigin = document.getElementById("btn-pick-origin");
    const btnDest = document.getElementById("btn-pick-dest");
    btnOrigin.classList.toggle("active-pick", mode === "origin");
    btnDest.classList.toggle("active-pick", mode === "destination");
  }

  function isEdgeCleared(u, v, key) {
    const uStr = String(u);
    const vStr = String(v);
    const kNum = Number(key || 0);
    return state.clearedEdges.some(function (item) {
      const iu = String(item.u);
      const iv = String(item.v);
      const ik = Number(item.key || 0);
      return ik === kNum && ((iu === uStr && iv === vStr) || (iu === vStr && iv === uStr));
    });
  }

  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      const netPill = document.getElementById("network-status-text");
      netPill.textContent = data.offline_ready ? "LOCAL WI-FI / OFFLINE READY" : "ONLINE";

      const tierBadge = document.getElementById("vlm-tier-badge");
      const lastTier = (data.vlm_tiers && data.vlm_tiers.last_tier_used) || "";
      if (lastTier.includes("tier_1")) {
        tierBadge.textContent = "TIER 1: CLOUD META API";
      } else if (lastTier.includes("tier_2")) {
        tierBadge.textContent = "TIER 2: EDGE MLX-VLM (4-BIT)";
      } else {
        tierBadge.textContent = "TIER 3: DETERMINISTIC GEOM";
      }

      const gsd = data.geospatial ? data.geospatial.gsd_m_per_px : 0.2804;
      const hwBadge = document.getElementById("hw-status-badge");
      hwBadge.textContent = `MPS/MLX fp16 • ${gsd} m/px`;
    } catch (err) {
      const netPill = document.getElementById("network-status-text");
      netPill.textContent = "LOCAL CACHE ACTIVE";
    }
  }

  async function loadInitialLayers() {
    const res = await fetch("/api/layers");
    const data = await res.json();

    state.roadsGeojson = data.roads;
    state.buildingsGeojson = data.buildings;
    state.boundsWgs84 = data.bounds_wgs84 || state.boundsWgs84;
    state.clearedEdges = data.cleared_edges || [];
    updateClearedCounter();

    if (!state.origin && data.default_origin) {
      state.origin = { lon: data.default_origin.lon, lat: data.default_origin.lat };
    }
    if (!state.destination && data.default_destination) {
      state.destination = { lon: data.default_destination.lon, lat: data.default_destination.lat };
    }

    const [minLon, minLat, maxLon, maxLat] = state.boundsWgs84;
    const imageBounds = [
      [minLat, minLon],
      [maxLat, maxLon],
    ];

    if (state.rasterOverlay) {
      state.map.removeLayer(state.rasterOverlay);
    }
    state.rasterOverlay = L.imageOverlay("/api/raster-preview.png", imageBounds, {
      opacity: 0.55,
      interactive: false,
    }).addTo(state.map);

    renderBuildingsLayer();
    renderRoadsLayer();
  }

  function renderBuildingsLayer() {
    state.buildingsLayer.clearLayers();
    const toggle = document.getElementById("toggle-buildings");
    if (!toggle.checked || !state.buildingsGeojson) {
      return;
    }

    const sevColors = {
      destroyed: "#ef4444",
      major_damage: "#f97316",
      minor_damage: "#eab308",
      no_damage: "#22c55e",
    };

    L.geoJSON(state.buildingsGeojson, {
      style: function (feature) {
        const sev = (feature.properties && feature.properties.damage_severity) || "no_damage";
        return {
          color: sevColors[sev] || "#64748b",
          fillColor: sevColors[sev] || "#64748b",
          weight: 1,
          fillOpacity: 0.35,
        };
      },
      onEachFeature: function (feature, layer) {
        const p = feature.properties || {};
        layer.bindPopup(
          `<strong>Building ${p.building_id || ""}</strong><br/>` +
          `Damage Severity: <code>${p.damage_severity}</code><br/>` +
          `High Severity Target: <strong>${p.is_high_severity ? "YES" : "NO"}</strong>`
        );
      },
    }).addTo(state.buildingsLayer);
  }

  function renderRoadsLayer() {
    state.roadsLayer.clearLayers();
    if (!state.roadsGeojson) {
      return;
    }

    const passKey = `passable_${state.vehicleClass}`;
    L.geoJSON(state.roadsGeojson, {
      style: function (feature) {
        const p = feature.properties || {};
        if (p.has_water_obstruction || Number(p.water_overlap_ratio) > 0) {
          return { color: "#a855f7", weight: 4, opacity: 0.9, dashArray: "6, 6" };
        }
        if (p[passKey]) {
          return { color: "#10b981", weight: 3, opacity: 0.75 };
        }
        return { color: "#f59e0b", weight: 4, opacity: 0.9 };
      },
      onEachFeature: function (feature, layer) {
        layer.on("click", function () {
          const p = feature.properties || {};
          state.selectedSegment = p;
          runVlmAssessmentForSegment(p);
        });
        const p = feature.properties || {};
        layer.bindTooltip(
          `<strong>${p.name}</strong><br/>` +
          `Min Clear Width: ${p.min_clear_width_m}m (GSD: ${p.gsd_m_per_px} m/px)<br/>` +
          `Debris: ${p.debris_area_sqm} m² | Water Ratio: ${p.water_overlap_ratio}`,
          { sticky: true }
        );
      },
    }).addTo(state.roadsLayer);
  }

  async function refreshRoute() {
    if (!state.origin || !state.destination) {
      return;
    }
    const payload = {
      origin: [state.origin.lon, state.origin.lat],
      destination: [state.destination.lon, state.destination.lat],
      vehicle_class: state.vehicleClass,
      algorithm: "dijkstra",
    };

    const res = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const routeData = await res.json();
    renderRouteOnMap(routeData);
    renderRouteBanner(routeData);
    renderTurnByTurnSteps(routeData.edge_steps || []);
  }

  function renderRouteOnMap(routeData) {
    state.routeLayer.clearLayers();
    state.markersLayer.clearLayers();

    if (state.origin) {
      L.circleMarker([state.origin.lat, state.origin.lon], {
        radius: 7,
        color: "#ffffff",
        fillColor: "#10b981",
        fillOpacity: 1,
        weight: 2,
      })
        .bindTooltip("Staging Origin", { permanent: false })
        .addTo(state.markersLayer);
    }

    if (state.destination) {
      L.circleMarker([state.destination.lat, state.destination.lon], {
        radius: 7,
        color: "#ffffff",
        fillColor: "#ef4444",
        fillOpacity: 1,
        weight: 2,
      })
        .bindTooltip("Target Destination", { permanent: false })
        .addTo(state.markersLayer);
    }

    if (routeData.reachable && routeData.geojson_geometry && routeData.geojson_geometry.coordinates.length > 0) {
      const latLngs = routeData.geojson_geometry.coordinates.map(function (c) {
        return [c[1], c[0]];
      });
      L.polyline(latLngs, {
        color: "#38bdf8",
        weight: 6,
        opacity: 0.95,
      }).addTo(state.routeLayer);
    }
  }

  function renderRouteBanner(routeData) {
    document.getElementById("route-class-label").textContent =
      VEHICLE_LABELS[state.vehicleClass] || state.vehicleClass;

    const statusEl = document.getElementById("route-status-val");
    if (routeData.reachable) {
      statusEl.textContent = "PASSABLE";
      statusEl.style.color = "#10b981";
    } else {
      statusEl.textContent = "BLOCKED";
      statusEl.style.color = "#ef4444";
    }

    document.getElementById("route-distance-val").textContent = `${routeData.total_distance_m.toFixed(1)} m`;
    document.getElementById("route-eta-val").textContent = `${routeData.estimated_transit_time_s.toFixed(1)} s`;
    document.getElementById("route-detours-val").textContent = String(
      (routeData.blocked_edges_avoided || []).length
    );
    document.getElementById("route-rationale-text").textContent = routeData.rationale || "";

    const summaryEl = document.getElementById("multi-class-comparison");
    const allSummary = routeData.all_classes_summary || {};
    const parts = [];
    ["atv", "ambulance", "fire_engine"].forEach(function (vc) {
      if (allSummary[vc]) {
        const info = allSummary[vc];
        const tag = info.reachable ? `${info.total_distance_m}m (${info.estimated_transit_time_s}s)` : "BLOCKED";
        parts.push(`<span><strong>${vc.toUpperCase()}:</strong> ${tag}</span>`);
      }
    });
    summaryEl.innerHTML = parts.join(" &bull; ");
  }

  function renderTurnByTurnSteps(steps) {
    document.getElementById("steps-count").textContent = String(steps.length);
    const listEl = document.getElementById("route-steps-list");
    if (!steps.length) {
      listEl.innerHTML = `<div class="roi-explanation">No passable route segments for selected vehicle class.</div>`;
      return;
    }

    listEl.innerHTML = steps
      .map(function (st, idx) {
        return `
          <div class="step-item">
            <div><strong>${idx + 1}. ${st.name}</strong> (${st.length_m}m)</div>
            <div class="step-meta">
              Min Clear Span: <strong>${st.min_clear_width_m}m</strong> &bull;
              Debris: ${st.debris_area_sqm} m² &bull;
              ETA: ${st.travel_time_s}s
            </div>
          </div>
        `;
      })
      .join("");
  }

  async function refreshClearanceRoi() {
    const res = await fetch(`/api/clearance-roi?vehicle_class=${encodeURIComponent(state.vehicleClass)}`);
    const data = await res.json();
    renderRoiQueue(data.recommendations || []);
  }

  function renderRoiQueue(recommendations) {
    const listEl = document.getElementById("roi-queue-list");
    if (!recommendations.length) {
      listEl.innerHTML = `<div class="roi-explanation">All primary corridors passable for ${VEHICLE_LABELS[state.vehicleClass]}.</div>`;
      return;
    }

    listEl.innerHTML = recommendations
      .map(function (rec) {
        const cleared = isEdgeCleared(rec.edge_u, rec.edge_v, rec.edge_key);
        return `
          <div class="roi-card ${cleared ? "cleared" : ""}">
            <div class="roi-card-header">
              <span class="roi-rank-badge">RANK #${rec.rank}</span>
              <span class="roi-score">ROI: ${rec.clearance_roi.toFixed(3)} bldg/m²</span>
            </div>
            <div class="roi-road-title">${rec.road_name}</div>
            <div class="roi-metrics-row">
              <span>Span: ${rec.current_clear_width_m}m &rarr; ${rec.restored_clear_width_m}m</span>
              <span>Debris: ${rec.debris_area_sqm} m²</span>
              <span>Unlocks: ${rec.high_severity_structures_unlocked} High-Sev Bldg</span>
            </div>
            <div class="roi-explanation">${rec.explanation}</div>
            <div class="roi-card-actions">
              <button
                type="button"
                class="btn-sim-clear ${cleared ? "is-cleared" : ""}"
                data-u="${rec.edge_u}"
                data-v="${rec.edge_v}"
                data-key="${rec.edge_key}"
              >
                ${cleared ? "✓ Clearance Simulated (Tap to Undo)" : "Simulate Clearance"}
              </button>
              <button
                type="button"
                class="action-btn btn-inspect-roi"
                data-name="${rec.road_name}"
                data-width="${rec.current_clear_width_m}"
                data-debris="${rec.debris_area_sqm}"
                data-gsd="${rec.gsd_m_per_px}"
              >
                VLM Triage
              </button>
            </div>
          </div>
        `;
      })
      .join("");

    listEl.querySelectorAll(".btn-sim-clear").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const u = btn.getAttribute("data-u");
        const v = btn.getAttribute("data-v");
        const key = Number(btn.getAttribute("data-key") || 0);
        toggleSimulateClearance(u, v, key);
      });
    });

    listEl.querySelectorAll(".btn-inspect-roi").forEach(function (btn) {
      btn.addEventListener("click", function () {
        switchTab("assess");
        runVlmAssessmentForSegment({
          name: btn.getAttribute("data-name"),
          min_clear_width_m: Number(btn.getAttribute("data-width")),
          debris_area_sqm: Number(btn.getAttribute("data-debris")),
          gsd_m_per_px: Number(btn.getAttribute("data-gsd")),
          water_overlap_ratio: 0.0,
          has_water_obstruction: false,
        });
      });
    });
  }

  async function toggleSimulateClearance(u, v, key) {
    const res = await fetch("/api/simulate-clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        edge_u: u,
        edge_v: v,
        edge_key: key,
        vehicle_class: state.vehicleClass,
      }),
    });
    const data = await res.json();
    state.clearedEdges = data.cleared_edges || [];
    state.roadsGeojson = data.roads || state.roadsGeojson;
    updateClearedCounter();
    renderRoadsLayer();
    renderRoiQueue(data.recommendations || []);
    await refreshRoute();
  }

  async function resetSimulatedClearances() {
    const res = await fetch("/api/simulate-clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reset: true,
        vehicle_class: state.vehicleClass,
      }),
    });
    const data = await res.json();
    state.clearedEdges = data.cleared_edges || [];
    state.roadsGeojson = data.roads || state.roadsGeojson;
    updateClearedCounter();
    renderRoadsLayer();
    renderRoiQueue(data.recommendations || []);
    await refreshRoute();
  }

  function updateClearedCounter() {
    document.getElementById("cleared-count").textContent = String(state.clearedEdges.length);
  }

  async function runVlmAssessmentForSegment(seg) {
    const modeSelect = document.getElementById("select-failover-mode");
    const boxEl = document.getElementById("assess-result-box");
    boxEl.innerHTML = "Running 3-Tier VLM Damage Grading &amp; Circuit Breaker...";

    const payload = {
      target_id: seg.name || "hwy98-segment",
      road_name: seg.name || "US Highway 98",
      min_clear_width_m: Number(seg.min_clear_width_m ?? 2.65),
      debris_area_m2: Number(seg.debris_area_sqm ?? 24.0),
      water_area_m2: Number(seg.water_overlap_ratio || 0) * 80.0,
      has_water_obstruction: Boolean(seg.has_water_obstruction || Number(seg.water_overlap_ratio) > 0),
      building_debris_overlap_ratio: Number(seg.debris_area_sqm || 0) > 20 ? 0.38 : 0.12,
      gsd_m_per_px: Number(seg.gsd_m_per_px || 0.2804),
      mode: modeSelect ? modeSelect.value : "auto",
    };

    const res = await fetch("/api/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    const tierBadge = document.getElementById("vlm-tier-badge");
    if (data.tier_used && data.tier_used.includes("tier_1")) {
      tierBadge.textContent = "TIER 1: CLOUD META API";
    } else if (data.tier_used && data.tier_used.includes("tier_2")) {
      tierBadge.textContent = "TIER 2: EDGE MLX-VLM (4-BIT)";
    } else {
      tierBadge.textContent = "TIER 3: DETERMINISTIC GEOM";
    }

    boxEl.innerHTML = `
      <div><strong>Target:</strong> ${data.target_id}</div>
      <div><strong>Active Tier Used:</strong> <code>${data.tier_used}</code> (${data.latency_ms} ms)</div>
      <div><strong>Fallback Chain:</strong> <code>${(data.fallback_chain || []).join(" &rarr; ")}</code></div>
      <div><strong>Building Severity:</strong> <code>${data.building_damage_severity}</code> (High-Sev: ${data.is_high_severity ? "YES" : "NO"})</div>
      <div><strong>Road Severity:</strong> <code>${data.road_damage_severity}</code></div>
      <div><strong>Unified Memory Hygiene:</strong> <code>memory_cleared=${data.memory_cleared}</code></div>
      <div class="roi-explanation">${data.passability_rationale}</div>
    `;
  }

  function switchTab(tabId) {
    document.querySelectorAll(".panel-tab").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-content").forEach(function (pane) {
      pane.classList.toggle("active", pane.id === `tab-${tabId}`);
    });
  }

  function bindUiEvents() {
    document.querySelectorAll(".vehicle-btn").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        document.querySelectorAll(".vehicle-btn").forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        state.vehicleClass = btn.getAttribute("data-vehicle");
        renderRoadsLayer();
        await Promise.all([refreshRoute(), refreshClearanceRoi()]);
      });
    });

    document.querySelectorAll(".panel-tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        switchTab(btn.getAttribute("data-tab"));
      });
    });

    document.getElementById("toggle-raster").addEventListener("change", function (ev) {
      if (!state.rasterOverlay) return;
      if (ev.target.checked) {
        state.rasterOverlay.addTo(state.map);
      } else {
        state.map.removeLayer(state.rasterOverlay);
      }
    });

    document.getElementById("toggle-buildings").addEventListener("change", function () {
      renderBuildingsLayer();
    });

    document.getElementById("btn-pick-origin").addEventListener("click", function () {
      setPickMode(state.pickMode === "origin" ? null : "origin");
    });

    document.getElementById("btn-pick-dest").addEventListener("click", function () {
      setPickMode(state.pickMode === "destination" ? null : "destination");
    });

    document.getElementById("btn-reset-sim").addEventListener("click", function () {
      resetSimulatedClearances();
    });

    document.getElementById("btn-run-assess").addEventListener("click", function () {
      runVlmAssessmentForSegment(
        state.selectedSegment || {
          name: "US Highway 98 E Bottleneck",
          min_clear_width_m: 2.65,
          debris_area_sqm: 28.4,
          gsd_m_per_px: 0.2804,
          water_overlap_ratio: 0.0,
        }
      );
    });
  }

  document.addEventListener("DOMContentLoaded", async function () {
    initMap();
    bindUiEvents();
    await fetchHealth();
    await loadInitialLayers();
    await Promise.all([refreshRoute(), refreshClearanceRoi()]);
  });
})();
