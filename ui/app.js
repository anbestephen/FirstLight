/**
 * FirstLight Offline Field Dashboard Application (`ui/app.js`).
 * Zero external CDN dependencies — operates 100% over local Wi-Fi when WAN is unplugged.
 */

(function () {
  "use strict";

  const state = {
    vehicleClass: "fire_engine",
    forceOffline: false,
    showCaliperRays: true,
    origin: null,
    destination: null,
    pickMode: null, // "origin" | "destination" | null
    clearedEdges: [],
    roadsGeojson: null,
    buildingsGeojson: null,
    lastRouteData: null,
    lastRoiRecommendations: [],
    activeEdgeId: null,
    inspectorViewMode: "crop", // "crop" | "schematic"
    highlightedUnlockedNodes: [],
    lastCaliperGeojson: null,
    boundsWgs84: [-85.4251, 29.9249, -85.3999, 29.9501],
    selectedSegment: null,
    map: null,
    rasterOverlay: null,
    buildingsLayer: null,
    roadsLayer: null,
    routeLayer: null,
    caliperLayer: null,
    markersLayer: null,
  };

  const VEHICLE_LABELS = {
    atv: "ATV / Foot (>=1.2m)",
    ambulance: "Ambulance / Pickup (>=2.2m)",
    fire_engine: "Fire Engine (>=3.5m)",
  };

  function formatTierLabel(tierCode) {
    const t = String(tierCode || "").toLowerCase();
    if (t.includes("tier_1")) {
      return "Tier 1: Meta Cloud API";
    }
    if (t.includes("tier_2")) {
      return "Tier 2: Apple Silicon MLX 4-bit";
    }
    return "Tier 3: Deterministic";
  }

  function initMap() {
    const centerLat = 29.9415;
    const centerLon = -85.4125;

    const map = L.map("map", {
      center: [centerLat, centerLon],
      zoom: 15,
      minZoom: 14,
      maxZoom: 19,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      maxBoundsViscosity: 1.0,
      zoomControl: true,
      attributionControl: false,
    });

    state.map = map;
    state.buildingsLayer = L.layerGroup().addTo(map);
    state.roadsLayer = L.layerGroup().addTo(map);
    state.routeLayer = L.layerGroup().addTo(map);
    state.caliperLayer = L.layerGroup().addTo(map);
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

    window.addEventListener("resize", function () {
      if (!state.map) return;
      state.map.invalidateSize();
      applyAoiViewportBounds(false);
    });
  }

  function applyAoiViewportBounds(snapToFit) {
    if (!state.map || !state.boundsWgs84) return;
    const [minLon, minLat, maxLon, maxLat] = state.boundsWgs84;
    const aoiBounds = L.latLngBounds([
      [minLat, minLon],
      [maxLat, maxLon],
    ]);
    const computedFitZoom = state.map.getBoundsZoom(aoiBounds, false, [24, 24]);
    const minFitZoom = Math.max(14, computedFitZoom);
    state.map.setMinZoom(minFitZoom);
    state.map.setMaxBounds(aoiBounds.pad(0.15));
    if (snapToFit || state.map.getZoom() < minFitZoom) {
      state.map.fitBounds(aoiBounds, { padding: [24, 24] });
    }
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
      netPill.textContent = state.forceOffline
        ? "COMMS BLACKOUT (EDGE ONLY)"
        : data.offline_ready
        ? "LOCAL WI-FI / OFFLINE READY"
        : "ONLINE";

      const tierBadge = document.getElementById("vlm-tier-badge");
      const activeTier = (data.vlm_tiers && data.vlm_tiers.active_tier) || "";
      tierBadge.textContent = formatTierLabel(activeTier).toUpperCase();

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

    applyAoiViewportBounds(false);
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

    const unlockedSet = new Set((state.highlightedUnlockedNodes || []).map(String));

    L.geoJSON(state.buildingsGeojson, {
      style: function (feature) {
        const p = feature.properties || {};
        const sev = p.damage_severity || "no_damage";
        const isUnlocked = unlockedSet.has(String(p.nearest_node || ""));
        if (isUnlocked && p.is_high_severity) {
          return {
            color: "#06b6d4",
            fillColor: "#10b981",
            weight: 2.5,
            fillOpacity: 0.85,
          };
        }
        return {
          color: sevColors[sev] || "#64748b",
          fillColor: sevColors[sev] || "#64748b",
          weight: 1,
          fillOpacity: 0.35,
        };
      },
      onEachFeature: function (feature, layer) {
        const p = feature.properties || {};
        const isUnlocked = unlockedSet.has(String(p.nearest_node || ""));
        layer.bindPopup(
          `<strong>Building ${p.building_id || ""}</strong><br/>` +
          `Damage Severity: <code>${p.damage_severity}</code><br/>` +
          `High Severity Target: <strong>${p.is_high_severity ? "YES" : "NO"}</strong>` +
          (isUnlocked ? `<br/><strong style="color:#10b981;">UNLOCKED BY CHOKE-POINT CLEARANCE</strong>` : "")
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
          const edgeId = p.edge_id || `${p.u}-${p.v}-${p.key || 0}`;
          inspectCorridorByEdgeId(edgeId, true);
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

  function updateInspectorImage() {
    if (!state.activeEdgeId) return;
    const cropImg = document.getElementById("inspector-crop-img");
    const captionEl = document.getElementById("inspector-crop-caption");
    const encodedId = encodeURIComponent(state.activeEdgeId);
    if (state.inspectorViewMode === "schematic") {
      cropImg.src = `/api/choke-point/${encodedId}/recon-schematic.png?force_offline=${Boolean(state.forceOffline)}`;
      if (captionEl) {
        captionEl.textContent = "Muse Image + SAM 3.1 Recon Schematic";
      }
    } else {
      cropImg.src = `/api/choke-point/${encodedId}/crop.png`;
      if (captionEl) {
        captionEl.textContent = "256×256 NOAA GeoTIFF Crop";
      }
    }
  }

  function updateCaliperThresholdGauge(minClearWidthM, hasWater) {
    const w = hasWater ? 0.0 : Math.max(0.0, Number(minClearWidthM || 0));
    const maxScaleM = 6.0;
    const pct = Math.min(100, Math.max(0, (w / maxScaleM) * 100));

    const labelEl = document.getElementById("gauge-measured-span-label");
    if (labelEl) {
      labelEl.textContent = hasWater
        ? `Measured: 0.00m (WATER=INF)`
        : `Measured: ${w.toFixed(2)}m`;
    }

    const fillEl = document.getElementById("caliper-gauge-fill");
    if (fillEl) {
      fillEl.style.width = `${pct.toFixed(1)}%`;
      fillEl.classList.remove("gauge-warning", "gauge-danger");
      if (hasWater || w < 1.2) {
        fillEl.classList.add("gauge-danger");
      } else if (w < 3.5) {
        fillEl.classList.add("gauge-warning");
      }
    }

    const checks = [
      { id: "gauge-badge-atv", label: "ATV (\u22651.2m)", pass: !hasWater && w >= 1.2 },
      { id: "gauge-badge-ambulance", label: "Ambulance (\u22652.2m)", pass: !hasWater && w >= 2.2 },
      { id: "gauge-badge-fire_engine", label: "Fire Engine (\u22653.5m)", pass: !hasWater && w >= 3.5 },
    ];
    checks.forEach(function (item) {
      const el = document.getElementById(item.id);
      if (!el) return;
      el.textContent = `${item.label}: ${item.pass ? "PASS" : "BLOCKED"}`;
      el.classList.toggle("pass", item.pass);
      el.classList.toggle("blocked", !item.pass);
    });
  }

  function updateUnlockedStructuresHighlight(edgeId) {
    const rec = (state.lastRoiRecommendations || []).find(function (r) {
      const rId = r.edge_id || `${r.edge_u}-${r.edge_v}-${r.edge_key}`;
      return String(rId) === String(edgeId);
    });
    const bannerEl = document.getElementById("unlocked-impact-banner");
    const textEl = document.getElementById("unlocked-impact-text");

    if (rec && rec.high_severity_structures_unlocked > 0) {
      state.highlightedUnlockedNodes = rec.unlocked_node_ids || [rec.edge_u, rec.edge_v];
      if (bannerEl && textEl) {
        textEl.textContent =
          `+${rec.high_severity_structures_unlocked} High-Severity Structures Unlocked | ` +
          `Clearance ROI: ${Number(rec.clearance_roi).toFixed(2)} structures/m² (${rec.road_name})`;
        bannerEl.classList.remove("hidden");
      }
    } else {
      state.highlightedUnlockedNodes = [];
      if (bannerEl) {
        bannerEl.classList.add("hidden");
      }
    }
    renderBuildingsLayer();
  }

  /**
   * Fetch 5m orthogonal caliper rays & SAM 3.1 masks for `edgeId`, render on map,
   * load `/api/choke-point/{edgeId}/crop.png` or `/recon-schematic.png`, and run live 3-tier VLM assessment.
   */
  async function inspectCorridorByEdgeId(edgeId, zoomToCorridor) {
    if (!edgeId) return;
    state.activeEdgeId = String(edgeId);

    // Highlight selected card in ROI list if present
    document.querySelectorAll(".roi-card").forEach(function (card) {
      card.classList.toggle("selected-corridor", card.getAttribute("data-edge-id") === state.activeEdgeId);
    });

    updateUnlockedStructuresHighlight(state.activeEdgeId);
    updateInspectorImage();

    // Fetch caliper GeoJSON and VLM assessment concurrently
    const [caliperRes] = await Promise.all([
      fetch(`/api/choke-point/${encodeURIComponent(state.activeEdgeId)}/caliper`),
      runVlmAssessmentForEdge(state.activeEdgeId),
    ]);

    if (caliperRes.ok) {
      const caliperGeojson = await caliperRes.json();
      state.lastCaliperGeojson = caliperGeojson;
      const cProps = caliperGeojson.properties || {};
      if (cProps.min_clear_width_m !== undefined) {
        updateCaliperThresholdGauge(cProps.min_clear_width_m, Boolean(cProps.has_water_obstruction));
      }
      renderCaliperLayer(caliperGeojson, Boolean(zoomToCorridor));
    }
    await refreshMetaModelsTelemetry();
  }

  function renderCaliperLayer(caliperGeojson, zoomToCorridor) {
    state.caliperLayer.clearLayers();
    if (!caliperGeojson || !state.showCaliperRays) {
      return;
    }

    const props = caliperGeojson.properties || {};
    const caliperReadingEl = document.getElementById("inspector-caliper-reading");
    if (caliperReadingEl && props.min_clear_width_m !== undefined) {
      caliperReadingEl.innerHTML =
        `Pinch Span: <strong>${Number(props.min_clear_width_m).toFixed(2)}m</strong> ` +
        `at station ${Number(props.pinch_point_station_m || 0).toFixed(1)}m ` +
        `(${props.ray_count || 0} rays @ ${props.gsd_m_per_px} m/px)`;
    }

    const geojsonLayer = L.geoJSON(caliperGeojson, {
      style: function (feature) {
        const fp = feature.properties || {};
        if (fp.feature_type === "road_corridor") {
          return {
            color: "#38bdf8",
            fillColor: "#0284c7",
            weight: 1,
            fillOpacity: 0.14,
            dashArray: "3, 3",
          };
        }
        if (fp.feature_type === "debris_mask") {
          return {
            color: "#ea580c",
            fillColor: "#f97316",
            weight: 2,
            fillOpacity: 0.55,
          };
        }
        if (fp.feature_type === "water_mask") {
          return {
            color: "#9333ea",
            fillColor: "#a855f7",
            weight: 2,
            fillOpacity: 0.6,
          };
        }
        if (fp.feature_type === "caliper_ray") {
          const w = Number(fp.clear_width_m || 0);
          if (fp.is_pinch_point || fp.water_intersected || w < 1.2) {
            return { color: "#ef4444", weight: 5, opacity: 1.0 };
          }
          if (w >= 3.5) {
            return { color: "#10b981", weight: 3, opacity: 0.9 };
          }
          return { color: "#f59e0b", weight: 3.5, opacity: 0.95 };
        }
        return { color: "#94a3b8", weight: 2 };
      },
      onEachFeature: function (feature, layer) {
        const fp = feature.properties || {};
        if (fp.feature_type === "caliper_ray") {
          const badge = fp.is_pinch_point ? " [MIN PINCH POINT]" : "";
          layer.bindTooltip(
            `<strong>5m Caliper Ray @ ${fp.station_m}m${badge}</strong><br/>` +
            `Clear Width: <strong>${fp.clear_width_m}m</strong><br/>` +
            `ATV: ${fp.atv ? "PASS" : "BLOCKED"} | Amb: ${fp.ambulance ? "PASS" : "BLOCKED"} | Fire: ${fp.fire_engine ? "PASS" : "BLOCKED"}`,
            { sticky: true }
          );
        } else if (fp.feature_type === "debris_mask") {
          layer.bindTooltip(
            `<strong>SAM 3.1 Debris Mask</strong><br/>Area: ${fp.area_m2} m² (GSD: ${fp.gsd_m_per_px} m/px)`,
            { sticky: true }
          );
        } else if (fp.feature_type === "water_mask") {
          layer.bindTooltip(
            `<strong>SAM 3.1 Standing Water Mask</strong><br/>Area: ${fp.area_m2} m² (Weight = &infin;)`,
            { sticky: true }
          );
        }
      },
    }).addTo(state.caliperLayer);

    if (zoomToCorridor && geojsonLayer.getBounds && geojsonLayer.getBounds().isValid()) {
      state.map.fitBounds(geojsonLayer.getBounds().pad(0.45), { maxZoom: 18 });
    }
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
    state.lastRouteData = routeData;
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
    state.lastRoiRecommendations = data.recommendations || [];
    renderRoiQueue(state.lastRoiRecommendations);

    // Automatically inspect Rank #1 choke point on initial load so caliper rays & crop are visible immediately
    if (!state.activeEdgeId && state.lastRoiRecommendations.length > 0) {
      const top = state.lastRoiRecommendations[0];
      const topEdgeId = top.edge_id || `${top.edge_u}-${top.edge_v}-${top.edge_key}`;
      inspectCorridorByEdgeId(topEdgeId, false);
    }
  }

  function renderRoiQueue(recommendations) {
    const listEl = document.getElementById("roi-queue-list");
    if (!recommendations.length) {
      listEl.innerHTML = `<div class="roi-explanation">All primary corridors passable for ${VEHICLE_LABELS[state.vehicleClass]}.</div>`;
      return;
    }

    listEl.innerHTML = recommendations
      .map(function (rec) {
        const edgeId = rec.edge_id || `${rec.edge_u}-${rec.edge_v}-${rec.edge_key}`;
        const cleared = isEdgeCleared(rec.edge_u, rec.edge_v, rec.edge_key);
        const isSelected = state.activeEdgeId === String(edgeId);
        return `
          <div class="roi-card ${cleared ? "cleared" : ""} ${isSelected ? "selected-corridor" : ""}" data-edge-id="${edgeId}">
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
                ${cleared ? "✓ Clearance Simulated (Undo)" : "Simulate Clearance"}
              </button>
              <button
                type="button"
                class="action-btn btn-inspect-roi"
                data-edge-id="${edgeId}"
              >
                5m Caliper &amp; VLM
              </button>
            </div>
          </div>
        `;
      })
      .join("");

    listEl.querySelectorAll(".roi-card").forEach(function (card) {
      card.addEventListener("click", function (ev) {
        if (ev.target && ev.target.classList.contains("btn-sim-clear")) {
          return;
        }
        const edgeId = card.getAttribute("data-edge-id");
        inspectCorridorByEdgeId(edgeId, true);
      });
    });

    listEl.querySelectorAll(".btn-sim-clear").forEach(function (btn) {
      btn.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const u = btn.getAttribute("data-u");
        const v = btn.getAttribute("data-v");
        const key = Number(btn.getAttribute("data-key") || 0);
        toggleSimulateClearance(u, v, key);
      });
    });

    listEl.querySelectorAll(".btn-inspect-roi").forEach(function (btn) {
      btn.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const edgeId = btn.getAttribute("data-edge-id");
        inspectCorridorByEdgeId(edgeId, true);
      });
    });
  }

  async function toggleSimulateClearance(u, v, key) {
    const clickedEdgeId = `${u}-${v}-${Number(key || 0)}`;
    const priorRec = (state.lastRoiRecommendations || []).find(function (r) {
      const rId = r.edge_id || `${r.edge_u}-${r.edge_v}-${r.edge_key}`;
      return String(rId) === clickedEdgeId;
    });

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
    state.lastRoiRecommendations = data.recommendations || [];
    updateClearedCounter();
    renderRoadsLayer();
    renderRoiQueue(state.lastRoiRecommendations);
    await refreshRoute();

    if (priorRec && priorRec.high_severity_structures_unlocked > 0) {
      state.highlightedUnlockedNodes = priorRec.unlocked_node_ids || [u, v];
      const bannerEl = document.getElementById("unlocked-impact-banner");
      const textEl = document.getElementById("unlocked-impact-text");
      if (bannerEl && textEl) {
        textEl.textContent =
          `+${priorRec.high_severity_structures_unlocked} High-Severity Structures Unlocked | ` +
          `Clearance ROI: ${Number(priorRec.clearance_roi).toFixed(2)} structures/m² (${priorRec.road_name})`;
        bannerEl.classList.remove("hidden");
      }
      renderBuildingsLayer();
    } else if (state.activeEdgeId) {
      await inspectCorridorByEdgeId(state.activeEdgeId, false);
    }
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
    state.lastRoiRecommendations = data.recommendations || [];
    updateClearedCounter();
    renderRoadsLayer();
    renderRoiQueue(state.lastRoiRecommendations);
    await refreshRoute();
    if (state.activeEdgeId) {
      await inspectCorridorByEdgeId(state.activeEdgeId, false);
    }
  }

  function updateClearedCounter() {
    document.getElementById("cleared-count").textContent = String(state.clearedEdges.length);
  }

  async function runVlmAssessmentForEdge(edgeId) {
    const modeSelect = document.getElementById("select-failover-mode");
    const boxEl = document.getElementById("assess-result-box");
    boxEl.innerHTML = "Running 3-Tier VLM Damage Grading &amp; Circuit Breaker...";

    const payload = {
      edge_id: String(edgeId),
      force_offline: Boolean(state.forceOffline),
      mode: modeSelect ? modeSelect.value : "auto",
    };

    const res = await fetch("/api/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    const formattedTier = formatTierLabel(data.tier_used);
    const tierBadge = document.getElementById("vlm-tier-badge");
    tierBadge.textContent = formattedTier.toUpperCase();

    // Update Corridor AI Inspector dock
    document.getElementById("inspector-tier-pill").textContent = formattedTier;
    document.getElementById("inspector-road-name").textContent = data.target_id || String(edgeId);
    document.getElementById("inspector-latency-row").innerHTML =
      `Latency: <strong>${data.latency_ms} ms</strong> &bull; ` +
      `Memory Cleared: <strong>${data.memory_cleared}</strong>`;
    document.getElementById("inspector-severity-row").innerHTML =
      `Building: <code>${data.building_damage_severity}</code> &bull; ` +
      `Road: <code>${data.road_damage_severity}</code>`;
    document.getElementById("inspector-rationale-text").textContent =
      data.passability_rationale || "";

    boxEl.innerHTML = `
      <div><strong>Edge ID:</strong> <code>${data.edge_id || data.target_id}</code></div>
      <div><strong>Comms Blackout Mode:</strong> <code>force_offline=${data.force_offline}</code></div>
      <div><strong>Active Tier Used:</strong> <code>${formattedTier}</code> (${data.latency_ms} ms)</div>
      <div><strong>Fallback Chain:</strong> <code>${(data.fallback_chain || []).join(" &rarr; ")}</code></div>
      <div><strong>Building Severity:</strong> <code>${data.building_damage_severity}</code> (High-Sev: ${data.is_high_severity ? "YES" : "NO"})</div>
      <div><strong>Road Severity:</strong> <code>${data.road_damage_severity}</code></div>
      <div><strong>Unified Memory Hygiene:</strong> <code>memory_cleared=${data.memory_cleared}</code></div>
      <div class="roi-explanation">${data.passability_rationale}</div>
    `;
  }

  /**
   * Export the currently computed route, turn-by-turn clearance steps, and top ROI
   * choke points as a local `.geojson` file for offline responder handoff.
   */
  function exportFieldManifestGeojson() {
    const features = [];

    if (state.lastRouteData && state.lastRouteData.geojson_geometry) {
      features.push({
        type: "Feature",
        geometry: state.lastRouteData.geojson_geometry,
        properties: {
          manifest_type: "primary_vehicle_route",
          vehicle_class: state.vehicleClass,
          reachable: state.lastRouteData.reachable,
          total_distance_m: state.lastRouteData.total_distance_m,
          estimated_transit_time_s: state.lastRouteData.estimated_transit_time_s,
          rationale: state.lastRouteData.rationale,
        },
      });

      (state.lastRouteData.edge_steps || []).forEach(function (step, idx) {
        if (step.coordinates_lonlat && step.coordinates_lonlat.length >= 2) {
          features.push({
            type: "Feature",
            geometry: {
              type: "LineString",
              coordinates: step.coordinates_lonlat,
            },
            properties: {
              manifest_type: "turn_by_turn_step",
              step_number: idx + 1,
              name: step.name,
              length_m: step.length_m,
              min_clear_width_m: step.min_clear_width_m,
              debris_area_sqm: step.debris_area_sqm,
              travel_time_s: step.travel_time_s,
              instruction: step.instruction,
            },
          });
        }
      });
    }

    (state.lastRoiRecommendations || []).forEach(function (rec) {
      features.push({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: rec.bottleneck_lonlat || [-85.4125, 29.9415],
        },
        properties: {
          manifest_type: "choke_point_clearance_roi",
          rank: rec.rank,
          edge_id: rec.edge_id,
          road_name: rec.road_name,
          current_clear_width_m: rec.current_clear_width_m,
          restored_clear_width_m: rec.restored_clear_width_m,
          debris_area_sqm: rec.debris_area_sqm,
          high_severity_structures_unlocked: rec.high_severity_structures_unlocked,
          clearance_roi: rec.clearance_roi,
          explanation: rec.explanation,
        },
      });
    });

    const manifestGeojson = {
      type: "FeatureCollection",
      metadata: {
        system: "FirstLight Tactical Disaster Routing Engine",
        aoi: "Mexico Beach, FL (Hurricane Michael - US Hwy 98 Corridor)",
        vehicle_class: state.vehicleClass,
        force_offline_mode: state.forceOffline,
        crs: "EPSG:4326",
        metric_crs: "EPSG:32616",
        gsd_m_per_px: 0.2804,
        exported_at: new Date().toISOString(),
      },
      features: features,
    };

    const blob = new Blob([JSON.stringify(manifestGeojson, null, 2)], {
      type: "application/geo+json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `firstlight_field_manifest_${state.vehicleClass}.geojson`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function refreshMetaModelsTelemetry() {
    try {
      const res = await fetch(`/api/meta-models?force_offline=${Boolean(state.forceOffline)}`);
      if (!res.ok) return;
      const data = await res.json();
      (data.models || []).forEach(function (m) {
        const chip = document.querySelector(`.meta-model-chip[data-model="${m.model_id}"]`);
        if (!chip) return;
        chip.classList.toggle("edge-mode", Boolean(state.forceOffline));
        const subEl = chip.querySelector(".chip-sub");
        if (subEl) {
          subEl.textContent = `${m.call_count}x (${m.last_latency_ms}ms)`;
        }
        chip.title = `${m.role} | Provider: ${m.provider} | Calls: ${m.call_count}`;
      });
    } catch (err) {
      // Ignore offline telemetry fetch errors
    }
  }

  async function runWonderBriefing() {
    const boxEl = document.getElementById("wonder-briefing-box");
    if (!boxEl) return;
    boxEl.classList.remove("hidden");
    boxEl.innerHTML = "Synthesizing Muse Wonder multi-corridor strategic triage plan...";

    const res = await fetch("/api/incident-briefing", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vehicle_class: state.vehicleClass,
        force_offline: Boolean(state.forceOffline),
        max_corridors: 5,
      }),
    });
    const data = await res.json();
    const sectorsHtml = (data.sector_priorities || [])
      .slice(0, 3)
      .map(function (sp) {
        return `<div>&bull; <strong>#${sp.priority_rank} ${sp.road_name}:</strong> ${sp.equipment_assigned} (${sp.debris_area_sqm} m², ${sp.span_widening_m}, Unlocks ${sp.high_severity_structures_unlocked} bldg)</div>`;
      })
      .join("");

    boxEl.innerHTML = `
      <div><strong>Muse Wonder Action Plan (${formatTierLabel(data.tier_used)} &bull; ${data.latency_ms} ms):</strong></div>
      <div>${data.executive_summary}</div>
      ${sectorsHtml}
    `;
    await refreshMetaModelsTelemetry();
  }

  async function runVoiceDispatch() {
    const statusEl = document.getElementById("voice-dispatch-status");
    if (statusEl) {
      statusEl.textContent = "Transcribing Tactical Radio via Muse Voice...";
    }

    const res = await fetch("/api/voice-dispatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice_command_text:
          "Dispatch Fire Engine and simulate clearing top choke point on US Highway 98",
        force_offline: Boolean(state.forceOffline),
        auto_apply_clearance: true,
      }),
    });
    const data = await res.json();

    if (statusEl) {
      statusEl.textContent = `"${data.transcript}" (${data.latency_ms}ms)`;
      statusEl.title = data.action_summary || data.transcript;
    }

    if (data.vehicle_class) {
      state.vehicleClass = data.vehicle_class;
      document.querySelectorAll(".vehicle-btn").forEach(function (b) {
        b.classList.toggle("active", b.getAttribute("data-vehicle") === state.vehicleClass);
      });
    }

    await loadInitialLayers();
    await Promise.all([refreshRoute(), refreshClearanceRoi()]);
    if (data.inspect_edge_id) {
      await inspectCorridorByEdgeId(data.inspect_edge_id, true);
    }
    await refreshMetaModelsTelemetry();
  }

  async function runBenchmarkTiers(edgeId) {
    const targetEdge = String(edgeId || state.activeEdgeId || "0");
    const boxEl = document.getElementById("benchmark-tiers-box");
    if (!boxEl) return;
    boxEl.classList.remove("hidden");
    boxEl.innerHTML = "Running Side-by-Side 3-Tier Failover Benchmark...";

    const res = await fetch("/api/benchmark-tiers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        edge_id: targetEdge,
        force_offline: Boolean(state.forceOffline),
      }),
    });
    const data = await res.json();
    const colsHtml = (data.tiers || [])
      .map(function (t) {
        return `
          <div class="benchmark-col">
            <div class="benchmark-col-title">${t.tier_label}</div>
            <div>Status: <code>${t.tier_status}</code></div>
            <div>Latency: <strong>${t.latency_ms} ms</strong></div>
            <div>Severity: <code>${t.damage_severity}</code></div>
            <div>Span: <strong>${t.min_clear_width_m}m</strong></div>
            <div>MemClear: <code>${t.memory_cleared}</code></div>
          </div>
        `;
      })
      .join("");

    boxEl.innerHTML = `
      <div class="benchmark-grid">${colsHtml}</div>
    `;
    await refreshMetaModelsTelemetry();
  }

  async function runDemoScenario(scenarioNum) {
    document.querySelectorAll(".demo-scenario-btn").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-scenario") === String(scenarioNum));
    });

    if (String(scenarioNum) === "1") {
      // Scenario 1: Differential Vehicle Routing (Pinched corridor passable for ATV/Ambulance, blocked for Fire Engine)
      state.vehicleClass = "fire_engine";
      document.querySelectorAll(".vehicle-btn").forEach(function (b) {
        b.classList.toggle("active", b.getAttribute("data-vehicle") === "fire_engine");
      });
      renderRoadsLayer();
      await Promise.all([refreshRoute(), refreshClearanceRoi()]);
      if (state.lastRoiRecommendations.length > 0) {
        const rec = state.lastRoiRecommendations[0];
        const edgeId = rec.edge_id || `${rec.edge_u}-${rec.edge_v}-${rec.edge_key}`;
        await inspectCorridorByEdgeId(edgeId, true);
      }
    } else if (String(scenarioNum) === "2") {
      // Scenario 2: SAM 3.1 5m Caliper & ROI Rescue
      state.showCaliperRays = true;
      const rayToggle = document.getElementById("toggle-caliper-rays");
      if (rayToggle) rayToggle.checked = true;
      state.inspectorViewMode = "schematic";
      const btnCrop = document.getElementById("btn-view-crop");
      const btnSchematic = document.getElementById("btn-view-schematic");
      if (btnCrop && btnSchematic) {
        btnSchematic.classList.add("active");
        btnCrop.classList.remove("active");
      }
      if (state.lastRoiRecommendations.length > 0) {
        const rec = state.lastRoiRecommendations[0];
        const edgeId = rec.edge_id || `${rec.edge_u}-${rec.edge_v}-${rec.edge_key}`;
        await inspectCorridorByEdgeId(edgeId, true);
      }
    } else if (String(scenarioNum) === "3") {
      // Scenario 3: Air-Gap Blackout & 3-Tier Benchmark
      state.forceOffline = true;
      const blackoutToggle = document.getElementById("toggle-comms-blackout");
      if (blackoutToggle) {
        blackoutToggle.checked = true;
        const labelParent = blackoutToggle.closest(".blackout-switch");
        if (labelParent) labelParent.classList.add("active-blackout");
      }
      const dot = document.getElementById("network-status-dot");
      const netText = document.getElementById("network-status-text");
      if (dot) {
        dot.classList.remove("online");
        dot.classList.add("offline");
      }
      if (netText) {
        netText.textContent = "COMMS BLACKOUT (EDGE ONLY)";
      }
      const targetEdge = state.activeEdgeId || "0";
      await Promise.all([
        runVlmAssessmentForEdge(targetEdge),
        runBenchmarkTiers(targetEdge),
      ]);
    }
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

    const blackoutToggle = document.getElementById("toggle-comms-blackout");
    blackoutToggle.addEventListener("change", async function (ev) {
      state.forceOffline = Boolean(ev.target.checked);
      const labelParent = blackoutToggle.closest(".blackout-switch");
      if (labelParent) {
        labelParent.classList.toggle("active-blackout", state.forceOffline);
      }
      const dot = document.getElementById("network-status-dot");
      const netText = document.getElementById("network-status-text");
      if (state.forceOffline) {
        dot.classList.remove("online");
        dot.classList.add("offline");
        netText.textContent = "COMMS BLACKOUT (EDGE ONLY)";
      } else {
        dot.classList.remove("offline");
        dot.classList.add("online");
        netText.textContent = "LOCAL WI-FI / OFFLINE READY";
      }
      if (state.activeEdgeId) {
        await runVlmAssessmentForEdge(state.activeEdgeId);
      }
    });

    document.getElementById("toggle-caliper-rays").addEventListener("change", function (ev) {
      state.showCaliperRays = Boolean(ev.target.checked);
      if (state.showCaliperRays && state.lastCaliperGeojson) {
        renderCaliperLayer(state.lastCaliperGeojson, false);
      } else {
        state.caliperLayer.clearLayers();
      }
    });

    document.getElementById("btn-export-manifest").addEventListener("click", function () {
      exportFieldManifestGeojson();
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

    const btnFitAoi = document.getElementById("btn-fit-aoi");
    if (btnFitAoi) {
      btnFitAoi.addEventListener("click", function () {
        applyAoiViewportBounds(true);
      });
    }

    document.getElementById("btn-run-assess").addEventListener("click", function () {
      const targetEdge = state.activeEdgeId || "0";
      runVlmAssessmentForEdge(targetEdge);
    });

    const btnCrop = document.getElementById("btn-view-crop");
    const btnSchematic = document.getElementById("btn-view-schematic");
    if (btnCrop && btnSchematic) {
      btnCrop.addEventListener("click", function () {
        state.inspectorViewMode = "crop";
        btnCrop.classList.add("active");
        btnSchematic.classList.remove("active");
        updateInspectorImage();
      });
      btnSchematic.addEventListener("click", async function () {
        state.inspectorViewMode = "schematic";
        btnSchematic.classList.add("active");
        btnCrop.classList.remove("active");
        updateInspectorImage();
        await refreshMetaModelsTelemetry();
      });
    }

    const btnWonder = document.getElementById("btn-wonder-briefing");
    if (btnWonder) {
      btnWonder.addEventListener("click", function () {
        runWonderBriefing();
      });
    }

    const btnVoice = document.getElementById("btn-voice-dispatch");
    if (btnVoice) {
      btnVoice.addEventListener("click", function () {
        runVoiceDispatch();
      });
    }

    const btnBenchmark = document.getElementById("btn-benchmark-tiers");
    if (btnBenchmark) {
      btnBenchmark.addEventListener("click", function () {
        runBenchmarkTiers(state.activeEdgeId || "0");
      });
    }

    document.querySelectorAll(".demo-scenario-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        runDemoScenario(btn.getAttribute("data-scenario"));
      });
    });
  }

  document.addEventListener("DOMContentLoaded", async function () {
    initMap();
    bindUiEvents();
    await fetchHealth();
    await loadInitialLayers();
    await Promise.all([refreshRoute(), refreshClearanceRoi(), refreshMetaModelsTelemetry()]);
  });
})();
