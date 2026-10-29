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
    schematicLabel: "Deterministic Caliper Overlay (Muse Image Offline)",
    highlightedUnlockedNodes: [],
    lastCaliperGeojson: null,
    caliperAnimToken: 0,
    boundsWgs84: [-85.4251, 29.9249, -85.3999, 29.9501],
    selectedSegment: null,
    map: null,
    rasterOverlay: null,
    rasterTiles: null,
    buildingsLayer: null,
    roadsLayer: null,
    routeLayer: null,
    caliperLayer: null,
    markersLayer: null,
  };

  // Map colours match the legend and the CSS tokens.
  const C = {
    open: "#34d399",
    narrow: "#f5a524",
    blocked: "#f87171",
    water: "#a78bfa",
    route: "#5b8cff",
    ink: "#0a0e15",
  };
  const VEHICLE_MIN_M = { atv: 1.2, ambulance: 2.2, fire_engine: 3.5 };
  const VEHICLE_SHORT = { atv: "ATV", ambulance: "Ambulance", fire_engine: "Fire engine" };

  const VEHICLE_LABELS = {
    atv: "ATV / Foot (>=1.2m)",
    ambulance: "Ambulance / Pickup (>=2.2m)",
    fire_engine: "Fire Engine (>=3.5m)",
  };

  function formatTierLabel(tierCode) {
    const t = String(tierCode || "").toLowerCase();
    if (t.includes("tier_1")) {
      return "Muse Spark · Meta Cloud";
    }
    if (t.includes("tier_2")) {
      return "Muse Glimmer · this laptop";
    }
    return "Rules only · no model";
  }

  // Top-bar AI pill: which Meta model answers right now (and warm-up progress).
  function setAiPill(tierCode) {
    const pill = document.getElementById("vlm-tier-badge");
    const t = String(tierCode || "").toLowerCase();
    pill.textContent = t.includes("tier_1") ? "AI · Meta Cloud" : t.includes("tier_2") ? "AI · On-device" : "AI · Rules only";
    pill.classList.toggle("edge", !t.includes("tier_1"));
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
      zoomControl: false,
      attributionControl: false,
    });
    L.control.zoom({ position: "bottomright" }).addTo(map);

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
    state.map.getContainer().classList.toggle("picking", Boolean(mode));
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

      setNetworkStatus();
      const tiers = data.vlm_tiers || {};
      setAiPill(state.forceOffline ? "tier_2" : tiers.active_tier);

      const gsd = data.geospatial ? data.geospatial.gsd_m_per_px : 0.2804;
      document.getElementById("hw-status-badge").textContent =
        `SAM 3.1 fp16 on MPS · Muse Glimmer 4-bit on MLX · ${gsd} m/px`;
      return renderWarmup(tiers.warmup || {});
    } catch (err) {
      document.getElementById("network-status-text").textContent = "Server unreachable";
      return false;
    }
  }

  function setNetworkStatus() {
    const dot = document.getElementById("network-status-dot");
    const text = document.getElementById("network-status-text");
    dot.classList.toggle("online", !state.forceOffline);
    dot.classList.toggle("offline", state.forceOffline);
    text.textContent = state.forceOffline ? "Blackout · on-device" : "Online";
  }

  // Warm-up loads Muse Glimmer and pre-grades the top choke points; returns true when finished.
  function renderWarmup(w) {
    const el = document.getElementById("warmup-status");
    const pill = document.getElementById("vlm-tier-badge");
    let text = "";
    let done = true;
    if (w.edge_model === "loading") {
      text = "Loading Muse Glimmer…";
      done = false;
    } else if (w.edge_cache_target !== undefined && w.edge_cache_prewarmed < w.edge_cache_target) {
      text = `Pre-grading roads on-device ${w.edge_cache_prewarmed}/${w.edge_cache_target}`;
      done = false;
    } else if (w.edge_model === "resident" && w.edge_cache_target === undefined && !w.cloud_cache_error) {
      text = `Pre-grading roads in the cloud ${w.cloud_cache_prewarmed || 0}/${w.cloud_cache_target || 0}`;
      done = false;
    } else if (w.edge_model === "resident") {
      text = "Ready";
    }
    el.textContent = text;
    pill.classList.toggle("warming", !done);
    pill.title = done ? "Which Meta model is answering; click for all models" : `Warming up: ${text}`;
    return done;
  }

  function pollWarmup() {
    fetchHealth().then(function (done) {
      if (!done) setTimeout(pollWarmup, 5000);
    });
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
    // Full-resolution offline tiles (0.28 m) on top of the low-res preview, which stays as a
    // fallback when tiles have not been built.
    if (state.rasterTiles) {
      state.map.removeLayer(state.rasterTiles);
    }
    state.rasterTiles = L.tileLayer("/tiles/post/{z}/{x}/{y}.jpg", {
      minZoom: 13,
      maxNativeZoom: 19,
      maxZoom: 20,
      bounds: L.latLngBounds(imageBounds),
      opacity: 0.7,
    }).addTo(state.map);

    applyAoiViewportBounds(false);
    renderBuildingsLayer();
    renderRoadsLayer();
    if (!state.fittedToRoads && state.roadsGeojson) {
      state.fittedToRoads = true;
      fitRoads();
    }
  }

  // Frame the measured road network between the KPI cards (top) and the command bar (bottom).
  function fitRoads() {
    const bounds = L.geoJSON(state.roadsGeojson).getBounds();
    if (bounds.isValid()) {
      state.map.fitBounds(bounds, { paddingTopLeft: [150, 120], paddingBottomRight: [70, 130] });
    }
  }

  function renderBuildingsLayer() {
    state.buildingsLayer.clearLayers();
    const toggle = document.getElementById("toggle-buildings");
    if (!toggle.checked || !state.buildingsGeojson) {
      return;
    }

    const sevColors = {
      destroyed: "#f87171",
      major_damage: "#fb923c",
      minor_damage: "#facc15",
      no_damage: "#64748b",
    };

    const unlockedSet = new Set((state.highlightedUnlockedNodes || []).map(String));

    L.geoJSON(state.buildingsGeojson, {
      style: function (feature) {
        const p = feature.properties || {};
        const sev = p.damage_severity || "no_damage";
        const isUnlocked = unlockedSet.has(String(p.nearest_node || ""));
        if (isUnlocked && p.is_high_severity) {
          return {
            color: C.open,
            fillColor: C.open,
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
          `Muse Spark grade: <code>${String(p.damage_severity || "").replace("_", " ")}</code>` +
          (isUnlocked ? `<br/><strong style="color:${C.open};">Reachable once the selected pile is cleared</strong>` : "")
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
          return { color: C.water, weight: 4, opacity: 0.95, dashArray: "6, 6" };
        }
        if (p[passKey]) {
          return { color: C.open, weight: 3, opacity: 0.8 };
        }
        // Some clear width, just not enough for this vehicle, vs no usable width at all.
        if (Number(p.min_clear_width_m) >= VEHICLE_MIN_M.atv) {
          return { color: C.narrow, weight: 4, opacity: 0.95 };
        }
        return { color: C.blocked, weight: 4, opacity: 0.95 };
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
          `Clear width ${Number(p.min_clear_width_m).toFixed(1)} m · debris ${Math.round(Number(p.debris_area_sqm) || 0)} m²` +
          (Number(p.water_overlap_ratio) > 0 ? " · standing water" : ""),
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
        captionEl.textContent =
          state.schematicLabel || "Deterministic Caliper Overlay (Muse Image Offline)";
      }
    } else {
      cropImg.src = `/api/choke-point/${encodedId}/crop.png`;
      if (captionEl) {
        captionEl.textContent = "NOAA post-storm aerial, Oct 11 2018";
      }
    }
  }

  function updateCaliperThresholdGauge(minClearWidthM, hasWater) {
    const w = hasWater ? 0.0 : Math.max(0.0, Number(minClearWidthM || 0));
    const maxScaleM = 6.0;
    const pct = Math.min(100, Math.max(0, (w / maxScaleM) * 100));

    const labelEl = document.getElementById("gauge-measured-span-label");
    if (labelEl) {
      labelEl.textContent = `${w.toFixed(1)} m`;
    }
    const chip = document.getElementById("inspector-status-chip");
    if (chip) {
      const passCount = ["atv", "ambulance", "fire_engine"].filter(function (vc) {
        return !hasWater && w >= VEHICLE_MIN_M[vc];
      }).length;
      chip.className = `chip ${passCount === 3 ? "ok" : passCount === 0 ? "bad" : "warn"}`;
      chip.textContent = hasWater
        ? "Standing water"
        : passCount === 3
        ? "Open to all vehicles"
        : passCount === 0
        ? "Blocked for all vehicles"
        : "Too narrow for some vehicles";
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
      { id: "gauge-badge-atv", label: "ATV", pass: !hasWater && w >= 1.2 },
      { id: "gauge-badge-ambulance", label: "Ambulance", pass: !hasWater && w >= 2.2 },
      { id: "gauge-badge-fire_engine", label: "Fire engine", pass: !hasWater && w >= 3.5 },
    ];
    checks.forEach(function (item) {
      const el = document.getElementById(item.id);
      if (!el) return;
      el.textContent = `${item.label} ${item.pass ? "passes" : "blocked"}`;
      el.classList.toggle("pass", item.pass);
      el.classList.toggle("blocked", !item.pass);
    });
  }

  function updateUnlockedStructuresHighlight(edgeId) {
    const rec = (state.lastRoiRecommendations || []).find(function (r) {
      const rId = r.edge_id || `${r.edge_u}-${r.edge_v}-${r.edge_key}`;
      return String(rId) === String(edgeId);
    });
    // Highlight only; the toast is reserved for an actual simulated clearance.
    if (rec && rec.high_severity_structures_unlocked > 0) {
      state.highlightedUnlockedNodes = rec.unlocked_node_ids || [rec.edge_u, rec.edge_v];
    } else {
      state.highlightedUnlockedNodes = [];
    }
    renderBuildingsLayer();
  }

  /**
   * Fetch 5m orthogonal caliper rays & SAM 3.1 masks for `edgeId`, render on map,
   * load `/api/choke-point/{edgeId}/crop.png` or `/recon-schematic.png`, and run live 3-tier VLM assessment.
   */
  function openInspector() {
    document.getElementById("corridor-inspector-dock").classList.remove("hidden");
    relayoutMap();
  }

  function closeInspector() {
    document.getElementById("corridor-inspector-dock").classList.add("hidden");
    relayoutMap();
  }

  // The map shares the row with the drawer and inspector; let Leaflet re-measure after they move.
  function relayoutMap() {
    requestAnimationFrame(function () {
      if (state.map) state.map.invalidateSize();
    });
  }

  async function inspectCorridorByEdgeId(edgeId, zoomToCorridor) {
    if (!edgeId) return;
    state.activeEdgeId = String(edgeId);
    if (zoomToCorridor) openInspector();

    // Highlight selected card in ROI list if present
    document.querySelectorAll(".roi-card").forEach(function (card) {
      card.classList.toggle("selected-corridor", card.getAttribute("data-edge-id") === state.activeEdgeId);
    });

    updateUnlockedStructuresHighlight(state.activeEdgeId);
    updateInspectorImage();

    // Render the SAM 3.1 caliper as soon as it arrives; VLM grading (seconds) runs alongside.
    const inspectedEdgeId = state.activeEdgeId;
    const vlmDone = runVlmAssessmentForEdge(inspectedEdgeId);
    const caliperRes = await fetch(`/api/choke-point/${encodeURIComponent(inspectedEdgeId)}/caliper`);

    if (caliperRes.ok && state.activeEdgeId === inspectedEdgeId) {
      const caliperGeojson = await caliperRes.json();
      state.lastCaliperGeojson = caliperGeojson;
      const cProps = caliperGeojson.properties || {};
      const animate = Boolean(zoomToCorridor) && !PREFERS_REDUCED_MOTION;
      if (cProps.min_clear_width_m !== undefined && !animate) {
        updateCaliperThresholdGauge(cProps.min_clear_width_m, Boolean(cProps.has_water_obstruction));
      }
      updateSamConfidenceAndReconBadge(cProps.sam_confidence, cProps.needs_drone_recon);
      // Replay the measurement when the user explicitly picks a corridor.
      renderCaliperLayer(caliperGeojson, Boolean(zoomToCorridor), animate);
    }
    await vlmDone;
    await refreshMetaModelsTelemetry();
  }

  function updateSamConfidenceAndReconBadge(samConfidence, needsDroneRecon) {
    const confValEl = document.getElementById("inspector-sam-conf-val");
    const badgeEl = document.getElementById("inspector-drone-recon-badge");
    if (confValEl && samConfidence !== undefined && samConfidence !== null) {
      confValEl.textContent = Number(samConfidence).toFixed(2);
    }
    if (badgeEl && needsDroneRecon !== undefined && needsDroneRecon !== null) {
      const flagged = Boolean(needsDroneRecon);
      badgeEl.textContent = flagged ? "Drone check advised" : "High confidence";
      badgeEl.classList.toggle("flagged", flagged);
      badgeEl.classList.toggle("high-conf", !flagged);
    }
  }

  const PREFERS_REDUCED_MOTION = Boolean(
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );

  function sleep(ms) {
    return new Promise(function (resolve) {
      setTimeout(resolve, ms);
    });
  }

  function maskSourceLabel(source) {
    const s = String(source || "").toLowerCase();
    if (s === "sam-3.1") return "SAM 3.1";
    if (s.includes("sim")) return "Simulated";
    return source || "Unknown";
  }

  function caliperRayStyle(fp, emphasizePinch) {
    const w = Number(fp.clear_width_m || 0);
    if ((emphasizePinch && fp.is_pinch_point) || fp.water_intersected || w < 1.2) {
      return { color: "#ef4444", weight: 5, opacity: 1.0 };
    }
    if (w >= 3.5) {
      return { color: "#10b981", weight: 3, opacity: 0.9 };
    }
    return { color: "#f59e0b", weight: 3.5, opacity: 0.95 };
  }

  function caliperFeatureStyle(feature) {
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
    if (fp.feature_type === "road_surface_mask") {
      return {
        color: "#38bdf8",
        fillColor: "#38bdf8",
        weight: 1,
        fillOpacity: 0.22,
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
      return caliperRayStyle(fp, true);
    }
    return { color: "#94a3b8", weight: 2 };
  }

  function bindCaliperTooltip(feature, layer) {
    const fp = feature.properties || {};
    if (fp.feature_type === "caliper_ray") {
      const badge = fp.is_pinch_point ? " [MIN PINCH POINT]" : "";
      const causeLabel = {
        clear: "visible pavement",
        narrowed_debris: "pavement narrowed by debris",
        debris: "pavement buried by debris",
        water: "standing water",
        not_visible: "no pavement visible &mdash; verify by drone",
      }[fp.cause];
      layer.bindTooltip(
        `<strong>5m Caliper Ray @ ${fp.station_m}m${badge}</strong><br/>` +
        `Clear Width: <strong>${fp.clear_width_m}m</strong>` +
        (causeLabel ? ` (${causeLabel})` : "") +
        `<br/>` +
        `ATV: ${fp.atv ? "PASS" : "BLOCKED"} | Amb: ${fp.ambulance ? "PASS" : "BLOCKED"} | Fire: ${fp.fire_engine ? "PASS" : "BLOCKED"}`,
        { sticky: true }
      );
    } else if (fp.feature_type === "road_surface_mask") {
      layer.bindTooltip(
        `<strong>SAM 3.1 Visible Pavement</strong><br/>Prompt: "road" (GSD: ${fp.gsd_m_per_px} m/px)`,
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
  }

  function showCaliperScanReadout(html, isVerdict) {
    const el = document.getElementById("caliper-scan-readout");
    if (!el) return;
    el.innerHTML = html;
    el.classList.toggle("verdict", Boolean(isVerdict));
    el.classList.remove("hidden");
  }

  function hideCaliperScanReadout() {
    const el = document.getElementById("caliper-scan-readout");
    if (el) el.classList.add("hidden");
  }

  function caliperVerdictText(props) {
    const minW = Number(props.min_clear_width_m || 0).toFixed(2);
    if (props.has_water_obstruction) {
      return `Standing water across the road &rarr; <strong>impassable for every vehicle</strong>`;
    }
    const pass = props.passability || {};
    const names = { fire_engine: "Fire engine", ambulance: "Ambulance", atv: "ATV" };
    const ok = [];
    const blocked = [];
    ["fire_engine", "ambulance", "atv"].forEach(function (vc) {
      (pass[vc] ? ok : blocked).push(names[vc]);
    });
    const verdict = [
      ok.length ? `<span class="verdict-pass">${ok.join(" &amp; ")} pass</span>` : "",
      blocked.length ? `<span class="verdict-block">${blocked.join(" &amp; ")} blocked</span>` : "",
    ]
      .filter(Boolean)
      .join(" &middot; ");
    return (
      `Narrowest clear span <strong>${minW} m</strong> at ` +
      `${Number(props.pinch_point_station_m || 0).toFixed(0)} m &rarr; ${verdict}`
    );
  }

  /**
   * Replay the SAM 3.1 caliper measurement on the map: masks fade in, then each 5 m
   * cross-section is cast in station order while the live readout and threshold gauge
   * track the narrowest clear span found so far. Every number shown comes from the
   * `/caliper` response; the animation only changes when it is revealed.
   */
  async function animateCaliperSweep(caliperGeojson, token) {
    const props = caliperGeojson.properties || {};
    const features = caliperGeojson.features || [];
    const maskFeatures = features.filter(function (f) {
      return (f.properties || {}).feature_type !== "caliper_ray";
    });
    const rays = features
      .filter(function (f) {
        return (f.properties || {}).feature_type === "caliper_ray";
      })
      .sort(function (a, b) {
        return Number(a.properties.station_m) - Number(b.properties.station_m);
      });
    const stillCurrent = function () {
      return token === state.caliperAnimToken && state.showCaliperRays;
    };

    L.geoJSON(
      { type: "FeatureCollection", features: maskFeatures },
      {
        style: function (f) {
          return Object.assign({}, caliperFeatureStyle(f), { className: "caliper-fade-in" });
        },
        onEachFeature: bindCaliperTooltip,
      }
    ).addTo(state.caliperLayer);

    const source = maskSourceLabel(props.mask_source);
    if (rays.length) {
      const first = rays[0].properties || {};
      updateCaliperThresholdGauge(first.clear_width_m, Boolean(first.water_intersected));
    }
    showCaliperScanReadout(
      `<span class="scan-label">${source} masks</span> Casting ${rays.length} cross-sections every 5 m&hellip;`
    );
    await sleep(500);

    const stepMs = Math.min(140, Math.max(40, 2400 / Math.max(1, rays.length)));
    let narrowest = Infinity;
    let sawWater = false;
    for (const ray of rays) {
      if (!stillCurrent()) return;
      const fp = ray.properties || {};
      const fullRay = (fp.full_ray_coords_wgs84 || []).map(function (c) {
        return [c[1], c[0]];
      });
      const probe =
        fullRay.length >= 2
          ? L.polyline(fullRay, {
              color: "#e2e8f0",
              weight: 1.5,
              opacity: 0.85,
              dashArray: "3, 4",
              interactive: false,
            }).addTo(state.caliperLayer)
          : null;
      L.geoJSON(ray, {
        style: function (f) {
          return caliperRayStyle(f.properties || {}, false);
        },
        onEachFeature: bindCaliperTooltip,
      }).addTo(state.caliperLayer);

      const w = Number(fp.clear_width_m || 0);
      narrowest = Math.min(narrowest, w);
      sawWater = sawWater || Boolean(fp.water_intersected);
      updateCaliperThresholdGauge(narrowest, sawWater);
      showCaliperScanReadout(
        `<span class="scan-label">${source} caliper</span> Station ${Number(fp.station_m).toFixed(0)} m &middot; ` +
        `<strong>${w.toFixed(2)} m</strong> clear &middot; narrowest so far ${narrowest.toFixed(2)} m`
      );
      await sleep(stepMs);
      if (probe) state.caliperLayer.removeLayer(probe);
    }
    if (!stillCurrent()) return;

    const pinch = rays.find(function (r) {
      return (r.properties || {}).is_pinch_point;
    });
    if (pinch) {
      L.geoJSON(pinch, {
        style: { color: "#ef4444", weight: 9, opacity: 1, className: "pinch-pulse" },
        onEachFeature: bindCaliperTooltip,
      }).addTo(state.caliperLayer);
    }
    updateCaliperThresholdGauge(props.min_clear_width_m, Boolean(props.has_water_obstruction));
    showCaliperScanReadout(caliperVerdictText(props), true);
    await sleep(7000);
    if (stillCurrent()) hideCaliperScanReadout();
  }

  function renderCaliperLayer(caliperGeojson, zoomToCorridor, animate) {
    // Any new render cancels an in-flight sweep.
    state.caliperAnimToken += 1;
    state.caliperLayer.clearLayers();
    hideCaliperScanReadout();
    if (!caliperGeojson || !state.showCaliperRays) {
      return;
    }

    const props = caliperGeojson.properties || {};
    const roadNameEl = document.getElementById("inspector-road-name");
    if (roadNameEl && (props.road_name || props.edge_id)) {
      // Keep the heading in sync with the measurement while VLM grading is still running.
      roadNameEl.textContent = props.road_name || props.edge_id;
    }
    const caliperReadingEl = document.getElementById("inspector-caliper-reading");
    if (caliperReadingEl && props.min_clear_width_m !== undefined) {
      caliperReadingEl.textContent =
        `narrowest at ${Number(props.pinch_point_station_m || 0).toFixed(0)} m · ` +
        `${props.ray_count || 0} cuts every 5 m`;
    }

    const bounds = L.geoJSON(caliperGeojson).getBounds();
    if (zoomToCorridor && bounds.isValid()) {
      state.map.fitBounds(bounds.pad(0.45), { maxZoom: 18 });
    }

    if (animate && !PREFERS_REDUCED_MOTION) {
      animateCaliperSweep(caliperGeojson, state.caliperAnimToken);
      return;
    }

    L.geoJSON(caliperGeojson, {
      style: caliperFeatureStyle,
      onEachFeature: bindCaliperTooltip,
    }).addTo(state.caliperLayer);
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
    pushFieldDispatch({
      vehicle_class: state.vehicleClass,
      origin: payload.origin,
      destination: payload.destination,
    });
  }

  /**
   * Push the commander's current assignment to field devices (`POST /api/dispatch`).
   * The server only re-broadcasts when something actually changed.
   */
  async function pushFieldDispatch(update) {
    try {
      await fetch("/api/dispatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(update),
      });
    } catch (err) {
      // Field sync is best-effort; the commander view keeps working without it.
    }
  }

  /**
   * Follow the dispatch stream as a `command` client to show how many field devices are
   * live and which URL they should open.
   */
  function connectFieldLinkStatus() {
    if (!window.EventSource) return;
    const pill = document.getElementById("field-link-pill");
    const textEl = document.getElementById("field-link-text");
    const source = new EventSource("/api/dispatch/stream?role=command");
    source.addEventListener("dispatch", function (ev) {
      let snapshot;
      try {
        snapshot = JSON.parse(ev.data);
      } catch (err) {
        return;
      }
      const count = (snapshot.connectivity && snapshot.connectivity.field_clients) || 0;
      pill.classList.toggle("live", count > 0);
      textEl.textContent = `${count} field device${count === 1 ? "" : "s"}`;
      pill.title = snapshot.field_url
        ? `Open ${snapshot.field_url} on a phone on this Wi-Fi`
        : "Field view: /field on this laptop's LAN address";
    });
    source.addEventListener("error", function () {
      pill.classList.remove("live");
    });
  }

  function renderRouteOnMap(routeData) {
    state.routeLayer.clearLayers();
    state.markersLayer.clearLayers();

    if (state.origin) {
      L.circleMarker([state.origin.lat, state.origin.lon], {
        radius: 7,
        color: C.ink,
        fillColor: C.route,
        fillOpacity: 1,
        weight: 2,
      })
        .bindTooltip("Command post", { permanent: false })
        .addTo(state.markersLayer);
    }

    if (state.destination) {
      L.circleMarker([state.destination.lat, state.destination.lon], {
        radius: 7,
        color: C.ink,
        fillColor: "#e7ecf4",
        fillOpacity: 1,
        weight: 2,
      })
        .bindTooltip("Destination", { permanent: false })
        .addTo(state.markersLayer);
    }

    if (routeData.reachable && routeData.geojson_geometry && routeData.geojson_geometry.coordinates.length > 0) {
      const latLngs = routeData.geojson_geometry.coordinates.map(function (c) {
        return [c[1], c[0]];
      });
      L.polyline(latLngs, { color: C.ink, weight: 10, opacity: 0.55 }).addTo(state.routeLayer);
      L.polyline(latLngs, { color: C.route, weight: 6, opacity: 1 }).addTo(state.routeLayer);
    }
  }

  function renderRouteBanner(routeData) {
    document.getElementById("route-class-label").textContent = VEHICLE_SHORT[state.vehicleClass] || state.vehicleClass;

    const statusEl = document.getElementById("route-status-val");
    statusEl.textContent = routeData.reachable ? "Open" : "Blocked";
    statusEl.className = `num ${routeData.reachable ? "ok" : "bad"}`;

    document.getElementById("route-distance-val").textContent = routeData.reachable
      ? `${(routeData.total_distance_m / 1000).toFixed(2)} km`
      : "no route";
    document.getElementById("route-eta-val").textContent = routeData.reachable
      ? `${(routeData.estimated_transit_time_s / 60).toFixed(1)} min`
      : "–";
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
        const tag = info.reachable
          ? `<span class="rc-val ok-text">${(info.total_distance_m / 1000).toFixed(2)} km · ${(info.estimated_transit_time_s / 60).toFixed(1)} min</span>`
          : `<span class="rc-val bad-text">no route</span>`;
        parts.push(
          `<div class="rc-row ${vc === state.vehicleClass ? "active" : ""}"><span>${VEHICLE_SHORT[vc]} <span class="muted">&gt;${VEHICLE_MIN_M[vc]} m</span></span>${tag}</div>`
        );
      }
    });
    summaryEl.innerHTML = parts.join("");
  }

  function renderTurnByTurnSteps(steps) {
    document.getElementById("steps-count").textContent = String(steps.length);
    const listEl = document.getElementById("route-steps-list");
    if (!steps.length) {
      listEl.innerHTML = `<div class="roi-explanation">No passable route for ${VEHICLE_SHORT[state.vehicleClass]}. Ask FirstLight which piles to clear.</div>`;
      return;
    }

    listEl.innerHTML = steps
      .map(function (st, idx) {
        return `
          <div class="step-item">
            <div><strong>${idx + 1}. ${st.name}</strong></div>
            <div class="step-meta">${Math.round(st.length_m)} m · narrowest ${Number(st.min_clear_width_m).toFixed(1)} m · ${Math.round(st.travel_time_s)} s</div>
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
    refreshSituation();

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
              <span class="roi-rank-badge">#${rec.rank}</span>
              <span class="roi-score">+${rec.high_severity_structures_unlocked} buildings</span>
            </div>
            <div class="roi-road-title">${rec.road_name}</div>
            <div class="roi-metrics-row">
              <span>${rec.current_clear_width_m} → ${rec.restored_clear_width_m} m</span>
              <span>${rec.debris_area_sqm} m² debris</span>
              <span>${(100 * rec.clearance_roi).toFixed(0)} bldgs / 100 m²</span>
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
                ${cleared ? "✓ Cleared (undo)" : "Simulate clearance"}
              </button>
              <button
                type="button"
                class="action-btn btn-inspect-roi"
                data-edge-id="${edgeId}"
              >
                Measure
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
    refreshSituation();

    if (priorRec && priorRec.high_severity_structures_unlocked > 0) {
      state.highlightedUnlockedNodes = priorRec.unlocked_node_ids || [u, v];
      const bannerEl = document.getElementById("unlocked-impact-banner");
      const textEl = document.getElementById("unlocked-impact-text");
      if (bannerEl && textEl) {
        textEl.textContent =
          `Cleared ${priorRec.road_name}: +${priorRec.high_severity_structures_unlocked} severely damaged buildings now reachable`;
        bannerEl.classList.remove("hidden");
        clearTimeout(state.toastTimer);
        state.toastTimer = setTimeout(function () {
          bannerEl.classList.add("hidden");
        }, 7000);
      }
      renderBuildingsLayer();
      await refreshActiveCaliper();
    } else if (state.activeEdgeId) {
      await inspectCorridorByEdgeId(state.activeEdgeId, false);
    }
  }

  /**
   * Re-read the selected corridor's caliper after a simulated clearance so the gauge, pinch
   * span and verdict show the restored width (no new VLM grading call).
   */
  async function refreshActiveCaliper() {
    if (!state.activeEdgeId) return;
    const res = await fetch(`/api/choke-point/${encodeURIComponent(state.activeEdgeId)}/caliper`);
    if (!res.ok) return;
    const caliperGeojson = await res.json();
    state.lastCaliperGeojson = caliperGeojson;
    const props = caliperGeojson.properties || {};
    renderCaliperLayer(caliperGeojson, false, false);
    updateCaliperThresholdGauge(props.min_clear_width_m, Boolean(props.has_water_obstruction));
    document.getElementById("inspector-rationale-text").textContent = props.reason || "";
    if (props.simulated_clearance) {
      showCaliperScanReadout(`${caliperVerdictText(props)} <span class="scan-label">simulated clearance</span>`, true);
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
    refreshSituation();
    if (state.activeEdgeId) {
      await inspectCorridorByEdgeId(state.activeEdgeId, false);
    }
  }

  function updateClearedCounter() {
    const badge = document.getElementById("cleared-count");
    badge.textContent = String(state.clearedEdges.length);
    badge.setAttribute("data-zero", String(state.clearedEdges.length === 0));
  }

  async function runVlmAssessmentForEdge(edgeId) {
    const modeSelect = document.getElementById("select-failover-mode");
    const boxEl = document.getElementById("assess-result-box");
    boxEl.innerHTML = `<span class="analyzing">Grading the selected road…</span>`;
    document.getElementById("inspector-latency-row").innerHTML = state.forceOffline
      ? `<span class="analyzing">On-device Muse Glimmer analyzing aerial crop…</span>`
      : `<span class="analyzing">Muse Spark 1.3 analyzing aerial crop…</span>`;
    // Never leave the previous corridor's grades on screen while the new one is graded.
    document.getElementById("inspector-severity-row").innerHTML = "Building: &hellip; &bull; Road: &hellip;";
    document.getElementById("inspector-rationale-text").textContent = "Waiting for damage grading…";

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

    // Update Corridor AI Inspector dock
    document.getElementById("inspector-tier-pill").textContent = formattedTier;
    // The caliper response carries the readable road name; only fall back to the id without it.
    const caliperProps = (state.lastCaliperGeojson && state.lastCaliperGeojson.properties) || {};
    if (caliperProps.edge_id !== String(edgeId)) {
      document.getElementById("inspector-road-name").textContent = data.target_id || String(edgeId);
    }
    document.getElementById("inspector-latency-row").innerHTML =
      `${(data.latency_ms / 1000).toFixed(1)} s` +
      (data.cloud_cache_hit
        ? ` · <span class="cache-hit" title="Real Muse Spark response computed earlier for this exact crop and measurement">cached Muse Spark response</span>`
        : "");
    document.getElementById("inspector-severity-row").innerHTML =
      `Buildings <strong>${String(data.building_damage_severity).replace("_", " ")}</strong> · ` +
      `road <strong>${String(data.road_damage_severity).replace("_", " ")}</strong>`;
    if (data.sam_confidence !== undefined) {
      updateSamConfidenceAndReconBadge(data.sam_confidence, data.needs_drone_recon);
    }
    document.getElementById("inspector-rationale-text").textContent =
      data.passability_rationale || "";

    boxEl.innerHTML = `
      <div><strong>${formattedTier}</strong> · ${(data.latency_ms / 1000).toFixed(1)} s</div>
      <div>Buildings <code>${data.building_damage_severity}</code> · road <code>${data.road_damage_severity}</code></div>
      <div class="muted">Fallback chain: <code>${(data.fallback_chain || []).join(" → ")}</code></div>
      <div class="muted">Blackout simulated: <code>${data.force_offline}</code> · memory released: <code>${data.memory_cleared}</code></div>
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
      if (data.schematic_label) {
        state.schematicLabel = data.schematic_label;
        const btnSchematic = document.getElementById("btn-view-schematic");
        if (btnSchematic) {
          btnSchematic.title = data.schematic_label;
        }
      }
      (data.models || []).forEach(function (m) {
        const chip = document.querySelector(`.meta-model-chip[data-model="${m.model_id}"]`);
        if (!chip) return;
        chip.classList.toggle("edge-mode", !m.cloud_primary_active);
        const subEl = chip.querySelector(".chip-sub");
        if (subEl) {
          const calls = `${m.call_count} call${m.call_count === 1 ? "" : "s"}`;
          subEl.textContent = !m.call_count
            ? "ready"
            : m.last_latency_ms > 0
            ? `${calls} · ${(m.last_latency_ms / 1000).toFixed(1)} s`
            : `${calls} · cached`;
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
    boxEl.innerHTML = `<span class="analyzing">Drafting the briefing…</span>`;

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

    const headerLabel =
      String(data.tier_used).includes("tier_1")
        ? "Muse Wonder 1.0 Cloud Action Plan"
        : "Deterministic Geometric Triage (No LLM Executed)";

    boxEl.innerHTML = `
      <div><strong>${headerLabel} (${formatTierLabel(data.tier_used)} &bull; ${data.latency_ms} ms):</strong></div>
      <div>${data.executive_summary}</div>
      ${sectorsHtml}
    `;
    await refreshMetaModelsTelemetry();
  }

  async function runBenchmarkTiers(edgeId) {
    const targetEdge = String(edgeId || state.activeEdgeId || "0");
    const boxEl = document.getElementById("benchmark-tiers-box");
    if (!boxEl) return;
    boxEl.classList.remove("hidden");
    boxEl.innerHTML = `<span class="analyzing">Grading this road on all three tiers…</span>`;

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
            <div>${(t.latency_ms / 1000).toFixed(1)} s · <code>${t.damage_severity}</code> · width ${t.min_clear_width_m} m</div>
            <div class="muted"><code>${t.tier_status}</code></div>
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
      setNetworkStatus();
      setAiPill("tier_2");
      pushFieldDispatch({ force_offline: true });
      const targetEdge = state.activeEdgeId || "0";
      await Promise.all([
        runVlmAssessmentForEdge(targetEdge),
        runBenchmarkTiers(targetEdge),
      ]);
    }
  }

  // ---- Ask FirstLight: commander agent (POST /api/agent/ask, streamed NDJSON) ----
  const AGENT_TIER_LABELS = {
    tier_1_cloud: "Muse Spark 1.3 · Meta Cloud",
    tier_2_edge: "Muse Glimmer · on this laptop",
    tier_3_deterministic: "No model · rules + measured tools",
  };

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function summarizeToolResult(name, r) {
    if (!r || r.error) return `⚠ ${r ? r.error : "no result"}`;
    if (name === "measure_road") {
      const ok = Object.keys(r.passable_end_to_end || {}).filter((k) => r.passable_end_to_end[k]);
      return `${r.road}: narrowest ${r.narrowest_clear_width_m} m · passable for ${ok.length ? ok.join(", ") : "none"}`;
    }
    if (name === "damage_hotspots") {
      return (r.hotspots || [])
        .map((h) => `${h.id} near ${h.near_road}: ${h.high_severity_buildings} severe`)
        .join(" · ");
    }
    if (name === "plan_route") {
      return r.reachable
        ? `${r.vehicle} → ${r.destination}: ${r.distance_km} km · ${r.eta_min} min`
        : `${r.vehicle} → ${r.destination}: no passable route`;
    }
    if (name === "plan_clearance_path") {
      if (r.reachable_now) return `${r.vehicle} → ${r.destination}: already reachable`;
      if (!r.clearance_opens_route) return `no clearance opens ${r.destination}`;
      const recon = r.segments_needing_drone_recon ? ` · ${r.segments_needing_drone_recon} need drone recon` : "";
      return `clear ${r.segment_count} segments (${r.measured_debris_m2} m² measured) → ${r.route_after_clearance.distance_km} km · ${r.route_after_clearance.eta_min} min${recon}`;
    }
    if (name === "rank_clearance") {
      const c = (r.choke_points || [])[0];
      return c ? `#1 ${c.road}: ${c.debris_m2} m² debris → +${c.high_severity_buildings_unlocked} severe buildings` : "no clearance helps";
    }
    if (name === "reachability_summary") {
      return `${r.reachable} of ${r.high_severity_buildings_total} severe buildings reachable`;
    }
    if (name === "dispatch_field_unit") {
      return r.dispatched
        ? `assignment sent to ${r.field_devices_notified} field device(s)`
        : `not sent: ${r.reason}`;
    }
    return "";
  }

  function highlightNumbers(text, unverified) {
    const bad = new Set(unverified || []);
    // Street ordinals ("16th") are names, not measurements, so they are not highlighted.
    return escapeHtml(text).replace(/(?<![\w.])\d+(?:[.,]\d+)?(?!\d|[.,]\d|st\b|nd\b|rd\b|th\b)/g, function (n) {
      return `<span class="${bad.has(n) ? "num-bad" : "num-ok"}">${n}</span>`;
    });
  }

  function applyAgentDispatch(result) {
    if (!result || !result.dispatched) return;
    state.vehicleClass = result.vehicle_class;
    document.querySelectorAll(".vehicle-btn").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-vehicle") === state.vehicleClass);
    });
    state.destination = { lon: result.destination_lonlat[0], lat: result.destination_lonlat[1] };
    renderRoadsLayer();
    refreshRoute();
    refreshClearanceRoi();
  }

  // Only the latest question owns the panel: a new ask (e.g. a suggestion chip clicked while the
  // agent is still working) cancels the previous stream, whose late events are then ignored.
  let agentRun = null;

  async function askAgent(question) {
    const q = String(question || "").trim();
    if (q.length < 3) return;
    if (agentRun) agentRun.abort();
    const run = new AbortController();
    agentRun = run;
    const panel = document.getElementById("agent-panel");
    const stepsEl = document.getElementById("agent-steps");
    const answerEl = document.getElementById("agent-answer");
    const verifyEl = document.getElementById("agent-verify");
    const tierEl = document.getElementById("agent-tier");
    const askBtn = document.getElementById("agent-ask-btn");
    panel.classList.remove("hidden");
    document.getElementById("agent-question").textContent = q;
    stepsEl.innerHTML = "";
    answerEl.innerHTML = `<span class="analyzing">Planning…</span>`;
    verifyEl.textContent = "";
    verifyEl.className = "agent-verify";
    askBtn.disabled = true;
    const pending = {};

    function onEvent(ev) {
      if (agentRun !== run) return;
      if (ev.type === "tier") {
        tierEl.textContent = AGENT_TIER_LABELS[ev.tier] || ev.tier;
        tierEl.classList.toggle("edge", ev.tier !== "tier_1_cloud");
      } else if (ev.type === "fallback") {
        const li = document.createElement("li");
        li.className = "fallback";
        li.textContent = `${AGENT_TIER_LABELS[ev.tier] || ev.tier} unavailable: ${ev.reason}. Falling back.`;
        stepsEl.appendChild(li);
      } else if (ev.type === "tool_call") {
        const li = document.createElement("li");
        const args = Object.values(ev.args || {}).map((a) => escapeHtml(a)).join(", ");
        li.innerHTML = `<span class="tool-name">${escapeHtml(ev.name)}</span>(${args})<span class="tool-summary analyzing">running…</span>`;
        stepsEl.appendChild(li);
        pending[ev.name] = (pending[ev.name] || []).concat([li]);
      } else if (ev.type === "tool_result") {
        const li = (pending[ev.name] || []).shift();
        if (li) {
          const s = li.querySelector(".tool-summary");
          s.classList.remove("analyzing");
          s.textContent = summarizeToolResult(ev.name, ev.result);
        }
        if (ev.name === "dispatch_field_unit") applyAgentDispatch(ev.result);
      } else if (ev.type === "revision") {
        const li = document.createElement("li");
        li.className = "fallback";
        li.textContent = `Numbers ${ev.unverified.join(", ")} not found in tool results; asking the model to rewrite.`;
        stepsEl.appendChild(li);
      } else if (ev.type === "answer") {
        const v = ev.verification || { total: 0, verified: 0, unverified: [] };
        answerEl.innerHTML = highlightNumbers(ev.text, v.unverified);
        const ok = v.unverified.length === 0;
        verifyEl.className = `agent-verify ${ok ? "ok" : "bad"}`;
        verifyEl.textContent =
          (ok ? `✓ ${v.verified}/${v.total} numbers traced to tool results` : `⚠ ${v.unverified.length} number(s) not traceable: ${v.unverified.join(", ")}`) +
          ` · ${AGENT_TIER_LABELS[ev.tier] || ev.tier} · ${(ev.latency_ms / 1000).toFixed(1)} s`;
      }
    }

    try {
      const res = await fetch("/api/agent/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, force_offline: Boolean(state.forceOffline) }),
        signal: run.signal,
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (line) onEvent(JSON.parse(line));
        }
      }
    } catch (err) {
      if (agentRun === run) answerEl.textContent = `Agent unavailable: ${err.message}`;
    } finally {
      if (agentRun === run) {
        agentRun = null;
        askBtn.disabled = false;
        refreshMetaModelsTelemetry();
      }
    }
  }

  // ---- Push-to-talk: microphone -> /api/voice/stream -> Muse Voice Transcribe -> agent ----
  // Captures mono PCM16 at 24 kHz (the rate Muse Voice expects), streams it live, shows the
  // words as they are recognised, and asks the final transcript like a typed question.
  const VOICE_WORKLET = `
    class PcmTap extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch) this.port.postMessage(ch.slice(0));
        return true;
      }
    }
    registerProcessor("pcm-tap", PcmTap);`;
  const VOICE_MAX_MS = 15000;
  let voice = null;

  function setVoiceStatus(text) {
    document.getElementById("agent-voice-status").textContent = text;
  }

  function setMicListening(on) {
    const btn = document.getElementById("agent-mic-btn");
    btn.classList.toggle("listening", on);
    btn.setAttribute("aria-pressed", String(on));
    btn.innerHTML = on ? "&#x25A0; Stop" : "&#x1F399; Talk";
  }

  async function startVoice() {
    if (state.forceOffline) {
      setVoiceStatus("Voice needs Meta Cloud: type the question");
      return;
    }
    const input = document.getElementById("agent-input");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      setVoiceStatus(`Microphone unavailable: ${err.name}`);
      return;
    }
    const ctx = new AudioContext({ sampleRate: 24000 });
    const workletUrl = URL.createObjectURL(new Blob([VOICE_WORKLET], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);
    const source = ctx.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(ctx, "pcm-tap");
    source.connect(tap);

    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/voice/stream`);
    ws.binaryType = "arraybuffer";
    const session = { ws, ctx, stream, ready: false, queued: [], startedAt: performance.now(), releasedAt: null };
    voice = session;

    tap.port.onmessage = function (ev) {
      const f32 = ev.data;
      const pcm = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      if (session.ready && ws.readyState === WebSocket.OPEN) ws.send(pcm.buffer);
      else session.queued.push(pcm.buffer); // speech before the Meta handshake completes
    };

    ws.onmessage = function (ev) {
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") {
        session.ready = true;
        session.queued.forEach((buf) => ws.send(buf));
        session.queued = [];
        setVoiceStatus("Listening… (Muse Voice)");
      } else if (msg.type === "transcript") {
        input.value = msg.text;
      } else if (msg.type === "done") {
        const after = session.releasedAt ? ((performance.now() - session.releasedAt) / 1000).toFixed(1) : "?";
        setVoiceStatus(`Muse Voice · ${(msg.audio_ms / 1000).toFixed(1)} s audio · text ${after} s after release`);
        if (msg.transcript && msg.transcript.length >= 3) {
          input.value = msg.transcript;
          askAgent(msg.transcript);
        } else {
          setVoiceStatus("Didn't catch that: try again or type");
        }
      } else if (msg.type === "error") {
        setVoiceStatus(msg.message);
      }
    };
    ws.onclose = function () {
      if (voice === session) stopVoice();
    };

    input.value = "";
    input.placeholder = "Listening…";
    setVoiceStatus("Connecting to Muse Voice…");
    setMicListening(true);
    session.timer = setTimeout(stopVoice, VOICE_MAX_MS);
  }

  function stopVoice() {
    const session = voice;
    if (!session) return;
    voice = null;
    clearTimeout(session.timer);
    session.releasedAt = performance.now();
    session.stream.getTracks().forEach((t) => t.stop());
    session.ctx.close();
    const finish = function () {
      session.queued.forEach((buf) => session.ws.send(buf));
      session.queued = [];
      session.ws.send(JSON.stringify({ type: "end" }));
    };
    if (session.ws.readyState === WebSocket.OPEN) finish();
    else if (session.ws.readyState === WebSocket.CONNECTING) session.ws.onopen = finish;
    document.getElementById("agent-input").placeholder = "e.g. Get an ambulance to the worst-hit block";
    setMicListening(false);
  }

  function bindAgentUi() {
    const form = document.getElementById("agent-form");
    const input = document.getElementById("agent-input");
    if (!form || !input) return;
    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      askAgent(input.value);
    });
    document.querySelectorAll(".agent-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        input.value = chip.getAttribute("data-q");
        askAgent(input.value);
      });
    });
    document.getElementById("agent-mic-btn").addEventListener("click", function () {
      if (voice) stopVoice();
      else startVoice();
    });
    document.getElementById("agent-close").addEventListener("click", function () {
      document.getElementById("agent-panel").classList.add("hidden");
    });
  }

  function switchTab(tabId) {
    // Clicking the open panel's rail button again returns to the map overview.
    const current = document.querySelector(".panel-tab.active");
    if (current && current.getAttribute("data-tab") === tabId && tabId !== "overview") tabId = "overview";
    document.querySelectorAll(".panel-tab").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-content").forEach(function (pane) {
      pane.classList.toggle("active", pane.id === `tab-${tabId}`);
    });
    document.getElementById("drawer").classList.toggle("hidden", tabId === "overview");
    relayoutMap();
  }

  // Popover menus in the top bar and map tools: one open at a time, closed by outside click or Esc.
  function bindMenus() {
    function closeAll(except) {
      document.querySelectorAll("[data-menu]").forEach(function (btn) {
        const menu = document.getElementById(btn.getAttribute("data-menu"));
        if (menu && menu !== except) {
          menu.classList.add("hidden");
          btn.setAttribute("aria-expanded", "false");
        }
      });
    }
    document.querySelectorAll("[data-menu]").forEach(function (btn) {
      btn.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const menu = document.getElementById(btn.getAttribute("data-menu"));
        const open = menu.classList.contains("hidden");
        closeAll(menu);
        menu.classList.toggle("hidden", !open);
        btn.setAttribute("aria-expanded", String(open));
        if (open && btn.id === "vlm-tier-badge") refreshMetaModelsTelemetry();
      });
    });
    document.querySelectorAll(".popover").forEach(function (menu) {
      menu.addEventListener("click", function (ev) {
        ev.stopPropagation();
        if (ev.target.closest(".menu-item")) closeAll(null);
      });
    });
    document.addEventListener("click", function () {
      closeAll(null);
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") closeAll(null);
    });
  }

  // KPI cards: every number comes from the API or the measured road layer.
  async function refreshSituation() {
    try {
      const res = await fetch(`/api/situation?vehicle_class=${encodeURIComponent(state.vehicleClass)}`);
      if (res.ok) {
        const d = await res.json();
        document.getElementById("kpi-reach-val").textContent = String(d.reachable);
        document.getElementById("kpi-reach-of").textContent = `/ ${d.high_severity_buildings_total}`;
        document.getElementById("kpi-reach-sub").textContent =
          `by ${VEHICLE_SHORT[state.vehicleClass].toLowerCase()} from the command post`;
      }
    } catch (err) {
      // KPI cards are informational; the map keeps working without them.
    }
    if (state.roadsGeojson) {
      const roads = new Map();
      state.roadsGeojson.features.forEach(function (f) {
        const p = f.properties || {};
        const key = [String(p.u), String(p.v)].sort().join("-") + `-${p.key || 0}`;
        roads.set(key, roads.get(key) || Boolean(p[`passable_${state.vehicleClass}`]));
      });
      const open = Array.from(roads.values()).filter(Boolean).length;
      document.getElementById("kpi-roads-label").textContent =
        `Road segments open to ${VEHICLE_SHORT[state.vehicleClass].toLowerCase()}s`;
      document.getElementById("kpi-roads-val").textContent = String(open);
      document.getElementById("kpi-roads-of").textContent = `/ ${roads.size}`;
      document.getElementById("roads-measured").textContent = String(roads.size);
    }
    const top = (state.lastRoiRecommendations || [])[0];
    document.getElementById("kpi-clear-val").textContent = top ? `+${top.high_severity_structures_unlocked}` : "0";
    document.getElementById("kpi-clear-sub").textContent = top
      ? `${top.debris_area_sqm} m² · ${top.road_name}`
      : "no single pile unlocks more";
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

    document.getElementById("inspector-close").addEventListener("click", closeInspector);

    const blackoutToggle = document.getElementById("toggle-comms-blackout");
    blackoutToggle.addEventListener("change", async function (ev) {
      state.forceOffline = Boolean(ev.target.checked);
      setNetworkStatus();
      if (state.forceOffline) setAiPill("tier_2");
      else fetchHealth();
      pushFieldDispatch({ force_offline: state.forceOffline });
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
      [state.rasterOverlay, state.rasterTiles].forEach(function (layer) {
        if (!layer) return;
        if (ev.target.checked) {
          layer.addTo(state.map);
        } else {
          state.map.removeLayer(layer);
        }
      });
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
        fitRoads();
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
    bindMenus();
    bindAgentUi();
    connectFieldLinkStatus();
    pollWarmup();
    await loadInitialLayers();
    await Promise.all([refreshRoute(), refreshClearanceRoi(), refreshMetaModelsTelemetry()]);
  });
})();
