/**
 * FirstLight Field View (`ui/field.js`).
 * Phone-first responder screen that follows the commander's live dispatch over local Wi-Fi
 * via Server-Sent Events (`/api/dispatch/stream`). Zero external CDN dependencies.
 */

(function () {
  "use strict";

  const VEHICLE_LABELS = {
    atv: "ATV / On foot · ≥1.2 m",
    ambulance: "Ambulance · ≥2.2 m",
    fire_engine: "Fire engine · ≥3.5 m",
  };

  const VEHICLE_SHORT = {
    atv: "ATV / on foot",
    ambulance: "ambulance",
    fire_engine: "fire engine",
  };

  const state = {
    map: null,
    roadsLayer: null,
    routeLayer: null,
    markersLayer: null,
    roadFeatures: [],
    lastVersion: null,
    lastAssignmentKey: null,
    lastReceivedAt: null,
    toastTimer: null,
  };

  function formatEta(seconds) {
    const s = Math.max(0, Math.round(Number(seconds) || 0));
    if (s < 60) return `${s} s`;
    const min = Math.floor(s / 60);
    const rem = s % 60;
    return rem >= 30 ? `${min + 1} min` : `${min} min`;
  }

  function formatDistance(meters) {
    const m = Math.max(0, Number(meters) || 0);
    if (m < 1000) return `${Math.round(m)} m`;
    return `${(m / 1000).toFixed(1)} km`;
  }

  function toastMessage(snapshot) {
    const detail = snapshot.reason_detail ? ` on ${snapshot.reason_detail}` : "";
    const vehicle = VEHICLE_SHORT[snapshot.vehicle_class] || snapshot.vehicle_class;
    switch (snapshot.reason) {
      case "debris_cleared":
        return { text: `Debris cleared${detail} — route updated`, warn: false };
      case "clearance_undone":
        return { text: `Clearance undone${detail} — route updated`, warn: true };
      case "clearances_reset":
        return { text: "All clearances reset — route updated", warn: true };
      case "vehicle_changed":
        return { text: `Now routing for ${vehicle}`, warn: false };
      case "route_reassigned":
        return { text: "New assignment from command", warn: false };
      case "comms_blackout_on":
        return { text: "Comms blackout — AI on the command laptop only", warn: true };
      case "comms_blackout_off":
        return { text: "Comms restored — Meta cloud AI available", warn: false };
      case "uplink_lost":
        return { text: "Internet lost — still live over local Wi-Fi", warn: true };
      case "uplink_restored":
        return { text: "Internet restored — Meta cloud AI back online", warn: false };
      default:
        return null;
    }
  }

  function initMap() {
    const map = L.map("field-map", {
      center: [29.9415, -85.4125],
      zoom: 15,
      minZoom: 13,
      maxZoom: 19,
      zoomControl: false,
      attributionControl: false,
    });
    state.map = map;
    state.roadsLayer = L.layerGroup().addTo(map);
    state.routeLayer = L.layerGroup().addTo(map);
    state.markersLayer = L.layerGroup().addTo(map);
  }

  async function loadBaseLayers() {
    const res = await fetch("/api/layers");
    if (!res.ok) throw new Error("layers unavailable");
    const data = await res.json();
    const [minLon, minLat, maxLon, maxLat] = data.bounds_wgs84;
    const bounds = L.latLngBounds([
      [minLat, minLon],
      [maxLat, maxLon],
    ]);
    L.imageOverlay("/api/raster-preview.png", bounds, {
      opacity: 0.6,
      interactive: false,
    }).addTo(state.map);
    state.map.setMaxBounds(bounds.pad(0.25));
    state.map.fitBounds(bounds);
    state.roadFeatures = (data.roads && data.roads.features) || [];
  }

  function renderRoads(snapshot) {
    state.roadsLayer.clearLayers();
    const blocked = new Set(snapshot.impassable_edge_ids || []);
    const water = new Set(snapshot.water_edge_ids || []);
    state.roadFeatures.forEach(function (feature) {
      const id = feature.properties && feature.properties.edge_id;
      const coords = (feature.geometry && feature.geometry.coordinates) || [];
      if (coords.length < 2) return;
      const latLngs = coords.map(function (c) {
        return [c[1], c[0]];
      });
      let style = { color: "#10b981", weight: 3, opacity: 0.55 };
      if (water.has(id)) {
        style = { color: "#a855f7", weight: 4, opacity: 0.9, dashArray: "6, 6" };
      } else if (blocked.has(id)) {
        style = { color: "#f59e0b", weight: 4, opacity: 0.9 };
      }
      L.polyline(latLngs, style).addTo(state.roadsLayer);
    });
  }

  function renderRoute(snapshot, fitToRoute, animateFit) {
    state.routeLayer.clearLayers();
    state.markersLayer.clearLayers();

    const route = snapshot.route || {};
    const coords = route.coordinates || [];
    let routeLine = null;
    if (route.reachable && coords.length >= 2) {
      const latLngs = coords.map(function (c) {
        return [c[1], c[0]];
      });
      L.polyline(latLngs, { color: "#0b1120", weight: 11, opacity: 0.85 }).addTo(state.routeLayer);
      routeLine = L.polyline(latLngs, { color: "#38bdf8", weight: 7, opacity: 1 }).addTo(
        state.routeLayer
      );
    }

    if (snapshot.origin) {
      L.circleMarker([snapshot.origin[1], snapshot.origin[0]], {
        radius: 9,
        color: "#ffffff",
        weight: 3,
        fillColor: "#10b981",
        fillOpacity: 1,
      }).addTo(state.markersLayer);
    }
    if (snapshot.destination) {
      L.circleMarker([snapshot.destination[1], snapshot.destination[0]], {
        radius: 9,
        color: "#ffffff",
        weight: 3,
        fillColor: "#ef4444",
        fillOpacity: 1,
      }).addTo(state.markersLayer);
    }

    if (fitToRoute) {
      const padding = {
        paddingTopLeft: [24, 90],
        paddingBottomRight: [24, 280],
        animate: Boolean(animateFit),
      };
      if (routeLine) {
        state.map.fitBounds(routeLine.getBounds(), padding);
      } else if (snapshot.origin && snapshot.destination) {
        state.map.fitBounds(
          L.latLngBounds([
            [snapshot.origin[1], snapshot.origin[0]],
            [snapshot.destination[1], snapshot.destination[0]],
          ]),
          padding
        );
      }
    }
  }

  function renderSheet(snapshot) {
    const route = snapshot.route || {};
    document.getElementById("vehicle-chip").textContent =
      VEHICLE_LABELS[snapshot.vehicle_class] || snapshot.vehicle_class;

    const statusEl = document.getElementById("route-status");
    const nextEl = document.getElementById("next-instruction");
    const steps = route.steps || [];

    if (route.reachable) {
      statusEl.textContent = "Route clear";
      statusEl.className = "route-status ok";
      document.getElementById("metric-eta").textContent = formatEta(route.estimated_transit_time_s);
      document.getElementById("metric-dist").textContent = formatDistance(route.total_distance_m);
      nextEl.className = "next-instruction";
      nextEl.textContent = steps.length ? steps[0].instruction : "You are at the destination";
    } else {
      statusEl.textContent = "No passable route";
      statusEl.className = "route-status blocked";
      document.getElementById("metric-eta").textContent = "—";
      document.getElementById("metric-dist").textContent = "—";
      nextEl.className = "next-instruction blocked";
      nextEl.textContent =
        `Destination is cut off for a ${VEHICLE_SHORT[snapshot.vehicle_class] || "vehicle"}. ` +
        "Hold position — command is clearing debris.";
    }
    document.getElementById("metric-avoided").textContent = String(route.blocked_segments_avoided || 0);

    document.getElementById("step-list").innerHTML = steps
      .map(function (st) {
        const drone = st.needs_drone_recon ? ` <span class="drone-flag">· verify by drone</span>` : "";
        return (
          `<li><span>${st.instruction}${drone}</span>` +
          `<span class="step-meta">${st.min_clear_width_m.toFixed(1)} m clear</span></li>`
        );
      })
      .join("");
  }

  function renderConnectivity(snapshot) {
    const conn = snapshot.connectivity || {};
    const aiPill = document.getElementById("ai-pill");
    const onEdge = conn.ai_tier === "tier_2_edge";
    aiPill.textContent = onEdge ? "AI: On-device Llama" : "AI: Meta Cloud";
    aiPill.classList.toggle("ai-edge", onEdge);
    aiPill.classList.toggle("ai-cloud", !onEdge);
    document
      .getElementById("uplink-banner")
      .classList.toggle("hidden", !(conn.uplink_online === false || conn.force_offline));
  }

  function showToast(message) {
    const el = document.getElementById("update-toast");
    el.textContent = message.text;
    el.classList.toggle("warn", Boolean(message.warn));
    el.classList.remove("hidden");
    // Restart the entrance animation for back-to-back updates.
    el.style.animation = "none";
    void el.offsetWidth;
    el.style.animation = "";
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(function () {
      el.classList.add("hidden");
    }, 4500);

    const sheet = document.getElementById("field-sheet");
    sheet.classList.remove("flash");
    void sheet.offsetWidth;
    sheet.classList.add("flash");

    if (navigator.vibrate) {
      navigator.vibrate([120, 60, 120]);
    }
  }

  function applySnapshot(snapshot) {
    const assignmentKey = JSON.stringify([
      snapshot.vehicle_class,
      snapshot.origin,
      snapshot.destination,
    ]);
    const isFirst = state.lastVersion === null;
    const versionChanged = !isFirst && snapshot.version !== state.lastVersion;
    const reassigned = assignmentKey !== state.lastAssignmentKey;

    renderRoads(snapshot);
    renderRoute(snapshot, isFirst || reassigned, !isFirst);
    renderSheet(snapshot);
    renderConnectivity(snapshot);

    if (versionChanged) {
      const message = toastMessage(snapshot);
      if (message) showToast(message);
    }

    state.lastVersion = snapshot.version;
    state.lastAssignmentKey = assignmentKey;
    state.lastReceivedAt = Date.now();
  }

  function setLinkStatus(live) {
    document.getElementById("link-dot").className = live ? "fdot live" : "fdot down";
    document.getElementById("link-text").textContent = live ? "Live" : "Reconnecting…";
  }

  function connectStream() {
    const source = new EventSource("/api/dispatch/stream?role=field");
    source.addEventListener("open", function () {
      setLinkStatus(true);
    });
    source.addEventListener("error", function () {
      // EventSource reconnects on its own (server sends `retry: 2000`).
      setLinkStatus(false);
    });
    source.addEventListener("dispatch", function (ev) {
      setLinkStatus(true);
      try {
        applySnapshot(JSON.parse(ev.data));
      } catch (err) {
        console.error("Bad dispatch payload", err);
      }
    });
  }

  function tickUpdatedAgo() {
    const el = document.getElementById("updated-ago");
    if (!state.lastReceivedAt) {
      el.textContent = "Waiting for command…";
      return;
    }
    const secs = Math.round((Date.now() - state.lastReceivedAt) / 1000);
    el.textContent = secs < 3 ? "Updated just now" : `Updated ${secs} s ago`;
  }

  function bindSheetToggle() {
    const sheet = document.getElementById("field-sheet");
    const handle = document.getElementById("sheet-handle");
    handle.addEventListener("click", function () {
      const expanded = sheet.classList.toggle("expanded");
      handle.setAttribute("aria-expanded", String(expanded));
    });
  }

  document.addEventListener("DOMContentLoaded", async function () {
    initMap();
    bindSheetToggle();
    setInterval(tickUpdatedAgo, 1000);
    try {
      await loadBaseLayers();
    } catch (err) {
      setLinkStatus(false);
    }
    connectStream();
  });
})();
