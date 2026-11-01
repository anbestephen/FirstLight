// Street-name labels for the dashboard and field maps (offline: names come from the OSM road layer).
// Each named street is labelled along its longest segments, at least LABEL_SPACING_M apart, rotated
// to the road and shown only from MIN_ZOOM so the overview stays readable.
(function () {
  "use strict";

  const MIN_ZOOM = 16;
  const MIN_SEGMENT_M = 40;
  const LABEL_SPACING_M = 250;
  const M_PER_DEG = 111320;

  function streetName(raw) {
    const name = String(raw || "").split(",")[0].trim();
    return name && !name.startsWith("unnamed road") ? name : null;
  }

  function planar(coords) {
    const cosLat = Math.cos((coords[0][1] * Math.PI) / 180);
    return coords.map(function (c) {
      return [c[0] * M_PER_DEG * cosLat, c[1] * M_PER_DEG];
    });
  }

  // Midpoint by length along the line, and the screen angle of the piece it falls on.
  function midpointAndAngle(coords) {
    const pts = planar(coords);
    const lengths = [];
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
      const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      lengths.push(d);
      total += d;
    }
    let walked = 0;
    for (let i = 1; i < pts.length; i++) {
      const d = lengths[i - 1];
      if (walked + d >= total / 2 && d > 0) {
        const t = (total / 2 - walked) / d;
        const a = coords[i - 1];
        const b = coords[i];
        let angle = (Math.atan2(-(pts[i][1] - pts[i - 1][1]), pts[i][0] - pts[i - 1][0]) * 180) / Math.PI;
        if (angle > 90) angle -= 180;
        if (angle < -90) angle += 180;
        return { latLng: [a[1] + t * (b[1] - a[1]), a[0] + t * (b[0] - a[0])], angle: angle, length: total };
      }
      walked += d;
    }
    return null;
  }

  function placements(features) {
    const candidates = [];
    features.forEach(function (feature) {
      const name = streetName(feature.properties && feature.properties.name);
      const coords = (feature.geometry && feature.geometry.coordinates) || [];
      if (!name || coords.length < 2) return;
      const mid = midpointAndAngle(coords);
      if (mid && mid.length >= MIN_SEGMENT_M) candidates.push(Object.assign({ name: name }, mid));
    });
    candidates.sort(function (a, b) {
      return b.length - a.length;
    });
    const placed = [];
    candidates.forEach(function (c) {
      const tooClose = placed.some(function (p) {
        const dLat = (p.latLng[0] - c.latLng[0]) * M_PER_DEG;
        const dLon = (p.latLng[1] - c.latLng[1]) * M_PER_DEG * Math.cos((c.latLng[0] * Math.PI) / 180);
        return p.name === c.name && Math.hypot(dLat, dLon) < LABEL_SPACING_M;
      });
      if (!tooClose) placed.push(c);
    });
    return placed;
  }

  function escapeText(s) {
    return s.replace(/[&<>"]/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch];
    });
  }

  // Returns a layer group that adds itself to the map at MIN_ZOOM and above.
  function create(map, features) {
    if (!map.getPane("streetLabels")) {
      const pane = map.createPane("streetLabels");
      pane.style.zIndex = 450; // above roads and routes, below markers and tooltips
      pane.style.pointerEvents = "none";
    }
    const group = L.layerGroup();
    placements(features).forEach(function (p) {
      L.marker(p.latLng, {
        pane: "streetLabels",
        interactive: false,
        keyboard: false,
        icon: L.divIcon({
          className: "street-label",
          html: `<span style="transform: translate(-50%, -50%) rotate(${p.angle.toFixed(1)}deg)">${escapeText(p.name)}</span>`,
          iconSize: [0, 0],
        }),
      }).addTo(group);
    });
    function sync() {
      const show = map.getZoom() >= MIN_ZOOM;
      if (show && !map.hasLayer(group)) group.addTo(map);
      if (!show && map.hasLayer(group)) map.removeLayer(group);
    }
    map.on("zoomend", sync);
    sync();
    return group;
  }

  window.FirstLightStreetLabels = { create: create };
})();
