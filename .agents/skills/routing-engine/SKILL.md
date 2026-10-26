---
name: routing-engine
description: Builds multi-class vehicle routing, network graph weighting, and Choke-Point Clearance ROI ranking using NetworkX.
---

# Routing Engine Guidelines
- **Vehicle Clearance Thresholds:**
  - ATV / On Foot: Minimum clear width > 1.2 meters
  - Ambulance / Pickup: Minimum clear width > 2.2 meters
  - Fire Engine: Minimum clear width > 3.5 meters
- **Impassable Rules:** Any road segment overlapping a SAM 3.1 water mask must have its weight set to `infinity` (unpassable), regardless of width.
- **Clearance ROI:** Compute the ratio of high-severity building structures unlocked per square meter of debris cleared at bottleneck nodes.

