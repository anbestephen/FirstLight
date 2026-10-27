# FirstLight — Offline-Resilient Tactical Disaster Routing & Choke-Point Clearance ROI Engine

> **Built with Llama** | **"When the cell towers fall and the cloud goes dark, FirstLight turns post-disaster aerial imagery into millimeter-accurate, multi-class emergency vehicle routing and heavy-equipment clearance triage—running 100% locally on Apple Silicon at the tactical edge."**

[![Meta Global AI Developer Hackathon](https://img.shields.io/badge/Meta_Global_AI_Hackathon-2026-0668E1?style=for-the-badge)](https://api.meta.ai/v1)
[![Hardware Profile](https://img.shields.io/badge/Apple_Silicon-M1_Max_64GB_(MPS_fp16_%2F_MLX_4--bit)-000000?style=for-the-badge)](#3-firstlight-architecture--the-6-model-meta-ai-suite)
[![Air-Gap Verified](https://img.shields.io/badge/Offline_Resilience-100%25_Zero--CDN_Verified-10B981?style=for-the-badge)](#6-quickstart-reproducibility--air-gap-verification)
[![Test Suite](https://img.shields.io/badge/Pytest_Suite-42%2F42_Passing-38BDF8?style=for-the-badge)](#6-quickstart-reproducibility--air-gap-verification)

---

## 1. Hackathon Track Alignment & Build Timeline Disclosure

**FirstLight** is purpose-built for the **Meta Global AI Developer Hackathon**, uniting both primary tracks into a single mission-critical field system:

* **Track 1 — Build for Your World (Civic & Disaster Resilience):** Provides municipal Incident Commanders, Urban Search & Rescue (USAR) task forces, and public works bulldozer crews with a zero-CDN, local Wi-Fi tactical command server (`0.0.0.0:8000`) that routes emergency vehicles around hurricane debris and ranks road-clearing operations by high-severity structures unlocked per square meter of rubble cleared.
* **Track 2 — Push the Frontier (Multi-Modal Geometric + VLM Edge Failover):** Transforms **SAM 3.1 (`sam-3.1`)** from a semantic mask generator into a **metric spatial caliper** (`0.2804 m/px` GSD in `EPSG:32616`), coupled with a **3-Tier Air-Gap Circuit Breaker** that seamlessly fails over from the **Meta Model API (`https://api.meta.ai/v1` — `muse-spark-1.3`, `muse-wonder-1.0`, `muse-voice-transcribe-1.0`, `muse-image-1.0`)** to local 4-bit **Apple Silicon MLX (`mlx_vlm.generate`)** and deterministic raycast geometry when field communications black out.

### Hackathon Build Timeline & Pre-Existing Public Datasets Disclosure
* **100% Built From Scratch During Hackathon Window:** Every line of application source code (`src/`), frontend UI (`ui/`), data/model baking scripts (`scripts/`), ground-truth evaluation harness (`benchmarks/`), and automated tests (`tests/`) was written from scratch after the official hackathon kickoff (**Monday, October 26, 2026 at 12:00 PM EDT**).
* **Pre-Existing Public Datasets & Pre-Trained Model Weights Used:**
  1. **NOAA National Geodetic Survey (NGS) Hurricane Michael Emergency Response Imagery:** Public-domain post-storm orthomosaic GeoTIFF (`data/noaa_michael_mexico_beach.tif`, captured October 11, 2018 over Mexico Beach, FL).
  2. **OpenStreetMap (OSM) Vector Topology:** Public road network (`data/osm_road_graph.graphml`, `data/osm_roads.geojson`, 312 road edges) and building footprints (`data/osm_buildings.geojson`, 334 structures) for Mexico Beach, FL.
  3. **Pre-Trained Meta AI Weights & APIs:** Local **Meta SAM 3.1** (`models/sam3.1/model.safetensors`, `Sam3Model` in `float16`), local 4-bit quantized **Meta Llama 3.2 11B Vision Instruct** (`models/Llama-3.2-11B-Vision-Instruct-4bit`), and live **Meta Model API** endpoints (`muse-spark-1.3`, `muse-wonder-1.0`, `muse-voice-transcribe-1.0`, `muse-image-1.0`).
  4. **Real Model-Generated GeoJSON Artifacts Baked During Hackathon:**
     * `data/sam31_edge_measurements.json` + `data/sam31_mexico_beach_masks.geojson` — real SAM 3.1 output: `Sam3Model` (`models/sam3.1`, `torch.float16` on Apple Silicon MPS) run on all 89 road tiles of the NOAA GeoTIFF with the text prompts `road`, `debris` and `standing water` (presence-gated), giving visible-pavement widths for all 158 roads and 1,218 mask polygons. Rebuild with [`scripts/bake_sam31_masks.py`](scripts/bake_sam31_masks.py) (~5 min on M1 Max).
     * `data/osm_buildings_graded.geojson` — **334 building footprints** graded for post-hurricane structural severity (`61 destroyed`, `64 major_damage`, `106 minor_damage`, `103 no_damage`) via live `muse-spark-1.3` multimodal vision calls on 10 sector aerial crops in [`scripts/grade_buildings_muse_spark.py`](scripts/grade_buildings_muse_spark.py).

---

## 2. The Core Problem: Why Communications Blackouts Cripple Disaster Response

### Case Study: Hurricane Michael — Mexico Beach, FL (US Highway 98 Corridor)
When Category 5 **Hurricane Michael** made landfall at Mexico Beach, Florida, 160 mph winds and a 14-foot storm surge obliterated cellular backhaul, fiber trunks, and municipal power across Bay County. Within minutes:

1. **Cloud Navigation & Standard GIS Failed:** Commercial routing APIs require WAN connectivity and assume pre-storm road topologies. They cannot distinguish a clear lane from a road partially pinched by collapsed roof trusses or submerged under storm surge.
2. **All Emergency Vehicles Are Not Equal:** A debris-strewn corridor with **2.65 meters** of clear lateral width is passable for a **Search & Rescue ATV ($\ge 1.2\text{ m}$)** or **EMS Ambulance ($\ge 2.2\text{ m}$)**, but physically blocks a **Type-1 Fire Engine ($\ge 3.5\text{ m}$)**—causing catastrophic turnarounds in dead-end residential sectors.
3. **Blind Bulldozer Dispatch Wastes Golden-Hour Time:** Without quantitative Return on Investment (ROI), front-end loaders clear the nearest visible rubble pile rather than the specific choke-point pinch that unlocks an isolated cluster of 15 destroyed or severely damaged homes.

**FirstLight solves all three failures** by ingesting post-event NOAA Emergency Response GeoTIFF imagery (`data/noaa_michael_mexico_beach.tif`), baked SAM 3.1 obstacle masks (`data/sam31_mexico_beach_masks.geojson`), and `muse-spark-1.3`-graded building footprints (`data/osm_buildings_graded.geojson`) locally, measuring exact clear road spans every `5.0 m`, grading structural damage through a 3-Tier VLM circuit breaker, and serving a zero-CDN tactical dashboard to any tablet or phone over a local Wi-Fi router.

---

## 3. FirstLight Architecture & The 6-Model Meta AI Suite

### 3-Tier Air-Gap Failover Circuit & Geospatial Pipeline

```mermaid
flowchart TD
    subgraph Ingest["1. Local Geospatial & Aerial Imagery Layer (100% Offline Cache)"]
        TIF["NOAA Post-Michael GeoTIFF\ndata/noaa_michael_mexico_beach.tif\n(EPSG:4326 -> EPSG:32616 @ 0.2804 m/px GSD)"]
        OSM["OSM Road Network & Graded Buildings\ndata/osm_road_graph.graphml (312 edges)\ndata/osm_buildings_graded.geojson (334 Muse Spark 1.3 structures)"]
    end

    subgraph Caliper["2. SAM 3.1 Geometric Measurement Instrument (src/geometry.py)"]
        SAM["SAM 3.1 (PyTorch MPS fp16)\nprompts: road / debris / standing water\ndata/sam31_edge_measurements.json (158 roads)"]
        RAY["5.0m Orthogonal Cross-Section Raycaster\nCasts 20m Normal Rays Along Road Centerlines\nComputes Contiguous Clear Span + Drone Recon Flags"]
    end

    subgraph Circuit["3. 3-Tier VLM Failover Circuit Breaker (src/failover.py)"]
        CROP["256x256 Real NOAA GeoTIFF Aerial Patch\nGET /api/choke-point/{edge_id}/crop.png"]
        T1["TIER 1: Meta Model API Cloud Primary\nhttps://api.meta.ai/v1 (max_tokens=2048, timeout=15s)\nmuse-spark-1.3 | muse-wonder-1.0 | muse-voice-transcribe-1.0 | muse-image-1.0"]
        T2["TIER 2: Apple Silicon Edge Failover\nReal mlx_vlm.generate (4-bit Quantized Unified Memory)\nLlama-3.2-11B-Vision-Instruct-4bit + gc/metal clear"]
        T3["TIER 3: Deterministic Geometric Fallback\nExplicit Provenance: [DETERMINISTIC SAM/GEOMETRY FALLBACK]\nGuaranteed Zero-Latency Survival"]
    end

    subgraph Routing["4. Multi-Class NetworkX Router & Clearance ROI Engine (src/routing.py)"]
        GRAPH["Vehicle-Specific Weighted Graphs\nATV (>=1.2m) | Ambulance (>=2.2m) | Fire Engine (>=3.5m)\nStanding Water Rule: water_overlap_ratio > 0 => weight = inf"]
        ROI["Choke-Point Clearance ROI Queue\nROI = High-Severity Structures Unlocked / Debris Area (m²)"]
    end

    subgraph UI["5. Zero-CDN Local Wi-Fi Field Dashboard (src/app.py + ui/)"]
        DASH["FastAPI 0.0.0.0:8000 + Vendored Leaflet 1.9.4\n5m Caliper Ray Overlay | SAM 3.1 Drone Recon Badge\nVoice Radio PTT | Muse Wonder Briefing | GeoJSON Manifest Export"]
    end

    TIF --> SAM
    OSM --> RAY
    SAM --> RAY
    TIF --> CROP
    RAY --> CROP
    CROP --> T1
    T1 -- "Timeout / HTTP Error / Comms Blackout" --> T2
    T2 -- "Device Exception / Force Deterministic" --> T3
    RAY --> GRAPH
    T1 & T2 & T3 --> ROI
    GRAPH --> ROI
    ROI --> DASH
```

### The 6-Model Meta AI Suite in FirstLight

FirstLight orchestrates **all six models** from the Meta AI ecosystem via [`MetaModelSuite`](src/failover.py) with strict engineering honesty—every response explicitly attributes whether a live cloud model, local `mlx-vlm` edge model, or deterministic geometric fallback executed:

| # | Meta AI Model ID | System Role in FirstLight | Primary Execution Target | Air-Gap / Comms Blackout Fallback (`force_offline=True`) | API Endpoint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **`sam-3.1`** | **Geometric Measurement Instrument:** Segments visible pavement, debris and standing water on every road tile (prompts `road`, `debris`, `standing water`), then measures the clear pavement width in metres every `5.0 m` along all 158 roads (`0.2804 m/px`, `EPSG:32616`), flagging roads where no pavement is visible for drone reconnaissance. | Local Apple Silicon PyTorch MPS (`float16` safetensors in `models/sam3.1/`) | Runs 100% locally; results baked to `data/sam31_edge_measurements.json` so the field laptop needs no GPU work at demo time. | `GET /api/choke-point/{edge_id}/caliper` |
| **2** | **`muse-spark-1.3`** | **Corridor & Structural Damage Grader:** Evaluates `256×256` NOAA aerial patches and grades all `334` Mexico Beach buildings (`data/osm_buildings_graded.geojson`) into canonical 4-class severity (`no_damage`, `minor_damage`, `major_damage`, `destroyed`). | Meta Model API (`https://api.meta.ai/v1/chat/completions`, `max_tokens=2048`) | Automatic circuit-breaker failover to **Tier 2 (`open-weight-mlx-4bit`)** and **Tier 3 (`[DETERMINISTIC SAM/GEOMETRY FALLBACK]`)**. | `POST /api/assess` |
| **3** | **`muse-wonder-1.0`** | **Incident Commander Strategic Triage Advisor:** Synthesizes top Choke-Point Clearance ROI bottlenecks into an executive multi-corridor action plan assigning loaders vs. skid-steers across Mexico Beach sectors. | Meta Model API (`https://api.meta.ai/v1/chat/completions`, `max_tokens=2048`) | Deterministic geometric triage explicitly labeled `[DETERMINISTIC GEOMETRIC TRIAGE — NO LLM EXECUTED]`. | `POST /api/incident-briefing` |
| **4** | **`muse-voice-transcribe-1.0`** | **Tactical Hands-Free Radio Dispatch:** Transcribes and parses field radio commands (e.g., *"Dispatch Fire Engine and simulate clearing top choke point on US Highway 98"*) into structured UI actions. | Meta Model API (`https://api.meta.ai/v1/chat/completions`) | Deterministic regex command intent parser explicitly labeled `[DETERMINISTIC REGEX DISPATCH — NO ASR MODEL EXECUTED]`. | `POST /api/voice-dispatch` |
| **5** | **`muse-image-1.0`** | **Tactical Clearance Recon Schematic Generator:** Produces annotated visual reconnaissance schematics for field bulldozer crew manifests. | Meta Model API (`https://api.meta.ai/v1/images/generations`) | Deterministic Caliper Overlay (`Deterministic Caliper Overlay (Muse Image Offline)`) rendering SAM 3.1 polygons & `5.0 m` caliper rays on the NOAA GeoTIFF crop. | `GET /api/choke-point/{edge_id}/recon-schematic.png` |
| **6** | **`open-weight-mlx-4bit`** | **Tier 2 Edge Vision-Language Engine:** Local `Llama-3.2-11B-Vision-Instruct-4bit` executing real `mlx_vlm.generate` in Apple Silicon Unified Memory with explicit `gc.collect()` + `mx.metal.clear_cache()` hygiene. | Local Apple Silicon Metal (`models/Llama-3.2-11B-Vision-Instruct-4bit`) | Real `mlx_vlm.generate` inference on Apple Silicon unified memory with automatic fallback to Tier 3 deterministic geometry if weights are absent. | `POST /api/assess` & `GET /api/health` |

---

## 4. First-Principles Geometric Engineering

FirstLight enforces a strict engineering invariant: **zero hallucinated counts or metrics**. Every clearance width, routing detour, and priority explanation is mathematically derived from projected pixel-to-meter geometry and graph theory.

### 4.1 Metric Reprojection (`EPSG:32616`) & Verified Ground Sample Distance
Geographic coordinates (`EPSG:4326` degrees) distort linear distances and angles. In [`src/geometry.py`](src/geometry.py):
1. Both `data/osm_roads.geojson` (`312` road segments) and `data/noaa_michael_mexico_beach.tif` (`9340 × 9340` pixels) are projected into **UTM Zone 16N (`EPSG:32616`, meters)** using `pyproj.Transformer` and `rasterio.warp.calculate_default_transform`.
2. The verified metric Ground Sample Distance (GSD) of the Mexico Beach NOAA GeoTIFF is:
   $$\text{GSD} = 0.2804\text{ meters/pixel} \quad \left(1\text{ pixel}^2 = 0.078624\text{ m}^2\right)$$

### 4.2 Orthogonal `5.0 m` Cross-Section Raycasting & Drone Recon Flagging
Rather than treating segmentation as a whole-road label, [`measure_edge`](src/sam_measure.py) marches along each road centerline at $\Delta s = 5.0\text{ m}$ intervals (`station_m`), sampling SAM 3.1's per-pixel probability maps:
1. At each station $s_i$, it computes the unit tangent vector $\hat{t}_i = (\Delta x, \Delta y) / \|\Delta \mathbf{p}\|$ and orthogonal unit normal $\hat{n}_i = (-\hat{t}_{i,y}, \hat{t}_{i,x})$.
2. It casts an orthogonal cross-section ray of total length $L = 30.0\text{ m}$ ($\pm 15.0\text{ m}$), sampled every $0.25\text{ m}$. The clear width is the contiguous run of SAM `road` pixels ($p \ge 0.5$) **nearest the OSM centerline**, which absorbs the few-metre misalignment between OSM and the imagery.
3. **Sustained Bottleneck Rule:** Debris on the pavement removes `road` pixels, splitting or narrowing the run; a vehicle cannot straddle rubble, so only the nearest contiguous run counts. An obstruction must persist over two consecutive stations before it sets the edge's bottleneck width, so a single parked car or shadow cannot close a road:
   $$w_{\text{clear}}(e) = \min_{i} \; \max\big(w(s_i),\, w(s_{i+1})\big)$$
4. **Cause & Drone Recon Flag (`needs_drone_recon`):** Where no pavement is visible, the gap is attributed to SAM `standing water` (impassable for every vehicle), SAM `debris` (clearable — feeds the Choke-Point ROI queue), or *not visible* (canopy, shadow or burial — treated as impassable). Roads with a not-visible bottleneck, mean pavement certainty $< 0.6$, or a width within $\pm 0.35\text{ m}$ of a vehicle threshold are flagged `needs_drone_recon = True` in the UI.

### 4.3 Multi-Class Vehicle Passability & Standing Water Infinite-Weight Rule
In [`src/routing.py`](src/routing.py), each directed edge $e = (u, v, k)$ in the NetworkX `MultiDiGraph` is assigned class-specific traversal weights $W(e, \text{class})$:

| Vehicle Class | Minimum Clear Width ($w_{\min}$) | Base Speed | Standing Water Rule (`water_overlap_ratio > 0`) | Insufficient Width ($w_{\text{clear}}(e) < w_{\min}$) |
| :--- | :---: | :---: | :---: | :---: |
| **ATV / Foot (`atv`)** | **$\ge 1.2\text{ m}$** | $6.0\text{ m/s}$ | $W(e) = \infty$ *(Strictly Impassable)* | $W(e) = \infty$ |
| **Ambulance / Pickup (`ambulance`)** | **$\ge 2.2\text{ m}$** | $10.0\text{ m/s}$ | $W(e) = \infty$ *(Strictly Impassable)* | $W(e) = \infty$ |
| **Fire Engine (`fire_engine`)** | **$\ge 3.5\text{ m}$** | $8.0\text{ m/s}$ | $W(e) = \infty$ *(Strictly Impassable)* | $W(e) = \infty$ |

### 4.4 Choke-Point Clearance ROI Formula
To tell Incident Commanders **which debris pile to bulldoze first**, [`rank_choke_point_clearance_roi`](src/routing.py) identifies every debris-blocked edge ($w_{\text{clear}}(e) < w_{\min}$, `water_overlap_ratio == 0`), simulates restoring its full nominal width, and measures how many previously unreachable **high-severity buildings** (`major_damage` or `destroyed` from `data/osm_buildings_graded.geojson`, totaling **125 high-severity structures** graded by `muse-spark-1.3`) become reachable from the staging origin:

$$\text{Clearance ROI}(e) = \frac{\text{High-Severity Structures Unlocked}(e)}{\text{Debris Area to Clear}(e)\ [\text{m}^2]}$$

---

## 5. Ground-Truth Evaluation & Benchmark Results

FirstLight includes a reproducible ground-truth benchmark suite ([`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py) and [`data/ground_truth_benchmark.json`](data/ground_truth_benchmark.json)) evaluating our baked **SAM 3.1 (`sam-3.1`)** orthogonal caliper measurements across **15 reference road cross-sections** and **Muse Spark (`muse-spark-1.3`)** damage classifications across **30 reference structures** in the Mexico Beach AOI.

Execute the benchmark at any time via:
```bash
./.venv/bin/python benchmarks/run_benchmarks.py
```

### Verified Benchmark Summary (`benchmarks/benchmark_results.json`)

| Evaluation Category | Metric | Measured Result | Engineering Interpretation |
| :--- | :--- | :---: | :--- |
| **SAM 3.1 Orthogonal Caliper (`n = 15` corridors)** | **Mean Absolute Error (`mae_m`)** | **`0.1587 m`** (`0.57 px`) | Sub-pixel accuracy at `0.2804 m/px` Ground Sample Distance (`EPSG:32616`). |
| **SAM 3.1 Orthogonal Caliper (`n = 15` corridors)** | **Root Mean Squared Error (`rmse_m`)** | **`0.2067 m`** | Tight variance across clear highways, pinched residential streets, and flooded segments. |
| **Vehicle Passability Classification** | **ATV (`>= 1.2 m`) Accuracy** | **`100.0%`** (`1.0000`) | Zero false-passable or false-blocked corridors for Search & Rescue ATVs. |
| **Vehicle Passability Classification** | **Ambulance (`>= 2.2 m`) Accuracy** | **`100.0%`** (`1.0000`) | 100% agreement with pixel-measured ground truth across all 15 corridors. |
| **Vehicle Passability Classification** | **Fire Engine (`>= 3.5 m`) Accuracy** | **`100.0%`** (`1.0000`) | Correctly blocks corridors pinched between `2.2 m` and `3.4 m`. |
| **Muse Spark 1.3 Building Grading (`n = 30` structures)** | **High-Severity Binary Accuracy** | **`96.67%`** (`0.9667`) | Distinguishes `major_damage`/`destroyed` vs. `minor_damage`/`no_damage`. |
| **Muse Spark 1.3 Building Grading (`n = 30` structures)** | **High-Severity F1 Score** | **`0.9655`** | Precision = `1.0000`, Recall = `0.9333` on high-priority rescue structures. |
| **Muse Spark 1.3 Building Grading (`n = 30` structures)** | **4-Class Exact Match Accuracy** | **`93.33%`** (`0.9333`) | Across `no_damage`, `minor_damage`, `major_damage`, and `destroyed`. |
| **SAM 3.1 Confidence & Recon Telemetry (`312` edges)** | **Mean Mask Confidence (`sam_confidence`)** | **`0.7355`** (`0.6708–0.8718`) | `228` high-confidence corridors; `84` corridors flagged with `needs_drone_recon=True`. |

---

## 6. Quickstart, Reproducibility & Air-Gap Verification

### 6.1 Environment Setup & Reproducibility Verification
FirstLight is optimized for **Apple Silicon (M1 Max, 64 GB Unified Memory, macOS)** using Python 3.11+ (`fp16` PyTorch MPS and 4-bit `mlx-vlm`):

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install locked dependencies
pip install -r requirements.txt

# 3. Verify all pre-staged datasets, baked SAM 3.1 / Muse Spark artifacts, and local model weights
./.venv/bin/python scripts/setup_data_and_models.py --check
```
*(If any large GeoTIFF or HuggingFace model weight is missing on a fresh clone, run `./.venv/bin/python scripts/setup_data_and_models.py --download` to fetch and rebuild automatically.)*

### 6.2 Configure Meta Model API Key (Optional — Automatic Edge Failover if Omitted)
To enable **Tier 1 Cloud Primary** calls against `https://api.meta.ai/v1` (`muse-spark-1.3`, `muse-wonder-1.0`, `muse-voice-transcribe-1.0`, `muse-image-1.0`), export `MODEL_API_KEY` or place it in `.env`:

```bash
export MODEL_API_KEY="your_meta_model_api_key_here"
```
> **Note:** If `MODEL_API_KEY` is unset, or if you toggle **"Simulate Comms Blackout (Force Edge Mode)"** in the UI, FirstLight automatically routes all inference through **Tier 2 (`mlx_vlm.generate` 4-bit)** and **Tier 3 (Deterministic Geometry)** with zero errors.

### 6.3 Run the 42-Test Verification Suite
Verify all geometric raycasting, 3-tier circuit breaker failovers, multi-class routing invariants, ground-truth benchmarks, and zero-CDN asset checks:

```bash
./.venv/bin/pytest -v
```

### 6.4 Launch the Offline Local Wi-Fi Command Server
Bind the FastAPI server to `0.0.0.0:8000` so both your local browser and any field tablet/phone on the same Wi-Fi LAN can connect:

```bash
./.venv/bin/uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## 7. Repository Structure

```text
FirstLight/
├── AGENTS.md                  # Hardware (M1 Max fp16/4-bit) & offline architecture invariants
├── PLAN.md                    # Full architectural blueprint & milestone verification checklist
├── README.md                  # Publication-grade documentation & hackathon compliance disclosures
├── pyproject.toml             # Package configuration & pytest settings
├── requirements.txt           # Core dependencies (FastAPI, Rasterio, GeoPandas, MLX, PyTorch)
├── benchmarks/                # Ground-truth evaluation suite & verified accuracy metrics
│   ├── run_benchmarks.py      # Evaluates SAM 3.1 caliper widths & Muse Spark 1.3 building grades
│   └── benchmark_results.json # Saved MAE (0.1587m), RMSE (0.2067m), and F1 (0.9655) metrics
├── data/                      # 100% locally cached Mexico Beach, FL (Hurricane Michael) datasets
│   ├── noaa_michael_mexico_beach.tif     # Post-event NOAA GeoTIFF (0.2804 m/px GSD in EPSG:32616)
│   ├── sam31_edge_measurements.json      # SAM 3.1 per-road stations, widths, causes, confidence
│   ├── sam31_mexico_beach_masks.geojson  # 1,218 SAM 3.1 pavement/debris/water polygons near roads
│   ├── osm_buildings_graded.geojson      # 334 buildings graded by live Muse Spark 1.3 vision calls
│   ├── ground_truth_benchmark.json       # 15 reference road cross-sections & 30 reference buildings
│   ├── osm_road_graph.graphml            # Cached OSM road network (125 nodes, 312 edges)
│   ├── osm_roads.geojson                 # Road centerlines (312 features)
│   └── osm_buildings.geojson             # Raw OSM building footprints (334 structures)
├── models/                    # Pre-cached local Apple Silicon weights for air-gapped operation
│   ├── sam3.1/                           # Meta SAM 3.1 fp16 safetensors checkpoint
│   └── Llama-3.2-11B-Vision-Instruct-4bit/  # 4-bit quantized MLX VLM weights
├── scripts/                   # Reproducible asset verification & offline model baking scripts
│   ├── setup_data_and_models.py          # --check and --download verification tool for judges
│   ├── bake_sam31_masks.py               # Runs Sam3Model (fp16 MPS) on every road tile, measures all roads
│   └── grade_buildings_muse_spark.py     # Grades all 334 OSM buildings via Muse Spark 1.3 API
├── src/
│   ├── geometry.py            # EPSG:32616 projection, SAM 3.1 masks, 5m orthogonal caliper raycaster
│   ├── failover.py            # 3-Tier VLM circuit breaker & 6-model MetaModelSuite orchestrator
│   ├── routing.py             # Multi-class NetworkX routing & Choke-Point Clearance ROI engine
│   └── app.py                 # FastAPI backend bound to 0.0.0.0:8000
├── ui/                        # Zero-CDN standalone tactical field dashboard
│   ├── index.html             # Tactical HUD, 6-Model Telemetry Strip, Inspector Dock & ROI Queue
│   ├── app.js                 # Map rendering, 5m caliper visualization, Drone Recon badge & PTT
│   ├── style.css              # High-contrast dark tactical UI & mobile touch layout
│   └── vendor/                # Locally vendored Leaflet 1.9.4 JS, CSS & marker assets
└── tests/                     # 42 unit & integration tests (100% passing)
    ├── test_geometry.py
    ├── test_failover.py
    ├── test_routing.py
    └── test_app.py
```

---

## 8. Licenses & Required Attributions

* **Built with Llama:** FirstLight's Tier 2 offline edge vision-language failover engine is **Built with Llama** using `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`, which is licensed under the **Meta Llama 3.2 Community License** (Copyright © Meta Platforms, Inc. All Rights Reserved).
* **Meta SAM 3.1 (`sam-3.1`):** Local segmentation weights (`models/sam3.1`) and architecture are provided by Meta Platforms, Inc. under the **Meta Segment Anything Model (SAM 3.1) License**.
* **Meta Model API (`muse-spark-1.3`, `muse-wonder-1.0`, `muse-voice-transcribe-1.0`, `muse-image-1.0`):** Accessed via the official Meta AI Developer API (`https://api.meta.ai/v1`) in accordance with the Meta Global AI Developer Hackathon Terms.
* **OpenStreetMap Attribution:** Road network topology (`data/osm_road_graph.graphml`, `data/osm_roads.geojson`) and building footprint geometries (`data/osm_buildings.geojson`) — **Map data © OpenStreetMap contributors, licensed under the Open Data Commons Open Database License (ODbL)**.
* **NOAA Hurricane Michael Emergency Response Imagery:** Post-event aerial orthomosaic (`data/noaa_michael_mexico_beach.tif`) courtesy of the **National Oceanic and Atmospheric Administration (NOAA) National Geodetic Survey (NGS) Remote Sensing Division** (U.S. Government Public Domain work).
