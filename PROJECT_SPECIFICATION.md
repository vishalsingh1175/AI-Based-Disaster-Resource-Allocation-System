# AI-Based Disaster Resource Allocation System
## Complete Technical Specification, System Design & Implementation Roadmap

---

## 1. Executive Summary & Problem Formulation

### 1.1 Problem Statement
During large-scale disasters (such as riverine floods, chemical hazards, structural collapses, and severe urban waterlogging), emergency resources—including medical teams, CATS ambulances, fire tenders, search and rescue squads (NDRF/Civil Defence), food/water supplies, and emergency shelters—are severely constrained.

Conventional disaster response often relies on fragmented phone reports, static spreadsheets, manual coordination, and localized judgment. This leads to critical operational failures:
- **Spatial Blindness:** Relief coordinators lack a single, unified geographic operational picture showing both disaster impact locations and live resource deployments simultaneously.
- **Suboptimal & Disproportionate Dispatch:** Resources are frequently dispatched on a first-come, first-served basis or based on media visibility, causing some accessible zones to be over-served while severely impacted pockets remain critically underserved during the vital "golden hours."
- **Inaccurate Proximity & Transit Estimation:** Dispatch decisions made without geospatial calculation fail to evaluate real travel times, proximity buffers, and road accessibility constraints.

### 1.2 System Purpose
The **AI-Based Disaster Resource Allocation System** is an integrated, geospatial decision-support platform designed to assist emergency operations centers (EOCs) and field coordinators. 

The system provides:
1. **Automated Geospatial Data Validation:** Standardizing, deduplicating, and bounding raw disaster reports and resource inventories.
2. **Dual-Layer GIS Operational Picture:** Overlaying demand-side disaster incidents and supply-side emergency resources on an interactive digital map.
3. **Automated Proximity & Transit Estimation:** Calculating Great-Circle geodesic distances and urban transit ETAs to determine reachable assets.
4. **Rule-Based Severity & Priority Scoring:** Evaluating multi-attribute indicators (hazard type, population exposed, reported structural damage) to rank incidents objectively.
5. **Human-in-the-Loop Resource Distribution Recommendations:** Providing dispatchers with ranked, actionable allocation suggestions while leaving final deployment authorization to emergency officials.
6. **Modular Architecture for AI & Optimization Extensions:** Structured data contracts and decoupled services that support the direct integration of machine learning severity models, graph-based road routing, and constrained mathematical optimization.

---

## 2. Core System Architecture & Workflow Pipeline

The system is designed as a modular, linear-yet-extensible pipeline where each stage performs a discrete computational task and exposes standardized data contracts.

```
┌────────────────────────────────────────────────────────┐
│             Disaster Incident Reports                  │
│       (Coordinates, Type, Population, Damage)          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Data Preprocessing Engine                │
│    • Bounding Box Check   • Deduplication & Cleaning   │
│    • Schema Validation    • Missing Value Handling     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│          Identification of Affected Zones              │
│    • Coordinate Mapping   • Buffer Radius Delineation  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────┐    ┌───────────────────────────────────┐
│     GIS-Based Mapping Module      │◄──►│ Resource Availability Data / Map  │
│  • OpenStreetMap / Folium Layers  │    │  • Fleet Status (Available/Busy)  │
│  • Visual Severity Color Grading  │    │  • Capacity & Station Telemetry   │
└─────────────────┬─────────────────┘    └───────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│        Severity & Priority Assessment Module           │
│    • Core Implementation: Rule-Based Heuristic Score   │
│    • [Planned AI Extension]: ML Supervised Classifier  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│        Recommended Resource Distribution Module        │
│    • Haversine Distance Matrix Computation             │
│    • Urban Transit Velocity ETA Estimation             │
│    • Ranked Nearest-Asset Candidate Selection          │
│    • [Planned AI Extension]: Multi-Constraint ILP      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│         Interactive GIS Operations Dashboard           │
│    • Streamlit Command Center with Real-Time Filters   │
│    • Visual Dispatch Trajectory Lines & Buffer Arcs    │
│    • Situational Analytics & CSV Audit Export          │
└────────────────────────────────────────────────────────┘
```

---

## 3. Data Models & Parameter Schema

The platform defines strict data schemas matching operational emergency protocols (consistent with NDMA/IRS standards).

### 3.1 Disaster Incident Entity (`DisasterIncident`)

| Parameter | Type | Validation / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `incident_id` | String | Unique Primary Key (`INC-DEL-XXX`) | Unique operational identifier for the incident. |
| `title` | String | Non-empty | Operational name of the incident. |
| `incident_type` | Enum | `Flood`, `Fire / Chemical Hazard`, `Structural Collapse`, `Earthquake`, `Medical Outbreak`, `Severe Storm` | Category of the disaster event. |
| `latitude` | Float | $28.30^\circ\text{N} \le \text{lat} \le 28.95^\circ\text{N}$ | WGS84 decimal latitude. |
| `longitude` | Float | $76.80^\circ\text{E} \le \text{lon} \le 77.55^\circ\text{E}$ | WGS84 decimal longitude. |
| `location_name` | String | Non-empty | Physical landmark or locality name. |
| `severity_level` | Enum | `Critical`, `High`, `Medium`, `Low` | Operational threat level to life and property. |
| `affected_population`| Integer | $\ge 0$ | Estimated count of exposed or displaced people. |
| `reported_damage` | String | Text description | Summary of structural and infrastructure impact. |
| `status` | Enum | `Active`, `Assigned`, `Contained`, `Resolved` | Current field response status. |
| `timestamp` | DateTime | ISO 8601 string | Date and time of the incident report. |
| `reported_by` | String | Non-empty | Reporting agency (e.g., Police PCR, DDMA, Residents). |

### 3.2 Emergency Resource Entity (`EmergencyResource`)

| Parameter | Type | Validation / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `resource_id` | String | Unique Primary Key (`RES-XXX-XXX`) | Unique inventory identifier. |
| `name` | String | Non-empty | Name of the unit, tender, or depot. |
| `resource_type` | Enum | `Ambulance Unit`, `NDRF / Rescue Team`, `Fire Service Unit`, `Food & Water Relief`, `Relief Shelter & Camp` | Classification of the emergency asset. |
| `latitude` | Float | $28.30^\circ\text{N} \le \text{lat} \le 28.95^\circ\text{N}$ | Base station / staging post latitude. |
| `longitude` | Float | $76.80^\circ\text{E} \le \text{lon} \le 77.55^\circ\text{E}$ | Base station / staging post longitude. |
| `base_station` | String | Non-empty | Physical facility or dispatch headquarters. |
| `capacity` | Integer | $\ge 1$ | Personnel count, beds, or vehicle throughput. |
| `availability_status`| Enum | `Available`, `Limited`, `Busy / Deployed`, `Unavailable` | Real-time deployment readiness. |
| `contact_number` | String | Valid phone / hotline string | Control room or dispatch phone number. |
| `dispatch_lead_time_min`| Integer | $\ge 0$ | Turnout preparation time before vehicle rolling. |

---

## 4. Algorithmic Formulations & Mathematical Models

### 4.1 Geospatial Distance: Haversine Formulation
To account for Earth's curvature without the computational overhead of projected ellipsoidal projections during interactive filtering, the platform uses the spherical **Haversine formula**:

$$\Delta \phi = \text{radians}(\phi_2 - \phi_1), \quad \Delta \lambda = \text{radians}(\lambda_2 - \lambda_1)$$

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\text{radians}(\phi_1)) \cdot \cos(\text{radians}(\phi_2)) \cdot \sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$

$$d = R \cdot c$$

Where:
- $R = 6371.0\text{ km}$ (Mean Earth radius)
- $\phi_1, \lambda_1$ are coordinates of the disaster incident.
- $\phi_2, \lambda_2$ are coordinates of the emergency resource depot.
- $d$ is the Great-Circle distance in kilometers.

### 4.2 Urban Emergency Transit & ETA Model
Travel time estimation incorporates baseline emergency transit velocity with a mandatory turnout/dispatch preparation lag:

$$\text{ETA (minutes)} = \left(\frac{d}{\max(v_{\text{transit}}, 5.0)} \times 60\right) + t_{\text{dispatch\_lag}}$$

Where:
- $v_{\text{transit}} = 35.0\text{ km/h}$ (Calibrated average speed for siren-equipped emergency vehicles operating in urban Indian traffic conditions).
- $t_{\text{dispatch\_lag}} = 3.0\text{ minutes}$ (Turnout, crew briefing, and siren startup lead time).

### 4.3 Rule-Based Severity & Priority Scoring
For the core implementation, the system calculates an objective priority index $P \in [0, 100]$:

$$P = \min\left(W_{\text{severity}} + \min\left(40.0, 10.0 \cdot \log_{10}(\max(N_{\text{pop}}, 1))\right), 100.0\right)$$

Where base severity weights $W_{\text{severity}}$ are defined as:
- $\text{Critical} = 50.0$ (Immediate structural hazard or threat to human life)
- $\text{High} = 35.0$ (Active fire, rapidly escalating flooding, hazardous materials)
- $\text{Medium} = 20.0$ (Localized infrastructure disruption, stable casualties)
- $\text{Low} = 10.0$ (Non-life-threatening waterlogging, minor property damage)

$N_{\text{pop}}$ is the reported affected population. The logarithmic scaling ensures that small casualty groups still receive high priority if critical, while large population exposures smoothly increase urgency without overflowing the scale.

---

## 5. Technology Stack & Component Architecture

### 5.1 Technology Matrix

| Layer | Tool / Library | Role & Technical Justification |
| :--- | :--- | :--- |
| **Core Programming** | **Python 3.13+** | Industry standard for data science, geospatial scripting, and backend logic. |
| **Data Ingestion & Cleaning** | **Pandas, NumPy** | Vectorized table processing, missing value imputation, coordinate bounding checks. |
| **Geospatial Processing** | **GeoPandas, Shapely** | Handling geometries, point-in-polygon operations, spatial join queries. |
| **Interactive Map Visualisation**| **Folium (Leaflet.js wrapper)** | Lightweight, browser-native tiled rendering supporting custom vector layers and popups. |
| **Base Cartography** | **OpenStreetMap (OSM)** | Open, unmetered, crowd-validated spatial road and landmark basemaps. |
| **User Interface / Dashboard** | **Streamlit** | Rapid reactive web framework enabling stateful interactive dashboards without frontend boilerplate. |
| **Geospatial Desktop Validation**| **QGIS (Desktop GIS)** | Offline verification, shapefile conversion, and coordinate reference system (CRS) audits. |
| **Future ML Engine** | **Scikit-Learn / XGBoost** | Supervised tabular models for automated severity regression/classification. |
| **Future Graph Routing** | **NetworkX / OSMnx** | Graph representation of road networks ($V, E$) for turn-by-turn Dijkstra and $A^*$ routing. |
| **Future Allocation Engine** | **Google OR-Tools** | Mixed-integer linear programming (MILP) solver for constrained vehicle routing problems (VRP). |

### 5.2 Component Breakdown (Core vs. Future Scope)

| Component | Responsibility | Current Implementation Status |
| :--- | :--- | :--- |
| **Disaster Data Input Module** | Ingests disaster reports, location tags, severity records. | **Core (Implemented)** |
| **Data Preprocessing Module** | Bounding box validation, deduplication, schema normalization. | **Core (Implemented)** |
| **GIS Mapping Module** | Renders affected areas, severity color indicators, response rings. | **Core (Implemented)** |
| **Resource Mapping Module** | Visualises available ambulances, fire tenders, NDRF units, relief depots. | **Core (Implemented)** |
| **Priority Assessment Module**| Evaluates population + damage criteria to rank incidents. | **Core (Rule-Based Implemented)** |
| **Resource Distribution Module**| Computes Haversine distances, estimates ETAs, recommends nearest units. | **Core (Implemented)** |
| **GIS Dashboard Module** | Interactive Streamlit web interface with analytics and CSV exports. | **Core (Implemented)** |
| **AI Allocation Engine** | Dynamic vehicle routing optimization under multi-resource constraints. | **Future Enhancement** |
| **Graph-Based Road Routing** | Real-time road blockage handling via graph shortest-path algorithms. | **Future Enhancement** |
| **Live Telemetry & IoT Feeds**| Automatic updates from water-level sensors and GPS vehicle trackers. | **Future Enhancement** |

---

## 6. Implementation Details: Core Features

### 6.1 Data Validation Pipeline (`src/data_manager.py`)
1. **Bounding Box Enforcement:** Checks all coordinates against the geographic boundary box of Delhi-NCR:
   $$\text{Lat} \in [28.30^\circ\text{N}, 28.95^\circ\text{N}], \quad \text{Lon} \in [76.80^\circ\text{E}, 77.55^\circ\text{E}]$$
2. **Duplicate Invariant:** Verifies zero duplicate records across primary keys (`incident_id`, `resource_id`).
3. **Imputation:** Automatic fallback defaults for missing secondary attributes (e.g., fallback helpline to `112`, capacity to `1`).
4. **Audit Metrics:** Returns real-time metrics displayed on the Dashboard's verification tab.

### 6.2 Proximity & Candidate Ranking Engine (`src/spatial_engine.py`)
1. **Dynamic Spatial Buffer:** Allows operators to adjust the radius buffer slider ($2.0\text{ km}$ to $20.0\text{ km}$).
2. **Availability Filtering:** Filters out units marked as `Busy / Deployed` or `Unavailable`, preventing invalid assignments.
3. **$k$-Nearest Neighbor Query:** Evaluates all eligible units against the target disaster location, returning the top $k$ candidates sorted by distance ascending.
4. **Dynamic Trajectory Overlay:** Renders dashed geospatial trajectory polylines on the Folium map connecting the incident to the top 3 recommended units.

### 6.3 Situational Registry & Analytics
1. **Categorical Breakdown:** Aggregates active incidents by disaster classification (Floods, Fires, Collapses, Outbreaks).
2. **Readiness Summary:** Visualizes asset distribution across operational availability states.
3. **Audited Data Export:** Provides direct one-click CSV export buttons for emergency field documentation.

---

## 7. Future Enhancements & Analytical Roadmap

The core architecture includes decoupled data contracts specifically designed to enable advanced AI and optimization extensions without restructuring the user interface:

### 7.1 Machine Learning Severity Prediction Model
- **Goal:** Replace rule-based heuristics with a supervised classification model.
- **Input Features ($X$):** Disaster category, building density, rainfall accumulation, reported casualties, historical vulnerability index, time of day.
- **Target Variable ($y$):** Priority level (`Critical`, `High`, `Medium`, `Low`) or continuous emergency severity score ($0.0 - 100.0$).
- **Candidate Models:** Random Forest, Gradient Boosted Trees (XGBoost/LightGBM), evaluated using Precision, Recall, and F1-score (with high recall for `Critical` cases).

### 7.2 Road Network Graph Routing (NetworkX / OSMnx)
- **Goal:** Replace Euclidean/Haversine distance with actual driving road-network distances and turn-by-turn navigation.
- **Graph Formulation:** Road intersections modeled as graph nodes $V$, road segments as directed edges $E$. Edge weights $w(u, v)$ represent estimated travel time:
  $$w(u, v) = \frac{\text{length}(u, v)}{v_{\text{segment}}} \times \kappa_{\text{congestion}}$$
- **Dynamic Invalidation:** When an area reports severe flooding or building debris, corresponding edges $e \in E_{\text{blocked}}$ have their weights set to $\infty$, prompting Dijkstra or $A^*$ to route around hazard zones.

### 7.3 Constrained Mathematical Optimization Engine (Google OR-Tools)
- **Goal:** Transition from independent nearest-neighbor dispatch to jurisdiction-wide optimal resource distribution.
- **Objective Function:**
  $$\min \quad Z = \alpha \sum_{i \in I} \sum_{j \in J} x_{ij} \cdot t_{ij} + \beta \sum_{i \in I} u_i \cdot \text{Severity}_i - \delta \sum_{i \in I} \sum_{j \in J} x_{ij} \cdot \text{Benefit}_{ij}$$
  Subject to:
  - $\sum_{j \in J} x_{ij} \le \text{Demand}_i$ (Demand limits per incident $i$)
  - $\sum_{i \in I} x_{ij} \le \text{Capacity}_j$ (Capacity limits per resource unit $j$)
  - $t_{ij} \le T_{\max}$ (Maximum permissible response time window)
  - $x_{ij} \in \{0, 1\}$ (Binary dispatch decisions)

### 7.4 Live Data Feeds & IoT Sensor Integration
- **Automated Weather Feeds:** Ingestion of live radar and rainfall data to pre-alert teams before flash waterlogging occurs.
- **Water Level Sensor Telemetry:** Real-time water-level sensors on Yamuna barrages feeding automatic flood warnings directly into the incident registry.

---

## 8. Verification & Operational Testing Guidelines

### 8.1 Automated Unit & Integration Tests
1. **Coordinate Boundary Tests:** Feeding test coordinates outside Delhi-NCR (e.g., coordinates in Mumbai or oceans) must be caught by `data_manager.py` and flagged.
2. **Haversine Distance Verification:** Comparing calculated output against standard geographic benchmarks (e.g., AIIMS to Connaught Place $\approx 7.2\text{ km}$).
3. **Availability Filter Tests:** Marking a resource as `Busy / Deployed` must immediately exclude it from the candidate recommendations list.

### 8.2 Live Dashboard Execution
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the Streamlit application
streamlit run app.py --server.port 8501
```
The application will launch on `http://localhost:8501`.
