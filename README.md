# AI-Based Disaster Resource Allocation System
**Micro Project Prototype & GIS Decision Support Dashboard**  
*Emergency Management & Geospatial Logistics Platform*

---

## 📌 Project Overview
During natural disasters (floods, earthquakes, fires, structural collapses), emergency response resources such as rescue teams, medical aid, food, drinking water, and relief shelters are critically limited. 

This project implements an integrated **GIS-based decision-support system** that captures, validates, and visualises disaster-affected locations and available emergency resources on a unified interactive digital map. It provides automated spatial proximity calculations, rule-based severity assessments, and ranked resource-distribution recommendations to assist emergency coordinators in rapid decision-making.

---

## 🏗️ System Workflow & Scope

### Fig. 1. System Pipeline
```text
Disaster Data Input ──► Data Preprocessing ──► Identification of Affected Areas ──► GIS-Based Mapping
                                                                                             ▲
Resource Availability Data / Mapping ────────────────────────────────────────────────────────┘
                               ▼
            Disaster Severity / Priority Assessment (Rule-Based)
                               ▼
        [Planned: AI-Based Prioritisation & Allocation (Future Enhancement)]
                               ▼
               Recommended Resource Distribution
                               ▼
                  GIS Dashboard / Visualisation
```

### Core Implementation vs. Future Enhancement
- **Core Micro Project Deliverables (Implemented)**:
  - Disaster Data Ingestion & Preprocessing (bounding box check, deduplication, schema validation).
  - OpenStreetMap & Folium Interactive GIS Mapping.
  - Resource Availability Mapping across Delhi-NCR.
  - Rule-Based Severity / Priority Assessment.
  - Recommended Resource Distribution (Haversine distance & urban transit ETA).
  - Web-based Operations Dashboard (Streamlit).
- **Future Scope & Enhancements**:
  - Supervised machine learning models for dynamic damage prediction.
  - Formal multi-constraint optimization algorithms (travel distance, capacity, time windows).
  - Real-time IoT / satellite data feeds and field responder mobile client.

---

## 📁 Repository Structure
```text
disaster-resource-allocation/
├── app.py                     # Streamlit + Folium GIS Operations Center
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & execution guide
├── data/
│   ├── seed_data.py           # Seed generator for authentic Delhi-NCR disaster data
│   ├── delhi_incidents.csv    # Validated active incident records
│   └── delhi_resources.csv    # Validated emergency units & depots
└── src/
    ├── __init__.py
    ├── models.py              # Core data models (Incident, Resource, Enums)
    ├── spatial_engine.py      # Haversine distance, ETA, nearest-neighbor queries
    └── data_manager.py        # Pipeline validation, bounds checking, deduplication
```

---

## 🚀 How to Run the Application

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Launch the GIS Operations Dashboard**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at **`http://localhost:8501`**.

---

## 🎯 Features Demonstrated
- **Interactive Delhi-NCR GIS Map**: Color-coded disaster incidents and emergency resource markers.
- **Dynamic Spatial Proximity**: Selecting an incident highlights its response buffer and displays trajectory lines to the top nearest available units with exact distance (km) and estimated transit ETA (min).
- **Multi-Parameter Filtering**: Filter by disaster category, minimum severity, resource category, and real-time operational status.
- **Situational Analytics & Data Export**: Distribution charts, capacity summaries, and full validated CSV downloads.
- **Preprocessing Audit & Parameter Schema**: Complete verification of Table III parameters with zero duplicates and coordinate boundary validation.
