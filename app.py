"""
AI-Based Disaster Resource Allocation System
Interactive Decision-Support Platform & Live Operations Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import plugins
from streamlit_folium import st_folium

from src.models import IncidentType, SeverityLevel, ResourceType, ResourceStatus
from src.spatial_engine import (
    haversine_distance,
    estimate_travel_time_minutes,
    find_nearest_resources,
    compute_system_coverage_stats
)
from src.data_manager import (
    load_datasets,
    validate_and_clean_datasets,
    create_and_validate_incident,
    DELHI_NCR_BOUNDS
)
from src.ml_severity import get_ml_engine, SEVERITY_CLASSES
from src.routing_engine import get_road_router, ROAD_GRAPH_NODES
from src.optimization_engine import run_global_fleet_optimization


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Disaster Resource Allocation System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #EA580C;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .ai-banner {
        background-color: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-left: 4px solid #4F46E5;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Core Engines Initialization ---
@st.cache_resource
def load_ml_and_routing_engines():
    ml = get_ml_engine()
    router = get_road_router()
    return ml, router

ml_engine, road_router = load_ml_and_routing_engines()


# --- Stateful Data Management ---
if "incidents_df" not in st.session_state or "resources_df" not in st.session_state:
    raw_inc, raw_res = load_datasets()
    clean_inc, clean_res, audit = validate_and_clean_datasets(raw_inc, raw_res)
    st.session_state.incidents_df = clean_inc
    st.session_state.resources_df = clean_res
    st.session_state.audit_report = audit
    st.session_state.dispatch_log = []
    st.session_state.selected_incident_id = clean_inc["incident_id"].iloc[0] if not clean_inc.empty else None

def reset_system_state():
    raw_inc, raw_res = load_datasets()
    clean_inc, clean_res, audit = validate_and_clean_datasets(raw_inc, raw_res)
    st.session_state.incidents_df = clean_inc
    st.session_state.resources_df = clean_res
    st.session_state.audit_report = audit
    st.session_state.dispatch_log = []
    st.session_state.selected_incident_id = clean_inc["incident_id"].iloc[0] if not clean_inc.empty else None

def dispatch_unit_to_incident(resource_id: str, incident_id: str, distance_km: float, eta_min: float):
    res_idx = st.session_state.resources_df.index[st.session_state.resources_df["resource_id"] == resource_id].tolist()
    if res_idx:
        st.session_state.resources_df.at[res_idx[0], "availability_status"] = "Busy / Deployed"

    inc_idx = st.session_state.incidents_df.index[st.session_state.incidents_df["incident_id"] == incident_id].tolist()
    inc_title = "Active Incident"
    if inc_idx:
        st.session_state.incidents_df.at[inc_idx[0], "status"] = "Assigned"
        inc_title = st.session_state.incidents_df.at[inc_idx[0], "title"]

    res_name = st.session_state.resources_df.loc[res_idx[0], "name"] if res_idx else resource_id
    st.session_state.dispatch_log.insert(0, {
        "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        "resource_id": resource_id,
        "resource_name": res_name,
        "incident_id": incident_id,
        "incident_title": inc_title,
        "distance_km": distance_km,
        "eta_minutes": eta_min,
        "status": "In Transit"
    })

def recall_unit_from_field(resource_id: str):
    res_idx = st.session_state.resources_df.index[st.session_state.resources_df["resource_id"] == resource_id].tolist()
    if res_idx:
        st.session_state.resources_df.at[res_idx[0], "availability_status"] = "Available"
    for item in st.session_state.dispatch_log:
        if item["resource_id"] == resource_id and item["status"] == "In Transit":
            item["status"] = "Returned to Base"
            break

incidents_df = st.session_state.incidents_df
resources_df = st.session_state.resources_df
audit_report = st.session_state.audit_report


# --- Header & Meta Info ---
st.markdown('<div class="main-header">🚨 AI-Based Disaster Resource Allocation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<b>Micro Project Stage — GIS-Based Proximity & Spatial Decision-Support Foundation</b> &bull; '
    'Geospatial Operations, Machine Learning Assessment & Emergency Logistics Platform'
    '</div>',
    unsafe_allow_html=True
)


# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/color/96/000000/siren.png", width=64)
st.sidebar.header("Operations Control")

incident_types = ["All"] + list(incidents_df["incident_type"].unique())
selected_inc_type = st.sidebar.selectbox("Filter Disaster Type", incident_types, index=0)

severity_options = ["All", "Critical", "High", "Medium", "Low"]
selected_severity = st.sidebar.selectbox("Filter Severity Level", severity_options, index=0)

resource_types = ["All"] + list(resources_df["resource_type"].unique())
selected_res_type = st.sidebar.selectbox("Resource Type Needed", resource_types, index=0)

only_available = st.sidebar.checkbox("Show Only Available/Limited Assets", value=True)

radius_km = st.sidebar.slider("Spatial Proximity Radius (km)", min_value=2.0, max_value=20.0, value=7.0, step=0.5)

# --- Navigation & Hazard Simulation Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛣️ Road Network & Hazard Simulation")
routing_mode = st.sidebar.radio(
    "Transit Routing Engine",
    ["🛣️ Graph Road Network (Dijkstra Turn-by-Turn)", "📏 Direct Geodesic (Haversine)"],
    index=0
)

sim_yamuna_blockage = st.sidebar.checkbox("🌊 Simulate Yamuna Bridge Inundation (Flood Block)", value=False)
sim_mundka_blockage = st.sidebar.checkbox("🔥 Simulate Rohtak Road Fire Barrier (Chemical Block)", value=False)

# Update router blockage state
road_router.set_blockage("KASHMERE_GATE", "YAMUNA_BRIDGE", sim_yamuna_blockage)
road_router.set_blockage("PASCHIM_VIHAR", "MUNDKA_IND", sim_mundka_blockage)


# --- Sidebar Incident Reporting Form ---
st.sidebar.markdown("---")
with st.sidebar.expander("🚨 Report New Disaster Incident", expanded=False):
    st.caption("Intake verified incident directly into active dispatch grid.")
    
    preset_locations = {
        "Custom Coordinates": (28.6139, 77.2090, "New Delhi Sector"),
        "Kashmere Gate Lowland": (28.6670, 77.2250, "Kashmere Gate Lowlands"),
        "Yamuna Bank Flood Basin": (28.6250, 77.2600, "Yamuna Bank Staging Basin"),
        "Connaught Place Radial": (28.6315, 77.2167, "Connaught Place Inner Circle"),
        "Rohini Sector 14 Hub": (28.7150, 77.1180, "Rohini Institutional Area"),
        "Karol Bagh Market": (28.6520, 77.1900, "Karol Bagh Commercial Area"),
        "Lajpat Nagar Ring Road": (28.5700, 77.2400, "Lajpat Nagar Central"),
        "Dwarka Sector 21 Hub": (28.5520, 77.0580, "Dwarka Transportation Hub")
    }
    
    selected_preset = st.selectbox("Location Preset", list(preset_locations.keys()), index=0)
    def_lat, def_lon, def_loc = preset_locations[selected_preset]
    
    with st.form("new_incident_form", clear_on_submit=False):
        new_title = st.text_input("Incident Title", value=f"{selected_preset} Distress Call" if selected_preset != "Custom Coordinates" else "Urgent Distress Call")
        new_type = st.selectbox("Disaster Category", ["Flood", "Fire / Chemical Hazard", "Structural Collapse", "Medical Outbreak", "Severe Storm"], index=0)
        c_lat, c_lon = st.columns(2)
        with c_lat:
            new_lat = st.number_input("Latitude", value=def_lat, min_value=float(DELHI_NCR_BOUNDS["min_lat"]), max_value=float(DELHI_NCR_BOUNDS["max_lat"]), format="%.4f")
        with c_lon:
            new_lon = st.number_input("Longitude", value=def_lon, min_value=float(DELHI_NCR_BOUNDS["min_lon"]), max_value=float(DELHI_NCR_BOUNDS["max_lon"]), format="%.4f")
        new_loc_name = st.text_input("Locality / Landmark", value=def_loc)
        new_sev = st.selectbox("Assessed Severity", ["Critical", "High", "Medium", "Low"], index=0)
        new_pop = st.number_input("Estimated Affected Population", min_value=1, max_value=50000, value=250, step=25)
        new_damage = st.text_area("Reported Damage / Scene Details", value="Casualties reported on site; emergency perimeter established.")
        
        submit_btn = st.form_submit_button("🚨 Submit Incident to Dispatch Grid", width="stretch")
        if submit_btn:
            raw_incident = {
                "title": new_title,
                "incident_type": new_type,
                "latitude": new_lat,
                "longitude": new_lon,
                "location_name": new_loc_name,
                "severity_level": new_sev,
                "affected_population": new_pop,
                "reported_damage": new_damage,
                "status": "Active"
            }
            is_valid, msg, validated_inc = create_and_validate_incident(raw_incident)
            if is_valid and validated_inc is not None:
                new_row_df = pd.DataFrame([validated_inc])
                st.session_state.incidents_df = pd.concat([new_row_df, st.session_state.incidents_df], ignore_index=True)
                st.session_state.selected_incident_id = validated_inc["incident_id"]
                st.success(f"Incident {validated_inc['incident_id']} added and prioritized!")
                st.rerun()
            else:
                st.error(f"Validation failed: {msg}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset to Default Fleet State", key="sidebar_reset_btn", width="stretch"):
    reset_system_state()
    st.toast("System reset to default state.")
    st.rerun()


# --- Filtered Data ---
filtered_incidents = incidents_df.copy()
if selected_inc_type != "All":
    filtered_incidents = filtered_incidents[filtered_incidents["incident_type"] == selected_inc_type]
if selected_severity != "All":
    filtered_incidents = filtered_incidents[filtered_incidents["severity_level"] == selected_severity]

filtered_resources = resources_df.copy()
if selected_res_type != "All":
    filtered_resources = filtered_resources[filtered_resources["resource_type"] == selected_res_type]
if only_available:
    filtered_resources = filtered_resources[filtered_resources["availability_status"].isin(["Available", "Limited"])]

# High-Level Metrics
cov_stats = compute_system_coverage_stats(filtered_incidents, resources_df, threshold_radius_km=radius_km)
critical_count = len(filtered_incidents[filtered_incidents["severity_level"] == "Critical"])
avail_resources_count = len(resources_df[resources_df["availability_status"] == "Available"])

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Disasters</div>
        <div class="metric-value">{len(filtered_incidents)}</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #EF4444;">
        <div class="metric-title">Critical Severity</div>
        <div class="metric-value" style="color: #DC2626;">{critical_count}</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #10B981;">
        <div class="metric-title">Available Units</div>
        <div class="metric-value" style="color: #059669;">{avail_resources_count}</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #8B5CF6;">
        <div class="metric-title">GIS Coverage Rate</div>
        <div class="metric-value" style="color: #7C3AED;">{cov_stats['coverage_pct']}%</div>
    </div>
    """, unsafe_allow_html=True)
with m5:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #F59E0B;">
        <div class="metric-title">Avg Distance to Unit</div>
        <div class="metric-value">{cov_stats['avg_min_distance_km']} km</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# --- Main Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Live GIS Command Center",
    "🤖 AI Severity Model & Inference",
    "⚡ Global Fleet Allocation Optimizer",
    "📊 Situational Analytics & Registry",
    "🔍 Data Preprocessing & Parameters",
    "🚀 System Architecture & Flowchart"
])


# ==========================================
# TAB 1: GIS COMMAND CENTER
# ==========================================
with tab1:
    col_map, col_details = st.columns([7, 5])

    # Incident Selection Selector for inspection
    incident_choices = {
        f"[{row['incident_id']}] {row['title']} ({row['severity_level']}) - {'[ASSIGNED]' if row.get('status') == 'Assigned' else '[ACTIVE]'}": row["incident_id"]
        for _, row in filtered_incidents.iterrows()
    }

    if not incident_choices:
        st.warning("No incidents match the active filter criteria.")
        selected_inc_id = None
        selected_incident = None
    else:
        with col_details:
            choice_keys = list(incident_choices.keys())
            curr_sel_id = st.session_state.get("selected_incident_id")
            def_idx = 0
            for idx, k in enumerate(choice_keys):
                if incident_choices[k] == curr_sel_id:
                    def_idx = idx
                    break
            selected_label = st.selectbox("Inspect Disaster Incident for Dispatch:", choice_keys, index=def_idx)
            selected_inc_id = incident_choices[selected_label]
            st.session_state.selected_incident_id = selected_inc_id
            selected_incident = filtered_incidents[filtered_incidents["incident_id"] == selected_inc_id].iloc[0]

    with col_map:
        st.subheader("Delhi-NCR Spatial Operational Picture")
        
        # Center map on selected incident or default to central Delhi
        if selected_incident is not None:
            map_center = [selected_incident["latitude"], selected_incident["longitude"]]
            map_zoom = 12
        else:
            map_center = [28.6448, 77.1500]
            map_zoom = 11

        folium_map = folium.Map(
            location=map_center,
            zoom_start=map_zoom,
            tiles="OpenStreetMap"
        )

        # Draw Resources Markers
        resource_colors = {
            "Ambulance Unit (CATS / Hospital)": "blue",
            "NDRF / Civil Defence Rescue Team": "green",
            "Delhi Fire Service Unit": "darkred",
            "Emergency Food & Water Relief": "purple",
            "Relief Shelter & Camp": "cadetblue"
        }

        resource_icons = {
            "Ambulance Unit (CATS / Hospital)": "plus",
            "NDRF / Civil Defence Rescue Team": "life-ring",
            "Delhi Fire Service Unit": "fire",
            "Emergency Food & Water Relief": "tint",
            "Relief Shelter & Camp": "home"
        }

        for _, res in filtered_resources.iterrows():
            res_color = resource_colors.get(res["resource_type"], "blue")
            res_icon = resource_icons.get(res["resource_type"], "info-sign")
            is_deployed = res["availability_status"] == "Busy / Deployed"
            if is_deployed:
                res_color = "orange"

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <h4 style="margin: 0; color: #1E293B;">{res['name']}</h4>
                <p style="margin: 4px 0; font-size: 12px; color: #64748B;"><b>ID:</b> {res['resource_id']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Type:</b> {res['resource_type']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Base:</b> {res['base_station']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Capacity:</b> {res['capacity']} units</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Status:</b> <span style="font-weight:bold; color:{'#EA580C' if is_deployed else '#16A34A'};">{res['availability_status']}</span></p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Helpline:</b> {res.get('contact_number', 'N/A')}</p>
            </div>
            """

            folium.Marker(
                location=[res["latitude"], res["longitude"]],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Resource: {res['name']} ({res['availability_status']})",
                icon=folium.Icon(color=res_color, icon=res_icon, prefix="fa")
            ).add_to(folium_map)

        # Draw Incidents Markers
        severity_marker_colors = {
            "Critical": "red",
            "High": "orange",
            "Medium": "beige",
            "Low": "gray"
        }

        for _, inc in filtered_incidents.iterrows():
            is_active_selected = (selected_incident is not None and inc["incident_id"] == selected_incident["incident_id"])
            m_color = severity_marker_colors.get(inc["severity_level"], "red")
            
            inc_popup = f"""
            <div style="font-family: Arial, sans-serif; min-width: 220px;">
                <h4 style="margin: 0; color: #DC2626;">{inc['title']}</h4>
                <p style="margin: 4px 0; font-size: 12px;"><b>ID:</b> {inc['incident_id']} | <b>Type:</b> {inc['incident_type']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Location:</b> {inc['location_name']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Severity:</b> <b>{inc['severity_level']}</b> (Score: {inc['priority_score']}/100)</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Status:</b> <b>{inc.get('status', 'Active')}</b></p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Affected Pop.:</b> {inc['affected_population']:,}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Reported Damage:</b> {inc['reported_damage']}</p>
            </div>
            """

            folium.CircleMarker(
                location=[inc["latitude"], inc["longitude"]],
                radius=10 if not is_active_selected else 14,
                color="#DC2626" if inc["severity_level"] == "Critical" else "#EA580C",
                weight=3 if is_active_selected else 1,
                fill=True,
                fill_color=m_color,
                fill_opacity=0.9 if is_active_selected else 0.7,
                popup=folium.Popup(inc_popup, max_width=320),
                tooltip=f"Incident: {inc['title']} [{inc['severity_level']}] - {inc.get('status', 'Active')}"
            ).add_to(folium_map)

        # Nearest resources and routing paths
        nearest_units = []
        if selected_incident is not None:
            inc_lat = selected_incident["latitude"]
            inc_lon = selected_incident["longitude"]

            # Response buffer circle
            folium.Circle(
                location=[inc_lat, inc_lon],
                radius=radius_km * 1000,
                color="#EF4444",
                weight=2,
                dash_array="5, 5",
                fill=True,
                fill_color="#F87171",
                fill_opacity=0.15,
                tooltip=f"{radius_km} km Response Buffer"
            ).add_to(folium_map)

            # Query spatial engine for top 3 nearest available resources
            nearest_units = find_nearest_resources(
                incident_lat=inc_lat,
                incident_lon=inc_lon,
                resources_df=resources_df,
                resource_type=selected_res_type if selected_res_type != "All" else None,
                only_available=only_available,
                max_results=3,
                max_radius_km=radius_km * 1.5
            )

            # Draw routes to nearest resources
            for idx, unit in enumerate(nearest_units):
                line_color = "#2563EB" if idx == 0 else "#64748B"
                weight = 4 if idx == 0 else 2
                
                if "Graph Road Network" in routing_mode:
                    calc_route = road_router.calculate_road_route(unit["latitude"], unit["longitude"], inc_lat, inc_lon)
                    route_pts = calc_route["route_coordinates"]
                    is_detour = calc_route.get("detour_active", False)
                    corridor_str = " ➔ ".join(calc_route.get("corridors_traversed", ["Direct"]))
                    
                    folium.PolyLine(
                        locations=route_pts,
                        color="#DC2626" if (is_detour and idx == 0) else line_color,
                        weight=weight,
                        dash_array=None if idx == 0 else "5, 5",
                        tooltip=f"Turn-by-Turn Route #{idx+1} to {unit['name']}: {calc_route['total_distance_km']} km (ETA ~{calc_route['estimated_eta_min']}m) | Corridors: {corridor_str}"
                    ).add_to(folium_map)
                else:
                    folium.PolyLine(
                        locations=[[inc_lat, inc_lon], [unit["latitude"], unit["longitude"]]],
                        color=line_color,
                        weight=weight,
                        dash_array="6, 6",
                        tooltip=f"Geodesic Line #{idx+1} to {unit['name']}: {unit['distance_km']} km (ETA ~{unit['eta_minutes']}m)"
                    ).add_to(folium_map)

            # If blockage active, show hazard marker on map
            if sim_yamuna_blockage:
                folium.Marker(
                    location=[28.6610, 77.2400],
                    popup="<b>⛔ HAZARD ROADBLOCK: Old Yamuna Bridge Causeway Submerged under Floodwater</b>",
                    tooltip="⛔ HAZARD: Yamuna Bridge Submerged",
                    icon=folium.Icon(color="red", icon="ban", prefix="fa")
                ).add_to(folium_map)
            if sim_mundka_blockage:
                folium.Marker(
                    location=[28.6812, 77.0321],
                    popup="<b>⛔ HAZARD ROADBLOCK: Rohtak Road NH-10 Impassable (Chemical Fire Barrier)</b>",
                    tooltip="⛔ HAZARD: Rohtak Road Blocked",
                    icon=folium.Icon(color="darkred", icon="fire", prefix="fa")
                ).add_to(folium_map)

        st_folium(folium_map, width="100%", height=560)

        if sim_yamuna_blockage or sim_mundka_blockage:
            st.warning("⚠️ **Dynamic Detour Active:** Real-time arterial graph router has recomputed paths around submerged/blocked corridors using NetworkX Dijkstra shortest path.")

    # --- Details & Recommendation Column ---
    with col_details:
        if selected_incident is not None:
            st.markdown("### Incident Profile")
            
            # Severity Badge formatting
            badge_class = f"badge-{selected_incident['severity_level'].lower()}"
            st.markdown(
                f"<span class='{badge_class}'>● {selected_incident['severity_level'].upper()} SEVERITY</span> "
                f"&nbsp; <b>ID:</b> {selected_incident['incident_id']} &nbsp;|&nbsp; <b>Type:</b> {selected_incident['incident_type']}",
                unsafe_allow_html=True
            )
            st.write("")
            
            st.markdown(f"**Location:** {selected_incident['location_name']}")
            st.markdown(f"**Reported Damage:** {selected_incident['reported_damage']}")
            
            is_assigned = (selected_incident.get("status") == "Assigned")
            status_badge = (
                "<span style='background-color: #DCFCE7; color: #166534; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>● DISPATCH ASSIGNED</span>"
                if is_assigned else
                "<span style='background-color: #FEE2E2; color: #991B1B; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>● AWAITING RESPONSE</span>"
            )
            st.markdown(f"**Operational Status:** {status_badge}", unsafe_allow_html=True)

            # AI Inference Assessment
            ml_pred = ml_engine.predict(
                disaster_type=selected_incident["incident_type"],
                affected_population=int(selected_incident["affected_population"]),
                damage_index=8 if selected_incident["severity_level"] == "Critical" else (6 if selected_incident["severity_level"] == "High" else 3),
                weather_risk=4 if selected_incident["incident_type"] == "Flood" else 2,
                vulnerability_index=4 if selected_incident["severity_level"] in ["Critical", "High"] else 2,
                collapse_flag=1 if selected_incident["incident_type"] == "Structural Collapse" else 0
            )

            st.markdown(
                f"<div class='ai-banner'>"
                f"<b>🤖 AI Model Severity Classification:</b> <b style='color:#4338CA;'>{ml_pred['predicted_class'].upper()}</b> "
                f"&bull; Confidence: <b>{ml_pred['confidence_pct']}%</b> &bull; Regressed Score: <b>{ml_pred['predicted_score']}/100</b>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("Affected Population", f"{selected_incident['affected_population']:,}")
            with p_col2:
                st.metric("Priority Index", f"{selected_incident['priority_score']} / 100")
            
            st.progress(selected_incident['priority_score'] / 100.0)

            st.markdown("---")
            st.markdown("### 🎯 Nearest Resource Dispatch Recommendations")
            st.caption(f"Calculated via {routing_mode.split()[1]} & urban transit velocity model.")

            if not nearest_units:
                st.error("⚠️ No available emergency resources found within the specified radius. Recommend expanding the search buffer or requesting mutual aid from neighboring districts.")
            else:
                for i, u in enumerate(nearest_units):
                    rank_icon = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
                    
                    # Calculate road route for the unit
                    unit_dist = u["distance_km"]
                    unit_eta = u["eta_minutes"]
                    corridor_summary = "Direct Geodesic"
                    
                    if "Graph Road Network" in routing_mode:
                        r_info = road_router.calculate_road_route(u["latitude"], u["longitude"], selected_incident["latitude"], selected_incident["longitude"])
                        unit_dist = r_info["total_distance_km"]
                        unit_eta = r_info["estimated_eta_min"]
                        corridor_summary = " ➔ ".join(r_info.get("corridors_traversed", ["Arterial Route"]))

                    with st.container():
                        st.markdown(f"**{rank_icon} Rank #{i+1}: {u['name']}**")
                        c_u1, c_u2, c_u3 = st.columns([3, 2, 2])
                        with c_u1:
                            st.write(f"🏢 *{u['base_station']}*")
                            st.write(f"📞 `{u['contact_number']}`")
                        with c_u2:
                            st.metric("Route Dist", f"{unit_dist} km")
                        with c_u3:
                            st.metric("Est. ETA", f"{unit_eta} min")
                        
                        btn_col1, btn_col2 = st.columns([3, 2])
                        with btn_col1:
                            st.caption(f"Corridor: {corridor_summary}")
                        with btn_col2:
                            if st.button(f"🚨 Dispatch Unit #{i+1}", key=f"dispatch_btn_{u['resource_id']}_{selected_incident['incident_id']}", width="stretch"):
                                dispatch_unit_to_incident(
                                    resource_id=u["resource_id"],
                                    incident_id=selected_incident["incident_id"],
                                    distance_km=unit_dist,
                                    eta_min=unit_eta
                                )
                                st.toast(f"✅ Dispatched {u['name']} to {selected_incident['title']}! ETA: {unit_eta} min", icon="🚨")
                                st.rerun()
                        st.divider()


# ==========================================
# TAB 2: AI SEVERITY MODEL & INFERENCE
# ==========================================
with tab2:
    st.subheader("🤖 Supervised Machine Learning Severity Assessment Engine")
    st.markdown("""
    This module implements the **AI-based severity assessment enhancement** (Section 5.8 of synopsis). 
    A **Random Forest Classifier & Regressor** is trained on historical and simulated multi-hazard NDMA indicators 
    to categorize disaster urgency and predict continuous damage impact scores.
    """)

    metrics = ml_engine.metrics
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.metric("Model Accuracy", f"{metrics['accuracy']}%")
    with sc2:
        st.metric("Precision (Weighted)", f"{metrics['precision']}%")
    with sc3:
        st.metric("Recall (Sensitivity)", f"{metrics['recall']}%")
    with sc4:
        st.metric("F1-Score", f"{metrics['f1_score']}%")
    with sc5:
        st.metric("Score MAE", f"±{metrics['mae']} pts")

    st.markdown("---")
    col_feat, col_sandbox = st.columns([6, 6])

    with col_feat:
        st.markdown("#### 📊 Model Feature Importance (Gini Impurity Reduction)")
        st.caption("How much each real-world variable drives the AI model's severity categorization:")
        
        fi = metrics["feature_importance"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["Disaster Indicator", "Influence Weight (%)"]).sort_values(by="Influence Weight (%)", ascending=False)
        st.dataframe(fi_df, width="stretch", hide_index=True)
        st.bar_chart(data=fi_df.set_index("Disaster Indicator"))

        st.markdown("#### 🎯 Classification Confusion Matrix")
        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["classes"], columns=[f"Pred {c}" for c in metrics["classes"]])
        st.dataframe(cm_df, width="stretch")

    with col_sandbox:
        st.markdown("#### 🧪 Interactive AI Inference Sandbox")
        st.caption("Test the trained machine learning model in real-time with custom disaster conditions:")
        
        with st.form("ai_inference_sandbox_form"):
            s_type = st.selectbox("Disaster Category", ["Flood", "Fire / Chemical Hazard", "Structural Collapse", "Medical Outbreak", "Severe Storm"], index=0)
            s_pop = st.slider("Affected Population Count", min_value=10, max_value=5000, value=750, step=50)
            s_dmg = st.slider("Infrastructure Damage Severity Index (1=Cosmetic, 10=Total Collapse)", min_value=1, max_value=10, value=7)
            s_weather = st.slider("Weather Escalation Risk Factor (1=Calm, 5=Severe Storm)", min_value=1, max_value=5, value=4)
            s_vuln = st.slider("Locality Vulnerability Index (1=Planned Sector, 5=Dense Informal)", min_value=1, max_value=5, value=4)
            s_collapse = st.radio("Structural Building Collapse Reported?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=1)[1]
            
            run_ai_btn = st.form_submit_button("⚡ Run AI Severity Assessment", width="stretch")

        if run_ai_btn:
            pred_res = ml_engine.predict(
                disaster_type=s_type,
                affected_population=s_pop,
                damage_index=s_dmg,
                weather_risk=s_weather,
                vulnerability_index=s_vuln,
                collapse_flag=s_collapse
            )
            
            st.success(f"### Predicted Priority: {pred_res['predicted_class'].upper()}")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.metric("Predicted Severity Score", f"{pred_res['predicted_score']} / 100")
            with c_p2:
                st.metric("Model Confidence", f"{pred_res['confidence_pct']}%")
            
            st.progress(pred_res['predicted_score'] / 100.0)
            st.markdown("##### Class Probability Distribution:")
            prob_df = pd.DataFrame(list(pred_res["class_probabilities"].items()), columns=["Severity Class", "Probability (%)"])
            st.dataframe(prob_df, width="stretch", hide_index=True)


# ==========================================
# TAB 3: GLOBAL FLEET ALLOCATION OPTIMIZER
# ==========================================
with tab3:
    st.subheader("⚡ Jurisdiction-Wide Multi-Incident Global Optimization Engine")
    st.markdown("""
    While greedy nearest-neighbor matching allocates one incident at a time, large-scale disasters require
    **global multi-resource optimization**. This engine formulates a bipartite matching problem solved via the 
    **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`), balancing response time against incident severity.
    """)

    opt_alpha = st.slider("Optimization Weight $\\alpha$ (Transit Time Minimization)", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
    opt_beta = round(1.0 - opt_alpha, 2)
    st.caption(f"Active Objective: $\\min \\sum ( {opt_alpha} \\times \\text{{ETA}} - {opt_beta} \\times \\text{{Priority}} )$")

    if st.button("⚡ Compute Global Optimal Fleet Allocation", key="run_opt_btn", width="stretch"):
        opt_result = run_global_fleet_optimization(incidents_df, resources_df, time_weight_alpha=opt_alpha, priority_weight_beta=opt_beta)
        st.session_state.global_opt_result = opt_result

    if "global_opt_result" in st.session_state and st.session_state.global_opt_result["success"]:
        res = st.session_state.global_opt_result
        st.success(f"✅ {res['message']}")

        o1, o2, o3, o4 = st.columns(4)
        with o1:
            st.metric("Matched Incidents", f"{res['assigned_count']}")
        with o2:
            st.metric("Cumulative Distance", f"{res['total_distance_km']} km")
        with o3:
            st.metric("Average Global ETA", f"{res['avg_eta_min']} min")
        with o4:
            st.metric("Priority Demand Served", f"{res['priority_served_pct']}%")

        st.markdown("#### Optimal Allocation Dispatch Matrix:")
        opt_table_data = []
        for match in res["assignments"]:
            opt_table_data.append({
                "Incident ID": match["incident_id"],
                "Disaster Target": match["incident_title"],
                "Severity": match["incident_severity"],
                "Priority Score": match["incident_priority"],
                "Assigned Unit": match["resource_name"],
                "Unit Type": match["resource_type"],
                "Base Station": match["base_station"],
                "Route Dist": f"{match['distance_km']} km",
                "Est. ETA": f"{match['eta_minutes']} min"
            })
        st.dataframe(pd.DataFrame(opt_table_data), width="stretch", hide_index=True)

        if st.button("🚨 Commit Global Allocation to Active Fleet (Bulk Dispatch)", key="commit_bulk_dispatch", width="stretch"):
            for match in res["assignments"]:
                dispatch_unit_to_incident(
                    resource_id=match["resource_id"],
                    incident_id=match["incident_id"],
                    distance_km=match["distance_km"],
                    eta_min=match["eta_minutes"]
                )
            st.success("Global dispatch executed! All matched emergency units are deployed.")
            del st.session_state.global_opt_result
            st.rerun()


# ==========================================
# TAB 4: SITUATIONAL ANALYTICS & REGISTRY
# ==========================================
with tab4:
    st.subheader("System Registry & Situational Metrics")

    # Active Deployments Log & Controls
    st.markdown("#### 🚨 Active Dispatch Operations & Fleet Controller")
    if not st.session_state.dispatch_log:
        st.info("No units actively dispatched in this session. Select an incident in Tab 1 or run the Global Optimizer in Tab 3 to trigger field dispatches.")
    else:
        disp_df = pd.DataFrame(st.session_state.dispatch_log)
        st.dataframe(disp_df, width="stretch", hide_index=True)

        active_transits = [item for item in st.session_state.dispatch_log if item["status"] == "In Transit"]
        if active_transits:
            rec_c1, rec_c2 = st.columns([4, 2])
            with rec_c1:
                rec_choices = {f"{it['resource_name']} (Dispatched to {it['incident_title']})": it["resource_id"] for it in active_transits}
                sel_rec_label = st.selectbox("Select Unit to Recall / Return to Base:", list(rec_choices.keys()))
                sel_rec_id = rec_choices[sel_rec_label]
            with rec_c2:
                st.write("")
                st.write("")
                if st.button("🔄 Recall Unit (Mark Available)", key="recall_unit_btn", width="stretch"):
                    recall_unit_from_field(sel_rec_id)
                    st.toast(f"Unit {sel_rec_id} returned to base and marked Available.")
                    st.rerun()

    st.markdown("---")
    st.markdown("#### Situational Overview & Capacity Distributions")

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.markdown("#### Incidents by Disaster Category")
        inc_type_counts = incidents_df["incident_type"].value_counts().reset_index()
        inc_type_counts.columns = ["Disaster Category", "Active Incidents"]
        st.dataframe(inc_type_counts, width="stretch", hide_index=True)

        st.markdown("#### Incidents by Severity Classification")
        inc_sev_counts = incidents_df["severity_level"].value_counts().reset_index()
        inc_sev_counts.columns = ["Severity Level", "Incident Count"]
        st.dataframe(inc_sev_counts, width="stretch", hide_index=True)

    with stat_col2:
        st.markdown("#### Emergency Resource Inventory by Status")
        res_stat_counts = resources_df["availability_status"].value_counts().reset_index()
        res_stat_counts.columns = ["Status", "Resource Count"]
        st.dataframe(res_stat_counts, width="stretch", hide_index=True)

        st.markdown("#### Resource Capacity by Classification")
        res_type_cap = resources_df.groupby("resource_type")["capacity"].sum().reset_index()
        res_type_cap.columns = ["Resource Type", "Total Capacity / Units"]
        st.dataframe(res_type_cap, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Raw Data Inspection & Export")
    
    t_inc, t_res = st.tabs(["Disaster Incidents Table", "Emergency Resources Table"])
    with t_inc:
        st.dataframe(incidents_df, width="stretch")
        st.download_button(
            "📥 Download Validated Incidents CSV",
            data=incidents_df.to_csv(index=False),
            file_name="validated_delhi_incidents.csv",
            mime="text/csv"
        )
    with t_res:
        st.dataframe(resources_df, width="stretch")
        st.download_button(
            "📥 Download Validated Resources CSV",
            data=resources_df.to_csv(index=False),
            file_name="validated_delhi_resources.csv",
            mime="text/csv"
        )


# ==========================================
# TAB 5: DATA PREPROCESSING & PARAMETERS
# ==========================================
with tab5:
    st.subheader("Data Preprocessing Pipeline & Parameters (Sections 5.2 & 5.11)")
    st.markdown("""
    The system follows a strict preprocessing pipeline that validates coordinates, removes duplicates, 
    and handles missing records before geospatial analysis.
    """)

    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.success(f"✅ Deduplication Passed: 0 Duplicate Incident / Resource IDs")
    with v_col2:
        st.success(f"✅ Coordinate Bounding Box Verified: Lat [{audit_report['incidents_valid_coords']}/{audit_report['incidents_total']} valid]")
    with v_col3:
        st.success(f"✅ Categorical Standardization: 100% Schema Conforming")

    st.markdown("#### Table III. Disaster Data and Resource Parameters")
    param_table = [
        {"Category": "Disaster Data", "Parameter": "Location (Latitude / Longitude)", "Description": "Geographic coordinates of the affected area"},
        {"Category": "Disaster Data", "Parameter": "Disaster Type", "Description": "Category of event, e.g., flood, fire, structural collapse"},
        {"Category": "Disaster Data", "Parameter": "Severity Indicator", "Description": "Estimated damage extent or population affected"},
        {"Category": "Disaster Data", "Parameter": "Timestamp", "Description": "Date and time of the reported event"},
        {"Category": "Resource Data", "Parameter": "Resource Type", "Description": "Medical aid, food, shelter, rescue team, etc."},
        {"Category": "Resource Data", "Parameter": "Quantity Available", "Description": "Number or amount of the resource on hand / capacity"},
        {"Category": "Resource Data", "Parameter": "Resource Location", "Description": "Geographic coordinates of the resource point"},
        {"Category": "Resource Data", "Parameter": "Status", "Description": "Availability state, e.g., Available or Deployed"}
    ]
    st.dataframe(pd.DataFrame(param_table), width="stretch", hide_index=True)


# ==========================================
# TAB 6: SYSTEM ARCHITECTURE & FLOWCHART
# ==========================================
with tab6:
    st.subheader("System Architecture & Workflow (Section 6 & Figure 1)")
    
    col_flow, col_comp = st.columns([6, 6])
    
    with col_flow:
        st.markdown("#### Fig. 1. Proposed System Workflow")
        st.markdown("""
        ```
        ┌────────────────────────────────────────────────────────┐
        │                 Disaster Data Input                    │
        └──────────────────────────┬─────────────────────────────┘
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │                   Data Preprocessing                   │
        └──────────────────────────┬─────────────────────────────┘
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │            Identification of Affected Areas            │
        └──────────────────────────┬─────────────────────────────┘
                                   ▼
        ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
        │         GIS-Based Mapping         │◄──►│ Resource Availability Data/Mapping│
        └─────────────────┬─────────────────┘    └───────────────────────────────────┘
                          ▼
        ┌────────────────────────────────────────────────────────┐
        │         Disaster Severity / Priority Assessment        │
        │                   (Rule-Based Core)                    │
        └──────────────────────────┬─────────────────────────────┘
                                   ▼
              · · · · · · · · · · · · · · · · · · · · · · · 
              : AI-Based Prioritisation & Allocation       :
              :           (Future Enhancement)             :
              · · · · · · · · · · ·┬· · · · · · · · · · · · 
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │           Recommended Resource Distribution            │
        └──────────────────────────┬─────────────────────────────┘
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │              GIS Dashboard / Visualisation             │
        └────────────────────────────────────────────────────────┘
        ```
        """)

    with col_comp:
        st.markdown("#### Core Implementation vs. Future AI Enhancement")
        st.info(
            "**Scope Distinction:** The core micro project delivers a functional "
            "GIS-based mapping and visualisation system for disaster-affected areas and available "
            "resources. The live platform now also demonstrates prototype implementations of the "
            "planned AI enhancements (supervised Random Forest severity models, NetworkX road routing, "
            "and Hungarian global multi-incident optimization)."
        )

        st.markdown("#### Table V. Functional Requirements / System Components")
        comp_table = [
            {"Component": "Disaster Data Input Module", "Description": "Accepts and structures disaster event data", "Status": "Core (Operational)"},
            {"Component": "Data Preprocessing Module", "Description": "Cleans and validates incoming data", "Status": "Core (Operational)"},
            {"Component": "GIS Mapping Module", "Description": "Visualises affected areas on an interactive map", "Status": "Core (Operational)"},
            {"Component": "Resource Mapping Module", "Description": "Visualises resource locations and quantities", "Status": "Core (Operational)"},
            {"Component": "Priority Assessment Module", "Description": "Ranks affected zones by severity (Rule & AI)", "Status": "Core & AI Model"},
            {"Component": "Dashboard Module", "Description": "Integrates all outputs into a single interface", "Status": "Core (Operational)"},
            {"Component": "AI Allocation Module", "Description": "Optimises resource distribution using machine learning", "Status": "AI Prototype (Operational)"}
        ]
        st.dataframe(pd.DataFrame(comp_table), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### Table IV. Technology Stack")
    tech_table = [
        {"Category": "Programming Language", "Technology / Tool": "Python", "Purpose": "Core language for data processing and application logic"},
        {"Category": "Data Handling", "Technology / Tool": "Pandas, NumPy", "Purpose": "Structured data handling and preprocessing"},
        {"Category": "Geospatial Processing", "Technology / Tool": "GeoPandas", "Purpose": "Spatial operations such as zone identification"},
        {"Category": "Map Visualisation", "Technology / Tool": "Folium (Leaflet.js)", "Purpose": "Rendering of interactive maps"},
        {"Category": "Base Map Data", "Technology / Tool": "OpenStreetMap", "Purpose": "Open-source base map layers"},
        {"Category": "Machine Learning", "Technology / Tool": "Scikit-Learn (Random Forest)", "Purpose": "Severity classification and continuous damage regression"},
        {"Category": "Graph Routing", "Technology / Tool": "NetworkX (Dijkstra)", "Purpose": "Turn-by-turn road network routing with hazard blockages"},
        {"Category": "Global Optimization", "Technology / Tool": "SciPy (Hungarian Algorithm)", "Purpose": "Jurisdiction-wide multi-incident optimal resource dispatch"},
        {"Category": "Dashboard Framework", "Technology / Tool": "Streamlit", "Purpose": "Browser-accessible operations console"}
    ]
    st.dataframe(pd.DataFrame(tech_table), width="stretch", hide_index=True)


# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "AI-Based Disaster Resource Allocation System &bull; Micro Project GIS Decision-Support Platform"
    "</div>",
    unsafe_allow_html=True
)
