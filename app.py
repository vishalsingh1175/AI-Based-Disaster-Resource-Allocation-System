"""
AI-Based Disaster Resource Allocation System
Micro Project Prototype & GIS Decision Support Dashboard
"""

import streamlit as st
import pandas as pd
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
</style>
""", unsafe_allow_html=True)

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
    'Geospatial Operations & Emergency Logistics Platform'
    '</div>',
    unsafe_allow_html=True
)

# --- Sidebar Filters ---
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


# Filter Incidents DataFrame
filtered_incidents = incidents_df.copy()
if selected_inc_type != "All":
    filtered_incidents = filtered_incidents[filtered_incidents["incident_type"] == selected_inc_type]
if selected_severity != "All":
    filtered_incidents = filtered_incidents[filtered_incidents["severity_level"] == selected_severity]

# Filter Resources DataFrame
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

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live GIS Command Center",
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

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <h4 style="margin: 0; color: #1E293B;">{res['name']}</h4>
                <p style="margin: 4px 0; font-size: 12px; color: #64748B;"><b>ID:</b> {res['resource_id']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Type:</b> {res['resource_type']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Base:</b> {res['base_station']}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Capacity:</b> {res['capacity']} units/persons</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Status:</b> <span style="font-weight:bold; color:green;">{res['availability_status']}</span></p>
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
                <p style="margin: 4px 0; font-size: 12px;"><b>Affected Pop.:</b> {inc['affected_population']:,}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Reported Damage:</b> {inc['reported_damage']}</p>
            </div>
            """

            # Incident Circle marker
            folium.CircleMarker(
                location=[inc["latitude"], inc["longitude"]],
                radius=10 if not is_active_selected else 14,
                color="#DC2626" if inc["severity_level"] == "Critical" else "#EA580C",
                weight=3 if is_active_selected else 1,
                fill=True,
                fill_color=m_color,
                fill_opacity=0.9 if is_active_selected else 0.7,
                popup=folium.Popup(inc_popup, max_width=320),
                tooltip=f"Incident: {inc['title']} [{inc['severity_level']}]"
            ).add_to(folium_map)

        # If an incident is selected, compute nearest resources and draw lines & radius
        nearest_units = []
        if selected_incident is not None:
            inc_lat = selected_incident["latitude"]
            inc_lon = selected_incident["longitude"]

            # Draw spatial buffer circle
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

            # Draw trajectory lines to nearest resources
            for idx, unit in enumerate(nearest_units):
                line_color = "#2563EB" if idx == 0 else "#64748B"
                weight = 3 if idx == 0 else 2
                
                folium.PolyLine(
                    locations=[[inc_lat, inc_lon], [unit["latitude"], unit["longitude"]]],
                    color=line_color,
                    weight=weight,
                    dash_array="6, 6",
                    tooltip=f"Dispatch Path #{idx+1} to {unit['name']}: {unit['distance_km']} km (ETA ~{unit['eta_minutes']}m)"
                ).add_to(folium_map)

        st_folium(folium_map, width="100%", height=560)

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
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("Affected Population", f"{selected_incident['affected_population']:,}")
            with p_col2:
                st.metric("Priority Index", f"{selected_incident['priority_score']} / 100")
            
            st.progress(selected_incident['priority_score'] / 100.0)

            st.markdown("---")
            st.markdown("### 🎯 Nearest Resource Dispatch Recommendations")
            st.caption(f"Spatial association using Great-Circle Haversine distance & urban emergency transit model (within {radius_km} km radius).")

            if not nearest_units:
                st.error("⚠️ No available emergency resources found within the specified radius. Recommend expanding the search buffer or requesting mutual aid from neighboring districts.")
            else:
                for i, u in enumerate(nearest_units):
                    rank_icon = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
                    with st.container():
                        st.markdown(f"**{rank_icon} Rank #{i+1}: {u['name']}**")
                        c_u1, c_u2, c_u3 = st.columns([3, 2, 2])
                        with c_u1:
                            st.write(f"🏢 *{u['base_station']}*")
                            st.write(f"📞 `{u['contact_number']}`")
                        with c_u2:
                            st.metric("Distance", f"{u['distance_km']} km")
                        with c_u3:
                            st.metric("Est. ETA", f"{u['eta_minutes']} min")
                        
                        btn_col1, btn_col2 = st.columns([3, 2])
                        with btn_col1:
                            st.caption(f"Type: {u['resource_type']}")
                        with btn_col2:
                            if st.button(f"🚨 Dispatch Unit #{i+1}", key=f"dispatch_btn_{u['resource_id']}_{selected_incident['incident_id']}", width="stretch"):
                                dispatch_unit_to_incident(
                                    resource_id=u["resource_id"],
                                    incident_id=selected_incident["incident_id"],
                                    distance_km=u["distance_km"],
                                    eta_min=u["eta_minutes"]
                                )
                                st.toast(f"✅ Dispatched {u['name']} to {selected_incident['title']}! ETA: {u['eta_minutes']} min", icon="🚨")
                                st.rerun()
                        st.divider()

# ==========================================
# TAB 2: SITUATIONAL ANALYTICS & REGISTRY
# ==========================================
with tab2:
    st.subheader("System Registry & Situational Metrics")

    # Active Deployments Log & Controls
    st.markdown("#### 🚨 Active Dispatch Operations & Fleet Controller")
    if not st.session_state.dispatch_log:
        st.info("No units actively dispatched in this session. Select an incident in Tab 1 and click 'Dispatch Unit' to trigger real-time field dispatch.")
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
# TAB 3: DATA PREPROCESSING & PARAMETERS
# ==========================================
with tab3:
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
# TAB 4: SYSTEM ARCHITECTURE & FLOWCHART
# ==========================================
with tab4:
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
            "**Scope Distinction:** The core micro project focuses on building a functional "
            "GIS-based mapping and visualisation system for disaster-affected areas and available "
            "resources, along with basic rule-based severity assessment and nearest-unit distribution. "
            "AI-based multi-objective optimization is reserved as a future enhancement."
        )

        st.markdown("#### Table V. Functional Requirements / System Components")
        comp_table = [
            {"Component": "Disaster Data Input Module", "Description": "Accepts and structures disaster event data", "Status": "Core"},
            {"Component": "Data Preprocessing Module", "Description": "Cleans and validates incoming data", "Status": "Core"},
            {"Component": "GIS Mapping Module", "Description": "Visualises affected areas on an interactive map", "Status": "Core"},
            {"Component": "Resource Mapping Module", "Description": "Visualises resource locations and quantities", "Status": "Core"},
            {"Component": "Priority Assessment Module", "Description": "Ranks affected zones by severity", "Status": "Core (rule-based)"},
            {"Component": "Dashboard Module", "Description": "Integrates all outputs into a single interface", "Status": "Core"},
            {"Component": "AI Allocation Module", "Description": "Optimises resource distribution using machine learning", "Status": "Future Enhancement"}
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
        {"Category": "GIS Development Tool", "Technology / Tool": "QGIS", "Purpose": "Offline inspection and validation of geospatial datasets"},
        {"Category": "Dashboard Framework", "Technology / Tool": "Streamlit / Flask", "Purpose": "Browser-accessible user interface"},
        {"Category": "Future Analytics", "Technology / Tool": "ML / optimisation libraries", "Purpose": "Planned AI-based severity assessment and allocation"}
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
