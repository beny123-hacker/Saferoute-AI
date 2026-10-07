import os
import json
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from route_engine import build_or_load_graph, find_safest_and_shortest

# ---------------------------------------------------------------------------
# 1. PAGE SETUP & DATA DIRECTORY RESOLUTION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SafeRoute AI - Route Prediction", layout="wide")

st.title("🛡️ SafeRoute AI: Risk-Aware Route Prediction System")
st.markdown("**Thoothukudi, Tamil Nadu** | AI-Powered Emergency Accessibility & Safe Path Engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") if os.path.exists(os.path.join(BASE_DIR, "data")) else BASE_DIR

# ---------------------------------------------------------------------------
# 2. CACHED ASSET LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def get_graph():
    return build_or_load_graph()

G = get_graph()

@st.cache_data
def get_facilities():
    hosp_path = os.path.join(DATA_DIR, "thoothukudi_hospitals.geojson")
    police_path = os.path.join(DATA_DIR, "thoothukudi_police.geojson")
    
    with open(hosp_path, "r", encoding="utf-8") as f:
        h_data = json.load(f)
    with open(police_path, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    return h_data, p_data

hospitals_json, police_json = get_facilities()

# ---------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------
if "route_results" not in st.session_state:
    st.session_state.route_results = None
if "current_points" not in st.session_state:
    st.session_state.current_points = None

# ---------------------------------------------------------------------------
# 4. SIDEBAR INPUTS
# ---------------------------------------------------------------------------
st.sidebar.header("📍 Trip Parameters")
preset = st.sidebar.selectbox(
    "Choose Preset Route in Thoothukudi:",
    [
        "Custom Coordinates",
        "Medical College Hospital -> Tuticorin Port Area",
        "Millerpuram -> V.E. Road Market",
        "Tiruchendur Road -> Old Bus Stand"
    ]
)

if preset == "Medical College Hospital -> Tuticorin Port Area":
    start_lat, start_lon = 8.7981, 78.1360
    end_lat, end_lon = 8.7536, 78.1857
elif preset == "Millerpuram -> V.E. Road Market":
    start_lat, start_lon = 8.7960, 78.1250
    end_lat, end_lon = 8.8023, 78.1505
elif preset == "Tiruchendur Road -> Old Bus Stand":
    start_lat, start_lon = 8.7877, 78.1408
    end_lat, end_lon = 8.8075, 78.1415
else:
    start_lat = st.sidebar.number_input("Start Latitude", value=8.7981, format="%.4f")
    start_lon = st.sidebar.number_input("Start Longitude", value=78.1360, format="%.4f")
    end_lat = st.sidebar.number_input("Destination Latitude", value=8.8060, format="%.4f")
    end_lon = st.sidebar.number_input("Destination Longitude", value=78.1480, format="%.4f")

show_hospitals = st.sidebar.checkbox("Show Hospitals", value=True)
show_police = st.sidebar.checkbox("Show Police Stations", value=True)

# ---------------------------------------------------------------------------
# 5. ROUTING INFERENCE
# ---------------------------------------------------------------------------
if st.sidebar.button("🚀 Calculate Safe Routes", type="primary"):
    with st.spinner("Calculating optimal risk-weighted paths..."):
        try:
            st.session_state.route_results = find_safest_and_shortest(G, start_lat, start_lon, end_lat, end_lon)
            st.session_state.current_points = (start_lat, start_lon, end_lat, end_lon)
        except Exception as e:
            st.error(f"Routing Error: {e}")

# ---------------------------------------------------------------------------
# 6. RESULTS & MAP VISUALIZATION
# ---------------------------------------------------------------------------
if st.session_state.route_results is not None:
    routes = st.session_state.route_results
    s_lat, s_lon, e_lat, e_lon = st.session_state.current_points

    # Route Comparison Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🔴 Shortest Route\n**(Distance Priority)**")
        st.metric("Distance", f"{routes['Shortest Route']['length_km']} km")
        st.metric("Avg Risk Score", f"{routes['Shortest Route']['avg_risk_score']}")
        st.metric("Illumination", f"{int(routes['Shortest Route']['avg_lighting_prob']*100)}%")
        
    with col2:
        st.success("### 🟢 Safest Route\n**(Maximum Security)**")
        st.metric("Distance", f"{routes['Safest Route']['length_km']} km")
        st.metric("Avg Risk Score", f"{routes['Safest Route']['avg_risk_score']}")
        st.metric("Illumination", f"{int(routes['Safest Route']['avg_lighting_prob']*100)}%")
        
    with col3:
        st.warning("### 🟡 Balanced Route\n**(Safe & Efficient)**")
        st.metric("Distance", f"{routes['Balanced Route']['length_km']} km")
        st.metric("Avg Risk Score", f"{routes['Balanced Route']['avg_risk_score']}")
        st.metric("Illumination", f"{int(routes['Balanced Route']['avg_lighting_prob']*100)}%")

    # Map Rendering using free OpenStreetMap tiles
    mid_lat = (s_lat + e_lat) / 2
    mid_lon = (s_lon + e_lon) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13, tiles="OpenStreetMap")

    # Draw Polylines for the three paths
    folium.PolyLine(
        routes["Shortest Route"]["coordinates"],
        color="#e74c3c", weight=5, opacity=0.8, tooltip="Shortest Route"
    ).add_to(m)

    folium.PolyLine(
        routes["Safest Route"]["coordinates"],
        color="#2ecc71", weight=6, opacity=0.9, tooltip="Safest Route"
    ).add_to(m)

    folium.PolyLine(
        routes["Balanced Route"]["coordinates"],
        color="#f39c12", weight=4, opacity=0.8, tooltip="Balanced Route", dash_array="5, 10"
    ).add_to(m)

    # Origin & Destination Pins
    folium.Marker([s_lat, s_lon], popup="Start Point", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([e_lat, e_lon], popup="Destination", icon=folium.Icon(color="black", icon="flag")).add_to(m)

    # Clustered Hospital Pins
    if show_hospitals:
        hosp_cluster = MarkerCluster(name="Hospitals").add_to(m)
        for feat in hospitals_json["features"]:
            coords = feat["geometry"]["coordinates"]
            name = feat.get("properties", {}).get("name", "Hospital")
            folium.Marker(
                [coords[1], coords[0]],
                popup=f"🏥 {name}",
                icon=folium.Icon(color="blue", icon="plus", prefix="fa")
            ).add_to(hosp_cluster)

    # Police Station Pins
    if show_police:
        for feat in police_json["features"]:
            coords = feat["geometry"]["coordinates"]
            name = feat.get("properties", {}).get("name", "Police Station")
            folium.Marker(
                [coords[1], coords[0]],
                popup=f"🚓 {name}",
                icon=folium.Icon(color="darkblue", icon="shield", prefix="fa")
            ).add_to(m)

    folium.LayerControl().add_to(m)

    # returned_objects=[] prevents map panning/zoom events from forcing Streamlit script re-runs
    st_folium(m, width="100%", height=600, returned_objects=[])

else:
    st.info("Select trip endpoints on the sidebar and click **'Calculate Safe Routes'** to run the routing engine.")