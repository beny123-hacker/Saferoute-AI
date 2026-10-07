import os
import json
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import shape, Polygon, LineString
from scipy.spatial import cKDTree

print("=== SafeRoute AI:- Routing Graph Engine ===")

GRAPH_CACHE = "graph_network.pkl"

def build_or_load_graph():
    if os.path.exists(GRAPH_CACHE):
        print(f"Loading cached graph network from '{GRAPH_CACHE}'...")
        return joblib.load(GRAPH_CACHE)
    
    print("Building road network graph from OSM GeoJSON and Master Dataset...")
    
    # 1. Load Master Dataset
    df = pd.read_csv("safe_route_master_dataset.csv", encoding="utf-8")
    df_indexed = df.set_index("road_id")
    
    # 2. Load Road Geometries
    with open("thoothukudi_roads.geojson", "r", encoding="utf-8") as f:
        roads_json = json.load(f)
        
    G = nx.Graph()
    
    for feat in roads_json["features"]:
        props = feat.get("properties", {})
        r_id = props.get("@id", f"way/{feat.get('id', 'unknown')}")
        
        if r_id not in df_indexed.index:
            continue
            
        geom = shape(feat["geometry"])
        if isinstance(geom, Polygon):
            geom = geom.exterior
            
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
            
        row = df_indexed.loc[r_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
            
        length_m = float(row["length_meters"])
        risk_score = float(row["risk_score"])
        risk_class = str(row["risk_class"])
        hosp_dist = float(row["hospital_dist_m"])
        pol_dist = float(row["police_dist_m"])
        light_prob = float(row["lighting_probability"])
        
        # Calculate dynamic edge costs
        cost_shortest = length_m
        cost_safest = length_m * (1.0 + 4.0 * (risk_score ** 1.5))
        cost_balanced = length_m * (1.0 + 1.5 * risk_score)
        
        total_geom_len = geom.length if geom.length > 0 else 1.0
        
        # Segment-by-segment edge insertion ensures intersections connect accurately
        for i in range(len(coords) - 1):
            u = (round(coords[i][0], 6), round(coords[i][1], 6))
            v = (round(coords[i+1][0], 6), round(coords[i+1][1], 6))
            
            sub_len = LineString([coords[i], coords[i+1]]).length
            frac = sub_len / total_geom_len
            
            G.add_edge(
                u, v,
                road_id=r_id,
                road_name=str(row["road_name"]),
                highway_type=str(row["highway_type"]),
                length_m=length_m * frac,
                cost_shortest=cost_shortest * frac,
                cost_safest=cost_safest * frac,
                cost_balanced=cost_balanced * frac,
                risk_score=risk_score,
                risk_class=risk_class,
                hospital_dist_m=hosp_dist,
                police_dist_m=pol_dist,
                lighting_prob=light_prob
            )
            
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    joblib.dump(G, GRAPH_CACHE)
    print(f"Graph successfully cached to '{GRAPH_CACHE}'.")
    return G

# ---------------------------------------------------------------------------
# PATHFINDING AND METRIC EXTRACTION
# ---------------------------------------------------------------------------
def compute_route_metrics(G, path):
    total_len = 0.0
    weighted_risk = 0.0
    hosp_dists = []
    pol_dists = []
    lighting_scores = []
    route_coords = []
    
    for u, v in zip(path[:-1], path[1:]):
        edge = G[u][v]
        l = edge["length_m"]
        total_len += l
        weighted_risk += edge["risk_score"] * l
        hosp_dists.append(edge["hospital_dist_m"])
        pol_dists.append(edge["police_dist_m"])
        lighting_scores.append(edge["lighting_prob"])
        route_coords.append([u[1], u[0]]) # [lat, lon] for Folium maps
        
    route_coords.append([path[-1][1], path[-1][0]])
    
    avg_risk = weighted_risk / total_len if total_len > 0 else 0.0
    avg_light = np.mean(lighting_scores) if lighting_scores else 0.5
    
    return {
        "length_km": round(total_len / 1000.0, 2),
        "avg_risk_score": round(avg_risk, 4),
        "avg_lighting_prob": round(avg_light, 2),
        "closest_hospital_m": round(min(hosp_dists), 1) if hosp_dists else 0.0,
        "closest_police_m": round(min(pol_dists), 1) if pol_dists else 0.0,
        "coordinates": route_coords
    }

def find_safest_and_shortest(G, start_lat, start_lon, end_lat, end_lon):
    # Match arbitrary user GPS coordinates to the nearest valid intersection nodes
    nodes = list(G.nodes())
    node_coords = np.array(nodes) # [lon, lat]
    tree = cKDTree(node_coords)
    
    _, start_idx = tree.query([start_lon, start_lat])
    _, end_idx = tree.query([end_lon, end_lat])
    
    start_node = nodes[start_idx]
    end_node = nodes[end_idx]
    
    # Check graph connectivity
    if not nx.has_path(G, start_node, end_node):
        raise nx.NetworkXNoPath("No navigable road connection exists between the selected points.")
        
    # Multi-objective path calculation
    path_shortest = nx.shortest_path(G, source=start_node, target=end_node, weight="cost_shortest")
    path_safest = nx.shortest_path(G, source=start_node, target=end_node, weight="cost_safest")
    path_balanced = nx.shortest_path(G, source=start_node, target=end_node, weight="cost_balanced")
    
    return {
        "Shortest Route": compute_route_metrics(G, path_shortest),
        "Safest Route": compute_route_metrics(G, path_safest),
        "Balanced Route": compute_route_metrics(G, path_balanced)
    }

if __name__ == "__main__":
    graph = build_or_load_graph()
    
    # Test coordinates across central Thoothukudi
    # Start: Subbiah Puram / Medical College area (8.7981°N, 78.1360°E)
    # End: Near V.E. Road / Railway Station area (8.8060°N, 78.1480°E)
    s_lat, s_lon = 8.7981, 78.1360
    e_lat, e_lon = 8.8060, 78.1480
    
    print(f"\nComputing routes from ({s_lat}, {s_lon}) to ({e_lat}, {e_lon})...")
    results = find_safest_and_shortest(graph, s_lat, s_lon, e_lat, e_lon)
    
    print("\n" + "="*60)
    for route_type, metrics in results.items():
        print(f"[{route_type}]")
        print(f"  * Total Distance:      {metrics['length_km']} km")
        print(f"  * Average Risk Score:  {metrics['avg_risk_score']} (Lower is safer)")
        print(f"  * Lighting Prob:       {int(metrics['avg_lighting_prob']*100)}% illuminated")
        print(f"  * Nearest Hospital:    {metrics['closest_hospital_m']} m")
        print(f"  * Nearest Police:      {metrics['closest_police_m']} m")
        print("-" * 60)