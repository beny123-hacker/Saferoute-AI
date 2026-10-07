import joblib
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load the graph network
G = joblib.load("graph_network.pkl")

print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")

# 2. Inspect a single road edge to see how Random Forest costs are applied
sample_edge = list(G.edges(data=True))[100]
u, v, data = sample_edge

print("\n--- SAMPLE GRAPH EDGE DATA ---")
print(f"Connecting Junctions: Node A {u} -> Node B {v}")
print(f"Road Name:            {data.get('road_name')}")
print(f"Highway Type:         {data.get('highway_type')}")
print(f"Physical Length:      {data.get('length_m'):.2f} meters")
print(f"Predicted Risk Score: {data.get('risk_score'):.4f}")
print(f"Shortest Cost:        {data.get('cost_shortest'):.2f}")
print(f"Safest Cost:          {data.get('cost_safest'):.2f}  <-- Penalized by Random Forest")

# 3. Plot a localized geographic subgraph of Thoothukudi center
# Filter nodes around Thoothukudi city center (Lat: 8.795 to 8.810, Lon: 78.130 to 78.145)
sub_nodes = [
    n for n in G.nodes() 
    if 8.795 <= n[1] <= 8.810 and 78.130 <= n[0] <= 78.145
]
sub_G = G.subgraph(sub_nodes)

print(f"\nVisualizing central neighborhood: {sub_G.number_of_nodes()} nodes, {sub_G.number_of_edges()} street edges...")

# Geographic positions are simply the node coordinates (lon, lat)
pos = {node: (node[0], node[1]) for node in sub_G.nodes()}

# Color edges by their predicted risk score
edge_risks = [sub_G[u][v].get("risk_score", 0.5) for u, v in sub_G.edges()]

plt.figure(figsize=(12, 10))
edges = nx.draw_networkx_edges(
    sub_G,
    pos,
    edge_color=edge_risks,
    edge_cmap=plt.cm.RdYlGn_r,  # Green = Low Risk, Red = High Risk
    width=2.0
)
nx.draw_networkx_nodes(sub_G, pos, node_size=10, node_color="blue", alpha=0.5)

cbar = plt.colorbar(edges)
cbar.set_label("Random Forest Predicted Risk Score")
plt.title("Thoothukudi Central Road Graph (Color-Coded by Edge Risk)", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("road_network_graph.png", dpi=200)
print("Saved local graph visualization as 'road_network_graph.png'. Open it in VS Code!")