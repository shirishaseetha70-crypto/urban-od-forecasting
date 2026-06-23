import os
import sys
import pickle
import pandas as pd
import numpy as np
import networkx as nx

# ── Path fix ──────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():
    zones_path = os.path.join(DATASET_DIR, "zone_metadata.csv")
    cleaned_path = os.path.join(DATASET_DIR, "od_cleaned.csv")
    output_path = os.path.join(DATASET_DIR, "road_graph.pkl")

    if not os.path.exists(zones_path):
        print(f"ERROR: {zones_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    if not os.path.exists(cleaned_path):
        print(f"ERROR: {cleaned_path} not found. Run dtw_cleaner.py first.")
        sys.exit(1)

    zones_df = pd.read_csv(zones_path)
    df = pd.read_csv(cleaned_path)

    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in zones_df.iterrows():
        G.add_node(
            row['zone_id'],
            lat=row['lat'],
            lon=row['lon'],
            zone_type=row['type']
        )

    # Add edges weighted by average OD flow
    print("Computing average OD flows for edges...")
    edge_flows = df.groupby(['origin_zone', 'dest_zone'])['od_flow'].mean().reset_index()

    edges_added = 0
    for _, row in edge_flows.iterrows():
        if row['od_flow'] > 5:
            G.add_edge(
                row['origin_zone'],
                row['dest_zone'],
                weight=float(row['od_flow'])
            )
            edges_added += 1

    with open(output_path, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph built successfully!")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Saved to: {output_path}")
    return G


if __name__ == "__main__":
    build_graph()
