# scripts/preprocess_data.py
import os
import networkx as nx
import pandas as pd
from tqdm import tqdm

def preprocess():
    raw_data_dir = 'data/raw/'
    processed_data_dir = 'data/processed/'
    os.makedirs(processed_data_dir, exist_ok=True)

    # Load raw data (assuming edge list format)
    edge_list = pd.read_csv(os.path.join(raw_data_dir, 'edges.csv'))

    # Build the graph
    G = nx.from_pandas_edgelist(edge_list, source='source', target='target', edge_attr=True)

    # Save the graph
    nx.write_gpickle(G, os.path.join(processed_data_dir, 'graph.gpickle'))

    # Additional preprocessing steps (e.g., node features, labels)
    node_features = pd.read_csv(os.path.join(raw_data_dir, 'node_features.csv'))
    node_features.to_pickle(os.path.join(processed_data_dir, 'node_features.pkl'))

if __name__ == "__main__":
    preprocess()
