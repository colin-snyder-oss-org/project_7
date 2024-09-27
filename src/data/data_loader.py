# src/data/data_loader.py
import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset
import networkx as nx
import pandas as pd
import numpy as np
import os
from functools import lru_cache

class SocialNetworkDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SocialNetworkDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graph.gpickle', 'node_features.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @lru_cache(maxsize=None)
    def download(self):
        pass  # Assume data is already downloaded

    def process(self):
        # Load data
        graph = nx.read_gpickle(os.path.join(self.raw_dir, 'graph.gpickle'))
        node_features = pd.read_pickle(os.path.join(self.raw_dir, 'node_features.pkl'))

        # Convert to PyTorch Geometric data
        x = torch.tensor(node_features.values, dtype=torch.float)
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        data = Data(x=x, edge_index=edge_index)

        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_data_loader(config):
    dataset = SocialNetworkDataset(root=config.data_path)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    return loader
