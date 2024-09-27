# src/models/gnn_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from abc import ABCMeta, abstractmethod

class MetaModel(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x, edge_index):
        pass

class ComplexGNNModel(nn.Module, MetaModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dynamically select convolutional layer
        conv_layer = self._select_conv_layer()

        self.convs = nn.ModuleList()
        for i in range(config.num_layers):
            in_dim = config.input_dim if i == 0 else config.hidden_dim
            out_dim = config.output_dim if i == config.num_layers - 1 else config.hidden_dim
            self.convs.append(conv_layer(in_dim, out_dim))

        self.activations = nn.ModuleList([nn.ReLU() for _ in range(config.num_layers - 1)])

    def _select_conv_layer(self):
        conv_dict = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'GraphSAGE': SAGEConv
        }
        return conv_dict.get(self.config.model_name, GCNConv)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activations[i](x)
        return x
