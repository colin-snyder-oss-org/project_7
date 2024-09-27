# src/visualization/visualize.py
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_networkx
from itertools import cycle

class Visualizer:
    @staticmethod
    def plot_graph(data):
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=50, with_labels=False)
        plt.show()

    @staticmethod
    def plot_embedding(embedding, labels):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        transformed = tsne.fit_transform(embedding)
        scatter = plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='rainbow', s=5)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.show()
