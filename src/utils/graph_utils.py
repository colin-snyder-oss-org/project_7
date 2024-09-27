# src/utils/graph_utils.py
import networkx as nx
from functools import lru_cache

class GraphUtils:
    @staticmethod
    @lru_cache(maxsize=None)
    def compute_shortest_path_lengths(graph):
        return dict(nx.all_pairs_shortest_path_length(graph))

    @staticmethod
    def recursive_node_count(graph):
        def count_nodes(node, visited):
            if node in visited:
                return 0
            visited.add(node)
            count = 1
            for neighbor in graph.neighbors(node):
                count += count_nodes(neighbor, visited)
            return count
        visited = set()
        total_nodes = sum(count_nodes(node, visited) for node in graph.nodes if node not in visited)
        return total_nodes
