import numpy as np
import pandas as pd
import os
import networkx as nx
from create_graph import get_geographical_nx_graph_from_json
from src.constants import Constants

def simulate_bimodal_traffic(G, num_intervals, K, output_dir):
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    for interval in range(num_intervals):
        traffic_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    s = np.random.uniform(0, 1)
                    if s > 0.8:
                        traffic_matrix[i, j] = np.random.normal(400, 100)
                    else:
                        traffic_matrix[i, j] = np.random.normal(800, 100)
        
        df = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
        file_path = os.path.join(output_dir, f"bimodal_tm_{interval}.dat")
        df.to_csv(file_path, sep=',', header=False, index=False)

def simulate_gravity_traffic(G, num_intervals, K, output_dir, cyclical=True, q=5):
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    def gravity_demand():
        traffic_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    Mi = sum([G[nodes[i]][k]['bw'] for k in G[nodes[i]]])
                    Mj = sum([G[k][nodes[j]]['bw'] for k in G[nodes[j]]])
                    Dij = G[nodes[i]][nodes[j]]['dist'] if G.has_edge(nodes[i], nodes[j]) else 1
                    traffic_matrix[i, j] = K * Mi * Mj / (Dij ** 2)
        return traffic_matrix
    
    if cyclical:
        base_matrices = [gravity_demand() for _ in range(q)]
        for interval in range(num_intervals):
            traffic_matrix = base_matrices[interval % q]
            df = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"gravity_cyclical_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)
    else:
        traffic_matrices = [gravity_demand() for _ in range(num_intervals)]
        for interval in range(num_intervals):
            avg_matrix = np.mean(traffic_matrices[max(0, interval - q + 1):interval + 1], axis=0)
            df = pd.DataFrame(avg_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"gravity_averaging_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)

def sparsify_traffic_matrix(matrix, p):
    sparsified_matrix = np.copy(matrix)
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if np.random.uniform(0, 1) > p:
                sparsified_matrix[i, j] = 0
    return sparsified_matrix

def simulate_sparsified_gravity_traffic(G, num_intervals, K, output_dir, p, cyclical=True, q=5):
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    def gravity_demand():
        traffic_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    Mi = sum([G[nodes[i]][k]['bw'] for k in G[nodes[i]]])
                    Mj = sum([G[k][nodes[j]]['bw'] for k in G[nodes[j]]])
                    Dij = G[nodes[i]][nodes[j]]['dist'] if G.has_edge(nodes[i], nodes[j]) else 1
                    traffic_matrix[i, j] = K * Mi * Mj / (Dij ** 2)
        return traffic_matrix
    
    if cyclical:
        base_matrices = [sparsify_traffic_matrix(gravity_demand(), p) for _ in range(q)]
        for interval in range(num_intervals):
            traffic_matrix = base_matrices[interval % q]
            df = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"sparsified_gravity_cyclical_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)
    else:
        traffic_matrices = [sparsify_traffic_matrix(gravity_demand(), p) for _ in range(num_intervals)]
        for interval in range(num_intervals):
            avg_matrix = np.mean(traffic_matrices[max(0, interval - q + 1):interval + 1], axis=0)
            df = pd.DataFrame(avg_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"sparsified_gravity_averaging_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)

# Initializes graph structure
G = get_geographical_nx_graph_from_json(Constants.geographical_data_filename)
# Sets output directory
output_dir = "./traffic_matrices"
os.makedirs(output_dir, exist_ok=True)

# Create one month's worth of TMs with cycles of 7 days
simulate_bimodal_traffic(G, 8640, 1, output_dir)

simulate_gravity_traffic(G, 8640, 1, output_dir, cyclical=True, q=2016)
simulate_gravity_traffic(G, 8640, 1, output_dir, cyclical=False, q=2016)

simulate_sparsified_gravity_traffic(G, 8640, 1, output_dir, p=0.5, cyclical=True, q=2016)
simulate_sparsified_gravity_traffic(G, 8640, 1, output_dir, p=0.5, cyclical=False, q=2016)
