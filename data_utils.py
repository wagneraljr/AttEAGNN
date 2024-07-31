import networkx as nx
import numpy as np
import torch
import csv
import os
import pandas as pd

# Generates graph information, node and edge features from a .gml file
def load_data(filepath):
    # Creates the RNP network graph from the .gml file
    G = nx.read_gml(filepath)

    # Maps node labels to integer indices
    label_to_index = {label: idx for idx, label in enumerate(G.nodes())}
    
    # Updates node labels to integer values
    G = nx.relabel_nodes(G, label_to_index)

    # Extract node features (one-hot encoding)
    node_features = np.eye(G.number_of_nodes())
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Extract edge feautures
    edge_indices = np.array(G.edges())
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Calculate implicit edge features
    # Calculate edge betweenness 
    edge_betweenness = nx.edge_betweenness_centrality(G)

    # Appends edge features
    edge_features = []
    for edge in G.edges(data=True):
        src, dst = edge[0], edge[1]
        
        # Explicit features, or a vector of zeros if none
        feature = edge[2].get('feature', [0]*8)
        
        # Edge betweenness
        feature.append(edge_betweenness[(src, dst)])
        
        # Edge degree, defined as the sum of the degrees of its nodes
        feature.append(G.degree[src] + G.degree[dst])
        
        # Edge clustering coefficient, defined as the average of its nodes coefficients
        clustering_src = nx.clustering(G, src)
        clustering_dst = nx.clustering(G, dst)
        avg_clustering = (clustering_src + clustering_dst) / 2
        feature.append(avg_clustering)
        
        edge_features.append(feature)
    
    # Convert edge features to tensor
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    return node_features, edge_indices, edge_features

# Auxiliary function for CAGNN graph processing
def create_graphs_from_nx(G):
    num_nodes = G.number_of_nodes()
    edges = list(G.edges())
    
    node_neighbors = [[] for _ in range(num_nodes)]
    edge_neighbors = [[] for _ in range(len(edges))]

    node_index = {node: i for i, node in enumerate(G.nodes())}

    for idx, (u, v) in enumerate(edges):
        node_neighbors[node_index[u]].append((node_index[v], idx))
        node_neighbors[node_index[v]].append((node_index[u], idx))

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edges[i][1] == edges[j][0]:
                edge_neighbors[i].append(j)
                edge_neighbors[j].append(i)

    return node_neighbors, edge_neighbors

# Generates graph information for the CAGNN model
def load_data_cagnn(filepath):
    # Creates the RNP network graph from the .gml file
    G = nx.read_gml(filepath)
    
    # Maps node labels to integer indices
    label_to_index = {label: idx for idx, label in enumerate(G.nodes())}
    
    # Updates node labels to integer values
    G = nx.relabel_nodes(G, label_to_index)
    
    # Extract node features (one-hot encoding)
    node_features = np.eye(G.number_of_nodes())
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Extract edge feautures
    edge_indices = np.array(G.edges())
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # Calculate implicit edge features
    # Calculate edge betweenness 
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # Appends edge features
    edge_features = []
    for edge in G.edges(data=True):
        src, dst = edge[0], edge[1]
        
        # Explicit features, or a vector of zeros if none
        feature = edge[2].get('feature', [0]*8)
        
        # Edge betweenness
        feature.append(edge_betweenness[(src, dst)])
        
        # Edge degree, defined as the sum of the degrees of its nodes
        feature.append(G.degree[src] + G.degree[dst])
        
        # Edge clustering coefficient, defined as the average of its nodes coefficients
        clustering_src = nx.clustering(G, src)
        clustering_dst = nx.clustering(G, dst)
        avg_clustering = (clustering_src + clustering_dst) / 2
        feature.append(avg_clustering)
        
        edge_features.append(feature)

    # Convert edge features to tensor
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    # Create lists of node and edge neighbors for the CAGNN model
    node_neighbors, edge_neighbors = create_graphs_from_nx(G)
    
    return node_features, edge_indices, edge_features, node_neighbors, edge_neighbors

def get_node_loads(traffic_matrix_filepath):
    
    # Process .dat files to extract traffic matrix data
    traffic_matrix = []
    with open(traffic_matrix_filepath, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row[0].startswith("#"):
                traffic_matrix.append([float(value) for value in row])
   
    # Get node loads
    node_loads_values = []
    for i in range(len(traffic_matrix)):
        node_load = sum(traffic_matrix[i]) + sum(row[i] for row in traffic_matrix)
        node_loads_values.append(node_load)
     
    # Get total load for normalization
    total_load = sum(node_loads_values)

    # Normalize node loads
    normalized_node_loads = [load / total_load for load in node_loads_values] 

    # Converts node_loads to tensor format
    node_loads = torch.tensor(normalized_node_loads, dtype=torch.float32).view(-1, 1)
    
    return node_loads

# Auxiliary function to normalize edge and node features
def normalize_features(features):
    return (features - features.mean(dim=0)) / features.std(dim=0)

# Function to convert data from .dat files to pandas
def load_traffic_pd(file_path):
    # Ignores comment lines
    data = pd.read_csv(file_path, comment='#', header=None)
    # Converts to pandas format
    numeric_data = data.apply(pd.to_numeric, errors='coerce', axis=1)
    numeric_data = normalize_features(numeric_data)
    return numeric_data
