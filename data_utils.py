import networkx as nx
import numpy as np
import torch
import csv
import os
import pandas as pd

# Função para carregar dados de um grafo e uma matriz de tráfego a partir de arquivos
def load_data(filepath):
    # Carrega o grafo da rede Abilene a partir de um arquivo GML
    G = nx.read_gml(filepath)

    # Cria um mapeamento de rótulos de nós para índices inteiros
    label_to_index = {label: idx for idx, label in enumerate(G.nodes())}
    
    # Atualiza o grafo para usar rótulos inteiros
    G = nx.relabel_nodes(G, label_to_index)

    # Extrai atributos dos nós (aqui, usando uma codificação one-hot)
    node_features = np.eye(G.number_of_nodes())
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Extrai índices das arestas
    edge_indices = np.array(G.edges())
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Calcula atributos implícitos das arestas
    # Calcula a centralidade de intermediação das arestas
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # Calcula atributos das arestas
    edge_features = []
    for edge in G.edges(data=True):
        src, dst = edge[0], edge[1]
        
        # Atributo original, caso não exista, usa um vetor de 8 zeros
        feature = edge[2].get('feature', [0]*8)
        #feature = edge[2].get('feature')
        
        # Centralidade de intermediação
        feature.append(edge_betweenness[(src, dst)])
        
        # Grau da aresta (soma dos graus dos dois nós conectados pela aresta)
        feature.append(G.degree[src] + G.degree[dst])
        
        # Coeficiente de agrupamento
        clustering_src = nx.clustering(G, src)
        clustering_dst = nx.clustering(G, dst)
        avg_clustering = (clustering_src + clustering_dst) / 2
        feature.append(avg_clustering)
        
        edge_features.append(feature)
    
    # Converte características das arestas para um tensor
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    return node_features, edge_indices, edge_features

def load_data_cagnn(filepath):
    # Carrega o grafo da rede Abilene a partir de um arquivo GML
    G = nx.read_gml(filepath)
    
    # Cria um mapeamento de rótulos de nós para índices inteiros
    label_to_index = {label: idx for idx, label in enumerate(G.nodes())}
    
    # Atualiza o grafo para usar rótulos inteiros
    G = nx.relabel_nodes(G, label_to_index)
    
    # Extrai atributos dos nós (aqui, usando uma codificação one-hot)
    node_features = np.eye(G.number_of_nodes())
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Extrai índices das arestas
    edge_indices = np.array(G.edges())
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # Calcula atributos implícitos das arestas
    # Calcula a centralidade de intermediação das arestas
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # Calcula atributos das arestas
    edge_features = []
    for edge in G.edges(data=True):
        src, dst = edge[0], edge[1]
        
        # Atributo original, caso não exista, usa um vetor de 8 zeros
        feature = edge[2].get('feature', [0]*8)
        
        # Centralidade de intermediação
        feature.append(edge_betweenness[(src, dst)])
        
        # Grau da aresta (soma dos graus dos dois nós conectados pela aresta)
        feature.append(G.degree[src] + G.degree[dst])
        
        # Coeficiente de agrupamento
        clustering_src = nx.clustering(G, src)
        clustering_dst = nx.clustering(G, dst)
        avg_clustering = (clustering_src + clustering_dst) / 2
        feature.append(avg_clustering)
        
        edge_features.append(feature)

    # Converte características das arestas para um tensor
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    # Cria listas de vizinhos para nós e arestas
    node_neighbors, edge_neighbors = create_graphs_from_nx(G)
    
    return node_features, edge_indices, edge_features, node_neighbors, edge_neighbors

def get_node_loads(traffic_matrix_filepath):
    
    # Processa o arquivo para extrair os valores da matriz de tráfego
    traffic_matrix = []
    with open(traffic_matrix_filepath, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row[0].startswith("#"):
                traffic_matrix.append([float(value) for value in row])

#    traffic_matrix = [row[1:] for row in traffic_matrix[1:]]
    
    # Calcula cargas dos nós
    node_loads_values = []
    for i in range(len(traffic_matrix)):
        node_load = sum(traffic_matrix[i]) + sum(row[i] for row in traffic_matrix)
        node_loads_values.append(node_load)
     
    # Calcula a carga total
    total_load = sum(node_loads_values)

    # Normaliza as cargas dos nós
    normalized_node_loads = [load / total_load for load in node_loads_values] 
    
    node_loads = torch.tensor(normalized_node_loads, dtype=torch.float32).view(-1, 1)
    
    return node_loads

def normalize_features(features):
    return (features - features.mean()) / features.std()

# Função para carregar dados de um arquivo .dat, ignorando cabeçalhos e linhas de comentário
def load_traffic_pd(file_path):
    # Ignorar linhas de comentário e carregar apenas os dados numéricos
    data = pd.read_csv(file_path, comment='#', header=None)
    #data = data.iloc[1:, 1:]
    # Converter dados para um DataFrame numérico
    numeric_data = data.apply(pd.to_numeric, errors='coerce', axis=1)
    numeric_data = normalize_features(numeric_data)
    return numeric_data
