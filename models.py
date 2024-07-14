import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv, global_mean_pool
import numpy as np

# Classe que implementa um mecanismo de atenção para arestas em redes neurais baseadas em grafos
class EdgeAttention(nn.Module):
    def __init__(self, edge_feature_dim, hidden_dim):
        super(EdgeAttention, self).__init__()
        # Uma camada linear que transforma os atributos das arestas
        self.edge_weight = nn.Linear(edge_feature_dim, hidden_dim)
        # Função de ativação LeakyReLU
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, edge_features):
        # Aplica a transformação linear aos atributos das arestas
        edge_transformed = self.edge_weight(edge_features)
        # Calcula a pontuação de atenção para cada aresta, usando produto escalar
        attention_scores = (edge_transformed * edge_transformed).sum(dim=1)
        # Aplica a função de ativação LeakyReLU às pontuações
        attention_scores = self.leaky_relu(attention_scores)
        # Normaliza as pontuações usando softmax para obter coeficientes de atenção
        attention_coeffs = F.softmax(attention_scores, dim=0)
        return attention_coeffs
    
# Classe que define um modelo GCN ciente dos atributos das arestas com mecanismo de atenção
class AttEdgeAwareGNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim, dropout=0.5):
        super(AttEdgeAwareGNN, self).__init__()
        print('node:', node_input_dim)
        print('output:', output_dim)
        # Define camadas convolucionais de grafos para os atributos dos nós
        self.gc1 = GraphConv(node_input_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        # Define uma camada SAGE para os atributos das arestas
        self.edge_gcn = SAGEConv(edge_input_dim, hidden_dim)
        # Inicializa o mecanismo de atenção para as arestas
        self.edge_attention = EdgeAttention(edge_input_dim, hidden_dim)
        #self.edge_attention = EdgeAttention(edge_input_dim, hidden_dim)
        # Camada linear para combinar os atributos dos nós e arestas
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, node_features, edge_index, edge_features):
        # Aplica a primeira camada convolucional e ativação ReLU aos nós
        x = F.relu(self.gc1(node_features, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # Aplica a segunda camada convolucional aos nós
        x = F.relu(self.gc2(x, edge_index))
        
        # Aplica a camada convolucional aos atributos das arestas
        e = F.relu(self.edge_gcn(edge_features, edge_index))
        
        # Calcula os coeficientes de atenção para as arestas
        attention_coeffs = self.edge_attention(edge_features)
        
        # Inicializa tensores para agregação de informações dos vizinhos e arestas
        row, col = edge_index
        aggregated_neighbors = torch.zeros_like(x)
        aggregated_edges = torch.zeros_like(x)

        # Realiza a agregação por soma ponderada pelas atenções
        for src, dest, edge, coeff in zip(row, col, e, attention_coeffs):
            aggregated_neighbors[dest] += coeff * x[src]
            aggregated_edges[dest] += coeff * edge
        
        # Combina os atributos dos nós com os atributos agregados das arestas
        x = torch.cat([x + aggregated_neighbors, aggregated_edges], dim=1)
        x = self.fc(x)
        
        return x

class AGGEdgeGraph(torch.nn.Module):
    def __init__(self, in_edge_feats, hidden_size):
        super(AGGEdgeGraph, self).__init__()
        self.edge_linear = torch.nn.Linear(in_edge_feats, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, edge_feats, neighbors):
        edge_feats_transformed = self.edge_linear(edge_feats)
        agg_feat = torch.zeros((edge_feats.size(0), self.hidden_size), device=edge_feats.device)
        for idx, (e_feat, neighs) in enumerate(zip(edge_feats_transformed, neighbors)):
            sum_neighs = torch.sum(torch.stack([edge_feats_transformed[n] for n in neighs]), dim=0)
            agg_feat[idx] = e_feat + sum_neighs
        return agg_feat

class COMEdgeGraph(torch.nn.Module):
    def __init__(self, hidden_size):
        super(COMEdgeGraph, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU()
        )

    def forward(self, agg_feat):
        return self.mlp(agg_feat)

class AGGNodeGraph(torch.nn.Module):
    def __init__(self, in_node_feats, hidden_size):
        super(AGGNodeGraph, self).__init__()
        self.node_linear = torch.nn.Linear(in_node_feats, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, node_feats, edge_feats, neighbors):
        node_feats_transformed = self.node_linear(node_feats)
        agg_feat = torch.zeros((node_feats.size(0), self.hidden_size), device=node_feats.device)
        for idx, (n_feat, neighs) in enumerate(zip(node_feats_transformed, neighbors)):
            sum_neighs = torch.sum(torch.stack([node_feats_transformed[n] + edge_feats[e] for n, e in neighs]), dim=0)
            agg_feat[idx] = n_feat + sum_neighs
        return agg_feat

class COMNodeGraph(torch.nn.Module):
    def __init__(self, hidden_size):
        super(COMNodeGraph, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU()
        )

    def forward(self, agg_feat):
        return self.mlp(agg_feat)

class CAGNNLayer(torch.nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_size):
        super(CAGNNLayer, self).__init__()
        self.node_agg = AGGNodeGraph(in_node_feats, hidden_size)
        self.edge_agg = AGGEdgeGraph(in_edge_feats, hidden_size)
        self.node_com = COMNodeGraph(hidden_size)
        self.edge_com = COMEdgeGraph(hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, node_neighbors, edge_neighbors, node_feats, edge_feats):
        # Atualizar as features das arestas
        edge_agg_feats = self.edge_agg(edge_feats, edge_neighbors)
        new_edge_feats = self.edge_com(edge_agg_feats)
        new_edge_feats = self.layer_norm(new_edge_feats)

        # Atualizar as features dos nós
        node_agg_feats = self.node_agg(node_feats, new_edge_feats, node_neighbors)
        new_node_feats = self.node_com(node_agg_feats)
        new_node_feats = self.layer_norm(new_node_feats)

        return new_node_feats, new_edge_feats

class CAGNN(torch.nn.Module):
    def __init__(self, num_layers, in_node_feats, in_edge_feats, hidden_size, out_feats):
        super(CAGNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(CAGNNLayer(in_node_feats, in_edge_feats, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(CAGNNLayer(hidden_size, hidden_size, hidden_size))
        self.final_layer = torch.nn.Linear(hidden_size, out_feats)

    def forward(self, node_neighbors, edge_neighbors, node_feats, edge_feats):
        for layer in self.layers:
            node_feats, edge_feats = layer(node_neighbors, edge_neighbors, node_feats, edge_feats)
        # Representação final dos nós
        out = self.final_layer(node_feats)
        return out
   
# Classe que define um modelo GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GraphSAGE, self).__init__()
        # Define duas camadas convolucionais SAGE
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    # Define o passo de propagação para frente (forward) do modelo
    def forward(self, x, edge_index):
        # Aplica a primeira camada convolucional, ReLU e dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # Aplica a segunda camada convolucional
        x = self.conv2(x, edge_index)
        return x

# Classe que define um modelo GCN (Graph Convolutional Network)
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        # Define duas camadas convolucionais de grafos
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    # Define o passo de propagação para frente do modelo
    def forward(self, x, edge_index):
        # Aplica a primeira camada convolucional, ReLU e dropout
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # Aplica a segunda camada convolucional
        x = self.gc2(x, edge_index)
        return x 
