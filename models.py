import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv, global_mean_pool
import numpy as np

# Class that implements the edge attention mechanism
class EdgeAttention(nn.Module):
    def __init__(self, edge_feature_dim, hidden_dim):
        super(EdgeAttention, self).__init__()
        # Linear layer for edge feature transforming
        self.edge_weight = nn.Linear(edge_feature_dim, hidden_dim)
        # LeakyReLU activation function
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, edge_features):
        # Applies linear transforming to edge features
        edge_transformed = self.edge_weight(edge_features)
        # Get attention scores to each edge using dot product
        attention_scores = (edge_transformed * edge_transformed).sum(dim=1)
        # Apply LeakyReLU to the scores
        attention_scores = self.leaky_relu(attention_scores)
        # Normalize scores using softmax to get attention coefficients
        attention_coeffs = F.softmax(attention_scores, dim=0)
        return attention_coeffs
    
# Defines the AttEAGNN model
class AttEdgeAwareGNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim, dropout=0.5):
        super(AttEdgeAwareGNN, self).__init__()
        # Node feature processing convolutional layers
        self.gc1 = GraphConv(node_input_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        # Edge feature processing SAGE layer
        self.edge_gcn = SAGEConv(edge_input_dim, hidden_dim)
        # Initialize edge attention mechanism
        self.edge_attention = EdgeAttention(edge_input_dim, hidden_dim)
        # Linear layer to combine edge and node features
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, node_features, edge_index, edge_features):
        # First node convolutional layer and ReLU activation
        x = F.relu(self.gc1(node_features, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # Second node convolutional layer
        x = F.relu(self.gc2(x, edge_index))
        
        # SAGE edge feature processing layer
        e = F.relu(self.edge_gcn(edge_features, edge_index))
        
        # Get edge attention coefficients
        attention_coeffs = self.edge_attention(edge_features)
        
        # Initializes tensors for edge and neighborhood information aggregation
        row, col = edge_index
        aggregated_neighbors = torch.zeros_like(x)
        aggregated_edges = torch.zeros_like(x)

        # Aggregates features via weighted sum
        for src, dest, edge, coeff in zip(row, col, e, attention_coeffs):
            aggregated_neighbors[dest] += coeff * x[src]
            aggregated_edges[dest] += coeff * edge
        
        # Concatenates node and aggregated edge features
        x = torch.cat([x + aggregated_neighbors, aggregated_edges], dim=1)
        x = self.fc(x)
        
        return x

# Process edge graph features for the CAGNN model
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
   
# Defines a GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GraphSAGE, self).__init__()
        # Defines two SAGEConv layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First convolutional layer, ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # Second convolutional layer
        x = self.conv2(x, edge_index)
        return x

# Defines a GCN model
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        # Defines two GraphConv layers
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First convolutional layer, ReLU and dropout
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # Second convolutional layer
        x = self.gc2(x, edge_index)
        return x 
