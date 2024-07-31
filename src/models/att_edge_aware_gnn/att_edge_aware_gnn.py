import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GraphConv, global_mean_pool
import numpy as np
from src.models.att_edge_aware_gnn.edge_attention import EdgeAttention
# Defines the AttEAGNN model
class AttEdgeAwareGNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim, dropout=0.5):
        super(AttEdgeAwareGNN, self).__init__()
        print('node:', node_input_dim)
        print('output:', output_dim)
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









   

