import torch.nn as nn
import torch.nn.functional as F
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