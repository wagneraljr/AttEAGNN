import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

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
