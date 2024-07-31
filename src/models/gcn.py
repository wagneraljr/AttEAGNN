import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GraphConv
import numpy as np
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
