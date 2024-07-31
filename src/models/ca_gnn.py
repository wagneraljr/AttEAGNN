import torch
from src.models.cagnn.cagnn_layer import CAGNNLayer
# CAGNN model implementation
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

        out = self.final_layer(node_feats)
        return out