import torch
from src.models.cagnn.agg_edge_graph import AGGEdgeGraph
from src.models.cagnn.agg_node_graph import AGGNodeGraph
from src.models.cagnn.com_node_graph import COMNodeGraph
from src.models.cagnn.com_edge_graph import COMEdgeGraph

# Defines a single CAGNN model layer
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
