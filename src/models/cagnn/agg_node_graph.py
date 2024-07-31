import torch
# Process node graph features for the CAGNN model
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
