import torch
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
