import torch
class COMNodeGraph(torch.nn.Module):
    def __init__(self, hidden_size):
        super(COMNodeGraph, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU()
        )

    def forward(self, agg_feat):
        return self.mlp(agg_feat)
