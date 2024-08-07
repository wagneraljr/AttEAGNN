import torch.nn as nn
from src.constants import Constants
from src.model_launcher import ModelLauncher
from configs.att_edge_aware_config import AttEdgeAwareGNNConfig
from configs.graphsage_config import GraphSAGEConfig
from configs.gcn_config import GCNConfig
from configs.cagnn_config import CAGNNConfig

loss_fn = nn.MSELoss()
path_to_gml_data = Constants.path_to_rnp_data

configs = [
    AttEdgeAwareGNNConfig(),
    GraphSAGEConfig(),
    GCNConfig(),
    CAGNNConfig()
    ]
for config in configs:
    model_laucher = ModelLauncher(config)
    model_laucher.train(path_to_gml_data,loss_fn)
