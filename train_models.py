import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from src.constants import Constants
from src.utils.data_util import DataUtil
from configs.att_edge_aware_config import AttEdgeAwareGNNConfig
from configs.graphsage_config import GraphSAGEConfig
from configs.gcn_config import GCNConfig
from configs.cagnn_config import CAGNNConfig
from src.model_launcher import ModelLauncher

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
