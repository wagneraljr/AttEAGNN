from configs.att_edge_aware_config import AttEdgeAwareGNNConfig
from src.model_launcher import ModelLauncher
from torch import nn
from src.constants import Constants

cfg = AttEdgeAwareGNNConfig()
# apply abilene/day overrides if present
cfg.apply_dataset_split('abilene', 'day')
# disable edge normalization to match EdgeAwareGNN-main
cfg.edge_norm_func = None

ml = ModelLauncher(cfg)
# Pass the path to the Abilene GML file (correct argument)
ml.train(Constants.path_to_abilene_data, nn.MSELoss(), dataset='abilene', tm_split='day')
