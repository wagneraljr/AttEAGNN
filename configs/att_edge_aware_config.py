from src.config import Config
from src.entities.edge_gnn_data import EdgeGNNData
from src.enums.enum_optim import EnumOptim
from src.enums.enum_scheduler import EnumScheduler
from src.utils.data_util import DataUtil
from src.models.att_edge_aware_gnn.att_edge_aware_gnn import AttEdgeAwareGNN
from src.constants import Constants
import os
class AttEdgeAwareGNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.seed = 48362
        
        self.model_name = 'AttEAGNN'
        self.results_path = Constants.path_results  + self.model_name+ os.sep

        self.class_data = EdgeGNNData

        self.enum_optim = EnumOptim.ADAM
        self.enum_scheduler = EnumScheduler.STEP_LR

        self.epochs = 300
        self.dropout = 0.26896778382768677
        self.hidden_dim = 32
        self.out_dim = 1

        self.lr = .0035191532755729435
        self.step_size = 75
        self.gamma_scheduler = 0.8953758492665873

        self.edge_norm_func = DataUtil.normalize_features
        self.load_data_func = DataUtil.load_data
        self.init_model_args = [28, 5, self.hidden_dim, self.out_dim, self.dropout]
        self.class_model = AttEdgeAwareGNN