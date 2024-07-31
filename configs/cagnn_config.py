from src.config import Config
from src.entities.cagnn_data import CAGNNData
from src.enums.enum_optim import EnumOptim
from src.enums.enum_scheduler import EnumScheduler
from src.utils.data_util import DataUtil
from src.models.ca_gnn import CAGNN
from src.constants import Constants
import os

class CAGNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.seed = 48362
        
        self.model_name = 'CAGNN'

        self.class_data = CAGNNData

        self.results_path = Constants.path_results + self.model_name + os.sep

        self.enum_optim = EnumOptim.ADAM
        self.enum_scheduler = EnumScheduler.STEP_LR

        self.epochs = 200
        self.hidden_dim = 64
        self.out_dim = 1

        self.lr = .005256305981655686
        self.step_size = 75
        self.gamma_scheduler = .8587686449302072


        self.load_data_func = DataUtil.load_data_cagnn
        self.edge_norm_func = DataUtil.normalize_features
        self.node_norm_func = DataUtil.normalize_features
        
        self.init_model_args = [2, 28, 5, self.hidden_dim, self.out_dim]
        self.class_model = CAGNN
