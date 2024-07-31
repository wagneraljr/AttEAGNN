from src.config import Config
from src.entities.gnn_data import GNNData
from src.enums.enum_optim import EnumOptim
from src.enums.enum_scheduler import EnumScheduler
from src.utils.data_util import DataUtil
from src.models.graph_sage import  GraphSAGE
from src.constants import Constants
import os
class GraphSAGEConfig(Config):
    def __init__(self):
        super().__init__()
        self.seed = 48362
        
        self.model_name = 'GraphSAGE'

        self.class_data = GNNData

        self.results_path = Constants.path_results + self.model_name + os.sep

        self.enum_optim = EnumOptim.ADAM
        self.enum_scheduler = EnumScheduler.STEP_LR

        self.epochs = 200
        self.dropout = 0.47150706333475556
        self.hidden_dim = 128
        self.out_dim = 1

        self.lr = 0.0079743582639907
        self.step_size = 75
        self.gamma_scheduler = 0.8599429449869641


        self.load_data_func = DataUtil.load_data_gnn
        self.init_model_args = [28,self.hidden_dim, self.out_dim, self.dropout]
        self.class_model = GraphSAGE 
    
