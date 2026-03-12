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
        self.results_path = Constants.path_results

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
        # Hyperparameter overrides per (dataset, split)
        self.hp_overrides = {
            ('abilene', 'day'):{
                'seed':3010,
                'epochs':300,
                'hidden_dim':128,
                'dropout':0.4486455892663148,
                'lr':0.002894236244459579,
                'step_size':25,
                'gamma_scheduler':0.648405713368948,
                'init_model_args':[None, None, 128, 1, 0.4486455892663148]
            },
            ('abilene','week'):{
                'seed':2011,
                'epochs':300,
                'hidden_dim':16,
                'dropout':0.2749459184344534,
                'lr':0.008619881892095895,
                'step_size':25,
                'gamma_scheduler':0.6430367566425178,
                'init_model_args':[None,None,16,1,0.2749459184344534]
            }
        }

    def apply_dataset_split(self, dataset: str, split: str):
        key = (dataset, split)
        if key not in getattr(self, 'hp_overrides', {}):
            return
        vals = self.hp_overrides[key]
        for k, v in vals.items():
            if k == 'init_model_args' and v is not None:
                self.init_model_args = v
                continue
            setattr(self, k, v)