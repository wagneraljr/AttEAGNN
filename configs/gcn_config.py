from src.config import Config
from src.entities.gnn_data import GNNData
from src.enums.enum_optim import EnumOptim
from src.enums.enum_scheduler import EnumScheduler
from src.utils.data_util import DataUtil
from src.models.gcn import GCN
from src.constants import Constants
import os

class GCNConfig(Config):
    def __init__(self):
        super().__init__()
        self.seed = 48362
        
        self.model_name = 'GCN'

        self.class_data = GNNData

        self.results_path = Constants.path_results


        self.enum_optim = EnumOptim.ADAM
        self.enum_scheduler = EnumScheduler.STEP_LR

        self.epochs = 200
        self.dropout = .2682185423306164
        self.hidden_dim = 128
        self.out_dim = 1

        self.lr = .004074977620942678
        self.step_size = 75
        self.gamma_scheduler = .9630977685525021
        
        self.load_data_func = DataUtil.load_data_gnn


        self.init_model_args = [28, self.hidden_dim, self.out_dim, self.dropout]
        self.class_model = GCN
        self.hp_overrides = {
            ('abilene','day'):{
                'seed':1959,
                'epochs':300,
                'hidden_dim':128,
                'dropout':0.22140133961816266,
                'lr':0.005314626197777135,
                'step_size':25,
                'gamma_scheduler':0.8372606595075531,
                'init_model_args':[None,128,1,0.22140133961816266]
            },
            ('abilene','week'):{
                'seed':99,
                'epochs':300,
                'hidden_dim':64,
                'dropout':0.3989285698124243,
                'lr':0.0055639430333264,
                'step_size':75,
                'gamma_scheduler':0.6767352225815311,
                'init_model_args':[None,64,1,0.3989285698124243]
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

