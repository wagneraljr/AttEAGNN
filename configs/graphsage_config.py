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

        self.results_path = Constants.path_results

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
        self.hp_overrides = {
            ('abilene','day'):{
                'seed':48362,
                'epochs':300,
                'hidden_dim':32,
                'dropout':0.2899968355254155,
                'lr':0.005632517545041001,
                'step_size':25,
                'gamma_scheduler':0.7348042711305643,
                'init_model_args':[None,32,1,0.2899968355254155]
            },
            ('abilene','week'):{
                'seed':99,
                'epochs':300,
                'hidden_dim':64,
                'dropout':0.40367790734911385,
                'lr':0.008176296053198579,
                'step_size':75,
                'gamma_scheduler':0.49667015759483746,
                'init_model_args':[None,64,1,0.40367790734911385]
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
    
