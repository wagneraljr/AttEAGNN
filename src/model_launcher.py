import os
from src.constants import Constants
from src.utils.initializer_util import InitializerUtil
from src.utils.train_util import TrainUtil
import torch
import numpy as np
from src.config import Config

class ModelLauncher:
    def __init__(self,config:Config):
        self.config = config
        if self.config.seed:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        self.model = self.config.class_model(*self.config.init_model_args)
        self.optimizer = InitializerUtil.init_optim_from_config(config,self.model)
        self.scheduler = InitializerUtil.init_scheduler_from_config(config,self.optimizer)


    def train(self,path_to_gml_data,loss_fn):
        print()
        print(f'Training Model {self.config.model_name}')
        print(f"{self.config.model_name} Configuration:")
        for k,v in self.config.__dict__.items():
            print(f'\t{k}:',v)
        print()
        
        data = self.config.load_data_func(path_to_gml_data)
        data = self.config.class_data(*data)

        if self.config.node_norm_func:
            data.norm_node_features(self.config.node_norm_func)
        
        if self.config.edge_norm_func:
            data.norm_edge_features(self.config.edge_norm_func)

        traffic_matrix_files = sorted([file for file in os.listdir(Constants.path_to_day_tm_files) if file.endswith('.dat')])

        losses = TrainUtil.train_epoch(self.config.epochs,data,traffic_matrix_files,self.model,
                                    self.optimizer,loss_fn,self.scheduler)
                
        predictions = self.model(*data.get_model_args()).detach().numpy().tolist()

        if self.config.results_path:
            os.makedirs(self.config.results_path,exist_ok=True)
            train_data = {'losses':losses,'weights':self.model.state_dict(),'predictions':predictions,'model_name':self.config.model_name} 
            torch.save(train_data,self.config.results_path+f'{self.config.model_name}.ckpt')
            print(f'Checkpoint saved in {self.config.results_path}{self.config.model_name}.ckpt')
            print()





