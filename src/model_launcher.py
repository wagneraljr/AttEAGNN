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
        # Defer model creation until data is loaded so input dims match dataset
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def train(self,path_to_gml_data,loss_fn, dataset: str = 'abilene', tm_split: str = 'day', return_val_loss: bool = False):
        # Adjust results_path to include dataset and split so printed config reflects actual save location
        if self.config.results_path:
            model_folder = os.path.basename(os.path.normpath(self.config.results_path))
            base_results = os.path.dirname(os.path.normpath(self.config.results_path))
            new_results_path = os.path.join(base_results, dataset.lower(), tm_split, model_folder) + os.sep
            self.config.results_path = new_results_path

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

        # Instantiate model with the correct node input dimension from data
        try:
            node_input_dim = data.node_features.shape[1]
        except Exception:
            node_input_dim = self.config.init_model_args[0] if len(self.config.init_model_args) > 0 else None

        try:
            edge_input_dim = data.edge_features.shape[1]
        except Exception:
            edge_input_dim = self.config.init_model_args[1] if len(self.config.init_model_args) > 1 else None

        init_args = list(self.config.init_model_args)
        if node_input_dim is not None and len(init_args) > 0:
            init_args[0] = node_input_dim
        if edge_input_dim is not None and len(init_args) > 1:
            init_args[1] = edge_input_dim

        self.model = self.config.class_model(*init_args)
        self.optimizer = InitializerUtil.init_optim_from_config(self.config,self.model)
        self.scheduler = InitializerUtil.init_scheduler_from_config(self.config,self.optimizer)

        # Choose traffic matrix folder based on requested dataset and split
        if dataset.lower() == 'abilene':
            traffic_matrix_folder = Constants.path_to_abilene_day_tm_files if tm_split == 'day' else Constants.path_to_abilene_week_tm_files
        else:
            # Replace 'day' with requested split for RNP-style folder paths
            traffic_matrix_folder = Constants.path_to_day_tm_files.replace('/day/', f'/{tm_split}/').replace('\\day\\', f'\\{tm_split}\\')

        if not os.path.isdir(traffic_matrix_folder):
            raise FileNotFoundError(f"Traffic matrix folder not found: {traffic_matrix_folder}")

        traffic_matrix_files = sorted([os.path.join(traffic_matrix_folder, file) for file in os.listdir(traffic_matrix_folder) if file.endswith('.dat')])

        losses = TrainUtil.train_epoch(self.config.epochs, data, traffic_matrix_files, self.model,
                                       self.optimizer, loss_fn, self.scheduler, dataset=dataset)
                
        predictions = self.model(*data.get_model_args()).detach().numpy().tolist()

        if self.config.results_path:
            # self.config.results_path already includes dataset and split; use it as save dir
            save_dir = os.path.normpath(self.config.results_path)
            os.makedirs(save_dir, exist_ok=True)
            train_data = {'losses': losses, 'weights': self.model.state_dict(), 'predictions': predictions, 'model_name': self.config.model_name}
            torch.save(train_data, os.path.join(save_dir, f'{self.config.model_name}.ckpt'))
            print(f'Checkpoint saved in {save_dir}{os.sep}{self.config.model_name}.ckpt')
            print()

        # By default this method preserves previous behavior (no meaningful return value).
        # If `return_val_loss=True`, compute and return a scalar validation loss so
        # callers (e.g., Optuna) receive a numeric objective value.
        # Calculation: average of the losses from the last epoch (i.e. last
        # `len(traffic_matrix_files)` training steps). If not enough values exist,
        # fall back to the last recorded loss. If no losses are recorded, return None.
        if not return_val_loss:
            return None

        try:
            if len(traffic_matrix_files) > 0 and len(losses) >= len(traffic_matrix_files):
                last_epoch_losses = losses[-len(traffic_matrix_files):]
                val_loss = sum(last_epoch_losses) / len(last_epoch_losses)
            elif len(losses) > 0:
                val_loss = losses[-1]
            else:
                val_loss = None
        except Exception:
            val_loss = None

        return val_loss





