from src.config import Config
from src.enums.enum_optim import EnumOptim
from src.enums.enum_scheduler import EnumScheduler
import torch
from  torch.optim.lr_scheduler import StepLR

class InitializerUtil:
    @staticmethod
    def init_optim_from_config(config: Config,model):
        if config.enum_optim == EnumOptim.ADAM:
            return torch.optim.Adam(list(model.parameters()), lr=config.lr) 
        else:
            raise ValueError(f"Optimizer {config.optim_enum} is not configured.")

    @staticmethod
    def init_scheduler_from_config(config: Config,optimizer):
        if config.enum_scheduler == EnumScheduler.STEP_LR:
            return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma_scheduler)

        else:
            print('Scheduler was not initialized.')
