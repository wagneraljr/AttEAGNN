import warnings
warnings.filterwarnings("ignore", "An issue occurred while importing 'torch-scatter'.")

import optuna
from src.model_launcher import ModelLauncher
from configs.att_edge_aware_config import AttEdgeAwareGNNConfig
from src.constants import Constants
import torch.nn as nn

def objective(trial):
    # Define the hyperparameters to optimize
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
    epochs = trial.suggest_categorical('epochs', [50, 100, 200, 300])
    lr = trial.suggest_float('lr', 0.001, 0.01)
    step_size = trial.suggest_categorical('step_size', [25, 50, 75])
    gamma_scheduler = trial.suggest_float('gamma_scheduler', 0.1, 0.99)

    # Create a config with the suggested hyperparameters
    config = AttEdgeAwareGNNConfig()
    config.dropout = dropout
    config.hidden_dim = hidden_dim
    config.lr = lr
    config.step_size = step_size
    config.gamma_scheduler = gamma_scheduler
    config.epochs = epochs

    # Train the model and return the validation loss (or any other metric you want to optimize)
    model_launcher = ModelLauncher(config)
    val_loss = model_launcher.train(Constants.path_to_rnp_data, nn.MSELoss(), return_val_loss=True)

    # Optuna requires a numeric objectiv        e; map missing values to +inf so the
    # trial is treated as very poor instead of returning None.
    if val_loss is None:
        return float('inf')

    return float(val_loss)
if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)