from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from src.constants import Constants
class EvalUtil:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {'MAE': mae, 'RMSE': rmse}
    @staticmethod
    # Plots MAE and RMSE for each model
    def plot_metrics(metrics_dict):
        labels = list(metrics_dict.keys())
        mae_scores = [metrics['MAE'] for metrics in metrics_dict.values()]
        rmse_scores = [metrics['RMSE'] for metrics in metrics_dict.values()]
        
        x = np.arange(len(labels))  # Label placement
        width = 0.3  # Bar width
        
        fig, ax = plt.subplots(figsize=(15, 7))
    
        rects1 = ax.bar(x - width/2, mae_scores, width, label='MAE')
        rects2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        plt.savefig(f'{Constants.path_results}metrics_day.png')
    @staticmethod
    def plot_loss_curves(losses_dict, markers):
        plt.figure(figsize=(10, 6))
        plt.rc('font', size=20)
        for model_name, loss_values in zip(losses_dict.keys(), losses_dict.values()):
            plt.plot(loss_values, label=model_name, marker=markers[model_name])
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.xlim(0, 60)
        plt.ylim(0, 0.05)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{Constants.path_results}loss_curves_day_rnp.png')
