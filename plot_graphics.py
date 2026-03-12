import warnings
warnings.filterwarnings("ignore", "An issue occurred while importing 'torch-scatter'.")
from src.constants import Constants
from src.utils.eval_util import EvalUtil
from src.utils.data_util import DataUtil
import matplotlib.markers as mkr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import os
import numpy as np
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['abilene', 'rnp'], default='rnp', help='Dataset to plot results for')
    return p.parse_args()


args = parse_args()

# Prefer dataset-specific results folder if it exists, otherwise fall back to top-level results
candidate_folder = os.path.join(Constants.path_results, args.dataset)
if os.path.exists(candidate_folder) and os.path.isdir(candidate_folder):
    results_folder = candidate_folder
else:
    results_folder = Constants.path_results

ckpts = {}
# list model subfolders (ignore files)
dirs = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]

for dirr in dirs:
    ckpt_path = os.path.join(results_folder, dirr, f'{dirr}.ckpt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        ckpts[dirr] = ckpt

losses_dict = {}
num_models =  len(ckpts)

# Choose target data depending on dataset
if results_folder.endswith(os.path.join(Constants.path_results, 'abilene')) or (hasattr(args, 'dataset') and args.dataset == 'abilene'):
    target_path = Constants.path_to_abilene_target_data
else:
    target_path = Constants.path_to_target_data

actual_node_loads = DataUtil.get_node_loads(target_path)
r2s = []
# Ensure ground-truth is a NumPy 1D array for sklearn metrics
if torch.is_tensor(actual_node_loads):
    actual_node_loads_np = actual_node_loads.detach().cpu().numpy().ravel()
else:
    actual_node_loads_np = np.array(actual_node_loads).ravel()


for model_name in dirs:
    preds = np.array(ckpts[model_name]['predictions']).ravel()
    r2 = r2_score(actual_node_loads_np, preds)
    r2s.append([model_name,r2])
    print(f'{model_name} R2 ', r2)

r2s = sorted(r2s,key= lambda x: x[1],reverse=True)

dirs = [model_name for model_name,_ in r2s]

r2s = [r2 for _,r2 in r2s]

metrics_dict = {}
for model_name in dirs:
    preds = np.array(ckpts[model_name]['predictions']).ravel()
    metrics_dict[model_name] = EvalUtil.compute_metrics(actual_node_loads_np, preds)

if num_models < 5:
    all_markers = ['o','x','s','^']
    map_model_to_marker = {
    'AttEAGNN': 'o',
    'CAGNN': 'x',
    'GCN': 's',
    'GraphSAGE': '^'
    }
else:
    all_markers = list(mkr.MarkerStyle.markers.keys())[:num_models]
    map_model_to_marker = None

losses_dict = {}
markers = {}

for i,model_name in enumerate(dirs):
    losses_dict[model_name] = ckpts[model_name]['losses']

    if map_model_to_marker:
        markers[model_name] = map_model_to_marker[model_name]
    else:
        markers[model_name] = all_markers[i]

EvalUtil.plot_loss_curves(losses_dict, markers)

# Plots R² scores
plt.figure(figsize=(10, 5))
plt.rc('font', size=20)
plt.bar(dirs, r2s)
plt.ylabel('R2')
# Save plots into the results folder that was used
plot_save_path = os.path.join(results_folder, f'r2_day_{args.dataset}.png') if hasattr(args, 'dataset') else os.path.join(Constants.path_results, 'r2_day_rnp.png')
plt.savefig(plot_save_path)

EvalUtil.plot_metrics(metrics_dict)

print(f'Graphics saved in {results_folder}')
