import warnings
warnings.filterwarnings("ignore", "An issue occurred while importing 'torch-scatter'.")
from src.constants import Constants
from src.utils.eval_util import EvalUtil
from src.utils.data_util import DataUtil
import json
import matplotlib.markers as mkr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import os
import numpy as np
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['abilene', 'rnp'], default='abilene', help='Dataset to plot results for')
    p.add_argument('--split', choices=['day', 'week'], default='day', help='Traffic-matrix split to plot (day or week)')
    return p.parse_args()


args = parse_args()

# Use dataset+split-specific results folder (required)
results_folder = os.path.join(Constants.path_results, args.dataset, args.split)
if not os.path.isdir(results_folder):
    raise FileNotFoundError(f"Results folder not found: {results_folder}")

ckpts = {}
# Gather checkpoints from the new layout:
# results/<dataset>/<split>/<model_name>/<model_name>.ckpt
model_dirs = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
for model_dir in model_dirs:
    model_path = os.path.join(results_folder, model_dir)
    meta_path = os.path.join(model_path, f'{model_dir}.meta.json')
    weights_path = os.path.join(model_path, f'{model_dir}.weights.pt')
    ckpt_path = os.path.join(model_path, f'{model_dir}.ckpt')
    # If metadata already exists, remove legacy .ckpt to avoid torch.load warnings
    if os.path.exists(meta_path) and os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
            print(f'Removed legacy checkpoint {ckpt_path}')
        except Exception:
            pass
    # Prefer metadata JSON
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        ckpts[model_dir] = {'predictions': meta.get('predictions', []), 'losses': meta.get('losses', [])}
        continue

    # If no metadata JSON, but an old .ckpt exists, migrate it to new format
    if os.path.exists(ckpt_path):
        print(f'Migrating legacy checkpoint for {model_dir} -> {meta_path}')
        # load full ckpt (one-time migration)
        legacy = torch.load(ckpt_path)
        preds = legacy.get('predictions', [])
        losses = legacy.get('losses', [])
        # save weights separately if present
        legacy_weights = legacy.get('weights', None)
        if legacy_weights is not None:
            try:
                torch.save(legacy_weights, weights_path)
            except Exception:
                pass
        # write metadata JSON
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({'predictions': preds, 'losses': losses, 'model_name': model_dir}, f)
            # remove legacy .ckpt once migration succeeded
            try:
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
            except Exception:
                pass
        except Exception:
            pass
        ckpts[model_dir] = {'predictions': preds, 'losses': losses}
        continue

    # no data for this model_dir
    print(f'No checkpoint or metadata found for {model_dir} in {model_path}')

losses_dict = {}
num_models = len(ckpts)
dirs = list(ckpts.keys())

# Choose target data depending on dataset
if args.dataset == 'abilene':
    target_path = Constants.path_to_abilene_target_data
    actual_node_loads = DataUtil.get_node_loads(target_path, abilene=True)
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

# Print detailed metrics and save a text summary
summary_lines = []
summary_lines.append(f'Dataset: {args.dataset}, split: {args.split}')
summary_lines.append('Model, R2, MAE, RMSE')
for i, model_name in enumerate(dirs):
    r2_val = r2s[i]
    m = metrics_dict[model_name]
    line = f"{model_name}, {r2_val:.6f}, {m['MAE']:.6f}, {m['RMSE']:.6f}"
    print(line)
    summary_lines.append(line)

summary_path = os.path.join(results_folder, 'metrics_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
print(f'Summary saved in {summary_path}')

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

EvalUtil.plot_loss_curves(losses_dict, markers, save_folder=results_folder)

# Plots R² scores
plt.figure(figsize=(10, 5))
plt.rc('font', size=20)
plt.bar(dirs, r2s)
plt.ylabel('R2')
# Save plots into the results folder that was used
plot_save_path = os.path.join(results_folder, f'r2_{args.split}_{args.dataset}.png')
plt.savefig(plot_save_path)

EvalUtil.plot_metrics(metrics_dict, save_folder=results_folder)

print(f'Graphics saved in {results_folder}')
