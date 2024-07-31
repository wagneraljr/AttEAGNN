from src.constants import Constants
from src.utils.eval_util import EvalUtil
from src.utils.data_util import DataUtil
import matplotlib.markers as mkr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import os
ckpts = {}
dirs =  [dirr for dirr in os.listdir(Constants.path_results) if not dirr.endswith('.png')]

for dirr in dirs:
    ckpt = torch.load(Constants.path_results + dirr + os.sep + dirr + '.ckpt')
    ckpts[dirr] = ckpt

losses_dict = {}
num_models =  len(ckpts)

# Calculates and prints R² scores for each model
actual_node_loads = DataUtil.get_node_loads(Constants.path_to_target_data)
r2s = []
# Calculates and plots MAE and RMSE
metrics_dict = {}

for model_name in dirs:
    r2 = r2_score(actual_node_loads, ckpts[model_name]['predictions'])
    r2s.append([model_name,r2])
    print(f'{model_name} R2 ', r2)

r2s = sorted(r2s,key= lambda x: x[1],reverse=True)

dirs = [model_name for model_name,_ in r2s]

r2s = [r2 for _,r2 in r2s]

for model_name in dirs:
    metrics_dict[model_name] = EvalUtil.compute_metrics(actual_node_loads,ckpts[model_name]['predictions'])

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
plt.savefig(f'{Constants.path_results}r2_day_rnp.png')

EvalUtil.plot_metrics(metrics_dict)

print(f'Graphics saved in {Constants.path_results}')
