import numpy as np
from models import GCN, GraphSAGE, AttEdgeAwareGNN, CAGNN
import torch
import torch.nn as nn
from data_utils import load_data, get_node_loads, load_data_cagnn, normalize_features
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Get RNP network information and normalize edge features
node_features, edge_indices, edge_features = load_data("rnp.gml")
edge_features = normalize_features(edge_features)
# Get target node loads
actual_node_loads = get_node_loads("traffic_matrices/sparsified_gravity_cyclical/target/sparsified_gravity_cyclical_tm_8639.dat")
# Defines loss function
loss_fn = nn.MSELoss()
# Get historical training traffic matrices
traffic_matrix_files = sorted([file for file in os.listdir("traffic_matrices/sparsified_gravity_cyclical/day/") if file.endswith('.dat')])

# Fixates a random seed for reproducibility
seed = 48362
torch.manual_seed(seed)
np.random.seed(seed)

# Optimized Hyper parameters
edge_aware_model = AttEdgeAwareGNN(node_features.size(1), edge_features.size(1), 32, 1, 0.26896778382768677)
optimizer_edgeaware = torch.optim.Adam(list(edge_aware_model.parameters()), lr=0.0035191532755729435) 
scheduler_edgeaware = StepLR(optimizer_edgeaware, step_size=75, gamma= 0.8953758492665873)
edge_aware_losses = []

# AttEAGNN Training Loop
for epoch in range(300):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "traffic_matrices/sparsified_gravity_cyclical/day/" + traffic_matrix_filepath
        node_loads = get_node_loads(tm)
                        
        optimizer_edgeaware.zero_grad()
        edge_aware_predictions = edge_aware_model(node_features, edge_indices, edge_features)

        edge_aware_loss = loss_fn(edge_aware_predictions, node_loads)
        edge_aware_losses.append(edge_aware_loss.item())
        
        edge_aware_loss.backward()
        optimizer_edgeaware.step()
        scheduler_edgeaware.step()
      
edge_aware_predictions = edge_aware_model(node_features, edge_indices, edge_features)
edge_aware_predictions = edge_aware_predictions.detach().numpy()

# Reset random seed for consistency
seed = 48362
torch.manual_seed(seed)
np.random.seed(seed)

graphsage_model = GraphSAGE(node_features.size(1), 128, 1, 0.47150706333475556)
optimizer_graphsage = torch.optim.Adam(list(graphsage_model.parameters()), lr=0.0079743582639907)
scheduler_graphsage = StepLR(optimizer_graphsage, step_size=75, gamma=0.8599429449869641)
graphsage_losses = []

# GraphSAGE training loop
for epoch in range(200):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "traffic_matrices/sparsified_gravity_cyclical/day/" + traffic_matrix_filepath
        node_loads = get_node_loads(tm)
        
        optimizer_graphsage.zero_grad()
        graphsage_predictions = graphsage_model(node_features, edge_indices)
        graphsage_loss = loss_fn(graphsage_predictions, node_loads)
        graphsage_losses.append(graphsage_loss.item())
        
        graphsage_loss.backward()
        optimizer_graphsage.step()
        scheduler_graphsage.step()
        
graphsage_predictions = graphsage_model(node_features, edge_indices)
graphsage_predictions = graphsage_predictions.detach().numpy()

seed = 48362
torch.manual_seed(seed)
np.random.seed(seed)
gcn_model = GCN(node_features.size(1), 128 , 1, 0.2682185423306164)
optimizer_gcn = torch.optim.Adam(list(gcn_model.parameters()), lr=0.004074977620942678)
scheduler_gcn = StepLR(optimizer_gcn, step_size=75, gamma=0.9630977685525021)
gcn_losses = []

# GCN training loop
for epoch in range(200):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "traffic_matrices/sparsified_gravity_cyclical/day/" + traffic_matrix_filepath
        node_loads = get_node_loads(tm)
        
        optimizer_gcn.zero_grad()
        gcn_predictions = gcn_model(node_features, edge_indices)
        gcn_loss = loss_fn(gcn_predictions, node_loads)
        gcn_losses.append(gcn_loss.item())
        
        gcn_loss.backward()
        optimizer_gcn.step()
        scheduler_gcn.step()

gcn_predictions = gcn_model(node_features, edge_indices)
gcn_predictions = gcn_predictions.detach().numpy()

# Reset random seed for consistency
seed = 48362
torch.manual_seed(seed)
np.random.seed(seed)

node_features, edge_indices, edge_features, node_neighbors, edge_neighbors = load_data_cagnn("rnp.gml")
node_features = normalize_features(node_features)
edge_features = normalize_features(edge_features)

cagnn_model = CAGNN(2, node_features.size(1), edge_features.size(1), 64, 1, 0.42181608156931477)
optimizer_cagnn = torch.optim.Adam(list(cagnn_model.parameters()), lr=0.005256305981655686)
scheduler_cagnn = StepLR(optimizer_cagnn, step_size=75, gamma=0.8587686449302072)
cagnn_losses = []

# CAGNN training loop
for epoch in range(200):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "traffic_matrices/sparsified_gravity_cyclical/day/" + traffic_matrix_filepath
        node_loads = get_node_loads(tm)
        
        optimizer_cagnn.zero_grad()
        cagnn_predictions = cagnn_model(node_neighbors, edge_neighbors, node_features, edge_features)
        cagnn_loss = loss_fn(cagnn_predictions, node_loads)
        cagnn_losses.append(cagnn_loss.item())
        
        cagnn_loss.backward()
        optimizer_cagnn.step()
        scheduler_cagnn.step()

cagnn_predictions = cagnn_model(node_neighbors, edge_neighbors, node_features, edge_features)
cagnn_predictions = cagnn_predictions.detach().numpy()

# Function for plotting loss curves comparison
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
    plt.savefig('loss_curves_day_rnp.png')

# Sets losses values for each model
losses_dict = {
    'AttEAGNN': edge_aware_losses,
    'CAGNN': cagnn_losses,
    'GCN': gcn_losses,
    'GraphSAGE': graphsage_losses
}
markers = {
    'AttEAGNN': 'o',
    'CAGNN': 'x',
    'GCN': 's',
    'GraphSAGE': '^'
}
plot_loss_curves(losses_dict, markers)

# Calculates and prints R² scores for each model
edge_aware_r2 = r2_score(actual_node_loads, edge_aware_predictions)
print('Edge Aware GNN R2: ', edge_aware_r2)
cagnn_r2 = r2_score(actual_node_loads, cagnn_predictions)
print('CAGNN R2: ', cagnn_r2)
graphsage_r2 = r2_score(actual_node_loads, graphsage_predictions)
print('GraphSAGE R2: ', graphsage_r2)
gcn_r2 = r2_score(actual_node_loads, gcn_predictions)
print('GCN R2: ', gcn_r2)

# Plots R² scores
plt.figure(figsize=(10, 5))
plt.rc('font', size=20)
plt.bar(['AttEAGNN', 'CAGNN', 'GraphSAGE', 'GCN'], [edge_aware_r2, cagnn_r2, graphsage_r2, gcn_r2])
plt.ylabel('R2')
plt.savefig('r2_day_rnp.png')

# Auxiliary function to calculate MAE and RMSE
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': mae, 'RMSE': rmse}

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

    plt.savefig('metrics_day.png')

# Calculates and plots MAE and RMSE
metrics_dict = {
    'AttEdge-Aware GNN': compute_metrics(actual_node_loads, edge_aware_predictions),
    'CAGNN': compute_metrics(actual_node_loads, cagnn_predictions),
    'GCN': compute_metrics(actual_node_loads, gcn_predictions),
    'GraphSAGE': compute_metrics(actual_node_loads, graphsage_predictions)
}

plot_metrics(metrics_dict)
