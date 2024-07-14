import numpy as np
import pandas as pd
from models import GCN, GraphSAGE, AttEdgeAwareGCN, CAGNN
import torch
import torch.nn as nn
from data_utils import load_data, get_node_loads, load_data_cagnn, normalize_features
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import StepLR

node_features, edge_indices, edge_features = load_data("rnp.gml.txt")
edge_features = normalize_features(edge_features)
actual_node_loads = get_node_loads("traffic_matrices/sparsified_gravity_cyclical/target/sparsified_gravity_cyclical_tm_8639.dat")
loss_fn = nn.MSELoss()

traffic_matrix_files = sorted([file for file in os.listdir("traffic_matrices/sparsified_gravity_cyclical/day/") if file.endswith('.dat')])

seed = 48362
torch.manual_seed(seed)
np.random.seed(seed)

# Optimized Hyper parameters
edge_aware_model = AttEdgeAwareGCN(node_features.size(1), edge_features.size(1), 32, 1, 0.26896778382768677)
optimizer_edgeaware = torch.optim.Adam(list(edge_aware_model.parameters()), lr=0.0035191532755729435) 
scheduler_edgeaware = StepLR(optimizer_edgeaware, step_size=75, gamma= 0.8953758492665873)
edge_aware_losses = []

# AttEAGNN Training Loop
for epoch in range(300):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "traffic_matrices/sparsified_gravity_cyclical/day/" + traffic_matrix_filepath
        node_loads = get_node_loads(tm)
        #print(node_loads.shape)
                
        optimizer_edgeaware.zero_grad()
        edge_aware_predictions = edge_aware_model(node_features, edge_indices, edge_features)
        #assert edge_aware_predictions.size(0) == node_loads.size(0), f"Tamanho do tensor predictions {edge_aware_predictions.size(0)} e node_loads {node_loads.size(0)} não batem"

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

seed = 48362
torch.manual_seed(seed)
np.random.seed(seed)

node_features, edge_indices, edge_features, node_neighbors, edge_neighbors = load_data_cagnn("rnp.gml.txt")
node_features = normalize_features(node_features)
edge_features = normalize_features(edge_features)

cagnn_model = CAGNN(2, node_features.size(1), edge_features.size(1), 64, 1)
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

# Função para plotar as curvas de perda dos modelos
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

# Plotando as curvas de perda
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

# Plotando a pontuação R2
plt.figure(figsize=(10, 5))
plt.rc('font', size=20)
plt.bar(['AttEAGNN', 'CAGNN', 'GraphSAGE', 'GCN'], [edge_aware_r2, cagnn_r2, graphsage_r2, gcn_r2])
plt.ylabel('R2')
plt.savefig('r2_day_rnp.png')

# Função auxiliar para calcular métricas MAE e RMSE
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': mae, 'RMSE': rmse}

# Função para plotar as métricas MAE e RMSE para cada modelo
def plot_metrics(metrics_dict):
    labels = list(metrics_dict.keys())
    mae_scores = [metrics['MAE'] for metrics in metrics_dict.values()]
    rmse_scores = [metrics['RMSE'] for metrics in metrics_dict.values()]
    
    x = np.arange(len(labels))  # Localização das etiquetas
    width = 0.3  # Largura das barras
    
    fig, ax = plt.subplots(figsize=(15, 7))
   
    rects1 = ax.bar(x - width/2, mae_scores, width, label='MAE')
    rects2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE')

  ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.savefig('metrics_day.png')

# Calculando e plotando outras métricas (MAE e RMSE)
metrics_dict = {
    'AttEdge-Aware GNN': compute_metrics(actual_node_loads, edge_aware_predictions),
    'CAGNN': compute_metrics(actual_node_loads, cagnn_predictions),
    'GCN': compute_metrics(actual_node_loads, gcn_predictions),
    'GraphSAGE': compute_metrics(actual_node_loads, graphsage_predictions)
}

plot_metrics(metrics_dict)
