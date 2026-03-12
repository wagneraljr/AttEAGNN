import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import GCN, GraphSAGE, AttEdgeAwareGCN
from data_utils import load_data, get_node_loads
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Definindo a semente para reprodutibilidade
seed = 3010
torch.manual_seed(seed)
np.random.seed(seed)

# Carregamento inicial dos dados do grafo
node_features, edge_indices, edge_features = load_data("Abilene.gml")
actual_node_loads = get_node_loads("Data/tm.2004-09-10.16-00-00.dat")

# Definindo a função de perda e o número de épocas
loss_fn = nn.MSELoss()
num_epochs = 300

# Criação do modelo AttEdgeAwareGCN e configuração do otimizador e escalonador
# Os parâmetros de todos os modelos foram definidos conforme os melhores resultados do optuna
edge_aware_model = AttEdgeAwareGCN(node_features.size(1), edge_features.size(1), 128, 1, 0.4486455892663148,)
optimizer_edgeaware = torch.optim.Adam(list(edge_aware_model.parameters()), lr=0.002894236244459579) 
scheduler_edgeaware = StepLR(optimizer_edgeaware, step_size=25, gamma=0.648405713368948)
edge_aware_losses = []  # Para armazenar as perdas ao longo do treinamento

# Obtendo lista de arquivos de matrizes de tráfego
traffic_matrix_files = sorted([file for file in os.listdir("./Measured/day") if file.endswith('.dat')])

# Loop de treinamento para o modelo AttEdgeAwareGCN
for epoch in range(num_epochs):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "Measured/day/" + traffic_matrix_filepath
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

# Resetando a semente para o próximo modelo
seed = 1959
torch.manual_seed(seed)
np.random.seed(seed)

# Configuração e treinamento para o modelo GCN
gcn_model = GCN(node_features.size(1), 128 , 1, 0.22140133961816266)
optimizer_gcn = torch.optim.Adam(list(gcn_model.parameters()), lr=0.005314626197777135)
scheduler_gcn = StepLR(optimizer_gcn, step_size=25, gamma=0.8372606595075531)
gcn_losses = []

for epoch in range(num_epochs - 200):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "Measured/day/" + traffic_matrix_filepath
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

# Configuração e treinamento para o modelo GraphSAGE
graphsage_model = GraphSAGE(node_features.size(1), 32, 1, 0.2899968355254155)
optimizer_graphsage = torch.optim.Adam(list(graphsage_model.parameters()), lr=0.005632517545041001)
scheduler_graphsage = StepLR(optimizer_graphsage, step_size=25, gamma=0.7348042711305643)
graphsage_losses = []

for epoch in range(num_epochs - 200):
    for traffic_matrix_filepath in traffic_matrix_files:
        tm = "Measured/day/" + traffic_matrix_filepath
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

# Preparando os dados reais para comparação
actual_node_loads = actual_node_loads.detach().numpy()

# Função para plotar as curvas de perda dos modelos
def plot_loss_curves(losses_dict, markers):
    plt.figure(figsize=(10, 6))
    plt.rc('font', size=20)
    for model_name, loss_values in zip(losses_dict.keys(), losses_dict.values()):
        plt.plot(loss_values, label=model_name, marker=markers[model_name])
    plt.xlabel('Épocas')
    plt.ylabel('Perda (MSE)')
    plt.xlim(0, 60)
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves_day.png')

# Plotando as curvas de perda
losses_dict = {
    'AttEdge-AwareGNN': edge_aware_losses,
    'GCN': gcn_losses,
    'GraphSAGE': graphsage_losses
}
markers = {
    'AttEdge-AwareGNN': 'o',
    'GCN': 's',
    'GraphSAGE': '^'
}
plot_loss_curves(losses_dict, markers)

# Função para plotar comparações entre previsões e valores reais
def plot_predicted_vs_actual(predictions_dict, actual_values):
    plt.figure(figsize=(15, 6))
    plt.rc('font', size=15)
    for idx, (model_name, predicted_values) in enumerate(predictions_dict.items(), 1):
        plt.subplot(1, len(predictions_dict), idx)
        plt.scatter(actual_values, predicted_values, alpha=0.5)
        plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'r--')
        plt.title(f'{model_name}')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictvsactual_day.png')

# Plotando previsões vs valores reais
predictions_dict = {
    'AttEdge-Aware GNN': edge_aware_predictions,
    'GCN': gcn_predictions,
    'GraphSAGE': graphsage_predictions
}
plot_predicted_vs_actual(predictions_dict, actual_node_loads)

# Calculando e imprimindo as métricas R2 para cada modelo
edge_aware_r2 = r2_score(actual_node_loads, edge_aware_predictions)
print('Edge Aware GNN R2: ', edge_aware_r2)
graphsage_r2 = r2_score(actual_node_loads, graphsage_predictions)
print('GraphSAGE R2: ', graphsage_r2)
gcn_r2 = r2_score(actual_node_loads, gcn_predictions)
print('GCN R2: ', gcn_r2)

# Plotando a pontuação R2
plt.figure(figsize=(10, 5))
plt.rc('font', size=20)
plt.bar(['AttEdge-Aware GNN', 'GraphSAGE', 'GCN'], [edge_aware_r2, graphsage_r2, gcn_r2])
plt.ylabel('R2')
plt.savefig('r2_day.png')

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

    ax.set_ylabel('Scores')
    ax.set_title('Scores by model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.savefig('metrics_day.png')

# Calculando e plotando outras métricas (MAE e RMSE)
metrics_dict = {
    'AttEdge-Aware GNN': compute_metrics(actual_node_loads, edge_aware_predictions),
    'GCN': compute_metrics(actual_node_loads, gcn_predictions),
    'GraphSAGE': compute_metrics(actual_node_loads, graphsage_predictions)
}
plot_metrics(metrics_dict)

# Salvando os modelos treinados
torch.save(edge_aware_model, 'models/edge_aware_model_day.pth')
torch.save(gcn_model, 'models/gcn_model_day.pth')
torch.save(graphsage_model, 'models/graphsage_model_day.pth')
