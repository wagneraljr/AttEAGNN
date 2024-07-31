from src.entities.edge_gnn_data import EdgeGNNData

class CAGNNData(EdgeGNNData):
    def __init__(self,node_features, edge_indices, edge_features, node_neighbors, edge_neighbors):
        super().__init__(node_features,edge_indices,edge_features)
        self.node_neighbors = node_neighbors
        self.edge_neighbors = edge_neighbors
    
    def get_model_args(self):
        return self.node_neighbors, self.edge_neighbors, self.node_features, self.edge_features
    

