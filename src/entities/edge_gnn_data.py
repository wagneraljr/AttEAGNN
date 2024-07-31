from src.entities.gnn_data import GNNData
class EdgeGNNData(GNNData):
    def __init__(self,node_features,edge_indices, edge_features):
        super().__init__(node_features,edge_indices)
        self.edge_features = edge_features
    
    def get_model_args(self):
        return self.node_features,self.edge_indices,self.edge_features

    def norm_edge_features(self,norm_func):
        self.edge_features = norm_func(self.edge_features)
