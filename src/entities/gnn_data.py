from src.entities.data_interface import DataInterface

class GNNData(DataInterface):
    def __init__(self,node_features,edge_indices):
        self.node_features = node_features
        self.edge_indices = edge_indices

    def get_model_args(self):
        return self.node_features,self.edge_indices
    
    def norm_node_features(self,norm_func):
        self.node_features = norm_func(self.node_features)

    def norm_edge_features(self,norm_func):
        pass
    
    
