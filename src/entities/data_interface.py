from abc import ABC, abstractmethod
class DataInterface(ABC):
    
    @abstractmethod
    def get_model_args(self):
        ...
    
    @abstractmethod
    def norm_node_features(self, norm_func):
        ...
    
    @abstractmethod
    def norm_edge_features(self, norm_func):
        ...
    