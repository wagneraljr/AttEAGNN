import networkx as nx
import json
from src.constants import Constants

def get_geographical_nx_graph_from_json(json_file):
    with open(json_file,'r',encoding='utf-8') as f:
        json_data = json.load(f)
    edge_keys = ['bandwith','distances']
    node_keys = ['coordinates']
    map_keys_to_attr_keys = {'bandwith': 'bw','distances':'dist','coordinates':'pos'}
    G = nx.Graph()
    for key in node_keys:
        for node,data in json_data[key].items():
            attr = {map_keys_to_attr_keys[key]:data}
            G.add_node(node,**attr)
    for key in edge_keys:
        for edge_str,data in json_data[key].items():
            src_edge,dst_edge = edge_str.split('-')
            attr = {map_keys_to_attr_keys[key]:data}
            G.add_edge(src_edge,dst_edge,**attr)
    return G
