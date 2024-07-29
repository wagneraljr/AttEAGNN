import numpy as np
import pandas as pd
import os
import networkx as nx

def simulate_bimodal_traffic(G, num_intervals, K, output_dir):
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    for interval in range(num_intervals):
        traffic_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    s = np.random.uniform(0, 1)
                    if s > 0.8:
                        traffic_matrix[i, j] = np.random.normal(400, 100)
                    else:
                        traffic_matrix[i, j] = np.random.normal(800, 100)
        
        df = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
        file_path = os.path.join(output_dir, f"bimodal_tm_{interval}.dat")
        df.to_csv(file_path, sep=',', header=False, index=False)

def simulate_gravity_traffic(G, num_intervals, K, output_dir, cyclical=True, q=5):
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    def gravity_demand():
        traffic_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    Mi = sum([G[nodes[i]][k]['bw'] for k in G[nodes[i]]])
                    Mj = sum([G[k][nodes[j]]['bw'] for k in G[nodes[j]]])
                    Dij = G[nodes[i]][nodes[j]]['dist'] if G.has_edge(nodes[i], nodes[j]) else 1
                    traffic_matrix[i, j] = K * Mi * Mj / (Dij ** 2)
        return traffic_matrix
    
    if cyclical:
        base_matrices = [gravity_demand() for _ in range(q)]
        for interval in range(num_intervals):
            traffic_matrix = base_matrices[interval % q]
            df = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"gravity_cyclical_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)
    else:
        traffic_matrices = [gravity_demand() for _ in range(num_intervals)]
        for interval in range(num_intervals):
            avg_matrix = np.mean(traffic_matrices[max(0, interval - q + 1):interval + 1], axis=0)
            df = pd.DataFrame(avg_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"gravity_averaging_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)

def sparsify_traffic_matrix(matrix, p):
    sparsified_matrix = np.copy(matrix)
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if np.random.uniform(0, 1) > p:
                sparsified_matrix[i, j] = 0
    return sparsified_matrix

def simulate_sparsified_gravity_traffic(G, num_intervals, K, output_dir, p, cyclical=True, q=5):
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    def gravity_demand():
        traffic_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    Mi = sum([G[nodes[i]][k]['bw'] for k in G[nodes[i]]])
                    Mj = sum([G[k][nodes[j]]['bw'] for k in G[nodes[j]]])
                    Dij = G[nodes[i]][nodes[j]]['dist'] if G.has_edge(nodes[i], nodes[j]) else 1
                    traffic_matrix[i, j] = K * Mi * Mj / (Dij ** 2)
        return traffic_matrix
    
    if cyclical:
        base_matrices = [sparsify_traffic_matrix(gravity_demand(), p) for _ in range(q)]
        for interval in range(num_intervals):
            traffic_matrix = base_matrices[interval % q]
            df = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"sparsified_gravity_cyclical_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)
    else:
        traffic_matrices = [sparsify_traffic_matrix(gravity_demand(), p) for _ in range(num_intervals)]
        for interval in range(num_intervals):
            avg_matrix = np.mean(traffic_matrices[max(0, interval - q + 1):interval + 1], axis=0)
            df = pd.DataFrame(avg_matrix, index=nodes, columns=nodes)
            file_path = os.path.join(output_dir, f"sparsified_gravity_averaging_tm_{interval}.dat")
            df.to_csv(file_path, sep=',', header=False, index=False)

# Initializes graph structure
G = nx.Graph()

# Add PoPs as nodes with geographical location information
G.add_node('Porto Alegre', pos = (-30.033056, -51.230000))
G.add_node('Florianopolis', pos = (-27.593500, -48.558540))
G.add_node('Curitiba', pos = (-25.441105, -49.276855))
G.add_node('Sao Paulo', pos = (-23.533773, -46.625290))
G.add_node('Campo Grande', pos = (-20.4435, -54.6478))
G.add_node('Goiania', pos = (-16.665136, -49.286041))
G.add_node('Cuiaba', pos = (-15.5989, -56.0949))
G.add_node('Porto Velho', pos = (-8.76183, -63.902))
G.add_node('Rio Branco', pos = (-9.974, -67.8076))
G.add_node('Brasilia', pos = (-15.793889, -47.882778))
G.add_node('Rio de Janeiro', pos = (-22.908333, -43.196388))
G.add_node('Belo Horizonte', pos = (-19.912998, -43.940933))
G.add_node('Salvador', pos = (-12.974722, -38.476665))
G.add_node('Fortaleza', pos = (-3.731862, -38.526669))
G.add_node('Sao Luis', pos = (-2.53073, -44.3068))
G.add_node('Manaus', pos = (-3.117034, -60.025780))
G.add_node('Palmas', pos = (-10.1689, -48.3317))
G.add_node('Vitoria', pos = (-20.3222, -40.3381))
G.add_node('Aracaju', pos = (-10.9095, -37.0748))
G.add_node('Campina Grande', pos = (-7.23072, -35.8817))
G.add_node('Teresina', pos = (-5.08921, -42.8016))
G.add_node('Belem', pos = (-1.45502, -48.5024))
G.add_node('Macapa', pos = (0.0344566, -51.0666))
G.add_node('Boa Vista', pos = (2.81954, -60.6714))
G.add_node('Natal', pos = (-5.79448, -35.211))
G.add_node('Joao Pessoa', pos = (-7.11532, -34.861))
G.add_node('Recife', pos = (-8.05428, -34.8813))
G.add_node('Maceio', pos = (-9.66625, -35.7351))

# Add links as edges with bandwidth and geographical distance as edge features
G.add_edge('Porto Alegre', 'Florianopolis', bw = 100, dist = 375.91)
G.add_edge('Porto Alegre', 'Curitiba', bw = 200, dist = 545.55)
G.add_edge('Porto Alegre', 'Sao Paulo', bw = 100, dist = 854.86)
G.add_edge('Florianopolis', 'Curitiba', bw = 100, dist = 249.78)
G.add_edge('Curitiba', 'Sao Paulo', bw = 300, dist = 342.00)
G.add_edge('Curitiba', 'Campo Grande', bw = 10, dist = 781.64)
G.add_edge('Campo Grande', 'Goiania', bw = 10, dist = 704.13)
G.add_edge('Campo Grande', 'Cuiaba', bw = 100, dist = 559.99)
G.add_edge('Cuiaba', 'Porto Velho', bw = 100, dist = 1138.87)
G.add_edge('Porto Velho', 'Rio Branco', bw = 6, dist = 449.18)
G.add_edge('Cuiaba', 'Brasilia', bw= 100, dist = 879.31)
G.add_edge('Brasilia', 'Goiania', bw = 20, dist = 178.41)
G.add_edge('Brasilia', 'Sao Paulo', bw = 20, dist = 870.63)
G.add_edge('Brasilia', 'Rio de Janeiro', bw = 100, dist = 931.19)
G.add_edge('Brasilia', 'Belo Horizonte', bw = 10, dist = 619.47)
G.add_edge('Brasilia', 'Salvador', bw = 100, dist = 1060.33)
G.add_edge('Brasilia', 'Fortaleza', bw = 10, dist = 1686.96)
G.add_edge('Brasilia', 'Sao Luis', bw = 100, dist = 1525.90)
G.add_edge('Brasilia', 'Manaus', bw = 3, dist = 1937.24)
G.add_edge('Brasilia', 'Palmas', bw = 100, dist = 627.36)
G.add_edge('Sao Paulo', 'Rio de Janeiro', bw = 200, dist = 357.21)
G.add_edge('Sao Paulo', 'Belo Horizonte', bw = 100, dist = 488.82)
G.add_edge('Sao Paulo', 'Fortaleza', bw = 200, dist = 2367.50)
G.add_edge('Rio de Janeiro', 'Belo Horizonte', bw = 20, dist = 341.87)
G.add_edge('Rio de Janeiro', 'Vitoria', bw = 100, dist = 412.28)
G.add_edge('Vitoria', 'Salvador', bw = 100, dist = 840.68)
G.add_edge('Vitoria', 'Belo Horizonte', bw = 10, dist = 378.91)
G.add_edge('Belo Horizonte', 'Salvador', bw = 100, dist = 966.57)
G.add_edge('Salvador', 'Aracaju', bw = 100, dist = 275.66)
G.add_edge('Salvador', 'Campina Grande', bw = 10, dist = 698.97)
G.add_edge('Salvador', 'Fortaleza', bw = 100, dist = 1027.77)
G.add_edge('Salvador', 'Teresina', bw = 100, dist = 997.00)
G.add_edge('Teresina', 'Fortaleza', bw = 100, dist = 497.38)
G.add_edge('Teresina', 'Sao Luis', bw = 100, dist = 329.88)
G.add_edge('Sao Luis', 'Belem', bw = 10, dist = 481.34)
G.add_edge('Belem', 'Macapa', bw = 100, dist = 329.71)
G.add_edge('Belem', 'Palmas', bw = 100, dist = 969.12)
G.add_edge('Macapa', 'Manaus', bw = 100, dist = 1055.59)
G.add_edge('Manaus', 'Boa Vista', bw = 1, dist = 664.01)
G.add_edge('Boa Vista', 'Fortaleza', bw = 1, dist = 2566.51)
G.add_edge('Fortaleza', 'Natal', bw = 100, dist = 433.10)
G.add_edge('Natal', 'Campina Grande', bw = 100, dist = 176.05)
G.add_edge('Campina Grande', 'Joao Pessoa', bw = 10, dist = 113.34)
G.add_edge('Campina Grande', 'Recife', bw = 100, dist = 143.32)
G.add_edge('Joao Pessoa', 'Recife', bw = 10, dist = 104.43)
G.add_edge('Recife', 'Maceio', bw = 10, dist = 202.30)
G.add_edge('Maceio', 'Aracaju', bw = 100, dist = 201.48)

# Sets output directory
output_dir = "./traffic_matrices"
os.makedirs(output_dir, exist_ok=True)

# Create one month's worth of TMs with cycles of 7 days
simulate_bimodal_traffic(G, 8640, 1, output_dir)

simulate_gravity_traffic(G, 8640, 1, output_dir, cyclical=True, q=2016)
simulate_gravity_traffic(G, 8640, 1, output_dir, cyclical=False, q=2016)

simulate_sparsified_gravity_traffic(G, 8640, 1, output_dir, p=0.5, cyclical=True, q=2016)
simulate_sparsified_gravity_traffic(G, 8640, 1, output_dir, p=0.5, cyclical=False, q=2016)
