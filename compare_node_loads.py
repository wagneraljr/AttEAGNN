import os
import sys
import numpy as np
from importlib.machinery import SourceFileLoader

# Add repo src to path
sys.path.insert(0, os.path.abspath(''))
from src.constants import Constants
from src.utils.data_util import DataUtil as RepoDataUtil

# Load EdgeAwareGNN-main data_utils via SourceFileLoader (folder name contains hyphen)
edge_main_path = os.path.join('EdgeAwareGNN-main', 'data_utils.py')
edge_main = SourceFileLoader('edge_main', edge_main_path).load_module()

# Find a sample .dat file in Abilene day folder
tm_folder = Constants.path_to_abilene_day_tm_files
files = [f for f in os.listdir(tm_folder) if f.endswith('.dat')]
if not files:
    print('No .dat files found in', tm_folder)
    sys.exit(1)

sample = os.path.join(tm_folder, sorted(files)[0])
print('Using TM file:', sample)

# Compute loads
repo_loads = RepoDataUtil.get_node_loads(sample, abilene=True)
edge_main_loads = edge_main.get_node_loads(sample)

# Convert to numpy
repo_np = repo_loads.detach().numpy().reshape(-1)
edge_np = edge_main_loads.detach().numpy().reshape(-1)

# Print shapes and first values
print('repo loads shape:', repo_np.shape)
print('edge_main loads shape:', edge_np.shape)

print('\nFirst 10 values (repo):', np.round(repo_np[:10], 6))
print('First 10 values (edge_main):', np.round(edge_np[:10], 6))

# Differences
min_len = min(len(repo_np), len(edge_np))
diff = repo_np[:min_len] - edge_np[:min_len]
print('\nMax abs difference:', float(np.max(np.abs(diff))))
print('Mean abs difference:', float(np.mean(np.abs(diff))))
print('L1 norm diff:', float(np.sum(np.abs(diff))))

# If lengths differ, report
if len(repo_np) != len(edge_np):
    print('\nLengths differ: repo', len(repo_np), 'edge_main', len(edge_np))
else:
    print('\nLengths equal:', len(repo_np))

# Exit

