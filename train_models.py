import warnings
warnings.filterwarnings("ignore", "An issue occurred while importing 'torch-scatter'.")

import argparse
import torch.nn as nn
from src.constants import Constants
from src.model_launcher import ModelLauncher
from configs.att_edge_aware_config import AttEdgeAwareGNNConfig
from configs.graphsage_config import GraphSAGEConfig
from configs.gcn_config import GCNConfig
from configs.cagnn_config import CAGNNConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train AttEAGNN on a chosen dataset')
    parser.add_argument('--dataset', choices=['abilene', 'rnp'], default='abilene',
                        help='Which dataset to use (abilene or rnp).')
    parser.add_argument('--split', choices=['day', 'week'], default='day',
                        help='Which TM split to use for training (day or week). Default: day')
    parser.add_argument('--no-original', action='store_false', dest='use_original',
                        help='Disable original edge attributes (default: enabled)')
    parser.add_argument('--no-betweenness', action='store_false', dest='use_betweenness',
                        help='Disable betweenness centrality feature (default: enabled)')
    parser.add_argument('--no-degree', action='store_false', dest='use_degree',
                        help='Disable edge degree feature (default: enabled)')
    parser.add_argument('--no-clustering', action='store_false', dest='use_clustering',
                        help='Disable clustering coefficient feature (default: enabled)')
    parser.add_argument('--no-edge-norm', action='store_false', dest='use_edge_norm',
                        help='Disable z-score normalization of edge features (default: enabled)')
    parser.add_argument('--dry-run', action='store_true', help='Only print chosen paths and exit.')
    parser.add_argument('--models', nargs='+', choices=['AttEAGNN','GraphSAGE','GCN','CAGNN'], default=['AttEAGNN'],
                        help='Which models to train (space separated). Default: AttEAGNN')
    return parser.parse_args()


def main():
    args = parse_args()
    loss_fn = nn.MSELoss()

    if args.dataset == 'abilene':
        path_to_gml_data = Constants.path_to_abilene_data
        dataset_name = 'abilene'
    else:
        path_to_gml_data = Constants.path_to_rnp_data
        dataset_name = 'rnp'

    if args.dry_run:
        print(f"Dry run: dataset={dataset_name}")
        print(f"GML path: {path_to_gml_data}")
        if dataset_name == 'abilene':
            tm_folder = Constants.path_to_abilene_day_tm_files if args.split == 'day' else Constants.path_to_abilene_week_tm_files
            print(f"TM folder: {tm_folder}")
        else:
            # attempt to show the appropriate RNP folder (replace 'day' with selected split)
            rnp_tm = Constants.path_to_day_tm_files.replace('/day/', f'/{args.split}/').replace('\\day\\', f'\\{args.split}\\')
            print(f"TM folder: {rnp_tm}")
        return

    # Build configs according to requested models
    name_to_config = {
        'AttEAGNN': AttEdgeAwareGNNConfig,
        'GraphSAGE': GraphSAGEConfig,
        'GCN': GCNConfig,
        'CAGNN': CAGNNConfig,
    }
    configs = []
    for m in args.models:
        cfg_cls = name_to_config.get(m)
        if cfg_cls:
            configs.append(cfg_cls())

    # Apply dataset/split-specific hyperparameter overrides from each config (if provided)
    for cfg in configs:
        apply_fn = getattr(cfg, 'apply_dataset_split', None)
        if callable(apply_fn):
            cfg.apply_dataset_split(dataset_name, args.split)
        # Apply runtime override for edge feature normalization
        if not args.use_edge_norm:
            cfg.edge_norm_func = None

    for config in configs:
        model_laucher = ModelLauncher(config)
        model_laucher.train(
            path_to_gml_data,
            loss_fn,
            dataset=dataset_name,
            tm_split=args.split,
            use_original=args.use_original,
            use_betweenness=args.use_betweenness,
            use_degree=args.use_degree,
            use_clustering=args.use_clustering,
        )


if __name__ == '__main__':
    main()
