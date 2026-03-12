import warnings
warnings.filterwarnings("ignore", "An issue occurred while importing 'torch-scatter'.")

import optuna
import argparse
import torch.nn as nn
from src.model_launcher import ModelLauncher
from src.constants import Constants
from configs.att_edge_aware_config import AttEdgeAwareGNNConfig
from configs.graphsage_config import GraphSAGEConfig
from configs.gcn_config import GCNConfig
from configs.cagnn_config import CAGNNConfig
import json


def get_config_class(name):
    return {
        'AttEAGNN': AttEdgeAwareGNNConfig,
        'GraphSAGE': GraphSAGEConfig,
        'GCN': GCNConfig,
        'CAGNN': CAGNNConfig,
    }.get(name, None)


def build_objective(config_cls, dataset, split):
    def objective(trial):
        # Generic hyperparameters
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
        epochs = trial.suggest_categorical('epochs', [50, 100, 200, 300])
        lr = trial.suggest_float('lr', 0.0005, 0.01)
        step_size = trial.suggest_categorical('step_size', [25, 50, 75])
        gamma_scheduler = trial.suggest_float('gamma_scheduler', 0.1, 0.99)

        # instantiate config and set suggested values
        cfg = config_cls()
        cfg.dropout = float(dropout)
        cfg.hidden_dim = int(hidden_dim)
        cfg.lr = float(lr)
        cfg.step_size = int(step_size)
        cfg.gamma_scheduler = float(gamma_scheduler)
        cfg.epochs = int(epochs)

        # Apply dataset/split overrides if present
        apply_fn = getattr(cfg, 'apply_dataset_split', None)
        if callable(apply_fn):
            cfg.apply_dataset_split(dataset, split)

        # Use the correct GML path
        path_to_gml = Constants.path_to_abilene_data if dataset == 'abilene' else Constants.path_to_rnp_data

        launcher = ModelLauncher(cfg)
        val_loss = launcher.train(path_to_gml, nn.MSELoss(), dataset=dataset, tm_split=split, return_val_loss=True)

        if val_loss is None:
            return float('inf')
        return float(val_loss)

    return objective


def parse_args():
    p = argparse.ArgumentParser(description='Hyperparameter optimization for models')
    p.add_argument('--models', nargs='+', choices=['AttEAGNN', 'GraphSAGE', 'GCN', 'CAGNN'], default=['AttEAGNN'], help='Models to optimize')
    p.add_argument('--dataset', choices=['abilene', 'rnp'], default='rnp', help='Dataset to use for optimization')
    p.add_argument('--split', choices=['day', 'week'], default='day', help='Traffic-matrix split')
    p.add_argument('--n-trials', type=int, default=20, help='Number of optuna trials per model')
    p.add_argument('--dry-run', action='store_true', help='Print planned runs and exit')
    p.add_argument('--no-edge-norm', action='store_false', dest='use_edge_norm',
                   help='Disable z-score normalization of edge features during optimization')
    return p.parse_args()


def main():
    args = parse_args()

    runs = []
    for m in args.models:
        cfg_cls = get_config_class(m)
        if cfg_cls is None:
            print(f'Unknown model {m}, skipping')
            continue
        runs.append((m, cfg_cls))

    if args.dry_run:
        print('Planned optimization runs:')
        for m, _ in runs:
            print(f'  Model: {m}, Dataset: {args.dataset}, Split: {args.split}, Trials: {args.n_trials}')
        return

    for model_name, cfg_cls in runs:
        print(f'Optimizing {model_name} on {args.dataset}/{args.split} for {args.n_trials} trials')
        # Build objective with a small wrapper to apply runtime options
        def make_objective():
            def objective(trial):
                # instantiate config inside objective
                cfg = cfg_cls()
                # Apply dataset/split overrides if present
                apply_fn = getattr(cfg, 'apply_dataset_split', None)
                if callable(apply_fn):
                    cfg.apply_dataset_split(args.dataset, args.split)
                # Apply runtime edge-norm override
                if not args.use_edge_norm:
                    cfg.edge_norm_func = None
                # Delegate to original builder which sets trial suggestions
                return build_objective(lambda: cfg, args.dataset, args.split)(trial)
            return objective

        study = optuna.create_study(direction='minimize')
        study.optimize(make_objective(), n_trials=args.n_trials)
        print(f'Best params for {model_name}: {study.best_params}')
        # Save study results
        out = {'model': model_name, 'dataset': args.dataset, 'split': args.split, 'best_params': study.best_params}
        fn = f'hp_{model_name}_{args.dataset}_{args.split}.json'
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f'Saved best params to {fn}')


if __name__ == '__main__':
    main()
