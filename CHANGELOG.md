# CHANGELOG

## 2026-03-12 — Improvements: dataset selection and fixes

Added/Modified:

- `train_models.py`
  - Added command-line interface options `--dataset {abilene,rnp}` and `--dry-run`.
  - Allows switching between datasets without editing source files.

- `src/model_launcher.py`
  - `train()` accepts a `dataset` parameter and automatically selects the corresponding traffic-matrix folder.
  - Instantiates the model only after loading data, dynamically adjusting input parameters (`node_input_dim`, `edge_input_dim`).

- `src/utils/train_util.py`
  - Fixed construction of traffic-matrix file paths; attempts the Abilene path when applicable.
  - Passes the `abilene=True` flag to `DataUtil.get_node_loads` to align Abilene TM formatting.

- `README.md`
  - Added a section with instructions to switch datasets and examples for `--dry-run`.

Motivation:

- Make it easier to experiment with both available datasets (Abilene and RNP) without manual code edits.
- Fix path and dimension mismatches observed when using the Abilene dataset.

Notes:

- If you want dataset selection to be persistent in configurations, consider adding a `dataset` field to the `Config` class.
- Review `src/constants.py` if you want to change the default dataset or traffic-matrix paths.

## 2026-03-12 — Feature: edge-feature CLI flags

Added:

- `train_models.py`
  - New CLI flags to enable/disable computed edge features at runtime: `--no-original`, `--no-betweenness`, `--no-degree`, `--no-clustering`. By default all features are used.

Motivation:

- Allow quick experiments to evaluate the impact of individual edge features without changing source code.

