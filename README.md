# AttEAGNN: Attention Based Edge-Aware GNN Applied to Network Load Prediction

## Requirements
* python3
* pip

## Initial Configuration
Install project dependecies:
```
pip install -r requirements.txt
```

## Project Structure
The project structure is organized as follow:
* **data:** Contains the RNP graph, traffic data and geographical data. 
* **results:** Contains the model weights, training information. Additionally, it includes comparative graphics between the models using different metrics.
* **src:** Contains the source code.
* **configs:** Contains the configurations used for each model.

## Reproducibility
### Training and Evaluation Flow of the Models

The training and evaluation of the models consist of the following steps:

1. First, the models are configured using the `Config` class in `src/config.py` and then added to the `train_models.py` file. Sample configurations can be viewed in the `configs/` directory.

2. The configurations are loaded using the `ModelLauncher` class in `src/model_launcher.py`. This class creates the model with the respective configurations and then performs training, saving the results such as losses, final load prediction, model weights, and model name in the `results/` directory.

3. Finally, the script `plot_graphics.py` is executed. The data saved in the previous step is loaded, and metrics such as MAE (Mean Absolute Error), MSE (Mean Squared Error), and R²(Coefficient of Determination) are calculated. The graphics containing these metrics are saved in the `results/` directory.

### Reproduction of the Experiments
To simplify the reproduction of the experiments, a shell script (`run_experiments.sh`) has been provided in the root folder of the project. This script will execute the Python script `train_models.py`, and at the end, the script `plot_graphics.py` is run to evaluate the models and generate the graphics. The training progress can be monitored through the terminal.

To execute the shell script, make sure it has execution permission, and if so, run:
```
./run_experiments.sh
```

If this file does not have execution permission, execute the command below, and then run the previous command:

```
chmod +x run_experiments.sh
```


## Tool Customization
### Train, Evaluate, and Compare Your Own Model

Our tool allows the inclusion of custom GNN models for training and testing on the RNP network by following these steps:

1. Create your model in `src/models/`.
2. Define your model and its configuration in the `configs/` directory using the `Config` class. Note that you should choose a class to encapsulate your data. 
    
    2.a. If your model is a traditional GNN, i.e., it only needs node features and edge indices, you can use the `GNNData` class. 
    
    2.b. If it also uses edge features, use the `EdgeGNNData` class. 
    
    2.c. Otherwise, you can create your own data class from the `DataInterface` interface, as done for the `CAGNNData` class.  These classes can be found in `src/entities`.  
    
    2.d. Note also that you need to define a function to load the network graph data, which will be used to initialize the data class (an extension of `DataInterface`). Use the implemented configurations as a reference.

3. Add your configuration class to `train_models.py`.

4. If you have already run the experiments, execute the training in `train_models.py` only for your model and run the `plot_graphics.py` script. Otherwise, run the `run_experiments.sh` including your model.

5. View the graphics in the `results/` folder.

## Switching between datasets

You can choose the dataset (`abilene` or `rnp`) directly from the command line when running `train_models.py`.

Examples:

- Train using the Abilene dataset (default):

```bash
python train_models.py --dataset abilene
```

- Train using the RNP dataset:

```bash
python train_models.py --dataset rnp
```

- Perform a dry run (prints selected paths without training):

```bash
python train_models.py --dry-run --dataset abilene
python train_models.py --dry-run --dataset rnp
```

Notes:

- Default paths used by the project:
    - Abilene GML: `data/abilene.gml` and TMs in `data/traffic_matrices/abilene/day/`.
    - RNP GML: `data/rnp.gml` and TMs in `data/traffic_matrices/sparsified_gravity_cyclical/day/`.
- To change these paths, edit `src/constants.py`.

Split selection (day/week)

Each dataset contains two TM splits under `data/traffic_matrices/...`: `day` and `week`.
You can choose which split to use during training with the `--split` flag (default: `day`).

Examples:

```bash
# Train on Abilene, using the week split
python train_models.py --dataset abilene --split week

# Train on RNP, using the day split (default)
python train_models.py --dataset rnp --split day

# Dry-run to print chosen paths
python train_models.py --dry-run --dataset abilene --split week
```

Note on results location:

Training outputs are saved under `results/<dataset>/<split>/<model_name>/`.
For example: `results/abilene/day/AttEAGNN/AttEAGNN.ckpt`.

