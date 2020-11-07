import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ray import tune

import genni

config = {}

config["seed"] = 0
config["device"] = "cpu"

# data params
dtype = torch.float

# N number of samples; inp_dim is input dimension;
# num_layerrs is hidden dimension; out_dim is output dimension.
N = 2 ** 14
inp_dim = 12
out_dim = 1
config["data_meta"] = {
    "N": N,
    "inp_dim": inp_dim,  # if conv net then will update it to make it square.
    "out_dim": out_dim,
}

num_channels = 1


# network architecture
config["net_name"] = "SimpleNet"

if config["net_name"] == "SimpleNet":
    width = 2  # tune.grid_search([64])
    num_layers = 1  #  tune.grid_search([16])
    add_bias = False
    config["net_params"] = [inp_dim, out_dim, width, num_layers, add_bias]
elif config["net_name"] == "LinearNet":
    config["net_params"] = [inp_dim, out_dim]
elif config["net_name"] == "BatchNormSimpleNet":
    config["net_params"] = [inp_dim, out_dim]
elif config["net_name"] == "LeNet":
    config["net_params"] = [inp_dim, inp_dim, num_channels, out_dim]
elif config["net_name"] == "KeskarC3":
    config["net_params"] = [inp_dim, inp_dim, num_channels, out_dim]

# Model Init
config["num_nets"] = 10  # random initialization
config["model_seed"] = tune.grid_search([10])

# Optimization method
config["optimizer"] = "SGD"  # "Adam"
config["learning_rate"] = 0.015  # tune.grid_search(list(np.linspace(0.00005, 0.015, 10)))
config["momentum"] = 0
config["batch_size"] = tune.grid_search([256])

# Run time of optimization
config["num_steps"] = 1000  # tune.grid_search([25000]) # roughly 50 * 500 / 16
config["mean_loss_threshold"] = None  # 0.01 # 0.15

# Parameters for saving and printing
config["save_model_freq"] = 25
config["print_stat_freq"] = 25


# --- Set up folder in which to store all results ---
folder_name = genni.utils.get_file_stamp()
cwd = os.environ["PATH_TO_GENNI_FOLDER"]
folder_path = os.path.join(cwd, "experiments", folder_name)
print(folder_path)
os.makedirs(folder_path)

# Run experiements
if config["device"] == "gpu":
    tune.run(
        lambda config_inp: genni.training.train(config_inp, folder_path),
        config=config,
        resources_per_trial={"gpu": 1},
    )  # have a lambda that init's different center models. and then each one has some number of models that search. right
else:
    tune.run(lambda config_inp: genni.training.train(config_inp, folder_path), config=config)
