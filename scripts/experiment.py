import os
import pickle
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ray import tune

import genni

with open('experiment_config.json') as json_file:
    config = json.load(json_file)

# data params
dtype = torch.float

# N number of samples; inp_dim is input dimension;
# num_layerrs is hidden dimension; out_dim is output dimension.
# N = 2 ** 14

num_channels = 1

# Check network architecture
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
config["model_seed"] = tune.grid_search(config["model_seed_param"])

# Optimization method
config["batch_size"] = tune.grid_search(config["batch_size_param"])

# Run time of optimization
# FIXME CHECK THIS config["mean_loss_threshold"] = None  # 0.01 # 0.15

# Run experiements
if config["device"] == "gpu":
    tune.run(
        lambda config_inp: genni.training.train(config_inp, folder_path),
        config=config,
        resources_per_trial={"gpu": 1},
    )  # have a lambda that init's different center models. and then each one has some number of models that search. right
else:
    tune.run(lambda config_inp: genni.training.train(config_inp, folder_path), config=config)
