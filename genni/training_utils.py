import numpy as np
import torch

from .utils import *
from .nets.Nets import SimpleNet, LeNet, LinearNet


def get_nets(net_name, net_params, num_nets, device=None):
    if net_name == "SimpleNet":
        nets = [SimpleNet(*net_params) for _ in range(num_nets)]
    elif net_name == "LeNet":
        nets = [LeNet(*net_params) for _ in range(num_nets)]
    elif net_name == "LinearNet":
        nets = [LinearNet(*net_params) for _ in range(num_nets)]
    elif net_name == "BatchNormSimpleNet":
        nets = [BatchNormSimpleNet(*net_params) for _ in range(num_nets)]
    elif net_name == "KeskarC3":
        nets = [KeskarC3(*net_params) for _ in range(num_nets)]
    else:
        raise NotImplementedError("{} is not implemented.".format(net_name))
    if device is not None:
        nets = [net.to(device) for net in nets]
    return nets


def get_stopping_criterion(num_steps, mean_loss_threshold):
    if (num_steps is not None) and (mean_loss_threshold is not None):
        stopping_criterion = lambda ml, s: (num_steps < s) or (ml < mean_loss_threshold)
    elif num_steps is not None:
        stopping_criterion = lambda ml, s: num_steps < s
    elif mean_loss_threshold is not None:
        stopping_criterion = lambda ml, s: ml < mean_loss_threshold
    else:
        raise Exception("Error: Did not provide a stopping criterion.")
    return stopping_criterion

