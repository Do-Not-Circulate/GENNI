import copy
import datetime
import os
import random
import socket

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_file_stamp():
    """Return time and hostname as string for saving files related to the current experiment"""

    host_name = socket.gethostname()
    mydate = datetime.datetime.now()
    return "{}_{}".format(mydate.strftime("%b%d_%H-%M-%S"), host_name)


def get_time_stamp():
    """Return time as string for saving files related to the current experiment"""
    mydate = datetime.datetime.now()
    return "{}".format(mydate.strftime("%b%d_%H-%M-%S"))


def set_seed(seed, force_cuda=False):
    """Seed all RNG's in use

    seed: seed to use
    force_cuda: force cuda to be deterministic, this leads to performance penalties"""
    if seed is None:
        seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if force_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_params_vec(net):
    param_vec = torch.cat([p.view(-1) for p in net.parameters()])
    return param_vec


def get_grad_params_vec(net):
    param_vec = torch.cat([p.grad.view(-1) for p in net.parameters()])
    return param_vec


def zero_model_gradients(model):
    for p in model.parameters():
        p.grad.data.zero_()


def vec_to_net(vec, net):
    new_net = copy.deepcopy(net)
    dict_new_net = dict(new_net.named_parameters())
    start_point = 0
    for name1, param1 in net.named_parameters():
        end_point = start_point + param1.numel()
        dict_new_net[name1].data.copy_(vec[start_point:end_point].reshape(param1.shape))
        dict_new_net[name1].data.requires_grad = True
        start_point = end_point
    return new_net


def get_model_num_params(model):
    num_params = 0
    for name1, param1 in model.named_parameters():
        num_params += param1.numel()
    return num_params


def get_net_loss(net, data_loader, criterion, full_dataset=False, device=None):
    loss_sum = 0
    for idx, (inputs, labels) in enumerate(data_loader):
        if device is not None:
            inputs, labels = (
                inputs.to(device).type(torch.cuda.FloatTensor),
                labels.to(device).type(torch.cuda.LongTensor),
            )
        else:
            inputs = inputs.float()
        outputs = net(inputs)
        loss_sum += float(criterion(outputs, labels))
        if not full_dataset:
            break

    return loss_sum / (idx + 1)


def get_model_outputs(net, data, softmax_outputs=False, device=None):
    inputs, labels = iter(DataLoader(data, batch_size=len(data))).next()
    if device is not None:
        inputs, labels = (
            inputs.to(device).type(torch.cuda.FloatTensor),
            labels.to(device).type(torch.cuda.LongTensor),
        )
    else:
        inputs = inputs.float()

    outputs = net(inputs)
    if softmax_outputs:
        m = torch.nn.Softmax(dim=-1)
        outputs = m(outputs)
    return outputs
