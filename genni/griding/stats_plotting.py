import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..utils import *
from .postprocessing import *


def different_cols(df):
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[0] != a[1:]).any(0)


def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict


def get_models_loss(models, data_set, criterion, device=None, seed=None):
    set_seed(seed)

    loss_dict = {}

    data_loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)

    for k, m in models.items():
        if device is not None:
            m = m.to(device)
        loss_dict[k] = get_net_loss(m, data_set, criterion, device=device)
    return loss_dict


def get_exp_loss(experiment_folder, step, num_datapoints=-1, seed=0, device=None):
    # init
    loss_dict = {}

    cfgs = load_configs(experiment_folder)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        models_dict = get_models(curr_path, step, device)

        data_set = get_data(cfgs.loc[exp_name], device=device)

        data_loader = DataLoader(
            data_set, batch_size=cfgs.loc[exp_name]["batch_size"], shuffle=False
        )

        if models_dict is None:
            continue
        loss_dict[exp_name] = get_models_loss(
            models_dict, data_loader, criterion, device=device
        )
        # cache data
        cache_data(experiment_folder, "loss", loss_dict, step=step)

    return loss_dict


def different_cols(df):
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[0] != a[1:]).any(0)


def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict


def get_end_stats(exp_folder, step=-1, with_min_max=False):

    loss, _ = load_cached_data(exp_folder, "loss", step=step)

    stats_dict = {}
    configs = load_configs(exp_folder)
    for exp_id in configs.index:

        num_nets = configs.loc[exp_id]["num_nets"]
        if runs is not None:
            num_steps = max(runs[exp_id], key=lambda x: int(x)) - 1

        stats_dict[str(exp_id)] = {}

        # make this multi index. Otherwise should be mostly godo
        for nn in range(num_nets):
            Loss_list = [loss[exp_id][str(nn)]]

            stats_dict[str(exp_id)]["Loss Mean"] = np.mean(Loss_list)

            if with_min_max:
                stats_dict[str(exp_id)]["Loss Max"] = np.max(Loss_list)
                stats_dict[str(exp_id)]["Loss Min"] = np.min(Loss_list)

    stats_pd = pd.DataFrame.from_dict(stats_dict, orient="index")

    # append hyperparameters to DataFrame
    cfs_hp = get_hp(configs)
    cfs_hp_df = configs[list(cfs_hp.keys())]
    stats_pd = pd.concat([stats_pd, cfs_hp_df], axis=1)

    return stats_pd


def main():
    # example use

    root_folder = os.environ["PATH_TO_GENNI_FOLDER"]
    exp = ""
    experiment_folder = os.path.join(root_folder, "experiments", exp)

    # init torch
    is_gpu = False
    if is_gpu:
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None

    # # compute all loss over time
    f = lambda step: get_exp_loss(
        experiment_folder, step, num_datapoints=-1, device=device
    )
    get_all_steps_f(experiment_folder, f)
