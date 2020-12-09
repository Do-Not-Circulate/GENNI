import os

import torch

from ..save_load import cache_data, get_all_models
from ..utils import get_net_loss, get_params_vec, vec_to_net
from .interpolation import (
    create_offset_orthonorm_basis,
    get_coordinates,
    get_model_interpolate_grid,
)


# TODO how should we get the data?
def main(experiment_folder, exp_id, config, data_loader):

    criterion = torch.nn.MSELoss()

    center_model = get_all_models(experiment_folder, config["center"]["step"])[exp_id][
        str(config["center"]["idx"])
    ]

    basis_vectors = [
        get_params_vec(
            get_all_models(experiment_folder, config["basis_vectors"]["steps"][i])[
                exp_id
            ][str(config["basis_vectors"]["model_idxs"][i])]
        )
        for i in range(len(config["basis_vectors"]["steps"]))
    ]

    basis_orthonorm_vectors = create_offset_orthonorm_basis(center_model, basis_vectors)
    basis_orthonorm_vectors = [torch.Tensor(v) for v in basis_orthonorm_vectors]

    config["basis_orthonorm_vectors"] = basis_orthonorm_vectors

    basis_coordinates = [
        get_coordinates(b, basis_orthonorm_vectors, get_params_vec(center_model))
        for b in basis_vectors
    ]
    basis_losses = [
        get_net_loss(
            vec_to_net(b, center_model), data_loader, criterion, full_dataset=True
        )
        for b in basis_vectors
    ]
    config["basis_coordinates"] = basis_coordinates
    config["basis_losses"] = basis_losses

    config["center_coordinates"] = [[0] * len(basis_vectors)]
    config["center_loss"] = get_net_loss(
        center_model, data_loader, criterion, full_dataset=True
    )

    num_inter_models = config["num_inter_models"]
    grid_bound = config["grid_bound"]

    # Get the loss on the grid
    func = lambda m: get_net_loss(
        m, data_loader, criterion, full_dataset=True, device=None
    )
    grid_vals = get_model_interpolate_grid(
        center_model, basis_orthonorm_vectors, num_inter_models, grid_bound, func
    )

    cache_folder = cache_data(
        experiment_folder,
        "{}d_loss".format(len(basis_vectors)),
        grid_vals,
        meta_dict=config,
        time_stamp=True,
    )

    config["cache_folder"] = cache_folder
    config["cache_name"] = {
        "loss": "{}d_loss".format(len(basis_vectors)),
        "timestamp": os.path.split(cache_folder)[-1],
    }

    return grid_vals, config
