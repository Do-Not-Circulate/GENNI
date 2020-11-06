import numpy as np
from ..utils import *
from ..data_getters import *
from ..postprocessing.postprocessing import *

import copy
import torch
from tqdm import tqdm
import itertools


def get_coordinates(v, basis_vectors, offset_v):

    adjusted_v = v - offset_v
    coeffs = [float(torch.dot(adjusted_v, b) / torch.norm(b)) for b in basis_vectors]
    return coeffs


def get_models_grid(offset, basis_vectors, num_inter_models, grid_bound):

    basis_m_dicts = [
        dict(vec_to_net(basis_vectors[i], offset).named_parameters())
        for i in range(len(basis_vectors))
    ]

    v_grid = []

    for coeffs in tqdm(
        itertools.product(
            np.linspace(grid_bound[0], grid_bound[1], num_inter_models),
            repeat=len(basis_vectors),
        )
    ):

        curr_model = copy.deepcopy(offset)
        dict_curr_model = dict(curr_model.named_parameters())

        # TODO if i am not computing anything i don't need to switch to model then back
        for name1, param1 in offset.named_parameters():
            v_to_add = sum(
                [
                    basis_m_dicts[i][name1].data * coeffs[-(i + 1)]
                    for i in range(len(basis_vectors))
                ]
            )
            dict_curr_model[name1].data.copy_(dict_curr_model[name1].data + v_to_add)

        v_grid.append(get_params_vec(curr_model).detach().numpy())

    v_grid = np.array(v_grid)
    v_grid = v_grid.reshape(*([num_inter_models] * len(basis_vectors)), -1)

    return v_grid


def get_model_interpolate_grid(
    offset, basis_vectors, num_inter_models, grid_bound, func
):

    basis_m_dicts = [
        dict(vec_to_net(basis_vectors[i], offset).named_parameters())
        for i in range(len(basis_vectors))
    ]

    val_arr = []

    for coeffs in tqdm(
        list(
            itertools.product(
                np.linspace(grid_bound[0], grid_bound[1], num_inter_models),
                repeat=len(basis_vectors),
            )
        )
    ):

        curr_model = copy.deepcopy(offset)
        dict_curr_model = dict(curr_model.named_parameters())

        for name1, param1 in offset.named_parameters():
            v_to_add = sum(
                [
                    basis_m_dicts[i][name1].data * coeffs[-(i + 1)]
                    for i in range(len(basis_vectors))
                ]
            )  # we are going in reverse because the first element in coeff corresponds to the last element of basis_vectors due to itertools.product
            dict_curr_model[name1].data.copy_(dict_curr_model[name1].data + v_to_add)

        to_append = func(curr_model)

        val_arr.append(to_append)

    val_arr = np.array(val_arr)
    val_arr = val_arr.reshape(*([num_inter_models] * len(basis_vectors)))

    return val_arr


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def create_offset_orthonorm_basis_new(center_model, basis_vectors):
    basis_vectors = [
        (basis_vectors[i] - get_params_vec(center_model)).detach().numpy()
        for i in range(len(basis_vectors))
    ]
    basis_vectors = np.array(basis_vectors)
    return gram_schmidt_columns(basis_vectors.T).T


def create_offset_basis(offset, model1, model2):
    """When time, replace usage of this function with "create_offset_orthonorm_basis_new" """
    v1_vec = get_params_vec(model1) - get_params_vec(offset)
    w2_vec = get_params_vec(model2) - get_params_vec(offset)
    proj_w2 = torch.matmul(v1_vec, w2_vec) * v1_vec / torch.norm(v1_vec) ** 2
    assert torch.norm(proj_w2) != torch.norm(w2_vec)  # make sure not lin dep
    v2_vec = w2_vec - proj_w2
    v2_vec = (v2_vec / torch.norm(v2_vec)) * torch.norm(v1_vec)
    # coordinates
    c_offset = (0, 0)  # (x, y)
    c_w1 = (1, 0)
    c_w2 = (
        float(torch.dot(w2_vec, v1_vec) / (torch.norm(v1_vec) ** 2)),
        float(torch.dot(w2_vec, v2_vec) / (torch.norm(v2_vec) ** 2)),
    )
    return offset, v1_vec, v2_vec, c_offset, c_w1, c_w2


if __name__ == "__main__":
    pass
