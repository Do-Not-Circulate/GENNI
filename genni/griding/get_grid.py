import torch


# TODO how should we get the data?
def main(experiment_folder, exp_id, config):

    center_model = get_all_models(experiment_folder, config["center"]["step"])[
        config["center"]["idx"]
    ][str(center_idx)]

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

    get_coordinates(
        basis_vectors[0], basis_orthonorm_vectors, get_params_vec(center_model)
    )
    get_coordinates(
        basis_vectors[1], basis_orthonorm_vectors, get_params_vec(center_model)
    )
    get_coordinates(
        basis_vectors[2], basis_orthonorm_vectors, get_params_vec(center_model)
    )

    coordinates = [
        get_coordinates(b, basis_orthonorm_vectors, get_params_vec(center_model))
        for b in basis_vectors
    ]
    losses = [
        get_net_loss(
            vec_to_net(b, center_model), data_loader, criterion, full_dataset=True
        )
        for b in basis_vectors
    ]

    # add center model
    coordinates = [[0] * len(basis_vectors)] + coordinates
    losses = [
        get_net_loss(center_model, data_loader, criterion, full_dataset=True)
    ] + losses

    num_inter_models = config["num_inter_models"]
    grid_bound = config["grid_bound"]

    # Get the loss on the grid
    func = lambda m: get_net_loss(
        m, data_loader, criterion, full_dataset=True, device=None
    )
    three_d_vals = get_model_interpolate_grid(
        center_model, basis_orthonorm_vectors, num_inter_models, grid_bound, func
    )

    cache_data(
        os.path.join(experiment_folder, exp_id),
        "{}d_loss".format(len(basis_vectors)),
        three_d_vals,
        meta_dict=cache_dict,
        time_stamp=True,
    )