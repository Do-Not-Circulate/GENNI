import itertools
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.utils.data import DataLoader

from . import data_getters, griding, save_load, utils
from .griding import interpolation

# Colors for 2/3d plots
COLOR_CENTER = "gold"
COLOR1 = "thistle"
COLOR2 = "lightsalmon"
COLOR3 = "skyblue"

## Data extraction
def extract_experiment(genni_config_path):
    with open(genni_config_path, "rb") as f:
        genni_config = yaml.safe_load(f)
    genni_home = Path(genni_config["genni_home"])
    genni_exp_dir = genni_home / "experiments"

    exp_dict = OrderedDict()
    for child in genni_exp_dir.iterdir():
        exp_dict[child.name] = [ids.name for ids in (child / "models").iterdir()]
    return genni_exp_dir, exp_dict


def create_experiment_folder_and_hyperparam_id(
    genni_exp_dir, exp_dict, experiment_idx=0, hyperparam_id=0
):
    experiment_name = list(exp_dict.keys())[experiment_idx]
    experiment_id = exp_dict[experiment_name][hyperparam_id]
    experiment_folder = genni_exp_dir / experiment_name
    return experiment_folder, experiment_id


def create_data_loader(experiment_folder, exp_id, N=500, device="cpu"):
    cache_dict = {}
    cfgs = save_load.load_configs(experiment_folder)
    cfg = cfgs.loc[exp_id]
    cfg["data_meta"]["N"] = N
    cache_dict["N"] = cfg["data_meta"]["N"]

    data_set = data_getters.get_data(cfg, device=device)
    data_loader = DataLoader(
        data_set, batch_size=cfgs.loc[exp_id]["batch_size"], shuffle=False
    )

    return cfg, cache_dict, data_loader


## Plotting
# Surface Plot
def plot_surface(vals_filtered, grid_arr, meta_data, update_layout=False):
    fig = go.Figure(
        data=[
            go.Surface(
                z=vals_filtered + 0,
                x=grid_arr,
                y=grid_arr,
                colorscale="viridis",
                name="log(J(θ))",  # cmax=0.001, cmin=0,
                colorbar=dict(
                    thickness=130,
                    title="log(J(θ))",
                    titlefont=dict(size=80),
                    lenmode="fraction",
                    len=0.80,
                    x=0.78,
                    y=0.45,
                    tickfont=dict(size=70),
                ),
            ),
            go.Scatter3d(
                x=np.array([0]),
                y=np.array([0]),
                z=[0.0 + meta_data["center_loss"]],
                mode="markers",
                marker=dict(size=20, color=[COLOR_CENTER]),
                showlegend=False,
            ),
            go.Scatter3d(
                x=np.array([meta_data["basis_coordinates"][0][0]]),
                y=np.array([meta_data["basis_coordinates"][0][1]]),
                z=[0.0 + meta_data["basis_losses"][0]],
                mode="markers",
                marker=dict(size=20, color=[COLOR1]),
                showlegend=False,
            ),
            go.Scatter3d(
                x=np.array([meta_data["basis_coordinates"][1][0]]),
                y=np.array([meta_data["basis_coordinates"][1][1]]),
                z=[0.0 + meta_data["basis_losses"][0]],
                mode="markers",
                marker=dict(size=20, color=[COLOR2]),
                showlegend=False,
            ),
        ]
    )
    if update_layout:
        fig.update_layout(
            title=None,
            autosize=False,
            width=1280,
            height=720,
            font=dict(size=27,),
            scene={
                "zaxis": dict(
                    title="log(J(θ))",
                    titlefont=dict(size=80),
                    range=[0, 0.001],
                    tickmode="linear",
                    tick0=0,
                    dtick=0.001,
                    tickangle=0,
                    ticks="outside",
                    tickwidth=0,
                ),
                "xaxis": dict(
                    range=[-10, 14],
                    title="c1",
                    titlefont=dict(size=80),
                    tickmode="array",
                    tickvals=[-8, 13],
                    tickangle=0,
                    ticks="outside",
                    tickwidth=0,
                ),
                "yaxis": dict(
                    range=[-14, 11],
                    title="c2",
                    titlefont=dict(size=80),
                    tickmode="array",
                    tickvals=[-13, 10],
                    tickangle=0,
                ),
            },
        )

    return fig


# Contour Plot
def plot_contour(
    vals_filtered, grid_arr, meta_data, update_layout=False, markersize=12
):
    fig = go.Figure(
        data=[
            go.Contour(
                z=np.log(vals_filtered),
                x=grid_arr,
                y=grid_arr,
                colorscale="viridis",
                colorbar=dict(title="log(J(θ))"),
            ),
            go.Scatter(
                x=np.array(0),
                y=np.array(0),
                marker=dict(color=COLOR_CENTER, size=markersize),
                showlegend=False,
            ),
            go.Scatter(
                x=np.array([meta_data["basis_coordinates"][0][0]]),
                y=np.array([meta_data["basis_coordinates"][0][1]]),
                marker=dict(color=COLOR1, size=markersize),
                showlegend=False,
            ),
            go.Scatter(
                x=np.array([meta_data["basis_coordinates"][1][0]]),
                y=np.array([meta_data["basis_coordinates"][1][1]]),
                marker=dict(color=COLOR2, size=markersize),
                showlegend=False,
            ),
        ]
    )
    if update_layout:
        fig.update_layout(
            title="",
            autosize=False,
            font=dict(size=22,),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            width=800,
            height=600,
            xaxis=dict(title="c1"),
            yaxis=dict(title="c2"),
        )

    return fig


def plot_3d(three_d_vals, grid_coeffs, grid_filter):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                z=grid_coeffs[:, 0],
                x=grid_coeffs[:, 2],
                y=grid_coeffs[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=three_d_vals.reshape(-1)[grid_filter],
                    colorscale="viridis",
                    opacity=1,
                    colorbar=dict(
                        thickness=130,
                        title="J(θ)",
                        titlefont=dict(size=80),
                        lenmode="fraction",
                        len=0.80,
                        x=0.78,
                        y=0.55,
                        tickfont=dict(size=70),
                    ),
                ),
            )
        ]
    )

    fig.update_layout(
        title=None,
        autosize=False,
        width=1280,
        height=720,
        font=dict(size=27,),
        scene={
            "zaxis": dict(title="c3", titlefont=dict(size=80)),
            "xaxis": dict(title="c1", titlefont=dict(size=80)),
            "yaxis": dict(title="c2", titlefont=dict(size=80)),
        },
    )

    return fig


def plot_umap(dataset, labels, projections, normalise=0.0025):
    fig, ax = plt.subplots(1, 1)
    sns.set_style(
        "white",
        {
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        },
    )
    ax = sns.scatterplot(
        x="x1",
        y="x2",
        hue="label",
        data=dataset,
        s=10,
        palette="viridis",
        edgecolor="none",
        ax=ax,
    )

    ax = sns.scatterplot(
        x=projections[-4:, 0],
        y=projections[-4:, 1],
        s=300,
        hue=[COLOR_CENTER, COLOR1, COLOR2, COLOR3],
        palette=[COLOR_CENTER, COLOR1, COLOR2, COLOR3],
        ax=ax,
    )

    ax.set(yticks=[], xticks=[], xlabel="", ylabel="")
    ax.get_legend().remove()

    norm = plt.Normalize(0, normalise)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    cax = inset_axes(
        ax,
        width="4%",  # width = 5% of parent_bbox width
        height="75%",  # height : 50%
        loc="center left",
        bbox_to_anchor=(1, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = ax.figure.colorbar(sm, cax=cax,)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)
    cbar.ax.get_yaxis().labelpad = 5
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("J(θ)", rotation=0, size=20)

    return fig, ax


def setup_2d_plot(
    cache_dict,
    experiment_folder,
    exp_id,
    data_loader,
    step=1001,
    center_vector_id=0,
    basis_vector_ids=[0, 1],
    grid_bound=[-20, 25],
    num_inter_models=35,
    filter_threshold=0.05,
    precomputed_data=False,
    precomputed_timestamp="",
):
    """Set up the data for the 2d plot

    Setup for plotting a 2d plot from three GENNI models following the optimisation
    trajectory until `step`. Basis vector span a space
    (after Gram-Schmidt normalisation) with the center vector being the center of
    reference such that colour indicates the distance in function space of the
    vector at a point on the grid with this center. grid_bound and num_inter_models specify
    the grid as np.linspace(grid_bound[0], grid_bound[1], num_inter_models).

    returns two_d_vals and meta_data necessary for plotting later.

    :param cache_dict: cache dictionary with experiment information
    :param experiment_folder: path to experiment folder
    :param exp_id: experiment unique identifier of the run
    :param step: step of the optimisation path from GENNI to visualise
    :param center_vector_id: id of the parameter vector at time `step` that we use as origin
    :param basis_vector_id: id of the parameter vectors at time `step` that we use as axis 1 and 2
    :param grid_bound: upper and lower bound of the grid
    :param num_inter_models: number of points in the grid
    :param filter_threshold: threshold below which we consider things to be equal
    :param precomputed_data: if precomputed, load the data from precomputed_timestamp dir
    :param precomputed_timestamp: specifying the precomputed data directory
    """
    if precomputed_data:
        two_d_vals, cache_dict = save_load.load_cached_data(
            experiment_folder, "2d_loss", time_stamp=precomputed_timestamp
        )
    else:
        # Center model
        cache_dict["center"] = {"step": step, "idx": center_vector_id}

        # Basis Vectors
        cache_dict["basis_vectors"] = {
            "steps": [step, step],
            "model_idxs": basis_vector_ids,
        }

        cache_dict["num_inter_models"] = num_inter_models
        cache_dict["grid_bound"] = grid_bound

        two_d_vals, meta_data = griding.get_grid.main(
            experiment_folder, exp_id, cache_dict, data_loader
        )

    vals_filtered = np.array(two_d_vals)
    vals_filtered[vals_filtered > filter_threshold] = None
    grid_arr = np.linspace(grid_bound[0], grid_bound[1], num_inter_models)
    return vals_filtered, two_d_vals, grid_arr, meta_data, cache_dict


def setup_3d_plot(
    cache_dict,
    experiment_folder,
    exp_id,
    data_loader,
    step=1001,
    center_vector_id=0,
    basis_vector_ids=[0, 1, 2],
    grid_bound=[-20, 25],
    num_inter_models=35,
    filter_threshold=0.05,
):
    """Set up the data for the 3d plot

    Setup for plotting a 3d plot from three GENNI models following the optimisation
    trajectory until `step`. Basis vector span a space
    (after Gram-Schmidt normalisation) with the center vector being the center of
    reference such that colour indicates the distance in function space of the
    vector at a point on the grid with this center. grid_bound and num_inter_models specify
    the grid as np.linspace(grid_bound[0], grid_bound[1], num_inter_models).

    returns two_d_vals and meta_data necessary for plotting later.

    :param cache_dict: cache dictionary with experiment information
    :param experiment_folder: path to experiment folder
    :param exp_id: experiment unique identifier of the run
    :param step: step of the optimisation path from GENNI to visualise
    :param center_vector_id: id of the parameter vector at time `step` that we use as origin
    :param basis_vector_id: id of the parameter vectors at time `step` that we use as axis 1 and 2 and third dimension
    :param grid_bound: upper and lower bound of the grid
    :param num_inter_models: number of points in the grid
    :param filter_threshold: threshold below which we consider things to be equal
    """
    # Center model
    cache_dict["center"] = {"step": step, "idx": center_vector_id}

    # Basis Vectors
    cache_dict["basis_vectors"] = {
        "steps": [step, step, step],
        "model_idxs": basis_vector_ids,
    }

    cache_dict["num_inter_models"] = num_inter_models
    cache_dict["grid_bound"] = grid_bound

    three_d_vals, meta_data = griding.get_grid.main(
        experiment_folder, exp_id, cache_dict, data_loader
    )
    grid_filter = three_d_vals.reshape(-1) < filter_threshold
    # First dim is z axis, and last dimension x axis. This is because of how we need to loop (from last basis vector to first)
    grid_coeffs = itertools.product(
        np.linspace(grid_bound[0], grid_bound[1], num_inter_models), repeat=3
    )
    grid_coeffs = np.array(list(grid_coeffs))[grid_filter]

    return three_d_vals, grid_coeffs, grid_filter, meta_data, cache_dict


def setup_umap(
    umap_model, grid_filter, three_d_vals, cache_dict_3d, experiment_folder, exp_id
):
    # Unpack saved data from setup_plot_3d
    center_step = cache_dict_3d["center"]["step"]
    center_idx = cache_dict_3d["center"]["idx"]
    center_model = save_load.get_all_models(experiment_folder, center_step)[exp_id][
        str(center_idx)
    ]

    # Unpack the other models, get the basis vectors for these
    model_steps = [center_step, center_step, center_step]
    model_idxs = cache_dict_3d["basis_vectors"]["model_idxs"]
    basis_vectors = [
        utils.get_params_vec(
            save_load.get_all_models(experiment_folder, model_steps[i])[exp_id][
                str(model_idxs[i])
            ]
        )
        for i in range(len(model_steps))
    ]

    # Set up grid
    num_inter_models = cache_dict_3d["num_inter_models"]
    grid_bound = cache_dict_3d["grid_bound"]
    basis_orthonorm_vectors = cache_dict_3d["basis_orthonorm_vectors"]
    grid = interpolation.get_models_grid(
        center_model, basis_orthonorm_vectors, num_inter_models, grid_bound
    )

    # Filter grid
    filtered_grid = grid.reshape(np.prod(grid.shape[:-1]), -1)[grid_filter]
    center_array = utils.get_params_vec(center_model).detach().numpy()
    bases_arrays = []
    bases_arrays.extend([b.detach().numpy() for b in basis_vectors])
    bases_arrays = np.vstack(bases_arrays)
    bases_arrays = np.vstack([center_array, bases_arrays])
    filtered_grid = np.concatenate([filtered_grid, bases_arrays])

    projections = umap_model.fit_transform(filtered_grid)

    lO = cache_dict_3d["center_loss"]
    l1 = cache_dict_3d["basis_losses"][0]
    l2 = cache_dict_3d["basis_losses"][1]
    l3 = cache_dict_3d["basis_losses"][2]

    # Merge fit with labels
    labels = np.concatenate([three_d_vals.reshape(-1)[grid_filter], [lO, l1, l2, l3]])

    dataset = pd.DataFrame(
        {"x1": projections[:-4, 0], "x2": projections[:-4, 1], "label": labels[:-4]}
    )
    return dataset, labels, projections
