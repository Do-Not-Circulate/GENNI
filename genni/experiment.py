from collections import OrderedDict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from torch.utils.data import DataLoader, Dataset

from . import data_getters, save_load


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
    cfgs = save_load.load_configs(experiment_folder)
    cfg = cfgs.loc[exp_id]
    cfg["data_meta"]["N"] = N
    cache_dict["N"] = cfg["data_meta"]["N"]

    data_set = genni.data_getters.get_data(cfg, device=device)
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
                name="J(" + "\u03B8" + ")",  # cmax=0.001, cmin=0,
                colorbar=dict(
                    thickness=130,
                    title="J(" + "\u03B8" + ")",
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
                    title="J(" + "\u03B8" + ")",
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
def plot_contour(vals_filtered, grid_arr, meta_data, update_layout=False):
    fig = go.Figure(
        data=[
            go.Contour(
                z=np.log(vals_filtered),
                x=grid_arr,
                y=grid_arr,
                colorscale="viridis",
                colorbar=dict(title="log({})".format("J(" + "\u03B8" + ")")),
            ),
            go.Scatter(
                x=np.array(0),
                y=np.array(0),
                marker=dict(color=COLOR_CENTER, size=12),
                showlegend=False,
            ),
            go.Scatter(
                x=np.array([meta_data["basis_coordinates"][0][0]]),
                y=np.array([meta_data["basis_coordinates"][0][1]]),
                marker=dict(color=COLOR1, size=12),
                showlegend=False,
            ),
            go.Scatter(
                x=np.array([meta_data["basis_coordinates"][1][0]]),
                y=np.array([meta_data["basis_coordinates"][1][1]]),
                marker=dict(color=COLOR2, size=12),
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


def set_up_plotting_arrays(two_d_vals, filter_threshold, cache_dict):
    vals_filtered = np.array(two_d_vals)
    vals_filtered[vals_filtered > filter_threshold] = None
    grid_arr = np.linspace(
        cache_dict["grid_bound"][0],
        cache_dict["grid_bound"][1],
        cache_dict["num_inter_models"],
    )
    return vals_filtered, grid_arr
