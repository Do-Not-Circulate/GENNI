import torch
from torch.utils.data import DataLoader, Dataset

from .save_load import get_nets
from .utils import set_seed

"""The vector for which we get equivalences is for now chosen by the model_seed given in the config file. """


def get_data(config, device=None):
    if True:
        # init eq_model
        set_seed(config["model_seed"])
        eq_net = get_nets(config["net_name"], config["net_params"], 1, device=device)[0]
        meta = config["data_meta"]
        N, inp_dim, out_dim = meta["N"], meta["inp_dim"], meta["out_dim"]
        # Create random input and output data
        if config["net_name"] in ["LeNet", "KeskarC3"]:  # give tag CNN or something
            x = torch.randn(N, 1, inp_dim, inp_dim, device=device)
        else:
            x = torch.randn(N, inp_dim, device=device)
        y = eq_net(x)
    else:
        raise NotImplementedError("{} is not implemented.".format("Not Yet"))

    return DataWrapper([x, y])


def get_random_data_subset(data, num_datapoints=1, seed=0):
    set_seed(seed)
    data_loader = DataLoader(data, batch_size=num_datapoints, shuffle=True)
    return DataWrapper(next(iter(data_loader)))


class DataWrapper(Dataset):
    """Wrapper for data of the form:
    data[0]: Torch Input data, data[1]: Torch Output Data"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])
