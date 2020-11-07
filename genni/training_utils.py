import torch
import torch.optim as optim

from .nets.Nets import BatchNormSimpleNet, KeskarC3, LeNet, LinearNet, SimpleNet


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


def get_optimizers(config, nets):
    num_nets = config["num_nets"]
    if config["optimizer"] == "SGD":
        optimizers = [
            optim.SGD(
                nets[i].parameters(),
                lr=config["learning_rate"],
                momentum=config["momentum"],
            )
            for i in range(num_nets)
        ]
        if (
            ("lr_decay" in config)
            and (config["lr_decay"] is not None)
            and (config["lr_decay"] != 0)
        ):
            optimizers = [
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=o, gamma=config["lr_decay"]
                )
                for o in optimizers
            ]
    elif config["optimizer"] == "Adam":
        optimizers = [
            optim.Adam(nets[i].parameters(), lr=config["learning_rate"])
            for i in range(num_nets)
        ]
    else:
        raise NotImplementedError("{} is not implemented.".format(config["optimizer"]))
    return optimizers


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
