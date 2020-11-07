import time

import torch
from torch import optim
from torch.utils.data import DataLoader

from .data_getters import get_data
from .save_load import init_summary_writer, save_config, save_models
from .training_utils import get_nets, get_stopping_criterion
from .utils import get_grad_params_vec, get_params_vec, set_seed


def train(config, folder_path):
    # init torch
    if config["device"] == "gpu":
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")

    # set seed for the rest of SGD
    set_seed(config["seed"])

    # get dataloaders
    # --- get data ---
    data_set = get_data(config, device)
    data_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

    # Init neural nets attempts
    num_nets = config["num_nets"]
    nets = get_nets(config["net_name"], config["net_params"], num_nets, device=device)

    #  Define a Loss function and optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizers = [
        optim.SGD(
            nets[i].parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
        )
        for i in range(num_nets)
    ]

    # define stopping criterion
    stopping_criterion = get_stopping_criterion(
        config["num_steps"], config["mean_loss_threshold"]
    )

    # init saving
    process_id = str(time.time())
    writer = init_summary_writer(folder_path, process_id)
    save_config(folder_path, process_id, config)
    save_models(
        nets, config["net_name"], config["net_params"], folder_path, process_id, step=0
    )

    # init number of steps
    curr_step = 1
    mean_loss = float("inf")

    # train
    while not stopping_criterion(mean_loss, curr_step):

        # get train loaders for each net
        net_data_loaders = [iter(data_loader) for _ in range(num_nets)]

        is_training_curr = True
        while is_training_curr and (not stopping_criterion(mean_loss, curr_step)):

            # do update step for each net
            nets, took_step, mean_loss_after_step = nets_training_step(
                nets,
                optimizers,
                net_data_loaders,
                criterion,
                writer=writer,
                curr_step=curr_step,
                device=device,
            )

            if not took_step:
                break
            else:
                mean_loss = mean_loss_after_step

            # save nets
            if (curr_step % config["save_model_freq"]) == 1:
                save_models(
                    nets,
                    config["net_name"],
                    config["net_params"],
                    folder_path,
                    process_id,
                    step=curr_step,
                )

            if (curr_step % config["print_stat_freq"]) == 1:
                print("Step: {}".format(curr_step))
                print("Mean Loss: {}".format(mean_loss))

            # update curr_step
            curr_step += 1

    # save final nets
    save_models(
        nets,
        config["net_name"],
        config["net_params"],
        folder_path,
        process_id,
        step=curr_step,
    )

    return nets


def nets_training_step(
    nets,
    net_optimizers,
    net_data_loaders,
    criterion,
    writer=None,
    curr_step=-1,
    device=None,
):
    """Does update step on all networks and computes the weights."""
    took_step = True

    mean_loss = 0

    for idx_net in range(len(nets)):

        # get net and optimizer
        net = nets[idx_net]
        optimizer = net_optimizers[idx_net]
        iter_data_loader = net_data_loaders[idx_net]

        net, curr_net_taking_step, loss = training_step(
            net,
            iter_data_loader,
            optimizer,
            criterion,
            writer=writer,
            curr_step=curr_step,
            idx_net=idx_net,
            device=device,
        )

        if not curr_net_taking_step:
            took_step = False
            break

        mean_loss += float(loss)

    assert took_step or (
        idx_net == 0
    )  # if we can't take a step for one network then we shouldn't be able to take a step for any network. Checking the first net should hence suffice.

    return nets, took_step, mean_loss / len(nets)


def training_step(
    net,
    iter_data_loader,
    optimizer,
    criterion,
    writer=None,
    curr_step=-1,
    idx_net=-1,
    device=None,
):
    """Does update step on all networks and computes the weights."""

    took_step = True

    # get the inputs; data is a list of [inputs, labels]
    try:
        data = next(iter_data_loader)
    except:
        took_step = False
        return net, took_step, None

    inputs, labels = data

    if device is not None:
        inputs, labels = (
            inputs.to(device).type(torch.cuda.FloatTensor),
            labels.to(device).type(torch.cuda.LongTensor),
        )

    # Compute gradients for input.
    inputs.requires_grad = True

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward(retain_graph=True)
    optimizer.step()

    # get gradient
    param_grads = get_grad_params_vec(net)
    curr_grad = torch.norm(param_grads)

    # store metrics for each net
    if writer is not None:
        writer.add_scalar("Loss/train/net_{}".format(idx_net), loss, curr_step)
        writer.add_scalar("Gradient/train/net_{}".format(idx_net), curr_grad, curr_step)
        writer.add_scalar(
            "Norm/net_{}".format(idx_net), torch.norm(get_params_vec(net)), curr_step
        )

    assert took_step or (idx_net == 0)

    return net, took_step, float(loss)
