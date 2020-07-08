"""For documentation, run
python run.py --help
"""
import argparse
from collections import namedtuple
import math
import sys
import time

import torch
import torchvision
import wandb

from mlp import MLP

Data = namedtuple("Data", ["train", "val"])


LEARNING_RATE_SCALING = 0.05


def main(args):
    setup_wandb(args)
    data = load_data()
    model = build_model(wandb.config)

    if wandb.config["lr"] is None:
        scaled_lr = LEARNING_RATE_SCALING * math.sqrt(wandb.config["batch_size"])
        wandb.config.update({"lr": scaled_lr}, allow_val_change=True)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=wandb.config["lr"])

    train_model(data, model, optimizer)


def build_model(config):
    model = MLP(config)
    return model


def train_model(data, model, optimizer):
    model.train()
    wandb.watch(model)

    start_time = time.time()
    step = 0
    for epoch in range(wandb.config["n_epochs"]):

        val_iterator = iter(data.val)
        for inpt, target in data.train:
            optimizer.zero_grad()
            output, loss, accuracy = run_forward_metrics(
                inpt, target, model)

            if wandb.config.track_images:
                labeled_images = get_labeled_images(inpt, output, max_n=4)
                wandb.log({"train-batch/examples": labeled_images},
                          commit=False, step=step)

            loss.backward()
            optimizer.step()

            val_inpt, val_target = next_val_batch(val_iterator, data)

            _, val_loss, val_accuracy = run_forward_metrics(
                val_inpt, val_target, model)

            step += 1
            wandb.log({"train-batch/loss": loss.item(),
                       "loss/train-batch": loss.item(),
                       "train-batch/accuracy": accuracy.item(),
                       "train-batch/epoch": epoch,
                       "val-batch/loss": val_loss.item(),
                       "loss/val-batch": val_loss.item(),
                       "val-batch/accuracy": val_accuracy.item(),
                       "val-batch/epoch": epoch,
                       }, step=step)

        running_loss, running_accuracy = 0, 0
        for inpt, target in data.val:
            _, val_loss, val_accuracy = run_forward_metrics(inpt, target, model)
            running_loss += val_loss.item()
            running_accuracy += val_accuracy.item()

        N = len(data.val)
        avg_loss, avg_accuracy = running_loss / N, running_accuracy / N
        # SLIGHTLY OVER-WEIGHTS LAST BATCH IF BATCH SIZE DOESN'T DIVIDE DATASET

        wandb.log({
                   "val-full/loss": avg_loss,
                   "loss/val-full": avg_loss,
                   "val-full/accuracy": avg_accuracy,
                   "val-full/epoch": epoch
                   }, step=step)

        if time.time() - start_time > wandb.config["timeout"]:
            break


# TODO: put in Model class
def run_forward_metrics(inpt, target, model):
    output = model(inpt)

    loss = model.loss_function(output, target)
    accuracy = model.accuracy(output, target)

    return output, loss, accuracy


def next_val_batch(val_iterator, data):
    try:
        val_inpt, val_target = next(val_iterator)
    except StopIteration:
        val_iterator = iter(data.val)
        val_inpt, val_target = next(val_iterator)
    return val_inpt, val_target


def load_data():

    to_float = torchvision.transforms.ToTensor()
    full_train = torchvision.datasets.MNIST(
        root="~/data", train=True, download=True, transform=to_float)

    # TODO: put in f'n
    N_full = len(full_train)
    train_val_split = 0.8
    train_size = int(train_val_split * N_full)
    test_size = N_full - train_size

    train_data, val_data = torch.utils.data.random_split(
        full_train, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=wandb.config["batch_size"])

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=wandb.config["batch_size"])

    data = Data(train=train_loader, val=val_loader)

    return data


def setup_wandb(args):

    wandb.init(entity="charlesfrye", project=args.project,
               config=args)
    return


def get_labeled_images(inputs, outputs, max_n=4):
    images = inputs.reshape(-1, 28, 28, 1)
    _, labels = torch.max(outputs, 1)

    images = images.numpy()
    labels = labels.numpy()

    images, labels = images[::max_n], labels[::max_n]
    labeled_images = [wandb.Image(image, caption=label)
                      for image, label in zip(images, labels)]
    return labeled_images


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run an MLP experiment on MNIST")
    default_lr = None
    default_batch_size = 32
    default_num_hidden = 32
    default_activation = "none"
    default_n_epochs = 20
    default_timeout = 15 * 60
    default_project = None

    # Optimization Parameters
    parser.add_argument("--lr", type=float, default=default_lr,
                        help="Learning rate for SGD. " +
                        "Default is None, " +
                        f"and lr is equal to {LEARNING_RATE_SCALING} * sqrt(batch_size).")
    parser.add_argument("--batch_size", type=int, default=default_batch_size,
                        help="Size of batches in SGD. " +
                        f"Default is {default_batch_size}.")
    parser.add_argument("--n_epochs", type=int, default=default_n_epochs,
                        help="Maximum number of epochs to run. " +
                        f"Default is {default_n_epochs}.")
    parser.add_argument("--timeout", type=int, default=default_timeout,
                        help="Maximum time (in seconds) to run.. " +
                        f"Default is {default_timeout}.")

    # Network Parameters
    parser.add_argument("--num_hidden", type=int, default=default_num_hidden,
                        help="Number of nodes in hidden layer." +
                        f"Default is {default_num_hidden}.")
    parser.add_argument("--activation", type=str, default=default_activation,
                        choices=MLP.activations,
                        help="Non-linear activation function for hidden layer. " +
                        f"Default is {default_activation}.")

    # Tracking Parameters
    parser.add_argument("--project", type=str, default=default_project,
                        help="Project name on wandb. Default is None.")
    parser.add_argument("--track_images", action="store_true",
                        help="Flag to track images and labels during training.")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(main(args))
