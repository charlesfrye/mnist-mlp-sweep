import torch
import torch.nn as nn

INPUT_DIMENSION = 28 * 28


class MLP(nn.Module):
    activations = nn.ModuleDict({
        "none": nn.Identity(),
        "gelu": nn.GELU(),
        "lrelu": nn.LeakyReLU(),
        "prelu": nn.PReLU(),
        "relu": nn.ReLU(),
        "selu": nn.SELU(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "tanh": nn.Tanh()
        })

    def __init__(self, config):
        super(MLP, self).__init__()

        self.feature_layer = nn.Linear(INPUT_DIMENSION, config["num_hidden"])
        self.set_activation(config["activation"])
        self.logit_layer = nn.Linear(config["num_hidden"], 10)

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, inpt):
        x = inpt.view(-1, INPUT_DIMENSION)
        x = self.feature_layer(x)
        x = self.activation(x)
        x = self.logit_layer(x)

        return x

    def set_activation(self, activation_str):
        self.activation = self.activations[activation_str]

    @staticmethod
    def accuracy(output, target):
        _, predicted = torch.max(output, 1)
        accuracy = torch.eq(predicted, target).float().mean()

        return accuracy
