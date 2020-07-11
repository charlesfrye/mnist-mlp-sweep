import torch
import torch.nn as nn


class MLP(nn.Module):
    input_dim = 28 * 28
    num_classes = 10
    activations = {
        "none": nn.Identity,
        "gelu": nn.GELU,
        "lrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh
        }
    max_depth = 15
    max_size_increment = 10

    def __init__(self, config):
        super(MLP, self).__init__()

        self.activation = self.activations[config.activation]

        self.parameter_budget = config.parameter_budget
        self.shape = config.shape
        self.layer_shapes = self.shapes_from_budget(config.parameter_budget, config.shape)

        self.build()

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, inpt):
        x = inpt.view(-1, self.input_dim)

        for layer in self.layers:
            x = layer(x)

        return x

    def build(self):
        blocks = []
        for layer_shape in self.layer_shapes[:-1]:
            linear = nn.Linear(*layer_shape)
            batchnorm = nn.BatchNorm1d(layer_shape[1])
            block = [linear, self.activation(), batchnorm]
            blocks.append(block)

        # handle logit block separately
        blocks.append([nn.Linear(*self.layer_shapes[-1])])

        self.layers = nn.ModuleList(
            sum(blocks, []))

    def shapes_from_budget(self, budget, shape):
        if shape == "wide":
            layer_shapes = self.shapes_from_budget_wide(budget)
        elif shape == "deep":
            layer_shapes = self.shapes_from_budget_deep(budget)
        elif shape == "pyramidal":
            layer_shapes = self.shapes_from_budget_pyramidal(budget)
        else:
            raise ValueError
        return layer_shapes

    def shapes_from_budget_wide(self, budget):
        size = 1
        layer_shapes = self.shapes_from_size(size)

        assert self.count_parameters(layer_shapes) <= budget

        while self.count_parameters(layer_shapes) <= budget:
            size += 1
            layer_shapes = self.shapes_from_size(size)

        size -= 1
        layer_shapes = self.shapes_from_size(size)

        return layer_shapes

    def shapes_from_budget_deep(self, budget):
        size = self.num_classes
        depth = 2

        layer_shapes = self.shapes_from_size_depth(size, depth)
        assert self.count_parameters(layer_shapes) <= budget

        depth_increment = 1

        size_increment = 1
        last_size_increment = 1

        while self.count_parameters(layer_shapes) <= budget:
            if depth < self.max_depth:
                depth += depth_increment
            else:
                depth_increment = 0

            size += size_increment
            last_size_increment = size_increment
            size_increment = min(size_increment * 2, self.max_size_increment)

            layer_shapes = self.shapes_from_size_depth(size, depth)

        size -= last_size_increment
        depth -= depth_increment
        layer_shapes = self.shapes_from_size_depth(size, depth)

        return layer_shapes

    def shapes_from_budget_pyramidal(self, budget):
        base = 16
        sizes = [base // (2 ** k) for k in range(10)
                 if base // (2 ** k) >= 16]

        layer_shapes = self.shapes_from_sizes(sizes)
        assert self.count_parameters(layer_shapes) <= budget

        while self.count_parameters(layer_shapes) <= budget:
            base *= 2
            sizes = [base // (2 ** k) for k in range(10)
                     if base // (2 ** k) >= 16]
            layer_shapes = self.shapes_from_sizes(sizes)

        base //= 2
        sizes = [base // (2 ** k) for k in range(10)
                 if base // (2 ** k) >= 16]
        layer_shapes = self.shapes_from_sizes(sizes)

        return layer_shapes

    def shapes_from_size(self, size):
        layer_shapes = [[self.input_dim, size], [size, self.num_classes]]
        return layer_shapes

    def shapes_from_size_depth(self, size, depth):
        layer_shapes = [[self.input_dim, size]]

        for _ in range(depth - 1):
            layer_shapes.append([size, size])

        layer_shapes.append([size, self.num_classes])

        return layer_shapes

    def shapes_from_sizes(self, sizes):
        layer_shapes = [[self.input_dim, sizes[0]]]

        for in_shp, out_shp in zip(sizes[:-1], sizes[1:]):
            layer_shapes.append([in_shp, out_shp])

        layer_shapes.append([sizes[-1], self.num_classes])

        return layer_shapes

    @staticmethod
    def count_parameters(layer_shapes):
        number_weights = sum([in_dim * out_dim for in_dim, out_dim in layer_shapes])
        number_biases = sum([out_dim for _, out_dim in layer_shapes])
        return number_weights + number_biases

    def count_trainable_parameters(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def set_activation(self, activation_str):
        self.activation = self.activations[activation_str]

    @staticmethod
    def accuracy(output, target):
        _, predicted = torch.max(output, 1)
        accuracy = torch.eq(predicted, target).float().mean()

        return accuracy
