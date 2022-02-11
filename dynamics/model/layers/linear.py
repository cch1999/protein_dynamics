import torch
import torch.nn as nn


class MLP(nn.Module):
    """Builds an MLP."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
    ):
        super(MLP, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):
    """Builds an ResNet."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
    ):
        super(ResNet, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        for _ in range(num_hidden_layers):
            layers.append(ResBlock(hidden_size))

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    """Builds an ResBlock."""

    def __init__(self, input_size: int):
        super(ResBlock, self).__init__()

        self.fc = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        # self.norm = nn.InstanceNorm1d(input_size, affine=True)

    def forward(self, x):
        residual = x
        x = self.fc(x)
        # x = self.norm(x)
        return self.relu(x + residual)


# TODO made layer norm work better (with dim size)
def MLP_with_layer_norm(
    input_size: int, hidden_size: int, num_hidden_layers: int, output_size: int
):
    return nn.Sequential(
        MLP(input_size, hidden_size, num_hidden_layers, output_size),
        nn.LayerNorm(output_size),
    )
