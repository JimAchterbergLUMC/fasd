import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOutputActivations(nn.Module):
    def __init__(
        self,
        output_sizes,
        activations,
    ):
        super().__init__()
        assert len(output_sizes) == len(
            activations
        ), "Each output feature must have an activation."
        self.output_sizes = output_sizes
        self.activations = activations

    def forward(self, x):
        split_x = torch.split(x, self.output_sizes, dim=1)
        activated = []
        for xi, act in zip(split_x, self.activations):
            if act == "tanh":
                activated.append(torch.tanh(xi))
            elif act == "sigmoid":
                activated.append(torch.sigmoid(xi))
            elif act == "relu":
                activated.append(F.relu(xi))
            elif act == "leaky_relu":
                activated.append(F.leaky_relu(xi))
            elif act == "softmax":
                activated.append(F.softmax(xi, dim=1))
            elif act == "gumbel_softmax":
                activated.append(F.gumbel_softmax(xi, tau=0.2, dim=1))
            elif act == "none" or act is None:
                activated.append(xi)
            else:
                raise ValueError(f"Unsupported activation: {act}")
        return torch.cat(activated, dim=1)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_config: list[tuple],  # list of (output_dim_i, activation_i)
        hidden_dims: list = [128, 64],
        nonlinearity: str = "relu",
        dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.residual = residual
        act_layer = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "none": nn.Identity,
        }[nonlinearity.lower()]

        layers = nn.ModuleList()
        prev_dim = input_dim

        for dim in hidden_dims:
            layer_input_dim = prev_dim + input_dim if residual else prev_dim
            layer = [nn.Linear(layer_input_dim, dim)]
            if batch_norm:
                layer.append(nn.BatchNorm1d(dim))
            layer.append(act_layer())
            if dropout > 0.0:
                layer.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*layer))
            prev_dim = dim

        self.body = layers

        output_sizes = [cfg[0] for cfg in output_config]
        self.output_layer = nn.Linear(prev_dim, sum(output_sizes))

        activations = [cfg[1] for cfg in output_config]
        self.mixed_activations = MixedOutputActivations(
            output_sizes,
            activations,
        )

    def forward(self, x):
        out = x
        for layer in self.body:
            if self.residual:
                out = layer(torch.cat([out, x], dim=1))
            else:
                out = layer(out)
        out = self.output_layer(out)
        out = self.mixed_activations(out)
        return out
