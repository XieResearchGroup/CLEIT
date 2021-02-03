from torch import nn
from types_ import *
from typing import List
import torch
from itertools import chain

def zero_grad(self, grad_input, grad_output):
    return grad_input * self.mask

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_of_modules):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features * num_of_modules, out_features * num_of_modules)
        self.mask = torch.block_diag(*chain([torch.ones(out_features, in_features)] * num_of_modules))
        self.linear.weight.data *= self.mask  # to zero it out first
        self.handle = self.register_backward_hook(zero_grad)

    def forward(self, input):
        return self.linear(input)

class MaskMLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List = None, dop: float = 0.1, **kwargs) -> None:
        super(MaskMLP, self).__init__()
        self.output_dim = output_dim
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]*output_dim, bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    MaskedLinear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1],num_of_modules=output_dim),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )

        self.module = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            MaskedLinear(hidden_dims[-1], 1, num_of_modules=output_dim),
            nn.Sigmoid()
        )

    def forward(self, input: Tensor) -> Tensor:
        embed = self.module(input)
        output = self.output_layer(embed)

        return output
