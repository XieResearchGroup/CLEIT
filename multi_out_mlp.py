from torch import nn
from types_ import *
from typing import List
import copy
import torch
from gradient_reversal import RevGrad



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MoMLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_shared_layers: int = 1, hidden_dims: List = None,
                 dop: float = 0.1, act_fn=nn.SELU, out_fn=None, gr_flag=False, **kwargs) -> None:
        super(MoMLP, self).__init__()
        self.output_dim = output_dim
        self.dop = dop
        assert num_shared_layers <= len(hidden_dims)
        self.num_shared_layers = num_shared_layers


        shared_modules = []
        ind_modules = []

        if num_shared_layers > 0:
            if gr_flag:
                shared_modules.append(RevGrad())
            shared_modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[0], bias=True),
                    # nn.BatchNorm1d(hidden_dims[0]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )
            for i in range(num_shared_layers - 1):
                shared_modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                        # nn.BatchNorm1d(hidden_dims[i + 1]),
                        act_fn(),
                        nn.Dropout(self.dop)
                    )
                )
            self.shared_module = nn.Sequential(*shared_modules)

        else:
            if gr_flag:
                ind_modules.append(RevGrad())
            ind_modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[0], bias=True),
                    # nn.BatchNorm1d(hidden_dims[0]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )

        for i in range(num_shared_layers-1, len(hidden_dims) - 1):
            ind_modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )

        if out_fn is None:
            output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], 1,bias=True)
            )
        else:
            output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], 1, bias=True),
                out_fn()
            )
        ind_modules.append(output_layer)
        ind_module = nn.Sequential(*ind_modules)
        self.output_modules = clones(ind_module, output_dim)

    def forward(self, input: Tensor) -> Tensor:
        if self.num_shared_layers > 0:
            input = self.shared_module(input)
        output = None
        for out_module in self.output_modules:
            o = out_module(input)
            output = o if output is None else torch.cat([output, o], dim=1)
        return output
