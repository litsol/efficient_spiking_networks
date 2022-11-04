# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""  module docstring """

import math

# import numpy as np
import torch

# from torch import nn
from torch.nn import functional as F

SURROGRATE_TYPE = "MG"
GAMMA = 0.5
LENS = 0.5
R_M = 1
BETA_VALUE = 0.184
B_J0_VALUE = 1.6


def gaussian(
    x: torch.Tensor,  # pylint: disable=C0103
    mu: float = 0.0,  # pylint: disable=C0103
    sigma: float = 0.5,
) -> torch.Tensor:
    """Gussian"""
    return (
        torch.exp(-((x - mu) ** 2) / (2 * sigma**2))
        / torch.sqrt(2 * torch.tensor(math.pi))
        / sigma
    )


class ActFunAdp(torch.autograd.Function):
    """class docstring"""

    @staticmethod
    def forward(ctx, inp):
        """function docstring
        inp = membrane potential- threshold"""
        ctx.save_for_backward(inp)
        return inp.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):
        """approximate the gradients"""
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(inp) < lens
        scale = 6.0
        hight = 0.15
        if SURROGRATE_TYPE == "G":
            temp = (
                torch.exp(-(inp**2) / (2 * LENS**2))
                / torch.sqrt(2 * torch.tensor(math.pi))
                / LENS
            )
        elif SURROGRATE_TYPE == "MG":
            temp = (
                gaussian(inp, mu=0.0, sigma=LENS) * (1.0 + hight)
                - gaussian(inp, mu=LENS, sigma=scale * LENS) * hight
                - gaussian(inp, mu=-LENS, sigma=scale * LENS) * hight
            )
        elif SURROGRATE_TYPE == "linear":
            temp = F.relu(1 - inp.abs())
        elif SURROGRATE_TYPE == "slayer":
            temp = torch.exp(-5 * inp.abs())
        return grad_input * temp.float() * GAMMA


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes spike_neuron.py; pylint-3 -d E0401 -f parseable spike_neuron.py" # NOQA, pylint: disable=C0301
# End:
