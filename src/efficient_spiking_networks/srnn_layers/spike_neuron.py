# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""  module docstring """

__all__ = ["ActFunADP"]

import math

# import numpy as np
import torch

# from torch import nn
from torch.nn import functional as F

SURROGRATE_TYPE: str = "MG"
GAMMA: float = 0.5
LENS: float = 0.5
R_M: float = 1
BETA_VALUE: float = 0.184
B_J0_VALUE: float = 1.6
SCALE: float = 6.0
HIGHT: float = 0.15

# act_fun_adp = ActFunADP.apply


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


def mem_update_adp(  # pylint: disable=R0913
    inputs,
    mem,
    spike,
    tau_adp,
    b,  # pylint: disable=C0103
    tau_m,
    dt=1,  # pylint: disable=C0103
    isAdapt=1,  # pylint: disable=C0103
    device=None,
):  # pylint: disable=C0103

    """Function Docstring"""

    alpha = torch.exp(-1.0 * dt / tau_m).to(device)
    ro = torch.exp(-1.0 * dt / tau_adp).to(device)  # pylint: disable=C0103
    if isAdapt:
        beta = BETA_VALUE
    else:
        beta = 0.0

    b = ro * b + (1 - ro) * spike
    B = B_J0_VALUE + beta * b  # pylint: disable=C0103

    mem = mem * alpha + (1 - alpha) * R_M * inputs - B * spike * dt
    inputs_ = mem - B
    spike = F.relu(inputs_)

    # For details about calling the 'apply' member function,
    # See: https://pytorch.org/docs/stable/autograd.html#function
    spike = ActFunADP.apply(inputs_)
    return mem, spike, B, b


def output_Neuron(
    inputs, mem, tau_m, dt=1, device=None
):  # pylint: disable=C0103
    """
    The read out neuron is leaky integrator without spike
    """
    alpha = torch.exp(-1.0 * dt / tau_m).to(device)
    mem = mem * alpha + (1 - alpha) * inputs
    return mem


class ActFunADP(torch.autograd.Function):
    """ActFunADP class docstring"""

    @staticmethod
    def forward(ctx, i):
        """forward member function docstring
        inp = membrane potential- threshold"""
        ctx.save_for_backward(i)
        return i.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):
        """approximate the gradients"""
        (result,) = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # temp = abs(result) < lens
        if SURROGRATE_TYPE == "G":
            temp = (
                torch.exp(-(result**2) / (2 * LENS**2))
                / torch.sqrt(2 * torch.tensor(math.pi))
                / LENS
            )
        elif SURROGRATE_TYPE == "MG":
            temp = (
                gaussian(result, mu=0.0, sigma=LENS) * (1.0 + HIGHT)
                - gaussian(result, mu=LENS, sigma=SCALE * LENS) * HIGHT
                - gaussian(result, mu=-LENS, sigma=SCALE * LENS) * HIGHT
            )
        elif SURROGRATE_TYPE == "linear":
            temp = F.relu(1 - result.abs())
        elif SURROGRATE_TYPE == "slayer":
            temp = torch.exp(-5 * result.abs())
        return grad_output * temp.float() * GAMMA


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes spike_neuron.py; pylint-3 -d E0401 -f parseable spike_neuron.py" # NOQA, pylint: disable=C0301
# End:
