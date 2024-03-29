# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
This module contains one class and three functions that together
aree used to calculate the membrane potential of the various spiking
neurons defined in this package. In particular, the functions
mem_update_adp and output_Neuron are called in the forward member
function of the SpikeDENSE, SpikeBIDENSE, SpikeRNN, SpikeCov1D and
SpikeCov2D layer classes and the readout_integration classes
respectively.
"""

import math

# import numpy as np
import torch
from loguru import logger

# from torch import nn
from torch.nn import functional as F

# all = ["output_Neuron, mem_update_adp"]

SURROGRATE_TYPE: str = "MG"
GAMMA: float = 0.5
LENS: float = 0.5
R_M: float = 1
BETA_VALUE: float = 0.184
B_J0_VALUE: float = 1.6
SCALE: float = 6.0
HIGHT: float = 0.15

# act_fun_adp = ActFunADP.apply


class NoSurrogateTypeException(Exception):
    pass


def gaussian(
    x: torch.Tensor,  # pylint: disable=C0103
    mu: float = 0.0,  # pylint: disable=C0103
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Gussian

    Used in the backward method of a custom autograd function class
    ActFunADP to approximate the gradiant in a surrogate function
    for back propogation.
    """

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
    """
    This function updates the membrane potential and adaptation
    variable of a spiking neural network.

    Inputs:
    inputs: the input spikes to the neuron
    mem: the current membrane potential of the neuron
    spike: the current adaptation variable of the neuron
    tau_adp: the time constant for the adaptation variable
    b: a value used in the adaptation variable update equation
    tau_m: the time constant for the membrane potential
    dt: the time step used in the simulation
    isAdapt: a boolean variable indicating whether or not to use the
    adaptation variable
    device: a variable indicating which device (e.g. CPU or GPU) to
    use for the computation

    Outputs:
    mem: the updated membrane potential
    spike: the updated adaptation variable
    B: a value used in the adaptation variable update equation
    b: the updated value of the adaptation variable

    The function first computes the exponential decay factors alpha
    and ro using the time constants tau_m and tau_adp, respectively.
    It then checks whether the isAdapt variable is True or False to
    determine the value of beta.  The adaptation variable b is then
    updated using the exponential decay rule, and B is computed using
    the value of beta and the initial value b_j0_value.  The function
    then updates the membrane potential mem using the input spikes, B,
    and the decay factor alpha, and computes the inputs_ variable as
    the difference between mem and B.  Finally, the adaptation
    variable spike is updated using the activation function defined in
    the act_fun_adp() function, and the updated values of mem, spike,
    B, and b are returned.
    """

    alpha = torch.exp(-1.0 * dt / tau_m).to(device)
    ro = torch.exp(-1.0 * dt / tau_adp).to(device)  # pylint: disable=C0103

    beta = BETA_VALUE if isAdapt else 0.0
    if isAdapt:
        beta = BETA_VALUE
    else:
        beta = 0.0

    b = ro * b + (1 - ro) * spike  # Hard reset equation 1.8 page 12.
    B = B_J0_VALUE + beta * b  # pylint: disable=C0103

    mem = mem * alpha + (1 - alpha) * R_M * inputs - B * spike * dt
    inputs_ = mem - B

    # Non spiking output
    spike = F.relu(inputs_)

    # For details about calling the 'apply' member function,
    # See: https://pytorch.org/docs/stable/autograd.html#function
    # Spiking output
    spike = ActFunADP.apply(inputs_)

    return mem, spike, B, b


def output_Neuron(
    inputs, mem, tau_m, dt=1, device=None
):  # pylint: disable=C0103
    """
    Output the membrane potential of a LIF neuron without spike

    The only appears of this function is in the forward member
    function of the ReadoutIntegrator layer class.
    """

    alpha = torch.exp(-1.0 * dt / tau_m).to(device)
    mem = mem * alpha + (1 - alpha) * inputs
    return mem


class ActFunADP(torch.autograd.Function):
    """
    ActFunADP

    Custom autograd function redefining how forward and backward
    passes are performed. This class is 'applied' in the
    mem_update_adp function to calculate the new spike value.

    For details about calling the 'apply' member function, See:
    https://pytorch.org/docs/stable/autograd.html#function
    """

    @staticmethod
    def forward(ctx, i):  # ? What is the type and dimension of i?
        """
        Redefine the default autograd forward pass function.
        inp = membrane potential- threshold

        Returns a tensor whose values are either 0 or 1 dependent
        upon their value in the input tensor i.
        """

        ctx.save_for_backward(i)
        return i.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):
        """
        Defines a formula for differentiating during back propogation.

        Since the spike function is nondifferentiable, we
        approximate the back propogation gradients with one of
        several surrogate functions.
        """

        (result,) = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # temp = abs(result) < lens
        if SURROGRATE_TYPE == "G":
            # temp = gaussian(result, mu=0.0, sigma=LENS)
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
        else:
            logger.critical(
                "No Surrogate type chosen, so temp tensor is undefined."
            )
            raise NoSurrogateTypeException("No Surrogate type chosen.")
        return grad_output * temp.float() * GAMMA


# finis

# Local Variables:
# compile-command: "pyflakes spike_neuron.py; pylint-3 -d E0401 -f parseable spike_neuron.py" # NOQA, pylint: disable=C0301
# End:
