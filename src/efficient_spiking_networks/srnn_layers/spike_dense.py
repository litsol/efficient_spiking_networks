# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""  module docstring """

__all__ = ["SpikeDENSE", "SpikeBIDENSE", "ReadoutIntegrator"]

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from . import spike_neuron as sn

B_J0: float = sn.B_J0_VALUE


def multi_normal_initilization(
    param, means=[10, 200], stds=[5, 20]
):  # pylint: disable=W0102
    """multi_normal_initialization function

    The tensor returned is composed of multiple, equal length
    partitions each drawn from a normal distribution described
    by a mean and std. The shape of the returned tensor is the same
    at the original input tensor."""

    shape_list = param.shape
    if len(shape_list) == 1:
        num_total = shape_list[0]
    elif len(shape_list) == 2:
        num_total = shape_list[0] * shape_list[1]

    num_per_group = int(num_total / len(means))
    # if num_total%len(means) != 0:
    num_last_group = num_total % len(means)
    a = []  # pylint: disable=C0103
    for i in range(len(means)):  # pylint: disable=C0200
        a = (  # pylint: disable=C0103
            a
            + np.random.normal(means[i], stds[i], size=num_per_group).tolist()
        )

        if i == len(means) - 1:
            a = (  # pylint: disable=C0103
                a
                + np.random.normal(
                    means[i], stds[i], size=num_per_group + num_last_group
                ).tolist()
            )
    p = np.array(a).reshape(shape_list)  # pylint: disable=C0103
    with torch.no_grad():
        param.copy_(torch.from_numpy(p).float())
    return param


class SpikeDENSE(nn.Module):
    """Spike_Dense class docstring"""

    def __init__(  # pylint: disable=R0913,W0231
        self,
        input_dim,
        output_dim,
        tau_m=20,
        tau_adp_inital=200,
        tau_initializer="normal",  # pylint: disable=W0613
        tau_m_inital_std=5,
        tau_adp_inital_std=5,
        is_adaptive=1,
        device="cpu",
        bias=True,
    ):
        """Class constructor member function docstring"""
        super().__init__()
        self.mem = None
        self.spike = None
        self.b = None  # pylint: disable=C0103
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device

        self.dense = nn.Linear(input_dim, output_dim, bias=bias)

        # Parameters are Tensor subclasses, that have a very special
        # property when used with Module s - when they’re assigned as
        # Module attributes they are automatically added to the list
        # of its parameters, and will appear e.g. in parameters() iterator.
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_initializer == "normal":
            # Initialize self.tau_m and self.tau_adp from a single
            # normal distributions.
            nn.init.normal_(self.tau_m, tau_m, tau_m_inital_std)
            nn.init.normal_(self.tau_adp, tau_adp_inital, tau_adp_inital_std)
        elif tau_initializer == "multi_normal":
            # Initialize self.tau_m and self.tau_adp from from
            # multiple normal distributions. tau_m and tar_adp_initial
            # must be lists of means and tar_m_initial_std and
            # tar_adp_initial_std must be lists of standard
            # deviations.
            self.tau_m = multi_normal_initilization(
                self.tau_m, tau_m, tau_m_inital_std
            )
            self.tau_adp = multi_normal_initilization(
                self.tau_adp, tau_adp_inital, tau_adp_inital_std
            )

    def parameters(self):
        """Return a list of parameters being trained."""
        # The latter two are module parameters; the first two aren't
        # Where is dense.weight defined or assigned?
        return [self.dense.weight, self.dense.bias, self.tau_m, self.tau_adp]

    def set_neuron_state(self, batch_size):
        """Initialize mem, spike and b tensors.

        The Variable API has been deprecated: Variables are no
        longer necessary to use autograd with tensors. Autograd
        automatically supports Tensors with requires_grad set to
        True.
        """
        # self.mem = (torch.rand(batch_size, self.output_dim) * self.b_j0).to(
        #     self.device
        # )
        self.mem = Variable(
            torch.zeros(batch_size, self.output_dim) * B_J0
        ).to(self.device)

        self.spike = Variable(torch.zeros(batch_size, self.output_dim)).to(
            self.device
        )

        self.b = Variable(torch.ones(batch_size, self.output_dim) * B_J0).to(
            self.device
        )

    def forward(self, input_spike):
        """SpikeDENSE forward pass"""

        d_input = self.dense(input_spike.float())
        (
            self.mem,
            self.spike,
            theta,  # pylint: disable=W0612
            self.b,
        ) = sn.mem_update_adp(
            d_input,
            self.mem,
            self.spike,
            self.tau_adp,
            self.b,
            self.tau_m,
            device=self.device,
            isAdapt=self.is_adaptive,
        )

        return self.mem, self.spike


class SpikeBIDENSE(nn.Module):  # pylint: disable=R0902
    """Spike_Bidense class docstring"""

    def __init__(  # pylint: disable=R0913
        self,
        input_dim1,
        input_dim2,
        output_dim,
        tau_m=20,
        tau_adp_inital=100,
        tau_initializer="normal",  # pylint: disable=W0613
        tau_m_inital_std=5,
        tau_adp_inital_std=5,
        is_adaptive=1,
        device="cpu",
    ):
        """Class constructor member function docstring"""
        super().__init__()
        self.mem = None
        self.spike = None
        self.b = None  # pylint: disable=C0103
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device

        self.dense = nn.Bilinear(input_dim1, input_dim2, output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_initializer == "normal":
            nn.init.normal_(self.tau_m, tau_m, tau_m_inital_std)
            nn.init.normal_(self.tau_adp, tau_adp_inital, tau_adp_inital_std)
        elif tau_initializer == "multi_normal":
            self.tau_m = multi_normal_initilization(
                self.tau_m, tau_m, tau_m_inital_std
            )
            self.tau_adp = multi_normal_initilization(
                self.tau_adp, tau_adp_inital, tau_adp_inital_std
            )

    def parameters(self):
        """parameter member function docstring"""
        return [self.dense.weight, self.dense.bias, self.tau_m, self.tau_adp]

    def set_neuron_state(self, batch_size):
        """set_neuron_state member function docstring"""
        self.mem = (torch.rand(batch_size, self.output_dim) * B_J0).to(
            self.device
        )
        self.spike = torch.zeros(batch_size, self.output_dim).to(self.device)
        self.b = (torch.ones(batch_size, self.output_dim) * B_J0).to(
            self.device
        )

    def forward(self, input_spike1, input_spike2):
        """forward member function docstring"""
        d_input = self.dense(input_spike1.float(), input_spike2.float())
        (
            self.mem,
            self.spike,
            theta,  # pylint: disable=W0612
            self.b,
        ) = sn.mem_update_adp(
            d_input,
            self.mem,
            self.spike,
            self.tau_adp,
            self.b,
            self.tau_m,
            device=self.device,
            isAdapt=self.is_adaptive,
        )

        return self.mem, self.spike


class ReadoutIntegrator(nn.Module):
    """Redout_Integrator class docstring"""

    def __init__(  # pylint: disable=R0913
        self,
        input_dim,
        output_dim,
        tau_m=20,
        tau_initializer="normal",  # pylint: disable=W0613
        tau_m_inital_std=5,
        device="cpu",
        bias=True,
    ):
        """Class constructor member function"""
        super().__init__()
        self.mem = None

        # UNUSED?!
        self.spike = None
        self.b = None  # pylint: disable=C0103

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        nn.init.normal_(self.tau_m, tau_m, tau_m_inital_std)

    def parameters(self):
        """parameters member function docstring"""
        return [self.dense.weight, self.dense.bias, self.tau_m]

    def set_neuron_state(self, batch_size):
        """set_neuron_state member function docstring"""
        # self.mem = torch.rand(batch_size,self.output_dim).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)

    def forward(self, input_spike):
        """forward member function docstring"""
        d_input = self.dense(input_spike.float())
        self.mem = sn.output_Neuron(
            d_input, self.mem, self.tau_m, device=self.device
        )
        return self.mem


# Local Variables:
# compile-command: "pyflakes spike_dense.py; pylint-3 -d E0401 -f parseable spike_dense.py" # NOQA, pylint: disable=C0301
# End:
