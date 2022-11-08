# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""  module docstring """

__all__ = ["Spike_Rnn"]

import torch
from torch import nn
from torch.autograd import Variable

from . import spike_dense as sd
from . import spike_neuron as sn

B_J0 = sn.B_J0_VALUE


class Spike_Rnn(nn.Module):  # pylint: disable=C0103,R0902
    """Spike_Rnn class docstring"""

    def __init__(  # pylint: disable=R0913,W0231
        self,
        input_dim,
        output_dim,
        tauM=20,
        tauAdp_inital=100,
        tau_initializer="normal",
        tauM_inital_std=5,
        tauAdp_inital_std=5,
        is_adaptive=1,
        device="cpu",
        bias=True,
    ):
        """Class constructor member function"""
        super()
        self.mem = None
        self.spike = None
        self.b = None  # pylint: disable=C0103
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device

        self.b_j0 = B_J0
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.recurrent = nn.Linear(output_dim, output_dim, bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_initializer == "normal":
            nn.init.normal_(self.tau_m, tauM, tauM_inital_std)
            nn.init.normal_(self.tau_adp, tauAdp_inital, tauAdp_inital_std)
        elif tau_initializer == "multi_normal":
            self.tau_m = sd.multi_normal_initilization(
                self.tau_m, tauM, tauM_inital_std
            )
            self.tau_adp = sd.multi_normal_initilization(
                self.tau_adp, tauAdp_inital, tauAdp_inital_std
            )

    def parameters(self):
        """parameters member function docstring"""
        return [
            self.dense.weight,
            self.dense.bias,
            self.recurrent.weight,
            self.recurrent.bias,
            self.tau_m,
            self.tau_adp,
        ]

    def set_neuron_state(self, batch_size):
        """set_neuron_state member function docstring"""

        self.mem = Variable(
            torch.zeros(batch_size, self.output_dim) * self.b_j0
        ).to(self.device)
        self.spike = Variable(torch.zeros(batch_size, self.output_dim)).to(
            self.device
        )
        self.b = Variable(
            torch.ones(batch_size, self.output_dim) * self.b_j0
        ).to(self.device)

    def forward(self, input_spike):
        """forward member function docstring"""
        d_input = self.dense(input_spike.float()) + self.recurrent(self.spike)
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


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes spike_rnn.py; pylint-3 -d E0401 -f parseable spike_rnn.py" # NOQA, pylint: disable=C0301
# End:
