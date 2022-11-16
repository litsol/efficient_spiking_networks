# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""  module docstring """

__all__ = ["SpikeCov1D", "SpikeCov2D"]

import numpy as np
import torch
from torch import nn

from . import spike_neuron as sn

B_J0 = 1.6


class SpikeCov1D(nn.Module):  # pylint: disable=R0902
    """Spike_Cov1D class docstring"""

    def __init__(  # pylint: disable=R0913,R0914
        self,
        input_size,
        output_dim,
        kernel_size=5,
        strides=1,
        pooling_type=None,
        pool_size=2,
        pool_strides=2,
        dilation=1,
        tau_m=20,
        tau_adp_inital=100,
        tau_initializer="normal",
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
        # input_size = [c,h]
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.dilation = dilation
        self.device = device

        if pooling_type is not None:
            if pooling_type == "max":
                self.pooling = nn.MaxPool1d(
                    kernel_size=pool_size, stride=pool_strides, padding=1
                )
            elif pooling_type == "avg":
                self.pooling = nn.AvgPool1d(
                    kernel_size=pool_size, stride=pool_strides, padding=1
                )
        else:
            self.pooling = None

        self.conv = nn.Conv1d(
            self.input_dim,
            self.output_dim,
            kernel_size=kernel_size,
            stride=strides,
            padding=(
                np.ceil(((kernel_size - 1) * self.dilation) / 2).astype(int),
            ),
            dilation=(self.dilation,),
        )

        self.output_size = self.compute_output_size()

        self.tau_m = nn.Parameter(torch.Tensor(self.output_size))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_size))

        if tau_initializer == "normal":
            nn.init.normal_(self.tau_m, tau_m, tau_m_inital_std)
            nn.init.normal_(self.tau_adp, tau_adp_inital, tau_adp_inital_std)

    def parameters(self):
        """parameters member function docstring"""
        return [self.dense.weight, self.dense.bias, self.tau_m, self.tau_adp]

    def set_neuron_state(self, batch_size):
        """se_neuron_state member function docstring"""
        self.mem = (
            torch.zeros(batch_size, self.output_size[0], self.output_size[1])
            * B_J0
        ).to(self.device)
        self.spike = torch.zeros(
            batch_size, self.output_size[0], self.output_size[1]
        ).to(self.device)
        self.b = (
            torch.ones(batch_size, self.output_size[0], self.output_size[1])
            * B_J0
        ).to(self.device)

    def forward(self, input_spike):
        """forward member function docstring"""
        d_input = self.conv(input_spike.float())
        if self.pooling is not None:
            d_input = self.pooling(d_input)
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

    def compute_output_size(self):
        """compute_output member function docstring"""
        x_emp = torch.randn([1, self.input_size[0], self.input_size[1]])
        out = self.conv(x_emp)
        if self.pooling is not None:
            out = self.pooling(out)
        # print(self.name+'\'s size: ', out.shape[1:])
        return out.shape[1:]


class SpikeCov2D(nn.Module):  # pylint: disable=R0902
    """Spike_Cov2D docstring"""

    def __init__(  # pylint: disable=R0913
        self,
        input_size,
        output_dim,
        kernel_size=5,
        strides=1,
        pooling_type=None,
        pool_size=2,
        pool_strides=2,
        tau_m=20,
        tau_adp_inital=100,
        tau_initializer="normal",
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

        # input_size = [c,w,h]
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device

        if pooling_type is not None:
            if pooling_type == "max":
                self.pooling = nn.MaxPool2d(
                    kernel_size=pool_size, stride=pool_strides, padding=1
                )
            elif pooling_type == "avg":
                self.pooling = nn.AvgPool2d(
                    kernel_size=pool_size, stride=pool_strides, padding=1
                )
        else:
            self.pooling = None

        self.conv = nn.Conv2d(  # Look at the original!!!!
            self.input_dim, self.output_dim, kernel_size, strides
        )

        self.output_size = self.compute_output_size()

        self.tau_m = nn.Parameter(torch.Tensor(self.output_size))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_size))

        if tau_initializer == "normal":
            nn.init.normal_(self.tau_m, tau_m, tau_m_inital_std)
            nn.init.normal_(self.tau_adp, tau_adp_inital, tau_adp_inital_std)

    def parameters(self):
        """parameters member function docstring"""
        return [self.dense.weight, self.dense.bias, self.tau_m, self.tau_adp]

    def set_neuron_state(self, batch_size):
        """set_neuron_state member function docstring"""
        self.mem = torch.rand(batch_size, self.output_size).to(self.device)
        self.spike = torch.zeros(batch_size, self.output_size).to(self.device)
        self.b = (torch.ones(batch_size, self.output_size) * B_J0).to(
            self.device
        )

    def forward(self, input_spike):
        """forward member function docstring"""
        d_input = self.conv(input_spike.float())
        if self.pooling is not None:
            d_input = self.pool(d_input)
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

    def compute_output_size(self):
        """compute_output_size member function docstring"""
        x_emp = torch.randn(
            [1, self.input_size[0], self.input_size[1], self.input_size[2]]
        )
        out = self.conv(x_emp)
        if self.pooling is not None:
            out = self.pooling(out)
        # print(self.name+'\'s size: ', out.shape[1:])
        return out.shape[1:]


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes spike_cnn.py; pylint-3 -d E0401 -f parseable spike_cnn.py" # NOQA, pylint: disable=C0301
# End:
