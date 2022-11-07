# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

""" test_esn """

import torch

import efficient_spiking_networks.srnn_layers.spike_neuron as sn
from efficient_spiking_networks import __version__

TARGET = torch.Tensor([0.1080, 0.1080, 0.1080, 0.1080, 0.1080])


def test_version():
    """sanity test"""
    assert __version__ == "0.1.0"


def test_gaussian_type():
    """Test gaussian type"""
    assert isinstance(TARGET, torch.Tensor)


def test_gaussian_value():
    """Test gaussian value"""
    assert torch.allclose(
        TARGET, sn.gaussian(torch.ones(5)), rtol=1e-04, atol=1e-05
    )


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes test_esn.py; pylint-3 -d E0401,E1101 -f parseable test_esn.py" # NOQA, pylint: disable=C0301
# End:
