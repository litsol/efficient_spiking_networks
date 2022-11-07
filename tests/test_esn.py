# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

""" test_esn """

import torch

import efficient_spiking_networks.srnn_layers.spike_neuron as sn
from efficient_spiking_networks import __version__


def test_version():
    """sanity test"""
    assert __version__ == "0.1.0"


def test_gaussian():
    """Test gaussian"""
    target = sn.gaussian(torch.randn(5))
    assert isinstance(target, torch.Tensor)


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes test_esn.py; pylint-3 -d E0401 -f parseable test_esn.py" # NOQA, pylint: disable=C0301
# End:
