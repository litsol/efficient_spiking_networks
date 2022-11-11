# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

""" test_esn """

from typing import Optional

import torch
from expecter import expect

import efficient_spiking_networks.srnn_layers.spike_neuron as sn
from efficient_spiking_networks import __version__
from efficient_spiking_networks.utilities.decorators import initializer
from efficient_spiking_networks.utilities.exceptions import InvalidContextError

TARGET = torch.Tensor([0.1080, 0.1080, 0.1080, 0.1080, 0.1080])


def test_version():
    """sanity test"""
    assert __version__ == "2022.1002-alpha"


def test_gaussian_type():
    """Test gaussian type"""
    assert isinstance(TARGET, torch.Tensor)


def test_gaussian_type_error():
    """Test gaussian type_error"""
    with expect.raises(TypeError):
        sn.gaussian(42)


def test_gaussian_value():
    """Test gaussian value"""
    assert torch.allclose(  # pylint: disable=E1101
        TARGET,
        sn.gaussian(torch.ones(5)),  # pylint: disable=E1101
        rtol=1e-04,
        atol=1e-05,
    )


def test_initializer_decorator():
    """Test initializer decorator"""

    class Bogus:  # pylint: disable=R0903
        """Class Docstring"""

        @initializer
        def __init__(  # pylint: disable=R0913
            self,
            alabama: int,
            alaska: int,
            arizona: Optional[int] = None,
            arkansas: Optional[str] = None,
            california: float = 3.0,
        ):
            """Class member function docstring"""

    bogus = Bogus(1, 2, arkansas=None)
    assert bogus.alabama == 1  # pylint: disable=E1101
    assert bogus.alaska == 2  # pylint: disable=E1101
    assert bogus.arizona is None  # pylint: disable=E1101
    assert bogus.arkansas is None  # pylint: disable=E1101
    assert bogus.california == 3.0  # pylint: disable=E1101


def test_initializer_decorator_context_exception():
    """Test initializer decorator invalid context"""
    with expect.raises(InvalidContextError):

        class Bogus2:  # pylint: disable=R0903,W0612
            """Class Docstring"""

            @initializer
            def improper_context():  # pylint: disable=E0211
                """Class member function docstring"""


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes test_esn.py; pylint-3 -d E0401 -f parseable test_esn.py" # NOQA, pylint: disable=C0301
# End:
