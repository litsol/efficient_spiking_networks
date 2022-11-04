# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""  module docstring """

import math

# import numpy as np
import torch

# from torch import nn
# from torch.nn import functional as F


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


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes spike_neuron.py; pylint-3 -d E0401 -f parseable spike_neuron.py" # NOQA, pylint: disable=C0301
# End:
