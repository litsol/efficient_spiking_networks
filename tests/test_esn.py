# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

""" test_esn """

from esn import __version__


def test_version():
    """sanity test"""
    assert __version__ == "0.1.0"


# finis
