# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
Custom exceptions
"""

__all__ = ["InvalidContextError"]


class InvalidContextError(Exception):
    """
    Raise this exception when you want to
    signal an invalid context.
    """


# finis

# Local Variables:
# compile-command: "pyflakes exceptions.py; pylint-3 -d E0401 -f parseable exceptions.py" # NOQA, pylint: disable=C0301
# End:
