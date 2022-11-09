# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

""" Custom function decorators """

__all__ = ["initializer"]

import inspect
from functools import wraps

from .exceptions import InvalidContextError


def initializer(fun):
    """This decorator takes a class constructor signature
    and makes corresponding class member variables."""

    if fun.__name__ != "__init__":
        raise InvalidContextError(
            "Only applicable context is decorating a class constructor."
        )

    specs = inspect.getfullargspec(fun)

    @wraps(fun)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(specs.args[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)
        for i in range(len(specs.defaults)):
            index = -(i + 1)
            if not hasattr(self, specs.args[index]):
                setattr(self, specs.args[index], specs.defaults[index])
        fun(self, *args, **kargs)

    return wrapper


# import-error / E0401
# Local Variables:
# compile-command: "pyflakes decorators.py; pylint-3 -d E0401 -f parseable decorators.py" # NOQA, pylint: disable=C0301
# End:
