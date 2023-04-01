# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

""" Custom function decorators """

__all__ = ["debug", "timeit", "initializer"]

import inspect
from datetime import datetime
from functools import wraps

from decorator import decorator
from loguru import logger

from .exceptions import InvalidContextError


@decorator
def debug(_func, *args, **kwargs):
    """Print the function signature and return value"""
    args_repr = [repr(arg) for arg in args]  # 1
    kwargs_repr = [f"{key}={val!r}" for key, val in kwargs.items()]  # 2
    signature = ", ".join(args_repr + kwargs_repr)  # 3
    logger.info(f"Calling {_func.__name__}({signature})")
    value = _func(*args, **kwargs)
    logger.info(f"{_func.__name__!r} returned {value!r}")  # 4
    return value


@decorator
def timeit(_func, *args, **kwargs):
    """Log the elasped time it took this function to run."""
    time_0 = datetime.now()
    rtn = _func(*args, **kwargs)
    time_1 = datetime.now()
    logger.info(f"This task took: {time_1 - time_0}")
    return rtn


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
        if specs.defaults is not None:
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
