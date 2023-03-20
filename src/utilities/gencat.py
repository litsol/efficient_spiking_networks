# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0
# gencat.py
#

""" Concatenate multiple generators into a single sequence """


def gen_cat(sources):
    """gencat"""
    for src in sources:
        yield from src


# Example use

if __name__ == "__main__":
    from pathlib import Path

    from .genopen import gen_open

    lognames = Path("www").rglob("access-log*")
    logfiles = gen_open(lognames)
    loglines = gen_cat(logfiles)
    for line in loglines:
        print(line, end="")

# finis

# Local Variables:
# compile-command: "pyflakes gencat.py; pylint-3 -d E0401 -f parseable gencat.py" # NOQA, pylint: disable=C0301
# End:
